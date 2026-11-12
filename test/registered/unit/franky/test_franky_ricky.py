"""Ricky's readers must add what the trained readers add, at the right layers,
or Ricky runs on inputs it never saw in training. A token of a request without
taps must get exactly zero.
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional

from sglang.srt.franky.cache.generated_spans import (
    FrankyGeneratedSpans,
    GeneratedSpan,
)
from sglang.srt.franky.cache.kv_allocator import FrankyKvAllocator
from sglang.srt.franky.cache.tap_cache import TAP_DTYPE
from sglang.srt.franky.parts.franky_mixin import FrankyMixin
from sglang.srt.franky.parts.franky_parts import FrankyParts
from sglang.srt.franky.parts.reader.query import reader as query_reader
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_HIDDEN_SIZE = 16
_TAP_SIZE = 8
_NUM_QUERIES = 2
_MLP_HIDDEN_SIZE = 6
_NUM_LAYERS = 4
_READER_LAYERS = [1, 3]
_PAGE_SIZE = 4
_CONTEXT = 32

# Request rows: 1 has taps for its whole prompt, 2 for less than it, 3 none; 4 is
# block 3 of a chain whose prompt is 5 tokens, in blocks of 4.
_TAP_LENS = {1: 5, 2: 3, 3: 0, 4: 13}


class _LanguageModel(FrankyMixin, nn.Module):
    def __init__(self, *, start_layer, end_layer):
        super().__init__()
        self.start_layer, self.end_layer = start_layer, end_layer
        self.config = SimpleNamespace(hidden_size=_HIDDEN_SIZE)


class _Model(nn.Module):
    def __init__(self, *, start_layer=0, end_layer=_NUM_LAYERS):
        super().__init__()
        self.model = _LanguageModel(start_layer=start_layer, end_layer=end_layer)


def _flash_attention_on_cpu(
    *, q, k_cache, v_cache, page_table, cache_seqlens, softmax_scale, causal, out
):
    """FlashAttention's paged KV cache call needs a GPU, so this does what it does,
    one query row at a time: a row attends to the first ``cache_seqlens`` keys of
    its pages, and a row with no key gets zero.
    """
    page_size = k_cache.shape[1]
    keys_of_slots = k_cache.reshape(-1, k_cache.shape[-1])
    for row in range(q.shape[0]):
        num_keys = int(cache_seqlens[row])
        if num_keys == 0:
            out[row] = 0
            continue
        slots = (
            page_table[row].long()[:, None] * page_size + torch.arange(page_size)
        ).flatten()[:num_keys]
        keys = keys_of_slots[slots].float()
        scores = q[row, 0].float() @ keys.T * softmax_scale
        out[row, 0] = (torch.softmax(scores, dim=-1) @ keys).to(out.dtype)
    return out


def _nan_filled(tensor, size, *, dtype=None):
    """Stands in for ``new_empty``: a buffer may hold anything, NaN too."""
    return torch.full(size, float("nan"), dtype=dtype or tensor.dtype)


def _pr_mlp(joined, weights, prefix):
    hidden = functional.linear(
        joined, weights[f"{prefix}mlp.0.weight"], weights[f"{prefix}mlp.0.bias"]
    )
    return functional.linear(
        functional.gelu(hidden),
        weights[f"{prefix}mlp.2.weight"],
        weights[f"{prefix}mlp.2.bias"],
    )


def _pr_query_reader(*, queries, taps, visible, weights, prefix):
    """The PR's ``QueryReader``: each token's queries are heads over one K=V head
    of taps, then the MLP.
    """
    heads = queries.float().transpose(0, 1)
    keys = taps.float()[None].expand(heads.shape[0], -1, -1)
    attended = functional.scaled_dot_product_attention(
        heads, keys, keys, attn_mask=visible
    )
    joined = attended.transpose(0, 1).flatten(start_dim=1).to(TAP_DTYPE).float()
    return _pr_mlp(joined, weights, prefix)


class _Files(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.directory = tempfile.mkdtemp(prefix="franky_ricky_")
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        torch.manual_seed(0)
        attention = patch.object(
            query_reader, "flash_attn_with_kvcache", _flash_attention_on_cpu
        )
        attention.start()
        self.addCleanup(attention.stop)
        # Row r holds its request's KV slots from slot 8 * r on.
        self.req_to_token = torch.zeros((5, _CONTEXT), dtype=torch.int32)
        for row in _TAP_LENS:
            self.req_to_token[row] = torch.arange(8 * row, 8 * row + _CONTEXT)

    def _write(self):
        config_path = os.path.join(self.directory, "franky.json")
        reader = {
            "type": "residual_query",
            "layers": _READER_LAYERS,
            "num_queries": _NUM_QUERIES,
            "mlp_hidden_size": _MLP_HIDDEN_SIZE,
        }
        with open(config_path, "w") as file:
            json.dump(
                {
                    "encoder": {
                        "type": "singular",
                        "layers": [2, 5],
                        "output_size": _TAP_SIZE,
                    },
                    "reader": reader,
                    "injector": {"type": "residual"},
                    "tap_page_size": _PAGE_SIZE,
                },
                file,
            )
        weights = {}
        for index in range(len(_READER_LAYERS)):
            prefix = f"readers.{index}."
            weights[f"{prefix}query_projection.weight"] = torch.randn(
                _NUM_QUERIES * _TAP_SIZE, _HIDDEN_SIZE
            )
            weights[f"{prefix}mlp.0.weight"] = torch.randn(
                _MLP_HIDDEN_SIZE, _NUM_QUERIES * _TAP_SIZE
            )
            weights[f"{prefix}mlp.0.bias"] = torch.randn(_MLP_HIDDEN_SIZE)
            weights[f"{prefix}mlp.2.weight"] = torch.randn(
                _HIDDEN_SIZE, _MLP_HIDDEN_SIZE
            )
            weights[f"{prefix}mlp.2.bias"] = torch.randn(_HIDDEN_SIZE)
        weights_path = os.path.join(self.directory, "franky.safetensors")
        save_file(weights, weights_path)
        return config_path, weights_path, weights

    def _attach(self, *, model, config_path, weights_path):
        parts = FrankyParts.attach(
            model=model,
            side="ricky",
            config_path=config_path,
            weights_path=weights_path,
            dtype=torch.float32,
            device="cpu",
        )
        parts.init_tap_cache(
            num_tokens=8 * len(self.req_to_token) + _CONTEXT,
            kv_allocator=FrankyKvAllocator.__new__(FrankyKvAllocator),
            req_to_token_pool=SimpleNamespace(req_to_token=self.req_to_token),
        )
        self.taps = {}
        for encoder_id in parts._tap_encoder_ids():
            for row, tap_len in _TAP_LENS.items():
                taps = torch.randn(tap_len, _TAP_SIZE, dtype=TAP_DTYPE)
                parts.tap_cache.write(
                    encoder_id=encoder_id,
                    taps=taps,
                    kv_slots=self.req_to_token[row, :tap_len].long(),
                )
                parts.tap_cache.set_generated_spans(
                    req_pool_indices=[row],
                    generated_spans=[
                        FrankyGeneratedSpans(spans=(GeneratedSpan(start=tap_len),))
                        if tap_len
                        else None
                    ],
                    num_input_tokens=[tap_len],
                )
                self.taps[encoder_id, row] = taps
        return parts


def _extend_batch(*, rows, prefix_lens, extend_lens):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.tensor(rows),
        req_pool_indices_cpu=torch.tensor(rows),
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32),
        extend_prefix_lens_cpu=list(prefix_lens),
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
        extend_seq_lens_cpu=list(extend_lens),
    )


class TestReaders(_Files):
    def _expected(self, *, encoder_id, row, num_taps, queries, weights):
        num_taps = torch.tensor(num_taps)
        reads = num_taps > 0
        expected = torch.zeros(len(num_taps), _HIDDEN_SIZE)
        if reads.any():
            expected[reads] = _pr_query_reader(
                queries=queries[reads],
                taps=self.taps[encoder_id, row],
                visible=torch.arange(_TAP_LENS[row])[None, :] < num_taps[reads, None],
                weights=weights,
                prefix=f"readers.{encoder_id}.",
            )
        return expected

    def test_residual_queries_add_the_pr_result(self):
        config_path, weights_path, weights = self._write()
        model = _Model()
        self._attach(model=model, config_path=config_path, weights_path=weights_path)
        # Row 1 runs its whole prompt; row 2 runs past its taps; row 3 has none.
        batch = _extend_batch(
            rows=[1, 2, 3], prefix_lens=[0, 2, 0], extend_lens=[5, 4, 2]
        )
        num_taps = {1: [0, 0, 0, 0, 5], 2: [3, 3, 3, 3], 3: [0, 0]}
        layer_output = torch.randn(11, _HIDDEN_SIZE)
        residual = torch.randn(11, _HIDDEN_SIZE)

        for encoder_id, layer in enumerate(_READER_LAYERS):
            hidden_states, residual_out = model.model.franky_after_layer_with_residual(
                layer_idx=layer - 1,
                hidden_states=layer_output,
                residual=residual,
                forward_batch=batch,
            )
            self.assertIs(residual_out, residual)
            queries = (
                functional.linear(
                    layer_output + residual,
                    weights[f"readers.{encoder_id}.query_projection.weight"],
                )
                .view(11, _NUM_QUERIES, _TAP_SIZE)
                .to(TAP_DTYPE)
            )
            start = 0
            for row, row_num_taps in num_taps.items():
                end = start + len(row_num_taps)
                expected = self._expected(
                    encoder_id=encoder_id,
                    row=row,
                    num_taps=row_num_taps,
                    queries=queries[start:end],
                    weights=weights,
                )
                torch.testing.assert_close(
                    hidden_states[start:end] - layer_output[start:end],
                    expected,
                    atol=2e-3,
                    rtol=2e-3,
                )
                start = end

    def test_a_later_blocks_tokens_read_the_taps_before_their_block(self):
        """Row 4's tokens 5..8 came from block 1 and 9..12 from block 2. Each
        token reads the taps before the block of the token it predicts, as that
        block's decode did; token 3 predicts a prompt token and reads nothing, and
        token 12 predicts block 3's first token.
        """
        config_path, weights_path, weights = self._write()
        model = _Model()
        parts = self._attach(
            model=model, config_path=config_path, weights_path=weights_path
        )
        parts.tap_cache.set_generated_spans(
            req_pool_indices=[4],
            generated_spans=[
                FrankyGeneratedSpans(spans=(GeneratedSpan(start=5),), block_size=4)
            ],
            num_input_tokens=[13],
        )
        layer_output = torch.randn(10, _HIDDEN_SIZE)

        hidden_states, _ = model.model.franky_after_layer_with_residual(
            layer_idx=0,
            hidden_states=layer_output,
            residual=None,
            forward_batch=_extend_batch(rows=[4], prefix_lens=[3], extend_lens=[10]),
        )

        queries = functional.linear(
            layer_output, weights["readers.0.query_projection.weight"]
        ).view(10, _NUM_QUERIES, _TAP_SIZE)
        expected = self._expected(
            encoder_id=0,
            row=4,
            num_taps=[0, 5, 5, 5, 5, 9, 9, 9, 9, 13],
            queries=queries.to(TAP_DTYPE),
            weights=weights,
        )
        torch.testing.assert_close(
            hidden_states - layer_output, expected, atol=2e-3, rtol=2e-3
        )

    def test_tokens_without_taps_and_padded_tokens_get_exactly_zero(self):
        """A CUDA graph pads the batch with tokens of no request."""
        config_path, weights_path, _ = self._write()
        model = _Model()
        self._attach(model=model, config_path=config_path, weights_path=weights_path)
        layer_output = torch.randn(4, _HIDDEN_SIZE)

        with patch.object(torch.Tensor, "new_empty", _nan_filled):
            hidden_states, _ = model.model.franky_after_layer_with_residual(
                layer_idx=0,
                hidden_states=layer_output,
                residual=None,
                forward_batch=_extend_batch(rows=[3], prefix_lens=[0], extend_lens=[2]),
            )

        self.assertTrue(torch.equal(hidden_states, layer_output))

    def _decode(self, *, model, rows):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE, req_pool_indices=torch.tensor(rows)
        )
        layer_output = torch.randn(len(rows), _HIDDEN_SIZE)
        hidden_states, _ = model.model.franky_after_layer_with_residual(
            layer_idx=0, hidden_states=layer_output, residual=None, forward_batch=batch
        )
        return hidden_states - layer_output, layer_output

    def test_a_decoded_token_reads_all_its_requests_taps(self):
        config_path, weights_path, weights = self._write()
        model = _Model()
        self._attach(model=model, config_path=config_path, weights_path=weights_path)

        added, layer_output = self._decode(model=model, rows=[1, 2, 3])

        queries = functional.linear(
            layer_output, weights["readers.0.query_projection.weight"]
        ).view(3, _NUM_QUERIES, _TAP_SIZE)
        for token, row in enumerate((1, 2, 3)):
            expected = self._expected(
                encoder_id=0,
                row=row,
                num_taps=[_TAP_LENS[row]],
                queries=queries[token : token + 1].to(TAP_DTYPE),
                weights=weights,
            )
            torch.testing.assert_close(
                added[token : token + 1], expected, atol=2e-3, rtol=2e-3
            )

    def test_layers_without_a_reader_are_left_alone(self):
        config_path, weights_path, _ = self._write()
        model = _Model()
        self._attach(model=model, config_path=config_path, weights_path=weights_path)
        layer_output = torch.randn(5, _HIDDEN_SIZE)

        for layer_idx in (1, 3):
            hidden_states, _ = model.model.franky_after_layer_with_residual(
                layer_idx=layer_idx,
                hidden_states=layer_output,
                residual=None,
                forward_batch=_extend_batch(rows=[1], prefix_lens=[0], extend_lens=[5]),
            )
            self.assertIs(hidden_states, layer_output)


class TestRickyParts(_Files):
    def test_a_pp_stage_loads_only_its_readers_and_holds_all_taps(self):
        """Reader layer ``L`` reads after the model's layer ``L``, counted from 1,
        so it belongs to the stage that ends at ``L``. Every stage holds all
        encoders' taps, since PD's transfer pairs entries by id.
        """
        config_path, weights_path, _ = self._write()
        edges = (0, 1, 3, _NUM_LAYERS)
        stages = [
            self._attach(
                model=_Model(start_layer=start_layer, end_layer=end_layer),
                config_path=config_path,
                weights_path=weights_path,
            )
            for start_layer, end_layer in zip(edges, edges[1:])
        ]

        self.assertEqual([tuple(stage._readers) for stage in stages], [(0,), (1,), ()])
        self.assertEqual([stage._tap_encoder_ids() for stage in stages], [(0, 1)] * 3)
        self.assertEqual(stages[2].kv_options().tap_bytes_per_token, 2 * _TAP_SIZE * 2)

    def test_updated_weights_reach_the_reader_on_their_stage_only(self):
        config_path, weights_path, _ = self._write()
        edges = (0, 1, _NUM_LAYERS)
        stages = [
            self._attach(
                model=_Model(start_layer=start_layer, end_layer=end_layer),
                config_path=config_path,
                weights_path=weights_path,
            )
            for start_layer, end_layer in zip(edges, edges[1:])
        ]
        before = stages[0]._readers[0].mlp[2].bias.clone()
        bias = torch.full_like(stages[1]._readers[1].mlp[2].bias, 0.5)

        for stage in stages:
            stage.load_weights([("readers.1.mlp.2.bias", bias)])

        self.assertTrue(torch.equal(stages[0]._readers[0].mlp[2].bias, before))
        self.assertTrue(torch.equal(stages[1]._readers[1].mlp[2].bias, bias))

    def test_readers_without_a_weights_file_start_at_zero(self):
        config_path, _, _ = self._write()
        parts = self._attach(
            model=_Model(start_layer=0, end_layer=_NUM_LAYERS),
            config_path=config_path,
            weights_path=None,
        )

        for reader in parts._readers.values():
            for parameter in reader.parameters():
                self.assertEqual(parameter.abs().sum().item(), 0)

    def test_weights_of_another_side_or_part_are_refused(self):
        config_path, weights_path, _ = self._write()
        parts = self._attach(
            model=_Model(start_layer=0, end_layer=_NUM_LAYERS),
            config_path=config_path,
            weights_path=weights_path,
        )

        for name in ("encoders.0.projection.weight", "readers.7.mlp.0.bias"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parts.load_weights([(name, torch.zeros(1))])


if __name__ == "__main__":
    unittest.main()
