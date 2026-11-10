"""Sandy's encoders must give the taps the trained encoders give, on any PP
split, or every Ricky token reads wrong taps.
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional

from sglang.srt.franky.cache.kv_allocator import FrankyKvAllocator
from sglang.srt.franky.cache.tap_cache import TAP_DTYPE, FrankyTapCache
from sglang.srt.franky.parts.encoder.all.encoder import AllEncoder
from sglang.srt.franky.parts.franky_mixin import FrankyMixin
from sglang.srt.franky.parts.franky_parts import FrankyParts
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_HIDDEN_SIZE = 16
_OUTPUT_SIZE = 8
_NUM_LAYERS = 6
_NUM_TOKENS = 5
_PAGE_SIZE = 4


def _pr_linear_encoder(captured_hiddens, weight):
    """The PR's ``LinearEncoder.forward``, step for step."""
    weight_blocks = weight.chunk(len(captured_hiddens), dim=1)
    output = functional.linear(captured_hiddens[0], weight_blocks[0])
    for hiddens, weight_block in zip(captured_hiddens[1:], weight_blocks[1:]):
        output = output + functional.linear(hiddens, weight_block)
    return output


class _LanguageModel(FrankyMixin, nn.Module):
    def __init__(self, *, start_layer, end_layer):
        super().__init__()
        self.start_layer, self.end_layer = start_layer, end_layer
        self.config = SimpleNamespace(hidden_size=_HIDDEN_SIZE)
        self.embed_tokens = nn.Embedding(10, _HIDDEN_SIZE)


class _Model(nn.Module):
    def __init__(self, *, start_layer=0, end_layer=_NUM_LAYERS, tied=False):
        super().__init__()
        self.model = _LanguageModel(start_layer=start_layer, end_layer=end_layer)
        self.lm_head = (
            self.model.embed_tokens if tied else nn.Linear(_HIDDEN_SIZE, 10, bias=False)
        )
        self.logits_processor = nn.Identity()


class _Files(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.directory = tempfile.mkdtemp(prefix="franky_sandy_")
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        torch.manual_seed(0)
        self.hiddens = {
            layer: torch.randn(_NUM_TOKENS, _HIDDEN_SIZE, dtype=TAP_DTYPE)
            for layer in range(1, _NUM_LAYERS + 1)
        }
        self.kv_slots = torch.arange(_PAGE_SIZE, _PAGE_SIZE + _NUM_TOKENS)

    def _write(self, *, encoder_type, layers, weights=None):
        config_path = os.path.join(self.directory, "franky.json")
        with open(config_path, "w") as file:
            json.dump(
                {
                    "encoder": {
                        "type": encoder_type,
                        "layers": layers,
                        "output_size": _OUTPUT_SIZE,
                    },
                    "reader": {
                        "type": "fixed_query",
                        "layers": list(range(1, len(layers) + 1)),
                        "num_queries": 1,
                        "mlp_hidden_size": 4,
                    },
                    "injector": {"type": "residual"},
                    "tap_page_size": _PAGE_SIZE,
                },
                file,
            )
        if weights is None:
            weights = {}
            for index in range(len(layers)):
                num_inputs = {"all": len(layers), "masked": index + 1}.get(
                    encoder_type, 1
                )
                weights[f"encoders.{index}.projection.weight"] = torch.randn(
                    _OUTPUT_SIZE, num_inputs * _HIDDEN_SIZE, dtype=TAP_DTYPE
                )
        weights_path = os.path.join(self.directory, "franky.safetensors")
        save_file(weights, weights_path)
        return config_path, weights_path, weights

    def _attach(self, *, model, config_path, weights_path):
        parts = FrankyParts.attach(
            model=model,
            side="sandy",
            config_path=config_path,
            weights_path=weights_path,
            dtype=TAP_DTYPE,
            device="cpu",
        )
        parts.init_tap_cache(
            num_tokens=64,
            kv_allocator=_kv_allocator(),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((2, 64), dtype=torch.int32)
            ),
        )
        return parts

    def _run_layers(self, *, model, start_layer, end_layer, proxies=None):
        """The model's layer loop, as Qwen3.5 calls the mixin."""
        forward_batch = SimpleNamespace(out_cache_loc=self.kv_slots)
        if proxies is not None:
            model.model.franky_read_pp(pp_proxy_tensors=proxies)
        for layer_idx in range(start_layer, end_layer):
            model.model.franky_after_layer_with_residual(
                layer_idx=layer_idx,
                hidden_states=self.hiddens[layer_idx + 1],
                residual=None,
                forward_batch=forward_batch,
            )
        proxies = PPProxyTensors({})
        model.model.franky_write_pp(pp_proxy_tensors=proxies)
        return proxies


def _kv_allocator():
    return FrankyKvAllocator.__new__(FrankyKvAllocator)


def _taps(parts: FrankyParts, index: int, kv_slots) -> torch.Tensor:
    cache: FrankyTapCache = parts.tap_cache
    return cache._buffers[index][cache.slots_of(kv_slots=kv_slots)]


class TestLinearEncoder(_Files):
    def test_the_running_sum_is_the_pr_formula(self):
        layers = (2, 4, 5)
        encoder = AllEncoder(
            input_layers=layers,
            output_size=_OUTPUT_SIZE,
            sandy_hidden_size=_HIDDEN_SIZE,
        ).to(TAP_DTYPE)

        state = None
        for layer in layers:
            state = encoder.accumulate(
                state=state, layer=layer, hiddens=self.hiddens[layer]
            )

        expected = _pr_linear_encoder(
            tuple(self.hiddens[layer] for layer in layers), encoder.projection.weight
        )
        self.assertTrue(torch.equal(encoder.finish(state=state), expected))


class TestSandyParts(_Files):
    def test_each_encoder_type_writes_the_pr_taps(self):
        for encoder_type in ("all", "masked", "singular"):
            with self.subTest(encoder_type):
                layers = [2, 4, 5]
                config_path, weights_path, weights = self._write(
                    encoder_type=encoder_type, layers=layers
                )
                model = _Model()
                parts = self._attach(
                    model=model, config_path=config_path, weights_path=weights_path
                )

                self._run_layers(model=model, start_layer=0, end_layer=_NUM_LAYERS)

                for index in range(len(layers)):
                    inputs = parts.config.encoder.input_layers(index=index)
                    expected = _pr_linear_encoder(
                        tuple(self.hiddens[layer] for layer in inputs),
                        weights[f"encoders.{index}.projection.weight"],
                    )
                    self.assertTrue(
                        torch.equal(_taps(parts, index, self.kv_slots), expected)
                    )

    def test_any_pp_split_gives_the_same_taps(self):
        """An encoder's sum crosses stages, from a first layer on a stage's edge,
        through a stage that has none of its layers.
        """
        config_path, weights_path, _ = self._write(encoder_type="all", layers=[2, 5])
        whole = _Model()
        whole_parts = self._attach(
            model=whole, config_path=config_path, weights_path=weights_path
        )
        self._run_layers(model=whole, start_layer=0, end_layer=_NUM_LAYERS)

        edges = (0, 2, 4, _NUM_LAYERS)
        proxies = None
        stage_parts = []
        for start_layer, end_layer in zip(edges, edges[1:]):
            model = _Model(start_layer=start_layer, end_layer=end_layer)
            stage_parts.append(
                self._attach(
                    model=model, config_path=config_path, weights_path=weights_path
                )
            )
            proxies = self._run_layers(
                model=model,
                start_layer=start_layer,
                end_layer=end_layer,
                proxies=proxies,
            )

        last_stage = stage_parts[-1]
        self.assertEqual(last_stage._tap_encoder_ids(), (0, 1))
        for index in (0, 1):
            self.assertTrue(
                torch.equal(
                    _taps(last_stage, index, self.kv_slots),
                    _taps(whole_parts, index, self.kv_slots),
                )
            )
        self.assertEqual(stage_parts[0]._tap_encoder_ids(), ())
        self.assertEqual(len(proxies), 0)

    def test_padded_tokens_write_to_the_dummy_page(self):
        config_path, weights_path, _ = self._write(encoder_type="singular", layers=[3])
        model = _Model()
        parts = self._attach(
            model=model, config_path=config_path, weights_path=weights_path
        )
        self.kv_slots = torch.tensor([_PAGE_SIZE, _PAGE_SIZE + 1, 0, 0, 0])

        self._run_layers(model=model, start_layer=0, end_layer=_NUM_LAYERS)

        real = _taps(parts, 0, torch.tensor([_PAGE_SIZE, _PAGE_SIZE + 1]))
        self.assertTrue(torch.all(real != 0))
        self.assertTrue(
            torch.all(parts.tap_cache._buffers[0][_PAGE_SIZE + 2 : 2 * _PAGE_SIZE] == 0)
        )


class TestAttach(_Files):
    def test_sandy_drops_a_separate_head_and_keeps_a_tied_one(self):
        """Attention DP slices the logits after the forward, so they have rows."""
        config_path, weights_path, _ = self._write(encoder_type="singular", layers=[3])
        separate, tied = _Model(), _Model(tied=True)

        for model in (separate, tied):
            self._attach(
                model=model, config_path=config_path, weights_path=weights_path
            )
            hidden_states = torch.zeros(_NUM_TOKENS, _HIDDEN_SIZE)
            output = model.logits_processor(None, hidden_states, model.lm_head, None)
            self.assertEqual(output.next_token_logits[:2].shape, (2, 0))

        self.assertIsNone(separate.lm_head)
        self.assertIs(tied.lm_head, tied.model.embed_tokens)

    def test_bad_weights_name_the_encoder(self):
        for name, weights in {
            "missing": {},
            "wrong shape": {
                "encoders.0.projection.weight": torch.zeros(_OUTPUT_SIZE, 3)
            },
        }.items():
            with self.subTest(name):
                config_path, weights_path, _ = self._write(
                    encoder_type="singular", layers=[3], weights=weights
                )
                with self.assertRaisesRegex(ValueError, "'encoders.0.'"):
                    self._attach(
                        model=_Model(),
                        config_path=config_path,
                        weights_path=weights_path,
                    )

    def test_a_model_without_one_franky_module_is_refused(self):
        config_path, weights_path, _ = self._write(encoder_type="singular", layers=[3])
        with self.assertRaisesRegex(ValueError, "found 0"):
            FrankyParts.attach(
                model=nn.Linear(2, 2),
                side="sandy",
                config_path=config_path,
                weights_path=weights_path,
                dtype=TAP_DTYPE,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
