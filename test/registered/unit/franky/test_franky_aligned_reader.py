"""An aligned reader writes into Ricky's prompt: each row whose input token is
outside its request's answers adds a map of the tap Sandy wrote at that same
token. Every other row - answer tokens, requests without taps, CUDA graph padding,
every decode row - must add exactly zero, bias included.
"""

import unittest
from types import SimpleNamespace

import msgspec
import torch
from torch.nn import functional

from sglang.srt.franky.cache.generated_spans import (
    FrankyGeneratedSpans,
    GeneratedSpan,
)
from sglang.srt.franky.cache.tap_cache import TAP_DTYPE, AlignedTaps, FrankyTapCache
from sglang.srt.franky.config import FrankyConfig
from sglang.srt.franky.parts.encoder.normed.encoder import SingularNormedEncoder
from sglang.srt.franky.parts.reader.aligned.reader import AlignedReader
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE_SIZE = 4
_NUM_TOKENS = 64
_OUTPUT_SIZE = 8
_RICKY_HIDDEN = 6


def _tap_cache() -> FrankyTapCache:
    cache = FrankyTapCache(
        encoder_ids=(0,),
        output_size=_OUTPUT_SIZE,
        page_size=_PAGE_SIZE,
        num_tokens=_NUM_TOKENS,
        req_to_token=torch.zeros((4, _NUM_TOKENS), dtype=torch.int32),
        device="cpu",
    )
    # Slot s holds a tap whose every entry is s, so a read names its slot.
    slots = torch.arange(_PAGE_SIZE, _NUM_TOKENS + _PAGE_SIZE)
    cache.write(
        encoder_id=0,
        taps=slots[:, None].to(TAP_DTYPE).expand(-1, _OUTPUT_SIZE),
        kv_slots=slots,
    )
    return cache


def _extend(*, requests: list[int], prefix_lens: list[int], counts: list[int], slots):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.tensor(requests),
        req_pool_indices_cpu=torch.tensor(requests),
        extend_prefix_lens=torch.tensor(prefix_lens),
        extend_prefix_lens_cpu=prefix_lens,
        extend_seq_lens_cpu=counts,
        out_cache_loc=torch.tensor(slots),
    )


class TestAlignedTaps(CustomTestCase):
    def test_a_prompt_row_reads_its_own_slots_tap_and_others_read_none(self):
        # Request 1 answers from token 5 and runs tokens 3-6; request 2 has no
        # taps and runs 2 tokens; a CUDA graph pads 2 rows.
        cache = _tap_cache()
        cache.set_generated_spans(
            req_pool_indices=[1, 2],
            generated_spans=[
                FrankyGeneratedSpans(spans=(GeneratedSpan(start=5),)),
                None,
            ],
            num_input_tokens=[5, 0],
        )
        slots = [20, 21, 22, 23, 30, 31]
        batch = _extend(
            requests=[1, 2], prefix_lens=[3, 0], counts=[4, 2], slots=slots
        )

        aligned = cache.aligned_taps(encoder_id=0, forward_batch=batch, num_rows=8)

        self.assertEqual(
            aligned.reads.tolist(), [True, True] + [False] * 6
        )
        self.assertEqual(aligned.taps[:6, 0].tolist(), [float(s) for s in slots])
        self.assertTrue(torch.all(aligned.taps[6:] == 0))

    def test_a_chain_block_recomputed_from_its_prompt_reads_only_its_prompt(self):
        cache = _tap_cache()
        cache.set_generated_spans(
            req_pool_indices=[1],
            generated_spans=[
                FrankyGeneratedSpans(spans=(GeneratedSpan(start=40),), block_size=32)
            ],
            num_input_tokens=[50],
        )
        batch = _extend(
            requests=[1], prefix_lens=[36], counts=[8], slots=list(range(10, 18))
        )

        aligned = cache.aligned_taps(encoder_id=0, forward_batch=batch, num_rows=8)

        self.assertEqual(aligned.reads.tolist(), [True] * 4 + [False] * 4)

    def test_earlier_answers_read_none_and_the_turn_between_them_reads(self):
        cache = _tap_cache()
        spans = FrankyGeneratedSpans(
            spans=(GeneratedSpan(start=3, end=6), GeneratedSpan(start=8))
        )
        cache.set_generated_spans(
            req_pool_indices=[1], generated_spans=[spans], num_input_tokens=[8]
        )
        batch = _extend(
            requests=[1], prefix_lens=[0], counts=[10], slots=list(range(10, 20))
        )

        aligned = cache.aligned_taps(encoder_id=0, forward_batch=batch, num_rows=10)

        self.assertEqual(
            aligned.reads.tolist(),
            [True, True, True, False, False, False, True, True, False, False],
        )

    def test_a_decode_forward_reads_none(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE, req_pool_indices=torch.tensor([1])
        )

        self.assertIsNone(
            _tap_cache().aligned_taps(encoder_id=0, forward_batch=batch, num_rows=1)
        )


class _FixedTaps:
    def __init__(self, aligned):
        self.aligned = aligned

    def aligned_taps(self, *, encoder_id, forward_batch, num_rows):
        return self.aligned


class TestAlignedReader(CustomTestCase):
    def _reader(self) -> AlignedReader:
        torch.manual_seed(0)
        reader = AlignedReader(
            input_size=_OUTPUT_SIZE, ricky_hidden_size=_RICKY_HIDDEN, output_scale=2.0
        )
        torch.nn.init.normal_(reader.projection.weight)
        torch.nn.init.ones_(reader.projection.bias)
        return reader

    def test_a_reading_row_adds_the_scaled_map_and_others_add_exact_zeros(self):
        reader = self._reader()
        taps = torch.randn(4, _OUTPUT_SIZE)
        reads = torch.tensor([True, False, True, False])

        added = reader(
            tap_cache=_FixedTaps(AlignedTaps(taps=taps, reads=reads)),
            encoder_id=0,
            residual=torch.zeros(4, _RICKY_HIDDEN),
            forward_batch=SimpleNamespace(forward_mode=ForwardMode.EXTEND),
        )

        # the taps pass through the tap dtype, as the tap cache holds them
        expected = 2.0 * reader.projection(taps.to(TAP_DTYPE).float())
        self.assertTrue(torch.allclose(added[reads], expected[reads], atol=1e-6))
        self.assertTrue(torch.all(added[~reads] == 0))

    def test_a_decode_forward_adds_zeros(self):
        added = self._reader()(
            tap_cache=_FixedTaps(None),
            encoder_id=0,
            residual=torch.ones(3, _RICKY_HIDDEN),
            forward_batch=SimpleNamespace(forward_mode=ForwardMode.DECODE),
        )

        self.assertEqual(tuple(added.shape), (3, _RICKY_HIDDEN))
        self.assertTrue(torch.all(added == 0))


class TestSingularNormedEncoder(CustomTestCase):
    def test_a_tap_is_the_linear_map_of_the_rms_normalised_hidden(self):
        torch.manual_seed(0)
        encoder = SingularNormedEncoder(
            input_layers=(16,), output_size=4, sandy_hidden_size=_OUTPUT_SIZE
        )
        hiddens = torch.randn(3, _OUTPUT_SIZE)

        state = encoder.accumulate(state=None, layer=16, hiddens=hiddens)

        normed = functional.rms_norm(hiddens, (_OUTPUT_SIZE,), eps=1e-6)
        self.assertTrue(
            torch.allclose(encoder.finish(state=state), encoder.projection(normed))
        )

    def test_a_tokens_tap_ignores_the_scale_of_its_hidden(self):
        encoder = SingularNormedEncoder(
            input_layers=(16,), output_size=4, sandy_hidden_size=_OUTPUT_SIZE
        )
        hiddens = torch.randn(2, _OUTPUT_SIZE)

        taps = encoder.accumulate(state=None, layer=16, hiddens=hiddens)
        scaled = encoder.accumulate(state=None, layer=16, hiddens=hiddens * 50)

        self.assertTrue(torch.allclose(taps, scaled, atol=1e-5))


class TestAlignedConfig(CustomTestCase):
    def test_the_config_file_names_the_normed_encoder_and_the_aligned_reader(self):
        config = msgspec.json.decode(
            msgspec.json.encode(
                {
                    "encoder": {
                        "type": "singular_normed",
                        "layers": [16],
                        "output_size": _OUTPUT_SIZE,
                    },
                    "reader": {"type": "aligned", "layers": [8], "output_scale": 3.5},
                    "injector": {"type": "residual"},
                    "tap_page_size": 64,
                }
            ),
            type=FrankyConfig,
        )

        reader = config.reader.build(
            input_size=config.encoder.output_size, ricky_hidden_size=_RICKY_HIDDEN
        )
        encoder = config.encoder.build(index=0, sandy_hidden_size=32)
        self.assertIsInstance(reader, AlignedReader)
        self.assertEqual(reader.output_scale, 3.5)
        self.assertIsInstance(encoder, SingularNormedEncoder)
        self.assertEqual(encoder.input_layers, (16,))

    def test_an_aligned_reader_without_a_positive_scale_is_refused(self):
        with self.assertRaises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"type": "aligned", "layers": [8], "output_scale": 0}',
                type=FrankyConfig.__annotations__["reader"],
            )


if __name__ == "__main__":
    unittest.main()
