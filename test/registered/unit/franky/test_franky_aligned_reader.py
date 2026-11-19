"""An aligned reader writes into Ricky's prompt where Sandy's view differs from
it: each row whose input token is in its request's write ranges adds a map of the
tap Sandy wrote at that same token. Every other row - prompt rows outside the
ranges, answer tokens, requests without taps, CUDA graph padding, every decode
row - must add exactly zero, bias included. Each attention-TP rank keeps the taps
of its stripes, so the ranks' reads sum to every row's tap.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
from torch.nn import functional

from sglang.srt.franky.cache.generated_spans import (
    FrankyGeneratedSpans,
    GeneratedSpan,
    checked_write_ranges,
)
from sglang.srt.franky.cache.tap_buffers import TAP_DTYPE, TapBuffers
from sglang.srt.franky.cache.tap_request_table import TapRequestTable
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.franky.config import FrankyConfig
from sglang.srt.franky.parts.encoder.normed.encoder import SingularNormedEncoder
from sglang.srt.franky.parts.reader.aligned import reader as aligned_reader
from sglang.srt.franky.parts.reader.aligned.own_taps import OwnTaps
from sglang.srt.franky.parts.reader.aligned.reader import AlignedReader
from sglang.srt.franky.parts.reader.config import ReaderConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE_SIZE = 4
_OUTPUT_SIZE = 8
_RICKY_HIDDEN = 6
_MAX_STRIPES = 5
# Request slot to the local pages of its stripes in order.
_STRIPE_PAGES = {1: [5, 2, 7, 3, 9], 2: [1, 6, 4, 0, 8]}


def _tap(slot: int, position: int) -> float:
    """Each tap's entries name its request and its token, exactly in bfloat16."""
    return 10.0 * slot + position


def _rank_pages(stripes: TapStripes) -> torch.Tensor:
    buffers = TapBuffers(
        encoder_ids=(0,),
        output_size=_OUTPUT_SIZE,
        page_size=_PAGE_SIZE,
        num_pages=10,
        device="cpu",
    )
    for slot, stripe_pages in _STRIPE_PAGES.items():
        for position in range(len(stripe_pages) * stripes.num_ranks * _PAGE_SIZE):
            page = position // _PAGE_SIZE
            if page % stripes.num_ranks != stripes.rank:
                continue
            local_page = stripe_pages[page // stripes.num_ranks]
            buffers.write(
                encoder_id=0,
                taps=torch.full((1, _OUTPUT_SIZE), _tap(slot, position), dtype=TAP_DTYPE),
                slots=torch.tensor([local_page * _PAGE_SIZE + position % _PAGE_SIZE]),
            )
    return buffers.pages(0)


def _request_table(spans: dict) -> TapRequestTable:
    table = TapRequestTable(num_slots=4, max_stripes=_MAX_STRIPES, device="cpu")
    slots = list(_STRIPE_PAGES)
    table.bind(
        req_pool_indices=slots,
        pages=[torch.tensor(_STRIPE_PAGES[slot]) for slot in slots],
        spans=[spans.get(slot) for slot in slots],
        num_input_tokens=[0 for _ in slots],
    )
    return table


def _extend(*, slots: list, prefix_lens: list, counts: list):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.tensor(slots),
        req_pool_indices_cpu=torch.tensor(slots),
        extend_prefix_lens=torch.tensor(prefix_lens),
        extend_prefix_lens_cpu=prefix_lens,
        extend_seq_lens_cpu=counts,
        rids=[f"rid-{slot}" for slot in slots],
    )


def _own_taps(spans, batch, *, num_rows, num_ranks=1):
    """The rows' taps summed over the ranks, and the rows' reads."""
    summed, reads = None, None
    for rank in range(num_ranks):
        stripes = TapStripes(num_ranks=num_ranks, rank=rank)
        own = OwnTaps.of(
            batch,
            pages=_rank_pages(stripes),
            request_table=_request_table(spans),
            stripes=stripes,
            num_rows=num_rows,
        )
        summed = own.taps.float() if summed is None else summed + own.taps.float()
        reads = own.reads
    return summed, reads


class TestOwnTaps(CustomTestCase):
    def test_a_prompt_row_reads_its_own_tap_and_others_read_none(self):
        # Request 1 answers from token 5 and runs tokens 3-6; request 2 has no
        # taps and runs 2 tokens; a CUDA graph pads 2 rows.
        spans = {
            1: FrankyGeneratedSpans(
                spans=(GeneratedSpan(start=5, sandy_start=5),), write_ranges=((2, 5),)
            )
        }
        batch = _extend(slots=[1, 2], prefix_lens=[3, 0], counts=[4, 2])
        stripes = TapStripes(num_ranks=1, rank=0)

        own = OwnTaps.of(
            batch,
            pages=_rank_pages(stripes),
            request_table=_request_table(spans),
            stripes=stripes,
            num_rows=8,
        )

        self.assertEqual(own.reads.tolist(), [True, True] + [False] * 6)
        self.assertEqual(own.taps[:2, 0].tolist(), [_tap(1, 3), _tap(1, 4)])
        self.assertTrue(torch.all(own.taps[2:] == 0))
        self.assertEqual(own.positions, [3, 4, 5, 6, 0, 1, -1, -1])
        self.assertEqual(own.requests, [0, 0, 0, 0, 1, 1, -1, -1])

    def test_the_ranks_reads_sum_to_every_rows_tap(self):
        spans = {
            slot: FrankyGeneratedSpans(
                spans=(GeneratedSpan(start=30, sandy_start=30),),
                write_ranges=((0, 30),),
            )
            for slot in (1, 2)
        }
        batch = _extend(slots=[1, 2], prefix_lens=[0, 6], counts=[20, 12])

        for num_ranks in (1, 2, 4):
            with self.subTest(num_ranks=num_ranks):
                summed, reads = _own_taps(
                    spans, batch, num_rows=32, num_ranks=num_ranks
                )

                expected = [_tap(1, t) for t in range(20)] + [
                    _tap(2, t) for t in range(6, 18)
                ]
                self.assertTrue(bool(reads.all()))
                self.assertEqual(summed[:, 0].tolist(), expected)

    def test_a_chain_block_recomputed_from_its_prompt_reads_only_its_prompt(self):
        spans = {
            1: FrankyGeneratedSpans(
                spans=(GeneratedSpan(start=12, sandy_start=12),),
                block_size=8,
                write_ranges=((6, 12),),
            )
        }
        batch = _extend(slots=[1], prefix_lens=[8], counts=[8])

        _, reads = _own_taps(spans, batch, num_rows=8)

        self.assertEqual(reads.tolist(), [True] * 4 + [False] * 4)

    def test_earlier_answers_read_none_and_the_turn_between_them_reads(self):
        spans = {
            1: FrankyGeneratedSpans(
                spans=(
                    GeneratedSpan(start=3, sandy_start=3, end=6),
                    GeneratedSpan(start=8, sandy_start=8),
                ),
                write_ranges=((0, 3), (6, 8)),
            )
        }
        batch = _extend(slots=[1], prefix_lens=[0], counts=[10])

        _, reads = _own_taps(spans, batch, num_rows=10)

        self.assertEqual(
            reads.tolist(),
            [True, True, True, False, False, False, True, True, False, False],
        )

    def test_a_prompt_row_outside_the_write_ranges_reads_none(self):
        spans = {
            1: FrankyGeneratedSpans(
                spans=(GeneratedSpan(start=6, sandy_start=6),),
                write_ranges=((1, 2), (4, 5)),
            )
        }
        batch = _extend(slots=[1], prefix_lens=[0], counts=[6])

        summed, reads = _own_taps(spans, batch, num_rows=6)

        self.assertEqual(reads.tolist(), [False, True, False, False, True, False])
        self.assertEqual(
            summed[:, 0].tolist(), [0, _tap(1, 1), 0, 0, _tap(1, 4), 0]
        )


class TestRowsWaitForTheirOwnTaps(CustomTestCase):
    def test_a_row_in_a_write_range_waits_for_its_own_tap(self):
        spans = FrankyGeneratedSpans(
            spans=(GeneratedSpan(start=20, sandy_start=20),),
            write_ranges=((3, 6), (10, 12)),
        )

        self.assertEqual(spans.rows_reading_at_most(0), 3)
        self.assertEqual(spans.rows_reading_at_most(4), 4)
        self.assertEqual(spans.rows_reading_at_most(6), 10)
        self.assertEqual(spans.rows_reading_at_most(12), 19)

    def test_answer_rows_in_a_write_range_do_not_wait(self):
        spans = FrankyGeneratedSpans(
            spans=(
                GeneratedSpan(start=2, sandy_start=2, end=5),
                GeneratedSpan(start=9, sandy_start=9),
            ),
            write_ranges=((0, 8),),
        )

        self.assertEqual(spans.rows_reading_at_most(1), 1)
        self.assertEqual(spans.rows_reading_at_most(2), 5)
        self.assertEqual(spans.rows_reading_at_most(5), 5)
        self.assertEqual(spans.rows_reading_at_most(8), 8)


class TestWriteRanges(CustomTestCase):
    def test_runs_in_order_inside_the_prompt_are_kept(self):
        self.assertEqual(
            checked_write_ranges([[0, 2], [5, 9]], prompt_length=9), ((0, 2), (5, 9))
        )

    def test_runs_out_of_order_empty_or_past_the_prompt_are_refused(self):
        for ranges in ([[5, 9], [0, 2]], [[2, 2]], [[3, 10]], [[1, 2, 3]]):
            with self.assertRaises(ValueError):
                checked_write_ranges(ranges, prompt_length=9)


class TestAlignedReader(CustomTestCase):
    def _reader(self) -> AlignedReader:
        torch.manual_seed(0)
        reader = AlignedReader(
            input_size=_OUTPUT_SIZE, ricky_hidden_size=_RICKY_HIDDEN, output_scale=2.0
        )
        torch.nn.init.normal_(reader.projection.weight)
        torch.nn.init.ones_(reader.projection.bias)
        return reader

    def _taps(self, spans) -> SimpleNamespace:
        stripes = TapStripes(num_ranks=1, rank=0)
        return SimpleNamespace(
            encoder_id=0,
            pages=_rank_pages(stripes),
            reading=SimpleNamespace(
                request_table=_request_table(spans), stripes=stripes
            ),
        )

    def _expected(self, reader, positions):
        taps = torch.tensor([[_tap(1, t)] * _OUTPUT_SIZE for t in positions])
        # the taps pass through the tap dtype, as the tap pages hold them
        return 2.0 * reader.projection(taps.to(TAP_DTYPE).float())

    def test_a_reading_row_adds_the_scaled_map_and_others_add_exact_zeros(self):
        reader = self._reader()
        spans = {
            1: FrankyGeneratedSpans(
                spans=(GeneratedSpan(start=4, sandy_start=4),),
                write_ranges=((0, 1), (2, 3)),
            )
        }

        added = reader(
            taps=self._taps(spans),
            residual=torch.zeros(4, _RICKY_HIDDEN),
            forward_batch=_extend(slots=[1], prefix_lens=[0], counts=[4]),
            rows=None,
        )

        reads = torch.tensor([True, False, True, False])
        expected = self._expected(reader, range(4))
        self.assertTrue(torch.allclose(added[reads], expected[reads], atol=1e-3))
        self.assertTrue(torch.all(added[~reads] == 0))

    def test_rows_of_a_tail_read_their_own_tokens_and_padding_reads_none(self):
        reader = self._reader()
        spans = {
            1: FrankyGeneratedSpans(
                spans=(GeneratedSpan(start=8, sandy_start=8),), write_ranges=((0, 8),)
            )
        }

        added = reader(
            taps=self._taps(spans),
            residual=torch.zeros(3, _RICKY_HIDDEN),
            forward_batch=_extend(slots=[1], prefix_lens=[0], counts=[6]),
            rows=torch.tensor([5, 2, 6]),
        )

        self.assertTrue(
            torch.allclose(added[:2], self._expected(reader, (5, 2)), atol=1e-3)
        )
        self.assertTrue(torch.all(added[2] == 0))

    def test_a_decode_forward_adds_zeros(self):
        added = self._reader()(
            taps=self._taps({}),
            residual=torch.ones(3, _RICKY_HIDDEN),
            forward_batch=SimpleNamespace(forward_mode=ForwardMode.DECODE),
            rows=None,
        )

        self.assertEqual(tuple(added.shape), (3, _RICKY_HIDDEN))
        self.assertTrue(torch.all(added == 0))

    def test_the_dump_lists_each_requests_reading_rows_and_their_taps(self):
        spans = {
            1: FrankyGeneratedSpans(
                spans=(GeneratedSpan(start=4, sandy_start=4),), write_ranges=((1, 3),)
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dump.jsonl")
            with patch.dict(os.environ, {aligned_reader.DUMP_VARIABLE: path}):
                self._reader()(
                    taps=self._taps(spans),
                    residual=torch.zeros(4, _RICKY_HIDDEN),
                    forward_batch=_extend(slots=[1], prefix_lens=[0], counts=[4]),
                    rows=None,
                )
            with open(path) as file:
                records = [json.loads(line) for line in file]

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["rid"], "rid-1")
        self.assertEqual(records[0]["encoder"], 0)
        self.assertEqual(records[0]["positions"], [1, 2])
        self.assertEqual(
            [tap[0] for tap in records[0]["taps"]], [_tap(1, 1), _tap(1, 2)]
        )


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
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "franky.json")
            with open(path, "w") as file:
                json.dump(
                    {
                        "encoder": {
                            "type": "singular_normed",
                            "layers": [16],
                            "output_size": _OUTPUT_SIZE,
                        },
                        "reader": {
                            "type": "aligned",
                            "layers": [8],
                            "output_scale": 3.5,
                        },
                        "injector": {
                            "sandy": {"type": "residual"},
                            "ricky": {"type": "residual"},
                        },
                        "tap_page_size": 64,
                    },
                    file,
                )
            config = FrankyConfig.load(path)

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
                type=ReaderConfig,
            )


if __name__ == "__main__":
    unittest.main()
