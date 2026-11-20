"""Ricky's readers attend over taps that the attention-TP ranks split by stripe:
each rank attends over the taps it keeps, and the ranks merge their results by
log-sum-exp. A row must read exactly the taps the reader rule gives it, whatever
the split, and a row that reads none must get exactly zero.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.franky_reader_merge import PACK_COLUMNS
from sglang.srt.franky.cache.generated_spans import (
    FrankyGeneratedSpans,
    GeneratedSpan,
)
from sglang.srt.franky.cache.tap_buffers import TAP_DTYPE, TapBuffers
from sglang.srt.franky.cache.tap_request_table import TapRequestTable
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.franky.parts.reader import tap_attention, tap_merge
from sglang.srt.franky.parts.reader.current_tap_reading import CurrentTapReading
from sglang.srt.franky.parts.reader.tap_attention import TapAttention
from sglang.srt.franky.parts.reader.tap_merge import TapMerge
from sglang.srt.franky.parts.reader.tap_reading import TapReading
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PAGE_SIZE = 4
_TAP_SIZE = 8
_NUM_QUERIES = 2
_NUM_ROWS = 4
_MAX_STRIPES = 4
_FREE_PAGES = {1: [3, 1, 4, 5], 2: [2, 6], 3: []}
# Row 1's earlier answer, tokens 2 and 3, reads 5 taps; its new answer reads 13.
# Row 2 reads 3 taps, all of them in rank 0's first page; row 3 reads none.
_SPANS = {
    1: FrankyGeneratedSpans(
        spans=(
            GeneratedSpan(start=2, sandy_start=5, end=4),
            GeneratedSpan(start=6, sandy_start=13),
        )
    ),
    2: FrankyGeneratedSpans(spans=(GeneratedSpan(start=3, sandy_start=3),)),
    3: None,
}
_NUM_INPUT_TOKENS = {1: 6, 2: 3, 3: 4}
_TAP_LENS = {1: 13, 2: 3, 3: 0}


def _flash_attention_on_cpu(
    *,
    q,
    k_cache,
    v_cache,
    page_table,
    cache_seqlens,
    softmax_scale,
    causal,
    out,
    return_softmax_lse=False,
):
    page_size = k_cache.shape[1]
    keys_of_slots = k_cache.reshape(-1, k_cache.shape[-1])
    lse = torch.full(q.shape[:1] + q.shape[2:3] + (1,), float("-inf"))
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
        lse[row, :, 0] = torch.logsumexp(scores, dim=-1)
    return (out, lse) if return_softmax_lse else out


def _pack_on_cpu(out, lse, local_taps):
    rows, queries, size = out.shape
    reads = (local_taps > 0)[:, None]
    packed = out.new_zeros((rows, queries, size + PACK_COLUMNS))
    packed[..., :size] = torch.where(reads[..., None], out, 0)
    packed.view(torch.float32)[..., size // 2] = torch.where(reads, lse, float("-inf"))
    return packed


def _merge_on_cpu(gathered, *, out, lse_is_base_e):
    size = gathered.shape[-1] - PACK_COLUMNS
    lse = gathered.view(torch.float32)[..., size // 2]
    lse = lse if lse_is_base_e else lse * torch.log(torch.tensor(2.0))
    weights = torch.softmax(lse, dim=0).nan_to_num(0.0)
    out.copy_((weights[..., None] * gathered[..., :size].float()).sum(dim=0))


class _Ranks:
    """The attention-TP group of ranks that run one after the other here: a
    gather hands each rank's place the last input that rank gave.
    """

    def __init__(self, world_size):
        self.world_size = world_size
        self.inputs = {}

    def group(self, rank):
        return SimpleNamespace(
            world_size=self.world_size,
            all_gather=lambda input, dim: self._gather_last_dim(rank, input),
        )

    def _gather_last_dim(self, rank, input):
        # As GroupCoordinator.all_gather along the last dim.
        output = input.new_empty((self.world_size * input.shape[0], *input.shape[1:]))
        self._gather(rank, output, input)
        return (
            output.view(self.world_size, *input.shape)
            .movedim(0, -2)
            .reshape(*input.shape[:-1], self.world_size * input.shape[-1])
        )

    def _gather(self, rank, output, input):
        self.inputs[rank] = input.clone()
        places = output.view(self.world_size, *input.shape)
        for other in range(self.world_size):
            places[other] = self.inputs.get(other, torch.zeros_like(input))


def _taps():
    torch.manual_seed(0)
    return {
        row: torch.randn(tap_len, _TAP_SIZE, dtype=TAP_DTYPE)
        for row, tap_len in _TAP_LENS.items()
    }


def _pages(row, *, num_ranks):
    stripe_size = num_ranks * _PAGE_SIZE
    num_stripes = -(-_TAP_LENS[row] // stripe_size)
    return _FREE_PAGES[row][:num_stripes] or None


def _rank_pages(taps, stripes):
    buffers = TapBuffers(
        encoder_ids=(0,),
        output_size=_TAP_SIZE,
        page_size=_PAGE_SIZE,
        num_pages=8,
        device="cpu",
    )
    for row, row_taps in taps.items():
        for tap in range(len(row_taps)):
            page = tap // _PAGE_SIZE
            if page % stripes.num_ranks != stripes.rank:
                continue
            local_page = _pages(row, num_ranks=stripes.num_ranks)[
                page // stripes.num_ranks
            ]
            buffers.write(
                encoder_id=0,
                taps=row_taps[tap : tap + 1],
                slots=torch.tensor([local_page * _PAGE_SIZE + tap % _PAGE_SIZE]),
            )
    return buffers.pages(0)


def _request_table(*, num_ranks):
    request_table = TapRequestTable(
        num_slots=_NUM_ROWS, max_stripes=_MAX_STRIPES, device="cpu"
    )
    rows = list(_SPANS)
    pages = [_pages(row, num_ranks=num_ranks) for row in rows]
    request_table.bind(
        req_pool_indices=rows,
        pages=[
            None if row_pages is None else torch.tensor(row_pages)
            for row_pages in pages
        ],
        spans=[_SPANS[row] for row in rows],
        num_input_tokens=[_NUM_INPUT_TOKENS[row] for row in rows],
    )
    return request_table


def _extend_batch():
    rows, prefix_lens, extend_lens = [1, 2, 3], [0, 0, 1], [6, 3, 3]
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.tensor(rows),
        req_pool_indices_cpu=torch.tensor(rows),
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32),
        extend_prefix_lens_cpu=prefix_lens,
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
        extend_seq_lens_cpu=extend_lens,
    )


_EXTEND_ROWS = [1] * 6 + [2] * 3 + [3] * 3
_EXTEND_TAPS = [0, 5, 5, 0, 0, 13, 0, 0, 3, 0, 0, 0]


def _decode_batch():
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE, req_pool_indices=torch.tensor([2, 1, 3])
    )


_DECODE_ROWS = [2, 1, 3]
_DECODE_TAPS = [3, 13, 0]


def _reference(queries, taps, *, rows, num_taps):
    expected = torch.zeros(queries.shape, dtype=torch.float32)
    for token, (row, num_read) in enumerate(zip(rows, num_taps)):
        if num_read == 0:
            continue
        values = taps[row][:num_read].float()
        weights = torch.softmax(
            queries[token].float() @ values.T * _TAP_SIZE**-0.5, dim=-1
        )
        expected[token] = weights @ values
    return expected


class TestTapReading(CustomTestCase):
    def _reading(self, batch, *, rank=0, num_ranks=2):
        return TapReading.of(
            batch,
            request_table=_request_table(num_ranks=num_ranks),
            stripes=TapStripes(num_ranks=num_ranks, rank=rank),
            page_size=_PAGE_SIZE,
            mask_dtype=torch.float32,
        )

    def test_an_extend_reading_gives_each_row_its_share_of_its_taps(self):
        for rank, local_taps in (
            (0, [0, 4, 4, 0, 0, 8, 0, 0, 3, 0, 0, 0]),
            (1, [0, 1, 1, 0, 0, 5, 0, 0, 0, 0, 0, 0]),
        ):
            with self.subTest(rank=rank):
                reading = self._reading(_extend_batch(), rank=rank)

                self.assertEqual(reading.num_rows, 12)
                self.assertEqual(reading.local_taps[:12].tolist(), local_taps)
                self.assertEqual(
                    (~reading.no_taps).tolist(),
                    [taps > 0 for taps in _EXTEND_TAPS] + [False],
                )
                self.assertEqual(
                    reading.has_taps[:, 0].tolist(),
                    [float(taps > 0) for taps in _EXTEND_TAPS] + [0.0],
                )
                self.assertEqual(
                    reading.page_table[[0, 6, 9]].tolist(), [[3, 1], [2, 0], [0, 0]]
                )
                self.assertEqual(reading.row_blocks.shape, (3, 3))

    def test_a_decode_reading_reads_each_rows_decoded_block(self):
        for rank, local_taps in ((0, [3, 8, 0]), (1, [0, 5, 0])):
            with self.subTest(rank=rank):
                reading = self._reading(_decode_batch(), rank=rank)

                self.assertEqual(reading.num_rows, 3)
                self.assertEqual(reading.local_taps[:3].tolist(), local_taps)
                self.assertEqual(
                    (~reading.no_taps).tolist(), [True, True, False, False]
                )
                self.assertIsNone(reading.row_blocks)
                self.assertEqual(
                    reading.page_table[:, :2].tolist(), [[2, 0], [3, 1], [0, 0]]
                )

    def test_a_selection_reads_what_its_rows_read_in_the_forward(self):
        reading = self._reading(_extend_batch(), rank=1)

        some = reading.select(torch.tensor([5, 8, 12]))

        self.assertEqual(some.num_rows, 3)
        self.assertEqual(some.local_taps.tolist(), [5, 0, 0])
        self.assertEqual((~some.no_taps).tolist(), [True, True, False])
        self.assertEqual(some.page_table.tolist(), [[3, 1], [2, 0], [0, 0]])

    def test_an_index_past_the_last_row_reads_nothing_without_padding(self):
        reading = self._reading(_extend_batch(), rank=0)

        some = reading.select(torch.tensor([5, 12]))

        self.assertEqual((~some.no_taps).tolist(), [True, False])
        self.assertEqual(some.local_taps.tolist(), [8, 0])


class TestTapAttention(CustomTestCase):
    def setUp(self):
        super().setUp()
        for module, name, stand_in in (
            (tap_attention, "flash_attn_with_kvcache", _flash_attention_on_cpu),
            (tap_merge, "franky_reader_pack", _pack_on_cpu),
            (tap_merge, "franky_reader_merge", _merge_on_cpu),
        ):
            stand_in_patch = patch.object(module, name, stand_in)
            stand_in_patch.start()
            self.addCleanup(stand_in_patch.stop)
        self.taps = _taps()

    def _attention(self, *, num_ranks, rank, group):
        stripes = TapStripes(num_ranks=num_ranks, rank=rank)
        return TapAttention(
            pages=_rank_pages(self.taps, stripes),
            reading=CurrentTapReading(
                request_table=_request_table(num_ranks=num_ranks),
                stripes=stripes,
                page_size=_PAGE_SIZE,
            ),
            builds_reading=True,
            merge=None if group is None else TapMerge(group=group, exchange=None),
            workspace=None,
        )

    def _attend(self, attention, queries, *, batch, padded_rows):
        attended = torch.full(
            (padded_rows, _NUM_QUERIES, _TAP_SIZE), float("nan"), dtype=TAP_DTYPE
        )
        has_taps = torch.full((padded_rows, 1), float("nan"))
        attention.attend(queries, attended, has_taps, forward_batch=batch, rows=None)
        return attended, has_taps

    def _attend_on_every_rank(self, queries, *, num_ranks, batch, padded_rows):
        ranks = _Ranks(num_ranks)
        attentions = [
            self._attention(num_ranks=num_ranks, rank=rank, group=ranks.group(rank))
            for rank in range(num_ranks)
        ]
        for attention in attentions:
            self._attend(attention, queries, batch=batch, padded_rows=padded_rows)
        return [
            self._attend(attention, queries, batch=batch, padded_rows=padded_rows)
            for attention in attentions
        ]

    def _assert_reads(self, attended, has_taps, queries, *, rows, num_taps):
        expected = _reference(
            queries[: len(rows)], self.taps, rows=rows, num_taps=num_taps
        )
        num_rows = len(rows)
        torch.testing.assert_close(
            attended[:num_rows].float(), expected, atol=2e-2, rtol=2e-2
        )
        self.assertEqual(
            has_taps[:num_rows, 0].tolist(), [float(taps > 0) for taps in num_taps]
        )
        self.assertTrue(torch.all(attended[num_rows:] == 0))
        self.assertTrue(torch.all(has_taps[num_rows:] == 0))

    def test_one_rank_reads_what_the_rule_allows(self):
        for batch, rows, num_taps in (
            (_extend_batch(), _EXTEND_ROWS, _EXTEND_TAPS),
            (_decode_batch(), _DECODE_ROWS, _DECODE_TAPS),
        ):
            with self.subTest(mode=batch.forward_mode):
                queries = torch.randn(
                    len(rows) + 2, _NUM_QUERIES, _TAP_SIZE, dtype=TAP_DTYPE
                )
                attention = self._attention(num_ranks=1, rank=0, group=None)

                attended, has_taps = self._attend(
                    attention, queries, batch=batch, padded_rows=len(rows) + 2
                )

                self._assert_reads(
                    attended, has_taps, queries, rows=rows, num_taps=num_taps
                )

    def test_every_rank_merges_to_what_one_rank_reads(self):
        for batch, rows, num_taps in (
            (_extend_batch(), _EXTEND_ROWS, _EXTEND_TAPS),
            (_decode_batch(), _DECODE_ROWS, _DECODE_TAPS),
        ):
            for num_ranks in (2, 3):
                with self.subTest(mode=batch.forward_mode, num_ranks=num_ranks):
                    queries = torch.randn(
                        len(rows) + 2, _NUM_QUERIES, _TAP_SIZE, dtype=TAP_DTYPE
                    )

                    results = self._attend_on_every_rank(
                        queries,
                        num_ranks=num_ranks,
                        batch=batch,
                        padded_rows=len(rows) + 2,
                    )

                    for attended, has_taps in results:
                        self._assert_reads(
                            attended, has_taps, queries, rows=rows, num_taps=num_taps
                        )

    def test_ranks_merge_rows_by_their_log_sum_exp(self):
        out = torch.randn(2, 1, _TAP_SIZE, dtype=TAP_DTYPE)
        lse = torch.tensor([[0.5], [float("-inf")]])
        other_out = torch.randn(2, 1, _TAP_SIZE, dtype=TAP_DTYPE)
        other_lse = torch.tensor([[1.5], [float("-inf")]])
        local_taps = torch.tensor([3, 0], dtype=torch.int32)
        no_taps = torch.tensor([False, True])
        ranks = _Ranks(2)
        merges = [TapMerge(group=ranks.group(rank), exchange=None) for rank in range(2)]
        merged = torch.full_like(out, float("nan"))
        has_taps = torch.full((2, 1), float("nan"))
        merges[1].merge(
            other_out,
            other_lse,
            local_taps,
            no_taps,
            out=torch.empty_like(out),
            has_taps=torch.empty_like(has_taps),
            lse_is_base_e=True,
        )

        merges[0].merge(
            out,
            lse,
            local_taps,
            no_taps,
            out=merged,
            has_taps=has_taps,
            lse_is_base_e=True,
        )

        weights = torch.softmax(torch.tensor([0.5, 1.5]), dim=0)
        expected = weights[0] * out[0].float() + weights[1] * other_out[0].float()
        torch.testing.assert_close(merged[0].float(), expected, atol=1e-2, rtol=1e-2)
        self.assertTrue(torch.all(merged[1] == 0))
        self.assertEqual(has_taps[:, 0].tolist(), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
