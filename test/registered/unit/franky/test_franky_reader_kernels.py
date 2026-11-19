"""Ricky's readers attend over tap pages with the prefill kernel, TRT-LLM's decode
kernels on Blackwell and FlashAttention elsewhere. For any mix of requests each
must read exactly the taps the reader rule allows, give the log-sum-exp the rank
merge expects, and stay right in a decode CUDA graph replayed with other
requests.
"""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.franky.cache.generated_spans import (
    FrankyGeneratedSpans,
    GeneratedSpan,
)
from sglang.srt.franky.cache.tap_buffers import TAP_DTYPE, TapBuffers
from sglang.srt.franky.cache.tap_read_rows import TapReadRows
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.franky.parts.reader.tap_attention import (
    TapAttention,
    TapReadings,
    TrtllmWorkspace,
)
from sglang.srt.franky.parts.reader.tap_reading import TapReading
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

_PAGE_SIZE = 16
_TAP_SIZE = 256
_NUM_ROWS = 7
_CONTEXT = 256
_MAX_STRIPES = _CONTEXT // _PAGE_SIZE


def _taps_to_read(token, *, generation_start, block_size):
    if generation_start is None or token < generation_start:
        return 0
    if block_size is None:
        return generation_start
    blocks = (token - generation_start) // block_size
    return generation_start + blocks * block_size


def _spans(generation_start, block_size):
    if generation_start is None:
        return None
    return FrankyGeneratedSpans(
        spans=(GeneratedSpan(start=generation_start, sandy_start=generation_start),),
        block_size=block_size,
    )


class _Ranks:
    def __init__(self, world_size):
        self.world_size = world_size
        self.inputs = {}

    def group(self, rank):
        return SimpleNamespace(
            world_size=self.world_size,
            all_gather_into_tensor=lambda output, input: self._gather(
                rank, output, input
            ),
        )

    def _gather(self, rank, output, input):
        self.inputs[rank] = input.clone()
        places = output.view(self.world_size, *input.shape)
        for other in range(self.world_size):
            places[other] = self.inputs.get(other, torch.zeros_like(input))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestReaderKernels(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.taps = torch.randn(
            _NUM_ROWS, _CONTEXT, _TAP_SIZE, dtype=TAP_DTYPE, device="cuda"
        )
        self.slots = (
            torch.randperm(_NUM_ROWS * _MAX_STRIPES).view(_NUM_ROWS, _MAX_STRIPES) + 1
        )

    def _attention(self, *, stripes, group, read_rows):
        stripe_size = stripes.num_ranks * _PAGE_SIZE
        buffers = TapBuffers(
            encoder_ids=(0,),
            output_size=_TAP_SIZE,
            page_size=_PAGE_SIZE,
            num_pages=_NUM_ROWS * _MAX_STRIPES + 1,
            device="cuda",
        )
        pages = buffers.pages(0).view(-1, _PAGE_SIZE, _TAP_SIZE)
        for row in range(_NUM_ROWS):
            for page in range(_CONTEXT // _PAGE_SIZE):
                if page % stripes.num_ranks != stripes.rank:
                    continue
                slot = self.slots[row, page // stripes.num_ranks]
                pages[slot] = self.taps[
                    row, page * _PAGE_SIZE : (page + 1) * _PAGE_SIZE
                ]
        num_stripes = _CONTEXT // stripe_size
        read_rows.stripe_slots[:, :num_stripes] = self.slots[:, :num_stripes].to(
            device="cuda", dtype=torch.int32
        )
        return TapAttention(
            pages=buffers.pages(0),
            readings=TapReadings(
                read_rows=read_rows, stripes=stripes, page_size=_PAGE_SIZE
            ),
            starts_forward=True,
            group=group,
            workspace=TrtllmWorkspace.for_device("cuda"),
        )

    def _read_rows(self, spans, num_input_tokens):
        read_rows = TapReadRows(
            num_rows=_NUM_ROWS, max_stripes=_MAX_STRIPES, device="cuda"
        )
        rows = list(spans)
        read_rows.bind(
            req_pool_indices=rows,
            slots=[None] * len(rows),
            spans=[spans[row] for row in rows],
            num_input_tokens=[num_input_tokens[row] for row in rows],
        )
        return read_rows

    def _reference(self, queries, *, rows, num_taps):
        out = torch.zeros(queries.shape, dtype=torch.float32, device="cuda")
        lse = torch.full(
            queries.shape[:2], float("-inf"), dtype=torch.float32, device="cuda"
        )
        for token, (row, num_read) in enumerate(zip(rows, num_taps)):
            if num_read == 0:
                continue
            values = self.taps[row, :num_read].float()
            scores = queries[token].float() @ values.T * _TAP_SIZE**-0.5
            out[token] = torch.softmax(scores, dim=-1) @ values
            lse[token] = torch.logsumexp(scores, dim=-1)
        return out, lse

    def _extend(self):
        answers = {1: (70, None), 2: (40, None), 3: (30, None), 4: (None, None)}
        answers |= {5: (200, None), 6: (40, 32)}
        num_input_tokens = {1: 70, 2: 40, 3: 30, 4: 0, 5: 200, 6: 200}
        prefix_lens, extend_lens = [0, 30, 35, 0, 120, 30], [70, 50, 9, 5, 64, 175]
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.tensor(list(answers), device="cuda"),
            req_pool_indices_cpu=torch.tensor(list(answers)),
            extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32).cuda(),
            extend_prefix_lens_cpu=prefix_lens,
            extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32).cuda(),
            extend_seq_lens_cpu=extend_lens,
        )
        rows, num_taps = [], []
        for (row, (generation_start, block_size)), prefix_len, extend_len in zip(
            answers.items(), prefix_lens, extend_lens
        ):
            rows += [row] * extend_len
            num_taps += [
                _taps_to_read(
                    token, generation_start=generation_start, block_size=block_size
                )
                for token in range(prefix_len + 1, prefix_len + extend_len + 1)
            ]
        spans = {row: _spans(*answer) for row, answer in answers.items()}
        return batch, spans, num_input_tokens, rows, num_taps

    def _attend(self, attention, queries, *, batch, padded_rows):
        attended = torch.full(
            (padded_rows,) + queries.shape[1:],
            float("nan"),
            dtype=TAP_DTYPE,
            device="cuda",
        )
        has_taps = torch.full((padded_rows, 1), float("nan"), device="cuda")
        attention.attend(queries, attended, has_taps, forward_batch=batch, rows=None)
        return attended, has_taps

    def _assert_matches(self, attended, has_taps, queries, *, rows, num_taps):
        expected, _ = self._reference(queries, rows=rows, num_taps=num_taps)
        num_rows = len(rows)
        torch.testing.assert_close(
            attended[:num_rows].float(), expected[:num_rows], atol=2e-2, rtol=2e-2
        )
        self.assertEqual(
            has_taps[:num_rows, 0].tolist(), [float(taps > 0) for taps in num_taps]
        )
        self.assertTrue(torch.all(attended[num_rows:] == 0))
        self.assertTrue(torch.all(has_taps[num_rows:] == 0))

    def test_extend_reads_what_the_rule_allows(self):
        batch, spans, num_input_tokens, rows, num_taps = self._extend()
        for num_queries in (1, 3):
            with self.subTest(num_queries=num_queries):
                attention = self._attention(
                    stripes=TapStripes(num_ranks=1, rank=0),
                    group=None,
                    read_rows=self._read_rows(spans, num_input_tokens),
                )
                queries = torch.randn(
                    len(rows), num_queries, _TAP_SIZE, dtype=TAP_DTYPE, device="cuda"
                )

                attended, has_taps = self._attend(
                    attention, queries, batch=batch, padded_rows=len(rows) + 6
                )

                self._assert_matches(
                    attended, has_taps, queries, rows=rows, num_taps=num_taps
                )

    def test_each_kernels_log_sum_exp_is_in_the_base_the_merge_takes(self):
        batch, spans, num_input_tokens, rows, num_taps = self._extend()
        decode_rows = [1, 2, 5]
        for name, forward_batch, query_rows, query_taps in (
            ("prefill", batch, rows, num_taps),
            (
                "decode",
                SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    req_pool_indices=torch.tensor(decode_rows, device="cuda"),
                ),
                decode_rows,
                [num_input_tokens[row] for row in decode_rows],
            ),
        ):
            with self.subTest(kernel=name):
                attention = self._attention(
                    stripes=TapStripes(num_ranks=1, rank=0),
                    group=None,
                    read_rows=self._read_rows(spans, num_input_tokens),
                )
                queries = torch.randn(
                    len(query_rows), 1, _TAP_SIZE, dtype=TAP_DTYPE, device="cuda"
                )
                reading = attention.readings.start(
                    forward_batch,
                    padded_rows=len(query_rows),
                    mask_dtype=torch.float32,
                )
                out = torch.empty_like(queries)
                lse = torch.empty(queries.shape[:2], dtype=torch.float32, device="cuda")

                lse_is_base_e = attention._attend_locally(
                    queries, reading, out=out, lse=lse
                )

                expected_out, expected_lse = self._reference(
                    queries, rows=query_rows, num_taps=query_taps
                )
                if not lse_is_base_e:
                    expected_lse = expected_lse / math.log(2)
                reads = torch.tensor(query_taps, device="cuda") > 0
                torch.testing.assert_close(
                    out[reads].float(), expected_out[reads], atol=2e-2, rtol=2e-2
                )
                torch.testing.assert_close(
                    lse[reads], expected_lse[reads], atol=2e-2, rtol=2e-3
                )

    def test_ranks_merge_to_what_one_rank_reads(self):
        batch, spans, num_input_tokens, rows, num_taps = self._extend()
        decode = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.tensor([4, 1, 6, 5], device="cuda"),
        )
        decode_rows = [4, 1, 6, 5]
        decode_taps = [num_input_tokens[row] for row in decode_rows]
        for name, forward_batch, query_rows, query_taps in (
            ("extend", batch, rows, num_taps),
            ("decode", decode, decode_rows, decode_taps),
        ):
            for num_ranks in (2, 4):
                with self.subTest(mode=name, num_ranks=num_ranks):
                    ranks = _Ranks(num_ranks)
                    attentions = [
                        self._attention(
                            stripes=TapStripes(num_ranks=num_ranks, rank=rank),
                            group=ranks.group(rank),
                            read_rows=self._read_rows(spans, num_input_tokens),
                        )
                        for rank in range(num_ranks)
                    ]
                    queries = torch.randn(
                        len(query_rows), 1, _TAP_SIZE, dtype=TAP_DTYPE, device="cuda"
                    )
                    for attention in attentions:
                        self._attend(
                            attention,
                            queries,
                            batch=forward_batch,
                            padded_rows=len(query_rows),
                        )

                    for attention in attentions:
                        attended, has_taps = self._attend(
                            attention,
                            queries,
                            batch=forward_batch,
                            padded_rows=len(query_rows),
                        )
                        self._assert_matches(
                            attended,
                            has_taps,
                            queries,
                            rows=query_rows,
                            num_taps=query_taps,
                        )

    def test_a_decode_cuda_graph_replays_with_other_requests(self):
        spans = {1: _spans(70, None), 2: _spans(40, None), 3: None, 4: _spans(12, None)}
        num_input_tokens = {1: 70, 2: 40, 3: 0, 4: 12}
        attention = self._attention(
            stripes=TapStripes(num_ranks=1, rank=0),
            group=None,
            read_rows=self._read_rows(spans, num_input_tokens),
        )
        req_pool_indices = torch.tensor([1, 2, 3], device="cuda")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE, req_pool_indices=req_pool_indices
        )
        queries = torch.randn(3, 3, _TAP_SIZE, dtype=TAP_DTYPE, device="cuda")
        attended = torch.empty_like(queries)
        has_taps = torch.empty((3, 1), device="cuda")

        def step():
            attention.attend(
                queries, attended, has_taps, forward_batch=batch, rows=None
            )

        step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        req_pool_indices.copy_(torch.tensor([4, 1, 3]))
        queries.copy_(torch.randn_like(queries))
        graph.replay()
        torch.cuda.synchronize()

        self._assert_matches(
            attended, has_taps, queries, rows=[4, 1, 3], num_taps=[12, 70, 0]
        )

    def test_a_reading_names_each_rows_local_taps_on_the_device(self):
        batch, spans, num_input_tokens, rows, num_taps = self._extend()
        reading = TapReading.of(
            batch,
            read_rows=self._read_rows(spans, num_input_tokens),
            stripes=TapStripes(num_ranks=2, rank=1),
            page_size=_PAGE_SIZE,
            padded_rows=len(rows),
            mask_dtype=torch.float32,
        )

        stripes = TapStripes(num_ranks=2, rank=1)
        expected = stripes.local_taps(torch.tensor(num_taps), page_size=_PAGE_SIZE)
        self.assertEqual(
            reading.local_taps[: len(rows)].cpu().tolist(), expected.tolist()
        )
        self.assertEqual(reading.local_taps.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
