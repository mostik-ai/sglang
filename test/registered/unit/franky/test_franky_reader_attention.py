"""Ricky's reader attention runs on FlashAttention's paged KV cache. For any mix
of requests it must read exactly the taps the reader rule allows: a prompt token
reads up to its own tap, and a later token the taps before the block of the token
it predicts, never past the request's last tap. That holds for tokens past the
last tap, a chain's later blocks, requests without taps, padded tokens, and a
decode CUDA graph replayed with other requests.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.franky.cache.generated_spans import (
    FrankyGeneratedSpans,
    GeneratedSpan,
)
from sglang.srt.franky.cache.tap_cache import TAP_DTYPE, FrankyTapCache
from sglang.srt.franky.parts.reader.query.reader import (
    _attend_taps,
    attention_workspace,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

_PAGE_SIZE = 16
_NUM_QUERIES = 3
_CONTEXT = 256
_NUM_ROWS = 6


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
        spans=(GeneratedSpan(start=generation_start),), block_size=block_size
    )


def _reference_token_by_token(*, tap_cache, queries, rows, num_taps):
    taps = tap_cache._buffers[0].float()
    out = torch.zeros(queries.shape, dtype=torch.float32, device="cuda")
    has_taps = torch.zeros(queries.shape[0], device="cuda")
    for token, (row, num_read) in enumerate(zip(rows, num_taps)):
        if num_read == 0:
            continue
        values = taps[tap_cache._req_to_token[row, :num_read].long()]
        weights = torch.softmax(
            queries[token].float() @ values.T * queries.shape[-1] ** -0.5, dim=-1
        )
        out[token] = weights @ values
        has_taps[token] = 1
    return out, has_taps


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestReaderAttention(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        # Each row's slots are whole pages, scattered over the pool.
        num_pages = _NUM_ROWS * _CONTEXT // _PAGE_SIZE
        pages = torch.randperm(num_pages, device="cuda") + 1
        req_to_token = torch.zeros(
            (_NUM_ROWS + 1, _CONTEXT), dtype=torch.int32, device="cuda"
        )
        req_to_token[1:] = (
            pages.view(_NUM_ROWS, -1)[:, :, None] * _PAGE_SIZE
            + torch.arange(_PAGE_SIZE, device="cuda")
        ).view(_NUM_ROWS, _CONTEXT)
        self.tap_cache = FrankyTapCache(
            encoder_ids=(0,),
            output_size=256,
            page_size=_PAGE_SIZE,
            num_tokens=num_pages * _PAGE_SIZE + _PAGE_SIZE,
            req_to_token=req_to_token,
            device="cuda",
            attention_workspace=attention_workspace(device="cuda"),
        )
        self.tap_cache._buffers[0].normal_()

    def _run(self, *, batch, num_tokens):
        queries = torch.randn(
            num_tokens, _NUM_QUERIES, 256, dtype=TAP_DTYPE, device="cuda"
        )
        attended = torch.full_like(queries, float("nan"))
        has_taps = torch.full((num_tokens, 1), float("nan"), device="cuda")
        _attend_taps(
            queries,
            attended,
            has_taps,
            tap_cache=self.tap_cache,
            encoder_id=0,
            forward_batch=batch,
        )
        return queries, attended, has_taps

    def _assert_matches(self, *, queries, attended, has_taps, rows, num_taps):
        expected, expected_has_taps = _reference_token_by_token(
            tap_cache=self.tap_cache, queries=queries, rows=rows, num_taps=num_taps
        )
        num_tokens = len(rows)
        torch.testing.assert_close(
            attended[:num_tokens].float(), expected[:num_tokens], atol=2e-2, rtol=2e-2
        )
        self.assertTrue(
            torch.equal(has_taps[:num_tokens, 0], expected_has_taps[:num_tokens])
        )
        # Padded tokens get exactly zero, so the reader adds nothing to them.
        self.assertTrue(torch.all(attended[num_tokens:] == 0))
        self.assertTrue(torch.all(has_taps[num_tokens:] == 0))

    def test_extend_reads_what_the_rule_allows(self):
        # Row 1 runs its whole prompt; row 2 crosses its last tap; row 3 runs only
        # past it, as a retracted request does; row 4 has no taps; row 5 runs a
        # chunk inside its taps; row 6 is block 6 of a chain whose prompt is 40
        # tokens, in blocks of 32, recomputed from inside its prompt.
        answers = {1: (70, None), 2: (40, None), 3: (30, None), 4: (None, None)}
        answers |= {5: (200, None), 6: (40, 32)}
        num_input_tokens = [70, 40, 30, 0, 200, 200]
        prefix_lens, extend_lens = [0, 30, 35, 0, 120, 30], [70, 50, 9, 5, 64, 175]
        self.tap_cache.set_generated_spans(
            req_pool_indices=list(answers),
            generated_spans=[_spans(*answer) for answer in answers.values()],
            num_input_tokens=num_input_tokens,
        )
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
        queries, attended, has_taps = self._run(batch=batch, num_tokens=len(rows) + 6)
        self._assert_matches(
            queries=queries,
            attended=attended,
            has_taps=has_taps,
            rows=rows,
            num_taps=num_taps,
        )

    def test_a_decode_cuda_graph_replays_with_other_requests(self):
        self.tap_cache.set_generated_spans(
            req_pool_indices=[1, 2, 3, 4],
            generated_spans=[
                _spans(70, None),
                _spans(40, None),
                None,
                _spans(12, None),
            ],
            num_input_tokens=[70, 40, 0, 12],
        )
        req_pool_indices = torch.tensor([1, 2, 3], device="cuda")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE, req_pool_indices=req_pool_indices
        )
        queries = torch.randn(3, _NUM_QUERIES, 256, dtype=TAP_DTYPE, device="cuda")
        attended = torch.empty_like(queries)
        has_taps = torch.empty((3, 1), device="cuda")

        def step():
            _attend_taps(
                queries,
                attended,
                has_taps,
                tap_cache=self.tap_cache,
                encoder_id=0,
                forward_batch=batch,
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
            queries=queries,
            attended=attended,
            has_taps=has_taps,
            rows=[4, 1, 3],
            num_taps=[12, 70, 0],
        )


if __name__ == "__main__":
    unittest.main()
