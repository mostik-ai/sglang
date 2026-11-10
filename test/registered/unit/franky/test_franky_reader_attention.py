"""Ricky's reader attention runs on FlashAttention's paged KV cache. For any mix
of requests it must read exactly the taps the reader rule allows: token ``p`` of
a request with ``tap_len`` taps reads taps ``0 .. min(p, tap_len - 1)``. That
holds for tokens past the last tap, requests without taps, padded tokens, and a
decode CUDA graph replayed with other requests.
"""

import unittest
from types import SimpleNamespace

import torch

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


def _reference_token_by_token(*, tap_cache, queries, rows, positions):
    taps = tap_cache._buffers[0].float()
    out = torch.zeros(queries.shape, dtype=torch.float32, device="cuda")
    has_taps = torch.zeros(queries.shape[0], device="cuda")
    for token, (row, position) in enumerate(zip(rows, positions)):
        last_tap = min(position, int(tap_cache._tap_lens[row]) - 1)
        if last_tap < 0:
            continue
        values = taps[tap_cache._req_to_token[row, : last_tap + 1].long()]
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

    def _assert_matches(self, *, queries, attended, has_taps, rows, positions):
        expected, expected_has_taps = _reference_token_by_token(
            tap_cache=self.tap_cache, queries=queries, rows=rows, positions=positions
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
        # chunk inside its taps.
        tap_lens = {1: 70, 2: 40, 3: 30, 4: 0, 5: 200}
        prefix_lens, extend_lens = [0, 30, 35, 0, 120], [70, 50, 9, 5, 64]
        self.tap_cache.set_tap_lens(
            req_pool_indices=list(tap_lens), tap_lens=list(tap_lens.values())
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.tensor(list(tap_lens), device="cuda"),
            extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32).cuda(),
            extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32).cuda(),
            extend_seq_lens_cpu=extend_lens,
        )
        rows, positions = [], []
        for row, prefix_len, extend_len in zip(tap_lens, prefix_lens, extend_lens):
            rows += [row] * extend_len
            positions += range(prefix_len, prefix_len + extend_len)
        queries, attended, has_taps = self._run(batch=batch, num_tokens=len(rows) + 6)
        self._assert_matches(
            queries=queries,
            attended=attended,
            has_taps=has_taps,
            rows=rows,
            positions=positions,
        )

    def test_a_decode_cuda_graph_replays_with_other_requests(self):
        self.tap_cache.set_tap_lens(
            req_pool_indices=[1, 2, 3, 4], tap_lens=[70, 40, 0, 12]
        )
        req_pool_indices = torch.tensor([1, 2, 3], device="cuda")
        seq_lens = torch.tensor([90, 20, 5], device="cuda")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
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
        seq_lens.copy_(torch.tensor([3, 200, 9]))
        queries.copy_(torch.randn_like(queries))
        graph.replay()
        torch.cuda.synchronize()

        self._assert_matches(
            queries=queries,
            attended=attended,
            has_taps=has_taps,
            rows=[4, 1, 3],
            positions=[2, 199, 8],
        )


if __name__ == "__main__":
    unittest.main()
