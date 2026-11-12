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

from sglang.srt.franky.cache.tap_cache import TAP_DTYPE, FrankyTapCache
from sglang.srt.franky.cache.tap_reading import FrankyTapReading
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


def _taps_read(position, reading):
    """The rule, token by token, from its definition."""
    token = position + 1
    if token < reading.generation_start:
        return 0
    if reading.block_size is None:
        return min(reading.generation_start, reading.tap_len)
    blocks = (token - reading.generation_start) // reading.block_size
    block_start = reading.generation_start + blocks * reading.block_size
    return min(block_start, reading.tap_len)


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
        readings = {
            1: FrankyTapReading(tap_len=70, generation_start=70),
            2: FrankyTapReading(tap_len=40, generation_start=40),
            3: FrankyTapReading(tap_len=30, generation_start=30),
            4: FrankyTapReading(tap_len=0, generation_start=0),
            5: FrankyTapReading(tap_len=200, generation_start=200),
            6: FrankyTapReading(tap_len=200, generation_start=40, block_size=32),
        }
        prefix_lens, extend_lens = [0, 30, 35, 0, 120, 30], [70, 50, 9, 5, 64, 175]
        self.tap_cache.set_readings(
            req_pool_indices=list(readings), readings=list(readings.values())
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.tensor(list(readings), device="cuda"),
            req_pool_indices_cpu=torch.tensor(list(readings)),
            extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32).cuda(),
            extend_prefix_lens_cpu=prefix_lens,
            extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32).cuda(),
            extend_seq_lens_cpu=extend_lens,
        )
        rows, num_taps = [], []
        for (row, reading), prefix_len, extend_len in zip(
            readings.items(), prefix_lens, extend_lens
        ):
            rows += [row] * extend_len
            num_taps += [
                _taps_read(position, reading)
                for position in range(prefix_len, prefix_len + extend_len)
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
        self.tap_cache.set_readings(
            req_pool_indices=[1, 2, 3, 4],
            readings=[
                FrankyTapReading(tap_len=tap_len, generation_start=tap_len)
                for tap_len in (70, 40, 0, 12)
            ],
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
