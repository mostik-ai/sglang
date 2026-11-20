"""Correctness test for the Franky readers' one-kernel merge across ranks.

Every rank stores its rows into every rank's symmetric buffer over NVLink P2P
and merges them by their log-sum-exp (``franky_reader_exchange_merge``). The
result, and whether any rank read taps for each row, must match merging the
NCCL-gathered rows in float32, for natural and base-2 log-sum-exp, rows some or
no rank reads taps for, eager and in a CUDA graph replayed with new inputs.

Usage::

    python test/registered/kernels/ops/communication/test_franky_reader_exchange.py --num-gpu 4
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.srt.franky.parts.reader.tap_merge import TapExchange, TapMerge
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=40, stage="extra-b", runner_config="8-gpu-h200")

TAP_SIZE = 256
MAX_ROWS = 1024
REPEATS = 4


@cache_once
def _world() -> ps.GroupCoordinator:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=ps._WORLD)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    return ps._WORLD


@cache_once
def _merge(num_queries: int) -> TapMerge:
    merge = TapMerge.build(
        group=_world(), max_rows=MAX_ROWS, num_queries=num_queries, tap_size=TAP_SIZE
    )
    assert merge.exchange is not None, "the test GPUs must reach each other over P2P"
    return merge


def _rank_inputs(rows: int, num_queries: int, *, seed: int):
    """This rank's results: some rows read no taps here, and row 0 reads none on
    any rank."""
    rank = _world().rank_in_group
    generator = torch.Generator(device="cuda").manual_seed(seed * 101 + rank)
    local_out = torch.randn(
        (rows, num_queries, TAP_SIZE), generator=generator, device="cuda"
    ).to(torch.bfloat16)
    lse = torch.randn((rows, num_queries), generator=generator, device="cuda") * 4
    local_taps = torch.randint(
        1, 300, (rows,), generator=generator, device="cuda", dtype=torch.int32
    )
    local_taps[torch.rand(rows, generator=generator, device="cuda") < 0.3] = 0
    local_taps[0] = 0
    # A rank that reads no taps for a row leaves garbage there, as the kernels do.
    local_out[local_taps == 0] = float("nan")
    return local_out, lse, local_taps


def _reference(local_out, lse, local_taps, *, lse_is_base_e: bool):
    """Returns:
    The merged rows, and whether any rank read taps for each row.
    """
    group = _world()

    def gather(tensor):
        return group.all_gather(tensor.unsqueeze(0).contiguous(), dim=0)

    outs, lses, taps = gather(local_out).float(), gather(lse), gather(local_taps)
    reads = (taps > 0)[..., None]
    lses = torch.where(reads, lses, float("-inf"))
    if not lse_is_base_e:
        lses = lses * torch.log(torch.tensor(2.0, device="cuda"))
    weights = torch.softmax(lses, dim=0).nan_to_num(0.0)
    merged = (torch.where(reads[..., None], outs, 0.0) * weights[..., None]).sum(0)
    return merged, (taps > 0).any(dim=0)[:, None].float()


@pytest.mark.parametrize("lse_is_base_e", [True, False])
@pytest.mark.parametrize("num_queries", [1, 2])
@pytest.mark.parametrize("rows", [1, 3, 64, 257, MAX_ROWS])
@torch.inference_mode()
def test_exchange_merge_matches_gathered_merge(rows, num_queries, lse_is_base_e):
    merge = _merge(num_queries)
    for repeat in range(REPEATS):
        local_out, lse, local_taps = _rank_inputs(rows, num_queries, seed=repeat)
        expected, expected_has_taps = _reference(
            local_out, lse, local_taps, lse_is_base_e=lse_is_base_e
        )
        out = torch.full_like(local_out, float("nan"))
        has_taps = torch.full((rows, 1), float("nan"), device="cuda")
        merge.merge(
            local_out,
            lse,
            local_taps,
            out=out,
            has_taps=has_taps,
            lse_is_base_e=lse_is_base_e,
        )
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(has_taps, expected_has_taps)
        assert torch.all(out[0] == 0), "a row no rank reads taps for must be zeros"


@pytest.mark.parametrize("rows", [1, 64])
@torch.inference_mode()
def test_exchange_merge_replays_in_a_cuda_graph(rows):
    merge = _merge(1)
    local_out, lse, local_taps = _rank_inputs(rows, 1, seed=7)
    out = torch.empty_like(local_out)
    has_taps = torch.empty((rows, 1), dtype=torch.bfloat16, device="cuda")

    def step():
        merge.merge(
            local_out, lse, local_taps, out=out, has_taps=has_taps, lse_is_base_e=False
        )

    step()
    torch.cuda.synchronize()
    # Merges over one buffer need a collective between them, as Ricky's layers
    # run between readers; here the reference's gathers are that collective.
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for repeat in range(REPEATS):
        new_out, new_lse, new_taps = _rank_inputs(rows, 1, seed=20 + repeat)
        local_out.copy_(new_out)
        lse.copy_(new_lse)
        local_taps.copy_(new_taps)
        graph.replay()
        expected, expected_has_taps = _reference(
            local_out, lse, local_taps, lse_is_base_e=False
        )
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(has_taps.float(), expected_has_taps)


def test_more_rows_than_a_slot_merge_over_nccl():
    merge = _merge(1)
    rows = merge.exchange.slot_rows + 5
    local_out, lse, local_taps = _rank_inputs(rows, 1, seed=3)
    out = torch.empty_like(local_out)
    has_taps = torch.empty((rows, 1), device="cuda")
    with torch.inference_mode():
        merge.merge(
            local_out, lse, local_taps, out=out, has_taps=has_taps, lse_is_base_e=True
        )
        expected, expected_has_taps = _reference(
            local_out, lse, local_taps, lse_is_base_e=True
        )
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(has_taps, expected_has_taps)


def test_exchange_is_built_on_one_node():
    assert isinstance(_merge(1).exchange, TapExchange)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4))
