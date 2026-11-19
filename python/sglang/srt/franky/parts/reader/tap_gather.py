from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    MultimemAllGatherer,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


class TapGather:
    """Gathers every attention-TP rank's packed reader results with SGLang's
    symmetric-memory all-gather: within one node each rank stores its rows into
    every rank's buffer in one kernel, over NVLink multicast where the GPUs have
    it and P2P stores elsewhere. It skips the gather's entry barrier: the layers
    between two readers run collectives, so no rank writes the next reader's rows
    while another still reads the last ones. Across nodes, and for more rows than
    the buffer holds, NCCL gathers them.
    """

    def __init__(self, *, gatherer: MultimemAllGatherer, num_ranks: int) -> None:
        self.gatherer = gatherer
        self.num_ranks = num_ranks

    @staticmethod
    def build(*, group: GroupCoordinator, max_rows: int, row_size: int) -> TapGather:
        """Collective: every rank of ``group`` builds it at the same point,
        outside CUDA graph capture.

        Args:
            max_rows: the most rows a gather takes over NVLink.
            row_size: the bfloat16 elements of one packed row, a multiple of 8.
        """
        gatherer = MultimemAllGatherer(
            max_rows, skip_entry_sync=True, group=group, p2p=True
        )
        gatherer.build(hidden_size=row_size * group.world_size)
        return TapGather(gatherer=gatherer, num_ranks=group.world_size)

    def gather(self, packed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            packed: [rows, queries, size] bfloat16, this rank's packed results.
        Returns:
            [ranks, rows, queries, size], every rank's.
        """
        rows, queries, size = packed.shape
        gathered = self.gatherer(packed.view(rows, queries * size))
        return gathered.view(rows, self.num_ranks, queries, size).transpose(0, 1)
