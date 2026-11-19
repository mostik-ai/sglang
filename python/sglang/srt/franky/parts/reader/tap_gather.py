from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    MultimemAllGatherState,
    all_gather_inner,
    create_state,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)


class TapGather:
    """Gathers every attention-TP rank's packed reader results. Within one node,
    each rank stores its rows into every rank's symmetric buffer in one kernel:
    one NVLink multicast store where the GPUs have multicast, else one P2P store
    per rank. It skips the gather's entry barrier: the layers between two readers
    run collectives, so no rank writes the next reader's rows while another still
    reads the last ones. Across nodes, and for more rows than the buffer holds,
    NCCL gathers them.
    """

    def __init__(
        self, *, group: GroupCoordinator, symmetric: Optional[MultimemAllGatherState]
    ) -> None:
        self.group = group
        self.symmetric = symmetric

    @staticmethod
    def build(*, group: GroupCoordinator, max_rows: int, row_size: int) -> TapGather:
        """Collective: every rank of ``group`` builds it at the same point,
        outside CUDA graph capture.

        Args:
            max_rows: the most rows a gather takes over NVLink.
            row_size: the bfloat16 elements of one packed row, a multiple of 8.
        """
        return TapGather(
            group=group,
            symmetric=_symmetric_state(group, max_rows=max_rows, row_size=row_size),
        )

    def gather(self, packed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            packed: [rows, queries, size] bfloat16, this rank's packed results.
        Returns:
            [ranks, rows, queries, size], every rank's.
        """
        rows, queries, size = packed.shape
        num_ranks = self.group.world_size
        if self.symmetric is not None and rows <= self.symmetric.max_token_num:
            gathered = all_gather_inner(
                self.symmetric,
                packed.view(rows, queries * size),
                tp_hidden_dim=queries * size * num_ranks,
                skip_entry_sync=True,
                safe=False,
            )
            return gathered.view(rows, num_ranks, queries, size).transpose(0, 1)
        gathered = packed.new_empty((num_ranks * rows, queries, size))
        self.group.all_gather_into_tensor(gathered, packed)
        return gathered.view(num_ranks, rows, queries, size)


def _symmetric_state(
    group: GroupCoordinator, *, max_rows: int, row_size: int
) -> Optional[MultimemAllGatherState]:
    if not all(in_the_same_node_as(group.cpu_group, source_rank=0)):
        logger.warning("Franky readers gather with NCCL: the ranks span nodes.")
        return None
    try:
        state = create_state(
            group=group.device_group,
            rank_in_group=group.rank_in_group,
            max_tokens=max_rows,
            hidden_size=row_size * group.world_size,
        )
    except Exception as error:
        logger.warning(
            "Franky readers gather with NCCL: no symmetric memory (%s).", error
        )
        return None
    logger.info(
        "Franky readers gather over NVLink %s.",
        "multicast" if state.symm_mem_hdl.multicast_ptr != 0 else "P2P stores",
    )
    return state
