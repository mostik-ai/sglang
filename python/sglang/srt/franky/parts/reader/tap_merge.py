from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch
import torch.distributed._symmetric_memory as symm_mem

from sglang.kernels.ops.attention.franky_reader_merge import (
    PACK_COLUMNS,
    franky_reader_merge,
    franky_reader_pack,
)
from sglang.kernels.ops.communication.franky_reader_exchange import (
    MAX_PROGRAMS,
    SIGNAL_PAD_BYTES_PER_RANK,
    franky_reader_exchange_merge,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)


class TapExchange(msgspec.Struct, frozen=True, kw_only=True):
    """The ranks' one-kernel merge within a node: each rank's symmetric buffer,
    two halves of one slot a rank, and this rank's merges so far per kernel
    program.

    Attributes:
        buffer: this rank's buffer, kept alive for its peers.
        handle: the buffer's rendezvous handle: every rank's buffer and signal
            pad.
        epochs: [MAX_PROGRAMS] int32, the merges each program ran on this rank.
        slot_rows: the query rows one slot holds.
    """

    buffer: torch.Tensor
    handle: Any
    epochs: torch.Tensor
    slot_rows: int

    @staticmethod
    def build(*, group: GroupCoordinator, slot_rows: int, row_size: int) -> TapExchange:
        """Collective: every rank of ``group``, on one node, builds it at the same
        point, outside CUDA graph capture.
        """
        symm_mem.set_signal_pad_size(
            max(
                symm_mem.get_signal_pad_size(),
                SIGNAL_PAD_BYTES_PER_RANK * group.world_size,
            )
        )
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        with torch.inference_mode(False), torch.no_grad():
            buffer = symm_mem.empty(
                (2 * group.world_size * slot_rows, row_size),
                dtype=torch.bfloat16,
                device=device,
            )
        return TapExchange(
            buffer=buffer,
            handle=symm_mem.rendezvous(buffer, group=group.device_group),
            epochs=torch.zeros(MAX_PROGRAMS, dtype=torch.int32, device=device),
            slot_rows=slot_rows,
        )

    def holds(self, num_query_rows: int) -> bool:
        return num_query_rows <= self.slot_rows

    def merge(
        self,
        local_out: torch.Tensor,
        lse: torch.Tensor,
        local_taps: torch.Tensor,
        no_taps: torch.Tensor,
        *,
        out: torch.Tensor,
        has_taps: torch.Tensor,
        lse_is_base_e: bool,
    ) -> None:
        franky_reader_exchange_merge(
            local_out,
            lse,
            local_taps,
            no_taps,
            out=out,
            has_taps=has_taps,
            buffer_ptrs=self.handle.buffer_ptrs_dev,
            signal_pad_ptrs=self.handle.signal_pad_ptrs_dev,
            epochs=self.epochs,
            slot_rows=self.slot_rows,
            rank=self.handle.rank,
            world_size=self.handle.world_size,
            lse_is_base_e=lse_is_base_e,
        )


class TapMerge:
    """Merges the results of the attention-TP ranks that split a reader's taps
    by their log-sum-exp, through ``TapExchange`` within a node. Across nodes,
    and for more rows than a slot holds, NCCL gathers the packed rows and a
    second kernel merges them.
    """

    def __init__(
        self, *, group: GroupCoordinator, exchange: Optional[TapExchange]
    ) -> None:
        self.group = group
        self.exchange = exchange

    @staticmethod
    def build(
        *,
        group: GroupCoordinator,
        max_rows: int,
        num_queries: int,
        tap_size: int,
        nnodes: int,
    ) -> TapMerge:
        """Collective: every rank of ``group`` builds it at the same point,
        outside CUDA graph capture; every rank takes the same path.

        Args:
            max_rows: the most rows one merge takes through the exchange.
            nnodes: the nodes of the deployment.
        """
        # The node probe is an all-reduce that crashes some EP setups; one node
        # needs none.
        if nnodes > 1 and not all(in_the_same_node_as(group.cpu_group, source_rank=0)):
            logger.info("Franky readers merge over NCCL: attention TP spans nodes")
            return TapMerge(group=group, exchange=None)
        exchange = TapExchange.build(
            group=group,
            slot_rows=max_rows * num_queries,
            row_size=tap_size + PACK_COLUMNS,
        )
        return TapMerge(group=group, exchange=exchange)

    def merge(
        self,
        local_out: torch.Tensor,
        lse: torch.Tensor,
        local_taps: torch.Tensor,
        no_taps: torch.Tensor,
        *,
        out: torch.Tensor,
        has_taps: torch.Tensor,
        lse_is_base_e: bool,
    ) -> None:
        """Write into ``out`` each row's attention over the taps of every rank,
        zeros for the rows that read none, and into ``has_taps`` whether it
        reads any.

        Args:
            local_out: [rows, queries, tap_size] bfloat16, this rank's results.
            lse: [rows, queries] float32, their log-sum-exp.
            local_taps: [rows] int32, the taps each row reads on this rank.
            no_taps: [rows] bool, whether each row reads no taps on any rank.
            out: [rows, queries, tap_size] bfloat16.
            has_taps: [rows, 1].
        """
        rows, num_queries, _ = local_out.shape
        if self.exchange is not None and self.exchange.holds(rows * num_queries):
            self.exchange.merge(
                local_out,
                lse,
                local_taps,
                no_taps,
                out=out,
                has_taps=has_taps,
                lse_is_base_e=lse_is_base_e,
            )
            return
        packed = franky_reader_pack(local_out, lse, local_taps)
        row_size = packed.shape[-1]
        gathered = self.group.all_gather(
            packed.view(rows, num_queries * row_size), dim=-1
        )
        every_rank = gathered.view(
            rows, self.group.world_size, num_queries, row_size
        ).transpose(0, 1)
        franky_reader_merge(every_rank, out=out, lse_is_base_e=lse_is_base_e)
        has_taps.copy_(~no_taps[:, None])
