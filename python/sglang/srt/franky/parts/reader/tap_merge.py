from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch

from sglang.kernels.ops.attention.franky_reader_merge import (
    EXCHANGE_MAX_PROGRAMS,
    PACK_COLUMNS,
    franky_reader_exchange_merge,
    franky_reader_merge,
    franky_reader_pack,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)


class TapExchange(msgspec.Struct, frozen=True, kw_only=True):
    """What the ranks' one-kernel merge shares: each rank's symmetric buffer of
    packed reader rows, one slot a rank, and the barrier's flags and epochs.

    Attributes:
        buffer: this rank's buffer, kept alive for its peers.
        buffer_handle: the buffer's rendezvous handle, with every rank's pointers.
        flags: this rank's barrier flags, kept alive for its peers.
        flag_handle: the flags' rendezvous handle.
        epochs: this rank's barriers so far, per program of the kernel.
        slot_rows: the query rows one slot holds.
    """

    buffer: torch.Tensor
    buffer_handle: Any
    flags: torch.Tensor
    flag_handle: Any
    epochs: torch.Tensor
    slot_rows: int

    @staticmethod
    def build(
        *, group: GroupCoordinator, max_query_rows: int, row_size: int
    ) -> Optional[TapExchange]:
        """Collective: every rank of ``group`` builds it at the same point,
        outside CUDA graph capture.

        Returns:
            None where the ranks cannot store into each other's memory: they
            span nodes, or symmetric memory is unavailable.
        """
        # Lazy import: torch's symmetric memory loads only on GPU hosts that
        # split taps.
        import torch.distributed._symmetric_memory as symm_mem

        if _spans_nodes(group):
            logger.info("Franky readers merge over NCCL: attention TP spans nodes")
            return None
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        try:
            with torch.inference_mode(False), torch.no_grad():
                buffer = symm_mem.empty(
                    (group.world_size * max_query_rows, row_size),
                    dtype=torch.bfloat16,
                    device=device,
                )
                flags = symm_mem.empty(
                    (EXCHANGE_MAX_PROGRAMS * group.world_size,),
                    dtype=torch.int32,
                    device=device,
                )
            # Zeroed before the rendezvous, which no rank leaves before every
            # rank reached it, so no peer signals into stale flags.
            flags.zero_()
            torch.cuda.synchronize(device)
            buffer_handle = symm_mem.rendezvous(buffer, group=group.device_group)
            flag_handle = symm_mem.rendezvous(flags, group=group.device_group)
        except Exception as error:  # no P2P between these GPUs
            logger.warning("Franky readers merge over NCCL: %s", error)
            return None
        return TapExchange(
            buffer=buffer,
            buffer_handle=buffer_handle,
            flags=flags,
            flag_handle=flag_handle,
            epochs=torch.zeros(EXCHANGE_MAX_PROGRAMS, dtype=torch.int32, device=device),
            slot_rows=max_query_rows,
        )


def _spans_nodes(group: GroupCoordinator) -> bool:
    # Lazy import avoids a module-load dependency on the distributed facade.
    from sglang.srt.distributed.parallel_state import in_the_same_node_as

    try:
        nnodes = get_parallel().nnodes
    except ValueError:  # unpublished: offline tools and tests probe the group
        nnodes = None
    # The probe is an all-reduce that crashes some EP setups; one node needs none.
    if nnodes == 1:
        return False
    return not all(in_the_same_node_as(group.cpu_group, source_rank=0))


class TapMerge:
    """Merges the results of the attention-TP ranks that split a reader's taps
    by their log-sum-exp. Within one node one kernel does it: every rank stores
    its rows into every rank's buffer over NVLink, then merges them. Across
    nodes, and for more rows than a slot holds, NCCL gathers the packed rows and
    a second kernel merges them.

    The kernel skips an entry barrier: the layers between two readers run
    collectives, so no rank stores the next reader's rows while another still
    merges the last ones.
    """

    def __init__(
        self, *, group: GroupCoordinator, exchange: Optional[TapExchange]
    ) -> None:
        self.group = group
        self.exchange = exchange

    @staticmethod
    def build(
        *, group: GroupCoordinator, max_rows: int, num_queries: int, tap_size: int
    ) -> TapMerge:
        """Collective: every rank of ``group`` builds it at the same point,
        outside CUDA graph capture.

        Args:
            max_rows: the most rows one merge takes through the exchange.
        """
        exchange = TapExchange.build(
            group=group,
            max_query_rows=max_rows * num_queries,
            row_size=tap_size + PACK_COLUMNS,
        )
        return TapMerge(group=group, exchange=exchange)

    def merge(
        self,
        local_out: torch.Tensor,
        lse: torch.Tensor,
        local_taps: torch.Tensor,
        *,
        out: torch.Tensor,
        has_taps: torch.Tensor,
        lse_is_base_e: bool,
    ) -> None:
        """Write into ``out`` each row's attention over the taps of every rank,
        and into ``has_taps`` whether any rank read taps for it; a row none did
        gets zeros.

        Args:
            local_out: [rows, queries, tap_size] bfloat16, this rank's results.
            lse: [rows, queries] float32, their log-sum-exp.
            local_taps: [rows] int32, the taps each row reads on this rank.
            out: [rows, queries, tap_size] bfloat16.
            has_taps: [rows, 1].
        """
        rows, num_queries, _ = local_out.shape
        exchange = self.exchange
        if exchange is not None and rows * num_queries <= exchange.slot_rows:
            franky_reader_exchange_merge(
                local_out,
                lse,
                local_taps,
                out=out,
                has_taps=has_taps,
                buffer_ptrs=exchange.buffer_handle.buffer_ptrs_dev,
                flag_ptrs=exchange.flag_handle.buffer_ptrs_dev,
                epochs=exchange.epochs,
                buffer_rows=exchange.slot_rows,
                rank=exchange.buffer_handle.rank,
                world_size=exchange.buffer_handle.world_size,
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
        has_taps.copy_(_reads_on_any_rank(every_rank, tap_size=out.shape[-1]))


def _reads_on_any_rank(every_rank: torch.Tensor, *, tap_size: int) -> torch.Tensor:
    """[rows, 1]: whether any rank's packed rows carry a finite log-sum-exp."""
    lse = every_rank.view(torch.float32)[..., tap_size // 2]
    return torch.isfinite(lse).any(dim=0).any(dim=-1, keepdim=True)
