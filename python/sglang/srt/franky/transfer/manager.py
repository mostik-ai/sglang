from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import numpy.typing as npt
import torch
from typing_extensions import override

from sglang.srt.disaggregation.base.conn import KVArgs
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.franky.transfer.landed_pages import FrankyLandedPages
from sglang.srt.runtime_context import get_disagg, get_parallel

if TYPE_CHECKING:
    from sglang.srt.disaggregation.common.conn import PrefillServerInfo
    from sglang.srt.disaggregation.common.utils import TransferKVChunk
    from sglang.srt.disaggregation.nixl.conn import TransferInfo
    from sglang.srt.franky.cache.tap_buffers import TapBuffers
    from sglang.srt.server_args import ServerArgs

_AUX_BYTES = 8


class FrankyTapManager(NixlKVManager):
    """PD's NIXL transfer, moving one rank's tap pages instead of its KV. Each
    attention-TP rank keeps its own stripe pages of every request, so every
    receiving rank pairs with every sending rank of every stage; page lists are
    whole requests with -1 where a rank keeps no page, and only the pages both
    ends keep move. Encoders pair by id.
    """

    def __init__(
        self,
        *,
        kv_args: KVArgs,
        side: DisaggregationMode,
        server_args: ServerArgs,
        bootstrap_port: Optional[int],
        aux_buffer: torch.Tensor,
    ) -> None:
        self.franky_bootstrap_port = bootstrap_port
        self.aux_buffer = aux_buffer
        self.landed = FrankyLandedPages()
        super().__init__(kv_args, side, server_args, is_mla_backend=True)

    @staticmethod
    def from_buffers(
        *,
        buffers: TapBuffers,
        side: DisaggregationMode,
        server_args: ServerArgs,
        bootstrap_port: Optional[int],
        engine_rank: int,
        pp_rank: int,
        gpu_id: int,
    ) -> FrankyTapManager:
        """
        Args:
            side: PREFILL on the sending engine, DECODE on a receiving one.
            bootstrap_port: the sending engine's bootstrap server port; None on a
                receiving engine.
            engine_rank: the rank within attention TP on a receiving engine.
        """
        aux_buffer = torch.zeros(_AUX_BYTES, dtype=torch.uint8)
        return FrankyTapManager(
            kv_args=_tap_kv_args(
                buffers=buffers,
                aux_buffer=aux_buffer,
                engine_rank=engine_rank,
                pp_rank=pp_rank,
                gpu_id=gpu_id,
            ),
            side=side,
            server_args=server_args,
            bootstrap_port=bootstrap_port,
            aux_buffer=aux_buffer,
        )

    def track_room(self, room: int) -> None:
        self.landed.open(room=room)

    def landed_pages(self, room: int) -> int:
        """
        Returns:
            The leading pages of ``room`` that every sending rank this rank reads
            has written here.
        """
        num_senders = self.required_prefill_response_num_table.get(room)
        if num_senders is None:
            return 0
        return self.landed.pages(room=room, num_senders=num_senders)

    def forget_room(self, room: int) -> None:
        self.landed.close(room=room)

    @override
    def after_chunk_sent(
        self, kv_chunk: TransferKVChunk, receivers: List[TransferInfo]
    ) -> None:
        for receiver in receivers:
            if receiver.is_dummy:
                continue
            self.send_chunk_ready(
                room=kv_chunk.room,
                chunk_idx=0,
                page_start=kv_chunk.index_slice.start,
                num_pages=kv_chunk.index_slice.stop - kv_chunk.index_slice.start,
                writer_id=receiver.agent_name,
                targets=[(receiver.endpoint, receiver.dst_port)],
            )

    @override
    def handle_chunk_ready(self, msg: List[bytes]) -> bool:
        if not msg or msg[0] != b"CHUNK_READY":
            return False
        page_start, num_pages = int(msg[3]), int(msg[4])
        self.landed.record(
            room=int(msg[1]), sender=int(msg[6]), page_end=page_start + num_pages
        )
        return True

    @override
    def send_kvcache(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        dst_mem_kind: str = "VRAM",
    ):
        kept = (prefill_kv_indices >= 0) & (dst_kv_indices >= 0)
        if not kept.any():
            return None
        return super().send_kvcache(
            peer_name,
            prefill_kv_indices[kept],
            dst_kv_ptrs,
            dst_kv_indices[kept],
            dst_gpu_id,
            dst_mem_kind=dst_mem_kind,
        )

    @override
    def _resolve_rank_mapping(self, info: PrefillServerInfo) -> None:
        if info.attn_cp_size != 1 or self.attn_cp_size != 1:
            raise ValueError(
                "Franky moves taps between engines without attention context "
                f"parallelism, got {info.attn_cp_size} and {self.attn_cp_size}."
            )
        info.target_tp_rank = None
        info.target_tp_ranks = list(range(info.attn_tp_size))
        info.target_cp_ranks = [0]
        info.target_pp_ranks = list(range(info.pp_size))
        info.required_dst_info_num = self.attn_tp_size * self.pp_size
        info.required_prefill_response_num = info.attn_tp_size * info.pp_size

    @override
    def _sync_bootstrap_port_across_nodes(self, local_port: int) -> int:
        return self.franky_bootstrap_port


def _tap_kv_args(
    *,
    buffers: TapBuffers,
    aux_buffer: torch.Tensor,
    engine_rank: int,
    pp_rank: int,
    gpu_id: int,
) -> KVArgs:
    transfer_buffers = buffers.transfer_buffers()
    encoder_ids = sorted(transfer_buffers)
    kv_args = KVArgs()
    kv_args.engine_rank = engine_rank
    kv_args.pp_rank = pp_rank
    kv_args.system_dp_rank = get_parallel().dp_rank
    kv_args.rust_http_port = None
    kv_args.kv_cache_dtype_str = f"franky taps, {buffers.page_bytes} bytes a page"
    kv_args.kv_layout = None
    kv_args.prefill_start_layer = 0
    kv_args.prefill_end_layer = None
    kv_args.kv_data_ptrs = [
        transfer_buffers[encoder_id][0] for encoder_id in encoder_ids
    ]
    kv_args.kv_data_lens = [
        transfer_buffers[encoder_id][1] for encoder_id in encoder_ids
    ]
    kv_args.kv_item_lens = [buffers.page_bytes for _ in encoder_ids]
    kv_args.kv_layer_ids = encoder_ids
    kv_args.page_size = buffers.page_size
    kv_args.aux_data_ptrs = [aux_buffer.data_ptr()]
    kv_args.aux_data_lens = [aux_buffer.nbytes]
    kv_args.aux_item_lens = [aux_buffer.nbytes]
    kv_args.ib_device = get_disagg().disaggregation_ib_device
    kv_args.gpu_id = gpu_id
    kv_args.state_types = []
    kv_args.state_data_ptrs = []
    kv_args.state_data_lens = []
    kv_args.state_item_lens = []
    kv_args.state_dim_per_tensor = []
    kv_args.state_slice_outer_counts = []
    kv_args.state_layer_ids = []
    kv_args.state_conv_shard_groups = []
    kv_args.swa_kv_cache_layout = None
    kv_args.mla_compression_ratios = None
    kv_args.is_hybrid_mla_backend = False
    return kv_args
