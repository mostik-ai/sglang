from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from typing_extensions import override

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.franky.cache.tap_reading import FrankyTapReading
from sglang.srt.franky.transfer.manager import FrankyTapManager
from sglang.srt.managers.io_struct import AbortReq, FrankyEndChainReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_device, get_franky, get_parallel
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class RickyDecodeScheduler(Scheduler):
    """A Ricky PD decode engine. Each request's taps come from Ricky prefill in the
    room of its KV, and PD hands a request to decode only once both landed.
    """

    @override
    def init_request_dispatcher(self) -> None:
        super().init_request_dispatcher()
        self._request_dispatcher += TypeBasedDispatcher(
            [(FrankyEndChainReq, self._handle_end_chain)]
        )

    @override
    def init_disaggregation(self) -> None:
        super().init_disaggregation()
        self.init_franky_relay()

    def init_franky_relay(self) -> None:
        """PD decode's end of the tap relay from Ricky prefill: this rank's tap
        manager, given to PD's KV manager so each KV room also brings its taps.
        """
        parallel = get_parallel()
        tap_cache = self.tp_worker.model_runner.franky_parts.tap_cache
        self.franky_relay_manager = FrankyTapManager.from_tap_cache(
            tap_cache=tap_cache,
            side=DisaggregationMode.DECODE,
            server_args=self.server_args,
            bootstrap_port=None,
            engine_rank=parallel.attn_tp_rank,
            pp_rank=parallel.pp_rank,
            gpu_id=get_device().gpu_id,
        )
        self.disagg_decode_prealloc_queue.kv_manager.attach_tap_relay(
            tap_relay=self.franky_relay_manager,
            tap_cache=tap_cache,
            relay_bootstrap_port=get_franky().franky_bootstrap_port,
        )

    @override
    def get_new_prebuilt_batch(
        self, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        batch = super().get_new_prebuilt_batch(running_batch)
        if batch is not None:
            franky_parts = self.tp_worker.model_runner.franky_parts
            req_pool_indices = [req.kv.req_pool_idx for req in batch.reqs]
            franky_parts.tap_cache.set_readings(
                req_pool_indices=req_pool_indices,
                readings=[
                    FrankyTapReading.of(
                        tap_len=(
                            len(req.origin_input_ids)
                            if req.franky_room is not None
                            else 0
                        ),
                        chain=req.franky_chain,
                        earlier_answers=req.franky_generated_spans,
                    )
                    for req in batch.reqs
                ],
            )
            # PD hands a request over once its taps landed; it only decodes here.
            franky_parts.keep_past_the_taps(req_pool_indices=req_pool_indices)
        return batch

    def _handle_end_chain(self, recv_req: FrankyEndChainReq) -> None:
        self.abort_request(AbortReq(rid=recv_req.chain_id))
