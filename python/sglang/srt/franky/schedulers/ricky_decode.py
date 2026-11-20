from __future__ import annotations

import functools
from typing import TYPE_CHECKING, List, Optional

from typing_extensions import override

from sglang.srt.franky.chains.messages import FrankyHoldExpired
from sglang.srt.franky.finish_rules.request_blocks import RequestBlocks
from sglang.srt.franky.ricky_taps.taps import RickyTaps
from sglang.srt.franky.transfer.relay.ends import relay_to_decode
from sglang.srt.franky.transfer.relay.ricky_target import RickyRelayTarget
from sglang.srt.managers.io_struct import AbortReq, FrankyEndChainReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    STALL_METRICS_INTERVAL_S,
)
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.franky.parts.ricky_parts import RickyParts
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class RickyDecodeScheduler(Scheduler):
    """A Ricky PD decode engine. Each request's taps come from Ricky prefill in
    the room of its KV, and PD hands a request to decode only once both landed.
    A request that reads taps takes their stripes in ``RickyTaps`` before PD
    prepares its KV, and gives them back once PD released its KV.
    """

    @override
    def init_request_dispatcher(self) -> None:
        super().init_request_dispatcher()
        self._request_dispatcher += TypeBasedDispatcher(
            [
                (FrankyHoldExpired, self._handle_hold_expired),
                (FrankyEndChainReq, self._handle_end_chain),
            ]
        )

    @override
    def init_disaggregation(self) -> None:
        super().init_disaggregation()
        self.init_franky_taps()
        self.init_franky_relay()

    def init_franky_taps(self) -> None:
        """This rank's taps; decode caches no prompts, so chain holds keep taps
        only.
        """
        self.franky_taps = RickyTaps.build(
            parts=self._franky_parts,
            tree_cache=None,
            mamba_room=None,
            metrics_collector_context=self.metrics_collector_context,
            metrics_interval_seconds=STALL_METRICS_INTERVAL_S,
        )

    def init_franky_relay(self) -> None:
        """PD decode's end of the tap relay from Ricky prefill: this rank's tap
        manager, given to PD's KV manager so each KV room also brings its taps.
        """
        self.franky_relay_manager = relay_to_decode(
            self.disagg_decode_prealloc_queue.kv_manager,
            buffers=self._franky_parts.tap_buffers,
            server_args=self.server_args,
            target=RickyRelayTarget(
                tap_requests=self.franky_taps.requests,
                tap_cache=self.franky_taps.cache,
            ),
        )

    @override
    def ingest_requests(self, stop_at_pause: bool = False) -> List:
        self._poll_franky_taps()
        self.franky_taps.publish_metrics()
        return super().ingest_requests(stop_at_pause=stop_at_pause)

    @override
    def _poll_timeout_aborts(self) -> List:
        return (
            super()._poll_timeout_aborts()
            + self.franky_taps.chain_holds.expired_messages()
        )

    @override
    def flush_cache(self, empty_cache: bool = True) -> bool:
        return self.franky_taps.flush(
            functools.partial(super().flush_cache, empty_cache=empty_cache)
        )

    @override
    def _add_request_to_queue(self, req: Req, is_retracted: bool = False) -> None:
        # The relay receiver of a refused request fails its room.
        self.franky_taps.requests.arrive(req)
        RequestBlocks.attach(req)
        super()._add_request_to_queue(req, is_retracted=is_retracted)

    @override
    def get_new_prebuilt_batch(
        self, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        batch = super().get_new_prebuilt_batch(running_batch)
        if batch is not None:
            self.franky_taps.requests.join_batch(batch.reqs)
        return batch

    def _poll_franky_taps(self) -> None:
        tap_requests = self.franky_taps.requests
        for req in tap_requests.reserved_requests():
            if req.finished() and not req.kv.holds_kv:
                tap_requests.end(
                    req, landed=not isinstance(req.finished_reason, FINISH_ABORT)
                )
        tap_requests.poll()

    @property
    def _franky_parts(self) -> RickyParts:
        return self.tp_worker.model_runner.franky_parts

    def _handle_hold_expired(self, message: FrankyHoldExpired) -> None:
        self.franky_taps.chain_holds.on_hold_expired(message=message)

    def _handle_end_chain(self, recv_req: FrankyEndChainReq) -> None:
        self.abort_request(AbortReq(rid=recv_req.chain_id))
        self.franky_taps.chain_holds.release(chain_id=recv_req.chain_id)
