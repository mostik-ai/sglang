from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from typing_extensions import override

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.franky.cache.generated_spans import FrankyGeneratedSpans
from sglang.srt.franky.cache.tap_pool_metrics import FrankyTapPoolMetrics
from sglang.srt.franky.cache.tap_reservations import FrankyTapReservations
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.chains.messages import FrankyHoldExpired
from sglang.srt.franky.sandy_input import sandy_input_ids
from sglang.srt.franky.transfer.manager import FrankyTapManager
from sglang.srt.managers.io_struct import AbortReq, FrankyEndChainReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    STALL_METRICS_INTERVAL_S,
)
from sglang.srt.runtime_context import get_device, get_franky, get_parallel
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.franky.parts.ricky_parts import RickyParts
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class RickyDecodeScheduler(Scheduler):
    """A Ricky PD decode engine. Each request's taps come from Ricky prefill in
    the room of its KV, and PD hands a request to decode only once both landed.
    A request that reads taps takes their stripes before PD prepares its KV; at
    its end the prefix tree keeps its whole stripes, locked for its chain's next
    block when it is a block that ended at its length.
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
        self.init_franky_relay()
        self.init_franky_tap_pool_metrics()

    def init_franky_tap_pool_metrics(self) -> None:
        self.franky_tap_pool_metrics = FrankyTapPoolMetrics.of_scheduler(
            tap_cache=self._franky_parts.tap_cache,
            metrics_collector_context=self.metrics_collector_context,
            interval_seconds=STALL_METRICS_INTERVAL_S,
        )

    def init_franky_relay(self) -> None:
        """PD decode's end of the tap relay from Ricky prefill: this rank's tap
        manager, given to PD's KV manager so each KV room also brings its taps,
        the requests' stripes by room, and the chain holds that keep a block's
        taps for the chain's next block.
        """
        parallel = get_parallel()
        tap_cache = self._franky_parts.tap_cache
        self.franky_chain_holds = FrankyChainHolds.from_config(
            tree_cache=None, tap_cache=tap_cache
        )
        self.franky_tap_reservations = FrankyTapReservations(
            tap_cache=tap_cache,
            chain_holds=self.franky_chain_holds,
            max_taps=self._franky_parts.max_taps,
        )
        self.franky_reading_reqs: dict[int, Req] = {}
        self.franky_relay_manager = FrankyTapManager.from_buffers(
            buffers=tap_cache.buffers,
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
            reservations=self.franky_tap_reservations,
        )

    @override
    def ingest_requests(self, stop_at_pause: bool = False) -> List:
        self._poll_franky_taps()
        if self.franky_tap_pool_metrics is not None:
            self.franky_tap_pool_metrics.maybe_publish()
        return super().ingest_requests(stop_at_pause=stop_at_pause)

    @override
    def _poll_timeout_aborts(self) -> List:
        return (
            super()._poll_timeout_aborts() + self.franky_chain_holds.expired_messages()
        )

    @override
    def flush_cache(self, empty_cache: bool = True) -> bool:
        self.franky_chain_holds.release_all()
        flushed = super().flush_cache(empty_cache=empty_cache)
        if flushed:
            self._franky_parts.tap_cache.reset()
        return flushed

    @override
    def _add_request_to_queue(self, req: Req, is_retracted: bool = False) -> None:
        if req.franky_room is not None and not is_retracted:
            self.franky_tap_reservations.expect(
                req.bootstrap_room, req=req, sandy_ids=sandy_input_ids(req)
            )
        super()._add_request_to_queue(req, is_retracted=is_retracted)

    @override
    def get_new_prebuilt_batch(
        self, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        batch = super().get_new_prebuilt_batch(running_batch)
        if batch is not None:
            self._bind_tap_rows(batch.reqs)
        return batch

    def _poll_franky_taps(self) -> None:
        for room, req in list(self.franky_reading_reqs.items()):
            if req.finished() and not req.kv.holds_kv:
                self._end_franky_taps(room, req)
        for room, req in self.franky_tap_reservations.poll():
            self.franky_chain_holds.release_chain_of(req)
            self.franky_reading_reqs[room] = req

    def _end_franky_taps(self, room: int, req: Req) -> None:
        del self.franky_reading_reqs[room]
        landed = not isinstance(req.finished_reason, FINISH_ABORT)
        hold = landed and isinstance(req.finished_reason, FINISH_LENGTH)
        tap_lock = self.franky_tap_reservations.end(
            room, landed=landed, keep_locked=hold
        )
        if hold:
            self.franky_chain_holds.hold_for_next_block(req, tap_lock=tap_lock)
        else:
            self.franky_chain_holds.release_chain_of(req)

    def _bind_tap_rows(self, reqs: list[Req]) -> None:
        slots = []
        for req in reqs:
            reservation = self.franky_tap_reservations.of(req.bootstrap_room)
            slots.append(None if reservation is None else reservation.slots)
        self._franky_parts.read_rows.bind(
            req_pool_indices=[req.kv.req_pool_idx for req in reqs],
            slots=slots,
            spans=[
                (
                    FrankyGeneratedSpans.of_request(req)
                    if req.franky_room is not None
                    else None
                )
                for req in reqs
            ],
            num_input_tokens=[len(req.origin_input_ids) for req in reqs],
        )

    @property
    def _franky_parts(self) -> RickyParts:
        return self.tp_worker.model_runner.franky_parts

    def _handle_hold_expired(self, message: FrankyHoldExpired) -> None:
        self.franky_chain_holds.on_hold_expired(message=message)

    def _handle_end_chain(self, recv_req: FrankyEndChainReq) -> None:
        self.abort_request(AbortReq(rid=recv_req.chain_id))
        self.franky_chain_holds.release(chain_id=recv_req.chain_id)
