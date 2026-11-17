from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from typing_extensions import override

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.franky.cache.generated_spans import FrankyGeneratedSpans
from sglang.srt.franky.cache.own_tap_pages_metrics import FrankyOwnTapPagesMetrics
from sglang.srt.franky.cache.own_tap_reservations import FrankyOwnTapReservations
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.chains.messages import FrankyHoldExpired
from sglang.srt.franky.transfer.manager import FrankyTapManager
from sglang.srt.managers.io_struct import AbortReq, FrankyEndChainReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    STALL_METRICS_INTERVAL_S,
)
from sglang.srt.runtime_context import get_device, get_franky, get_parallel
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class RickyDecodeScheduler(Scheduler):
    """A Ricky PD decode engine. Each request's taps come from Ricky prefill in the
    room of its KV, and PD hands a request to decode only once both landed. When
    requests own their tap pages, each reserves them before PD prepares its KV,
    and a chain's next block takes over the pages of its last block.
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
        self.init_franky_own_tap_pages_metrics()

    def init_franky_own_tap_pages_metrics(self) -> None:
        self.franky_own_tap_pages_metrics = FrankyOwnTapPagesMetrics.of_scheduler(
            own_pages=self.tp_worker.model_runner.franky_parts.tap_cache.own_pages,
            metrics_collector_context=self.metrics_collector_context,
            interval_seconds=STALL_METRICS_INTERVAL_S,
        )

    def init_franky_relay(self) -> None:
        """PD decode's end of the tap relay from Ricky prefill: this rank's tap
        manager, given to PD's KV manager so each KV room also brings its taps,
        the reservations of requests that own their tap pages, and the chain
        holds that keep those pages between two blocks.
        """
        parallel = get_parallel()
        tap_cache = self.tp_worker.model_runner.franky_parts.tap_cache
        self.franky_chain_holds = FrankyChainHolds.from_config(
            tree_cache=None, own_pages=tap_cache.own_pages
        )
        self.franky_own_taps = (
            FrankyOwnTapReservations(
                own_pages=tap_cache.own_pages,
                page_size=tap_cache.page_size,
                chain_holds=self.franky_chain_holds,
            )
            if tap_cache.own_pages is not None
            else None
        )
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
            own_taps=self.franky_own_taps,
        )

    @override
    def ingest_requests(self, stop_at_pause: bool = False) -> List:
        if self.franky_own_taps is not None:
            self.franky_own_taps.poll()
        if self.franky_own_tap_pages_metrics is not None:
            self.franky_own_tap_pages_metrics.maybe_publish()
        return super().ingest_requests(stop_at_pause=stop_at_pause)

    @override
    def _poll_timeout_aborts(self) -> List:
        return (
            super()._poll_timeout_aborts() + self.franky_chain_holds.expired_messages()
        )

    @override
    def flush_cache(self, empty_cache: bool = True) -> bool:
        self.franky_chain_holds.release_all()
        return super().flush_cache(empty_cache=empty_cache)

    @override
    def _add_request_to_queue(self, req: Req, is_retracted: bool = False) -> None:
        if (
            self.franky_own_taps is not None
            and req.franky_room is not None
            and req.franky_sandy_input_len is not None
        ):
            self.franky_own_taps.expect(req)
        super()._add_request_to_queue(req, is_retracted=is_retracted)

    @override
    def get_new_prebuilt_batch(
        self, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        batch = super().get_new_prebuilt_batch(running_batch)
        if batch is not None:
            tap_cache = self.tp_worker.model_runner.franky_parts.tap_cache
            tap_cache.set_generated_spans(
                req_pool_indices=[req.kv.req_pool_idx for req in batch.reqs],
                generated_spans=[
                    (
                        FrankyGeneratedSpans.of(
                            earlier_answers=req.franky_generated_spans,
                            num_input_tokens=len(req.origin_input_ids),
                            chain=req.franky_chain,
                            sandy_input_len=req.franky_sandy_input_len,
                        )
                        if req.franky_room is not None
                        else None
                    )
                    for req in batch.reqs
                ],
                num_input_tokens=[len(req.origin_input_ids) for req in batch.reqs],
            )
            if self.franky_own_taps is not None:
                self._bind_own_tap_pages(batch.reqs)
        return batch

    def _bind_own_tap_pages(self, reqs: list[Req]) -> None:
        owning = []
        for req in reqs:
            pages = self.franky_own_taps.pages_of(room=req.bootstrap_room)
            if pages is not None:
                owning.append((req.kv.req_pool_idx, pages))
        if owning:
            req_pool_indices, pages = zip(*owning)
            tap_cache = self.tp_worker.model_runner.franky_parts.tap_cache
            tap_cache.own_pages.bind(
                req_pool_indices=list(req_pool_indices), pages=list(pages)
            )

    def _handle_hold_expired(self, message: FrankyHoldExpired) -> None:
        self.franky_chain_holds.on_hold_expired(message=message)

    def _handle_end_chain(self, recv_req: FrankyEndChainReq) -> None:
        self.abort_request(AbortReq(rid=recv_req.chain_id))
        self.franky_chain_holds.release(chain_id=recv_req.chain_id)
