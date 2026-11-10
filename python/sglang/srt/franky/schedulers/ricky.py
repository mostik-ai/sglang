from __future__ import annotations

import time
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Tuple

from typing_extensions import override

from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore
from sglang.srt.disaggregation.utils import DisaggregationMode, prepare_abort
from sglang.srt.environ import envs
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.chains.messages import FrankyHoldExpired
from sglang.srt.franky.schedulers.parked_chunks import FrankyParkedChunks
from sglang.srt.franky.schedulers.prefill_adder import FrankyPrefillAdder, taps_landed
from sglang.srt.franky.transfer.manager import FrankyTapManager
from sglang.srt.franky.transfer.messages import FrankyRoomClosed, FrankyTapsReady
from sglang.srt.franky.transfer.rank_reports import FrankyRankReports
from sglang.srt.franky.transfer.receive_queue import FrankyReceiveQueue
from sglang.srt.managers.io_struct import AbortReq, FrankyEndChainReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler, _make_abort_req
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.runtime_context import (
    get_device,
    get_disagg,
    get_franky,
    get_parallel,
    get_serving,
)
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.schedule_policy import PrefillAdder


class RickyScheduler(Scheduler):
    """A Ricky engine that runs prefills: without PD, or as PD prefill. A request
    takes KV pages for its prompt at arrival, and joins a batch only for tokens
    whose taps have landed. PD prefill relays the taps to Ricky decode with the KV.
    A cache flush drops every chain hold.
    """

    @override
    def init_request_dispatcher(self) -> None:
        super().init_request_dispatcher()
        self._request_dispatcher += TypeBasedDispatcher(
            [
                (FrankyTapsReady, self._handle_taps_ready),
                (FrankyRoomClosed, self._handle_room_closed),
                (FrankyHoldExpired, self._handle_hold_expired),
                (FrankyEndChainReq, self._handle_end_chain),
            ]
        )

    @override
    def init_disaggregation(self) -> None:
        super().init_disaggregation()
        self.init_franky_receiving()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.init_franky_relay()

    def init_franky_receiving(self) -> None:
        """Ricky's end of tap transfers: this rank's tap manager, the queue of
        requests with their taken pages, and the chunked requests waiting for taps.
        """
        parallel = get_parallel()
        franky = get_franky()
        tap_cache = self.tp_worker.model_runner.franky_parts.tap_cache
        self.franky_chain_holds = FrankyChainHolds(
            tree_cache=self.tree_cache,
            max_held_tokens=int(
                franky.franky_chain_hold_ratio * self.max_total_num_tokens
            ),
            hold_seconds=franky.franky_chain_hold_seconds,
        )
        self.franky_tap_manager = FrankyTapManager.from_tap_cache(
            tap_cache=tap_cache,
            side=DisaggregationMode.DECODE,
            server_args=self.server_args,
            bootstrap_port=None,
            engine_rank=parallel.attn_tp_rank,
            pp_rank=parallel.pp_rank,
            gpu_id=get_device().gpu_id,
        )
        self.franky_receive_queue = FrankyReceiveQueue(
            tap_cache=tap_cache,
            kv_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            tap_manager=self.franky_tap_manager,
            rank_reports=FrankyRankReports(
                attn_tp_cpu_group=self.attn_tp_cpu_group,
                attn_cp_cpu_group=self.attn_cp_cpu_group,
                pp_store=(
                    PPConsensusStore(parallel.pp_size, parallel.pp_rank, self.pp_group)
                    if parallel.pp_size > 1
                    else None
                ),
            ),
            chain_holds=self.franky_chain_holds,
            page_size=self.page_size,
            reserved_decode_tokens=get_disagg().num_reserved_decode_tokens,
            release_timeout_seconds=(
                envs.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT.get()
            ),
            fail_request=self._fail_franky_request,
        )
        self.franky_parked_chunks = FrankyParkedChunks()

    def init_franky_relay(self) -> None:
        """PD prefill's end of the tap relay to Ricky decode: a bootstrap server on
        the first rank and a sending tap manager on every rank, given to PD's KV
        manager so each KV room also carries its taps.
        """
        parallel = get_parallel()
        port = get_franky().franky_bootstrap_port
        tap_cache = self.tp_worker.model_runner.franky_parts.tap_cache
        self.franky_relay_server = (
            CommonKVBootstrapServer(host=get_serving().host, port=port)
            if parallel.tp_rank == 0 and parallel.pp_rank == 0
            else None
        )
        self.franky_relay_manager = FrankyTapManager.from_tap_cache(
            tap_cache=tap_cache,
            side=DisaggregationMode.PREFILL,
            server_args=self.server_args,
            bootstrap_port=port,
            engine_rank=parallel.tp_rank,
            pp_rank=parallel.pp_rank,
            gpu_id=get_device().gpu_id,
        )
        self.disagg_prefill_bootstrap_queue.kv_manager.attach_tap_relay(
            tap_relay=self.franky_relay_manager,
            tap_cache=tap_cache,
            relay_bootstrap_port=port,
        )

    @override
    def get_prefill_adder_class(self) -> type[PrefillAdder]:
        return FrankyPrefillAdder

    @override
    def prefillable_queue_len(self) -> int:
        return sum(1 for req in self.waiting_queue if taps_landed(req))

    @override
    def ingest_requests(self, stop_at_pause: bool = False) -> List:
        self.franky_receive_queue.poll()
        return super().ingest_requests(stop_at_pause=stop_at_pause)

    @override
    def _poll_timeout_aborts(self) -> List:
        return (
            super()._poll_timeout_aborts()
            + self.franky_receive_queue.ready_messages()
            + self.franky_chain_holds.expired_messages()
        )

    @override
    def _add_request_to_queue(self, req: Req, is_retracted: bool = False) -> None:
        self.franky_receive_queue.add(req=req)
        super()._add_request_to_queue(req, is_retracted=is_retracted)

    @override
    def _release_aborted_request(self, req: Req) -> None:
        self.franky_receive_queue.forget(req=req)
        super()._release_aborted_request(req)

    @override
    def _get_new_batch_prefill_raw(
        self, prefill_delayer_single_pass, running_batch: ScheduleBatch
    ) -> Tuple[Optional[ScheduleBatch], ScheduleBatch]:
        self.chunked_req = self.franky_parked_chunks.pick(current=self.chunked_req)
        new_batch, running_batch = super()._get_new_batch_prefill_raw(
            prefill_delayer_single_pass, running_batch
        )
        if new_batch is not None:
            franky_parts = self.tp_worker.model_runner.franky_parts
            franky_parts.tap_cache.set_tap_lens(
                req_pool_indices=[req.kv.req_pool_idx for req in new_batch.reqs],
                tap_lens=[req.franky.tap_len for req in new_batch.reqs],
            )
            if self.disaggregation_mode == DisaggregationMode.NULL:
                franky_parts.keep_past_the_taps(
                    req_pool_indices=[
                        req.kv.req_pool_idx
                        for req in new_batch.reqs
                        if req is not self.chunked_req
                    ]
                )
        return new_batch, running_batch

    @override
    def abort_request(self, recv_req: AbortReq) -> None:
        for req in self.franky_parked_chunks.take_matching(recv_req=recv_req):
            self._abort_parked_chunk(
                req,
                reason=FINISH_ABORT("Aborted"),
                reason_for_tokenizer=recv_req.finished_reason,
            )
        super().abort_request(recv_req)

    @override
    def flush_cache(self, empty_cache: bool = True) -> bool:
        self.franky_chain_holds.release_all()
        return super().flush_cache(empty_cache=empty_cache)

    @override
    def on_idle(self) -> None:
        super().on_idle()
        # NIXL's transfer threads need the GIL; PD's schedulers yield it the same way.
        time.sleep(0)

    @override
    def is_fully_idle(self, for_health_check=False, ignore_waiting=False) -> bool:
        return (
            self.franky_receive_queue.is_empty()
            and self.franky_parked_chunks.is_empty()
            and super().is_fully_idle(
                for_health_check=for_health_check, ignore_waiting=ignore_waiting
            )
        )

    def _handle_taps_ready(self, message: FrankyTapsReady) -> None:
        self.franky_receive_queue.on_taps_ready(message=message)

    def _handle_room_closed(self, message: FrankyRoomClosed) -> None:
        self.franky_receive_queue.on_room_closed(message=message)

    def _handle_hold_expired(self, message: FrankyHoldExpired) -> None:
        self.franky_chain_holds.on_hold_expired(message=message)

    def _handle_end_chain(self, recv_req: FrankyEndChainReq) -> None:
        self.abort_request(AbortReq(rid=recv_req.chain_id))
        self.franky_chain_holds.release(chain_id=recv_req.chain_id)

    def _fail_franky_request(self, req: Req, reason: str) -> None:
        finished_reason = FINISH_ABORT(
            reason, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
        if req is self.chunked_req:
            req.discard_output_reason = finished_reason
            self._pending_chunked_abort_req = req
        elif self.franky_parked_chunks.take(req):
            self._abort_parked_chunk(
                req,
                reason=finished_reason,
                reason_for_tokenizer=finished_reason.to_json(),
            )
        elif req in self.waiting_queue:
            self.waiting_queue.remove(req)
            prepare_abort(req, reason, status_code=finished_reason.status_code)
            self._release_aborted_request(req)
            self.send_abort_output(
                _make_abort_req(req, finished_reason=finished_reason.to_json()), req
            )
        else:
            prepare_abort(req, reason, status_code=finished_reason.status_code)
            self._release_aborted_request(req)
            req.disagg_kv_sender.abort()

    def _abort_parked_chunk(
        self,
        req: Req,
        *,
        reason: FINISH_ABORT,
        reason_for_tokenizer: Optional[dict],
    ) -> None:
        prepare_abort(req, reason.message, status_code=reason.status_code)
        self._release_aborted_request(req)
        release_kv_cache(req, self.tree_cache, checkpoint=False)
        self.send_abort_output(
            _make_abort_req(req, finished_reason=reason_for_tokenizer), req
        )
