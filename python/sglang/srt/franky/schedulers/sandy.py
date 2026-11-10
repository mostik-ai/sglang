from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, Optional, Tuple

from typing_extensions import override

from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore
from sglang.srt.disaggregation.utils import DisaggregationMode, is_aborted
from sglang.srt.franky.chains.chain import put_continuations_first
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.chains.messages import FrankyHoldExpired
from sglang.srt.franky.schedulers.sandy_batch_result_processor import (
    SandyBatchResultProcessor,
)
from sglang.srt.franky.transfer.manager import FrankyTapManager
from sglang.srt.franky.transfer.messages import FrankySendDone
from sglang.srt.franky.transfer.rank_reports import FrankyRankReports
from sglang.srt.franky.transfer.send_queue import FrankySendQueue
from sglang.srt.franky.transfer.senders.no_room import FrankyNoRoomSender
from sglang.srt.franky.transfer.senders.room import FrankyRoomSender
from sglang.srt.managers.io_struct import AbortReq, FrankyEndChainReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_device, get_franky, get_parallel, get_serving
from sglang.srt.utils.common import ceil_div
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.franky.transfer.senders.sender import FrankyTapSender
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class SandyScheduler(Scheduler):
    """The Sandy engine regime: prefill on input tokens, send taps, never decode.
    It runs the normal event loops; a request ends at its prefill and answers
    with no tokens once its taps are sent.
    """

    @override
    def init_request_dispatcher(self) -> None:
        super().init_request_dispatcher()
        self._request_dispatcher += TypeBasedDispatcher(
            [
                (FrankySendDone, self._handle_send_done),
                (FrankyHoldExpired, self._handle_hold_expired),
                (FrankyEndChainReq, self._handle_end_chain),
            ]
        )

    @override
    def init_batch_result_processor(self) -> None:
        super().init_batch_result_processor()
        self.init_franky_sending()
        self.init_franky_chain_holds()
        self.init_franky_send_queue()
        self.batch_result_processor = SandyBatchResultProcessor.extend(
            self.batch_result_processor, franky_send_queue=self.franky_send_queue
        )

    def init_franky_sending(self) -> None:
        """Sandy's end of tap transfers over PD's NIXL classes: the bootstrap
        server on the first rank, and a tap manager on every rank, since Ricky
        waits for every rank and the server for every rank to register.
        """
        parallel = get_parallel()
        port = get_franky().franky_bootstrap_port
        self.franky_bootstrap_server = (
            CommonKVBootstrapServer(host=get_serving().host, port=port)
            if parallel.tp_rank == 0 and parallel.pp_rank == 0
            else None
        )
        self.franky_tap_manager = FrankyTapManager.from_tap_cache(
            tap_cache=self.tp_worker.model_runner.franky_parts.tap_cache,
            side=DisaggregationMode.PREFILL,
            server_args=self.server_args,
            bootstrap_port=port,
            engine_rank=parallel.tp_rank,
            pp_rank=parallel.pp_rank,
            gpu_id=get_device().gpu_id,
        )

    def init_franky_chain_holds(self) -> None:
        self.franky_chain_holds = FrankyChainHolds.from_config(
            tree_cache=self.tree_cache, max_total_num_tokens=self.max_total_num_tokens
        )

    def init_franky_send_queue(self) -> None:
        pp_size, pp_rank = get_parallel().pp_size, get_parallel().pp_rank
        self.franky_send_queue = FrankySendQueue(
            tap_cache=self.tp_worker.model_runner.franky_parts.tap_cache,
            req_to_token_pool=self.req_to_token_pool,
            tree_cache=self.tree_cache,
            output_streamer=self.output_streamer,
            rank_reports=FrankyRankReports(
                attn_tp_cpu_group=self.attn_tp_cpu_group,
                attn_cp_cpu_group=self.attn_cp_cpu_group,
                pp_store=(
                    PPConsensusStore(pp_size, pp_rank, self.pp_group)
                    if pp_size > 1
                    else None
                ),
            ),
            chain_holds=self.franky_chain_holds,
            page_size=self.page_size,
            open_room=self._open_franky_room,
        )

    @override
    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
        *,
        mm_input_error: Optional[str] = None,
    ) -> None:
        # SGLang's server warmup sends requests without a room, which no Ricky reads.
        has_embeds = recv_req.mm_inputs is not None or recv_req.input_embeds is not None
        if has_embeds and recv_req.franky_room is not None:
            mm_input_error = "--franky-side sandy takes token ids only in a room."
        # Sandy runs the prompt and answers no tokens, so sampling has no use.
        recv_req.sampling_params.max_new_tokens = 0
        recv_req.sampling_params.json_schema = None
        recv_req.sampling_params.regex = None
        recv_req.sampling_params.ebnf = None
        recv_req.sampling_params.structural_tag = None
        recv_req.return_logprob = False
        super().handle_generate_request(recv_req, mm_input_error=mm_input_error)

    @override
    def _add_request_to_queue(self, req: Req, is_retracted: bool = False) -> None:
        if not is_retracted and not is_aborted(req):
            self.franky_send_queue.add(req=req)
        super()._add_request_to_queue(req, is_retracted=is_retracted)

    @override
    def _release_aborted_request(self, req: Req) -> None:
        self.franky_send_queue.forget(req=req)
        super()._release_aborted_request(req)

    @override
    def ingest_requests(self, stop_at_pause: bool = False) -> List:
        self.franky_send_queue.poll()
        return super().ingest_requests(stop_at_pause=stop_at_pause)

    @override
    def _poll_timeout_aborts(self) -> List:
        return (
            super()._poll_timeout_aborts()
            + self.franky_send_queue.ready_messages()
            + self.franky_chain_holds.expired_messages()
        )

    @override
    def _get_new_batch_prefill_raw(
        self, prefill_delayer_single_pass, running_batch: ScheduleBatch
    ) -> Tuple[Optional[ScheduleBatch], ScheduleBatch]:
        put_continuations_first(self.waiting_queue)
        new_batch, running_batch = super()._get_new_batch_prefill_raw(
            prefill_delayer_single_pass, running_batch
        )
        if new_batch is None and self._waits_without_running(running_batch):
            self.franky_chain_holds.release_oldest()
        return new_batch, running_batch

    @override
    def get_chunked_prefill_size_for_batch(self) -> Optional[int]:
        return self.franky_chain_holds.chunk_size_for_batch(
            super().get_chunked_prefill_size_for_batch()
        )

    @override
    def abort_request(self, recv_req: AbortReq) -> None:
        self.franky_send_queue.abort(recv_req=recv_req)
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
        return self.franky_send_queue.is_empty() and super().is_fully_idle(
            for_health_check=for_health_check, ignore_waiting=ignore_waiting
        )

    def _handle_send_done(self, message: FrankySendDone) -> None:
        self.franky_send_queue.on_send_done(message=message)

    def _handle_hold_expired(self, message: FrankyHoldExpired) -> None:
        self.franky_chain_holds.on_hold_expired(message=message)

    def _handle_end_chain(self, recv_req: FrankyEndChainReq) -> None:
        self.abort_request(AbortReq(rid=recv_req.chain_id))
        self.franky_chain_holds.release(chain_id=recv_req.chain_id)

    def _waits_without_running(self, running_batch: ScheduleBatch) -> bool:
        return (
            len(self.waiting_queue) > 0
            and self.chunked_req is None
            and running_batch.is_empty()
        )

    def _open_franky_room(self, req: Req) -> FrankyTapSender:
        if req.franky_room is None:
            return FrankyNoRoomSender()
        return FrankyRoomSender(
            manager=self.franky_tap_manager,
            franky_room=req.franky_room,
            num_pages=ceil_div(len(req.origin_input_ids), self.page_size),
        )
