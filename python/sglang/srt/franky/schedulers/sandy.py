from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, Optional

from typing_extensions import override

from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore
from sglang.srt.disaggregation.utils import DisaggregationMode, is_aborted
from sglang.srt.franky.finish_rules.tap_step import TapStep
from sglang.srt.franky.request import FrankyRequest
from sglang.srt.franky.schedulers.chain import FrankyChainScheduler
from sglang.srt.franky.schedulers.sandy_batch_result_processor import (
    SandyBatchResultProcessor,
)
from sglang.srt.franky.transfer.manager import FrankyTapManager
from sglang.srt.franky.transfer.messages import FrankySendDone
from sglang.srt.franky.transfer.rank_reports import FrankyRankReports
from sglang.srt.franky.transfer.send_queue import FrankySendQueue
from sglang.srt.franky.transfer.senders.room import FrankyRoomSender
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.runtime_context import get_device, get_franky, get_parallel, get_serving
from sglang.srt.utils.common import ceil_div
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.franky.cache.sandy_tap_cache import SandyTapCache
    from sglang.srt.franky.transfer.senders.sender import FrankyTapSender
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs


class SandyScheduler(FrankyChainScheduler):
    """The Sandy engine regime: runs requests on the frozen model as any engine
    does, and sends the taps of every token their forwards run over to their
    room. A request answers when its text is written, at its prefill end when
    it writes none, and keeps its KV until its taps are sent.
    """

    @override
    def init_request_dispatcher(self) -> None:
        super().init_request_dispatcher()
        self._request_dispatcher += TypeBasedDispatcher(
            [(FrankySendDone, self._handle_send_done)]
        )

    @override
    def init_batch_result_processor(self) -> None:
        super().init_batch_result_processor()
        self.init_franky_sending()
        self.init_franky_send_queue()
        self.batch_result_processor = SandyBatchResultProcessor.extend(
            self.batch_result_processor, franky_send_queue=self.franky_send_queue
        )

    def init_franky_sending(self) -> None:
        self.franky_bootstrap_server, self.franky_tap_manager = sandy_tap_sending(
            tap_cache=self.tp_worker.model_runner.franky_parts.tap_cache,
            server_args=self.server_args,
        )

    def init_franky_send_queue(self) -> None:
        pp_size, pp_rank = get_parallel().pp_size, get_parallel().pp_rank
        self.franky_send_queue = FrankySendQueue(
            tap_cache=self.tp_worker.model_runner.franky_parts.tap_cache,
            req_to_token_pool=self.req_to_token_pool,
            tree_cache=self.tree_cache,
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
        refusal = prepare_sandy_request(recv_req)
        super().handle_generate_request(
            recv_req, mm_input_error=refusal or mm_input_error
        )

    @override
    def init_req_max_new_tokens(self, req: Req) -> None:
        franky = FrankyRequest.of(req)
        writes_taps = franky.room is not None and not franky.taps_only
        if writes_taps:
            TapStep.ask_extra_token(req)
        super().init_req_max_new_tokens(req)
        if writes_taps:
            TapStep.attach(req)

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
        return super()._poll_timeout_aborts() + self.franky_send_queue.ready_messages()

    @override
    def abort_request(self, recv_req: AbortReq) -> None:
        self.franky_send_queue.abort(recv_req=recv_req)
        super().abort_request(recv_req)

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

    def _open_franky_room(self, req: Req, num_tokens: int) -> FrankyTapSender:
        return FrankyRoomSender(
            manager=self.franky_tap_manager,
            franky_room=FrankyRequest.of(req).room,
            num_pages=ceil_div(
                num_tokens,
                self.tp_worker.model_runner.franky_parts.tap_cache.buffers.page_size,
            ),
        )


def prepare_sandy_request(recv_req: TokenizedGenerateReqInput) -> Optional[str]:
    """Sandy writes plain text: no constrained decoding and no logprobs; a
    taps-only request writes nothing.

    Returns:
        Why the request is refused, or None.
    """
    if recv_req.franky is not None and recv_req.franky.taps_only:
        recv_req.sampling_params.max_new_tokens = 0
    recv_req.sampling_params.json_schema = None
    recv_req.sampling_params.regex = None
    recv_req.sampling_params.ebnf = None
    recv_req.sampling_params.structural_tag = None
    recv_req.return_logprob = False
    has_embeds = recv_req.mm_inputs is not None or recv_req.input_embeds is not None
    # SGLang's server warmup sends requests without a room, which no Ricky reads.
    if has_embeds and recv_req.franky is not None and recv_req.franky.room:
        return "--franky-side sandy takes token ids only in a room."
    return None


def sandy_tap_sending(
    *, tap_cache: SandyTapCache, server_args: ServerArgs
) -> tuple[Optional[CommonKVBootstrapServer], FrankyTapManager]:
    """A Sandy's sending end of tap transfers over PD's NIXL classes: the
    bootstrap server on the first rank, and a tap manager on every rank, since
    a receiver waits for every rank and the server for every rank to register.

    Returns:
        The bootstrap server, None off the first rank, and the tap manager.
    """
    parallel = get_parallel()
    port = get_franky().franky_bootstrap_port
    server = (
        CommonKVBootstrapServer(host=get_serving().host, port=port)
        if parallel.tp_rank == 0 and parallel.pp_rank == 0
        else None
    )
    manager = FrankyTapManager.from_buffers(
        buffers=tap_cache.buffers,
        side=DisaggregationMode.PREFILL,
        server_args=server_args,
        bootstrap_port=port,
        engine_rank=parallel.tp_rank,
        pp_rank=parallel.pp_rank,
        gpu_id=get_device().gpu_id,
    )
    return server, manager
