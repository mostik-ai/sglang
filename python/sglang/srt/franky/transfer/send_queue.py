from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.franky.transfer.host_pages import FrankyHostPages
from sglang.srt.franky.transfer.messages import FrankySendDone
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.mem_cache.common import release_kv_cache

if TYPE_CHECKING:
    from sglang.srt.franky.cache.tap_cache import FrankyTapCache
    from sglang.srt.franky.chains.holds import FrankyChainHolds
    from sglang.srt.franky.transfer.rank_reports import FrankyRankReports
    from sglang.srt.franky.transfer.senders.sender import FrankyTapSender
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class _Send(msgspec.Struct, kw_only=True):
    """One request's sending. All ends count tokens from the prompt's start."""

    req: Any
    sender: Any
    landed_end: int = 0
    # Pages landed in order, still on their way to the host for the sender.
    copying: list[Any] = []
    # The prefill ended; the request waits here for its sends.
    held: bool = False
    # The last poll state all ranks of this stage agreed on.
    agreed: Optional[int] = None
    # The leader rank put its release in the request stream.
    announced: bool = False


class FrankySendQueue:
    """Sandy's requests, from arrival until every Ricky engine has their taps.

    A tap lands once its forward ended and the prefix tree took its KV page, so
    the page is final. The tree takes a chunk when the next one is scheduled, so
    without the overlap scheduler a chunk's taps land at the next chunk's result.
    A request does not wait for Ricky to start; late engines
    catch up. A request whose prefill ended stays here, holding its KV and taps,
    as PD prefill holds one until its transfer ends. Its release goes through
    the request stream, so every rank of the DP rank releases it at one step.
    A released block of a chain leaves its prompt held for the chain's next block.
    """

    def __init__(
        self,
        *,
        tap_cache: FrankyTapCache,
        req_to_token_pool: ReqToTokenPool,
        tree_cache: BasePrefixCache,
        output_streamer: SchedulerOutputStreamer,
        rank_reports: FrankyRankReports,
        chain_holds: FrankyChainHolds,
        page_size: int,
        open_room: Callable[[Req], FrankyTapSender],
    ) -> None:
        self._tap_cache = tap_cache
        self._req_to_token = req_to_token_pool.req_to_token
        self._tree_cache = tree_cache
        self._output_streamer = output_streamer
        self._rank_reports = rank_reports
        self._chain_holds = chain_holds
        self._page_size = page_size
        self._open_room = open_room
        self._sends: dict[str, _Send] = {}

    def add(self, *, req: Req) -> None:
        """A new request: open its room."""
        self._sends[req.rid] = _Send(req=req, sender=self._open_room(req))

    def after_chunk(self, *, req: Req, end: int, last: bool) -> None:
        """The forward of the request's tokens up to ``end`` ended. Send the taps
        whose KV pages the prefix tree also holds, as ``req.prefix_len`` says, so
        the pages are final. After the last chunk, keep the request until its
        sends end.
        """
        if (send := self._sends.get(req.rid)) is None:
            return
        self._land(send, end=min(req.prefix_len, end), last=last)
        send.held = last

    def poll(self) -> None:
        """Every rank, at each step: hand copied pages to the senders, and agree
        on the held requests' sends.
        """
        for send in self._sends.values():
            self._hand_over(send)
        held = [send for send in self._sends.values() if send.held]
        if not held:
            return
        agreed = self._rank_reports.report(
            numbers={send.req.rid: send.sender.poll() for send in held}
        )
        for send in held:
            send.agreed = agreed[send.req.rid]

    def ready_messages(self) -> list[FrankySendDone]:
        """The leader rank only: the held requests whose sends ended on every
        rank, successfully or not.
        """
        messages = []
        for send in self._sends.values():
            if not send.held or send.announced or send.agreed is None:
                continue
            state = self._rank_reports.lowest(key=send.req.rid, agreed=send.agreed)
            if state in (KVPoll.Success, KVPoll.Failed):
                messages.append(
                    FrankySendDone(
                        request_rid=send.req.rid, failed=state == KVPoll.Failed
                    )
                )
                send.announced = True
        return messages

    def on_send_done(self, *, message: FrankySendDone) -> None:
        """Every rank, at the same step: release the request and answer it."""
        send = self._sends.get(message.request_rid)
        if send is None or not send.held:
            return
        del self._sends[message.request_rid]
        self._rank_reports.forget(key=message.request_rid)
        req = send.req
        req.time_stats.set_completion_time()
        release_kv_cache(req, self._tree_cache, checkpoint=True)
        if message.failed:
            self._tree_cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)
            prepare_abort(
                req,
                f"Franky taps of request {req.rid} were not sent.",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        else:
            self._hold_for_the_next_block(req)
        self._output_streamer.stream_output([req], False)
        send.sender.close()

    def forget(self, *, req: Req) -> None:
        """The scheduler aborted a request before its prefill ended, and frees
        it: stop its sends.
        """
        send = self._sends.get(req.rid)
        if send is None or send.held:
            return
        send.sender.abort()
        send.sender.close()
        del self._sends[req.rid]

    def abort(self, *, recv_req: AbortReq) -> None:
        """Stop the sends of the matching held requests; each is released when
        its poll fails. The scheduler aborts the others itself.
        """
        for rid, send in self._sends.items():
            if send.held and (recv_req.abort_all or rid.startswith(recv_req.rid)):
                send.sender.abort()

    def is_empty(self) -> bool:
        return not self._sends

    def _land(self, send: _Send, *, end: int, last: bool) -> None:
        """Hand the sender the taps from its last call up to ``end``. Only the
        last call may end inside a page.
        """
        if not last:
            end -= end % self._page_size
        if end <= send.landed_end:
            return
        # The first KV slot of each page from landed_end, which is page aligned.
        first_slots = self._req_to_token[
            send.req.kv.req_pool_idx, send.landed_end : end : self._page_size
        ]
        kv_pages = first_slots // self._page_size
        send.copying.append(
            FrankyHostPages(self._tap_cache.pages_of(kv_pages=kv_pages))
        )
        send.landed_end = end
        self._hand_over(send)

    def _hold_for_the_next_block(self, req: Req) -> None:
        chain = req.franky_chain
        if chain is None or chain.last_block:
            return
        self._chain_holds.hold(
            chain_id=chain.id,
            node=req.last_node,
            num_tokens=len(req.origin_input_ids),
        )

    def _hand_over(self, send: _Send) -> None:
        while send.copying and send.copying[0].ready():
            send.sender.land(tap_pages=send.copying.pop(0).numpy())
