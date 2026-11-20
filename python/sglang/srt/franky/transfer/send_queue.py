from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.franky.request import FrankyRequest
from sglang.srt.franky.transfer.host_pages import FrankyHostPages
from sglang.srt.franky.transfer.messages import FrankySendDone
from sglang.srt.mem_cache.common import release_kv_cache

if TYPE_CHECKING:
    from sglang.srt.franky.cache.sandy_tap_cache import SandyTapCache
    from sglang.srt.franky.chains.holds import FrankyChainHolds
    from sglang.srt.franky.transfer.rank_reports import FrankyRankReports
    from sglang.srt.franky.transfer.senders.sender import FrankyTapSender
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


class _Send(msgspec.Struct, kw_only=True):
    """One request's sending. All ends count tokens from the prompt's start."""

    req: Any
    sender: Any
    landed_end: int = 0
    # Pages landed in order, still on their way to the host for the sender.
    copying: list[Any] = []
    # The request ended; it waits here for its sends.
    held: bool = False
    # Whether the prefix tree keeps the request's KV when it is released.
    checkpoint: bool = True
    # The last poll state all ranks of this stage agreed on.
    agreed: Optional[int] = None
    # The leader rank put its release in the request stream.
    announced: bool = False


class FrankySendQueue:
    """Sandy's requests with a room, until every Ricky engine has their taps.

    A request that writes no tokens opens its room at arrival, and its prompt's
    taps land as its prefill goes, while Ricky's leg runs beside it. A tap lands
    once its forward ended and the prefix tree took its KV page, so the page is
    final. The tree takes a chunk when the next one is scheduled, so without the
    overlap scheduler a chunk's taps land at the next chunk's result. A request
    that writes tokens opens its room when it ends, since Ricky's leg reads its
    text and comes after it.

    A request answers when it ends, as on any engine, and stays here holding its
    KV and taps until they are sent, as PD prefill holds one until its transfer
    ends. Its release goes through the request stream, so every rank of the DP
    rank releases it at one step. Sends that fail leave the Ricky leg without
    its taps, which fails it.
    """

    def __init__(
        self,
        *,
        tap_cache: SandyTapCache,
        req_to_token_pool: ReqToTokenPool,
        tree_cache: BasePrefixCache,
        rank_reports: FrankyRankReports,
        chain_holds: FrankyChainHolds,
        page_size: int,
        open_room: Callable[[Req, int], FrankyTapSender],
    ) -> None:
        """
        Args:
            open_room: opens a request's room for the taps of its first tokens,
                as many as the number it is given.
        """
        self._tap_cache = tap_cache
        self._req_to_token = req_to_token_pool.req_to_token
        self._tree_cache = tree_cache
        self._rank_reports = rank_reports
        self._chain_holds = chain_holds
        self._page_size = page_size
        self._open_room = open_room
        self._sends: dict[str, _Send] = {}

    def add(self, *, req: Req) -> None:
        """A new request: open the room of one that writes no tokens."""
        if FrankyRequest.of(req).room is None or req.sampling_params.max_new_tokens:
            return
        self._sends[req.rid] = _Send(
            req=req, sender=self._open_room(req, len(req.origin_input_ids))
        )

    def after_chunk(self, *, req: Req, end: int) -> None:
        """The forward of a middle chunk ended at token ``end``. Land the taps
        whose KV pages the prefix tree also holds, as ``req.prefix_len`` says, so
        the pages are final.
        """
        send = self._sends.get(req.rid)
        if send is not None:
            self._land(send, end=min(req.prefix_len, end), last=False)

    def after_end(self, *, req: Req, checkpoint: bool) -> None:
        """A request with a room ended well and answers now. Land the taps of
        every token its forwards ran over, opening its room if it wrote tokens,
        and hold it until they are sent.

        Args:
            checkpoint: whether the prefix tree keeps its KV at its release.
        """
        num_tokens = req.owned_kv_len()
        send = self._sends.get(req.rid)
        if send is None:
            send = _Send(req=req, sender=self._open_room(req, num_tokens))
            self._sends[req.rid] = send
        send.checkpoint = checkpoint
        self._land(send, end=num_tokens, last=True)
        send.held = True

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
        """Every rank, at the same step: release the request, and hold what it
        left for its chain's next request when its taps were sent.
        """
        send = self._sends.get(message.request_rid)
        if send is None or not send.held:
            return
        del self._sends[message.request_rid]
        self._rank_reports.forget(key=message.request_rid)
        release_kv_cache(send.req, self._tree_cache, checkpoint=send.checkpoint)
        if message.failed:
            logger.warning(
                "Franky taps of request %s were not sent; the Ricky leg that "
                "reads them fails.",
                send.req.rid,
            )
            self._chain_holds.release_chain_of(send.req)
        else:
            self._chain_holds.hold_for_next_block(send.req)
        send.sender.close()

    def forget(self, *, req: Req) -> None:
        """The scheduler aborted a request before it ended, and frees it: stop
        its sends.
        """
        send = self._sends.get(req.rid)
        if send is None or send.held:
            return
        send.sender.abort()
        send.sender.close()
        del self._sends[req.rid]
        self._chain_holds.release_chain_of(req)

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
        slots = self._req_to_token[send.req.kv.req_pool_idx, send.landed_end : end]
        send.copying.append(
            FrankyHostPages(self._tap_cache.local_pages(kv_slots=slots))
        )
        send.landed_end = end
        self._hand_over(send)

    def _hand_over(self, send: _Send) -> None:
        while send.copying and send.copying[0].ready():
            send.sender.land(tap_pages=send.copying.pop(0).numpy())
