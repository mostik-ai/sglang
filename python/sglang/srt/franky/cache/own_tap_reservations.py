from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch

from sglang.srt.franky.chains.holds import FrankyHeldTaps
from sglang.srt.managers.schedule_batch import FINISH_LENGTH
from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.franky.cache.own_tap_pages import FrankyOwnTapPages
    from sglang.srt.franky.chains.holds import FrankyChainHolds
    from sglang.srt.managers.schedule_batch import Req


class _Reservation(msgspec.Struct, frozen=True, kw_only=True):
    req: Any
    pages: torch.Tensor
    taps_held: int


class FrankyOwnTapReservations:
    """Ricky decode's requests whose taps live in tap pages of their own, as when
    Sandy tokenizes on its own, by PD room. Each takes its pages in arrival order
    before PD registers its KV, so the relay has somewhere to land, and gives
    them back once it ended and PD released its KV, after which no relay writes
    them. A block that ended at its length leaves them to its chain's next block
    instead, whose relay lands only the taps past their whole pages. Every rank
    takes and frees at the same steps, as it decides on KV.
    """

    def __init__(
        self,
        *,
        own_pages: FrankyOwnTapPages,
        page_size: int,
        chain_holds: FrankyChainHolds,
    ) -> None:
        self._own_pages = own_pages
        self._page_size = page_size
        self._chain_holds = chain_holds
        self._waiting: dict[int, Req] = {}
        self._last_block_taps: dict[int, FrankyHeldTaps] = {}
        self._reserved: dict[int, _Reservation] = {}

    def expect(self, req: Req) -> None:
        """``req`` arrived with taps to relay; it takes pages at a later poll. A
        block takes the tap pages its chain's last block left."""
        self._waiting[req.bootstrap_room] = req
        if req.franky_chain is not None:
            self._take_taps_of_the_last_block(req)

    def poll(self) -> None:
        """Every rank, at each step: hold or free the pages of ended requests, then
        take pages for waiting ones in order while they fit.
        """
        for room, reservation in list(self._reserved.items()):
            if reservation.req.finished() and not reservation.req.kv.holds_kv:
                self._hold_or_give_back(reservation)
                del self._reserved[room]
        for room, req in list(self._waiting.items()):
            if req.finished():
                self._forget_waiting(room)
                continue
            if self._too_large(req) or not self._fits_after_releasing_holds(room, req):
                return
            self._reserve(room, req)

    def waits(self, *, room: int) -> bool:
        """Whether the request in ``room`` still waits for its pages."""
        return room in self._waiting

    def too_large(self, *, room: int) -> bool:
        """Whether the request in ``room`` has more taps than a request may own."""
        req = self._waiting.get(room)
        return req is not None and self._too_large(req)

    def pages_of(self, *, room: int) -> Optional[torch.Tensor]:
        """
        Returns:
            int32 host pages of the request in ``room``, or None if it holds none.
        """
        reservation = self._reserved.get(room)
        return None if reservation is None else reservation.pages

    def taps_held_of(self, *, room: int) -> int:
        """
        Returns:
            The first taps of the request in ``room``, in whole pages, that its
            pages held when it took them; the relay skips them.
        """
        return self._reserved[room].taps_held

    def _take_taps_of_the_last_block(self, req: Req) -> None:
        taps = self._chain_holds.take_taps(chain_id=req.franky_chain.id)
        self._chain_holds.release_chain_of(req)
        if taps is None:
            return
        if taps.num_taps > req.franky_sandy_input_len:
            self._own_pages.free(taps.pages)
            return
        self._last_block_taps[req.bootstrap_room] = taps

    def _hold_or_give_back(self, reservation: _Reservation) -> None:
        req = reservation.req
        if not isinstance(req.finished_reason, FINISH_LENGTH):
            self._chain_holds.release_chain_of(req)
            self._own_pages.free(reservation.pages)
            return
        self._chain_holds.hold_for_next_block(
            req,
            taps=FrankyHeldTaps(
                pages=reservation.pages, num_taps=req.franky_sandy_input_len
            ),
        )

    def _forget_waiting(self, room: int) -> None:
        del self._waiting[room]
        taps = self._last_block_taps.pop(room, None)
        if taps is not None:
            self._own_pages.free(taps.pages)

    def _fits_after_releasing_holds(self, room: int, req: Req) -> bool:
        num_new_pages = self._num_new_pages(room, req)
        free_pages = self._own_pages.num_free()
        if free_pages + self._chain_holds.held_tap_pages < num_new_pages:
            return False
        while self._own_pages.num_free() < num_new_pages:
            if not self._chain_holds.release_newest_taps():
                return False
        return True

    def _reserve(self, room: int, req: Req) -> None:
        new_pages = self._own_pages.take(num_pages=self._num_new_pages(room, req))
        last_block_taps = self._last_block_taps.pop(room, None)
        if last_block_taps is None:
            pages, taps_held = new_pages, 0
        else:
            pages = torch.cat([last_block_taps.pages, new_pages])
            taps_held = last_block_taps.num_reused_taps(
                num_pages=len(pages), page_size=self._page_size
            )
        self._reserved[room] = _Reservation(req=req, pages=pages, taps_held=taps_held)
        del self._waiting[room]

    def _num_new_pages(self, room: int, req: Req) -> int:
        taps = self._last_block_taps.get(room)
        return self._num_pages(req) - (0 if taps is None else len(taps.pages))

    def _num_pages(self, req: Req) -> int:
        return ceil_div(req.franky_sandy_input_len, self._page_size)

    def _too_large(self, req: Req) -> bool:
        return self._num_pages(req) > self._own_pages.max_pages_per_request
