from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.franky.cache.own_tap_pages import FrankyOwnTapPages
    from sglang.srt.managers.schedule_batch import Req


class FrankyOwnTapReservations:
    """Ricky decode's requests whose taps live in tap pages of their own, as when
    Sandy tokenizes on its own, by PD room. Each takes its pages in arrival order
    before PD registers its KV, so the relay has somewhere to land, and gives
    them back once it ended and PD released its KV, after which no relay writes
    them. Every rank takes and frees at the same steps, as it decides on KV.
    """

    def __init__(self, *, own_pages: FrankyOwnTapPages, page_size: int) -> None:
        self._own_pages = own_pages
        self._page_size = page_size
        self._waiting: dict[int, Req] = {}
        self._held: dict[int, tuple[Req, torch.Tensor]] = {}

    def expect(self, req: Req) -> None:
        """``req`` arrived with taps to relay; it takes pages at a later poll."""
        self._waiting[req.bootstrap_room] = req

    def poll(self) -> None:
        """Every rank, at each step: free the pages of ended requests, then take
        pages for waiting ones in order while they fit.
        """
        for room, (req, pages) in list(self._held.items()):
            if req.finished() and not req.kv.holds_kv:
                self._own_pages.free(pages)
                del self._held[room]
        for room, req in list(self._waiting.items()):
            if req.finished():
                del self._waiting[room]
                continue
            if self._too_large(req):
                return
            pages = self._own_pages.take(num_pages=self._num_pages(req))
            if pages is None:
                return
            self._held[room] = (req, pages)
            del self._waiting[room]

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
        held = self._held.get(room)
        return None if held is None else held[1]

    def _num_pages(self, req: Req) -> int:
        return ceil_div(req.franky_sandy_input_len, self._page_size)

    def _too_large(self, req: Req) -> bool:
        return self._num_pages(req) > self._own_pages.max_pages_per_request
