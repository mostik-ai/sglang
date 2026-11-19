from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import msgspec

if TYPE_CHECKING:
    from sglang.srt.franky.ricky_taps.prefill_tap_limit import PrefillTapLimit
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req


class _Parking(msgspec.Struct, kw_only=True):
    since: float
    landed_at: Optional[float] = None


class FrankyParkedChunks:
    """Ricky's chunked requests whose next chunk waits for taps. The scheduler
    runs one chunked request at a time, so a waiting one steps aside. A request
    that leaves adds two times to its time stats: the seconds until its next
    chunk's taps landed, and the seconds after that until the scheduler took it.
    """

    def __init__(self, *, tap_limit: PrefillTapLimit) -> None:
        self.tap_limit = tap_limit
        self.reqs: list[Req] = []
        self.parkings: dict[str, _Parking] = {}

    def pick(self, *, current: Optional[Req]) -> Optional[Req]:
        """
        Returns:
            ``current`` if its next chunk's taps landed; else, after parking
            ``current``, a parked request whose next chunk's taps landed, or None.
        """
        now = time.perf_counter()
        self._note_landings(now=now)
        if current is not None and self.tap_limit.can_run(current):
            return current
        if current is not None:
            self.reqs.append(current)
            self.parkings[current.rid] = _Parking(since=now)
        for req in self.reqs:
            if self.tap_limit.can_run(req):
                self.reqs.remove(req)
                self._unpark(req, now=now)
                return req
        return None

    def take(self, req: Req) -> bool:
        if req not in self.reqs:
            return False
        self.reqs.remove(req)
        del self.parkings[req.rid]
        return True

    def take_matching(self, *, recv_req: AbortReq) -> list[Req]:
        matching = [
            req
            for req in self.reqs
            if recv_req.abort_all or req.rid.startswith(recv_req.rid)
        ]
        self.reqs = [req for req in self.reqs if req not in matching]
        for req in matching:
            del self.parkings[req.rid]
        return matching

    def is_empty(self) -> bool:
        return not self.reqs

    def _note_landings(self, *, now: float) -> None:
        for req in self.reqs:
            parking = self.parkings[req.rid]
            if parking.landed_at is None and self.tap_limit.can_run(req):
                parking.landed_at = now

    def _unpark(self, req: Req, *, now: float) -> None:
        parking = self.parkings.pop(req.rid)
        req.time_stats.add_franky_parked_seconds(
            waiting_for_taps=parking.landed_at - parking.since,
            after_taps_landed=now - parking.landed_at,
        )
