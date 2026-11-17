from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req


class FrankyParkedChunks:
    """Ricky's chunked requests whose next chunk waits for taps. The scheduler
    runs one chunked request at a time, so a waiting one steps aside. A request
    that leaves after its taps landed adds its time here to its time stats.
    """

    def __init__(self) -> None:
        self.reqs: list[Req] = []
        self.parked_since: dict[str, float] = {}

    def pick(self, *, current: Optional[Req]) -> Optional[Req]:
        """
        Returns:
            ``current`` if its next chunk's taps landed; else, after parking
            ``current``, a parked request whose next chunk's taps landed, or None.
        """
        if current is not None and _next_chunk_landed(current):
            return current
        if current is not None:
            self.reqs.append(current)
            self.parked_since[current.rid] = time.perf_counter()
        for req in self.reqs:
            if _next_chunk_landed(req):
                self.reqs.remove(req)
                self._unpark(req)
                return req
        return None

    def take(self, req: Req) -> bool:
        if req not in self.reqs:
            return False
        self.reqs.remove(req)
        self.parked_since.pop(req.rid, None)
        return True

    def take_matching(self, *, recv_req: AbortReq) -> list[Req]:
        matching = [
            req
            for req in self.reqs
            if recv_req.abort_all or req.rid.startswith(recv_req.rid)
        ]
        self.reqs = [req for req in self.reqs if req not in matching]
        for req in matching:
            self.parked_since.pop(req.rid, None)
        return matching

    def is_empty(self) -> bool:
        return not self.reqs

    def _unpark(self, req: Req) -> None:
        parked_seconds = time.perf_counter() - self.parked_since.pop(req.rid)
        req.time_stats.add_franky_parked_seconds(parked_seconds)


def _next_chunk_landed(req: Req) -> bool:
    return req.franky.rows_ready() > req.prefix_len
