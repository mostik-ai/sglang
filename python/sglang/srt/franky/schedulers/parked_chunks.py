from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req


class FrankyParkedChunks:
    """Ricky's chunked requests whose next chunk waits for taps. The scheduler
    runs one chunked request at a time, so a waiting one steps aside.
    """

    def __init__(self) -> None:
        self.reqs: list[Req] = []

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
        for req in self.reqs:
            if _next_chunk_landed(req):
                self.reqs.remove(req)
                return req
        return None

    def take(self, req: Req) -> bool:
        if req not in self.reqs:
            return False
        self.reqs.remove(req)
        return True

    def take_matching(self, *, recv_req: AbortReq) -> list[Req]:
        matching = [
            req
            for req in self.reqs
            if recv_req.abort_all or req.rid.startswith(recv_req.rid)
        ]
        self.reqs = [req for req in self.reqs if req not in matching]
        return matching

    def is_empty(self) -> bool:
        return not self.reqs


def _next_chunk_landed(req: Req) -> bool:
    return req.franky.taps_ready > req.prefix_len
