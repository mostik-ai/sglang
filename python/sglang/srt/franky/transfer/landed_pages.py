from __future__ import annotations

import threading


class FrankyLandedPages:
    """How far each sending rank has written the pages of each open room on this
    rank. Messages for a room that is not open are dropped.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.page_ends: dict[int, dict[int, int]] = {}

    def open(self, *, room: int) -> None:
        with self.lock:
            self.page_ends[room] = {}

    def record(self, *, room: int, sender: int, page_end: int) -> None:
        with self.lock:
            senders = self.page_ends.get(room)
            if senders is None:
                return
            senders[sender] = max(senders.get(sender, 0), page_end)

    def pages(self, *, room: int, num_senders: int) -> int:
        """
        Returns:
            The leading pages of ``room`` that all ``num_senders`` sending ranks
            have written; 0 until each of them has written some.
        """
        with self.lock:
            senders = self.page_ends[room]
            if len(senders) < num_senders:
                return 0
            return min(senders.values())

    def close(self, *, room: int) -> None:
        with self.lock:
            self.page_ends.pop(room, None)
