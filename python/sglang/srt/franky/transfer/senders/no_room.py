from __future__ import annotations

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll


class FrankyNoRoomSender:
    """A request sent without a room: its taps go nowhere."""

    def __init__(self) -> None:
        self.aborted = False

    def land(self, *, tap_pages: np.ndarray) -> None:
        pass

    def poll(self) -> int:
        return KVPoll.Failed if self.aborted else KVPoll.Success

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        pass
