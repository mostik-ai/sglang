from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl.conn import NixlKVSender

if TYPE_CHECKING:
    from sglang.srt.franky.transfer.manager import FrankyTapManager
    from sglang.srt.franky.transfer.messages import FrankyRoom


class FrankyRoomSender(NixlKVSender):
    """One request's taps leaving one Sandy rank over PD's NIXL transfer. Pages
    that land before the room's receivers arrive wait here for them.
    """

    def __init__(
        self, *, manager: FrankyTapManager, franky_room: FrankyRoom, num_pages: int
    ) -> None:
        super().__init__(manager, franky_room.bootstrap_address, franky_room.room)
        self.num_pages = num_pages
        self.waiting_pages: list[np.ndarray] = []
        self.receivers_arrived = False

    def land(self, *, tap_pages: np.ndarray) -> None:
        self.waiting_pages.append(tap_pages)
        self._send_waiting_pages()

    @override
    def poll(self) -> int:
        self._send_waiting_pages()
        return super().poll()

    def close(self) -> None:
        self.clear()

    def _send_waiting_pages(self) -> None:
        if not self.receivers_arrived:
            if self.kv_mgr.check_status(self.bootstrap_room) != KVPoll.WaitingForInput:
                return
            self.init(self.num_pages, aux_index=0)
            self.receivers_arrived = True
        if self.waiting_pages:
            self.send(np.concatenate(self.waiting_pages))
            self.waiting_pages = []
