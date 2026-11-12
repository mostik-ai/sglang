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
        self.unsent_prefix_pages = 0

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
            self._start_past_the_receivers_prefix()
        if not self.waiting_pages:
            return
        pages = self._drop_the_receivers_prefix(np.concatenate(self.waiting_pages))
        self.waiting_pages = []
        if len(pages) > 0:
            self.send(pages)

    def _start_past_the_receivers_prefix(self) -> None:
        prefix_len = self.pop_decode_prefix_len()
        page_size = self.kv_mgr.kv_args.page_size
        if prefix_len % page_size != 0 or prefix_len >= self.num_pages * page_size:
            raise ValueError(
                f"Franky room {self.bootstrap_room}: the receivers' cached prefix "
                f"of {prefix_len} tokens is not whole pages of {page_size} tokens "
                f"inside the prompt's {self.num_pages} pages."
            )
        self.unsent_prefix_pages = prefix_len // page_size
        self.init(self.num_pages - self.unsent_prefix_pages, aux_index=0)
        self.receivers_arrived = True

    def _drop_the_receivers_prefix(self, pages: np.ndarray) -> np.ndarray:
        dropped = min(self.unsent_prefix_pages, len(pages))
        self.unsent_prefix_pages -= dropped
        return pages[dropped:]
