from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from sglang.srt.disaggregation.nixl.conn import NixlKVSender
from sglang.srt.franky.transfer.messages import FrankyRoom
from sglang.srt.franky.transfer.senders.room import FrankyRoomSender

if TYPE_CHECKING:
    from sglang.srt.franky.transfer.relay.manager import FrankyRelayManager


class FrankyRelaySender(NixlKVSender):
    """PD's KV sender on Ricky prefill that also relays the request's taps to
    Ricky decode, and ends only when both transfers ended: the taps of each KV
    page it sends, or, for a request that owns its tap pages, all of them at the
    last chunk.
    """

    kv_mgr: FrankyRelayManager

    def __init__(
        self,
        mgr: FrankyRelayManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ) -> None:
        super().__init__(
            mgr, bootstrap_addr, bootstrap_room, req_has_disagg_prefill_dp_rank
        )
        self.tap_sender: Optional[FrankyRoomSender] = None

    @override
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None) -> None:
        super().init(num_kv_indices, aux_index)
        tap_cache = self.kv_mgr.tap_cache
        if tap_cache.own_pages is None:
            self.tap_sender = self._open_tap_room(
                num_pages=num_kv_indices * tap_cache.pages_per_kv_page
            )

    @override
    def send(
        self,
        kv_indices: npt.NDArray,
        state_indices: Optional[list] = None,
        num_kv_tokens: Optional[int] = None,
        indices_event: Optional[object] = None,
    ) -> None:
        super().send(
            kv_indices,
            state_indices,
            num_kv_tokens=num_kv_tokens,
            indices_event=indices_event,
        )
        tap_cache = self.kv_mgr.tap_cache
        if tap_cache.own_pages is None:
            self.tap_sender.land(
                tap_pages=tap_cache.pages_of_kv_pages(kv_pages=kv_indices)
            )

    def relay_own_taps(self, *, tap_pages: np.ndarray) -> None:
        """Relay the taps of a request that owns its tap pages, all at once: its
        last chunk read them all.
        """
        if self.tap_sender is None:
            self.tap_sender = self._open_tap_room(num_pages=len(tap_pages))
            self.tap_sender.land(tap_pages=tap_pages)

    @override
    def poll(self) -> int:
        state = super().poll()
        return state if self.tap_sender is None else min(state, self.tap_sender.poll())

    @override
    def abort(self) -> None:
        super().abort()
        if self.tap_sender is not None:
            self.tap_sender.abort()

    @override
    def clear(self) -> None:
        super().clear()
        if self.tap_sender is not None:
            self.tap_sender.close()

    def _open_tap_room(self, *, num_pages: int) -> FrankyRoomSender:
        return FrankyRoomSender(
            manager=self.kv_mgr.tap_relay,
            franky_room=self._relay_room(),
            num_pages=num_pages,
        )

    def _relay_room(self) -> FrankyRoom:
        host = self.bootstrap_server_url.rsplit(":", 1)[0].strip("[]")
        return FrankyRoom(
            bootstrap_host=host,
            bootstrap_port=self.kv_mgr.relay_bootstrap_port,
            room=self.bootstrap_room,
        )
