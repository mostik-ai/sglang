from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from typing_extensions import override

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl.conn import NixlKVSender
from sglang.srt.franky.transfer.messages import FrankyRoom
from sglang.srt.franky.transfer.senders.room import FrankyRoomSender

if TYPE_CHECKING:
    from sglang.srt.franky.transfer.relay.manager import FrankyRelayManager


class FrankyRelaySender(NixlKVSender):
    """PD's KV sender on Ricky prefill that also relays the request's taps to
    Ricky decode in the same room, once they all landed here, and ends only
    when both transfers ended.
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

    def relay_taps(self, *, tap_pages: np.ndarray) -> None:
        """Relay all of the request's taps at once.

        Args:
            tap_pages: int32, per page of the request, its local page here or -1.
        """
        if self.tap_sender is not None:
            return
        self.tap_sender = FrankyRoomSender(
            manager=self.kv_mgr.tap_relay,
            franky_room=self._relay_room(),
            num_pages=len(tap_pages),
        )
        self.tap_sender.land(tap_pages=tap_pages)

    @property
    def relays_taps(self) -> bool:
        return self.tap_sender is not None

    @override
    def poll(self) -> int:
        """The less advanced of the KV transfer and the relay; a relay still
        waiting for its receiver counts as waiting for input."""
        state = super().poll()
        if self.tap_sender is None:
            return state
        taps = self.tap_sender.poll()
        if taps == KVPoll.Bootstrapping:
            taps = KVPoll.WaitingForInput
        return min(state, taps)

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

    def _relay_room(self) -> FrankyRoom:
        host = self.bootstrap_server_url.rsplit(":", 1)[0].strip("[]")
        return FrankyRoom(
            bootstrap_host=host,
            bootstrap_port=self.kv_mgr.relay_bootstrap_port,
            room=self.bootstrap_room,
        )
