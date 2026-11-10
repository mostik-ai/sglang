from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferDestination
from sglang.srt.disaggregation.nixl.conn import NixlKVReceiver
from sglang.srt.franky.transfer.messages import FrankyRoom
from sglang.srt.franky.transfer.receiver import FrankyReceiver
from sglang.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from sglang.srt.franky.transfer.relay.manager import FrankyRelayManager


class FrankyRelayReceiver(NixlKVReceiver):
    """PD's KV receiver on Ricky decode that also receives the relayed taps of
    the pages it registers, and ends only when both have landed.
    """

    kv_mgr: FrankyRelayManager

    def __init__(
        self,
        mgr: FrankyRelayManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ) -> None:
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self.tap_receiver: Optional[FrankyReceiver] = None

    @override
    def send_metadata(
        self,
        kv_indices: npt.NDArray,
        aux_index: Optional[int] = None,
        state_indices: Optional[list] = None,
        decode_prefix_len: Optional[int] = None,
        destination: KVTransferDestination = KVTransferDestination.DEVICE,
        device_kv_indices: Optional[npt.NDArray] = None,
    ) -> None:
        super().send_metadata(
            kv_indices,
            aux_index,
            state_indices,
            decode_prefix_len,
            destination,
            device_kv_indices,
        )
        if self.conclude_state == KVPoll.Failed:
            return
        tap_pages = self.kv_mgr.tap_cache.pages_of(kv_pages=kv_indices)
        self.tap_receiver = FrankyReceiver(
            manager=self.kv_mgr.tap_relay,
            franky_room=self._relay_room(),
            tap_pages=np.asarray(tap_pages, dtype=np.int32),
            num_tokens=len(tap_pages) * self.kv_mgr.tap_cache.page_size,
        )
        self.kv_mgr.track_tap_receiver(self.bootstrap_room, self.tap_receiver)

    @override
    def poll(self) -> int:
        state = super().poll()
        return (
            state if self.tap_receiver is None else min(state, self.tap_receiver.poll())
        )

    @override
    def abort(self) -> None:
        super().abort()
        if self.tap_receiver is not None:
            self.tap_receiver.abort()

    @override
    def clear(self) -> None:
        super().clear()
        if self.tap_receiver is not None:
            self.tap_receiver.close()
            self.kv_mgr.forget_tap_receiver(self.bootstrap_room)

    def _relay_room(self) -> FrankyRoom:
        return FrankyRoom(
            bootstrap_host=NetworkAddress.parse(self.bootstrap_addr).host,
            bootstrap_port=self.kv_mgr.relay_bootstrap_port,
            room=self.bootstrap_room,
        )
