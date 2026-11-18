from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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
    """PD's KV receiver on Ricky decode that also receives the relayed taps, and
    ends only when both have landed: in the tap pages of the KV pages it
    registers, or in the tap pages its request reserved when requests own them.
    Until that reservation, it reports the handshake unfinished, so PD prepares
    no KV for the request.
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
        tap_pages = self._tap_pages(kv_indices)
        if tap_pages is None:
            return
        self.tap_receiver = FrankyReceiver(
            manager=self.kv_mgr.tap_relay,
            franky_room=self._relay_room(),
            tap_pages=tap_pages,
            num_tokens=len(tap_pages) * self.kv_mgr.tap_cache.page_size,
        )
        self.kv_mgr.track_tap_receiver(self.bootstrap_room, self.tap_receiver)

    @override
    def poll(self) -> int:
        own_taps = self.kv_mgr.own_taps
        if own_taps is not None and self.conclude_state is None:
            if own_taps.too_large(room=self.bootstrap_room):
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    "Sandy's tokens of the request need more tap pages than a "
                    "request may own on Ricky decode.",
                )
                self.conclude_state = KVPoll.Failed
            elif own_taps.waits(room=self.bootstrap_room):
                return KVPoll.Bootstrapping
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

    def _tap_pages(self, kv_indices: npt.NDArray) -> Optional[npt.NDArray]:
        """The tap pages to land in, or None when the request owns none: it reads
        no taps.
        """
        own_taps = self.kv_mgr.own_taps
        if own_taps is None:
            return self.kv_mgr.tap_cache.pages_of_kv_pages(kv_pages=kv_indices)
        pages = own_taps.pages_of(room=self.bootstrap_room)
        return None if pages is None else pages.numpy()

    def _relay_room(self) -> FrankyRoom:
        return FrankyRoom(
            bootstrap_host=NetworkAddress.parse(self.bootstrap_addr).host,
            bootstrap_port=self.kv_mgr.relay_bootstrap_port,
            room=self.bootstrap_room,
        )
