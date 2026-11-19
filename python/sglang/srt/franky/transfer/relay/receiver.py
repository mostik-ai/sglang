from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy.typing as npt
from typing_extensions import override

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferDestination
from sglang.srt.disaggregation.nixl.conn import NixlKVReceiver
from sglang.srt.franky.transfer.messages import FrankyRoom
from sglang.srt.franky.transfer.receiver import FrankyReceiver
from sglang.srt.utils.common import ceil_div
from sglang.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from sglang.srt.franky.transfer.relay.manager import FrankyRelayManager


class FrankyRelayReceiver(NixlKVReceiver):
    """PD's KV receiver on Ricky decode that also receives the relayed taps, in
    the stripes its request reserved past their cached prefix, and ends only
    once both landed. Until the request reserves its stripes, it reports the
    handshake unfinished, so PD prepares no KV for the request.
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
        reservation = self.kv_mgr.reservations.of(self.bootstrap_room)
        if reservation is None:
            return
        tap_cache = self.kv_mgr.tap_cache
        page_size = tap_cache.buffers.page_size
        num_cached = reservation.cached.num_taps
        self.tap_receiver = FrankyReceiver(
            manager=self.kv_mgr.tap_relay,
            franky_room=self._relay_room(),
            tap_pages=tap_cache.local_pages(
                slots=reservation.slots,
                first_page=num_cached // page_size,
                num_pages=ceil_div(reservation.num_taps, page_size),
            ),
            num_tokens=reservation.num_taps,
            prefix_len=num_cached,
        )
        self.kv_mgr.track_tap_receiver(self.bootstrap_room, self.tap_receiver)

    @override
    def poll(self) -> int:
        reservations = self.kv_mgr.reservations
        if self.conclude_state is None:
            refusal = reservations.refusal(self.bootstrap_room)
            if refusal is not None:
                self.kv_mgr.record_failure(self.bootstrap_room, refusal)
                self.conclude_state = KVPoll.Failed
            elif reservations.waits(self.bootstrap_room):
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

    def _relay_room(self) -> FrankyRoom:
        return FrankyRoom(
            bootstrap_host=NetworkAddress.parse(self.bootstrap_addr).host,
            bootstrap_port=self.kv_mgr.relay_bootstrap_port,
            room=self.bootstrap_room,
        )
