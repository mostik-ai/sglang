from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from typing_extensions import override

from sglang.srt.disaggregation.nixl.conn import NixlKVManager

if TYPE_CHECKING:
    from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
    from sglang.srt.franky.cache.tap_reservations import FrankyTapReservations
    from sglang.srt.franky.transfer.manager import FrankyTapManager
    from sglang.srt.franky.transfer.receiver import FrankyReceiver


class FrankyRelayManager(NixlKVManager):
    """PD's NIXL KV manager on a Ricky engine, beside the tap manager that relays
    each request's taps from Ricky prefill to Ricky decode in the same room. An
    aborted room's pages are released only once both transfers stopped writing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tap_relay: Optional[FrankyTapManager] = None
        self.tap_cache: Optional[RickyTapCache] = None
        self.relay_bootstrap_port: Optional[int] = None
        self.reservations: Optional[FrankyTapReservations] = None
        self.tap_receivers: dict[int, FrankyReceiver] = {}

    def attach_tap_relay(
        self,
        *,
        tap_relay: FrankyTapManager,
        tap_cache: RickyTapCache,
        relay_bootstrap_port: int,
        reservations: Optional[FrankyTapReservations] = None,
    ) -> None:
        """
        Args:
            relay_bootstrap_port: the port of Ricky prefill's relay bootstrap
                server, on the host of its PD bootstrap server.
            reservations: on Ricky decode, the requests' stripes by PD room.
        """
        self.tap_relay = tap_relay
        self.tap_cache = tap_cache
        self.relay_bootstrap_port = relay_bootstrap_port
        self.reservations = reservations

    def track_tap_receiver(self, room: int, receiver: FrankyReceiver) -> None:
        self.tap_receivers[room] = receiver

    def forget_tap_receiver(self, room: int) -> None:
        self.tap_receivers.pop(room, None)

    @override
    def is_abort_release_safe(self, bootstrap_room: int, required_acks: int) -> bool:
        receiver = self.tap_receivers.get(bootstrap_room)
        taps_stopped = receiver is None or receiver.writes_stopped()
        return taps_stopped and super().is_abort_release_safe(
            bootstrap_room, required_acks
        )
