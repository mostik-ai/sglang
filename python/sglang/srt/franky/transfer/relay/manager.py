from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from typing_extensions import override

from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.runtime_context import get_franky

if TYPE_CHECKING:
    from sglang.srt.franky.transfer.manager import FrankyTapManager
    from sglang.srt.franky.transfer.receiver import FrankyReceiver
    from sglang.srt.franky.transfer.relay.target import FrankyRelayTarget


class FrankyRelayManager(NixlKVManager):
    """PD's NIXL KV manager on an engine with taps, beside the tap manager that
    relays each request's taps from PD prefill to PD decode in the same room.
    A prefill registers its relay port with its PD bootstrap info, where its
    decodes find it. An aborted room's pages are released only once both
    transfers stopped writing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tap_relay: Optional[FrankyTapManager] = None
        self.relay_target: Optional[FrankyRelayTarget] = None
        self.tap_receivers: dict[int, FrankyReceiver] = {}

    def attach_tap_relay(
        self,
        *,
        tap_relay: FrankyTapManager,
        relay_target: Optional[FrankyRelayTarget] = None,
    ) -> None:
        """
        Args:
            relay_target: on PD decode, where each room's relayed taps land.
        """
        self.tap_relay = tap_relay
        self.relay_target = relay_target

    @override
    def extra_registration_fields(self) -> dict:
        return {"franky_relay_port": get_franky().franky_bootstrap_port}

    def relay_port(self, bootstrap_addr: str) -> int:
        """
        Returns:
            The port where the PD prefill at ``bootstrap_addr`` relays its taps.
        Raises:
            RuntimeError: the prefill registered no relay port.
        """
        port = self.prefill_info_table[bootstrap_addr].franky_relay_port
        if port is None:
            raise RuntimeError(
                f"The PD prefill at {bootstrap_addr} registered no Franky relay "
                "port; run it with the same Franky config as this decode."
            )
        return port

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
