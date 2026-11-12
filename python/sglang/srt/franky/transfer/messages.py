"""Stage 0's leader rank puts ``FrankyTapsReady`` and ``FrankySendDone`` in the
request stream, so every rank of the DP rank applies them at the same step.
Sending one twice is harmless.

The request's id is ``request_rid``, not ``rid``: the scheduler takes anything
with a health-check ``rid`` for a health check and drops it while busy.
"""

from __future__ import annotations

import msgspec

from sglang.srt.utils.network import NetworkAddress


class FrankyRoom(msgspec.Struct, frozen=True, kw_only=True):
    """Where a request's taps meet: the sending engine's bootstrap server, and
    the room the router gave both engines' requests.
    """

    bootstrap_host: str
    bootstrap_port: int
    room: int

    @property
    def bootstrap_address(self) -> str:
        return NetworkAddress(
            self.bootstrap_host, self.bootstrap_port
        ).to_host_port_str()


class FrankyTapsReady(msgspec.Struct, frozen=True, kw_only=True):
    request_rid: str
    taps_ready: int
    failed: bool = False


class FrankySendDone(msgspec.Struct, frozen=True, kw_only=True):
    request_rid: str
    failed: bool = False


class FrankyRoomClosed(msgspec.Struct, frozen=True, kw_only=True):
    request_rid: str
