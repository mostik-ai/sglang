from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from typing_extensions import override

from sglang.srt.franky.transfer.relay.target import (
    FrankyRelayLanding,
    FrankyRelayTarget,
)

if TYPE_CHECKING:
    from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
    from sglang.srt.franky.ricky_taps.tap_requests import RickyTapRequests


class RickyRelayTarget(FrankyRelayTarget):
    """Ricky decode's end of the relay: a room's taps land in the stripes its
    request reserved, past their cached prefix. Both Ricky ends split tap pages
    by the request's page.
    """

    senders_split_alike = True

    def __init__(
        self, *, tap_requests: RickyTapRequests, tap_cache: RickyTapCache
    ) -> None:
        self._tap_requests = tap_requests
        self._tap_cache = tap_cache

    @override
    def waits(self, *, room: int) -> bool:
        rid = self._tap_requests.rid_of_room(room)
        return rid is not None and self._tap_requests.waits(rid)

    @override
    def refusal(self, *, room: int) -> Optional[str]:
        rid = self._tap_requests.rid_of_room(room)
        return None if rid is None else self._tap_requests.refusal(rid)

    @override
    def landing(
        self, *, room: int, kv_pages: np.ndarray, prefix_len: int
    ) -> Optional[FrankyRelayLanding]:
        rid = self._tap_requests.rid_of_room(room)
        reservation = None if rid is None else self._tap_requests.reservation(rid)
        if reservation is None:
            return None
        return FrankyRelayLanding(
            tap_pages=self._tap_cache.pages_to_land(reservation),
            prefix_len=reservation.cached.num_taps,
        )
