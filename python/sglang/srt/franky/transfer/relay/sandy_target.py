from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from typing_extensions import override

from sglang.srt.franky.transfer.relay.target import (
    FrankyRelayLanding,
    FrankyRelayTarget,
)

if TYPE_CHECKING:
    from sglang.srt.franky.cache.sandy_tap_cache import SandyTapCache


class SandyRelayTarget(FrankyRelayTarget):
    """Sandy decode's end of the relay: a room's taps land at the request's own
    KV slots, as all of Sandy's taps do, so it never waits and never refuses.
    Both Sandy ends split tap pages by KV slot, which differ between them.
    """

    senders_split_alike = False

    def __init__(self, *, tap_cache: SandyTapCache, kv_page_size: int) -> None:
        self._tap_cache = tap_cache
        self._kv_page_size = kv_page_size

    @override
    def landing(
        self, *, room: int, kv_pages: np.ndarray, prefix_len: int
    ) -> Optional[FrankyRelayLanding]:
        tap_pages = self._tap_cache.pages_of_kv_pages(
            kv_pages=kv_pages, kv_page_size=self._kv_page_size
        )
        return FrankyRelayLanding(tap_pages=tap_pages, prefix_len=prefix_len)
