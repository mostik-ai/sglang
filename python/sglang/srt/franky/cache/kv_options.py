from __future__ import annotations

import msgspec


class FrankyKvOptions(msgspec.Struct, frozen=True, kw_only=True):
    """What Franky adds to KV pool setup. None with Franky off.

    Attributes:
        tap_bytes_per_token: this rank's tap bytes per token of the KV pool,
            which the KV budget sets aside for the tap cache.
    """

    tap_bytes_per_token: int
