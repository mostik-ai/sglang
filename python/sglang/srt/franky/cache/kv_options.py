from __future__ import annotations

import msgspec


class FrankyKvOptions(msgspec.Struct, frozen=True, kw_only=True):
    """What Franky adds to KV pool setup. None with Franky off."""

    tap_bytes_per_token: int
    paged_allocator_cls: type
