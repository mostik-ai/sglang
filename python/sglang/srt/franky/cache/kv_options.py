from __future__ import annotations

import msgspec


class FrankyKvOptions(msgspec.Struct, frozen=True, kw_only=True):
    """What Franky adds to KV pool setup. None with Franky off."""

    tap_bytes_per_token: int
    # The allocators KV setup builds instead of the paged and sliding-window ones.
    paged_allocator_cls: type
    swa_allocator_cls: type
