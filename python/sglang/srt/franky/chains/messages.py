from __future__ import annotations

import msgspec


class FrankyHoldExpired(msgspec.Struct, frozen=True, kw_only=True):
    """The leader rank's notice that what the engine keeps for a chain, its
    hold or its memory that the chain ended, is past its time."""

    chain_id: str
    order: int
