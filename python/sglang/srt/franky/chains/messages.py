from __future__ import annotations

import msgspec


class FrankyHoldExpired(msgspec.Struct, frozen=True, kw_only=True):
    chain_id: str
    order: int
