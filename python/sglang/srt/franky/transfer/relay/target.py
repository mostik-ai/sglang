from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import msgspec
import numpy as np


class FrankyRelayLanding(msgspec.Struct, frozen=True, kw_only=True):
    """Where one room's relayed taps land on one decode rank.

    Attributes:
        tap_pages: int32, per tap page past ``prefix_len``, its local page on
            this rank, or -1 where another rank keeps it.
        num_tokens: the tokens whose taps the room holds, the prefix included.
        prefix_len: the leading tokens, in whole tap pages, whose taps this
            rank already holds; the prefill skips them.
    """

    tap_pages: np.ndarray
    num_tokens: int
    prefix_len: int


class FrankyRelayTarget(ABC):
    """Where the taps a PD prefill relays to its decode land, room by room: the
    decode side's own tap pages of the room's request.
    """

    def waits(self, *, room: int) -> bool:
        """Whether the room's request cannot take its tap pages yet; PD prepares
        no KV for it until it can.
        """
        return False

    def refusal(self, *, room: int) -> Optional[str]:
        """Why the room's request can never take its tap pages; None while it
        can.
        """
        return None

    @abstractmethod
    def landing(
        self, *, room: int, kv_pages: np.ndarray, prefix_len: int
    ) -> Optional[FrankyRelayLanding]:
        """
        Args:
            kv_pages: the request's KV pages past its cached prefix on this
                engine, as PD's transfer sends them.
            prefix_len: the request's cached prefix on this engine, in tokens.
        Returns:
            Where the room's taps land on this rank; None for a request that
            reads no taps.
        """
