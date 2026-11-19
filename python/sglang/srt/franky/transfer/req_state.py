from __future__ import annotations

import sys
from typing import Optional

import msgspec

from sglang.srt.franky.cache.generated_spans import FrankyGeneratedSpans


class FrankyReqState(msgspec.Struct, kw_only=True):
    """One Ricky request's taps on this rank. Taps count Sandy's tokens.

    Attributes:
        tap_len: the taps the request reads.
        taps_ready: its first taps that landed on every rank.
        generated_spans: which taps each row reads; None when it reads none.
    """

    tap_len: int
    taps_ready: int
    generated_spans: Optional[FrankyGeneratedSpans] = None

    def rows_ready(self) -> int:
        """
        Returns:
            How many of the request's first rows may run: rows whose forward
            reads only landed taps.
        """
        if self.generated_spans is None:
            return sys.maxsize
        return self.generated_spans.rows_reading_at_most(self.taps_ready)

    def taps_complete(self) -> bool:
        return self.taps_ready >= self.tap_len
