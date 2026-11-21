from __future__ import annotations

import sys
from typing import Optional

import msgspec

from sglang.srt.franky.tap_runs import FrankyTapRuns


class TapProgress(msgspec.Struct, kw_only=True):
    """How far one Ricky request's taps landed on this rank. Taps count Sandy's
    tokens.

    Attributes:
        tap_len: the taps the request reads.
        taps_ready: its first taps that landed on every rank.
        tap_runs: which taps each of its tokens' forwards reads; None when it reads
            none.
    """

    tap_len: int
    taps_ready: int
    tap_runs: Optional[FrankyTapRuns] = None

    def rows_ready(self) -> int:
        """
        Returns:
            How many of the request's first tokens may run: tokens whose forward
            reads only landed taps.
        """
        if self.tap_runs is None:
            return sys.maxsize
        return self.tap_runs.rows_reading_at_most(self.taps_ready)

    def taps_complete(self) -> bool:
        return self.taps_ready >= self.tap_len
