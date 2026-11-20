from __future__ import annotations

import sys

import msgspec
import numpy as np


class FrankyTapRuns(msgspec.Struct, frozen=True, kw_only=True):
    """Which taps the forward that makes each token of a request reads, as runs
    the router sets: a token from ``starts[i]`` on, up to the next run, reads
    ``taps[i]`` taps. Tokens before the first run read none. The router makes a
    token of an earlier answer read Sandy's tokens before that answer, and a
    token of a chain's answer read the taps of the Sandy call before it.

    Attributes:
        starts: the first token of each run, going up.
        taps: the taps each run's tokens read.
    """

    starts: tuple[int, ...]
    taps: tuple[int, ...]

    @staticmethod
    def of_runs(
        runs: list[list[int]], *, num_input_tokens: int, num_taps: int
    ) -> FrankyTapRuns:
        """
        Args:
            runs: ``franky_tap_runs``, ``[start, taps]`` pairs.
            num_taps: Sandy's tokens of the request's input.
        Raises:
            ValueError: the runs are not pairs going up inside the input, read
                more taps than the input has, or end reading fewer than all.
        """
        malformed = (
            not runs
            or any(len(run) != 2 for run in runs)
            or any(start < 1 or start > num_input_tokens for start, _ in runs)
            or any(taps < 0 or taps > num_taps for _, taps in runs)
            or any(later[0] <= earlier[0] for earlier, later in zip(runs, runs[1:]))
            or runs[-1][1] != num_taps
        )
        if malformed:
            raise ValueError(
                "franky_tap_runs must be [start, taps] pairs whose starts go up "
                f"from 1 to the input's {num_input_tokens} tokens, reading at most "
                f"all {num_taps} taps of the input and all of them last, got {runs}."
            )
        starts, taps = zip(*runs)
        return FrankyTapRuns(starts=starts, taps=taps)

    @staticmethod
    def after_input(*, num_input_tokens: int, num_taps: int) -> FrankyTapRuns:
        """One answer after the input, whose tokens read all of its taps."""
        return FrankyTapRuns(starts=(num_input_tokens,), taps=(num_taps,))

    def taps_to_read(self, tokens: np.ndarray) -> np.ndarray:
        """
        Args:
            tokens: int64 indices of tokens of the request.
        Returns:
            Per token, how many taps the forward that makes it reads.
        """
        starts = np.asarray(self.starts, dtype=tokens.dtype)
        taps = np.asarray((0, *self.taps), dtype=tokens.dtype)
        return taps[np.searchsorted(starts, tokens, side="right")]

    def rows_reading_at_most(self, num_taps: int) -> int:
        """
        Returns:
            How many of the request's first rows read at most ``num_taps`` taps;
            the forward of row ``t`` makes token ``t + 1``.
        """
        for start, taps in zip(self.starts, self.taps):
            if taps > num_taps:
                return start - 1
        return sys.maxsize
