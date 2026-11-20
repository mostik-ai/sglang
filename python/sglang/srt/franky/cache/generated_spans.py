from __future__ import annotations

import bisect
import sys
from typing import TYPE_CHECKING, Optional

import msgspec
import numpy as np

if TYPE_CHECKING:
    from sglang.srt.franky.chains.chain import FrankyChain


class GeneratedSpan(msgspec.Struct, frozen=True, kw_only=True):
    """Tokens ``[start, end)`` of a request that Ricky generated; ``end`` is None
    while the answer is still being generated. Sandy's tokens of the same answer
    start at ``sandy_start``, which is ``start`` when both sides tokenize alike.

    Attributes:
        sandy_block_starts: with blocks and a Sandy that tokenizes on its own,
            Sandy's tokens before each block of the answer after its first; past
            the last, every block reads as many. None when Sandy's copy of each
            block is as long as Ricky's.
    """

    start: int
    sandy_start: int
    end: Optional[int] = None
    sandy_block_starts: Optional[tuple[int, ...]] = None


class FrankyGeneratedSpans(msgspec.Struct, frozen=True, kw_only=True):
    """A request's answers in order, the last one being generated now. The
    forward of each answer token reads the taps of Sandy's tokens before its
    block: before the answer's copy in Sandy's tokens.

    Attributes:
        block_size: tokens per block, counted from each answer's start; None: one.
    """

    spans: tuple[GeneratedSpan, ...]
    block_size: Optional[int] = None

    @staticmethod
    def of(
        *,
        earlier_answers: tuple[GeneratedSpan, ...],
        num_input_tokens: int,
        chain: Optional[FrankyChain],
        sandy_input_len: Optional[int],
    ) -> FrankyGeneratedSpans:
        """The spans of a request whose new answer starts after the chain's
        earlier blocks, or right after its input without a chain.

        Args:
            sandy_input_len: Sandy's tokens of the input, where Sandy's copy of
                the new answer starts; None when Sandy reads Ricky's tokens.
        """
        if chain is None:
            current = GeneratedSpan(
                start=num_input_tokens,
                sandy_start=(
                    num_input_tokens if sandy_input_len is None else sandy_input_len
                ),
            )
            return FrankyGeneratedSpans(spans=(*earlier_answers, current))
        start = num_input_tokens - chain.num_generated_before
        if sandy_input_len is None:
            current = GeneratedSpan(start=start, sandy_start=start)
            return FrankyGeneratedSpans(
                spans=(*earlier_answers, current), block_size=chain.block_size
            )
        # Sandy tokenizes on its own: each block of the answer reads Sandy's tokens
        # before it, this one all of Sandy's input. An earlier answer's blocks
        # are not known in Sandy's tokens, so all of its tokens read the taps
        # before it.
        starts = (*chain.sandy_block_starts, sandy_input_len)
        current = GeneratedSpan(
            start=start, sandy_start=starts[0], sandy_block_starts=starts[1:]
        )
        earlier = tuple(
            msgspec.structs.replace(span, sandy_block_starts=())
            for span in earlier_answers
        )
        return FrankyGeneratedSpans(
            spans=(*earlier, current), block_size=chain.block_size
        )

    def taps_to_read(self, tokens: np.ndarray) -> np.ndarray:
        """
        Args:
            tokens: int64 indices of tokens of the request.
        Returns:
            Per token, how many taps its forward reads: Sandy's tokens before its
            block, or 0.
        """
        taps = np.zeros_like(tokens)
        for span in self.spans:
            inside = tokens >= span.start
            if span.end is not None:
                inside &= tokens < span.end
            taps = np.where(inside, self._block_start(tokens, span), taps)
        return taps

    def rows_reading_at_most(self, num_taps: int) -> int:
        """
        Returns:
            How many of the request's first rows read at most ``num_taps`` taps;
            the forward of row ``t`` makes token ``t + 1``.
        """
        for span in self.spans:
            if span.sandy_start > num_taps:
                return span.start - 1
            if self.block_size is None:
                continue
            if span.sandy_block_starts is None:
                past_blocks = (num_taps - span.sandy_start) // self.block_size + 1
            else:
                past_blocks = 1 + bisect.bisect_right(span.sandy_block_starts, num_taps)
                if past_blocks > len(span.sandy_block_starts):
                    continue
            first_token = span.start + past_blocks * self.block_size
            if span.end is None or first_token < span.end:
                return first_token - 1
        return sys.maxsize

    def _block_start(self, tokens: np.ndarray, span: GeneratedSpan) -> np.ndarray:
        if self.block_size is None:
            return np.full_like(tokens, span.sandy_start)
        blocks = (tokens - span.start) // self.block_size
        if span.sandy_block_starts is None:
            return span.sandy_start + blocks * self.block_size
        starts = np.array(
            (span.sandy_start, *span.sandy_block_starts), dtype=tokens.dtype
        )
        return starts[blocks.clip(0, len(starts) - 1)]


def checked_earlier_answers(
    answers: list[list[int]],
    *,
    generation_start: int,
    sandy_answers: Optional[list[list[int]]],
    sandy_generation_start: Optional[int],
) -> tuple[GeneratedSpan, ...]:
    """
    Args:
        sandy_answers: the same answers in Sandy's tokens, one per answer; None
            with ``sandy_generation_start`` when Sandy reads Ricky's tokens.
        sandy_generation_start: Sandy's tokens of the input.
    Returns:
        The answers as spans.
    Raises:
        ValueError: the answers are not non-empty runs in order inside the
            prompt, on either side, or the sides do not match one to one.
    """
    _check_runs(
        answers, generation_start=generation_start, field="franky_generated_spans"
    )
    if sandy_generation_start is None:
        if sandy_answers is not None:
            raise ValueError("franky_sandy_spans come with franky_sandy_input_ids.")
        return tuple(
            GeneratedSpan(start=start, sandy_start=start, end=end)
            for start, end in answers
        )
    sandy_answers = sandy_answers or []
    _check_runs(
        sandy_answers,
        generation_start=sandy_generation_start,
        field="franky_sandy_spans",
    )
    if len(sandy_answers) != len(answers):
        raise ValueError(
            f"Each of the {len(answers)} answers in franky_generated_spans has a "
            f"run in Sandy's tokens, but franky_sandy_spans has {len(sandy_answers)}."
        )
    return tuple(
        GeneratedSpan(start=start, sandy_start=sandy_start, end=end)
        for (start, end), (sandy_start, _) in zip(answers, sandy_answers)
    )


def _check_runs(runs: list[list[int]], *, generation_start: int, field: str) -> None:
    bounds = [1, *(position for run in runs for position in run), generation_start]
    if (
        any(len(run) != 2 for run in runs)
        or any(lower > upper for lower, upper in zip(bounds, bounds[1:]))
        or any(start == end for start, end in runs)
    ):
        raise ValueError(
            f"{field} must be non-empty [start, end) runs in order, after the first "
            f"token and before position {generation_start}, got {runs}."
        )
