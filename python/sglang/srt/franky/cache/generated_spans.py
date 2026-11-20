from __future__ import annotations

import bisect
import sys
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

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
        write_ranges: the [start, end) runs of the prompt where Sandy's view
            differs from it, which an aligned reader writes at.
    """

    spans: tuple[GeneratedSpan, ...]
    block_size: Optional[int] = None
    write_ranges: tuple[tuple[int, int], ...] = ()

    @staticmethod
    def of(
        *,
        earlier_answers: tuple[GeneratedSpan, ...],
        num_input_tokens: int,
        chain: Optional[FrankyChain],
        sandy_input_len: Optional[int],
        write_ranges: tuple[tuple[int, int], ...] = (),
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
            return FrankyGeneratedSpans(
                spans=(*earlier_answers, current), write_ranges=write_ranges
            )
        start = num_input_tokens - chain.num_generated_before
        if sandy_input_len is None:
            current = GeneratedSpan(start=start, sandy_start=start)
            return FrankyGeneratedSpans(
                spans=(*earlier_answers, current),
                block_size=chain.block_size,
                write_ranges=write_ranges,
            )
        # Sandy tokenizes on its own: each block of the answer reads Sandy's tokens
        # before it, this one all of Sandy's input. An earlier answer whose blocks
        # are not known in Sandy's tokens reads the taps before it with all of
        # its tokens.
        starts = (*chain.sandy_block_starts, sandy_input_len)
        current = GeneratedSpan(
            start=start, sandy_start=starts[0], sandy_block_starts=starts[1:]
        )
        earlier = tuple(
            (
                span
                if span.sandy_block_starts is not None
                else msgspec.structs.replace(span, sandy_block_starts=())
            )
            for span in earlier_answers
        )
        return FrankyGeneratedSpans(
            spans=(*earlier, current),
            block_size=chain.block_size,
            write_ranges=write_ranges,
        )

    def taps_to_read(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: indices of tokens of the request.
        Returns:
            Per token, how many taps its forward reads: Sandy's tokens before its
            block, or 0.
        """
        taps = torch.zeros_like(tokens)
        for span in self.spans:
            inside = tokens >= span.start
            if span.end is not None:
                inside &= tokens < span.end
            taps = torch.where(inside, self._block_start(tokens, span), taps)
        return taps

    def rows_reading_at_most(self, num_taps: int) -> int:
        """
        Returns:
            How many of the request's first rows read at most ``num_taps`` taps;
            the forward of row ``t`` makes token ``t + 1``. A row that an aligned
            reader writes at reads the tap of its own token ``t``.
        """
        return min(
            self._rows_before_unlanded_blocks(num_taps),
            self._rows_before_unlanded_writes(num_taps),
        )

    def written_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: indices of tokens of the request.
        Returns:
            Per token, whether an aligned reader writes at it: inside a write
            range and outside every answer.
        """
        inside = torch.zeros_like(tokens, dtype=torch.bool)
        for start, end in self.write_ranges:
            inside |= (tokens >= start) & (tokens < end)
        return inside & ~self.answer_tokens(tokens)

    def answer_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: indices of tokens of the request.
        Returns:
            Per token, whether it is a token of one of the answers.
        """
        inside = torch.zeros_like(tokens, dtype=torch.bool)
        for span in self.spans:
            here = tokens >= span.start
            if span.end is not None:
                here &= tokens < span.end
            inside |= here
        return inside

    def _rows_before_unlanded_blocks(self, num_taps: int) -> int:
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

    def _rows_before_unlanded_writes(self, num_taps: int) -> int:
        for start, end in self.write_ranges:
            token = self._first_token_outside_answers(max(start, num_taps))
            if token < end:
                return token
        return sys.maxsize

    def _first_token_outside_answers(self, token: int) -> int:
        for span in self.spans:
            if span.start <= token and (span.end is None or token < span.end):
                if span.end is None:
                    return sys.maxsize
                token = span.end
        return token

    def _block_start(self, tokens: torch.Tensor, span: GeneratedSpan) -> torch.Tensor:
        if self.block_size is None:
            return torch.full_like(tokens, span.sandy_start)
        blocks = (tokens - span.start).div(self.block_size, rounding_mode="floor")
        if span.sandy_block_starts is None:
            return span.sandy_start + blocks * self.block_size
        starts = torch.tensor(
            (span.sandy_start, *span.sandy_block_starts),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        return starts[blocks.clamp(min=0, max=len(starts) - 1)]


def checked_earlier_answers(
    answers: list[list[int]],
    *,
    generation_start: int,
    sandy_answers: Optional[list[list[int]]],
    sandy_generation_start: Optional[int],
    sandy_block_starts: Optional[list[list[int]]] = None,
) -> tuple[GeneratedSpan, ...]:
    """
    Args:
        sandy_answers: the same answers in Sandy's tokens, one per answer; None
            with ``sandy_generation_start`` when Sandy reads Ricky's tokens.
        sandy_generation_start: Sandy's tokens of the input.
        sandy_block_starts: per answer, Sandy's tokens before each of its
            blocks after the first, as a chain of blocks reported them; None
            when unknown, and then every token of an answer reads the taps
            before it.
    Returns:
        The answers as spans.
    Raises:
        ValueError: the answers are not non-empty runs in order inside the
            prompt, on either side, the sides do not match one to one, or an
            answer's block starts are not in order inside its run in Sandy's
            tokens.
    """
    _check_runs(
        answers, generation_start=generation_start, field="franky_generated_spans"
    )
    if sandy_generation_start is None:
        if sandy_answers is not None or sandy_block_starts is not None:
            raise ValueError(
                "franky_sandy_spans and franky_sandy_block_starts come with "
                "franky_sandy_input_ids."
            )
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
    block_starts = _checked_block_starts(sandy_block_starts, sandy_answers)
    return tuple(
        GeneratedSpan(
            start=start,
            sandy_start=sandy_start,
            end=end,
            sandy_block_starts=later_starts,
        )
        for (start, end), (sandy_start, _), later_starts in zip(
            answers, sandy_answers, block_starts
        )
    )


def _checked_block_starts(
    block_starts: Optional[list[list[int]]], sandy_answers: list[list[int]]
) -> list[tuple[int, ...]]:
    """Per answer, Sandy's tokens before each of its blocks after the first."""
    if block_starts is None:
        return [()] * len(sandy_answers)
    if len(block_starts) != len(sandy_answers):
        raise ValueError(
            f"franky_sandy_block_starts has {len(block_starts)} lists for the "
            f"{len(sandy_answers)} answers of franky_sandy_spans; it needs one each."
        )
    for starts, (sandy_start, sandy_end) in zip(block_starts, sandy_answers):
        bounds = [sandy_start, *starts, sandy_end]
        if any(lower > upper for lower, upper in zip(bounds, bounds[1:])):
            raise ValueError(
                "Each list of franky_sandy_block_starts must stay in order inside "
                f"its answer's run in Sandy's tokens [{sandy_start}, {sandy_end}), "
                f"got {starts}."
            )
    return [tuple(starts) for starts in block_starts]


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


def checked_write_ranges(
    ranges: list[list[int]], *, prompt_length: int
) -> tuple[tuple[int, int], ...]:
    """
    Returns:
        The ranges as [start, end) pairs.
    Raises:
        ValueError: the ranges are not non-empty runs in order inside the prompt.
    """
    bounds = [0, *(position for run in ranges for position in run), prompt_length]
    if (
        any(len(run) != 2 for run in ranges)
        or any(lower > upper for lower, upper in zip(bounds, bounds[1:]))
        or any(start == end for start, end in ranges)
    ):
        raise ValueError(
            "franky_write_ranges must be non-empty [start, end) runs in order "
            f"inside a prompt of {prompt_length} tokens, got {ranges}."
        )
    return tuple((start, end) for start, end in ranges)
