from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.franky.chains.chain import FrankyChain


class GeneratedSpan(msgspec.Struct, frozen=True, kw_only=True):
    """Tokens ``[start, end)`` of a request that Ricky generated; ``end`` is None
    while the answer is still being generated."""

    start: int
    end: Optional[int] = None


class FrankyGeneratedSpans(msgspec.Struct, frozen=True, kw_only=True):
    """A request's answers in order, the last one being generated now; the
    forward of each answer token reads the taps of the tokens before its block.

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
        write_ranges: tuple[tuple[int, int], ...] = (),
    ) -> FrankyGeneratedSpans:
        """The spans of a request whose new answer starts after the chain's
        earlier blocks, or right after its input without a chain."""
        if chain is None:
            current = GeneratedSpan(start=num_input_tokens)
            return FrankyGeneratedSpans(
                spans=(*earlier_answers, current), write_ranges=write_ranges
            )
        current = GeneratedSpan(start=num_input_tokens - chain.num_generated_before)
        return FrankyGeneratedSpans(
            spans=(*earlier_answers, current),
            block_size=chain.block_size,
            write_ranges=write_ranges,
        )

    def taps_to_read(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: indices of tokens of the request.
        Returns:
            Per token, how many taps its forward reads: its block's start, or 0.
        """
        taps = torch.zeros_like(tokens)
        for span in self.spans:
            inside = tokens >= span.start
            if span.end is not None:
                inside &= tokens < span.end
            taps = torch.where(inside, self._block_start(tokens, span.start), taps)
        return taps

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

    def _block_start(self, tokens: torch.Tensor, answer_start: int) -> torch.Tensor:
        if self.block_size is None:
            return torch.full_like(tokens, answer_start)
        blocks = (tokens - answer_start).div(self.block_size, rounding_mode="floor")
        return answer_start + blocks * self.block_size


def checked_earlier_answers(
    answers: list[list[int]], *, generation_start: int
) -> tuple[GeneratedSpan, ...]:
    """
    Returns:
        The answers as spans.
    Raises:
        ValueError: the answers are not non-empty runs in order inside the prompt.
    """
    bounds = [
        1,
        *(position for answer in answers for position in answer),
        generation_start,
    ]
    if (
        any(len(answer) != 2 for answer in answers)
        or any(lower > upper for lower, upper in zip(bounds, bounds[1:]))
        or any(start == end for start, end in answers)
    ):
        raise ValueError(
            "franky_generated_spans must be non-empty [start, end) runs in order, "
            f"after the first token and before position {generation_start}, got "
            f"{answers}."
        )
    return tuple(GeneratedSpan(start=start, end=end) for start, end in answers)


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
