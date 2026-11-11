from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec

if TYPE_CHECKING:
    from sglang.srt.franky.chains.chain import FrankyChain


class TapRun(msgspec.Struct, frozen=True, kw_only=True):
    """Consecutive tokens of one request that read their taps one way: each up to
    its own position, the last at ``causal_len``; or all of the first ``full_len``.
    """

    num_tokens: int
    causal_len: int
    full_len: int


class FrankyTapReading(msgspec.Struct, frozen=True, kw_only=True):
    """Which taps each token of a request reads. The token at position ``p``
    predicts token ``p + 1``. When token ``p + 1`` was generated, it reads the
    taps before the block of token ``p + 1``, as that block's decode did, so a
    later block or turn that computes generated tokens again repeats their decode.
    Any other token reads taps ``0 .. p``.

    Args:
        tap_len: the request's taps; no token reads past them.
        generation_start: the position of the first token its chain generated.
        block_size: the chain's block size; None for answers of one block each.
        earlier_answers: the ``[start, end)`` runs of tokens Ricky generated in
            earlier turns, in order, all before ``generation_start``. Each one
            was cut into blocks from its own start.
    """

    tap_len: int
    generation_start: int
    block_size: Optional[int] = None
    earlier_answers: tuple[tuple[int, int], ...] = ()

    @staticmethod
    def of(
        *,
        tap_len: int,
        chain: Optional[FrankyChain],
        earlier_answers: tuple[tuple[int, int], ...] = (),
    ) -> FrankyTapReading:
        if tap_len == 0:
            return FrankyTapReading(tap_len=0, generation_start=0)
        if chain is None:
            return FrankyTapReading(
                tap_len=tap_len,
                generation_start=tap_len,
                earlier_answers=earlier_answers,
            )
        return FrankyTapReading(
            tap_len=tap_len,
            generation_start=tap_len - chain.num_generated_before,
            block_size=chain.block_size,
            earlier_answers=earlier_answers,
        )

    def runs(self, *, first_position: int, num_tokens: int) -> list[TapRun]:
        """The runs of the ``num_tokens`` positions from ``first_position``, in
        order."""
        end = first_position + num_tokens
        runs = []
        position = first_position
        answers = (*self.earlier_answers, (self.generation_start, None))
        for answer_start, answer_end in answers:
            causal_end = min(end, answer_start)
            if position < causal_end:
                runs.append(
                    TapRun(
                        num_tokens=causal_end - position,
                        causal_len=causal_end,
                        full_len=0,
                    )
                )
                position = causal_end
            predicting_end = end if answer_end is None else min(end, answer_end - 1)
            while position < predicting_end:
                limit = self._limit_at(position, answer_start)
                run_end = (
                    predicting_end
                    if self.block_size is None or limit == self.tap_len
                    else min(predicting_end, limit + self.block_size - 1)
                )
                runs.append(
                    TapRun(num_tokens=run_end - position, causal_len=0, full_len=limit)
                )
                position = run_end
        return runs

    def _limit_at(self, position: int, answer_start: int) -> int:
        if self.block_size is None:
            return min(self.tap_len, answer_start)
        block = (position + 1 - answer_start) // self.block_size
        return min(self.tap_len, answer_start + block * self.block_size)


def check_earlier_answers(
    answers: list[list[int]], *, generation_start: int
) -> tuple[tuple[int, int], ...]:
    """
    Returns:
        The answers as ``(start, end)`` pairs.
    Raises:
        ValueError: the answers are not non-empty runs in order inside the prompt.
    """
    pairs = tuple((start, end) for start, end in answers)
    bounds = [1, *(position for pair in pairs for position in pair), generation_start]
    if any(lower > upper for lower, upper in zip(bounds, bounds[1:])) or any(
        start == end for start, end in pairs
    ):
        raise ValueError(
            "franky_generated_spans must be non-empty [start, end) runs in order, "
            f"after the first token and before position {generation_start}, got "
            f"{answers}."
        )
    return pairs
