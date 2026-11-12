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
    predicts token ``p + 1``. In the prompt it reads taps ``0 .. p``; after the
    prompt it reads the taps before the block of the token it predicts, as that
    block's decode did, so a later block's prefill repeats the decode.

    Args:
        tap_len: the request's taps; no token reads past them.
        generation_start: the position of the first token its chain generated.
        block_size: the chain's block size; None for a request that is one block.
    """

    tap_len: int
    generation_start: int
    block_size: Optional[int] = None

    @staticmethod
    def of(*, tap_len: int, chain: Optional[FrankyChain]) -> FrankyTapReading:
        if chain is None or tap_len == 0:
            return FrankyTapReading(tap_len=tap_len, generation_start=tap_len)
        return FrankyTapReading(
            tap_len=tap_len,
            generation_start=tap_len - chain.num_generated_before,
            block_size=chain.block_size,
        )

    def runs(self, *, first_position: int, num_tokens: int) -> list[TapRun]:
        """The runs of the ``num_tokens`` positions from ``first_position``, in
        order."""
        end = first_position + num_tokens
        prompt_end = min(end, self.generation_start)
        runs = []
        if first_position < prompt_end:
            runs.append(
                TapRun(
                    num_tokens=prompt_end - first_position,
                    causal_len=prompt_end,
                    full_len=0,
                )
            )
        position = max(first_position, prompt_end)
        while position < end:
            limit = self._limit_at(position)
            run_end = (
                end if limit == self.tap_len else min(end, limit + self.block_size - 1)
            )
            runs.append(
                TapRun(num_tokens=run_end - position, causal_len=0, full_len=limit)
            )
            position = run_end
        return runs

    def _limit_at(self, position: int) -> int:
        if self.block_size is None:
            return self.tap_len
        block = (position + 1 - self.generation_start) // self.block_size
        return min(self.tap_len, self.generation_start + block * self.block_size)
