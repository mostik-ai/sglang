from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Optional

import msgspec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class FrankyChain(msgspec.Struct, frozen=True, kw_only=True):
    """The chain of blocks a client request runs as, seen from one of its blocks.

    Args:
        id: the chain's id; the rid of each of its blocks starts with it.
        block: this block's number, from 1.
        block_size: the tokens a block generates; only the last may generate fewer.
        last_block: no block follows this one, so the engines hold nothing after it.
        sandy_block_starts: with a Sandy that tokenizes on its own, Sandy's tokens
            before each earlier block of the answer, from the first; this block
            starts at Sandy's input length. Empty for a first block, and when
            Sandy reads Ricky's tokens: its copy of each block is as long.
    """

    id: Annotated[str, msgspec.Meta(min_length=1)]
    block: Annotated[int, msgspec.Meta(ge=1)] = 1
    block_size: Annotated[int, msgspec.Meta(ge=1)]
    last_block: bool = False
    sandy_block_starts: tuple[Annotated[int, msgspec.Meta(ge=0)], ...] = ()

    @staticmethod
    def from_request(
        *,
        franky_chain: dict[str, Any],
        rid: str,
        prompt_len: int,
        sandy_input_len: Optional[int] = None,
        on_sandy: bool = False,
    ) -> FrankyChain:
        """
        Args:
            prompt_len: this engine's tokens of the block's prompt.
            sandy_input_len: on Ricky, Sandy's tokens of the block's prompt when
                Sandy tokenizes on its own.
            on_sandy: whether this engine is the chain's Sandy.
        Raises:
            ValueError: ``rid`` does not start with the chain id, so ending the
                chain would not reach the request; the block's prompt cannot hold
                the earlier blocks' tokens; or Sandy's block starts do not match
                the block's number, its order or Sandy's tokens.
        """
        chain = msgspec.convert(franky_chain, FrankyChain)
        if not rid.startswith(chain.id):
            raise ValueError(
                f"A block of Franky chain {chain.id!r} needs a rid that starts "
                f"with the chain id, not {rid!r}."
            )
        chain._check_sandy_block_starts(
            sandy_input_len=sandy_input_len, on_sandy=on_sandy
        )
        if on_sandy and chain.sandy_block_starts:
            # Sandy tokenizes on its own: its prompt ends with its own tokens of
            # the earlier blocks.
            if prompt_len < chain.sandy_block_starts[-1]:
                raise ValueError(
                    f"Block {chain.block} of Franky chain {chain.id!r} has a Sandy "
                    f"prompt of {prompt_len} tokens, too short for its earlier "
                    f"blocks, which start at {list(chain.sandy_block_starts)}."
                )
        elif prompt_len <= chain.num_generated_before:
            raise ValueError(
                f"Block {chain.block} of Franky chain {chain.id!r} has a prompt of "
                f"{prompt_len} tokens, too short for its earlier blocks' "
                f"{chain.num_generated_before} tokens after the chain's prompt."
            )
        return chain

    @property
    def continues(self) -> bool:
        return self.block > 1

    @property
    def num_generated_before(self) -> int:
        return (self.block - 1) * self.block_size

    def sandy_answer_start(self, *, sandy_input_len: int) -> int:
        """
        Args:
            sandy_input_len: Sandy's tokens of this block's prompt, when Sandy
                tokenizes on its own.
        Returns:
            Sandy's tokens before the chain's first block: before the answer.
        """
        if self.sandy_block_starts:
            return self.sandy_block_starts[0]
        return sandy_input_len

    def _check_sandy_block_starts(
        self, *, sandy_input_len: Optional[int], on_sandy: bool
    ) -> None:
        starts = self.sandy_block_starts
        if starts and len(starts) != self.block - 1:
            raise ValueError(
                f"Block {self.block} of Franky chain {self.id!r} has "
                f"{self.block - 1} earlier blocks, but sandy_block_starts names "
                f"{len(starts)}: {list(starts)}."
            )
        if any(lower > upper for lower, upper in zip(starts, starts[1:])):
            raise ValueError(
                f"sandy_block_starts of Franky chain {self.id!r} must be in order, "
                f"got {list(starts)}."
            )
        if on_sandy:
            return
        if sandy_input_len is None:
            if starts:
                raise ValueError(
                    "sandy_block_starts come with franky_sandy_input_ids: they "
                    "count the tokens of a Sandy that tokenizes on its own."
                )
            return
        if self.continues and not starts:
            raise ValueError(
                f"Block {self.block} of Franky chain {self.id!r} reads a Sandy that "
                "tokenizes on its own, so it needs Sandy's tokens before each "
                "earlier block in sandy_block_starts."
            )
        if starts and starts[-1] > sandy_input_len:
            raise ValueError(
                f"Block {self.block} of Franky chain {self.id!r} starts at Sandy's "
                f"token {sandy_input_len}, before its earlier blocks' starts "
                f"{list(starts)}."
            )


def put_continuations_first(requests: list[Req]) -> None:
    """Keeps the order within the continuing blocks and within the others."""
    requests.sort(key=_is_not_a_continuation)


def _is_not_a_continuation(req: Req) -> bool:
    request = req.franky_request
    return request is None or request.chain is None or not request.chain.continues
