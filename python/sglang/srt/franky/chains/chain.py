from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import msgspec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class FrankyChain(msgspec.Struct, frozen=True, kw_only=True):
    """The chain of blocks a client request runs as, seen from one of its blocks.

    Args:
        id: the chain's id; the rid of each of its blocks starts with it.
        block: this block's number, from 1.
        last_block: no block follows this one, so the engines hold nothing after it.
    """

    id: Annotated[str, msgspec.Meta(min_length=1)]
    block: Annotated[int, msgspec.Meta(ge=1)] = 1
    last_block: bool = False

    @staticmethod
    def from_request(*, franky_chain: dict[str, Any], rid: str) -> FrankyChain:
        """
        Raises:
            ValueError: ``rid`` does not start with the chain id, so ending the
                chain would not reach the request.
        """
        chain = msgspec.convert(franky_chain, FrankyChain)
        if not rid.startswith(chain.id):
            raise ValueError(
                f"A block of Franky chain {chain.id!r} needs a rid that starts "
                f"with the chain id, not {rid!r}."
            )
        return chain

    @property
    def continues(self) -> bool:
        return self.block > 1


def put_continuations_first(requests: list[Req]) -> None:
    """Keeps the order within the continuing blocks and within the others."""
    requests.sort(key=_is_not_a_continuation)


def _is_not_a_continuation(req: Req) -> bool:
    return req.franky_chain is None or not req.franky_chain.continues
