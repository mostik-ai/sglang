from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Optional

import msgspec

from sglang.srt.franky.chains.blocks.config import FrankyBlocksConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class FrankyChain(msgspec.Struct, frozen=True, kw_only=True):
    """The chain of Sandy calls and Ricky legs a client request runs as, seen
    from one of its requests. Each engine holds what a chain's request left for
    the chain's next request, until the router ends the chain.

    Attributes:
        id: the chain's id; the rid of each of its requests starts with it.
        block: from 1, the number of the block a Ricky leg writes first, or of
            the block that follows a Sandy call.
        blocks: on a Ricky leg, how Ricky ends its blocks and decides whether
            Sandy writes next; None on a Sandy call.
    """

    id: Annotated[str, msgspec.Meta(min_length=1)]
    block: Annotated[int, msgspec.Meta(ge=1)] = 1
    blocks: Optional[FrankyBlocksConfig] = None

    @staticmethod
    def from_request(*, franky_chain: dict[str, Any], rid: str) -> FrankyChain:
        """
        Raises:
            ValueError: the chain is malformed, or ``rid`` does not start with the
                chain id, so ending the chain would not reach the request.
        """
        chain = msgspec.convert(franky_chain, FrankyChain)
        if not rid.startswith(chain.id):
            raise ValueError(
                f"A request of Franky chain {chain.id!r} needs a rid that starts "
                f"with the chain id, not {rid!r}."
            )
        return chain

    @property
    def continues(self) -> bool:
        return self.block > 1


def put_continuations_first(requests: list[Req]) -> None:
    """Keeps the order within the continuing requests and within the others."""
    requests.sort(key=_is_not_a_continuation)


def _is_not_a_continuation(req: Req) -> bool:
    request = req.franky
    return request is None or request.chain is None or not request.chain.continues
