from __future__ import annotations

from typing import Any

import msgspec


class FrankyChain(msgspec.Struct, frozen=True, kw_only=True):
    """The chain of blocks one client request runs as: the router sends each
    block as a request whose rid starts with ``id``. After a block that is not
    ``last_block``, each engine holds the chain's cached prefix for the next one.
    """

    id: str
    last_block: bool = False

    @staticmethod
    def from_request(*, franky_chain: dict[str, Any], rid: str) -> FrankyChain:
        """
        Raises:
            ValueError: ``rid`` does not start with the chain id, so ending the
                chain would not reach the request.
        """
        chain = msgspec.convert(franky_chain, FrankyChain)
        if not chain.id or not rid.startswith(chain.id):
            raise ValueError(
                f"A block of Franky chain {chain.id!r} needs a rid that starts "
                f"with the chain id, not {rid!r}."
            )
        return chain
