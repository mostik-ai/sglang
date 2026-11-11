from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

import msgspec

from sglang.srt.franky.chains.messages import FrankyHoldExpired
from sglang.srt.runtime_context import get_franky

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, TreeLock


class _Hold(msgspec.Struct, kw_only=True):
    lock: Optional[TreeLock]
    num_tokens: int
    order: int
    expires_at: float
    announced: bool = False


class FrankyChainHolds:
    """Each chain's tree lock on its cached prompt (KV and taps) between two of its
    blocks. All ranks hold and release alike; only the leader rank decides expiry.
    """

    def __init__(
        self,
        *,
        tree_cache: BasePrefixCache,
        max_held_tokens: int,
        hold_seconds: float,
        chunked_prefill_size: int,
    ) -> None:
        self._tree_cache = tree_cache
        self._max_held_tokens = max_held_tokens
        self._hold_seconds = hold_seconds
        self._chunked_prefill_size = chunked_prefill_size
        self._holds: dict[str, _Hold] = {}
        self._held_tokens = 0
        self._next_order = 0

    @staticmethod
    def from_config(
        *, tree_cache: BasePrefixCache, max_total_num_tokens: int
    ) -> FrankyChainHolds:
        """The holds of an engine whose KV pool has ``max_total_num_tokens``, as
        the ``--franky-chain-*`` flags say."""
        franky = get_franky()
        return FrankyChainHolds(
            tree_cache=tree_cache,
            max_held_tokens=int(franky.franky_chain_hold_ratio * max_total_num_tokens),
            hold_seconds=franky.franky_chain_hold_seconds,
            chunked_prefill_size=franky.franky_chain_chunked_prefill_size,
        )

    def hold_for_next_block(self, req: Req) -> None:
        """Lock the cached prompt of ``req`` for its chain's next block, in place of
        the chain's last hold; past the cap, the oldest holds go. A last block
        releases the chain's hold instead."""
        chain = req.franky_chain
        if chain is None:
            return
        if chain.last_block:
            self.release(chain_id=chain.id)
            return
        self._hold(
            chain_id=chain.id,
            node=req.last_node,
            num_tokens=len(req.origin_input_ids),
        )

    def release_chain_of(self, req: Req) -> None:
        if req.franky_chain is not None:
            self.release(chain_id=req.franky_chain.id)

    def release(self, *, chain_id: str) -> None:
        hold = self._holds.pop(chain_id, None)
        if hold is None:
            return
        self._tree_cache.unlock(hold.lock)
        self._held_tokens -= hold.num_tokens

    def release_all(self) -> None:
        for chain_id in list(self._holds):
            self.release(chain_id=chain_id)

    def release_oldest(self) -> bool:
        """
        Returns:
            Whether there was a hold to release.
        """
        if not self._holds:
            return False
        self.release(chain_id=next(iter(self._holds)))
        return True

    def expired_messages(self) -> list[FrankyHoldExpired]:
        """The leader rank only: the holds past their expiry, each once."""
        now = time.monotonic()
        messages = []
        for chain_id, hold in self._holds.items():
            if hold.expires_at <= now and not hold.announced:
                hold.announced = True
                messages.append(FrankyHoldExpired(chain_id=chain_id, order=hold.order))
        return messages

    def on_hold_expired(self, *, message: FrankyHoldExpired) -> None:
        hold = self._holds.get(message.chain_id)
        if hold is not None and hold.order == message.order:
            self.release(chain_id=message.chain_id)

    def chunk_size_for_batch(
        self, chunked_prefill_size: Optional[int], *, num_blocks_in_flight: int = 0
    ) -> Optional[int]:
        """The engine's chunk size for its next prefill batch, cut to the chains'
        chunk size while any chain holds its prompt here or has a later block in
        flight here (``num_blocks_in_flight``); no chunking stays."""
        if chunked_prefill_size is None or not (self._holds or num_blocks_in_flight):
            return chunked_prefill_size
        return min(chunked_prefill_size, self._chunked_prefill_size)

    def locked_nodes(self) -> set[Any]:
        """The tree nodes the holds lock, each once."""
        return {
            hold.lock.node for hold in self._holds.values() if hold.lock is not None
        }

    def _hold(self, *, chain_id: str, node: Any, num_tokens: int) -> None:
        self.release(chain_id=chain_id)
        self._holds[chain_id] = _Hold(
            lock=self._tree_cache.lock(node),
            num_tokens=num_tokens,
            order=self._next_order,
            expires_at=time.monotonic() + self._hold_seconds,
        )
        self._next_order += 1
        self._held_tokens += num_tokens
        while self._held_tokens > self._max_held_tokens:
            self.release_oldest()

    @property
    def held_tokens(self) -> int:
        return self._held_tokens
