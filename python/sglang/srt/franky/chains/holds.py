from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch

from sglang.srt.franky.chains.messages import FrankyHoldExpired
from sglang.srt.runtime_context import get_franky

if TYPE_CHECKING:
    from sglang.srt.franky.cache.own_tap_pages import FrankyOwnTapPages
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, TreeLock


class FrankyHeldTaps(msgspec.Struct, frozen=True, kw_only=True):
    """The tap pages a block of a chain owned, kept for the chain's next block,
    whose Sandy input extends this block's.

    Attributes:
        pages: int32 host tap pages of Sandy's tokens from token 0, in order.
        num_taps: Sandy's tokens whose taps the pages hold: the block's Sandy input.
    """

    pages: torch.Tensor
    num_taps: int

    def num_reused_taps(self, *, num_pages: int, page_size: int) -> int:
        """
        Args:
            num_pages: the tap pages of the next block's Sandy input.
        Returns:
            The taps of the whole pages the next block reads here instead of
            receiving them again. Its last page lands again, since a transfer
            room ends only once a page lands.
        """
        return min(self.num_taps // page_size, num_pages - 1) * page_size


class _Hold(msgspec.Struct, kw_only=True):
    lock: Optional[TreeLock]
    num_tokens: int
    order: int
    expires_at: float
    announced: bool = False
    taps: Optional[FrankyHeldTaps] = None


class FrankyChainHolds:
    """Each chain's tree lock on its cached prompt (KV and taps) between two of its
    blocks, and on a Ricky whose Sandy tokenizes on its own, the tap pages its last
    block owned. All ranks hold and release alike; only the leader rank decides
    expiry.
    """

    def __init__(
        self,
        *,
        tree_cache: Optional[BasePrefixCache],
        max_held_tokens: int,
        hold_seconds: float,
        own_pages: Optional[FrankyOwnTapPages] = None,
    ) -> None:
        """
        Args:
            tree_cache: None on an engine that caches no prompts, such as PD
                decode; its holds keep tap pages only.
            own_pages: where held tap pages go back when their hold ends; None on
                an engine whose requests own no tap pages.
        """
        self._tree_cache = tree_cache
        self._max_held_tokens = max_held_tokens
        self._hold_seconds = hold_seconds
        self._own_pages = own_pages
        self._holds: dict[str, _Hold] = {}
        self._held_tokens = 0
        self._next_order = 0

    @staticmethod
    def from_config(
        *,
        tree_cache: Optional[BasePrefixCache],
        max_total_num_tokens: int,
        own_pages: Optional[FrankyOwnTapPages] = None,
    ) -> FrankyChainHolds:
        """The holds of an engine whose KV pool has ``max_total_num_tokens``, as
        the ``--franky-chain-*`` flags say."""
        franky = get_franky()
        return FrankyChainHolds(
            tree_cache=tree_cache,
            max_held_tokens=int(franky.franky_chain_hold_ratio * max_total_num_tokens),
            hold_seconds=franky.franky_chain_hold_seconds,
            own_pages=own_pages,
        )

    def hold_for_next_block(
        self, req: Req, *, taps: Optional[FrankyHeldTaps] = None
    ) -> None:
        """Lock the cached prompt of ``req`` for its chain's next block, in place of
        the chain's last hold; past the cap, the oldest holds go. A last block
        releases the chain's hold instead.

        Args:
            taps: the tap pages ``req`` owned, which the holds own from now on:
                they keep them for the chain's next block or give them back.
        """
        chain = req.franky_chain
        if chain is None or chain.last_block:
            self.release_chain_of(req)
            self._give_back(taps)
            return
        self.release(chain_id=chain.id)
        self._hold(chain_id=chain.id, hold=self._new_hold(req, taps=taps))

    def take_taps(self, *, chain_id: str) -> Optional[FrankyHeldTaps]:
        """
        Returns:
            The tap pages the chain's hold keeps, which the caller owns from now
            on; None when it keeps none.
        """
        hold = self._holds.get(chain_id)
        if hold is None or hold.taps is None:
            return None
        taps, hold.taps = hold.taps, None
        return taps

    def release_chain_of(self, req: Req) -> None:
        if req.franky_chain is not None:
            self.release(chain_id=req.franky_chain.id)

    def release(self, *, chain_id: str) -> None:
        hold = self._holds.pop(chain_id, None)
        if hold is None:
            return
        if hold.lock is not None:
            self._tree_cache.unlock(hold.lock)
        self._held_tokens -= hold.num_tokens
        self._give_back(hold.taps)

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

    def release_newest_taps(self) -> bool:
        """Give back the tap pages of the newest hold that keeps some; its prompt
        stays locked. Chains come back in the order their blocks ended, so its
        taps are the last ones a block would read again.

        Returns:
            Whether a hold kept tap pages.
        """
        for hold in reversed(self._holds.values()):
            if hold.taps is not None:
                self._give_back(hold.taps)
                hold.taps = None
                return True
        return False

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

    def locked_nodes(self) -> set[Any]:
        """The tree nodes the holds lock, each once."""
        return {
            hold.lock.node for hold in self._holds.values() if hold.lock is not None
        }

    def _new_hold(self, req: Req, *, taps: Optional[FrankyHeldTaps]) -> _Hold:
        caches_prompts = self._tree_cache is not None
        hold = _Hold(
            lock=self._tree_cache.lock(req.last_node) if caches_prompts else None,
            num_tokens=len(req.origin_input_ids) if caches_prompts else 0,
            order=self._next_order,
            expires_at=time.monotonic() + self._hold_seconds,
            taps=taps,
        )
        self._next_order += 1
        return hold

    def _hold(self, *, chain_id: str, hold: _Hold) -> None:
        self._holds[chain_id] = hold
        self._held_tokens += hold.num_tokens
        while self._held_tokens > self._max_held_tokens:
            self.release_oldest()

    def _give_back(self, taps: Optional[FrankyHeldTaps]) -> None:
        if taps is not None:
            self._own_pages.free(taps.pages)

    @property
    def held_tokens(self) -> int:
        return self._held_tokens

    @property
    def held_tap_pages(self) -> int:
        return sum(
            len(hold.taps.pages)
            for hold in self._holds.values()
            if hold.taps is not None
        )
