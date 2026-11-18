from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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
    block: int
    num_tokens: int
    order: int
    expires_at: float
    announced: bool = False
    taps: Optional[FrankyHeldTaps] = None


class FrankyChainHolds:
    """Each chain's tree lock on its cached prompt (KV and taps) between two of its
    blocks, and on a Ricky whose Sandy tokenizes on its own, the tap pages its last
    block owned. A chain between two of its blocks is in flight, as a decoding
    request is: nothing caps what holds keep, and a new request waits for room
    instead of taking a hold's. All ranks hold and release alike; only the leader
    rank decides expiry.
    """

    def __init__(
        self,
        *,
        tree_cache: Optional[BasePrefixCache],
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
        self._hold_seconds = hold_seconds
        self._own_pages = own_pages
        self._holds: dict[str, _Hold] = {}
        self._next_order = 0

    @staticmethod
    def from_config(
        *,
        tree_cache: Optional[BasePrefixCache],
        own_pages: Optional[FrankyOwnTapPages] = None,
    ) -> FrankyChainHolds:
        """The holds of an engine, as ``--franky-chain-hold-seconds`` says."""
        return FrankyChainHolds(
            tree_cache=tree_cache,
            hold_seconds=get_franky().franky_chain_hold_seconds,
            own_pages=own_pages,
        )

    def hold_for_next_block(
        self, req: Req, *, taps: Optional[FrankyHeldTaps] = None
    ) -> None:
        """Lock the cached prompt of ``req`` for its chain's next block, in place of
        the chain's last hold. A last block releases the chain's hold instead.

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
        self._holds[chain.id] = self._new_hold(req, taps=taps)

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

    def holds_chain_of(self, req: Req) -> bool:
        return req.franky_chain is not None and req.franky_chain.id in self._holds

    def put_held_chains_first(self, requests: list[Req]) -> None:
        """Keeps the order within the blocks of held chains and within the others."""
        requests.sort(key=self._is_not_held)

    def release_chain_of(self, req: Req) -> None:
        if req.franky_chain is not None:
            self.release(chain_id=req.franky_chain.id)

    def release(self, *, chain_id: str) -> None:
        hold = self._holds.pop(chain_id, None)
        if hold is None:
            return
        if hold.lock is not None:
            self._tree_cache.unlock(hold.lock)
        self._give_back(hold.taps)

    def release_all(self) -> None:
        for chain_id in list(self._holds):
            self.release(chain_id=chain_id)

    def release_least_progress(self, *, sparing_chain_id: Optional[str]) -> bool:
        """Release one hold so that a held chain's next block fits, as SGLang
        retracts decoding requests: the chain with the fewest blocks done goes
        first, then the one that keeps the most tokens.

        Args:
            sparing_chain_id: the chain whose next block needs the room.
        Returns:
            Whether there was a hold to release.
        """
        candidates = [
            (hold.block, -hold.num_tokens, chain_id)
            for chain_id, hold in self._holds.items()
            if chain_id != sparing_chain_id
        ]
        if not candidates:
            return False
        block, _, chain_id = min(candidates)
        logger.info(
            "Franky released the hold of chain %s after block %d, so that the next "
            "block of chain %s fits.",
            chain_id,
            block,
            sparing_chain_id,
        )
        self.release(chain_id=chain_id)
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

    def num_held_tap_pages(self) -> int:
        return sum(
            len(hold.taps.pages)
            for hold in self._holds.values()
            if hold.taps is not None
        )

    def locked_nodes(self) -> set[Any]:
        """The tree nodes the holds lock, each once."""
        return {
            hold.lock.node for hold in self._holds.values() if hold.lock is not None
        }

    def _new_hold(self, req: Req, *, taps: Optional[FrankyHeldTaps]) -> _Hold:
        hold = _Hold(
            lock=(
                None
                if self._tree_cache is None
                else self._tree_cache.lock(req.last_node)
            ),
            block=req.franky_chain.block,
            num_tokens=self._num_held_tokens(req, taps=taps),
            order=self._next_order,
            expires_at=time.monotonic() + self._hold_seconds,
            taps=taps,
        )
        self._next_order += 1
        return hold

    def _num_held_tokens(self, req: Req, *, taps: Optional[FrankyHeldTaps]) -> int:
        if self._tree_cache is not None:
            return len(req.origin_input_ids)
        return 0 if taps is None else taps.num_taps

    def _is_not_held(self, req: Req) -> bool:
        return not self.holds_chain_of(req)

    def _give_back(self, taps: Optional[FrankyHeldTaps]) -> None:
        if taps is not None:
            self._own_pages.free(taps.pages)
