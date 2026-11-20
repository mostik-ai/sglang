from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

import msgspec

from sglang.srt.franky.chains.messages import FrankyHoldExpired
from sglang.srt.franky.request import FrankyRequest
from sglang.srt.runtime_context import get_franky

if TYPE_CHECKING:
    from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, TreeLock

logger = logging.getLogger(__name__)


class _Hold(msgspec.Struct, kw_only=True):
    kv_lock: Optional[TreeLock]
    tap_lock: Optional[TreeLock]
    block: int
    num_tokens: int
    order: int
    expires_at: float
    announced: bool = False


class FrankyChainHolds:
    """Each chain's locks on what its last block left in the prefix trees,
    between two of its blocks: the KV prefix tree's lock on the block's prompt,
    and on Ricky the tap prefix tree's lock on its taps. A chain between two of
    its blocks is in flight, as a decoding request is: nothing caps what holds
    keep, and a new request waits for room instead of taking a hold's. All ranks
    hold and release alike; only the leader rank decides expiry.
    """

    def __init__(
        self,
        *,
        tree_cache: Optional[BasePrefixCache],
        tap_cache: Optional[RickyTapCache],
        hold_seconds: float,
    ) -> None:
        """
        Args:
            tree_cache: None on an engine that caches no prompts, such as PD
                decode.
            tap_cache: None on Sandy, whose taps live with its KV.
        """
        self._tree_cache = tree_cache
        self._tap_cache = tap_cache
        self._hold_seconds = hold_seconds
        self._holds: dict[str, _Hold] = {}
        self._next_order = 0

    @staticmethod
    def from_config(
        *,
        tree_cache: Optional[BasePrefixCache],
        tap_cache: Optional[RickyTapCache] = None,
    ) -> FrankyChainHolds:
        """The holds of an engine, as ``--franky-chain-hold-seconds`` says."""
        return FrankyChainHolds(
            tree_cache=tree_cache,
            tap_cache=tap_cache,
            hold_seconds=get_franky().franky_chain_hold_seconds,
        )

    def hold_for_next_block(
        self, req: Req, *, tap_lock: Optional[TreeLock] = None
    ) -> None:
        """``req`` ended well: lock what it left for its chain's next request,
        in place of the chain's last hold. Only the router knows when a chain
        ends; it ends the chain on every engine.

        Args:
            tap_lock: the tap prefix tree's lock on the taps ``req`` left, which
                the holds own from now on.
        """
        chain = FrankyRequest.of(req).chain
        if chain is None:
            self._release_tap_lock(tap_lock)
            return
        self.release(chain_id=chain.id)
        self._holds[chain.id] = self._new_hold(req, tap_lock=tap_lock)

    def holds_chain_of(self, req: Req) -> bool:
        chain = FrankyRequest.of(req).chain
        return chain is not None and chain.id in self._holds

    def put_held_chains_first(self, requests: list[Req]) -> None:
        """Keeps the order within the blocks of held chains and within the others."""
        requests.sort(key=self._is_not_held)

    def release_chain_of(self, req: Req) -> None:
        chain = FrankyRequest.of(req).chain
        if chain is not None:
            self.release(chain_id=chain.id)

    def release(self, *, chain_id: str) -> None:
        hold = self._holds.pop(chain_id, None)
        if hold is None:
            return
        if hold.kv_lock is not None:
            self._tree_cache.unlock(hold.kv_lock)
        self._release_tap_lock(hold.tap_lock)

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

    def locked_nodes(self) -> set[Any]:
        """The KV prefix tree nodes the holds lock, each once."""
        return {
            hold.kv_lock.node
            for hold in self._holds.values()
            if hold.kv_lock is not None
        }

    def _new_hold(self, req: Req, *, tap_lock: Optional[TreeLock]) -> _Hold:
        hold = _Hold(
            kv_lock=(
                None
                if self._tree_cache is None
                else self._tree_cache.lock(req.last_node)
            ),
            tap_lock=tap_lock,
            block=FrankyRequest.of(req).chain.block,
            num_tokens=len(req.origin_input_ids),
            order=self._next_order,
            expires_at=time.monotonic() + self._hold_seconds,
        )
        self._next_order += 1
        return hold

    def _release_tap_lock(self, tap_lock: Optional[TreeLock]) -> None:
        if tap_lock is not None:
            self._tap_cache.release(tap_lock)

    def _is_not_held(self, req: Req) -> bool:
        return not self.holds_chain_of(req)
