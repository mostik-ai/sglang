from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import msgspec

from sglang.srt.franky.chains.messages import FrankyHoldExpired

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class _Hold(msgspec.Struct, kw_only=True):
    lock: Any
    num_tokens: int
    order: int
    expires_at: float
    announced: bool = False


class FrankyChainHolds:
    """Each chain's tree lock on its cached prefix between two of its blocks, at
    most one per chain. Taps live in the tap pages of their KV pages, so the lock
    keeps both. Every rank takes and releases the same holds at the same step;
    only expiry is the leader rank's to decide, from its own clock.
    """

    def __init__(
        self,
        *,
        tree_cache: BasePrefixCache,
        max_held_tokens: int,
        hold_seconds: float,
    ) -> None:
        self._tree_cache = tree_cache
        self._max_held_tokens = max_held_tokens
        self._hold_seconds = hold_seconds
        self._holds: dict[str, _Hold] = {}
        self._held_tokens = 0
        self._next_order = 0

    def hold(self, *, chain_id: str, node: Any, num_tokens: int) -> None:
        """Lock ``node`` for the chain's next block, in place of the chain's last
        hold. Over the cap, the oldest holds go.

        Args:
            num_tokens: the tokens on the path to ``node``.
        """
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

    def release(self, *, chain_id: str) -> None:
        hold = self._holds.pop(chain_id, None)
        if hold is None:
            return
        self._tree_cache.unlock(hold.lock)
        self._held_tokens -= hold.num_tokens

    def release_all(self) -> None:
        """Before the tree drops every node, as a cache flush does."""
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

    @property
    def held_tokens(self) -> int:
        return self._held_tokens
