from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch

from sglang.srt.franky.cache.ricky_tap_cache import CachedTaps
from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
    from sglang.srt.franky.chains.holds import FrankyChainHolds
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import TreeLock


class TapReservation(msgspec.Struct, frozen=True, kw_only=True):
    """One request's stripes on Ricky, from its arrival to its end.

    Attributes:
        sandy_ids: the Sandy tokens whose taps the request reads.
        slots: int64 host stripe slots in order, those of ``cached`` first.
        cached: the prefix the request took from the prefix tree.
    """

    sandy_ids: Sequence[int]
    slots: torch.Tensor
    cached: CachedTaps

    @property
    def num_taps(self) -> int:
        return len(self.sandy_ids)


class _Waiting(msgspec.Struct, frozen=True, kw_only=True):
    req: Any
    sandy_ids: Sequence[int]


class FrankyTapReservations:
    """The stripes of Ricky's requests that read taps, by a key the caller
    picks, from arrival until the request ends.

    A request takes its stripes in arrival order, the blocks of held chains
    first: the cached prefix of its taps from the prefix tree, and new slots for
    the rest, of which at least one page lands, since a transfer ends only once
    a page lands. It waits until they fit; only a held chain's next block may
    release other chains' holds for room. At its end a request whose taps all
    landed leaves its whole stripes in the prefix tree, where a held block keeps
    them locked for its chain's next block. Every rank decides at the same steps,
    so stripe slots stay the same on every rank.
    """

    def __init__(
        self,
        *,
        tap_cache: RickyTapCache,
        chain_holds: FrankyChainHolds,
        max_taps: int,
    ) -> None:
        """
        Args:
            max_taps: the most taps one request may read.
        """
        self._tap_cache = tap_cache
        self._chain_holds = chain_holds
        self._max_taps = max_taps
        self._waiting: dict[Hashable, _Waiting] = {}
        self._reserved: dict[Hashable, TapReservation] = {}
        self._refused: dict[Hashable, _Waiting] = {}

    def expect(self, key: Hashable, *, req: Req, sandy_ids: Sequence[int]) -> None:
        """``req`` arrived and reads the taps of ``sandy_ids``, at least one; it
        waits for its stripes, or is refused when it could never take them.
        """
        waiting = _Waiting(req=req, sandy_ids=sandy_ids)
        if len(sandy_ids) > self._max_taps:
            self._refused[key] = waiting
            return
        self._waiting[key] = waiting

    def refusal(self, key: Hashable) -> Optional[str]:
        """Why the request of ``key`` was refused, or None."""
        refused = self._refused.get(key)
        if refused is None:
            return None
        return (
            f"Sandy's {len(refused.sandy_ids)} tokens of request {refused.req.rid} "
            f"are past the {self._max_taps} taps a request may read on Ricky."
        )

    def poll(self) -> list[tuple[Hashable, Req]]:
        """Every rank, at each step: reserve the waiting requests in order while
        they fit.

        Returns:
            The keys and requests reserved at this step, in order.
        """
        self._forget_finished()
        reserved = []
        for key in sorted(self._waiting, key=self._is_not_of_a_held_chain):
            waiting = self._waiting[key]
            reservation = self._reserve(waiting)
            if reservation is None:
                break
            del self._waiting[key]
            self._reserved[key] = reservation
            reserved.append((key, waiting.req))
        return reserved

    def of(self, key: Hashable) -> Optional[TapReservation]:
        return self._reserved.get(key)

    def waits(self, key: Hashable) -> bool:
        return key in self._waiting

    def forget(self, key: Hashable) -> None:
        """A request that waits or was refused ended before it reserved anything."""
        self._waiting.pop(key, None)
        self._refused.pop(key, None)

    def end(
        self, key: Hashable, *, landed: bool, keep_locked: bool = False
    ) -> Optional[TreeLock]:
        """A reserved request ended, and no transfer writes its slots any more.

        Args:
            landed: whether all of its taps landed; only then the prefix tree
                keeps its whole stripes.
            keep_locked: lock what the prefix tree kept for a later request.
        Returns:
            With ``keep_locked`` and ``landed``, the lock on the kept stripes,
            which the caller ends through ``RickyTapCache.release``; else None.
        """
        reservation = self._reserved.pop(key)
        if not landed:
            self._tap_cache.free(reservation.slots[len(reservation.cached.slots) :])
            self._tap_cache.release(reservation.cached.lock)
            return None
        lock = self._tap_cache.keep(
            sandy_ids=reservation.sandy_ids,
            slots=reservation.slots,
            cached=reservation.cached,
        )
        if keep_locked:
            return lock
        self._tap_cache.release(lock)
        return None

    def num_reserved_slots(self) -> int:
        return sum(len(reservation.slots) for reservation in self._reserved.values())

    def is_empty(self) -> bool:
        return not self._waiting and not self._reserved and not self._refused

    def _forget_finished(self) -> None:
        for requests in (self._waiting, self._refused):
            for key in [
                key for key, waiting in requests.items() if waiting.req.finished()
            ]:
                del requests[key]

    def _reserve(self, waiting: _Waiting) -> Optional[TapReservation]:
        cached = self._tap_cache.match(
            waiting.sandy_ids[: self._max_cached_taps(waiting)]
        )
        num_new = self._num_stripes(len(waiting.sandy_ids)) - len(cached.slots)
        if not self._fits_or_releases_holds(waiting.req, num_new):
            self._tap_cache.release(cached.lock)
            return None
        new_slots = self._tap_cache.take(num_new)
        if new_slots is None:
            raise RuntimeError(
                f"Franky could not take {num_new} stripe slots that fit: "
                f"{self._tap_cache.num_free_slots()} free."
            )
        return TapReservation(
            sandy_ids=waiting.sandy_ids,
            slots=torch.cat([cached.slots, new_slots]),
            cached=cached,
        )

    def _fits_or_releases_holds(self, req: Req, num_new: int) -> bool:
        if not self._chain_holds.holds_chain_of(req):
            return self._tap_cache.num_free_slots() >= num_new
        while self._tap_cache.num_free_slots() < num_new:
            if not self._chain_holds.release_least_progress(
                sparing_chain_id=req.franky_chain.id
            ):
                return False
        return True

    def _max_cached_taps(self, waiting: _Waiting) -> int:
        page_size = self._tap_cache.buffers.page_size
        return (ceil_div(len(waiting.sandy_ids), page_size) - 1) * page_size

    def _num_stripes(self, num_taps: int) -> int:
        return ceil_div(num_taps, self._tap_cache.stripe_size)

    def _is_not_of_a_held_chain(self, key: Hashable) -> bool:
        return not self._chain_holds.holds_chain_of(self._waiting[key].req)
