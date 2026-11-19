from __future__ import annotations

from array import array
from collections.abc import Sequence
from typing import Optional

import msgspec
import numpy as np
import torch

from sglang.srt.franky.cache.tap_buffers import TapBuffers
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
    TreeLock,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.utils.common import ceil_div


class CachedTaps(msgspec.Struct, frozen=True, kw_only=True):
    """A cached prefix of one request's taps, locked until the request ends.

    Attributes:
        lock: the prefix tree's lock on the prefix.
        pages: int64 host, the local pages of the prefix's stripes in order.
        num_taps: the taps of the prefix, whole stripes.
    """

    lock: TreeLock
    pages: torch.Tensor
    num_taps: int


class TapReservation(msgspec.Struct, frozen=True, kw_only=True):
    """One request's stripes on Ricky, from its arrival to its end.

    Attributes:
        sandy_ids: the Sandy tokens whose taps the request reads.
        pages: int64 host, the local pages of its stripes in order, those of
            ``cached`` first.
        cached: the prefix it took from the prefix tree.
        extra_key: the request's cache namespace, which keys its taps beside
            ``sandy_ids``: Sandy may read a view of the request's tokens.
    """

    sandy_ids: Sequence[int]
    pages: torch.Tensor
    cached: CachedTaps
    extra_key: Optional[str] = None

    @property
    def num_taps(self) -> int:
        return len(self.sandy_ids)


class TapPoolUsage(msgspec.Struct, frozen=True, kw_only=True):
    """Ricky's tap pool in request tap pages over all attention-TP ranks.

    Attributes:
        pages: all of them.
        free_pages: the ones nothing holds.
        evictable_pages: the ones only the prefix tree keeps.
    """

    pages: int
    free_pages: int
    evictable_pages: int


class RickyTapCache:
    """Ricky's tap pool on this rank, on Ricky prefill and Ricky decode alike, and
    whichever tokenizer Sandy uses: taps count Sandy's tokens. The model's parts
    own the pages; this cache hands them to requests and keeps finished ones.

    Request page ``k`` holds the taps of Sandy's tokens ``[k * page_size,
    (k + 1) * page_size)``, and ``stripes`` says which rank keeps it. Each stripe
    of a request takes one local page, which every rank keeps at the same index,
    so a request's pages are the same on every rank as long as every rank takes
    and frees them at the same steps. A prefix tree over Sandy's token ids keeps
    the whole stripes that requests leave, for later requests that start with the
    same tokens; pages it keeps go back only when room is needed.
    """

    def __init__(self, *, buffers: TapBuffers, stripes: TapStripes) -> None:
        self.buffers = buffers
        self.stripes = stripes
        self.stripe_size = stripes.num_ranks * buffers.page_size
        self._pages = PagedTokenToKVPoolAllocator(
            size=(buffers.num_pages - 1) * self.stripe_size,
            page_size=self.stripe_size,
            dtype=torch.int64,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self._tree = RadixCache.create_simulated(
            mock_allocator=self._pages, page_size=self.stripe_size
        )
        self._stripe_offsets = torch.arange(self.stripe_size, dtype=torch.int64)

    def match(
        self, sandy_ids: Sequence[int], *, extra_key: Optional[str] = None
    ) -> CachedTaps:
        """
        Args:
            extra_key: the request's cache namespace.
        Returns:
            The longest cached prefix of the taps of ``sandy_ids``, locked.
        """
        result = self._tree.match_prefix(
            MatchPrefixParams(key=_key(sandy_ids, extra_key=extra_key))
        )
        node = result.last_device_node
        return CachedTaps(
            lock=self._tree.lock(node),
            pages=self._pages_of_path(node),
            num_taps=result.device_prefix_len,
        )

    def take(self, num_stripes: int) -> torch.Tensor:
        """
        Args:
            num_stripes: at most ``num_takeable_stripes()``.
        Returns:
            int64 host local pages of ``num_stripes`` new stripes, after evicting
            the cached stripes that no request holds as needed.
        """
        num_tokens = num_stripes * self.stripe_size
        shortfall = num_tokens - self._pages.available_size()
        if shortfall > 0:
            self._tree.evict(EvictParams(num_tokens=shortfall))
        indices = self._pages.alloc(num_tokens)
        return indices[:: self.stripe_size] // self.stripe_size

    def cache(self, reservation: TapReservation) -> TreeLock:
        """Leave a request's whole stripes in the prefix tree, as SGLang's tree
        caches a finished request's KV, and give back its other pages and its
        hold on what it took from the tree.

        Args:
            reservation: a request whose taps all landed.
        Returns:
            A lock on the cached stripes, for a holder that keeps them for a later
            request; ``release`` ends it.
        """
        num_whole = reservation.num_taps // self.stripe_size
        pages = reservation.pages
        values = (
            pages[:num_whole, None] * self.stripe_size + self._stripe_offsets
        ).reshape(-1)
        result = self._tree.insert(
            InsertParams(
                key=_key(
                    reservation.sandy_ids[: num_whole * self.stripe_size],
                    extra_key=reservation.extra_key,
                ),
                value=values,
            )
        )
        num_cached = reservation.cached.num_taps // self.stripe_size
        num_found = result.prefix_len // self.stripe_size
        self.free(pages[num_cached:num_found])
        self.free(pages[num_whole:])
        lock = self._tree.lock(result.last_device_node)
        self._tree.unlock(reservation.cached.lock)
        return lock

    def drop(self, reservation: TapReservation) -> None:
        """Give back a request's pages and its hold on what it took from the tree,
        caching nothing: its taps did not all land.
        """
        self.free(reservation.pages[len(reservation.cached.pages) :])
        self._tree.unlock(reservation.cached.lock)

    def free(self, pages: torch.Tensor) -> None:
        """Give back local pages that no tree node holds."""
        self._pages.free_page_ids(pages)

    def release(self, lock: TreeLock) -> None:
        """End a lock from ``match`` or ``cache``."""
        self._tree.unlock(lock)

    def pages_to_land(self, reservation: TapReservation) -> np.ndarray:
        """
        Returns:
            int32, per request page past the cached prefix, its local page here,
            or -1 where another rank keeps it: where its transfer lands.
        """
        page_size = self.buffers.page_size
        return self._local_pages(
            reservation.pages,
            first_page=reservation.cached.num_taps // page_size,
            num_pages=ceil_div(reservation.num_taps, page_size),
        )

    def pages_to_relay(self, reservation: TapReservation) -> np.ndarray:
        """
        Returns:
            int32, per request page, its local page here, or -1 where another
            rank keeps it: what this rank relays to Ricky decode.
        """
        return self._local_pages(
            reservation.pages,
            first_page=0,
            num_pages=ceil_div(reservation.num_taps, self.buffers.page_size),
        )

    def num_takeable_stripes(self) -> int:
        """Stripes that ``take`` can return: free ones and evictable ones."""
        tokens = self._pages.available_size() + self._tree.evictable_size()
        return tokens // self.stripe_size

    def usage(self) -> TapPoolUsage:
        pages_per_token = self.stripes.num_ranks / self.stripe_size
        return TapPoolUsage(
            pages=(self.buffers.num_pages - 1) * self.stripes.num_ranks,
            free_pages=int(self._pages.available_size() * pages_per_token),
            evictable_pages=int(self._tree.evictable_size() * pages_per_token),
        )

    def reset(self) -> None:
        """Drop every cached stripe; no request may hold any."""
        self._tree.reset()
        self._pages.clear()

    def _local_pages(
        self, pages: torch.Tensor, *, first_page: int, num_pages: int
    ) -> np.ndarray:
        request_pages = np.arange(first_page, num_pages)
        stripe_pages = pages.numpy()[self.stripes.stripe(request_pages)]
        return np.where(self.stripes.owns(request_pages), stripe_pages, -1).astype(
            np.int32
        )

    def _pages_of_path(self, node: object) -> torch.Tensor:
        indices = self._tree.path_device_indices(node)
        return indices[:: self.stripe_size] // self.stripe_size


def _key(sandy_ids: Sequence[int], *, extra_key: Optional[str]) -> RadixKey:
    return RadixKey(token_ids=array("q", sandy_ids), extra_key=extra_key)
