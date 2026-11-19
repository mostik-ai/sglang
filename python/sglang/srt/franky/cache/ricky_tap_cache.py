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


class CachedTaps(msgspec.Struct, frozen=True, kw_only=True):
    """A cached prefix of one request's taps, locked until the request ends.

    Attributes:
        lock: the prefix tree's lock on the prefix.
        slots: int64 host, the stripe slots of the prefix in order.
        num_taps: the taps of the prefix, whole stripes.
    """

    lock: TreeLock
    slots: torch.Tensor
    num_taps: int


class RickyTapCache:
    """Ricky's taps on this rank, on Ricky prefill and Ricky decode alike, and
    whichever tokenizer Sandy uses: taps count Sandy's tokens.

    Request page ``k`` holds the taps of Sandy's tokens ``[k * page_size,
    (k + 1) * page_size)``, and ``stripes`` says which rank keeps it. Each stripe
    of a request takes one stripe slot, which every rank keeps at the same local
    page, so a request's slots are the same on every rank as long as every rank
    takes and frees them at the same steps. A prefix tree over Sandy's token ids
    keeps the whole stripes that requests leave, for later requests that start
    with the same tokens; slots it keeps go back only when room is needed.
    """

    def __init__(self, *, buffers: TapBuffers, stripes: TapStripes) -> None:
        self.buffers = buffers
        self.stripes = stripes
        self.stripe_size = stripes.num_ranks * buffers.page_size
        self._slots = PagedTokenToKVPoolAllocator(
            size=(buffers.num_pages - 1) * self.stripe_size,
            page_size=self.stripe_size,
            dtype=torch.int64,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self._tree = RadixCache.create_simulated(
            mock_allocator=self._slots, page_size=self.stripe_size
        )
        self._stripe_offsets = torch.arange(self.stripe_size, dtype=torch.int64)

    @staticmethod
    def build(
        *,
        encoder_ids: tuple[int, ...],
        output_size: int,
        page_size: int,
        num_taps: int,
        stripes: TapStripes,
        device: str,
    ) -> RickyTapCache:
        """
        Args:
            num_taps: the taps the cache holds over all ranks, whole stripes.
        """
        num_slots = num_taps // (stripes.num_ranks * page_size)
        return RickyTapCache(
            buffers=TapBuffers(
                encoder_ids=encoder_ids,
                output_size=output_size,
                page_size=page_size,
                num_pages=num_slots + 1,
                device=device,
            ),
            stripes=stripes,
        )

    def match(self, sandy_ids: Sequence[int]) -> CachedTaps:
        """
        Returns:
            The longest cached prefix of the taps of ``sandy_ids``, locked.
        """
        result = self._tree.match_prefix(MatchPrefixParams(key=_key(sandy_ids)))
        node = result.last_device_node
        return CachedTaps(
            lock=self._tree.lock(node),
            slots=self._slots_of_path(node),
            num_taps=result.device_prefix_len,
        )

    def take(self, num_stripes: int) -> Optional[torch.Tensor]:
        """
        Returns:
            int64 host slots of ``num_stripes`` new stripes, after evicting the
            cached stripes that no request holds as needed; None when they do
            not fit.
        """
        num_slots_tokens = num_stripes * self.stripe_size
        shortfall = num_slots_tokens - self._slots.available_size()
        if shortfall > 0:
            self._tree.evict(EvictParams(num_tokens=shortfall))
        indices = self._slots.alloc(num_slots_tokens)
        if indices is None:
            return None
        return indices[:: self.stripe_size] // self.stripe_size

    def keep(
        self, *, sandy_ids: Sequence[int], slots: torch.Tensor, cached: CachedTaps
    ) -> TreeLock:
        """Leave a request's whole stripes in the prefix tree and give back its
        other slots and its hold on ``cached``.

        Args:
            sandy_ids: the Sandy tokens whose taps all landed in ``slots``.
            slots: the request's stripe slots, those of ``cached`` first.
            cached: what the request took from the tree at its arrival.
        Returns:
            A lock on the kept stripes, for a holder that keeps them for a later
            request; ``release`` ends it.
        """
        num_whole = len(sandy_ids) // self.stripe_size
        values = (
            slots[:num_whole, None] * self.stripe_size + self._stripe_offsets
        ).reshape(-1)
        result = self._tree.insert(
            InsertParams(
                key=_key(sandy_ids[: num_whole * self.stripe_size]), value=values
            )
        )
        num_cached = cached.num_taps // self.stripe_size
        num_found = result.prefix_len // self.stripe_size
        self.free(slots[num_cached:num_found])
        self.free(slots[num_whole:])
        lock = self._tree.lock(result.last_device_node)
        self._tree.unlock(cached.lock)
        return lock

    def local_pages(
        self, *, slots: torch.Tensor, first_page: int, num_pages: int
    ) -> np.ndarray:
        """
        Args:
            slots: a request's stripe slots in order.
            first_page, num_pages: the request's pages ``[first_page, num_pages)``.
        Returns:
            int32, per page in that range, its local page on this rank, which is
            its stripe's slot, or -1 where another rank keeps it.
        """
        pages = np.arange(first_page, num_pages)
        stripe_slots = slots.numpy()[self.stripes.stripe(pages)]
        return np.where(self.stripes.keeps(pages), stripe_slots, -1).astype(np.int32)

    def free(self, slots: torch.Tensor) -> None:
        """Give back stripe slots that no tree node holds."""
        self._slots.free_page_ids(slots)

    def release(self, lock: Optional[TreeLock]) -> None:
        """End a lock from ``match`` or ``keep``."""
        self._tree.unlock(lock)

    def num_free_slots(self) -> int:
        """Stripe slots that ``take`` can return: free ones and evictable ones."""
        free_tokens = self._slots.available_size() + self._tree.evictable_size()
        return free_tokens // self.stripe_size

    def num_free_pages(self) -> tuple[int, int]:
        """
        Returns:
            The request pages of free slots, and of slots that only the prefix
            tree holds, which it gives back when room is needed.
        """
        pages_per_token = self.stripes.num_ranks / self.stripe_size
        return (
            int(self._slots.available_size() * pages_per_token),
            int(self._tree.evictable_size() * pages_per_token),
        )

    def reset(self) -> None:
        """Drop every cached stripe; no request may hold any."""
        self._tree.reset()
        self._slots.clear()

    def _slots_of_path(self, node: object) -> torch.Tensor:
        indices = self._tree.path_device_indices(node)
        return indices[:: self.stripe_size] // self.stripe_size

    @property
    def num_slots(self) -> int:
        return self.buffers.num_pages - 1

    @property
    def num_pages(self) -> int:
        """The request pages the cache holds over all ranks."""
        return self.num_slots * self.stripes.num_ranks


def _key(sandy_ids: Sequence[int]) -> RadixKey:
    return RadixKey(token_ids=array("q", sandy_ids))
