from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
from typing_extensions import override

from sglang.srt.franky.cache.prefill_budget import FrankySwaPrefillBudget
from sglang.srt.franky.cache.taken_slots import FrankyTakenSlots
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.utils.common import get_num_new_pages

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class FrankySwaKvAllocator(SWATokenToKVPoolAllocator):
    """Ricky's allocator for models whose KV keeps sliding windows apart from a
    full pool, such as DeepSeek V4: a request takes full-pool pages at arrival,
    and its taps live at those slots for as long as its KV does. Its chunks run on
    them and take window slots as they would without Franky. Without
    ``expect_extend`` it behaves as its base class.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._taken = FrankyTakenSlots()

    @property
    def arrival_size(self) -> int:
        """All full-pool slots an arrival could ever take."""
        return self.size_full

    def free_at_arrival(self, *, tree_cache: BasePrefixCache) -> int:
        """Full-pool slots an arrival may take, counting what the prefix tree can
        evict."""
        return self.full_available_size() + tree_cache.full_evictable_size()

    def take_at_arrival(
        self, *, tree_cache: BasePrefixCache, num_slots: int
    ) -> Optional[torch.Tensor]:
        """
        Returns:
            ``num_slots`` full-pool slots, whole pages, evicting the tree as
            needed; None when they do not fit. No window slot is taken.
        """
        shortfall = num_slots - self.full_available_size()
        if shortfall > 0:
            tree_cache.evict_for_alloc(EvictParams(num_tokens=shortfall))
        return self.full_attn_allocator.alloc(num_slots)

    def expect_extend(self, *, reqs: List[Req]) -> None:
        """The next ``alloc_extend`` returns these requests' taken slots, in this
        order, and takes window slots for them only.
        """
        self._taken.expect(reqs=reqs)

    def prefill_budget(
        self, *, tree_cache: BasePrefixCache, num_mixed_decode_tokens: int
    ) -> FrankySwaPrefillBudget:
        """The budget of Ricky's prefill batches, which run on taken full-pool
        slots and take window slots."""
        return FrankySwaPrefillBudget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

    @override
    def evict_to_free_tokens(self, tree_cache, num_tokens: int) -> None:
        if not self._taken.expected:
            super().evict_to_free_tokens(tree_cache, num_tokens)
            return
        swa_shortfall = num_tokens - self.swa_available_size()
        if tree_cache is not None and swa_shortfall > 0:
            tree_cache.evict_for_alloc(EvictParams(swa_num_tokens=swa_shortfall))

    @override
    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if not self._taken.expected:
            return super().alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
            )
        full_indices = self._taken.hand_out(
            prefix_lens_cpu=prefix_lens_cpu, seq_lens_cpu=seq_lens_cpu
        )
        if self._swa_req_ring:
            return full_indices
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        swa_indices = (
            self.swa_attn_allocator.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                self.translate_loc_from_full_to_swa(last_loc),
                extend_num_tokens,
                num_new_pages=num_new_pages,
            )
            if self.new_pages_available(0, num_new_pages)
            else None
        )
        if swa_indices is None:
            raise RuntimeError(
                f"Franky found no window slots for {extend_num_tokens} tokens: "
                f"{self.swa_available_size()} available."
            )
        self.set_full_to_swa_mapping(full_indices, swa_indices)
        return full_indices
