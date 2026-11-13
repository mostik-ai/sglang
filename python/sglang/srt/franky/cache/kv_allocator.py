from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
from typing_extensions import override

from sglang.srt.franky.cache.prefill_budget import FrankyPrefillBudget
from sglang.srt.franky.cache.taken_slots import FrankyTakenSlots
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import evict_from_tree_cache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class FrankyKvAllocator(PagedTokenToKVPoolAllocator):
    """Lets Ricky run chunks on the slots its requests took at arrival, so
    ``alloc_for_extend`` stays as it is. Without ``expect_extend`` it behaves as
    its base class. Solo models never build it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._taken = FrankyTakenSlots()

    @property
    def arrival_size(self) -> int:
        """All slots an arrival could ever take."""
        return self.size

    def free_at_arrival(self, *, tree_cache: BasePrefixCache) -> int:
        """Slots an arrival may take, counting what the prefix tree can evict."""
        return self.available_size() + tree_cache.evictable_size()

    def take_at_arrival(
        self, *, tree_cache: BasePrefixCache, num_slots: int
    ) -> Optional[torch.Tensor]:
        """
        Returns:
            ``num_slots`` slots, whole pages, evicting the tree as needed; None
            when they do not fit.
        """
        if self.available_size() < num_slots:
            evict_from_tree_cache(tree_cache, num_slots)
        return self.alloc(num_slots)

    def expect_extend(self, *, reqs: List[Req]) -> None:
        """The next ``alloc_extend`` returns these requests' taken slots, in this
        order, and takes no page.
        """
        self._taken.expect(reqs=reqs)

    def prefill_budget(
        self, *, tree_cache: BasePrefixCache, num_mixed_decode_tokens: int
    ) -> FrankyPrefillBudget:
        """The budget of Ricky's prefill batches, which run on taken slots."""
        return FrankyPrefillBudget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

    @override
    def get_extend_allocation_demand(
        self,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        *,
        conservative_num_tokens: int,
        shard_size: int,
    ) -> int:
        if self._taken.expected:
            return 0
        return super().get_extend_allocation_demand(
            prefix_lens_cpu,
            seq_lens_cpu,
            conservative_num_tokens=conservative_num_tokens,
            shard_size=shard_size,
        )

    @override
    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: Optional[int] = None,
    ):
        if not self._taken.expected:
            return super().alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages,
            )
        return self._taken.hand_out(
            prefix_lens_cpu=prefix_lens_cpu, seq_lens_cpu=seq_lens_cpu
        )
