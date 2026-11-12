from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
from typing_extensions import override

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class FrankyKvAllocator(PagedTokenToKVPoolAllocator):
    """Lets Ricky run chunks on the slots its requests took at arrival, so
    ``alloc_for_extend`` stays as it is. Without ``expect_extend`` it behaves as
    its base class. Solo models never build it.
    """

    _expected_reqs: Optional[List[Req]] = None

    def expect_extend(self, *, reqs: List[Req]) -> None:
        """The next ``alloc_extend`` returns these requests' taken slots, in this
        order, and takes no page.
        """
        self._expected_reqs = reqs

    @override
    def get_extend_allocation_demand(
        self,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        *,
        conservative_num_tokens: int,
        shard_size: int,
    ) -> int:
        if self._expected_reqs is not None:
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
        reqs, self._expected_reqs = self._expected_reqs, None
        if reqs is None:
            return super().alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages,
            )
        return torch.cat(
            [
                req.franky.hand_to_chunk(start=start, end=end)
                for req, start, end in zip(
                    reqs, prefix_lens_cpu.tolist(), seq_lens_cpu.tolist(), strict=True
                )
            ]
        )
