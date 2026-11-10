from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from typing_extensions import override

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.prefill_budget import PrefillBudget

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class FrankyPrefillBudget(PrefillBudget):
    """Ricky's prefill takes no KV memory, since each request took its pages at
    arrival, so only chunk sizes bound a batch.
    """

    @override
    def has_capacity(self) -> bool:
        return True

    @override
    def check_prefill(
        self,
        *,
        extend_input_len: int,
        total_tokens: int,
        max_new_tokens: int,
        input_tokens: int,
        swa_host_hit_length: int,
        chunk_limit: Optional[int],
        auxiliary_swa_host_hit_length: int = 0,
    ) -> tuple[bool, Optional[int]]:
        return True, chunk_limit

    @override
    def can_allocate_prefill(
        self,
        *,
        paged_input: int,
        extend_input_len: int,
        max_new_tokens: int,
        chunk_limit: Optional[int],
    ) -> bool:
        return True

    @override
    def available_chunk_tokens(self, chunk_limit: int) -> int:
        return chunk_limit


class FrankyPrefillAdder(PrefillAdder):
    """Ricky's admission: a chunk ends at the landed taps and runs on the slots
    its request took at arrival. While a chunked request goes on, another
    request joins only if it ends in this batch: the scheduler keeps one
    chunked request.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.memory_budget = FrankyPrefillBudget(
            self.token_to_kv_pool_allocator, self.tree_cache
        )

    @override
    def add_chunked_req(self, req: Req) -> Optional[Req]:
        taps_limit = req.franky.taps_ready - req.prefix_len
        self.chunked_req_limit = (
            taps_limit
            if self.chunked_req_limit is None
            else min(self.chunked_req_limit, taps_limit)
        )
        chunked_req = super().add_chunked_req(req)
        self._expect_taken_slots()
        return chunked_req

    @override
    def add_one_req(
        self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
    ) -> AddReqResult:
        if not self._can_start(req, has_chunked_req=has_chunked_req):
            return AddReqResult.CONTINUE
        result = super().add_one_req(req, has_chunked_req, truncation_align_size)
        self._expect_taken_slots()
        return result

    @override
    def _check_prefill_budget(
        self,
        req: Req,
        *,
        extend_input_len: int,
        total_tokens: int,
        swa_host_hit_length: int,
        chunk_limit: Optional[int],
    ) -> tuple[bool, Optional[int]]:
        if _taps_complete(req):
            return True, chunk_limit
        taps_limit = req.franky.taps_ready - req.prefix_len
        return True, taps_limit if chunk_limit is None else min(chunk_limit, taps_limit)

    def _can_start(self, req: Req, *, has_chunked_req: bool) -> bool:
        if not taps_landed(req):
            return False
        if not has_chunked_req and self.new_chunked_req is None:
            return True
        return _taps_complete(req) and self._fits_this_batch(req)

    def _fits_this_batch(self, req: Req) -> bool:
        if self.rem_chunk_tokens is None:
            return True
        remaining = len(req.full_untruncated_fill_ids) - req.prefix_len
        return self.ceil_paged_tokens(remaining) <= self.rem_chunk_tokens

    def _expect_taken_slots(self) -> None:
        if self.can_run_list:
            self.token_to_kv_pool_allocator.expect_extend(reqs=self.can_run_list)


def taps_landed(req: Req) -> bool:
    """Whether taps have landed for some of the request's tokens still to run."""
    return req.franky is not None and req.franky.taps_ready > req.prefix_len


def _taps_complete(req: Req) -> bool:
    return req.franky.taps_ready >= len(req.full_untruncated_fill_ids)
