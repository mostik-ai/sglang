from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional

from typing_extensions import override

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class FrankyPrefillAdder(PrefillAdder):
    """Ricky's admission: a chunk ends at the first row that reads taps that
    have not landed. While a chunked request goes on, another request joins only
    if it ends in this batch: the scheduler keeps one chunked request.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Mamba checkpoints land only on page ends, so their chunks end there.
        self._chunks_may_end_inside_pages = not (
            self.tree_cache.supports_mamba()
            and self.tree_cache.supports_prefix_sharing()
        )

    @override
    def _select_prefill_admission(
        self,
        req: Req,
        *,
        total_tokens: int,
        host_hit_length: int,
        swa_host_hit_length: int,
        truncation_align_size: Optional[int],
        has_chunked_req: bool = False,
    ):
        """A new request whose taps have not all landed runs every row whose
        taps it does not wait for. Cut to whole pages, the rows past the last
        whole page would take one more forward, and a forward's fixed cost holds
        the scheduler for every request while the taps land.
        """
        select = functools.partial(
            super()._select_prefill_admission,
            req,
            total_tokens=total_tokens,
            host_hit_length=host_hit_length,
            swa_host_hit_length=swa_host_hit_length,
            truncation_align_size=truncation_align_size,
            has_chunked_req=has_chunked_req,
        )
        if _taps_complete(req) or not self._chunks_may_end_inside_pages:
            return select()
        exact_chunk_fill, self.exact_chunk_fill = self.exact_chunk_fill, True
        try:
            return select()
        finally:
            self.exact_chunk_fill = exact_chunk_fill

    @override
    def add_chunked_req(self, req: Req) -> Optional[Req]:
        taps_limit = req.franky.rows_ready() - req.prefix_len
        self.chunked_req_limit = (
            taps_limit
            if self.chunked_req_limit is None
            else min(self.chunked_req_limit, taps_limit)
        )
        return super().add_chunked_req(req)

    @override
    def add_one_req(
        self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
    ) -> AddReqResult:
        if not self._can_start(req, has_chunked_req=has_chunked_req):
            return AddReqResult.CONTINUE
        return super().add_one_req(req, has_chunked_req, truncation_align_size)

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
        if not _taps_complete(req):
            taps_limit = req.franky.rows_ready() - req.prefix_len
            chunk_limit = (
                taps_limit if chunk_limit is None else min(chunk_limit, taps_limit)
            )
        return super()._check_prefill_budget(
            req,
            extend_input_len=extend_input_len,
            total_tokens=total_tokens,
            swa_host_hit_length=swa_host_hit_length,
            chunk_limit=chunk_limit,
        )

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


def taps_landed(req: Req) -> bool:
    """Whether some of the request's rows still to run read only landed taps."""
    return req.franky is not None and req.franky.rows_ready() > req.prefix_len


def _taps_complete(req: Req) -> bool:
    return req.franky.rows_ready() >= len(req.full_untruncated_fill_ids)
