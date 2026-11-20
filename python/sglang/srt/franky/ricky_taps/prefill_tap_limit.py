from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class PrefillTapLimit(msgspec.Struct, frozen=True, kw_only=True):
    """How far a Ricky request's prefill may run while its taps land: up to the
    first token that reads a tap that has not landed. Mamba checkpoints land only
    on page ends, so with a mamba prefix cache a prefill that stops early stops on
    a page end; the tokens past it wait for the taps and run with the last ones.

    Attributes:
        chunk_grid: the tokens an early stop is a multiple of; 1 where it may stop
            on any token.
    """

    chunk_grid: int

    @staticmethod
    def of(tree_cache: BasePrefixCache, *, page_size: int) -> PrefillTapLimit:
        mamba_checkpoints = (
            tree_cache.supports_mamba() and tree_cache.supports_prefix_sharing()
        )
        return PrefillTapLimit(chunk_grid=page_size if mamba_checkpoints else 1)

    def end(self, req: Req) -> int:
        """The token the request's prefill may run up to now."""
        num_tokens = _num_tokens(req)
        ready = req.franky_taps.rows_ready()
        if ready >= num_tokens:
            return num_tokens
        return ready // self.chunk_grid * self.chunk_grid

    def can_run(self, req: Req) -> bool:
        """Whether some of the request's tokens still to run may run now."""
        return req.franky_taps is not None and self.end(req) > req.prefix_len

    @staticmethod
    def complete(req: Req) -> bool:
        """Whether every token of the request may run."""
        return req.franky_taps.rows_ready() >= _num_tokens(req)


def _num_tokens(req: Req) -> int:
    return len(req.origin_input_ids) + len(req.output_ids)
