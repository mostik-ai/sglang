from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class FrankyMambaRoom:
    """SGLang starts requests while each keeps ``states_per_request`` mamba states;
    a state that Franky's locks pin (chain holds, waiting blocks) takes that room."""

    def __init__(
        self, *, req_to_token_pool: HybridReqToTokenPool, states_per_request: int
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.states_per_request = states_per_request

    @staticmethod
    def from_engine(
        *,
        req_to_token_pool: ReqToTokenPool,
        kv_cache_configurator: KVCacheConfigurator,
    ) -> Optional[FrankyMambaRoom]:
        """
        Returns:
            None for an engine without a mamba state pool of fixed slots.
        """
        if not isinstance(req_to_token_pool, HybridReqToTokenPool) or not isinstance(
            req_to_token_pool.mamba_allocator, MambaSlotAllocator
        ):
            return None
        return FrankyMambaRoom(
            req_to_token_pool=req_to_token_pool,
            states_per_request=kv_cache_configurator.calculate_mamba_ratio(),
        )

    def num_startable_requests(self, *, num_pinned_states: int) -> int:
        """Requests that may start beside the ones that hold request rows."""
        free_states = (
            self._num_states
            - num_pinned_states
            - self.states_per_request * self._num_rows_in_use
        )
        return max(free_states // self.states_per_request, 0)

    def fits_pinned_states(self, *, num_pinned_states: int) -> bool:
        """Whether that many pinned states leave each request that holds a row
        its states, and leave one request room to start while none does."""
        num_budgeted_rows = max(self._num_rows_in_use, 1)
        return (
            num_pinned_states + self.states_per_request * num_budgeted_rows
            <= self._num_states
        )

    @property
    def _num_states(self) -> int:
        return self.req_to_token_pool.mamba_allocator.size

    @property
    def _num_rows_in_use(self) -> int:
        return self.req_to_token_pool.size - self.req_to_token_pool.available_size()
