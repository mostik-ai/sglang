from __future__ import annotations

import sys

from typing_extensions import override

from sglang.srt.mem_cache.prefill_budget import PrefillBudget, SWAPrefillBudget

# Ricky's requests took their full-pool pages at arrival, so a prefill never waits
# on that pool; large enough for every check, small enough to stay an int.
_TAKEN_AT_ARRIVAL = sys.maxsize // 4


class FrankyPrefillBudget(PrefillBudget):
    """Ricky's prefill takes no KV memory, since each request took its pages at
    arrival, so only chunk sizes bound a batch.
    """

    @override
    def _available_and_evictable(self) -> int:
        return _TAKEN_AT_ARRIVAL


class FrankySwaPrefillBudget(SWAPrefillBudget):
    """Ricky's prefill on a model with sliding windows: the full pool was taken
    at arrival, and only the windows' slots bound a batch, as without Franky.
    """

    @override
    def _available_and_evictable(self) -> int:
        return _TAKEN_AT_ARRIVAL
