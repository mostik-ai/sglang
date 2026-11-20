from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from sglang.srt.disaggregation.utils import is_aborted
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)

if TYPE_CHECKING:
    from sglang.srt.franky.chains.holds import FrankyChainHolds
    from sglang.srt.managers.schedule_batch import Req


# A subclass of a dataclass must stay one to add a field.
@dataclasses.dataclass(kw_only=True, frozen=True)
class FrankyChainBatchResultProcessor(SchedulerBatchResultProcessor):
    """Results on an engine that serves Franky chains: a chain request that
    finished well leaves its prefix held for the chain's next request once its
    KV is in the tree; an aborted one ends its chain's hold.
    """

    franky_chain_holds: FrankyChainHolds

    @classmethod
    def extend(
        cls,
        base: SchedulerBatchResultProcessor,
        *,
        franky_chain_holds: FrankyChainHolds,
    ) -> FrankyChainBatchResultProcessor:
        return cls(**fields_of(base), franky_chain_holds=franky_chain_holds)

    @override
    def release_finished_kv(self, req: Req, *, checkpoint: bool) -> None:
        super().release_finished_kv(req, checkpoint=checkpoint)
        if is_aborted(req):
            self.franky_chain_holds.release_chain_of(req)
        else:
            self.franky_chain_holds.hold_for_next_block(req)


def fields_of(processor: SchedulerBatchResultProcessor) -> dict[str, Any]:
    """The fields of ``processor``, for a subclass that adds its own."""
    return {
        field.name: getattr(processor, field.name)
        for field in dataclasses.fields(processor)
    }
