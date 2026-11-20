from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Union

from typing_extensions import override

from sglang.srt.disaggregation.utils import is_aborted
from sglang.srt.franky.request import FrankyRequest
from sglang.srt.franky.schedulers.chain_batch_result_processor import (
    FrankyChainBatchResultProcessor,
    fields_of,
)

if TYPE_CHECKING:
    from sglang.srt.franky.transfer.send_queue import FrankySendQueue
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
    )
    from sglang.srt.managers.utils import EmbeddingBatchResult, GenerationBatchResult


# A subclass of a dataclass must stay one to add a field.
@dataclasses.dataclass(kw_only=True, frozen=True)
class SandyBatchResultProcessor(FrankyChainBatchResultProcessor):
    """Sandy's results, processed as on any engine. The taps of a prefill chunk
    land once its forward ended, and a request with a room that finished goes to
    the send queue, which keeps its KV until its taps are sent.
    """

    franky_send_queue: FrankySendQueue

    @classmethod
    def extend(
        cls,
        base: SchedulerBatchResultProcessor,
        *,
        franky_send_queue: FrankySendQueue,
    ) -> SandyBatchResultProcessor:
        return cls(**fields_of(base), franky_send_queue=franky_send_queue)

    @override
    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ) -> None:
        middle_chunks = [
            (req, prefix_len + extend_len)
            for req, prefix_len, extend_len in zip(
                batch.reqs, batch.prefix_lens, batch.extend_lens, strict=True
            )
            if req.inflight_middle_chunks > 0
        ]
        super().process_batch_result_prefill(batch, result)
        for req, end in middle_chunks:
            self.franky_send_queue.after_chunk(req=req, end=end)

    @override
    def release_finished_kv(self, req: Req, *, checkpoint: bool) -> None:
        if FrankyRequest.of(req).room is None or is_aborted(req):
            super().release_finished_kv(req, checkpoint=checkpoint)
            return
        self.franky_send_queue.after_end(req=req, checkpoint=checkpoint)
