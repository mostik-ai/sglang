from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Union

from typing_extensions import override

from sglang.srt.managers.schedule_batch import FINISH_LENGTH
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache

if TYPE_CHECKING:
    from sglang.srt.franky.transfer.send_queue import FrankySendQueue
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.utils import EmbeddingBatchResult, GenerationBatchResult


# A subclass of a dataclass must stay one to add a field.
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class SandyBatchResultProcessor(SchedulerBatchResultProcessor):
    """Sandy's prefill results. A request whose prefill ended is held until its
    taps are sent, as PD prefill holds one until its transfer ends; it leaves
    the running batch as finished, so it never decodes.
    """

    franky_send_queue: FrankySendQueue

    @classmethod
    def extend(
        cls,
        base: SchedulerBatchResultProcessor,
        *,
        franky_send_queue: FrankySendQueue,
    ) -> SandyBatchResultProcessor:
        fields = {
            field.name: getattr(base, field.name) for field in dataclasses.fields(base)
        }
        return cls(**fields, franky_send_queue=franky_send_queue)

    @override
    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ) -> None:
        if result.copy_done is not None:
            result.copy_done.synchronize()
        # Waits for the forward, so the batch's taps are written.
        result.next_token_ids.tolist()
        self.token_to_kv_pool_allocator.free_group_begin()
        aborted = []
        for req, prefix_len, extend_len in zip(
            batch.reqs, batch.prefix_lens, batch.extend_lens, strict=True
        ):
            if req.inflight_middle_chunks > 0:
                req.inflight_middle_chunks -= 1
                req.time_stats.set_last_chunked_prefill_finish_time()
                self.franky_send_queue.after_chunk(
                    req=req, end=prefix_len + extend_len, last=False
                )
            elif req.finished():
                # An aborted chunked request, which its abort released.
                continue
            elif req.to_finish is not None:
                # Aborted while it ran: end it as the normal path does.
                self.franky_send_queue.forget(req=req)
                req.update_finish_state(0)
                release_kv_cache(req, self.tree_cache, checkpoint=True)
                req.time_stats.set_completion_time()
                aborted.append(req)
            else:
                req.time_stats.set_prefill_finished_time()
                checkpoint_kv_cache(req, self.tree_cache)
                self.franky_send_queue.after_chunk(
                    req=req, end=prefix_len + extend_len, last=True
                )
                req.finished_reason = FINISH_LENGTH(length=0)
        self.token_to_kv_pool_allocator.free_group_end()
        if aborted:
            self.output_streamer.stream_output(aborted, False)
        if batch.prefill_stats is not None:
            self.metrics_reporter.report_prefill_stats(
                batch=batch,
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=result.can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )
