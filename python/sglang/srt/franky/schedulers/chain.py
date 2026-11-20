from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from typing_extensions import override

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.franky.cache.mamba_room import FrankyMambaRoom
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.chains.messages import FrankyHoldExpired
from sglang.srt.franky.config import FrankySide
from sglang.srt.franky.finish_rules.request_blocks import start_blocks
from sglang.srt.franky.request import FrankyRequest
from sglang.srt.franky.schedulers.chain_batch_result_processor import (
    FrankyChainBatchResultProcessor,
)
from sglang.srt.managers.io_struct import AbortReq, FrankyEndChainReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_franky
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from typing import List

    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class FrankyChainScheduler(Scheduler):
    """An engine that serves Franky chains: Sandy or Ricky of a pair that only
    writes text, and the base of Sandy's engine that also sends taps. It holds a
    chain's prefix between the chain's requests, runs held chains' requests
    first, ends a chain when the router asks, and where Ricky writes a chain's
    blocks ends each request at the block where the decider calls Sandy. When a
    held chain's request cannot start with nothing else running, the chain with
    the least progress gives up its hold.
    """

    @override
    def init_request_dispatcher(self) -> None:
        super().init_request_dispatcher()
        self._request_dispatcher += TypeBasedDispatcher(
            [
                (FrankyHoldExpired, self._handle_hold_expired),
                (FrankyEndChainReq, self._handle_end_chain),
            ]
        )

    @override
    def init_batch_result_processor(self) -> None:
        super().init_batch_result_processor()
        self.init_franky_chain_holds()
        self.init_franky_mamba_room()
        self.batch_result_processor = FrankyChainBatchResultProcessor.extend(
            self.batch_result_processor, franky_chain_holds=self.franky_chain_holds
        )

    def init_franky_chain_holds(self) -> None:
        """PD decode caches no prompts, so it holds nothing."""
        self.franky_chain_holds = FrankyChainHolds.from_config(
            tree_cache=(
                None
                if self.disaggregation_mode == DisaggregationMode.DECODE
                else self.tree_cache
            )
        )

    def init_franky_mamba_room(self) -> None:
        self.franky_mamba_room = FrankyMambaRoom.from_engine(
            req_to_token_pool=self.req_to_token_pool,
            kv_cache_configurator=self.tp_worker.model_runner.kv_cache_configurator,
        )
        self.franky_num_startable_reqs: Optional[int] = None

    @override
    def _add_request_to_queue(self, req: Req, is_retracted: bool = False) -> None:
        if self._writes_blocks:
            start_blocks(req)
        super()._add_request_to_queue(req, is_retracted=is_retracted)

    @override
    def _poll_timeout_aborts(self) -> List:
        return (
            super()._poll_timeout_aborts() + self.franky_chain_holds.expired_messages()
        )

    @override
    def _get_new_batch_prefill_raw(
        self, prefill_delayer_single_pass, running_batch: ScheduleBatch
    ) -> Tuple[Optional[ScheduleBatch], ScheduleBatch]:
        self.franky_chain_holds.put_held_chains_first(self.waiting_queue)
        self.franky_num_startable_reqs = self._num_franky_startable_reqs()
        new_batch, running_batch = super()._get_new_batch_prefill_raw(
            prefill_delayer_single_pass, running_batch
        )
        if new_batch is None and self._a_held_chain_waits_alone(running_batch):
            self.franky_chain_holds.release_least_progress(
                sparing_chain_id=FrankyRequest.of(self.waiting_queue[0]).chain.id
            )
        return new_batch, running_batch

    @override
    def get_num_allocatable_reqs(
        self,
        running_bs: int,
        beam_width: Optional[int] = None,
        running_batch: Optional[ScheduleBatch] = None,
        num_held_rows: int = 0,
    ) -> int:
        allocatable = super().get_num_allocatable_reqs(
            running_bs,
            beam_width,
            running_batch=running_batch,
            num_held_rows=num_held_rows,
        )
        if self.franky_num_startable_reqs is None:
            return allocatable
        return min(allocatable, num_held_rows + self.franky_num_startable_reqs)

    @override
    def flush_cache(self, empty_cache: bool = True) -> bool:
        self.franky_chain_holds.release_all()
        return super().flush_cache(empty_cache=empty_cache)

    @property
    def _writes_blocks(self) -> bool:
        return (
            FrankySide(get_franky().franky_side) == FrankySide.RICKY
            and self.disaggregation_mode != DisaggregationMode.PREFILL
        )

    def _handle_hold_expired(self, message: FrankyHoldExpired) -> None:
        self.franky_chain_holds.on_hold_expired(message=message)

    def _handle_end_chain(self, recv_req: FrankyEndChainReq) -> None:
        self.abort_request(AbortReq(rid=recv_req.chain_id))
        self.franky_chain_holds.release(chain_id=recv_req.chain_id)

    def _num_franky_startable_reqs(self) -> Optional[int]:
        if self.franky_mamba_room is None:
            return None
        return self.franky_mamba_room.num_startable_requests(
            num_pinned_states=len(self.franky_chain_holds.locked_nodes())
        )

    def _a_held_chain_waits_alone(self, running_batch: ScheduleBatch) -> bool:
        return (
            len(self.waiting_queue) > 0
            and self.chunked_req is None
            and running_batch.is_empty()
            and self.franky_chain_holds.holds_chain_of(self.waiting_queue[0])
        )
