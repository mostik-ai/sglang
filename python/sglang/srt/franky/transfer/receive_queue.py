from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.franky.request import FrankyRequest
from sglang.srt.franky.transfer.messages import FrankyRoomClosed, FrankyTapsReady
from sglang.srt.franky.transfer.receiver import FrankyReceiver

if TYPE_CHECKING:
    from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
    from sglang.srt.franky.ricky_taps.tap_requests import RickyTapRequests
    from sglang.srt.franky.transfer.manager import FrankyTapManager
    from sglang.srt.franky.transfer.rank_reports import FrankyRankReports
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class _Reception(msgspec.Struct, kw_only=True):
    req: Any
    receiver: Optional[FrankyReceiver]
    landed: int
    ended: bool = False
    release_deadline: float = 0.0
    announced_taps: int = 0
    announced_failure: bool = False
    announced_close: bool = False


class FrankyReceiveQueue:
    """The taps landing from Sandy on Ricky prefill and aggregated engines: one
    reception per request from the reservation of its stripes until its taps
    landed, or, for a request that ended first, until its room stopped writing.
    Ranks agree on what landed through ``FrankyRankReports``, and decisions reach
    every rank and stage at one step as leader messages.
    """

    def __init__(
        self,
        *,
        tap_cache: RickyTapCache,
        tap_requests: RickyTapRequests,
        tap_manager: FrankyTapManager,
        rank_reports: FrankyRankReports,
        release_timeout_seconds: float,
        fail_request: Callable[[Req, str], None],
    ) -> None:
        self.tap_cache = tap_cache
        self.tap_requests = tap_requests
        self.tap_manager = tap_manager
        self.rank_reports = rank_reports
        self.release_timeout_seconds = release_timeout_seconds
        self.fail_request = fail_request
        self.receptions: dict[str, _Reception] = {}
        self.agreed: dict[str, int] = {}
        self._failing: list[tuple[Req, str]] = []

    def add(self, *, req: Req) -> None:
        """A request entered the waiting queue: new, or back after a retraction
        released its KV, keeping its taps. One that can never read its taps fails
        at the next poll, once the waiting queue holds it.
        """
        problem = self.tap_requests.arrive(req)
        if problem is not None:
            self._failing.append((req, problem))

    def forget(self, *, req: Req) -> None:
        """The scheduler aborted ``req``; its stripes wait for its room to close."""
        self.tap_requests.forget(req)
        if req.rid in self.receptions:
            self._end(self.receptions[req.rid])

    def poll(self) -> None:
        """Every rank, at each step: fail what never fits, end the receptions of
        ended requests, open receptions for the requests that took their stripes,
        and agree on landings with ``FrankyRankReports``.
        """
        failing, self._failing = self._failing, []
        for req, problem in failing:
            self.fail_request(req, problem)
        for reception in self._live_receptions():
            if reception.req.finished():
                self._end(reception)
        for req in self.tap_requests.poll():
            self._open(req)
        for reception in self._live_receptions():
            self._keep_kv_past_the_prompt_out_of_the_cache(reception.req)
        self.agreed = self.rank_reports.report(numbers=self._rank_numbers())

    def ready_messages(self) -> list[FrankyTapsReady | FrankyRoomClosed]:
        """The leader rank only: what every rank agreed on since the last step."""
        messages = []
        for rid, reception in self.receptions.items():
            message = (
                self._room_closed_message(rid, reception)
                if reception.ended
                else self._taps_ready_message(rid, reception)
            )
            if message is not None:
                messages.append(message)
        return messages

    def on_taps_ready(self, *, message: FrankyTapsReady) -> None:
        reception = self.receptions.get(message.request_rid)
        if reception is None or reception.ended:
            return
        if message.failed:
            self.fail_request(
                reception.req,
                f"Franky taps of request {message.request_rid} did not arrive.",
            )
            return
        state = reception.req.franky_taps
        state.taps_ready = max(state.taps_ready, message.taps_ready)
        self._note_first_chunk_ready(reception.req)
        if state.taps_complete():
            self._forget_keys(message.request_rid, "landed", "ok")

    def on_room_closed(self, *, message: FrankyRoomClosed) -> None:
        reception = self.receptions.get(message.request_rid)
        if reception is not None and reception.ended:
            self._release(reception)

    def is_empty(self) -> bool:
        return (
            self.tap_requests.is_empty() and not self.receptions and not self._failing
        )

    def _open(self, req: Req) -> None:
        reservation = self.tap_requests.reservation(req.rid)
        num_cached = reservation.cached.num_taps
        self.receptions[req.rid] = _Reception(
            req=req,
            receiver=FrankyReceiver.of_reservation(
                manager=self.tap_manager,
                franky_room=FrankyRequest.of(req).room,
                tap_cache=self.tap_cache,
                reservation=reservation,
            ),
            landed=num_cached,
            announced_taps=num_cached,
        )
        req.franky_taps.taps_ready = max(req.franky_taps.taps_ready, num_cached)
        req.time_stats.set_franky_pages_taken_time(
            tap_len=reservation.num_taps, taps_held=num_cached
        )
        self._note_first_chunk_ready(req)

    def _note_first_chunk_ready(self, req: Req) -> None:
        if req.franky_taps.rows_ready() > req.prefix_len:
            req.time_stats.set_franky_taps_ready_time()

    def _live_receptions(self) -> list[_Reception]:
        return [
            reception for reception in self.receptions.values() if not reception.ended
        ]

    def _keep_kv_past_the_prompt_out_of_the_cache(self, req: Req) -> None:
        if req.franky_taps.tap_len > 0 and req.prefix_len >= len(req.origin_input_ids):
            req.skip_radix_cache_insert = True

    def _rank_numbers(self) -> dict[str, int]:
        numbers = {}
        for rid, reception in self.receptions.items():
            if reception.ended:
                numbers[f"closed:{rid}"] = int(self._room_is_closed(reception))
            elif not reception.req.franky_taps.taps_complete():
                ok = self._poll_receiver(reception)
                numbers[f"landed:{rid}"] = reception.landed
                numbers[f"ok:{rid}"] = int(ok)
        return numbers

    def _poll_receiver(self, reception: _Reception) -> bool:
        receiver = reception.receiver
        if receiver is None:
            return True
        state = receiver.poll()
        reception.landed = receiver.landed()
        if state == KVPoll.Success:
            receiver.close()
            reception.receiver = None
            reception.landed = reception.req.franky_taps.tap_len
        return state != KVPoll.Failed

    def _taps_ready_message(
        self, rid: str, reception: _Reception
    ) -> Optional[FrankyTapsReady]:
        if reception.req.franky_taps.taps_complete() or reception.announced_failure:
            return None
        ok = self._lowest(f"ok:{rid}")
        if ok == 0:
            reception.announced_failure = True
            return FrankyTapsReady(request_rid=rid, taps_ready=0, failed=True)
        landed = self._lowest(f"landed:{rid}")
        if landed is None or landed <= reception.announced_taps:
            return None
        reception.announced_taps = landed
        return FrankyTapsReady(request_rid=rid, taps_ready=landed)

    def _room_closed_message(
        self, rid: str, reception: _Reception
    ) -> Optional[FrankyRoomClosed]:
        if reception.announced_close or self._lowest(f"closed:{rid}") != 1:
            return None
        reception.announced_close = True
        return FrankyRoomClosed(request_rid=rid)

    def _lowest(self, key: str) -> Optional[int]:
        if key not in self.agreed:
            return None
        return self.rank_reports.lowest(key=key, agreed=self.agreed[key])

    def _end(self, reception: _Reception) -> None:
        reception.ended = True
        if reception.req.franky_taps.taps_complete():
            self._release(reception)
            return
        reception.release_deadline = time.monotonic() + self.release_timeout_seconds
        if reception.receiver is not None:
            reception.receiver.abort()

    def _room_is_closed(self, reception: _Reception) -> bool:
        receiver = reception.receiver
        if receiver is None or receiver.writes_stopped():
            return True
        if time.monotonic() < reception.release_deadline:
            return False
        logger.warning(
            "Franky room %s did not confirm that its taps stopped landing after "
            "%.0fs; releasing its stripes anyway.",
            receiver.bootstrap_room,
            self.release_timeout_seconds,
        )
        return True

    def _release(self, reception: _Reception) -> None:
        req = reception.req
        self.tap_requests.end(req, landed=req.franky_taps.taps_complete())
        if reception.receiver is not None:
            self.tap_manager.clear_deferred_abort_state(
                reception.receiver.bootstrap_room
            )
            reception.receiver.close()
        self._forget_keys(req.rid, "landed", "ok", "closed")
        del self.receptions[req.rid]

    def _forget_keys(self, rid: str, *names: str) -> None:
        for name in names:
            key = f"{name}:{rid}"
            self.agreed.pop(key, None)
            self.rank_reports.forget(key=key)
