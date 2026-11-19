from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec
import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.franky.cache.generated_spans import FrankyGeneratedSpans
from sglang.srt.franky.sandy_input import sandy_input_ids
from sglang.srt.franky.transfer.messages import FrankyRoomClosed, FrankyTapsReady
from sglang.srt.franky.transfer.receiver import FrankyReceiver
from sglang.srt.franky.transfer.req_state import FrankyReqState
from sglang.srt.managers.schedule_batch import FINISH_LENGTH

if TYPE_CHECKING:
    from sglang.srt.franky.cache.mamba_room import FrankyMambaRoom
    from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
    from sglang.srt.franky.cache.tap_reservations import (
        FrankyTapReservations,
        TapReservation,
    )
    from sglang.srt.franky.chains.holds import FrankyChainHolds
    from sglang.srt.franky.transfer.manager import FrankyTapManager
    from sglang.srt.franky.transfer.rank_reports import FrankyRankReports
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

_NO_TAPS_NAMESPACE = "franky=no-taps"


class _Reception(msgspec.Struct, kw_only=True):
    req: Any
    receiver: Optional[FrankyReceiver]
    landed: int
    ended: bool = False
    hold: bool = False
    release_deadline: float = 0.0
    announced_taps: int = 0
    announced_failure: bool = False
    announced_close: bool = False


class FrankyReceiveQueue:
    """Ricky's requests from arrival until they end. A request that reads taps
    takes their stripes at arrival in ``FrankyTapReservations``, and its taps
    land from Sandy in the stripes past its cached prefix. Its rows run as soon
    as the taps they read landed; its KV is SGLang's, as without Franky. Ranks
    agree on what landed through ``FrankyRankReports``, and decisions reach
    every rank and stage at one step as leader messages.
    """

    def __init__(
        self,
        *,
        tap_cache: RickyTapCache,
        reservations: FrankyTapReservations,
        tap_manager: FrankyTapManager,
        rank_reports: FrankyRankReports,
        chain_holds: FrankyChainHolds,
        mamba_room: Optional[FrankyMambaRoom],
        shared_tokenizer: bool,
        release_timeout_seconds: float,
        fail_request: Callable[[Req, str], None],
    ) -> None:
        """
        Args:
            shared_tokenizer: whether Sandy reads Ricky's tokens, as the Franky
                config says; a request then sends no Sandy ids of its own.
        """
        self.tap_cache = tap_cache
        self.reservations = reservations
        self.tap_manager = tap_manager
        self.rank_reports = rank_reports
        self.chain_holds = chain_holds
        self.mamba_room = mamba_room
        self.shared_tokenizer = shared_tokenizer
        self.release_timeout_seconds = release_timeout_seconds
        self.fail_request = fail_request
        self.receptions: dict[str, _Reception] = {}
        self.agreed: dict[str, int] = {}
        self._failing: list[tuple[Req, str]] = []

    def add(self, *, req: Req) -> None:
        """A request entered the waiting queue: new, or back after a retraction
        released its KV, keeping its taps. A request without a room reads no
        taps, so its KV is cached apart from KV that read taps.
        """
        if req.franky is not None:
            return
        if req.franky_room is None:
            req.extra_key = (
                f"{req.extra_key}|{_NO_TAPS_NAMESPACE}"
                if req.extra_key
                else _NO_TAPS_NAMESPACE
            )
            req.franky = FrankyReqState(tap_len=0, taps_ready=0)
            return
        problem = self._tokenization_problem(req)
        if problem is not None:
            self._failing.append((req, problem))
            return
        sandy_ids = sandy_input_ids(req)
        self.reservations.expect(req.rid, req=req, sandy_ids=sandy_ids)
        if (refusal := self.reservations.refusal(req.rid)) is not None:
            self.reservations.forget(req.rid)
            self._failing.append((req, refusal))
            return
        req.franky = FrankyReqState(
            tap_len=len(sandy_ids),
            taps_ready=0,
            generated_spans=FrankyGeneratedSpans.of_request(req),
        )

    def forget(self, *, req: Req) -> None:
        """The scheduler aborted ``req``; its stripes wait for its room to close."""
        self.reservations.forget(req.rid)
        self.chain_holds.release_chain_of(req)
        if req.rid in self.receptions:
            self._end(self.receptions[req.rid], hold=False)

    def poll(self) -> None:
        """Every rank, at each step: end the receptions of ended requests, take
        stripes for arrivals in order, and agree on landings with
        ``FrankyRankReports``.
        """
        failing, self._failing = self._failing, []
        for req, problem in failing:
            self.fail_request(req, problem)
        for reception in self._live_receptions():
            if reception.req.finished():
                self._end(reception, hold=_ended_at_its_length(reception.req))
        self._release_holds_past_the_mamba_room()
        for rid, req in self.reservations.poll():
            self._open(rid, req)
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
        state = reception.req.franky
        state.taps_ready = max(state.taps_ready, message.taps_ready)
        self._note_first_chunk_ready(reception.req)
        if state.taps_complete():
            self._forget_keys(message.request_rid, "landed", "ok")

    def on_room_closed(self, *, message: FrankyRoomClosed) -> None:
        reception = self.receptions.get(message.request_rid)
        if reception is not None and reception.ended:
            self._release(reception)

    def on_started(self, *, reqs: list[Req]) -> None:
        """The scheduler started ``reqs``; each now locks its cached KV prefix
        itself, so its chain's hold on it may go.
        """
        for req in reqs:
            if req.franky_chain is not None and req.franky_room is not None:
                self.chain_holds.release_chain_of(req)

    def num_startable_requests(self) -> Optional[int]:
        """
        Returns:
            How many requests may start beside the running ones as far as the
            mamba state pool goes, or None without such a pool.
        """
        if self.mamba_room is None:
            return None
        return self.mamba_room.num_startable_requests(
            num_pinned_states=len(self.chain_holds.locked_nodes())
        )

    def is_empty(self) -> bool:
        return (
            self.reservations.is_empty() and not self.receptions and not self._failing
        )

    def _open(self, rid: str, req: Req) -> None:
        reservation = self.reservations.of(rid)
        num_cached = reservation.cached.num_taps
        self.receptions[rid] = _Reception(
            req=req,
            receiver=FrankyReceiver(
                manager=self.tap_manager,
                franky_room=req.franky_room,
                tap_pages=self._local_pages(reservation),
                num_tokens=reservation.num_taps,
                prefix_len=num_cached,
            ),
            landed=num_cached,
            announced_taps=num_cached,
        )
        req.franky.taps_ready = max(req.franky.taps_ready, num_cached)
        req.time_stats.set_franky_pages_taken_time(
            tap_len=reservation.num_taps, taps_held=num_cached
        )
        self._note_first_chunk_ready(req)

    def _local_pages(self, reservation: TapReservation) -> np.ndarray:
        page_size = self.tap_cache.buffers.page_size
        return self.tap_cache.local_pages(
            slots=reservation.slots,
            first_page=reservation.cached.num_taps // page_size,
            num_pages=-(-reservation.num_taps // page_size),
        )

    def _note_first_chunk_ready(self, req: Req) -> None:
        if req.franky.rows_ready() > req.prefix_len:
            req.time_stats.set_franky_taps_ready_time()

    def _tokenization_problem(self, req: Req) -> Optional[str]:
        if self.shared_tokenizer and req.franky_sandy_input_ids is not None:
            return (
                f"Request {req.rid} sends franky_sandy_input_ids, but this Ricky's "
                "Sandy reads its tokens."
            )
        if not self.shared_tokenizer and req.franky_sandy_input_ids is None:
            return (
                f"Request {req.rid} reads taps of a Sandy that tokenizes on its own, "
                "so it needs franky_sandy_input_ids."
            )
        return None

    def _live_receptions(self) -> list[_Reception]:
        return [
            reception for reception in self.receptions.values() if not reception.ended
        ]

    def _keep_kv_past_the_prompt_out_of_the_cache(self, req: Req) -> None:
        if req.franky.tap_len > 0 and req.prefix_len >= len(req.origin_input_ids):
            req.skip_radix_cache_insert = True

    def _release_holds_past_the_mamba_room(self) -> None:
        while not self._fits_mamba_room():
            if not self.chain_holds.release_least_progress(sparing_chain_id=None):
                return

    def _fits_mamba_room(self) -> bool:
        if self.mamba_room is None:
            return True
        return self.mamba_room.fits_pinned_states(
            num_pinned_states=len(self.chain_holds.locked_nodes())
        )

    def _rank_numbers(self) -> dict[str, int]:
        numbers = {}
        for rid, reception in self.receptions.items():
            if reception.ended:
                numbers[f"closed:{rid}"] = int(self._room_is_closed(reception))
            elif not reception.req.franky.taps_complete():
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
            reception.landed = reception.req.franky.tap_len
        return state != KVPoll.Failed

    def _taps_ready_message(
        self, rid: str, reception: _Reception
    ) -> Optional[FrankyTapsReady]:
        if reception.req.franky.taps_complete() or reception.announced_failure:
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

    def _end(self, reception: _Reception, *, hold: bool) -> None:
        reception.ended = True
        reception.hold = hold
        if reception.req.franky.taps_complete():
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
        landed = req.franky.taps_complete()
        hold = reception.hold and landed
        tap_lock = self.reservations.end(req.rid, landed=landed, keep_locked=hold)
        if hold:
            self.chain_holds.hold_for_next_block(req, tap_lock=tap_lock)
        else:
            self.chain_holds.release_chain_of(req)
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


def _ended_at_its_length(req: Req) -> bool:
    return isinstance(req.finished_reason, FINISH_LENGTH)
