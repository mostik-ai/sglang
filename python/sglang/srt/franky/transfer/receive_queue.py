from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import msgspec
import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.franky.cache.tap_cache import FrankyReqState
from sglang.srt.franky.transfer.host_pages import FrankyHostPages
from sglang.srt.franky.transfer.messages import FrankyRoomClosed, FrankyTapsReady
from sglang.srt.franky.transfer.receiver import FrankyReceiver
from sglang.srt.managers.schedule_batch import FINISH_LENGTH
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.franky.cache.mamba_room import FrankyMambaRoom
    from sglang.srt.franky.cache.tap_cache import FrankyTapCache
    from sglang.srt.franky.chains.holds import FrankyChainHolds
    from sglang.srt.franky.transfer.manager import FrankyTapManager
    from sglang.srt.franky.transfer.rank_reports import FrankyRankReports
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, TreeLock

logger = logging.getLogger(__name__)

_NO_TAPS_NAMESPACE = "franky=no-taps"


class _Reception(msgspec.Struct, kw_only=True):
    req: Any
    # The prompt's tap pages on their way to the host; the receiver opens on arrival.
    pages: Optional[FrankyHostPages]
    receiver: Optional[FrankyReceiver]
    prefix_lock: Any
    landed: int
    ended: bool = False
    release_deadline: float = 0.0
    announced_taps: int = 0
    announced_failure: bool = False
    announced_close: bool = False


class FrankyReceiveQueue:
    """Ricky's requests from arrival until they end. At arrival a request takes
    KV pages for its prompt past its cached prefix, as PD decode does, and its
    taps land from Sandy in the tap pages of those pages. Decisions reach every
    rank and stage at one step as leader messages, as Sandy's releases do.
    """

    def __init__(
        self,
        *,
        tap_cache: FrankyTapCache,
        kv_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        tap_manager: FrankyTapManager,
        rank_reports: FrankyRankReports,
        chain_holds: FrankyChainHolds,
        mamba_room: Optional[FrankyMambaRoom],
        page_size: int,
        reserved_decode_tokens: int,
        release_timeout_seconds: float,
        fail_request: Callable[[Req, str], None],
    ) -> None:
        self.tap_cache = tap_cache
        self.kv_allocator = kv_allocator
        self.tree_cache = tree_cache
        self.tap_manager = tap_manager
        self.rank_reports = rank_reports
        self.chain_holds = chain_holds
        self.mamba_room = mamba_room
        self.page_size = page_size
        self.reserved_decode_tokens = reserved_decode_tokens
        self.release_timeout_seconds = release_timeout_seconds
        self.fail_request = fail_request
        self.arrivals: list[Req] = []
        self.returning_tap_lens: dict[str, int] = {}
        self.receptions: dict[str, _Reception] = {}
        self.agreed: dict[str, int] = {}

    def add(self, *, req: Req) -> None:
        """A request entered the waiting queue: new, or back after its KV was
        released by a retraction. A request without a room reads no taps, so its
        KV is cached apart from KV that read taps.
        """
        if req.rid in self.receptions:
            self.returning_tap_lens[req.rid] = req.franky.tap_len
            self._release(self.receptions[req.rid])
            req.franky = None
        elif req.franky_room is None:
            req.extra_key = (
                f"{req.extra_key}|{_NO_TAPS_NAMESPACE}"
                if req.extra_key
                else _NO_TAPS_NAMESPACE
            )
        self.arrivals.append(req)

    def forget(self, *, req: Req) -> None:
        """The scheduler aborted ``req``; its pages wait for its room to close."""
        if req in self.arrivals:
            self.arrivals.remove(req)
        self.returning_tap_lens.pop(req.rid, None)
        self.chain_holds.release_chain_of(req)
        if req.rid in self.receptions:
            self._end(self.receptions[req.rid])

    def poll(self) -> None:
        """Every rank, at each step: notice ended requests before an arrival can
        evict their prompts, take pages for arrivals in order, and agree on
        landings with ``FrankyRankReports``.
        """
        for reception in self._live_receptions():
            if reception.req.finished():
                self._hold_or_release_the_chain(reception.req)
                self._end(reception)
        self._release_holds_past_the_mamba_room()
        self._take_pages_for_arrivals()
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
        if self._taps_complete(reception):
            self._forget_keys(message.request_rid, "landed", "ok")

    def on_room_closed(self, *, message: FrankyRoomClosed) -> None:
        reception = self.receptions.get(message.request_rid)
        if reception is not None and reception.ended:
            self._release(reception)

    def on_started(self, *, reqs: list[Req]) -> None:
        """The scheduler started ``reqs``; each now locks its cached prefix itself."""
        for req in reqs:
            reception = self.receptions.get(req.rid)
            if reception is not None:
                self.tree_cache.unlock(reception.prefix_lock)
                reception.prefix_lock = None

    def num_startable_requests(self) -> Optional[int]:
        """
        Returns:
            How many requests may start beside the running ones as far as the
            mamba state pool goes, or None without such a pool.
        """
        if self.mamba_room is None:
            return None
        return self.mamba_room.num_startable_requests(
            num_pinned_states=len(self._pinned_nodes())
        )

    def is_empty(self) -> bool:
        return not self.arrivals and not self.receptions

    def _take_pages_for_arrivals(self) -> None:
        while self.arrivals:
            req = self.arrivals.pop(0)
            if not self._take_pages(req):
                self.arrivals.insert(0, req)
                return

    def _take_pages(self, req: Req) -> bool:
        returning = req.rid in self.returning_tap_lens
        tap_len = self._tap_len_of(req)
        req.init_next_round_input(self.tree_cache, cow_mamba=False)
        num_tokens = len(req.full_untruncated_fill_ids)
        num_taken = num_tokens - req.prefix_len
        num_page_slots = ceil_div(num_taken, self.page_size) * self.page_size
        if num_page_slots + self.reserved_decode_tokens > self.kv_allocator.size:
            self.fail_request(
                req, f"Request {req.rid} does not fit in Ricky's KV memory."
            )
            return True
        if returning and req.prefix_len < tap_len:
            self.fail_request(
                req, f"The taps of request {req.rid} left the prefix cache."
            )
            return True
        prefix_lock = (
            self.tree_cache.lock(req.last_node) if req.prefix_len > 0 else None
        )
        if not self._fits_after_releasing_holds(
            num_page_slots, prefix_lock=prefix_lock
        ):
            self.tree_cache.unlock(prefix_lock)
            return False
        self.chain_holds.release_chain_of(req)
        self.returning_tap_lens.pop(req.rid, None)
        taken = self._take_slots(num_page_slots)[:num_taken]
        receives_taps = not returning and tap_len > 0
        req.franky = FrankyReqState(
            tap_len=tap_len,
            taps_ready=req.prefix_len if receives_taps else num_tokens,
            taken_kv_slots=taken,
            taken_from=req.prefix_len,
            used_end=req.prefix_len,
        )
        self.receptions[req.rid] = _Reception(
            req=req,
            pages=self._tap_pages(taken) if receives_taps else None,
            receiver=None,
            prefix_lock=prefix_lock,
            landed=req.prefix_len if receives_taps else tap_len,
            announced_taps=req.prefix_len,
        )
        return True

    def _live_receptions(self) -> list[_Reception]:
        return [
            reception for reception in self.receptions.values() if not reception.ended
        ]

    def _hold_or_release_the_chain(self, req: Req) -> None:
        if isinstance(req.finished_reason, FINISH_LENGTH):
            self.chain_holds.hold_for_next_block(req)
        else:
            self.chain_holds.release_chain_of(req)

    def _keep_kv_past_the_prompt_out_of_the_cache(self, req: Req) -> None:
        if req.franky.tap_len > 0 and req.prefix_len >= req.franky.tap_len:
            req.skip_radix_cache_insert = True

    def _tap_len_of(self, req: Req) -> int:
        if req.rid in self.returning_tap_lens:
            return self.returning_tap_lens[req.rid]
        return len(req.origin_input_ids) if req.franky_room is not None else 0

    def _fits_after_releasing_holds(
        self, num_page_slots: int, *, prefix_lock: Optional[TreeLock]
    ) -> bool:
        if self._free_slots() + self.chain_holds.held_tokens < num_page_slots:
            return False
        while not self._fits(num_page_slots, prefix_lock=prefix_lock):
            if not self.chain_holds.release_oldest():
                return False
        return True

    def _fits(self, num_page_slots: int, *, prefix_lock: Optional[TreeLock]) -> bool:
        return self._free_slots() >= num_page_slots and self._fits_mamba_room(
            arriving_prefix_lock=prefix_lock
        )

    def _release_holds_past_the_mamba_room(self) -> None:
        while not self._fits_mamba_room():
            if not self.chain_holds.release_oldest():
                return

    def _fits_mamba_room(
        self, *, arriving_prefix_lock: Optional[TreeLock] = None
    ) -> bool:
        if self.mamba_room is None:
            return True
        pinned_nodes = self._pinned_nodes(arriving_prefix_lock=arriving_prefix_lock)
        return self.mamba_room.fits_pinned_states(num_pinned_states=len(pinned_nodes))

    def _pinned_nodes(
        self, *, arriving_prefix_lock: Optional[TreeLock] = None
    ) -> set[Any]:
        locks = [reception.prefix_lock for reception in self.receptions.values()]
        locks.append(arriving_prefix_lock)
        nodes = {lock.node for lock in locks if lock is not None}
        return nodes | self.chain_holds.locked_nodes()

    def _free_slots(self) -> int:
        free = self.kv_allocator.available_size() + self.tree_cache.evictable_size()
        return free - self.reserved_decode_tokens * (len(self.receptions) + 1)

    def _take_slots(self, num_page_slots: int) -> torch.Tensor:
        if self.kv_allocator.available_size() < num_page_slots:
            evict_from_tree_cache(self.tree_cache, num_page_slots)
        slots = self.kv_allocator.alloc(num_page_slots)
        if slots is None:
            raise RuntimeError(
                f"Franky could not take {num_page_slots} KV slots that fit: "
                f"{self.kv_allocator.available_size()} available."
            )
        return slots

    def _tap_pages(self, taken: torch.Tensor) -> FrankyHostPages:
        kv_pages = taken[:: self.page_size] // self.page_size
        tap_pages = self.tap_cache.pages_of(kv_pages=kv_pages)
        return FrankyHostPages(tap_pages.to(torch.int32))

    def _rank_numbers(self) -> dict[str, int]:
        numbers = {}
        for rid, reception in self.receptions.items():
            if reception.ended:
                numbers[f"closed:{rid}"] = int(self._room_is_closed(reception))
            elif not self._taps_complete(reception):
                ok = self._poll_receiver(reception)
                numbers[f"landed:{rid}"] = reception.landed
                numbers[f"ok:{rid}"] = int(ok)
        return numbers

    def _poll_receiver(self, reception: _Reception) -> bool:
        if reception.pages is not None and reception.pages.ready():
            reception.receiver = FrankyReceiver(
                manager=self.tap_manager,
                franky_room=reception.req.franky_room,
                tap_pages=reception.pages.numpy(),
                num_tokens=reception.req.franky.tap_len,
                prefix_len=reception.req.franky.taken_from,
            )
            reception.pages = None
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
        if self._taps_complete(reception) or reception.announced_failure:
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
        if self._taps_complete(reception):
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
            "%.0fs; releasing its pages anyway.",
            receiver.bootstrap_room,
            self.release_timeout_seconds,
        )
        return True

    def _release(self, reception: _Reception) -> None:
        req = reception.req
        state = req.franky
        unused_from = ceil_div(state.used_end - state.taken_from, self.page_size)
        self.kv_allocator.free(state.taken_kv_slots[unused_from * self.page_size :])
        self.tree_cache.unlock(reception.prefix_lock)
        if reception.receiver is not None:
            self.tap_manager.clear_deferred_abort_state(
                reception.receiver.bootstrap_room
            )
            reception.receiver.close()
        self._forget_keys(req.rid, "landed", "ok", "closed")
        del self.receptions[req.rid]

    def _taps_complete(self, reception: _Reception) -> bool:
        state = reception.req.franky
        return state.taps_ready >= state.tap_len

    def _forget_keys(self, rid: str, *names: str) -> None:
        for name in names:
            key = f"{name}:{rid}"
            self.agreed.pop(key, None)
            self.rank_reports.forget(key=key)
