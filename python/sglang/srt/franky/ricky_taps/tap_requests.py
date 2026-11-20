from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch

from sglang.srt.franky.cache.ricky_tap_cache import TapReservation
from sglang.srt.franky.request import FrankyRequest
from sglang.srt.franky.ricky_taps.tap_progress import TapProgress
from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.franky.cache.mamba_room import FrankyMambaRoom
    from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
    from sglang.srt.franky.cache.tap_request_table import TapRequestTable
    from sglang.srt.franky.chains.holds import FrankyChainHolds
    from sglang.srt.managers.schedule_batch import Req

_NO_TAPS_NAMESPACE = "franky=no-taps"


class _Waiting(msgspec.Struct, frozen=True, kw_only=True):
    req: Any
    sandy_tokens: Sequence[int]


class _Refused(msgspec.Struct, frozen=True, kw_only=True):
    req: Any
    why: str


class _Reserved(msgspec.Struct, frozen=True, kw_only=True):
    req: Any
    reservation: TapReservation


class RickyTapRequests:
    """Ricky's requests that read taps, by rid, from arrival to their end, on
    Ricky prefill, decode and aggregated engines alike.

    A request takes its stripes in arrival order, the blocks of held chains
    first: the cached prefix of its taps from the prefix tree, and new stripes
    for the rest, of which at least one page lands, since a transfer ends only
    once a page lands. It waits until they fit; only a held chain's next block
    may release other chains' holds for room. When it joins a batch its tokens
    are set for the readers and its chain's hold may go. At its end a request
    whose taps all landed leaves its whole stripes in the prefix tree, held for
    its chain's next block unless it ended its chain; a request aborted before
    that ends its chain's hold. Every rank decides at the same
    steps, so stripe pages stay the same on every rank.

    Where the engine has a mamba state pool, the chains' holds stay within its
    room: a hold pins the mamba state of its block's prompt.
    """

    def __init__(
        self,
        *,
        tap_cache: RickyTapCache,
        request_table: TapRequestTable,
        chain_holds: FrankyChainHolds,
        shared_tokenizer: bool,
        mamba_room: Optional[FrankyMambaRoom] = None,
    ) -> None:
        """
        Args:
            shared_tokenizer: whether Sandy reads Ricky's tokens, as the Franky
                config says; a request then sends no Sandy tokens of its own.
            mamba_room: the engine's mamba state pool; None without one, or on
                an engine that caches no prompts.
        """
        self._tap_cache = tap_cache
        self._request_table = request_table
        self._chain_holds = chain_holds
        self._shared_tokenizer = shared_tokenizer
        self._mamba_room = mamba_room
        self._max_taps = request_table.max_stripes * tap_cache.stripe_size
        self._waiting: dict[str, _Waiting] = {}
        self._reserved: dict[str, _Reserved] = {}
        self._refused: dict[str, _Refused] = {}
        self._rids_by_room: dict[int, str] = {}
        # The first waiting request that did not fit, and the tap pool's version
        # then: it cannot fit before the pool changes.
        self._blocked: Optional[tuple[str, int]] = None

    def arrive(self, req: Req) -> Optional[str]:
        """A request came; one back from a retraction keeps what it had. A request
        without a room reads no taps, so its KV is cached apart from KV that read
        taps.

        Returns:
            Why the request can never read its taps, or None.
        """
        if req.franky_taps is not None:
            return None
        request = FrankyRequest.of(req)
        if request.room is None:
            req.extra_key = (
                f"{req.extra_key}|{_NO_TAPS_NAMESPACE}"
                if req.extra_key
                else _NO_TAPS_NAMESPACE
            )
            req.franky_taps = TapProgress(tap_len=0, taps_ready=0)
            return None
        if req.bootstrap_room is not None:
            self._rids_by_room[req.bootstrap_room] = req.rid
        sandy_tokens = request.sandy_tokens(req.origin_input_ids)
        problem = self._problem_of(req, request, num_taps=len(sandy_tokens))
        if problem is not None:
            self._refused[req.rid] = _Refused(req=req, why=problem)
            return problem
        req.franky_taps = TapProgress(
            tap_len=len(sandy_tokens),
            taps_ready=0,
            spans=request.tap_runs,
        )
        self._waiting[req.rid] = _Waiting(req=req, sandy_tokens=sandy_tokens)
        return None

    def poll(self) -> list[Req]:
        """Every rank, at each step: release the holds past the mamba room, drop
        the waiting requests that finished, then reserve waiting ones in order
        while they fit.

        Returns:
            The requests reserved at this step, in order.
        """
        self._release_holds_past_the_mamba_room()
        self._forget_finished()
        in_order = [waiting.req for waiting in self._waiting.values()]
        self._chain_holds.put_held_chains_first(in_order)
        reserved = []
        for req in in_order:
            if self._blocked == (req.rid, self._tap_cache.version):
                break
            reservation = self._reserve(self._waiting[req.rid])
            if reservation is None:
                self._blocked = (req.rid, self._tap_cache.version)
                break
            del self._waiting[req.rid]
            self._reserved[req.rid] = _Reserved(req=req, reservation=reservation)
            reserved.append(req)
        return reserved

    def join_batch(self, reqs: list[Req]) -> None:
        """``reqs`` joined a batch: set their requests' entries for the readers.
        A block's prefix is now locked by the scheduler and by its reservation, so
        its chain's hold may go.
        """
        pages = []
        for req in reqs:
            reserved = self._reserved.get(req.rid)
            pages.append(None if reserved is None else reserved.reservation.pages)
            if reserved is not None:
                self._chain_holds.release_chain_of(req)
        self._request_table.bind(
            req_pool_indices=[req.kv.req_pool_idx for req in reqs],
            pages=pages,
            spans=[
                None if req.franky_taps is None else req.franky_taps.spans
                for req in reqs
            ],
            num_input_tokens=[len(req.origin_input_ids) for req in reqs],
        )

    def end(self, req: Req, *, landed: bool) -> None:
        """A reserved request ended, and no transfer writes its stripes any more.

        Args:
            landed: whether all of its taps landed; only then the prefix tree
                caches its whole stripes.
        """
        reservation = self._reserved.pop(req.rid).reservation
        self._rids_by_room.pop(req.bootstrap_room, None)
        if not landed:
            self._tap_cache.drop(reservation)
            self._chain_holds.release_chain_of(req)
            return
        self._chain_holds.hold_for_next_block(
            req, tap_lock=self._tap_cache.cache(reservation)
        )

    def forget(self, req: Req) -> None:
        """A request ended before it reserved anything, or was aborted while its
        stripes wait for its room to close; its chain's hold ends either way.
        """
        self._waiting.pop(req.rid, None)
        self._refused.pop(req.rid, None)
        self._rids_by_room.pop(req.bootstrap_room, None)
        self._chain_holds.release_chain_of(req)

    def num_startable_requests(self) -> Optional[int]:
        """
        Returns:
            How many requests may start beside the running ones as far as the
            mamba state pool goes, or None without such a pool.
        """
        if self._mamba_room is None:
            return None
        return self._mamba_room.num_startable_requests(
            num_pinned_states=len(self._chain_holds.locked_nodes())
        )

    def reservation(self, rid: str) -> Optional[TapReservation]:
        reserved = self._reserved.get(rid)
        return None if reserved is None else reserved.reservation

    def rid_of_room(self, room: int) -> Optional[str]:
        """The request of a PD room, for Ricky decode's relay."""
        return self._rids_by_room.get(room)

    def waits(self, rid: str) -> bool:
        return rid in self._waiting

    def refusal(self, rid: str) -> Optional[str]:
        """Why the request of ``rid`` was refused, or None."""
        refused = self._refused.get(rid)
        return None if refused is None else refused.why

    def reserved_requests(self) -> list[Req]:
        return [reserved.req for reserved in self._reserved.values()]

    def is_empty(self) -> bool:
        return not self._waiting and not self._reserved and not self._refused

    def _reserve(self, waiting: _Waiting) -> Optional[TapReservation]:
        page_size = self._tap_cache.buffers.page_size
        num_taps = len(waiting.sandy_tokens)
        max_cached_taps = (ceil_div(num_taps, page_size) - 1) * page_size
        cached = self._tap_cache.match(waiting.sandy_tokens[:max_cached_taps])
        num_new = ceil_div(num_taps, self._tap_cache.stripe_size) - len(cached.pages)
        if not self._fits_or_releases_holds(waiting.req, num_new):
            self._tap_cache.release(cached.lock)
            return None
        return TapReservation(
            sandy_ids=waiting.sandy_tokens,
            pages=torch.cat([cached.pages, self._tap_cache.take(num_new)]),
            cached=cached,
        )

    def _fits_or_releases_holds(self, req: Req, num_new: int) -> bool:
        if not self._chain_holds.holds_chain_of(req):
            return self._tap_cache.num_takeable_stripes() >= num_new
        while self._tap_cache.num_takeable_stripes() < num_new:
            if not self._chain_holds.release_least_progress(
                sparing_chain_id=FrankyRequest.of(req).chain.id
            ):
                return False
        return True

    def _release_holds_past_the_mamba_room(self) -> None:
        if self._mamba_room is None:
            return
        while not self._mamba_room.fits_pinned_states(
            num_pinned_states=len(self._chain_holds.locked_nodes())
        ):
            if not self._chain_holds.release_least_progress(sparing_chain_id=None):
                return

    def _problem_of(
        self, req: Req, request: FrankyRequest, *, num_taps: int
    ) -> Optional[str]:
        if self._shared_tokenizer and request.sandy_input_ids is not None:
            return (
                f"Request {req.rid} sends franky_sandy_input_ids, but this Ricky's "
                "Sandy reads its tokens."
            )
        if not self._shared_tokenizer and request.sandy_input_ids is None:
            return (
                f"Request {req.rid} reads taps of a Sandy that tokenizes on its own, "
                "so it needs franky_sandy_input_ids."
            )
        if num_taps > self._max_taps:
            return (
                f"Sandy's {num_taps} tokens of request {req.rid} are past the "
                f"{self._max_taps} taps a request may read on Ricky."
            )
        return None

    def _forget_finished(self) -> None:
        finished = [
            entry.req
            for entry in (*self._waiting.values(), *self._refused.values())
            if entry.req.finished()
        ]
        for req in finished:
            self.forget(req)
