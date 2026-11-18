from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import msgspec
import numpy as np
import torch

from sglang.kernels.ops.attention.franky_reader_blocks import (
    franky_reader_row_blocks,
)
from sglang.srt.franky.cache.generated_spans import FrankyGeneratedSpans
from sglang.srt.franky.cache.own_tap_pages import FrankyOwnTapPages
from sglang.srt.utils.common import async_h2d

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

TAP_DTYPE = torch.bfloat16


class FrankyReqState(msgspec.Struct, kw_only=True):
    """One Ricky request's Franky state on this rank. Taps count Sandy's tokens.

    Attributes:
        generated_spans: which taps each row reads; None without a room.
        own_tap_pages: the pages the request's taps land in when Sandy tokenizes
            on its own; None when they land at its KV slots.
    """

    tap_len: int
    taps_ready: int
    # KV slots taken at arrival for the prompt from token ``taken_from`` on.
    # They stay out of ``req.kv`` until a chunk runs them, so KV code sees a plain
    # chunked request.
    taken_kv_slots: torch.Tensor
    taken_from: int
    used_end: int
    generated_spans: Optional[FrankyGeneratedSpans] = None
    own_tap_pages: Optional[torch.Tensor] = None

    def rows_ready(self) -> int:
        """
        Returns:
            How many of the request's first rows may run: rows whose forward reads
            only landed taps. With taps at KV slots, also only rows whose own taps
            landed, since the prefix tree hands their KV pages on with the taps.
        """
        if self.own_tap_pages is None:
            return self.taps_ready
        return self.generated_spans.rows_reading_at_most(self.taps_ready)

    def hand_to_chunk(self, *, start: int, end: int) -> torch.Tensor:
        """The taken KV slots of prompt tokens ``[start, end)``, which a chunk now
        uses.
        """
        self.used_end = end
        return self.taken_kv_slots[start - self.taken_from : end - self.taken_from]


class TrtllmWorkspace(msgspec.Struct, frozen=True, kw_only=True):
    """What the readers' TRT-LLM kernels keep across launches on Blackwell.

    Attributes:
        buffer: scratch for split-KV partials, zeroed once.
        counters: the split-KV completion tickets, zeroed once; every launch
            resets them. Without them FlashInfer allocates fresh tickets per
            launch, whose memory the allocator, or a CUDA graph's pool, hands
            to a later kernel while the launch may still run.
    """

    buffer: torch.Tensor
    counters: torch.Tensor


class RowMasks(msgspec.Struct, frozen=True, kw_only=True):
    """Which rows of a forward read taps, padding rows reading none, made once
    for all of its readers.

    Attributes:
        no_taps: [rows] bool, rows that read no taps.
        has_taps: [rows, 1], 1 for rows that read taps and 0 for the rest.
    """

    no_taps: torch.Tensor
    has_taps: torch.Tensor


class TapView(msgspec.Struct, frozen=True, kw_only=True):
    """One encoder's taps for the rows of one forward, as the paged attention
    kernels take them.

    Attributes:
        taps: [pages, page_size, 1, output_size], both the keys and the values.
        request_pages: [requests, pages] int32, the tap pages of each request.
        row_requests: [rows], the request of each row; None when row i is
            request i.
        taps_to_read: [rows] int32, the taps each row's query reads, from tap 0.
        num_tokens: the real rows; a CUDA graph pads after them.
        trtllm_workspace: TRT-LLM's workspace on Blackwell; None where FA3 runs.
        row_blocks: the prefill kernel's row blocks when the rows are every
            row of the forward, whole requests one after another; else None.
        row_masks: when the rows are every row of the forward, the
            ``RowMasks`` its readers share, padding rows included; else None.
    """

    taps: torch.Tensor
    request_pages: torch.Tensor
    row_requests: Optional[torch.Tensor]
    taps_to_read: torch.Tensor
    num_tokens: int
    trtllm_workspace: Optional[TrtllmWorkspace]
    row_blocks: Optional[torch.Tensor] = None
    row_masks: Optional[RowMasks] = None

    @property
    def page_table(self) -> torch.Tensor:
        """[rows, pages] int32, the pages of each row's request."""
        if self.row_requests is None:
            return self.request_pages
        return self.request_pages[self.row_requests]


class FrankyTapCache:
    """This rank's taps: one tensor per encoder, a tap at each KV slot.

    A token's tap lives at its KV slot, so taps are taken and freed with KV, one
    memory budget covers both, and neither needs an allocator of its own. KV
    never reads taps, and other code goes through ``slots_of`` and ``pages_of``.

    Outside the KV allocator taps are seen in tap pages of ``page_size`` tokens:
    the transfer moves them and the readers attend over them. A tap page is a view
    of the slots of a KV page, so KV pages may be any multiple of it and two
    engines with different KV pages still exchange taps page to page, in place.

    On a Ricky whose Sandy tokenizes on its own, taps belong to Sandy's tokens,
    which have no KV slots here; requests then own tap pages of the same buffers,
    from ``own_pages``, and readers find them there.
    """

    def __init__(
        self,
        *,
        encoder_ids: tuple[int, ...],
        output_size: int,
        page_size: int,
        kv_page_size: int,
        num_tokens: int,
        req_to_token: torch.Tensor,
        device: str,
        own_pages: bool = False,
        attention_workspace: Optional[TrtllmWorkspace] = None,
    ) -> None:
        """
        Args:
            page_size: tokens of a tap page; it divides ``kv_page_size``.
            num_tokens: tokens whose taps fit past the first KV page: the KV
                slots, or with ``own_pages`` as many as the tap budget holds.
            own_pages: whether requests own tap pages apart from their KV slots.
        """
        if kv_page_size % page_size != 0:
            raise ValueError(
                f"A KV page of {kv_page_size} tokens is not whole tap pages of "
                f"{page_size} tokens."
            )
        self.page_size = page_size
        self.page_bytes = page_size * output_size * TAP_DTYPE.itemsize
        self.pages_per_kv_page = kv_page_size // page_size
        self._output_size = output_size
        self._req_to_token = req_to_token
        self._attention_workspace = attention_workspace
        self._decode_taps_to_read = torch.zeros(
            req_to_token.shape[0], dtype=torch.int32, device=device
        )
        self._generated_spans: list[Optional[FrankyGeneratedSpans]] = [
            None
        ] * req_to_token.shape[0]
        self._plan: Optional[_RowPlan] = None
        self._last_encoder_id = -1
        self._max_pages = -(-req_to_token.shape[1] // page_size)
        self.own_pages = (
            FrankyOwnTapPages(
                first_page=self.pages_per_kv_page,
                num_pages=(num_tokens + kv_page_size) // page_size,
                max_pages_per_request=self._max_pages,
                num_request_rows=req_to_token.shape[0],
                device=device,
            )
            if own_pages
            else None
        )
        # One more KV page, as in the KV pool: padded tokens write to page 0, and
        # padded requests read its zeros.
        self._buffers = {
            encoder_id: torch.zeros(
                (num_tokens + kv_page_size, output_size),
                dtype=TAP_DTYPE,
                device=device,
            )
            for encoder_id in encoder_ids
        }

    @property
    def encoder_ids(self) -> tuple[int, ...]:
        return tuple(self._buffers)

    def transfer_buffers(self) -> dict[int, tuple[int, int]]:
        """Encoder id to the address and bytes of its tap buffer; tap page ``p``
        is at ``address + p * page_bytes``.
        """
        return {
            encoder_id: (buffer.data_ptr(), buffer.nbytes)
            for encoder_id, buffer in self._buffers.items()
        }

    def pages_of(self, *, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: consecutive tokens' KV slots, from a tap page's start.
        Returns:
            The tap pages that hold them, in order.
        """
        return slots[:: self.page_size] // self.page_size

    def pages_of_kv_pages(self, *, kv_pages: np.ndarray) -> np.ndarray:
        """
        Returns:
            int32, the tap pages of ``kv_pages``, in order.
        """
        first_pages = np.asarray(kv_pages, dtype=np.int64) * self.pages_per_kv_page
        offsets = np.arange(self.pages_per_kv_page, dtype=np.int64)
        return (first_pages[:, None] + offsets).reshape(-1).astype(np.int32)

    def slots_of(self, *, kv_slots: torch.Tensor) -> torch.Tensor:
        return kv_slots

    def write(
        self, *, encoder_id: int, taps: torch.Tensor, kv_slots: torch.Tensor
    ) -> None:
        self._buffers[encoder_id][self.slots_of(kv_slots=kv_slots)] = taps

    def set_generated_spans(
        self,
        *,
        req_pool_indices: list[int],
        generated_spans: list[Optional[FrankyGeneratedSpans]],
        num_input_tokens: list[int],
    ) -> None:
        """Which taps the forward of each token of these requests reads; a request
        with None reads none.

        Args:
            num_input_tokens: per request, its input length: its decoded block's start.
        """
        device = self._decode_taps_to_read.device
        decoded = [
            0 if spans is None else int(spans.taps_to_read(torch.tensor([first]))[0])
            for spans, first in zip(generated_spans, num_input_tokens)
        ]
        rows = async_h2d(req_pool_indices, dtype=torch.int64, device=device)
        self._decode_taps_to_read[rows] = async_h2d(
            decoded, dtype=torch.int32, device=device
        )
        for req_pool_index, spans in zip(req_pool_indices, generated_spans):
            self._generated_spans[req_pool_index] = spans

    def view(
        self,
        *,
        encoder_id: int,
        forward_batch: ForwardBatch,
        rows: Optional[torch.Tensor] = None,
        padded_rows: Optional[int] = None,
        mask_dtype: Optional[torch.dtype] = None,
    ) -> TapView:
        """Each row's query reads the taps that its request's spans give the token
        the row makes.

        Args:
            rows: the rows of ``forward_batch`` the queries belong to, as indices
                into them; None for all of them. An index past the last row reads
                nothing.
            padded_rows, mask_dtype: with ``rows`` None, the rows the readers'
                outputs have, padding included, and the dtype of their
                ``has_taps``; the view then carries the forward's ``RowMasks``.
        """
        plan = self._plan_of(forward_batch, encoder_id=encoder_id)
        if rows is None:
            return TapView(
                row_masks=(
                    None
                    if padded_rows is None
                    else plan.row_masks(padded_rows=padded_rows, dtype=mask_dtype)
                ),
                taps=self._paged_taps(encoder_id),
                request_pages=(
                    plan.request_pages[: plan.num_rows]
                    if plan.row_requests is None
                    else plan.request_pages
                ),
                row_requests=(
                    None
                    if plan.row_requests is None
                    else plan.row_requests[: plan.num_rows]
                ),
                taps_to_read=plan.taps_to_read[: plan.num_rows],
                num_tokens=plan.num_rows,
                trtllm_workspace=self._attention_workspace,
                row_blocks=plan.row_blocks,
            )
        selected = rows.clamp(max=plan.num_rows)
        return TapView(
            taps=self._paged_taps(encoder_id),
            request_pages=(
                plan.request_pages[selected]
                if plan.row_requests is None
                else plan.request_pages
            ),
            row_requests=(
                None if plan.row_requests is None else plan.row_requests[selected]
            ),
            taps_to_read=plan.taps_to_read[selected],
            num_tokens=rows.shape[0],
            trtllm_workspace=self._attention_workspace,
        )

    def _plan_of(self, forward_batch: ForwardBatch, *, encoder_id: int) -> _RowPlan:
        """The rows' pages and tap counts of one forward, built by its first reader
        for all of them. Readers view their encoders in increasing order within a
        forward, so an encoder at or below the last one starts the next forward.
        A CUDA graph runner warms a batch up and captures it on the same tensors;
        a plan kept across forwards would leave the captured forward reading the
        warmup's freed plan instead of building its own.
        """
        if self._plan is None or encoder_id <= self._last_encoder_id:
            self._plan = (
                self._decode_plan(forward_batch)
                if forward_batch.forward_mode.is_decode_or_idle()
                else self._extend_plan(forward_batch)
            )
        self._last_encoder_id = encoder_id
        return self._plan

    def _extend_plan(self, forward_batch: ForwardBatch) -> _RowPlan:
        req_pool_indices = _on_host(
            forward_batch.req_pool_indices_cpu, forward_batch.req_pool_indices
        )
        first_rows = _on_host(
            forward_batch.extend_prefix_lens_cpu, forward_batch.extend_prefix_lens
        )
        num_rows = list(forward_batch.extend_seq_lens_cpu)
        taps_to_read = torch.cat(
            [
                self._taps_to_read(req_pool_index, first_row=first_row, num_rows=count)
                for req_pool_index, first_row, count in zip(
                    req_pool_indices, first_rows, num_rows
                )
            ]
            + [torch.zeros(1, dtype=torch.int64)]
        )
        row_requests = torch.repeat_interleave(
            torch.arange(len(num_rows) + 1),
            torch.tensor(num_rows + [1], dtype=torch.int64),
        )
        device = self._decode_taps_to_read.device
        num_pages = max(1, -(-int(taps_to_read.max()) // self.page_size))
        page_table = self._page_table(
            forward_batch.req_pool_indices, num_pages=num_pages
        )
        return _RowPlan(
            request_pages=torch.cat([page_table, page_table.new_zeros((1, num_pages))]),
            row_requests=_to_device(row_requests.to(torch.int32), device=device),
            taps_to_read=_to_device(taps_to_read.to(torch.int32), device=device),
            num_rows=sum(num_rows),
            row_blocks=franky_reader_row_blocks(num_rows, device=device),
        )

    def _decode_plan(self, forward_batch: ForwardBatch) -> _RowPlan:
        req_pool_indices = forward_batch.req_pool_indices
        page_table = self._page_table(req_pool_indices, num_pages=self._max_pages)
        taps_to_read = self._decode_taps_to_read[req_pool_indices]
        return _RowPlan(
            request_pages=torch.cat([page_table, page_table[:1] * 0]),
            row_requests=None,
            taps_to_read=torch.cat([taps_to_read, taps_to_read[:1] * 0]),
            num_rows=req_pool_indices.shape[0],
        )

    def _taps_to_read(
        self, req_pool_index: int, *, first_row: int, num_rows: int
    ) -> torch.Tensor:
        rows = torch.arange(first_row, first_row + num_rows)
        spans = self._generated_spans[req_pool_index]
        if spans is None:
            return torch.zeros_like(rows)
        made_tokens = rows + 1
        return spans.taps_to_read(made_tokens)

    def _paged_taps(self, encoder_id: int) -> torch.Tensor:
        return self._buffers[encoder_id].view(-1, self.page_size, 1, self._output_size)

    def _page_table(
        self, req_pool_indices: torch.Tensor, *, num_pages: int
    ) -> torch.Tensor:
        if self.own_pages is not None:
            return self.own_pages.page_table[req_pool_indices, :num_pages]
        first_slots = self._req_to_token[
            req_pool_indices, : num_pages * self.page_size : self.page_size
        ]
        return (first_slots // self.page_size).to(torch.int32)


class _RowPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Per row of one forward: its request's tap pages and how many taps its query
    reads, with one more row that reads nothing.
    """

    request_pages: torch.Tensor
    row_requests: Optional[torch.Tensor]
    taps_to_read: torch.Tensor
    num_rows: int
    row_blocks: Optional[torch.Tensor] = None
    made_masks: dict[tuple[int, torch.dtype], RowMasks] = msgspec.field(
        default_factory=dict
    )

    def row_masks(self, *, padded_rows: int, dtype: torch.dtype) -> RowMasks:
        key = (padded_rows, dtype)
        if key not in self.made_masks:
            num_rows = min(self.num_rows, padded_rows)
            no_taps = torch.ones(
                padded_rows, dtype=torch.bool, device=self.taps_to_read.device
            )
            no_taps[:num_rows] = self.taps_to_read[:num_rows] <= 0
            self.made_masks[key] = RowMasks(
                no_taps=no_taps, has_taps=(~no_taps).to(dtype)[:, None]
            )
        return self.made_masks[key]


def _to_device(host: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return host
    return host.pin_memory().to(device, non_blocking=True)


def _on_host(
    host: Optional[Sequence[int] | torch.Tensor], device: torch.Tensor
) -> list[int]:
    # The prefill graph's warmup and capture batches keep these on the device only.
    values = device if host is None else host
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)
