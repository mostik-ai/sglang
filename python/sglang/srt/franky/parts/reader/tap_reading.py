from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.kernels.ops.attention.franky_reader_blocks import (
    franky_reader_row_blocks,
)
from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.franky.cache.tap_read_rows import TapReadRows
    from sglang.srt.franky.cache.tap_stripes import TapStripes
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TapReading(msgspec.Struct, frozen=True, kw_only=True):
    """What the rows of one forward read on this rank, made once for all of its
    readers. Every rank attends over the taps it keeps, then the ranks merge.

    Attributes:
        request_slots: [requests, stripes] int32, the stripe slots of each
            row's request; a last request reads nothing, for padding rows.
        row_requests: [rows] int32, the request of each row; None when row
            ``i`` is request ``i``.
        local_taps: [rows + 1] int32, how many of the taps each row reads this
            rank keeps; the last entry is a row that reads nothing.
        reads: [rows + 1] bool, whether each row reads any tap.
        no_taps: [rows + 1] bool, the opposite of ``reads``.
        has_taps: [rows + 1, 1], 1 for rows that read taps and 0 for the rest.
        num_rows: the real rows; a CUDA graph pads after them.
        merged_rows: the rows whose results the ranks merge; None for all real
            rows.
        row_blocks: the prefill kernel's row blocks; None for a decode forward.
    """

    request_slots: torch.Tensor
    row_requests: Optional[torch.Tensor]
    local_taps: torch.Tensor
    reads: torch.Tensor
    no_taps: torch.Tensor
    has_taps: torch.Tensor
    num_rows: int
    merged_rows: Optional[torch.Tensor] = None
    row_blocks: Optional[torch.Tensor] = None

    @staticmethod
    def of(
        forward_batch: ForwardBatch,
        *,
        read_rows: TapReadRows,
        stripes: TapStripes,
        page_size: int,
        mask_dtype: torch.dtype,
    ) -> TapReading:
        """
        Args:
            mask_dtype: the dtype of ``has_taps``.
        """
        reading = (
            _decode_reading
            if forward_batch.forward_mode.is_decode_or_idle()
            else _extend_reading
        )
        return reading(
            forward_batch,
            read_rows=read_rows,
            stripes=stripes,
            page_size=page_size,
            mask_dtype=mask_dtype,
        )

    def select(self, rows: torch.Tensor) -> TapReading:
        """The reading of some rows of the forward, as indices into them; an
        index past the last row reads nothing.
        """
        selected = rows.clamp(max=self.num_rows)
        return TapReading(
            request_slots=(
                self.request_slots[selected]
                if self.row_requests is None
                else self.request_slots
            ),
            row_requests=(
                None if self.row_requests is None else self.row_requests[selected]
            ),
            local_taps=self.local_taps[selected],
            reads=self.reads[selected],
            no_taps=self.no_taps[selected],
            has_taps=self.has_taps[selected],
            num_rows=rows.shape[0],
        )

    @property
    def slot_table(self) -> torch.Tensor:
        """[rows, stripes] int32, the stripe slots of each real row's request."""
        if self.row_requests is None:
            return self.request_slots[: self.num_rows]
        return self.request_slots[self.row_requests[: self.num_rows]]


def _extend_reading(
    forward_batch: ForwardBatch,
    *,
    read_rows: TapReadRows,
    stripes: TapStripes,
    page_size: int,
    mask_dtype: torch.dtype,
) -> TapReading:
    req_pool_indices = _on_host(
        forward_batch.req_pool_indices_cpu, forward_batch.req_pool_indices
    )
    first_rows = _on_host(
        forward_batch.extend_prefix_lens_cpu, forward_batch.extend_prefix_lens
    )
    num_rows = list(forward_batch.extend_seq_lens_cpu)
    taps = torch.cat(
        [
            _taps_of_rows(read_rows, row, first_row=first_row, num_rows=count)
            for row, first_row, count in zip(req_pool_indices, first_rows, num_rows)
        ]
        + [torch.zeros(1, dtype=torch.int64)]
    )
    device = read_rows.stripe_slots.device
    total_rows = sum(num_rows)
    num_stripes = max(1, ceil_div(int(taps.max()), stripes.num_ranks * page_size))
    request_slots = read_rows.stripe_slots[forward_batch.req_pool_indices, :num_stripes]
    reads = taps > 0
    reads_on_device = _to_device(reads, device=device)
    return TapReading(
        request_slots=torch.cat(
            [request_slots, request_slots.new_zeros((1, num_stripes))]
        ),
        row_requests=_to_device(
            torch.repeat_interleave(
                torch.arange(len(num_rows) + 1, dtype=torch.int32),
                torch.tensor(num_rows + [1], dtype=torch.int64),
            ),
            device=device,
        ),
        local_taps=_to_device(
            stripes.local_taps(taps, page_size=page_size).to(torch.int32),
            device=device,
        ),
        reads=reads_on_device,
        no_taps=~reads_on_device,
        has_taps=reads_on_device.to(mask_dtype)[:, None],
        num_rows=total_rows,
        merged_rows=_to_device(torch.nonzero(reads).flatten(), device=device),
        row_blocks=franky_reader_row_blocks(num_rows, device=device),
    )


def _decode_reading(
    forward_batch: ForwardBatch,
    *,
    read_rows: TapReadRows,
    stripes: TapStripes,
    page_size: int,
    mask_dtype: torch.dtype,
) -> TapReading:
    req_pool_indices = forward_batch.req_pool_indices
    request_slots = read_rows.stripe_slots[req_pool_indices]
    taps = torch.cat(
        [read_rows.decode_taps[req_pool_indices], read_rows.decode_taps[:1] * 0]
    )
    reads = taps > 0
    return TapReading(
        request_slots=torch.cat([request_slots, request_slots[:1] * 0]),
        row_requests=None,
        local_taps=stripes.local_taps(taps, page_size=page_size),
        reads=reads,
        no_taps=~reads,
        has_taps=reads.to(mask_dtype)[:, None],
        num_rows=req_pool_indices.shape[0],
    )


def _taps_of_rows(
    read_rows: TapReadRows, row: int, *, first_row: int, num_rows: int
) -> torch.Tensor:
    spans = read_rows.spans_of(row)
    made_tokens = torch.arange(first_row + 1, first_row + num_rows + 1)
    if spans is None:
        return torch.zeros_like(made_tokens)
    return spans.taps_to_read(made_tokens)


def _to_device(host: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return host
    return host.pin_memory().to(device, non_blocking=True)


def _on_host(
    host: Optional[Sequence[int] | torch.Tensor], device: torch.Tensor
) -> list[int]:
    values = device if host is None else host
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)
