from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import msgspec
import numpy as np
import torch

from sglang.kernels.ops.attention.franky_reader_blocks import (
    franky_reader_row_blocks,
)
from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.franky.cache.tap_request_table import TapRequestTable
    from sglang.srt.franky.cache.tap_stripes import TapStripes
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TapReading(msgspec.Struct, frozen=True, kw_only=True):
    """What the rows of one forward read on this rank, made once for all of its
    readers. Every rank attends over the taps it keeps, then the ranks merge.

    Attributes:
        request_pages: [requests, stripes] int32, the local stripe pages of each
            row's request; a last request reads nothing, for padding rows.
        row_requests: [rows] int32, the request of each row; None when row
            ``i`` is request ``i``.
        local_taps: [rows + 1] int32, how many of the taps each row reads this
            rank keeps; the last entry is a row that reads nothing.
        no_taps: [rows + 1] bool, whether each row reads no tap.
        has_taps: [rows + 1, 1], 1 for rows that read taps and 0 for the rest.
        num_rows: the real rows; a CUDA graph pads after them.
        row_blocks: the prefill kernel's row blocks; None for a decode forward.
    """

    request_pages: torch.Tensor
    row_requests: Optional[torch.Tensor]
    local_taps: torch.Tensor
    no_taps: torch.Tensor
    has_taps: torch.Tensor
    num_rows: int
    row_blocks: Optional[torch.Tensor] = None

    @staticmethod
    def of(
        forward_batch: ForwardBatch,
        *,
        request_table: TapRequestTable,
        stripes: TapStripes,
        page_size: int,
        mask_dtype: torch.dtype,
    ) -> Optional[TapReading]:
        """
        Args:
            mask_dtype: the dtype of ``has_taps``.
        Returns:
            The reading; None for an extend forward none of whose rows reads a
            tap, which the host knows. A decode forward's taps live on the
            device, so it always has one.
        """
        reading = (
            _decode_reading
            if forward_batch.forward_mode.is_decode_or_idle()
            else _extend_reading
        )
        return reading(
            forward_batch,
            request_table=request_table,
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
            request_pages=(
                self.request_pages[selected]
                if self.row_requests is None
                else self.request_pages
            ),
            row_requests=(
                None if self.row_requests is None else self.row_requests[selected]
            ),
            local_taps=self.local_taps[selected],
            no_taps=self.no_taps[selected],
            has_taps=self.has_taps[selected],
            num_rows=rows.shape[0],
        )

    @property
    def page_table(self) -> torch.Tensor:
        """[rows, stripes] int32, the local stripe pages of each real row's
        request."""
        if self.row_requests is None:
            return self.request_pages[: self.num_rows]
        return self.request_pages[self.row_requests[: self.num_rows]]


def _extend_reading(
    forward_batch: ForwardBatch,
    *,
    request_table: TapRequestTable,
    stripes: TapStripes,
    page_size: int,
    mask_dtype: torch.dtype,
) -> Optional[TapReading]:
    req_pool_indices = _on_host(
        forward_batch.req_pool_indices_cpu, forward_batch.req_pool_indices
    )
    first_tokens = _on_host(
        forward_batch.extend_prefix_lens_cpu, forward_batch.extend_prefix_lens
    )
    num_rows = list(forward_batch.extend_seq_lens_cpu)
    request_taps = [
        _taps_of_tokens(request_table, slot, first_token=first, num_tokens=count)
        for slot, first, count in zip(req_pool_indices, first_tokens, num_rows)
    ]
    if all(taps is None for taps in request_taps):
        return None
    taps = np.concatenate(
        [
            np.zeros(count, dtype=np.int64) if taps is None else taps
            for taps, count in zip(request_taps, num_rows)
        ]
        + [np.zeros(1, dtype=np.int64)]
    )
    max_taps = int(taps.max())
    if max_taps == 0:
        return None
    # One host-to-device copy: each row's request, its local taps, whether it
    # reads any, then the prefill kernel's row blocks.
    num_entries = taps.shape[0]
    host = torch.from_numpy(
        np.concatenate(
            [
                np.repeat(np.arange(len(num_rows) + 1, dtype=np.int32), num_rows + [1]),
                stripes.local_taps(taps, page_size=page_size).astype(np.int32),
                (taps > 0).astype(np.int32),
                franky_reader_row_blocks(num_rows, device=torch.device("cpu"))
                .numpy()
                .reshape(-1),
            ]
        )
    )
    device = request_table.stripe_pages.device
    values = _to_device(host, device=device)
    reads = values[2 * num_entries : 3 * num_entries]
    num_stripes = ceil_div(max_taps, stripes.num_ranks * page_size)
    request_pages = request_table.stripe_pages[
        forward_batch.req_pool_indices, :num_stripes
    ]
    return TapReading(
        request_pages=torch.cat(
            [request_pages, request_pages.new_zeros((1, num_stripes))]
        ),
        row_requests=values[:num_entries],
        local_taps=values[num_entries : 2 * num_entries],
        no_taps=reads == 0,
        has_taps=reads.to(mask_dtype)[:, None],
        num_rows=sum(num_rows),
        row_blocks=values[3 * num_entries :].view(-1, 3),
    )


def _decode_reading(
    forward_batch: ForwardBatch,
    *,
    request_table: TapRequestTable,
    stripes: TapStripes,
    page_size: int,
    mask_dtype: torch.dtype,
) -> TapReading:
    req_pool_indices = forward_batch.req_pool_indices
    request_pages = request_table.stripe_pages[req_pool_indices]
    taps = torch.cat(
        [
            request_table.decode_taps[req_pool_indices],
            request_table.decode_taps[:1] * 0,
        ]
    )
    reads = taps > 0
    return TapReading(
        request_pages=torch.cat([request_pages, request_pages[:1] * 0]),
        row_requests=None,
        local_taps=stripes.local_taps(taps, page_size=page_size),
        no_taps=~reads,
        has_taps=reads.to(mask_dtype)[:, None],
        num_rows=req_pool_indices.shape[0],
    )


def _taps_of_tokens(
    request_table: TapRequestTable, slot: int, *, first_token: int, num_tokens: int
) -> Optional[np.ndarray]:
    """
    Returns:
        int64, per token the forward makes, how many taps it reads; None for a
        request that reads none.
    """
    spans = request_table.spans_of(slot)
    if spans is None:
        return None
    return spans.taps_to_read(
        np.arange(first_token + 1, first_token + num_tokens + 1, dtype=np.int64)
    )


def _to_device(host: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return host
    return host.pin_memory().to(device, non_blocking=True)


def _on_host(
    host: Optional[Sequence[int] | torch.Tensor], device: torch.Tensor
) -> list[int]:
    values = device if host is None else host
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)
