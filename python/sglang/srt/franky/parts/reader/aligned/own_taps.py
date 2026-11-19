from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.utils.common import async_h2d

if TYPE_CHECKING:
    from sglang.srt.franky.cache.tap_request_table import TapRequestTable
    from sglang.srt.franky.cache.tap_stripes import TapStripes
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class OwnTaps(msgspec.Struct, frozen=True, kw_only=True):
    """One encoder's tap at each row's own input token, for the rows of one
    extend forward. A rank holds only the taps of its stripes, so each rank reads
    the rows whose tap page it keeps and gives zeros elsewhere; summed over the
    attention-TP ranks, every row has its tap.

    Attributes:
        taps: [rows, output_size], this rank's part of the tap Sandy wrote at
            each row's input token. CUDA graph padding gets zeros.
        reads: [rows] bool, whether the row reads it: its input token is in its
            request's write ranges. Answer rows and CUDA graph padding read none.
        positions: per row, its input token's index in its request; -1 for padding.
        requests: per row, its request's index in the batch; -1 for padding.
    """

    taps: torch.Tensor
    reads: torch.Tensor
    positions: list[int]
    requests: list[int]

    @staticmethod
    def of(
        forward_batch: ForwardBatch,
        *,
        pages: torch.Tensor,
        request_table: TapRequestTable,
        stripes: TapStripes,
        num_rows: int,
    ) -> OwnTaps:
        """
        Args:
            pages: [pages, page_size, 1, output_size], this rank's tap pages of
                the encoder.
            num_rows: the forward's rows, CUDA graph padding included.
        """
        req_pool_indices = _on_host(
            forward_batch.req_pool_indices_cpu, forward_batch.req_pool_indices
        )
        first_rows = _on_host(
            forward_batch.extend_prefix_lens_cpu, forward_batch.extend_prefix_lens
        )
        reads: list[bool] = []
        positions: list[int] = []
        requests: list[int] = []
        slots: list[int] = []
        for request, (slot, first_row, count) in enumerate(
            zip(req_pool_indices, first_rows, forward_batch.extend_seq_lens_cpu)
        ):
            positions.extend(range(first_row, first_row + count))
            requests.extend([request] * count)
            slots.extend([slot] * count)
            spans = request_table.spans_of(slot)
            if spans is None:
                reads.extend([False] * count)
                continue
            tokens = torch.arange(first_row, first_row + count)
            reads.extend(spans.written_tokens(tokens).tolist())
        padding = num_rows - len(reads)
        reads.extend([False] * padding)
        positions.extend([-1] * padding)
        requests.extend([-1] * padding)
        slots.extend([0] * padding)
        num_pages, page_size, _, output_size = pages.shape
        device = pages.device
        position = async_h2d(
            [max(row, 0) for row in positions], dtype=torch.int64, device=device
        )
        reading = async_h2d(reads, dtype=torch.bool, device=device)
        request_pages = position // page_size
        kept = reading & stripes.owns(request_pages)
        # Rows that read nothing look up their request's first stripe.
        local_pages = request_table.stripe_pages[
            async_h2d(slots, dtype=torch.int64, device=device),
            torch.where(reading, stripes.stripe(request_pages), 0),
        ].to(torch.int64)
        taps = pages.view(num_pages * page_size, output_size)[
            local_pages * page_size + position % page_size
        ]
        return OwnTaps(
            taps=torch.where(kept[:, None], taps, 0),
            reads=reading,
            positions=positions,
            requests=requests,
        )


def _on_host(
    host: Optional[Sequence[int] | torch.Tensor], device: torch.Tensor
) -> list[int]:
    # The prefill graph's warmup and capture batches keep these on the device only.
    values = device if host is None else host
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)
