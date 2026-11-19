from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.franky.cache.generated_spans import FrankyGeneratedSpans
from sglang.srt.utils.common import async_h2d, async_h2d_stack


class TapReadRows:
    """What each request row of Ricky's ``req_to_token`` pool reads: the stripe
    slots of its taps, its generated spans, and, once it decodes, how many taps
    the forward of its next token reads. A request's row is set when it joins a
    batch.

    Attributes:
        stripe_slots: [rows, stripes] int32, each row's stripe slots in order.
        decode_taps: [rows] int32, the taps a decoding row's forward reads.
    """

    def __init__(self, *, num_rows: int, max_stripes: int, device: str) -> None:
        self.stripe_slots = torch.zeros(
            (num_rows, max_stripes), dtype=torch.int32, device=device
        )
        self.decode_taps = torch.zeros(num_rows, dtype=torch.int32, device=device)
        self._spans: list[Optional[FrankyGeneratedSpans]] = [None] * num_rows

    def bind(
        self,
        *,
        req_pool_indices: list[int],
        slots: list[Optional[torch.Tensor]],
        spans: list[Optional[FrankyGeneratedSpans]],
        num_input_tokens: list[int],
    ) -> None:
        """Set the rows of requests that join a batch; a request without slots
        reads no taps.

        Args:
            slots: per request, int64 host stripe slots in order, or None.
            num_input_tokens: per request, its input length, where its decoded
                block starts.
        """
        device = self.decode_taps.device
        rows = async_h2d(req_pool_indices, dtype=torch.int64, device=device)
        self.decode_taps[rows] = async_h2d(
            [
                _decode_taps(row_spans, first)
                for row_spans, first in zip(spans, num_input_tokens)
            ],
            dtype=torch.int32,
            device=device,
        )
        for row, row_spans in zip(req_pool_indices, spans):
            self._spans[row] = row_spans
        reading = [
            (row, row_slots)
            for row, row_slots in zip(req_pool_indices, slots)
            if row_slots is not None
        ]
        if reading:
            self._bind_slots(reading)

    def spans_of(self, row: int) -> Optional[FrankyGeneratedSpans]:
        return self._spans[row]

    def _bind_slots(self, reading: list[tuple[int, torch.Tensor]]) -> None:
        width = max(len(row_slots) for _, row_slots in reading)
        padded = [
            torch.nn.functional.pad(
                row_slots.to(torch.int32), (0, width - len(row_slots))
            )
            for _, row_slots in reading
        ]
        device = self.stripe_slots.device
        rows = async_h2d([row for row, _ in reading], dtype=torch.int64, device=device)
        self.stripe_slots[rows, :width] = async_h2d_stack(padded, device=device)


def _decode_taps(spans: Optional[FrankyGeneratedSpans], first_decoded: int) -> int:
    if spans is None:
        return 0
    return int(spans.taps_to_read(torch.tensor([first_decoded]))[0])
