from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from sglang.srt.franky.cache.generated_spans import FrankyGeneratedSpans
from sglang.srt.utils.common import async_h2d, async_h2d_stack


class TapRequestTable:
    """Per request slot of Ricky's ``req_to_token`` pool, what its readers read:
    the local pages of the request's stripes, its generated spans, and, once it
    decodes, how many taps the forward of its next token reads. A request's slot
    is set when it joins a batch.

    Attributes:
        stripe_pages: [slots, stripes] int32, each request's local stripe pages in
            order.
        decode_taps: [slots] int32, the taps a decoding request's next forward
            reads.
    """

    def __init__(self, *, num_slots: int, max_stripes: int, device: str) -> None:
        self.stripe_pages = torch.zeros(
            (num_slots, max_stripes), dtype=torch.int32, device=device
        )
        self.decode_taps = torch.zeros(num_slots, dtype=torch.int32, device=device)
        self._spans: list[Optional[FrankyGeneratedSpans]] = [None] * num_slots

    def bind(
        self,
        *,
        req_pool_indices: list[int],
        pages: list[Optional[torch.Tensor]],
        spans: list[Optional[FrankyGeneratedSpans]],
        num_input_tokens: list[int],
    ) -> None:
        """Set the slots of requests that join a batch; a request without pages
        reads no taps.

        Args:
            pages: per request, int64 host local stripe pages in order, or None.
            num_input_tokens: per request, its input length, where its decoded
                block starts.
        """
        device = self.decode_taps.device
        slots = async_h2d(req_pool_indices, dtype=torch.int64, device=device)
        self.decode_taps[slots] = async_h2d(
            [
                _decode_taps(request_spans, first)
                for request_spans, first in zip(spans, num_input_tokens)
            ],
            dtype=torch.int32,
            device=device,
        )
        for slot, slot_spans in zip(req_pool_indices, spans):
            self._spans[slot] = slot_spans
        reading = [
            (slot, slot_pages)
            for slot, slot_pages in zip(req_pool_indices, pages)
            if slot_pages is not None
        ]
        if reading:
            self._bind_pages(reading)

    def spans_of(self, slot: int) -> Optional[FrankyGeneratedSpans]:
        return self._spans[slot]

    def _bind_pages(self, reading: list[tuple[int, torch.Tensor]]) -> None:
        width = max(len(slot_pages) for _, slot_pages in reading)
        padded = [
            torch.nn.functional.pad(
                slot_pages.to(torch.int32), (0, width - len(slot_pages))
            )
            for _, slot_pages in reading
        ]
        device = self.stripe_pages.device
        slots = async_h2d(
            [slot for slot, _ in reading], dtype=torch.int64, device=device
        )
        self.stripe_pages[slots, :width] = async_h2d_stack(padded, device=device)

    @property
    def max_stripes(self) -> int:
        return self.stripe_pages.shape[1]


def _decode_taps(spans: Optional[FrankyGeneratedSpans], first_decoded: int) -> int:
    if spans is None:
        return 0
    return int(spans.taps_to_read(np.array([first_decoded]))[0])
