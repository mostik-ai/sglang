from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.franky.parts.reader.tap_reading import TapReading

if TYPE_CHECKING:
    from sglang.srt.franky.cache.tap_request_table import TapRequestTable
    from sglang.srt.franky.cache.tap_stripes import TapStripes
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class CurrentTapReading:
    """The ``TapReading`` of the forward that runs on this rank, shared by the
    readers of one PP stage: the stage's first reader makes it, the others read
    it.
    """

    def __init__(
        self, *, request_table: TapRequestTable, stripes: TapStripes, page_size: int
    ) -> None:
        self.request_table = request_table
        self.stripes = stripes
        self.page_size = page_size
        self._reading: Optional[TapReading] = None

    def start(
        self, forward_batch: ForwardBatch, *, mask_dtype: torch.dtype
    ) -> TapReading:
        self._reading = TapReading.of(
            forward_batch,
            request_table=self.request_table,
            stripes=self.stripes,
            page_size=self.page_size,
            mask_dtype=mask_dtype,
        )
        return self._reading

    def get(self) -> TapReading:
        return self._reading
