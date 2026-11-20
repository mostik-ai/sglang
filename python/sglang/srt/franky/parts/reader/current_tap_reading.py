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
    it. None for a forward that reads no taps.
    """

    def __init__(
        self, *, request_table: TapRequestTable, stripes: TapStripes, page_size: int
    ) -> None:
        self.request_table = request_table
        self.stripes = stripes
        self.page_size = page_size
        self._reading: Optional[TapReading] = None
        self._scratch_key: Optional[tuple] = None
        self._scratch: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def start(
        self, forward_batch: ForwardBatch, *, mask_dtype: torch.dtype
    ) -> Optional[TapReading]:
        self._reading = TapReading.of(
            forward_batch,
            request_table=self.request_table,
            stripes=self.stripes,
            page_size=self.page_size,
            mask_dtype=mask_dtype,
        )
        return self._reading

    def get(self) -> Optional[TapReading]:
        return self._reading

    def scratch(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A reader's buffers for its rank's results and their log-sum-exp before
        the ranks merge them. The stage's readers run one after another on one
        stream, so they share one pair per shape; a CUDA graph capture takes its
        own.
        """
        if queries.is_cuda and torch.cuda.is_current_stream_capturing():
            return _scratch_like(queries)
        key = (queries.shape, queries.dtype, queries.device)
        if key != self._scratch_key:
            self._scratch_key = key
            self._scratch = _scratch_like(queries)
        return self._scratch


def _scratch_like(queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(queries), torch.empty(
        queries.shape[:2], dtype=torch.float32, device=queries.device
    )
