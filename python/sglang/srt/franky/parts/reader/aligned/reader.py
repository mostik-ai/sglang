from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from typing_extensions import override

from sglang.srt.franky.parts.reader.reader import Reader

if TYPE_CHECKING:
    from sglang.srt.franky.cache.tap_cache import FrankyTapCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class AlignedReader(Reader):
    """Adds, at each row whose input token is outside its request's answers, a
    linear map of the tap Sandy wrote at that same token, times a fixed scale: the
    two sides' views align token for token. Answer rows, so every decode row, add
    zero.
    """

    def __init__(
        self, *, input_size: int, ricky_hidden_size: int, output_scale: float
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, ricky_hidden_size)
        self.output_scale = output_scale

    @override
    def forward(
        self,
        *,
        tap_cache: FrankyTapCache,
        encoder_id: int,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        num_rows = residual.shape[0]
        aligned = tap_cache.aligned_taps(
            encoder_id=encoder_id, forward_batch=forward_batch, num_rows=num_rows
        )
        if aligned is None:
            return residual.new_zeros(num_rows, self.projection.out_features)
        weight = self.projection.weight
        added = self.projection(aligned.taps.to(weight.dtype)) * self.output_scale
        added = added.to(residual.dtype)
        return torch.where(aligned.reads[:, None], added, torch.zeros_like(added))
