from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from typing_extensions import override

from sglang.srt.franky.cache.tap_cache import TAP_DTYPE
from sglang.srt.franky.parts.reader.reader import Reader
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    is_in_prefill_graph_scope,
)

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
        if forward_batch.forward_mode.is_decode_or_idle():
            return residual.new_zeros(num_rows, self.projection.out_features)
        taps = residual.new_empty(
            (num_rows, self.projection.in_features), dtype=TAP_DTYPE
        )
        reads = residual.new_empty((num_rows, 1))
        _aligned_taps(
            taps,
            reads,
            tap_cache=tap_cache,
            encoder_id=encoder_id,
            forward_batch=forward_batch,
        )
        weight = self.projection.weight
        added = self.projection(taps.to(weight.dtype)) * self.output_scale
        return added.to(residual.dtype) * reads.to(residual.dtype)


@eager_on_graph
def _aligned_taps(
    taps: torch.Tensor,
    reads: torch.Tensor,
    *,
    tap_cache: FrankyTapCache,
    encoder_id: int,
    forward_batch: ForwardBatch,
) -> None:
    """Write each row's own tap into ``taps``, and 1 or 0 into ``reads`` by
    whether it reads it."""
    # A prefill CUDA graph replays this break with the batch of the replay.
    if is_in_prefill_graph_scope():
        forward_batch = get_forward_batch()
    aligned = tap_cache.aligned_taps(
        encoder_id=encoder_id, forward_batch=forward_batch, num_rows=taps.shape[0]
    )
    if aligned is None:
        taps.zero_()
        reads.zero_()
        return
    taps.copy_(aligned.taps)
    reads[:, 0] = aligned.reads.to(reads.dtype)
