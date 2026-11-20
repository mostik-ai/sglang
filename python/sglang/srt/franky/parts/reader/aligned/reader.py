from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from typing_extensions import override

from sglang.srt.distributed.parallel_state import get_attn_tp_group
from sglang.srt.franky.cache.tap_buffers import TAP_DTYPE
from sglang.srt.franky.parts.reader.aligned.own_taps import OwnTaps
from sglang.srt.franky.parts.reader.reader import Reader
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    is_in_prefill_graph_scope,
)

if TYPE_CHECKING:
    from sglang.srt.franky.parts.reader.tap_attention import TapAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Test-only: a file each prefill appends its reading rows' positions and taps to, so
# a parity test can compare the engine's writes with the trainer's row for row.
DUMP_VARIABLE = "SGLANG_FRANKY_ALIGNED_DUMP"


class AlignedReader(Reader):
    """Adds, at each row whose input token is in its request's write ranges, a
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
        taps: TapAttention,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        rows: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_rows = residual.shape[0]
        if forward_batch.forward_mode.is_decode_or_idle():
            return residual.new_zeros(num_rows, self.projection.out_features)
        own = residual.new_empty(
            (num_rows, self.projection.in_features), dtype=TAP_DTYPE
        )
        reads = residual.new_empty((num_rows, 1))
        _own_taps(own, reads, taps=taps, forward_batch=forward_batch, rows=rows)
        weight = self.projection.weight
        added = self.projection(own.to(weight.dtype)) * self.output_scale
        return added.to(residual.dtype) * reads.to(residual.dtype)


@eager_on_graph
def _own_taps(
    own: torch.Tensor,
    reads: torch.Tensor,
    *,
    taps: TapAttention,
    forward_batch: ForwardBatch,
    rows: Optional[torch.Tensor],
) -> None:
    """Write each row's own tap into ``own``, and 1 or 0 into ``reads`` by
    whether it reads it."""
    # A prefill CUDA graph replays this break with the batch of the replay.
    if is_in_prefill_graph_scope():
        forward_batch = get_forward_batch()
    stripes = taps.reading.stripes
    num_forward_rows = (
        own.shape[0] if rows is None else sum(forward_batch.extend_seq_lens_cpu)
    )
    own_taps = OwnTaps.of(
        forward_batch,
        pages=taps.pages,
        request_table=taps.reading.request_table,
        stripes=stripes,
        num_rows=num_forward_rows,
    )
    summed = own_taps.taps
    if stripes.num_ranks > 1:
        summed = get_attn_tp_group().all_reduce(summed)
    dump = os.environ.get(DUMP_VARIABLE)
    if dump:
        _dump(dump, own_taps, summed, forward_batch, taps.encoder_id)
    row_reads = own_taps.reads
    if rows is not None:
        # An index past the forward's rows is padding, which reads nothing.
        selected = rows.clamp(max=num_forward_rows)
        summed = torch.cat([summed, summed.new_zeros(1, summed.shape[1])])[selected]
        row_reads = torch.cat([row_reads, row_reads.new_zeros(1)])[selected]
    own.copy_(summed)
    reads[:, 0] = row_reads.to(reads.dtype)


def _dump(
    path: str,
    own_taps: OwnTaps,
    summed: torch.Tensor,
    forward_batch: ForwardBatch,
    encoder_id: int,
) -> None:
    reading = own_taps.reads.tolist()
    rids = forward_batch.rids or []
    by_request: dict[int, list[int]] = {}
    for row, read in enumerate(reading):
        if read:
            by_request.setdefault(own_taps.requests[row], []).append(row)
    with open(path, "a") as file:  # test-only dump, never in a serving run
        for request, request_rows in by_request.items():
            record = {
                "rid": rids[request] if request < len(rids) else None,
                "encoder": encoder_id,
                "positions": [own_taps.positions[row] for row in request_rows],
                "taps": summed[request_rows].float().tolist(),
            }
            file.write(json.dumps(record) + "\n")
