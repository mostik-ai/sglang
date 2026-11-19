from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.srt.franky.parts.reader.tap_reading import TapReading
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    is_in_prefill_graph_scope,
)

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.franky.cache.tap_read_rows import TapReadRows
    from sglang.srt.franky.cache.tap_stripes import TapStripes
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Split-KV completion tickets, one int32 a query head of a row, for 2^18 rows of
# 8 heads: a forward's rows stay far below.
_TRTLLM_COUNTER_BYTES = (1 << 18) * 8 * 4
# TRT-LLM's kernels keep the log-sum-exp in base 2, as FlashInfer's MLA ones do.
_TRTLLM_LSE_IS_BASE_E = False


class TrtllmWorkspace(msgspec.Struct, frozen=True, kw_only=True):
    """What the readers' TRT-LLM kernels keep across launches on Blackwell.

    Attributes:
        buffer: scratch for split-KV partials, zeroed once.
        counters: the split-KV completion tickets, zeroed once; every launch
            resets them. Without them FlashInfer allocates fresh tickets per
            launch, whose memory the allocator, or a CUDA graph's pool, hands
            to a later kernel while the launch may still run.
    """

    buffer: torch.Tensor
    counters: torch.Tensor

    @staticmethod
    def for_device(device: str) -> Optional[TrtllmWorkspace]:
        """
        Returns:
            The zeroed workspace where the readers run TRT-LLM's kernels, which
            is Blackwell, or None where they run FA3.
        """
        if torch.device(device).type != "cuda":
            return None
        if torch.cuda.get_device_capability(device)[0] < 10:
            return None
        return TrtllmWorkspace(
            buffer=torch.zeros(256 << 20, dtype=torch.uint8, device=device),
            counters=torch.zeros(
                _TRTLLM_COUNTER_BYTES, dtype=torch.uint8, device=device
            ),
        )


class TapReadings:
    """The ``TapReading`` of the forward that runs on this rank, which the
    forward's first reader makes for the readers after it.
    """

    def __init__(
        self, *, read_rows: TapReadRows, stripes: TapStripes, page_size: int
    ) -> None:
        self.read_rows = read_rows
        self.stripes = stripes
        self.page_size = page_size
        self._current: Optional[TapReading] = None

    def start(
        self, forward_batch: ForwardBatch, *, mask_dtype: torch.dtype
    ) -> TapReading:
        self._current = TapReading.of(
            forward_batch,
            read_rows=self.read_rows,
            stripes=self.stripes,
            page_size=self.page_size,
            mask_dtype=mask_dtype,
        )
        return self._current

    def current(self) -> TapReading:
        return self._current


class TapAttention:
    """One reader's attention over the taps of its encoder on this rank. Each
    rank attends over the taps it keeps; with more than one rank, the ranks
    gather each row's result and log-sum-exp and merge them.
    """

    def __init__(
        self,
        *,
        pages: torch.Tensor,
        readings: TapReadings,
        starts_forward: bool,
        group: Optional[GroupCoordinator],
        workspace: Optional[TrtllmWorkspace],
    ) -> None:
        """
        Args:
            pages: [pages, page_size, 1, output_size], this rank's tap pages of
                the encoder.
            starts_forward: whether this reader runs first in each forward and
                makes the forward's reading.
            group: the attention-TP group whose ranks split the taps; None
                when this rank keeps them all.
        """
        self.pages = pages
        self.readings = readings
        self.starts_forward = starts_forward
        self.group = group
        self.workspace = workspace

    def attend(
        self,
        queries: torch.Tensor,
        attended: torch.Tensor,
        has_taps: torch.Tensor,
        *,
        forward_batch: ForwardBatch,
        rows: Optional[torch.Tensor],
    ) -> None:
        """Write each row's attended taps into ``attended``, and 1 or 0 into
        ``has_taps`` by whether it read any.

        Args:
            queries: [rows, queries, output_size].
            rows: the rows of ``forward_batch`` the queries belong to, as
                indices into them; None for all of them.
        """
        _attend_taps(
            queries,
            attended,
            has_taps,
            attention=self,
            forward_batch=forward_batch,
            rows=rows,
        )

    def run(
        self,
        queries: torch.Tensor,
        attended: torch.Tensor,
        has_taps: torch.Tensor,
        *,
        forward_batch: ForwardBatch,
        rows: Optional[torch.Tensor],
    ) -> None:
        reading = (
            self.readings.start(forward_batch, mask_dtype=has_taps.dtype)
            if self.starts_forward
            else self.readings.current()
        )
        if rows is not None:
            reading = reading.select(rows)
        num_rows = reading.num_rows
        has_taps[:num_rows].copy_(reading.has_taps[:num_rows])
        has_taps[num_rows:].zero_()
        attended[num_rows:].zero_()
        if num_rows == 0:
            return
        real = attended[:num_rows]
        real.copy_(self._attend_rows(queries[:num_rows], reading))
        real.masked_fill_(~reading.reads[:num_rows, None, None], 0)

    def _attend_rows(self, queries: torch.Tensor, reading: TapReading) -> torch.Tensor:
        merges = self.group is not None
        out = torch.empty_like(queries)
        lse = (
            torch.empty(queries.shape[:2], dtype=torch.float32, device=queries.device)
            if merges
            else None
        )
        lse_is_base_e = self._attend_locally(queries, reading, out=out, lse=lse)
        if not merges:
            return out
        kept_none = (reading.local_taps[: queries.shape[0]] <= 0)[:, None]
        out.masked_fill_(kept_none[:, :, None], 0)
        lse.masked_fill_(kept_none, float("-inf"))
        if reading.merged_rows is None:
            return _merge_ranks(out, lse, group=self.group, lse_is_base_e=lse_is_base_e)
        merged_rows = reading.merged_rows
        out[merged_rows] = _merge_ranks(
            out[merged_rows],
            lse[merged_rows],
            group=self.group,
            lse_is_base_e=lse_is_base_e,
        )
        return out

    def _attend_locally(
        self,
        queries: torch.Tensor,
        reading: TapReading,
        *,
        out: torch.Tensor,
        lse: Optional[torch.Tensor],
    ) -> bool:
        if reading.row_blocks is not None and queries.shape[1] == 1 and queries.is_cuda:
            _prefill_attend(queries, self.pages, reading, out=out, lse=lse)
            return True
        if self.workspace is None:
            _fa3_attend(queries, self.pages, reading, out=out, lse=lse)
            return True
        _trtllm_attend(queries, self.pages, reading, self.workspace, out=out, lse=lse)
        return _TRTLLM_LSE_IS_BASE_E


@eager_on_graph
def _attend_taps(
    queries: torch.Tensor,
    attended: torch.Tensor,
    has_taps: torch.Tensor,
    *,
    attention: TapAttention,
    forward_batch: ForwardBatch,
    rows: Optional[torch.Tensor],
) -> None:
    # A prefill CUDA graph replays this break with the batch of the replay.
    if is_in_prefill_graph_scope():
        forward_batch = get_forward_batch()
    attention.run(queries, attended, has_taps, forward_batch=forward_batch, rows=rows)


def _merge_ranks(
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    group: GroupCoordinator,
    lse_is_base_e: bool,
) -> torch.Tensor:
    if out.shape[0] == 0:
        return out
    num_rows, num_heads, size = out.shape
    lse_columns = torch.float32.itemsize // out.dtype.itemsize
    packed = out.new_empty((num_rows, num_heads, size + lse_columns))
    packed[..., :size] = out
    packed.view(torch.float32)[..., size // lse_columns] = lse
    gathered = packed.new_empty(
        (group.world_size * num_rows, num_heads, size + lse_columns)
    )
    group.all_gather_into_tensor(gathered, packed)
    gathered = gathered.view(group.world_size, num_rows, num_heads, size + lse_columns)
    merged, _ = dcp_lse_combine_triton(
        gathered[..., :size],
        gathered.view(torch.float32)[..., size // lse_columns],
        is_lse_base_on_e=lse_is_base_e,
    )
    return merged


def _prefill_attend(
    queries: torch.Tensor,
    pages: torch.Tensor,
    reading: TapReading,
    *,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
) -> None:
    from sglang.kernels.ops.attention.franky_reader_prefill import (
        franky_reader_prefill,
    )

    num_pages, page_size, _, tap_size = pages.shape
    franky_reader_prefill(
        queries.view(-1, tap_size),
        pages.view(num_pages, page_size, tap_size),
        reading.request_slots,
        reading.local_taps,
        reading.row_blocks,
        out=out.view(-1, tap_size),
        lse=None if lse is None else lse.view(-1),
    )


def _fa3_attend(
    queries: torch.Tensor,
    pages: torch.Tensor,
    reading: TapReading,
    *,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
) -> None:
    result = flash_attn_with_kvcache(
        q=queries.unsqueeze(1),
        k_cache=pages,
        v_cache=pages,
        page_table=reading.slot_table,
        cache_seqlens=reading.local_taps[: queries.shape[0]],
        softmax_scale=queries.shape[-1] ** -0.5,
        causal=False,
        out=out.unsqueeze(1),
        return_softmax_lse=lse is not None,
    )
    if lse is not None:
        lse.copy_(result[1].reshape(lse.shape))


def _trtllm_attend(
    queries: torch.Tensor,
    pages: torch.Tensor,
    reading: TapReading,
    workspace: TrtllmWorkspace,
    *,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
) -> None:
    # FA4 refuses paged KV with 256-wide heads; flashinfer exists only on GPU hosts.
    from flashinfer import decode

    # With one head, [pages, page_size, 1, size] is also the HND layout.
    num_pages, page_size, _, tap_size = pages.shape
    taps = pages.view(num_pages, 1, page_size, tap_size)
    slot_table = reading.slot_table
    result = decode.trtllm_batch_decode_with_kv_cache(
        query=queries,
        kv_cache=(taps, taps),
        workspace_buffer=workspace.buffer,
        multi_ctas_kv_counter_buffer=workspace.counters,
        block_tables=slot_table,
        seq_lens=reading.local_taps[: queries.shape[0]],
        max_seq_len=slot_table.shape[1] * page_size,
        bmm1_scale=tap_size**-0.5,
        bmm2_scale=1.0,
        out=out,
        lse=lse,
    )
    if isinstance(result, tuple):
        result = result[0]
    if result is not out:
        out.copy_(result)
