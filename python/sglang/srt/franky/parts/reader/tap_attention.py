from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.franky_reader_merge import (
    franky_reader_merge,
    franky_reader_pack,
)
from sglang.kernels.ops.attention.franky_reader_prefill import franky_reader_prefill
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    is_in_prefill_graph_scope,
)

if TYPE_CHECKING:
    from sglang.srt.franky.parts.reader.current_tap_reading import CurrentTapReading
    from sglang.srt.franky.parts.reader.tap_gather import TapGather
    from sglang.srt.franky.parts.reader.tap_reading import TapReading
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Split-KV completion tickets, one int32 a query head of a row, for 2^18 rows of
# 8 heads: a forward's rows stay far below.
_TRTLLM_COUNTER_BYTES = (1 << 18) * 8 * 4


class LocalKernel(enum.Enum):
    """A kernel a rank attends over the taps it keeps with."""

    PREFILL = "franky_reader_prefill"
    FA3 = "flash_attn_with_kvcache"
    TRTLLM = "trtllm_batch_decode_with_kv_cache"

    @property
    def lse_is_base_e(self) -> bool:
        """Whether its log-sum-exp is natural; TRT-LLM's kernels keep it in base 2,
        as FlashInfer's MLA ones do.
        """
        return self is not LocalKernel.TRTLLM


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


class TapAttention:
    """One reader's attention over the taps of its encoder on this rank. Each
    rank attends over the taps it keeps; with more than one rank, the ranks
    gather each row's result and log-sum-exp and merge them.
    """

    def __init__(
        self,
        *,
        encoder_id: int,
        pages: torch.Tensor,
        reading: CurrentTapReading,
        builds_reading: bool,
        gather: Optional[TapGather],
        workspace: Optional[TrtllmWorkspace],
    ) -> None:
        """
        Args:
            encoder_id: the encoder whose taps the reader reads.
            pages: [pages, page_size, 1, output_size], this rank's tap pages of
                the encoder.
            reading: the stage's reading of the forward that runs.
            builds_reading: whether this reader runs first in each forward and
                makes the forward's reading for the stage.
            gather: how the attention-TP ranks that split the taps gather their
                results; None when this rank keeps them all.
        """
        self.encoder_id = encoder_id
        self.pages = pages
        self.reading = reading
        self.builds_reading = builds_reading
        self.gather = gather
        self.workspace = workspace

    @eager_on_graph
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
        ``has_taps`` by whether it read any. A graph break: a prefill CUDA graph
        replays it with the batch of the replay.

        Args:
            queries: [rows, queries, output_size].
            rows: the rows of ``forward_batch`` the queries belong to, as
                indices into them; None for all of them.
        """
        if is_in_prefill_graph_scope():
            forward_batch = get_forward_batch()
        reading = (
            self.reading.start(forward_batch, mask_dtype=has_taps.dtype)
            if self.builds_reading
            else self.reading.get()
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
        self._attend_rows(queries[:num_rows], reading, out=real)
        real.masked_fill_(reading.no_taps[:num_rows, None, None], 0)

    def _attend_rows(
        self, queries: torch.Tensor, reading: TapReading, *, out: torch.Tensor
    ) -> None:
        kernel = self._local_kernel(queries, reading)
        if self.gather is None:
            self._attend_locally(kernel, queries, reading, out=out, lse=None)
            return
        local_out = torch.empty_like(queries)
        lse = torch.empty(queries.shape[:2], dtype=torch.float32, device=queries.device)
        self._attend_locally(kernel, queries, reading, out=local_out, lse=lse)
        local_taps = reading.local_taps[: queries.shape[0]]
        rows = reading.merged_rows
        if rows is None:
            self._merge(
                local_out, lse, local_taps, out=out, lse_is_base_e=kernel.lse_is_base_e
            )
            return
        merged = local_out.new_empty((rows.shape[0], *local_out.shape[1:]))
        self._merge(
            local_out[rows],
            lse[rows],
            local_taps[rows],
            out=merged,
            lse_is_base_e=kernel.lse_is_base_e,
        )
        out[rows] = merged

    def _merge(
        self,
        local_out: torch.Tensor,
        lse: torch.Tensor,
        local_taps: torch.Tensor,
        *,
        out: torch.Tensor,
        lse_is_base_e: bool,
    ) -> None:
        if local_out.shape[0] == 0:
            return
        packed = franky_reader_pack(local_out, lse, local_taps)
        franky_reader_merge(
            self.gather.gather(packed), out=out, lse_is_base_e=lse_is_base_e
        )

    def _local_kernel(self, queries: torch.Tensor, reading: TapReading) -> LocalKernel:
        if reading.row_blocks is not None and queries.shape[1] == 1 and queries.is_cuda:
            return LocalKernel.PREFILL
        if self.workspace is None:
            return LocalKernel.FA3
        return LocalKernel.TRTLLM

    def _attend_locally(
        self,
        kernel: LocalKernel,
        queries: torch.Tensor,
        reading: TapReading,
        *,
        out: torch.Tensor,
        lse: Optional[torch.Tensor],
    ) -> None:
        if kernel is LocalKernel.PREFILL:
            _prefill_attend(queries, self.pages, reading, out=out, lse=lse)
        elif kernel is LocalKernel.FA3:
            _fa3_attend(queries, self.pages, reading, out=out, lse=lse)
        else:
            _trtllm_attend(
                queries, self.pages, reading, self.workspace, out=out, lse=lse
            )


def _prefill_attend(
    queries: torch.Tensor,
    pages: torch.Tensor,
    reading: TapReading,
    *,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
) -> None:
    num_pages, page_size, _, tap_size = pages.shape
    franky_reader_prefill(
        queries.view(-1, tap_size),
        pages.view(num_pages, page_size, tap_size),
        reading.request_pages,
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
        page_table=reading.page_table,
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
    page_table = reading.page_table
    result = decode.trtllm_batch_decode_with_kv_cache(
        query=queries,
        kv_cache=(taps, taps),
        workspace_buffer=workspace.buffer,
        multi_ctas_kv_counter_buffer=workspace.counters,
        block_tables=page_table,
        seq_lens=reading.local_taps[: queries.shape[0]],
        max_seq_len=page_table.shape[1] * page_size,
        bmm1_scale=tap_size**-0.5,
        bmm2_scale=1.0,
        out=out,
        lse=lse,
    )
    if isinstance(result, tuple):
        result = result[0]
    if result is not out:
        out.copy_(result)
