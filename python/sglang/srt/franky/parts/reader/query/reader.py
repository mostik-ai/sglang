from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from typing_extensions import override

from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
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
    from sglang.srt.franky.cache.tap_cache import FrankyTapCache, TapView
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class QueryReader(Reader):
    """Queries attend, with one head, over the taps, which are the keys and the
    values, as the PR's ``QueryReader``; an MLP maps every query's result,
    joined, to Ricky's residual. Subclasses only say where the queries come from.
    """

    def __init__(
        self,
        *,
        num_queries: int,
        input_size: int,
        mlp_hidden_size: int,
        ricky_hidden_size: int,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.input_size = input_size
        self.mlp = nn.Sequential(
            nn.Linear(num_queries * input_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, ricky_hidden_size),
        )

    @abstractmethod
    def _queries(self, *, residual: torch.Tensor) -> torch.Tensor:
        """Return ``[tokens, num_queries, input_size]``."""
        raise NotImplementedError

    @override
    def forward(
        self,
        *,
        tap_cache: FrankyTapCache,
        encoder_id: int,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        num_tokens = residual.shape[0]
        attended = residual.new_empty(
            (num_tokens, self.num_queries, self.input_size), dtype=TAP_DTYPE
        )
        has_taps = residual.new_empty((num_tokens, 1))
        _attend_taps(
            self._queries(residual=residual).to(TAP_DTYPE).contiguous(),
            attended,
            has_taps,
            tap_cache=tap_cache,
            encoder_id=encoder_id,
            forward_batch=forward_batch,
        )
        return self._map(attended, has_taps)

    def _map(self, attended: torch.Tensor, has_taps: torch.Tensor) -> torch.Tensor:
        joined = attended.flatten(start_dim=1).to(has_taps.dtype)
        return self.mlp(joined) * has_taps


@eager_on_graph
def _attend_taps(
    queries: torch.Tensor,
    attended: torch.Tensor,
    has_taps: torch.Tensor,
    *,
    tap_cache: FrankyTapCache,
    encoder_id: int,
    forward_batch: ForwardBatch,
) -> None:
    # A prefill CUDA graph replays this break with the batch of the replay.
    if is_in_prefill_graph_scope():
        forward_batch = get_forward_batch()
    attend_taps(
        queries,
        attended,
        has_taps,
        view=tap_cache.view(encoder_id=encoder_id, forward_batch=forward_batch),
    )


def attend_taps(
    queries: torch.Tensor,
    attended: torch.Tensor,
    has_taps: torch.Tensor,
    *,
    view: TapView,
) -> None:
    """Write each row's attended taps into ``attended``, and 1 or 0 into
    ``has_taps`` by whether it read any.
    """
    padding = slice(view.num_tokens, None)
    attended[padding].zero_()
    has_taps[padding].zero_()
    if view.num_tokens == 0:
        return
    rows = slice(0, view.num_tokens)
    reads = view.taps_to_read > 0
    _attend(queries[rows], view, out=attended[rows])
    attended[rows] = torch.where(reads[:, None, None], attended[rows], 0)
    has_taps[rows, 0] = reads.to(has_taps.dtype)


def attention_workspace(*, device: str) -> Optional[torch.Tensor]:
    """
    Returns:
        A zeroed workspace for the TRT-LLM kernels the readers run on Blackwell,
        or None where they run FA3.
    """
    if torch.device(device).type != "cuda":
        return None
    if torch.cuda.get_device_capability(device)[0] < 10:
        return None
    # Arbitrary size; the kernels need it zeroed once.
    return torch.zeros(256 << 20, dtype=torch.uint8, device=device)


def _attend(queries: torch.Tensor, view: TapView, *, out: torch.Tensor) -> None:
    if view.trtllm_workspace is None:
        _fa3_attend(queries, view, out=out)
    else:
        _trtllm_attend(queries, view, out=out)


def _fa3_attend(queries: torch.Tensor, view: TapView, *, out: torch.Tensor) -> None:
    flash_attn_with_kvcache(
        q=queries.unsqueeze(1),
        k_cache=view.taps,
        v_cache=view.taps,
        page_table=view.page_table,
        cache_seqlens=view.taps_to_read,
        softmax_scale=queries.shape[-1] ** -0.5,
        causal=False,
        out=out.unsqueeze(1),
    )


def _trtllm_attend(queries: torch.Tensor, view: TapView, *, out: torch.Tensor) -> None:
    # FA4 refuses paged KV with 256-wide heads; flashinfer exists only on GPU hosts.
    from flashinfer import decode

    # With one head, [pages, page_size, 1, size] is also the HND layout.
    num_pages, page_size, _, tap_size = view.taps.shape
    taps = view.taps.view(num_pages, 1, page_size, tap_size)
    result = decode.trtllm_batch_decode_with_kv_cache(
        query=queries,
        kv_cache=(taps, taps),
        workspace_buffer=view.trtllm_workspace,
        block_tables=view.page_table,
        seq_lens=view.taps_to_read,
        max_seq_len=view.page_table.shape[1] * page_size,
        bmm1_scale=tap_size**-0.5,
        bmm2_scale=1.0,
        out=out,
    )
    if result is not out:
        out.copy_(result)
