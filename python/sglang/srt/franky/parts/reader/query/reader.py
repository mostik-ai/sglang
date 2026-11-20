from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from typing_extensions import override

from sglang.srt.franky.cache.tap_buffers import TAP_DTYPE
from sglang.srt.franky.parts.reader.reader import Reader

if TYPE_CHECKING:
    from sglang.srt.franky.parts.reader.tap_attention import TapAttention
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
        taps: TapAttention,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        rows: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_tokens = residual.shape[0]
        attended = residual.new_empty(
            (num_tokens, self.num_queries, self.input_size), dtype=TAP_DTYPE
        )
        has_taps = residual.new_empty((num_tokens, 1))
        taps.attend(
            self._queries(residual=residual).to(TAP_DTYPE).contiguous(),
            attended,
            has_taps,
            forward_batch=forward_batch,
            rows=rows,
        )
        return self._map(attended, has_taps)

    def _map(self, attended: torch.Tensor, has_taps: torch.Tensor) -> torch.Tensor:
        joined = attended.flatten(start_dim=1).to(has_taps.dtype)
        return self.mlp(joined) * has_taps
