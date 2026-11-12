from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from typing_extensions import override

from sglang.srt.franky.cache.tap_cache import TAP_DTYPE
from sglang.srt.franky.parts.reader.query.reader import QueryReader, attend_taps

if TYPE_CHECKING:
    from sglang.srt.franky.cache.tap_cache import FrankyTapCache


class FixedQueryReader(QueryReader):
    """The queries are learned and the same for every token, so every token at
    or past a request's last tap reads the same.
    """

    same_past_the_taps = True

    def __init__(
        self,
        *,
        num_queries: int,
        input_size: int,
        mlp_hidden_size: int,
        ricky_hidden_size: int,
    ) -> None:
        super().__init__(
            num_queries=num_queries,
            input_size=input_size,
            mlp_hidden_size=mlp_hidden_size,
            ricky_hidden_size=ricky_hidden_size,
        )
        self.queries = nn.Parameter(torch.empty(num_queries, input_size))

    @override
    def _queries(self, *, residual: torch.Tensor) -> torch.Tensor:
        return self.queries.expand(residual.shape[0], -1, -1)

    @override
    def read_past_the_taps(
        self,
        *,
        tap_cache: FrankyTapCache,
        encoder_id: int,
        req_pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_requests = req_pool_indices.shape[0]
        queries = self.queries.to(TAP_DTYPE).expand(num_requests, -1, -1).contiguous()
        attended = torch.empty_like(queries)
        has_taps = self.queries.new_empty((num_requests, 1))
        attend_taps(
            queries,
            attended,
            has_taps,
            view=tap_cache.view_past_the_taps(
                encoder_id=encoder_id, req_pool_indices=req_pool_indices
            ),
        )
        return self._map(attended, has_taps)
