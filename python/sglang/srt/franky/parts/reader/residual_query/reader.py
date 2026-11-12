from __future__ import annotations

import torch
from torch import nn
from typing_extensions import override

from sglang.srt.franky.parts.reader.query.reader import QueryReader


class ResidualQueryReader(QueryReader):
    """The queries are a projection of Ricky's residual after the reader's layer."""

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
        self.query_projection = nn.Linear(
            ricky_hidden_size, num_queries * input_size, bias=False
        )

    @override
    def _queries(self, *, residual: torch.Tensor) -> torch.Tensor:
        return self.query_projection(residual).view(
            residual.shape[0], self.num_queries, self.input_size
        )
