from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional
from typing_extensions import override

from sglang.srt.franky.parts.encoder.encoder import Encoder


class LinearEncoder(Encoder):
    """One Linear without bias over the concatenated input layers, as the PR's
    ``LinearEncoder``. It sums each layer's product with its block of the weight,
    in input order, so the state is a running sum.
    """

    def __init__(
        self,
        *,
        input_layers: tuple[int, ...],
        output_size: int,
        sandy_hidden_size: int,
    ) -> None:
        super().__init__(
            input_layers=input_layers,
            output_size=output_size,
            sandy_hidden_size=sandy_hidden_size,
        )
        self.projection = nn.Linear(
            len(input_layers) * sandy_hidden_size, output_size, bias=False
        )

    @override
    def accumulate(
        self, *, state: Optional[torch.Tensor], layer: int, hiddens: torch.Tensor
    ) -> torch.Tensor:
        weight_block = self.projection.weight.chunk(len(self.input_layers), dim=1)[
            self.input_layers.index(layer)
        ]
        product = functional.linear(hiddens, weight_block)
        return product if state is None else state + product

    @override
    def finish(self, *, state: torch.Tensor) -> torch.Tensor:
        return state
