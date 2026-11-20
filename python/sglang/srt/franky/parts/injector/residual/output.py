from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.franky.parts.injector.layer_output import LayerOutput

if TYPE_CHECKING:
    import torch


class StreamOutput(LayerOutput, frozen=True, kw_only=True):
    """A layer's output in a model whose residual is one stream, kept in
    ``forward_batch`` by ``residual_batch`` until the next layer adds it.
    """

    hidden_states: torch.Tensor
