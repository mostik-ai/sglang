from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

from sglang.srt.franky.parts.injector.layer_output import LayerOutput

if TYPE_CHECKING:
    import torch
    from torch import nn

    from sglang.srt.layers.attn_residual import AttnResidual


class AttentionBankRead(msgspec.Struct, frozen=True, kw_only=True):
    """How the layer after a Franky layer mixes the attention-residual bank: its
    score weights and the bank rows it reads.
    """

    bank: AttnResidual
    score_proj: nn.Module
    score_norm: nn.Module
    num_valid_blocks: int


class AttentionBankOutput(LayerOutput, frozen=True, kw_only=True):
    """A layer's output in a model with attention residuals (Kimi K3): a layer
    reads a learned mix of the bank's block sums and the running sum since the
    last block, which ``residual_batch`` keeps in ``forward_batch``.
    """

    hidden_states: torch.Tensor
    next_read: AttentionBankRead
