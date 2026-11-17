from __future__ import annotations

from typing import Optional

import torch
from torch.nn import functional
from typing_extensions import override

from sglang.srt.franky.parts.encoder.singular.encoder import SingularEncoder

EPS = 1e-6


class SingularNormedEncoder(SingularEncoder):
    """Each encoder takes only its own layer, RMS-normalised over the hidden size in
    float32 before its Linear, so its taps keep one scale across tokens."""

    @override
    def accumulate(
        self, *, state: Optional[torch.Tensor], layer: int, hiddens: torch.Tensor
    ) -> torch.Tensor:
        normed = functional.rms_norm(hiddens.float(), (hiddens.shape[-1],), eps=EPS)
        return super().accumulate(
            state=state, layer=layer, hiddens=normed.to(hiddens.dtype)
        )
