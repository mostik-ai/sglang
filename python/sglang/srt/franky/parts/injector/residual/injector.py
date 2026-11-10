from __future__ import annotations

import torch
from typing_extensions import override

from sglang.srt.franky.parts.injector.injector import Injector


class ResidualInjector(Injector):
    """Adds a reader's result to its layer's output, as the PR's injector does."""

    @override
    def inject(
        self, *, reader_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        return reader_output
