from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

if TYPE_CHECKING:
    from sglang.srt.franky.parts.reader.tap_attention import TapAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class Reader(nn.Module, ABC):
    """Runs on Ricky: turns one encoder's taps into what the injector adds.

    A row reads the taps that its request's ``FrankyTapRuns`` give the
    token it makes; a row that reads none must get exactly zero.

    To add an implementation: make a folder here with its reader and its
    config, then add the config to ``ReaderConfig`` in ``config.py``.
    """

    @abstractmethod
    def forward(
        self,
        *,
        taps: TapAttention,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        rows: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return ``[tokens, ricky_hidden_size]``. ``taps`` attends over the
        reader's encoder's taps; ``residual`` is the whole residual after the
        reader's layer, for readers that use it; ``rows`` are the rows of
        ``forward_batch`` it holds, as indices, or None for all.
        """
        raise NotImplementedError
