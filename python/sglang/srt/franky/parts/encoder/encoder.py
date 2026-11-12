from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class Encoder(nn.Module, ABC):
    """Runs on Sandy: merges the hiddens of its input layers into one tap per token.

    To add an implementation: make a folder here with its encoder and its
    config, then add the config to ``EncoderConfig`` in ``config.py``.
    """

    def __init__(
        self,
        *,
        input_layers: tuple[int, ...],
        output_size: int,
        sandy_hidden_size: int,
    ) -> None:
        super().__init__()
        self.input_layers = input_layers
        self.output_size = output_size
        self.sandy_hidden_size = sandy_hidden_size

    @staticmethod
    @abstractmethod
    def select_inputs(*, layers: tuple[int, ...], index: int) -> tuple[int, ...]:
        """Which of the encoder layers encoder ``index`` reads, in input order."""
        raise NotImplementedError

    @abstractmethod
    def accumulate(
        self, *, state: Optional[torch.Tensor], layer: int, hiddens: torch.Tensor
    ) -> torch.Tensor:
        """Fold one input layer into ``state``. The state crosses PP stages, so
        it must be one tensor, the same on every TP rank.
        """
        raise NotImplementedError

    @abstractmethod
    def finish(self, *, state: torch.Tensor) -> torch.Tensor:
        """Turn ``state`` into taps, ``[tokens, output_size]``."""
        raise NotImplementedError
