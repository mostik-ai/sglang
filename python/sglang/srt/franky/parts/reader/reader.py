from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn

if TYPE_CHECKING:
    from sglang.srt.franky.cache.tap_cache import FrankyTapCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class Reader(nn.Module, ABC):
    """Runs on Ricky: turns one encoder's taps into what the injector adds.

    A token at position ``p`` may read taps ``0 .. min(p, tap_len - 1)``; a token
    of a request without taps must get exactly zero.

    To add an implementation: make a folder here with its reader and its
    config, then add the config to ``ReaderConfig`` in ``config.py``.
    """

    # True when every token past a request's last tap reads the same result.
    same_past_the_taps: ClassVar[bool] = False

    @abstractmethod
    def forward(
        self,
        *,
        tap_cache: FrankyTapCache,
        encoder_id: int,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Return ``[tokens, ricky_hidden_size]``. ``residual`` is the whole
        residual after the reader's layer, for readers that use it.
        """
        raise NotImplementedError

    def read_past_the_taps(
        self,
        *,
        tap_cache: FrankyTapCache,
        encoder_id: int,
        req_pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        """For a reader with ``same_past_the_taps``: what a token of each request
        at or past its last tap gets, ``[requests, ricky_hidden_size]``.
        """
        raise NotImplementedError
