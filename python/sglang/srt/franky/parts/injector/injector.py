from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Injector(ABC):
    """Puts a reader's result into Ricky's model, after the reader's layer.

    To add an implementation: make a folder here with its injector and its
    config, then add the config to ``InjectorConfig`` in ``config.py``.
    """

    @abstractmethod
    def inject(
        self, *, reader_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Return what to add to the output of the reader's layer. ``residual``
        is the whole residual after that layer.
        """
        raise NotImplementedError
