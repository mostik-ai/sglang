from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import torch

    from sglang.srt.franky.parts.injector.layer_output import LayerOutput


class Injector(ABC):
    """Where Franky's parts meet a model's residual right after a layer: what
    encoders and readers read from it, and how a reader's result goes into what
    the next layer reads. Each side's model keeps its residual in one form, and
    the config names an injector for that form per side.

    To add an implementation: make a folder here with its injector, its
    ``LayerOutput`` and its config, then add the config to ``InjectorConfig`` in
    ``config.py``.
    """

    output_cls: ClassVar[type[LayerOutput]]

    @abstractmethod
    def read(self, *, output: LayerOutput) -> torch.Tensor:
        """
        Returns:
            ``[rows, hidden_size]``, one hidden per row of ``output``; ``output``
            is left as it is.
        """
        raise NotImplementedError

    def check_stage(
        self, *, layer_indices: frozenset[int], end_layer: int, num_layers: int
    ) -> None:
        """Refuses parts that this PP stage cannot run.

        Args:
            layer_indices: the stage's layers with a part, counted from 0.
            end_layer: the stage runs the layers before this one.
            num_layers: the model's layers.
        Raises:
            ValueError: a part needs what only another stage holds.
        """

    @abstractmethod
    def inject(self, *, output: LayerOutput, addition: torch.Tensor) -> LayerOutput:
        """
        Args:
            addition: ``[rows, hidden_size]``, a reader's result for the rows of
                ``output``.
        Returns:
            The residual the next layer reads, with ``addition`` in it.
        """
        raise NotImplementedError
