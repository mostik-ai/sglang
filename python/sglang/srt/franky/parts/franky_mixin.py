from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional

import torch

from sglang.srt.franky.parts.injector.residual.output import StreamOutput

if TYPE_CHECKING:
    from sglang.srt.franky.parts.franky_parts import FrankyParts
    from sglang.srt.franky.parts.injector.layer_output import LayerOutput
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )


class FrankyMixin:
    """Makes a language model able to join Franky. Every call does nothing until
    ``franky_parts`` is set. ``README.md`` in this folder shows where the calls go.

    Attributes:
        franky_output_cls: the ``LayerOutput`` form the model's layer loop hands
            to Franky; each side's injector must take it.
    """

    franky_parts: Optional[FrankyParts] = None
    franky_output_cls: ClassVar[type[LayerOutput]]

    def franky_after_layer(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """For models whose residual is one stream that ``residual_batch`` keeps in
        ``forward_batch``.
        """
        if not self._franky_runs_after(layer_idx):
            return hidden_states
        output = self.franky_parts.after_layer(
            layer_idx=layer_idx,
            output=StreamOutput(
                hidden_states=hidden_states, forward_batch=forward_batch
            ),
        )
        return output.hidden_states

    def franky_read_pp(self, *, pp_proxy_tensors: PPProxyTensors) -> None:
        if self.franky_parts is not None:
            self.franky_parts.read_pp(pp_proxy_tensors=pp_proxy_tensors)

    def franky_write_pp(self, *, pp_proxy_tensors: PPProxyTensors) -> None:
        if self.franky_parts is not None:
            self.franky_parts.write_pp(pp_proxy_tensors=pp_proxy_tensors)

    def _franky_runs_after(self, layer_idx: int) -> bool:
        return self.franky_parts is not None and self.franky_parts.runs_after(
            layer_idx=layer_idx
        )
