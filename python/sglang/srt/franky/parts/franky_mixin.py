from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.franky.parts.franky_parts import FrankyParts
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )


class FrankyMixin:
    """Makes a language model able to join Franky. Every call does nothing until
    ``franky_parts`` is set. ``README.md`` in this folder shows where the calls go.
    """

    franky_parts: Optional[FrankyParts] = None

    def franky_after_layer(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """For models whose layers return ``hidden_states``."""
        if self.franky_parts is None:
            return hidden_states
        return self.franky_parts.after_layer(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

    def franky_after_layer_with_residual(
        self,
        *,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """For models whose layers return ``(hidden_states, residual)``."""
        if self.franky_parts is None:
            return hidden_states, residual
        return self.franky_parts.after_layer_with_residual(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

    def franky_read_pp(self, *, pp_proxy_tensors: PPProxyTensors) -> None:
        if self.franky_parts is not None:
            self.franky_parts.read_pp(pp_proxy_tensors=pp_proxy_tensors)

    def franky_write_pp(self, *, pp_proxy_tensors: PPProxyTensors) -> None:
        if self.franky_parts is not None:
            self.franky_parts.write_pp(pp_proxy_tensors=pp_proxy_tensors)
