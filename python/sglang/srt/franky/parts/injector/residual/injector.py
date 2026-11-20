from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from typing_extensions import override

from sglang.srt.franky.parts.injector.injector import Injector
from sglang.srt.franky.parts.injector.residual.output import StreamOutput
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

if TYPE_CHECKING:
    import torch


class ResidualInjector(Injector):
    """The residual is one stream: parts read its whole sum, and a reader's result
    is added to the layer's output, as the PR's injector does.
    """

    output_cls = StreamOutput

    @override
    def read(self, *, output: StreamOutput) -> torch.Tensor:
        return residual_batch.snapshot(output.hidden_states, output.forward_batch)

    @override
    def inject(self, *, output: StreamOutput, addition: torch.Tensor) -> StreamOutput:
        hidden_states = residual_batch.add_to_output(
            output.hidden_states, output.forward_batch, addition
        )
        return msgspec.structs.replace(output, hidden_states=hidden_states)
