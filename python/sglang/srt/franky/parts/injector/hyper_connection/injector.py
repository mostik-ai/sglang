from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from typing_extensions import override

from sglang.kernels.ops.layernorm.mhc import mhc_stream_mean
from sglang.srt.franky.parts.injector.hyper_connection.output import (
    HyperConnectionOutput,
)
from sglang.srt.franky.parts.injector.injector import Injector

if TYPE_CHECKING:
    import torch


class HyperConnectionInjector(Injector):
    """Manifold hyper-connections: parts read the mean of the residual streams,
    as DSpark's capture does, and a reader's result is added to every stream, so
    any mix of the streams gains it in proportion to its weights.
    """

    output_cls = HyperConnectionOutput

    @override
    def read(self, *, output: HyperConnectionOutput) -> torch.Tensor:
        return mhc_stream_mean(output.state.residual)

    @override
    def inject(
        self, *, output: HyperConnectionOutput, addition: torch.Tensor
    ) -> HyperConnectionOutput:
        residual = output.state.residual
        streams = residual + addition.to(residual.dtype)[:, None, :]
        return msgspec.structs.replace(
            output, state=output.state.with_residual(streams)
        )
