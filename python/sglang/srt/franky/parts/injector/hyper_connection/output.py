from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.franky.parts.injector.layer_output import LayerOutput

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v4_mhc import HcState


class HyperConnectionOutput(LayerOutput, frozen=True, kw_only=True):
    """A layer's output in a model with manifold hyper-connections (DeepSeek
    V4): the residual is several streams, ``state.residual`` of shape
    ``[rows, streams, hidden_size]``, that each sublayer mixes. The state is
    complete: no post of the layer is still outstanding.
    """

    state: HcState
