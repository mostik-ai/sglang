from __future__ import annotations

from typing import ClassVar

from sglang.srt.franky.parts.encoder.encoder import Encoder
from sglang.srt.franky.parts.part_config import LayersPartConfig, PositiveInt


class BaseEncoderConfig(LayersPartConfig, frozen=True, kw_only=True):
    """``layers`` lists where the encoders sit on Sandy, one encoder per entry."""

    encoder_cls: ClassVar[type[Encoder]]

    output_size: PositiveInt

    def input_layers(self, *, index: int) -> tuple[int, ...]:
        return self.encoder_cls.select_inputs(layers=self.layers, index=index)

    def build(self, *, index: int, sandy_hidden_size: int) -> Encoder:
        return self.encoder_cls(
            input_layers=self.input_layers(index=index),
            output_size=self.output_size,
            sandy_hidden_size=sandy_hidden_size,
        )
