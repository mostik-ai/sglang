from __future__ import annotations

from typing import Annotated

import msgspec
from typing_extensions import override

from sglang.srt.franky.parts.reader.aligned.reader import AlignedReader
from sglang.srt.franky.parts.reader.base_config import BaseReaderConfig
from sglang.srt.franky.parts.reader.reader import Reader


class AlignedReaderConfig(
    BaseReaderConfig, frozen=True, kw_only=True, tag="aligned"
):
    """``output_scale`` multiplies what each reader adds, so that the map starts on
    the scale of Ricky's residual at its layer. With ``replace``, a writing row's
    residual moves ``replace_weight`` of the way to the scaled map instead of gaining
    it (1: becomes the map): a fitted map from Sandy's states to Ricky's, which sets
    his state rather than adding to it."""

    output_scale: Annotated[float, msgspec.Meta(gt=0)]
    replace: bool = False
    replace_weight: Annotated[float, msgspec.Meta(gt=0, le=1)] = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.replace_weight != 1.0 and not self.replace:
            raise ValueError("replace_weight takes effect only with replace")

    @override
    def build(self, *, input_size: int, ricky_hidden_size: int) -> Reader:
        return AlignedReader(
            input_size=input_size,
            ricky_hidden_size=ricky_hidden_size,
            output_scale=self.output_scale,
            replace=self.replace,
            replace_weight=self.replace_weight,
        )
