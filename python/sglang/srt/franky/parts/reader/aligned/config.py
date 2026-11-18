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
    residual becomes the scaled map instead of gaining it: a fitted map from
    Sandy's states to Ricky's, which sets his state rather than adding to it."""

    output_scale: Annotated[float, msgspec.Meta(gt=0)]
    replace: bool = False

    @override
    def build(self, *, input_size: int, ricky_hidden_size: int) -> Reader:
        return AlignedReader(
            input_size=input_size,
            ricky_hidden_size=ricky_hidden_size,
            output_scale=self.output_scale,
            replace=self.replace,
        )
