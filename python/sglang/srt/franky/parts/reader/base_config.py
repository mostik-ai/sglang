from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.franky.parts.part_config import LayersPartConfig

if TYPE_CHECKING:
    from sglang.srt.franky.parts.reader.reader import Reader


class BaseReaderConfig(LayersPartConfig, frozen=True, kw_only=True):
    """``layers`` lists where the readers sit on Ricky; reader ``i`` reads encoder ``i``."""

    def build(self, *, input_size: int, ricky_hidden_size: int) -> Reader:
        raise NotImplementedError
