from __future__ import annotations

from typing import ClassVar

from typing_extensions import override

from sglang.srt.franky.parts.part_config import PositiveInt
from sglang.srt.franky.parts.reader.base_config import BaseReaderConfig
from sglang.srt.franky.parts.reader.query.reader import QueryReader
from sglang.srt.franky.parts.reader.reader import Reader


class QueryReaderConfig(BaseReaderConfig, frozen=True, kw_only=True):
    reader_cls: ClassVar[type[QueryReader]]

    num_queries: PositiveInt
    mlp_hidden_size: PositiveInt

    @override
    def build(self, *, input_size: int, ricky_hidden_size: int) -> Reader:
        return self.reader_cls(
            num_queries=self.num_queries,
            input_size=input_size,
            mlp_hidden_size=self.mlp_hidden_size,
            ricky_hidden_size=ricky_hidden_size,
        )
