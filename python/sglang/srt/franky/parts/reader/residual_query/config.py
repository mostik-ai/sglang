from __future__ import annotations

from sglang.srt.franky.parts.reader.query.config import QueryReaderConfig
from sglang.srt.franky.parts.reader.residual_query.reader import ResidualQueryReader


class ResidualQueryReaderConfig(
    QueryReaderConfig, frozen=True, kw_only=True, tag="residual_query"
):
    reader_cls = ResidualQueryReader
