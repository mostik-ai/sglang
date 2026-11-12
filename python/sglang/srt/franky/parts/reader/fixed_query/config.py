from __future__ import annotations

from sglang.srt.franky.parts.reader.fixed_query.reader import FixedQueryReader
from sglang.srt.franky.parts.reader.query.config import QueryReaderConfig


class FixedQueryReaderConfig(
    QueryReaderConfig, frozen=True, kw_only=True, tag="fixed_query"
):
    reader_cls = FixedQueryReader
