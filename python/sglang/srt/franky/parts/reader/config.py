from __future__ import annotations

from typing import Union

from sglang.srt.franky.parts.reader.fixed_query.config import FixedQueryReaderConfig
from sglang.srt.franky.parts.reader.residual_query.config import (
    ResidualQueryReaderConfig,
)

ReaderConfig = Union[FixedQueryReaderConfig, ResidualQueryReaderConfig]
