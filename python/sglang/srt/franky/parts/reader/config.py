from __future__ import annotations

from typing import Union

from sglang.srt.franky.parts.reader.aligned.config import AlignedReaderConfig
from sglang.srt.franky.parts.reader.residual_query.config import (
    ResidualQueryReaderConfig,
)

ReaderConfig = Union[ResidualQueryReaderConfig, AlignedReaderConfig]
