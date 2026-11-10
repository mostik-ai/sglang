from __future__ import annotations

from typing import Union

from sglang.srt.franky.parts.encoder.all.config import AllEncoderConfig
from sglang.srt.franky.parts.encoder.masked.config import MaskedEncoderConfig
from sglang.srt.franky.parts.encoder.singular.config import SingularEncoderConfig

EncoderConfig = Union[AllEncoderConfig, MaskedEncoderConfig, SingularEncoderConfig]
