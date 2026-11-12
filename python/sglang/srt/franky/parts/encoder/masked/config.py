from __future__ import annotations

from sglang.srt.franky.parts.encoder.base_config import BaseEncoderConfig
from sglang.srt.franky.parts.encoder.masked.encoder import MaskedEncoder


class MaskedEncoderConfig(BaseEncoderConfig, frozen=True, kw_only=True, tag="masked"):
    encoder_cls = MaskedEncoder
