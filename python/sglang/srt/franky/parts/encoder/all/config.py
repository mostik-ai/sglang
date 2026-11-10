from __future__ import annotations

from sglang.srt.franky.parts.encoder.all.encoder import AllEncoder
from sglang.srt.franky.parts.encoder.base_config import BaseEncoderConfig


class AllEncoderConfig(BaseEncoderConfig, frozen=True, kw_only=True, tag="all"):
    encoder_cls = AllEncoder
