from __future__ import annotations

from sglang.srt.franky.parts.encoder.base_config import BaseEncoderConfig
from sglang.srt.franky.parts.encoder.singular.encoder import SingularEncoder


class SingularEncoderConfig(
    BaseEncoderConfig, frozen=True, kw_only=True, tag="singular"
):
    encoder_cls = SingularEncoder
