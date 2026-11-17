from __future__ import annotations

from sglang.srt.franky.parts.encoder.base_config import BaseEncoderConfig
from sglang.srt.franky.parts.encoder.normed.encoder import SingularNormedEncoder


class SingularNormedEncoderConfig(
    BaseEncoderConfig, frozen=True, kw_only=True, tag="singular_normed"
):
    encoder_cls = SingularNormedEncoder
