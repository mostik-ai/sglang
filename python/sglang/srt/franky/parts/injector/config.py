from __future__ import annotations

from typing import Union

import msgspec

from sglang.srt.franky.parts.injector.attention_bank.config import (
    AttentionBankInjectorConfig,
)
from sglang.srt.franky.parts.injector.residual.config import ResidualInjectorConfig

InjectorConfig = Union[AttentionBankInjectorConfig, ResidualInjectorConfig]


class InjectorsConfig(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """One injector per side: Sandy's and Ricky's models may keep their residuals
    in different forms. Sandy's injector only reads.
    """

    sandy: InjectorConfig
    ricky: InjectorConfig
