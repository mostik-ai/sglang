from __future__ import annotations

from typing_extensions import override

from sglang.srt.franky.parts.injector.attention_bank.injector import (
    AttentionBankInjector,
)
from sglang.srt.franky.parts.injector.base_config import BaseInjectorConfig
from sglang.srt.franky.parts.injector.injector import Injector


class AttentionBankInjectorConfig(
    BaseInjectorConfig, frozen=True, kw_only=True, tag="attention_bank"
):
    @override
    def build(self) -> Injector:
        return AttentionBankInjector()
