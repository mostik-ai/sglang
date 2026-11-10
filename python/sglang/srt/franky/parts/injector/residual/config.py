from __future__ import annotations

from typing_extensions import override

from sglang.srt.franky.parts.injector.base_config import BaseInjectorConfig
from sglang.srt.franky.parts.injector.injector import Injector
from sglang.srt.franky.parts.injector.residual.injector import ResidualInjector


class ResidualInjectorConfig(
    BaseInjectorConfig, frozen=True, kw_only=True, tag="residual"
):
    @override
    def build(self) -> Injector:
        return ResidualInjector()
