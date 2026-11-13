from __future__ import annotations

from typing_extensions import override

from sglang.srt.franky.parts.injector.base_config import BaseInjectorConfig
from sglang.srt.franky.parts.injector.hyper_connection.injector import (
    HyperConnectionInjector,
)
from sglang.srt.franky.parts.injector.injector import Injector


class HyperConnectionInjectorConfig(
    BaseInjectorConfig, frozen=True, kw_only=True, tag="hyper_connection"
):
    @override
    def build(self) -> Injector:
        return HyperConnectionInjector()
