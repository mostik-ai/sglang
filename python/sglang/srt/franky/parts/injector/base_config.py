from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from sglang.srt.franky.parts.injector.injector import Injector


class BaseInjectorConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
    tag_field="type",
):
    def build(self) -> Injector:
        raise NotImplementedError
