from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from sglang.srt.franky.chains.deciders.decider import SandyCallDecider


class BaseDeciderConfig(
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    forbid_unknown_fields=True,
    tag_field="type",
):
    def build(self) -> SandyCallDecider:
        raise NotImplementedError
