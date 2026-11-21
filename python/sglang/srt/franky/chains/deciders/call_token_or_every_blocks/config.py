from __future__ import annotations

from typing_extensions import override

from sglang.srt.franky.chains.deciders.base_config import BaseDeciderConfig
from sglang.srt.franky.chains.deciders.call_token_or_every_blocks.decider import (
    CallTokenOrEveryBlocks,
)
from sglang.srt.franky.chains.deciders.decider import SandyCallDecider
from sglang.srt.franky.parts.part_config import PositiveInt


class CallTokenOrEveryBlocksConfig(
    BaseDeciderConfig, frozen=True, kw_only=True, tag="call_token_or_every_blocks"
):
    every_blocks: PositiveInt = 1

    @override
    def build(self) -> SandyCallDecider:
        return CallTokenOrEveryBlocks(every_blocks=self.every_blocks)
