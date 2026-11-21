from __future__ import annotations

from typing import Annotated, Optional

import msgspec

from sglang.srt.franky.chains.deciders.call_token_or_every_blocks.config import (
    CallTokenOrEveryBlocksConfig,
)
from sglang.srt.franky.chains.deciders.config import DeciderConfig
from sglang.srt.franky.parts.part_config import PositiveInt


class FrankyBlocksConfig(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """How Ricky splits a chain's answer into blocks, and who decides at each
    block end whether Sandy writes before the next one. The router's chain
    config sets it for every Ricky leg.

    Attributes:
        block_size: the most tokens of one block.
        call_token_id: a token of Ricky's vocabulary that also ends a block;
            None for none.
        decider: what decides at each block end; by default Sandy writes after
            every block.
    """

    block_size: PositiveInt
    call_token_id: Optional[Annotated[int, msgspec.Meta(ge=0)]] = None
    decider: DeciderConfig = CallTokenOrEveryBlocksConfig()
