from __future__ import annotations

from typing_extensions import override

from sglang.srt.franky.chains.deciders.decider import EndedBlock, SandyCallDecider


class CallTokenOrEveryBlocks(SandyCallDecider):
    """Calls Sandy after a block the call token ended, and after blocks
    ``every_blocks``, ``2 * every_blocks``, ... of the answer.
    """

    def __init__(self, *, every_blocks: int) -> None:
        self.every_blocks = every_blocks

    @override
    def calls_sandy(self, block: EndedBlock) -> bool:
        return block.ends_with_call_token or block.number % self.every_blocks == 0
