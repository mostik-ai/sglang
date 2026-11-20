from __future__ import annotations

from typing_extensions import override

from sglang.srt.managers.schedule_batch import BaseFinishReason


class FrankySandyCall(BaseFinishReason):
    """A Ricky request of a chain ended at a block end, and Sandy writes before
    the chain's next block ``next_block``.
    """

    def __init__(self, *, next_block: int) -> None:
        super().__init__()
        self.next_block = next_block

    @override
    def to_json(self) -> dict:
        return {"type": "franky_sandy_call", "next_block": self.next_block}
