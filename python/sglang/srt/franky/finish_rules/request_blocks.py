from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

from typing_extensions import override

from sglang.srt.franky.chains.blocks.sandy_call import FrankySandyCall
from sglang.srt.franky.chains.deciders.decider import EndedBlock
from sglang.srt.franky.finish_rules.finish_rule import FinishRule
from sglang.srt.franky.request import FrankyRequest

if TYPE_CHECKING:
    from sglang.srt.franky.chains.blocks.config import FrankyBlocksConfig
    from sglang.srt.franky.chains.deciders.decider import SandyCallDecider
    from sglang.srt.managers.schedule_batch import Req


class RequestBlocks(FinishRule):
    """One Ricky leg's blocks, ``block`` being the chain's number of the block
    the leg writes now. It counts the tokens the leg writes, ends a block at the
    block size or at the call token, and ends the leg when the decider calls
    Sandy at a block end. The call token stays in the output; EOS, stop rules
    and the limit end the answer first.
    """

    def __init__(
        self,
        *,
        config: FrankyBlocksConfig,
        decider: SandyCallDecider,
        block: int,
    ) -> None:
        self._config = config
        self._decider = decider
        self._block = block
        self._num_tokens = 0

    @staticmethod
    def attach(req: Req) -> None:
        """On an engine that writes a chain's blocks: set the rule on a leg whose
        chain has blocks. A request back from a retraction keeps its count.
        """
        chain = FrankyRequest.of(req).chain
        if chain is None or chain.blocks is None or req.franky_finish_rule is not None:
            return
        req.franky_finish_rule = RequestBlocks(
            config=chain.blocks,
            decider=chain.blocks.decider.build(),
            block=chain.block,
        )

    @override
    def after_checks(self, req: Req, new_tokens: Sequence[int]) -> None:
        if req.finished():
            return
        for offset, token in enumerate(new_tokens):
            block = self._end_of_block(token)
            if block is not None and self._decider.calls_sandy(block):
                self._end_for_sandy(req, num_unread=len(new_tokens) - offset - 1)
                return

    def _end_of_block(self, token: int) -> Optional[EndedBlock]:
        self._num_tokens += 1
        ends_with_call_token = token == self._config.call_token_id
        if not ends_with_call_token and self._num_tokens < self._config.block_size:
            return None
        block = EndedBlock(
            number=self._block, ends_with_call_token=ends_with_call_token
        )
        self._block += 1
        self._num_tokens = 0
        return block

    def _end_for_sandy(self, req: Req, *, num_unread: int) -> None:
        req.finished_reason = FrankySandyCall(next_block=self._block)
        req.finished_len = len(req.output_ids) - num_unread
