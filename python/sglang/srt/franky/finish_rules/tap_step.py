from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from typing_extensions import override

from sglang.srt.franky.finish_rules.finish_rule import FinishRule
from sglang.srt.managers.schedule_batch import FINISH_LENGTH

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams


class TapStep(FinishRule):
    """A Sandy write with a room writes one token past the ones it returns, so
    the forward of its last returned token runs and that token has a tap. A
    write that ends at its token limit returns one token less than it wrote; one
    that stops at a stop token returns it as SGLang does, since the stop token is
    the one whose forward never ran.
    """

    @staticmethod
    def take(sampling_params: SamplingParams) -> TapStep:
        """Ask for the extra token of a write."""
        sampling_params.max_new_tokens += 1
        return TapStep()

    @override
    def after_checks(self, req: Req, new_tokens: Sequence[int]) -> None:
        if isinstance(req.finished_reason, FINISH_LENGTH):
            num_returned = req.finished_len - 1
            req.finished_reason = FINISH_LENGTH(length=num_returned)
            req.finished_len = num_returned
