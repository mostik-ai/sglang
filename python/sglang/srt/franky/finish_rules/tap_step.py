from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from typing_extensions import override

from sglang.srt.franky.finish_rules.finish_rule import FinishRule
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class TapStep(FinishRule):
    """A Sandy write with a room writes one token past the ones it asks for, so
    the forward of its last token runs and that token has a tap. The extra token
    is never returned: a write that ends on it, at its limit or at a stop, ends
    at the tokens it asked for. A write that stops before it returns its stop
    token as SGLang does, since the stop token's forward never ran.
    """

    def __init__(self, *, asked: int) -> None:
        self._asked = asked

    @staticmethod
    def ask_extra_token(req: Req) -> None:
        """Raise ``req``'s limit by the extra token, before SGLang clamps it. A
        write without a limit runs to the clamp, which ends it as well.
        """
        if req.sampling_params.max_new_tokens is not None:
            req.sampling_params.max_new_tokens += 1

    @staticmethod
    def attach(req: Req) -> None:
        """Set the rule on ``req``, whose clamped limit ends at the extra token."""
        asked = max(req.sampling_params.max_new_tokens - 1, 0)
        req.franky_finish_rule = TapStep(asked=asked)

    @override
    def after_checks(self, req: Req, new_tokens: Sequence[int]) -> None:
        if not req.finished() or isinstance(req.finished_reason, FINISH_ABORT):
            return
        if len(req.output_ids) > self._asked:
            req.finished_reason = FINISH_LENGTH(length=self._asked)
            req.finished_len = self._asked
