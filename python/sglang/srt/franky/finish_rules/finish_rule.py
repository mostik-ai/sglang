from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class FinishRule(ABC):
    """Franky's rule for where a request's output ends, beside SGLang's own.
    A request carries at most one, as ``Req.franky_finish_rule``.

    To add a rule: make a module here with its class, and set it on the
    requests it serves where they arrive.
    """

    @abstractmethod
    def after_checks(self, req: Req, new_tokens: Sequence[int]) -> None:
        """SGLang has checked whether ``req`` finished after writing
        ``new_tokens``, whatever it decided. End the request, or change how it
        ended, through ``finished_reason`` and ``finished_len``.
        """
        raise NotImplementedError
