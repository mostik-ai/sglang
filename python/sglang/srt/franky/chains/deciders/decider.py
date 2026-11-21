from __future__ import annotations

from abc import ABC, abstractmethod

import msgspec


class EndedBlock(msgspec.Struct, frozen=True, kw_only=True):
    """A block Ricky ended without ending the answer.

    Attributes:
        number: the block's number in the answer, from 1.
        ends_with_call_token: whether its last token is the call token.
    """

    number: int
    ends_with_call_token: bool


class SandyCallDecider(ABC):
    """Runs on Ricky at each block end that does not end the answer, and decides
    whether Sandy writes before the next block.

    To add an implementation: make a folder here with its decider and its
    config, then add the config to ``DeciderConfig`` in ``config.py``.
    """

    @abstractmethod
    def calls_sandy(self, block: EndedBlock) -> bool:
        raise NotImplementedError
