from __future__ import annotations

from typing import Protocol

import numpy as np


class FrankyTapSender(Protocol):
    """One request's taps leaving one Sandy rank. Pages land in prompt order."""

    def land(self, *, tap_pages: np.ndarray) -> None:
        """
        Args:
            tap_pages: int32 tap pages of the prompt's next whole pages, or of
                all its remaining pages after its last chunk.
        """

    def poll(self) -> int:
        """
        Returns:
            The room's ``KVPoll`` state on this rank.
        """

    def abort(self) -> None: ...

    def close(self) -> None: ...
