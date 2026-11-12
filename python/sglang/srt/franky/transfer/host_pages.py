from __future__ import annotations

import numpy as np
import torch

from sglang.srt.utils.common import async_d2h


class FrankyHostPages:
    """Page ids on their way to the host; the copy may wait behind the forward in
    flight, so callers check ``ready`` instead of blocking."""

    def __init__(self, pages: torch.Tensor) -> None:
        self._host = async_d2h(pages)
        self._done = None
        if pages.device.type != "cpu":
            self._done = torch.get_device_module(pages.device).Event()
            self._done.record()

    def ready(self) -> bool:
        return self._done is None or self._done.query()

    def numpy(self) -> np.ndarray:
        """The pages, once ``ready``."""
        return self._host.numpy()
