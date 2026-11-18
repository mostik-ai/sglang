from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class FrankyTakenSlots:
    """The requests whose KV slots, taken at arrival, the next extend runs on."""

    def __init__(self) -> None:
        self._reqs: Optional[list[Req]] = None

    def expect(self, *, reqs: list[Req]) -> None:
        self._reqs = reqs

    @property
    def expected(self) -> bool:
        return self._reqs is not None

    def hand_out(
        self, *, prefix_lens_cpu: torch.Tensor, seq_lens_cpu: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns:
            The expected requests' taken slots of tokens ``[prefix, seq)``, in
            request order; no request stays expected.
        """
        reqs, self._reqs = self._reqs, None
        return torch.cat(
            [
                req.franky.hand_to_chunk(start=start, end=end)
                for req, start, end in zip(
                    reqs, prefix_lens_cpu.tolist(), seq_lens_cpu.tolist(), strict=True
                )
            ]
        )
