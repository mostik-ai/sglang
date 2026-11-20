from __future__ import annotations

from typing import TypeVar

import msgspec
import numpy as np
import torch

from sglang.srt.runtime_context import get_parallel

Pages = TypeVar("Pages", np.ndarray, torch.Tensor)


class TapStripes(msgspec.Struct, frozen=True, kw_only=True):
    """How the attention-TP ranks of one engine split tap pages. A stripe is
    ``num_ranks`` pages in a row; rank ``r`` keeps page ``r`` of every stripe.

    Attributes:
        num_ranks: the attention-TP ranks.
        rank: this rank among them.
    """

    num_ranks: int
    rank: int

    @staticmethod
    def of_attention_tp() -> TapStripes:
        """How this engine's attention-TP ranks split the taps."""
        parallel = get_parallel()
        return TapStripes(num_ranks=parallel.attn_tp_size, rank=parallel.attn_tp_rank)

    def owns(self, pages: Pages) -> Pages:
        """Whether this rank keeps each page."""
        return pages % self.num_ranks == self.rank

    def stripe(self, pages: Pages) -> Pages:
        """The stripe of each page."""
        return pages // self.num_ranks

    def local_taps(self, taps: torch.Tensor, *, page_size: int) -> torch.Tensor:
        """
        Args:
            taps: per row, how many taps it reads, from the first one on.
        Returns:
            Per row, how many of them this rank keeps: its page of each whole
            stripe, then its part of the last stripe.
        """
        stripe_taps = self.num_ranks * page_size
        last_stripe = (taps % stripe_taps - self.rank * page_size).clamp(
            min=0, max=page_size
        )
        return taps // stripe_taps * page_size + last_stripe
