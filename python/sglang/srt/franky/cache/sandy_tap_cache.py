from __future__ import annotations

import torch

from sglang.srt.franky.cache.tap_buffers import TapBuffers
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.utils.common import ceil_div


class SandyTapCache:
    """Sandy's taps on this rank. A token's tap lives with its KV slot, so taps
    share the lifetime and the memory budget of Sandy's KV: tap page ``t`` holds
    the taps of KV slots ``[t * page_size, (t + 1) * page_size)``, and
    ``stripes`` says which rank keeps it, at the local page of its stripe. The
    taps of tokens that other ranks keep go to the last local page, which nothing
    reads.
    """

    def __init__(self, *, buffers: TapBuffers, stripes: TapStripes) -> None:
        self.buffers = buffers
        self.stripes = stripes
        self._discarded_page = buffers.num_pages - 1

    @staticmethod
    def build(
        *,
        encoder_ids: tuple[int, ...],
        output_size: int,
        page_size: int,
        num_kv_slots: int,
        stripes: TapStripes,
        device: str,
    ) -> SandyTapCache:
        """
        Args:
            num_kv_slots: Sandy's KV slots, padding included.
        """
        num_tap_pages = ceil_div(num_kv_slots, page_size)
        return SandyTapCache(
            buffers=TapBuffers(
                encoder_ids=encoder_ids,
                output_size=output_size,
                page_size=page_size,
                num_pages=ceil_div(num_tap_pages, stripes.num_ranks) + 1,
                device=device,
            ),
            stripes=stripes,
        )

    def write(
        self, *, encoder_id: int, taps: torch.Tensor, kv_slots: torch.Tensor
    ) -> None:
        self.buffers.write(
            encoder_id=encoder_id, taps=taps, slots=self._local_slots(kv_slots)
        )

    def local_pages(self, *, kv_slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kv_slots: consecutive tokens' KV slots, from a tap page's start.
        Returns:
            int32, per tap page of these tokens in order, its local page on this
            rank, or -1 where another rank keeps it.
        """
        page_size = self.buffers.page_size
        pages = kv_slots[::page_size] // page_size
        local = torch.where(self.stripes.owns(pages), self.stripes.stripe(pages), -1)
        return local.to(torch.int32)

    def _local_slots(self, kv_slots: torch.Tensor) -> torch.Tensor:
        page_size = self.buffers.page_size
        pages = kv_slots // page_size
        offsets = kv_slots % page_size
        kept = self.stripes.stripe(pages) * page_size + offsets
        discarded = self._discarded_page * page_size + offsets
        return torch.where(self.stripes.owns(pages), kept, discarded)
