from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.utils.common import async_h2d, async_h2d_stack


class FrankyOwnTapPages:
    """Tap pages that Ricky's requests own when Sandy tokenizes on its own. A tap
    then belongs to one of Sandy's tokens, which has no KV slot on Ricky, so a
    request takes pages for its taps when it arrives and gives them back when
    it ends; readers find them in its row of ``page_table``. Nothing shares them:
    the prefix tree keys Ricky's tokens.
    """

    def __init__(
        self,
        *,
        first_page: int,
        num_pages: int,
        max_pages_per_request: int,
        num_request_rows: int,
        device: str,
    ) -> None:
        """
        Args:
            first_page: pages before it hold the zeros that padded rows read.
        """
        self._free_pages = torch.arange(first_page, num_pages, dtype=torch.int32)
        self.num_ownable_pages = num_pages - first_page
        self.page_table = torch.zeros(
            (num_request_rows, max_pages_per_request), dtype=torch.int32, device=device
        )

    @property
    def max_pages_per_request(self) -> int:
        return self.page_table.shape[1]

    def num_free(self) -> int:
        return len(self._free_pages)

    def take(self, *, num_pages: int) -> Optional[torch.Tensor]:
        """
        Returns:
            int32 pages on the host, or None when fewer are free.
        """
        if num_pages > len(self._free_pages):
            return None
        pages = self._free_pages[:num_pages]
        self._free_pages = self._free_pages[num_pages:]
        return pages

    def free(self, pages: torch.Tensor) -> None:
        self._free_pages = torch.cat([self._free_pages, pages])

    def bind(self, *, req_pool_indices: list[int], pages: list[torch.Tensor]) -> None:
        """The reads of the requests in these rows go to their pages from now on."""
        width = max(len(request_pages) for request_pages in pages)
        padded = [
            torch.nn.functional.pad(request_pages, (0, width - len(request_pages)))
            for request_pages in pages
        ]
        device = self.page_table.device
        rows = async_h2d(req_pool_indices, dtype=torch.int64, device=device)
        self.page_table[rows, :width] = async_h2d_stack(padded, device=device)
