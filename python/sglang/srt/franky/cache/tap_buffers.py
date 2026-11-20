from __future__ import annotations

import torch

TAP_DTYPE = torch.bfloat16


class TapBuffers:
    """One rank's tap pages: a tensor of ``num_pages`` pages per encoder. The
    readers' attention takes a page as both its keys and its values, and the
    transfer moves whole pages.
    """

    def __init__(
        self,
        *,
        encoder_ids: tuple[int, ...],
        output_size: int,
        page_size: int,
        num_pages: int,
        device: str,
    ) -> None:
        self.page_size = page_size
        self.output_size = output_size
        self.num_pages = num_pages
        self.page_bytes = page_size * output_size * TAP_DTYPE.itemsize
        self._buffers = {
            encoder_id: torch.zeros(
                (num_pages, page_size, 1, output_size), dtype=TAP_DTYPE, device=device
            )
            for encoder_id in encoder_ids
        }

    def transfer_buffers(self) -> dict[int, tuple[int, int]]:
        """Encoder id to the address and bytes of its buffer; page ``p`` is at
        ``address + p * page_bytes``.
        """
        return {
            encoder_id: (buffer.data_ptr(), buffer.nbytes)
            for encoder_id, buffer in self._buffers.items()
        }

    def write(
        self, *, encoder_id: int, taps: torch.Tensor, slots: torch.Tensor
    ) -> None:
        """
        Args:
            taps: [rows, output_size].
            slots: per row, ``page * page_size + offset``.
        """
        self._buffers[encoder_id].view(-1, self.output_size)[slots] = taps

    def pages(self, encoder_id: int) -> torch.Tensor:
        """[num_pages, page_size, 1, output_size]."""
        return self._buffers[encoder_id]
