from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import msgspec
import numpy as np
import torch

from sglang.srt.franky.cache.generated_spans import FrankyGeneratedSpans
from sglang.srt.utils.common import async_h2d

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

TAP_DTYPE = torch.bfloat16


class FrankyReqState(msgspec.Struct, kw_only=True):
    """One Ricky request's Franky state on this rank."""

    tap_len: int
    taps_ready: int
    # KV slots taken at arrival for the prompt from token ``taken_from`` on.
    # They stay out of ``req.kv`` until a chunk runs them, so KV code sees a plain
    # chunked request.
    taken_kv_slots: torch.Tensor
    taken_from: int
    used_end: int

    def hand_to_chunk(self, *, start: int, end: int) -> torch.Tensor:
        """The taken KV slots of prompt tokens ``[start, end)``, which a chunk now
        uses.
        """
        self.used_end = end
        return self.taken_kv_slots[start - self.taken_from : end - self.taken_from]


class TapView(msgspec.Struct, frozen=True, kw_only=True):
    """One encoder's taps for the rows of one forward, as the paged attention
    kernels take them.

    Attributes:
        taps: [pages, page_size, 1, output_size], both the keys and the values.
        page_table: [rows, pages] int32, the pages of each row's request.
        taps_to_read: [rows] int32, the taps each row's query reads, from tap 0.
        num_tokens: the real rows; a CUDA graph pads after them.
        trtllm_workspace: TRT-LLM's workspace on Blackwell; None where FA3 runs.
    """

    taps: torch.Tensor
    page_table: torch.Tensor
    taps_to_read: torch.Tensor
    num_tokens: int
    trtllm_workspace: Optional[torch.Tensor]


class FrankyTapCache:
    """This rank's taps: one tensor per encoder, a tap at each KV slot.

    A token's tap lives at its KV slot, so taps are taken and freed with KV, one
    memory budget covers both, and neither needs an allocator of its own. KV
    never reads taps, and other code goes through ``slots_of`` and ``pages_of``.

    Outside the KV allocator taps are seen in tap pages of ``page_size`` tokens:
    the transfer moves them and the readers attend over them. A tap page is a view
    of the slots of a KV page, so KV pages may be any multiple of it and two
    engines with different KV pages still exchange taps page to page, in place.
    """

    def __init__(
        self,
        *,
        encoder_ids: tuple[int, ...],
        output_size: int,
        page_size: int,
        kv_page_size: int,
        num_tokens: int,
        req_to_token: torch.Tensor,
        device: str,
        attention_workspace: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            page_size: tokens of a tap page; it divides ``kv_page_size``.
            num_tokens: KV slots past the KV pool's first page.
        """
        if kv_page_size % page_size != 0:
            raise ValueError(
                f"A KV page of {kv_page_size} tokens is not whole tap pages of "
                f"{page_size} tokens."
            )
        self.page_size = page_size
        self.page_bytes = page_size * output_size * TAP_DTYPE.itemsize
        self.pages_per_kv_page = kv_page_size // page_size
        self._output_size = output_size
        self._req_to_token = req_to_token
        self._attention_workspace = attention_workspace
        self._decode_taps_to_read = torch.zeros(
            req_to_token.shape[0], dtype=torch.int32, device=device
        )
        self._generated_spans: list[Optional[FrankyGeneratedSpans]] = [
            None
        ] * req_to_token.shape[0]
        # One more KV page, as in the KV pool: padded tokens write to page 0, and
        # padded requests read its zeros.
        self._buffers = {
            encoder_id: torch.zeros(
                (num_tokens + kv_page_size, output_size),
                dtype=TAP_DTYPE,
                device=device,
            )
            for encoder_id in encoder_ids
        }

    @property
    def encoder_ids(self) -> tuple[int, ...]:
        return tuple(self._buffers)

    def transfer_buffers(self) -> dict[int, tuple[int, int]]:
        """Encoder id to the address and bytes of its tap buffer; tap page ``p``
        is at ``address + p * page_bytes``.
        """
        return {
            encoder_id: (buffer.data_ptr(), buffer.nbytes)
            for encoder_id, buffer in self._buffers.items()
        }

    def pages_of(self, *, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: consecutive tokens' KV slots, from a tap page's start.
        Returns:
            The tap pages that hold them, in order.
        """
        return slots[:: self.page_size] // self.page_size

    def pages_of_kv_pages(self, *, kv_pages: np.ndarray) -> np.ndarray:
        """
        Returns:
            int32, the tap pages of ``kv_pages``, in order.
        """
        first_pages = np.asarray(kv_pages, dtype=np.int64) * self.pages_per_kv_page
        offsets = np.arange(self.pages_per_kv_page, dtype=np.int64)
        return (first_pages[:, None] + offsets).reshape(-1).astype(np.int32)

    def slots_of(self, *, kv_slots: torch.Tensor) -> torch.Tensor:
        return kv_slots

    def write(
        self, *, encoder_id: int, taps: torch.Tensor, kv_slots: torch.Tensor
    ) -> None:
        self._buffers[encoder_id][self.slots_of(kv_slots=kv_slots)] = taps

    def set_generated_spans(
        self,
        *,
        req_pool_indices: list[int],
        generated_spans: list[Optional[FrankyGeneratedSpans]],
        num_input_tokens: list[int],
    ) -> None:
        """Which taps the forward of each token of these requests reads; a request
        with None reads none.

        Args:
            num_input_tokens: per request, its input length: its decoded block's start.
        """
        device = self._decode_taps_to_read.device
        decoded = [
            0 if spans is None else int(spans.taps_to_read(torch.tensor([first]))[0])
            for spans, first in zip(generated_spans, num_input_tokens)
        ]
        rows = async_h2d(req_pool_indices, dtype=torch.int64, device=device)
        self._decode_taps_to_read[rows] = async_h2d(
            decoded, dtype=torch.int32, device=device
        )
        for req_pool_index, spans in zip(req_pool_indices, generated_spans):
            self._generated_spans[req_pool_index] = spans

    def view(self, *, encoder_id: int, forward_batch: ForwardBatch) -> TapView:
        """Each row's query reads the taps that its request's spans give the token
        the row makes."""
        if forward_batch.forward_mode.is_decode_or_idle():
            return self._decode_view(
                encoder_id=encoder_id, req_pool_indices=forward_batch.req_pool_indices
            )
        return self._extend_view(encoder_id=encoder_id, forward_batch=forward_batch)

    def _extend_view(self, *, encoder_id: int, forward_batch: ForwardBatch) -> TapView:
        req_pool_indices = _on_host(
            forward_batch.req_pool_indices_cpu, forward_batch.req_pool_indices
        )
        first_rows = _on_host(
            forward_batch.extend_prefix_lens_cpu, forward_batch.extend_prefix_lens
        )
        num_rows = list(forward_batch.extend_seq_lens_cpu)
        taps_to_read = [
            self._taps_to_read(req_pool_index, first_row=first_row, num_rows=count)
            for req_pool_index, first_row, count in zip(
                req_pool_indices, first_rows, num_rows
            )
        ]
        device = self._decode_taps_to_read.device
        row_requests = [
            request for request, count in enumerate(num_rows) for _ in range(count)
        ]
        page_table = self._page_table(forward_batch.req_pool_indices)
        return TapView(
            taps=self._paged_taps(encoder_id),
            page_table=page_table[
                async_h2d(row_requests, dtype=torch.int64, device=device)
            ],
            taps_to_read=async_h2d(
                [taps for request in taps_to_read for taps in request.tolist()],
                dtype=torch.int32,
                device=device,
            ),
            num_tokens=len(row_requests),
            trtllm_workspace=self._attention_workspace,
        )

    def _taps_to_read(
        self, req_pool_index: int, *, first_row: int, num_rows: int
    ) -> torch.Tensor:
        rows = torch.arange(first_row, first_row + num_rows)
        spans = self._generated_spans[req_pool_index]
        if spans is None:
            return torch.zeros_like(rows)
        made_tokens = rows + 1
        return spans.taps_to_read(made_tokens)

    def _decode_view(
        self, *, encoder_id: int, req_pool_indices: torch.Tensor
    ) -> TapView:
        return TapView(
            taps=self._paged_taps(encoder_id),
            page_table=self._page_table(req_pool_indices),
            taps_to_read=self._decode_taps_to_read[req_pool_indices],
            num_tokens=req_pool_indices.shape[0],
            trtllm_workspace=self._attention_workspace,
        )

    def _paged_taps(self, encoder_id: int) -> torch.Tensor:
        return self._buffers[encoder_id].view(-1, self.page_size, 1, self._output_size)

    def _page_table(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        return (
            self._req_to_token[req_pool_indices, :: self.page_size] // self.page_size
        ).to(torch.int32)


def _on_host(
    host: Optional[Sequence[int] | torch.Tensor], device: torch.Tensor
) -> list[int]:
    # The prefill graph's warmup and capture batches keep these on the device only.
    values = device if host is None else host
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)
