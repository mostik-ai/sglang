from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import torch
from torch.nn import functional

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
    """One encoder's taps for the tokens of one forward, as FlashAttention's paged
    KV cache with one head: the taps are both keys and values.

    The tokens are in the sequences of ``cu_seqlens_q``. A causal pass lets each
    sequence attend to the first ``causal_lens`` taps of its page table row, with
    its last token at the last of them. With ``full_lens``, a second pass lets
    every token of a sequence attend to all of its first ``full_lens`` taps: a
    request's tokens past its last tap form a sequence of their own for it.
    """

    taps: torch.Tensor  # [pages, page_size, 1, output_size]
    page_table: torch.Tensor  # [sequences, pages], int32
    cu_seqlens_q: torch.Tensor  # [sequences + 1], int32
    max_seqlen_q: int
    causal_lens: torch.Tensor  # [sequences], int32
    full_lens: Optional[torch.Tensor]  # [sequences], int32; None in decode
    num_tokens: int
    # The TRT-LLM kernels' workspace on Blackwell; None where FA3 runs.
    trtllm_workspace: Optional[torch.Tensor]


class FrankyTapCache:
    """This rank's taps: one tensor per encoder, in pages like KV.

    Each KV page is paired with one tap page, once, at start. So a pair is taken
    and freed together, one memory budget covers both, and neither needs an
    allocator of its own. KV never reads taps, and other code goes through
    ``slots_of`` and ``pages_of``.
    """

    def __init__(
        self,
        *,
        encoder_ids: tuple[int, ...],
        output_size: int,
        page_size: int,
        num_tokens: int,
        req_to_token: torch.Tensor,
        device: str,
        attention_workspace: Optional[torch.Tensor] = None,
    ) -> None:
        self.page_size = page_size
        self.page_bytes = page_size * output_size * TAP_DTYPE.itemsize
        self._output_size = output_size
        self._req_to_token = req_to_token
        self._attention_workspace = attention_workspace
        # Taps per request row; 0 for rows without taps, such as row 0, which
        # padded requests use.
        self._tap_lens = torch.zeros(
            req_to_token.shape[0], dtype=torch.int32, device=device
        )
        # One more page, as in the KV pool: padded tokens write to page 0, and
        # padded requests read its zeros.
        self._buffers = {
            encoder_id: torch.zeros(
                (num_tokens + page_size, output_size), dtype=TAP_DTYPE, device=device
            )
            for encoder_id in encoder_ids
        }

    @property
    def encoder_ids(self) -> tuple[int, ...]:
        return tuple(self._buffers)

    def transfer_buffers(self) -> dict[int, tuple[int, int]]:
        """Encoder id to the address and bytes of its tap buffer; page ``p`` is at
        ``address + p * page_bytes``.
        """
        return {
            encoder_id: (buffer.data_ptr(), buffer.nbytes)
            for encoder_id, buffer in self._buffers.items()
        }

    def pages_of(self, *, kv_pages: torch.Tensor) -> torch.Tensor:
        return kv_pages

    def slots_of(self, *, kv_slots: torch.Tensor) -> torch.Tensor:
        return kv_slots

    def write(
        self, *, encoder_id: int, taps: torch.Tensor, kv_slots: torch.Tensor
    ) -> None:
        self._buffers[encoder_id][self.slots_of(kv_slots=kv_slots)] = taps

    def set_tap_lens(self, *, req_pool_indices: list[int], tap_lens: list[int]) -> None:
        device = self._tap_lens.device
        rows = async_h2d(req_pool_indices, dtype=torch.int64, device=device)
        self._tap_lens[rows] = async_h2d(tap_lens, dtype=torch.int32, device=device)

    def view(self, *, encoder_id: int, forward_batch: ForwardBatch) -> TapView:
        """Token ``p`` of a request with ``tap_len`` taps reads taps
        ``0 .. min(p, tap_len - 1)``.
        """
        req_pool_indices = forward_batch.req_pool_indices
        if forward_batch.forward_mode.is_decode_or_idle():
            return self._one_token_view(
                encoder_id=encoder_id,
                req_pool_indices=req_pool_indices,
                seq_lens=forward_batch.seq_lens,
            )
        taps = self._buffers[encoder_id].view(-1, self.page_size, 1, self._output_size)
        page_table = self._page_table(req_pool_indices)
        tap_lens = self._tap_lens[req_pool_indices]
        first_positions = forward_batch.extend_prefix_lens
        extend_lens = forward_batch.extend_seq_lens
        within = (tap_lens - first_positions).clamp(min=0).minimum(extend_lens)
        past = extend_lens - within
        zero = torch.zeros_like(within)
        sequence_lens = torch.stack([within, past], dim=1).flatten()
        return TapView(
            taps=taps,
            page_table=page_table.repeat_interleave(2, dim=0),
            cu_seqlens_q=functional.pad(sequence_lens.cumsum(0), (1, 0)).to(
                torch.int32
            ),
            max_seqlen_q=max(forward_batch.extend_seq_lens_cpu, default=0),
            causal_lens=torch.stack([first_positions + within, zero], dim=1)
            .flatten()
            .to(torch.int32),
            full_lens=torch.stack([zero, torch.where(past > 0, tap_lens, zero)], dim=1)
            .flatten()
            .to(torch.int32),
            num_tokens=sum(forward_batch.extend_seq_lens_cpu),
            trtllm_workspace=self._attention_workspace,
        )

    def view_past_the_taps(
        self, *, encoder_id: int, req_pool_indices: torch.Tensor
    ) -> TapView:
        """One token per request, at or past its last tap: it reads all taps."""
        return self._one_token_view(
            encoder_id=encoder_id, req_pool_indices=req_pool_indices, seq_lens=None
        )

    def _one_token_view(
        self,
        *,
        encoder_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: Optional[torch.Tensor],
    ) -> TapView:
        tap_lens = self._tap_lens[req_pool_indices]
        num_requests = req_pool_indices.shape[0]
        num_keys = tap_lens if seq_lens is None else torch.minimum(seq_lens, tap_lens)
        return TapView(
            taps=self._buffers[encoder_id].view(
                -1, self.page_size, 1, self._output_size
            ),
            page_table=self._page_table(req_pool_indices),
            cu_seqlens_q=torch.arange(
                num_requests + 1, dtype=torch.int32, device=tap_lens.device
            ),
            max_seqlen_q=1,
            causal_lens=num_keys.to(torch.int32),
            full_lens=None,
            num_tokens=num_requests,
            trtllm_workspace=self._attention_workspace,
        )

    def _page_table(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        return (
            self._req_to_token[req_pool_indices, :: self.page_size] // self.page_size
        ).to(torch.int32)
