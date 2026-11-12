from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.franky.cache.tap_reading import FrankyTapReading
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

    The tokens are in the sequences of ``cu_seqlens_q``. Every token of a sequence
    attends to the first ``lens[sequence]`` taps of its page table row, an int32
    count; a sequence with 0 reads nothing. In decode, each sequence is one token.
    """

    taps: torch.Tensor  # [pages, page_size, 1, output_size]
    page_table: torch.Tensor  # [sequences, pages], int32
    cu_seqlens_q: torch.Tensor  # [sequences + 1], int32
    max_seqlen_q: int
    lens: torch.Tensor
    decode: bool
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
        self._readings = [
            FrankyTapReading(tap_len=0, generation_start=0)
        ] * req_to_token.shape[0]
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

    def set_readings(
        self, *, req_pool_indices: list[int], readings: list[FrankyTapReading]
    ) -> None:
        device = self._tap_lens.device
        rows = async_h2d(req_pool_indices, dtype=torch.int64, device=device)
        self._tap_lens[rows] = async_h2d(
            [reading.tap_len for reading in readings], dtype=torch.int32, device=device
        )
        for row, reading in zip(req_pool_indices, readings):
            self._readings[row] = reading

    def view(self, *, encoder_id: int, forward_batch: ForwardBatch) -> TapView:
        """Each token reads the taps its request's ``FrankyTapReading`` gives it."""
        if forward_batch.forward_mode.is_decode_or_idle():
            return self._decode_view(
                encoder_id=encoder_id, req_pool_indices=forward_batch.req_pool_indices
            )
        return self._extend_view(encoder_id=encoder_id, forward_batch=forward_batch)

    def _extend_view(self, *, encoder_id: int, forward_batch: ForwardBatch) -> TapView:
        runs, run_requests = [], []
        for index, (row, first_position, num_tokens) in enumerate(
            zip(
                _on_host(
                    forward_batch.req_pool_indices_cpu, forward_batch.req_pool_indices
                ),
                _on_host(
                    forward_batch.extend_prefix_lens_cpu,
                    forward_batch.extend_prefix_lens,
                ),
                forward_batch.extend_seq_lens_cpu,
            )
        ):
            request_runs = self._readings[row].runs(
                first_position=first_position, num_tokens=num_tokens
            )
            runs += request_runs
            run_requests += [index] * len(request_runs)
        device = self._tap_lens.device
        run_lens = [run.num_tokens for run in runs]
        page_table = self._page_table(forward_batch.req_pool_indices)
        return TapView(
            taps=self._buffers[encoder_id].view(
                -1, self.page_size, 1, self._output_size
            ),
            page_table=page_table[
                async_h2d(run_requests, dtype=torch.int64, device=device)
            ],
            cu_seqlens_q=async_h2d(
                [0, *itertools.accumulate(run_lens)], dtype=torch.int32, device=device
            ),
            max_seqlen_q=max(run_lens, default=0),
            lens=async_h2d(
                [run.num_taps for run in runs], dtype=torch.int32, device=device
            ),
            decode=False,
            num_tokens=sum(run_lens),
            trtllm_workspace=self._attention_workspace,
        )

    def _decode_view(
        self, *, encoder_id: int, req_pool_indices: torch.Tensor
    ) -> TapView:
        tap_lens = self._tap_lens[req_pool_indices]
        num_requests = req_pool_indices.shape[0]
        return TapView(
            taps=self._buffers[encoder_id].view(
                -1, self.page_size, 1, self._output_size
            ),
            page_table=self._page_table(req_pool_indices),
            cu_seqlens_q=torch.arange(
                num_requests + 1, dtype=torch.int32, device=tap_lens.device
            ),
            max_seqlen_q=1,
            lens=tap_lens,
            decode=True,
            num_tokens=num_requests,
            trtllm_workspace=self._attention_workspace,
        )

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
