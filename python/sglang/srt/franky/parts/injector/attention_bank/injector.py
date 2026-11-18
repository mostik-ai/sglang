from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
from typing_extensions import override

from sglang.srt.franky.parts.injector.attention_bank.output import AttentionBankOutput
from sglang.srt.franky.parts.injector.injector import Injector
from sglang.srt.layers.attn_residual import aggregate_stream
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

if TYPE_CHECKING:
    import torch

    from sglang.srt.franky.parts.injector.attention_bank.output import (
        AttentionBankRead,
    )


class AttentionBankInjector(Injector):
    """Attention residuals: parts read the mix the next layer reads, as DSpark's
    capture does. A reader's result goes into every row of that mix, the bank's
    rows and the running sum; the mix weights sum to one, so every later mix gains
    it, as a plain residual add would.
    """

    output_cls = AttentionBankOutput

    @override
    def read(self, *, output: AttentionBankOutput) -> torch.Tensor:
        read = output.next_read
        _check_whole_rows(read)
        running_sum = residual_batch.snapshot(
            output.hidden_states, output.forward_batch
        )
        return aggregate_stream(
            running_sum,
            read.bank.block_residual,
            read.num_valid_blocks,
            read.score_proj,
            read.score_norm,
        )

    @override
    def inject(
        self, *, output: AttentionBankOutput, addition: torch.Tensor
    ) -> AttentionBankOutput:
        read = output.next_read
        _check_whole_rows(read)
        read.bank.block_residual[:, : read.num_valid_blocks] += addition[:, None, :]
        hidden_states = residual_batch.add_to_output(
            output.hidden_states, output.forward_batch, addition
        )
        return msgspec.structs.replace(output, hidden_states=hidden_states)


def _check_whole_rows(read: AttentionBankRead) -> None:
    if read.bank.written_on_slice:
        raise ValueError(
            "Franky reads every row of the attention-residual bank, but this rank "
            "keeps only its attention-TP slice of it; unset SGLANG_K3_SP_ATTN_RES."
        )
