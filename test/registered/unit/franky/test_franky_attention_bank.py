"""Kimi K3 keeps attention residuals: a layer reads a learned mix of a bank of
block sums and the running sum. Sandy's encoders must read the mix the next layer
reads, as DSpark's capture does, or the taps describe a hidden state no layer
ever saw; an injection must reach every row of that mix.
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.franky.parts.injector.attention_bank.injector import (
    AttentionBankInjector,
)
from sglang.srt.franky.parts.injector.attention_bank.output import (
    AttentionBankOutput,
    AttentionBankRead,
)
from sglang.srt.layers.attn_residual import AttnResidual
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Not a multiple of the fused kernel's block, so the torch mix runs on CPU.
_HIDDEN_SIZE = 12
_NUM_TOKENS = 3
_NUM_BLOCKS = 4
_NUM_VALID_BLOCKS = 2


class _ScoreProjection(nn.Module):
    """A ReplicatedLinear's call shape: the output and no bias."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, _HIDDEN_SIZE))

    def forward(self, rows):
        return rows @ self.weight.T, None


class _Norm(nn.Module):
    def forward(self, rows):
        return rows / rows.norm(dim=-1, keepdim=True)


def _mix(*, bank_rows, running_sum, score_proj, score_norm):
    """The read's math: a softmax over the bank's valid rows and the running sum,
    scored by the next layer's weights."""
    rows = torch.cat([bank_rows, running_sum[:, None]], dim=1)
    scores = score_proj(score_norm(rows))[0].squeeze(-1)
    return (torch.softmax(scores, dim=-1)[..., None] * rows).sum(dim=1)


class TestAttentionBankInjector(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.bank = AttnResidual(
            torch.zeros(_NUM_TOKENS, _HIDDEN_SIZE), _NUM_BLOCKS, None
        )
        self.bank.block_residual.copy_(
            torch.randn(_NUM_TOKENS, _NUM_BLOCKS, _HIDDEN_SIZE)
        )
        self.bank.num_valid_blocks = _NUM_VALID_BLOCKS
        self.read = AttentionBankRead(
            bank=self.bank,
            score_proj=_ScoreProjection(),
            score_norm=_Norm(),
            num_valid_blocks=_NUM_VALID_BLOCKS,
        )
        self.layer_output = torch.randn(_NUM_TOKENS, _HIDDEN_SIZE)
        self.residual = torch.randn(_NUM_TOKENS, _HIDDEN_SIZE)

    def _output(self):
        stream = ResidualStream(self.residual.clone())
        hidden_states = stream.record(self.layer_output.clone(), PLAIN_ADD)
        return AttentionBankOutput(
            hidden_states=hidden_states,
            forward_batch=SimpleNamespace(residual_stream=stream),
            next_read=self.read,
        )

    def test_read_is_the_mix_the_next_layer_reads(self):
        output = self._output()

        read = AttentionBankInjector().read(output=output)

        expected = _mix(
            bank_rows=self.bank.block_residual[:, :_NUM_VALID_BLOCKS],
            running_sum=self.layer_output + self.residual,
            score_proj=self.read.score_proj,
            score_norm=self.read.score_norm,
        )
        torch.testing.assert_close(read, expected)

    def test_read_leaves_the_bank_and_the_running_sum_as_they_are(self):
        bank_before = self.bank.block_residual.clone()
        output = self._output()

        AttentionBankInjector().read(output=output)

        self.assertTrue(torch.equal(self.bank.block_residual, bank_before))
        torch.testing.assert_close(
            residual_batch.fold(output.hidden_states, output.forward_batch),
            self.layer_output + self.residual,
        )

    def test_inject_reaches_every_row_the_next_layer_mixes(self):
        bank_before = self.bank.block_residual.clone()
        addition = torch.randn(_NUM_TOKENS, _HIDDEN_SIZE)

        injected = AttentionBankInjector().inject(
            output=self._output(), addition=addition
        )

        valid = slice(0, _NUM_VALID_BLOCKS)
        torch.testing.assert_close(
            self.bank.block_residual[:, valid],
            bank_before[:, valid] + addition[:, None],
        )
        self.assertTrue(
            torch.equal(
                self.bank.block_residual[:, _NUM_VALID_BLOCKS:],
                bank_before[:, _NUM_VALID_BLOCKS:],
            )
        )
        torch.testing.assert_close(
            residual_batch.fold(injected.hidden_states, injected.forward_batch),
            self.layer_output + self.residual + addition,
        )

    def test_a_bank_kept_in_slices_is_refused(self):
        self.bank.written_on_slice = True

        with self.assertRaisesRegex(ValueError, "SGLANG_K3_SP_ATTN_RES"):
            AttentionBankInjector().read(output=self._output())


if __name__ == "__main__":
    unittest.main()
