"""DeepSeek V4 keeps its residual as several streams that each sublayer mixes.
Readers must see the mean of the streams, as DSpark's draft does, and a reader's
result must reach every stream; a stale per-row input left on the state would let
the next layer read the residual from before the injection.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.franky.parts.injector.hyper_connection.injector import (
    HyperConnectionInjector,
)
from sglang.srt.franky.parts.injector.hyper_connection.output import (
    HyperConnectionOutput,
)
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_NUM_TOKENS = 3
_NUM_STREAMS = 4
_HIDDEN_SIZE = 8


class TestHyperConnectionInjector(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self.streams = torch.randn(_NUM_TOKENS, _NUM_STREAMS, _HIDDEN_SIZE)
        self.pre = torch.randn(_NUM_TOKENS, _NUM_STREAMS)
        self.state = HcState(
            self.streams.clone(), self.pre, stats=(torch.zeros(_NUM_TOKENS),)
        )
        self.output = HyperConnectionOutput(
            state=self.state, forward_batch=SimpleNamespace()
        )

    def test_read_is_the_mean_of_the_streams(self):
        read = HyperConnectionInjector().read(output=self.output)

        torch.testing.assert_close(read, self.streams.mean(dim=1))
        self.assertTrue(torch.equal(self.state.residual, self.streams))

    def test_inject_adds_to_every_stream_and_drops_what_the_old_rows_gave(self):
        addition = torch.randn(_NUM_TOKENS, _HIDDEN_SIZE)

        injected = HyperConnectionInjector().inject(
            output=self.output, addition=addition
        )

        torch.testing.assert_close(
            injected.state.residual, self.streams + addition[:, None, :]
        )
        self.assertIs(injected.state.pre, self.pre)
        self.assertIsNone(injected.state.stats)
        self.assertIsNone(injected.state.input)

    def test_any_mix_of_the_streams_gains_the_addition_by_its_weight(self):
        addition = torch.randn(_NUM_TOKENS, _HIDDEN_SIZE)
        weights = torch.rand(_NUM_TOKENS, _NUM_STREAMS)

        injected = HyperConnectionInjector().inject(
            output=self.output, addition=addition
        )

        mix_before = (weights[..., None] * self.streams).sum(dim=1)
        mix_after = (weights[..., None] * injected.state.residual).sum(dim=1)
        torch.testing.assert_close(
            mix_after, mix_before + weights.sum(dim=1, keepdim=True) * addition
        )


if __name__ == "__main__":
    unittest.main()
