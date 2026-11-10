"""``FrankyParts`` turns each model's layer output into one residual tensor and back,
so a wrong sum here would silently change every token after a Franky layer.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.franky.cache.kv_allocator import FrankyKvAllocator
from sglang.srt.franky.cache.tap_cache import TAP_DTYPE
from sglang.srt.franky.parts.franky_parts import FrankyParts
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_PART_LAYER = 3
_OTHER_LAYER = 4


class _Parts(FrankyParts):
    """Adds ``extra`` after its layer; None reads only, as Sandy's encoders do."""

    def __init__(self, *, extra=None):
        super().__init__(
            config=SimpleNamespace(
                encoder=SimpleNamespace(output_size=256), tap_page_size=4
            ),
            layer_indices=frozenset({_PART_LAYER}),
            device="cpu",
        )
        self.extra = extra
        self.seen = None

    @classmethod
    def build(cls, **kwargs):
        raise NotImplementedError

    def _tap_encoder_ids(self):
        return (0, 2)

    def _after_layer(self, *, layer_idx, residual, forward_batch):
        self.seen = residual.clone()
        return self.extra


def _stream_output(*, layer_output, residual):
    """The state a layer leaves behind: its output, and a residual not yet added."""
    stream = ResidualStream(residual)
    return stream.record(layer_output, PLAIN_ADD), SimpleNamespace(
        residual_stream=stream
    )


class TestFrankyParts(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.extra = torch.tensor([[100.0, 200.0]])
        self.parts = _Parts(extra=self.extra)
        self.layer_output = torch.tensor([[1.0, 2.0]])
        self.residual = torch.tensor([[10.0, 20.0]])
        self.full_residual = torch.tensor([[11.0, 22.0]])

    def test_stream_form_adds_to_the_layer_output(self):
        hidden_states, forward_batch = _stream_output(
            layer_output=self.layer_output, residual=self.residual
        )
        # The add happens in place, as the model's own add_to_output calls do.
        expected_output = self.layer_output + self.extra

        result = self.parts.after_layer(
            layer_idx=_PART_LAYER,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        self.assertTrue(torch.equal(self.parts.seen, self.full_residual))
        self.assertTrue(torch.equal(result, expected_output))
        # The residual is still added by the next layer, as without Franky.
        self.assertTrue(
            torch.equal(
                residual_batch.fold(result, forward_batch),
                self.full_residual + self.extra,
            )
        )

    def test_pair_form_adds_to_the_layer_output(self):
        """The output gets the addition; the residual stays the residual."""
        hidden_states, residual = self.parts.after_layer_with_residual(
            layer_idx=_PART_LAYER,
            hidden_states=self.layer_output,
            residual=self.residual,
            forward_batch=None,
        )

        self.assertTrue(torch.equal(self.parts.seen, self.full_residual))
        self.assertTrue(torch.equal(hidden_states, self.layer_output + self.extra))
        self.assertIs(residual, self.residual)

    def test_pair_form_without_a_residual_adds_to_the_output(self):
        """With no residual yet, the output is the whole residual."""
        hidden_states, residual = self.parts.after_layer_with_residual(
            layer_idx=_PART_LAYER,
            hidden_states=self.full_residual,
            residual=None,
            forward_batch=None,
        )

        self.assertTrue(torch.equal(self.parts.seen, self.full_residual))
        self.assertTrue(torch.equal(hidden_states, self.full_residual + self.extra))
        self.assertIsNone(residual)

    def test_a_part_that_only_reads_changes_nothing(self):
        parts = _Parts(extra=None)
        hidden_states, forward_batch = _stream_output(
            layer_output=self.layer_output, residual=self.residual
        )

        result = parts.after_layer(
            layer_idx=_PART_LAYER,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        pair = parts.after_layer_with_residual(
            layer_idx=_PART_LAYER,
            hidden_states=self.layer_output,
            residual=self.residual,
            forward_batch=None,
        )

        self.assertIs(result, hidden_states)
        self.assertTrue(
            torch.equal(residual_batch.fold(result, forward_batch), self.full_residual)
        )
        self.assertIs(pair[0], self.layer_output)
        self.assertIs(pair[1], self.residual)

    def test_a_layer_without_a_part_is_left_alone(self):
        hidden_states, forward_batch = _stream_output(
            layer_output=self.layer_output, residual=self.residual
        )

        result = self.parts.after_layer(
            layer_idx=_OTHER_LAYER,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        pair = self.parts.after_layer_with_residual(
            layer_idx=_OTHER_LAYER,
            hidden_states=self.layer_output,
            residual=self.residual,
            forward_batch=None,
        )

        self.assertIs(result, hidden_states)
        self.assertIs(pair[0], self.layer_output)
        self.assertIs(pair[1], self.residual)
        self.assertIsNone(self.parts.seen)

    def test_the_budget_charges_the_encoders_of_this_stage(self):
        options = self.parts.kv_options()

        self.assertEqual(options.tap_bytes_per_token, 2 * 256 * TAP_DTYPE.itemsize)
        self.assertIs(options.paged_allocator_cls, FrankyKvAllocator)

    def test_any_other_kv_allocator_is_refused_at_start(self):
        with self.assertRaisesRegex(ValueError, "plain paged KV allocator"):
            self.parts.init_tap_cache(
                num_tokens=8, kv_allocator=object(), req_to_token_pool=object()
            )


if __name__ == "__main__":
    unittest.main()
