"""``FrankyParts`` reads and changes a model's residual only through its injector,
so a wrong sum there would silently change every token after a Franky layer.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.franky.cache.kv_allocator import FrankyKvAllocator
from sglang.srt.franky.cache.tap_cache import TAP_DTYPE
from sglang.srt.franky.parts.franky_mixin import FrankyMixin
from sglang.srt.franky.parts.franky_parts import FrankyParts
from sglang.srt.franky.parts.injector.residual.injector import ResidualInjector
from sglang.srt.franky.parts.injector.residual.output import StreamOutput
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
            injector=ResidualInjector(),
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

    def _after_layer(self, *, layer_idx, output):
        self.seen = self.injector.read(output=output)
        if self.extra is None:
            return output
        return self.injector.inject(output=output, addition=self.extra)


class _Model(FrankyMixin):
    franky_output_cls = StreamOutput


def _stream_output(*, layer_output, residual):
    """The state a layer leaves behind: its output, and a residual not yet added."""
    stream = ResidualStream(residual)
    forward_batch = SimpleNamespace(residual_stream=stream)
    hidden_states = stream.record(layer_output, PLAIN_ADD)
    return StreamOutput(hidden_states=hidden_states, forward_batch=forward_batch)


class TestResidualInjector(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.layer_output = torch.tensor([[1.0, 2.0]])
        self.residual = torch.tensor([[10.0, 20.0]])
        self.full_residual = torch.tensor([[11.0, 22.0]])

    def test_read_is_the_whole_stream_and_leaves_it_as_it_is(self):
        output = _stream_output(layer_output=self.layer_output, residual=self.residual)

        read = ResidualInjector().read(output=output)

        self.assertTrue(torch.equal(read, self.full_residual))
        self.assertTrue(
            torch.equal(
                residual_batch.fold(output.hidden_states, output.forward_batch),
                self.full_residual,
            )
        )

    def test_inject_adds_to_the_layer_output_and_the_next_layer_adds_the_rest(self):
        output = _stream_output(layer_output=self.layer_output, residual=self.residual)
        addition = torch.tensor([[100.0, 200.0]])

        injected = ResidualInjector().inject(output=output, addition=addition)

        self.assertTrue(
            torch.equal(
                residual_batch.fold(injected.hidden_states, injected.forward_batch),
                self.full_residual + addition,
            )
        )


class TestFrankyParts(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.extra = torch.tensor([[100.0, 200.0]])
        self.layer_output = torch.tensor([[1.0, 2.0]])
        self.residual = torch.tensor([[10.0, 20.0]])
        self.full_residual = torch.tensor([[11.0, 22.0]])

    def test_a_part_reads_and_injects_through_the_injector(self):
        parts = _Parts(extra=self.extra)
        model = _Model()
        model.franky_parts = parts
        output = _stream_output(layer_output=self.layer_output, residual=self.residual)

        hidden_states = model.franky_after_layer(
            layer_idx=_PART_LAYER,
            hidden_states=output.hidden_states,
            forward_batch=output.forward_batch,
        )

        self.assertTrue(torch.equal(parts.seen, self.full_residual))
        self.assertTrue(
            torch.equal(
                residual_batch.fold(hidden_states, output.forward_batch),
                self.full_residual + self.extra,
            )
        )

    def test_a_part_that_only_reads_changes_nothing(self):
        parts = _Parts(extra=None)
        output = _stream_output(layer_output=self.layer_output, residual=self.residual)

        result = parts.after_layer(layer_idx=_PART_LAYER, output=output)

        self.assertIs(result, output)
        self.assertTrue(
            torch.equal(
                residual_batch.fold(result.hidden_states, result.forward_batch),
                self.full_residual,
            )
        )

    def test_a_layer_without_a_part_is_left_alone(self):
        parts = _Parts(extra=self.extra)
        model = _Model()
        model.franky_parts = parts
        hidden_states = object()

        result = model.franky_after_layer(
            layer_idx=_OTHER_LAYER, hidden_states=hidden_states, forward_batch=None
        )

        self.assertIs(result, hidden_states)
        self.assertIsNone(parts.seen)

    def test_the_budget_charges_the_encoders_of_this_stage(self):
        options = _Parts().kv_options()

        self.assertEqual(options.tap_bytes_per_token, 2 * 256 * TAP_DTYPE.itemsize)
        self.assertIs(options.paged_allocator_cls, FrankyKvAllocator)

    def test_any_other_kv_allocator_is_refused_at_start(self):
        with self.assertRaisesRegex(ValueError, "plain paged KV allocator"):
            _Parts().init_tap_cache(
                num_tokens=8, kv_allocator=object(), req_to_token_pool=object()
            )


if __name__ == "__main__":
    unittest.main()
