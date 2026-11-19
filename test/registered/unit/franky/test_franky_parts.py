"""``FrankyParts`` reads and changes a model's residual only through its injector,
so a wrong sum there would silently change every token after a Franky layer.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
from sglang.srt.franky.cache.tap_buffers import TAP_DTYPE
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.franky.parts import ricky_parts
from sglang.srt.franky.parts.franky_mixin import FrankyMixin
from sglang.srt.franky.parts.franky_parts import FrankyParts
from sglang.srt.franky.parts.injector.residual.injector import ResidualInjector
from sglang.srt.franky.parts.injector.residual.output import StreamOutput
from sglang.srt.franky.parts.ricky_parts import RickyParts
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

    def __init__(self, *, extra=None, num_ranks=1):
        super().__init__(
            config=SimpleNamespace(
                encoder=SimpleNamespace(output_size=256), tap_page_size=4
            ),
            injector=ResidualInjector(),
            layer_indices=frozenset({_PART_LAYER}),
            device="cpu",
            stripes=TapStripes(num_ranks=num_ranks, rank=0),
        )
        self.extra = extra
        self.seen = None

    @classmethod
    def build(cls, **kwargs):
        raise NotImplementedError

    def _weight_prefix(self):
        return "encoders"

    def _parts_by_index(self):
        return {}

    def _num_parts_on_all_stages(self):
        return 0

    def init_taps(self, **kwargs):
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

    def test_the_budget_charges_this_ranks_share_of_the_encoders_of_this_stage(
        self,
    ):
        tap_bytes = 2 * 256 * TAP_DTYPE.itemsize

        self.assertEqual(
            [
                _Parts(num_ranks=num_ranks).tap_bytes_per_kv_token()
                for num_ranks in (1, 2, 3)
            ],
            [tap_bytes, tap_bytes // 2, -(-tap_bytes // 3)],
        )


class TestRickyTapPool(CustomTestCase):
    """Ricky's taps get a pool of their own, a share of the KV tokens split over
    the attention-TP ranks; a share too small for one request at the context
    length would park it for good.
    """

    def setUp(self):
        super().setUp()
        group = patch.object(ricky_parts, "get_attn_tp_group", lambda: "attn-tp")
        group.start()
        self.addCleanup(group.stop)

    def _parts(self, *, tap_pool_ratio, num_ranks=1):
        return RickyParts(
            config=SimpleNamespace(
                encoder=SimpleNamespace(output_size=256, layers=[1, 2]),
                reader=SimpleNamespace(layers=[]),
                tap_page_size=4,
            ),
            injector=ResidualInjector(),
            readers={},
            device="cpu",
            stripes=TapStripes(num_ranks=num_ranks, rank=num_ranks - 1),
            tap_pool_ratio=tap_pool_ratio,
        )

    def _init(self, parts, *, context_len, kv_page_size=8):
        """The pool the scheduler builds over the pages the parts own."""
        parts.init_taps(
            num_kv_tokens=64,
            kv_page_size=kv_page_size,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((2, context_len), dtype=torch.int32)
            ),
        )
        return RickyTapCache(buffers=parts.tap_buffers, stripes=parts.stripes)

    def test_the_budget_charges_the_ratio_of_the_taps_per_kv_token_on_this_rank(
        self,
    ):
        tap_bytes = 2 * 256 * TAP_DTYPE.itemsize

        self.assertEqual(
            [
                self._parts(
                    tap_pool_ratio=0.25, num_ranks=num_ranks
                ).tap_bytes_per_kv_token()
                for num_ranks in (1, 2)
            ],
            [tap_bytes // 4, tap_bytes // 8],
        )

    def test_the_pool_holds_the_ratio_of_the_kv_tokens_in_whole_stripes(self):
        tap_cache = self._init(self._parts(tap_pool_ratio=0.25), context_len=16)

        self.assertEqual(tap_cache.num_takeable_stripes(), 4)
        self.assertEqual(tap_cache.usage().pages, 4)
        self.assertEqual(tap_cache.buffers.transfer_buffers()[0][1], 5 * 4 * 256 * 2)

    def test_each_rank_keeps_its_share_of_the_pool(self):
        one = self._init(self._parts(tap_pool_ratio=0.5), context_len=16)
        two = self._init(self._parts(tap_pool_ratio=0.5, num_ranks=2), context_len=16)

        self.assertEqual((one.usage().pages, two.usage().pages), (8, 8))
        self.assertEqual((one.buffers.num_pages, two.buffers.num_pages), (9, 5))
        self.assertEqual(two.stripes, TapStripes(num_ranks=2, rank=1))

    def test_a_pool_short_of_one_request_at_the_context_length_is_refused(self):
        with self.assertRaisesRegex(ValueError, "fewer than one request"):
            self._init(self._parts(tap_pool_ratio=0.25), context_len=32)

    def test_any_kv_page_size_works_for_ricky(self):
        tap_cache = self._init(
            self._parts(tap_pool_ratio=1.0), context_len=16, kv_page_size=6
        )

        self.assertEqual(tap_cache.num_takeable_stripes(), 16)


if __name__ == "__main__":
    unittest.main()
