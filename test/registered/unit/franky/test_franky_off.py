"""With ``--franky-side`` unset, a solo model must run the default code."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.franky.parts.franky_mixin import FrankyMixin
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler, get_scheduler_class
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestFrankyOff(CustomTestCase):
    def test_the_default_scheduler_is_used(self):
        with patch.object(
            scheduler_module,
            "get_franky",
            return_value=SimpleNamespace(franky_side="null"),
        ):
            self.assertIs(get_scheduler_class(), Scheduler)

    def test_the_mixin_hands_back_the_same_objects(self):
        model = FrankyMixin()
        hidden_states, residual = object(), object()

        self.assertIs(
            model.franky_after_layer(
                layer_idx=0, hidden_states=hidden_states, forward_batch=None
            ),
            hidden_states,
        )
        self.assertEqual(
            model.franky_after_layer_with_residual(
                layer_idx=0,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=None,
            ),
            (hidden_states, residual),
        )
        model.franky_read_pp(pp_proxy_tensors=None)
        model.franky_write_pp(pp_proxy_tensors=None)


if __name__ == "__main__":
    unittest.main()
