"""When Sandy tokenizes on its own, Ricky's requests own tap pages from a pool of
their own, which can run out before the KV pool does. Its gauges let a viewer
show how full that pool was over a run, and how much of it chain holds kept for
their chains' next blocks.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.franky.cache import own_tap_pages_metrics as metrics_module
from sglang.srt.franky.cache.own_tap_pages import FrankyOwnTapPages
from sglang.srt.franky.cache.own_tap_pages_metrics import FrankyOwnTapPagesMetrics
from sglang.srt.franky.chains.chain import FrankyChain
from sglang.srt.franky.chains.holds import FrankyChainHolds, FrankyHeldTaps
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Gauge:
    def __init__(self) -> None:
        self.value = None

    def set(self, value) -> None:
        self.value = value


def _own_pages() -> FrankyOwnTapPages:
    return FrankyOwnTapPages(
        first_page=2,
        num_pages=12,
        max_pages_per_request=4,
        num_request_rows=2,
        device="cpu",
    )


def _block(chain_id: str):
    return SimpleNamespace(franky_chain=FrankyChain(id=chain_id, block_size=8))


class TestOwnTapPagesMetrics(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.own_pages = _own_pages()
        self.chain_holds = FrankyChainHolds(
            tree_cache=None, hold_seconds=60.0, own_pages=self.own_pages
        )
        self.metrics = FrankyOwnTapPagesMetrics(
            own_pages=self.own_pages,
            chain_holds=self.chain_holds,
            pages=_Gauge(),
            free_pages=_Gauge(),
            held_pages=_Gauge(),
            interval_seconds=0.25,
        )

    def _publish_at(self, seconds: float) -> None:
        with mock.patch.object(
            metrics_module.time, "perf_counter", return_value=seconds
        ):
            self.metrics.maybe_publish()

    def _hold(self, chain_id: str, *, num_pages: int) -> None:
        taps = FrankyHeldTaps(
            pages=self.own_pages.take(num_pages=num_pages), num_taps=num_pages * 16
        )
        self.chain_holds.hold_for_next_block(_block(chain_id), taps=taps)

    def test_publishes_the_pages_a_request_can_own_and_the_free_ones(self):
        self.own_pages.take(num_pages=3)
        self._publish_at(10.0)
        self.assertEqual(self.metrics.pages.value, 10)
        self.assertEqual(self.metrics.free_pages.value, 7)

    def test_publishes_the_pages_holds_keep_until_each_hold_gives_them_up(self):
        self._hold("a", num_pages=3)
        self._hold("b", num_pages=2)
        self._publish_at(10.0)
        self.assertEqual(self.metrics.held_pages.value, 5)
        self.assertEqual(self.metrics.free_pages.value, 5)

        self.chain_holds.take_taps(chain_id="a")
        self.chain_holds.release(chain_id="b")
        self._publish_at(10.25)

        self.assertEqual(self.metrics.held_pages.value, 0)
        self.assertEqual(self.metrics.free_pages.value, 7)

    def test_publishes_at_most_once_per_interval(self):
        self._publish_at(10.0)
        self.own_pages.take(num_pages=3)
        self._publish_at(10.1)
        self.assertEqual(self.metrics.free_pages.value, 10)
        self._publish_at(10.25)
        self.assertEqual(self.metrics.free_pages.value, 7)

    def test_no_gauges_without_own_pages_or_on_a_quiet_rank(self):
        context = SimpleNamespace(
            current_scheduler_metrics_enabled=True,
            collector=SimpleNamespace(labels={}),
        )
        self.assertIsNone(
            FrankyOwnTapPagesMetrics.of_scheduler(
                own_pages=None,
                chain_holds=self.chain_holds,
                metrics_collector_context=context,
                interval_seconds=1,
            )
        )
        context.current_scheduler_metrics_enabled = False
        self.assertIsNone(
            FrankyOwnTapPagesMetrics.of_scheduler(
                own_pages=self.own_pages,
                chain_holds=self.chain_holds,
                metrics_collector_context=context,
                interval_seconds=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
