import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.franky.cache import tap_pool_metrics as metrics_module
from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache, TapReservation
from sglang.srt.franky.cache.tap_buffers import TapBuffers
from sglang.srt.franky.cache.tap_pool_metrics import FrankyTapPoolMetrics
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Gauge:
    def __init__(self) -> None:
        self.value = None

    def set(self, value) -> None:
        self.value = value


def _tap_cache() -> RickyTapCache:
    return RickyTapCache(
        buffers=TapBuffers(
            encoder_ids=(0,), output_size=8, page_size=4, num_pages=11, device="cpu"
        ),
        stripes=TapStripes(num_ranks=2, rank=1),
    )


class TestTapPoolMetrics(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tap_cache = _tap_cache()
        self.metrics = FrankyTapPoolMetrics(
            tap_cache=self.tap_cache,
            pages=_Gauge(),
            free_pages=_Gauge(),
            evictable_pages=_Gauge(),
            interval_seconds=0.25,
        )

    def _publish_at(self, seconds: float) -> None:
        with mock.patch.object(
            metrics_module.time, "perf_counter", return_value=seconds
        ):
            self.metrics.maybe_publish()

    def _cache(self, sandy_ids: list[int]) -> None:
        cached = self.tap_cache.match(sandy_ids)
        pages = self.tap_cache.take(len(sandy_ids) // self.tap_cache.stripe_size)
        self.tap_cache.release(
            self.tap_cache.cache(
                TapReservation(
                    sandy_ids=sandy_ids,
                    pages=torch.cat([cached.pages, pages]),
                    cached=cached,
                )
            )
        )

    def test_publishes_request_pages_over_all_ranks(self):
        self.tap_cache.take(3)

        self._publish_at(10.0)

        self.assertEqual(self.metrics.pages.value, 20)
        self.assertEqual(self.metrics.free_pages.value, 14)
        self.assertEqual(self.metrics.evictable_pages.value, 0)

    def test_publishes_what_only_the_prefix_tree_keeps_as_evictable(self):
        self._cache(list(range(16)))

        self._publish_at(10.0)

        self.assertEqual(self.metrics.free_pages.value, 16)
        self.assertEqual(self.metrics.evictable_pages.value, 4)

    def test_a_locked_prefix_is_neither_free_nor_evictable(self):
        self._cache(list(range(16)))
        cached = self.tap_cache.match(list(range(16)))

        self._publish_at(10.0)

        self.assertEqual(self.metrics.free_pages.value, 16)
        self.assertEqual(self.metrics.evictable_pages.value, 0)
        self.tap_cache.release(cached.lock)

    def test_publishes_at_most_once_per_interval(self):
        self._publish_at(10.0)
        self.tap_cache.take(3)
        self._publish_at(10.1)
        self.assertEqual(self.metrics.free_pages.value, 20)
        self._publish_at(10.25)
        self.assertEqual(self.metrics.free_pages.value, 14)

    def test_no_gauges_on_a_quiet_rank(self):
        context = SimpleNamespace(
            current_scheduler_metrics_enabled=False,
            collector=SimpleNamespace(labels={}),
        )

        self.assertIsNone(
            FrankyTapPoolMetrics.of_scheduler(
                tap_cache=self.tap_cache,
                metrics_collector_context=context,
                interval_seconds=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
