"""How long Full KV stays in device memory before eviction, kept by the Python
TreeCore. Each node is stamped when its KV enters device memory (insert, or a
host load-back); its age at eviction counts once per evicted token, so the
histogram's distribution is over tokens, not nodes.
"""

import unittest
from array import array
from unittest import mock

import prometheus_client
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache import unified_tree_core
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.tree_core_registry import _TREE_CORE_REGISTRY
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import RadixCacheMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now


def _cache():
    with (
        mock.patch.dict(_TREE_CORE_REGISTRY),
        envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"),
    ):
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
                page_size=1,
                tree_components=(ComponentType.FULL,),
            )
        )
    cache.tree_core.device_eviction_ages = []
    return cache


def _insert(core, token_ids):
    step = core.begin_insert(
        InsertParams(
            key=RadixKey(array("q", token_ids)),
            value=torch.arange(len(token_ids), dtype=torch.int64),
        )
    )
    while step.result is None:
        step = core.resume_insert()
    core.end_insert()
    return step.result.last_device_node


def _evict(core, node_id):
    result = core.evict_device_leaf(node_id, False)
    # Drain what the step freed, as the cache does after every step.
    result.device_frees.clear()
    result.host_frees.clear()
    result.cache_actions.clear()


class TestEvictedKvAge(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.clock = _Clock()
        clock_patch = mock.patch.object(unified_tree_core, "time", self.clock)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.cache = _cache()
        self.core = self.cache.tree_core

    def test_a_leaf_reports_its_age_and_tokens_once(self):
        leaf = _insert(self.core, list(range(48)))
        self.clock.now = 107.5
        _evict(self.core, leaf)

        self.assertEqual(self.core.take_device_eviction_ages(), [(7.5, 48)])
        self.assertEqual(self.core.take_device_eviction_ages(), [])

    def test_a_split_keeps_the_time_the_kv_entered(self):
        _insert(self.core, list(range(32)))
        self.clock.now = 103.0
        branch = _insert(self.core, list(range(16)) + [900, 901, 902, 903])
        self.clock.now = 110.0
        _evict(self.core, branch)

        self.assertEqual(self.core.take_device_eviction_ages(), [(7.0, 4)])

    def test_nothing_accumulates_with_metrics_off(self):
        self.core.device_eviction_ages = None
        leaf = _insert(self.core, list(range(8)))
        _evict(self.core, leaf)
        self.assertEqual(self.core.take_device_eviction_ages(), [])


class TestEvictedKvAgeHistogram(CustomTestCase):
    def _collector(self):
        # A private registry: the default one already holds the radix-cache
        # metrics of any collector another test built in this process.
        collector = object.__new__(RadixCacheMetricsCollector)
        collector.labels = {"cache_type": "test-ages"}
        collector.evicted_kv_age_seconds = prometheus_client.Histogram(
            name="sglang:evicted_kv_age_seconds",
            documentation="test",
            labelnames=["cache_type"],
            buckets=[0.1, 1.0, 10.0],
            registry=prometheus_client.CollectorRegistry(),
        )
        return collector

    def test_counts_one_per_token(self):
        collector = self._collector()
        collector.observe_evicted_kv_ages([(0.1, 3), (7.0, 5), (60.0, 2)])

        histogram = collector.evicted_kv_age_seconds.labels(cache_type="test-ages")
        self.assertEqual([bucket.get() for bucket in histogram._buckets], [3, 0, 5, 2])
        self.assertAlmostEqual(histogram._sum.get(), 0.1 * 3 + 7.0 * 5 + 60.0 * 2)


if __name__ == "__main__":
    unittest.main()
