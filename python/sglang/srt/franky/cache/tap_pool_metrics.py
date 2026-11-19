from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prometheus_client.metrics import Gauge as LabeledGauge

    from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
    from sglang.srt.observability.metrics_collector import (
        SchedulerMetricsCollectorContext,
    )


class FrankyTapPoolMetrics:
    """Ricky's tap pool as gauges a scrape reads, in request tap pages over all
    attention-TP ranks: all of them, the free ones, and the ones only the prefix
    tree keeps, which it gives back when room is needed.
    """

    def __init__(
        self,
        *,
        tap_cache: RickyTapCache,
        pages: LabeledGauge,
        free_pages: LabeledGauge,
        evictable_pages: LabeledGauge,
        interval_seconds: float,
    ) -> None:
        self.tap_cache = tap_cache
        self.pages = pages
        self.free_pages = free_pages
        self.evictable_pages = evictable_pages
        self.interval_seconds = interval_seconds
        self.last_publish_seconds = 0.0

    @staticmethod
    def of_scheduler(
        *,
        tap_cache: RickyTapCache,
        metrics_collector_context: SchedulerMetricsCollectorContext,
        interval_seconds: float,
    ) -> Optional[FrankyTapPoolMetrics]:
        """
        Args:
            interval_seconds: the least time between two publishes.
        Returns:
            None when this rank publishes no metrics.
        """
        if not metrics_collector_context.current_scheduler_metrics_enabled:
            return None
        return FrankyTapPoolMetrics.create(
            tap_cache=tap_cache,
            labels=metrics_collector_context.collector.labels,
            interval_seconds=interval_seconds,
        )

    @staticmethod
    def create(
        *, tap_cache: RickyTapCache, labels: dict[str, str], interval_seconds: float
    ) -> FrankyTapPoolMetrics:
        """
        Args:
            labels: the scheduler's metric labels, so the gauges sit beside its own.
        """
        # SGLang imports prometheus_client only once metrics are on.
        from prometheus_client import Gauge

        def gauge(name: str, documentation: str) -> LabeledGauge:
            return Gauge(
                name=name,
                documentation=documentation,
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            ).labels(**labels)

        return FrankyTapPoolMetrics(
            tap_cache=tap_cache,
            pages=gauge("sglang:franky_tap_pages", "Ricky's tap pages."),
            free_pages=gauge(
                "sglang:franky_tap_pages_free", "Tap pages that nothing holds."
            ),
            evictable_pages=gauge(
                "sglang:franky_tap_pages_evictable",
                "Tap pages that only the prefix tree keeps.",
            ),
            interval_seconds=interval_seconds,
        )

    def maybe_publish(self) -> None:
        now = time.perf_counter()
        if now < self.last_publish_seconds + self.interval_seconds:
            return
        self.last_publish_seconds = now
        usage = self.tap_cache.usage()
        self.pages.set(usage.pages)
        self.free_pages.set(usage.free_pages)
        self.evictable_pages.set(usage.evictable_pages)
