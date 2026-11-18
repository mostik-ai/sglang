from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prometheus_client.metrics import Gauge as LabeledGauge

    from sglang.srt.franky.cache.own_tap_pages import FrankyOwnTapPages
    from sglang.srt.franky.chains.holds import FrankyChainHolds
    from sglang.srt.observability.metrics_collector import (
        SchedulerMetricsCollectorContext,
    )


class FrankyOwnTapPagesMetrics:
    """The pool of tap pages Ricky's requests own, as gauges a scrape reads: how
    many pages a request can own, how many no request owns, and how many chain
    holds keep for their chains' next blocks.
    """

    def __init__(
        self,
        *,
        own_pages: FrankyOwnTapPages,
        chain_holds: FrankyChainHolds,
        pages: LabeledGauge,
        free_pages: LabeledGauge,
        held_pages: LabeledGauge,
        interval_seconds: float,
    ) -> None:
        self.own_pages = own_pages
        self.chain_holds = chain_holds
        self.pages = pages
        self.free_pages = free_pages
        self.held_pages = held_pages
        self.interval_seconds = interval_seconds
        self.last_publish_seconds = 0.0

    @staticmethod
    def of_scheduler(
        *,
        own_pages: Optional[FrankyOwnTapPages],
        chain_holds: FrankyChainHolds,
        metrics_collector_context: SchedulerMetricsCollectorContext,
        interval_seconds: float,
    ) -> Optional[FrankyOwnTapPagesMetrics]:
        """
        Args:
            interval_seconds: the least time between two publishes.
        Returns:
            None when requests own no tap pages or this rank publishes no metrics.
        """
        if (
            own_pages is None
            or not metrics_collector_context.current_scheduler_metrics_enabled
        ):
            return None
        return FrankyOwnTapPagesMetrics.create(
            own_pages=own_pages,
            chain_holds=chain_holds,
            labels=metrics_collector_context.collector.labels,
            interval_seconds=interval_seconds,
        )

    @staticmethod
    def create(
        *,
        own_pages: FrankyOwnTapPages,
        chain_holds: FrankyChainHolds,
        labels: dict[str, str],
        interval_seconds: float,
    ) -> FrankyOwnTapPagesMetrics:
        """
        Args:
            labels: the scheduler's metric labels, so the gauges sit beside its own.
        """
        # SGLang imports prometheus_client only once metrics are on.
        from prometheus_client import Gauge

        return FrankyOwnTapPagesMetrics(
            own_pages=own_pages,
            chain_holds=chain_holds,
            pages=Gauge(
                name="sglang:franky_own_tap_pages",
                documentation="Tap pages that Ricky's requests can own.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            ).labels(**labels),
            free_pages=Gauge(
                name="sglang:franky_own_tap_pages_free",
                documentation="Tap pages that no request owns.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            ).labels(**labels),
            held_pages=Gauge(
                name="sglang:franky_own_tap_pages_held",
                documentation="Tap pages that chain holds keep for their chains' "
                "next blocks.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            ).labels(**labels),
            interval_seconds=interval_seconds,
        )

    def maybe_publish(self) -> None:
        now = time.perf_counter()
        if now < self.last_publish_seconds + self.interval_seconds:
            return
        self.last_publish_seconds = now
        self.pages.set(self.own_pages.num_ownable_pages)
        self.free_pages.set(self.own_pages.num_free())
        self.held_pages.set(self.chain_holds.num_held_tap_pages())
