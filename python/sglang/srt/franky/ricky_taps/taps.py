from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
from sglang.srt.franky.cache.tap_pool_metrics import FrankyTapPoolMetrics
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.ricky_taps.tap_requests import RickyTapRequests

if TYPE_CHECKING:
    from sglang.srt.franky.cache.mamba_room import FrankyMambaRoom
    from sglang.srt.franky.parts.ricky_parts import RickyParts
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.observability.metrics_collector import (
        SchedulerMetricsCollectorContext,
    )


class RickyTaps:
    """Ricky's taps on this rank, on prefill, decode and aggregated engines alike:
    the pool over the model's tap pages, the chains' holds on what a block left,
    the requests that read taps, and the pool's gauges.

    Attributes:
        cache: the pool and its prefix tree over Sandy's tokens.
        chain_holds: each chain's locks between two of its blocks.
        requests: the requests that read taps, from arrival to their end.
        pool_metrics: the pool's gauges; None where this rank publishes none.
    """

    def __init__(
        self,
        *,
        cache: RickyTapCache,
        chain_holds: FrankyChainHolds,
        requests: RickyTapRequests,
        pool_metrics: Optional[FrankyTapPoolMetrics],
    ) -> None:
        self.cache = cache
        self.chain_holds = chain_holds
        self.requests = requests
        self.pool_metrics = pool_metrics

    @staticmethod
    def build(
        *,
        parts: RickyParts,
        tree_cache: Optional[BasePrefixCache],
        mamba_room: Optional[FrankyMambaRoom],
        metrics_collector_context: SchedulerMetricsCollectorContext,
        metrics_interval_seconds: float,
    ) -> RickyTaps:
        """
        Args:
            tree_cache: the engine's KV prefix tree; None on an engine that caches
                no prompts, such as PD decode.
            mamba_room: the engine's mamba state pool; None without one.
            metrics_interval_seconds: the least time between two publishes.
        """
        cache = RickyTapCache(buffers=parts.tap_buffers, stripes=parts.stripes)
        chain_holds = FrankyChainHolds.from_config(
            tree_cache=tree_cache, tap_cache=cache
        )
        return RickyTaps(
            cache=cache,
            chain_holds=chain_holds,
            requests=RickyTapRequests(
                tap_cache=cache,
                request_table=parts.request_table,
                chain_holds=chain_holds,
                shared_tokenizer=parts.config.shared_tokenizer,
                mamba_room=mamba_room,
            ),
            pool_metrics=FrankyTapPoolMetrics.of_scheduler(
                tap_cache=cache,
                metrics_collector_context=metrics_collector_context,
                interval_seconds=metrics_interval_seconds,
            ),
        )

    def publish_metrics(self) -> None:
        if self.pool_metrics is not None:
            self.pool_metrics.maybe_publish()

    def flush(self, flush_engine: Callable[[], bool]) -> bool:
        """Flush the engine's caches with the taps': the holds go first, as they
        lock the engine's prefix tree, and the cached stripes go once it flushed.

        Args:
            flush_engine: flushes the engine's own caches, and says whether it did.
        """
        self.chain_holds.release_all()
        flushed = flush_engine()
        if flushed:
            self.cache.reset()
        return flushed
