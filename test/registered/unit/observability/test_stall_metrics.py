"""Queue and pool gauges are written by batch reports and by full idleness only.
While requests wait without a batch (PD bootstrap and transfers, a Franky Ricky
waiting for taps, a Sandy holding requests for their tap sends) a scrape would
read old values, and every distribution over time built from it would be wrong.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers.scheduler_components import metrics_reporter
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    STALL_METRICS_INTERVAL_S,
    SchedulerMetricsReporter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _reporter(*, last_log_time: float, publishes: bool = True):
    reporter = object.__new__(SchedulerMetricsReporter)
    reporter.current_scheduler_metrics_enabled = publishes
    reporter.metrics_collector = SimpleNamespace(last_log_time=last_log_time)
    reporter._log_snapshot_stats = mock.Mock()
    return reporter


class TestStallMetrics(CustomTestCase):
    def test_a_stall_publishes_once_the_interval_passed(self):
        reporter = _reporter(last_log_time=10.0)
        with mock.patch.object(
            metrics_reporter.time,
            "perf_counter",
            return_value=10.0 + STALL_METRICS_INTERVAL_S,
        ):
            reporter.maybe_log_stall_metrics()
        reporter._log_snapshot_stats.assert_called_once()

    def test_a_recent_report_is_not_repeated(self):
        reporter = _reporter(last_log_time=10.0)
        with mock.patch.object(
            metrics_reporter.time,
            "perf_counter",
            return_value=10.0 + STALL_METRICS_INTERVAL_S / 2,
        ):
            reporter.maybe_log_stall_metrics()
        reporter._log_snapshot_stats.assert_not_called()

    def test_a_rank_that_publishes_no_metrics_stays_quiet(self):
        reporter = _reporter(last_log_time=0.0, publishes=False)
        reporter.maybe_log_stall_metrics()
        reporter._log_snapshot_stats.assert_not_called()


if __name__ == "__main__":
    unittest.main()
