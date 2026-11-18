"""Tests for WatchdogRaw in watchdog.py.

No watchdog thread starts: each case calls _watchdog_once() on a fake clock.
It returns when the watchdog fires; _ClockEnd means it did not fire.
"""

import unittest
from unittest import mock

from sglang.srt.utils import watchdog
from sglang.srt.utils.watchdog import WatchdogRaw
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")

TIMEOUT = 300.0
# Like time.sleep, the fake sleep never wakes early and wakes a little late.
OVERSLEEP = 0.001
IDLE = 10 * TIMEOUT
# The watchdog checks every TIMEOUT / 2; start the batch early, midway and
# late between two checks.
BATCH_STARTS = [IDLE + f * TIMEOUT / 2 for f in (0.01, 0.5, 0.99)]


class _ClockEnd(Exception):
    pass


class _FakeTime:
    """Stands in for the time module inside watchdog.py."""

    def __init__(self, end: float):
        self.now = 0.0
        self.end = end

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds + OVERSLEEP
        if self.now >= self.end:
            raise _ClockEnd()


def _has_batch(start: float, end: float = float("inf")):
    return lambda now: start <= now < end


class TestWatchdogRaw(CustomTestCase):
    def setUp(self):
        for name in ("threading", "pyspy_dump_schedulers"):
            patcher = mock.patch.object(watchdog, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _fire_time(self, is_active, get_counter, end):
        """Run the watchdog until fake time `end`; return when it fired, or None."""
        clock = _FakeTime(end)
        raw = WatchdogRaw(
            debug_name="Test",
            get_counter=lambda: get_counter(clock.now),
            is_active=lambda: is_active(clock.now),
            watchdog_timeout=TIMEOUT,
            soft=True,
            dump_info=lambda: "batch info",
        )
        with mock.patch.object(watchdog, "time", clock):
            try:
                raw._watchdog_once()
            except _ClockEnd:
                return None
        return clock.now

    def test_idle_time_does_not_count_toward_a_batch(self):
        """A scheduler idle for longer than the timeout gets a batch that runs
        less than the timeout, and its forward counter does not move while the
        batch runs (a PP rank waiting on the previous stage). The watchdog must
        not fire. It used to measure the batch from the last progress before
        the idle period, so its first check during the batch fired.
        """
        for start in BATCH_STARTS:
            with self.subTest(start=start):
                fired_at = self._fire_time(
                    is_active=_has_batch(start, start + 0.9 * TIMEOUT),
                    get_counter=lambda now: 0,
                    end=start + 3 * TIMEOUT,
                )
                self.assertIsNone(fired_at)

    def test_stuck_batch_after_idle_fires_after_the_timeout(self):
        """A batch that never makes progress after an idle period still trips
        the watchdog, and only once it has run at least the timeout."""
        for start in BATCH_STARTS:
            with self.subTest(start=start):
                with self.assertLogs(watchdog.logger, "ERROR"):
                    fired_at = self._fire_time(
                        is_active=_has_batch(start),
                        get_counter=lambda now: 0,
                        end=start + 3 * TIMEOUT,
                    )
                self.assertGreaterEqual(fired_at - start, TIMEOUT)
                self.assertLess(fired_at - start, 2 * TIMEOUT)

    def test_busy_scheduler_never_fires(self):
        """A busy scheduler whose forward counter keeps moving never trips the
        watchdog."""
        fired_at = self._fire_time(
            is_active=_has_batch(0.0),
            get_counter=lambda now: int(now),
            end=IDLE,
        )
        self.assertIsNone(fired_at)


if __name__ == "__main__":
    unittest.main()
