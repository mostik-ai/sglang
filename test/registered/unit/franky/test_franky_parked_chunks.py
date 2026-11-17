"""A Ricky request that waits for taps between two of its chunks is parked. The
viewer splits a request's prefill into the time its chunks ran and the time they
sat parked, so a parked request must report every second it waited, once.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.franky.schedulers import parked_chunks as parked_chunks_module
from sglang.srt.franky.schedulers.parked_chunks import FrankyParkedChunks
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Taps:
    def __init__(self, rows_ready: int) -> None:
        self.ready = rows_ready

    def rows_ready(self) -> int:
        return self.ready


def _chunked_req(rid: str, *, prefix_len: int, rows_ready: int):
    return SimpleNamespace(
        rid=rid,
        prefix_len=prefix_len,
        franky=_Taps(rows_ready),
        time_stats=SchedulerReqTimeStats(),
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def perf_counter(self) -> float:
        return self.now


class TestParkedSeconds(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.clock = _Clock()
        clock_patch = patch.object(parked_chunks_module, "time", self.clock)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.parked = FrankyParkedChunks()

    def test_a_request_adds_the_time_it_sat_parked(self):
        req = _chunked_req("r", prefix_len=64, rows_ready=64)
        self.assertIsNone(self.parked.pick(current=req))

        self.clock.now = 100.75
        req.franky.ready = 128
        picked = self.parked.pick(current=None)

        self.assertIs(picked, req)
        self.assertEqual(req.time_stats.franky_parked_s, 0.75)

    def test_every_parking_adds_up(self):
        req = _chunked_req("r", prefix_len=64, rows_ready=64)
        for parked_seconds in (0.5, 0.25):
            self.parked.pick(current=req)
            self.clock.now += parked_seconds
            req.franky.ready = req.prefix_len + 64
            self.parked.pick(current=None)
            req.prefix_len += 64
        self.assertEqual(req.time_stats.franky_parked_s, 0.75)

    def test_a_request_whose_taps_landed_never_parks(self):
        req = _chunked_req("r", prefix_len=64, rows_ready=128)
        self.assertIs(self.parked.pick(current=req), req)
        self.assertEqual(req.time_stats.franky_parked_s, 0.0)

    def test_an_aborted_request_adds_nothing(self):
        req = _chunked_req("r", prefix_len=64, rows_ready=64)
        self.parked.pick(current=req)
        self.clock.now = 101.0
        self.assertTrue(self.parked.take(req))
        self.assertEqual(req.time_stats.franky_parked_s, 0.0)
        self.assertEqual(self.parked.parked_since, {})


if __name__ == "__main__":
    unittest.main()
