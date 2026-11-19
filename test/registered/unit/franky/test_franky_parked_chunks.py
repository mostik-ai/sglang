"""A Ricky request that waits for taps between two of its chunks is parked. The
viewer splits a request's prefill into the time its chunks ran, the time they sat
parked until their taps landed, and the time they stayed parked after that, so a
parked request must report every second it waited, once, on the right side.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.franky.ricky_taps.prefill_tap_limit import PrefillTapLimit
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


def _chunked_req(rid: str, *, prefix_len: int, rows_ready: int, num_rows: int = 1024):
    return SimpleNamespace(
        rid=rid,
        prefix_len=prefix_len,
        franky_taps=_Taps(rows_ready),
        origin_input_ids=[0] * num_rows,
        output_ids=[],
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
        self.parked = FrankyParkedChunks(tap_limit=PrefillTapLimit(chunk_grid=1))

    def test_a_request_adds_the_time_it_sat_parked(self):
        req = _chunked_req("r", prefix_len=64, rows_ready=64)
        self.assertIsNone(self.parked.pick(current=req))

        self.clock.now = 100.75
        req.franky_taps.ready = 128
        picked = self.parked.pick(current=None)

        self.assertIs(picked, req)
        self.assertEqual(req.time_stats.franky_parked_s, 0.75)
        self.assertEqual(req.time_stats.franky_parked_landed_s, 0.0)

    def test_time_parked_after_the_taps_landed_counts_apart(self):
        parked = _chunked_req("parked", prefix_len=64, rows_ready=64)
        running = _chunked_req("running", prefix_len=64, rows_ready=1024)
        self.parked.pick(current=parked)

        self.clock.now = 100.5
        parked.franky_taps.ready = 128
        self.assertIs(self.parked.pick(current=running), running)
        self.clock.now = 101.25
        self.assertIs(self.parked.pick(current=None), parked)

        self.assertEqual(parked.time_stats.franky_parked_s, 0.5)
        self.assertEqual(parked.time_stats.franky_parked_landed_s, 0.75)

    def test_a_landed_request_counts_its_wait_behind_another_apart(self):
        first = _chunked_req("first", prefix_len=64, rows_ready=64)
        second = _chunked_req("second", prefix_len=64, rows_ready=64)
        self.parked.pick(current=first)
        self.parked.pick(current=second)

        self.clock.now = 100.5
        first.franky_taps.ready = second.franky_taps.ready = 128
        self.assertIs(self.parked.pick(current=None), first)
        self.clock.now = 101.0
        self.assertIs(self.parked.pick(current=None), second)

        self.assertEqual(first.time_stats.franky_parked_s, 0.5)
        self.assertEqual(first.time_stats.franky_parked_landed_s, 0.0)
        self.assertEqual(second.time_stats.franky_parked_s, 0.5)
        self.assertEqual(second.time_stats.franky_parked_landed_s, 0.5)

    def test_every_parking_adds_up(self):
        req = _chunked_req("r", prefix_len=64, rows_ready=64)
        for parked_seconds in (0.5, 0.25):
            self.parked.pick(current=req)
            self.clock.now += parked_seconds
            req.franky_taps.ready = req.prefix_len + 64
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
        self.assertEqual(req.time_stats.franky_parked_landed_s, 0.0)
        self.assertEqual(self.parked.parkings, {})


class TestPrefillTapLimit(CustomTestCase):
    def test_without_mamba_checkpoints_a_prefill_stops_on_any_row(self):
        readiness = PrefillTapLimit(chunk_grid=1)
        req = _chunked_req("r", prefix_len=960, rows_ready=1023)

        self.assertEqual(readiness.end(req), 1023)
        self.assertTrue(readiness.can_run(req))
        self.assertFalse(readiness.complete(req))

    def test_with_mamba_checkpoints_an_early_stop_is_a_page_end(self):
        readiness = PrefillTapLimit(chunk_grid=64)
        req = _chunked_req("r", prefix_len=960, rows_ready=1023)

        self.assertEqual(readiness.end(req), 960)
        self.assertFalse(readiness.can_run(req))
        req.franky_taps.ready = 1024
        self.assertEqual(readiness.end(req), 1024)
        self.assertTrue(readiness.can_run(req))
        self.assertTrue(readiness.complete(req))

    def test_the_grid_is_the_page_only_for_a_mamba_prefix_cache(self):
        mamba = SimpleNamespace(
            supports_mamba=lambda: True, supports_prefix_sharing=lambda: True
        )
        plain = SimpleNamespace(
            supports_mamba=lambda: False, supports_prefix_sharing=lambda: True
        )

        self.assertEqual(PrefillTapLimit.of(mamba, page_size=64).chunk_grid, 64)
        self.assertEqual(PrefillTapLimit.of(plain, page_size=64).chunk_grid, 1)

    def test_a_request_without_tap_state_cannot_run(self):
        req = _chunked_req("r", prefix_len=0, rows_ready=0)
        req.franky_taps = None

        self.assertFalse(PrefillTapLimit(chunk_grid=1).can_run(req))

    def test_a_parked_mamba_request_waits_for_its_last_rows_taps(self):
        parked = FrankyParkedChunks(tap_limit=PrefillTapLimit(chunk_grid=64))
        req = _chunked_req("r", prefix_len=960, rows_ready=1023)

        self.assertIsNone(parked.pick(current=req))
        req.franky_taps.ready = 1024
        self.assertIs(parked.pick(current=None), req)


if __name__ == "__main__":
    unittest.main()
