import sys
import unittest

import numpy as np

from sglang.srt.franky.tap_runs import FrankyTapRuns
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# An earlier answer at tokens 5 to 10 reads Sandy's 5 tokens before it; the
# answer written now starts at token 14, where Sandy's first call wrote nothing,
# and Sandy's second call wrote tokens 18 to 21.
_RUNS = [[5, 5], [11, 0], [14, 14], [18, 22]]


def _spans(runs=_RUNS, *, num_input_tokens=22, num_taps=22):
    return FrankyTapRuns.of_runs(
        runs, num_input_tokens=num_input_tokens, num_taps=num_taps
    )


class TestFrankyTapRuns(CustomTestCase):
    def test_a_token_reads_the_taps_of_its_run_and_tokens_before_any_none(self):
        taps = _spans().taps_to_read(np.arange(26))

        self.assertEqual(
            taps.tolist(), [0] * 5 + [5] * 6 + [0] * 3 + [14] * 4 + [22] * 8
        )

    def test_an_answer_after_the_input_reads_all_of_its_taps(self):
        spans = FrankyTapRuns.after_input(num_input_tokens=356, num_taps=340)

        self.assertEqual(spans, FrankyTapRuns(starts=(356,), taps=(340,)))
        self.assertEqual(
            spans.taps_to_read(np.array([355, 356, 400])).tolist(), [0, 340, 340]
        )

    def test_rows_before_the_first_row_reading_unlanded_taps_may_run(self):
        spans = _spans()

        # Row t makes token t + 1, so row 4 makes the earlier answer's first token.
        self.assertEqual(spans.rows_reading_at_most(0), 4)
        self.assertEqual(spans.rows_reading_at_most(5), 13)
        self.assertEqual(spans.rows_reading_at_most(14), 17)
        self.assertEqual(spans.rows_reading_at_most(21), 17)
        self.assertEqual(spans.rows_reading_at_most(22), sys.maxsize)

    def test_rows_reading_at_most_stops_at_the_first_row_past_the_taps(self):
        spans = _spans()
        reads = spans.taps_to_read(np.arange(1, 40))

        for num_taps in range(22):
            with self.subTest(num_taps=num_taps):
                rows = spans.rows_reading_at_most(num_taps)
                self.assertTrue(bool((reads[:rows] <= num_taps).all()))
                self.assertGreater(int(reads[rows]), num_taps)

    def test_runs_go_up_inside_the_input_and_end_reading_all_taps(self):
        for runs in (
            [],
            [[0, 22]],
            [[5, 5, 1], [14, 22]],
            [[14, 14], [14, 22]],
            [[18, 22], [14, 14]],
            [[23, 22]],
            [[14, 23]],
            [[14, -1], [18, 22]],
            [[14, 14]],
        ):
            with (
                self.subTest(runs=runs),
                self.assertRaisesRegex(ValueError, "franky_tap_runs"),
            ):
                _spans(runs)


if __name__ == "__main__":
    unittest.main()
