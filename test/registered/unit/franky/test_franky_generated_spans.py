import sys
import unittest

import torch

from sglang.srt.franky.cache.generated_spans import (
    FrankyGeneratedSpans,
    GeneratedSpan,
    checked_earlier_answers,
)
from sglang.srt.franky.chains.chain import FrankyChain
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ANSWERS_5_TO_10_AND_FROM_14 = (
    GeneratedSpan(start=5, sandy_start=5, end=11),
    GeneratedSpan(start=14, sandy_start=14),
)
# The same answers where Sandy tokenizes on its own: its copy of the first
# answer starts at its token 3, and its input is 12 tokens long.
_SANDY_ANSWERS_FROM_3_AND_FROM_12 = (
    GeneratedSpan(start=5, sandy_start=3, end=11),
    GeneratedSpan(start=14, sandy_start=12),
)


class TestFrankyGeneratedSpans(CustomTestCase):
    def test_an_answer_token_reads_the_taps_before_its_block_and_others_read_none(
        self,
    ):
        spans = FrankyGeneratedSpans(spans=_ANSWERS_5_TO_10_AND_FROM_14, block_size=4)

        taps = spans.taps_to_read(torch.arange(18))

        self.assertEqual(
            taps.tolist(), [0, 0, 0, 0, 0, 5, 5, 5, 5, 9, 9, 0, 0, 0, 14, 14, 14, 14]
        )

    def test_without_a_block_size_each_answer_is_one_block(self):
        spans = FrankyGeneratedSpans(spans=_ANSWERS_5_TO_10_AND_FROM_14)

        taps = spans.taps_to_read(torch.arange(18))

        self.assertEqual(taps.tolist(), [0] * 5 + [5] * 6 + [0] * 3 + [14] * 4)

    def test_an_answer_token_reads_sandys_tokens_before_its_copy_of_the_answer(self):
        spans = FrankyGeneratedSpans(spans=_SANDY_ANSWERS_FROM_3_AND_FROM_12)

        taps = spans.taps_to_read(torch.arange(18))

        self.assertEqual(taps.tolist(), [0] * 5 + [3] * 6 + [0] * 3 + [12] * 4)

    def test_rows_before_the_first_row_reading_unlanded_taps_may_run(self):
        spans = FrankyGeneratedSpans(spans=_SANDY_ANSWERS_FROM_3_AND_FROM_12)

        # Row t makes token t + 1, so row 4 makes the first answer token.
        self.assertEqual(spans.rows_reading_at_most(0), 4)
        self.assertEqual(spans.rows_reading_at_most(3), 13)
        self.assertEqual(spans.rows_reading_at_most(11), 13)
        self.assertEqual(spans.rows_reading_at_most(12), sys.maxsize)

    def test_rows_reading_at_most_stops_at_the_first_block_past_the_taps(self):
        spans = FrankyGeneratedSpans(spans=_ANSWERS_5_TO_10_AND_FROM_14, block_size=4)
        reads = spans.taps_to_read(torch.arange(1, 40))

        for num_taps in range(16):
            with self.subTest(num_taps=num_taps):
                rows = spans.rows_reading_at_most(num_taps)
                self.assertTrue(bool((reads[:rows] <= num_taps).all()))
                self.assertGreater(int(reads[rows]), num_taps)

    def test_a_chain_blocks_answer_starts_before_its_earlier_blocks(self):
        chain = FrankyChain(id="c", block=3, block_size=128)

        spans = FrankyGeneratedSpans.of(
            earlier_answers=_ANSWERS_5_TO_10_AND_FROM_14[:1],
            num_input_tokens=356,
            chain=chain,
            sandy_input_len=None,
        )

        self.assertEqual(
            spans,
            FrankyGeneratedSpans(
                spans=(
                    _ANSWERS_5_TO_10_AND_FROM_14[0],
                    GeneratedSpan(start=100, sandy_start=100),
                ),
                block_size=128,
            ),
        )

    def test_with_sandy_on_its_own_each_block_reads_sandys_tokens_before_it(self):
        # A prompt of 22 tokens and two earlier 4-token blocks; Sandy's copy of
        # the answer starts at its token 20, its blocks at 23 and 27.
        chain = FrankyChain(
            id="c", block=3, block_size=4, sandy_block_starts=(20, 23)
        )

        spans = FrankyGeneratedSpans.of(
            earlier_answers=_SANDY_ANSWERS_FROM_3_AND_FROM_12[:1],
            num_input_tokens=30,
            chain=chain,
            sandy_input_len=27,
        )

        self.assertEqual(
            spans.spans,
            (
                GeneratedSpan(start=5, sandy_start=3, end=11, sandy_block_starts=()),
                GeneratedSpan(start=22, sandy_start=20, sandy_block_starts=(23, 27)),
            ),
        )
        taps = spans.taps_to_read(torch.arange(36))
        self.assertEqual(
            taps.tolist(),
            [0] * 5 + [3] * 6 + [0] * 11 + [20] * 4 + [23] * 4 + [27] * 6,
        )

    def test_with_sandy_on_its_own_rows_stop_at_the_first_block_past_the_taps(self):
        chain = FrankyChain(
            id="c", block=4, block_size=4, sandy_block_starts=(20, 23, 23)
        )
        spans = FrankyGeneratedSpans.of(
            earlier_answers=_SANDY_ANSWERS_FROM_3_AND_FROM_12[:1],
            num_input_tokens=34,
            chain=chain,
            sandy_input_len=29,
        )
        reads = spans.taps_to_read(torch.arange(1, 60))

        for num_taps in range(29):
            with self.subTest(num_taps=num_taps):
                rows = spans.rows_reading_at_most(num_taps)
                self.assertTrue(bool((reads[:rows] <= num_taps).all()))
                self.assertGreater(int(reads[rows]), num_taps)
        self.assertEqual(spans.rows_reading_at_most(29), sys.maxsize)

    def test_without_a_chain_the_answer_starts_after_the_input(self):
        spans = FrankyGeneratedSpans.of(
            earlier_answers=(), num_input_tokens=356, chain=None, sandy_input_len=None
        )

        self.assertEqual(
            spans,
            FrankyGeneratedSpans(spans=(GeneratedSpan(start=356, sandy_start=356),)),
        )

    def test_sandys_copy_of_the_answer_starts_after_its_own_input(self):
        spans = FrankyGeneratedSpans.of(
            earlier_answers=(), num_input_tokens=356, chain=None, sandy_input_len=340
        )

        self.assertEqual(
            spans,
            FrankyGeneratedSpans(spans=(GeneratedSpan(start=356, sandy_start=340),)),
        )

    def test_earlier_answers_must_be_runs_in_order_inside_the_prompt(self):
        self.assertEqual(
            checked_earlier_answers(
                [[2, 5], [7, 9]],
                generation_start=12,
                sandy_answers=None,
                sandy_generation_start=None,
            ),
            (
                GeneratedSpan(start=2, sandy_start=2, end=5),
                GeneratedSpan(start=7, sandy_start=7, end=9),
            ),
        )
        for answers in (
            [[0, 3]],
            [[2, 5], [4, 6]],
            [[2, 2]],
            [[8, 13]],
            [[5, 3]],
            [[2, 5, 7]],
        ):
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                checked_earlier_answers(
                    answers,
                    generation_start=12,
                    sandy_answers=None,
                    sandy_generation_start=None,
                )

    def test_each_answer_has_its_run_in_sandys_tokens(self):
        self.assertEqual(
            checked_earlier_answers(
                [[2, 5], [7, 9]],
                generation_start=12,
                sandy_answers=[[3, 4], [8, 12]],
                sandy_generation_start=15,
            ),
            (
                GeneratedSpan(start=2, sandy_start=3, end=5),
                GeneratedSpan(start=7, sandy_start=8, end=9),
            ),
        )
        for sandy_answers, sandy_generation_start in (
            ([[3, 4]], 15),
            ([[3, 4], [8, 16]], 15),
            ([[3, 4], [8, 12]], None),
        ):
            with (
                self.subTest(sandy_answers=sandy_answers),
                self.assertRaises(ValueError),
            ):
                checked_earlier_answers(
                    [[2, 5], [7, 9]],
                    generation_start=12,
                    sandy_answers=sandy_answers,
                    sandy_generation_start=sandy_generation_start,
                )

    def test_a_prompt_without_answers_names_only_sandys_input_length(self):
        self.assertEqual(
            checked_earlier_answers(
                [], generation_start=12, sandy_answers=None, sandy_generation_start=9
            ),
            (),
        )


if __name__ == "__main__":
    unittest.main()
