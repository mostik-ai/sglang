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

_ANSWERS_5_TO_10_AND_FROM_14 = (GeneratedSpan(start=5, end=11), GeneratedSpan(start=14))


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

    def test_a_chain_blocks_answer_starts_before_its_earlier_blocks(self):
        chain = FrankyChain(id="c", block=3, block_size=128)

        spans = FrankyGeneratedSpans.of(
            earlier_answers=_ANSWERS_5_TO_10_AND_FROM_14[:1],
            num_input_tokens=356,
            chain=chain,
        )

        self.assertEqual(
            spans,
            FrankyGeneratedSpans(
                spans=(_ANSWERS_5_TO_10_AND_FROM_14[0], GeneratedSpan(start=100)),
                block_size=128,
            ),
        )

    def test_without_a_chain_the_answer_starts_after_the_input(self):
        spans = FrankyGeneratedSpans.of(
            earlier_answers=(), num_input_tokens=356, chain=None
        )

        self.assertEqual(spans, FrankyGeneratedSpans(spans=(GeneratedSpan(start=356),)))

    def test_earlier_answers_must_be_runs_in_order_inside_the_prompt(self):
        self.assertEqual(
            checked_earlier_answers([[2, 5], [7, 9]], generation_start=12),
            (GeneratedSpan(start=2, end=5), GeneratedSpan(start=7, end=9)),
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
                checked_earlier_answers(answers, generation_start=12)


if __name__ == "__main__":
    unittest.main()
