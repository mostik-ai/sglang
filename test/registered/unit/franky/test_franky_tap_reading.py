import unittest

from sglang.srt.franky.cache.tap_reading import (
    FrankyTapReading,
    TapRun,
    check_earlier_answers,
)
from sglang.srt.franky.chains.chain import FrankyChain
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Block 3 of a chain whose prompt is 100 tokens, in blocks of 128: block 1 made
# tokens 100..227 and block 2 tokens 228..355.
_THIRD_BLOCK = FrankyTapReading(tap_len=356, generation_start=100, block_size=128)


def _causal(num_tokens, end):
    return TapRun(num_tokens=num_tokens, causal_len=end, full_len=0)


def _full(num_tokens, limit):
    return TapRun(num_tokens=num_tokens, causal_len=0, full_len=limit)


class TestFrankyTapReading(CustomTestCase):
    def test_a_one_block_request_reads_up_to_each_token_then_all_its_taps(self):
        reading = FrankyTapReading(tap_len=200, generation_start=200)

        runs = reading.runs(first_position=120, num_tokens=100)

        self.assertEqual(runs, [_causal(80, 200), _full(20, 200)])

    def test_a_later_block_rereads_each_token_as_its_decode_did(self):
        """Tokens 192..226 were made from taps 0..99, tokens 227..354 from taps
        0..227, and token 355 makes the next block's first token from all taps."""
        runs = _THIRD_BLOCK.runs(first_position=192, num_tokens=164)

        self.assertEqual(runs, [_full(35, 100), _full(128, 228), _full(1, 356)])

    def test_a_recompute_from_inside_the_prompt_crosses_every_block(self):
        runs = _THIRD_BLOCK.runs(first_position=64, num_tokens=292)

        self.assertEqual(
            runs,
            [_causal(36, 100), _full(127, 100), _full(128, 228), _full(1, 356)],
        )

    def test_tokens_past_the_taps_read_all_of_them(self):
        runs = _THIRD_BLOCK.runs(first_position=356, num_tokens=10)

        self.assertEqual(runs, [_full(10, 356)])

    def test_a_request_without_taps_reads_none(self):
        reading = FrankyTapReading.of(
            tap_len=0, chain=FrankyChain(id="c", block=2, block_size=8)
        )

        self.assertEqual(reading.runs(first_position=0, num_tokens=5), [_full(5, 0)])

    def test_a_chain_blocks_reading_starts_after_its_earlier_blocks(self):
        chain = FrankyChain(id="c", block=3, block_size=128)

        self.assertEqual(FrankyTapReading.of(tap_len=356, chain=chain), _THIRD_BLOCK)
        self.assertEqual(
            FrankyTapReading.of(tap_len=356, chain=None),
            FrankyTapReading(tap_len=356, generation_start=356),
        )

    def test_a_later_turn_rereads_an_earlier_answer_as_its_decode_did(self):
        """Tokens 11..19 of the earlier answer were made from taps 0..9; the
        answer's last token and the tool output after it read up to themselves."""
        reading = FrankyTapReading.of(
            tap_len=30, chain=None, earlier_answers=((10, 20),)
        )

        runs = reading.runs(first_position=0, num_tokens=40)

        self.assertEqual(
            runs, [_causal(10, 10), _full(9, 10), _causal(11, 30), _full(10, 30)]
        )

    def test_each_earlier_answer_keeps_the_blocks_it_was_cut_into(self):
        chain = FrankyChain(id="c", block=2, block_size=8)
        reading = FrankyTapReading.of(
            tap_len=50, chain=chain, earlier_answers=((10, 30),)
        )

        runs = reading.runs(first_position=0, num_tokens=50)

        self.assertEqual(
            runs,
            [
                _causal(10, 10),
                _full(7, 10),
                _full(8, 18),
                _full(4, 26),
                _causal(13, 42),
                _full(7, 42),
                _full(1, 50),
            ],
        )

    def test_earlier_answers_must_be_runs_in_order_inside_the_prompt(self):
        self.assertEqual(
            check_earlier_answers([[2, 5], [7, 9]], generation_start=12),
            ((2, 5), (7, 9)),
        )
        for answers in ([[0, 3]], [[2, 5], [4, 6]], [[2, 2]], [[8, 13]], [[5, 3]]):
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                check_earlier_answers(answers, generation_start=12)


if __name__ == "__main__":
    unittest.main()
