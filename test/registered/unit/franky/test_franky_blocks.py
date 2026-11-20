import unittest

import msgspec

from sglang.srt.franky.chains.blocks.sandy_call import FrankySandyCall
from sglang.srt.franky.chains.chain import FrankyChain
from sglang.srt.franky.finish_rules.request_blocks import RequestBlocks
from sglang.srt.franky.finish_rules.tap_step import TapStep
from sglang.srt.franky.request import FrankyRequest
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_CALL_TOKEN = 99
_EOS = 2


def _chain(blocks, *, block=1):
    return FrankyChain.from_request(
        franky_chain={"id": "c", "block": block, "blocks": blocks}, rid="c-1"
    )


def _request(*, chain=None, max_new_tokens=100):
    params = SamplingParams(max_new_tokens=max_new_tokens, stop_token_ids=[_EOS])
    params.normalize(None)
    franky = None if chain is None else FrankyRequest(chain=chain)
    return Req("c-1", "", [1, 2, 3], params, franky=franky)


def _leg(blocks, *, block=1, max_new_tokens=100):
    req = _request(chain=_chain(blocks, block=block), max_new_tokens=max_new_tokens)
    RequestBlocks.attach(req)
    return req


def _write(req, tokens):
    """Write tokens one decode step at a time until the request ends; return how
    many it wrote."""
    for written, token in enumerate(tokens, start=1):
        req.output_ids.append(token)
        req.update_finish_state()
        if req.finished():
            return written
    return len(tokens)


class TestRequestBlocks(CustomTestCase):
    def test_sandy_is_called_after_every_kth_block_of_the_answer(self):
        req = _leg(
            {
                "block_size": 2,
                "decider": {"type": "call_token_or_every_blocks", "every_blocks": 3},
            },
            block=2,
        )

        written = _write(req, [10] * 20)

        self.assertEqual(written, 4)
        self.assertIsInstance(req.finished_reason, FrankySandyCall)
        self.assertEqual(
            req.finished_reason.to_json(),
            {"type": "franky_sandy_call", "next_block": 4},
        )
        self.assertEqual(req.finished_len, 4)

    def test_the_call_token_ends_its_block_calls_sandy_and_stays_in_the_output(self):
        req = _leg(
            {
                "block_size": 8,
                "call_token_id": _CALL_TOKEN,
                "decider": {"type": "call_token_or_every_blocks", "every_blocks": 100},
            }
        )

        written = _write(req, [10, 11, _CALL_TOKEN, 12])

        self.assertEqual(written, 3)
        self.assertEqual(req.finished_reason.next_block, 2)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, _CALL_TOKEN])

    def test_a_block_end_without_a_call_keeps_the_leg_writing(self):
        req = _leg(
            {
                "block_size": 2,
                "decider": {"type": "call_token_or_every_blocks", "every_blocks": 3},
            }
        )

        _write(req, [10, 11, 12, 13, 14])

        self.assertFalse(req.finished())

    def test_eos_and_the_token_limit_end_the_answer_even_at_a_block_end(self):
        at_eos = _leg({"block_size": 2})
        at_limit = _leg({"block_size": 2}, max_new_tokens=2)

        _write(at_eos, [10, _EOS])
        _write(at_limit, [10, 11])

        self.assertIsInstance(at_eos.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertIsInstance(at_limit.finished_reason, FINISH_LENGTH)

    def test_the_default_decider_calls_sandy_after_every_block(self):
        req = _leg({"block_size": 3}, block=5)

        written = _write(req, [10] * 9)

        self.assertEqual(written, 3)
        self.assertEqual(req.finished_reason.next_block, 6)

    def test_a_request_back_from_a_retraction_keeps_its_count(self):
        req = _leg({"block_size": 3})
        _write(req, [10, 11])

        RequestBlocks.attach(req)
        written = _write(req, [12, 13])

        self.assertEqual(written, 1)
        self.assertEqual(req.finished_reason.next_block, 2)

    def test_a_request_outside_chains_or_a_chain_without_blocks_has_no_rule(self):
        outside = _request()
        without_blocks = _request(chain=FrankyChain(id="c"))

        for req in (outside, without_blocks):
            RequestBlocks.attach(req)
            _write(req, [10, 11, 12])
            self.assertIsNone(req.franky_finish_rule)
            self.assertFalse(req.finished())

    def test_a_bad_blocks_config_is_refused(self):
        for blocks in (
            {"block_size": 0},
            {"block_size": 4, "call_token_id": -1},
            {"block_size": 4, "decider": {"type": "never"}},
            {
                "block_size": 4,
                "decider": {"type": "call_token_or_every_blocks", "every_blocks": 0},
            },
            {"block_size": 4, "size": 4},
        ):
            with (
                self.subTest(blocks=blocks),
                self.assertRaises(msgspec.ValidationError),
            ):
                _chain(blocks)


class TestTapStep(CustomTestCase):
    def _write_with_tap_step(self, tokens, *, max_new_tokens, clamped_to=None):
        req = _request(max_new_tokens=max_new_tokens)
        TapStep.ask_extra_token(req)
        if clamped_to is not None:
            req.sampling_params.max_new_tokens = clamped_to
        TapStep.attach(req)
        _write(req, tokens)
        return req

    def test_a_write_at_its_limit_returns_one_token_less_than_it_wrote(self):
        req = self._write_with_tap_step([10, 11, 12, 13, 14], max_new_tokens=3)

        self.assertEqual(len(req.output_ids), 4)
        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 12])
        self.assertEqual(req.finished_reason.to_json()["length"], 3)

    def test_a_write_clamped_by_sglang_returns_one_token_less_than_its_limit(self):
        req = self._write_with_tap_step([10, 11, 12], max_new_tokens=3, clamped_to=2)

        self.assertEqual(list(req.output_ids_through_stop), [10])
        self.assertEqual(req.finished_reason.to_json()["length"], 1)

    def test_a_write_without_a_limit_ends_one_token_before_sglangs_clamp(self):
        req = self._write_with_tap_step([10, 11, 12], max_new_tokens=None, clamped_to=2)

        self.assertEqual(list(req.output_ids_through_stop), [10])
        self.assertEqual(req.finished_reason.to_json()["length"], 1)

    def test_a_write_that_stops_returns_its_stop_token(self):
        req = self._write_with_tap_step([10, _EOS, 12], max_new_tokens=3)

        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(list(req.output_ids_through_stop), [10, _EOS])

    def test_a_write_that_stops_on_its_extra_token_ends_at_the_tokens_it_asked(self):
        req = self._write_with_tap_step([10, 11, 12, _EOS], max_new_tokens=3)

        self.assertEqual(list(req.output_ids_through_stop), [10, 11, 12])
        self.assertEqual(req.finished_reason.to_json()["length"], 3)

    def test_an_aborted_write_keeps_its_abort(self):
        req = _request(max_new_tokens=3)
        TapStep.ask_extra_token(req)
        TapStep.attach(req)
        req.to_finish = FINISH_ABORT("gone")

        req.update_finish_state(0)

        self.assertIsInstance(req.finished_reason, FINISH_ABORT)


if __name__ == "__main__":
    unittest.main()
