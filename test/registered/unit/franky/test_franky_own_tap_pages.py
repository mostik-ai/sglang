"""When Sandy tokenizes on its own, a request's taps count Sandy's tokens, which
have no KV slots on Ricky. The request then owns tap pages for them; a reader
that looked at its KV slots instead, or a relay that sent KV page views, would
feed it another request's taps without any error.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl.conn import NixlKVReceiver, NixlKVSender
from sglang.srt.franky.cache.generated_spans import FrankyGeneratedSpans, GeneratedSpan
from sglang.srt.franky.cache.own_tap_pages import FrankyOwnTapPages
from sglang.srt.franky.cache.own_tap_reservations import FrankyOwnTapReservations
from sglang.srt.franky.cache.tap_cache import TAP_DTYPE, FrankyTapCache
from sglang.srt.franky.chains.chain import FrankyChain
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.transfer.relay import receiver as relay_receiver_module
from sglang.srt.franky.transfer.relay import sender as relay_sender_module
from sglang.srt.franky.transfer.relay.receiver import FrankyRelayReceiver
from sglang.srt.franky.transfer.relay.sender import FrankyRelaySender
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE_SIZE = 4
_NUM_TOKENS = 32
_OUTPUT_SIZE = 8
_NUM_REQUEST_ROWS = 4


def _extend_batch(*, rows, prefix_lens, extend_lens):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.tensor(rows),
        req_pool_indices_cpu=torch.tensor(rows),
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32),
        extend_prefix_lens_cpu=list(prefix_lens),
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
        extend_seq_lens_cpu=list(extend_lens),
    )


def _decode_batch(*, rows):
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE, req_pool_indices=torch.tensor(rows)
    )


class TestReadingOwnTapPages(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.cache = FrankyTapCache(
            encoder_ids=(0,),
            output_size=_OUTPUT_SIZE,
            page_size=_PAGE_SIZE,
            kv_page_size=_PAGE_SIZE,
            num_tokens=_NUM_TOKENS,
            req_to_token=torch.zeros(
                (_NUM_REQUEST_ROWS, _NUM_TOKENS), dtype=torch.int32
            ),
            device="cpu",
            own_pages=True,
        )
        self.own_pages = self.cache.own_pages
        # Ricky's 8 prompt tokens hold an answer at 3..4 whose copy starts at
        # Sandy's token 2; Sandy's input is 6 tokens, two pages of taps.
        self.pages = self.own_pages.take(num_pages=2)
        self.own_pages.bind(req_pool_indices=[1], pages=[self.pages])
        self.cache.set_generated_spans(
            req_pool_indices=[1],
            generated_spans=[
                FrankyGeneratedSpans(
                    spans=(
                        GeneratedSpan(start=3, sandy_start=2, end=5),
                        GeneratedSpan(start=8, sandy_start=6),
                    )
                )
            ],
            num_input_tokens=[8],
        )

    def test_pages_past_the_zero_page_are_taken_and_given_back(self):
        free = self.own_pages.num_free()

        self.assertNotIn(0, self.pages.tolist())
        self.assertIsNone(self.own_pages.take(num_pages=free + 1))
        self.own_pages.free(self.pages)
        self.assertEqual(self.own_pages.num_free(), _NUM_TOKENS // _PAGE_SIZE)

    def test_a_prefill_row_reads_its_owned_pages_up_to_sandys_answer_start(self):
        view = self.cache.view(
            encoder_id=0,
            forward_batch=_extend_batch(rows=[1], prefix_lens=[0], extend_lens=[9]),
        )

        # Row t makes token t + 1: rows 2 and 3 make the earlier answer's tokens,
        # rows 7 and 8 the new answer's.
        self.assertEqual(view.taps_to_read.tolist(), [0, 0, 2, 2, 0, 0, 0, 6, 6])
        self.assertEqual(view.page_table.tolist(), [self.pages.tolist()] * 9)

    def test_decode_reads_all_of_sandys_input_from_the_owned_pages(self):
        view = self.cache.view(encoder_id=0, forward_batch=_decode_batch(rows=[1]))

        self.assertEqual(view.taps_to_read.tolist(), [6])
        self.assertEqual(view.page_table[0, :2].tolist(), self.pages.tolist())

    def test_each_forward_builds_its_plan_even_on_the_same_tensors(self):
        """A CUDA graph runner warms a batch up and captures it on the same
        tensors: the captured forward must build its own plan."""
        batch = _decode_batch(rows=[1])
        first = self.cache.view(encoder_id=0, forward_batch=batch)
        moved = self.own_pages.take(num_pages=2)
        self.own_pages.bind(req_pool_indices=[1], pages=[moved])

        second = self.cache.view(encoder_id=0, forward_batch=batch)

        self.assertEqual(first.page_table[0, :2].tolist(), self.pages.tolist())
        self.assertEqual(second.page_table[0, :2].tolist(), moved.tolist())

    def test_an_owned_page_is_the_bytes_the_transfer_writes(self):
        page = int(self.pages[1])
        start = page * self.cache.page_bytes // TAP_DTYPE.itemsize
        flat = self.cache._buffers[0].view(-1)
        flat[start : start + _PAGE_SIZE * _OUTPUT_SIZE] = 7

        view = self.cache.view(encoder_id=0, forward_batch=_decode_batch(rows=[1]))

        self.assertTrue(bool((view.taps[page] == 7).all()))


def _request(room, *, sandy_input_len, chain=None):
    request = SimpleNamespace(
        bootstrap_room=room,
        franky_sandy_input_len=sandy_input_len,
        franky_chain=chain,
        kv=SimpleNamespace(holds_kv=False),
        done=False,
        finished_reason=None,
    )
    request.finished = lambda: request.done
    return request


def _decode_holds(own_pages):
    return FrankyChainHolds(
        tree_cache=None, max_held_tokens=0, hold_seconds=60.0, own_pages=own_pages
    )


def _reservations(own_pages, *, holds):
    return FrankyOwnTapReservations(
        own_pages=own_pages, page_size=_PAGE_SIZE, chain_holds=holds
    )


class TestOwnTapReservations(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.own_pages = FrankyOwnTapPages(
            first_page=1,
            num_pages=13,
            max_pages_per_request=6,
            num_request_rows=2,
            device="cpu",
        )
        self.holds = _decode_holds(self.own_pages)
        self.reservations = _reservations(self.own_pages, holds=self.holds)

    def _expect(self, room, *, sandy_input_len, chain=None):
        request = _request(room, sandy_input_len=sandy_input_len, chain=chain)
        self.reservations.expect(request)
        return request

    def _block(self, block, *, room, sandy_input_len, starts=()):
        chain = FrankyChain(
            id="c", block=block, block_size=8, sandy_block_starts=starts
        )
        return self._expect(room, sandy_input_len=sandy_input_len, chain=chain)

    def _end(self, request, reason):
        request.done = True
        request.finished_reason = reason
        self.reservations.poll()

    def test_requests_take_pages_in_arrival_order(self):
        self._expect(1, sandy_input_len=24)
        self._expect(2, sandy_input_len=21)
        self._expect(3, sandy_input_len=1)

        self.reservations.poll()

        self.assertEqual(len(self.reservations.pages_of(room=1)), 6)
        self.assertEqual(len(self.reservations.pages_of(room=2)), 6)
        self.assertIsNone(self.reservations.pages_of(room=3))
        self.assertTrue(self.reservations.waits(room=3))

    def test_pages_come_back_once_the_request_ended_and_released_its_kv(self):
        first = self._expect(1, sandy_input_len=24)
        self._expect(2, sandy_input_len=24)
        self._expect(3, sandy_input_len=4)
        self.reservations.poll()
        first.done = True
        first.kv.holds_kv = True

        self.reservations.poll()

        self.assertTrue(self.reservations.waits(room=3))
        first.kv.holds_kv = False
        self.reservations.poll()
        self.assertIsNone(self.reservations.pages_of(room=1))
        self.assertIsNotNone(self.reservations.pages_of(room=3))

    def test_a_block_at_its_length_leaves_its_pages_to_the_next_block(self):
        first = self._block(1, room=1, sandy_input_len=13)
        self.reservations.poll()
        first_pages = self.reservations.pages_of(room=1).tolist()
        self._end(first, FINISH_LENGTH(length=8))
        self.assertEqual(self.own_pages.num_free(), 8)

        self._block(2, room=2, sandy_input_len=22, starts=(13,))
        self.reservations.poll()

        pages = self.reservations.pages_of(room=2).tolist()
        self.assertEqual(pages[:4], first_pages)
        self.assertEqual(len(pages), 6)
        self.assertEqual(self.own_pages.num_free(), 6)
        self.assertEqual(
            self.reservations.taps_held_of(room=2), 12, "3 whole pages of 13 taps"
        )

    def test_a_block_that_stopped_or_waited_in_vain_gives_its_pages_back(self):
        stopped = self._block(1, room=1, sandy_input_len=13)
        self.reservations.poll()
        self._end(stopped, FINISH_MATCHED_TOKEN(matched=0))
        self.assertEqual(self.own_pages.num_free(), 12)

        first = self._block(1, room=2, sandy_input_len=13)
        self.reservations.poll()
        self._end(first, FINISH_LENGTH(length=8))
        aborted = self._block(2, room=3, sandy_input_len=22, starts=(13,))
        self._end(aborted, FINISH_ABORT("Aborted"))

        self.assertEqual(self.own_pages.num_free(), 12)

    def test_a_request_short_of_pages_takes_the_newest_held_ones(self):
        for room, chain_id in ((1, "a"), (2, "b")):
            block = self._expect(
                room,
                sandy_input_len=24,
                chain=FrankyChain(id=chain_id, block_size=8),
            )
            self.reservations.poll()
            self._end(block, FINISH_LENGTH(length=8))

        self._expect(3, sandy_input_len=24)
        self.reservations.poll()

        self.assertIsNotNone(self.reservations.pages_of(room=3))
        self.assertEqual(self.holds.held_tap_pages, 6)
        self.assertIsNone(self.holds.take_taps(chain_id="b"))
        self.assertIsNotNone(self.holds.take_taps(chain_id="a"))

    def test_a_request_past_the_pages_one_may_own_is_too_large(self):
        too_large = self._expect(1, sandy_input_len=25)
        self._expect(2, sandy_input_len=4)

        self.reservations.poll()

        self.assertTrue(self.reservations.too_large(room=1))
        self.assertTrue(self.reservations.waits(room=2))
        too_large.done = True
        self.reservations.poll()
        self.assertFalse(self.reservations.waits(room=1))
        self.assertIsNotNone(self.reservations.pages_of(room=2))


class TestRelayOfOwnTapPages(CustomTestCase):
    def _sender(self):
        sender = FrankyRelaySender.__new__(FrankyRelaySender)
        sender.kv_mgr = SimpleNamespace(
            tap_cache=SimpleNamespace(own_pages=object(), pages_per_kv_page=4),
            tap_relay=object(),
            relay_bootstrap_port=9000,
        )
        sender.bootstrap_server_url = "prefill:8998"
        sender.bootstrap_room = 7
        sender.tap_sender = None
        return sender

    def test_prefill_relays_the_owned_pages_once_and_no_kv_page_views(self):
        sender = self._sender()
        with (
            patch.object(NixlKVSender, "init"),
            patch.object(NixlKVSender, "send"),
            patch.object(relay_sender_module, "FrankyRoomSender") as room_sender,
        ):
            sender.init(3)
            sender.send(np.array([1, 2, 3]))
            room_sender.assert_not_called()

            sender.relay_own_taps(tap_pages=np.array([5, 6], dtype=np.int32))
            sender.relay_own_taps(tap_pages=np.array([5, 6], dtype=np.int32))

        room_sender.assert_called_once()
        self.assertEqual(room_sender.call_args.kwargs["num_pages"], 2)
        [land] = room_sender.return_value.land.call_args_list
        self.assertEqual(land.kwargs["tap_pages"].tolist(), [5, 6])

    def _receiver(self, reservations):
        receiver = FrankyRelayReceiver.__new__(FrankyRelayReceiver)
        receiver.kv_mgr = SimpleNamespace(
            own_taps=reservations,
            tap_cache=SimpleNamespace(page_size=_PAGE_SIZE),
            tap_relay=object(),
            relay_bootstrap_port=9000,
            track_tap_receiver=MagicMock(),
            record_failure=MagicMock(),
        )
        receiver.bootstrap_room = 7
        receiver.bootstrap_addr = "prefill:8998"
        receiver.conclude_state = None
        receiver.tap_receiver = None
        return receiver

    def _own_pages(self, *, max_pages_per_request):
        return FrankyOwnTapPages(
            first_page=1,
            num_pages=9,
            max_pages_per_request=max_pages_per_request,
            num_request_rows=2,
            device="cpu",
        )

    def test_decode_registers_the_pages_its_request_reserved(self):
        own_pages = self._own_pages(max_pages_per_request=4)
        reservations = _reservations(own_pages, holds=_decode_holds(own_pages))
        reservations.expect(_request(7, sandy_input_len=6))
        receiver = self._receiver(reservations)

        with patch.object(
            NixlKVReceiver, "poll", return_value=KVPoll.WaitingForInput
        ) as base_poll:
            self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
            base_poll.assert_not_called()
            reservations.poll()
            self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)

        with (
            patch.object(NixlKVReceiver, "send_metadata"),
            patch.object(relay_receiver_module, "FrankyReceiver") as tap_receiver,
        ):
            receiver.send_metadata(np.array([3, 4, 5]), 0)

        tap_pages = tap_receiver.call_args.kwargs["tap_pages"]
        self.assertEqual(tap_pages.tolist(), reservations.pages_of(room=7).tolist())
        self.assertEqual(tap_receiver.call_args.kwargs["num_tokens"], 2 * _PAGE_SIZE)
        self.assertEqual(tap_receiver.call_args.kwargs["prefix_len"], 0)

    def test_decode_receives_only_the_pages_past_its_chains_whole_held_pages(self):
        own_pages = self._own_pages(max_pages_per_request=6)
        reservations = _reservations(own_pages, holds=_decode_holds(own_pages))
        first = _request(6, sandy_input_len=13, chain=FrankyChain(id="c", block_size=8))
        reservations.expect(first)
        reservations.poll()
        first.done = True
        first.finished_reason = FINISH_LENGTH(length=8)
        reservations.poll()
        second_chain = FrankyChain(
            id="c", block=2, block_size=8, sandy_block_starts=(13,)
        )
        reservations.expect(_request(7, sandy_input_len=22, chain=second_chain))
        reservations.poll()
        receiver = self._receiver(reservations)

        with (
            patch.object(NixlKVReceiver, "send_metadata"),
            patch.object(relay_receiver_module, "FrankyReceiver") as tap_receiver,
        ):
            receiver.send_metadata(np.array([3, 4, 5]), 0)

        pages = reservations.pages_of(room=7).tolist()
        kwargs = tap_receiver.call_args.kwargs
        self.assertEqual(kwargs["tap_pages"].tolist(), pages[3:])
        self.assertEqual(kwargs["prefix_len"], 3 * _PAGE_SIZE)
        self.assertEqual(kwargs["num_tokens"], 6 * _PAGE_SIZE)

    def test_decode_fails_a_request_with_more_taps_than_one_may_own(self):
        own_pages = self._own_pages(max_pages_per_request=1)
        reservations = _reservations(own_pages, holds=_decode_holds(own_pages))
        reservations.expect(_request(7, sandy_input_len=6))
        receiver = self._receiver(reservations)

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        receiver.kv_mgr.record_failure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
