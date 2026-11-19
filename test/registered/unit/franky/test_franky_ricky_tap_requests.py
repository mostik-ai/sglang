"""Ricky's requests take tap stripes in arrival order and give them back at their
end, the same on every rank. A request that passed another could starve it; a
page taken twice would mix two requests' taps.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl.conn import NixlKVReceiver, NixlKVSender
from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
from sglang.srt.franky.cache.tap_buffers import TapBuffers
from sglang.srt.franky.cache.tap_request_table import TapRequestTable
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.franky.chains.chain import FrankyChain
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.request import FrankyRequest
from sglang.srt.franky.ricky_taps.tap_requests import RickyTapRequests
from sglang.srt.franky.transfer import receiver as receiver_module
from sglang.srt.franky.transfer.messages import FrankyRoom
from sglang.srt.franky.transfer.relay import sender as relay_sender_module
from sglang.srt.franky.transfer.relay.receiver import FrankyRelayReceiver
from sglang.srt.franky.transfer.relay.sender import FrankyRelaySender
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, FINISH_MATCHED_TOKEN
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PAGE_SIZE = 4
_STRIPE_SIZE = 2 * _PAGE_SIZE
_NUM_STRIPES = 8
_MAX_STRIPES = 8
_MAX_TAPS = _MAX_STRIPES * _STRIPE_SIZE
_NUM_ROWS = 4
_ROOM = FrankyRoom(bootstrap_host="sandy", bootstrap_port=8996, room=7)


def _tap_cache(*, rank: int = 0) -> RickyTapCache:
    return RickyTapCache(
        buffers=TapBuffers(
            encoder_ids=(0,),
            output_size=8,
            page_size=_PAGE_SIZE,
            num_pages=_NUM_STRIPES + 1,
            device="cpu",
        ),
        stripes=TapStripes(num_ranks=2, rank=rank),
    )


def _tap_requests(tap_cache, *, shared_tokenizer=True):
    return RickyTapRequests(
        tap_cache=tap_cache,
        request_table=TapRequestTable(
            num_slots=_NUM_ROWS, max_stripes=_MAX_STRIPES, device="cpu"
        ),
        chain_holds=FrankyChainHolds(
            tree_cache=None, tap_cache=tap_cache, hold_seconds=60.0
        ),
        shared_tokenizer=shared_tokenizer,
    )


def _request(
    rid,
    input_ids,
    *,
    chain=None,
    room=_ROOM,
    sandy_input_ids=None,
    bootstrap_room=None,
    row=0,
):
    """A request whose input is ``input_ids``, which Sandy reads too unless the
    request names ``sandy_input_ids``.
    """
    request = SimpleNamespace(
        rid=rid,
        franky_taps=None,
        franky=(
            None
            if room is None and chain is None and sandy_input_ids is None
            else FrankyRequest(room=room, chain=chain, sandy_input_ids=sandy_input_ids)
        ),
        origin_input_ids=list(input_ids),
        bootstrap_room=bootstrap_room,
        extra_key=None,
        last_node=f"node-{rid}",
        kv=SimpleNamespace(req_pool_idx=row),
        finished_reason=None,
        done=False,
    )
    request.finished = lambda: request.done
    return request


def _block(chain_id, input_ids, *, block):
    return _request(
        f"{chain_id}-{block}",
        input_ids,
        chain=FrankyChain(id=chain_id, block=block, block_size=8),
    )


def _finish(request, reason=FINISH_MATCHED_TOKEN(matched=0)):
    request.done = True
    request.finished_reason = reason


class _Requests(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tap_cache = _tap_cache()
        self.requests = _tap_requests(self.tap_cache)
        self.holds = self.requests._chain_holds

    def _arrive(self, request):
        self.assertIsNone(self.requests.arrive(request))
        return request

    def _reserve(self, request):
        self._arrive(request)
        self.requests.poll()
        return self.requests.reservation(request.rid)

    def _end_landed(self, request, reason=FINISH_MATCHED_TOKEN(matched=0)):
        _finish(request, reason)
        self.requests.end(request, landed=True)

    def _hold(self, request):
        """A block that ended at its length, with all its taps landed."""
        self._reserve(request)
        self._end_landed(request, FINISH_LENGTH(length=8))

    def _free_and_evictable(self):
        usage = self.tap_cache.usage()
        return usage.free_pages, usage.evictable_pages


class TestRickyTapRequests(_Requests):
    def test_requests_take_stripes_in_arrival_order_and_wait_when_full(self):
        first = self._arrive(_request("a", range(40)))
        second = self._arrive(_request("b", range(100, 124)))
        self._arrive(_request("c", range(200, 208)))

        reserved = self.requests.poll()

        self.assertEqual(reserved, [first, second])
        self.assertEqual(len(self.requests.reservation("a").pages), 5)
        self.assertEqual(len(self.requests.reservation("b").pages), 3)
        self.assertIsNone(self.requests.reservation("c"))
        self.assertTrue(self.requests.waits("c"))

    def test_a_later_request_never_passes_a_waiting_one(self):
        self._reserve(_request("a", range(40)))
        self._arrive(_request("b", range(100, 132)))
        self._arrive(_request("c", range(200, 208)))

        self.assertEqual(self.requests.poll(), [])
        self.assertTrue(self.requests.waits("c"))

    def test_a_request_that_ends_before_its_taps_land_gives_its_stripes_back(self):
        first = _request("a", range(64))
        self._reserve(first)
        second = self._arrive(_request("b", range(100, 108)))
        self.assertEqual(self.requests.poll(), [])

        self.requests.end(first, landed=False)

        self.assertEqual(self.tap_cache.num_takeable_stripes(), _NUM_STRIPES)
        self.assertEqual(self.tap_cache.match(list(range(64))).num_taps, 0)
        self.assertEqual(self.requests.poll(), [second])

    def test_a_landed_request_leaves_its_whole_stripes_for_later_requests(self):
        first = _request("a", range(20))
        first_reservation = self._reserve(first)
        self._end_landed(first)

        second = self._reserve(_request("b", range(30)))

        self.assertEqual(second.cached.num_taps, 16)
        self.assertEqual(
            second.pages[:2].tolist(), first_reservation.pages[:2].tolist()
        )
        self.assertEqual(len(second.pages), 4)

    def test_at_least_one_page_of_a_cached_request_lands_again(self):
        first = _request("a", range(16))
        self._reserve(first)
        self._end_landed(first)

        again = self._reserve(_request("b", range(16)))

        self.assertEqual(again.cached.num_taps, 8)
        self.assertEqual(len(again.pages), 2)

    def test_a_request_that_ends_before_its_length_leaves_its_stripes_evictable(
        self,
    ):
        request = _request("a", range(24))
        self._reserve(request)

        self._end_landed(request)

        self.assertEqual(self.tap_cache.num_takeable_stripes(), _NUM_STRIPES)
        self.assertEqual(self._free_and_evictable(), (10, 6))
        self.assertTrue(self.requests.is_empty())

    def test_a_request_past_the_taps_one_may_read_is_refused(self):
        too_many = _request("a", range(_MAX_TAPS + 1))
        problem = self.requests.arrive(too_many)
        fits = self._arrive(_request("b", range(8)))

        self.assertIn("past the 64 taps", problem)
        self.assertEqual(self.requests.poll(), [fits])
        self.assertEqual(self.requests.refusal("a"), problem)
        self.assertFalse(self.requests.waits("a"))
        self.assertIsNone(self.requests.reservation("a"))
        self.assertIsNone(too_many.franky_taps)
        too_many.done = True
        self.requests.poll()
        self.assertIsNone(self.requests.refusal("a"))

    def test_a_request_names_sandys_tokens_exactly_when_sandy_tokenizes_alone(self):
        own = _tap_requests(self.tap_cache, shared_tokenizer=False)
        for requests, request, problem in (
            (
                self.requests,
                _request("shared", range(16), sandy_input_ids=[1, 2]),
                "sends franky_sandy_input_ids",
            ),
            (own, _request("own", range(16)), "needs franky_sandy_input_ids"),
        ):
            with self.subTest(request.rid):
                self.assertIn(problem, requests.arrive(request))
                self.assertIsNone(request.franky_taps)
                self.assertEqual(requests.poll(), [])

    def test_with_sandys_own_tokens_the_stripes_count_sandys_tokens(self):
        self.requests = _tap_requests(self.tap_cache, shared_tokenizer=False)
        sandy_ids = list(range(100, 120))

        reservation = self._reserve(_request("r", range(70), sandy_input_ids=sandy_ids))

        self.assertEqual(reservation.num_taps, 20)
        self.assertEqual(len(reservation.pages), 3)

    def test_a_request_without_a_room_reads_no_taps_and_caches_apart(self):
        plain = _request("plain", range(16), room=None)
        plain.extra_key = "user"
        bare = _request("bare", range(16), room=None)

        for request in (plain, bare):
            self._arrive(request)

        self.assertEqual(plain.extra_key, "user|franky=no-taps")
        self.assertEqual(bare.extra_key, "franky=no-taps")
        self.assertEqual(
            (plain.franky_taps.tap_len, plain.franky_taps.spans), (0, None)
        )
        self.assertEqual(self.requests.poll(), [])
        self.assertTrue(self.requests.is_empty())

    def test_a_returning_request_keeps_what_it_had(self):
        request = _request("r", range(20))
        reservation = self._reserve(request)
        state = request.franky_taps

        self._arrive(request)

        self.assertIs(request.franky_taps, state)
        self.assertEqual(self.requests.poll(), [])
        self.assertIs(self.requests.reservation("r"), reservation)

    def test_a_waiting_request_that_ended_is_dropped(self):
        first = _request("a", range(64))
        self._reserve(first)
        waiting = self._arrive(_request("b", range(100, 108)))

        waiting.done = True
        self.requests.poll()

        self.assertFalse(self.requests.waits("b"))
        self.requests.end(first, landed=False)
        self.assertEqual(self.requests.poll(), [])
        self.assertTrue(self.requests.is_empty())

    def test_forget_drops_a_waiting_or_refused_request(self):
        first = _request("a", range(64))
        self._reserve(first)
        waiting = self._arrive(_request("b", range(8)))
        refused = _request("c", range(_MAX_TAPS + 1))
        self.requests.arrive(refused)

        self.requests.forget(waiting)
        self.requests.forget(refused)

        self.assertFalse(self.requests.waits("b"))
        self.assertIsNone(self.requests.refusal("c"))
        self.requests.end(first, landed=False)
        self.assertTrue(self.requests.is_empty())

    def test_every_rank_reserves_the_same_pages(self):
        pages = []
        for rank in range(2):
            requests = _tap_requests(_tap_cache(rank=rank))
            first = _request("a", range(20))
            requests.arrive(first)
            requests.poll()
            _finish(first)
            requests.end(first, landed=True)
            requests.arrive(_request("b", range(30)))
            requests.poll()
            pages.append(requests.reservation("b").pages.tolist())

        self.assertEqual(pages[0], pages[1])

    def test_a_pd_room_names_its_request_until_the_request_ends(self):
        request = _request("r", range(20), bootstrap_room=7)
        self._reserve(request)

        self.assertEqual(self.requests.rid_of_room(7), "r")
        self.assertIsNone(self.requests.rid_of_room(8))
        self._end_landed(request)
        self.assertIsNone(self.requests.rid_of_room(7))

    def test_a_batch_binds_the_rows_of_its_reserved_requests(self):
        reading = _request("reading", range(20), row=2)
        reservation = self._reserve(reading)
        plain = self._arrive(_request("plain", range(8), room=None, row=1))
        request_table = self.requests._request_table

        self.requests.join_batch([reading, plain])

        self.assertEqual(
            request_table.stripe_pages[2, :3].tolist(), reservation.pages.tolist()
        )
        self.assertEqual(request_table.stripe_pages[1].tolist(), [0] * _MAX_STRIPES)
        self.assertEqual(request_table.decode_taps.tolist(), [0, 0, 20, 0])
        self.assertIs(request_table.spans_of(2), reading.franky_taps.spans)
        self.assertIsNone(request_table.spans_of(1))
        self.assertEqual(reservation.num_taps, reading.franky_taps.tap_len)


class TestHeldChains(_Requests):
    def test_a_held_block_keeps_its_stripes_locked_for_the_next_block(self):
        self._hold(_block("c", range(16), block=1))

        self.assertEqual(self.tap_cache.num_takeable_stripes(), _NUM_STRIPES - 2)
        second = self._reserve(_block("c", range(24), block=2))

        self.assertEqual(second.cached.num_taps, 16)
        self.assertEqual(len(second.pages), 3)

    def test_a_block_lets_its_chains_hold_go_once_it_joins_a_batch(self):
        self._hold(_block("c", range(16), block=1))
        second = _block("c", range(24), block=2)
        self._reserve(second)
        self.assertTrue(self.holds.holds_chain_of(second))

        self.requests.join_batch([second])

        self.assertFalse(self.holds.holds_chain_of(second))
        self.assertEqual(self.tap_cache.num_takeable_stripes(), _NUM_STRIPES - 3)

    def test_a_last_block_or_one_that_stopped_releases_its_chains_hold(self):
        for name, first_id, last_block, reason in (
            ("last", 0, True, FINISH_LENGTH(length=8)),
            ("stopped", 1000, False, FINISH_MATCHED_TOKEN(matched=0)),
        ):
            with self.subTest(name):
                self._hold(_block(name, range(first_id, first_id + 16), block=1))
                second = _request(
                    f"{name}-2",
                    range(first_id, first_id + 24),
                    chain=FrankyChain(
                        id=name, block=2, block_size=8, last_block=last_block
                    ),
                )
                self._reserve(second)

                self._end_landed(second, reason)

                self.assertFalse(self.holds.holds_chain_of(second))
                self.assertEqual(self.tap_cache.num_takeable_stripes(), _NUM_STRIPES)

    def test_a_block_whose_taps_did_not_land_releases_its_chains_hold(self):
        self._hold(_block("c", range(16), block=1))
        second = _block("c", range(24), block=2)
        self._reserve(second)
        _finish(second, FINISH_LENGTH(length=8))

        self.requests.end(second, landed=False)

        self.assertFalse(self.holds.holds_chain_of(second))
        self.assertEqual(self._free_and_evictable(), (12, 4))

    def test_held_chains_next_blocks_reserve_before_earlier_requests(self):
        self._hold(_block("c", range(16), block=1))
        self._hold(_block("x", range(100, 116), block=1))
        self._arrive(_request("plain", range(200, 240)))
        block = self._arrive(_block("c", range(32), block=2))

        reserved = self.requests.poll()

        self.assertEqual(reserved, [block])
        self.assertTrue(self.requests.waits("plain"))
        self.assertTrue(self.holds.holds_chain_of(_block("x", range(8), block=2)))

    def test_a_new_request_short_of_room_waits_and_the_holds_stay(self):
        self._hold(_block("a", range(32), block=1))
        self._hold(_block("b", range(100, 132), block=1))

        self._arrive(_request("plain", range(200, 208)))
        self.requests.poll()

        self.assertTrue(self.requests.waits("plain"))
        self.assertTrue(self.holds.holds_chain_of(_block("a", range(8), block=2)))
        self.assertTrue(self.holds.holds_chain_of(_block("b", range(8), block=2)))

    def test_a_held_chains_next_block_short_of_room_releases_the_least_progress(
        self,
    ):
        self._hold(_block("c", range(16), block=1))
        self._hold(_block("far", range(100, 124), block=3))
        self._hold(_block("early", range(200, 224), block=1))

        block = self._arrive(_block("c", range(40), block=2))
        reserved = self.requests.poll()

        self.assertEqual(reserved, [block])
        self.assertFalse(self.holds.holds_chain_of(_block("early", range(8), block=2)))
        self.assertTrue(self.holds.holds_chain_of(_block("far", range(8), block=4)))

    def test_a_held_chains_next_block_waits_when_no_hold_is_left_to_release(self):
        self._hold(_block("c", range(16), block=1))
        self._reserve(_request("big", range(100, 148)))

        self._arrive(_block("c", range(32), block=2))

        self.assertEqual(self.requests.poll(), [])
        self.assertTrue(self.requests.waits("c-2"))
        self.assertTrue(self.holds.holds_chain_of(_block("c", range(8), block=2)))

    def test_an_aborted_waiting_block_releases_its_chains_hold(self):
        for name, first_id, abort in (
            ("forgotten", 0, self.requests.forget),
            ("finished", 1000, self._finish_and_poll),
        ):
            with self.subTest(name):
                self._hold(_block(name, range(first_id, first_id + 16), block=1))
                big = _request(f"{name}-big", range(first_id + 100, first_id + 148))
                self._reserve(big)
                block = self._arrive(
                    _block(name, range(first_id, first_id + 32), block=2)
                )
                self.assertEqual(self.requests.poll(), [])

                abort(block)

                self.assertFalse(self.requests.waits(block.rid))
                self.assertFalse(self.holds.holds_chain_of(block))
                self.requests.end(big, landed=False)
                self.assertEqual(self.tap_cache.num_takeable_stripes(), _NUM_STRIPES)

    def _finish_and_poll(self, request):
        _finish(request)
        self.requests.poll()


class TestTapRelay(_Requests):
    def _sender(self):
        sender = FrankyRelaySender.__new__(FrankyRelaySender)
        sender.kv_mgr = SimpleNamespace(tap_relay=object(), relay_bootstrap_port=9000)
        sender.bootstrap_server_url = "prefill:8998"
        sender.bootstrap_room = 7
        sender.tap_sender = None
        return sender

    def test_prefill_relays_all_taps_once(self):
        sender = self._sender()
        with patch.object(relay_sender_module, "FrankyRoomSender") as room_sender:
            self.assertFalse(sender.relays_taps)
            sender.relay_taps(tap_pages=np.array([5, -1, 6], dtype=np.int32))
            sender.relay_taps(tap_pages=np.array([5, -1, 6], dtype=np.int32))

        room_sender.assert_called_once()
        kwargs = room_sender.call_args.kwargs
        self.assertEqual(kwargs["num_pages"], 3)
        self.assertEqual(kwargs["franky_room"].room, 7)
        self.assertEqual(kwargs["franky_room"].bootstrap_port, 9000)
        [land] = room_sender.return_value.land.call_args_list
        self.assertEqual(land.kwargs["tap_pages"].tolist(), [5, -1, 6])
        self.assertTrue(sender.relays_taps)

    def test_prefill_ends_only_once_kv_and_taps_ended(self):
        sender = self._sender()
        sender.tap_sender = SimpleNamespace(poll=lambda: KVPoll.Transferring)

        with patch.object(NixlKVSender, "poll", return_value=KVPoll.Success):
            self.assertEqual(sender.poll(), KVPoll.Transferring)
            sender.tap_sender = None
            self.assertEqual(sender.poll(), KVPoll.Success)

    def _receiver(self):
        receiver = FrankyRelayReceiver.__new__(FrankyRelayReceiver)
        receiver.kv_mgr = SimpleNamespace(
            tap_requests=self.requests,
            tap_cache=self.tap_cache,
            tap_relay=object(),
            relay_bootstrap_port=9000,
            track_tap_receiver=MagicMock(),
            record_failure=MagicMock(),
        )
        receiver.bootstrap_room = 7
        receiver.bootstrap_addr = "prefill:8998"
        receiver.conclude_state = None
        receiver.tap_receiver = None
        receiver._lifecycle_lock = threading.Lock()
        return receiver

    def test_decode_waits_for_its_stripes_before_pd_prepares_kv(self):
        big = _request("big", range(64))
        self._reserve(big)
        self._arrive(_request("r", range(8), bootstrap_room=7))
        receiver = self._receiver()

        with patch.object(
            NixlKVReceiver, "poll", return_value=KVPoll.WaitingForInput
        ) as base_poll:
            self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
            base_poll.assert_not_called()
            self.requests.end(big, landed=False)
            self.requests.poll()
            self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)

    def test_decode_registers_the_pages_this_rank_keeps_past_the_cached_prefix(self):
        first = _request("a", range(16))
        self._reserve(first)
        self._end_landed(first)
        reservation = self._reserve(_request("r", range(30), bootstrap_room=7))
        receiver = self._receiver()

        # FrankyReceiver.of_reservation builds the receiver this module names.
        with (
            patch.object(NixlKVReceiver, "send_metadata"),
            patch.object(receiver_module, "FrankyReceiver") as tap_receiver,
        ):
            receiver.send_metadata(np.array([3, 4, 5]), 0)

        kwargs = tap_receiver.call_args.kwargs
        pages = reservation.pages.tolist()
        self.assertEqual(kwargs["tap_pages"].tolist(), [pages[2], -1, pages[3], -1])
        self.assertEqual(kwargs["prefix_len"], 16)
        self.assertEqual(kwargs["num_tokens"], 30)
        self.assertEqual(
            kwargs["franky_room"],
            FrankyRoom(bootstrap_host="prefill", bootstrap_port=9000, room=7),
        )
        receiver.kv_mgr.track_tap_receiver.assert_called_once_with(
            7, tap_receiver.return_value
        )

    def test_decode_registers_no_taps_for_a_request_that_reads_none(self):
        self._arrive(_request("plain", range(8), room=None, bootstrap_room=7))
        receiver = self._receiver()

        with (
            patch.object(NixlKVReceiver, "send_metadata"),
            patch.object(receiver_module, "FrankyReceiver") as tap_receiver,
        ):
            receiver.send_metadata(np.array([3, 4, 5]), 0)

        tap_receiver.assert_not_called()
        self.assertIsNone(receiver.tap_receiver)

    def test_decode_fails_a_refused_request(self):
        self.requests.arrive(_request("r", range(_MAX_TAPS + 1), bootstrap_room=7))
        receiver = self._receiver()

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        receiver.kv_mgr.record_failure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
