import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl.conn import NixlKVReceiver, NixlKVSender
from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
from sglang.srt.franky.cache.tap_reservations import FrankyTapReservations
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.franky.chains.chain import FrankyChain
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.transfer.relay import receiver as relay_receiver_module
from sglang.srt.franky.transfer.relay import sender as relay_sender_module
from sglang.srt.franky.transfer.relay.receiver import FrankyRelayReceiver
from sglang.srt.franky.transfer.relay.sender import FrankyRelaySender
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PAGE_SIZE = 4
_STRIPE_SIZE = 2 * _PAGE_SIZE
_NUM_SLOTS = 8
_MAX_TAPS = _NUM_SLOTS * _STRIPE_SIZE


def _tap_cache(*, rank: int = 0) -> RickyTapCache:
    return RickyTapCache.build(
        encoder_ids=(0,),
        output_size=8,
        page_size=_PAGE_SIZE,
        num_taps=_NUM_SLOTS * _STRIPE_SIZE,
        stripes=TapStripes(num_ranks=2, rank=rank),
        device="cpu",
    )


def _request(rid, *, chain=None, num_input_tokens=1):
    request = SimpleNamespace(
        rid=rid,
        franky_chain=chain,
        last_node=None,
        origin_input_ids=[0] * num_input_tokens,
        done=False,
    )
    request.finished = lambda: request.done
    return request


def _block(chain_id, *, block, num_input_tokens=1):
    return _request(
        f"{chain_id}-{block}",
        chain=FrankyChain(id=chain_id, block=block, block_size=8),
        num_input_tokens=num_input_tokens,
    )


class _Reservations(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tap_cache = _tap_cache()
        self.holds = FrankyChainHolds(
            tree_cache=None, tap_cache=self.tap_cache, hold_seconds=60.0
        )
        self.reservations = FrankyTapReservations(
            tap_cache=self.tap_cache, chain_holds=self.holds, max_taps=_MAX_TAPS
        )

    def _expect(self, request, sandy_ids):
        self.reservations.expect(request.rid, req=request, sandy_ids=sandy_ids)
        return request

    def _reserve(self, request, sandy_ids):
        self._expect(request, sandy_ids)
        self.reservations.poll()
        return self.reservations.of(request.rid)

    def _hold(self, request, sandy_ids):
        self._reserve(request, sandy_ids)
        tap_lock = self.reservations.end(request.rid, landed=True, keep_locked=True)
        self.holds.hold_for_next_block(request, tap_lock=tap_lock)


class TestTapReservations(_Reservations):
    def test_requests_take_stripes_in_arrival_order_and_wait_when_full(self):
        first = self._expect(_request("a"), list(range(40)))
        second = self._expect(_request("b"), list(range(100, 124)))
        self._expect(_request("c"), list(range(200, 208)))

        reserved = self.reservations.poll()

        self.assertEqual(reserved, [("a", first), ("b", second)])
        self.assertEqual(len(self.reservations.of("a").slots), 5)
        self.assertEqual(len(self.reservations.of("b").slots), 3)
        self.assertIsNone(self.reservations.of("c"))
        self.assertTrue(self.reservations.waits("c"))

    def test_a_later_request_never_passes_a_waiting_one(self):
        self._reserve(_request("a"), list(range(40)))
        self._expect(_request("b"), list(range(100, 132)))
        self._expect(_request("c"), list(range(200, 208)))

        self.assertEqual(self.reservations.poll(), [])
        self.assertTrue(self.reservations.waits("c"))

    def test_a_request_that_ends_before_its_taps_land_gives_its_stripes_back(self):
        self._reserve(_request("a"), list(range(64)))
        self._expect(_request("b"), list(range(100, 108)))
        self.assertEqual(self.reservations.poll(), [])

        self.assertIsNone(self.reservations.end("a", landed=False))

        self.assertEqual(self.tap_cache.num_free_slots(), _NUM_SLOTS)
        self.assertEqual(self.tap_cache.match(list(range(64))).num_taps, 0)
        self.assertEqual([key for key, _ in self.reservations.poll()], ["b"])

    def test_a_landed_request_leaves_its_whole_stripes_for_later_requests(self):
        first = self._reserve(_request("a"), list(range(20)))
        self.reservations.end("a", landed=True)

        second = self._reserve(_request("b"), list(range(30)))

        self.assertEqual(second.cached.num_taps, 16)
        self.assertEqual(second.slots[:2].tolist(), first.slots[:2].tolist())
        self.assertEqual(len(second.slots), 4)

    def test_at_least_one_page_of_a_cached_request_lands_again(self):
        self._reserve(_request("a"), list(range(16)))
        self.reservations.end("a", landed=True)

        again = self._reserve(_request("b"), list(range(16)))

        self.assertEqual(again.cached.num_taps, 8)
        self.assertEqual(len(again.slots), 2)

    def test_end_without_keep_locked_leaves_the_stripes_evictable(self):
        self._reserve(_request("a"), list(range(24)))

        self.assertIsNone(self.reservations.end("a", landed=True))

        self.assertEqual(self.tap_cache.num_free_slots(), _NUM_SLOTS)
        self.assertEqual(self.tap_cache.num_free_pages(), (10, 6))

    def test_a_request_past_the_taps_one_may_read_is_refused(self):
        too_many = self._expect(_request("a"), list(range(_MAX_TAPS + 1)))
        self._expect(_request("b"), list(range(8)))

        self.assertEqual([key for key, _ in self.reservations.poll()], ["b"])

        self.assertIn("past the 64 taps", self.reservations.refusal("a"))
        self.assertFalse(self.reservations.waits("a"))
        self.assertIsNone(self.reservations.of("a"))
        too_many.done = True
        self.reservations.poll()
        self.assertIsNone(self.reservations.refusal("a"))

    def test_a_waiting_request_that_ended_is_dropped(self):
        self._reserve(_request("a"), list(range(64)))
        waiting = self._expect(_request("b"), list(range(100, 108)))

        waiting.done = True
        self.reservations.poll()

        self.assertFalse(self.reservations.waits("b"))
        self.reservations.end("a", landed=False)
        self.assertEqual(self.reservations.poll(), [])
        self.assertTrue(self.reservations.is_empty())

    def test_forget_drops_a_waiting_or_refused_request(self):
        self._reserve(_request("a"), list(range(64)))
        self._expect(_request("b"), list(range(8)))
        self._expect(_request("c"), list(range(_MAX_TAPS + 1)))

        self.reservations.forget("b")
        self.reservations.forget("c")

        self.assertFalse(self.reservations.waits("b"))
        self.assertIsNone(self.reservations.refusal("c"))
        self.reservations.end("a", landed=False)
        self.assertTrue(self.reservations.is_empty())

    def test_every_rank_reserves_the_same_slots(self):
        slots = []
        for rank in range(2):
            tap_cache = _tap_cache(rank=rank)
            reservations = FrankyTapReservations(
                tap_cache=tap_cache,
                chain_holds=FrankyChainHolds(
                    tree_cache=None, tap_cache=tap_cache, hold_seconds=60.0
                ),
                max_taps=_MAX_TAPS,
            )
            for rid, sandy_ids in (("a", range(20)), ("b", range(30))):
                reservations.expect(rid, req=_request(rid), sandy_ids=list(sandy_ids))
                reservations.poll()
                if rid == "a":
                    reservations.end("a", landed=True)
            slots.append(reservations.of("b").slots.tolist())

        self.assertEqual(slots[0], slots[1])


class TestHeldChains(_Reservations):
    def test_a_held_block_keeps_its_stripes_locked_for_the_next_block(self):
        first = _block("c", block=1)
        self._hold(first, list(range(16)))

        self.assertEqual(self.tap_cache.num_free_slots(), _NUM_SLOTS - 2)
        second = self._reserve(_block("c", block=2), list(range(24)))

        self.assertEqual(second.cached.num_taps, 16)
        self.assertEqual(len(second.slots), 3)

    def test_held_chains_next_blocks_reserve_before_earlier_requests(self):
        self._hold(_block("c", block=1), list(range(16)))
        self._hold(_block("x", block=1), list(range(100, 116)))
        self._expect(_request("plain"), list(range(200, 240)))
        self._expect(_block("c", block=2), list(range(32)))

        reserved = self.reservations.poll()

        self.assertEqual([key for key, _ in reserved], ["c-2"])
        self.assertTrue(self.reservations.waits("plain"))
        self.assertTrue(self.holds.holds_chain_of(_block("x", block=2)))

    def test_a_new_request_short_of_room_waits_and_the_holds_stay(self):
        self._hold(_block("a", block=1), list(range(32)))
        self._hold(_block("b", block=1), list(range(100, 132)))

        self._expect(_request("plain"), list(range(200, 208)))
        self.reservations.poll()

        self.assertTrue(self.reservations.waits("plain"))
        self.assertTrue(self.holds.holds_chain_of(_block("a", block=2)))
        self.assertTrue(self.holds.holds_chain_of(_block("b", block=2)))

    def test_a_held_chains_next_block_short_of_room_releases_the_least_progress(
        self,
    ):
        self._hold(_block("c", block=1), list(range(16)))
        self._hold(_block("far", block=3), list(range(100, 124)))
        self._hold(_block("early", block=1), list(range(200, 224)))

        self._expect(_block("c", block=2), list(range(40)))
        reserved = self.reservations.poll()

        self.assertEqual([key for key, _ in reserved], ["c-2"])
        self.assertFalse(self.holds.holds_chain_of(_block("early", block=2)))
        self.assertTrue(self.holds.holds_chain_of(_block("far", block=4)))

    def test_a_held_chains_next_block_waits_when_no_hold_is_left_to_release(self):
        self._hold(_block("c", block=1), list(range(16)))
        self._reserve(_request("big"), list(range(100, 148)))

        self._expect(_block("c", block=2), list(range(32)))

        self.assertEqual(self.reservations.poll(), [])
        self.assertTrue(self.reservations.waits("c-2"))
        self.assertTrue(self.holds.holds_chain_of(_block("c", block=2)))


class TestTapRelay(_Reservations):
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
            reservations=self.reservations,
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
        self._reserve(_request("big"), list(range(64)))
        self.reservations.expect(7, req=_request("r"), sandy_ids=list(range(8)))
        receiver = self._receiver()

        with patch.object(
            NixlKVReceiver, "poll", return_value=KVPoll.WaitingForInput
        ) as base_poll:
            self.assertEqual(receiver.poll(), KVPoll.Bootstrapping)
            base_poll.assert_not_called()
            self.reservations.end("big", landed=False)
            self.reservations.poll()
            self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)

    def test_decode_registers_the_pages_this_rank_keeps_past_the_cached_prefix(self):
        self._reserve(_request("a"), list(range(16)))
        self.reservations.end("a", landed=True)
        self.reservations.expect(7, req=_request("r"), sandy_ids=list(range(30)))
        self.reservations.poll()
        reservation = self.reservations.of(7)
        receiver = self._receiver()

        with (
            patch.object(NixlKVReceiver, "send_metadata"),
            patch.object(relay_receiver_module, "FrankyReceiver") as tap_receiver,
        ):
            receiver.send_metadata(np.array([3, 4, 5]), 0)

        kwargs = tap_receiver.call_args.kwargs
        slots = reservation.slots.tolist()
        self.assertEqual(kwargs["tap_pages"].tolist(), [slots[2], -1, slots[3], -1])
        self.assertEqual(kwargs["prefix_len"], 16)
        self.assertEqual(kwargs["num_tokens"], 30)
        receiver.kv_mgr.track_tap_receiver.assert_called_once()

    def test_decode_registers_no_taps_for_a_request_that_reads_none(self):
        receiver = self._receiver()

        with (
            patch.object(NixlKVReceiver, "send_metadata"),
            patch.object(relay_receiver_module, "FrankyReceiver") as tap_receiver,
        ):
            receiver.send_metadata(np.array([3, 4, 5]), 0)

        tap_receiver.assert_not_called()
        self.assertIsNone(receiver.tap_receiver)

    def test_decode_fails_a_refused_request(self):
        self.reservations.expect(
            7, req=_request("r"), sandy_ids=list(range(_MAX_TAPS + 1))
        )
        receiver = self._receiver()

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        receiver.kv_mgr.record_failure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
