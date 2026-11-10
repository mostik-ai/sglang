import unittest
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.franky.transfer.messages import FrankyRoom
from sglang.srt.franky.transfer.receiver import FrankyReceiver
from sglang.srt.franky.transfer.senders.room import FrankyRoomSender
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PAGE_SIZE = 16
_ROOM = FrankyRoom(bootstrap_host="sandy", bootstrap_port=8996, room=7)


class _SendingManager:
    sender_trace_module = None
    deferred_bootstrap = None
    is_dummy_cp_rank = False
    kv_shard_size = 1
    enable_all_cp_ranks_for_transfer = False

    def __init__(self, *, decode_prefix_len):
        self.kv_args = SimpleNamespace(page_size=_PAGE_SIZE)
        self.req_to_decode_prefix_len = {_ROOM.room: decode_prefix_len}
        self.request_status = {}
        self.transfers = []

    def update_status(self, room, status):
        self.request_status[room] = status

    def check_status(self, room):
        return self.request_status[room]

    def prepare_send_state_indices(self, state_indices):
        return state_indices

    def add_transfer_request(self, room, kv_indices, index_slice, is_last_chunk, **_):
        self.transfers.append((kv_indices.tolist(), index_slice, is_last_chunk))


class _ReceivingManager:
    defer_decode_allocation = False

    def __init__(self):
        self.kv_args = SimpleNamespace(page_size=_PAGE_SIZE)
        self.request_status = {}
        self.addr_to_rooms_tracker = defaultdict(set)
        self.prefill_response_tracker = {}
        self.pages_landed = 0

    def update_status(self, room, status):
        self.request_status[room] = status

    def track_room(self, room):
        pass

    def landed_pages(self, room):
        return self.pages_landed


class TestFrankyTapsPastTheCachedPrefix(CustomTestCase):
    def setUp(self):
        super().setUp()
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_the_sender_never_sends_the_receivers_cached_pages(self):
        manager = _SendingManager(decode_prefix_len=2 * _PAGE_SIZE)
        sender = FrankyRoomSender(manager=manager, franky_room=_ROOM, num_pages=5)

        sender.land(tap_pages=np.array([10, 11, 12], dtype=np.int32))
        self.assertEqual(manager.transfers, [], "the receivers have not arrived")
        manager.update_status(_ROOM.room, KVPoll.WaitingForInput)
        sender.land(tap_pages=np.array([13], dtype=np.int32))
        sender.land(tap_pages=np.array([14], dtype=np.int32))

        self.assertEqual(
            manager.transfers,
            [([12, 13], slice(0, 2), False), ([14], slice(2, 3), True)],
        )

    def test_a_landing_inside_the_cached_prefix_sends_nothing(self):
        manager = _SendingManager(decode_prefix_len=2 * _PAGE_SIZE)
        sender = FrankyRoomSender(manager=manager, franky_room=_ROOM, num_pages=3)
        manager.update_status(_ROOM.room, KVPoll.WaitingForInput)

        sender.land(tap_pages=np.array([10], dtype=np.int32))
        sender.land(tap_pages=np.array([11, 12], dtype=np.int32))

        self.assertEqual(manager.transfers, [([12], slice(0, 1), True)])

    def test_a_cached_prefix_off_a_page_edge_is_refused(self):
        manager = _SendingManager(decode_prefix_len=_PAGE_SIZE + 1)
        sender = FrankyRoomSender(manager=manager, franky_room=_ROOM, num_pages=3)
        manager.update_status(_ROOM.room, KVPoll.WaitingForInput)

        with self.assertRaisesRegex(ValueError, "17 tokens"):
            sender.land(tap_pages=np.array([10], dtype=np.int32))

    def test_the_receiver_counts_landed_taps_from_its_cached_prefix(self):
        manager = _ReceivingManager()
        receiver = FrankyReceiver(
            manager=manager,
            franky_room=_ROOM,
            tap_pages=np.array([5, 6, 7], dtype=np.int32),
            num_tokens=70,
            prefix_len=32,
        )

        self.assertEqual(receiver.landed(), 32)
        manager.pages_landed = 2
        self.assertEqual(receiver.landed(), 64)
        manager.pages_landed = 3
        self.assertEqual(receiver.landed(), 70)


if __name__ == "__main__":
    unittest.main()
