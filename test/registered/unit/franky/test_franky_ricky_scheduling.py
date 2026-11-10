import socket
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.franky.cache.kv_allocator import FrankyKvAllocator
from sglang.srt.franky.cache.tap_cache import FrankyTapCache
from sglang.srt.franky.chains.chain import FrankyChain
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.schedulers.prefill_adder import FrankyPrefillAdder
from sglang.srt.franky.schedulers.ricky import RickyScheduler
from sglang.srt.franky.transfer import receive_queue as receive_queue_module
from sglang.srt.franky.transfer.messages import (
    FrankyRoom,
    FrankyRoomClosed,
    FrankyTapsReady,
)
from sglang.srt.franky.transfer.rank_reports import FrankyRankReports
from sglang.srt.franky.transfer.receive_queue import FrankyReceiveQueue
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    Req,
)
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefResult,
    IncLockRefResult,
    TreeLock,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_PAGE_SIZE = 16
_NUM_TOKENS = 1024


class _Receiver:
    def __init__(self, *, manager, franky_room, tap_pages, num_tokens, prefix_len):
        self.manager = manager
        self.tap_pages = tap_pages.tolist()
        self.prefix_len = prefix_len
        self.bootstrap_room = franky_room.room
        self.state = KVPoll.WaitingForInput
        self.landed_tokens = 0
        self.pages_registered = True
        self.abort_notified = False
        self.required_prefill_response_num = 1
        self.closed = False

    def poll(self):
        return self.state

    def landed(self):
        return self.landed_tokens

    def abort(self):
        self.abort_notified = True
        self.state = KVPoll.Failed

    def writes_stopped(self):
        return self.bootstrap_room in self.manager.acked_rooms

    def close(self):
        self.closed = True


class _Tree:
    """Unlocking a node's last lock makes its tokens evictable."""

    def __init__(self):
        self.locked = []
        self.node_tokens = {}
        self.evictable = 0

    def lock(self, node):
        self.locked.append(node)
        return node

    def unlock(self, lock):
        self.locked.remove(lock)
        if lock not in self.locked:
            self.evictable += self.node_tokens.get(lock, 0)

    def evictable_size(self):
        return self.evictable


class _Req:
    def __init__(self, rid, *, prompt_len, cached_prefix_len=0, room=True, chain=None):
        self.rid = rid
        self.franky_chain = chain
        self.finished_reason = None
        self.origin_input_ids = list(range(prompt_len))
        self.output_ids = []
        self.cached_prefix_len = cached_prefix_len
        self.franky = None
        self.franky_room = (
            FrankyRoom(bootstrap_host="sandy", bootstrap_port=8996, room=7)
            if room
            else None
        )
        self.last_node = f"node-{rid}"
        self.prefix_len = 0
        self.extra_key = None
        self.skip_radix_cache_insert = False
        self.done = False

    def init_next_round_input(self, tree_cache):
        self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids
        self.prefix_len = self.cached_prefix_len

    def finished(self):
        return self.done


class _PPStore(dict):
    def __init__(self):
        super().__init__()
        self.other_stages = {}

    def collect(self, key):
        return [self.get(key), self.other_stages.get(key)]


class _Manager:
    enable_deferred_decode_kv_release = True

    def __init__(self):
        self.acked_rooms = set()

    def is_abort_release_safe(self, room, required_acks):
        return room in self.acked_rooms

    def clear_deferred_abort_state(self, room):
        self.acked_rooms.discard(room)


def _init_single_process_group():
    if dist.is_initialized():
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )


class TestFrankyReceiveQueue(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _init_single_process_group()

    def setUp(self):
        super().setUp()
        self.receivers = []
        self.failed = []
        self.tree = _Tree()
        self.manager = _Manager()
        self.store = _PPStore()
        self.allocator = FrankyKvAllocator(
            _NUM_TOKENS,
            page_size=_PAGE_SIZE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.allocator.use_triton_kernels = False
        receiver = patch.object(receive_queue_module, "FrankyReceiver", self._receiver)
        receiver.start()
        self.addCleanup(receiver.stop)
        self.holds = FrankyChainHolds(
            tree_cache=self.tree, max_held_tokens=_NUM_TOKENS, hold_seconds=60
        )
        self.queue = FrankyReceiveQueue(
            tap_cache=FrankyTapCache(
                encoder_ids=(0,),
                output_size=2,
                page_size=_PAGE_SIZE,
                num_tokens=_NUM_TOKENS,
                req_to_token=torch.zeros((2, 16), dtype=torch.int32),
                device="cpu",
            ),
            kv_allocator=self.allocator,
            tree_cache=self.tree,
            tap_manager=self.manager,
            rank_reports=FrankyRankReports(
                attn_tp_cpu_group=dist.group.WORLD,
                attn_cp_cpu_group=dist.group.WORLD,
                pp_store=self.store,
            ),
            chain_holds=self.holds,
            page_size=_PAGE_SIZE,
            reserved_decode_tokens=64,
            release_timeout_seconds=600,
            fail_request=lambda req, reason: self.failed.append(req.rid),
        )

    def _receiver(self, **kwargs):
        receiver = _Receiver(**kwargs)
        self.receivers.append(receiver)
        return receiver

    def _step(self, *, other_stage=None):
        self.queue.poll()
        for key, number in (other_stage or {}).items():
            self.store.other_stages[key] = number
        messages = self.queue.ready_messages()
        for message in messages:
            if isinstance(message, FrankyTapsReady):
                self.queue.on_taps_ready(message=message)
            else:
                self.queue.on_room_closed(message=message)
        return messages

    def _arrive(self, req):
        self.queue.add(req=req)
        self.queue.poll()
        return req

    def test_taps_land_only_in_the_pages_past_the_cached_prefix(self):
        req = self._arrive(_Req("r", prompt_len=70, cached_prefix_len=32))

        [receiver] = self.receivers
        taken_pages = (req.franky.taken_kv_slots[::_PAGE_SIZE] // _PAGE_SIZE).tolist()
        self.assertEqual(receiver.tap_pages, taken_pages)
        self.assertEqual(receiver.prefix_len, 32)
        self.assertEqual(req.franky.taps_ready, 32)
        self.assertEqual(len(req.franky.taken_kv_slots), 70 - 32)
        self.assertEqual(req.franky.taken_from, 32)

    def test_taps_count_from_the_cached_prefix(self):
        req = self._arrive(_Req("r", prompt_len=70, cached_prefix_len=32))
        self.receivers[0].landed_tokens = 48

        messages = self._step(other_stage={"landed:r": 48, "ok:r": 1})

        self.assertEqual(messages, [FrankyTapsReady(request_rid="r", taps_ready=48)])
        self.assertEqual(req.franky.taps_ready, 48)

    def test_an_arrival_that_does_not_fit_holds_back_the_ones_behind_it(self):
        self._arrive(_Req("first", prompt_len=512))
        second = self._arrive(_Req("second", prompt_len=400))
        third = self._arrive(_Req("third", prompt_len=16))

        self.assertIsNone(second.franky)
        self.assertIsNone(third.franky)
        self.assertEqual(self.failed, [])

    def test_taps_are_ready_up_to_the_lowest_landing_of_every_stage(self):
        req = self._arrive(_Req("r", prompt_len=64))
        self.receivers[0].landed_tokens = 48

        self.assertEqual(self._step(), [], "stage 1 has not reported yet")
        messages = self._step(other_stage={"landed:r": 32, "ok:r": 1})

        self.assertEqual(messages, [FrankyTapsReady(request_rid="r", taps_ready=32)])
        self.assertEqual(req.franky.taps_ready, 32)

    def test_a_request_that_ends_early_keeps_its_pages_until_its_room_closes(self):
        req = self._arrive(_Req("r", prompt_len=64, cached_prefix_len=16))
        receiver = self.receivers[0]
        req.franky.hand_to_chunk(start=16, end=32)
        available = self.allocator.available_size()

        self.queue.forget(req=req)
        self._step(other_stage={"closed:r": 1})

        self.assertTrue(receiver.abort_notified)
        self.assertEqual(self.allocator.available_size(), available)
        self.assertEqual(self.tree.locked, ["node-r"])

        self.manager.acked_rooms.add(receiver.bootstrap_room)
        messages = self._step(other_stage={"closed:r": 1})

        self.assertEqual(messages, [FrankyRoomClosed(request_rid="r")])
        self.assertEqual(self.allocator.available_size(), available + 2 * _PAGE_SIZE)
        self.assertEqual(self.tree.locked, [])
        self.assertTrue(receiver.closed)
        self.assertTrue(self.queue.is_empty())

    def test_a_failed_room_fails_its_request(self):
        self._arrive(_Req("r", prompt_len=64))
        self.receivers[0].state = KVPoll.Failed

        self._step(other_stage={"landed:r": 0, "ok:r": 1})

        self.assertEqual(self.failed, ["r"])

    def test_a_returning_request_whose_taps_left_the_cache_fails(self):
        req = self._arrive(_Req("r", prompt_len=64))
        req.franky.taps_ready = 64
        req.cached_prefix_len = 32

        self._arrive(req)

        self.assertEqual(self.failed, ["r"])

    def test_a_returning_request_never_runs_on_the_pages_it_gave_back(self):
        req = self._arrive(_Req("r", prompt_len=64, cached_prefix_len=16))
        req.franky.taps_ready = 64
        req.output_ids = list(range(8))
        req.cached_prefix_len = 64

        self.queue.add(req=req)

        self.assertIsNone(req.franky)
        self.queue.poll()
        self.assertEqual(req.franky.taken_from, 64)
        self.assertEqual(len(req.franky.taken_kv_slots), 8)
        self.assertEqual(req.franky.tap_len, 64)

    def test_a_request_without_a_room_runs_at_once_and_caches_apart(self):
        req = _Req("r", prompt_len=64, room=False)
        req.extra_key = "user"
        self._arrive(req)

        self.assertEqual(self.receivers, [])
        self.assertEqual(req.franky.taps_ready, 64)
        self.assertEqual(req.franky.tap_len, 0)
        self.assertEqual(req.extra_key, "user|franky=no-taps")

    def _finish(self, req, reason):
        req.franky.taps_ready = req.franky.tap_len
        req.done = True
        req.finished_reason = reason

    def test_a_block_at_its_token_limit_holds_its_prompt_for_the_next_block(self):
        req = self._arrive(_Req("c-1", prompt_len=64, chain=FrankyChain(id="c")))

        self._finish(req, FINISH_LENGTH(length=8))
        self.queue.poll()

        self.assertEqual(self.tree.locked, ["node-c-1"])
        self.assertTrue(self.queue.is_empty())

    def test_a_stopped_or_last_block_holds_nothing(self):
        stopped = self._arrive(_Req("s-1", prompt_len=64, chain=FrankyChain(id="s")))
        last_chain = FrankyChain(id="l", last_block=True)
        last = self._arrive(_Req("l-1", prompt_len=64, chain=last_chain))

        self._finish(stopped, FINISH_MATCHED_TOKEN(matched=0))
        self._finish(last, FINISH_LENGTH(length=8))
        self.queue.poll()

        self.assertEqual(self.tree.locked, [])

    def test_an_arrival_that_does_not_fit_releases_the_oldest_holds_first(self):
        self.tree.node_tokens = {"held-a": 400, "held-b": 400}
        self.holds.hold(chain_id="a", node="held-a", num_tokens=400)
        self.holds.hold(chain_id="b", node="held-b", num_tokens=400)
        self._arrive(_Req("first", prompt_len=512))

        second = self._arrive(_Req("second", prompt_len=400))

        self.assertIsNotNone(second.franky)
        self.assertEqual(self.tree.locked, ["held-b", "node-first", "node-second"])

    def test_holds_too_small_to_make_room_are_kept(self):
        self.tree.node_tokens = {"held-a": 8}
        self.holds.hold(chain_id="a", node="held-a", num_tokens=8)
        self._arrive(_Req("first", prompt_len=512))

        second = self._arrive(_Req("second", prompt_len=400))

        self.assertIsNone(second.franky)
        self.assertEqual(self.tree.locked, ["held-a", "node-first"])

    def test_kv_past_the_prompt_never_enters_the_cache(self):
        req = self._arrive(_Req("r", prompt_len=64))
        req.prefix_len = 48
        self.queue.poll()
        self.assertFalse(req.skip_radix_cache_insert)

        req.prefix_len = 64
        self.queue.poll()

        self.assertTrue(req.skip_radix_cache_insert)


class TestFrankyPrefillAdder(CustomTestCase):
    def setUp(self):
        super().setUp()
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.tree_cache = MagicMock()
        self.tree_cache.evictable_size.return_value = 0
        self.tree_cache.disable = False
        self.tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        self.tree_cache.dec_lock_ref.return_value = DecLockRefResult()
        self.tree_cache.lock.side_effect = lambda node: TreeLock(
            node, self.tree_cache.inc_lock_ref(node).to_dec_params()
        )
        self.tree_cache.buffer_pipeline = None
        self.allocator = MagicMock()
        self.allocator.page_size = _PAGE_SIZE
        self.allocator.available_size.return_value = 0
        self.allocator.request_slot_reserve.return_value = 0

    def _adder(self, *, rem_chunk_tokens=256):
        running_batch = MagicMock()
        running_batch.reqs = []
        return FrankyPrefillAdder(
            page_size=_PAGE_SIZE,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.allocator,
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=10000,
            rem_chunk_tokens=rem_chunk_tokens,
        )

    def _req(self, rid, *, prompt_len, prefix_len=0, taps_ready):
        req = MagicMock(spec=Req)
        type(req).extend_len = Req.extend_len
        req.rid = rid
        req.kv = SimpleNamespace(req_pool_idx=None, holds_mamba=False)
        req.cache_request_handle = CacheRequestHandle(rid, 0)
        req.last_node = f"node-{rid}"
        req.prefix_len = prefix_len
        req.extend_end = None
        req.full_untruncated_fill_ids = list(range(prompt_len))
        req.output_ids = []
        req.sampling_params = SimpleNamespace(max_new_tokens=8, ignore_eos=False)
        req.retracted_stain = False
        req.host_hit_length = 0
        req.swa_host_hit_length = 0
        req.auxiliary_swa_host_hit_length = 0
        req.mamba_host_hit_length = 0
        req.storage_hit_length = 0
        req.storage_hit_start = None
        req.host_hit_is_storage = False
        req.host_loaded_length = 0
        req.materialized_host_hit_len.return_value = 0
        req.fulfilled_storage_hit_len.return_value = 0
        req.finished.return_value = False
        req.needs_host_load_back.return_value = False
        req.franky = SimpleNamespace(taps_ready=taps_ready)
        return req

    def test_a_first_chunk_ends_at_the_landed_taps_on_taken_slots(self):
        adder = self._adder()
        req = self._req("r", prompt_len=100, taps_ready=32)

        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

        self.assertEqual(req.extend_end, 32)
        self.assertIs(adder.new_chunked_req, req)
        self.allocator.expect_extend.assert_called_with(reqs=[req])

    def test_a_request_without_new_taps_waits_and_lets_others_in(self):
        adder = self._adder()
        waiting = self._req("waiting", prompt_len=100, prefix_len=32, taps_ready=32)

        result = adder.add_one_req(
            waiting, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(adder.can_run_list, [])

    def test_while_a_chunked_request_goes_on_only_a_whole_request_joins(self):
        adder = self._adder()
        partial = self._req("partial", prompt_len=100, taps_ready=32)
        whole = self._req("whole", prompt_len=40, taps_ready=40)

        for req in (partial, whole):
            adder.add_one_req(req, has_chunked_req=True, truncation_align_size=None)

        self.assertEqual(adder.can_run_list, [whole])
        self.assertIsNone(adder.new_chunked_req)

    def test_a_continuing_chunk_ends_at_the_landed_taps(self):
        adder = self._adder()
        req = self._req("r", prompt_len=100, prefix_len=32, taps_ready=64)

        self.assertIs(adder.add_chunked_req(req), req)
        self.assertEqual(req.extend_end, 64)

    def test_the_prefill_delayer_counts_only_requests_with_landed_taps(self):
        """Taps land one request at a time; counting requests that still wait on
        taps would let every lone ready request start its own prefill."""
        scheduler = object.__new__(RickyScheduler)
        scheduler.waiting_queue = [
            SimpleNamespace(franky=SimpleNamespace(taps_ready=64), prefix_len=0),
            SimpleNamespace(franky=SimpleNamespace(taps_ready=32), prefix_len=32),
            SimpleNamespace(franky=None, prefix_len=0),
        ]

        self.assertEqual(scheduler.prefillable_queue_len(), 1)


if __name__ == "__main__":
    unittest.main()
