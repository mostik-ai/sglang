import socket
import sys
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch.distributed as dist

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.franky.cache.mamba_room import FrankyMambaRoom
from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
from sglang.srt.franky.cache.tap_buffers import TapBuffers
from sglang.srt.franky.cache.tap_request_table import TapRequestTable
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.franky.chains.blocks.sandy_call import FrankySandyCall
from sglang.srt.franky.chains.chain import FrankyChain
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.request import FrankyRequest
from sglang.srt.franky.ricky_taps.prefill_tap_limit import PrefillTapLimit
from sglang.srt.franky.ricky_taps.tap_progress import TapProgress
from sglang.srt.franky.ricky_taps.tap_requests import RickyTapRequests
from sglang.srt.franky.ricky_taps.taps import RickyTaps
from sglang.srt.franky.schedulers.parked_chunks import FrankyParkedChunks
from sglang.srt.franky.schedulers.prefill_adder import FrankyPrefillAdder
from sglang.srt.franky.schedulers.ricky import RickyScheduler
from sglang.srt.franky.schedulers.sandy import SandyScheduler
from sglang.srt.franky.tap_runs import FrankyTapRuns
from sglang.srt.franky.transfer import receiver as receiver_module
from sglang.srt.franky.transfer.messages import (
    FrankyRoom,
    FrankyRoomClosed,
    FrankyTapsReady,
)
from sglang.srt.franky.transfer.rank_reports import FrankyRankReports
from sglang.srt.franky.transfer.receive_queue import FrankyReceiveQueue
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    Req,
)
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefResult,
    IncLockRefResult,
    TreeLock,
)
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_PAGE_SIZE = 16
_NUM_STRIPES = 16
_MAX_TAPS = 512
_SANDY_CALL = FrankySandyCall(next_block=2)


class _Receiver:
    def __init__(self, *, manager, franky_room, tap_pages, prefix_len):
        self.manager = manager
        self.tap_pages = tap_pages.tolist()
        self.prefix_len = prefix_len
        self.bootstrap_room = franky_room.room
        self.state = KVPoll.WaitingForInput
        self.landed_tokens = prefix_len
        self.abort_notified = False
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
    def __init__(self):
        self.locked = []

    def lock(self, node):
        self.locked.append(node)
        return SimpleNamespace(node=node)

    def unlock(self, lock):
        if lock is not None:
            self.locked.remove(lock.node)


class _Req:
    def __init__(
        self,
        rid,
        *,
        prompt_len,
        first_id=0,
        room=True,
        chain=None,
        sandy_input_ids=None,
        tap_runs=None,
        row=0,
    ):
        self.rid = rid
        num_taps = prompt_len if sandy_input_ids is None else len(sandy_input_ids)
        # As the tokenizer manager builds it: none for a request without Franky
        # fields, and the answer after the input reading all taps without runs.
        self.franky = (
            FrankyRequest(
                room=(
                    FrankyRoom(bootstrap_host="sandy", bootstrap_port=8996, room=7)
                    if room
                    else None
                ),
                chain=chain,
                tap_runs=(
                    None
                    if not room
                    else FrankyTapRuns.of_runs(
                        tap_runs or [[prompt_len, num_taps]],
                        num_input_tokens=prompt_len,
                        num_taps=num_taps,
                    )
                ),
                sandy_input_ids=sandy_input_ids,
            )
            if room or chain is not None or sandy_input_ids is not None
            else None
        )
        self.bootstrap_room = None
        self.kv = SimpleNamespace(req_pool_idx=row)
        self.finished_reason = None
        self.origin_input_ids = list(range(first_id, first_id + prompt_len))
        self.output_ids = []
        self.franky_taps = None
        self.last_node = f"node-{rid}"
        self.prefix_len = 0
        self.extra_key = None
        self.skip_radix_cache_insert = False
        self.done = False
        self.time_stats = SchedulerReqTimeStats()

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

    def clear_deferred_abort_state(self, room):
        self.acked_rooms.discard(room)


def _held(chain_id, *, node, num_tokens, block=1):
    return SimpleNamespace(
        franky=FrankyRequest(chain=FrankyChain(id=chain_id, block=block)),
        last_node=node,
        origin_input_ids=[0] * num_tokens,
        finished_reason=_SANDY_CALL,
    )


def _single_turn(prompt_len):
    return FrankyTapRuns(starts=(prompt_len,), taps=(prompt_len,))


def _tap_cache(*, num_ranks=1, rank=0, num_stripes=_NUM_STRIPES):
    return RickyTapCache(
        buffers=TapBuffers(
            encoder_ids=(0,),
            output_size=2,
            page_size=_PAGE_SIZE,
            num_pages=num_stripes + 1,
            device="cpu",
        ),
        stripes=TapStripes(num_ranks=num_ranks, rank=rank),
    )


def _tap_requests(
    tap_cache, *, chain_holds, shared_tokenizer=True, num_slots=4, mamba_room=None
):
    """Requests that may read up to ``_MAX_TAPS`` taps."""
    return RickyTapRequests(
        tap_cache=tap_cache,
        request_table=TapRequestTable(
            num_slots=num_slots,
            max_stripes=_MAX_TAPS // tap_cache.stripe_size,
            device="cpu",
        ),
        chain_holds=chain_holds,
        shared_tokenizer=shared_tokenizer,
        mamba_room=mamba_room,
    )


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
        self.statuses = []
        self.tree = _Tree()
        self.manager = _Manager()
        self.store = _PPStore()
        # FrankyReceiver.of_reservation builds the receiver this module names.
        receiver = patch.object(receiver_module, "FrankyReceiver", self._receiver)
        receiver.start()
        self.addCleanup(receiver.stop)
        self.queue = self._queue()

    def _queue(self, *, num_ranks=1, rank=0, shared_tokenizer=True):
        self.tap_cache = _tap_cache(num_ranks=num_ranks, rank=rank)
        self.holds = FrankyChainHolds(
            tree_cache=self.tree,
            tap_cache=self.tap_cache,
            hold_seconds=60,
        )
        return self._queue_on(shared_tokenizer=shared_tokenizer, mamba_room=None)

    def _queue_on(self, *, shared_tokenizer, mamba_room):
        self.tap_requests = _tap_requests(
            self.tap_cache,
            chain_holds=self.holds,
            shared_tokenizer=shared_tokenizer,
            mamba_room=mamba_room,
        )
        return FrankyReceiveQueue(
            tap_cache=self.tap_cache,
            tap_requests=self.tap_requests,
            tap_manager=self.manager,
            rank_reports=FrankyRankReports(
                attn_tp_cpu_group=dist.group.WORLD,
                attn_cp_cpu_group=dist.group.WORLD,
                pp_store=self.store,
            ),
            release_timeout_seconds=600,
            fail_request=self._fail,
        )

    def _fail(self, req, reason, status_code):
        """As the scheduler fails a request: it ends, and the queue forgets it."""
        self.failed.append(req.rid)
        self.statuses.append(status_code)
        req.done = True
        req.finished_reason = FINISH_ABORT(reason, status_code=status_code)
        self.queue.forget(req=req)

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

    def _finish(self, req, reason):
        req.franky_taps.taps_ready = req.franky_taps.tap_len
        req.done = True
        req.finished_reason = reason

    def _run(self, req, reason=FINISH_MATCHED_TOKEN(matched=0)):
        self._arrive(req)
        pages = self._pages(req.rid)
        self._finish(req, reason)
        self.queue.poll()
        return pages

    def _pages(self, rid):
        return self.tap_requests.reservation(rid).pages.tolist()

    def _cached_taps(self, sandy_ids):
        cached = self.tap_cache.match(sandy_ids)
        self.tap_cache.release(cached.lock)
        return cached.num_taps

    def _free_and_evictable(self):
        usage = self.tap_cache.usage()
        return usage.free_pages, usage.evictable_pages

    def test_an_arrival_takes_a_stripe_per_page_and_all_its_taps_land(self):
        req = self._arrive(_Req("r", prompt_len=70))

        [receiver] = self.receivers
        pages = self._pages("r")
        self.assertEqual(len(pages), 5)
        self.assertEqual(receiver.tap_pages, pages)
        self.assertEqual(receiver.prefix_len, 0)
        self.assertEqual(req.franky_taps.tap_len, 70)
        self.assertEqual(req.franky_taps.taps_ready, 0)
        self.assertEqual(self.tap_cache.num_takeable_stripes(), _NUM_STRIPES - 5)

    def test_each_rank_lists_only_the_pages_it_keeps(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                self.receivers = []
                self.queue = self._queue(num_ranks=2, rank=rank)

                self._arrive(_Req("r", prompt_len=70))

                pages = self._pages("r")
                self.assertEqual(len(pages), 3)
                self.assertEqual(
                    self.receivers[0].tap_pages,
                    [pages[page // 2] if page % 2 == rank else -1 for page in range(5)],
                )

    def test_a_later_request_reads_cached_whole_stripes_and_sandy_skips_them(self):
        first_pages = self._run(_Req("first", prompt_len=70))

        later = self._arrive(_Req("later", prompt_len=90))

        receiver = self.receivers[-1]
        pages = self._pages("later")
        self.assertEqual(pages[:4], first_pages[:4])
        self.assertEqual(receiver.prefix_len, 64)
        self.assertEqual(receiver.tap_pages, pages[4:])
        self.assertEqual(later.franky_taps.taps_ready, 64)
        self.assertEqual(later.time_stats.franky_taps_held, 64)

    def test_a_fully_cached_input_still_lands_its_last_page(self):
        first_pages = self._run(_Req("first", prompt_len=64))

        self._arrive(_Req("again", prompt_len=64))

        receiver = self.receivers[-1]
        pages = self._pages("again")
        self.assertEqual(pages[:3], first_pages[:3])
        self.assertEqual(receiver.prefix_len, 48)
        self.assertEqual(receiver.tap_pages, pages[3:])
        self.assertEqual(len(receiver.tap_pages), 1)

    def test_a_request_whose_taps_all_landed_leaves_its_whole_stripes_cached(self):
        self._run(_Req("r", prompt_len=70))

        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self.tap_cache.num_takeable_stripes(), _NUM_STRIPES)
        self.assertEqual(self._free_and_evictable(), (_NUM_STRIPES - 4, 4))
        self.assertEqual(self._cached_taps(list(range(70))), 64)

    def test_rows_that_read_no_taps_run_before_any_tap_lands(self):
        for shared_tokenizer, sandy_input_ids, tap_len in (
            (True, None, 70),
            (False, list(range(100, 150)), 50),
        ):
            with self.subTest(shared_tokenizer=shared_tokenizer):
                self.queue = self._queue(shared_tokenizer=shared_tokenizer)
                req = _Req("r", prompt_len=70, sandy_input_ids=sandy_input_ids)

                self.queue.add(req=req)

                self.assertEqual(req.franky_taps.tap_len, tap_len)
                self.assertEqual(req.franky_taps.taps_ready, 0)
                self.assertEqual(req.franky_taps.rows_ready(), 69)
                self.assertEqual(req.time_stats.franky_taps_ready_time, 0.0)

                self.queue.poll()

                self.assertGreaterEqual(
                    req.time_stats.franky_taps_ready_time,
                    req.time_stats.franky_pages_taken_time,
                )
                self.assertGreater(req.time_stats.franky_pages_taken_time, 0.0)

    def test_with_sandys_own_tokens_the_stripes_count_sandys_tokens(self):
        self.queue = self._queue(shared_tokenizer=False)
        sandy_ids = list(range(100, 150))

        req = self._arrive(_Req("r", prompt_len=70, sandy_input_ids=sandy_ids))

        [receiver] = self.receivers
        self.assertEqual(len(receiver.tap_pages), 4)
        self.assertEqual(req.franky_taps.tap_len, 50)
        self.assertEqual(req.franky_taps.tap_runs.taps[-1], 50)

    def test_sandys_own_tokens_key_the_cached_taps(self):
        self.queue = self._queue(shared_tokenizer=False)
        sandy_ids = list(range(100, 170))
        self._run(_Req("first", prompt_len=20, sandy_input_ids=sandy_ids))

        self._arrive(_Req("same", prompt_len=30, sandy_input_ids=sandy_ids + [1]))
        self._arrive(
            _Req("other", prompt_len=20, sandy_input_ids=list(range(200, 270)))
        )

        self.assertEqual([r.prefix_len for r in self.receivers[1:]], [64, 0])

    def test_a_request_notes_when_it_took_its_stripes_and_its_first_chunk_could_run(
        self,
    ):
        req = _Req("r", prompt_len=70)
        req.prefix_len = 69
        stats = req.time_stats

        self._arrive(req)

        self.assertGreater(stats.franky_pages_taken_time, 0.0)
        self.assertEqual(stats.franky_tap_len, 70)
        self.assertEqual(stats.franky_taps_held, 0)
        self.assertEqual(stats.franky_taps_ready_time, 0.0)
        self.receivers[0].landed_tokens = 70

        self._step(other_stage={"landed:r": 70, "ok:r": 1})

        self.assertGreater(stats.franky_taps_ready_time, stats.franky_pages_taken_time)

    def test_taps_count_from_the_cached_prefix(self):
        self._run(_Req("first", prompt_len=70))
        later = self._arrive(_Req("later", prompt_len=90))
        self.receivers[-1].landed_tokens = 80

        messages = self._step(other_stage={"landed:later": 80, "ok:later": 1})

        self.assertEqual(
            messages, [FrankyTapsReady(request_rid="later", taps_ready=80)]
        )
        self.assertEqual(later.franky_taps.taps_ready, 80)

    def test_taps_are_ready_up_to_the_lowest_landing_of_every_stage(self):
        req = self._arrive(_Req("r", prompt_len=64))
        self.receivers[0].landed_tokens = 48

        self.assertEqual(self._step(), [], "stage 1 has not reported yet")
        messages = self._step(other_stage={"landed:r": 32, "ok:r": 1})

        self.assertEqual(messages, [FrankyTapsReady(request_rid="r", taps_ready=32)])
        self.assertEqual(req.franky_taps.taps_ready, 32)

    def test_an_arrival_that_does_not_fit_holds_back_the_ones_behind_it(self):
        self._arrive(_Req("first", prompt_len=200))
        second = self._arrive(_Req("second", prompt_len=100))
        third = self._arrive(_Req("third", prompt_len=16))

        self.assertEqual(len(self.receivers), 1)
        self.assertTrue(self.tap_requests.waits("second"))
        self.assertTrue(self.tap_requests.waits("third"))
        self.assertEqual(self.failed, [])
        self.assertEqual(second.franky_taps.rows_ready(), 99)
        self.assertEqual(third.franky_taps.rows_ready(), 15)

    def test_a_waiting_arrival_takes_its_stripes_once_they_come_back(self):
        first = self._arrive(_Req("first", prompt_len=200))
        self._arrive(_Req("second", prompt_len=100))

        self._finish(first, FINISH_MATCHED_TOKEN(matched=0))
        self.queue.poll()

        self.assertEqual(len(self.receivers), 2)
        self.assertIsNotNone(self.tap_requests.reservation("second"))

    def test_a_request_that_ends_early_keeps_its_stripes_until_its_room_closes(self):
        req = self._arrive(_Req("r", prompt_len=64))
        receiver = self.receivers[0]
        takeable = self.tap_cache.num_takeable_stripes()

        self.queue.forget(req=req)
        self._step(other_stage={"closed:r": 1})

        self.assertTrue(receiver.abort_notified)
        self.assertEqual(self.tap_cache.num_takeable_stripes(), takeable)

        self.manager.acked_rooms.add(receiver.bootstrap_room)
        messages = self._step(other_stage={"closed:r": 1})

        self.assertEqual(messages, [FrankyRoomClosed(request_rid="r")])
        self.assertEqual(self.tap_cache.num_takeable_stripes(), _NUM_STRIPES)
        self.assertEqual(self._cached_taps(list(range(64))), 0)
        self.assertTrue(receiver.closed)
        self.assertTrue(self.queue.is_empty())

    def test_a_request_that_ends_after_its_taps_landed_keeps_them_cached(self):
        req = self._arrive(_Req("r", prompt_len=64))
        req.franky_taps.taps_ready = 64

        self.queue.forget(req=req)

        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self._cached_taps(list(range(64))), 64)

    def test_an_aborted_waiting_request_takes_no_stripes(self):
        self._arrive(_Req("first", prompt_len=200))
        waiting = self._arrive(_Req("waiting", prompt_len=100))

        self.queue.forget(req=waiting)
        self.queue.poll()

        self.assertFalse(self.tap_requests.waits("waiting"))
        self.assertEqual(len(self.receivers), 1)

    def test_a_failed_room_fails_its_request_as_its_sandys_failure(self):
        """The router blames the room's Sandy for a leg that failed upstream."""
        self._arrive(_Req("r", prompt_len=64))
        self.receivers[0].state = KVPoll.Failed

        self._step(other_stage={"landed:r": 0, "ok:r": 1})

        self.assertEqual(self.failed, ["r"])
        self.assertEqual(self.statuses, [HTTPStatus.BAD_GATEWAY])

    def test_a_request_names_sandys_tokens_exactly_when_sandy_tokenizes_alone(self):
        self.queue.add(req=_Req("shared", prompt_len=16, sandy_input_ids=[1, 2]))
        self.assertEqual(self.failed, [], "failures wait for the next poll")
        self.queue.poll()
        self.assertEqual(self.statuses, [HTTPStatus.BAD_REQUEST])
        self.queue = self._queue(shared_tokenizer=False)
        for req in (
            _Req("own", prompt_len=16),
            _Req(
                "too-long",
                prompt_len=16,
                sandy_input_ids=list(range(_MAX_TAPS + 1)),
            ),
        ):
            self.queue.add(req=req)
        self.queue.poll()

        self.assertEqual(self.failed, ["shared", "own", "too-long"])
        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self.receivers, [])

    def test_a_returning_request_keeps_its_taps_and_its_receiver(self):
        req = self._arrive(_Req("r", prompt_len=64))
        reservation = self.tap_requests.reservation("r")
        req.franky_taps.taps_ready = 32
        state = req.franky_taps

        self._arrive(req)

        self.assertIs(req.franky_taps, state)
        self.assertEqual(req.franky_taps.taps_ready, 32)
        self.assertIs(self.tap_requests.reservation("r"), reservation)
        self.assertEqual(len(self.receivers), 1)

    def test_a_request_without_a_room_runs_at_once_and_caches_apart(self):
        req = _Req("r", prompt_len=64, room=False)
        req.extra_key = "user"
        self._arrive(req)

        self.assertEqual(self.receivers, [])
        self.assertEqual(req.franky_taps.tap_len, 0)
        self.assertEqual(req.franky_taps.rows_ready(), sys.maxsize)
        self.assertEqual(req.extra_key, "user|franky=no-taps")
        self.assertTrue(self.queue.is_empty())

    def test_kv_past_the_prompt_never_enters_the_cache(self):
        req = self._arrive(_Req("r", prompt_len=64))
        req.prefix_len = 48
        self.queue.poll()
        self.assertFalse(req.skip_radix_cache_insert)

        req.prefix_len = 64
        self.queue.poll()

        self.assertTrue(req.skip_radix_cache_insert)

    def _block(self, block, *, prompt_len, chain_id="c", first_id=0):
        """Block ``block`` of a chain of 8-token blocks where Sandy writes
        nothing: each block reads the taps of the context before it.
        """
        answer_start = prompt_len - (block - 1) * 8
        return _Req(
            f"{chain_id}-{block}",
            prompt_len=prompt_len,
            first_id=first_id,
            chain=FrankyChain(id=chain_id, block=block),
            tap_runs=[
                [start, start] for start in range(answer_start, prompt_len + 1, 8)
            ],
        )

    def test_a_block_that_calls_sandy_holds_its_prompt_and_taps_for_the_next(self):
        self._run(self._block(1, prompt_len=64), _SANDY_CALL)

        self.assertEqual(self.tree.locked, ["node-c-1"])
        self.assertEqual(self.tap_cache.num_takeable_stripes(), _NUM_STRIPES - 4)
        self.assertEqual(self._free_and_evictable(), (_NUM_STRIPES - 4, 0))
        self.assertTrue(self.queue.is_empty())

    def test_the_next_block_reads_the_held_taps_and_lets_the_hold_go_once_it_runs(
        self,
    ):
        first_pages = self._run(self._block(1, prompt_len=64), _SANDY_CALL)

        second = self._arrive(self._block(2, prompt_len=72))

        receiver = self.receivers[-1]
        pages = self._pages("c-2")
        self.assertEqual(pages[:4], first_pages)
        self.assertEqual(receiver.prefix_len, 64)
        self.assertEqual(receiver.tap_pages, pages[4:])
        self.assertEqual(second.franky_taps.taps_ready, 64)
        self.assertEqual(
            second.franky_taps.rows_ready(),
            71,
            "rows up to 70 make the first block's tokens, which read 64 taps",
        )
        self.assertEqual(self.tree.locked, ["node-c-1"])

        self.tap_requests.join_batch([second])

        self.assertEqual(self.tree.locked, [])

    def test_a_chains_last_request_holds_until_the_router_ends_the_chain(self):
        for chain_id, first_id, reason in (
            ("s", 1000, FINISH_MATCHED_TOKEN(matched=0)),
            ("l", 2000, FINISH_LENGTH(length=8)),
        ):
            with self.subTest(chain_id=chain_id):
                self._run(
                    self._block(1, prompt_len=64, chain_id=chain_id, first_id=first_id),
                    _SANDY_CALL,
                )
                self._run(
                    self._block(2, prompt_len=72, chain_id=chain_id, first_id=first_id),
                    reason,
                )
                self.assertEqual(self.tree.locked, [f"node-{chain_id}-2"])

                self.holds.release(chain_id=chain_id)

                self.assertEqual(self.tree.locked, [])
                self.assertEqual(self.tap_cache.num_takeable_stripes(), _NUM_STRIPES)

    def test_a_new_arrival_short_of_stripes_waits_and_the_holds_stay(self):
        self._run(
            self._block(1, prompt_len=96, chain_id="a", first_id=1000),
            _SANDY_CALL,
        )
        self._run(
            self._block(1, prompt_len=96, chain_id="b", first_id=2000),
            _SANDY_CALL,
        )

        req = self._arrive(_Req("r", prompt_len=96, first_id=3000))

        self.assertTrue(self.tap_requests.waits("r"))
        self.assertEqual(self.failed, [])
        self.assertEqual(self.tree.locked, ["node-a-1", "node-b-1"])
        self.assertEqual(req.franky_taps.rows_ready(), 95)

    def test_a_held_chains_next_block_short_of_stripes_releases_another(self):
        self._run(
            self._block(3, prompt_len=96, chain_id="late", first_id=1000),
            _SANDY_CALL,
        )
        self._run(
            self._block(1, prompt_len=96, chain_id="early", first_id=2000),
            _SANDY_CALL,
        )
        self._run(self._block(1, prompt_len=32), _SANDY_CALL)

        second = self._arrive(self._block(2, prompt_len=72))

        self.assertIsNotNone(self.tap_requests.reservation("c-2"))
        self.assertEqual(second.franky_taps.taps_ready, 32)
        self.assertEqual(
            self.tree.locked, ["node-late-3", "node-c-1"], "the fewest blocks go"
        )

    def test_held_chains_next_blocks_take_stripes_before_earlier_arrivals(self):
        self._run(self._block(1, prompt_len=16), _SANDY_CALL)
        new = _Req("new", prompt_len=16, first_id=1000)
        block = self._block(2, prompt_len=24)
        self.queue.add(req=new)
        self.queue.add(req=block)

        self.queue.poll()

        self.assertEqual(list(self.queue.receptions), ["c-2", "new"])

    def test_a_block_aborted_while_waiting_releases_its_chains_hold(self):
        self._run(self._block(1, prompt_len=64), _SANDY_CALL)
        self._arrive(_Req("big", prompt_len=192, first_id=1000))
        second = self._arrive(self._block(2, prompt_len=72))
        self.assertTrue(self.tap_requests.waits("c-2"))

        self.queue.forget(req=second)

        self.assertEqual(self.holds.locked_nodes(), set())
        self.assertEqual(self._free_and_evictable(), (0, 4))

    def _with_mamba_room(self, *, num_states):
        pool = SimpleNamespace(
            size=4,
            available_size=lambda: 4,
            mamba_allocator=SimpleNamespace(size=num_states),
        )
        self.queue = self._queue_on(
            shared_tokenizer=True,
            mamba_room=FrankyMambaRoom(req_to_token_pool=pool, states_per_request=5),
        )

    def test_holds_past_the_mamba_room_go_least_progress_first(self):
        self._with_mamba_room(num_states=7)
        for chain_id, block in (("a", 3), ("b", 1), ("c", 2)):
            self.holds.hold_for_next_request(
                _held(chain_id, node=f"held-{chain_id}", num_tokens=16, block=block)
            )

        self.queue.poll()

        self.assertEqual(self.tree.locked, ["held-a", "held-c"])

    def test_only_chain_holds_pin_mamba_states(self):
        self._with_mamba_room(num_states=22)
        self.holds.hold_for_next_request(_held("c", node="held-c", num_tokens=16))
        for rid in ("a", "b"):
            self._arrive(_Req(rid, prompt_len=64, first_id=1000))

        self.assertEqual(self.tap_requests.num_startable_requests(), 4)

    def test_an_engine_without_a_mamba_pool_sets_no_limit(self):
        self._arrive(_Req("r", prompt_len=64))

        self.assertIsNone(self.tap_requests.num_startable_requests())


class TestRickySchedulerTaps(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tap_cache = _tap_cache(num_ranks=2, rank=1)
        self.request_table = TapRequestTable(num_slots=4, max_stripes=8, device="cpu")
        chain_holds = FrankyChainHolds(
            tree_cache=None,
            tap_cache=self.tap_cache,
            hold_seconds=60,
        )
        self.tap_requests = RickyTapRequests(
            tap_cache=self.tap_cache,
            request_table=self.request_table,
            chain_holds=chain_holds,
            shared_tokenizer=True,
        )
        self.scheduler = object.__new__(RickyScheduler)
        self.scheduler.franky_taps = RickyTaps(
            cache=self.tap_cache,
            chain_holds=chain_holds,
            requests=self.tap_requests,
            pool_metrics=None,
        )

    def _reserved(self, rid, *, row, num_taps):
        req = _Req(rid, prompt_len=num_taps, row=row)
        req.disagg_kv_sender = None
        self.tap_requests.arrive(req)
        self.tap_requests.poll()
        return req

    def test_a_prefill_batch_binds_each_requests_stripes_and_spans(self):
        reading = self._reserved("reading", row=2, num_taps=100)
        plain = _Req("plain", prompt_len=8, room=False, row=1)
        self.tap_requests.arrive(plain)
        self.scheduler.franky_parked_chunks = FrankyParkedChunks(
            tap_limit=PrefillTapLimit(chunk_grid=1)
        )
        self.scheduler.chunked_req = None
        self.scheduler.waiting_queue = []
        idle = SimpleNamespace(is_empty=lambda: True)

        with patch.object(
            Scheduler,
            "_get_new_batch_prefill_raw",
            return_value=(SimpleNamespace(reqs=[reading, plain]), idle),
        ):
            self.scheduler._get_new_batch_prefill_raw(None, idle)

        pages = self.tap_requests.reservation("reading").pages.tolist()
        self.assertEqual(self.request_table.stripe_pages[2, :4].tolist(), pages)
        self.assertEqual(self.request_table.stripe_pages[1].tolist(), [0] * 8)
        self.assertEqual(self.request_table.decode_taps.tolist(), [0, 0, 100, 0])
        self.assertIs(self.request_table.tap_runs_of(2), reading.franky_taps.tap_runs)
        self.assertIsNone(self.request_table.tap_runs_of(1))

    def test_pd_prefill_relays_the_taps_once_they_all_landed(self):
        req = self._reserved("r", row=0, num_taps=100)
        relayed = []
        req.disagg_kv_sender = SimpleNamespace(
            relays_taps=False,
            relay_taps=lambda *, tap_pages: relayed.append(tap_pages.tolist()),
        )

        self.scheduler._relay_landed_taps(req)
        self.assertEqual(relayed, [])

        req.franky_taps.taps_ready = 100
        self.scheduler._relay_landed_taps(req)

        pages = self.tap_requests.reservation("r").pages.tolist()
        self.assertEqual(
            relayed,
            [[pages[page // 2] if page % 2 == 1 else -1 for page in range(7)]],
        )

    def test_a_request_without_taps_relays_nothing(self):
        sender = MagicMock(relays_taps=False)
        req = SimpleNamespace(
            franky_taps=TapProgress(tap_len=0, taps_ready=0), disagg_kv_sender=sender
        )

        self.scheduler._relay_landed_taps(req)

        sender.relay_taps.assert_not_called()


class TestFrankyStartableRequests(CustomTestCase):
    def test_pinned_mamba_states_cap_the_requests_a_prefill_may_start(self):
        for scheduler_class in (RickyScheduler, SandyScheduler):
            scheduler = object.__new__(scheduler_class)
            with (
                self.subTest(scheduler=scheduler_class.__name__),
                patch.object(Scheduler, "get_num_allocatable_reqs", return_value=10),
            ):
                scheduler.franky_num_startable_reqs = 2
                self.assertEqual(
                    scheduler.get_num_allocatable_reqs(0, num_held_rows=1), 3
                )
                scheduler.franky_num_startable_reqs = None
                self.assertEqual(
                    scheduler.get_num_allocatable_reqs(0, num_held_rows=1), 10
                )


class TestSandyHeldChains(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tree = _Tree()
        self.holds = FrankyChainHolds(
            tree_cache=self.tree, tap_cache=None, hold_seconds=60
        )
        self.holds.hold_for_next_request(_held("asker", node="asker", num_tokens=16))
        self.holds.hold_for_next_request(
            _held("far", node="far", num_tokens=16, block=4)
        )
        self.holds.hold_for_next_request(_held("near", node="near", num_tokens=16))
        self.new = SimpleNamespace(franky=None)
        self.block = SimpleNamespace(
            franky=FrankyRequest(chain=FrankyChain(id="asker", block=2))
        )

    def _schedule_nothing(self, *, waiting):
        sandy = object.__new__(SandyScheduler)
        sandy.waiting_queue = waiting
        sandy.chunked_req = None
        sandy.franky_chain_holds = self.holds
        sandy.franky_mamba_room = None
        idle = SimpleNamespace(is_empty=lambda: True)
        with patch.object(
            Scheduler, "_get_new_batch_prefill_raw", return_value=(None, idle)
        ):
            sandy._get_new_batch_prefill_raw(None, idle)
        return sandy

    def test_a_stuck_block_of_a_held_chain_releases_the_least_progress(self):
        sandy = self._schedule_nothing(waiting=[self.new, self.block])

        self.assertEqual(sandy.waiting_queue, [self.block, self.new])
        self.assertEqual(self.tree.locked, ["asker", "far"])

    def test_stuck_new_requests_release_no_hold(self):
        self._schedule_nothing(waiting=[self.new])

        self.assertEqual(self.tree.locked, ["asker", "far", "near"])


class TestFrankyPrefillAdder(CustomTestCase):
    def setUp(self):
        super().setUp()
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.tree_cache = MagicMock()
        self.tree_cache.evictable_size.return_value = 0
        self.tree_cache.full_evictable_size.return_value = 0
        self.tree_cache.supports_mamba.return_value = False
        self.tree_cache.disable = False
        self.tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        self.tree_cache.dec_lock_ref.return_value = DecLockRefResult()
        self.tree_cache.lock.side_effect = lambda node: TreeLock(
            node, self.tree_cache.inc_lock_ref(node).to_dec_params()
        )
        self.tree_cache.buffer_pipeline = None
        self.allocator = MagicMock()
        self.allocator.page_size = _PAGE_SIZE
        self.allocator.available_size.return_value = 100_000
        self.allocator.request_slot_reserve.return_value = 0
        self.allocator.create_prefill_budget.side_effect = lambda tree_cache, **kwargs: (
            PrefillBudget(self.allocator, tree_cache, **kwargs)
        )

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

    def _req(self, rid, *, prompt_len, prefix_len=0, taps_ready, tap_runs=None):
        req = MagicMock(spec=Req)
        type(req).extend_len = Req.extend_len
        req.rid = rid
        req.kv = SimpleNamespace(req_pool_idx=None, holds_mamba=False)
        req.cache_request_handle = CacheRequestHandle(rid, 0)
        req.last_node = f"node-{rid}"
        req.prefix_len = prefix_len
        req.extend_end = None
        req.full_untruncated_fill_ids = list(range(prompt_len))
        req.origin_input_ids = list(range(prompt_len))
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
        req.franky_taps = TapProgress(
            tap_len=prompt_len,
            taps_ready=taps_ready,
            tap_runs=tap_runs or _single_turn(prompt_len),
        )
        return req

    def test_a_first_chunk_runs_every_row_that_reads_no_unlanded_tap(self):
        adder = self._adder()
        req = self._req("r", prompt_len=100, taps_ready=0)

        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

        self.assertEqual(req.extend_end, 99)
        self.assertIs(adder.new_chunked_req, req)
        self.assertFalse(adder.exact_chunk_fill)

    def test_a_request_whose_next_row_reads_unlanded_taps_waits(self):
        adder = self._adder()
        waiting = self._req("waiting", prompt_len=100, prefix_len=99, taps_ready=64)

        result = adder.add_one_req(
            waiting, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(adder.can_run_list, [])

    def test_a_request_whose_taps_all_landed_runs_whole(self):
        adder = self._adder()
        req = self._req("r", prompt_len=100, taps_ready=100)

        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

        self.assertEqual(adder.can_run_list, [req])
        self.assertIsNone(adder.new_chunked_req)

    def test_kv_memory_bounds_a_request_whose_taps_all_landed(self):
        self.allocator.available_size.return_value = 32
        adder = self._adder()
        req = self._req("r", prompt_len=100, taps_ready=100)

        result = adder.add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )

        self.assertNotEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(adder.can_run_list, [])

    def test_while_a_chunked_request_goes_on_only_a_whole_request_joins(self):
        adder = self._adder()
        partial = self._req("partial", prompt_len=100, taps_ready=0)
        whole = self._req("whole", prompt_len=40, taps_ready=40)

        for req in (partial, whole):
            adder.add_one_req(req, has_chunked_req=True, truncation_align_size=None)

        self.assertEqual(adder.can_run_list, [whole])
        self.assertIsNone(adder.new_chunked_req)

    def test_with_sandys_own_tokens_a_chunk_ends_before_the_first_unlanded_read(
        self,
    ):
        self.tree_cache.supports_mamba.return_value = True
        self.tree_cache.supports_prefix_sharing.return_value = True
        adder = self._adder()
        req = self._own_tokens_req()

        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

        self.assertEqual(req.franky_taps.rows_ready(), 99)
        self.assertEqual(req.extend_end, 96)
        self.assertIs(adder.new_chunked_req, req)

    def test_without_mamba_a_first_chunk_runs_every_row_its_taps_allow(self):
        adder = self._adder()
        req = self._own_tokens_req()

        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

        self.assertEqual(req.extend_end, 99)
        self.assertIs(adder.new_chunked_req, req)
        self.assertFalse(adder.exact_chunk_fill)

    def _own_tokens_req(self, *, prefix_len=0, taps_ready=30):
        return self._req(
            "r",
            prompt_len=100,
            prefix_len=prefix_len,
            taps_ready=taps_ready,
            tap_runs=FrankyTapRuns.of_runs(
                [[40, 30], [60, 0], [100, 80]], num_input_tokens=100, num_taps=80
            ),
        )

    def test_a_continuing_chunk_ends_at_the_rows_its_landed_taps_allow(self):
        adder = self._adder()
        req = self._own_tokens_req(prefix_len=32, taps_ready=29)

        self.assertIs(adder.add_chunked_req(req), req)
        self.assertEqual(req.extend_end, 39)

    def test_with_mamba_rows_short_of_a_page_wait_for_the_last_rows_taps(self):
        # A chunk that ends inside a page would leave no mamba checkpoint and
        # take one more forward; the rows wait and run with the last one.
        self.tree_cache.supports_mamba.return_value = True
        self.tree_cache.supports_prefix_sharing.return_value = True
        adder = self._adder()
        req = self._req("r", prompt_len=100, prefix_len=96, taps_ready=0)

        self.assertIs(adder.add_chunked_req(req), req)
        self.assertEqual(adder.can_run_list, [])

        req.franky_taps.taps_ready = 100
        adder = self._adder()
        adder.add_chunked_req(req)
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(req.extend_end, 100)

    def test_the_prefill_delayer_counts_only_requests_with_rows_to_run(self):
        scheduler = object.__new__(RickyScheduler)
        scheduler.franky_tap_limit = PrefillTapLimit(chunk_grid=1)
        rows = dict(origin_input_ids=[0] * 64, output_ids=[])
        scheduler.waiting_queue = [
            SimpleNamespace(
                franky_taps=TapProgress(
                    tap_len=64, taps_ready=0, tap_runs=_single_turn(64)
                ),
                prefix_len=0,
                **rows,
            ),
            SimpleNamespace(
                franky_taps=TapProgress(
                    tap_len=64, taps_ready=32, tap_runs=_single_turn(64)
                ),
                prefix_len=63,
                **rows,
            ),
            SimpleNamespace(franky_taps=None, prefix_len=0, **rows),
        ]

        self.assertEqual(scheduler.prefillable_queue_len(), 1)


if __name__ == "__main__":
    unittest.main()
