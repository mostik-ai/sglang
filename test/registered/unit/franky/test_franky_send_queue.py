"""Sandy sends a tap only once its forward ended and the prefix tree took its KV
page, and keeps a request, with its KV and taps, until every rank and stage
agrees its sends ended. A tap sent early can be overwritten before Ricky reads
it; a request freed early can lose its pages under a running send.
"""

import dataclasses
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.franky.cache.sandy_tap_cache import SandyTapCache
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.franky.chains.chain import FrankyChain
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.request import FrankyRequest
from sglang.srt.franky.schedulers.sandy_batch_result_processor import (
    SandyBatchResultProcessor,
)
from sglang.srt.franky.transfer import send_queue as send_queue_module
from sglang.srt.franky.transfer.landed_pages import FrankyLandedPages
from sglang.srt.franky.transfer.messages import (
    FrankyRoom,
    FrankySendDone,
    FrankyTapsReady,
)
from sglang.srt.franky.transfer.rank_reports import FrankyRankReports
from sglang.srt.franky.transfer.send_queue import FrankySendQueue
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.scheduler_components import (
    batch_result_processor as core_processor_module,
)
from sglang.srt.managers.utils import is_health_check_generate_req
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_PAGE_SIZE = 4
_ROW = 1
# Request row 1 holds KV slots on pages 5, 2, 7, so pages are not in order.
_ROW_SLOTS = [20, 21, 22, 23, 8, 9, 10, 11, 28, 29]
_ROOM = FrankyRoom(bootstrap_host="sandy", bootstrap_port=8996, room=7)


def _pages(start, end):
    return [slot // _PAGE_SIZE for slot in _ROW_SLOTS[start:end:_PAGE_SIZE]]


class _Sender:
    def __init__(self):
        self.landed = []
        self.state = KVPoll.Transferring
        self.aborted = False

    def land(self, *, tap_pages):
        self.landed.append(tap_pages.tolist())

    def poll(self):
        return KVPoll.Failed if self.aborted else self.state

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True


class _LockTree:
    def __init__(self):
        self.locked = []

    def lock(self, node):
        self.locked.append(node)
        return node

    def unlock(self, lock):
        self.locked.remove(lock)


class _PPStore(dict):
    """Stage 0's view: other stages' reports, set by the test."""

    def __init__(self):
        super().__init__()
        self.other_stages = {}

    def collect(self, key):
        return [self.get(key), self.other_stages.get(key)]


def _sandy_cache(*, page_size=_PAGE_SIZE, num_ranks=1, rank=0) -> SandyTapCache:
    return SandyTapCache.build(
        encoder_ids=(0,),
        output_size=2,
        page_size=page_size,
        num_kv_slots=32 + _PAGE_SIZE,
        stripes=TapStripes(num_ranks=num_ranks, rank=rank),
        device="cpu",
    )


def _req(rid="r", prompt_len=10, chain=None, max_new_tokens=0, room=True, kept=None):
    """A Sandy request; one that writes keeps the KV of ``kept`` tokens."""
    return SimpleNamespace(
        rid=rid,
        franky=FrankyRequest(
            room=_ROOM if room else None,
            chain=chain,
            taps_only=room and not max_new_tokens,
        ),
        last_node=f"node-{rid}",
        origin_input_ids=list(range(prompt_len)),
        prefix_len=0,
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
        kv=SimpleNamespace(req_pool_idx=_ROW),
        finished_reason=None,
        to_finish=None,
        owned_kv_len=lambda: prompt_len if kept is None else kept,
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


class _Queue(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _init_single_process_group()

    def setUp(self):
        super().setUp()
        req_to_token = torch.zeros((2, 16), dtype=torch.int32)
        req_to_token[_ROW, : len(_ROW_SLOTS)] = torch.tensor(_ROW_SLOTS)
        self.sender = _Sender()
        self.opened = []
        self.released = []
        self.store = _PPStore()
        self.lock_tree = _LockTree()
        release = patch.object(
            send_queue_module,
            "release_kv_cache",
            lambda req, tree_cache, checkpoint: self.released.append(
                (req.rid, checkpoint)
            ),
        )
        release.start()
        self.addCleanup(release.stop)
        self.queue = FrankySendQueue(
            tap_cache=_sandy_cache(),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            tree_cache=None,
            rank_reports=FrankyRankReports(
                attn_tp_cpu_group=dist.group.WORLD,
                attn_cp_cpu_group=dist.group.WORLD,
                pp_store=self.store,
            ),
            chain_holds=FrankyChainHolds(
                tree_cache=self.lock_tree, tap_cache=None, hold_seconds=60
            ),
            page_size=_PAGE_SIZE,
            open_room=self._open_room,
        )

    def _open_room(self, req, num_tokens):
        self.opened.append((req.rid, num_tokens))
        return self.sender

    def _hold(self, req):
        self.queue.add(req=req)
        req.prefix_len = len(req.origin_input_ids)
        self.queue.after_end(req=req, checkpoint=True)

    def _step(self):
        """One scheduler step: every rank polls, the leader announces, every
        rank applies the announcements.
        """
        self.queue.poll()
        messages = self.queue.ready_messages()
        for message in messages:
            self.queue.on_send_done(message=message)
        return messages


class TestLanding(_Queue):
    def test_taps_land_only_as_far_as_the_tree_holds_them(self):
        """The tree takes a chunk when the next one is scheduled: before the
        chunk's result with the overlap scheduler, after it without.
        """
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                self.sender.landed.clear()
                req = _req(rid=f"r{overlap}")
                self.queue.add(req=req)
                # The tree held tokens 0..4; chunk 1 runs 4..8, chunk 2 runs 8..10.
                req.prefix_len = 8 if overlap else 4
                self.queue.after_chunk(req=req, end=8)
                sent = req.prefix_len
                self.assertEqual(self.sender.landed, [_pages(0, sent)])
                req.prefix_len = 10
                self.queue.after_end(req=req, checkpoint=True)
                self.assertEqual(self.sender.landed[-1], _pages(sent, 10))

    def test_taps_never_land_past_the_forward(self):
        req = _req()
        self.queue.add(req=req)
        req.prefix_len = 10

        self.queue.after_chunk(req=req, end=4)

        self.assertEqual(self.sender.landed, [_pages(0, 4)])

    def test_a_middle_chunk_lands_whole_pages_only(self):
        req = _req()
        self.queue.add(req=req)
        req.prefix_len = 7

        self.queue.after_chunk(req=req, end=7)

        self.assertEqual(self.sender.landed, [_pages(0, 4)])

    def test_pages_still_copying_to_the_host_land_later_in_order(self):
        """The copy of a chunk's pages waits behind the forward in flight; the
        scheduler must not block on it, and the sender must get pages in prompt
        order.
        """
        copied = {"ready": False}
        ready = patch.object(
            send_queue_module.FrankyHostPages, "ready", lambda self: copied["ready"]
        )
        ready.start()
        self.addCleanup(ready.stop)
        req = _req()
        self.queue.add(req=req)
        req.prefix_len = 8
        self.queue.after_chunk(req=req, end=8)
        req.prefix_len = 10
        self.queue.after_end(req=req, checkpoint=True)
        self._step()
        self.assertEqual(self.sender.landed, [])

        copied["ready"] = True
        self._step()

        self.assertEqual(self.sender.landed, [_pages(0, 8), _pages(8, 10)])

    def test_a_write_opens_its_room_when_it_ends_with_every_token_it_ran(self):
        """Ricky's leg reads the write's text, so its room opens at its end."""
        req = _req(max_new_tokens=5, kept=10)
        self.queue.add(req=req)
        self.assertEqual(self.opened, [])

        self.queue.after_end(req=req, checkpoint=False)

        self.assertEqual(self.opened, [("r", 10)])
        self.assertEqual(self.sender.landed, [_pages(0, 10)])

    def test_a_request_without_a_room_never_joins(self):
        req = _req(room=False)

        self.queue.add(req=req)

        self.assertTrue(self.queue.is_empty())


class TestLandingInTapPages(_Queue):
    def test_a_kv_page_lands_as_its_tap_pages(self):
        """With tap pages smaller than Sandy's KV pages, each KV page lands as the
        tap pages it holds, in prompt order."""
        self.queue._tap_cache = _sandy_cache(page_size=_PAGE_SIZE // 2)
        req = _req()
        self.queue.add(req=req)
        req.prefix_len = 10

        self.queue.after_end(req=req, checkpoint=True)

        self.assertEqual(self.sender.landed, [[10, 11, 4, 5, 14]])


class TestLandingStripedTaps(_Queue):
    def test_each_rank_lands_every_page_marking_the_pages_other_ranks_keep(self):
        landed = []
        for rank in range(2):
            self.sender.landed = []
            self.queue._tap_cache = _sandy_cache(num_ranks=2, rank=rank)
            req = _req(rid=f"r{rank}")
            self.queue.add(req=req)
            req.prefix_len = 10

            self.queue.after_end(req=req, checkpoint=True)

            landed.append(self.sender.landed)

        self.assertEqual(landed, [[[-1, 1, -1]], [[2, -1, 3]]])


class TestRelease(_Queue):
    def test_a_request_is_never_released_before_it_ends(self):
        req = _req()
        self.queue.add(req=req)
        req.prefix_len = 8
        self.queue.after_chunk(req=req, end=8)
        self.sender.state = KVPoll.Success
        self.store.other_stages["r"] = KVPoll.Success

        self.assertEqual(self._step(), [])
        self.assertEqual(self.released, [])

    def test_a_held_request_waits_for_every_rank_and_stage(self):
        req = _req()
        self._hold(req)

        self.assertEqual(self._step(), [])
        self.sender.state = KVPoll.Success
        self.assertEqual(self._step(), [], "stage 1 has not reported yet")
        self.store.other_stages["r"] = KVPoll.Success
        self.assertEqual(len(self._step()), 1)

        self.assertEqual(self.released, [("r", True)])
        self.assertTrue(self.queue.is_empty())
        self.assertNotIn("r", self.store)

    def test_a_release_keeps_the_kv_as_the_request_ended_asking(self):
        req = _req(max_new_tokens=4, kept=10)
        self.queue.add(req=req)
        self.queue.after_end(req=req, checkpoint=False)
        self.sender.state = KVPoll.Success
        self.store.other_stages["r"] = KVPoll.Success

        self._step()

        self.assertEqual(self.released, [("r", False)])

    def test_a_sent_request_of_a_chain_holds_its_prompt_for_the_next(self):
        req = _req(chain=FrankyChain(id="chain"))
        self._hold(req)
        self.sender.state = KVPoll.Success
        self.store.other_stages["r"] = KVPoll.Success

        self._step()

        self.assertEqual(self.lock_tree.locked, ["node-r"])

    def _send_block(self, rid, chain, state):
        req = _req(rid, chain=chain)
        self._hold(req)
        self.sender.state = state
        self.store.other_stages[rid] = KVPoll.Success
        self._step()

    def test_a_chains_next_request_takes_over_its_hold(self):
        self._send_block("c-1", FrankyChain(id="c"), KVPoll.Success)
        self.assertEqual(self.lock_tree.locked, ["node-c-1"])

        self._send_block("c-2", FrankyChain(id="c", block=2), KVPoll.Success)

        self.assertEqual([rid for rid, _ in self.released], ["c-1", "c-2"])
        self.assertEqual(self.lock_tree.locked, ["node-c-2"])

    def test_a_failed_send_frees_the_request_and_releases_its_chains_hold(self):
        self._send_block("c-1", FrankyChain(id="c"), KVPoll.Success)

        self._send_block("c-2", FrankyChain(id="c", block=2), KVPoll.Failed)

        self.assertEqual([rid for rid, _ in self.released], ["c-1", "c-2"])
        self.assertEqual(self.lock_tree.locked, [])

    def test_aborting_a_held_request_fails_its_send(self):
        req = _req()
        self._hold(req)
        self.store.other_stages["r"] = KVPoll.Success

        self.queue.abort(recv_req=AbortReq(rid="r"))
        self._step()

        self.assertTrue(self.sender.aborted)
        self.assertEqual(self.released, [("r", True)])
        self.assertTrue(self.queue.is_empty())

    def test_a_request_aborted_before_it_ended_is_forgotten(self):
        """The scheduler frees it; the queue only stops its sends."""
        req = _req()
        self.queue.add(req=req)

        self.queue.forget(req=req)
        req.prefix_len = 8
        self.queue.after_chunk(req=req, end=8)

        self.assertTrue(self.sender.aborted)
        self.assertEqual(self.sender.landed, [])
        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self.released, [])

    def test_a_release_announced_twice_is_applied_once(self):
        req = _req()
        self._hold(req)
        self.sender.state = KVPoll.Success
        self.store.other_stages["r"] = KVPoll.Success
        self.queue.poll()
        [message] = self.queue.ready_messages()

        self.queue.on_send_done(message=message)
        self.queue.on_send_done(message=message)

        self.assertEqual(self.released, [("r", True)])
        self.assertEqual(self.queue.ready_messages(), [])


class TestLandedPages(CustomTestCase):
    def test_taps_land_only_once_every_sending_rank_wrote_them(self):
        """Ricky reads taps up to the slowest sending rank; a rank that has not
        written yet holds every page back.
        """
        landed = FrankyLandedPages()
        landed.open(room=7)

        landed.record(room=7, sender=0, page_end=3)
        self.assertEqual(landed.pages(room=7, num_senders=2), 0)
        landed.record(room=7, sender=1, page_end=2)
        self.assertEqual(landed.pages(room=7, num_senders=2), 2)

    def test_a_late_message_does_not_reopen_a_closed_room(self):
        landed = FrankyLandedPages()
        landed.open(room=7)
        landed.close(room=7)

        landed.record(room=7, sender=0, page_end=3)

        self.assertEqual(landed.page_ends, {})


class TestMessages(CustomTestCase):
    def test_a_message_about_a_health_check_is_not_taken_for_one(self):
        """The scheduler drops health checks while busy, and a held request keeps
        Sandy busy, so its release would never come.
        """
        rid = "HEALTH_CHECK_1"
        for message in (
            FrankySendDone(request_rid=rid),
            FrankyTapsReady(request_rid=rid, taps_ready=1),
        ):
            with self.subTest(type(message).__name__):
                self.assertFalse(is_health_check_generate_req(message))


class TestSandyBatchResultProcessor(_Queue):
    def setUp(self):
        super().setUp()
        self.base_released = []
        base_prefill = patch.object(
            core_processor_module.SchedulerBatchResultProcessor,
            "process_batch_result_prefill",
            self._base_prefill,
        )
        base_release = patch.object(
            core_processor_module,
            "release_kv_cache",
            lambda req, tree_cache, checkpoint: self.base_released.append(req.rid),
        )
        for patcher in (base_prefill, base_release):
            patcher.start()
            self.addCleanup(patcher.stop)
        fields = {
            field.name: None for field in dataclasses.fields(SandyBatchResultProcessor)
        }
        fields.update(
            franky_send_queue=self.queue,
            franky_chain_holds=self.queue._chain_holds,
        )
        self.processor = SandyBatchResultProcessor(**fields)

    @staticmethod
    def _base_prefill(processor, batch, result):
        """SGLang's prefill result: a middle chunk counts down, a last chunk ends
        a request that writes nothing through the release step."""
        for req in batch.reqs:
            if req.inflight_middle_chunks > 0:
                req.inflight_middle_chunks -= 1
            else:
                req.finished_reason = FINISH_LENGTH(length=0)
                processor.release_finished_kv(req, checkpoint=True)

    def test_a_chunk_lands_its_taps_and_an_ended_request_is_held(self):
        senders = {}
        self.queue._open_room = lambda req, num_tokens: senders.setdefault(
            req.rid, _Sender()
        )
        middle, last = _req("middle"), _req("last")
        middle.inflight_middle_chunks, last.inflight_middle_chunks = 1, 0
        # The overlap scheduler: the tree took the middle chunk before its result.
        middle.prefix_len, last.prefix_len = 8, 10
        for req in (middle, last):
            self.queue.add(req=req)
        batch = SimpleNamespace(
            reqs=[middle, last], prefix_lens=[0, 4], extend_lens=[8, 6]
        )

        self.processor.process_batch_result_prefill(batch, result=None)

        self.assertEqual(senders["middle"].landed, [_pages(0, 8)])
        self.assertEqual(senders["last"].landed, [_pages(0, 10)])
        self.assertEqual(self.base_released, [], "held until its taps are sent")
        self.assertFalse(self.queue.is_empty())

    def test_a_request_without_a_room_or_aborted_is_released_as_usual(self):
        plain = _req("plain", room=False, chain=FrankyChain(id="plain"))
        aborted = _req("aborted")
        aborted.finished_reason = FINISH_ABORT("gone")

        for req in (plain, aborted):
            self.processor.release_finished_kv(req, checkpoint=True)

        self.assertEqual(self.base_released, ["plain", "aborted"])
        self.assertEqual(self.lock_tree.locked, ["node-plain"])
        self.assertTrue(self.queue.is_empty())


if __name__ == "__main__":
    unittest.main()
