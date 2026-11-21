import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl.conn import NixlKVSender
from sglang.srt.franky.cache.sandy_tap_cache import SandyTapCache
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.srt.franky.schedulers.sandy_prefill import SandyPrefillScheduler
from sglang.srt.franky.transfer.relay.sandy_target import SandyRelayTarget
from sglang.srt.franky.transfer.relay.sender import FrankyRelaySender
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_TAP_PAGE_SIZE = 16
_KV_PAGE_SIZE = 64


def _tap_cache(*, rank):
    return SandyTapCache.build(
        encoder_ids=(0,),
        output_size=8,
        page_size=_TAP_PAGE_SIZE,
        num_kv_slots=4096,
        stripes=TapStripes(num_ranks=2, rank=rank),
        device="cpu",
    )


class TestSandyRelayTarget(CustomTestCase):
    def test_a_rooms_taps_land_at_the_tap_pages_of_its_kv_pages_this_rank_keeps(self):
        for rank, expected in (
            (0, [6, -1, 7, -1, 14, -1, 15, -1]),
            (1, [-1, 6, -1, 7, -1, 14, -1, 15]),
        ):
            with self.subTest(rank=rank):
                target = SandyRelayTarget(
                    tap_cache=_tap_cache(rank=rank), kv_page_size=_KV_PAGE_SIZE
                )

                landing = target.landing(
                    room=7, kv_pages=np.array([3, 7], dtype=np.int32), prefix_len=64
                )

                self.assertEqual(landing.tap_pages.tolist(), expected)
                self.assertEqual(landing.prefix_len, 64)
                self.assertFalse(target.waits(room=7))
                self.assertIsNone(target.refusal(room=7))


class TestSandyPrefillRelay(CustomTestCase):
    def _scheduler(self):
        scheduler = SandyPrefillScheduler.__new__(SandyPrefillScheduler)
        req_to_token = torch.zeros((1, 512), dtype=torch.int64)
        req_to_token[0, :130] = torch.arange(192, 322)
        scheduler.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        scheduler.page_size = _KV_PAGE_SIZE
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                franky_parts=SimpleNamespace(tap_cache=_tap_cache(rank=0))
            )
        )
        return scheduler

    def _req(self, sender):
        return SimpleNamespace(
            disagg_kv_sender=sender,
            kv=SimpleNamespace(req_pool_idx=0),
            origin_input_ids=list(range(130)),
        )

    def test_the_last_chunk_relays_the_tap_pages_of_the_prompts_kv_pages_once(self):
        scheduler = self._scheduler()
        sender = FrankyRelaySender.__new__(FrankyRelaySender)
        sender.tap_sender = None
        sender.relay_taps = MagicMock(
            side_effect=lambda tap_pages: setattr(sender, "tap_sender", object())
        )
        req = self._req(sender)

        scheduler._relay_taps(req)
        scheduler._relay_taps(req)

        [call] = sender.relay_taps.call_args_list
        self.assertEqual(
            call.kwargs["tap_pages"].tolist(),
            [6, -1, 7, -1, 8, -1, 9, -1, 10, -1, 11, -1],
        )

    def test_a_request_without_a_relay_sends_nothing(self):
        scheduler = self._scheduler()
        fake_sender = MagicMock(spec=["poll", "relay_taps"])

        scheduler._relay_taps(self._req(fake_sender))

        fake_sender.relay_taps.assert_not_called()


class TestFrankyRelaySenderPoll(CustomTestCase):
    def _poll(self, *, kv, taps):
        sender = FrankyRelaySender.__new__(FrankyRelaySender)
        sender.tap_sender = None if taps is None else SimpleNamespace(poll=lambda: taps)
        with patch.object(NixlKVSender, "poll", return_value=kv):
            return sender.poll()

    def test_taps_waiting_for_the_decodes_receiver_keep_the_request_in_flight(self):
        for kv in (KVPoll.WaitingForInput, KVPoll.Transferring, KVPoll.Success):
            with self.subTest(kv=kv):
                self.assertEqual(
                    self._poll(kv=kv, taps=KVPoll.Bootstrapping),
                    KVPoll.WaitingForInput,
                )

    def test_the_request_ends_as_the_less_advanced_transfer(self):
        cases = (
            (KVPoll.Success, KVPoll.Transferring, KVPoll.Transferring),
            (KVPoll.Success, KVPoll.Success, KVPoll.Success),
            (KVPoll.Success, KVPoll.Failed, KVPoll.Failed),
            (KVPoll.Failed, KVPoll.Bootstrapping, KVPoll.Failed),
        )
        for kv, taps, expected in cases:
            with self.subTest(kv=kv, taps=taps):
                self.assertEqual(self._poll(kv=kv, taps=taps), expected)

    def test_before_the_relay_starts_the_request_is_its_kv_transfer(self):
        self.assertEqual(
            self._poll(kv=KVPoll.Bootstrapping, taps=None), KVPoll.Bootstrapping
        )


if __name__ == "__main__":
    unittest.main()
