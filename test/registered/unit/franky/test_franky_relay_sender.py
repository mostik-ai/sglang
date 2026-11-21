import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl.conn import NixlKVSender
from sglang.srt.franky.transfer.relay.sender import FrankyRelaySender
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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
