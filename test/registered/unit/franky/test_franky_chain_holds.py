import unittest

from sglang.srt.franky.chains.chain import FrankyChain
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.chains.messages import FrankyHoldExpired
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Tree:
    def __init__(self):
        self.locked = []

    def lock(self, node):
        self.locked.append(node)
        return node

    def unlock(self, lock):
        self.locked.remove(lock)


class TestFrankyChainHolds(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tree = _Tree()

    def _holds(self, *, max_held_tokens=100, hold_seconds=60.0):
        return FrankyChainHolds(
            tree_cache=self.tree,
            max_held_tokens=max_held_tokens,
            hold_seconds=hold_seconds,
        )

    def test_a_chains_next_hold_replaces_its_last(self):
        holds = self._holds()

        holds.hold(chain_id="c", node="block-1", num_tokens=10)
        holds.hold(chain_id="c", node="block-2", num_tokens=20)

        self.assertEqual(self.tree.locked, ["block-2"])
        self.assertEqual(holds.held_tokens, 20)

    def test_past_the_cap_the_oldest_holds_go(self):
        holds = self._holds(max_held_tokens=50)
        holds.hold(chain_id="a", node="a", num_tokens=20)
        holds.hold(chain_id="b", node="b", num_tokens=20)
        holds.hold(chain_id="a", node="a2", num_tokens=20)

        holds.hold(chain_id="c", node="c", num_tokens=20)

        self.assertEqual(self.tree.locked, ["a2", "c"])
        self.assertEqual(holds.held_tokens, 40)

    def test_release_oldest_goes_in_the_order_holds_were_taken(self):
        holds = self._holds()
        holds.hold(chain_id="a", node="a", num_tokens=1)
        holds.hold(chain_id="b", node="b", num_tokens=1)

        self.assertTrue(holds.release_oldest())
        self.assertEqual(self.tree.locked, ["b"])
        self.assertTrue(holds.release_oldest())
        self.assertFalse(holds.release_oldest())

    def test_releasing_a_chain_without_a_hold_does_nothing(self):
        holds = self._holds()
        holds.hold(chain_id="a", node="a", num_tokens=1)

        holds.release(chain_id="other")

        self.assertEqual(self.tree.locked, ["a"])

    def test_an_expired_hold_is_announced_once_and_released_on_every_rank(self):
        leader = self._holds(hold_seconds=0)
        leader.hold(chain_id="a", node="a", num_tokens=1)

        messages = leader.expired_messages()

        self.assertEqual(messages, [FrankyHoldExpired(chain_id="a", order=0)])
        self.assertEqual(leader.expired_messages(), [])
        leader.on_hold_expired(message=messages[0])
        self.assertEqual(self.tree.locked, [])

    def test_a_live_hold_is_not_announced(self):
        holds = self._holds(hold_seconds=3600)
        holds.hold(chain_id="a", node="a", num_tokens=1)

        self.assertEqual(holds.expired_messages(), [])

    def test_an_expiry_of_an_older_hold_keeps_the_newer_one(self):
        holds = self._holds()
        holds.hold(chain_id="a", node="block-1", num_tokens=1)
        holds.hold(chain_id="a", node="block-2", num_tokens=1)

        holds.on_hold_expired(message=FrankyHoldExpired(chain_id="a", order=0))

        self.assertEqual(self.tree.locked, ["block-2"])


class TestFrankyChain(CustomTestCase):
    def test_a_block_rid_must_start_with_the_chain_id(self):
        chain = FrankyChain.from_request(
            franky_chain={"id": "abc", "last_block": True}, rid="abc-3"
        )
        self.assertEqual(chain, FrankyChain(id="abc", last_block=True))

        with self.assertRaisesRegex(ValueError, "starts with the chain id"):
            FrankyChain.from_request(franky_chain={"id": "abc"}, rid="xyz-1")


if __name__ == "__main__":
    unittest.main()
