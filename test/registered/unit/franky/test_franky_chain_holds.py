import unittest
from types import SimpleNamespace

import msgspec

from sglang.srt.franky.chains.chain import FrankyChain, put_continuations_first
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


def _block(chain_id, *, node, num_tokens=1, last_block=False):
    return SimpleNamespace(
        franky_chain=FrankyChain(id=chain_id, last_block=last_block),
        last_node=node,
        origin_input_ids=[0] * num_tokens,
    )


class TestFrankyChainHolds(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tree = _Tree()

    def _holds(self, *, max_held_tokens=100, hold_seconds=60.0):
        return FrankyChainHolds(
            tree_cache=self.tree,
            max_held_tokens=max_held_tokens,
            hold_seconds=hold_seconds,
            chunked_prefill_size=2048,
        )

    def test_a_chains_next_hold_replaces_its_last(self):
        holds = self._holds()

        holds.hold_for_next_block(_block("c", node="block-1", num_tokens=10))
        holds.hold_for_next_block(_block("c", node="block-2", num_tokens=20))

        self.assertEqual(self.tree.locked, ["block-2"])
        self.assertEqual(holds.held_tokens, 20)

    def test_a_last_block_releases_its_chains_hold(self):
        holds = self._holds()
        holds.hold_for_next_block(_block("c", node="block-1"))

        holds.hold_for_next_block(_block("c", node="block-2", last_block=True))

        self.assertEqual(self.tree.locked, [])
        self.assertEqual(holds.held_tokens, 0)

    def test_a_request_outside_chains_holds_nothing(self):
        holds = self._holds()
        plain = SimpleNamespace(franky_chain=None, last_node="n", origin_input_ids=[0])

        holds.hold_for_next_block(plain)
        holds.release_chain_of(plain)

        self.assertEqual(self.tree.locked, [])

    def test_past_the_cap_the_oldest_holds_go(self):
        holds = self._holds(max_held_tokens=50)
        holds.hold_for_next_block(_block("a", node="a", num_tokens=20))
        holds.hold_for_next_block(_block("b", node="b", num_tokens=20))
        holds.hold_for_next_block(_block("a", node="a2", num_tokens=20))

        holds.hold_for_next_block(_block("c", node="c", num_tokens=20))

        self.assertEqual(self.tree.locked, ["a2", "c"])
        self.assertEqual(holds.held_tokens, 40)

    def test_release_oldest_goes_in_the_order_holds_were_taken(self):
        holds = self._holds()
        holds.hold_for_next_block(_block("a", node="a"))
        holds.hold_for_next_block(_block("b", node="b"))

        self.assertTrue(holds.release_oldest())
        self.assertEqual(self.tree.locked, ["b"])
        self.assertTrue(holds.release_oldest())
        self.assertFalse(holds.release_oldest())

    def test_release_all_leaves_nothing_locked(self):
        holds = self._holds()
        holds.hold_for_next_block(_block("a", node="a", num_tokens=3))
        holds.hold_for_next_block(_block("b", node="b", num_tokens=4))

        holds.release_all()

        self.assertEqual(self.tree.locked, [])
        self.assertEqual(holds.held_tokens, 0)

    def test_releasing_a_chain_without_a_hold_does_nothing(self):
        holds = self._holds()
        holds.hold_for_next_block(_block("a", node="a"))

        holds.release(chain_id="other")
        holds.release_chain_of(_block("other", node="o"))

        self.assertEqual(self.tree.locked, ["a"])

    def test_an_expired_hold_is_announced_once_and_released_on_every_rank(self):
        leader = self._holds(hold_seconds=0)
        leader.hold_for_next_block(_block("a", node="a"))

        messages = leader.expired_messages()

        self.assertEqual(messages, [FrankyHoldExpired(chain_id="a", order=0)])
        self.assertEqual(leader.expired_messages(), [])
        leader.on_hold_expired(message=messages[0])
        self.assertEqual(self.tree.locked, [])

    def test_a_live_hold_is_not_announced(self):
        holds = self._holds(hold_seconds=3600)
        holds.hold_for_next_block(_block("a", node="a"))

        self.assertEqual(holds.expired_messages(), [])

    def test_prefill_chunks_shrink_only_while_a_chain_holds_its_prompt(self):
        holds = self._holds()
        self.assertEqual(holds.chunk_size_for_batch(8192), 8192)

        holds.hold_for_next_block(_block("a", node="a"))

        self.assertEqual(holds.chunk_size_for_batch(8192), 2048)
        self.assertEqual(holds.chunk_size_for_batch(1024), 1024)
        self.assertIsNone(holds.chunk_size_for_batch(None))

    def test_an_expiry_of_an_older_hold_keeps_the_newer_one(self):
        holds = self._holds()
        holds.hold_for_next_block(_block("a", node="block-1"))
        holds.hold_for_next_block(_block("a", node="block-2"))

        holds.on_hold_expired(message=FrankyHoldExpired(chain_id="a", order=0))

        self.assertEqual(self.tree.locked, ["block-2"])


class TestFrankyChain(CustomTestCase):
    def test_continuations_go_first_and_keep_their_order(self):
        def req(name, block):
            chain = None if block is None else FrankyChain(id=name, block=block)
            return SimpleNamespace(name=name, franky_chain=chain)

        requests = [
            req("new", 1),
            req("plain", None),
            req("c1", 3),
            req("new2", 1),
            req("c2", 2),
        ]

        put_continuations_first(requests)

        self.assertEqual(
            [request.name for request in requests], ["c1", "c2", "new", "plain", "new2"]
        )

    def test_a_chain_needs_an_id_and_blocks_count_from_one(self):
        for franky_chain in ({"id": "", "block": 1}, {"id": "abc", "block": 0}):
            with self.subTest(franky_chain), self.assertRaises(msgspec.ValidationError):
                FrankyChain.from_request(franky_chain=franky_chain, rid="abc-0")

    def test_a_block_rid_must_start_with_the_chain_id(self):
        chain = FrankyChain.from_request(
            franky_chain={"id": "abc", "last_block": True}, rid="abc-3"
        )
        self.assertEqual(chain, FrankyChain(id="abc", last_block=True))

        with self.assertRaisesRegex(ValueError, "starts with the chain id"):
            FrankyChain.from_request(franky_chain={"id": "abc"}, rid="xyz-1")


if __name__ == "__main__":
    unittest.main()
