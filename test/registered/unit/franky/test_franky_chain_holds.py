import unittest
from types import SimpleNamespace

import msgspec

from sglang.srt.franky.chains.chain import FrankyChain, put_continuations_first
from sglang.srt.franky.chains.holds import FrankyChainHolds
from sglang.srt.franky.chains.messages import FrankyHoldExpired
from sglang.srt.franky.request import FrankyRequest
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Tree:
    def __init__(self):
        self.locked = []

    def lock(self, node):
        self.locked.append(node)
        return SimpleNamespace(node=node)

    def unlock(self, lock):
        self.locked.remove(lock.node)


def _block(chain_id, *, node, num_tokens=1, block=1, last_block=False):
    return SimpleNamespace(
        franky=FrankyRequest(
            chain=FrankyChain(
                id=chain_id, block=block, block_size=8, last_block=last_block
            )
        ),
        last_node=node,
        origin_input_ids=[0] * num_tokens,
    )


def _chain_id(req):
    request = FrankyRequest.of(req)
    return None if request.chain is None else request.chain.id


class TestFrankyChainHolds(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tree = _Tree()

    def _holds(self, *, hold_seconds=60.0):
        return FrankyChainHolds(
            tree_cache=self.tree, tap_cache=None, hold_seconds=hold_seconds
        )

    def test_a_chains_next_hold_replaces_its_last(self):
        holds = self._holds()

        holds.hold_for_next_block(_block("c", node="block-1", num_tokens=10))
        holds.hold_for_next_block(_block("c", node="block-2", num_tokens=20))

        self.assertEqual(self.tree.locked, ["block-2"])

    def test_a_last_block_releases_its_chains_hold(self):
        holds = self._holds()
        holds.hold_for_next_block(_block("c", node="block-1"))

        holds.hold_for_next_block(_block("c", node="block-2", last_block=True))

        self.assertEqual(self.tree.locked, [])

    def test_a_request_outside_chains_holds_nothing(self):
        holds = self._holds()
        plain = SimpleNamespace(franky=None, last_node="n", origin_input_ids=[0])

        holds.hold_for_next_block(plain)
        holds.release_chain_of(plain)

        self.assertEqual(self.tree.locked, [])

    def test_holds_keep_every_chain_however_many_tokens(self):
        holds = self._holds()

        for chain_id in "abcdefgh":
            holds.hold_for_next_block(
                _block(chain_id, node=chain_id, num_tokens=1_000_000)
            )

        self.assertEqual(self.tree.locked, list("abcdefgh"))

    def test_least_progress_goes_first_then_the_most_tokens_but_not_the_asker(self):
        holds = self._holds()
        holds.hold_for_next_block(_block("far", node="far", num_tokens=90, block=5))
        holds.hold_for_next_block(_block("small", node="small", num_tokens=10))
        holds.hold_for_next_block(_block("big", node="big", num_tokens=50))
        holds.hold_for_next_block(_block("asker", node="asker", num_tokens=99))

        released = [
            holds.release_least_progress(sparing_chain_id="asker") for _ in range(4)
        ]

        self.assertEqual(released, [True, True, True, False])
        self.assertEqual(self.tree.locked, ["asker"])

    def test_the_first_release_frees_the_biggest_chain_of_the_fewest_blocks(self):
        holds = self._holds()
        holds.hold_for_next_block(_block("far", node="far", num_tokens=90, block=5))
        holds.hold_for_next_block(_block("small", node="small", num_tokens=10))
        holds.hold_for_next_block(_block("big", node="big", num_tokens=50))

        holds.release_least_progress(sparing_chain_id=None)

        self.assertEqual(self.tree.locked, ["far", "small"])

    def test_held_chains_go_first_and_keep_their_order(self):
        holds = self._holds()
        holds.hold_for_next_block(_block("held-1", node="h1"))
        holds.hold_for_next_block(_block("held-2", node="h2"))
        requests = [
            _block("new", node="n"),
            _block("held-2", node="h2", block=2),
            SimpleNamespace(franky=None),
            _block("held-1", node="h1", block=2),
        ]

        holds.put_held_chains_first(requests)

        self.assertEqual(
            [_chain_id(r) for r in requests], ["held-2", "held-1", "new", None]
        )

    def test_locked_nodes_name_each_held_node_once(self):
        holds = self._holds()
        holds.hold_for_next_block(_block("a", node="shared"))
        holds.hold_for_next_block(_block("b", node="shared"))
        holds.hold_for_next_block(_block("c", node="c"))

        self.assertEqual(holds.locked_nodes(), {"shared", "c"})

    def test_release_all_leaves_nothing_locked(self):
        holds = self._holds()
        holds.hold_for_next_block(_block("a", node="a", num_tokens=3))
        holds.hold_for_next_block(_block("b", node="b", num_tokens=4))

        holds.release_all()

        self.assertEqual(self.tree.locked, [])

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

    def test_an_expiry_of_an_older_hold_keeps_the_newer_one(self):
        holds = self._holds()
        holds.hold_for_next_block(_block("a", node="block-1"))
        holds.hold_for_next_block(_block("a", node="block-2"))

        holds.on_hold_expired(message=FrankyHoldExpired(chain_id="a", order=0))

        self.assertEqual(self.tree.locked, ["block-2"])


class _TapCache:
    def __init__(self):
        self.released = []

    def release(self, lock):
        self.released.append(lock)


class TestFrankyChainHoldsOfTaps(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tree = _Tree()
        self.tap_cache = _TapCache()
        self.holds = self._holds()

    def _holds(self, *, tree_cache=None):
        return FrankyChainHolds(
            tree_cache=self.tree if tree_cache is None else tree_cache,
            tap_cache=self.tap_cache,
            hold_seconds=60.0,
        )

    def test_a_hold_keeps_its_blocks_tap_lock_beside_its_kv_lock(self):
        self.holds.hold_for_next_block(_block("c", node="c"), tap_lock="taps-c")

        self.assertTrue(self.holds.holds_chain_of(_block("c", node="c", block=2)))
        self.assertEqual(self.tree.locked, ["c"])
        self.assertEqual(self.tap_cache.released, [])

    def test_every_way_a_hold_ends_releases_its_tap_lock(self):
        ends = {
            "release": lambda holds: holds.release(chain_id="c"),
            "least progress": lambda holds: holds.release_least_progress(
                sparing_chain_id=None
            ),
            "release_all": lambda holds: holds.release_all(),
            "expiry": lambda holds: holds.on_hold_expired(
                message=FrankyHoldExpired(chain_id="c", order=0)
            ),
            "next hold": lambda holds: holds.hold_for_next_block(
                _block("c", node="c-2")
            ),
        }
        for name, end in ends.items():
            with self.subTest(name):
                self.tap_cache.released.clear()
                holds = self._holds()
                holds.hold_for_next_block(_block("c", node="c"), tap_lock="taps-c")

                end(holds)

                self.assertEqual(self.tap_cache.released, ["taps-c"])
                holds.release_all()

    def test_a_last_block_or_one_outside_chains_releases_its_tap_lock_at_once(self):
        plain = SimpleNamespace(franky=None, last_node="n", origin_input_ids=[0])
        last = _block("c", node="c-2", last_block=True)
        self.holds.hold_for_next_block(_block("c", node="c-1"), tap_lock="taps-1")

        self.holds.hold_for_next_block(plain, tap_lock="taps-plain")
        self.holds.hold_for_next_block(last, tap_lock="taps-last")

        self.assertEqual(
            sorted(self.tap_cache.released), ["taps-1", "taps-last", "taps-plain"]
        )
        self.assertEqual(self.tree.locked, [])

    def test_a_decode_hold_locks_no_prompt_and_ranks_by_its_prompt(self):
        holds = FrankyChainHolds(
            tree_cache=None, tap_cache=self.tap_cache, hold_seconds=60.0
        )
        holds.hold_for_next_block(
            _block("few", node=None, num_tokens=20), tap_lock="taps-few"
        )
        holds.hold_for_next_block(
            _block("many", node=None, num_tokens=40), tap_lock="taps-many"
        )

        holds.release_least_progress(sparing_chain_id=None)

        self.assertEqual(self.tap_cache.released, ["taps-many"])
        self.assertTrue(holds.holds_chain_of(_block("few", node=None, block=2)))
        self.assertEqual(holds.locked_nodes(), set())
        self.assertEqual(self.tree.locked, [])

    def test_a_hold_without_taps_releases_no_tap_lock(self):
        self.holds.hold_for_next_block(_block("c", node="c"))

        self.holds.release(chain_id="c")

        self.assertEqual(self.tap_cache.released, [])
        self.assertEqual(self.tree.locked, [])


class TestFrankyChain(CustomTestCase):
    def test_continuations_go_first_and_keep_their_order(self):
        def req(name, block):
            chain = (
                None
                if block is None
                else FrankyChain(id=name, block=block, block_size=8)
            )
            return SimpleNamespace(
                name=name,
                franky=None if chain is None else FrankyRequest(chain=chain),
            )

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

    def test_a_chain_needs_an_id_a_block_size_and_blocks_count_from_one(self):
        for franky_chain in (
            {"id": "", "block": 1, "block_size": 8},
            {"id": "abc", "block": 0, "block_size": 8},
            {"id": "abc", "block": 1},
            {"id": "abc", "block": 1, "block_size": 0},
        ):
            with self.subTest(franky_chain), self.assertRaises(msgspec.ValidationError):
                FrankyChain.from_request(
                    franky_chain=franky_chain, rid="abc-0", prompt_len=20
                )

    def test_a_block_rid_must_start_with_the_chain_id(self):
        chain = FrankyChain.from_request(
            franky_chain={"id": "abc", "block_size": 8, "last_block": True},
            rid="abc-3",
            prompt_len=20,
        )
        self.assertEqual(chain, FrankyChain(id="abc", block_size=8, last_block=True))

        with self.assertRaisesRegex(ValueError, "starts with the chain id"):
            FrankyChain.from_request(
                franky_chain={"id": "abc", "block_size": 8}, rid="xyz-1", prompt_len=20
            )

    def test_a_blocks_prompt_must_hold_the_earlier_blocks_and_a_prompt(self):
        third_block = {"id": "abc", "block": 3, "block_size": 8}

        chain = FrankyChain.from_request(
            franky_chain=third_block, rid="abc-3", prompt_len=17
        )

        self.assertEqual(chain.num_generated_before, 16)
        with self.assertRaisesRegex(ValueError, "too short"):
            FrankyChain.from_request(
                franky_chain=third_block, rid="abc-3", prompt_len=16
            )

    def test_with_sandy_on_its_own_a_block_names_sandys_start_of_each_earlier_block(
        self,
    ):
        third_block = {"id": "abc", "block": 3, "block_size": 8}

        chain = FrankyChain.from_request(
            franky_chain={**third_block, "sandy_block_starts": [11, 18]},
            rid="abc-3",
            prompt_len=30,
            sandy_input_len=25,
        )

        self.assertEqual(chain.sandy_block_starts, (11, 18))
        self.assertEqual(chain.sandy_answer_start(sandy_input_len=25), 11)
        for starts, sandy_input_len, problem in (
            ([11], 25, "earlier blocks"),
            ([18, 11], 25, "in order"),
            ([11, 18], None, "come with franky_sandy_input_ids"),
            ([], 25, "needs Sandy's tokens before each earlier block"),
            ([11, 26], 25, "before its earlier blocks"),
        ):
            with (
                self.subTest(starts=starts, sandy_input_len=sandy_input_len),
                self.assertRaisesRegex(ValueError, problem),
            ):
                FrankyChain.from_request(
                    franky_chain={**third_block, "sandy_block_starts": starts},
                    rid="abc-3",
                    prompt_len=30,
                    sandy_input_len=sandy_input_len,
                )

    def test_a_first_block_reads_sandys_input_as_its_answer_start(self):
        chain = FrankyChain.from_request(
            franky_chain={"id": "abc", "block_size": 8},
            rid="abc-1",
            prompt_len=12,
            sandy_input_len=9,
        )

        self.assertEqual(chain.sandy_answer_start(sandy_input_len=9), 9)

    def test_on_a_sandy_of_its_own_a_blocks_prompt_counts_sandys_tokens(self):
        third_block = {"id": "abc", "block": 3, "block_size": 8}
        starts = {"sandy_block_starts": [11, 13]}

        # Sandy's tokens of two 8-token blocks may be fewer than Ricky's 16.
        chain = FrankyChain.from_request(
            franky_chain={**third_block, **starts},
            rid="abc-3",
            prompt_len=15,
            on_sandy=True,
        )

        self.assertEqual(chain.sandy_block_starts, (11, 13))
        with self.assertRaisesRegex(ValueError, "too short"):
            FrankyChain.from_request(
                franky_chain={**third_block, **starts},
                rid="abc-3",
                prompt_len=12,
                on_sandy=True,
            )


if __name__ == "__main__":
    unittest.main()
