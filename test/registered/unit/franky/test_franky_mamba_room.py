import unittest
from types import SimpleNamespace

from sglang.srt.franky.cache.mamba_room import FrankyMambaRoom
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _room(*, num_states, num_rows=4, rows_in_use=0):
    pool = SimpleNamespace(
        size=num_rows,
        available_size=lambda: num_rows - rows_in_use,
        mamba_allocator=SimpleNamespace(size=num_states),
    )
    return FrankyMambaRoom(req_to_token_pool=pool, states_per_request=5)


class TestFrankyMambaRoom(CustomTestCase):
    def test_pinned_states_take_the_room_of_requests_that_could_start(self):
        for num_pinned_states, num_startable in ((0, 4), (3, 3), (5, 3), (6, 2)):
            with self.subTest(num_pinned_states=num_pinned_states):
                self.assertEqual(
                    _room(num_states=20).num_startable_requests(
                        num_pinned_states=num_pinned_states
                    ),
                    num_startable,
                )

    def test_requests_that_hold_rows_keep_their_states(self):
        room = _room(num_states=20, rows_in_use=1)

        self.assertEqual(room.num_startable_requests(num_pinned_states=5), 2)

    def test_a_full_pool_starts_no_request(self):
        room = _room(num_states=20, rows_in_use=3)

        self.assertEqual(room.num_startable_requests(num_pinned_states=10), 0)

    def test_pins_leave_one_request_room_to_start_while_none_runs(self):
        room = _room(num_states=20)

        self.assertTrue(room.fits_pinned_states(num_pinned_states=15))
        self.assertFalse(room.fits_pinned_states(num_pinned_states=16))

    def test_pins_leave_running_requests_their_states(self):
        room = _room(num_states=20, rows_in_use=2)

        self.assertTrue(room.fits_pinned_states(num_pinned_states=10))
        self.assertFalse(room.fits_pinned_states(num_pinned_states=11))

    def test_an_engine_with_a_mamba_slot_pool_counts_its_requests_states(self):
        pool = object.__new__(HybridReqToTokenPool)
        pool.mamba_allocator = MambaSlotAllocator(size=20, device="cpu")
        configurator = SimpleNamespace(calculate_mamba_ratio=lambda: 5)

        room = FrankyMambaRoom.from_engine(
            req_to_token_pool=pool, kv_cache_configurator=configurator
        )

        self.assertEqual(room.states_per_request, 5)

    def test_an_engine_without_a_mamba_slot_pool_has_no_room(self):
        pool = SimpleNamespace(size=4, available_size=lambda: 4)

        self.assertIsNone(
            FrankyMambaRoom.from_engine(
                req_to_token_pool=pool, kv_cache_configurator=SimpleNamespace()
            )
        )


if __name__ == "__main__":
    unittest.main()
