import unittest

import numpy as np
import torch

from sglang.srt.franky.cache.ricky_tap_cache import RickyTapCache
from sglang.srt.franky.cache.sandy_tap_cache import SandyTapCache
from sglang.srt.franky.cache.tap_buffers import TAP_DTYPE
from sglang.srt.franky.cache.tap_stripes import TapStripes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PAGE_SIZE = 4
_OUTPUT_SIZE = 8
_NUM_KV_SLOTS = 32


def _sandy_cache(*, num_ranks: int, rank: int) -> SandyTapCache:
    return SandyTapCache.build(
        encoder_ids=(0, 1),
        output_size=_OUTPUT_SIZE,
        page_size=_PAGE_SIZE,
        num_kv_slots=_NUM_KV_SLOTS,
        stripes=TapStripes(num_ranks=num_ranks, rank=rank),
        device="cpu",
    )


def _ricky_cache(*, num_ranks: int = 2, rank: int = 0, num_taps: int = 64):
    return RickyTapCache.build(
        encoder_ids=(0,),
        output_size=_OUTPUT_SIZE,
        page_size=_PAGE_SIZE,
        num_taps=num_taps,
        stripes=TapStripes(num_ranks=num_ranks, rank=rank),
        device="cpu",
    )


def _slot_values(num_slots: int) -> torch.Tensor:
    values = torch.arange(1, num_slots + 1, dtype=TAP_DTYPE)[:, None]
    return values.expand(-1, _OUTPUT_SIZE)


def _first_feature(cache: SandyTapCache, encoder_id: int) -> torch.Tensor:
    return cache.buffers.pages(encoder_id)[:, :, 0, 0]


class TestTapStripes(CustomTestCase):
    def test_one_rank_keeps_every_page_as_its_own_stripe(self):
        stripes = TapStripes(num_ranks=1, rank=0)
        pages = torch.arange(6)

        self.assertTrue(bool(stripes.keeps(pages).all()))
        self.assertEqual(stripes.stripe(pages).tolist(), pages.tolist())
        self.assertEqual(
            stripes.local_taps(torch.tensor([0, 3, 17]), page_size=4).tolist(),
            [0, 3, 17],
        )

    def test_ranks_keep_pages_in_turn(self):
        pages = np.arange(10)

        kept = {
            rank: pages[TapStripes(num_ranks=4, rank=rank).keeps(pages)].tolist()
            for rank in range(4)
        }

        self.assertEqual(kept, {0: [0, 4, 8], 1: [1, 5, 9], 2: [2, 6], 3: [3, 7]})
        stripes = TapStripes(num_ranks=4, rank=1)
        self.assertEqual(stripes.stripe(np.array([1, 5, 9])).tolist(), [0, 1, 2])

    def test_a_rank_reads_whole_stripes_then_its_part_of_the_last(self):
        taps = torch.tensor([21])

        local = [
            int(TapStripes(num_ranks=4, rank=rank).local_taps(taps, page_size=4)[0])
            for rank in range(4)
        ]

        self.assertEqual(local, [8, 5, 4, 4])

    def test_the_ranks_together_read_every_tap_once(self):
        taps = torch.arange(0, 70)

        total = sum(
            TapStripes(num_ranks=4, rank=rank).local_taps(taps, page_size=4)
            for rank in range(4)
        )

        self.assertEqual(total.tolist(), taps.tolist())


class TestSandyTapCache(CustomTestCase):
    def test_one_rank_keeps_each_tap_at_its_kv_slot(self):
        cache = _sandy_cache(num_ranks=1, rank=0)
        kv_slots = torch.arange(_NUM_KV_SLOTS)

        cache.write(encoder_id=0, taps=_slot_values(_NUM_KV_SLOTS), kv_slots=kv_slots)

        written = _first_feature(cache, 0)[:-1].reshape(-1)
        self.assertEqual(written.tolist(), list(range(1, _NUM_KV_SLOTS + 1)))
        self.assertEqual(cache.buffers.num_pages, _NUM_KV_SLOTS // _PAGE_SIZE + 1)

    def test_each_rank_keeps_its_tap_pages_at_the_local_page_of_their_stripe(self):
        kv_slots = torch.arange(_NUM_KV_SLOTS)
        taps = _slot_values(_NUM_KV_SLOTS)
        for rank in range(2):
            with self.subTest(rank=rank):
                cache = _sandy_cache(num_ranks=2, rank=rank)

                cache.write(encoder_id=0, taps=taps, kv_slots=kv_slots)

                kept_pages = range(rank, _NUM_KV_SLOTS // _PAGE_SIZE, 2)
                expected = [
                    [page * _PAGE_SIZE + offset + 1 for offset in range(_PAGE_SIZE)]
                    for page in kept_pages
                ]
                self.assertEqual(
                    _first_feature(cache, 0)[: len(expected)].tolist(), expected
                )

    def test_taps_another_rank_keeps_go_to_the_discarded_page_only(self):
        cache = _sandy_cache(num_ranks=2, rank=0)
        other_ranks_page = torch.arange(_PAGE_SIZE, 2 * _PAGE_SIZE)

        cache.write(
            encoder_id=0, taps=_slot_values(_PAGE_SIZE), kv_slots=other_ranks_page
        )

        pages = _first_feature(cache, 0)
        self.assertTrue(bool((pages[:-1] == 0).all()))
        self.assertEqual(pages[-1].tolist(), [1, 2, 3, 4])

    def test_encoders_do_not_share_taps(self):
        cache = _sandy_cache(num_ranks=2, rank=1)

        cache.write(
            encoder_id=0,
            taps=_slot_values(_PAGE_SIZE),
            kv_slots=torch.arange(_PAGE_SIZE, 2 * _PAGE_SIZE),
        )

        self.assertTrue(bool((cache.buffers.pages(1) == 0).all()))

    def test_local_pages_mark_the_pages_other_ranks_keep(self):
        kv_slots = torch.cat(
            [
                torch.arange(page * _PAGE_SIZE, (page + 1) * _PAGE_SIZE)
                for page in (3, 4, 7)
            ]
        )

        rank_pages = [
            _sandy_cache(num_ranks=2, rank=rank).local_pages(kv_slots=kv_slots)
            for rank in range(2)
        ]

        self.assertEqual(rank_pages[0].dtype, torch.int32)
        self.assertEqual(rank_pages[0].tolist(), [-1, 2, -1])
        self.assertEqual(rank_pages[1].tolist(), [1, -1, 3])

    def test_transfer_sees_whole_pages_of_each_buffer(self):
        cache = _sandy_cache(num_ranks=2, rank=0)
        buffers = cache.buffers.transfer_buffers()
        page_bytes = _PAGE_SIZE * _OUTPUT_SIZE * TAP_DTYPE.itemsize

        self.assertEqual(cache.buffers.page_bytes, page_bytes)
        self.assertEqual(sorted(buffers), [0, 1])
        self.assertEqual(len({address for address, _ in buffers.values()}), 2)
        self.assertEqual(
            [size for _, size in buffers.values()],
            [page_bytes * cache.buffers.num_pages] * 2,
        )


class TestRickyTapCache(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.cache = _ricky_cache()

    def _hold_and_keep(self, cache: RickyTapCache, sandy_ids: list[int]):
        cached = cache.match(sandy_ids)
        num_stripes = -(-len(sandy_ids) // cache.stripe_size)
        new_slots = cache.take(num_stripes - len(cached.slots))
        slots = torch.cat([cached.slots, new_slots])
        return slots, cache.keep(sandy_ids=sandy_ids, slots=slots, cached=cached)

    def _keep(self, sandy_ids: list[int], cache=None) -> torch.Tensor:
        cache = cache or self.cache
        slots, lock = self._hold_and_keep(cache, sandy_ids)
        cache.release(lock)
        return slots

    def test_build_counts_slots_of_whole_stripes_over_all_ranks(self):
        self.assertEqual(self.cache.stripe_size, 8)
        self.assertEqual(self.cache.num_slots, 8)
        self.assertEqual(self.cache.num_pages, 16)
        self.assertEqual(self.cache.buffers.num_pages, 9)

    def test_take_gives_distinct_slots_past_the_padding_page(self):
        slots = self.cache.take(5)

        self.assertEqual(slots.dtype, torch.int64)
        self.assertEqual(len(set(slots.tolist())), 5)
        self.assertTrue(all(1 <= slot <= 8 for slot in slots.tolist()))
        self.assertIsNone(self.cache.take(4))
        self.assertEqual(self.cache.num_free_slots(), 3)

    def test_freed_slots_can_be_taken_again(self):
        slots = self.cache.take(8)

        self.cache.free(slots[:3])

        self.assertEqual(self.cache.num_free_slots(), 3)
        self.assertEqual(
            sorted(self.cache.take(3).tolist()), sorted(slots[:3].tolist())
        )

    def test_a_kept_request_is_found_by_a_later_one_with_the_same_first_tokens(self):
        slots = self._keep(list(range(20)))

        cached = self.cache.match(list(range(20)) + [99, 98])

        self.assertEqual(cached.num_taps, 16)
        self.assertEqual(cached.slots.tolist(), slots[:2].tolist())
        self.cache.release(cached.lock)

    def test_a_match_counts_whole_stripes_only(self):
        self._keep(list(range(20)))

        cached = self.cache.match(list(range(15)))

        self.assertEqual(cached.num_taps, 8)
        self.cache.release(cached.lock)

    def test_a_match_without_a_cached_prefix_holds_nothing(self):
        cached = self.cache.match(list(range(20)))

        self.assertEqual(cached.num_taps, 0)
        self.assertEqual(cached.slots.tolist(), [])
        self.cache.release(cached.lock)

    def test_keep_gives_back_the_partial_last_stripe(self):
        self._keep(list(range(20)))

        self.assertEqual(self.cache.num_free_slots(), 8)
        self.assertEqual(self.cache.num_free_pages(), (12, 4))

    def test_keep_gives_back_stripes_another_request_kept_first(self):
        first = self.cache.match(list(range(16)))
        second = self.cache.match(list(range(16)))
        first_slots = self.cache.take(2)
        second_slots = self.cache.take(2)

        self.cache.release(
            self.cache.keep(sandy_ids=list(range(16)), slots=first_slots, cached=first)
        )
        self.cache.release(
            self.cache.keep(
                sandy_ids=list(range(16)), slots=second_slots, cached=second
            )
        )

        self.assertEqual(self.cache.num_free_pages(), (12, 4))
        cached = self.cache.match(list(range(16)))
        self.assertEqual(cached.slots.tolist(), first_slots.tolist())
        self.cache.release(cached.lock)

    def test_a_request_that_started_from_a_cached_prefix_keeps_only_its_new_stripes(
        self,
    ):
        prefix_slots = self._keep(list(range(16)))

        slots = self._keep(list(range(32)))

        self.assertEqual(slots[:2].tolist(), prefix_slots[:2].tolist())
        self.assertEqual(self.cache.num_free_pages(), (8, 8))
        cached = self.cache.match(list(range(32)))
        self.assertEqual(cached.slots.tolist(), slots.tolist())
        self.cache.release(cached.lock)

    def test_a_locked_stripe_is_never_evicted(self):
        _, lock = self._hold_and_keep(self.cache, list(range(16)))

        self.assertEqual(self.cache.num_free_slots(), 6)
        self.assertIsNone(self.cache.take(7))
        self.cache.release(lock)
        self.assertEqual(len(self.cache.take(8)), 8)
        self.assertEqual(self.cache.match(list(range(16))).num_taps, 0)

    def test_take_evicts_stripes_that_only_the_tree_keeps(self):
        self._keep(list(range(32)))
        self._keep(list(range(100, 132)))

        taken = self.cache.take(6)

        self.assertEqual(len(taken), 6)
        self.assertEqual(self.cache.num_free_slots(), 2)

    def test_a_match_holds_its_prefix_until_released(self):
        self._keep(list(range(16)))
        cached = self.cache.match(list(range(16)))

        self.assertIsNone(self.cache.take(7))
        self.cache.release(cached.lock)
        self.assertEqual(len(self.cache.take(7)), 7)

    def test_local_pages_mark_the_pages_other_ranks_keep(self):
        cache = _ricky_cache(num_ranks=2, rank=1)

        pages = cache.local_pages(
            slots=torch.tensor([5, 2, 7]), first_page=1, num_pages=6
        )

        self.assertEqual(pages.dtype, np.int32)
        self.assertEqual(pages.tolist(), [5, -1, 2, -1, 7])

    def test_one_rank_lists_a_slot_for_every_page(self):
        cache = _ricky_cache(num_ranks=1, rank=0, num_taps=32)

        pages = cache.local_pages(
            slots=torch.tensor([3, 1, 6]), first_page=0, num_pages=3
        )

        self.assertEqual(pages.tolist(), [3, 1, 6])

    def test_every_rank_takes_the_same_slots_for_the_same_requests(self):
        caches = [_ricky_cache(num_ranks=2, rank=rank) for rank in range(2)]

        slots = []
        for cache in caches:
            self._keep(list(range(24)), cache=cache)
            slots.append(cache.take(3).tolist())

        self.assertEqual(slots[0], slots[1])

    def test_reset_drops_every_cached_stripe(self):
        self._keep(list(range(32)))

        self.cache.reset()

        self.assertEqual(self.cache.match(list(range(32))).num_taps, 0)
        self.assertEqual(self.cache.num_free_pages(), (16, 0))


if __name__ == "__main__":
    unittest.main()
