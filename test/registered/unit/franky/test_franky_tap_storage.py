import unittest

import numpy as np
import torch

from sglang.srt.franky.cache.ricky_tap_cache import (
    CachedTaps,
    RickyTapCache,
    TapReservation,
)
from sglang.srt.franky.cache.sandy_tap_cache import SandyTapCache
from sglang.srt.franky.cache.tap_buffers import TAP_DTYPE, TapBuffers
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


def _ricky_cache(*, num_ranks: int = 2, rank: int = 0, num_stripes: int = 8):
    return RickyTapCache(
        buffers=TapBuffers(
            encoder_ids=(0,),
            output_size=_OUTPUT_SIZE,
            page_size=_PAGE_SIZE,
            num_pages=num_stripes + 1,
            device="cpu",
        ),
        stripes=TapStripes(num_ranks=num_ranks, rank=rank),
    )


def _reservation(*, pages: list[int], num_taps: int, num_cached: int):
    """A request's stripes as ``RickyTapRequests`` takes them; the page lists
    read no lock."""
    return TapReservation(
        sandy_ids=list(range(num_taps)),
        pages=torch.tensor(pages),
        cached=CachedTaps(lock=None, pages=torch.tensor([]), num_taps=num_cached),
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

        self.assertTrue(bool(stripes.owns(pages).all()))
        self.assertEqual(stripes.stripe(pages).tolist(), pages.tolist())
        self.assertEqual(
            stripes.local_taps(torch.tensor([0, 3, 17]), page_size=4).tolist(),
            [0, 3, 17],
        )

    def test_ranks_keep_pages_in_turn(self):
        pages = np.arange(10)

        kept = {
            rank: pages[TapStripes(num_ranks=4, rank=rank).owns(pages)].tolist()
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

    def _reserve(self, cache: RickyTapCache, sandy_ids: list[int]) -> TapReservation:
        cached = cache.match(sandy_ids)
        num_stripes = -(-len(sandy_ids) // cache.stripe_size)
        new_pages = cache.take(num_stripes - len(cached.pages))
        return TapReservation(
            sandy_ids=sandy_ids,
            pages=torch.cat([cached.pages, new_pages]),
            cached=cached,
        )

    def _hold_and_cache(self, cache: RickyTapCache, sandy_ids: list[int]):
        reservation = self._reserve(cache, sandy_ids)
        return reservation.pages, cache.cache(reservation)

    def _cache(self, sandy_ids: list[int], cache=None) -> torch.Tensor:
        cache = cache or self.cache
        pages, lock = self._hold_and_cache(cache, sandy_ids)
        cache.release(lock)
        return pages

    def _free_and_evictable(self, cache=None) -> tuple[int, int]:
        usage = (cache or self.cache).usage()
        return usage.free_pages, usage.evictable_pages

    def test_the_pool_is_the_buffers_whole_stripes_over_all_ranks(self):
        self.assertEqual(self.cache.stripe_size, 8)
        self.assertEqual(self.cache.num_takeable_stripes(), 8)
        self.assertEqual(self.cache.usage().pages, 16)
        self.assertEqual(self.cache.buffers.num_pages, 9)

    def test_take_gives_distinct_pages_past_the_padding_page(self):
        pages = self.cache.take(5)

        self.assertEqual(pages.dtype, torch.int64)
        self.assertEqual(len(set(pages.tolist())), 5)
        self.assertTrue(all(1 <= page <= 8 for page in pages.tolist()))
        self.assertEqual(self.cache.num_takeable_stripes(), 3)

    def test_freed_pages_can_be_taken_again(self):
        pages = self.cache.take(8)

        self.cache.free(pages[:3])

        self.assertEqual(self.cache.num_takeable_stripes(), 3)
        self.assertEqual(
            sorted(self.cache.take(3).tolist()), sorted(pages[:3].tolist())
        )

    def test_a_cached_request_is_found_by_a_later_one_with_the_same_first_tokens(
        self,
    ):
        pages = self._cache(list(range(20)))

        cached = self.cache.match(list(range(20)) + [99, 98])

        self.assertEqual(cached.num_taps, 16)
        self.assertEqual(cached.pages.tolist(), pages[:2].tolist())
        self.cache.release(cached.lock)

    def test_a_match_counts_whole_stripes_only(self):
        self._cache(list(range(20)))

        cached = self.cache.match(list(range(15)))

        self.assertEqual(cached.num_taps, 8)
        self.cache.release(cached.lock)

    def test_a_match_without_a_cached_prefix_holds_nothing(self):
        cached = self.cache.match(list(range(20)))

        self.assertEqual(cached.num_taps, 0)
        self.assertEqual(cached.pages.tolist(), [])
        self.cache.release(cached.lock)

    def test_cache_gives_back_the_partial_last_stripe(self):
        self._cache(list(range(20)))

        self.assertEqual(self.cache.num_takeable_stripes(), 8)
        self.assertEqual(self._free_and_evictable(), (12, 4))

    def test_cache_gives_back_stripes_another_request_cached_first(self):
        first = self._reserve(self.cache, list(range(16)))
        second = self._reserve(self.cache, list(range(16)))

        self.cache.release(self.cache.cache(first))
        self.cache.release(self.cache.cache(second))

        self.assertEqual(self._free_and_evictable(), (12, 4))
        cached = self.cache.match(list(range(16)))
        self.assertEqual(cached.pages.tolist(), first.pages.tolist())
        self.cache.release(cached.lock)

    def test_a_request_that_started_from_a_cached_prefix_caches_only_its_new_stripes(
        self,
    ):
        prefix_pages = self._cache(list(range(16)))

        pages = self._cache(list(range(32)))

        self.assertEqual(pages[:2].tolist(), prefix_pages[:2].tolist())
        self.assertEqual(self._free_and_evictable(), (8, 8))
        cached = self.cache.match(list(range(32)))
        self.assertEqual(cached.pages.tolist(), pages.tolist())
        self.cache.release(cached.lock)

    def test_drop_gives_back_the_new_stripes_and_leaves_the_cached_prefix(self):
        self._cache(list(range(16)))
        reservation = self._reserve(self.cache, list(range(40)))
        self.assertEqual(self.cache.num_takeable_stripes(), 3)

        self.cache.drop(reservation)

        self.assertEqual(self.cache.num_takeable_stripes(), 8)
        self.assertEqual(self._free_and_evictable(), (12, 4))
        cached = self.cache.match(list(range(40)))
        self.assertEqual(cached.num_taps, 16)
        self.cache.release(cached.lock)

    def test_a_locked_stripe_is_never_evicted(self):
        _, lock = self._hold_and_cache(self.cache, list(range(16)))

        self.assertEqual(self.cache.num_takeable_stripes(), 6)
        self.assertEqual(self._free_and_evictable(), (12, 0))
        self.cache.release(lock)
        self.assertEqual(len(self.cache.take(8)), 8)
        self.assertEqual(self.cache.match(list(range(16))).num_taps, 0)

    def test_take_evicts_stripes_that_only_the_tree_keeps(self):
        self._cache(list(range(32)))
        self._cache(list(range(100, 132)))

        taken = self.cache.take(6)

        self.assertEqual(len(taken), 6)
        self.assertEqual(self.cache.num_takeable_stripes(), 2)

    def test_a_match_holds_its_prefix_until_released(self):
        self._cache(list(range(16)))
        cached = self.cache.match(list(range(16)))

        self.assertEqual(self.cache.num_takeable_stripes(), 6)
        self.cache.release(cached.lock)
        self.assertEqual(len(self.cache.take(7)), 7)

    def test_pages_to_land_mark_the_pages_other_ranks_keep_past_the_prefix(self):
        reservation = _reservation(pages=[5, 2, 7], num_taps=22, num_cached=8)

        rank_pages = [
            _ricky_cache(num_ranks=2, rank=rank).pages_to_land(reservation)
            for rank in range(2)
        ]

        self.assertEqual(rank_pages[0].dtype, np.int32)
        self.assertEqual(rank_pages[0].tolist(), [2, -1, 7, -1])
        self.assertEqual(rank_pages[1].tolist(), [-1, 2, -1, 7])

    def test_pages_to_relay_mark_every_page_from_the_first(self):
        reservation = _reservation(pages=[5, 2, 7], num_taps=22, num_cached=8)

        pages = _ricky_cache(num_ranks=2, rank=1).pages_to_relay(reservation)

        self.assertEqual(pages.dtype, np.int32)
        self.assertEqual(pages.tolist(), [-1, 5, -1, 2, -1, 7])

    def test_one_rank_lists_a_page_for_every_request_page(self):
        cache = _ricky_cache(num_ranks=1, rank=0)
        reservation = _reservation(pages=[3, 1, 6], num_taps=12, num_cached=0)

        self.assertEqual(cache.pages_to_land(reservation).tolist(), [3, 1, 6])
        self.assertEqual(cache.pages_to_relay(reservation).tolist(), [3, 1, 6])

    def test_every_rank_takes_the_same_pages_for_the_same_requests(self):
        caches = [_ricky_cache(num_ranks=2, rank=rank) for rank in range(2)]

        pages = []
        for cache in caches:
            self._cache(list(range(24)), cache=cache)
            pages.append(cache.take(3).tolist())

        self.assertEqual(pages[0], pages[1])

    def test_reset_drops_every_cached_stripe(self):
        self._cache(list(range(32)))

        self.cache.reset()

        self.assertEqual(self.cache.match(list(range(32))).num_taps, 0)
        self.assertEqual(self._free_and_evictable(), (16, 0))


if __name__ == "__main__":
    unittest.main()
