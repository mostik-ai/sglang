"""Taps share KV's life because each KV page has one fixed tap page, and Ricky's
chunks run on slots taken at arrival. A wrong slot here silently feeds a reader
another token's tap.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.franky.cache.kv_allocator import FrankyKvAllocator
from sglang.srt.franky.cache.tap_cache import (
    TAP_DTYPE,
    FrankyReqState,
    FrankyTapCache,
)
from sglang.srt.mem_cache import allocation
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_PAGE_SIZE = 4
_NUM_TOKENS = 32
_OUTPUT_SIZE = 8
_NUM_REQUEST_ROWS = 4


def _allocator() -> FrankyKvAllocator:
    allocator = FrankyKvAllocator(
        _NUM_TOKENS,
        page_size=_PAGE_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    allocator.use_triton_kernels = False
    return allocator


def _tap_cache() -> FrankyTapCache:
    return FrankyTapCache(
        encoder_ids=(0, 1),
        output_size=_OUTPUT_SIZE,
        page_size=_PAGE_SIZE,
        num_tokens=_NUM_TOKENS,
        req_to_token=torch.zeros((_NUM_REQUEST_ROWS, _NUM_TOKENS), dtype=torch.int32),
        device="cpu",
    )


def _taps(num_tokens: int, value: float) -> torch.Tensor:
    return torch.full((num_tokens, _OUTPUT_SIZE), value, dtype=TAP_DTYPE)


class TestFrankyTapCache(CustomTestCase):
    def test_every_slot_of_a_full_kv_pool_has_its_own_tap(self):
        allocator, cache = _allocator(), _tap_cache()
        kv_slots = allocator.alloc(_NUM_TOKENS)
        self.assertIsNone(allocator.alloc(_PAGE_SIZE))

        values = torch.arange(1, _NUM_TOKENS + 1, dtype=TAP_DTYPE)[:, None]
        cache.write(
            encoder_id=0, taps=values.expand(-1, _OUTPUT_SIZE), kv_slots=kv_slots
        )

        tap_slots = cache.slots_of(kv_slots=kv_slots)
        self.assertTrue(torch.equal(cache._buffers[0][tap_slots, :1], values))

    def test_padded_tokens_touch_no_real_slot(self):
        """Padded tokens write to KV slot 0, on a page the allocator never gives."""
        allocator, cache = _allocator(), _tap_cache()
        kv_slots = allocator.alloc(_NUM_TOKENS)
        cache.write(encoder_id=0, taps=_taps(_NUM_TOKENS, 1.0), kv_slots=kv_slots)

        cache.write(
            encoder_id=0, taps=_taps(1, 9.0), kv_slots=torch.zeros(1, dtype=torch.int64)
        )

        tap_slots = cache.slots_of(kv_slots=kv_slots)
        self.assertTrue(torch.all(cache._buffers[0][tap_slots] == 1.0))

    def test_padded_requests_read_zeros(self):
        """A padded request reads page 0, and a reader must add nothing for it."""
        self.assertTrue(torch.all(_tap_cache()._buffers[1][:_PAGE_SIZE] == 0))

    def test_encoders_do_not_share_taps(self):
        cache = _tap_cache()
        kv_slots = torch.arange(_PAGE_SIZE, 2 * _PAGE_SIZE)

        cache.write(encoder_id=0, taps=_taps(_PAGE_SIZE, 1.0), kv_slots=kv_slots)

        self.assertTrue(torch.all(cache._buffers[1] == 0))

    def test_transfer_sees_whole_pages_of_each_buffer(self):
        """NIXL reaches page ``p`` of a buffer at ``address + p * page_bytes``."""
        cache = _tap_cache()
        buffers = cache.transfer_buffers()
        page_bytes = _PAGE_SIZE * _OUTPUT_SIZE * TAP_DTYPE.itemsize
        num_pages = _NUM_TOKENS // _PAGE_SIZE + 1

        self.assertEqual(cache.page_bytes, page_bytes)
        self.assertEqual(sorted(buffers), [0, 1])
        self.assertEqual(len({address for address, _ in buffers.values()}), 2)
        self.assertEqual(
            [size for _, size in buffers.values()], [page_bytes * num_pages] * 2
        )


class TestFrankyKvAllocator(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.allocator = _allocator()
        self.tree_cache = SimpleNamespace(token_to_kv_pool_allocator=self.allocator)
        self.make_room_demands = []

    def _take_arrival_slots(self, *, prefix_len: int, num_tokens: int):
        taken = self.allocator.alloc(num_tokens)
        return SimpleNamespace(
            franky=FrankyReqState(
                tap_len=0,
                taps_ready=prefix_len + num_tokens,
                taken_kv_slots=taken,
                taken_from=prefix_len,
                used_end=prefix_len,
            )
        )

    def _extend(self, *, prefix_lens: list[int], seq_lens: list[int]):
        prefix_lens_cpu = torch.tensor(prefix_lens)
        seq_lens_cpu = torch.tensor(seq_lens)
        with patch.object(
            allocation,
            "evict_from_tree_cache",
            lambda tree_cache, num_tokens: self.make_room_demands.append(num_tokens),
        ):
            return allocation.alloc_paged_token_slots_extend(
                self.tree_cache,
                prefix_lens=prefix_lens_cpu,
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=seq_lens_cpu,
                seq_lens_cpu=seq_lens_cpu,
                last_loc=torch.full((len(seq_lens),), -1),
                extend_num_tokens=sum(seq_lens) - sum(prefix_lens),
            )

    def test_chunks_run_on_taken_slots_in_order(self):
        req = self._take_arrival_slots(prefix_len=4, num_tokens=8)
        taken = req.franky.taken_kv_slots
        free_pages_before = self.allocator.available_size()

        chunks = []
        for start, end in ((4, 7), (7, 12)):
            self.allocator.expect_extend(reqs=[req])
            chunks.append(self._extend(prefix_lens=[start], seq_lens=[end]))

        self.assertTrue(torch.equal(torch.cat(chunks), taken))
        self.assertEqual(self.allocator.available_size(), free_pages_before)

    def test_taken_slots_ask_for_no_eviction(self):
        req = self._take_arrival_slots(prefix_len=0, num_tokens=8)

        self.allocator.expect_extend(reqs=[req])
        self._extend(prefix_lens=[0], seq_lens=[8])

        self.assertEqual(self.make_room_demands, [0])

    def test_a_batch_keeps_its_request_order(self):
        first = self._take_arrival_slots(prefix_len=0, num_tokens=4)
        second = self._take_arrival_slots(prefix_len=8, num_tokens=4)

        self.allocator.expect_extend(reqs=[second, first])
        slots = self._extend(prefix_lens=[8, 0], seq_lens=[10, 3])

        self.assertTrue(
            torch.equal(
                slots,
                torch.cat(
                    (second.franky.taken_kv_slots[:2], first.franky.taken_kv_slots[:3])
                ),
            )
        )

    def test_a_wrong_batch_size_is_refused(self):
        req = self._take_arrival_slots(prefix_len=0, num_tokens=4)

        self.allocator.expect_extend(reqs=[req])
        with self.assertRaises(ValueError):
            self._extend(prefix_lens=[0, 0], seq_lens=[2, 2])

    def test_it_takes_new_pages_when_nothing_is_expected(self):
        """Such as PD decode, where ``_pre_alloc`` takes KV through this call."""
        free_before = self.allocator.available_size()

        slots = self._extend(prefix_lens=[0], seq_lens=[_PAGE_SIZE])

        self.assertEqual(len(slots), _PAGE_SIZE)
        self.assertEqual(self.allocator.available_size(), free_before - _PAGE_SIZE)
        self.assertGreater(self.make_room_demands[0], 0)

    def test_expect_extend_covers_one_batch_only(self):
        req = self._take_arrival_slots(prefix_len=0, num_tokens=4)
        self.allocator.expect_extend(reqs=[req])
        self._extend(prefix_lens=[0], seq_lens=[4])
        free_before = self.allocator.available_size()

        self._extend(prefix_lens=[0], seq_lens=[_PAGE_SIZE])

        self.assertEqual(self.allocator.available_size(), free_before - _PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
