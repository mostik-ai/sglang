"""DeepSeek V4 keeps sliding windows apart from its full pool. Ricky's requests
take full-pool pages at arrival, where their taps live for as long as their KV;
a chunk must run on those pages and take window slots as without Franky. A
window slot taken at arrival would pin a whole prompt in the small window pool,
and a full page taken twice would hand one token's taps to another.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.franky.cache.swa_kv_allocator import FrankySwaKvAllocator
from sglang.srt.franky.cache.tap_cache import FrankyReqState
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PAGE_SIZE = 4
_FULL_TOKENS = 64
_SWA_TOKENS = 16


def _allocator() -> FrankySwaKvAllocator:
    kvcache = MagicMock(spec=BaseSWAKVPool)
    kvcache.swa_req_ring_size = None
    allocator = FrankySwaKvAllocator(
        _FULL_TOKENS,
        _SWA_TOKENS,
        page_size=_PAGE_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kvcache,
        need_sort=False,
    )
    allocator.full_attn_allocator.use_triton_kernels = False
    allocator.swa_attn_allocator.use_triton_kernels = False
    # The real pool registers the allocator's mapping and reads it back.
    mapping = allocator.full_to_swa_index_mapping
    kvcache.full_to_swa_index_mapping = mapping
    kvcache.translate_loc_from_full_to_swa.side_effect = lambda slots: mapping[slots]
    return allocator


def _req(taken: torch.Tensor, *, prefix_len: int = 0):
    return SimpleNamespace(
        franky=FrankyReqState(
            tap_len=prefix_len + len(taken),
            taps_ready=prefix_len + len(taken),
            taken_kv_slots=taken,
            taken_from=prefix_len,
            used_end=prefix_len,
        )
    )


class _Tree:
    def __init__(self):
        self.evictions = []

    def evict_for_alloc(self, params):
        self.evictions.append(params)

    def full_evictable_size(self):
        return 8


class TestFrankySwaKvAllocator(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.allocator = _allocator()
        self.tree = _Tree()

    def _chunk(self, req, *, start, end):
        self.allocator.expect_extend(reqs=[req])
        return self.allocator.alloc_extend(
            prefix_lens=torch.tensor([start]),
            prefix_lens_cpu=torch.tensor([start]),
            seq_lens=torch.tensor([end]),
            seq_lens_cpu=torch.tensor([end]),
            last_loc=torch.tensor(
                [-1 if start == 0 else int(req.franky.taken_kv_slots[start - 1])]
            ),
            extend_num_tokens=end - start,
        )

    def test_an_arrival_takes_full_pages_and_no_window_slot(self):
        swa_before = self.allocator.swa_available_size()

        taken = self.allocator.take_at_arrival(tree_cache=self.tree, num_slots=32)

        self.assertEqual(len(taken), 32)
        self.assertEqual(self.allocator.full_available_size(), _FULL_TOKENS - 32)
        self.assertEqual(self.allocator.swa_available_size(), swa_before)
        self.assertTrue(torch.all(self.allocator.full_to_swa_index_mapping[taken] == 0))
        self.assertEqual(self.tree.evictions, [])

    def test_an_arrival_evicts_only_the_full_pool_shortfall(self):
        self.allocator.take_at_arrival(tree_cache=self.tree, num_slots=_FULL_TOKENS - 8)

        self.allocator.take_at_arrival(tree_cache=self.tree, num_slots=16)

        self.assertEqual(self.tree.evictions, [EvictParams(num_tokens=8)])

    def test_a_chunk_runs_on_its_taken_slots_and_takes_window_slots(self):
        taken = self.allocator.take_at_arrival(tree_cache=self.tree, num_slots=16)
        req = _req(taken)

        full = self._chunk(req, start=0, end=8)

        self.assertTrue(torch.equal(full, taken[:8]))
        self.assertEqual(self.allocator.full_available_size(), _FULL_TOKENS - 16)
        self.assertEqual(self.allocator.swa_available_size(), _SWA_TOKENS - 8)
        mapping = self.allocator.full_to_swa_index_mapping
        self.assertTrue(torch.all(mapping[taken[:8]] > 0))
        self.assertTrue(torch.all(mapping[taken[8:]] == 0))
        self.assertEqual(req.franky.used_end, 8)

    def test_freeing_a_request_returns_its_used_and_unused_slots(self):
        taken = self.allocator.take_at_arrival(tree_cache=self.tree, num_slots=16)
        self._chunk(_req(taken), start=0, end=8)

        self.allocator.free(taken)

        self.assertEqual(self.allocator.full_available_size(), _FULL_TOKENS)
        self.assertEqual(self.allocator.swa_available_size(), _SWA_TOKENS)

    def test_a_chunk_evicts_only_window_slots(self):
        self.allocator.expect_extend(reqs=[])

        self.allocator.evict_to_free_tokens(self.tree, _SWA_TOKENS + 4)

        self.assertEqual(self.tree.evictions, [EvictParams(swa_num_tokens=4)])

    def test_without_taken_slots_it_is_the_plain_sliding_window_allocator(self):
        full = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([8]),
            seq_lens_cpu=torch.tensor([8]),
            last_loc=torch.tensor([-1]),
            extend_num_tokens=8,
        )

        self.assertEqual(len(full), 8)
        self.assertEqual(self.allocator.full_available_size(), _FULL_TOKENS - 8)
        self.assertEqual(self.allocator.swa_available_size(), _SWA_TOKENS - 8)


class TestFrankySwaPrefillBudget(CustomTestCase):
    def test_only_the_windows_bound_a_prefill(self):
        allocator = _allocator()
        allocator.take_at_arrival(tree_cache=_Tree(), num_slots=_FULL_TOKENS)
        tree_cache = MagicMock()
        tree_cache.swa_evictable_size.return_value = 0
        tree_cache.sliding_window_size = 8

        budget = allocator.prefill_budget(
            tree_cache=tree_cache, num_mixed_decode_tokens=0
        )

        self.assertTrue(budget.has_capacity())
        self.assertTrue(
            budget.can_allocate_prefill(
                paged_input=_SWA_TOKENS // 2,
                extend_input_len=_SWA_TOKENS // 2,
                max_new_tokens=0,
                chunk_limit=None,
            )
        )
        self.assertFalse(
            budget.can_allocate_prefill(
                paged_input=2 * _SWA_TOKENS,
                extend_input_len=2 * _SWA_TOKENS,
                max_new_tokens=0,
                chunk_limit=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
