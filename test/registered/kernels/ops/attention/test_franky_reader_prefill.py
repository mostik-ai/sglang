import unittest

import torch

from sglang.kernels.ops.attention.franky_reader_blocks import franky_reader_row_blocks
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_PAGE_SIZE, _TAP_SIZE = 64, 256


def _reference(queries, taps, request_pages, taps_to_read, row_requests):
    rows = []
    for query, count, request in zip(queries, taps_to_read.tolist(), row_requests):
        if count == 0:
            rows.append(torch.zeros_like(query, dtype=torch.float32))
            continue
        pages = request_pages[request, : -(-count // _PAGE_SIZE)].long()
        read = taps[pages].reshape(-1, _TAP_SIZE)[:count].float()
        weights = torch.softmax(read @ query.float() * _TAP_SIZE**-0.5, dim=0)
        rows.append(weights @ read)
    return torch.stack(rows)


class TestFrankyReaderPrefill(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_rows_read_their_own_requests_prefixes(self):
        from sglang.kernels.ops.attention.franky_reader_prefill import (
            franky_reader_prefill,
        )

        generator = torch.Generator(device="cuda").manual_seed(0)
        rows_per_request = [1, 70, 0, 200, 3]
        taps_per_request = [5, 130, 0, 640, 64]
        num_pages = 64
        taps = torch.randn(
            num_pages, _PAGE_SIZE, _TAP_SIZE, generator=generator, device="cuda"
        ).to(torch.bfloat16)
        page_order = torch.randperm(num_pages, generator=generator, device="cuda")
        request_pages = page_order[: 5 * 10].reshape(5, 10).to(torch.int32)
        counts, row_requests = [], []
        for request, (num_rows, num_taps) in enumerate(
            zip(rows_per_request, taps_per_request)
        ):
            for row in range(num_rows):
                counts.append(0 if row % 17 == 3 else min(num_taps, row * 4 + 1))
                row_requests.append(request)
        taps_to_read = torch.tensor(counts, dtype=torch.int32, device="cuda")
        queries = torch.randn(
            len(counts), _TAP_SIZE, generator=generator, device="cuda"
        ).to(torch.bfloat16)
        out = torch.full_like(queries, float("nan"))

        franky_reader_prefill(
            queries,
            taps,
            request_pages,
            taps_to_read,
            franky_reader_row_blocks(rows_per_request, device=torch.device("cuda")),
            out=out,
        )

        expected = _reference(queries, taps, request_pages, taps_to_read, row_requests)
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
