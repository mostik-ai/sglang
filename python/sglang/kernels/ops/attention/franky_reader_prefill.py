import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.franky_reader_blocks import ROWS_PER_BLOCK


@triton.jit
def _franky_reader_prefill_kernel(
    queries,
    taps,
    request_pages,
    taps_to_read,
    row_blocks,
    out,
    query_row_stride,
    tap_page_stride,
    request_pages_stride,
    out_row_stride,
    scale,
    PAGE_SIZE: tl.constexpr,
    TAP_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
):
    block = tl.program_id(0)
    first_row = tl.load(row_blocks + block * 3)
    end_row = tl.load(row_blocks + block * 3 + 1)
    request = tl.load(row_blocks + block * 3 + 2).to(tl.int64)
    rows = first_row + tl.arange(0, ROWS)
    in_block = rows < end_row
    features = tl.arange(0, TAP_SIZE)
    page_offsets = tl.arange(0, PAGE_SIZE)
    row_taps = tl.load(taps_to_read + rows, mask=in_block, other=0)
    most_taps = tl.max(row_taps, axis=0)
    query = tl.load(
        queries + rows[:, None] * query_row_stride + features[None, :],
        mask=in_block[:, None],
        other=0.0,
    )
    running_max = tl.full([ROWS], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([ROWS], dtype=tl.float32)
    attended = tl.zeros([ROWS, TAP_SIZE], dtype=tl.float32)
    for first_tap in range(0, most_taps, PAGE_SIZE):
        page = tl.load(
            request_pages + request * request_pages_stride + first_tap // PAGE_SIZE
        ).to(tl.int64)
        page_taps = tl.load(
            taps
            + page * tap_page_stride
            + page_offsets[:, None] * TAP_SIZE
            + features[None, :]
        )
        scores = tl.dot(query, tl.trans(page_taps)) * scale
        readable = (first_tap + page_offsets)[None, :] < row_taps[:, None]
        scores = tl.where(readable, scores, float("-inf"))
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        shift = tl.where(new_max == float("-inf"), 0.0, new_max)
        weights = tl.exp(scores - shift[:, None])
        decay = tl.exp(running_max - shift)
        running_sum = running_sum * decay + tl.sum(weights, axis=1)
        attended = attended * decay[:, None] + tl.dot(
            weights.to(page_taps.dtype), page_taps
        )
        running_max = new_max
    attended = attended / tl.where(running_sum > 0, running_sum, 1.0)[:, None]
    tl.store(
        out + rows[:, None] * out_row_stride + features[None, :],
        attended.to(out.dtype.element_ty),
        mask=in_block[:, None],
    )


def franky_reader_prefill(
    queries: torch.Tensor,
    taps: torch.Tensor,
    request_pages: torch.Tensor,
    taps_to_read: torch.Tensor,
    row_blocks: torch.Tensor,
    *,
    out: torch.Tensor,
) -> None:
    """Attention of every row over the first ``taps_to_read[row]`` taps of its
    request, the taps being both keys and values; a row with none gets zeros.

    Args:
        queries: [rows, tap_size], one query a row.
        taps: [pages, page_size, tap_size] tap pages.
        request_pages: [requests, pages] int32, each request's tap pages in order.
        taps_to_read: [rows] int32.
        row_blocks: ``franky_reader_row_blocks`` of the same rows.
        out: [rows, tap_size], written.
    """
    if row_blocks.shape[0] == 0:
        return
    _, page_size, tap_size = taps.shape
    _franky_reader_prefill_kernel[(row_blocks.shape[0],)](
        queries,
        taps,
        request_pages,
        taps_to_read,
        row_blocks,
        out,
        queries.stride(0),
        taps.stride(0),
        request_pages.stride(0),
        out.stride(0),
        tap_size**-0.5,
        PAGE_SIZE=page_size,
        TAP_SIZE=tap_size,
        ROWS=ROWS_PER_BLOCK,
        num_warps=8,
        num_stages=1,
    )
