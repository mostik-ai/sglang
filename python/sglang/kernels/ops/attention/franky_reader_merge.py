import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dcp_kernels import _dcp_lse_combine_kernel

LSE_COLUMNS = 2


@triton.jit
def _franky_reader_pack_kernel(
    out,
    lse,
    local_taps,
    packed,
    packed_lse,
    out_row_stride,
    packed_row_stride,
    packed_lse_row_stride,
    QUERIES: tl.constexpr,
    TAP_SIZE: tl.constexpr,
):
    query_row = tl.program_id(0)
    reads = tl.load(local_taps + query_row // QUERIES) > 0
    features = tl.arange(0, TAP_SIZE)
    values = tl.load(out + query_row * out_row_stride + features)
    tl.store(
        packed + query_row * packed_row_stride + features,
        tl.where(reads, values, 0.0).to(packed.dtype.element_ty),
    )
    row_lse = tl.load(lse + query_row)
    tl.store(
        packed_lse + query_row * packed_lse_row_stride + TAP_SIZE // 2,
        tl.where(reads, row_lse, float("-inf")),
    )


def franky_reader_pack(
    out: torch.Tensor, lse: torch.Tensor, local_taps: torch.Tensor
) -> torch.Tensor:
    """One rank's reader results, packed for the ranks to gather: each query's
    attended taps, then its log-sum-exp in the next two columns; a row that
    reads no taps on this rank gets zeros and -inf.

    Args:
        out: [rows, queries, tap_size] bfloat16.
        lse: [rows, queries] float32.
        local_taps: [rows] int32, the taps each row reads on this rank.
    Returns:
        [rows, queries, tap_size + 2] bfloat16.
    """
    rows, queries, tap_size = out.shape
    packed = out.new_empty((rows, queries, tap_size + LSE_COLUMNS))
    if rows == 0:
        return packed
    flat_out = out.view(rows * queries, tap_size)
    flat_packed = packed.view(rows * queries, tap_size + LSE_COLUMNS)
    flat_packed_lse = flat_packed.view(torch.float32)
    _franky_reader_pack_kernel[(rows * queries,)](
        flat_out,
        lse.view(-1),
        local_taps,
        flat_packed,
        flat_packed_lse,
        flat_out.stride(0),
        flat_packed.stride(0),
        flat_packed_lse.stride(0),
        QUERIES=queries,
        TAP_SIZE=tap_size,
    )
    return packed


def franky_reader_merge(
    gathered: torch.Tensor, *, out: torch.Tensor, lse_is_base_e: bool
) -> None:
    """Merge the ranks' packed results by their log-sum-exp into ``out``.

    Args:
        gathered: [ranks, rows, queries, tap_size + 2], every rank's
            ``franky_reader_pack`` result.
        out: [rows, queries, tap_size], written.
        lse_is_base_e: whether the log-sum-exp is natural, else base 2.
    """
    ranks, rows, queries, packed_size = gathered.shape
    if rows == 0:
        return
    tap_size = packed_size - LSE_COLUMNS
    results = gathered[..., :tap_size]
    lse = gathered.view(torch.float32)[..., tap_size // LSE_COLUMNS]
    _dcp_lse_combine_kernel[(rows, queries)](
        results,
        lse,
        out,
        lse.new_empty(0),
        results.stride(0),
        results.stride(1),
        results.stride(2),
        results.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N=ranks,
        HEAD_DIM=tap_size,
        IS_BASE_E=lse_is_base_e,
        RETURN_LSE=False,
    )
