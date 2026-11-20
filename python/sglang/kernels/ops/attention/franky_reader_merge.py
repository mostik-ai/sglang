import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dcp_kernels import _dcp_lse_combine_kernel

# SGLang's symmetric-memory all-gather keeps these PTX helpers private; the
# exchange below loads and syncs threads as that kernel does.
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    _local_ld_128,
    _sync_threads,
)

# Each packed row is the attended taps, then the log-sum-exp in two columns and
# padding, so rows stay whole 16-byte vectors for NVLink multicast stores.
PACK_COLUMNS = 8


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
    attended taps, then its log-sum-exp in the next two columns, then padding to
    ``PACK_COLUMNS``; a row that reads no taps on this rank gets zeros and -inf.

    Args:
        out: [rows, queries, tap_size] bfloat16.
        lse: [rows, queries] float32.
        local_taps: [rows] int32, the taps each row reads on this rank.
    Returns:
        [rows, queries, tap_size + PACK_COLUMNS] bfloat16.
    """
    rows, queries, tap_size = out.shape
    packed = out.new_empty((rows, queries, tap_size + PACK_COLUMNS))
    if rows == 0:
        return packed
    flat_out = out.view(rows * queries, tap_size)
    flat_packed = packed.view(rows * queries, tap_size + PACK_COLUMNS)
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
        gathered: [ranks, rows, queries, tap_size + PACK_COLUMNS], every
            rank's ``franky_reader_pack`` result.
        out: [rows, queries, tap_size], written.
        lse_is_base_e: whether the log-sum-exp is natural, else base 2.
    """
    ranks, rows, queries, packed_size = gathered.shape
    if rows == 0:
        return
    tap_size = packed_size - PACK_COLUMNS
    results = gathered[..., :tap_size]
    lse = gathered.view(torch.float32)[..., tap_size // 2]
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


# Query rows one program handles at a time, and the most programs an exchange
# runs: the barrier holds one flag per program and rank.
_EXCHANGE_BLOCK = 4
EXCHANGE_MAX_PROGRAMS = 64


@triton.jit
def _signal_epoch(ptr, epoch):
    """Post ``epoch`` to a rank's flag, after this thread's earlier stores."""
    return tl.inline_asm_elementwise(
        "st.release.sys.global.u32 [$1], $2;",
        "=r,l,r",
        [ptr, epoch],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_epoch(ptr, epoch):
    """Spin until this rank's flag reaches ``epoch``; the comparison is signed
    so the flag may wrap around.
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %flag;
            .reg .s32 %behind;
            .reg .pred %waits;
            wait_epoch:
                ld.acquire.sys.global.u32 %flag, [$1];
                sub.s32 %behind, %flag, $2;
                setp.lt.s32 %waits, %behind, 0;
                @%waits bra wait_epoch;
            mov.u32 $0, %flag;
        }
        """,
        "=r,l,r",
        [ptr, epoch],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _epoch_barrier(flag_ptrs, epochs, RANK: tl.constexpr, WORLD_SIZE: tl.constexpr):
    """Every rank's program ``pid`` meets here: each posts its next epoch to its
    slot of every rank's flags, one store a rank and no round trip, then waits
    until every rank's epoch reached its own flags. ``epochs`` counts each
    program's barriers on this rank, which every rank runs alike.
    """
    pid = tl.program_id(0)
    epoch = tl.load(epochs + pid) + 1
    ranks = tl.arange(0, WORLD_SIZE)
    flags = flag_ptrs.to(tl.pointer_type(tl.uint64))
    peer_flags = tl.load(flags + ranks).to(tl.pointer_type(tl.uint32))
    own_flags = tl.load(flags + RANK).to(tl.pointer_type(tl.uint32))
    every_epoch = tl.zeros_like(ranks) + epoch
    _signal_epoch(peer_flags + pid * WORLD_SIZE + RANK, every_epoch)
    _wait_epoch(own_flags + pid * WORLD_SIZE + ranks, every_epoch)
    _sync_threads()
    tl.store(epochs + pid, epoch)


@triton.jit
def _st_128(ptr, x, y, z, w, mask):
    """Store 16 bytes to a peer's memory over NVLink P2P."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            st.relaxed.sys.global.v4.b32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[ptr, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _franky_reader_exchange_merge_kernel(
    local_out,
    lse,
    local_taps,
    out,
    has_taps,
    buffer_ptrs,
    flag_ptrs,
    epochs,
    num_query_rows,
    buffer_rows,
    has_taps_stride,
    QUERIES: tl.constexpr,
    TAP_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    LSE_IS_BASE_E: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_blocks = tl.cdiv(num_query_rows, BLOCK)
    buffers = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    row_words: tl.constexpr = ROW_SIZE // 4
    chunks: tl.constexpr = TAP_SIZE // 8
    chunk_ids = tl.arange(0, BLOCK * chunks)
    block_rows = tl.arange(0, BLOCK)

    # Store this rank's rows, 16 bytes at a time, into its slot of every rank's
    # buffer: the attended taps, then the log-sum-exp, -inf where it reads none.
    block = pid
    while block < num_blocks:
        rows = block * BLOCK + chunk_ids // chunks
        columns = chunk_ids % chunks
        in_rows = rows < num_query_rows
        source = (
            local_out.to(tl.pointer_type(tl.uint64))
            + rows * (TAP_SIZE // 4)
            + columns * 2
        )
        x, y, z, w = _local_ld_128(source, in_rows)
        query_rows = block * BLOCK + block_rows
        in_query_rows = query_rows < num_query_rows
        row_lse = tl.load(lse + query_rows, mask=in_query_rows, other=float("-inf"))
        reads = (
            tl.load(local_taps + query_rows // QUERIES, mask=in_query_rows, other=0) > 0
        )
        row_lse = tl.where(reads & (row_lse == row_lse), row_lse, float("-inf"))
        lse_bits = row_lse.to(tl.uint32, bitcast=True)
        zeros = tl.zeros_like(lse_bits)
        for peer in tl.static_range(WORLD_SIZE):
            peer_buffer = tl.load(buffers + peer).to(tl.pointer_type(tl.uint64))
            slot = peer_buffer + RANK * buffer_rows * row_words
            _st_128(slot + rows * row_words + columns * 2, x, y, z, w, in_rows)
            _st_128(
                slot + query_rows * row_words + TAP_SIZE // 4,
                lse_bits,
                zeros,
                zeros,
                zeros,
                in_query_rows,
            )
        block += num_programs

    # Every rank's program `pid` has stored the same rows.
    _sync_threads()
    _epoch_barrier(flag_ptrs, epochs, RANK, WORLD_SIZE)

    # Merge each row's results by their log-sum-exp; a row no rank read taps
    # for gets zeros, and a 0 in ``has_taps``.
    own = tl.load(buffers + RANK).to(tl.pointer_type(tl.bfloat16))
    own_lse = own.to(tl.pointer_type(tl.float32))
    features = tl.arange(0, TAP_SIZE)
    block = pid
    while block < num_blocks:
        query_rows = block * BLOCK + block_rows
        in_query_rows = query_rows < num_query_rows
        lse_max = tl.full([BLOCK], float("-inf"), tl.float32)
        for rank in tl.static_range(WORLD_SIZE):
            slot_rows = rank * buffer_rows + query_rows
            rank_lse = tl.load(
                own_lse + slot_rows * (ROW_SIZE // 2) + TAP_SIZE // 2,
                mask=in_query_rows,
                other=float("-inf"),
                volatile=True,
            )
            lse_max = tl.maximum(lse_max, rank_lse)
        center = tl.where(lse_max == float("-inf"), 0.0, lse_max)
        acc = tl.zeros([BLOCK, TAP_SIZE], dtype=tl.float32)
        weight_sum = tl.zeros([BLOCK], dtype=tl.float32)
        for rank in tl.static_range(WORLD_SIZE):
            slot_rows = rank * buffer_rows + query_rows
            rank_lse = tl.load(
                own_lse + slot_rows * (ROW_SIZE // 2) + TAP_SIZE // 2,
                mask=in_query_rows,
                other=float("-inf"),
                volatile=True,
            )
            if LSE_IS_BASE_E:
                weight = tl.exp(rank_lse - center)
            else:
                weight = tl.exp2(rank_lse - center)
            values = tl.load(
                own + slot_rows[:, None] * ROW_SIZE + features[None, :],
                mask=in_query_rows[:, None],
                other=0.0,
                volatile=True,
            ).to(tl.float32)
            acc += tl.where(weight[:, None] > 0, values * weight[:, None], 0.0)
            weight_sum += weight
        merged = acc / tl.maximum(weight_sum, 1.0)[:, None]
        tl.store(
            out + query_rows[:, None] * TAP_SIZE + features[None, :],
            merged.to(out.dtype.element_ty),
            mask=in_query_rows[:, None],
        )
        reads = (lse_max > float("-inf")).to(has_taps.dtype.element_ty)
        tl.store(
            has_taps + query_rows // QUERIES * has_taps_stride,
            reads,
            mask=in_query_rows & (query_rows % QUERIES == 0),
        )
        block += num_programs


def franky_reader_exchange_merge(
    local_out: torch.Tensor,
    lse: torch.Tensor,
    local_taps: torch.Tensor,
    *,
    out: torch.Tensor,
    has_taps: torch.Tensor,
    buffer_ptrs: torch.Tensor,
    flag_ptrs: torch.Tensor,
    epochs: torch.Tensor,
    buffer_rows: int,
    rank: int,
    world_size: int,
    lse_is_base_e: bool,
) -> None:
    """Merge the attention-TP ranks' reader results in one kernel: each rank
    stores its rows, packed as ``franky_reader_pack`` packs them, into its slot
    of every rank's symmetric buffer over NVLink P2P; once every rank has stored
    them, each rank merges them by their log-sum-exp. A row no rank read taps
    for gets zeros, and a 0 in ``has_taps``; the others a 1.

    Every rank of the group runs it with the same rows, at the same point of its
    stream; between two runs over one buffer some collective must order them.

    Args:
        local_out: [rows, queries, tap_size] bfloat16, this rank's results.
        lse: [rows, queries] float32, their log-sum-exp.
        local_taps: [rows] int32, the taps each row reads on this rank.
        out: [rows, queries, tap_size] bfloat16, written.
        has_taps: [rows, 1], written: whether any rank read taps for each row.
        buffer_ptrs: every rank's buffer of ``world_size * buffer_rows`` packed
            rows, as int64 device pointers.
        flag_ptrs: every rank's ``EXCHANGE_MAX_PROGRAMS * world_size`` uint32
            barrier flags, zeroed before the first exchange, as int64 device
            pointers.
        epochs: [EXCHANGE_MAX_PROGRAMS] int32, this rank's barriers so far per
            program, zeroed with the flags.
        buffer_rows: the query rows of one rank's slot of a buffer.
    """
    rows, queries, tap_size = local_out.shape
    num_query_rows = rows * queries
    if num_query_rows == 0:
        return
    assert num_query_rows <= buffer_rows, (num_query_rows, buffer_rows)
    assert local_out.is_contiguous() and lse.is_contiguous() and out.is_contiguous()
    assert local_out.data_ptr() % 16 == 0 and tap_size % 8 == 0
    num_blocks = triton.cdiv(num_query_rows, _EXCHANGE_BLOCK)
    _franky_reader_exchange_merge_kernel[(min(num_blocks, EXCHANGE_MAX_PROGRAMS),)](
        local_out,
        lse,
        local_taps,
        out,
        has_taps,
        buffer_ptrs,
        flag_ptrs,
        epochs,
        num_query_rows,
        buffer_rows,
        has_taps.stride(0),
        QUERIES=queries,
        TAP_SIZE=tap_size,
        ROW_SIZE=tap_size + PACK_COLUMNS,
        BLOCK=_EXCHANGE_BLOCK,
        RANK=rank,
        WORLD_SIZE=world_size,
        LSE_IS_BASE_E=lse_is_base_e,
        num_warps=4,
    )
