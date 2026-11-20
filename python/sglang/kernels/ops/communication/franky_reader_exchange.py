"""The Franky readers' merge across the attention-TP ranks that split their taps,
in one kernel within a node.

Each rank stores its rows, packed as ``franky_reader_pack`` packs them, into its
slot of every rank's symmetric buffer over NVLink P2P; the ranks meet at a
barrier of epoch flags, one posted store a peer; then each rank merges every
rank's rows by their log-sum-exp. A block of rows always belongs to the same
program, and each program alternates between the two halves of the buffer from
one merge to the next: a rank that runs ahead into the next merge stores into
the other half, and passes that merge's barrier only once every rank started
it. So merges need nothing between them.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.franky_reader_merge import PACK_COLUMNS

# Query rows one program handles at a time. An exchange runs at most a fixed
# number of programs: block b of rows always belongs to program b % programs,
# and the signal pad keeps one flag a program and rank.
BLOCK_ROWS = 4


def signal_pad_bytes(*, world_size: int, programs: int) -> int:
    return programs * world_size * 4


@triton.jit
def _sync_threads():
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _load_128(ptr, mask):
    """16 bytes as four words, read past any cache that a peer's stores miss."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @!%p0 bra end;
            ld.relaxed.sys.global.v4.b32 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _store_128(ptr, x, y, z, w, mask):
    """16 bytes into a rank's memory, over NVLink P2P for a peer's."""
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
def _post_epoch(ptr, epoch):
    """Post ``epoch`` to a flag, after the thread's earlier stores."""
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
    """Spin until a flag reaches ``epoch``; signed, so epochs may wrap around."""
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
def _store_rows(
    local_out,
    lse,
    local_taps,
    no_taps,
    buffers,
    slot,
    num_query_rows,
    QUERIES: tl.constexpr,
    TAP_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    """Store this program's rows that read taps into ``slot`` of every rank's
    buffer: the attended taps, then the log-sum-exp, -inf where this rank keeps
    none of the row's taps.
    """
    pid = tl.program_id(0)
    row_words: tl.constexpr = ROW_SIZE // 4
    chunks: tl.constexpr = TAP_SIZE // 8
    chunk_ids = tl.arange(0, BLOCK * chunks)
    block_rows = tl.arange(0, BLOCK)
    block = pid
    while block < tl.cdiv(num_query_rows, BLOCK):
        # Every load first, none waiting on another; the stores skip the rows
        # that read no taps.
        rows = block * BLOCK + chunk_ids // chunks
        columns = chunk_ids % chunks
        in_rows = rows < num_query_rows
        source = (
            local_out.to(tl.pointer_type(tl.uint64))
            + rows * (TAP_SIZE // 4)
            + columns * 2
        )
        x, y, z, w = _load_128(source, in_rows)
        reads = in_rows & (
            tl.load(no_taps + rows // QUERIES, mask=in_rows, other=1) == 0
        )
        query_rows = block * BLOCK + block_rows
        in_query_rows = query_rows < num_query_rows
        row_lse = tl.load(lse + query_rows, mask=in_query_rows, other=float("-inf"))
        kept = (
            tl.load(local_taps + query_rows // QUERIES, mask=in_query_rows, other=0) > 0
        )
        row_reads = in_query_rows & (
            tl.load(no_taps + query_rows // QUERIES, mask=in_query_rows, other=1) == 0
        )
        row_lse = tl.where(kept & (row_lse == row_lse), row_lse, float("-inf"))
        lse_bits = row_lse.to(tl.uint32, bitcast=True)
        zeros = tl.zeros_like(lse_bits)
        for peer in tl.static_range(WORLD_SIZE):
            peer_slot = (
                tl.load(buffers + peer).to(tl.pointer_type(tl.uint64))
                + slot * row_words
            )
            _store_128(peer_slot + rows * row_words + columns * 2, x, y, z, w, reads)
            _store_128(
                peer_slot + query_rows * row_words + TAP_SIZE // 4,
                lse_bits,
                zeros,
                zeros,
                zeros,
                row_reads,
            )
        block += PROGRAMS


@triton.jit
def _flags(
    signal_pads, RANKS: tl.constexpr, RANK: tl.constexpr, WORLD_SIZE: tl.constexpr
):
    """Per lane, a rank's signal pad, and the rank whose flag the lane reads here.
    ``RANKS`` is ``WORLD_SIZE`` rounded up to a power of two; the extra lanes
    stand for this rank.
    """
    lanes = tl.arange(0, RANKS)
    ranks = tl.where(lanes < WORLD_SIZE, lanes, RANK)
    pads = signal_pads.to(tl.pointer_type(tl.uint64))
    peer_pads = tl.load(pads + ranks).to(tl.pointer_type(tl.uint32))
    own_pad = tl.load(pads + RANK).to(tl.pointer_type(tl.uint32))
    return peer_pads, own_pad, ranks


@triton.jit
def _meet(
    signal_pads,
    epoch,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    RANKS: tl.constexpr,
):
    """Every rank's program ``pid`` meets here: each posts ``epoch`` to its flag
    on every rank, then waits until every rank's flag here reaches it.
    """
    pid = tl.program_id(0)
    peer_pads, own_pad, ranks = _flags(signal_pads, RANKS, RANK, WORLD_SIZE)
    epochs = tl.zeros_like(ranks) + epoch
    _post_epoch(peer_pads + pid * WORLD_SIZE + RANK, epochs)
    _wait_epoch(own_pad + pid * WORLD_SIZE + ranks, epochs)
    _sync_threads()


@triton.jit
def _merge_rows(
    no_taps,
    out,
    has_taps,
    buffers,
    half_row,
    num_query_rows,
    slot_rows,
    has_taps_stride,
    QUERIES: tl.constexpr,
    TAP_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    LSE_IS_BASE_E: tl.constexpr,
):
    """Merge every rank's stored rows of this program by their log-sum-exp; a row
    that reads no taps gets zeros and a 0 in ``has_taps``, the others a 1.
    """
    pid = tl.program_id(0)
    own = tl.load(buffers + RANK).to(tl.pointer_type(tl.bfloat16))
    own_lse = own.to(tl.pointer_type(tl.float32))
    features = tl.arange(0, TAP_SIZE)
    block_rows = tl.arange(0, BLOCK)
    block = pid
    while block < tl.cdiv(num_query_rows, BLOCK):
        query_rows = block * BLOCK + block_rows
        in_rows = query_rows < num_query_rows
        reads = in_rows & (
            tl.load(no_taps + query_rows // QUERIES, mask=in_rows, other=1) == 0
        )
        lse_max = tl.full([BLOCK], float("-inf"), tl.float32)
        for rank in tl.static_range(WORLD_SIZE):
            stored = half_row + rank * slot_rows + query_rows
            rank_lse = tl.load(
                own_lse + stored * (ROW_SIZE // 2) + TAP_SIZE // 2,
                mask=in_rows,
                other=float("-inf"),
                volatile=True,
            )
            lse_max = tl.maximum(lse_max, tl.where(reads, rank_lse, float("-inf")))
        center = tl.where(lse_max == float("-inf"), 0.0, lse_max)
        acc = tl.zeros([BLOCK, TAP_SIZE], dtype=tl.float32)
        weight_sum = tl.zeros([BLOCK], dtype=tl.float32)
        for rank in tl.static_range(WORLD_SIZE):
            stored = half_row + rank * slot_rows + query_rows
            rank_lse = tl.load(
                own_lse + stored * (ROW_SIZE // 2) + TAP_SIZE // 2,
                mask=in_rows,
                other=float("-inf"),
                volatile=True,
            )
            rank_lse = tl.where(reads, rank_lse, float("-inf"))
            if LSE_IS_BASE_E:
                weight = tl.exp(rank_lse - center)
            else:
                weight = tl.exp2(rank_lse - center)
            values = tl.load(
                own + stored[:, None] * ROW_SIZE + features[None, :],
                mask=reads[:, None] & (weight[:, None] > 0),
                other=0.0,
                volatile=True,
            ).to(tl.float32)
            acc += values * weight[:, None]
            weight_sum += weight
        merged = acc / tl.maximum(weight_sum, 1.0)[:, None]
        tl.store(
            out + query_rows[:, None] * TAP_SIZE + features[None, :],
            merged.to(out.dtype.element_ty),
            mask=in_rows[:, None],
        )
        tl.store(
            has_taps + query_rows // QUERIES * has_taps_stride,
            reads.to(has_taps.dtype.element_ty),
            mask=in_rows & (query_rows % QUERIES == 0),
        )
        block += PROGRAMS


@triton.jit
def _exchange_merge_kernel(
    local_out,
    lse,
    local_taps,
    no_taps,
    out,
    has_taps,
    buffer_ptrs,
    signal_pad_ptrs,
    epochs,
    num_query_rows,
    slot_rows,
    has_taps_stride,
    QUERIES: tl.constexpr,
    TAP_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    RANKS: tl.constexpr,
    PROGRAMS: tl.constexpr,
    LSE_IS_BASE_E: tl.constexpr,
):
    # epochs[pid]: the merges this program ran on this rank; every rank runs the
    # same merges with the same programs, so their epochs agree.
    pid = tl.program_id(0)
    epoch = tl.load(epochs + pid) + 1
    half_row = (epoch & 1) * WORLD_SIZE * slot_rows
    buffers = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    _store_rows(
        local_out,
        lse,
        local_taps,
        no_taps,
        buffers,
        half_row + RANK * slot_rows,
        num_query_rows,
        QUERIES,
        TAP_SIZE,
        ROW_SIZE,
        BLOCK,
        WORLD_SIZE,
        PROGRAMS,
    )
    _sync_threads()
    _meet(signal_pad_ptrs, epoch, RANK, WORLD_SIZE, RANKS)
    _merge_rows(
        no_taps,
        out,
        has_taps,
        buffers,
        half_row,
        num_query_rows,
        slot_rows,
        has_taps_stride,
        QUERIES,
        TAP_SIZE,
        ROW_SIZE,
        BLOCK,
        RANK,
        WORLD_SIZE,
        PROGRAMS,
        LSE_IS_BASE_E,
    )
    tl.store(epochs + pid, epoch)


def franky_reader_exchange_merge(
    local_out: torch.Tensor,
    lse: torch.Tensor,
    local_taps: torch.Tensor,
    no_taps: torch.Tensor,
    *,
    out: torch.Tensor,
    has_taps: torch.Tensor,
    buffer_ptrs: torch.Tensor,
    signal_pad_ptrs: torch.Tensor,
    epochs: torch.Tensor,
    slot_rows: int,
    rank: int,
    world_size: int,
    programs: int,
    lse_is_base_e: bool,
) -> None:
    """Merge the attention-TP ranks' reader results by their log-sum-exp; every
    rank of the group runs it with the same rows.

    Args:
        local_out: [rows, queries, tap_size] bfloat16, contiguous, this rank's
            results.
        lse: [rows, queries] float32, their log-sum-exp.
        local_taps: [rows] int32, the taps each row reads on this rank.
        no_taps: [rows] bool, whether each row reads no taps on any rank.
        out: [rows, queries, tap_size] bfloat16, contiguous, written.
        has_taps: [rows, 1], written: 1 where the row reads taps, else 0.
        buffer_ptrs: every rank's buffer of two halves of ``world_size`` slots
            of ``slot_rows`` packed query rows, as int64 device pointers.
        signal_pad_ptrs: every rank's buffer signal pad, zeroed at allocation,
            of at least ``signal_pad_bytes(world_size, programs)`` bytes.
        epochs: [programs] int32 on this rank's device, zeroed before the
            first merge.
        slot_rows: at least ``rows * queries``.
        programs: the exchange's most programs, the same for all its merges.
    """
    rows, queries, tap_size = local_out.shape
    num_query_rows = rows * queries
    grid = (min(triton.cdiv(num_query_rows, BLOCK_ROWS), programs),)
    _exchange_merge_kernel[grid](
        local_out,
        lse,
        local_taps,
        no_taps,
        out,
        has_taps,
        buffer_ptrs,
        signal_pad_ptrs,
        epochs,
        num_query_rows,
        slot_rows,
        has_taps.stride(0),
        QUERIES=queries,
        TAP_SIZE=tap_size,
        ROW_SIZE=tap_size + PACK_COLUMNS,
        BLOCK=BLOCK_ROWS,
        RANK=rank,
        WORLD_SIZE=world_size,
        RANKS=triton.next_power_of_2(world_size),
        PROGRAMS=programs,
        LSE_IS_BASE_E=lse_is_base_e,
        num_warps=4,
    )
