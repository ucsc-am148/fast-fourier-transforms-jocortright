"""STUDENT FILE: implement the Triton kernels and pipeline drivers.

You implement:
  - Six @triton.jit kernels: f1_kernel, f2_kernel, transpose_kernel,
    f4_kernel_L2, dft_kernel, bailey_scale_kernel.
  - The f1_launch and f2_launch grid-choice wrappers around them.
  - The pipeline drivers: f3_launch, f5_launch, _f6_rec, _f7_rec.
  - f6_factor: the chunk-recipe for F6/F7.

You do NOT implement (left given below):
  - The thin launch wrappers _transpose, _fft_chunk, _scale, _lookup_tw.
    These are mechanical "pick the grid and launch one kernel" helpers.
  - The tuning constants F4_L2_BLOCK_B, DFT_BLOCK_B, SCALE_BLOCK,
    TRANSPOSE_BLOCK.

The signatures below are the ones the harness calls -- your job is to fill
the bodies. When your code passes sanity_check.py, you're done.
"""

import math

import torch
import triton
import triton.language as tl


# Tunings -- GIVEN.
F4_L2_BLOCK_B = 2
DFT_BLOCK_B = 16
SCALE_BLOCK = 32
TRANSPOSE_BLOCK = 32


# =============================================================================
# Device-function helper: complex matmul
# =============================================================================
# Implement this once -- f1_kernel, f4_kernel_L2, and dft_kernel all call it.

@triton.jit
def _cdot(a_re, a_im, b_re, b_im):
    """Complex matmul Y = A @ B as four real tl.dot calls.

    Returns (y_re, y_im) in fp32 (out_dtype=tl.float32). Caller is responsible
    for any fp16 down-cast on store. Works at any matmul shape tl.dot accepts.

    Used by f1_kernel, f4_kernel_L2, and dft_kernel. Don't reimplement the
    four-tl.dot expansion at each call site -- implement once here, call
    everywhere.

    TODO: implement.
    """
    # dots
    rere = tl.dot(a_re, b_re, out_dtype = tl.float32)
    imim = tl.dot(a_im, b_im, out_dtype = tl.float32)
    imre = tl.dot(a_im, b_re, out_dtype = tl.float32)
    reim = tl.dot(a_re, b_im, out_dtype = tl.float32)

    # sums
    y_re = rere - imim
    y_im = imre + reim

    # outputs
    return y_re, y_im


# =============================================================================
# Chunk factorization for F6 / F7
# =============================================================================

def f6_factor(N: int) -> list[int]:
    """Factor N = 2^k into FFT chunks.

    Recipe: prefer 256-length chunks (radix-256, handled by f4_kernel_L2), then
    16-length (handled by dft_kernel via the padded radix-16 path), then a
    small leftover in {2, 4, 8} for the remaining bits. chunks[0] is the
    innermost (fastest) input axis. Examples:
        256 -> [256]                4096 -> [256, 16]
        65536 -> [256, 256]         1048576 -> [256, 256, 16]
        64 -> [16, 4]               2 -> [2]
    """
    assert N >= 2 and (N & (N - 1)) == 0, f"N must be a power of 2 >= 2; got {N}"
    k = N.bit_length() - 1
    n256, rb = divmod(k, 8)
    n16, rb2 = divmod(rb, 4)
    rsmall = 1 << rb2
    chunks = [256] * n256 + [16] * n16
    if (rsmall > 1):
        chunks.append(rsmall)
    assert math.prod(chunks) == N
    return chunks

f7_factor = f6_factor   # F7 reuses F6's chunk recipe


# =============================================================================
# F1: DFT as one dense complex matmul (four tl.dot)
# =============================================================================

@triton.jit
def f1_kernel(
    x_re_ptr, x_im_ptr,    # (B, N) fp16
    W_re_ptr, W_im_ptr,    # (N, N) fp16; W[n, k]
    y_re_ptr, y_im_ptr,    # (B, N) fp32
    B,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Y = X @ W^T as four (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) tl.dot calls.

    Y[b, n] = sum_k X[b, k] * W[n, k]. Load W in transposed access
    (W_T[k, n] = W[n, k]) so tl.dot reads it the way it wants.

    Use `_cdot(x_re, x_im, W_T_re, W_T_im)` for the per-block complex matmul;
    accumulate its fp32 output into `acc_re` / `acc_im`.

    Dtype contract (same as F4): loads are fp16, `tl.dot` runs with
    `out_dtype=tl.float32` (handled by `_cdot`), accumulator is fp32, store
    is fp32. Allocations in `f1_alloc` already match this -- x_re/x_im are
    fp16, y_re/y_im are fp32.
    """
    # program ids
    b_pid = tl.program_id(axis = 0)
    n_pid = tl.program_id(axis = 1)
    
    # offsets for rows and column
    b_offs = b_pid*BLOCK_M + tl.arange(0, BLOCK_M) # how far across for x
    n_offs = n_pid*BLOCK_N + tl.arange(0, BLOCK_N) # how far across for W
    
    # define masks
    b_mask = b_offs[:, None] < B
    n_mask = n_offs[None, :] < N    

    # initialize accumulators
    acc_re = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
    acc_im = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for k in range (0, N, BLOCK_K):
        # define offset and mask for k
        k_offs = k + tl.arange(0, BLOCK_K)
        x_k_mask = k_offs[None, :] < N
        W_T_k_mask = k_offs[:, None] < N

        # define offsets for x
        x_offs = b_offs[:, None]*N + k_offs[None, :]

        # define offsets for W_T
        W_T_offs = n_offs[None, :]*N + k_offs[:, None]
        
        # define masks
        x_mask = b_mask & x_k_mask
        W_T_mask = W_T_k_mask & n_mask

        # load vectors and matrices
        x_re = tl.load(x_re_ptr + x_offs, mask = x_mask, other = 0.0)
        x_im = tl.load(x_im_ptr + x_offs, mask = x_mask, other = 0.0)
        W_T_re = tl.load(W_re_ptr + W_T_offs, mask = W_T_mask, other = 0.0)
        W_T_im = tl.load(W_im_ptr + W_T_offs, mask = W_T_mask, other = 0.0)

        # update accumulators
        y_re, y_im = _cdot(x_re, x_im, W_T_re, W_T_im)
        acc_re += y_re
        acc_im += y_im
        
    # define offset
    offs = b_offs[:, None]*N + n_offs[None, :]
    
    # define mask
    mask = b_mask & n_mask

    # store results
    tl.store(y_re_ptr + offs, acc_re, mask = mask)
    tl.store(y_im_ptr + offs, acc_im, mask = mask)


def f1_launch(x_re, x_im, W_re, W_im, y_re, y_im):
    """Grid: (cdiv(B, BLOCK_M), cdiv(N, BLOCK_N)). One program tiles a
    (BLOCK_M, BLOCK_N) output square. tl.dot needs all three dims >=16, so B
    should be >= 16.
    """
    # dimensions
    B, N = x_re.shape
 
    # block sizes
    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_K = 16
    
    # grid
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # function call
    f1_kernel[grid](x_re, x_im, W_re, W_im, y_re, y_im, B, N,
                    BLOCK_M = BLOCK_M, BLOCK_N = BLOCK_N, BLOCK_K = BLOCK_K)


# =============================================================================
# F2: radix-2 Cooley-Tukey, single program per signal
# =============================================================================
# F3 reuses this kernel! For F2, only BAILEY_EPILOGUE=False, STRIDED_STORE=False need to be implemented.
#
# Call-site cheatsheet:
#   F2 vanilla:  pid -> one signal in (B, N). Grid: (B,).
#                BAILEY_EPILOGUE=False, STRIDED_STORE=False.
#                OUTER_DIM and N_TOTAL unused (pass 1 / 0).
#                bt_*_ptr: pass tw_*_ptr again (sentinel; never read).
#   F2-A (F3):   pid -> (b, n1). Grid: (B*N1,). FFT length N=N2.
#                BAILEY_EPILOGUE=True, STRIDED_STORE=False.
#                OUTER_DIM=N1 (n1 = pid % N1).
#                bt_*_ptr: real Bailey twiddles shape (N1, N2).
#   F2-B (F3):   pid -> (b, k2). Grid: (B*N2,). FFT length N=N1.
#                BAILEY_EPILOGUE=False, STRIDED_STORE=True.
#                OUTER_DIM=N2, N_TOTAL=N1*N2.
#                bt_*_ptr: sentinel.

@triton.jit
def f2_kernel(
    x_re_ptr, x_im_ptr,        # (B, N) fp32 input
    y_re_ptr, y_im_ptr,        # (B, N) fp32 output (layout depends on STRIDED_STORE)
    tw_re_ptr, tw_im_ptr,      # (N/2,) fp32 radix-2 twiddles
    perm_ptr,                   # (N,) int32 bit-reversal index
    bt_re_ptr, bt_im_ptr,       # (OUTER_DIM, N) fp32 Bailey twiddles (BAILEY_EPILOGUE only)
    OUTER_DIM, N_TOTAL,
    N: tl.constexpr,
    LOG2_N: tl.constexpr,
    BAILEY_EPILOGUE: tl.constexpr,
    STRIDED_STORE: tl.constexpr,
):
    """Radix-2 Cooley-Tukey FFT in registers, with optional Bailey epilogue and
    strided store. log2(N) butterfly stages via tl.gather for partner shuffle.
    """
    # program id
    pid = tl.program_id(axis = 0)

    # define indices spanning N, used multiple times
    j = tl.arange(0, N) 

    # load perm vector
    perm = tl.load(perm_ptr + j)
    
    # define x offset
    x_offs = pid*N + perm

    # load x vectors
    x_re = tl.load(x_re_ptr + x_offs)
    x_im = tl.load(x_im_ptr + x_offs)

    # define duplicate vectors to be manipulated
    v_re = x_re
    v_im = x_im

    # run each butterfly level
    for s in tl.static_range(0, LOG2_N):
        # define variable to represent 2^s in order to divide vectors evenly
        half = 1 << s

        # determine offset partner by bitwise XOR
        partner = j ^ half

        # pick out partner elements using tl.gather
        v_partner_re = tl.gather(v_re, partner, axis = 0)
        v_partner_im = tl.gather(v_im, partner, axis = 0)

        # divide N by 2^(s+1) to determine how many twiddles are needed
        num_twiddles = N >> (s + 1)

        # mask and calculate j mod 2^s to specify which twiddles to load
        mask = j & (half - 1)

        # define twiddle offset and load
        tw_offs = mask * num_twiddles
        tw_re = tl.load(tw_re_ptr + tw_offs)
        tw_im = tl.load(tw_im_ptr + tw_offs)

        # decide which half by comparing leading bit; if not 0 then upper half
        is_high = (j & half) != 0

        # load values depending on boolean; partner will never be in same half
        lo_re = tl.where(is_high, v_partner_re, v_re)
        lo_im = tl.where(is_high, v_partner_im, v_im)
        hi_re = tl.where(is_high, v_re, v_partner_re)
        hi_im = tl.where(is_high, v_im, v_partner_im)

        # calculate twiddles for high values
        tw_hi_re = tw_re*hi_re - tw_im*hi_im
        tw_hi_im = tw_re*hi_im + tw_im*hi_re

        # now the butterfly, add if upper half and subtract if lower half
        new_lo_re = lo_re - tw_hi_re
        new_lo_im = lo_im - tw_hi_im
        new_hi_re = lo_re + tw_hi_re
        new_hi_im = lo_im + tw_hi_im

        # store, ready for next loop/steps
        v_re = tl.where(is_high, new_lo_re, new_hi_re)
        v_im = tl.where(is_high, new_lo_im, new_hi_im)
    
# for F3 -----------------------------------------------------------------------
    if BAILEY_EPILOGUE:
        # calculate bailey twiddle offsets
        bt_offs = (pid % OUTER_DIM)*N + j

        # load bailey twiddle matrices
        bt_re = tl.load(bt_re_ptr + bt_offs)
        bt_im = tl.load(bt_im_ptr + bt_offs)

        # calculate new v
        v_re_new = v_re*bt_re - v_im*bt_im
        v_im_new = v_re*bt_im + v_im*bt_re

        # update v
        v_re = v_re_new
        v_im = v_im_new

    if STRIDED_STORE:
        # calculate stride
        stride = pid//OUTER_DIM

        # calculate y offsets
        y_offs = pid + OUTER_DIM*j + stride*(N_TOTAL - OUTER_DIM)
    
    else:
        # calculate y offsets without striding
        y_offs = pid*N + j
    
    # store results
    tl.store(y_re_ptr + y_offs, v_re)
    tl.store(y_im_ptr + y_offs, v_im)


def f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm):
    """Grid: (B,). One program per length-N signal. Vanilla mode.
    """
    # dimensions
    B, N = x_re.shape
    LOG2_N = math.ceil(math.log2(N))

    # booleans
    BAILEY_EPILOGUE = False
    STRIDED_STORE = False

    # unused constants/variables
    bt_re = 0
    bt_im = 0
    OUTER_DIM = 0
    N_TOTAL = 0

    # grid definition
    grid = (B,)

    #function call
    f2_kernel[grid](x_re, x_im, y_re, y_im, tw_re, tw_im, perm, bt_re, bt_im,
                    OUTER_DIM = OUTER_DIM, N_TOTAL = N_TOTAL, N = N,
                    LOG2_N = LOG2_N, BAILEY_EPILOGUE = BAILEY_EPILOGUE,
                    STRIDED_STORE = STRIDED_STORE)


# =============================================================================
# transpose_kernel: (B, R, C) -> (B, C, R), paired re/im
# =============================================================================

@triton.jit
def transpose_kernel(
    x_re_ptr, x_im_ptr,     # (B*R*C,) fp16 or fp32 input
    y_re_ptr, y_im_ptr,     # (B*R*C,) fp16 or fp32 output
    R, C,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Logical (B, R, C) -> (B, C, R) transpose. Grid: (cdiv(R, BLOCK_R),
    cdiv(C, BLOCK_C), B). Each program copies a (BLOCK_R, BLOCK_C) tile.
    """
    # define program ids
    r_pid = tl.program_id(axis = 0)
    c_pid = tl.program_id(axis = 1)
    b_pid = tl.program_id(axis = 2)

    # define offsets
    r_offs = r_pid*BLOCK_R + tl.arange(0, BLOCK_R)
    c_offs = c_pid*BLOCK_C + tl.arange(0, BLOCK_C)
    x_offs = b_pid*R*C + r_offs[:, None]*C + c_offs[None, :]
    y_offs = b_pid*R*C + c_offs[None, :]*R + r_offs[:, None]

    # define masks
    r_mask = r_offs[:, None] < R
    c_mask = c_offs[None, :] < C
    mask = r_mask & c_mask
    
    # load vectors
    x_re = tl.load(x_re_ptr + x_offs, mask = mask, other = 0.0)
    x_im = tl.load(x_im_ptr + x_offs, mask = mask, other = 0.0)

    # store matrices
    tl.store(y_re_ptr + y_offs, x_re, mask = mask)
    tl.store(y_im_ptr + y_offs, x_im, mask = mask)


# =============================================================================
# F4: tcFFT radix-16 single-program FFT (N = 256, L = 2)
# =============================================================================
# See the kernel docstring for the tl.permute tuple-literal gotcha.

@triton.jit
def f4_kernel_L2(
    x_re_ptr, x_im_ptr,    # (B, 256) fp16
    y_re_ptr, y_im_ptr,    # (B, 256) or (B//M, 256, M) fp16
    F_re_ptr, F_im_ptr,    # (16, 16) fp16 -- F_16 DFT matrix
    tw_re_ptr, tw_im_ptr,  # (L=2, 16, 16) fp16 stacked stage twiddles
    B, M,
    BLOCK_B: tl.constexpr,
    STAGE_STOP: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """tcFFT length-256 FFT as two stages of (permute + per-stage twiddle +
    length-16 DFT via four tl.dot). fp16 storage, fp32 matmul accumulators.

    `STAGE_STOP` and `M` are both degenerate in vanilla F4 (`STAGE_STOP=L=2`,
    `M=1`). They exist so the same kernel handles two extra uses:
      - `STAGE_STOP=1`: stop after the s=0 stage, for the sanity_check.py
        stage-1 isolation test (no twiddles, no second matmul).
      - `M>1` with `STORE_T=True`: F7's fused FFT-m_0+T3, writing the
        transposed (rows_outer, 256, M) layout the next level expects.

    STORE_T=False (M=1): natural (B, 256) row-major output.
    STORE_T=True  (M>1): transposed (B//M, 256, M) output for F7 fusion.

    Each stage's four-`tl.dot` is one `_cdot` call; cast its fp32 output to
    fp16 before the next stage.

    Dtype contract:
        Loads:           fp16
        Reshape/permute: fp16 (free)
        tl.dot inputs:   fp16, out_dtype=tl.float32  (use _cdot)
        Twiddle mul:     fp32 * fp16 -> fp32
        Inter-stage:     .to(tl.float16) before next iter's reshape
        Store:           fp16
    Forgetting the inter-stage cast doubles register pressure and passes the
    L=2 tolerance, but fails as soon as F6 stacks more stages.

    Triton 3.6 gotcha -- tl.permute requires LITERAL tuples:
        tl.permute(x, (1, 0, 2))                  # works
        perm = (1, 0, 2); tl.permute(x, perm)     # fails
    Inline each stage's permute tuple at the call site; don't store the
    schedule in a loop variable.
    """
    # program id
    pid = tl.program_id(axis = 0)

    # constants for dimension size
    BASE: tl.constexpr = 16
    L: tl.constexpr = 2
    N: tl.constexpr = 256 # = BASE^L

    # define offsets for x
    b_offs = pid*BLOCK_B + tl.arange(0, BLOCK_B)[:, None]
    n_offs = tl.arange(0, N)[None, :]
    x_offs = b_offs*N + n_offs
    
    # define offsets for F
    b_offs_F = tl.arange(0, BASE)[:, None]
    n_offs_F = tl.arange(0, BASE)[None, :]
    F_offs = b_offs_F*BASE + n_offs_F

    # define masks
    b_mask = b_offs < B
    mask = b_mask

    # load vectors and matrices
    x_re = tl.load(x_re_ptr + x_offs, mask = mask, other = 0.0)
    x_im = tl.load(x_im_ptr + x_offs, mask = mask, other = 0.0)
    F_re = tl.load(F_re_ptr + F_offs)
    F_im = tl.load(F_im_ptr + F_offs)

# this is where the calculations start------------------------------------------ 
    # calculate tile of x
    x_tile_re = tl.reshape(x_re, (BLOCK_B, BASE, BASE))
    x_tile_im = tl.reshape(x_im, (BLOCK_B, BASE, BASE))

    # permute tiles
    permute_x_tile_re = tl.permute(x_tile_re, (0, 2, 1))
    permute_x_tile_im = tl.permute(x_tile_im, (0, 2, 1))

    # convert to cdot-productable format (2D)
    dot_x_tile_re = tl.reshape(permute_x_tile_re, (BLOCK_B*BASE, BASE))
    dot_x_tile_im = tl.reshape(permute_x_tile_im, (BLOCK_B*BASE, BASE))

    # calculate cdot product
    dot_y_re, dot_y_im = _cdot(dot_x_tile_re, dot_x_tile_im, F_re, F_im)
    
    # convert dot_y_re and dot_y_im to fp16
    dot_y_re = dot_y_re.to(tl.float16)
    dot_y_im = dot_y_im.to(tl.float16)

    # reshape and permute to get back to original dimensions
    reshaped_y_re = tl.reshape(dot_y_re, (BLOCK_B, BASE, BASE))
    reshaped_y_im = tl.reshape(dot_y_im, (BLOCK_B, BASE, BASE))
    y_tile_re = tl.permute(reshaped_y_re, (0, 2, 1))
    y_tile_im = tl.permute(reshaped_y_im, (0, 2, 1))

    # all of the above is only for s=1, so we need to cover other cases
    if STAGE_STOP > 1:
        # reuse the output from the last stage
        x_tile_re = y_tile_re
        x_tile_im = y_tile_im

        # permute up here to make sure the right dimensions are twiddled
        permute_x_tile_re = tl.permute(x_tile_re, (0, 2, 1))
        permute_x_tile_im = tl.permute(x_tile_im, (0, 2, 1))

        # we need to load twiddles after stage 0
        m_offs = tl.arange(0, BASE)[:, None]
        c_offs = tl.arange(0, BASE)[None, :]
        tw_offs = N + m_offs*BASE + c_offs
        
        # load twiddles
        tw_re = tl.load(tw_re_ptr + tw_offs)
        tw_im = tl.load(tw_im_ptr + tw_offs)

        # twiddle tiles
        twiddled_re = permute_x_tile_re*tw_re - permute_x_tile_im*tw_im
        twiddled_im = permute_x_tile_re*tw_im + permute_x_tile_im*tw_re

        # convert to fp16
        x_tile_re = twiddled_re.to(tl.float16)
        x_tile_im = twiddled_im.to(tl.float16)

        # now proceed as usual
        # permute tiles
        permute_x_tile_re = tl.permute(x_tile_re, (0, 2, 1))
        permute_x_tile_im = tl.permute(x_tile_im, (0, 2, 1))

        # convert to cdot-productable format (2D)
        dot_x_tile_re = tl.reshape(permute_x_tile_re, (BLOCK_B*BASE, BASE))
        dot_x_tile_im = tl.reshape(permute_x_tile_im, (BLOCK_B*BASE, BASE))

        # calculate cdot product
        dot_y_re, dot_y_im = _cdot(dot_x_tile_re, dot_x_tile_im, F_re, F_im)
        
        # convert dot_y_re and dot_y_im to fp16
        dot_y_re = dot_y_re.to(tl.float16)
        dot_y_im = dot_y_im.to(tl.float16)

        # reshape and permute to get back to original dimensions
        reshaped_y_re = tl.reshape(dot_y_re, (BLOCK_B, BASE, BASE))
        reshaped_y_im = tl.reshape(dot_y_im, (BLOCK_B, BASE, BASE))
        y_tile_re = tl.permute(reshaped_y_re, (0, 2, 1))
        y_tile_im = tl.permute(reshaped_y_im, (0, 2, 1))     

    # reshape to prepare for final store
    y_re = tl.reshape(y_tile_re, (BLOCK_B, N))
    y_im = tl.reshape(y_tile_im, (BLOCK_B, N))

    # the offsets will be different if STORE_T is active
    if STORE_T:
        # calculate new offsets
        outer = b_offs//M
        inner = b_offs % M

        # combine to get total offset
        y_offs = outer*M*N + n_offs*M + inner
    else:
        # we don't really need to do anything fancy if STORE_T is inactive
        y_offs = x_offs

    tl.store(y_re_ptr + y_offs, y_re, mask = mask)
    tl.store(y_im_ptr + y_offs, y_im, mask = mask)


# =============================================================================
# dft_kernel: padded length-R DFT for the small chunks (R in {2, 4, 8, 16})
# =============================================================================

@triton.jit
def dft_kernel(
    x_re_ptr, x_im_ptr,     # (rows, R) fp16
    y_re_ptr, y_im_ptr,     # (rows, R) or (rows//M, R, M) fp16
    M_re_ptr, M_im_ptr,     # (16, 16) fp16 padded-R DFT matrix
    rows, M,
    R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Padded length-R DFT via a (16, 16) tl.dot. STORE_T toggles natural
    vs transposed output (same pattern as f4_kernel_L2).

    One `_cdot(x_re, x_im, MT_re, MT_im)` call replaces the four `tl.dot`
    expansions; cast its fp32 result to fp16 on store.
    """
    # program id
    pid = tl.program_id(axis = 0)

    # define constants
    MAX_R: tl.constexpr = 16

    # define offsets for x
    b_offs = pid*BLOCK_B + tl.arange(0, BLOCK_B)[:, None]
    r_offs = tl.arange(0, MAX_R)[None, :]
    x_offs = b_offs*R + r_offs

    # define offsets for MT
    m_offs = tl.arange(0, MAX_R)[:, None]
    n_offs = tl.arange(0, MAX_R)[None, :]
    MT_offs = n_offs*MAX_R + m_offs # since MT and not M, offsets go down first

    # define masks
    b_mask = b_offs < rows
    r_mask = r_offs < R
    mask = b_mask & r_mask

    # load vectors and matrices
    x_re = tl.load(x_re_ptr + x_offs, mask = mask, other = 0.0)
    x_im = tl.load(x_im_ptr + x_offs, mask = mask, other = 0.0)
    MT_re = tl.load(M_re_ptr + MT_offs)
    MT_im = tl.load(M_im_ptr + MT_offs)

    # calculate cdot product
    y_re, y_im = _cdot(x_re, x_im, MT_re, MT_im)

    # now we define the offsets based on the value of STORE_T
    if STORE_T:
        # calculate new offsets
        outer = b_offs//M
        inner = b_offs % M
        
        # combine to get total offset
        y_offs = outer*M*R + r_offs*M + inner
    else:
        # regular y offset
        y_offs = x_offs
    
    # store matrices
    tl.store(y_re_ptr + y_offs, y_re, mask = mask)
    tl.store(y_im_ptr + y_offs, y_im, mask = mask)


# =============================================================================
# bailey_scale_kernel: elementwise w_N^{n1 kM} multiply with optional fused T2
# =============================================================================

@triton.jit
def bailey_scale_kernel(
    x_re_ptr, x_im_ptr,     # (rows*m0*M,) fp16 input (logical (rows, m0, M))
    y_re_ptr, y_im_ptr,     # (rows*m0*M,) fp16 output ((rows, m0, M) or (rows, M, m0))
    tw_re_ptr, tw_im_ptr,   # (m0, M) fp16
    m0, M,
    BLOCK_M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Elementwise complex multiply by bt[n1, kM] over the (rows, m0, M) view.
    fp32 arithmetic, fp16 result. STORE_T=True fuses with a transpose to
    produce (rows, M, m0).

    Grid: (cdiv(m0, BLOCK_M0), cdiv(M, BLOCK_M), rows).
    """
    # program ids
    m0_pid = tl.program_id(axis = 0)
    M_pid = tl.program_id(axis = 1)
    rows_pid = tl.program_id(axis = 2)

    # define offsets
    m0_offs = m0_pid*BLOCK_M0 + tl.arange(0, BLOCK_M0)[:, None]
    M_offs = M_pid*BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
    x_offs = rows_pid*m0*M + m0_offs*M + M_offs
    tw_offs = m0_offs*M + M_offs

    # define masks
    m0_mask = m0_offs < m0
    M_mask = M_offs < M
    mask = m0_mask & M_mask

    # load vectors
    x_re = tl.load(x_re_ptr + x_offs, mask = mask, other = 0.0)
    x_im = tl.load(x_im_ptr + x_offs, mask = mask, other = 0.0)
    tw_re = tl.load(tw_re_ptr + tw_offs, mask = mask, other = 0.0)
    tw_im = tl.load(tw_im_ptr + tw_offs, mask = mask, other = 0.0)

    # convert to fp32 for the twiddling
    x_re = x_re.to(tl.float32)
    x_im = x_im.to(tl.float32)
    tw_re = tw_re.to(tl.float32)
    tw_im = tw_im.to(tl.float32)

    # compute twiddles
    y_re = x_re*tw_re - x_im*tw_im
    y_im = x_re*tw_im + x_im*tw_re

    # now we have to convert back to fp16 for storing
    y_re = y_re.to(tl.float16)
    y_im = y_im.to(tl.float16)

    # now check for STORE_T
    if STORE_T:
        # essentially, x_offs but transposed
        y_offs = rows_pid*m0*M + M_offs*m0 + m0_offs
    else:
        # if not then just regular x_offs
        y_offs = x_offs

    tl.store(y_re_ptr + y_offs, y_re, mask = mask)
    tl.store(y_im_ptr + y_offs, y_im, mask = mask)


# =============================================================================
# Thin launch wrappers -- GIVEN, do not edit
# =============================================================================

def _transpose(in_re, in_im, out_re, out_im, B, R, C):
    """Logical (B, R, C) -> (B, C, R) transpose, paired re/im."""
    grid = (triton.cdiv(R, TRANSPOSE_BLOCK), triton.cdiv(C, TRANSPOSE_BLOCK), B)
    transpose_kernel[grid](
        in_re, in_im, out_re, out_im, R, C,
        BLOCK_R=TRANSPOSE_BLOCK, BLOCK_C=TRANSPOSE_BLOCK,
    )


def _fft_chunk(in_re, in_im, out_re, out_im, rows, m, plan, M=1, store_t=False):
    """Length-m FFT over `rows` contiguous (rows, m) signals.

    M / store_t control the output layout:
      store_t=False, M=1: natural (rows, m) row-major (F6 leaf path)
      store_t=True,  M>1: transposed (rows//M, m, M) (F7 fused FFT-m0+T3)
    """
    if m == 256:
        f4_plan = plan['f4_plan']
        f4_kernel_L2[(triton.cdiv(rows, F4_L2_BLOCK_B),)](
            in_re.view(rows, 256), in_im.view(rows, 256),
            out_re.view(rows, 256), out_im.view(rows, 256),
            f4_plan['F_re'], f4_plan['F_im'],
            f4_plan['tw_re'], f4_plan['tw_im'],
            rows, M,
            BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=f4_plan['L'], STORE_T=store_t,
            num_warps=4, num_stages=1,
        )
    else:
        M_re, M_im = plan['dft_mats'][m]
        dft_kernel[(triton.cdiv(rows, DFT_BLOCK_B),)](
            in_re.view(rows, m), in_im.view(rows, m),
            out_re.view(rows, m), out_im.view(rows, m),
            M_re, M_im, rows, M,
            R=m, BLOCK_B=DFT_BLOCK_B, STORE_T=store_t,
        )


def _scale(in_re, in_im, out_re, out_im, rows, m0, M, twr, twi, store_t=False):
    """Bailey scale over logical (rows, m0, M)."""
    grid = (triton.cdiv(m0, SCALE_BLOCK), triton.cdiv(M, SCALE_BLOCK), rows)
    bailey_scale_kernel[grid](
        in_re, in_im, out_re, out_im, twr, twi,
        m0, M, BLOCK_M0=SCALE_BLOCK, BLOCK_M=SCALE_BLOCK, STORE_T=store_t,
    )


def _lookup_tw(plan, m0, M, N_i):
    """Find the precomputed Bailey twiddle table for (m0, M, N_i) in plan['tw']."""
    for (a, b, n, tr, ti) in plan['tw']:
        if a == m0 and b == M and n == N_i:
            return tr, ti
    raise KeyError(f"no twiddle table for (m0={m0}, M={M}, N={N_i})")


# =============================================================================
# F3 pipeline: 4-step Bailey six-step (T1 -> F2-A -> T2 -> F2-B)
# =============================================================================

def f3_launch(in_re, in_im, out_re, out_im, mid_re, mid_im, plan, B):
    """Run the 4-step F3 pipeline. Buffer ping-pong: in -> mid -> out -> mid
    -> out. The Bailey twiddle fuses into F2-A (BAILEY_EPILOGUE=True), and
    the would-be T3 is absorbed by F2-B (STRIDED_STORE=True).

    Steps:
      1. T1 (transpose): x[b, n2, n1] -> A[b, n1, n2]
      2. F2-A:           length-N2 FFT over (B*N1) signals with Bailey epilogue
      3. T2 (transpose): Z[b, n1, k2] -> Z'[b, k2, n1]
      4. F2-B:           length-N1 FFT over (B*N2) signals with strided store
    """
    # access dimensions
    N1 = plan['N1']
    N2 = plan['N2']
    N = plan['N']
    
    # access and define variables for f2a
    tw_re_f2a = plan['tw_re_n2']
    tw_im_f2a = plan['tw_im_n2']
    perm_f2a = plan['perm_n2']
    LOG2_N_f2a = plan['LOG2_N2']
    BAILEY_EPILOGUE_f2a = True
    STRIDED_STORE_f2a = False
    OUTER_DIM_f2a = N1
    N_f2a = N2
    
    # access and define variables for f2b
    tw_re_f2b = plan['tw_re_n1']
    tw_im_f2b = plan['tw_im_n1']
    perm_f2b = plan['perm_n1']
    LOG2_N_f2b = plan['LOG2_N1']
    BAILEY_EPILOGUE_f2b = False
    STRIDED_STORE_f2b = True
    OUTER_DIM_f2b = N2
    N_f2b = N1
    
    # define total N and bailey twiddles; both do not change between f2a and f2b
    N_TOTAL = N
    bt_re = plan['bt_re']
    bt_im = plan['bt_im']
    
    # calculate grid sizes
    grid_f2a = (B*N1,)
    grid_f2b = (B*N2,)

    # calculate first transpose (T1)
    _transpose(in_re, in_im, mid_re, mid_im, B, N2, N1)

    # first f2 function call (F2-A)
    f2_kernel[grid_f2a](mid_re, mid_im, out_re, out_im, tw_re_f2a, tw_im_f2a,
                      perm_f2a, bt_re, bt_im, OUTER_DIM = OUTER_DIM_f2a,
                      N_TOTAL = N_TOTAL, N = N_f2a, LOG2_N = LOG2_N_f2a,
                      BAILEY_EPILOGUE = BAILEY_EPILOGUE_f2a,
                      STRIDED_STORE = STRIDED_STORE_f2a)

    # calculate second transpose (T2)
    _transpose(out_re, out_im, mid_re, mid_im, B, N1, N2)

    # second f2 function call (F2-B)
    f2_kernel[grid_f2b](mid_re, mid_im, out_re, out_im, tw_re_f2b, tw_im_f2b,
                      perm_f2b, bt_re, bt_im, OUTER_DIM = OUTER_DIM_f2b,
                      N_TOTAL = N_TOTAL, N = N_f2b, LOG2_N = LOG2_N_f2b,
                      BAILEY_EPILOGUE = BAILEY_EPILOGUE_f2b,
                      STRIDED_STORE = STRIDED_STORE_f2b)


# =============================================================================
# F5 pipeline: 6-step Bailey at N1=N2=256 with F4 as inner FFT
# =============================================================================

def f5_launch(in_re, in_im, b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, plan, B):
    """Run the 6-step F5 pipeline at N = 65536 = 256 * 256.

    Buffer ping-pong: in -> b0 -> b1 -> b0 -> b1 -> b2 -> b0 (final).
    The Bailey twiddle is NOT fused into F4 (F4 stays unmodified), so this is
    6 launches; F7 generalizes the fusion idea recursively.

    Steps:
      1. T1:    x[b, n2, n1] -> A[b, n1, n2]
      2. FFT-A: length-256 FFT along last axis -> Y[b, n1, k2]
      3. Scale: Z[b, n1, k2] = Y[b, n1, k2] * bt[n1, k2]
      4. T2:    Z[b, n1, k2] -> Z'[b, k2, n1]
      5. FFT-B: length-256 FFT along last axis -> V[b, k2, k1]
      6. T3:    V[b, k2, k1] -> X[b, k1, k2]   (final in b0)
    """
    # access dimensions and variables
    N1 = plan['N1']
    N2 = plan['N2']
    bt_re = plan['bt_re']
    bt_im = plan['bt_im']

    # define inputs for FFT-A
    rows_fft_a = B*N1
    m_fft_a = N2

    # define inputs for scale
    rows_scale = B
    m0_scale = N1
    M_scale = N2

    # define inputs for FFT-B
    rows_fft_b = B*N2
    m_fft_b = N1

    # calculate first transpose (T1) and store in b0
    _transpose(in_re, in_im, b0_re, b0_im, B, N2, N1)

    # calculate first fft (FFT-A) and store in b1
    _fft_chunk(b0_re, b0_im, b1_re, b1_im,
              rows = rows_fft_a, m = m_fft_a, plan = plan)

    # calculate scaled matrix and store back in b0
    _scale(b1_re, b1_im, b0_re, b0_im, rows = rows_scale,
            m0 = m0_scale, M = M_scale, twr = bt_re, twi = bt_im)

    # calculate second transpose (T2) and store in b1 again
    _transpose(b0_re, b0_im, b1_re, b1_im, B, N1, N2)

    # calculate second fft (FFT-B) and store in b2
    _fft_chunk(b1_re, b1_im, b2_re, b2_im,
              rows = rows_fft_b, m = m_fft_b, plan = plan)

    # calculate third and final transpose (T3) and store back in b0
    _transpose(b2_re, b2_im, b0_re, b0_im, B, N2, N1)

# =============================================================================
# F6 / F7 recursion
# =============================================================================
# Per level i with chunks = [m_0, m_1, ..., m_{p-1}], M = prod(chunks[1:]):
#   T1 :       (rows, M, m_0) -> (rows, m_0, M)
#   recurse:   length-M FFT over (rows*m_0, M)
#   Scale :    y *= w_{N_i}^{n_1 k_M}            (n_1 = the m_0 digit)
#   T2 :       (rows, m_0, M) -> (rows, M, m_0)
#   FFT-m_0 :  length-m_0 FFT over (rows*M, m_0)
#   T3 :       (rows, M, m_0) -> (rows, m_0, M)   [F6 only; F7 fuses]

def _f6_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Recursive 2-factor Bailey split. Leaf (len(chunks)==1) is one
    _fft_chunk call; non-leaf is the 6-step pipeline above.

    Returns the (re, im) cycler-managed buffers holding the (rows, prod(chunks))
    FFT result.
    """
    # define variables
    m0 = chunks[0]
    m_other = chunks[1:] # split chunks into m0, m1 through m{p-1}
    M = math.prod(m_other)
    N_i = m0*M
    
    # define inputs for recursion
    rows_f6_rec = rows*m0
    chunks_f6_rec = m_other

    # define inputs for FFT-m0
    rows_fft_m0 = rows*M

    # leaf case, just _fft_chunk once
    if (len(chunks) == 1):
        y_re, y_im = cyc.next()
        _fft_chunk(cur_re, cur_im, y_re, y_im, rows = rows, m = m0, plan = plan)
        return y_re, y_im

    # note that we have to access these AFTER we run the leaf case
    tw_re, tw_im = _lookup_tw(plan = plan, m0 = m0, M = M, N_i = N_i)

    # first transpose (T1)
    transpose_1_re, transpose_1_im = cyc.next()
    _transpose(cur_re, cur_im, transpose_1_re, transpose_1_im, rows, M, m0)

    # first FFT, uses recursion and is of length M
    rec_re, rec_im = _f6_rec(transpose_1_re, transpose_1_im, rows = rows_f6_rec,
                              chunks = chunks_f6_rec, plan = plan, cyc = cyc)

    # scale
    scale_re, scale_im = cyc.next()
    _scale(rec_re, rec_im, scale_re, scale_im, rows = rows,
            m0 = m0, M = M, twr = tw_re, twi = tw_im)

    # second transpose (T2)
    transpose_2_re, transpose_2_im = cyc.next()
    _transpose(scale_re, scale_im, transpose_2_re, transpose_2_im, rows, m0, M)

    # second FFT, uses plain _fft_chunk and is of length m_0 (FFT-m_0)
    fftm0_re, fftm0_im = cyc.next()
    _fft_chunk(transpose_2_re, transpose_2_im, fftm0_re, fftm0_im,
                rows = rows_fft_m0, m = m0, plan = plan)

    # final transpose (T3)
    y_re, y_im = cyc.next()
    _transpose(fftm0_re, fftm0_im, y_re, y_im, rows, M, m0)

    return y_re, y_im
    

def _f7_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Same recursion as _f6_rec but with Scale+T2 fused (store_t=True on
    bailey_scale_kernel) and FFT-m_0+T3 fused (store_t=True, M=M on the inner
    FFT kernel). Output should be bitwise-equal to _f6_rec.
    """
    # define variables
    m0 = chunks[0]
    m_other = chunks[1:] # split chunks into m0, m1 through m{p-1}
    M = math.prod(m_other)
    N_i = m0*M
    
    # define inputs for recursion
    rows_f7_rec = rows*m0
    chunks_f7_rec = m_other

    # define inputs for FFT-m0
    rows_fft_m0 = rows*M

    # leaf case, just _fft_chunk once
    if (len(chunks) == 1):
        y_re, y_im = cyc.next()
        _fft_chunk(cur_re, cur_im, y_re, y_im, rows = rows, m = m0, plan = plan)
        return y_re, y_im

    # note that we have to access these AFTER we run the leaf case
    tw_re, tw_im = _lookup_tw(plan = plan, m0 = m0, M = M, N_i = N_i)

    # first transpose (T1)
    transpose_1_re, transpose_1_im = cyc.next()
    _transpose(cur_re, cur_im, transpose_1_re, transpose_1_im, rows, M, m0)

    # first FFT, uses recursion and is of length M
    rec_re, rec_im = _f7_rec(transpose_1_re, transpose_1_im, rows = rows_f7_rec,
                            chunks = chunks_f7_rec, plan = plan, cyc = cyc)

    # scale and second transpose (scale + T2)
    scale_trans_2_re, scale_trans_2_im = cyc.next()
    _scale(rec_re, rec_im, scale_trans_2_re, scale_trans_2_im,
          rows = rows, m0 = m0, M = M, twr = tw_re, twi = tw_im, store_t = True)

    # second FFT and third and final transpose (FFT-m0 + T3)
    y_re, y_im = cyc.next()
    _fft_chunk(scale_trans_2_re, scale_trans_2_im, y_re, y_im,
              rows = rows_fft_m0, m = m0, plan = plan, M = M, store_t = True)

    return y_re, y_im