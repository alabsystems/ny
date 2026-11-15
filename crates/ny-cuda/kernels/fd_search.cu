// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// #wide-fd-search (ENGINE_PROGRAM M4 / §1.9): the frozen-dual tier's LEVEL
// SEARCH on the device. UNTRUSTED: every value written here is a HINT that the
// lane re-scores in CPU f64 with its own certified evaluator (and splits the
// node when that fails). THE DEVICE NEVER CERTIFIES.
//
// REFERENCE. The kernels reproduce ny-bbe's CPU f32 search
// (crates/ny-bbe/src/fd.rs, `fd_level::<f32>`) BIT FOR BIT on every output
// the lane reads: the same operations, on the same operands, in the same
// order, in IEEE binary32 round-to-nearest with no FMA contraction (every
// multiply and add below is an explicit __fmul_rn / __fadd_rn, which the
// compiler never fuses; the build also passes --fmad=false). The sign of a
// zero may differ where the CPU adds `0 * a` to a lane whose multiplier is 0
// and a kernel lane skips the row; no output reads that sign (it only enters
// `|x|`, `x < 0`, `x > 0` or a product that is added to a nonzero term). The
// known-answer tests in ny-cli (`cuda_fd_*`) pin the bit identity against the
// CPU backend itself.
//
// SHAPE (one level chunk, "slab", per launch sequence):
//   fd_level_warm    one CTA per FD_NB nodes: sanitize, WARM value / S / rc /
//                    supergradient, slack clamp, projected supergradient,
//                    the log-bucket line search (two column passes, rc and
//                    the direction recomputed per tile: no n-sized scratch),
//                    the clamped FAR point;
//   fd_compact       one CTA: the far-first work lists in node order
//                    (phase 0: moved far points; phase 1: clamped / floor
//                    points of nodes neither warm nor far scores clearly
//                    positive), deterministic integer scan;
//   fd_level_points  one CTA per FD_NB listed (node, point) items: value / S;
//   fd_level_finish  one thread per node: the children's warm start and the
//                    stored proposals.
//
// DETERMINISM (R6). Every float reduction is a fixed-order sequential sum
// owned by one thread; the only cross-thread reductions are min / max (exact,
// order-free) and integer scans. No atomics. A node's outputs depend on
// neither the grid, the slab cut nor which nodes share its CTA.
//
// LAYOUT. Row-major f32 rows `a[i * n + j]`; per-row pairs packed as
// PAIRS[i * 8 + {0,1: rho, 2,3: c0s, 4,5: b0, 6,7: b1}] ([Off, On]);
// OWN[j] = the split row whose own column is j (or -1). Per-node vectors live
// in VEC[(v * cap + node) * D + i] (v one of the VEC_* below; `cap` is the
// vectors' node stride, which the host sets to the launch's own node count so
// a launch of `count` nodes at depth D touches `FD_NVEC * count * D` floats);
// item rows of the points kernel in LM[item * D + i].

#define FD_NB 8
#define FD_THREADS 256
#define FD_TILE 256
#define FD_BUCKETS 256
#define FD_MROWS 64
#define FD_POINTS 4
#define FD_COMPACT_THREADS 1024

#define PT_WARM 0
#define PT_CLAMPED 1
#define PT_FLOOR 2
#define PT_FAR 3

#define VEC_L0 0
#define VEC_L1 1
#define VEC_WM 2
#define VEC_P 3
#define VEC_C0 4
#define VEC_C1 5
#define VEC_SP0 6
#define VEC_SP1 7
#define VEC_DIR 8
#define VEC_F0 9
#define VEC_F1 10
#define FD_NVEC 11

#define K_TOL 0
#define K_FLOOR_E 1
#define K_LW_MIN 2
#define K_KB 3
#define K_FAR_SCALE 4
#define FD_NK 5

#define FL_ACTIVE 1u
#define FL_CLAMPED_DIFFERS 2u
#define FL_ANY_MULT 4u
#define FL_FAR_DIFFERS 8u
#define FL_SKIP 16u

#define EVALS_REFUSED 0x80000000u

// ---------------------------------------------------------------------------
//  IEEE binary32 arithmetic, never contracted
// ---------------------------------------------------------------------------

__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float fdiv(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float fbits(unsigned int b) { return __uint_as_float(b); }

#define F32_INF fbits(0x7f800000u)
#define F32_MAX fbits(0x7f7fffffu)

// fd.rs `fast_ln_f32`, operation for operation.
__device__ __forceinline__ float fast_ln(float x) {
    if (x == F32_INF) {
        return x;
    }
    const unsigned int bits = __float_as_uint(x);
    int e = (int)((bits >> 23) & 0xffu) - 127;
    float m = __uint_as_float((bits & 0x007fffffu) | 0x3f800000u);
    if (m > fbits(0x3fb504f3u)) {  // SQRT_2
        m = fmul(m, 0.5f);
        e += 1;
    }
    const float s = fdiv(fsub(m, 1.0f), fadd(m, 1.0f));
    const float s2 = fmul(s, s);
    float poly = fadd(fbits(0x3e4ccccdu), fmul(s2, fbits(0x3e124925u)));  // 0.2 + s2 (1/7)
    poly = fadd(fbits(0x3eaaaaabu), fmul(s2, poly));                        // 1/3 + s2 (..)
    poly = fadd(1.0f, fmul(s2, poly));
    const float series = fmul(fmul(2.0f, s), poly);
    return fadd(fmul((float)e, fbits(0x3f317218u)), series);  // e LN_2 + series
}

// fd.rs `pow2_f32` (k in -126..=127).
__device__ __forceinline__ float pow2f(int k) { return __uint_as_float((unsigned int)(k + 127) << 23); }

// fd.rs `fast_exp_f32`, operation for operation.
__device__ __forceinline__ float fast_exp(float x) {
    if (isnan(x)) {
        return x;
    }
    if (x > fbits(0x42b17217u)) {
        return F32_INF;
    }
    if (x < fbits(0xc2aeac4fu)) {
        return 0.0f;
    }
    const float nf = floorf(fadd(fmul(x, fbits(0x3fb8aa3bu)), 0.5f));
    const float r = fsub(fsub(x, fmul(nf, fbits(0x3f318000u))), fmul(nf, fbits(0xb95e8083u)));
    float p = fadd(fbits(0x3c088889u), fmul(r, fbits(0x3ab60b61u)));  // 1/120 + r / 720
    p = fadd(fbits(0x3d2aaaabu), fmul(r, p));                           // 1/24 + r (..)
    p = fadd(fbits(0x3e2aaaabu), fmul(r, p));                           // 1/6 + r (..)
    p = fadd(0.5f, fmul(r, p));
    p = fadd(1.0f, fmul(r, p));
    p = fadd(1.0f, fmul(r, p));
    const int n = (int)nf;
    const int h = n / 2;
    return fmul(fmul(p, pow2f(h)), pow2f(n - h));
}

// fd.rs `sanitize`: finite and > 0, else 0.
__device__ __forceinline__ float sanitize(float v) { return (isfinite(v) && v > 0.0f) ? v : 0.0f; }

// fd.rs `store`: NaN and non-positive -> 0, beyond f32::MAX -> f32::MAX.
__device__ __forceinline__ float store(float x) {
    if (isnan(x) || x <= 0.0f) {
        return 0.0f;
    }
    return x >= F32_MAX ? F32_MAX : x;
}

// fd.rs `search_block`'s `strong`: a clearly positive point.
__device__ __forceinline__ bool strong(float v, float s, float rel) {
    return isfinite(v) && isfinite(s) && v > fmul(rel, s);
}

// ---------------------------------------------------------------------------
//  Shared pieces
// ---------------------------------------------------------------------------

struct Prob {
    const float* A;
    const int* OWN;
    const float* DV;
    const float* PAIRS;
    const float* ROWABS;
    int n;
    float c0;
    float abs_d;
};

__device__ __forceinline__ float* vec_row(float* VEC, int v, int cap, int node, int D) {
    return VEC + ((size_t)v * (size_t)cap + (size_t)node) * (size_t)D;
}

// fd.rs `prologue` for one lane: the row multipliers (written to `mult`) and
// the five rc-independent sums. `m0`/`m1` null = the floor point (all 0).
// Returns false on a non-finite multiplier (the point is refused, `mult`
// zeroed, exactly as the CPU zeroes a refused lane).
__device__ bool prologue(const Prob& P, int D, const unsigned char* sides, const float* m0,
                         const float* m1, float* mult, float out[5]) {
    float c0path = P.c0;
    float c0abs = fabsf(P.c0);
    float lag = 0.0f;
    float lagabs = 0.0f;
    float tsum = 0.0f;
    bool ok = true;
    for (int i = 0; i < D; ++i) {
        const float a = m0 ? fmaxf(m0[i], 0.0f) : 0.0f;
        const float b = m1 ? fmaxf(m1[i], 0.0f) : 0.0f;
        if (!isfinite(a) || !isfinite(b)) {
            ok = false;
            break;
        }
        const int si = sides[i] ? 1 : 0;
        const float sign = si == 0 ? 1.0f : -1.0f;
        const float* pr = P.PAIRS + (size_t)i * 8;
        const float rho = pr[si];
        mult[i] = fadd(rho, fmul(sign, fsub(a, b)));
        const float cs = pr[2 + si];
        c0path = fadd(c0path, cs);
        c0abs = fadd(c0abs, fabsf(cs));
        const float b0 = pr[4 + si];
        const float b1 = pr[6 + si];
        lag = fadd(lag, fadd(fmul(a, b0), fmul(b, b1)));
        lagabs = fadd(lagabs, fadd(fmul(a, fabsf(b0)), fmul(b, fabsf(b1))));
        tsum = fadd(tsum, fmul(fadd(fadd(fabsf(rho), a), b), P.ROWABS[i]));
    }
    if (!ok) {
        for (int i = 0; i < D; ++i) {
            mult[i] = 0.0f;
        }
    }
    out[0] = c0path;
    out[1] = c0abs;
    out[2] = lag;
    out[3] = lagabs;
    out[4] = tsum;
    return ok;
}

// fd.rs `finish`: (value, S) from the sums and sum |rc|.
__device__ __forceinline__ void finish(const Prob& P, const float s[5], float sumabs, float* value,
                                       float* big_s) {
    *value = fsub(fsub(s[0], sumabs), s[2]);
    *big_s = fadd(fadd(fadd(fadd(s[1], sumabs), P.abs_d), s[4]), s[3]);
}

// One column tile of every lane's `rc`: out[q][t] = base_j + sum_i m_q[i] a[i][j]
// for column j = j0 + t (t < tn), rows in ascending order. `rows` is the
// [slot][D] matrix of lane rows and `slot[q]` lane q's row (lanes >= nl are
// idle). `with_base`: start from d with the split rows' own columns zeroed
// (fd.rs `d_base`); else from 0 (the line-search direction). A row whose
// multiplier is 0 in lane q adds nothing to lane q (the CPU adds `0 * a`,
// the same value up to the sign of a zero). Called by EVERY thread of the
// CTA (it synchronises); ends synchronised.
__device__ void rc_tile(const Prob& P, int D, int j0, int tn, int nl, const float* rows,
                        const int* slot, bool with_base, float (*out)[FD_TILE + 1],
                        float (*s_m)[FD_MROWS]) {
    const int t = threadIdx.x;
    float r[FD_NB];
    float base = 0.0f;
    if (with_base && t < tn) {
        const int j = j0 + t;
        const int own = P.OWN[j];
        base = (own >= 0 && own < D) ? 0.0f : P.DV[j];
    }
#pragma unroll
    for (int q = 0; q < FD_NB; ++q) {
        r[q] = base;
    }
    for (int i0 = 0; i0 < D; i0 += FD_MROWS) {
        const int rws = min(FD_MROWS, D - i0);
        __syncthreads();
        for (int k = t; k < FD_NB * FD_MROWS; k += FD_THREADS) {
            const int q = k / FD_MROWS;
            const int ii = k % FD_MROWS;
            s_m[q][ii] =
                (q < nl && ii < rws) ? rows[(size_t)slot[q] * (size_t)D + (size_t)(i0 + ii)] : 0.0f;
        }
        __syncthreads();
        if (t < tn) {
            const float* col = P.A + (size_t)i0 * (size_t)P.n + (size_t)(j0 + t);
            for (int ii = 0; ii < rws; ++ii) {
                const float av = col[(size_t)ii * (size_t)P.n];
#pragma unroll
                for (int q = 0; q < FD_NB; ++q) {
                    const float m = s_m[q][ii];
                    if (m != 0.0f) {
                        r[q] = fadd(r[q], fmul(m, av));
                    }
                }
            }
        }
    }
    if (t < tn) {
#pragma unroll
        for (int q = 0; q < FD_NB; ++q) {
            out[q][t] = r[q];
        }
    }
    __syncthreads();
}

// Lane q's sum |rc| over one tile, folded into its running sum (fd.rs
// `tile_abs` + the per-tile fold): one thread per lane, columns in order.
__device__ __forceinline__ void abs_fold(int nl, int tn, float (*tile)[FD_TILE + 1], float* sumabs) {
    const int t = threadIdx.x;
    if (t < nl) {
        float part = 0.0f;
        for (int u = 0; u < tn; ++u) {
            part = fadd(part, fabsf(tile[t][u]));
        }
        sumabs[t] = fadd(sumabs[t], part);
    }
}

// ---------------------------------------------------------------------------
//  fd_level_warm
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(FD_THREADS)
    fd_level_warm(const float* A, const int* OWN, const float* DV, const float* PAIRS,
                  const float* ROWABS, int n, float c0, float abs_d, int D, int count, int cap,
                  const unsigned char* SIDES, const float* L0IN, const float* L1IN, const float* K,
                  float* VEC, float* VAL, float* ABS, unsigned int* FLAGS, float* SLOPE,
                  float* ETA) {
    __shared__ float s_tile[FD_NB][FD_TILE + 1];
    __shared__ float s_aux[FD_NB][FD_TILE + 1];
    __shared__ float s_m[FD_NB][FD_MROWS];
    __shared__ float s_bucket[FD_NB][FD_BUCKETS];
    __shared__ float s_sum[FD_NB][5];
    __shared__ float s_sumabs[FD_NB];
    __shared__ float s_red[FD_NB][FD_THREADS / 32][2];
    __shared__ float s_emax[FD_NB];
    __shared__ float s_le[FD_NB];
    __shared__ float s_lw[FD_NB];
    __shared__ int s_slot[FD_NB];
    __shared__ int s_ok[FD_NB];
    __shared__ int s_active[FD_NB];

    const int t = threadIdx.x;
    const int first = blockIdx.x * FD_NB;
    if (first >= count) {
        return;  // uniform over the CTA
    }
    const int nb = min(FD_NB, count - first);
    const Prob P = {A, OWN, DV, PAIRS, ROWABS, n, c0, abs_d};
    const float tol = K[K_TOL];
    const float floor_e = K[K_FLOOR_E];
    const float lw_min = K[K_LW_MIN];
    const float kb = K[K_KB];
    const float far_scale = K[K_FAR_SCALE];
    const float nan = __uint_as_float(0x7fc00000u);  // f32::NAN (LevelOut::prepare)

    // 0. load: sanitized warm multipliers, zeroed supergradient, NaN points.
    for (int k = t; k < nb * D; k += FD_THREADS) {
        const int q = k / D;
        const int i = k % D;
        const size_t g = (size_t)(first + q) * (size_t)D + (size_t)i;
        vec_row(VEC, VEC_L0, cap, first + q, D)[i] = sanitize(L0IN[g]);
        vec_row(VEC, VEC_L1, cap, first + q, D)[i] = sanitize(L1IN[g]);
        vec_row(VEC, VEC_P, cap, first + q, D)[i] = 0.0f;
    }
    if (t < FD_NB) {
        s_slot[t] = first + (t < nb ? t : 0);
        s_sumabs[t] = 0.0f;
        s_ok[t] = 0;
        s_active[t] = 0;
    }
    if (t < nb) {
        const int node = first + t;
        for (int p = 0; p < FD_POINTS; ++p) {
            VAL[(size_t)node * FD_POINTS + p] = nan;
            ABS[(size_t)node * FD_POINTS + p] = nan;
        }
        FLAGS[node] = 0u;
        SLOPE[node] = 0.0f;
        ETA[node] = 0.0f;
    }
    __syncthreads();

    // 1. the warm prologue (one thread per node).
    if (t < nb) {
        const int node = first + t;
        float sums[5];
        s_ok[t] = prologue(P, D, SIDES + (size_t)node * (size_t)D,
                           vec_row(VEC, VEC_L0, cap, node, D), vec_row(VEC, VEC_L1, cap, node, D),
                           vec_row(VEC, VEC_WM, cap, node, D), sums)
                      ? 1
                      : 0;
        for (int k = 0; k < 5; ++k) {
            s_sum[t][k] = sums[k];
        }
    }
    __syncthreads();

    // 2. warm tiles: rc, sum |rc|, supergradient partials.
    const float* wm_rows = VEC + (size_t)VEC_WM * (size_t)cap * (size_t)D;
    for (int j0 = 0; j0 < n; j0 += FD_TILE) {
        const int tn = min(FD_TILE, n - j0);
        rc_tile(P, D, j0, tn, nb, wm_rows, s_slot, true, s_tile, s_m);
        abs_fold(nb, tn, s_tile, s_sumabs);
        for (int k = t; k < nb * D; k += FD_THREADS) {
            const int q = k / D;
            const int i = k % D;
            const float* arow = A + (size_t)i * (size_t)n + (size_t)j0;
            float part = 0.0f;
            for (int u = 0; u < tn; ++u) {
                const float av = arow[u];
                part = fadd(part, s_tile[q][u] < 0.0f ? av : -av);
            }
            float* pv = vec_row(VEC, VEC_P, cap, first + q, D);
            pv[i] = fadd(pv[i], part);
        }
    }
    __syncthreads();

    // 3. finish warm; slack clamp, projected supergradient, direction.
    if (t < nb) {
        const int node = first + t;
        bool active = false;
        float value = 0.0f;
        float big_s = 0.0f;
        if (s_ok[t]) {
            finish(P, s_sum[t], s_sumabs[t], &value, &big_s);
            active = isfinite(value) && isfinite(big_s);
        }
        float* dir = vec_row(VEC, VEC_DIR, cap, node, D);
        if (active) {
            VAL[(size_t)node * FD_POINTS + PT_WARM] = value;
            ABS[(size_t)node * FD_POINTS + PT_WARM] = big_s;
            const unsigned char* sides = SIDES + (size_t)node * (size_t)D;
            const float* l0 = vec_row(VEC, VEC_L0, cap, node, D);
            const float* l1 = vec_row(VEC, VEC_L1, cap, node, D);
            const float* pv = vec_row(VEC, VEC_P, cap, node, D);
            float* c0v = vec_row(VEC, VEC_C0, cap, node, D);
            float* c1v = vec_row(VEC, VEC_C1, cap, node, D);
            float* sp0 = vec_row(VEC, VEC_SP0, cap, node, D);
            float* sp1 = vec_row(VEC, VEC_SP1, cap, node, D);
            bool clamped_differs = false;
            bool any_mult = false;
            float slope0 = 0.0f;
            for (int i = 0; i < D; ++i) {
                const int si = sides[i] ? 1 : 0;
                const float sign = si == 0 ? 1.0f : -1.0f;
                const float* pr = PAIRS + (size_t)i * 8;
                const float s0 = fsub(fmul(sign, pv[i]), pr[4 + si]);
                const float s1 = fsub(fmul(-sign, pv[i]), pr[6 + si]);
                const float a = s0 < 0.0f ? 0.0f : l0[i];
                const float c = s1 < 0.0f ? 0.0f : l1[i];
                clamped_differs = clamped_differs || a != l0[i] || c != l1[i];
                any_mult = any_mult || l0[i] != 0.0f || l1[i] != 0.0f;
                c0v[i] = a;
                c1v[i] = c;
                const float p0 = (l0[i] <= tol && s0 < 0.0f) ? 0.0f : s0;
                const float p1 = (l1[i] <= tol && s1 < 0.0f) ? 0.0f : s1;
                sp0[i] = p0;
                sp1[i] = p1;
                slope0 = fadd(slope0, fadd(fmul(p0, s0), fmul(p1, s1)));
                dir[i] = fmul(sign, fsub(p0, p1));
            }
            FLAGS[node] = FL_ACTIVE | (clamped_differs ? FL_CLAMPED_DIFFERS : 0u) |
                          (any_mult ? FL_ANY_MULT : 0u);
            SLOPE[node] = slope0;
        } else {
            for (int i = 0; i < D; ++i) {
                dir[i] = 0.0f;
            }
        }
        s_active[t] = active ? 1 : 0;
    }
    __syncthreads();

    // 4. line search, pass 1: the breakpoints' range per node (min / max are
    //    exact, so the reduction order is irrelevant).
    const float* dir_rows = VEC + (size_t)VEC_DIR * (size_t)cap * (size_t)D;
    float mn[FD_NB];
    float mx[FD_NB];
#pragma unroll
    for (int q = 0; q < FD_NB; ++q) {
        mn[q] = F32_INF;
        mx[q] = -F32_INF;
    }
    for (int j0 = 0; j0 < n; j0 += FD_TILE) {
        const int tn = min(FD_TILE, n - j0);
        rc_tile(P, D, j0, tn, nb, wm_rows, s_slot, true, s_tile, s_m);
        rc_tile(P, D, j0, tn, nb, dir_rows, s_slot, false, s_aux, s_m);
        if (t < tn) {
#pragma unroll
            for (int q = 0; q < FD_NB; ++q) {
                if (q < nb && s_active[q]) {
                    const float r = s_tile[q][t];
                    const float v = s_aux[q][t];
                    if ((v > 0.0f && r < 0.0f) || (v < 0.0f && r > 0.0f)) {
                        const float e = fdiv(-r, v);
                        mn[q] = fminf(mn[q], e);
                        mx[q] = fmaxf(mx[q], e);
                    }
                }
            }
        }
    }
    const int warp = t / 32;
    const int lane = t % 32;
#pragma unroll
    for (int q = 0; q < FD_NB; ++q) {
        float a = mn[q];
        float b = mx[q];
        for (int off = 16; off > 0; off >>= 1) {
            a = fminf(a, __shfl_xor_sync(0xffffffffu, a, off));
            b = fmaxf(b, __shfl_xor_sync(0xffffffffu, b, off));
        }
        if (lane == 0) {
            s_red[q][warp][0] = a;
            s_red[q][warp][1] = b;
        }
    }
    for (int k = t; k < FD_NB * FD_BUCKETS; k += FD_THREADS) {
        s_bucket[k / FD_BUCKETS][k % FD_BUCKETS] = 0.0f;
    }
    __syncthreads();
    if (t < nb) {
        float a = F32_INF;
        float b = -F32_INF;
        for (int w = 0; w < FD_THREADS / 32; ++w) {
            a = fminf(a, s_red[t][w][0]);
            b = fmaxf(b, s_red[t][w][1]);
        }
        const float emin = fmaxf(a, floor_e);
        const float emax = fmaxf(b, floor_e);
        const float le = fast_ln(emin);
        s_emax[t] = emax;
        s_le[t] = le;
        s_lw[t] = fdiv(fmaxf(fsub(fast_ln(emax), le), lw_min), kb);
    }
    __syncthreads();

    // 5. line search, pass 2: the weighted histogram of the breakpoints' log
    //    positions. Warp q owns node q and adds each column's weight in column
    //    order: lanes of one 32-column chunk that hit the same bucket are
    //    folded by the lowest such lane in ascending lane (= column) order, so
    //    every bucket is the CPU's sequential sum bit for bit.
    for (int j0 = 0; j0 < n; j0 += FD_TILE) {
        const int tn = min(FD_TILE, n - j0);
        rc_tile(P, D, j0, tn, nb, wm_rows, s_slot, true, s_tile, s_m);
        rc_tile(P, D, j0, tn, nb, dir_rows, s_slot, false, s_aux, s_m);
        if (t < tn) {
#pragma unroll
            for (int q = 0; q < FD_NB; ++q) {
                const float r = s_tile[q][t];
                const float v = s_aux[q][t];
                float w = 0.0f;
                if (q < nb && s_active[q] && ((v > 0.0f && r < 0.0f) || (v < 0.0f && r > 0.0f))) {
                    s_tile[q][t] = fdiv(-r, v);
                    w = fmul(2.0f, fabsf(v));
                }
                s_aux[q][t] = w;
            }
        }
        __syncthreads();
        if (warp < nb && s_active[warp]) {
            const int q = warp;
            for (int c = 0; c < tn; c += 32) {
                const int u = c + lane;
                int idx = -1;
                if (u < tn) {
                    const float w = s_aux[q][u];
                    if (w > 0.0f) {
                        const float raw =
                            fdiv(fsub(fast_ln(fmaxf(s_tile[q][u], floor_e)), s_le[q]), s_lw[q]);
                        const float f = floorf(raw);
                        idx = f > 0.0f ? (f >= (float)(FD_BUCKETS - 1) ? FD_BUCKETS - 1 : (int)f)
                                       : 0;
                    }
                }
                const unsigned int peers = __match_any_sync(0xffffffffu, idx);
                if (idx >= 0 && lane == __ffs(peers) - 1) {
                    float acc = s_bucket[q][idx];
                    for (unsigned int m = peers; m != 0u; m &= m - 1u) {
                        acc = fadd(acc, s_aux[q][c + __ffs(m) - 1]);
                    }
                    s_bucket[q][idx] = acc;
                }
                __syncwarp();
            }
        }
    }
    __syncthreads();

    // 6. the step: first bucket whose cumulative weight reaches the slope,
    //    then the clamped far probe.
    if (t < nb && s_active[t]) {
        const int node = first + t;
        const float slope0 = SLOPE[node];
        float cum = 0.0f;
        int first_k = -1;
        for (int k = 0; k < FD_BUCKETS; ++k) {
            cum = fadd(cum, s_bucket[t][k]);
            if (cum >= slope0) {
                first_k = k;
                break;
            }
        }
        float eta = first_k >= 0 ? fast_exp(fadd(s_le[t], fmul((float)first_k, s_lw[t]))) : s_emax[t];
        if (!isfinite(eta)) {
            eta = 0.0f;
        }
        ETA[node] = eta;
        const float* l0 = vec_row(VEC, VEC_L0, cap, node, D);
        const float* l1 = vec_row(VEC, VEC_L1, cap, node, D);
        const float* sp0 = vec_row(VEC, VEC_SP0, cap, node, D);
        const float* sp1 = vec_row(VEC, VEC_SP1, cap, node, D);
        float ebnd = F32_INF;
        for (int i = 0; i < D; ++i) {
            if (sp0[i] < -tol) {
                ebnd = fminf(ebnd, fdiv(-l0[i], sp0[i]));
            }
            if (sp1[i] < -tol) {
                ebnd = fminf(ebnd, fdiv(-l1[i], sp1[i]));
            }
        }
        const float far = isfinite(ebnd) ? ebnd : fmul(s_emax[t], far_scale);
        float eta_c = first_k >= 0 ? eta : far;
        if (!(slope0 > tol && isfinite(eta_c))) {
            eta_c = 0.0f;
        }
        eta_c = fmaxf(eta_c, 0.0f);
        float* f0 = vec_row(VEC, VEC_F0, cap, node, D);
        float* f1 = vec_row(VEC, VEC_F1, cap, node, D);
        bool far_differs = false;
        for (int i = 0; i < D; ++i) {
            const float a = fmaxf(fadd(l0[i], fmul(eta_c, sp0[i])), 0.0f);
            const float c = fmaxf(fadd(l1[i], fmul(eta_c, sp1[i])), 0.0f);
            far_differs = far_differs || a != l0[i] || c != l1[i];
            f0[i] = a;
            f1[i] = c;
        }
        if (far_differs) {
            FLAGS[node] |= FL_FAR_DIFFERS;
        }
    }
}

// ---------------------------------------------------------------------------
//  fd_compact
// ---------------------------------------------------------------------------

// Node `node`'s items of `phase`, applying the phase's copies of the warm
// value (idempotent, so the scan's two passes may both apply them). Writes
// the items at `dst` when non-null. Returns the item count.
__device__ unsigned int phase_items(int phase, int node, float rel, unsigned int* FLAGS,
                                    float* VAL, float* ABS, unsigned int* dst) {
    unsigned int flags = FLAGS[node];
    if (!(flags & FL_ACTIVE)) {
        return 0u;
    }
    const size_t at = (size_t)node * FD_POINTS;
    const unsigned int id = (unsigned int)node << 2;
    if (phase == 0) {
        if (flags & FL_FAR_DIFFERS) {
            if (dst) {
                dst[0] = id | PT_FAR;
            }
            return 1u;
        }
        VAL[at + PT_FAR] = VAL[at + PT_WARM];
        ABS[at + PT_FAR] = ABS[at + PT_WARM];
        return 0u;
    }
    const float vw = VAL[at + PT_WARM];
    const float sw = ABS[at + PT_WARM];
    const float vf = VAL[at + PT_FAR];
    const float sf = ABS[at + PT_FAR];
    if (!(flags & FL_CLAMPED_DIFFERS)) {
        VAL[at + PT_CLAMPED] = vw;
        ABS[at + PT_CLAMPED] = sw;
    }
    if (!(flags & FL_ANY_MULT)) {
        VAL[at + PT_FLOOR] = vw;
        ABS[at + PT_FLOOR] = sw;
    }
    if (strong(vw, sw, rel) || strong(vf, sf, rel)) {
        FLAGS[node] = flags | FL_SKIP;
        return 0u;
    }
    unsigned int k = 0u;
    if (flags & FL_CLAMPED_DIFFERS) {
        if (dst) {
            dst[k] = id | PT_CLAMPED;
        }
        k += 1u;
    }
    if (flags & FL_ANY_MULT) {
        if (dst) {
            dst[k] = id | PT_FLOOR;
        }
        k += 1u;
    }
    return k;
}

extern "C" __global__ void __launch_bounds__(FD_COMPACT_THREADS)
    fd_compact(int phase, int count, float rel, unsigned int* FLAGS, float* VAL, float* ABS,
               unsigned int* ITEMS, unsigned int* ITEM_COUNT) {
    __shared__ unsigned int s_cnt[FD_COMPACT_THREADS];
    const int t = threadIdx.x;
    const int per = (count + FD_COMPACT_THREADS - 1) / FD_COMPACT_THREADS;
    const int lo = min(count, t * per);
    const int hi = min(count, lo + per);
    unsigned int mine = 0u;
    for (int node = lo; node < hi; ++node) {
        mine += phase_items(phase, node, rel, FLAGS, VAL, ABS, nullptr);
    }
    s_cnt[t] = mine;
    __syncthreads();
    for (int off = 1; off < FD_COMPACT_THREADS; off <<= 1) {
        const unsigned int v = t >= off ? s_cnt[t - off] : 0u;
        __syncthreads();
        s_cnt[t] += v;
        __syncthreads();
    }
    unsigned int at = s_cnt[t] - mine;
    for (int node = lo; node < hi; ++node) {
        at += phase_items(phase, node, rel, FLAGS, VAL, ABS, ITEMS + at);
    }
    if (t == FD_COMPACT_THREADS - 1) {
        ITEM_COUNT[0] = s_cnt[t];
    }
}

// ---------------------------------------------------------------------------
//  fd_level_points
// ---------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(FD_THREADS)
    fd_level_points(const float* A, const int* OWN, const float* DV, const float* PAIRS,
                    const float* ROWABS, int n, float c0, float abs_d, int D, int cap,
                    const unsigned char* SIDES, float* VEC, float* LM, float* VAL, float* ABS,
                    const unsigned int* ITEMS, const unsigned int* ITEM_COUNT) {
    __shared__ float s_tile[FD_NB][FD_TILE + 1];
    __shared__ float s_m[FD_NB][FD_MROWS];
    __shared__ float s_sum[FD_NB][5];
    __shared__ float s_sumabs[FD_NB];
    __shared__ int s_slot[FD_NB];
    __shared__ int s_node[FD_NB];
    __shared__ int s_point[FD_NB];
    __shared__ int s_ok[FD_NB];

    const int t = threadIdx.x;
    const int base = blockIdx.x * FD_NB;
    const int total = (int)ITEM_COUNT[0];
    if (base >= total) {
        return;  // uniform over the CTA
    }
    const int nl = min(FD_NB, total - base);
    const Prob P = {A, OWN, DV, PAIRS, ROWABS, n, c0, abs_d};
    if (t < FD_NB) {
        s_slot[t] = base + (t < nl ? t : 0);
        s_sumabs[t] = 0.0f;
        s_ok[t] = 0;
    }
    if (t < nl) {
        const unsigned int item = ITEMS[base + t];
        const int node = (int)(item >> 2);
        const int point = (int)(item & 3u);
        s_node[t] = node;
        s_point[t] = point;
        const float* m0 = nullptr;
        const float* m1 = nullptr;
        if (point == PT_FAR) {
            m0 = vec_row(VEC, VEC_F0, cap, node, D);
            m1 = vec_row(VEC, VEC_F1, cap, node, D);
        } else if (point == PT_CLAMPED) {
            m0 = vec_row(VEC, VEC_C0, cap, node, D);
            m1 = vec_row(VEC, VEC_C1, cap, node, D);
        }
        float sums[5];
        s_ok[t] = prologue(P, D, SIDES + (size_t)node * (size_t)D, m0, m1,
                           LM + (size_t)(base + t) * (size_t)D, sums)
                      ? 1
                      : 0;
        for (int k = 0; k < 5; ++k) {
            s_sum[t][k] = sums[k];
        }
    }
    __syncthreads();
    for (int j0 = 0; j0 < n; j0 += FD_TILE) {
        const int tn = min(FD_TILE, n - j0);
        rc_tile(P, D, j0, tn, nl, LM, s_slot, true, s_tile, s_m);
        abs_fold(nl, tn, s_tile, s_sumabs);
    }
    __syncthreads();
    if (t < nl && s_ok[t]) {
        float value;
        float big_s;
        finish(P, s_sum[t], s_sumabs[t], &value, &big_s);
        if (isfinite(value) && isfinite(big_s)) {
            const size_t at = (size_t)s_node[t] * FD_POINTS + (size_t)s_point[t];
            VAL[at] = value;
            ABS[at] = big_s;
        }
    }
}

// ---------------------------------------------------------------------------
//  fd_level_finish
// ---------------------------------------------------------------------------

extern "C" __global__ void fd_level_finish(int D, int count, int cap, const float* K,
                                           const float* VEC, const float* VAL,
                                           const unsigned int* FLAGS, const float* SLOPE,
                                           const float* ETA, float* OUT_CLAMPED, float* OUT_FAR,
                                           float* OUT_NEXT, unsigned int* EVALS) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) {
        return;
    }
    const float tol = K[K_TOL];
    const unsigned int flags = FLAGS[node];
    const size_t mo = (size_t)node * 2u * (size_t)D;
    float* VECW = (float*)VEC;
    const float* l0 = vec_row(VECW, VEC_L0, cap, node, D);
    const float* l1 = vec_row(VECW, VEC_L1, cap, node, D);
    if (!(flags & FL_ACTIVE)) {
        // Refused: no value; every proposal is the parent's point.
        for (int i = 0; i < D; ++i) {
            const float a = store(l0[i]);
            const float c = store(l1[i]);
            OUT_CLAMPED[mo + i] = a;
            OUT_FAR[mo + i] = a;
            OUT_NEXT[mo + i] = a;
            OUT_CLAMPED[mo + D + i] = c;
            OUT_FAR[mo + D + i] = c;
            OUT_NEXT[mo + D + i] = c;
        }
        EVALS[node] = EVALS_REFUSED;
        return;
    }
    const float* v = VAL + (size_t)node * FD_POINTS;
    float pre_best = v[PT_WARM];
    if (isfinite(v[PT_CLAMPED])) {
        pre_best = fmaxf(pre_best, v[PT_CLAMPED]);
    }
    if (isfinite(v[PT_FLOOR])) {
        pre_best = fmaxf(pre_best, v[PT_FLOOR]);
    }
    const bool pending = pre_best <= tol;
    const float slope0 = SLOPE[node];
    const float eta_s = (pending && slope0 > tol) ? fmaxf(ETA[node], 0.0f) : 0.0f;
    const float* c0v = vec_row(VECW, VEC_C0, cap, node, D);
    const float* c1v = vec_row(VECW, VEC_C1, cap, node, D);
    const float* f0 = vec_row(VECW, VEC_F0, cap, node, D);
    const float* f1 = vec_row(VECW, VEC_F1, cap, node, D);
    const float* sp0 = vec_row(VECW, VEC_SP0, cap, node, D);
    const float* sp1 = vec_row(VECW, VEC_SP1, cap, node, D);
    for (int i = 0; i < D; ++i) {
        OUT_CLAMPED[mo + i] = store(c0v[i]);
        OUT_CLAMPED[mo + D + i] = store(c1v[i]);
        OUT_FAR[mo + i] = store(f0[i]);
        OUT_FAR[mo + D + i] = store(f1[i]);
        OUT_NEXT[mo + i] = store(fadd(l0[i], fmul(eta_s, sp0[i])));
        OUT_NEXT[mo + D + i] = store(fadd(l1[i], fmul(eta_s, sp1[i])));
    }
    unsigned int evals = 1u + ((flags & FL_FAR_DIFFERS) ? 1u : 0u);
    if (!(flags & FL_SKIP)) {
        evals += ((flags & FL_CLAMPED_DIFFERS) ? 1u : 0u) + ((flags & FL_ANY_MULT) ? 1u : 0u);
    }
    EVALS[node] = evals;
}

// ---------------------------------------------------------------------------
//  Known-answer helpers (the backend's init self-test)
// ---------------------------------------------------------------------------

// out[k] = fast_exp(x[k]) and out[count + k] = fast_ln(x[k]) (the host
// compares them bit for bit with its own copies).
extern "C" __global__ void fd_math_kat(const float* x, int count, float* out) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count) {
        out[k] = fast_exp(x[k]);
        out[count + k] = fast_ln(x[k]);
    }
}
