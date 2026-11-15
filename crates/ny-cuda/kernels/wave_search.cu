// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// #wave-beta-bab (ENGINE_PROGRAM M6 / §1.9): the margin-row WAVE search on the
// device (`margin_row.tree_mode: wave_cuda`). UNTRUSTED: every value written
// here is a HINT the lane certifies with its own Outward evaluator (hints
// clamped at ingest) or refutes and splits. THE DEVICE NEVER CERTIFIES.
//
// REFERENCE. The kernels reproduce ny-bbe's CPU f32 wave search
// (crates/ny-bbe/src/wave.rs, `backward_batch::<f32>` / `branch_scores::<f32>`)
// BIT FOR BIT on every output the lane reads: the same operations, on the same
// operands, in the same order, in IEEE binary32 (the passes) and binary64 (the
// Adam moments, the BaBSR areas) round-to-nearest, never contracted (every
// multiply, add, divide and square root below is an explicit __f*_rn /
// __d*_rn intrinsic; the build also passes --fmad=false). The host lowering
// (crates/ny-cuda/src/search/wave_plan.rs) reproduces the reference's f64 ->
// f32 roundings, chords and warm slopes value for value. The sign of a zero
// and a NaN's payload are not outputs (a zero only enters `== 0`, `>= 0`,
// `|x|` or a sum with a term of its own; no output reads a payload).
//
// ORDER. The reference's backward walk SCATTERS each op's output coefficients
// into its input in (output index, tap) order, and sums every bias term and
// the concretisation over the input box as ONE sequential chain per domain.
// The kernels GATHER instead — one thread per input element, adding its
// contributions in exactly the scatter's order (per conv input element, its
// (output position, tap) list sorted as the scatter meets it; per dense input,
// the outputs in order) — and run each chain on one thread of a dedicated CTA
// whose other threads compute the chain's terms in parallel. Every float
// reduction is therefore a fixed-order sequential sum owned by one thread; no
// atomics. A domain's outputs depend on neither the grid, the slab it shares
// nor which domains share it (R6).
//
// SHAPE (one slab = up to `nd` domains; per iteration of the ascent):
//   wave_init        phase 0: per (domain, universe position) the warm slope
//                    (ingest), best copy, zero gradient / moments / pin, and
//                    the per-domain state; phase 1: per literal the warm
//                    multiplier (ingest), best copy, zero gradient / moments,
//                    and the domain's pin; phase 2: per domain the objective
//                    seed (`+1` target, `-1` class) and zero bias sums;
//   wave_backward    one launch per op, REVERSE order: input coefficients
//                    (gather), plus the op's bias chain on the last CTA;
//   wave_concretize  the bound and its absolute-term sum over the input box
//                    (one chain), x* for the forward walk, and the domain's
//                    keep-best / early-stop / next-step state;
//   wave_forward     one launch per op, in order, for the domains that step:
//                    the linearised walk at x*, harvesting the slope and
//                    multiplier gradients;
//   wave_adam        per (domain, position) and per literal: the keep-best
//                    copy and one projected f64 Adam ascent step;
//   wave_babsr       (branch_scores) one thread per domain: the BaBSR
//                    intercept top-k of its unsplit unstable positions.
//
// LAYOUT. Per domain `d`: a tensor block of `T` floats (tensor 0 is the input)
// in COEF (backward coefficients) and VAL (forward values) at d * T; slopes,
// their best copy, gradient, Adam moments and pins at d * stride + p;
// literals (every domain's, concatenated) at g, with LIT_DOM[g] their domain;
// PIN = 0 for a free position, else 1 + the literal pinning it. Op
// descriptors are OP_FIELDS ints (the F_* fields); parameters live in FP / IP
// at the descriptors' offsets (wave_plan.rs documents each table).

#define WAVE_THREADS 256
#define WAVE_BABSR_THREADS 128
#define WAVE_MAX_K 16
#define OP_FIELDS 16

#define OP_CONV 0
#define OP_AFFINE 1
#define OP_RELU 2
#define OP_ADD 3
#define OP_COPY 4
#define OP_GEMM 5

#define F_KIND 0
#define F_OUT 1
#define F_NOUT 2
#define F_IN 3
#define F_NIN 4
#define F_RHS 5
#define F_A 6
#define F_B 7
#define F_C 8
#define F_W 9
#define F_BIAS 10
#define F_G 11
#define F_INV_START 12
#define F_INV_ENT 13
#define F_WT 14
#define F_SITE 15

#define ST_ACTIVE 1u
#define ST_GRAD 2u
#define ST_IMPROVED 4u
#define ST_EARLY 8u
#define ST_HAS_BEST 16u

#define DOM_F 6
#define DF_BIAS 0
#define DF_BIAS_ABS 1
#define DF_LB 2
#define DF_SCALE 3
#define DF_BEST_LB 4
#define DF_BEST_SCALE 5
#define DOM_U 2
#define DU_STATE 0
#define DU_EVALS 1

#define PHASE_DOMAINS 0
#define PHASE_LITERALS 1
#define PHASE_SEED 2

#define WAVE_NONE 0xffffffffu

// ---------------------------------------------------------------------------
//  The launch ABI (wave_plan.rs `WaveArgs` / `AdamConsts`, field for field)
// ---------------------------------------------------------------------------

struct WaveArgs {
    const int* ops;
    const float* fp;
    const unsigned int* ip;
    const unsigned int* classes;
    float* coef;
    float* val;
    float* alpha;
    float* best_alpha;
    float* g_alpha;
    double* ma;
    double* va;
    unsigned int* pin;
    const unsigned int* lit_pos;
    const unsigned int* lit_act;
    const unsigned int* lit_dom;
    float* beta;
    float* best_beta;
    float* g_beta;
    double* mb;
    double* vb;
    const unsigned int* rows;
    float* dom_f;
    unsigned int* dom_u;
    unsigned int* bpos;
    float* bscore;
    long long t;
    int nops;
    int n_in;
    int out_off;
    int target;
    int stride;
    int nd;
    int nl;
    int has_alpha;
    int s_l;
    int s_u;
    int s_slope;
    int s_icpt;
    int s_adapt;
    int s_pos;
    int alpha0;
    int mid;
    int rad;
    int uni_flat;
    int uni_lam;
    int k;
};
static_assert(sizeof(WaveArgs) == 288, "WaveArgs must match wave_plan.rs WAVE_ARGS_BYTES");

struct AdamConsts {
    double b1;
    double omb1;
    double b2;
    double omb2;
    double eps;
    double c1;
    double c2;
    double lr_a;
    double lr_b;
};
static_assert(sizeof(AdamConsts) == 72, "AdamConsts must match wave_plan.rs ADAM_CONSTS_BYTES");

// ---------------------------------------------------------------------------
//  IEEE arithmetic, never contracted
// ---------------------------------------------------------------------------

__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ double rn_dadd(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ double rn_dsub(double a, double b) { return __dsub_rn(a, b); }
__device__ __forceinline__ double rn_dmul(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ double rn_ddiv(double a, double b) { return __ddiv_rn(a, b); }

#define F32_MAX __uint_as_float(0x7f7fffffu)
#define F32_NAN __uint_as_float(0x7fc00000u)

// wave.rs `warm_alpha`: a finite slope clamped into [0, 1] (f64 clamp of an
// f32 is exact, so f32 comparisons give the same value), else `fall`.
__device__ __forceinline__ float warm_alpha(float a, float fall) {
    if (!isfinite(a)) {
        return fall;
    }
    if (a < 0.0f) {
        return 0.0f;
    }
    if (a > 1.0f) {
        return 1.0f;
    }
    return a;
}

// wave.rs `warm_beta`: finite and positive, else 0.
__device__ __forceinline__ float warm_beta(float b) { return (isfinite(b) && b > 0.0f) ? b : 0.0f; }

__device__ __forceinline__ unsigned int state_of(const WaveArgs& A, size_t d) {
    return A.dom_u[d * DOM_U + DU_STATE];
}

// The pin of universe position `p` of domain `d`: 0 free, else 1 + its literal.
__device__ __forceinline__ unsigned int pin_of(const WaveArgs& A, size_t d, unsigned int p) {
    return p == WAVE_NONE ? 0u : A.pin[d * (size_t)A.stride + p];
}

// ---------------------------------------------------------------------------
//  wave_init
// ---------------------------------------------------------------------------

extern "C" __global__ void wave_init(const WaveArgs A, const int phase) {
    const size_t e = (size_t)blockIdx.x * WAVE_THREADS + threadIdx.x;
    const size_t S = (size_t)A.stride;
    const size_t ND = (size_t)A.nd;
    if (phase == PHASE_DOMAINS) {
        const size_t na = ND * S;
        if (e < na) {
            const size_t p = e % S;
            const float fall = A.fp[(size_t)A.alpha0 + p];
            const float a = A.has_alpha ? warm_alpha(A.alpha[e], fall) : fall;
            A.alpha[e] = a;
            A.best_alpha[e] = a;
            A.g_alpha[e] = 0.0f;
            A.ma[e] = 0.0;
            A.va[e] = 0.0;
            A.pin[e] = 0u;
        } else if (e < na + ND) {
            const size_t d = e - na;
            A.dom_u[d * DOM_U + DU_STATE] = ST_ACTIVE;
            A.dom_u[d * DOM_U + DU_EVALS] = 0u;
            for (int f = 0; f < DOM_F; ++f) {
                A.dom_f[d * DOM_F + f] = F32_NAN;
            }
        }
    } else if (phase == PHASE_LITERALS) {
        if (e < (size_t)A.nl) {
            const float b = warm_beta(A.beta[e]);
            A.beta[e] = b;
            A.best_beta[e] = b;
            A.g_beta[e] = 0.0f;
            A.mb[e] = 0.0;
            A.vb[e] = 0.0;
            A.pin[(size_t)A.lit_dom[e] * S + A.lit_pos[e]] = (unsigned int)e + 1u;
        }
    } else if (phase == PHASE_SEED) {
        if (e < ND && (state_of(A, e) & ST_ACTIVE)) {
            float* c = A.coef + e * (size_t)A.t + A.out_off;
            const unsigned int cls = A.classes[A.rows[e]];
            c[A.target] = fadd(c[A.target], 1.0f);
            c[cls] = fsub(c[cls], 1.0f);
            A.dom_f[e * DOM_F + DF_BIAS] = 0.0f;
            A.dom_f[e * DOM_F + DF_BIAS_ABS] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
//  wave_backward
// ---------------------------------------------------------------------------

// The ReLU backward coefficient of neuron j (wave.rs `pass`, the Relu arm):
// `*chord` is set when the upper-chord branch (with its bias term) was taken.
__device__ __forceinline__ float relu_back(const WaveArgs& A, size_t d, int sb, size_t j, float v,
                                           bool* chord, float* icpt) {
    const size_t f = (size_t)sb + j;
    const float l = A.fp[(size_t)A.s_l + f];
    const float u = A.fp[(size_t)A.s_u + f];
    const unsigned int p = A.ip[(size_t)A.s_pos + f];
    const unsigned int code = pin_of(A, d, p);
    const int pj = code == 0u ? 0 : (A.lit_act[code - 1u] ? 1 : -1);
    float c = 0.0f;
    *chord = false;
    if (pj > 0 || (pj == 0 && l >= 0.0f)) {
        c = v;
    } else if (pj < 0 || u <= 0.0f) {
        // Inactive: the output is exactly zero.
    } else if (v >= 0.0f) {
        const float a = p == WAVE_NONE ? A.fp[(size_t)A.s_adapt + f] : A.alpha[d * (size_t)A.stride + p];
        c = fmul(a, v);
    } else {
        c = fmul(A.fp[(size_t)A.s_slope + f], v);
        *chord = true;
        *icpt = A.fp[(size_t)A.s_icpt + f];
    }
    if (code != 0u) {
        // The Lagrangian term `- beta_k s_k z_k`.
        const float b = A.beta[code - 1u];
        c = A.lit_act[code - 1u] ? fsub(c, b) : fadd(c, b);
    }
    return c;
}

// The op's bias chain over its output coefficients, on the whole CTA: every
// thread forms one term, thread 0 adds them in index order (wave.rs `bias +=
// t; bias_abs += t.abs()` for every nonzero-coefficient term of the op).
__device__ void bias_chain(const WaveArgs& A, const int* O, size_t d, const float* C) {
    __shared__ float term[WAVE_THREADS];
    __shared__ unsigned char used[WAVE_THREADS];
    const int kind = O[F_KIND];
    if (kind == OP_ADD || kind == OP_COPY) {
        return;
    }
    const int n = O[F_NOUT];
    const float* co = C + O[F_OUT];
    float bias = 0.0f;
    float babs = 0.0f;
    if (threadIdx.x == 0) {
        bias = A.dom_f[d * DOM_F + DF_BIAS];
        babs = A.dom_f[d * DOM_F + DF_BIAS_ABS];
    }
    for (int base = 0; base < n; base += WAVE_THREADS) {
        const int i = base + (int)threadIdx.x;
        float t = 0.0f;
        unsigned char use = 0;
        if (i < n) {
            const float v = co[i];
            if (kind == OP_CONV) {
                if (v != 0.0f) {
                    t = fmul(A.fp[(size_t)O[F_BIAS] + (size_t)(i / O[F_C])], v);
                    use = 1;
                }
            } else if (kind == OP_AFFINE || kind == OP_GEMM) {
                if (v != 0.0f) {
                    t = fmul(A.fp[(size_t)O[F_BIAS] + (size_t)i], v);
                    use = 1;
                }
            } else if (kind == OP_RELU) {
                bool chord = false;
                float icpt = 0.0f;
                (void)relu_back(A, d, O[F_SITE], (size_t)i, v, &chord, &icpt);
                if (chord) {
                    t = fmul(icpt, v);
                    use = 1;
                }
            }
        }
        term[threadIdx.x] = t;
        used[threadIdx.x] = use;
        __syncthreads();
        if (threadIdx.x == 0) {
            const int lim = min(WAVE_THREADS, n - base);
            for (int q = 0; q < lim; ++q) {
                if (used[q]) {
                    bias = fadd(bias, term[q]);
                    babs = fadd(babs, fabsf(term[q]));
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        A.dom_f[d * DOM_F + DF_BIAS] = bias;
        A.dom_f[d * DOM_F + DF_BIAS_ABS] = babs;
    }
}

// One op of the backward walk for domain blockIdx.y: CTAs 0..gridDim.x-2
// gather the input coefficients (one thread per input element), the LAST CTA
// runs the op's bias chain.
extern "C" __global__ void wave_backward(const WaveArgs A, const int op) {
    const size_t d = blockIdx.y;
    if (!(state_of(A, d) & ST_ACTIVE)) {
        return;
    }
    const int* O = A.ops + (size_t)op * OP_FIELDS;
    float* C = A.coef + d * (size_t)A.t;
    if (blockIdx.x == gridDim.x - 1) {
        bias_chain(A, O, d, C);
        return;
    }
    const size_t j = (size_t)blockIdx.x * WAVE_THREADS + threadIdx.x;
    const int kind = O[F_KIND];
    const float* co = C + O[F_OUT];
    float* ci = C + O[F_IN];
    if (j >= (size_t)O[F_NIN]) {
        return;
    }
    switch (kind) {
    case OP_CONV: {
        const int cout = O[F_A];
        const int taps = O[F_B];
        const int p = O[F_C];
        const float* w = A.fp + O[F_W];
        const unsigned int* start = A.ip + O[F_INV_START];
        const unsigned int* ent = A.ip + O[F_INV_ENT];
        const unsigned int e0 = start[j];
        const unsigned int e1 = start[j + 1];
        float acc = ci[j];
        for (int o = 0; o < cout; ++o) {
            const float* vrow = co + (size_t)o * (size_t)p;
            const float* wrow = w + (size_t)o * (size_t)taps;
            for (unsigned int e = e0; e < e1; ++e) {
                const float v = vrow[ent[2u * e]];
                if (v == 0.0f) {
                    continue;
                }
                const float wt = wrow[ent[2u * e + 1u]];
                if (wt == 0.0f) {
                    continue;
                }
                acc = fadd(acc, fmul(wt, v));
            }
        }
        ci[j] = acc;
        break;
    }
    case OP_AFFINE: {
        const float v = co[j];
        if (v != 0.0f) {
            ci[j] = fadd(ci[j], fmul(A.fp[(size_t)O[F_W] + j], v));
        }
        break;
    }
    case OP_RELU: {
        bool chord = false;
        float icpt = 0.0f;
        const float c = relu_back(A, d, O[F_SITE], j, co[j], &chord, &icpt);
        if (c != 0.0f) {
            ci[j] = fadd(ci[j], c);
        }
        break;
    }
    case OP_ADD: {
        const float v = co[j];
        if (v != 0.0f) {
            float* cr = C + O[F_RHS];
            ci[j] = fadd(ci[j], v);
            cr[j] = fadd(cr[j], v);
        }
        break;
    }
    case OP_COPY: {
        const float v = co[j];
        if (v != 0.0f) {
            ci[j] = fadd(ci[j], v);
        }
        break;
    }
    case OP_GEMM: {
        const int n_out = O[F_A];
        const size_t n_in = (size_t)O[F_B];
        const float* w = A.fp + O[F_W];
        float acc = ci[j];
        for (int o = 0; o < n_out; ++o) {
            const float v = co[o];
            if (v == 0.0f) {
                continue;
            }
            const float wi = w[(size_t)o * n_in + j];
            if (wi == 0.0f) {
                continue;
            }
            acc = fadd(acc, fmul(wi, v));
        }
        ci[j] = acc;
        break;
    }
    default:
        break;
    }
}

// ---------------------------------------------------------------------------
//  wave_concretize
// ---------------------------------------------------------------------------

// The bound of domain blockIdx.y over the input box and x* (every CTA writes
// x* for its inputs; CTA 0 runs the chain and the state transition, which
// only it reads or writes).
extern "C" __global__ void wave_concretize(const WaveArgs A, const int last, const double stop_above) {
    __shared__ float lt[WAVE_THREADS];
    __shared__ float st[WAVE_THREADS];
    const size_t d = blockIdx.y;
    const float* C = A.coef + d * (size_t)A.t;
    float* X = A.val + d * (size_t)A.t;
    const float* mid = A.fp + A.mid;
    const float* rad = A.fp + A.rad;
    const int n = A.n_in;
    {
        const int i = (int)(blockIdx.x * WAVE_THREADS + threadIdx.x);
        if (i < n) {
            const float c = C[i];
            const float m = mid[i];
            const float r = rad[i];
            X[i] = c > 0.0f ? fsub(m, r) : (c < 0.0f ? fadd(m, r) : m);
        }
    }
    if (blockIdx.x != 0) {
        return;
    }
    const unsigned int s0 = state_of(A, d);
    if (!(s0 & ST_ACTIVE)) {
        if (threadIdx.x == 0) {
            A.dom_u[d * DOM_U + DU_STATE] = s0 & ~(ST_IMPROVED | ST_GRAD);
        }
        return;
    }
    float lb = 0.0f;
    float sc = 0.0f;
    if (threadIdx.x == 0) {
        lb = A.dom_f[d * DOM_F + DF_BIAS];
        sc = A.dom_f[d * DOM_F + DF_BIAS_ABS];
    }
    for (int base = 0; base < n; base += WAVE_THREADS) {
        const int i = base + (int)threadIdx.x;
        if (i < n) {
            const float c = C[i];
            const float m = mid[i];
            const float r = rad[i];
            const float ac = fabsf(c);
            lt[threadIdx.x] = fsub(fmul(c, m), fmul(ac, r));
            st[threadIdx.x] = fmul(ac, fadd(fabsf(m), r));
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int lim = min(WAVE_THREADS, n - base);
            for (int q = 0; q < lim; ++q) {
                lb = fadd(lb, lt[q]);
                sc = fadd(sc, st[q]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) {
        return;
    }
    // wave.rs `run_domain`, after each pass: keep-best, then leave on a
    // non-finite bound, on a bound above `stop_above` (early) or after the
    // final (bound-only) pass; otherwise take a step.
    unsigned int s = s0 & ~(ST_IMPROVED | ST_GRAD);
    A.dom_u[d * DOM_U + DU_EVALS] += 1u;
    A.dom_f[d * DOM_F + DF_LB] = lb;
    A.dom_f[d * DOM_F + DF_SCALE] = sc;
    if (isfinite(lb) && (!(s & ST_HAS_BEST) || lb > A.dom_f[d * DOM_F + DF_BEST_LB])) {
        A.dom_f[d * DOM_F + DF_BEST_LB] = lb;
        A.dom_f[d * DOM_F + DF_BEST_SCALE] = sc;
        s |= ST_IMPROVED | ST_HAS_BEST;
    }
    if (!isfinite(lb)) {
        s &= ~ST_ACTIVE;
    } else if ((double)lb > stop_above) {
        s = (s | ST_EARLY) & ~ST_ACTIVE;
    } else if (last) {
        s &= ~ST_ACTIVE;
    } else {
        s |= ST_GRAD;
    }
    A.dom_u[d * DOM_U + DU_STATE] = s;
}

// ---------------------------------------------------------------------------
//  wave_forward
// ---------------------------------------------------------------------------

// One op of the linearised forward walk at x* for domain blockIdx.y (only a
// domain that takes a step this iteration), one thread per output element.
extern "C" __global__ void wave_forward(const WaveArgs A, const int op) {
    const size_t d = blockIdx.y;
    if (!(state_of(A, d) & ST_GRAD)) {
        return;
    }
    const int* O = A.ops + (size_t)op * OP_FIELDS;
    const size_t j = (size_t)blockIdx.x * WAVE_THREADS + threadIdx.x;
    if (j >= (size_t)O[F_NOUT]) {
        return;
    }
    float* V = A.val + d * (size_t)A.t;
    const float* src = V + O[F_IN];
    float* dst = V + O[F_OUT];
    switch (O[F_KIND]) {
    case OP_CONV: {
        const int taps = O[F_B];
        const size_t p = (size_t)O[F_C];
        const size_t o = j / p;
        const size_t sp = j % p;
        const unsigned int* gat = A.ip + O[F_G];
        const float* wrow = A.fp + O[F_W] + o * (size_t)taps;
        float acc = A.fp[(size_t)O[F_BIAS] + o];
        for (int t = 0; t < taps; ++t) {
            const unsigned int g = gat[(size_t)t * p + sp];
            if (g == WAVE_NONE) {
                continue;
            }
            const float wt = wrow[t];
            if (wt == 0.0f) {
                continue;
            }
            acc = fadd(acc, fmul(wt, src[g]));
        }
        dst[j] = acc;
        break;
    }
    case OP_AFFINE:
        dst[j] = fadd(fmul(A.fp[(size_t)O[F_W] + j], src[j]), A.fp[(size_t)O[F_BIAS] + j]);
        break;
    case OP_RELU: {
        const size_t f = (size_t)O[F_SITE] + j;
        const float z = src[j];
        const float l = A.fp[(size_t)A.s_l + f];
        const float u = A.fp[(size_t)A.s_u + f];
        const unsigned int p = A.ip[(size_t)A.s_pos + f];
        const unsigned int code = pin_of(A, d, p);
        const int pj = code == 0u ? 0 : (A.lit_act[code - 1u] ? 1 : -1);
        const float lj = A.coef[d * (size_t)A.t + (size_t)O[F_OUT] + j];
        float y;
        if (pj > 0 || (pj == 0 && l >= 0.0f)) {
            y = z;
        } else if (pj < 0 || u <= 0.0f) {
            y = 0.0f;
        } else if (lj >= 0.0f) {
            if (p == WAVE_NONE) {
                y = fmul(A.fp[(size_t)A.s_adapt + f], z);
            } else {
                const size_t q = d * (size_t)A.stride + p;
                A.g_alpha[q] = fadd(A.g_alpha[q], fmul(lj, z));
                y = fmul(A.alpha[q], z);
            }
        } else {
            y = fadd(fmul(A.fp[(size_t)A.s_slope + f], z), A.fp[(size_t)A.s_icpt + f]);
        }
        if (code != 0u) {
            A.g_beta[code - 1u] = A.lit_act[code - 1u] ? -z : z;
        }
        dst[j] = y;
        break;
    }
    case OP_ADD:
        dst[j] = fadd(src[j], V[(size_t)O[F_RHS] + j]);
        break;
    case OP_COPY:
        dst[j] = src[j];
        break;
    case OP_GEMM: {
        const size_t n_out = (size_t)O[F_A];
        const int n_in = O[F_B];
        const float* wt = A.fp + O[F_WT];
        float acc = A.fp[(size_t)O[F_BIAS] + j];
        for (int i = 0; i < n_in; ++i) {
            const float wi = wt[(size_t)i * n_out + j];
            if (wi == 0.0f) {
                continue;
            }
            acc = fadd(acc, fmul(wi, src[i]));
        }
        dst[j] = acc;
        break;
    }
    default:
        break;
    }
}

// ---------------------------------------------------------------------------
//  wave_adam
// ---------------------------------------------------------------------------

// wave.rs `Adam::step` for one parameter: f64 moments, a non-finite gradient
// skips it, a zero first moment skips the update, `project` then a finite
// result narrows back to f32.
__device__ __forceinline__ void adam_one(float* x, double* m, double* v, float g, const AdamConsts& K,
                                         double lr, bool alpha) {
    const double gi = (double)g;
    if (!isfinite(gi)) {
        return;
    }
    const double mm = rn_dadd(rn_dmul(K.b1, *m), rn_dmul(K.omb1, gi));
    const double vv = rn_dadd(rn_dmul(K.b2, *v), rn_dmul(rn_dmul(K.omb2, gi), gi));
    *m = mm;
    *v = vv;
    if (mm == 0.0) {
        return;
    }
    const double mh = rn_ddiv(mm, K.c1);
    const double vh = rn_ddiv(vv, K.c2);
    double next = rn_dadd((double)*x, rn_ddiv(rn_dmul(lr, mh), rn_dadd(__dsqrt_rn(vh), K.eps)));
    if (alpha) {
        // f64::clamp(0, 1): NaN stays NaN.
        if (next < 0.0) {
            next = 0.0;
        }
        if (next > 1.0) {
            next = 1.0;
        }
    } else {
        // f64::max(0): a NaN operand yields the other.
        next = fmax(next, 0.0);
    }
    if (isfinite(next)) {
        *x = __double2float_rn(next);
    }
}

// Per (domain, universe position), then per literal: the keep-best copy of an
// improved domain, then one ascent step of a stepping domain.
extern "C" __global__ void wave_adam(const WaveArgs A, const AdamConsts K) {
    const size_t e = (size_t)blockIdx.x * WAVE_THREADS + threadIdx.x;
    const size_t S = (size_t)A.stride;
    const size_t na = (size_t)A.nd * S;
    if (e < na) {
        const unsigned int s = state_of(A, e / S);
        if (s & ST_IMPROVED) {
            A.best_alpha[e] = A.alpha[e];
        }
        if (s & ST_GRAD) {
            adam_one(&A.alpha[e], &A.ma[e], &A.va[e], A.g_alpha[e], K, K.lr_a, true);
            A.g_alpha[e] = 0.0f;
        }
    } else if (e < na + (size_t)A.nl) {
        const size_t g = e - na;
        const unsigned int s = state_of(A, A.lit_dom[g]);
        if (s & ST_IMPROVED) {
            A.best_beta[g] = A.beta[g];
        }
        if (s & ST_GRAD) {
            adam_one(&A.beta[g], &A.mb[g], &A.vb[g], A.g_beta[g], K, K.lr_b, false);
        }
    }
}

// ---------------------------------------------------------------------------
//  wave_babsr
// ---------------------------------------------------------------------------

// wave.rs `score_domain` for domain d: after one bound pass, every unsplit
// universe position unstable in the frozen box, by descending
// |lam| * (-u l / (u - l)) (f64, narrowed to f32; non-finite -> f32::MAX),
// ties to the lower position; the first `k` in BPOS / BSCORE at d * k, padded
// with WAVE_NONE / NaN.
extern "C" __global__ void wave_babsr(const WaveArgs A) {
    const size_t d = (size_t)blockIdx.x * WAVE_BABSR_THREADS + threadIdx.x;
    if (d >= (size_t)A.nd) {
        return;
    }
    const int k = A.k;
    unsigned int tp[WAVE_MAX_K];
    float ts[WAVE_MAX_K];
    int len = 0;
    const float* C = A.coef + d * (size_t)A.t;
    for (int p = 0; p < A.stride; ++p) {
        if (A.pin[d * (size_t)A.stride + (size_t)p] != 0u) {
            continue;
        }
        const size_t f = A.ip[(size_t)A.uni_flat + (size_t)p];
        const float l = A.fp[(size_t)A.s_l + f];
        const float u = A.fp[(size_t)A.s_u + f];
        if (!(l < 0.0f && u > 0.0f)) {
            continue;
        }
        const double dl = (double)l;
        const double du = (double)u;
        const double lam = fabs((double)C[A.ip[(size_t)A.uni_lam + (size_t)p]]);
        const double area = (isfinite(dl) && isfinite(du)) ? rn_ddiv(rn_dmul(-du, dl), rn_dsub(du, dl)) : 1.0;
        float score = __double2float_rn(rn_dmul(lam, area));
        if (!isfinite(score)) {
            score = F32_MAX;
        }
        int at = len;
        for (int q = 0; q < len; ++q) {
            if (score > ts[q] || (score == ts[q] && (unsigned int)p < tp[q])) {
                at = q;
                break;
            }
        }
        if (at < k) {
            const int top = len < k ? len : k - 1;
            for (int q = top; q > at; --q) {
                ts[q] = ts[q - 1];
                tp[q] = tp[q - 1];
            }
            ts[at] = score;
            tp[at] = (unsigned int)p;
            if (len < k) {
                ++len;
            }
        }
    }
    for (int c = 0; c < k; ++c) {
        A.bpos[d * (size_t)k + (size_t)c] = c < len ? tp[c] : WAVE_NONE;
        A.bscore[d * (size_t)k + (size_t)c] = c < len ? ts[c] : F32_NAN;
    }
}
