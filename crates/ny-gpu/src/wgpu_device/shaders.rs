// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// #u4: the combine and EFT-min-combine taint twins plus the live on-device
// row-OR fold live in `shaders_taint.rs`
// (this file is past comfortable editing size); re-exported here so `ops/`
// keeps its one `use super::super::shaders as sh;` import path.
#[allow(unused_imports)] // consumed by the #u4 taint device probes
pub(super) use super::shaders_taint::{
    CONV_COL2IM_TAINT_SHADER, CONV_RESHAPE_TAINT_SHADER, CROWN_AW_ERROR_COMBINE_TAINT_SHADER,
    CROWN_EFT_MIN_COMBINE_TAINT_SHADER, GEMM_F32_SMALL_K_TAINT_SHADER, TAINT_ROW_OR_SHADER,
};

/// Shared WGSL prelude concatenated (NOT re-pasted) into every SOUND-IBP shader
/// (`docs/SOUND_GPU_IBP_PLAN.md` §2.2). Defines the one `round_up_pos`,
/// `is_non_finite`, the elementwise coefficient-≤1 outward widen (`widen_lo`/
/// `widen_hi`), and the NORMAL-range floors so no sound-IBP shader carries a
/// divergent copy. Prepend this string to a sound-IBP shader body; the fast
/// (unsound) shaders keep their own inline copies untouched.
///
/// # Rounding discipline (committed, one path per kind)
/// Every outward move is `center ∓ POSITIVE radius`, the radius formed by
/// `round_up_pos` on a strictly-non-negative quantity whose floor is NORMAL-range.
/// There is NO signed `next_up`/`next_down` on a signed endpoint anywhere — that
/// pattern needs `next_up(0) = 2^-149` (a subnormal Metal FTZ flushes to 0),
/// silently dropping the floor. One discipline across all kinds closes that trap.
///
/// # Constants (each bit-exact; verified by round-trip to the intended f32 bits)
/// - `U = 2^-24` (`0x33800000`), `EPS_REL = 8·U = 2^-21` (`0x35000000`),
/// - `F32_MIN_NORMAL = 2^-126` (`0x00800000`) — smallest NORMAL, survives FTZ,
/// - `ADDITIVE1 = ftz_safe_underflow_floor(1) = 2^-122` (`0x02800000`).
///
/// NOTE: the plan §2.2 rendered `ADDITIVE1` as `1.8816388e-37`, which round-trips
/// to `0x02800ec5` (~3781 ULPs ABOVE `2^-122`). That is still sound (a larger
/// additive floor only widens the interval) but does NOT equal the documented
/// `ftz_safe_underflow_floor(1)`. We use the bit-exact `1.8807910e-37` so the
/// on-device floor matches the host `ny_core::ftz_safe_underflow_floor(1)` exactly.
#[allow(dead_code)] // consumed by the sound-IBP shaders that land in the Keystone phase (§3)
pub(super) const IBP_SOUND_PRELUDE: &str = r#"
const FALLBACK_BOUND: f32 = 1e10;            // == crate::FALLBACK_BOUND (gemm.rs:37 saturation sentinel)
const U: f32            = 5.9604645e-8;      // 2^-24 exact f32 unit roundoff
const EPS_REL: f32      = 4.7683716e-7;      // 8*U: op-round + store-round + >=1-ULP CPU parity (elementwise)
const F32_MIN_NORMAL: f32 = 1.1754944e-38;   // 2^-126 smallest NORMAL — survives FTZ, base of the flush floor
const TWO_PROD_EXACT_FLOOR_F32: f32 = 3.9443045e-31; // 2^-101 == ny_core::eft::TWO_PROD_EXACT_FLOOR_F32 (from_bits 0x0D000000)
const ADDITIVE1: f32   = 1.8807910e-37;      // ftz_safe_underflow_floor(1) = 2^-122 (coefficient-<=1 kinds)
fn is_non_finite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u; }
// Smallest FTZ-safe f32 >= x for x >= 0. Classification is integer-only: a DAZ
// float comparison must not turn a positive subnormal radius into zero.
fn round_up_pos(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return x; }
    if ((bits & 0x80000000u) != 0u || magnitude == 0u) { return 0.0; }
    if (magnitude < 0x00800000u) { return F32_MIN_NORMAL; }
    return bitcast<f32>(bits + 1u);
}
// elementwise coefficient-<=1 outward widen (Add/Transpose/ReLU):
fn widen_lo(x: f32) -> f32 { return x - round_up_pos(EPS_REL * abs(x) + ADDITIVE1); }
fn widen_hi(x: f32) -> f32 { return x + round_up_pos(EPS_REL * abs(x) + ADDITIVE1); }
"#;

/// Body of the SOUND linear-layer IBP shader (`docs/SOUND_GPU_IBP_PLAN.md` §3.1),
/// concatenated AFTER [`IBP_SOUND_PRELUDE`] (which supplies `FALLBACK_BOUND`, `U`,
/// `F32_MIN_NORMAL`, `TWO_PROD_EXACT_FLOOR_F32`, `is_non_finite`, `round_up_pos`).
/// Transcribed verbatim from the
/// verified spec: the `weight_pos`/`weight_neg` split, the §0 amplified-flush
/// accumulator `flushacc`, the strict `3γ·S + 4N·U·|endpoint| + flush` radius, the
/// `is_non_finite` product guards, and the `FALLBACK_BOUND` degrade. NO f64 in the
/// body; every additive floor is NORMAL-range (Metal FTZ-safe). Do NOT mutate — build
/// the full source via [`linear_ibp_sound_source`].
const LINEAR_IBP_SOUND_BODY: &str = r#"
struct Params { batch_size:u32, in_features:u32, out_features:u32, n_ulps:u32,   // n_ulps = 2*(in_features+2)
                gamma_k:f32, slack:f32, additive:f32, _pad:u32 }                  // 32 bytes, std140-clean
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       input_lower: array<f32>;
@group(0) @binding(2) var<storage, read>       input_upper: array<f32>;
@group(0) @binding(3) var<storage, read>       weight_pos:  array<f32>;   // max(W,0) >= 0
@group(0) @binding(4) var<storage, read>       weight_neg:  array<f32>;   // min(W,0) <= 0
@group(0) @binding(5) var<storage, read>       bias:        array<f32>;
@group(0) @binding(6) var<storage, read_write> output_lower:array<f32>;
@group(0) @binding(7) var<storage, read_write> output_upper:array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.batch_size * params.out_features) { return; }
    let xoff = (idx / params.out_features) * params.in_features;
    let woff = (idx % params.out_features) * params.in_features;

    var low: f32 = 0.0; var high: f32 = 0.0;
    var s: f32 = 0.0;                 // Sigma|W|*max(|xl|,|xu|) — RTN f32 (UNDER-reports; slack recovers)
    var flushacc: f32 = 1.0;          // §0 amplified-flush base (init 1 covers bias-add + store flush)
    var bad: bool = false;

    for (var i: u32 = 0u; i < params.in_features; i = i + 1u) {
        let xl = input_lower[xoff + i]; let xu = input_upper[xoff + i];
        let wp = weight_pos[woff + i];  let wn = weight_neg[woff + i];
        let pl1 = wp*xl; let pl2 = wn*xu; let ph1 = wp*xu; let ph2 = wn*xl;
        if (is_non_finite(pl1)||is_non_finite(pl2)||is_non_finite(ph1)||is_non_finite(ph2)) { bad = true; }
        else {
            low  = low  + (pl1 + pl2);
            high = high + (ph1 + ph2);
            let absw = wp - wn;                       // |W| EXACTLY (one of wp,wn is 0)
            let xmax = max(abs(xl), abs(xu));
            s        = s + absw * xmax;
            flushacc = flushacc + max(max(absw, xmax), 1.0);   // §0: >= FLT_MIN operand-flush cover per tap
        }
    }
    let bj = bias[idx % params.out_features];
    low = low + bj; high = high + bj;
    s = s + abs(bj);                                  // bias magnitude into the error base (bias-last, single RN op)

    // Radius. flush = amplified-operand-flush floor (§0) + accumulation-underflow floor.
    let flush = params.additive + flushacc * params.slack * F32_MIN_NORMAL;
    let s_safe = s * params.slack;
    // STRICT GPU superset CPU (default): 3g*S covers GPU-vs-CPU center diff (<=2g*S) + concrete error (g*S);
    // 4*N*U*|endpoint| over-bounds the CPU N-ULP widen.
    let g3s = 3.0 * params.gamma_k * s_safe;
    let cf  = f32(params.n_ulps);
    let r_lo = round_up_pos(g3s + 4.0*cf*U*abs(low)  + flush);
    let r_hi = round_up_pos(g3s + 4.0*cf*U*abs(high) + flush);

    var lo = low - r_lo; var hi = high + r_hi;
    if (bad || is_non_finite(lo)) { lo = -FALLBACK_BOUND; }
    if (bad || is_non_finite(hi)) { hi =  FALLBACK_BOUND; }
    if (lo > hi) { lo = -FALLBACK_BOUND; hi = FALLBACK_BOUND; }
    output_lower[idx] = lo; output_upper[idx] = hi;
}
"#;

/// Body of the SOUND ReLU IBP shader (`docs/SOUND_GPU_IBP_PLAN.md` §3.7),
/// concatenated AFTER [`IBP_SOUND_PRELUDE`]. `phi(x)=max(x,0)` is EXACT and
/// monotone (coefficient <= 1 ⇒ the naïve `widen_lo`/`widen_hi` floor suffices).
/// Adds the `is_non_finite` guard the FAST ReLU shader lacks — Metal `fmax(NaN,0)`
/// returns 0 and would silently drop a NaN bound the CPU rejects as infeasible.
const RELU_IBP_SOUND_BODY: &str = r#"
struct Params { num_elements:u32, _p0:u32, _p1:u32, _p2:u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> lower: array<f32>;
@group(0) @binding(2) var<storage, read_write> upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_elements) { return; }
    let lo_in = lower[idx]; let hi_in = upper[idx];
    if (is_non_finite(lo_in) || is_non_finite(hi_in)) {   // NaN/inf box -> sound wide bound (superset of empty)
        lower[idx] = -FALLBACK_BOUND; upper[idx] = FALLBACK_BOUND; return;
    }
    lower[idx] = widen_lo(max(lo_in, 0.0));
    upper[idx] = widen_hi(max(hi_in, 0.0));
}
"#;

/// Full WGSL source of the sound linear IBP shader: [`IBP_SOUND_PRELUDE`] followed
/// by [`LINEAR_IBP_SOUND_BODY`]. Built at pipeline-creation time so the prelude
/// lives in exactly one place (§2.2).
pub(super) fn linear_ibp_sound_source() -> String {
    format!("{IBP_SOUND_PRELUDE}{LINEAR_IBP_SOUND_BODY}")
}

/// Full WGSL source of the sound ReLU IBP shader: [`IBP_SOUND_PRELUDE`] followed by
/// [`RELU_IBP_SOUND_BODY`].
pub(super) fn relu_ibp_sound_source() -> String {
    format!("{IBP_SOUND_PRELUDE}{RELU_IBP_SOUND_BODY}")
}

/// Body of the SOUND Conv2d IBP shader (`docs/SOUND_GPU_IBP_PLAN.md` §3.2), a delta
/// on [`CONV2D_IBP_IM2COL_SHADER`]. Identical `weight_pos`/`weight_neg` structure to
/// the sound Linear keystone: on every VALID (non-padding) tap accumulate the §0
/// amplified-flush base `flushacc += max(|W|, xmax, 1)` and error base
/// `s += |W|·xmax`, fold `|bias|` into `s`, then the SAME strict
/// `3γ·S + 4N·U·|endpoint| + flush` radius and `FALLBACK_BOUND` degrade. `k` and
/// `n_ulps` are computed host-side over the FULL window (padding taps not
/// subtracted ⇒ sound but looser at the image border). Grouped conv early-outs to
/// `[-FALLBACK, +FALLBACK]` (a maximal sound superset). NO f64; every floor NORMAL.
const CONV2D_IBP_SOUND_BODY: &str = r#"
struct Params {
    batch_size:u32, in_channels:u32, out_channels:u32, input_h:u32, input_w:u32,
    out_h:u32, out_w:u32, kernel_h:u32, kernel_w:u32, stride_h:u32,
    stride_w:u32, pad_h:u32, pad_w:u32, groups:u32, n_ulps:u32,
    gamma_k:f32, slack:f32, additive:f32, _pad0:u32, _pad1:u32
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       input_lower: array<f32>;
@group(0) @binding(2) var<storage, read>       input_upper: array<f32>;
@group(0) @binding(3) var<storage, read>       weight_pos:  array<f32>;
@group(0) @binding(4) var<storage, read>       weight_neg:  array<f32>;
@group(0) @binding(5) var<storage, read>       bias:        array<f32>;
@group(0) @binding(6) var<storage, read_write> output_lower:array<f32>;
@group(0) @binding(7) var<storage, read_write> output_upper:array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let out_spatial = params.out_h * params.out_w;
    let batch_stride = params.out_channels * out_spatial;
    let total_outputs = params.batch_size * batch_stride;
    if (idx >= total_outputs) { return; }
    if (params.groups != 1u) {
        output_lower[idx] = -FALLBACK_BOUND; output_upper[idx] = FALLBACK_BOUND; return;
    }
    let batch_idx = idx / batch_stride;
    let batch_offset = idx % batch_stride;
    let out_channel = batch_offset / out_spatial;
    let spatial_offset = batch_offset % out_spatial;
    let oh = spatial_offset / params.out_w;
    let ow = spatial_offset % params.out_w;

    var low: f32 = 0.0; var high: f32 = 0.0;
    var s: f32 = 0.0;
    var flushacc: f32 = 1.0;
    var bad: bool = false;

    for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
        for (var kh: u32 = 0u; kh < params.kernel_h; kh = kh + 1u) {
            let ih = i32(oh * params.stride_h + kh) - i32(params.pad_h);
            if (ih < 0 || ih >= i32(params.input_h)) { continue; }
            for (var kw: u32 = 0u; kw < params.kernel_w; kw = kw + 1u) {
                let iw = i32(ow * params.stride_w + kw) - i32(params.pad_w);
                if (iw < 0 || iw >= i32(params.input_w)) { continue; }
                let input_offset =
                    (((batch_idx * params.in_channels + ic) * params.input_h + u32(ih)) * params.input_w)
                    + u32(iw);
                let weight_offset =
                    ((((out_channel * params.in_channels) + ic) * params.kernel_h + kh) * params.kernel_w)
                    + kw;
                let xl = input_lower[input_offset]; let xu = input_upper[input_offset];
                let wp = weight_pos[weight_offset]; let wn = weight_neg[weight_offset];
                let pl1 = wp*xl; let pl2 = wn*xu; let ph1 = wp*xu; let ph2 = wn*xl;
                if (is_non_finite(pl1)||is_non_finite(pl2)||is_non_finite(ph1)||is_non_finite(ph2)) { bad = true; }
                else {
                    low  = low  + (pl1 + pl2);
                    high = high + (ph1 + ph2);
                    let absw = wp - wn;
                    let xmax = max(abs(xl), abs(xu));
                    s        = s + absw * xmax;
                    flushacc = flushacc + max(max(absw, xmax), 1.0);
                }
            }
        }
    }
    let bj = bias[out_channel];
    low = low + bj; high = high + bj;
    s = s + abs(bj);

    let flush = params.additive + flushacc * params.slack * F32_MIN_NORMAL;
    let s_safe = s * params.slack;
    let g3s = 3.0 * params.gamma_k * s_safe;
    let cf  = f32(params.n_ulps);
    let r_lo = round_up_pos(g3s + 4.0*cf*U*abs(low)  + flush);
    let r_hi = round_up_pos(g3s + 4.0*cf*U*abs(high) + flush);

    var lo = low - r_lo; var hi = high + r_hi;
    if (bad || is_non_finite(lo)) { lo = -FALLBACK_BOUND; }
    if (bad || is_non_finite(hi)) { hi =  FALLBACK_BOUND; }
    if (lo > hi) { lo = -FALLBACK_BOUND; hi = FALLBACK_BOUND; }
    output_lower[idx] = lo; output_upper[idx] = hi;
}
"#;

/// Body of the SOUND MatMul IBP shader (`docs/SOUND_GPU_IBP_PLAN.md` §3.3), a delta
/// on [`MATMUL_IBP_SHADER`]. Keeps the exact 4-corner `min/max` interval product per
/// contraction tap for `low/high`; BOTH operands are interval-valued so the §0
/// amplifier is `max(|a|, |b|, 1)` per tap. Uses the CORE radius
/// `round_up_pos(γ·S·slack + flush)` (NO `n_ulps` term — the CPU MatMul forward has
/// no matching f32 `γ·S` gold, so the oracle brute-forces the true product interval
/// directly). `k = contraction + 3`.
const MATMUL_IBP_SOUND_BODY: &str = r#"
struct Params { batch_size:u32, m:u32, k:u32, n:u32, gamma_k:f32, slack:f32, additive:f32, _pad:u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       a_lower: array<f32>;
@group(0) @binding(2) var<storage, read>       a_upper: array<f32>;
@group(0) @binding(3) var<storage, read>       b_lower: array<f32>;
@group(0) @binding(4) var<storage, read>       b_upper: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let output_matrix_size = params.m * params.n;
    let total_outputs = params.batch_size * output_matrix_size;
    if (idx >= total_outputs) { return; }
    let batch_idx = idx / output_matrix_size;
    let matrix_idx = idx % output_matrix_size;
    let i = matrix_idx / params.n;
    let j = matrix_idx % params.n;
    let a_offset = batch_idx * params.m * params.k;
    let b_offset = batch_idx * params.k * params.n;

    var low: f32 = 0.0; var high: f32 = 0.0;
    var s: f32 = 0.0;
    var flushacc: f32 = 1.0;
    var bad: bool = false;

    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        let a_l = a_lower[a_offset + i * params.k + kk];
        let a_u = a_upper[a_offset + i * params.k + kk];
        let b_l = b_lower[b_offset + kk * params.n + j];
        let b_u = b_upper[b_offset + kk * params.n + j];
        let p1 = a_l*b_l; let p2 = a_l*b_u; let p3 = a_u*b_l; let p4 = a_u*b_u;
        if (is_non_finite(p1)||is_non_finite(p2)||is_non_finite(p3)||is_non_finite(p4)) { bad = true; }
        else {
            low  = low  + min(min(p1, p2), min(p3, p4));
            high = high + max(max(p1, p2), max(p3, p4));
            let amax = max(abs(a_l), abs(a_u));
            let bmax = max(abs(b_l), abs(b_u));
            s        = s + amax * bmax;
            flushacc = flushacc + max(max(amax, bmax), 1.0);
        }
    }

    let flush = params.additive + flushacc * params.slack * F32_MIN_NORMAL;
    let r = round_up_pos(params.gamma_k * s * params.slack + flush);
    var lo = low - r; var hi = high + r;
    if (bad || is_non_finite(lo)) { lo = -FALLBACK_BOUND; }
    if (bad || is_non_finite(hi)) { hi =  FALLBACK_BOUND; }
    if (lo > hi) { lo = -FALLBACK_BOUND; hi = FALLBACK_BOUND; }
    output_lower[idx] = lo; output_upper[idx] = hi;
}
"#;

/// Body of the SOUND AvgPool IBP shader (`docs/SOUND_GPU_IBP_PLAN.md` §3.4), a delta
/// on [`AVGPOOL_IBP_SHADER`]. Coefficient `1/D ≤ 1` ⇒ NO amplifier, the naïve
/// NORMAL floor suffices. Accumulate `s += max(|il|,|iu|)` over valid taps; the
/// Higham sum error `γ_k·s·slack` is divided by `D` (any reduction order) and the
/// per-endpoint `EPS_REL·|avg|` covers the (possibly non-power-of-two) division
/// round. `k = kernel_h·kernel_w + 3` (global pool: `input_h·input_w + 3`). A γ
/// saturation on enormous global pools drives the non-finite clamp to
/// `[-FALLBACK, +FALLBACK]` (matches the CPU saturation-to-∞ guard).
const AVGPOOL_IBP_SOUND_BODY: &str = r#"
struct Params {
    num_elements:u32, channels:u32, input_h:u32, input_w:u32,
    output_h:u32, output_w:u32, kernel_h:u32, kernel_w:u32,
    stride_h:u32, stride_w:u32, pad_h:u32, pad_w:u32,
    count_include_pad:u32, gamma_k:f32, slack:f32, additive:f32
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       input_lower: array<f32>;
@group(0) @binding(2) var<storage, read>       input_upper: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.num_elements) { return; }
    let out_hw = params.output_h * params.output_w;
    let c = idx / out_hw;
    let rem = idx % out_hw;
    let oh = rem / params.output_w;
    let ow = rem % params.output_w;
    let in_hw = params.input_h * params.input_w;
    let ih_start = oh * params.stride_h;
    let iw_start = ow * params.stride_w;

    var sum_lower: f32 = 0.0;
    var sum_upper: f32 = 0.0;
    var s: f32 = 0.0;
    var count: u32 = 0u;
    var bad: bool = false;

    for (var kh: u32 = 0u; kh < params.kernel_h; kh = kh + 1u) {
        for (var kw: u32 = 0u; kw < params.kernel_w; kw = kw + 1u) {
            let ih_raw = i32(ih_start + kh) - i32(params.pad_h);
            let iw_raw = i32(iw_start + kw) - i32(params.pad_w);
            if (ih_raw >= 0 && u32(ih_raw) < params.input_h &&
                iw_raw >= 0 && u32(iw_raw) < params.input_w) {
                let flat = c * in_hw + u32(ih_raw) * params.input_w + u32(iw_raw);
                let il = input_lower[flat]; let iu = input_upper[flat];
                if (is_non_finite(il) || is_non_finite(iu)) { bad = true; }
                else {
                    sum_lower = sum_lower + il;
                    sum_upper = sum_upper + iu;
                    s = s + max(abs(il), abs(iu));
                }
                count = count + 1u;
            } else if (params.count_include_pad != 0u) {
                count = count + 1u;
            }
        }
    }

    var divisor: f32;
    if (params.count_include_pad != 0u) {
        divisor = f32(params.kernel_h * params.kernel_w);
    } else {
        divisor = f32(max(count, 1u));
    }

    var low = sum_lower / divisor;
    var high = sum_upper / divisor;
    let sum_err = params.gamma_k * s * params.slack;
    let r_lo = round_up_pos(sum_err/divisor + EPS_REL*abs(low)  + params.additive);
    let r_hi = round_up_pos(sum_err/divisor + EPS_REL*abs(high) + params.additive);
    low = low - r_lo; high = high + r_hi;

    if (bad || is_non_finite(low))  { low  = -FALLBACK_BOUND; }
    if (bad || is_non_finite(high)) { high =  FALLBACK_BOUND; }
    if (low > high) { low = -FALLBACK_BOUND; high = FALLBACK_BOUND; }
    output_lower[idx] = low; output_upper[idx] = high;
}
"#;

/// Body of the SOUND element-wise Add IBP shader (`docs/SOUND_GPU_IBP_PLAN.md`
/// §3.5), a delta on [`ADD_IBP_SHADER`]. A single RN add per endpoint is correctly
/// rounded even under cancellation ⇒ NO `γ·S`; coefficient 1 ⇒ the elementwise
/// `widen_lo`/`widen_hi` (EPS_REL·|·| + ADDITIVE1) suffices. Non-finite → ±FALLBACK;
/// a FINITE endpoint is passed through at any magnitude — `FALLBACK_BOUND` is the
/// ±inf sentinel, never a cap, and clamping past it would narrow a valid interval
/// (`ny_core::gemm::FALLBACK_BOUND`, and S2 parity with the CPU
/// `RepairStrategy::Conservative`).
const ADD_IBP_SOUND_BODY: &str = r#"
struct Params { num_elements:u32, _p0:u32, _p1:u32, _p2:u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       input_a_lower: array<f32>;
@group(0) @binding(2) var<storage, read>       input_a_upper: array<f32>;
@group(0) @binding(3) var<storage, read>       input_b_lower: array<f32>;
@group(0) @binding(4) var<storage, read>       input_b_upper: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_elements) { return; }
    var low  = input_a_lower[idx] + input_b_lower[idx];
    var high = input_a_upper[idx] + input_b_upper[idx];
    low = widen_lo(low); high = widen_hi(high);
    if (is_non_finite(low))  { low  = -FALLBACK_BOUND; }
    if (is_non_finite(high)) { high =  FALLBACK_BOUND; }
    if (low > high) { low = -FALLBACK_BOUND; high = FALLBACK_BOUND; }
    output_lower[idx] = low; output_upper[idx] = high;
}
"#;

/// Body of the SOUND Transpose IBP shader (`docs/SOUND_GPU_IBP_PLAN.md` §3.6), a
/// delta on [`TRANSPOSE_IBP_SHADER`]. The axis permutation is EXACT (strictly
/// tighter than the CPU 1-ULP widen ⇒ would violate S2), so after the index-remap
/// gather apply `widen_lo`/`widen_hi`. Non-finite → ±FALLBACK — the sentinel
/// encoding of the CPU ±inf passthrough, NOT a superset (as plain reals the finite
/// interval is a strict subset of one with an ±inf endpoint); applied to non-finite
/// endpoints only, never as a magnitude cap on finite ones; also resolves the NaN
/// divergence uniformly.
const TRANSPOSE_IBP_SOUND_BODY: &str = r#"
struct Params { batch_size:u32, rows:u32, cols:u32, _pad:u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       input_lower: array<f32>;
@group(0) @binding(2) var<storage, read>       input_upper: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let matrix_size = params.rows * params.cols;
    let total_elements = params.batch_size * matrix_size;
    if (idx >= total_elements) { return; }
    let batch_idx = idx / matrix_size;
    let out_matrix_idx = idx % matrix_size;
    let out_row = out_matrix_idx / params.rows;
    let out_col = out_matrix_idx % params.rows;
    let in_row = out_col;
    let in_col = out_row;
    let in_idx = batch_idx * matrix_size + in_row * params.cols + in_col;
    let vl = input_lower[in_idx];
    let vh = input_upper[in_idx];
    if (is_non_finite(vl) || is_non_finite(vh)) {
        output_lower[idx] = -FALLBACK_BOUND; output_upper[idx] = FALLBACK_BOUND; return;
    }
    var low  = widen_lo(vl);
    var high = widen_hi(vh);
    if (low > high) { low = -FALLBACK_BOUND; high = FALLBACK_BOUND; }
    output_lower[idx] = low; output_upper[idx] = high;
}
"#;

/// Body of the SOUND Scale IBP shader (`docs/SOUND_GPU_IBP_PLAN.md` §3.8), a delta
/// on [`SCALE_IBP_SHADER`]. Scale has an amplifier `|s|`, so it uses a HOST-computed
/// `|s|`-scaled floor `scale_floor ≥ |s|·FLT_MIN` (the fixed `ADDITIVE1` is UNSOUND
/// for `|s|>16`, ubiquitous in batchnorm/gain layers). `s==0` is special-cased
/// BEFORE the multiply to guard `Inf·0 = NaN`. `EPS_REL·|·| + scale_floor` widen.
const SCALE_IBP_SOUND_BODY: &str = r#"
struct Params { total_elements:u32, scale:f32, scale_floor:f32, zero_ulp_floor:f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       input_lower: array<f32>;
@group(0) @binding(2) var<storage, read>       input_upper: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_upper: array<f32>;

fn nan_safe_lower(x: f32) -> f32 { if (is_non_finite(x)) { return -FALLBACK_BOUND; } return x; }
fn nan_safe_upper(x: f32) -> f32 { if (is_non_finite(x)) { return  FALLBACK_BOUND; } return x; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) { return; }
    let il = input_lower[idx]; let iu = input_upper[idx];
    let s = params.scale; let sf = params.scale_floor;
    var low: f32; var high: f32;
    if (s == 0.0) { low = 0.0; high = 0.0; }
    else if (s > 0.0) { low = nan_safe_lower(s*il); high = nan_safe_upper(s*iu); }
    else              { low = nan_safe_lower(s*iu); high = nan_safe_upper(s*il); }
    low  = low  - round_up_pos(EPS_REL*abs(low)  + sf);
    high = high + round_up_pos(EPS_REL*abs(high) + sf);
    if (is_non_finite(low) || is_non_finite(high) || low > high) { low = -FALLBACK_BOUND; high = FALLBACK_BOUND; }
    output_lower[idx] = low; output_upper[idx] = high;
}
"#;

/// Full WGSL source of the sound Conv2d IBP shader: prelude + [`CONV2D_IBP_SOUND_BODY`].
pub(super) fn conv2d_ibp_sound_source() -> String {
    format!("{IBP_SOUND_PRELUDE}{CONV2D_IBP_SOUND_BODY}")
}

/// Full WGSL source of the sound MatMul IBP shader: prelude + [`MATMUL_IBP_SOUND_BODY`].
pub(super) fn matmul_ibp_sound_source() -> String {
    format!("{IBP_SOUND_PRELUDE}{MATMUL_IBP_SOUND_BODY}")
}

/// Full WGSL source of the sound AvgPool IBP shader: prelude + [`AVGPOOL_IBP_SOUND_BODY`].
pub(super) fn avgpool_ibp_sound_source() -> String {
    format!("{IBP_SOUND_PRELUDE}{AVGPOOL_IBP_SOUND_BODY}")
}

/// Full WGSL source of the sound Add IBP shader: prelude + [`ADD_IBP_SOUND_BODY`].
pub(super) fn add_ibp_sound_source() -> String {
    format!("{IBP_SOUND_PRELUDE}{ADD_IBP_SOUND_BODY}")
}

/// Full WGSL source of the sound Transpose IBP shader: prelude + [`TRANSPOSE_IBP_SOUND_BODY`].
pub(super) fn transpose_ibp_sound_source() -> String {
    format!("{IBP_SOUND_PRELUDE}{TRANSPOSE_IBP_SOUND_BODY}")
}

/// Full WGSL source of the sound Scale IBP shader: prelude + [`SCALE_IBP_SOUND_BODY`].
pub(super) fn scale_ibp_sound_source() -> String {
    format!("{IBP_SOUND_PRELUDE}{SCALE_IBP_SOUND_BODY}")
}

/// Body of the SOUND MaxPool2d CROWN-backward COEFFICIENT gather (`docs/SOUND_GPU_IBP_PLAN.md`
/// T1.2), concatenated AFTER [`IBP_SOUND_PRELUDE`]. Transposes an incoming linear
/// frontier on the maxpool OUTPUT into a frontier on the INPUT, using the proven CPU
/// winner/i* relaxation (`layers/pooling/max.rs::propagate_linear_with_bounds`):
///   - definite-winner window (l_{i*} ≥ every other u): route BOTH rows through i*.
///   - else (i* = argmax lower): route the lower row through i* iff `la>0`, the upper
///     row iff `ua<0` (the `la<0`/`ua>0` arms are CONSTANTS folded into the bias on
///     the host — NOT here). Only the `istar == j` window taps this thread's input.
///
/// Per-window `i*`+definite metadata is precomputed on the host (`window_meta[w]`:
/// low 31 bits = i* flat input index, bit 31 = is_definite; `0xFFFFFFFF` = empty
/// window). The coefficient is a COEFFICIENT-1 accumulation (routing selects incoming
/// `la`/`ua`, no multiplier) ⇒ the NORMAL-range `additive` floor suffices (no §0
/// amplifier). The per-coefficient certified error `round_up_pos(3·γ_k·S·slack +
/// additive)` (S = Σ|routed term|) DOMINATES the CPU f64 per-coeff error so the GPU
/// coefficient interval ⊇ the CPU one. Emits the coefficient + its error separately
/// (the CROWN concretize applies `err·|input|`). 7 buffers (Metal-safe). NO f64.
const MAXPOOL_CROWN_SOUND_BODY: &str = r#"
struct Params {
    num_outputs:u32, input_size:u32, output_size:u32, channels:u32,
    in_h:u32, in_w:u32, out_h:u32, out_w:u32,
    kh:u32, kw:u32, sh:u32, sw:u32, ph:u32, pw:u32,
    gamma_k:f32, slack:f32, additive:f32, total:u32, _p0:u32, _p1:u32
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       lower_a: array<f32>;
@group(0) @binding(2) var<storage, read>       upper_a: array<f32>;
@group(0) @binding(3) var<storage, read>       window_meta: array<u32>;
@group(0) @binding(4) var<storage, read_write> new_lower_a: array<f32>;
@group(0) @binding(5) var<storage, read_write> new_upper_a: array<f32>;
@group(0) @binding(6) var<storage, read_write> err_comb: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_outputs * params.input_size) { return; }
    let out_idx = idx / params.input_size;
    let j = idx % params.input_size;             // flat input index (c, ih, iw)
    let in_hw = params.in_h * params.in_w;
    let c = j / in_hw;
    let rem = j % in_hw;
    let ih = rem / params.in_w;
    let iw = rem % params.in_w;
    let out_hw = params.out_h * params.out_w;

    var lo: f32 = 0.0; var hi: f32 = 0.0;
    var s_lo: f32 = 0.0; var s_hi: f32 = 0.0;
    var bad: bool = false;

    // Enumerate the (kh_off, kw_off) taps that map input j → a window (oh, ow).
    for (var khf: u32 = 0u; khf < params.kh; khf = khf + 1u) {
        let num = i32(ih) + i32(params.ph) - i32(khf);
        if (num < 0 || (num % i32(params.sh)) != 0) { continue; }
        let oh = num / i32(params.sh);
        if (oh < 0 || oh >= i32(params.out_h)) { continue; }
        for (var kwf: u32 = 0u; kwf < params.kw; kwf = kwf + 1u) {
            let numw = i32(iw) + i32(params.pw) - i32(kwf);
            if (numw < 0 || (numw % i32(params.sw)) != 0) { continue; }
            let ow = numw / i32(params.sw);
            if (ow < 0 || ow >= i32(params.out_w)) { continue; }
            let w = c * out_hw + u32(oh) * params.out_w + u32(ow);
            let wmeta = window_meta[w];
            if (wmeta == 0xFFFFFFFFu) { continue; }        // empty (all-padding) window
            let istar = wmeta & 0x7FFFFFFFu;
            if (istar != j) { continue; }                 // j is not this window's route
            let isdef = (wmeta >> 31u) & 1u;
            let la = lower_a[out_idx * params.output_size + w];
            let ua = upper_a[out_idx * params.output_size + w];
            if (is_non_finite(la) || is_non_finite(ua)) { bad = true; }
            else if (isdef == 1u) {
                lo = lo + la; hi = hi + ua;
                s_lo = s_lo + abs(la); s_hi = s_hi + abs(ua);
            } else {
                if (la > 0.0) { lo = lo + la; s_lo = s_lo + abs(la); }   // y >= x_{i*}
                if (ua < 0.0) { hi = hi + ua; s_hi = s_hi + abs(ua); }   // ua*y <= ua*x_{i*}
            }
        }
    }

    // Coefficient-1 accumulation ⇒ NORMAL floor only (no §0 amplifier). 3·γ·S·slack
    // dominates the CPU f64 per-coeff error (center f32-vs-f64 diff + γ^f64·S).
    let r_lo = round_up_pos(3.0 * params.gamma_k * s_lo * params.slack + params.additive);
    let r_hi = round_up_pos(3.0 * params.gamma_k * s_hi * params.slack + params.additive);
    if (bad) {
        // A non-finite incoming coefficient ⇒ maximal (sound) coefficient error so the
        // downstream concretize widens to the FALLBACK regime.
        new_lower_a[idx] = 0.0; new_upper_a[idx] = 0.0;
        err_comb[idx] = FALLBACK_BOUND; err_comb[params.total + idx] = FALLBACK_BOUND;
        return;
    }
    new_lower_a[idx] = lo;
    new_upper_a[idx] = hi;
    err_comb[idx] = r_lo;
    err_comb[params.total + idx] = r_hi;
}
"#;

/// Full WGSL source of the sound MaxPool CROWN coefficient shader: prelude +
/// [`MAXPOOL_CROWN_SOUND_BODY`].
pub(super) fn maxpool_crown_sound_source() -> String {
    format!("{IBP_SOUND_PRELUDE}{MAXPOOL_CROWN_SOUND_BODY}")
}

/// WGSL shader for linear layer IBP.
///
/// This shader computes:
/// - lower = W_pos @ x_l + W_neg @ x_u + bias
/// - upper = W_pos @ x_u + W_neg @ x_l + bias
///
/// where W_pos = max(W, 0), W_neg = min(W, 0)
pub(super) const LINEAR_IBP_SHADER: &str = r#"
struct Params {
    batch_size: u32,
    in_features: u32,
    out_features: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_lower: array<f32>;
@group(0) @binding(2) var<storage, read> input_upper: array<f32>;
@group(0) @binding(3) var<storage, read> weight_pos: array<f32>;
@group(0) @binding(4) var<storage, read> weight_neg: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(7) var<storage, read_write> output_upper: array<f32>;

const FALLBACK_BOUND: f32 = 1e10;  // Must match crate::FALLBACK_BOUND (#2258)

fn is_non_finite(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

fn nan_safe_lower(x: f32) -> f32 {
    if (is_non_finite(x)) {
        return -FALLBACK_BOUND;
    }
    return x;  // Preserve finite values unchanged (#2549)
}

fn nan_safe_upper(x: f32) -> f32 {
    if (is_non_finite(x)) {
        return FALLBACK_BOUND;
    }
    return x;  // Preserve finite values unchanged (#2549)
}

fn linear_term_lower(wp: f32, xl: f32, wn: f32, xu: f32) -> f32 {
    let p1 = wp * xl;
    let p2 = wn * xu;
    if (is_non_finite(p1) || is_non_finite(p2)) {
        // Any non-finite output contribution widens lower bound conservatively.
        return -FALLBACK_BOUND;
    }
    return p1 + p2;
}

fn linear_term_upper(wp: f32, xu: f32, wn: f32, xl: f32) -> f32 {
    let p1 = wp * xu;
    let p2 = wn * xl;
    if (is_non_finite(p1) || is_non_finite(p2)) {
        // Any non-finite output contribution widens upper bound conservatively.
        return FALLBACK_BOUND;
    }
    return p1 + p2;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_outputs = params.batch_size * params.out_features;

    if (idx >= total_outputs) {
        return;
    }

    // Compute batch index and output feature index
    let batch_idx = idx / params.out_features;
    let out_idx = idx % params.out_features;

    // Input offset for this batch element
    let input_offset = batch_idx * params.in_features;

    // Weight offset for this output feature (row-major: [out_features, in_features])
    let weight_offset = out_idx * params.in_features;

    // Compute dot products with interval arithmetic
    var low: f32 = 0.0;
    var high: f32 = 0.0;

    for (var i: u32 = 0u; i < params.in_features; i = i + 1u) {
        let xl = input_lower[input_offset + i];
        let xu = input_upper[input_offset + i];
        let wp = weight_pos[weight_offset + i];
        let wn = weight_neg[weight_offset + i];

        // lower = W_pos @ x_l + W_neg @ x_u
        low = nan_safe_lower(low + linear_term_lower(wp, xl, wn, xu));
        // upper = W_pos @ x_u + W_neg @ x_l
        high = nan_safe_upper(high + linear_term_upper(wp, xu, wn, xl));
    }

    // Add bias
    low = nan_safe_lower(low + bias[out_idx]);
    high = nan_safe_upper(high + bias[out_idx]);

    if (low > high) {
        low = -FALLBACK_BOUND;
        high = FALLBACK_BOUND;
    }

    // Write output
    output_lower[idx] = low;
    output_upper[idx] = high;
}
"#;

/// WGSL shader for GEMM (C = A @ B) used by CROWN linear backward.
///
/// A: [m, k] row-major  (CROWN coefficient matrix)
/// B: [k, n] row-major  (network weight matrix)
/// C: [m, n] row-major
///
/// Uses shared-memory tiling (#3397 Step 6): each workgroup loads TILE×TILE
/// sub-matrices of A and B into `var<workgroup>` shared memory, reducing
/// global memory reads by a factor of TILE (16×).
///
/// NaN/Inf guard runs once per output, on the FINAL sum only (matching
/// `GEMM_F32_SMALL_K_SHADER`) — never on a partial sum, since the tile reduction is
/// signed and a clamped partial can cancel back into range, hiding both the lost
/// magnitude and the sentinel (§ `nan_safe_clamp`).
/// - **NaN** (from 0*Inf, Inf-Inf): preserved for downstream detection (#2708).
/// - **Inf** (from overflow of large finite products): clamp to ±FALLBACK_BOUND.
///
/// Partial sums are left unclamped without overflow risk: operands are bounded by
/// `CROWN_COEFF_MAX`, so a k-tap reduction stays far under the f32 range; and an
/// overflow that did occur reaches the final write as ±Inf ⇒ the ±FALLBACK_BOUND
/// sentinel ⇒ concretize degrades the row (fail-closed).
///
/// Note: 32×32 tiles were tested (#3599) but showed no improvement on Apple
/// Silicon (unified memory architecture, GPU L1/L2 cache already provides
/// equivalent data reuse). Kept at 16×16 for simplicity.
///
/// Reference: #2366, #2258 (unified FALLBACK_BOUND), #3397 (tiled GEMM), #3599
pub(super) const GEMM_F32_SHADER: &str = r#"
struct Params {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const FALLBACK_BOUND: f32 = 1e10;  // Must match crate::FALLBACK_BOUND (#2258)
const TILE: u32 = 16u;

fn nan_safe_clamp(x: f32) -> f32 {
    // #2708: Preserve NaN (not replace with 0.0). Replacing NaN with 0.0 silently
    // drops a coefficient whose true sign is unknown — this can make bounds tighter
    // than correct (unsound). NaN propagates through subsequent backward steps and
    // is caught at concretize time, which degrades the affected row to maximally
    // loose bounds (matching the CPU conservative fallback path).
    // Inf from overflow saturates to ±FALLBACK_BOUND: that exact magnitude is the
    // sentinel concretize degrades on, so it must only ever be produced by a FINAL
    // value — apply this to the output element, never to a running sum.
    if (x != x) {
        return x;
    }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

var<workgroup> tile_a: array<f32, 256>;  // TILE × TILE shared A sub-matrix
var<workgroup> tile_b: array<f32, 256>;  // TILE × TILE shared B sub-matrix

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = gid.x;
    let row = gid.y;
    let lc = lid.x;
    let lr = lid.y;

    var sum: f32 = 0.0;
    let num_tiles = (params.k + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Cooperative load: each thread loads one element of tile_a and tile_b.
        // Out-of-bounds elements are zero-filled (neutral for addition).
        let a_col = t * TILE + lc;
        if (row < params.m && a_col < params.k) {
            tile_a[lr * TILE + lc] = a[row * params.k + a_col];
        } else {
            tile_a[lr * TILE + lc] = 0.0;
        }

        let b_row = t * TILE + lr;
        if (b_row < params.k && col < params.n) {
            tile_b[lr * TILE + lc] = b[b_row * params.n + col];
        } else {
            tile_b[lr * TILE + lc] = 0.0;
        }

        workgroupBarrier();

        // Partial dot product from shared memory — 16 multiply-accumulates
        // per tile. The running sum is NEVER guarded: `nan_safe_clamp` may only
        // ever observe the FINAL value. Clamping a partial sum rewrites it to a
        // finite ±FALLBACK_BOUND that a later tile's opposite-signed term can pull
        // back inside the range, erasing both the discarded magnitude and the
        // sentinel `out` must carry for concretize to degrade the row. NaN/Inf
        // reach the final write on their own (NaN+x = NaN, ±Inf+finite = ±Inf).
        for (var kk: u32 = 0u; kk < TILE; kk = kk + 1u) {
            sum = sum + tile_a[lr * TILE + kk] * tile_b[kk * TILE + lc];
        }

        workgroupBarrier();
    }

    if (row < params.m && col < params.n) {
        out[row * params.n + col] = nan_safe_clamp(sum);
    }
}
"#;

/// `#u4` — GEMM with an OUT-OF-BAND STICKY TAINT CHANNEL.
///
/// Same math as [`GEMM_F32_SHADER`], plus two `u32` buffers carrying "this
/// coefficient's true magnitude is UNKNOWN and strictly larger than what the
/// f32 holds". Bindings 4/5/6 are additive; the shared GEMM's four bindings are
/// untouched, so no other caller is affected.
///
/// # Why a separate channel at all
///
/// The value channel saturates a finite overflow to `±FALLBACK_BOUND` (`1e10`)
/// and both downstream consumers test that MAGNITUDE. A magnitude is destroyed
/// by arithmetic: `ops/sentinel_taint_selfcheck.rs` measures one weight of
/// `1e-20` turning the sentinel into `1e-10` with a `5e-17` error budget, and
/// one activation slope of `1e-25` turning the `1e30` degrade marker into an
/// ordinary `2.0e5` charge — below every guard. The stored `1e10` stands for a
/// true coefficient up to `~3.4e38`, so the chain then publishes a CONFIDENT
/// number up to 28 orders of magnitude too small. No finite float can survive
/// arbitrary downscaling, so stickiness cannot live in the value.
///
/// # Why not saturate to ±inf instead
///
/// Infinity IS sticky under multiplication and is what the CPU reference does,
/// which is why it looks like the cheaper fix. It is not, and the reason is
/// `inf * 0 = NaN`: a DEAD RELU (slope exactly 0) is the most common event in a
/// deep network, and today it annihilates a tainted coefficient EXACTLY and
/// correctly — `R * 0 == 0` for every finite real `R`, and the sentinel always
/// stands for a finite real. Under ±inf every such annihilation would instead
/// produce NaN and degrade the whole row. That trades a laundering bug for a
/// tightness collapse on the hot path. The bitmask keeps annihilation exact.
///
/// # The propagation rule
///
/// ```text
/// taint_out[row, col] = OR over k of
///       (taint_a[row, k] AND (b[k, col] != 0 OR taint_b[k, col]))
///    OR (taint_b[k, col] AND (a[row, k] != 0 OR taint_a[row, k]))
///    OR  the output itself saturated
/// ```
///
/// A stored zero annihilates only when its word is clean. A tainted partner's
/// real value is unknown, so its stored zero cannot justify clearing the other
/// word. These conjuncts make the probe lanes come out right:
///
/// * a tainted coefficient times a NONZERO weight keeps the taint however small
///   the weight is — closes lanes 2 and 5, the two that launder today;
/// * a tainted coefficient times a CLEAN EXACT ZERO drops it — keeps lanes 1
///   and 4, where `R * 0 == 0` makes dropping it arithmetically justified;
/// * saturation at this op seeds the taint — keeps lane 3, the armed control.
///
/// Taint is only ever OR'd, never multiplied; only an authenticated clean-zero
/// multiplier can clear it. `1u` means tainted; the buffer is zero-initialized.
pub(super) const GEMM_F32_TAINT_SHADER: &str = r#"
struct Params {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> taint_a: array<u32>;
@group(0) @binding(5) var<storage, read> taint_b: array<u32>;
@group(0) @binding(6) var<storage, read_write> taint_out: array<u32>;

const FALLBACK_BOUND: f32 = 1e10;  // Must match crate::FALLBACK_BOUND (#2258)
const TILE: u32 = 16u;

fn nan_safe_clamp(x: f32) -> f32 {
    if (x != x) {
        return x;
    }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;
var<workgroup> tile_ta: array<u32, 256>;
var<workgroup> tile_tb: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = gid.x;
    let row = gid.y;
    let lc = lid.x;
    let lr = lid.y;

    var sum: f32 = 0.0;
    var taint: u32 = 0u;
    let num_tiles = (params.k + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + lc;
        if (row < params.m && a_col < params.k) {
            tile_a[lr * TILE + lc] = a[row * params.k + a_col];
            tile_ta[lr * TILE + lc] = taint_a[row * params.k + a_col];
        } else {
            // Zero-padded tail: neutral for the sum AND untainted, so a padded
            // tap can never invent taint.
            tile_a[lr * TILE + lc] = 0.0;
            tile_ta[lr * TILE + lc] = 0u;
        }

        let b_row = t * TILE + lr;
        if (b_row < params.k && col < params.n) {
            tile_b[lr * TILE + lc] = b[b_row * params.n + col];
            tile_tb[lr * TILE + lc] = taint_b[b_row * params.n + col];
        } else {
            tile_b[lr * TILE + lc] = 0.0;
            tile_tb[lr * TILE + lc] = 0u;
        }

        workgroupBarrier();

        for (var kk: u32 = 0u; kk < TILE; kk = kk + 1u) {
            let av = tile_a[lr * TILE + kk];
            let bv = tile_b[kk * TILE + lc];
            sum = sum + av * bv;
            // OR, never multiply. A clean exact-zero partner annihilates; a
            // TAINTED stored zero is not known to be real zero and therefore
            // cannot clear the other operand's word.
            let taw = tile_ta[lr * TILE + kk];
            let tbw = tile_tb[kk * TILE + lc];
            if (taw != 0u && (bv != 0.0 || tbw != 0u)) { taint = 1u; }
            if (tbw != 0u && (av != 0.0 || taw != 0u)) { taint = 1u; }
        }

        workgroupBarrier();
    }

    if (row < params.m && col < params.n) {
        let guarded = nan_safe_clamp(sum);
        // Saturation HERE seeds the taint: the stored value is a sentinel, not
        // the magnitude. NaN counts too — it is the other "unknown" marker.
        if (guarded != guarded || abs(guarded) >= FALLBACK_BOUND) { taint = 1u; }
        out[row * params.n + col] = guarded;
        taint_out[row * params.n + col] = taint;
    }
}
"#;

/// WGSL shader for GEMM (C = A @ B) optimized for small K (≤ 64).
///
/// Same bindings and semantics as `GEMM_F32_SHADER` but eliminates the
/// tiled shared-memory approach. For CROWN backward on competition workloads
/// (K=24-64, M up to 1.57M), the tiling overhead (barriers, shared memory
/// management, K padding) dominates actual compute time.
///
/// Each thread computes `ROWS_PER_THREAD=4` consecutive output rows at one
/// column. This gives:
///   - **0 barriers** (vs 2×ceil(K/16) in the tiled shader)
///   - **0 shared memory** (B column stays in L1/L2 cache)
///   - **4× dispatch reduction in M** (ceil(M/64) vs ceil(M/16)), keeping
///     soundnessbench-shaped workloads (M=1.57M) within the 65535 dispatch limit
///     without M-batching
///   - NaN clamp only on final write (1 check per output element)
///
/// B access pattern: threads in the same warp read adjacent columns B[kk, col],
/// giving coalesced reads. A access: all 16 threads in a warp row read the same
/// A[row, kk], giving broadcast. Both patterns are GPU-friendly.
///
/// Threshold: K ≤ `GEMM_SMALL_K_THRESHOLD` (64). Above this, the tiled shader's
/// shared-memory reuse of B outweighs the barrier cost.
///
/// Reference: #3599 (optimization target 3), #3397 (plan cache)
pub(super) const GEMM_F32_SMALL_K_SHADER: &str = r#"
struct Params {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const FALLBACK_BOUND: f32 = 1e10;  // Must match crate::FALLBACK_BOUND (#2258)
const ROWS_PER_THREAD: u32 = 4u;

fn nan_safe_clamp(x: f32) -> f32 {
    // #2708: Preserve NaN for downstream detection at concretize.
    if (x != x) {
        return x;
    }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let col = gid.x;
    let base_row = gid.y * ROWS_PER_THREAD;

    if (col >= params.n) { return; }

    // Each thread computes ROWS_PER_THREAD consecutive output rows at one column.
    // For small K (≤ 64), the inner dot product is cheap and doesn't need tiled
    // shared memory — the B column is small enough to stay in L1/L2 cache.
    for (var r: u32 = 0u; r < ROWS_PER_THREAD; r = r + 1u) {
        let row = base_row + r;
        if (row >= params.m) { return; }

        var sum: f32 = 0.0;
        let a_base = row * params.k;
        for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
            sum = sum + a[a_base + kk] * b[kk * params.n + col];
        }
        out[row * params.n + col] = nan_safe_clamp(sum);
    }
}
"#;

/// WGSL shader for batched matrix multiplication IBP.
///
/// This shader computes [A_l, A_u] @ [B_l, B_u] with interval arithmetic.
/// For each output element, computes 4 products and takes min/max.
pub(super) const MATMUL_IBP_SHADER: &str = r#"
struct Params {
    batch_size: u32,
    m: u32,           // rows of A
    k: u32,           // cols of A = rows of B
    n: u32,           // cols of B
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_lower: array<f32>;
@group(0) @binding(2) var<storage, read> a_upper: array<f32>;
@group(0) @binding(3) var<storage, read> b_lower: array<f32>;
@group(0) @binding(4) var<storage, read> b_upper: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_upper: array<f32>;

const FALLBACK_BOUND: f32 = 1e10;  // Must match crate::FALLBACK_BOUND (#2258)

fn is_non_finite(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

fn nan_safe_lower(x: f32) -> f32 {
    if (is_non_finite(x)) {
        return -FALLBACK_BOUND;
    }
    return x;  // Preserve finite values unchanged (#2549)
}

fn nan_safe_upper(x: f32) -> f32 {
    if (is_non_finite(x)) {
        return FALLBACK_BOUND;
    }
    return x;  // Preserve finite values unchanged (#2549)
}

fn min_product_nan_safe(p1: f32, p2: f32, p3: f32, p4: f32) -> f32 {
    // Guard non-finite products before min/max reduction.
    if (is_non_finite(p1) || is_non_finite(p2) || is_non_finite(p3) || is_non_finite(p4)) {
        return -FALLBACK_BOUND;
    }
    return min(min(p1, p2), min(p3, p4));
}

fn max_product_nan_safe(p1: f32, p2: f32, p3: f32, p4: f32) -> f32 {
    // Mirror CPU interval widening: any non-finite corner product => widest bound.
    if (is_non_finite(p1) || is_non_finite(p2) || is_non_finite(p3) || is_non_finite(p4)) {
        return FALLBACK_BOUND;
    }
    return max(max(p1, p2), max(p3, p4));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let output_matrix_size = params.m * params.n;
    let total_outputs = params.batch_size * output_matrix_size;

    if (idx >= total_outputs) {
        return;
    }

    // Decompose index: idx = batch * (m * n) + i * n + j
    let batch_idx = idx / output_matrix_size;
    let matrix_idx = idx % output_matrix_size;
    let i = matrix_idx / params.n;  // row in output
    let j = matrix_idx % params.n;  // col in output

    // Matrix offsets
    let a_matrix_size = params.m * params.k;
    let b_matrix_size = params.k * params.n;
    let a_offset = batch_idx * a_matrix_size;
    let b_offset = batch_idx * b_matrix_size;

    // Compute dot product with interval arithmetic
    var low: f32 = 0.0;
    var high: f32 = 0.0;

    for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
        // A[batch, i, kk]
        let a_l = a_lower[a_offset + i * params.k + kk];
        let a_u = a_upper[a_offset + i * params.k + kk];
        // B[batch, kk, j]
        let b_l = b_lower[b_offset + kk * params.n + j];
        let b_u = b_upper[b_offset + kk * params.n + j];

        // Interval multiplication: compute all 4 products, take min/max
        let p1 = a_l * b_l;
        let p2 = a_l * b_u;
        let p3 = a_u * b_l;
        let p4 = a_u * b_u;

        let min_prod = min_product_nan_safe(p1, p2, p3, p4);
        let max_prod = max_product_nan_safe(p1, p2, p3, p4);

        low = nan_safe_lower(low + min_prod);
        high = nan_safe_upper(high + max_prod);
    }

    if (low > high) {
        low = -FALLBACK_BOUND;
        high = FALLBACK_BOUND;
    }

    // Write output
    output_lower[idx] = low;
    output_upper[idx] = high;
}
"#;

/// WGSL shader for softmax IBP - Pass 1: Reduction.
///
/// For each row, computes:
/// - max_upper = max of upper bounds (for numerical stability)
/// - exp_lower[i] = exp(input_lower[i] - max_upper)
/// - exp_upper[i] = exp(input_upper[i] - max_upper)
/// - sum_exp_lower = sum of exp_lower
/// - sum_exp_upper = sum of exp_upper
///
/// This pass runs one thread per row to perform the reduction.
///
/// NaN/Inf guard: max(x, NaN) is implementation-defined in WGSL — some GPUs return x,
/// others return NaN. If max_u becomes NaN, all subsequent exp() calls produce NaN,
/// poisoning the entire row. We skip non-finite values during max reduction and guard
/// exp outputs. If the entire row is non-finite, we fall back conservatively.
/// Reference: #2390, #2258 (unified FALLBACK_BOUND)
pub(super) const SOFTMAX_REDUCE_SHADER: &str = r#"
struct Params {
    num_rows: u32,
    row_size: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_lower: array<f32>;
@group(0) @binding(2) var<storage, read> input_upper: array<f32>;
@group(0) @binding(3) var<storage, read_write> exp_lower: array<f32>;
@group(0) @binding(4) var<storage, read_write> exp_upper: array<f32>;
@group(0) @binding(5) var<storage, read_write> sum_exp_lower: array<f32>;
@group(0) @binding(6) var<storage, read_write> sum_exp_upper: array<f32>;
@group(0) @binding(7) var<storage, read_write> max_upper_out: array<f32>;

const FALLBACK_BOUND: f32 = 1e10;  // Must match crate::FALLBACK_BOUND (#2258)

fn is_non_finite(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

// Guard exp output: exp() can produce Inf for large inputs, and NaN if input is NaN.
// For softmax bounds, exp values must be non-negative and finite.
fn safe_exp(x: f32) -> f32 {
    let result = exp(x);
    if (is_non_finite(result)) {
        // exp(x) overflowed or x was NaN. Use FALLBACK_BOUND as a conservative
        // upper estimate. This is sound because it widens the softmax bounds.
        return FALLBACK_BOUND;
    }
    return result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.num_rows) {
        return;
    }

    let row_offset = row * params.row_size;

    // Pass 1: Find max of upper bounds for numerical stability.
    // Skip non-finite values to avoid poisoning the max reduction.
    // WGSL spec: max(x, NaN) is implementation-defined, so we must filter explicitly.
    var max_u: f32 = -3.4028235e+38;  // f32::MIN
    var found_finite: bool = false;
    for (var i: u32 = 0u; i < params.row_size; i = i + 1u) {
        let val = input_upper[row_offset + i];
        if (!is_non_finite(val)) {
            max_u = max(max_u, val);
            found_finite = true;
        }
    }
    // If entire row is non-finite, use 0.0 as max so exp(x - 0) = exp(x).
    // The safe_exp guard will catch any resulting Inf/NaN.
    if (!found_finite) {
        max_u = 0.0;
    }
    max_upper_out[row] = max_u;

    // Pass 2: Compute exp(x - max) and sums with NaN/Inf guards.
    var sum_l: f32 = 0.0;
    var sum_u: f32 = 0.0;
    for (var i: u32 = 0u; i < params.row_size; i = i + 1u) {
        let idx = row_offset + i;
        let raw_l = input_lower[idx];
        let raw_u = input_upper[idx];

        // Guard non-finite inputs: if input is NaN/Inf, use a conservative exp
        // argument that produces a widening bound.
        var arg_l = raw_l - max_u;
        var arg_u = raw_u - max_u;
        if (is_non_finite(raw_l)) { arg_l = -FALLBACK_BOUND; }
        if (is_non_finite(raw_u)) { arg_u = -FALLBACK_BOUND; }

        let el = safe_exp(arg_l);
        let eu = safe_exp(arg_u);
        exp_lower[idx] = el;
        exp_upper[idx] = eu;
        sum_l = sum_l + el;
        sum_u = sum_u + eu;
    }

    // Guard accumulated sums against overflow.
    if (is_non_finite(sum_l)) { sum_l = FALLBACK_BOUND; }
    if (is_non_finite(sum_u)) { sum_u = FALLBACK_BOUND; }

    sum_exp_lower[row] = sum_l;
    sum_exp_upper[row] = sum_u;
}
"#;

/// WGSL shader for softmax IBP - Pass 2: Apply bounds formula.
///
/// Using the Auto-LiRPA formula:
/// - output_lower[i] = exp_lower[i] / (sum_exp_upper - exp_upper[i] + exp_lower[i] + epsilon)
/// - output_upper[i] = exp_upper[i] / (sum_exp_lower - exp_lower[i] + exp_upper[i] + epsilon)
///
/// This pass runs one thread per element.
///
/// NaN/Inf guard: Division can produce NaN (0/0) or Inf (x/0) if exp values or sums
/// are corrupted. Softmax output bounds must be in [0, 1] since they represent
/// probability bounds. Any non-finite result is clamped to [0, 1] conservatively.
/// Reference: #2390, #2258 (unified FALLBACK_BOUND)
pub(super) const SOFTMAX_APPLY_SHADER: &str = r#"
struct Params {
    num_rows: u32,
    row_size: u32,
    _padding0: u32,
    _padding1: u32,
}

const EPSILON: f32 = 1e-12;

fn is_non_finite(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> exp_lower: array<f32>;
@group(0) @binding(2) var<storage, read> exp_upper: array<f32>;
@group(0) @binding(3) var<storage, read> sum_exp_lower: array<f32>;
@group(0) @binding(4) var<storage, read> sum_exp_upper: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = params.num_rows * params.row_size;

    if (idx >= total_elements) {
        return;
    }

    let row = idx / params.row_size;

    let el = exp_lower[idx];
    let eu = exp_upper[idx];
    let sum_l = sum_exp_lower[row];
    let sum_u = sum_exp_upper[row];

    // Auto-LiRPA softmax bounds formula
    let denom_lower = sum_u - eu + el + EPSILON;
    let denom_upper = sum_l - el + eu + EPSILON;

    var low = el / denom_lower;
    var high = eu / denom_upper;

    // Guard: softmax probability bounds must be in [0, 1].
    // Non-finite results (NaN from 0/0, Inf from x/~0) get conservative bounds.
    if (is_non_finite(low) || low < 0.0) {
        low = 0.0;
    }
    if (is_non_finite(high) || high > 1.0) {
        high = 1.0;
    }
    if (low > high) {
        low = 0.0;
        high = 1.0;
    }

    output_lower[idx] = low;
    output_upper[idx] = high;
}
"#;

/// WGSL shader for transpose IBP.
///
/// Transposes the last two dimensions of a bounded tensor.
/// Input: [batch, rows, cols] -> Output: [batch, cols, rows]
pub(super) const TRANSPOSE_IBP_SHADER: &str = r#"
struct Params {
    batch_size: u32,
    rows: u32,       // Input second-to-last dim (becomes output last dim)
    cols: u32,       // Input last dim (becomes output second-to-last dim)
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_lower: array<f32>;
@group(0) @binding(2) var<storage, read> input_upper: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let matrix_size = params.rows * params.cols;
    let total_elements = params.batch_size * matrix_size;

    if (idx >= total_elements) {
        return;
    }

    // Decompose output index: idx = batch * (cols * rows) + col * rows + row
    // Output shape is [batch, cols, rows]
    let batch_idx = idx / matrix_size;
    let out_matrix_idx = idx % matrix_size;
    let out_row = out_matrix_idx / params.rows;  // Actually col in input
    let out_col = out_matrix_idx % params.rows;  // Actually row in input

    // Input index: [batch, rows, cols] layout
    let in_row = out_col;
    let in_col = out_row;
    let in_idx = batch_idx * matrix_size + in_row * params.cols + in_col;

    // Transpose just copies with remapped indices
    output_lower[idx] = input_lower[in_idx];
    output_upper[idx] = input_upper[in_idx];
}
"#;

/// WGSL shader for scale IBP.
///
/// Element-wise multiplication by a scalar with interval arithmetic.
/// For positive scale: [l, u] -> [scale*l, scale*u]
/// For negative scale: [l, u] -> [scale*u, scale*l]
///
/// NaN/Inf guard: scale * bound can produce NaN (e.g., 0 * Inf) or Inf (overflow).
/// Uses the same is_non_finite / nan_safe_lower / nan_safe_upper pattern as
/// LINEAR_IBP_SHADER and MATMUL_IBP_SHADER.
/// Reference: #2390, #2258 (unified FALLBACK_BOUND)
pub(super) const SCALE_IBP_SHADER: &str = r#"
struct Params {
    total_elements: u32,
    scale: f32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_lower: array<f32>;
@group(0) @binding(2) var<storage, read> input_upper: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_upper: array<f32>;

const FALLBACK_BOUND: f32 = 1e10;  // Must match crate::FALLBACK_BOUND (#2258)

fn is_non_finite(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

fn nan_safe_lower(x: f32) -> f32 {
    if (is_non_finite(x)) {
        return -FALLBACK_BOUND;
    }
    return x;  // Preserve finite values unchanged (#2549)
}

fn nan_safe_upper(x: f32) -> f32 {
    if (is_non_finite(x)) {
        return FALLBACK_BOUND;
    }
    return x;  // Preserve finite values unchanged (#2549)
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.total_elements) {
        return;
    }

    let il = input_lower[idx];
    let iu = input_upper[idx];
    let s = params.scale;

    // Interval multiplication by scalar with NaN/Inf guards.
    // s * bound can produce NaN (0 * Inf) or Inf (large values).
    var low: f32;
    var high: f32;
    if (s >= 0.0) {
        low = nan_safe_lower(s * il);
        high = nan_safe_upper(s * iu);
    } else {
        low = nan_safe_lower(s * iu);
        high = nan_safe_upper(s * il);
    }

    if (low > high) {
        low = -FALLBACK_BOUND;
        high = FALLBACK_BOUND;
    }

    output_lower[idx] = low;
    output_upper[idx] = high;
}
"#;

/// WGSL shader for CROWN activation backward (element-wise relaxation + bias).
///
/// For each (spec_row, neuron) pair, updates A-matrix coefficients:
/// - Positive coefficients: lower uses lower_slope, upper uses upper_slope
/// - Negative coefficients: lower uses upper_slope, upper uses lower_slope
///
/// Also accumulates activation intercept contributions into bias via
/// workgroup-shared-memory reduction (one workgroup per spec row).
///
/// Reference: compose.rs compose_lower/compose_upper
/// Design: designs/2026-03-06-gpu-crown-backward.md §2
pub(super) const CROWN_ACTIVATION_BACKWARD_SHADER: &str = r#"
struct Params {
    num_specs: u32,
    num_neurons: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_lower_src: array<f32>;
@group(0) @binding(2) var<storage, read> a_upper_src: array<f32>;
// Packed slopes buffer: [lower_slope | upper_slope | lower_intercept | upper_intercept],
// each sub-array is num_neurons long. Single binding avoids aliasing bug (#3444).
@group(0) @binding(3) var<storage, read> slopes: array<f32>;
@group(0) @binding(4) var<storage, read_write> a_lower_dst: array<f32>;
@group(0) @binding(5) var<storage, read_write> a_upper_dst: array<f32>;
@group(0) @binding(6) var<storage, read_write> bias_lower: array<f32>;
@group(0) @binding(7) var<storage, read_write> bias_upper: array<f32>;

var<workgroup> shared_lower: array<f32, 256>;
var<workgroup> shared_upper: array<f32, 256>;

const FALLBACK_BOUND: f32 = 1e10;  // Must match crate::FALLBACK_BOUND (#2258)

fn is_non_finite(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

// #2708: For A-matrix coefficients, preserve NaN for downstream detection at
// concretize. Clamp only finite overflow to ±FALLBACK_BOUND.
fn clamp_coeff(x: f32) -> f32 {
    if (x != x) { return x; }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

// #2708: For lower bias, NaN/Inf → -Inf (maximally loose lower bound).
// Uses Inf (not NaN) for reliable propagation through WGSL workgroup reduction.
fn clamp_bias_lower(x: f32) -> f32 {
    if (is_non_finite(x)) { return bitcast<f32>(0xFF800000u); }  // -Inf
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

// #2708: For upper bias, NaN/Inf → +Inf (maximally loose upper bound).
fn clamp_bias_upper(x: f32) -> f32 {
    if (is_non_finite(x)) { return bitcast<f32>(0x7F800000u); }  // +Inf
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let spec_row = wg_id.x;
    let local_id = lid.x;

    if (spec_row >= params.num_specs) { return; }

    var local_lb: f32 = 0.0;
    var local_ub: f32 = 0.0;

    // Strided loop: each thread handles neurons at offsets local_id, local_id+256, ...
    let n = params.num_neurons;
    var neuron = local_id;
    while (neuron < n) {
        let idx = spec_row * n + neuron;
        let a_l = a_lower_src[idx];
        let a_u = a_upper_src[idx];

        // Packed slopes layout: [lower_slope(0..n) | upper_slope(n..2n) |
        //                        lower_intercept(2n..3n) | upper_intercept(3n..4n)]
        let ls = slopes[neuron];
        let us = slopes[n + neuron];
        let li = slopes[2u * n + neuron];
        let ui = slopes[3u * n + neuron];

        // Coefficient update: positive coefficients preserve bound direction,
        // negative coefficients flip. Intercept contribution follows same sign rule.
        // Reference: compose.rs:50 (compose_lower), compose.rs:90 (compose_upper)
        var new_a_l: f32;
        var new_a_u: f32;
        if (a_l >= 0.0) {
            new_a_l = a_l * ls;
            local_lb = local_lb + a_l * li;
        } else {
            new_a_l = a_l * us;
            local_lb = local_lb + a_l * ui;
        }
        if (a_u >= 0.0) {
            new_a_u = a_u * us;
            local_ub = local_ub + a_u * ui;
        } else {
            new_a_u = a_u * ls;
            local_ub = local_ub + a_u * li;
        }
        a_lower_dst[idx] = clamp_coeff(new_a_l);
        a_upper_dst[idx] = clamp_coeff(new_a_u);

        neuron = neuron + 256u;
    }

    // Workgroup reduction for bias accumulation
    shared_lower[local_id] = local_lb;
    shared_upper[local_id] = local_ub;
    workgroupBarrier();

    // Tree reduction (256 -> 128 -> 64 -> ... -> 1)
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_lower[local_id] = shared_lower[local_id] + shared_lower[local_id + stride];
            shared_upper[local_id] = shared_upper[local_id] + shared_upper[local_id + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Thread 0 writes the accumulated bias for this spec row
    if (local_id == 0u) {
        bias_lower[spec_row] = clamp_bias_lower(bias_lower[spec_row] + shared_lower[0]);
        bias_upper[spec_row] = clamp_bias_upper(bias_upper[spec_row] + shared_upper[0]);
    }
}
"#;

/// WGSL shader for ReLU dual-alpha CROWN activation backward (#4313).
///
/// Four-slice packed layout: [lower_pos_slope | cross_slope | upper_neg_slope | cross_intercept].
/// Routes coefficient updates by sign to preserve exact dual-alpha semantics:
/// - lower bound, a >= 0: `a * lower_pos_slope` (alpha_lower, through origin)
/// - lower bound, a < 0:  `a * cross_slope`, bias += `a * cross_intercept`
/// - upper bound, a >= 0: `a * cross_slope`, bias += `a * cross_intercept`
/// - upper bound, a < 0:  `a * upper_neg_slope` (alpha_upper, through origin)
///
/// Same bind-group layout and CrownActivationParams as the standard activation shader.
/// Reference: designs/2026-03-21-issue-4313-relu-dual-alpha-four-slice-abi.md
pub(super) const CROWN_ACTIVATION_RELU_DUAL_ALPHA_BACKWARD_SHADER: &str = r#"
struct Params {
    num_specs: u32,
    num_neurons: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_lower_src: array<f32>;
@group(0) @binding(2) var<storage, read> a_upper_src: array<f32>;
// Packed slopes buffer: [lower_pos_slope | cross_slope | upper_neg_slope | cross_intercept],
// each sub-array is num_neurons long.
@group(0) @binding(3) var<storage, read> slopes: array<f32>;
@group(0) @binding(4) var<storage, read_write> a_lower_dst: array<f32>;
@group(0) @binding(5) var<storage, read_write> a_upper_dst: array<f32>;
@group(0) @binding(6) var<storage, read_write> bias_lower: array<f32>;
@group(0) @binding(7) var<storage, read_write> bias_upper: array<f32>;

var<workgroup> shared_lower: array<f32, 256>;
var<workgroup> shared_upper: array<f32, 256>;

const FALLBACK_BOUND: f32 = 1e10;

fn is_non_finite(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

fn clamp_coeff(x: f32) -> f32 {
    if (x != x) { return x; }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

fn clamp_bias_lower(x: f32) -> f32 {
    if (is_non_finite(x)) { return bitcast<f32>(0xFF800000u); }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

fn clamp_bias_upper(x: f32) -> f32 {
    if (is_non_finite(x)) { return bitcast<f32>(0x7F800000u); }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let spec_row = wg_id.x;
    let local_id = lid.x;

    if (spec_row >= params.num_specs) { return; }

    var local_lb: f32 = 0.0;
    var local_ub: f32 = 0.0;

    let n = params.num_neurons;
    var neuron = local_id;
    while (neuron < n) {
        let idx = spec_row * n + neuron;
        let a_l = a_lower_src[idx];
        let a_u = a_upper_src[idx];

        // Packed layout: [lower_pos_slope(0..n) | cross_slope(n..2n) |
        //                 upper_neg_slope(2n..3n) | cross_intercept(3n..4n)]
        let lps = slopes[neuron];          // lower_pos_slope (alpha_lower)
        let cs  = slopes[n + neuron];      // cross_slope (chord)
        let uns = slopes[2u * n + neuron]; // upper_neg_slope (alpha_upper)
        let ci  = slopes[3u * n + neuron]; // cross_intercept (chord intercept)

        // Lower bound: positive A uses alpha_lower (through origin, no bias),
        //              negative A uses chord (with intercept).
        var new_a_l: f32;
        if (a_l >= 0.0) {
            new_a_l = a_l * lps;
            // alpha through origin: no intercept contribution
        } else {
            new_a_l = a_l * cs;
            local_lb = local_lb + a_l * ci;
        }

        // Upper bound: positive A uses chord (with intercept),
        //              negative A uses alpha_upper (through origin, no bias).
        var new_a_u: f32;
        if (a_u >= 0.0) {
            new_a_u = a_u * cs;
            local_ub = local_ub + a_u * ci;
        } else {
            new_a_u = a_u * uns;
            // alpha through origin: no intercept contribution
        }
        a_lower_dst[idx] = clamp_coeff(new_a_l);
        a_upper_dst[idx] = clamp_coeff(new_a_u);

        neuron = neuron + 256u;
    }

    // Workgroup reduction for bias accumulation
    shared_lower[local_id] = local_lb;
    shared_upper[local_id] = local_ub;
    workgroupBarrier();

    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_lower[local_id] = shared_lower[local_id] + shared_lower[local_id + stride];
            shared_upper[local_id] = shared_upper[local_id] + shared_upper[local_id + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (local_id == 0u) {
        bias_lower[spec_row] = clamp_bias_lower(bias_lower[spec_row] + shared_lower[0]);
        bias_upper[spec_row] = clamp_bias_upper(bias_upper[spec_row] + shared_upper[0]);
    }
}
"#;

/// WGSL shader for CROWN linear bias accumulation.
///
/// Computes b[i] += sum_j(A[i,j] * layer_bias[j]) for each spec row i.
/// Uses the A-matrix from BEFORE the linear GEMM (source buffer, not destination).
/// One workgroup per spec row with shared-memory reduction.
///
/// Design: designs/2026-03-06-gpu-crown-backward.md §3
pub(super) const CROWN_BIAS_ACCUMULATE_SHADER: &str = r#"
struct Params {
    num_specs: u32,
    num_features: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_lower: array<f32>;
@group(0) @binding(2) var<storage, read> a_upper: array<f32>;
@group(0) @binding(3) var<storage, read> layer_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> bias_lower: array<f32>;
@group(0) @binding(5) var<storage, read_write> bias_upper: array<f32>;

var<workgroup> shared_lower: array<f32, 256>;
var<workgroup> shared_upper: array<f32, 256>;

const FALLBACK_BOUND: f32 = 1e10;  // Must match crate::FALLBACK_BOUND (#2258)

fn is_non_finite(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

// #2708: Directional bias degradation — NaN/Inf in lower bias → -Inf (sound).
fn clamp_bias_lower(x: f32) -> f32 {
    if (is_non_finite(x)) { return bitcast<f32>(0xFF800000u); }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

fn clamp_bias_upper(x: f32) -> f32 {
    if (is_non_finite(x)) { return bitcast<f32>(0x7F800000u); }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let spec_row = wg_id.x;
    let local_id = lid.x;

    if (spec_row >= params.num_specs) { return; }

    var local_lb: f32 = 0.0;
    var local_ub: f32 = 0.0;

    // Strided loop: each thread accumulates a portion of the dot product
    var j = local_id;
    while (j < params.num_features) {
        let a_l = a_lower[spec_row * params.num_features + j];
        let a_u = a_upper[spec_row * params.num_features + j];
        let b = layer_bias[j];
        local_lb = local_lb + a_l * b;
        local_ub = local_ub + a_u * b;
        j = j + 256u;
    }

    // Workgroup reduction
    shared_lower[local_id] = local_lb;
    shared_upper[local_id] = local_ub;
    workgroupBarrier();

    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_lower[local_id] = shared_lower[local_id] + shared_lower[local_id + stride];
            shared_upper[local_id] = shared_upper[local_id] + shared_upper[local_id + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (local_id == 0u) {
        bias_lower[spec_row] = clamp_bias_lower(bias_lower[spec_row] + shared_lower[0]);
        bias_upper[spec_row] = clamp_bias_upper(bias_upper[spec_row] + shared_upper[0]);
    }
}
"#;

/// WGSL shader for MaxPool2d CROWN backward.
///
/// One workgroup handles one spec row. Threads:
/// 1. zero the destination input-space A-matrix for that row
/// 2. scatter coefficients for definite-winner output positions
/// 3. reduce IBP fallback bias contributions for ambiguous windows
pub(super) const CROWN_MAXPOOL2D_BACKWARD_SHADER: &str = r#"
struct Params {
    num_specs: u32,
    input_dim: u32,
    output_dim: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_lower_src: array<f32>;
@group(0) @binding(2) var<storage, read> a_upper_src: array<f32>;
@group(0) @binding(3) var<storage, read> routing: array<u32>;
// Packed bounds buffer: [ibp_lower | ibp_upper], each output_dim long.
@group(0) @binding(4) var<storage, read> ibp_bounds: array<f32>;
@group(0) @binding(5) var<storage, read_write> a_lower_dst: array<f32>;
@group(0) @binding(6) var<storage, read_write> a_upper_dst: array<f32>;
@group(0) @binding(7) var<storage, read_write> bias_lower: array<f32>;
@group(0) @binding(8) var<storage, read_write> bias_upper: array<f32>;

var<workgroup> shared_lower: array<f32, 256>;
var<workgroup> shared_upper: array<f32, 256>;

const FALLBACK_BOUND: f32 = 1e10;  // Must match crate::FALLBACK_BOUND (#2258)
const MAXPOOL_IBP_FALLBACK: u32 = 0xFFFFFFFFu;

fn is_non_finite(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

fn clamp_coeff(x: f32) -> f32 {
    if (x != x) { return x; }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

fn clamp_bias_lower(x: f32) -> f32 {
    if (is_non_finite(x)) { return bitcast<f32>(0xFF800000u); }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

fn clamp_bias_upper(x: f32) -> f32 {
    if (is_non_finite(x)) { return bitcast<f32>(0x7F800000u); }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let spec_row = wg_id.x;
    let local_id = lid.x;

    if (spec_row >= params.num_specs) { return; }

    var input_idx = local_id;
    while (input_idx < params.input_dim) {
        let dst = spec_row * params.input_dim + input_idx;
        a_lower_dst[dst] = 0.0;
        a_upper_dst[dst] = 0.0;
        input_idx = input_idx + 256u;
    }
    workgroupBarrier();

    var local_lb: f32 = 0.0;
    var local_ub: f32 = 0.0;
    var out_idx = local_id;
    while (out_idx < params.output_dim) {
        let src = spec_row * params.output_dim + out_idx;
        let route = routing[out_idx];
        let a_l = a_lower_src[src];
        let a_u = a_upper_src[src];

        if (route != MAXPOOL_IBP_FALLBACK) {
            let dst = spec_row * params.input_dim + route;
            a_lower_dst[dst] = clamp_coeff(a_l);
            a_upper_dst[dst] = clamp_coeff(a_u);
        } else {
            let max_lower = ibp_bounds[out_idx];
            let max_upper = ibp_bounds[params.output_dim + out_idx];

            if (a_l > 0.0) {
                local_lb = local_lb + a_l * max_lower;
            } else if (a_l < 0.0) {
                local_lb = local_lb + a_l * max_upper;
            }

            if (a_u > 0.0) {
                local_ub = local_ub + a_u * max_upper;
            } else if (a_u < 0.0) {
                local_ub = local_ub + a_u * max_lower;
            }
        }

        out_idx = out_idx + 256u;
    }

    shared_lower[local_id] = local_lb;
    shared_upper[local_id] = local_ub;
    workgroupBarrier();

    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_lower[local_id] = shared_lower[local_id] + shared_lower[local_id + stride];
            shared_upper[local_id] = shared_upper[local_id] + shared_upper[local_id + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (local_id == 0u) {
        bias_lower[spec_row] = clamp_bias_lower(bias_lower[spec_row] + shared_lower[0]);
        bias_upper[spec_row] = clamp_bias_upper(bias_upper[spec_row] + shared_upper[0]);
    }
}
"#;

/// WGSL shader for CROWN concretization.
///
/// Final bound computation from A-matrices and input bounds:
///   lb[i] = sum_j(max(0, A_l[i,j]) * x_l[j] + min(0, A_l[i,j]) * x_u[j]) + bias_l[i]
///   ub[i] = sum_j(max(0, A_u[i,j]) * x_u[j] + min(0, A_u[i,j]) * x_l[j]) + bias_u[i]
///
/// One workgroup per spec row with shared-memory reduction.
/// Reference: concretize.rs:56 concretize_f64_inner
/// Design: designs/2026-03-06-gpu-crown-backward.md §4
pub(super) const CROWN_CONCRETIZE_SHADER: &str = r#"
struct Params {
    num_specs: u32,
    input_dim: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_lower: array<f32>;
@group(0) @binding(2) var<storage, read> a_upper: array<f32>;
@group(0) @binding(3) var<storage, read> input_lower: array<f32>;
@group(0) @binding(4) var<storage, read> input_upper: array<f32>;
@group(0) @binding(5) var<storage, read> bias_lower: array<f32>;
@group(0) @binding(6) var<storage, read> bias_upper: array<f32>;
@group(0) @binding(7) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(8) var<storage, read_write> output_upper: array<f32>;

var<workgroup> shared_lower: array<f32, 256>;
var<workgroup> shared_upper: array<f32, 256>;

const FALLBACK_BOUND: f32 = 1e10;  // Must match crate::FALLBACK_BOUND (#2258)

fn is_non_finite(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

fn nan_safe_lower(x: f32) -> f32 {
    if (is_non_finite(x)) { return -FALLBACK_BOUND; }
    return x;
}

fn nan_safe_upper(x: f32) -> f32 {
    if (is_non_finite(x)) { return FALLBACK_BOUND; }
    return x;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let spec_row = wg_id.x;
    let local_id = lid.x;

    if (spec_row >= params.num_specs) { return; }

    // Each thread accumulates a portion of the concretization sum.
    // #2708: Track per-thread degradation flags. WGSL max/min swallow NaN
    // (returning the non-NaN operand per spec), so we must check A-coefficients
    // for NaN and overflow BEFORE the positive/negative split. When detected,
    // the flag causes the thread to emit ±Inf into the workgroup reduction,
    // which propagates reliably (WGSL guarantees Inf + finite = Inf).
    var local_lb: f32 = 0.0;
    var local_ub: f32 = 0.0;
    var lb_degraded: bool = false;
    var ub_degraded: bool = false;

    var j = local_id;
    while (j < params.input_dim) {
        let a_l = a_lower[spec_row * params.input_dim + j];
        let a_u = a_upper[spec_row * params.input_dim + j];
        let x_l = input_lower[j];
        let x_u = input_upper[j];

        // #2708: Check for NaN (x != x) or any coefficient that reached the
        // GPU overflow sentinel magnitude. Earlier shaders clamp finite overflow
        // to exactly ±FALLBACK_BOUND; once that sentinel appears, the original
        // coefficient is unknown, so concretize must degrade the row instead of
        // treating the sentinel as a legitimate finite coefficient.
        if (!lb_degraded) {
            if (a_l != a_l || abs(a_l) >= FALLBACK_BOUND) {
                lb_degraded = true;
            } else {
                let a_l_pos = max(a_l, 0.0);
                let a_l_neg = min(a_l, 0.0);
                local_lb = local_lb + a_l_pos * x_l + a_l_neg * x_u;
            }
        }

        if (!ub_degraded) {
            if (a_u != a_u || abs(a_u) >= FALLBACK_BOUND) {
                ub_degraded = true;
            } else {
                let a_u_pos = max(a_u, 0.0);
                let a_u_neg = min(a_u, 0.0);
                local_ub = local_ub + a_u_pos * x_u + a_u_neg * x_l;
            }
        }

        j = j + 256u;
    }

    // Set degraded bounds to ±Inf for reliable propagation through reduction.
    // WGSL guarantees: Inf + finite = Inf, so one degraded thread poisons the sum.
    // nan_safe_lower/upper then catches it: -Inf → -FALLBACK_BOUND, +Inf → FALLBACK_BOUND.
    if (lb_degraded) { local_lb = bitcast<f32>(0xFF800000u); }  // -Inf
    if (ub_degraded) { local_ub = bitcast<f32>(0x7F800000u); }  // +Inf

    // Workgroup reduction
    shared_lower[local_id] = local_lb;
    shared_upper[local_id] = local_ub;
    workgroupBarrier();

    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_lower[local_id] = shared_lower[local_id] + shared_lower[local_id + stride];
            shared_upper[local_id] = shared_upper[local_id] + shared_upper[local_id + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (local_id == 0u) {
        let lb = nan_safe_lower(shared_lower[0] + bias_lower[spec_row]);
        let ub = nan_safe_upper(shared_upper[0] + bias_upper[spec_row]);

        // Guard against inversions
        if (lb > ub) {
            output_lower[spec_row] = -FALLBACK_BOUND;
            output_upper[spec_row] = FALLBACK_BOUND;
        } else {
            output_lower[spec_row] = lb;
            output_upper[spec_row] = ub;
        }
    }
}
"#;

/// Elementwise `dst[i] = abs(src[i])` — fills `|A|`/`|W|` for the sound resident
/// backward's error GEMMs (`S = |A|@|W|`). 3-binding layout.
pub(super) const ABS_COPY_SHADER: &str = r#"
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) { dst[i] = abs(src[i]); }
}
"#;

/// Per-ReLU analytic alpha gradient for the GPU-resident alpha-CROWN warmup
/// (cifar100/tinyimagenet unsat keystone, step 1). Mirrors the CPU
/// `compute_graph_chain_rule_gradients` (ny-propagate): for the lower-bound
/// relaxation `y >= alpha*x` of an unstable ReLU (l < 0 < u), the bound's gradient
/// w.r.t. that neuron's `alpha` is `Σ_j max(A_lower[j,i], 0) * l_i`, where
/// `A_lower` (num_specs × num_neurons, row-major) is the lower coefficient entering
/// the ReLU and `l_i = pre_lower[i]` its pre-activation lower bound. Computing this
/// on-device (one thread per neuron, reducing over the spec rows) lets the warmup
/// alpha optimization run GPU-resident instead of round-tripping the dense
/// coefficient to the CPU per iteration. Stable neurons (caller-masked via
/// `pre_lower`/`unstable`) contribute 0; here the caller passes the unstable mask
/// folded into `pre_lower` (0 for stable) so this kernel stays branchless on it.
/// Domain-blocked for the wide/batched lane (#w4 wide α+β ascent): with
/// `num_specs_per_dom = nsp` and `num_specs = N = n_domains*nsp` stacked rows,
/// thread t covers (domain `t/nn`, neuron `t%nn`), reducing ONLY over that
/// domain's row block `[dom*nsp, (dom+1)*nsp)` and reading the domain's
/// pre-activation block `pre_lower[dom*nn + i]` — per-domain gradients, never
/// blended across domains. The single-domain case (`nsp == num_specs`) is
/// byte-identical to the original arithmetic: dom = 0, t = i, rows 0..N.
pub(super) const CROWN_ALPHA_GRADIENT_SHADER: &str = r#"
struct Params { num_specs: u32, num_neurons: u32, num_specs_per_dom: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a_lower: array<f32>;
@group(0) @binding(2) var<storage, read> pre_lower: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    // 0 = unset (legacy single-domain callers): reduce over all rows as one domain.
    let nsp = select(p.num_specs_per_dom, p.num_specs, p.num_specs_per_dom == 0u);
    let n_domains = p.num_specs / max(nsp, 1u);
    if (t < n_domains * p.num_neurons) {
        let dom = t / p.num_neurons;
        let i = t % p.num_neurons;
        var s: f32 = 0.0;
        for (var j: u32 = dom * nsp; j < (dom + 1u) * nsp; j = j + 1u) {
            let a = a_lower[j * p.num_neurons + i];
            if (a > 0.0) { s = s + a; }
        }
        grad[t] = pre_lower[dom * p.num_neurons + i] * s;
    }
}
"#;

/// Batched, strided capture of selected columns from the resident lower-A
/// coefficient matrix. One invocation writes one `(spec row, requested column)`
/// output, replacing the legacy host encoder loop that emitted one four-byte
/// `copy_buffer_to_buffer` command per output. This channel is advisory (β and
/// Complete Clip decision scoring only) and does not write the coefficient or
/// certified bound state.
///
/// Invalid requested columns retain the historical gather behavior: write
/// exactly `+0.0`. The host validates all dimensions and dispatch limits before
/// launching the kernel.
pub(super) const CROWN_STRIDED_GATHER_SHADER: &str = r#"
struct Params { num_specs: u32, num_neurons: u32, num_indices: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a_lower: array<f32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> gathered: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let total = p.num_specs * p.num_indices;
    if (t < total) {
        let row = t / p.num_indices;
        let slot = t % p.num_indices;
        let col = indices[slot];
        if (col < p.num_neurons) {
            gathered[t] = a_lower[row * p.num_neurons + col];
        } else {
            gathered[t] = 0.0;
        }
    }
}
"#;

/// Combine the two error GEMM products into the new certified coefficient error:
/// `err_new[i] = round_up((γ_k·S[i] + prop[i])·slack + additive)`, mirroring
/// `GemmEngine::crown_aw_error_step` (gemm.rs). CRITICAL (soundness): both `S` and
/// `prop` are F32-ACCUMULATED GEMM products (`fl(|A|@|W|)`, `fl(err@|W|)`), so each
/// can UNDER-report its exact value by up to a factor `γ_k` (catastrophic when a
/// large partial sum absorbs the trailing terms). `slack` is therefore host-computed
/// per contraction `k` as `combine_slack_f32(k) ≥ 1/(1−γ_k)` (NOT a fixed
/// `1.000001`, which only covered the combine's own ULPs and silently under-counted
/// the GEMM error for wide `k` → false proofs), and the result is rounded UP. Plus
/// `additive` (subnormal underflow floor). All keep `err_new` an OUTWARD (≥ true)
/// bound — the soundness requirement. `slack`, `γ_k`, `additive` are passed via the
/// uniform; non-finite collapses to a large finite taint. Before the downstream
/// concretize dispatch, its host preflight proves the complete affine radius is
/// strictly below `FALLBACK_BOUND`; the taint therefore causes refusal rather
/// than authorizing a finite sentinel as an enclosure.
///
/// # Weight-amplified DAZ floor (#gpu-metal-daz — CLOSED)
/// `additive` is only the weight-INDEPENDENT `ftz_safe_underflow_floor(k)`. `A·W` is
/// a weight-AMPLIFIED reduction, so on a Metal/DAZ adapter a subnormal *operand*
/// zeroed before the multiply loses up to `|w|·FLT_MIN` — which that floor cannot
/// cover (see `ny_core::gemm` weight-amplified-floor doc + `docs/SOUND_GPU_IBP_PLAN.md`
/// §0). Mirroring the now-fixed CPU reference `crown_aw_error_step` (gemm.rs, test
/// `crown_aw_error_step_daz_operand_flush_stays_outward`) and the IBP MatMul shader,
/// this shader now adds `flushacc[i]·p.slack·F32_MIN_NORMAL` with the SEPARABLE
/// over-bound `flushacc = 1 + k + ‖a_i‖₁ + max_j‖w_j‖₁` — `row_abs_a[i/out_cols]` is
/// the per-spec-row `‖a_i‖₁` (host/GPU-reduced `|A|@ones`) and `w_l1_max` a scalar
/// host bound on `max_j‖w_j‖₁` (`≥ ‖w_col‖₁` for every column, keeping the term
/// OUTWARD). The term is universally widening; whether it pays a real hardware
/// loss is adapter/loading-path specific. Plain GB10 WGSL flushes, while its
/// DenormPreserve core multiply is conformant; Apple and the remaining paths
/// retain separate measured/refusal evidence.
pub(super) const CROWN_AW_ERROR_COMBINE_SHADER: &str = r#"
struct Params { n: u32, slack: f32, gamma_k: f32, additive: f32,
                k: u32, out_cols: u32, w_l1_max: f32, _pad: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> s_prod: array<f32>;
@group(0) @binding(2) var<storage, read> prop: array<f32>;
@group(0) @binding(3) var<storage, read_write> err_out: array<f32>;
@group(0) @binding(4) var<storage, read> row_abs_a: array<f32>;   // per-spec-row ‖a_i‖₁
const F32_MIN_NORMAL: f32 = 1.1754944e-38;   // 2^-126 smallest NORMAL — survives Metal FTZ
const FALLBACK_BOUND: f32 = 1e10;            // == crate::FALLBACK_BOUND (the GEMM saturation sentinel)
fn is_nonfinite(x: f32) -> bool { let b = bitcast<u32>(x); return (b & 0x7f800000u) == 0x7f800000u; }
// smallest f32 >= x (for x >= 0): the successor covers the final f32 op rounding DOWN.
fn round_up_pos(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return x; }
    if ((bits & 0x80000000u) != 0u || magnitude == 0u) { return 0.0; }
    if (magnitude < 0x00800000u) { return F32_MIN_NORMAL; }
    return bitcast<f32>(bits + 1u);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) {
        // s_prod = fl(|A|@|W|), prop = fl(err@|W|): both f32-accumulated over the
        // contraction k, so each UNDER-reports its exact sum by up to a factor γ_k.
        // `p.slack` (host: >= 1/(1-γ_k) with combine-ULP headroom) recovers an
        // OUTWARD bound; round_up_pos covers the remaining f32 op rounding.
        // §0 weight-amplified operand-flush cover: 1 + k + ‖a_i‖₁ + max_j‖w_j‖₁ ≥
        // 1 + Σ_l max(|a_il|,|w_lj|,1), the exact per-tap subnormal-flush over-bound.
        var flushacc = 1.0 + f32(p.k) + p.w_l1_max;
        if (p.out_cols > 0u) { flushacc = flushacc + row_abs_a[i / p.out_cols]; }
        let flush = p.additive + flushacc * p.slack * F32_MIN_NORMAL;
        var e = round_up_pos((p.gamma_k * s_prod[i] + prop[i]) * p.slack + flush);
        if (is_nonfinite(e) || e < 0.0) { e = 1e30; }
        // Both inputs are NON-NEGATIVE reductions run through the GEMM, so each is
        // monotone in its partials and saturates to EXACTLY FALLBACK_BOUND once the
        // true sum passes it. At the sentinel the true |A|@|W| / err@|W| is unknown
        // and strictly larger, so `e` would UNDER-cover: degrade instead. (The signed
        // A@W coefficient can cancel back under the sentinel and reach concretize
        // looking legitimate; this row is the only place that saturation is visible.)
        if (s_prod[i] >= FALLBACK_BOUND || prop[i] >= FALLBACK_BOUND) { e = 1e30; }
        err_out[i] = e;
    }
}
"#;

/// EFT twin GEMM (#eft-err, `docs/EFT_COMPENSATED_CERTIFIED_ERROR_DESIGN.md`):
/// recompute `V = fl(A@W)` with a DETERMINISTIC, compiler-immune op sequence
/// while measuring or conservatively charging the rounding residual of that
/// sequence per output element. Primary products/adds use the separately
/// qualified core operations; residual recovery uses explicit `fma` barriers
/// (probe-pinned and gated per adapter by `verify_eft_primitives`):
///
/// * product: `p = a * w` = RN(a·w), deliberately using the rung-3-qualified
///   core multiply rather than `fma(a,w,0)` (the measured GB10 FMA path
///   DAZ-zeroes subnormal multiplicands even under DenormPreserve); residual
///   `ep = fma(a, w, −p)` is EXACT when the FMA honors its operands and `|p| >=
///   TWO_PROD_EXACT_FLOOR_F32`. If that residual FMA DAZ-zeroes a subnormal
///   multiplicand while `p` is normal, it returns `−p`, whose magnitude is a
///   conservative over-charge. Smaller products use `F32_MIN_NORMAL` instead;
/// * accumulate: `s = acc + p` (single RN add) with the fma-barrier TwoSum
///   residual `es`; it is exact on a conforming FMA, while the admitted
///   subnormal-result zero-flush is covered by the scaled rung-3 floor.
///
/// On a conforming FMA the terms telescope exactly:
/// `Σ a·w = V + Σ ep + Σ es`. Under the admitted measured FMA behavior,
/// conservative residuals and explicit floors replace that identity with the
/// needed outward inequality. Thus `R = Σ|ep| + Σ|es| (+floors)` bounds
/// `|exact − V|` for THIS sequence. `R`'s own f32 accumulation is charged on
/// the host via `eft_r_slack_f32`. The twin
/// value `V` is NOT the shipped coefficient — the min-combine below measures
/// `d = |V − value_kernel_output|` on device and charges it, so NO assumption
/// about the value kernel's compilation is needed (sound for any fusion).
pub(super) const GEMM_F32_EFT_TWIN_SHADER: &str = r#"
struct Params { m: u32, k: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;   // m x k
@group(0) @binding(2) var<storage, read> w_in: array<f32>;   // k x n
@group(0) @binding(3) var<storage, read_write> v_out: array<f32>; // m x n twin value
@group(0) @binding(4) var<storage, read_write> r_out: array<f32>; // m x n residual sum
const F32_MIN_NORMAL: f32 = 1.1754944e-38;
const TWO_PROD_EXACT_FLOOR_F32: f32 = 3.9443045e-31; // 2^-101 == ny_core::eft::TWO_PROD_EXACT_FLOOR_F32
const TILE: u32 = 16u;
var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;
// Tiled exactly like GEMM_F32_SHADER (same cooperative loads / memory pattern —
// the residual channel is ORDER-AGNOSTIC: it measures the sequence it executes,
// so tiling changes only performance, never soundness). Zero-padded OOB taps
// contribute exact zeros (0*0 products, exact adds of 0) and cost no residual.
@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = gid.x;
    let row = gid.y;
    let lc = lid.x;
    let lr = lid.y;
    var acc: f32 = 0.0;
    var rsum: f32 = 0.0;
    let num_tiles = (params.k + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + lc;
        if (row < params.m && a_col < params.k) {
            tile_a[lr * TILE + lc] = a_in[row * params.k + a_col];
        } else {
            tile_a[lr * TILE + lc] = 0.0;
        }
        let b_row = t * TILE + lr;
        if (b_row < params.k && col < params.n) {
            tile_b[lr * TILE + lc] = w_in[b_row * params.n + col];
        } else {
            tile_b[lr * TILE + lc] = 0.0;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < TILE; kk = kk + 1u) {
            let a = tile_a[lr * TILE + kk];
            let w = tile_b[kk * TILE + lc];
            // Primary product via the rung-3-qualified core multiply. The
            // residual alone uses FMA; see the host doc above for why FMA DAZ
            // then over-charges or enters the explicit small-product floor.
            let prod = a * w;
            let ep = fma(a, w, -prod);
            var eterm = abs(ep);
            if (a != 0.0 && w != 0.0 && abs(prod) < TWO_PROD_EXACT_FLOOR_F32) {
                // Below the TwoProdFMA exactness range the residual may itself
                // round; replace it with a sound normal-range charge.
                // Guard at 2^-101, NOT 2^-126: throughout [2^-126, 2^-101) the
                // residual `ep` is ITSELF rounded (often to 0), so trusting it
                // publishes a radius that does not enclose. Charging 2^-126 there
                // is sound because |a·w − prod| <= ½·ulp(prod) <= 2^-126.
                eterm = F32_MIN_NORMAL;
            }
            // fma-barrier TwoSum: s = RN(acc + prod), es exact.
            let s = acc + prod;
            let bb = fma(-1.0, acc, s);
            let sb = fma(-1.0, bb, s);
            let da = fma(-1.0, sb, acc);
            let db = fma(-1.0, bb, prod);
            let es = da + db;
            rsum = rsum + eterm + abs(es);
            acc = s;
        }
        workgroupBarrier();
    }
    if (row < params.m && col < params.n) {
        v_out[row * params.n + col] = acc;
        r_out[row * params.n + col] = rsum;
    }
}
"#;

/// EFT min-combine (#eft-err): tighten the certified error written by
/// `CROWN_AW_ERROR_COMBINE_SHADER` with the a-posteriori EFT bound
/// `err_eft = round_up(((R + |V − value|)·r_slack + prop·slack) + flush)`,
/// taking `err_out = min(err_higham, err_eft)` per element. Both are valid
/// sound bounds on `|exact_new_coeff − shipped_coeff|` (the EFT side because
/// `|exact − value| ≤ |exact − V| + |V − value| ≤ R·r_slack + d`, with `d`
/// MEASURED on device), so the min is sound. Fail-closed per element: any
/// non-finite input or a saturated `prop` keeps the Higham value untouched.
/// Dispatched ONLY under the `NY_EFT_ERR` gate + `verify_eft_primitives` —
/// gate off ⇒ this shader never runs ⇒ byte-identical.
pub(super) const CROWN_EFT_MIN_COMBINE_SHADER: &str = r#"
struct Params { n: u32, r_slack: f32, slack: f32, additive: f32,
                k: u32, out_cols: u32, w_l1_max: f32, _pad: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> v_twin: array<f32>;
@group(0) @binding(2) var<storage, read> r_in: array<f32>;
@group(0) @binding(3) var<storage, read> value: array<f32>;   // the shipped A@W output
@group(0) @binding(4) var<storage, read> prop: array<f32>;    // fl(err@|W|)
@group(0) @binding(5) var<storage, read_write> err_out: array<f32>;
@group(0) @binding(6) var<storage, read> row_abs_a: array<f32>;
@group(0) @binding(7) var<storage, read> s_prod: array<f32>;  // fl(|A|@|W|)
const F32_MIN_NORMAL: f32 = 1.1754944e-38;
const FALLBACK_BOUND: f32 = 1e10;
fn is_nonfinite(x: f32) -> bool { let b = bitcast<u32>(x); return (b & 0x7f800000u) == 0x7f800000u; }
fn round_up_pos(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return x; }
    if ((bits & 0x80000000u) != 0u || magnitude == 0u) { return 0.0; }
    if (magnitude < 0x00800000u) { return F32_MIN_NORMAL; }
    return bitcast<f32>(bits + 1u);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let v = v_twin[i];
    let val = value[i];
    let r = r_in[i];
    let pr = prop[i];
    // Fail-closed: anything non-finite or a saturated propagated term keeps
    // the (already sound) Higham value.
    if (is_nonfinite(v) || is_nonfinite(val) || is_nonfinite(r) || is_nonfinite(pr)) { return; }
    if (pr >= FALLBACK_BOUND) { return; }
    // SENTINEL STICKINESS (#gpu-typed-authority). The Higham combine
    // DELIBERATELY degrades to `e = 1e30` when EITHER `s_prod = fl(|A|@|W|)` or
    // `prop` saturates at FALLBACK_BOUND, because past saturation the true
    // reduction is unknown and strictly larger. This min-combine previously
    // observed only `prop` — it had no `s_prod` binding at all — so on a row
    // where `s_prod` saturated but `prop` did not, `min(err_out, e_eft)` would
    // silently LOWER that deliberately-degraded 1e30 charge back to a measured
    // one, erasing the finite overflow transport sentinel. Whether the EFT
    // identity happens to survive there is exactly the kind of unwritten
    // argument the quarantine exists to demand, so we refuse instead: the
    // sentinel is now STICKY across this arm. Strictly a WIDENING (the Higham
    // charge ships unchanged), never a tightening.
    if (s_prod[i] >= FALLBACK_BOUND) { return; }
    // d = |V - value|: RN of an exact-difference class; r_slack (host, outward)
    // carries headroom for this op and R's own f32 accumulation.
    let d = abs(v - val);
    // Same §0 operand-flush cover as the Higham combine (the twin reads the
    // SAME possibly-DAZ-flushed operands, so exact(unflushed) needs the floor).
    var flushacc = 1.0 + f32(p.k) + p.w_l1_max;
    if (p.out_cols > 0u) { flushacc = flushacc + row_abs_a[i / p.out_cols]; }
    let flush = p.additive + flushacc * p.slack * F32_MIN_NORMAL;
    let e_eft = round_up_pos((r + d) * p.r_slack + pr * p.slack + flush);
    if (is_nonfinite(e_eft) || e_eft < 0.0) { return; }
    err_out[i] = min(err_out[i], e_eft);
}
"#;

/// Accumulate a linear layer's bias contribution into the running CROWN bias and
/// its certified error, on-device (one workgroup per spec row, tree reduction).
/// For row s over the layer's `k` outputs:
///   bias_out[s]     += Σ_k a[s,k]·bias[k]
///   bias_err_out[s] = add_up(
///       bias_err_out[s],
///       round_up(γ_k·Σ_k|a[s,k]·bias[k]| + Σ_k err[s,k]·|bias[k]|),
///       flush,
///   )
/// The `γ_k·Σ|a·bias|` term covers the f32 reduction's own rounding (the host
/// accumulates in f64, so the GPU bound is at most this term looser — still
/// sound). `+=` is a read-modify-write of one element by thread 0 (no race).
pub(super) const CROWN_BIAS_ERR_ACCUMULATE_SHADER: &str = r#"
// #eft-err (former padding _pad0/_pad1): eft_mode=1 ⇒ the a·bias reduction's
// rounding is charged from a core-RN product plus an FMA residual (exact on a
// conforming FMA; conservative/floored under the measured FMA-operand DAZ),
// with barrier-TwoSum residuals through the tree and final running-sum add,
// all charged ·eft_r_slack,
// replacing the a-priori γ_k·Σ|a·bias| term (k = 10^4–10^5 on conv layers).
// The propagated Σ err·|bias| stays. 0 ⇒ byte-identical legacy.
struct Params { num_specs: u32, k: u32, gamma_k: f32, additive: f32, slack: f32, eft_mode: u32, eft_r_slack: f32, _pad2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> a_err: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> bias_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> bias_err_out: array<f32>;

const F32_MIN_NORMAL: f32 = 1.1754944e-38;   // 2^-126 smallest NORMAL — survives Metal FTZ
const TWO_PROD_EXACT_FLOOR_F32: f32 = 3.9443045e-31; // 2^-101 == ny_core::eft::TWO_PROD_EXACT_FLOOR_F32

var<workgroup> sv: array<f32, 256>;   // Σ a·bias
var<workgroup> sa: array<f32, 256>;   // Σ |a·bias|
var<workgroup> se: array<f32, 256>;   // Σ err·|bias|
var<workgroup> sf: array<f32, 256>;   // §0 amplified-flush accumulator
var<workgroup> sr: array<f32, 256>;   // #eft-err measured residuals

fn round_up_pos(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return x; }
    if ((bits & 0x80000000u) != 0u || magnitude == 0u) { return 0.0; }
    if (magnitude < 0x00800000u) { return F32_MIN_NORMAL; }
    return bitcast<f32>(bits + 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let s = wg.x;
    let t = lid.x;
    if (s >= p.num_specs) { return; }
    let eft = p.eft_mode == 1u;
    var v: f32 = 0.0;
    var av: f32 = 0.0;
    var ev: f32 = 0.0;
    var fa: f32 = 1.0;   // §0 amplified-flush base (per-thread; reduced below)
    var rv: f32 = 0.0;   // #eft-err per-thread measured residuals
    var j = t;
    while (j < p.k) {
        let aj = a[s * p.k + j];
        let bj = bias[j];
        if (!eft) {
            v = v + aj * bj;
            av = av + abs(aj * bj);
        } else {
            // Measured product + add residuals (exact; small products use the
            // TwoProdFMA exactness guard and a normal-range charge).
            let prod = aj * bj;
            var ep = abs(fma(aj, bj, -prod));
            if (aj != 0.0 && bj != 0.0 && abs(prod) < TWO_PROD_EXACT_FLOOR_F32) { ep = F32_MIN_NORMAL; }
            let s2 = v + prod;
            let bb = fma(-1.0, v, s2);
            let sb = fma(-1.0, bb, s2);
            rv = rv + ep + abs(fma(-1.0, sb, v) + fma(-1.0, bb, prod));
            v = s2;
        }
        ev = ev + a_err[s * p.k + j] * abs(bj);
        // §0: a subnormal aj flushed to 0 by Metal FTZ then amplified by |bj| loses
        // up to |bj|·FLT_MIN from `v`; the γ_k·Σ|a·bias| term reads the same flushed
        // aj as 0 so misses it. max(|aj|,|bj|,1) per tap dominates the lost product.
        fa = fa + max(max(abs(aj), abs(bj)), 1.0);
        j = j + 256u;
    }
    sv[t] = v; sa[t] = av; se[t] = ev; sf[t] = fa; sr[t] = rv;
    workgroupBarrier();
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (t < stride) {
            // Value adds keep the exact legacy expression; their fma-barrier
            // TwoSum residuals go into the r lane (multiplied by 0 in legacy mode).
            let a0 = sv[t];
            let b0 = sv[t + stride];
            let s0 = a0 + b0;
            let bb0 = fma(-1.0, a0, s0);
            let sb0 = fma(-1.0, bb0, s0);
            let r0 = abs(fma(-1.0, sb0, a0) + fma(-1.0, bb0, b0));
            sv[t] = s0;
            sa[t] = sa[t] + sa[t + stride];
            se[t] = se[t] + se[t + stride];
            sf[t] = sf[t] + sf[t + stride];
            sr[t] = sr[t] + sr[t + stride] + r0;
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (t == 0u) {
        let old = bias_out[s];
        let sum = old + sv[0];
        bias_out[s] = sum;
        // Every non-negative term is a certified radius. Assemble the flush
        // and the update outward; ordinary RN addition can otherwise absorb a
        // tiny but real floor into an existing O(1) bias error.
        let flush_scaled = round_up_pos(round_up_pos(sf[0] * p.slack) * F32_MIN_NORMAL);
        let flush = round_up_pos(p.additive + flush_scaled);
        let old_err = bias_err_out[s];
        if (!eft) {
            // `sa` and `se` are non-negative f32 reductions. Recover their
            // possible k-term undercount before publishing the local radius.
            let reduced_err = round_up_pos(p.gamma_k * sa[0] + se[0]);
            let local_err = round_up_pos(reduced_err * p.slack);
            bias_err_out[s] = round_up_pos(round_up_pos(old_err + local_err) + flush);
        } else {
            // Final running-sum add's residual, measured like the rest.
            let bbf = fma(-1.0, old, sum);
            let sbf = fma(-1.0, bbf, sum);
            let rf = abs(fma(-1.0, sbf, old) + fma(-1.0, bbf, sv[0]));
            let residual_err = round_up_pos((sr[0] + rf) * p.eft_r_slack);
            let propagated_err = round_up_pos(se[0] * p.slack);
            let local_err = round_up_pos(residual_err + propagated_err);
            bias_err_out[s] = round_up_pos(round_up_pos(old_err + local_err) + flush);
        }
    }
}
"#;

/// Resident activation-backward coefficient + error (elementwise), mirroring
/// `crown_activation_error_step`. One side (lower or upper, via `is_upper`) per
/// dispatch to stay under the 8-storage-buffer limit. For element idx (neuron
/// i = idx % num_neurons):
///   sel  = lower: a>=0?ls:us ; upper: a>=0?us:ls
///   a_out = a*sel
///   err_out = (err*(|ls|+|us|) + gap + additive)*SLACK
/// where the host's exact `gap = |a*sel - fl(a*sel)|` (needs f64) is OVER-bounded
/// on-device by `u*|a*sel|` (u=2^-24, max f32 product rounding) — sound, ULP-loose.
/// `beta_signed` (binding 7, per-neuron, num_neurons wide; all-zero ⇒ inert/byte-
/// identical) adds the β-CROWN split-constraint Lagrangian dual term post-slope
/// (#unsat-keystone step 4). Mirrors the CPU `apply_constrained_relu_beta_contribution`:
/// for a split ReLU neuron i with signed_beta = β·sign (sign +1 active / −1 inactive,
/// β≥0), the POST-transform coefficient gets `lower -= signed_beta`, `upper += signed_beta`,
/// the SAME constant for every spec row, independent of `a`. A β-CROWN bound is a valid
/// Lagrangian dual for ANY β≥0, so this is sound regardless of the β values; the only new
/// f32 op (the ± add) is over-bounded by an extra `U·|a*sel|` in `gap` (gated on β≠0 so the
/// no-β path stays byte-identical).
pub(super) const CROWN_ACTIVATION_RESIDENT_SHADER: &str = r#"
// #batched-bab: `num_specs_per_dom` is the per-domain spec-row count. The slope/β
// buffers are per-domain-block ([n_domains * num_neurons]); row `idx/num_neurons`
// belongs to domain `dom = (idx/num_neurons) / num_specs_per_dom`, so its slopes
// live at `dom*num_neurons + i`. With `num_specs_per_dom == num_specs` (single
// domain, n_domains=1) `dom` is always 0 → `[i]`, BYTE-IDENTICAL to the pre-batch
// path (buffers stay num_neurons wide).
// #eft-err (former padding `_p0` → `eft_mode`): 1 ⇒ (a) the a·sel and ∓β rounding
// is charged by an FMA product residual + fma-barrier TwoSum residual (exact on
// a conforming FMA; conservative/floored under measured FMA-operand DAZ — zero
// for the stable sel∈{0,1} majority where the ops are exact) instead of the
// a-priori U·|coeff| charge, and (b) the propagated error uses the LIPSCHITZ
// factor of the piecewise-linear activation map v ↦ v·sel(v)∓β — `|sel|` when
// the coefficient's sign is certain (|a| > err), `max(|ls|,|us|)` otherwise —
// instead of the conservative `|ls|+|us|` (which DOUBLES the propagated error at
// every stable-active ReLU). 0 ⇒ byte-identical legacy behavior.
struct Params { num_specs: u32, num_neurons: u32, is_upper: u32, additive: f32, num_specs_per_dom: u32, eft_mode: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;
@group(0) @binding(2) var<storage, read> err_in: array<f32>;
@group(0) @binding(3) var<storage, read> ls: array<f32>;
@group(0) @binding(4) var<storage, read> us: array<f32>;
@group(0) @binding(5) var<storage, read_write> a_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> err_out: array<f32>;
@group(0) @binding(7) var<storage, read> beta_signed: array<f32>;
const U: f32 = 0.00000005960464477539063; // 2^-24
const SLACK: f32 = 1.000001;
const F32_MIN_NORMAL_ACT: f32 = 1.1754944e-38;
const TWO_PROD_EXACT_FLOOR_F32: f32 = 3.9443045e-31; // 2^-101 == ny_core::eft::TWO_PROD_EXACT_FLOOR_F32
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.num_specs * p.num_neurons;
    if (idx >= total) { return; }
    let i = idx % p.num_neurons;
    let dom = (idx / p.num_neurons) / p.num_specs_per_dom;
    let sbase = dom * p.num_neurons;
    let a = a_in[idx];
    let lsv = ls[sbase + i];
    let usv = us[sbase + i];
    var sel: f32;
    if (p.is_upper == 0u) { sel = select(usv, lsv, a >= 0.0); }
    else { sel = select(lsv, usv, a >= 0.0); }
    let base = a * sel;
    // beta_signed[i] = β·sign; lower subtracts, upper adds (CPU relu.rs L123-126).
    let bv = beta_signed[sbase + i];
    var coeff: f32;
    if (p.is_upper == 0u) { coeff = base - bv; } else { coeff = base + bv; }
    a_out[idx] = coeff;
    let e_in = err_in[idx];
    if (p.eft_mode == 0u) {
        let slope_sum = abs(lsv) + abs(usv);
        // gap: U·|coeff| bounds the a*sel rounding (no-β case); when β≠0 the extra ± add
        // is bounded by an additional U·|base|. select keeps β=0 byte-identical to before.
        let extra = select(0.0, abs(base) * U, bv != 0.0);
        let gap = abs(coeff) * U + extra;
        err_out[idx] = (e_in * slope_sum + gap + p.additive) * SLACK;
    } else {
        // Measured gap: e_prod = fma(a,sel,−base) is the exact product residual
        // on a conforming FMA and a conservative over-charge (or explicit
        // small-product floor) under measured FMA-operand DAZ. It is 0 for the
        // stable sel∈{0,1} majority; the β step's residual via the
        // fma-barrier TwoSum (exact; 0 when β=0). Small products use the
        // TwoProdFMA exactness floor, while the charged radius stays FLT_MIN.
        var e_prod = abs(fma(a, sel, -base));
        if (a != 0.0 && sel != 0.0 && abs(base) < TWO_PROD_EXACT_FLOOR_F32) { e_prod = F32_MIN_NORMAL_ACT; }
        let sb2 = select(bv, -bv, p.is_upper == 0u); // coeff = base + sb2
        let bb = fma(-1.0, base, coeff);
        let sbb = fma(-1.0, bb, coeff);
        let e_sub = abs(fma(-1.0, sbb, base) + fma(-1.0, bb, sb2));
        // Lipschitz propagation of the piecewise-linear map: |sel| when the
        // coefficient's sign is certain (a_exact has sign(a)); the continuous-
        // at-0 max slope otherwise.
        let prop = select(max(abs(lsv), abs(usv)), abs(sel), abs(a) > e_in);
        err_out[idx] = (e_in * prop + e_prod + e_sub + p.additive) * SLACK;
    }
}
"#;

/// #u4 taint twin of [`CROWN_ACTIVATION_RESIDENT_SHADER`]: same arithmetic,
/// same bindings 0-7, plus four additive `u32` taint buffers (8-11). The value
/// and error taints follow the GEMM taint channel's propagation rule, adapted
/// to this op's two multiplications (`a * sel` on the value channel and
/// `e_in * slope-factor` on the error channel):
///
/// ```text
/// slopes_live = lower_slope != 0 OR upper_slope != 0
/// taint_a_out = taint_a_in AND slopes_live
/// taint_e_out = (taint_e_in OR taint_a_in) AND slopes_live
/// ```
///
/// A set coefficient/error word makes the observed sign and sign-selected
/// slope untrustworthy, so a selected zero cannot annihilate while the other
/// slope is nonzero. Only a stable zero map (`ls == us == 0`) clears the word.
/// This keeps dead-ReLU annihilation exact, makes value taint flow into error
/// taint, and closes both the asymmetric-sign case and lane 5's tiny-slope
/// laundering case.
pub(super) const CROWN_ACTIVATION_RESIDENT_TAINT_SHADER: &str = r#"
struct Params { num_specs: u32, num_neurons: u32, is_upper: u32, additive: f32, num_specs_per_dom: u32, eft_mode: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;
@group(0) @binding(2) var<storage, read> err_in: array<f32>;
@group(0) @binding(3) var<storage, read> ls: array<f32>;
@group(0) @binding(4) var<storage, read> us: array<f32>;
@group(0) @binding(5) var<storage, read_write> a_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> err_out: array<f32>;
@group(0) @binding(7) var<storage, read> beta_signed: array<f32>;
@group(0) @binding(8) var<storage, read> taint_a_in: array<u32>;
@group(0) @binding(9) var<storage, read> taint_e_in: array<u32>;
@group(0) @binding(10) var<storage, read_write> taint_a_out: array<u32>;
@group(0) @binding(11) var<storage, read_write> taint_e_out: array<u32>;
const U: f32 = 0.00000005960464477539063; // 2^-24
const SLACK: f32 = 1.000001;
const F32_MIN_NORMAL_ACT: f32 = 1.1754944e-38;
const TWO_PROD_EXACT_FLOOR_F32: f32 = 3.9443045e-31; // 2^-101 == ny_core::eft::TWO_PROD_EXACT_FLOOR_F32
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.num_specs * p.num_neurons;
    if (idx >= total) { return; }
    let i = idx % p.num_neurons;
    let dom = (idx / p.num_neurons) / p.num_specs_per_dom;
    let sbase = dom * p.num_neurons;
    let a = a_in[idx];
    let lsv = ls[sbase + i];
    let usv = us[sbase + i];
    var sel: f32;
    if (p.is_upper == 0u) { sel = select(usv, lsv, a >= 0.0); }
    else { sel = select(lsv, usv, a >= 0.0); }
    let base = a * sel;
    let bv = beta_signed[sbase + i];
    var coeff: f32;
    if (p.is_upper == 0u) { coeff = base - bv; } else { coeff = base + bv; }
    a_out[idx] = coeff;
    let e_in = err_in[idx];
    if (p.eft_mode == 0u) {
        let slope_sum = abs(lsv) + abs(usv);
        let extra = select(0.0, abs(base) * U, bv != 0.0);
        let gap = abs(coeff) * U + extra;
        err_out[idx] = (e_in * slope_sum + gap + p.additive) * SLACK;
    } else {
        var e_prod = abs(fma(a, sel, -base));
        if (a != 0.0 && sel != 0.0 && abs(base) < TWO_PROD_EXACT_FLOOR_F32) { e_prod = F32_MIN_NORMAL_ACT; }
        let sb2 = select(bv, -bv, p.is_upper == 0u);
        let bb = fma(-1.0, base, coeff);
        let sbb = fma(-1.0, bb, coeff);
        let e_sub = abs(fma(-1.0, sbb, base) + fma(-1.0, bb, sb2));
        let prop = select(max(abs(lsv), abs(usv)), abs(sel), abs(a) > e_in);
        err_out[idx] = (e_in * prop + e_prod + e_sub + p.additive) * SLACK;
    }
    // #u4: out-of-band taint transport. A tainted coefficient/error has no
    // authenticated sign, so the observed sign-selected slope cannot justify
    // annihilation. Only BOTH exact-zero slopes define the stable zero map.
    let ta = taint_a_in[idx];
    let te = taint_e_in[idx];
    let slopes_live = lsv != 0.0 || usv != 0.0;
    let ta_kept = select(0u, ta, slopes_live);
    taint_a_out[idx] = ta_kept;
    taint_e_out[idx] = select(0u, te, slopes_live) | ta_kept;
}
"#;

/// Resident activation INTERCEPT -> running bias (reduction per spec row), the
/// piece the host folds during the activation layer. For row s (one side):
///   bias_out[s]     += sum_i a[s,i]*sel_int(i)
///   bias_err_out[s] = add_up(
///       bias_err_out[s],
///       round_up(gamma*sum_i|a[s,i]*sel_int(i)| + sum_i err[s,i]*(|li|+|ui|)),
///       flush,
///   )
/// sel_int = lower: a>=0?lower_int:upper_int ; upper: a>=0?upper_int:lower_int. The
/// `gamma*sum|a*sel_int|` term certifies the f32 reduction rounding (host uses f64).
pub(super) const CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER: &str = r#"
// #batched-bab: `num_specs_per_dom` reuses a padding slot. Intercepts are per-
// domain-block ([n_domains*num_neurons]); spec row `s` uses domain
// `s/num_specs_per_dom`, so its intercepts live at `dom*num_neurons + j`. Single
// domain (num_specs_per_dom == num_specs) → dom 0 → `[j]`, byte-identical.
// #eft-err (former padding _p2 → eft_mode): 1 ⇒ the a·sel_int reduction's
// rounding is charged (core products plus conservative/floored FMA residuals
// through taps, tree, and the final add),
// charged ·r_slack — carried in the OTHERWISE-UNUSED gamma_k field — replacing
// the a-priori γ_k·Σ|a·sel_int|; and the propagated err uses the Lipschitz
// factor of v ↦ v·int(v) (piecewise-linear through 0): |sel_int| when the
// coefficient sign is certain, max(|li|,|ui|) otherwise, instead of |li|+|ui|.
// 0 ⇒ byte-identical legacy.
struct Params { num_specs: u32, num_neurons: u32, is_upper: u32, gamma_k: f32, additive: f32, slack: f32, num_specs_per_dom: u32, eft_mode: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> err: array<f32>;
@group(0) @binding(3) var<storage, read> lower_int: array<f32>;
@group(0) @binding(4) var<storage, read> upper_int: array<f32>;
@group(0) @binding(5) var<storage, read_write> bias_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> bias_err_out: array<f32>;
const F32_MIN_NORMAL: f32 = 1.1754944e-38;   // 2^-126 smallest NORMAL — survives Metal FTZ
const TWO_PROD_EXACT_FLOOR_F32: f32 = 3.9443045e-31; // 2^-101 == ny_core::eft::TWO_PROD_EXACT_FLOOR_F32
var<workgroup> sv: array<f32, 256>;
var<workgroup> sa: array<f32, 256>;
var<workgroup> se: array<f32, 256>;
var<workgroup> sf: array<f32, 256>;   // §0 amplified-flush accumulator
var<workgroup> sr: array<f32, 256>;   // #eft-err measured residuals
fn round_up_pos(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return x; }
    if ((bits & 0x80000000u) != 0u || magnitude == 0u) { return 0.0; }
    if (magnitude < 0x00800000u) { return F32_MIN_NORMAL; }
    return bitcast<f32>(bits + 1u);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let s = wg.x;
    let t = lid.x;
    if (s >= p.num_specs) { return; }
    // #batched-bab domain block (num_specs_per_dom==num_specs → sbase 0, identical).
    let sbase = (s / max(p.num_specs_per_dom, 1u)) * p.num_neurons;
    let eft = p.eft_mode == 1u;
    var v: f32 = 0.0; var av: f32 = 0.0; var ev: f32 = 0.0; var fa: f32 = 1.0;
    var rv: f32 = 0.0;
    var j = t;
    while (j < p.num_neurons) {
        let idx = s * p.num_neurons + j;
        let a_v = a[idx];
        let li = lower_int[sbase + j];
        let ui = upper_int[sbase + j];
        var sel: f32;
        if (p.is_upper == 0u) { sel = select(ui, li, a_v >= 0.0); }
        else { sel = select(li, ui, a_v >= 0.0); }
        if (!eft) {
            v = v + a_v * sel;
            av = av + abs(a_v * sel);
            ev = ev + err[idx] * (abs(li) + abs(ui));
        } else {
            let prod = a_v * sel;
            var ep = abs(fma(a_v, sel, -prod));
            if (a_v != 0.0 && sel != 0.0 && abs(prod) < TWO_PROD_EXACT_FLOOR_F32) { ep = F32_MIN_NORMAL; }
            let s2 = v + prod;
            let bb = fma(-1.0, v, s2);
            let sb = fma(-1.0, bb, s2);
            rv = rv + ep + abs(fma(-1.0, sb, v) + fma(-1.0, bb, prod));
            v = s2;
            // Lipschitz intercept propagation: v ↦ v·int(v) is piecewise-linear
            // through 0, so max(|li|,|ui|) always suffices; |sel| when the
            // coefficient's sign is certain.
            let e_in = err[idx];
            let prop = select(max(abs(li), abs(ui)), abs(sel), abs(a_v) > e_in);
            ev = ev + e_in * prop;
        }
        // §0: a subnormal a_v flushed by Metal FTZ then amplified by |sel| loses up
        // to |sel|·FLT_MIN from `v`; the γ_k·Σ|a·sel| term reads the same flushed a_v
        // as 0 so misses it. max(|a_v|, max(|li|,|ui|), 1) dominates the lost product.
        fa = fa + max(max(abs(a_v), max(abs(li), abs(ui))), 1.0);
        j = j + 256u;
    }
    sv[t] = v; sa[t] = av; se[t] = ev; sf[t] = fa; sr[t] = rv;
    workgroupBarrier();
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (t < stride) {
            let a0 = sv[t];
            let b0 = sv[t + stride];
            let s0 = a0 + b0;
            let bb0 = fma(-1.0, a0, s0);
            let sb0 = fma(-1.0, bb0, s0);
            let r0 = abs(fma(-1.0, sb0, a0) + fma(-1.0, bb0, b0));
            sv[t] = s0;
            sa[t] = sa[t] + sa[t + stride];
            se[t] = se[t] + se[t + stride];
            sf[t] = sf[t] + sf[t + stride];
            sr[t] = sr[t] + sr[t + stride] + r0;
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (t == 0u) {
        let old = bias_out[s];
        let sum = old + sv[0];
        bias_out[s] = sum;
        // Preserve every non-negative certified term. Without directed
        // assembly an existing O(1) radius can swallow the tiny rung-3 floor.
        let flush_scaled = round_up_pos(round_up_pos(sf[0] * p.slack) * F32_MIN_NORMAL);
        let flush = round_up_pos(p.additive + flush_scaled);
        let old_err = bias_err_out[s];
        if (!eft) {
            // Both non-negative reduction lanes need the k-scaled recovery;
            // one final ULP cannot recover trailing terms swallowed by `se`.
            let reduced_err = round_up_pos(p.gamma_k * sa[0] + se[0]);
            let local_err = round_up_pos(reduced_err * p.slack);
            bias_err_out[s] = round_up_pos(round_up_pos(old_err + local_err) + flush);
        } else {
            // In EFT mode `gamma_k` carries r_slack (the γ term is unused).
            let bbf = fma(-1.0, old, sum);
            let sbf = fma(-1.0, bbf, sum);
            let rf = abs(fma(-1.0, sbf, old) + fma(-1.0, bbf, sv[0]));
            let residual_err = round_up_pos((sr[0] + rf) * p.gamma_k);
            let propagated_err = round_up_pos(se[0] * p.slack);
            let local_err = round_up_pos(residual_err + propagated_err);
            bias_err_out[s] = round_up_pos(round_up_pos(old_err + local_err) + flush);
        }
    }
}
"#;

/// Resident Conv2d coefficient-error over-bound (the per-row broadcast the host
/// uses, NOT a second conv pass). One workgroup per spec row reduces
/// rowmax|a| and rowmax|err| over the OC·OH·OW inputs, then broadcasts
///   err_out[s,·] = round_up( gamma·rowmax|a[s]|·kernel_l1 + rowmax|err[s]|·kernel_l1 )
/// to every one of the IC·IH·IW outputs. `gamma = γ_{OC·KH·KW}` and `kernel_l1 =
/// Σ|weight_col|` are host scalars. Sound (over-bounds the true conv-transpose
/// coefficient error), ULP-scale looser than a per-element pass.
pub(super) const CROWN_CONV_ERROR_ROWMAX_SHADER: &str = r#"
struct Params { num_specs: u32, out_dim: u32, new_dim: u32, _p0: u32, gamma: f32, kernel_l1: f32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> err: array<f32>;
@group(0) @binding(3) var<storage, read_write> err_out: array<f32>;
var<workgroup> rma: array<f32, 256>;
var<workgroup> rme: array<f32, 256>;
const F32_MIN_NORMAL: f32 = 1.1754944e-38;
fn round_up_pos(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return x; }
    if ((bits & 0x80000000u) != 0u || magnitude == 0u) { return 0.0; }
    if (magnitude < 0x00800000u) { return F32_MIN_NORMAL; }
    return bitcast<f32>(bits + 1u);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let s = wg.x;
    let t = lid.x;
    if (s >= p.num_specs) { return; }
    var ma: f32 = 0.0; var me: f32 = 0.0;
    var j = t;
    while (j < p.out_dim) {
        ma = max(ma, abs(a[s * p.out_dim + j]));
        me = max(me, abs(err[s * p.out_dim + j]));
        j = j + 256u;
    }
    rma[t] = ma; rme[t] = me;
    workgroupBarrier();
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (t < stride) {
            rma[t] = max(rma[t], rma[t + stride]);
            rme[t] = max(rme[t], rme[t + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let val = round_up_pos(p.gamma * rma[0] * p.kernel_l1 + rme[0] * p.kernel_l1);
    var q = t;
    while (q < p.new_dim) {
        err_out[s * p.new_dim + q] = val;
        q = q + 256u;
    }
}
"#;

/// Sound CROWN concretize: like `CROWN_CONCRETIZE_SHADER` but widens each bound
/// by a certified f32 rounding/coefficient-error penalty so the result is a SOUND
/// enclosure even though the dot product runs in round-to-nearest f32.
///
/// For each spec row the lower bound is `Σ_j (a_l[j]>=0 ? a_l·x_l : a_l·x_u)`,
/// computed in f32, then widened DOWN by
///   `penalty_l = Σ_j (err_l[j] + γ_n·|a_l[j]|)·max(|x_l[j]|,|x_u[j]|) + additive`,
/// where `err_l[j]` is the accumulated coefficient error fed in from the backward
/// pass and `γ_n = n·u/(1−n·u)` (`u=2⁻²⁴`) bounds the concretize's own dot
/// rounding (`additive` covers subnormal underflow). The upper bound is widened
/// UP symmetrically. This is the on-device form of the CPU `γ_n·S` certified
/// error — the piece that lets the GPU CROWN backward decide a verdict soundly.
pub(super) const CROWN_CONCRETIZE_SOUND_SHADER: &str = r#"
struct Params {
    num_specs: u32,
    input_dim: u32,
    gamma_n: f32,
    additive: f32,
    slack: f32,       // §0 amplified-flush combine slack (>= 1)
    // #batched-bab: per-domain spec-row count (reuses a padding slot). Each domain
    // has its OWN input box ([n_domains*input_dim]); spec row `s` concretizes
    // against `dom = s/num_specs_per_dom`. Single domain (== num_specs) → dom 0 →
    // `[j]`, byte-identical. Concretizing against the WRONG box = false-VERIFIED.
    num_specs_per_dom: u32,
    // #eft-err (former padding): 1 ⇒ the concretize dot runs the barrier-fma EFT
    // sequence and charges its MEASURED residual sum (·eft_r_slack) instead of
    // the a-priori γ_n·|a| term. 0 (every legacy writer) ⇒ byte-identical.
    eft_mode: u32,
    eft_r_slack: f32,
}

// Storage buffers are limited to 8 per compute stage on Metal's default limits,
// so the lower/upper bias and lower/upper coefficient-error matrices are each
// packed into a single buffer (upper half offset by num_specs / by coeff count).
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_lower: array<f32>;
@group(0) @binding(2) var<storage, read> a_upper: array<f32>;
@group(0) @binding(3) var<storage, read> input_lower: array<f32>;
@group(0) @binding(4) var<storage, read> input_upper: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;       // [lower | upper]
@group(0) @binding(6) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(7) var<storage, read_write> output_upper: array<f32>;
@group(0) @binding(8) var<storage, read> a_err: array<f32>;      // [lower_err | upper_err]

var<workgroup> sh_lb: array<f32, 256>;
var<workgroup> sh_ub: array<f32, 256>;
var<workgroup> sh_pl: array<f32, 256>;
var<workgroup> sh_pu: array<f32, 256>;
var<workgroup> sh_fa: array<f32, 256>;   // §0 amplified-flush accumulator (reduced)
var<workgroup> sh_rl: array<f32, 256>;   // #eft-err measured-residual lanes (0 in legacy mode)
var<workgroup> sh_ru: array<f32, 256>;

const FALLBACK_BOUND: f32 = 1e10;
const F32_MIN_NORMAL: f32 = 1.1754944e-38;   // 2^-126 smallest NORMAL — survives Metal FTZ
const TWO_PROD_EXACT_FLOOR_F32: f32 = 3.9443045e-31; // 2^-101 == ny_core::eft::TWO_PROD_EXACT_FLOOR_F32

fn is_non_finite(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

// Outward helpers for the verdict-deciding final assembly.  The normal floor is
// intentional: a subnormal one-ULP step can be flushed back to zero on Metal.
fn round_up_pos(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return x; }
    if ((bits & 0x80000000u) != 0u || magnitude == 0u) { return 0.0; }
    if (magnitude < 0x00800000u) { return F32_MIN_NORMAL; }
    return bitcast<f32>(bits + 1u);
}

fn next_down_f32_normal(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let magnitude = bits & 0x7fffffffu;
    let negative = (bits & 0x80000000u) != 0u;
    if (magnitude >= 0x7f800000u) { return x; }
    if (magnitude == 0u) { return -F32_MIN_NORMAL; }
    // Toward -Inf: zero is a valid outward replacement for a positive
    // subnormal; a negative subnormal needs the negative normal floor.
    if (magnitude < 0x00800000u) {
        return select(0.0, -F32_MIN_NORMAL, negative);
    }
    let y_bits = select(bits - 1u, bits + 1u, negative);
    if ((y_bits & 0x7fffffffu) < 0x00800000u) {
        return select(0.0, -F32_MIN_NORMAL, negative);
    }
    return bitcast<f32>(y_bits);
}

fn next_up_f32_normal(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let magnitude = bits & 0x7fffffffu;
    let negative = (bits & 0x80000000u) != 0u;
    if (magnitude >= 0x7f800000u) { return x; }
    if (magnitude == 0u) { return F32_MIN_NORMAL; }
    // Toward +Inf: zero is a valid outward replacement for a negative
    // subnormal; a positive subnormal needs the positive normal floor.
    if (magnitude < 0x00800000u) {
        return select(F32_MIN_NORMAL, bitcast<f32>(0x80000000u), negative);
    }
    let y_bits = select(bits + 1u, bits - 1u, negative);
    if ((y_bits & 0x7fffffffu) < 0x00800000u) {
        return select(F32_MIN_NORMAL, bitcast<f32>(0x80000000u), negative);
    }
    return bitcast<f32>(y_bits);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let spec_row = wg_id.x;
    let local_id = lid.x;
    if (spec_row >= params.num_specs) { return; }

    var local_lb: f32 = 0.0;
    var local_ub: f32 = 0.0;
    var pen_l: f32 = 0.0;
    var pen_u: f32 = 0.0;
    var r_l: f32 = 0.0;        // #eft-err measured residuals (lower dot)
    var r_u: f32 = 0.0;
    var flushacc: f32 = 1.0;   // §0 amplified-flush base (per-thread; reduced below)
    var degraded: bool = false;

    let coeff = params.num_specs * params.input_dim;
    var j = local_id;
    while (j < params.input_dim) {
        let idx = spec_row * params.input_dim + j;
        // #batched-bab domain block (num_specs_per_dom==num_specs → dbase 0, identical).
        let dbase = (spec_row / max(params.num_specs_per_dom, 1u)) * params.input_dim;
        let a_l = a_lower[idx];
        let a_u = a_upper[idx];
        let e_l = a_err[idx];
        let e_u = a_err[coeff + idx];
        let x_l = input_lower[dbase + j];
        let x_u = input_upper[dbase + j];

        if (a_l != a_l || abs(a_l) >= FALLBACK_BOUND || a_u != a_u || abs(a_u) >= FALLBACK_BOUND) {
            degraded = true;
        } else {
            let a_l_pos = max(a_l, 0.0);
            let a_l_neg = min(a_l, 0.0);
            let a_u_pos = max(a_u, 0.0);
            let a_u_neg = min(a_u, 0.0);
            let xmax = max(abs(x_l), abs(x_u));
            if (params.eft_mode == 0u) {
                local_lb = local_lb + a_l_pos * x_l + a_l_neg * x_u;
                local_ub = local_ub + a_u_pos * x_u + a_u_neg * x_l;

                // |x| upper, and certified penalty per coefficient: accumulated error
                // plus the concretize dot's own f32 rounding bound γ_n·|a|.
                pen_l = pen_l + (e_l + params.gamma_n * abs(a_l)) * xmax;
                pen_u = pen_u + (e_u + params.gamma_n * abs(a_u)) * xmax;
            } else {
                // #eft-err: qualified core value sequence with measured or
                // conservatively charged FMA residuals
                // (core RN product + FMA residual, conservative/floored under
                // measured FMA-operand DAZ; add via plain RN + fma-barrier
                // TwoSum residual). The measured/conservative residual sum
                // replaces the a-priori γ_n·|a| charge; the propagated coefficient
                // error e·xmax stays. Sound for THIS executed sequence — the final
                // penalty applies eft_r_slack (covers the residual lanes' own f32
                // accumulation + final-assembly ops, host-computed outward).
                var p = a_l_pos * x_l;
                var ep = abs(fma(a_l_pos, x_l, -p));
                if (a_l_pos != 0.0 && x_l != 0.0 && abs(p) < TWO_PROD_EXACT_FLOOR_F32) { ep = F32_MIN_NORMAL; }
                var s = local_lb + p;
                var bb = fma(-1.0, local_lb, s);
                var sb = fma(-1.0, bb, s);
                r_l = r_l + ep + abs(fma(-1.0, sb, local_lb) + fma(-1.0, bb, p));
                local_lb = s;
                p = a_l_neg * x_u;
                ep = abs(fma(a_l_neg, x_u, -p));
                if (a_l_neg != 0.0 && x_u != 0.0 && abs(p) < TWO_PROD_EXACT_FLOOR_F32) { ep = F32_MIN_NORMAL; }
                s = local_lb + p;
                bb = fma(-1.0, local_lb, s);
                sb = fma(-1.0, bb, s);
                r_l = r_l + ep + abs(fma(-1.0, sb, local_lb) + fma(-1.0, bb, p));
                local_lb = s;

                p = a_u_pos * x_u;
                ep = abs(fma(a_u_pos, x_u, -p));
                if (a_u_pos != 0.0 && x_u != 0.0 && abs(p) < TWO_PROD_EXACT_FLOOR_F32) { ep = F32_MIN_NORMAL; }
                s = local_ub + p;
                bb = fma(-1.0, local_ub, s);
                sb = fma(-1.0, bb, s);
                r_u = r_u + ep + abs(fma(-1.0, sb, local_ub) + fma(-1.0, bb, p));
                local_ub = s;
                p = a_u_neg * x_l;
                ep = abs(fma(a_u_neg, x_l, -p));
                if (a_u_neg != 0.0 && x_l != 0.0 && abs(p) < TWO_PROD_EXACT_FLOOR_F32) { ep = F32_MIN_NORMAL; }
                s = local_ub + p;
                bb = fma(-1.0, local_ub, s);
                sb = fma(-1.0, bb, s);
                r_u = r_u + ep + abs(fma(-1.0, sb, local_ub) + fma(-1.0, bb, p));
                local_ub = s;

                pen_l = pen_l + e_l * xmax;
                pen_u = pen_u + e_u * xmax;
            }

            // §0 amplified operand-flush: a subnormal a_l/a_u flushed to 0 by Metal
            // FTZ then amplified by |x| (or a subnormal x flushed then amplified by
            // |a|) loses up to max(|a|,|x|)·FLT_MIN — NOT covered by `pen` (which also
            // reads the flushed |a| as 0) nor a weight-independent floor. Accumulate
            // max(|a|,|x|,1) per tap; whichever operand FTZ zeroed, the survivor
            // dominates the lost product. The live gradual-underflow gate, not a
            // backend-name assumption, decides whether core subnormals are preserved.
            flushacc = flushacc + max(max(max(abs(a_l), abs(a_u)), xmax), 1.0);
        }
        j = j + 256u;
    }

    if (degraded) {
        local_lb = bitcast<f32>(0xFF800000u); // -Inf
        local_ub = bitcast<f32>(0x7F800000u); // +Inf
    }

    sh_lb[local_id] = local_lb;
    sh_ub[local_id] = local_ub;
    sh_pl[local_id] = pen_l;
    sh_pu[local_id] = pen_u;
    sh_fa[local_id] = flushacc;
    sh_rl[local_id] = r_l;
    sh_ru[local_id] = r_u;
    workgroupBarrier();

    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_id < stride) {
            // #eft-err: the value adds keep the EXACT legacy expression (mode-0
            // byte-identity); their fma-barrier TwoSum residuals are captured on
            // the side into the r lanes, which mode 0 multiplies by 0 at the end.
            let al0 = sh_lb[local_id];
            let bl0 = sh_lb[local_id + stride];
            let sl0 = al0 + bl0;
            let blb = fma(-1.0, al0, sl0);
            let slb = fma(-1.0, blb, sl0);
            let rl_add = abs(fma(-1.0, slb, al0) + fma(-1.0, blb, bl0));
            let au0 = sh_ub[local_id];
            let bu0 = sh_ub[local_id + stride];
            let su0 = au0 + bu0;
            let bub = fma(-1.0, au0, su0);
            let sbu = fma(-1.0, bub, su0);
            let ru_add = abs(fma(-1.0, sbu, au0) + fma(-1.0, bub, bu0));
            sh_lb[local_id] = sl0;
            sh_ub[local_id] = su0;
            sh_pl[local_id] = sh_pl[local_id] + sh_pl[local_id + stride];
            sh_pu[local_id] = sh_pu[local_id] + sh_pu[local_id + stride];
            sh_fa[local_id] = sh_fa[local_id] + sh_fa[local_id + stride];
            sh_rl[local_id] = sh_rl[local_id] + sh_rl[local_id + stride] + rl_add;
            sh_ru[local_id] = sh_ru[local_id] + sh_ru[local_id + stride] + ru_add;
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (local_id == 0u) {
        // Widen OUTWARD by the penalty + amplified underflow floor. The host has
        // already proved the complete exact affine radius is strictly below
        // FALLBACK_BOUND, which is the necessary precondition making the
        // non-finite repair to ±FALLBACK_BOUND an enclosure.
        // `flush` = weight-independent floor + §0 amplified operand-flush term.
        // #eft-err: `rs` = 0 in legacy mode (adding literal 0.0 to the non-negative
        // penalty is value-identical), else the measured-residual slack.
        let rs = select(0.0, params.eft_r_slack, params.eft_mode == 1u);
        let flush_scaled = round_up_pos(round_up_pos(sh_fa[0] * params.slack) * F32_MIN_NORMAL);
        let flush = round_up_pos(params.additive + flush_scaled);
        // #concretize-assembly-round: assemble every verdict-facing operation in
        // its safe direction.  In particular, the dominant penalty itself must
        // participate in the final-subtraction rounding charge: charging only
        // |endpoint|+|bias| is unsound when a small endpoint subtracts a large
        // propagated-error penalty.
        //
        // `params.slack` recovers positive accumulation/multiply under-reporting
        // in sh_pl/sh_pu (and is explicitly part of the EFT prop-error contract).
        // The residual lane has its independent `rs` recovery factor.
        let prop_l = round_up_pos(sh_pl[0] * params.slack);
        let prop_u = round_up_pos(sh_pu[0] * params.slack);
        let resid_l = round_up_pos(sh_rl[0] * rs);
        let resid_u = round_up_pos(sh_ru[0] * rs);
        let pen_l = round_up_pos(round_up_pos(prop_l + resid_l) + flush);
        let pen_u = round_up_pos(round_up_pos(prop_u + resid_u) + flush);
        let cl = next_down_f32_normal(sh_lb[0] + bias[spec_row]);
        let cu = next_up_f32_normal(sh_ub[0] + bias[params.num_specs + spec_row]);
        var lb = next_down_f32_normal(cl - pen_l);
        var ub = next_up_f32_normal(cu + pen_u);
        if (is_non_finite(lb)) { lb = -FALLBACK_BOUND; }
        if (is_non_finite(ub)) { ub = FALLBACK_BOUND; }
        if (lb > ub) {
            output_lower[spec_row] = -FALLBACK_BOUND;
            output_upper[spec_row] = FALLBACK_BOUND;
        } else {
            output_lower[spec_row] = lb;
            output_upper[spec_row] = ub;
        }
    }
}
"#;

/// WGSL shader for Conv2d A-matrix reshape (#3397).
///
/// Transforms the CROWN A-matrix from conv output layout to GEMM input layout:
///   Source: A[s, oc * spatial + pos] — shape (num_specs, out_c * oh * ow)
///   Dest:   A_reshaped[s * spatial + pos, oc] — shape (num_specs * oh * ow, out_c)
///
/// Each thread handles one element of the output.
/// Reference: designs/2026-03-06-conv-crown-backward-gemm.md (Step 1)
pub(super) const CONV_RESHAPE_SHADER: &str = r#"
struct Params {
    num_specs: u32,
    out_channels: u32,
    spatial: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.num_specs * params.spatial * params.out_channels;
    if (idx >= total) {
        return;
    }

    // Output layout: (S * spatial + pos, out_channels)
    // idx = flat_row * out_channels + oc
    let flat_row = idx / params.out_channels;
    let oc = idx % params.out_channels;
    let s = flat_row / params.spatial;
    let pos = flat_row % params.spatial;

    // Source layout: (S, out_channels * spatial)
    // src_idx = s * (out_channels * spatial) + oc * spatial + pos
    let src_idx = s * (params.out_channels * params.spatial) + oc * params.spatial + pos;
    dst[idx] = src[src_idx];
}
"#;

/// WGSL shader for Conv2d col2im gather (#3397).
///
/// Gathers from GEMM output (S*OH*OW, kernel_cols) into input-space A-matrix
/// (S, IC*IH*IW) using the inverse convolution mapping.
///
/// Each thread computes one output element by gathering contributions from all
/// kernel positions that map to this input position. For a 3×3 kernel, the
/// inner loop has at most 9 iterations.
///
/// Reference: designs/2026-03-06-conv-crown-backward-gemm.md (Step 3/col2im)
pub(super) const CONV_COL2IM_SHADER: &str = r#"
struct Params {
    num_specs: u32,
    flat_input_dim: u32,
    out_h: u32,
    out_w: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    kernel_cols: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gemm_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    let total = params.num_specs * params.flat_input_dim;
    if (thread_id >= total) {
        return;
    }

    let s = thread_id / params.flat_input_dim;
    let flat_idx = thread_id % params.flat_input_dim;

    // Decode flat_idx → (ic, ih, iw)
    let in_hw = params.in_h * params.in_w;
    let ic = flat_idx / in_hw;
    let rem = flat_idx % in_hw;
    let ih = rem / params.in_w;
    let iw = rem % params.in_w;

    var sum: f32 = 0.0;
    let spatial = params.out_h * params.out_w;

    // Gather from all kernel positions that contribute to (ih, iw)
    for (var ki: u32 = 0u; ki < params.kernel_h; ki = ki + 1u) {
        // gy = (ih + pad_h - ki) / stride_h, must be non-negative integer
        let ih_plus_ph = ih + params.pad_h;
        if (ih_plus_ph < ki) {
            continue;
        }
        let numerator_h = ih_plus_ph - ki;
        if (numerator_h % params.stride_h != 0u) {
            continue;
        }
        let gy = numerator_h / params.stride_h;
        if (gy >= params.out_h) {
            continue;
        }

        for (var kj: u32 = 0u; kj < params.kernel_w; kj = kj + 1u) {
            let iw_plus_pw = iw + params.pad_w;
            if (iw_plus_pw < kj) {
                continue;
            }
            let numerator_w = iw_plus_pw - kj;
            if (numerator_w % params.stride_w != 0u) {
                continue;
            }
            let gx = numerator_w / params.stride_w;
            if (gx >= params.out_w) {
                continue;
            }

            let gemm_row = s * spatial + gy * params.out_w + gx;
            let gemm_col = ic * params.kernel_h * params.kernel_w + ki * params.kernel_w + kj;
            sum = sum + gemm_out[gemm_row * params.kernel_cols + gemm_col];
        }
    }

    dst[thread_id] = sum;
}
"#;

/// #seg-resident: on-device twin of the CPU `merge_streams`/`add_skip_stream`
/// lane-pair merge. VALUE lane: `s = a + b` — an f32 RN add IS the correctly
/// rounded f64 sum of two f32s, so this is BIT-IDENTICAL to the CPU's
/// f64-sum-then-round. ERR lane: the CPU computes
/// `up_f32(err_a + err_b + |s|·u)` in f64; the f32 evaluation under-reports by
/// ≤ γ₃, covered OUTWARD by `slack ≥ (1+u)⁴` plus the final `round_up_pos`.
/// Dispatched once per lane pair (lower/upper coeff, and the bias pairs for
/// projection merges).
pub(super) const RESIDENT_SEG_MERGE_SHADER: &str = r#"
struct Params { n: u32, slack: f32, stride: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> err_a: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read> err_b: array<f32>;
const U: f32 = 0.00000005960464477539063; // 2^-24
const F32_MIN_NORMAL: f32 = 1.1754944e-38;
fn round_up_pos(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return x; }
    if ((bits & 0x80000000u) != 0u || magnitude == 0u) { return 0.0; }
    if (magnitude < 0x00800000u) { return F32_MIN_NORMAL; }
    return bitcast<f32>(bits + 1u);
}
// Grid-stride over n (stride = total dispatched threads): a wide frontier
// (num_specs*dim > 65535*256) exceeds the per-dim workgroup limit, so the
// dispatch is capped and each thread walks its stride class. Bit-identical
// per element regardless of the grid shape (each i is touched exactly once).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var i = gid.x;
    loop {
        if (i >= p.n) { break; }
        let s = a[i] + b[i];
        err_a[i] = round_up_pos(((err_a[i] + err_b[i]) + abs(s) * U) * p.slack);
        a[i] = s;
        i += p.stride;
    }
}
"#;

/// #eft-err conv col2im twin: the EFT residual channel through the Conv2d
/// col2im gather. Identical index mapping to [`CONV_COL2IM_SHADER`], but
/// gathers BOTH the EFT twin-GEMM value and residual streams: the value is
/// re-accumulated with the fma-barrier TwoSum (exact per-add residuals), and
/// the residual output is `Σ r_gemm + Σ|e_sum|` — so
/// `|exact_conv_coeff − v_dst| ≤ r_dst · r_slack` telescopes through the whole
/// reshape→GEMM→col2im chain (the reshape moves data without arithmetic).
/// Host `eft_r_slack_f32(oc·kh·kw)` covers all f32 self-accumulation.
pub(super) const CONV_COL2IM_EFT_TWIN_SHADER: &str = r#"
struct Params {
    num_specs: u32,
    flat_input_dim: u32,
    out_h: u32,
    out_w: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    kernel_cols: u32,
    _padding: vec2<u32>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> v_gemm: array<f32>;
@group(0) @binding(2) var<storage, read> r_gemm: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_dst: array<f32>;
@group(0) @binding(4) var<storage, read_write> r_dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    let total = params.num_specs * params.flat_input_dim;
    if (thread_id >= total) { return; }
    let s = thread_id / params.flat_input_dim;
    let flat_idx = thread_id % params.flat_input_dim;
    let in_hw = params.in_h * params.in_w;
    let ic = flat_idx / in_hw;
    let rem = flat_idx % in_hw;
    let ih = rem / params.in_w;
    let iw = rem % params.in_w;
    var acc: f32 = 0.0;
    var rsum: f32 = 0.0;
    let spatial = params.out_h * params.out_w;
    for (var ki: u32 = 0u; ki < params.kernel_h; ki = ki + 1u) {
        let ih_plus_ph = ih + params.pad_h;
        if (ih_plus_ph < ki) { continue; }
        let numerator_h = ih_plus_ph - ki;
        if (numerator_h % params.stride_h != 0u) { continue; }
        let gy = numerator_h / params.stride_h;
        if (gy >= params.out_h) { continue; }
        for (var kj: u32 = 0u; kj < params.kernel_w; kj = kj + 1u) {
            let iw_plus_pw = iw + params.pad_w;
            if (iw_plus_pw < kj) { continue; }
            let numerator_w = iw_plus_pw - kj;
            if (numerator_w % params.stride_w != 0u) { continue; }
            let gx = numerator_w / params.stride_w;
            if (gx >= params.out_w) { continue; }
            let gemm_row = s * spatial + gy * params.out_w + gx;
            let gemm_col = ic * params.kernel_h * params.kernel_w + ki * params.kernel_w + kj;
            let src = gemm_row * params.kernel_cols + gemm_col;
            let v = v_gemm[src];
            // fma-barrier TwoSum: acc + v with the exact add residual.
            let s2 = acc + v;
            let bb = fma(-1.0, acc, s2);
            let sb = fma(-1.0, bb, s2);
            let da = fma(-1.0, sb, acc);
            let db = fma(-1.0, bb, v);
            let es = da + db;
            rsum = rsum + r_gemm[src] + abs(es);
            acc = s2;
        }
    }
    v_dst[thread_id] = acc;
    r_dst[thread_id] = rsum;
}
"#;

/// WGSL shader for resident Conv2d IBP forward (#4275).
///
/// Each thread computes one output element in flattened NCHW order:
/// `(batch, out_channel, out_h, out_w)`.
///
/// The shader gathers the receptive field in registers and accumulates
/// `W+ @ x_lower + W- @ x_upper` for the lower bound and
/// `W+ @ x_upper + W- @ x_lower` for the upper bound.
pub(super) const CONV2D_IBP_IM2COL_SHADER: &str = r#"
struct Params {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    input_h: u32,
    input_w: u32,
    out_h: u32,
    out_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    groups: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_lower: array<f32>;
@group(0) @binding(2) var<storage, read> input_upper: array<f32>;
@group(0) @binding(3) var<storage, read> weight_pos: array<f32>;
@group(0) @binding(4) var<storage, read> weight_neg: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(7) var<storage, read_write> output_upper: array<f32>;

const FALLBACK_BOUND: f32 = 1e10;  // Must match crate::FALLBACK_BOUND (#2258)

fn is_non_finite(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7f800000u) == 0x7f800000u;
}

fn nan_safe_lower(x: f32) -> f32 {
    if (is_non_finite(x)) {
        return -FALLBACK_BOUND;
    }
    return x;
}

fn nan_safe_upper(x: f32) -> f32 {
    if (is_non_finite(x)) {
        return FALLBACK_BOUND;
    }
    return x;
}

fn linear_term_lower(wp: f32, xl: f32, wn: f32, xu: f32) -> f32 {
    let p1 = wp * xl;
    let p2 = wn * xu;
    if (is_non_finite(p1) || is_non_finite(p2)) {
        return -FALLBACK_BOUND;
    }
    return p1 + p2;
}

fn linear_term_upper(wp: f32, xu: f32, wn: f32, xl: f32) -> f32 {
    let p1 = wp * xu;
    let p2 = wn * xl;
    if (is_non_finite(p1) || is_non_finite(p2)) {
        return FALLBACK_BOUND;
    }
    return p1 + p2;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let out_spatial = params.out_h * params.out_w;
    let batch_stride = params.out_channels * out_spatial;
    let total_outputs = params.batch_size * batch_stride;

    if (idx >= total_outputs) {
        return;
    }
    if (params.groups != 1u) {
        output_lower[idx] = -FALLBACK_BOUND;
        output_upper[idx] = FALLBACK_BOUND;
        return;
    }

    let batch_idx = idx / batch_stride;
    let batch_offset = idx % batch_stride;
    let out_channel = batch_offset / out_spatial;
    let spatial_offset = batch_offset % out_spatial;
    let oh = spatial_offset / params.out_w;
    let ow = spatial_offset % params.out_w;

    var low: f32 = 0.0;
    var high: f32 = 0.0;

    for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
        for (var kh: u32 = 0u; kh < params.kernel_h; kh = kh + 1u) {
            let ih = i32(oh * params.stride_h + kh) - i32(params.pad_h);
            if (ih < 0 || ih >= i32(params.input_h)) {
                continue;
            }

            for (var kw: u32 = 0u; kw < params.kernel_w; kw = kw + 1u) {
                let iw = i32(ow * params.stride_w + kw) - i32(params.pad_w);
                if (iw < 0 || iw >= i32(params.input_w)) {
                    continue;
                }

                let input_offset =
                    (((batch_idx * params.in_channels + ic) * params.input_h + u32(ih)) * params.input_w)
                    + u32(iw);
                let weight_offset =
                    ((((out_channel * params.in_channels) + ic) * params.kernel_h + kh) * params.kernel_w)
                    + kw;

                let xl = input_lower[input_offset];
                let xu = input_upper[input_offset];
                let wp = weight_pos[weight_offset];
                let wn = weight_neg[weight_offset];

                low = nan_safe_lower(low + linear_term_lower(wp, xl, wn, xu));
                high = nan_safe_upper(high + linear_term_upper(wp, xu, wn, xl));
            }
        }
    }

    low = nan_safe_lower(low + bias[out_channel]);
    high = nan_safe_upper(high + bias[out_channel]);

    if (low > high) {
        low = -FALLBACK_BOUND;
        high = FALLBACK_BOUND;
    }

    output_lower[idx] = low;
    output_upper[idx] = high;
}
"#;

/// WGSL shader for in-place ReLU IBP: max(x, 0) applied to both bounds.
///
/// This is an elementwise kernel that modifies lower and upper buffers in-place.
/// Used by the resident `GpuIbpForward` to chain ReLU after Linear without a
/// host roundtrip.
///
/// Reference: designs/2026-03-18-issue-4081-gpu-ibp-forward-gap2-addendum.md §7
/// Part of #4081.
pub(super) const RELU_IBP_SHADER: &str = r#"
struct Params {
    num_elements: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> lower: array<f32>;
@group(0) @binding(2) var<storage, read_write> upper: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.num_elements) {
        return;
    }
    lower[idx] = max(lower[idx], 0.0);
    upper[idx] = max(upper[idx], 0.0);
}
"#;

/// WGSL shader for element-wise Add IBP (residual connections, #4319).
///
/// For interval arithmetic: `[a_l, a_u] + [b_l, b_u] = [a_l + b_l, a_u + b_u]`
///
/// Reads from two separate input buffer pairs and writes to a destination pair.
///
/// FAST PATH ONLY — not verdict-legal: it magnitude-clamps finite endpoints to
/// ±FALLBACK_BOUND (an unsound tightening) and applies no rounding widen. The
/// verdict path uses [`ADD_IBP_SOUND_BODY`] via `add_ibp_sound_source`.
pub(super) const ADD_IBP_SHADER: &str = r#"
struct Params {
    num_elements: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_a_lower: array<f32>;
@group(0) @binding(2) var<storage, read> input_a_upper: array<f32>;
@group(0) @binding(3) var<storage, read> input_b_lower: array<f32>;
@group(0) @binding(4) var<storage, read> input_b_upper: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_upper: array<f32>;

const FALLBACK_BOUND: f32 = 1e10;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.num_elements) {
        return;
    }
    var low = input_a_lower[idx] + input_b_lower[idx];
    var high = input_a_upper[idx] + input_b_upper[idx];

    // FAST-PATH-ONLY NaN/Inf defense: this clamps FINITE endpoints beyond
    // ±FALLBACK_BOUND, which narrows a valid interval and is therefore UNSOUND.
    // This shader must never feed a verdict-legal consumer; the verdict path
    // uses ADD_IBP_SOUND_BODY, whose repair is non-finite-only. Do not copy
    // this clamp into a sound shader.
    if (low != low || low < -FALLBACK_BOUND) {
        low = -FALLBACK_BOUND;
    }
    if (high != high || high > FALLBACK_BOUND) {
        high = FALLBACK_BOUND;
    }
    if (low > high) {
        low = -FALLBACK_BOUND;
        high = FALLBACK_BOUND;
    }

    output_lower[idx] = low;
    output_upper[idx] = high;
}
"#;

/// Average pool IBP shader (#4320).
///
/// Each thread computes one output element (indexed by global_invocation_id).
/// IBP is exact because average pooling is linear:
///   out_lower = avg_pool(in_lower), out_upper = avg_pool(in_upper)
pub(super) const AVGPOOL_IBP_SHADER: &str = r#"
struct Params {
    num_elements: u32,
    channels: u32,
    input_h: u32,
    input_w: u32,
    output_h: u32,
    output_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    count_include_pad: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_lower: array<f32>;
@group(0) @binding(2) var<storage, read> input_upper: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_lower: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_upper: array<f32>;

const FALLBACK_BOUND: f32 = 1e10;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.num_elements) {
        return;
    }

    let out_hw = params.output_h * params.output_w;
    let c = idx / out_hw;
    let rem = idx % out_hw;
    let oh = rem / params.output_w;
    let ow = rem % params.output_w;

    let in_hw = params.input_h * params.input_w;

    let ih_start = oh * params.stride_h;
    let iw_start = ow * params.stride_w;

    var sum_lower: f32 = 0.0;
    var sum_upper: f32 = 0.0;
    var count: u32 = 0u;

    for (var kh: u32 = 0u; kh < params.kernel_h; kh = kh + 1u) {
        for (var kw: u32 = 0u; kw < params.kernel_w; kw = kw + 1u) {
            let ih_raw = i32(ih_start + kh) - i32(params.pad_h);
            let iw_raw = i32(iw_start + kw) - i32(params.pad_w);

            if (ih_raw >= 0 && u32(ih_raw) < params.input_h &&
                iw_raw >= 0 && u32(iw_raw) < params.input_w) {
                let flat = c * in_hw + u32(ih_raw) * params.input_w + u32(iw_raw);
                sum_lower = sum_lower + input_lower[flat];
                sum_upper = sum_upper + input_upper[flat];
                count = count + 1u;
            } else if (params.count_include_pad != 0u) {
                count = count + 1u;
            }
        }
    }

    var divisor: f32;
    if (params.count_include_pad != 0u) {
        divisor = f32(params.kernel_h * params.kernel_w);
    } else {
        divisor = f32(max(count, 1u));
    }

    var low = sum_lower / divisor;
    var high = sum_upper / divisor;

    if (low != low || low < -FALLBACK_BOUND) {
        low = -FALLBACK_BOUND;
    }
    if (high != high || high > FALLBACK_BOUND) {
        high = FALLBACK_BOUND;
    }
    if (low > high) {
        low = -FALLBACK_BOUND;
        high = FALLBACK_BOUND;
    }

    output_lower[idx] = low;
    output_upper[idx] = high;
}
"#;

// ============================================================================
// TRUE joint α-gradient — ON-DEVICE adjoint (docs/BATCHED_BAB_JOINT_ALPHA_GRADIENT.md §3)
//
// These shaders implement the coefficient-channel forward fold + reverse-mode
// adjoint of `ny_core::joint_alpha_grad` (the FD-proven CPU oracle) entirely on
// device, so the correct joint gradient no longer pays the per-domain CPU re-fold.
// All are NON-soundness-critical: the gradient only proposes the next α∈[0,1];
// the verdict bound is always the sound fold. Each mirrors the CPU element loop
// exactly (one thread per output element), so it matches the CPU oracle to f32
// reduction order. The forward pass tracks ONLY the lower coefficient A (no bias
// accumulator, no certified error — the adjoint gradient needs neither).
// ============================================================================

/// ξ seed (design doc §2 terminal): `abar[s,j] = (A⁰[s,j] ≥ 0 ? in_lo[j] : in_hi[j])`.
/// One thread per (s,j) element of the folded input-level coefficient A⁰.
pub(super) const JOINT_XI_SEED_SHADER: &str = r#"
struct P { num_specs: u32, input_dim: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a0: array<f32>;
@group(0) @binding(2) var<storage, read> in_lo: array<f32>;
@group(0) @binding(3) var<storage, read> in_hi: array<f32>;
@group(0) @binding(4) var<storage, read_write> abar: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.num_specs * p.input_dim) { return; }
    let j = t % p.input_dim;
    abar[t] = select(in_hi[j], in_lo[j], a0[t] >= 0.0);
}
"#;

/// Forward ReLU (coefficient channel): `A'[s,i] = A[s,i]·σ`, σ = lower_slope[i]
/// if `A[s,i] ≥ 0` else upper_slope[i] (design doc §1 lower stream). The input
/// buffer `a` is the captured pre-transform coefficient `A_preᵏ` (kept resident
/// for the adjoint); this writes a fresh output so `a` is preserved.
pub(super) const JOINT_RELU_FWD_SHADER: &str = r#"
struct P { num_specs: u32, nn: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> lower_slope: array<f32>;
@group(0) @binding(3) var<storage, read> upper_slope: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.num_specs * p.nn) { return; }
    let i = t % p.nn;
    let av = a[t];
    let sig = select(upper_slope[i], lower_slope[i], av >= 0.0);
    out[t] = av * sig;
}
"#;

/// Forward Conv2d fold (coefficient channel, transposed conv `A ⊛ Wᵀ`), fused
/// GEMM+col2im as a per-input-element GATHER — mirrors `fold_layer_forward`
/// Conv2d (scatter) in the equivalent gather form. Thread per input-space element
/// `(s, cin, ih, iw)`; sums over `(oc, ky, kx)` whose receptive map hits it.
pub(super) const JOINT_CONV_T_FWD_SHADER: &str = r#"
struct C { num_specs: u32, oc: u32, ic: u32, oh: u32,
           ow: u32, ih: u32, iw: u32, kh: u32,
           kw: u32, sh: u32, sw: u32, ph: u32,
           pw: u32, has_bias: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> c: C;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let in_dim = c.ic * c.ih * c.iw;
    if (t >= c.num_specs * in_dim) { return; }
    let s = t / in_dim;
    let f = t % in_dim;
    let cin = f / (c.ih * c.iw);
    let r = f % (c.ih * c.iw);
    let iyy = r / c.iw;
    let ixx = r % c.iw;
    let a_dim = c.oc * c.oh * c.ow;
    let khkw = c.kh * c.kw;
    var acc: f32 = 0.0;
    for (var ky: u32 = 0u; ky < c.kh; ky = ky + 1u) {
        let numy = iyy + c.ph;
        if (numy < ky) { continue; }
        let ny = numy - ky;
        if (ny % c.sh != 0u) { continue; }
        let y = ny / c.sh;
        if (y >= c.oh) { continue; }
        for (var kx: u32 = 0u; kx < c.kw; kx = kx + 1u) {
            let numx = ixx + c.pw;
            if (numx < kx) { continue; }
            let nx = numx - kx;
            if (nx % c.sw != 0u) { continue; }
            let x = nx / c.sw;
            if (x >= c.ow) { continue; }
            for (var oc: u32 = 0u; oc < c.oc; oc = oc + 1u) {
                let av = a[s * a_dim + (oc * c.oh + y) * c.ow + x];
                let wv = w[oc * (c.ic * khkw) + cin * khkw + ky * c.kw + kx];
                acc = acc + av * wv;
            }
        }
    }
    out[t] = acc;
}
"#;

/// Elementwise add `out = x + y` (residual merge in the forward fold; skip / branch
/// fan-out sum in the adjoint). Length `n`.
pub(super) const JOINT_ADD_SHADER: &str = r#"
struct P { n: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.n) { return; }
    out[t] = x[t] + y[t];
}
"#;

/// Row-broadcast add `out[s,i] = x[s,i] + v[i]` — the adjoint linear/conv BIAS
/// CHANNEL (`Ā_in = Ā_out·Wᵀ + bias`, design doc §2). `dim` = the per-row width.
pub(super) const JOINT_ROWVEC_ADD_SHADER: &str = r#"
struct P { num_specs: u32, dim: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.num_specs * p.dim) { return; }
    let i = t % p.dim;
    out[t] = x[t] + v[i];
}
"#;

/// Adjoint ReLU HARVEST (design doc §2, the joint gradient): per neuron `i`,
/// `grad[i] = Σ_s Ā_out[s,i] · max(A_preᵏ[s,i], 0)`. This is CROWN_ALPHA_GRADIENT
/// GENERALIZED to carry the per-row adjoint `Ā_out[s,i]` INSIDE the sum instead of
/// the scalar `pre_lower[i]`. One thread per neuron `i` (single domain = all rows).
pub(super) const JOINT_RELU_HARVEST_SHADER: &str = r#"
struct P { num_specs: u32, nn: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> abar: array<f32>;
@group(0) @binding(2) var<storage, read> a_pre: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.nn) { return; }
    var acc: f32 = 0.0;
    for (var s: u32 = 0u; s < p.num_specs; s = s + 1u) {
        let idx = s * p.nn + i;
        acc = acc + abar[idx] * max(a_pre[idx], 0.0);
    }
    grad[i] = acc;
}
"#;

/// Adjoint ReLU PROPAGATE (design doc §2): `Ā_in[s,i] = Ā_out[s,i]·σ + τ`, with
/// σ,τ sign-selected from the FROZEN forward coefficient `A_preᵏ` (positive →
/// lower relaxation slope/intercept, negative → upper chord). `bias_channel`
/// toggles the `+ τ` term (the ~0.7× degradation A/B when dropped).
pub(super) const JOINT_RELU_PROP_SHADER: &str = r#"
struct P { num_specs: u32, nn: u32, bias_channel: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> abar: array<f32>;
@group(0) @binding(2) var<storage, read> a_pre: array<f32>;
@group(0) @binding(3) var<storage, read> lower_slope: array<f32>;
@group(0) @binding(4) var<storage, read> upper_slope: array<f32>;
@group(0) @binding(5) var<storage, read> lower_intercept: array<f32>;
@group(0) @binding(6) var<storage, read> upper_intercept: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p.num_specs * p.nn) { return; }
    let i = t % p.nn;
    let pos = a_pre[t] >= 0.0;
    let sig = select(upper_slope[i], lower_slope[i], pos);
    var v = abar[t] * sig;
    if (p.bias_channel != 0u) {
        let ta = select(upper_intercept[i], lower_intercept[i], pos);
        v = v + ta;
    }
    out[t] = v;
}
"#;

/// Adjoint Conv2d (design doc §2): the PLAIN conv (transpose of the fold's
/// conv-transpose), gathering per output-space element `(s, oc, oh, ow)` over
/// `(cin, ky, kx)`, plus the bias channel `+ bias_expanded[(oc,oh,ow)]`. Mirrors
/// `adjoint_layer` Conv2d exactly.
pub(super) const JOINT_CONV_ADJ_SHADER: &str = r#"
struct C { num_specs: u32, oc: u32, ic: u32, oh: u32,
           ow: u32, ih: u32, iw: u32, kh: u32,
           kw: u32, sh: u32, sw: u32, ph: u32,
           pw: u32, has_bias: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> c: C;
@group(0) @binding(1) var<storage, read> abar: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let out_dim = c.oc * c.oh * c.ow;
    if (t >= c.num_specs * out_dim) { return; }
    let s = t / out_dim;
    let f = t % out_dim;
    let oc = f / (c.oh * c.ow);
    let r = f % (c.oh * c.ow);
    let y = r / c.ow;
    let x = r % c.ow;
    let in_dim = c.ic * c.ih * c.iw;
    let khkw = c.kh * c.kw;
    var acc: f32 = 0.0;
    for (var ky: u32 = 0u; ky < c.kh; ky = ky + 1u) {
        let iy = y * c.sh + ky;
        if (iy < c.ph) { continue; }
        let iyy = iy - c.ph;
        if (iyy >= c.ih) { continue; }
        for (var kx: u32 = 0u; kx < c.kw; kx = kx + 1u) {
            let ix = x * c.sw + kx;
            if (ix < c.pw) { continue; }
            let ixx = ix - c.pw;
            if (ixx >= c.iw) { continue; }
            for (var cin: u32 = 0u; cin < c.ic; cin = cin + 1u) {
                let av = abar[s * in_dim + (cin * c.ih + iyy) * c.iw + ixx];
                let wv = w[oc * (c.ic * khkw) + cin * khkw + ky * c.kw + kx];
                acc = acc + av * wv;
            }
        }
    }
    if (c.has_bias != 0u) {
        acc = acc + bias[f];
    }
    out[t] = acc;
}
"#;
