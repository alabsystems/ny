// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#u4` — out-of-band TAINT TWIN shaders that do not live in `shaders.rs`.
//!
//! `ops/sentinel_taint_selfcheck.rs` measures the defect this family closes:
//! every value kernel saturates a finite overflow to the FINITE sentinel
//! `±FALLBACK_BOUND` (`1e10`) and the combine degrades to `1e30`, but both
//! downstream verdict guards are MAGNITUDE tests, so one small weight (lane 2:
//! `1e10 * 1e-20 = 1e-10`, error budget `5.4e-17`) or one activation slope
//! (lane 5: `1e30 * 1e-25 = 2.0000019e5`) launders the taint into a small,
//! finite, CONFIDENT number. The fix, proven on the GB10 by the GEMM twin
//! (`shaders.rs::GEMM_F32_TAINT_SHADER`) and the activation twin
//! (`shaders.rs::CROWN_ACTIVATION_RESIDENT_TAINT_SHADER`, 10/10 probe lanes
//! green), is a `u32` taint word carried BESIDE the value, OR'd and never
//! multiplied, with clean exact-zero multiplicative partners annihilating
//! (`R * 0 == 0` for every finite real the sentinel stands for). A tainted
//! stored zero cannot authenticate annihilation.
//!
//! This module holds the twins added after `shaders.rs` grew past comfortable
//! editing size; `shaders.rs` re-exports them so every `ops/` consumer keeps
//! its single `use super::super::shaders as sh;` import path.
//!
//! Besides the twins it holds Conv's exact-value word transport and the
//! ON-DEVICE row-fold kernel
//! ([`TAINT_ROW_OR_SHADER`]) that replaced the resident walk's per-layer
//! blocking word readbacks (the measured 2.3–3.1× gate-ON tax of
//! `taint_gate_overhead_report`): admitted no-twin transports conservatively
//! fold their words into a per-spec-row device accumulator, read back once at
//! walk end. [`TAINT_G13_SEED_SHADER`] is compiled but deliberately dormant;
//! boundary reseeding is not accepted as a substitute for internal transport.

/// `#u4` exact-value taint twin of `GEMM_F32_SMALL_K_SHADER`.
///
/// The value loop, row schedule, guard, and write order deliberately mirror the
/// small-K base shader statement-for-statement.  The extra word loop is purely
/// observational, so selecting this twin under the word gate preserves every
/// value bit while removing the former large-M/small-K authority refusal.
pub(super) const GEMM_F32_SMALL_K_TAINT_SHADER: &str = r#"
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

const FALLBACK_BOUND: f32 = 1e10;
const ROWS_PER_THREAD: u32 = 4u;

fn nan_safe_clamp(x: f32) -> f32 {
    if (x != x) { return x; }
    return clamp(x, -FALLBACK_BOUND, FALLBACK_BOUND);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let base_row = gid.y * ROWS_PER_THREAD;
    if (col >= params.n) { return; }

    for (var r: u32 = 0u; r < ROWS_PER_THREAD; r = r + 1u) {
        let row = base_row + r;
        if (row >= params.m) { return; }

        var sum: f32 = 0.0;
        var taint: u32 = 0u;
        let a_base = row * params.k;
        for (var kk: u32 = 0u; kk < params.k; kk = kk + 1u) {
            let ai = a_base + kk;
            let bi = kk * params.n + col;
            let av = a[ai];
            let bv = b[bi];
            sum = sum + av * bv;
            let taw = taint_a[ai];
            let tbw = taint_b[bi];
            if (taw != 0u && (bv != 0.0 || tbw != 0u)) { taint = 1u; }
            if (tbw != 0u && (av != 0.0 || taw != 0u)) { taint = 1u; }
        }
        let guarded = nan_safe_clamp(sum);
        if (guarded != guarded || abs(guarded) >= FALLBACK_BOUND) { taint = 1u; }
        out[row * params.n + col] = guarded;
        taint_out[row * params.n + col] = taint;
    }
}
"#;

/// `#u4` exact-value taint twin of `CONV_RESHAPE_SHADER`.
///
/// Reshape is a permutation, so values and words use the identical asymmetric
/// `(spec, channel, position) -> (spec, position, channel)` index mapping.
pub(super) const CONV_RESHAPE_TAINT_SHADER: &str = r#"
struct Params {
    num_specs: u32,
    out_channels: u32,
    spatial: u32,
    _padding: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> taint_src: array<u32>;
@group(0) @binding(4) var<storage, read_write> taint_dst: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.num_specs * params.spatial * params.out_channels;
    if (idx >= total) { return; }
    let flat_row = idx / params.out_channels;
    let oc = idx % params.out_channels;
    let s = flat_row / params.spatial;
    let pos = flat_row % params.spatial;
    let src_idx = s * (params.out_channels * params.spatial) + oc * params.spatial + pos;
    dst[idx] = src[src_idx];
    taint_dst[idx] = taint_src[src_idx];
}
"#;

/// `#u4` exact-value taint twin of `CONV_COL2IM_SHADER`.
///
/// Col2im is additive: no clean-zero multiplier can annihilate an incoming
/// word, so all gathered words are OR-carried.  In addition, every running
/// value is inspected *after* the same add used by the base shader.  This
/// catches a sentinel/non-finite partial that a later opposite-signed gather
/// cancels back into an innocent-looking final value—the Conv-specific hole
/// that boundary reseeding could not close.  The inspection never rewrites
/// `sum`, preserving gate-on/off f32 value bits.
pub(super) const CONV_COL2IM_TAINT_SHADER: &str = r#"
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
@group(0) @binding(3) var<storage, read> taint_gemm: array<u32>;
@group(0) @binding(4) var<storage, read_write> taint_dst: array<u32>;
const FALLBACK_BOUND: f32 = 1e10;
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
    var sum: f32 = 0.0;
    var taint: u32 = 0u;
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
            sum = sum + gemm_out[src];
            taint = taint | taint_gemm[src];
            if (sum != sum || abs(sum) >= FALLBACK_BOUND) { taint = taint | 1u; }
        }
    }
    dst[thread_id] = sum;
    taint_dst[thread_id] = taint;
}
"#;

/// `#u4` taint twin of [`super::shaders::CROWN_AW_ERROR_COMBINE_SHADER`]: the
/// same arithmetic byte-for-byte on the same bindings 0-4, plus three additive
/// `u32` taint bindings (5-7):
///
/// * `taint_sprod_in` (5) — the word beside `s_prod = fl(|A|@|W|)`, written by
///   the `|A|@|W|` GEMM twin;
/// * `taint_prop_in` (6) — the word beside `prop = fl(err@|W|)`, written by the
///   `err@|W|` GEMM twin;
/// * `taint_e_out` (7) — the word beside the combined error `err_out`.
///
/// 7 storage bindings + 1 uniform, comfortably under the 12-binding house
/// limit. Pipeline rw flags for probe authors:
/// `&[false, false, true, false, false, false, true]`.
///
/// # The propagation rule
///
/// ```text
/// taint_e_out[i] = taint_sprod_in[i] | taint_prop_in[i]
///               | 1u when s_prod[i] >= FALLBACK_BOUND || prop[i] >= FALLBACK_BOUND
///               | 1u when the computed e was nonfinite/negative (the e = 1e30 repair arm)
/// ```
///
/// The two seed arms are exactly the two places the base shader writes its
/// `e = 1e30` degrade — i.e. the places taint is BORN in this op. `1e30` is a
/// MAGNITUDE, and `sentinel_taint_selfcheck` lane 5 measures it being laundered
/// ONE op later: a single activation slope of `1e-25` turns it into an ordinary
/// `2.0000019e5` charge, below every downstream guard, while the true error it
/// stood for is UNKNOWN and strictly larger than `1e10`. The `u32` word cannot
/// be laundered because nothing ever multiplies it: it is OR'd here and at
/// every other twin, so the only way it clears is an upstream EXACT-zero
/// annihilation that was arithmetically justified.
///
/// # Why there are NO annihilation conjuncts in this twin
///
/// The GEMM twin guards every OR with `partner != 0` because a GEMM tap has a
/// per-element multiplicative partner that can be exactly zero (a dead ReLU's
/// weight), and `R * 0 == 0` for every finite real the sentinel stands for.
/// The combine has NO such partner: it consumes two NON-NEGATIVE reductions
/// ADDITIVELY (`γ_k·s_prod + prop`, then a host-constant `slack >= 1` scale and
/// a non-negative `flush` add). Every exact-zero annihilation that could ever
/// justify dropping the word already happened upstream, per tap, inside the
/// GEMM twins — which then emit an UNTAINTED zero, so this shader never sees a
/// set word on a legitimately-annihilated element. An incoming set word here
/// means "this reduction's stored value is not trustworthy", and no arithmetic
/// fact available at this op can justify clearing that; conjuncts like
/// `s_prod[i] != 0.0` would only re-open a laundering hole (a tainted input
/// whose STORED value happens to be zero is precisely a value that cannot be
/// trusted to be zero).
///
/// # The value channel is untouched
///
/// `err_out` is computed exactly as the base shader computes it, INCLUDING both
/// `1e30` degrades, so every existing magnitude-guard decision is unchanged and
/// the twin is drop-in for the base wherever the taint words are wired. The
/// word is purely additive: consulting it downstream can only REFUSE more,
/// never grant more — the fail-closed direction.
// Consumed by the #u4 combine-twin probes (device tests) AND, since the
// NY_GPU_TAINT_WORDS resident wiring, built unconditionally into
// `ResidentBackwardPipelines::combine_taint` (dispatched only under the gate).
pub(super) const CROWN_AW_ERROR_COMBINE_TAINT_SHADER: &str = r#"
struct Params { n: u32, slack: f32, gamma_k: f32, additive: f32,
                k: u32, out_cols: u32, w_l1_max: f32, _pad: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> s_prod: array<f32>;
@group(0) @binding(2) var<storage, read> prop: array<f32>;
@group(0) @binding(3) var<storage, read_write> err_out: array<f32>;
@group(0) @binding(4) var<storage, read> row_abs_a: array<f32>;   // per-spec-row ‖a_i‖₁
@group(0) @binding(5) var<storage, read> taint_sprod_in: array<u32>;
@group(0) @binding(6) var<storage, read> taint_prop_in: array<u32>;
@group(0) @binding(7) var<storage, read_write> taint_e_out: array<u32>;
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
        // #u4: incoming taint from the two GEMM twins, OR'd — never multiplied,
        // never cleared. This op is a pure additive combine of two NON-NEGATIVE
        // reductions, so it has no exact-zero multiplicative partner that could
        // justify annihilation; legitimate annihilation already happened per
        // tap inside the GEMM twins.
        var taint: u32 = taint_sprod_in[i] | taint_prop_in[i];
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
        // #u4 seed: the repair arm writes 1e30 because the computed charge is
        // not a usable number — that is a birth of taint, so record it out of
        // band where no downstream Lipschitz factor can scale it away.
        if (is_nonfinite(e) || e < 0.0) { e = 1e30; taint = taint | 1u; }
        // Both inputs are NON-NEGATIVE reductions run through the GEMM, so each is
        // monotone in its partials and saturates to EXACTLY FALLBACK_BOUND once the
        // true sum passes it. At the sentinel the true |A|@|W| / err@|W| is unknown
        // and strictly larger, so `e` would UNDER-cover: degrade instead. (The signed
        // A@W coefficient can cancel back under the sentinel and reach concretize
        // looking legitimate; this row is the only place that saturation is visible.)
        // #u4 seed: this degrade arm is the in-band 1e30 marker that
        // sentinel_taint_selfcheck lane 5 measures being laundered by a 1e-25
        // slope one op later; the taint word is its unlaunderable shadow.
        if (s_prod[i] >= FALLBACK_BOUND || prop[i] >= FALLBACK_BOUND) { e = 1e30; taint = taint | 1u; }
        err_out[i] = e;
        taint_e_out[i] = taint;
    }
}
"#;

/// `#u4` consult twin of [`super::shaders::CROWN_EFT_MIN_COMBINE_SHADER`] — the
/// C2 choke point of `ops/TAINT_GUARD_AUDIT.md` §4: the same arithmetic
/// byte-for-byte on the same bindings 0-7, plus two read-only `u32` taint
/// bindings appended after the existing ones (8-9):
///
/// * `taint_s` (8) — the word beside `s_prod = fl(|A|@|W|)`, written by that
///   GEMM twin's binding 6 `taint_out` (`shaders.rs::GEMM_F32_TAINT_SHADER`);
/// * `taint_p` (9) — the word beside `prop = fl(err@|W|)`, from the other GEMM
///   twin dispatch's binding 6 `taint_out`.
///
/// 9 storage bindings + 1 uniform, under the 12-binding house limit (the audit
/// counts 8 bindings today → 10 with the words). Pipeline rw flags for probe
/// authors: `&[false, false, false, false, true, false, false, false, false]`.
///
/// # The consult rule (refusal, not transport)
///
/// ```text
/// taint_s[i] != 0 || taint_p[i] != 0  =>  return   (err_out untouched)
/// ```
///
/// `min(err_out, e_eft)` is the ONLY operation in the whole chain that can
/// LOWER an error charge, and it happens mid-chain, per element, never read
/// back — a laundered `s_prod` here erases a degrade that the C1 preflight
/// consult would otherwise have seen (C1 catches what survives; C2 prevents
/// the one op that can un-survive it). The refusal order inside `main`:
///
/// 1. the base non-finite refusal FIRST — base order preserved, so NaN never
///    reaches the NaN-blind `>=` comparisons and both shaders refuse
///    non-finite elements identically;
/// 2. the word consult — the launder-proof gate. A `u32` word cannot be
///    downscaled, so the lane-2 shape (`s_prod = 1e-10` after one `1e-20`
///    weight, every magnitude innocent) refuses HERE, where the magnitude
///    arms below are provably blind;
/// 3. the magnitude arms, kept as redundant belt-and-suspenders: they fire
///    only at exactly `>= FALLBACK_BOUND`, which the GEMM twins ALSO
///    taint-seed, so the arms can never disagree with the words in the unsafe
///    direction.
///
/// # Why there is NO taint output binding
///
/// This twin is a CONSUMER (audit C2), not a transport like
/// [`CROWN_AW_ERROR_COMBINE_TAINT_SHADER`]. On refusal `err_out` is untouched
/// and its own word — written beside it by the combine twin's binding 7
/// `taint_e_out` — still stands. On a tightening the words were clean and
/// `e_eft` is computed from measured, untainted quantities, so no taint is
/// born (the non-finite `e_eft` arm refuses rather than degrades, exactly
/// like the base). Either way the word beside `err_out` needs no update from
/// this op.
///
/// # Production status
///
/// Dispatched by the resident walk under its AUTO/default word gate when the
/// taint twins are available; `NY_GPU_TAINT_WORDS=0` opts out and `=1` requires
/// them. The C1 `PRODUCTION_GUARDS_CONSULT_TAINT_WORD` source gate is armed, so
/// absent or tainted row words fail closed at concretization. This C2 refusal
/// can only WIDEN an error, never tighten one. U5/U6 and B0 are discharged and
/// the independent raw-device authority source gate is now open; its typed
/// request and full per-device live-probe conjunction still fail closed.
// Consumed by the #u4 C2 device probes (`ops/eft_min_combine_taint_probe.rs`)
// AND, since the NY_GPU_TAINT_WORDS resident wiring, built unconditionally into
// `ResidentBackwardPipelines::eft_min_combine_taint` (dispatched only under the
// gate, and only on the eft-armed Linear path).
pub(super) const CROWN_EFT_MIN_COMBINE_TAINT_SHADER: &str = r#"
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
@group(0) @binding(8) var<storage, read> taint_s: array<u32>; // #u4 word beside s_prod
@group(0) @binding(9) var<storage, read> taint_p: array<u32>; // #u4 word beside prop
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
    // the (already sound) Higham value. This refusal stays FIRST — identical
    // to the base — so NaN never reaches the NaN-blind `>=` arms below.
    if (is_nonfinite(v) || is_nonfinite(val) || is_nonfinite(r) || is_nonfinite(pr)) { return; }
    // #u4 consult (C2): a set word beside either reduction means its STORED
    // value is not trustworthy — refuse the tightening outright. The word is
    // OR-carried and never multiplied, so the lane-2 launder (`s_prod = 1e-10`
    // after one 1e-20 weight, every magnitude innocent) refuses here, where
    // the magnitude arms below are blind. `min(err_out, e_eft)` is the only
    // op in the chain that can LOWER an error charge; on refusal the Higham
    // value ships unchanged — strictly a WIDENING, never a tightening.
    if (taint_s[i] != 0u || taint_p[i] != 0u) { return; }
    // Magnitude arms, kept as redundant belt-and-suspenders (#u4): both
    // reductions saturate to EXACTLY FALLBACK_BOUND and the GEMM twins
    // taint-seed at that same threshold, so these arms can never disagree
    // with the word consult in the unsafe direction. See the base shader's
    // SENTINEL STICKINESS block (#gpu-typed-authority) for the original
    // per-arm argument.
    if (pr >= FALLBACK_BOUND) { return; }
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

/// `#u4` on-device word→row TRANSPORT — the kernel form of the resident walk's
/// fail-closed no-twin transports (`or_taint_words_into_rows` and the
/// annihilation-conjunct companions `bias_fold_taint` / `intercept_fold_taint`
/// in crown_backward_sound_host.rs, which stay as the CPU test references).
/// Each thread owns one element of a `[rows × cols]` word buffer and ORs a set
/// word into its spec row of a device accumulator, so no mid-walk host
/// readback is needed: the walk reads the accumulator ONCE at walk end.
///
/// Bindings (uniform 0 + 3 storage — fits every granted limit, so the pipeline
/// is built unconditionally):
///
/// * `words` (1, ro) — `u32` taint words, `[rows*cols]` row-major;
/// * `partner` (2, ro) — `f32` multiplicative partners; only read when
///   `use_partner != 0` (bind any placeholder buffer otherwise — the walk
///   binds `words` itself);
/// * `rows_out` (3, rw) — per-spec-row accumulator `[rows]`,
///   `array<atomic<u32>>` in WGSL, plain `u32` on the host side (rw flags
///   `&[false, false, true]`).
///
/// # The partner-mode encoding (`Params::use_partner`)
///
/// * `0` — UNCONDITIONAL: every set word condemns its row (the fail-closed
///   no-twin form: conv incoming row-OR, the §0 row-L1 word, the final
///   coefficient/error fold, the batched-domain intercept fallback).
/// * `1` — per-COLUMN partner: element `i` keeps its word iff
///   `partner[i % cols] != 0.0` (`partner` len ≥ `cols`) — the `bias[k]`
///   conjunct of `bias_fold_taint`. The two-intercept disjunction
///   `li != 0 || ui != 0` of `intercept_fold_taint` is expressed as TWO
///   dispatches of this mode (one per intercept vector): a word survives iff
///   EITHER dispatch keeps it, which is exactly the disjunction.
/// * `2` — per-ELEMENT partner: element `i` keeps its word iff
///   `partner[i] != 0.0` (`partner` same shape as `words`).
///
/// # The propagation rule
///
/// ```text
/// word != 0 && (use_partner == 0 || its partner != 0.0)
///     => atomicOr(&rows_out[i / cols], word)
/// ```
///
/// The word VALUE is OR'd (not `1u`) so multi-bit words survive exactly as in
/// the host companions (`rows[s] |= word`); the twins only ever emit `0`/`1`,
/// where the two are identical. Annihilation matches the host conjunct
/// bit-for-bit: `partner == 0.0` is true for ±0 (drop — `R·0 == 0` for every
/// finite real the sentinel stands for) and false for NaN (keep — an unknown
/// partner cannot justify clearing), the same outcomes as the host's
/// `partner != 0.0`. The accumulator is monotone: nothing ever clears a row
/// bit, so dispatch order between transports is irrelevant (atomicOr is the
/// only write).
// Consumed by the resident walk's on-device transports under the resolved word
// gate (AUTO/default or forced `NY_GPU_TAINT_WORDS=1`; `=0` never dispatches it)
// and pinned through the walk by
// `taint_walk_bias_conjunct_annihilates_on_device`.
pub(super) const TAINT_ROW_OR_SHADER: &str = r#"
struct Params { rows: u32, cols: u32, use_partner: u32, _pad: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> words: array<u32>;
@group(0) @binding(2) var<storage, read> partner: array<f32>;
@group(0) @binding(3) var<storage, read_write> rows_out: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.rows * p.cols) { return; }
    let w = words[i];
    if (w == 0u) { return; }
    // Annihilation conjunct: an EXACT-zero multiplicative partner is the one
    // arithmetically-justified word drop (R * 0 == 0). NaN partners compare
    // false to 0.0 and therefore KEEP the word — same as the host `!= 0.0`.
    if (p.use_partner == 1u && partner[i % p.cols] == 0.0) { return; }
    if (p.use_partner == 2u && partner[i] == 0.0) { return; }
    atomicOr(&rows_out[i / p.cols], w);
}
"#;

/// `#u4` dormant on-device G13 word-seed reference — the kernel form of the
/// host `taint_seed_word` rule. It marks a non-finite value or one at/beyond
/// `CROWN_COEFF_MAX` (`== FALLBACK_BOUND`). Production does not use it as a
/// boundary patch: Conv's exact-op twins above catch a word before internal
/// cancellation, while seed ingestion uses the host rule directly.
///
/// Bindings (uniform 0 + 2 storage; rw flags `&[false, true]`):
///
/// * `values` (1, ro) — `f32` value buffer `[n]`;
/// * `words_out` (2, rw) — `u32` word buffer `[n]`, fully overwritten with
///   `seed(value)` rather than OR-accumulated.
///
/// # The seed rule (mirror of `taint_seed_word`, bit-for-bit)
///
/// ```text
/// word = 1u  when  !finite(v) || abs(v) >= CROWN_COEFF_MAX   else 0u
/// ```
///
/// `is_nonfinite` is a bit test (NaN-safe: NaN reaches neither the NaN-blind
/// `>=` nor `abs`); the threshold literal `1e10` rounds to the same f32 in
/// WGSL and Rust, so the device seed can never disagree with the host rule.
#[allow(dead_code)] // Reference only; exact-op twins make boundary reseeding unnecessary.
pub(super) const TAINT_G13_SEED_SHADER: &str = r#"
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read_write> words_out: array<u32>;
const CROWN_COEFF_MAX: f32 = 1e10;  // == ny_core::CROWN_COEFF_MAX == FALLBACK_BOUND (pinned)
fn is_nonfinite(x: f32) -> bool { let b = bitcast<u32>(x); return (b & 0x7f800000u) == 0x7f800000u; }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.n) {
        let v = values[i];
        words_out[i] = select(0u, 1u, is_nonfinite(v) || abs(v) >= CROWN_COEFF_MAX);
    }
}
"#;
