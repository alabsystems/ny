// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#u1` TREE-KERNEL SETTLING PROBE (campaign item B2): do the PRODUCTION
//! 256-thread strided-loop + 8-level tree-reduction kernels compile to the op
//! sequence their certified-error channel claims to measure?
//!
//! # The obligation (ops/sound_authority.rs, U1)
//!
//! Every probe in the authority ladder is a `workgroup_size(1)` straight-line
//! shader. The GEMM twin's settling test exists
//! (`ops/twin_composition_probe.rs`, measured 2026-08-06: V bit-exact
//! everywhere, R re-associated within 2 ULP, always inside the
//! order-independent Higham envelope) — but the BIAS / ACTIVATION-INTERCEPT /
//! CONCRETIZE kernels use a different composition entirely: per-lane strided
//! chains (lane `t` consumes taps `j = t, t+256, …` ascending), a fixed
//! 8-level `var<workgroup>` tree (`stride = 128 … 1`) with the shipped
//! 3-addend `sr[t] = sr[t] + sr[t+stride] + r0`, and a thread-0 tail
//! (`round_up_pos`, the flush term, the final running-sum `rf`). The col2im
//! EFT twin adds a per-element serial gather chain. None of those were ever
//! bit-compared. This file is that comparison.
//!
//! # What it does
//!
//! Verbatim copies of the four production WGSL constants
//! (`CROWN_BIAS_ERR_ACCUMULATE_SHADER`, `CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER`,
//! `CROWN_CONCRETIZE_SOUND_SHADER`, `CONV_COL2IM_EFT_TWIN_SHADER`) are compiled
//! on the production `WgpuDevice` (same features/limits/loading seam,
//! including `NY_GPU_DENORM_PRESERVE=1` passthrough parity) and dispatched
//! exactly as production does (`pass_simple` shape: binding 0 = params,
//! one-workgroup-per-spec-row for the tree kernels; per-element grid for
//! col2im). A CPU twin transcribes each WGSL line-for-line (`f32::mul_add`
//! for every `fma`, host FMA validity asserted via `ny_core::eft`), and the
//! ONLY observables are compared: `(bias_out[s], bias_err_out[s])` for the
//! bias kernels, `(output_lower[s], output_upper[s])` for concretize,
//! `(v_dst, r_dst)` per element for col2im.
//!
//! This lives in `tests/` (not the lib) because the lib API needed
//! (`resident_backward_pipelines`, `pass_simple`, `read_u32_buffer`) is
//! `pub(crate)` and this probe must not widen the lib surface. Byte-identity
//! of the WGSL copies with the shipped source is ENFORCED, not hoped for:
//! [`shader_copies_match_production_source`] `include_str!`s
//! `src/wgpu_device/shaders.rs` and asserts substring containment, so a
//! production shader edit fails this suite until the copy is refreshed. The
//! same wgpu/naga toolchain then compiles identical bytes to identical
//! pipelines — the compiled-sequence question is preserved.
//!
//! `eft_mode = 1` is driven at the PARAMS level, never via the authority
//! ladder (precedent: `u1_composed_sequence_integrity` in
//! `crown_backward_sound_resident.rs`): on this box `eft_primitives_cached() =
//! false` under `#u2b` keeps the arm dark at the production gates, and U1
//! measures COMPILATION, so the test builds the uniforms itself.
//!
//! # Localization tiers (input construction only; zero shader edits)
//!
//! * **T1 composed** — full word compare at the CROWN-shaped case matrix
//!   (`k ∈ {1, 15, 255, 256, 257, 1000, 3072, 4096, 14400}` straddles the
//!   chain-length 1→2 boundary and reaches cifar100_resnet_medium's largest
//!   production reduction). Verdict semantics: V bit-exact is HARD (the value
//!   chain feeds the barriered TwoSum, so the compiler has no reassociation
//!   freedom there); an R word that drifts must stay within the 8-ULP pin
//!   (the GEMM twin's measured ceiling), must match an ENUMERATED association
//!   hypothesis or be reported, must never match a dropped-term control, and
//!   the f64 ENCLOSURE (|exact − V| ≤ published radius) must hold with zero
//!   violations in EFT mode.
//! * **T2 chain isolation** — one nonzero lane `t0 ∈ {0, 1, 17, 255}`, chain
//!   lengths 1/2/5, operands engineered so every TwoSum residual is exactly 0
//!   and the TwoProduct residuals are distinct powers of two: the R word is
//!   then association-INDEPENDENT (any binary tree over the same multiset is
//!   exact), so bit-inequality is a hard failure that NAMES the strided chain.
//! * **T3 pair meeting-level** — exactly two nonzero lanes `(a, a+s)` with
//!   `a < s`, per `s ∈ {128, …, 1}`: the msb rule makes level `s` the unique
//!   meeting point, so the pair's tree residual `r0` is the only nonzero
//!   residual in the whole dispatch and a failure NAMES the tree level. All
//!   15 distinct pairs (+ the stride-1 repeat with different values) ride as
//!   16 spec rows of ONE dispatch.
//! * **T4 tail isolation** — `k = 1` cases separately pin residual recovery,
//!   propagated-error combine-slack recovery, and staged outward publication
//!   beside an existing radius. The explicit `round_up_pos` barriers make the
//!   new tail sequence bit-defined; every GPU word must equal its CPU twin.
//!
//! # Fail-closed discipline
//!
//! Every GPU error path panics (a refused dispatch must fail the settling
//! probe, not skip it); pure-write outputs are sentinel-prefilled so a
//! silently no-op'd dispatch (the async bind-group trap) reads back as a
//! mismatch, never as agreement; read-modify-write outputs use known preloads
//! whose faithful increments are asserted nonzero on discriminating rows. The
//! probe authorizes nothing by itself: it is the U1 evidence consumed by the
//! later B0 review that opened `PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED`.
//! Runtime authority still requires the explicit request and the independent
//! live five-rung ladder.

#![cfg(all(feature = "wgpu", feature = "gpu-tests"))]
#![allow(
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    // This file deliberately transcribes WGSL expression/assignment syntax so
    // its settling hypotheses stay visually auditable against the shader.
    clippy::assign_op_pattern,
    clippy::manual_is_multiple_of
)]

use std::sync::{Mutex, OnceLock};

use ny_gpu::WgpuDevice;

// ===========================================================================
// §1 — VERBATIM production WGSL copies + the drift guard that keeps them honest
// ===========================================================================

/// The shipped source of every constant below. The drift guard asserts each
/// copy appears BYTE-FOR-BYTE in this file, so the comparison is always
/// against the code production actually compiles.
const PRODUCTION_SHADERS_RS: &str = include_str!("../src/wgpu_device/shaders.rs");

/// Verbatim copy of `CROWN_BIAS_ERR_ACCUMULATE_SHADER` (shaders.rs).
const BIAS_ERR_ACCUMULATE_WGSL: &str = r#"
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

/// Verbatim copy of `CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER` (shaders.rs).
const ACTIVATION_INTERCEPT_BIAS_WGSL: &str = r#"
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

/// Verbatim copy of `CROWN_CONCRETIZE_SOUND_SHADER` (shaders.rs).
const CONCRETIZE_SOUND_WGSL: &str = r#"
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

/// Verbatim copy of `CONV_COL2IM_EFT_TWIN_SHADER` (shaders.rs).
const CONV_COL2IM_EFT_TWIN_WGSL: &str = r#"
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

/// CPU-only drift guard: a copy that stops matching the shipped constant makes
/// every GPU verdict in this file fail until the copy is refreshed — the
/// settling test can never silently measure stale WGSL.
#[test]
fn shader_copies_match_production_source() {
    // Normalize line endings before the byte comparison. The `copy` constants
    // below are ordinary string literals, and RUSTC NORMALIZES CRLF TO LF
    // inside a literal — but `include_str!` does not, it hands back the file's
    // bytes verbatim. So under core.autocrlf the shipped source carries CRLF
    // while every copy carries LF, and `contains` could never match on Windows:
    // the drift guard reported stale WGSL for four shaders that were in fact
    // byte-identical. Normalizing keeps this a comparison of CONTENT, which is
    // what "verbatim copy" is meant to assert.
    let production = PRODUCTION_SHADERS_RS.replace("\r\n", "\n");
    for (name, copy) in [
        ("CROWN_BIAS_ERR_ACCUMULATE_SHADER", BIAS_ERR_ACCUMULATE_WGSL),
        (
            "CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER",
            ACTIVATION_INTERCEPT_BIAS_WGSL,
        ),
        ("CROWN_CONCRETIZE_SOUND_SHADER", CONCRETIZE_SOUND_WGSL),
        ("CONV_COL2IM_EFT_TWIN_SHADER", CONV_COL2IM_EFT_TWIN_WGSL),
    ] {
        assert!(
            production.contains(&copy.replace("\r\n", "\n")),
            "{name}: the verbatim copy in tests/u1_tree_settling.rs no longer \
             matches src/wgpu_device/shaders.rs — refresh the copy before \
             trusting any measurement in this file"
        );
    }
}

// ===========================================================================
// §2 — Local transcriptions of the host-side sound helpers
//       (sound_consts.rs is pub(crate); these mirror it and are property-pinned)
// ===========================================================================

/// 2^-126, the smallest f32 NORMAL — the shaders' flush-safe residual charge.
const F32_MIN_NORMAL: f32 = 1.1754944e-38;
/// 2^-101 — below this product magnitude the TwoProdFMA residual may itself
/// round, and the shaders substitute [`F32_MIN_NORMAL`].
const TWO_PROD_EXACT_FLOOR_F32: f32 = 3.9443045e-31;
/// The shaders' saturation bound (`ny_core::FALLBACK_BOUND` as the WGSL literal).
const FALLBACK_BOUND: f32 = 1e10;
/// NaN payload pre-written into pure-write outputs: an element the grid never
/// wrote shows up as a mismatch instead of a plausible zero.
const UNWRITTEN_SENTINEL: u32 = 0x7FC0_1234;
/// The GEMM twin's measured re-association ceiling, reused as the tripwire for
/// the tree kernels' R words: drift beyond this is a composition CHANGE.
const PINNED_MAX_ULP_DRIFT: i64 = 8;
/// f32 unit roundoff 2^-24 as f64 (mirrors `sound_consts::U`).
const U64: f64 = 5.960_464_477_539_063e-8;
/// `sound_consts::TREE_REDUCTION_RESIDUAL_ADDS` — 2 adds x 8 levels.
const TREE_REDUCTION_RESIDUAL_ADDS: usize = 16;

/// Round an f64 UP to f32 (mirrors `sound_consts::up_f32` on the positive
/// domain these slacks live in).
fn up_f32_local(x: f64) -> f32 {
    let y = x as f32; // round-to-nearest
    if f64::from(y) >= x {
        y
    } else {
        // Positive normal domain: bits+1 is next-up.
        f32::from_bits(y.to_bits() + 1)
    }
}

/// γ_k = k·u/(1−k·u), rounded outward (mirrors `sound_consts::gamma_k_f32`).
fn gamma_k_local(k: usize) -> f32 {
    let ku = (k as f64) * U64;
    assert!(
        ku.is_finite() && ku < 1.0,
        "gamma_k_local: k={k} outside the Higham regime — the production \
         helper fails closed here and so does the probe"
    );
    up_f32_local(ku / (1.0 - ku))
}

/// EFT residual recovery slack (mirrors `sound_consts::eft_r_slack_f32`):
/// γ over `2k + 2 + 16` terms (the U3-discharged exact count: 2k chain terms,
/// +2 for `rf` and the final assembly, +16 tree adds), inverted, with the
/// `(1+u)^6` final-assembly headroom, rounded up.
fn eft_r_slack_local(k: usize) -> f32 {
    let terms = 2 * k + 2 + TREE_REDUCTION_RESIDUAL_ADDS;
    let g = f64::from(gamma_k_local(terms));
    assert!(g < 1.0, "eft_r_slack_local: gamma >= 1 for k={k}");
    let inv = 1.0 / (1.0 - g);
    let headroom = (1.0 + U64).powi(6);
    up_f32_local(inv * headroom)
}

/// `round_up_pos` transcription (bit-defined via `bitcast<u32>`, portable —
/// identical text in all three tree shaders).
fn round_up_pos(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= 0x7f80_0000 {
        return x;
    }
    if (bits & 0x8000_0000) != 0 || magnitude == 0 {
        return 0.0;
    }
    if magnitude < 0x0080_0000 {
        return F32_MIN_NORMAL;
    }
    f32::from_bits(bits + 1)
}

/// Thread-0 publication used by both scalar bias-error writers after `flush`
/// itself has been rounded outward.
fn outward_bias_update(old_error: f32, local_error: f32, flush: f32) -> f32 {
    round_up_pos(round_up_pos(old_error + local_error) + flush)
}

/// `next_down_f32_normal` transcription (concretize tail).
fn next_down_f32_normal(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let negative = (bits & 0x8000_0000) != 0;
    if magnitude >= 0x7f80_0000 {
        return x;
    }
    if magnitude == 0 {
        return -F32_MIN_NORMAL;
    }
    if magnitude < 0x0080_0000 {
        return if negative { -F32_MIN_NORMAL } else { 0.0 };
    }
    let y_bits = if negative { bits + 1 } else { bits - 1 };
    if (y_bits & 0x7fff_ffff) < 0x0080_0000 {
        return if negative { -F32_MIN_NORMAL } else { 0.0 };
    }
    f32::from_bits(y_bits)
}

/// `next_up_f32_normal` transcription (concretize tail).
fn next_up_f32_normal(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let negative = (bits & 0x8000_0000) != 0;
    if magnitude >= 0x7f80_0000 {
        return x;
    }
    if magnitude == 0 {
        return F32_MIN_NORMAL;
    }
    if magnitude < 0x0080_0000 {
        return if negative { -0.0 } else { F32_MIN_NORMAL };
    }
    let y_bits = if negative { bits - 1 } else { bits + 1 };
    if (y_bits & 0x7fff_ffff) < 0x0080_0000 {
        return if negative { -0.0 } else { F32_MIN_NORMAL };
    }
    f32::from_bits(y_bits)
}

/// WGSL `is_non_finite` transcription (concretize).
fn is_non_finite(x: f32) -> bool {
    (x.to_bits() & 0x7f80_0000) == 0x7f80_0000
}

/// The fma-barrier TwoSum shared by every kernel under test, transcribed
/// op-for-op (`fma` → `mul_add`; host FMA validity is asserted by
/// `ny_core::eft::eft_self_check` before any comparison). Returns `(s, es)`;
/// callers apply `abs` exactly where the WGSL does.
#[inline]
fn barriered_two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = (-1.0f32).mul_add(a, s);
    let sb = (-1.0f32).mul_add(bb, s);
    let da = (-1.0f32).mul_add(sb, a);
    let db = (-1.0f32).mul_add(bb, b);
    (s, da + db)
}

/// Signed ULP distance between two f32 bit patterns (ordered-integer mapping,
/// valid for any finite pair including opposite signs).
fn ulp_delta(a_bits: u32, b_bits: u32) -> i64 {
    fn ord(b: u32) -> i64 {
        if b & 0x8000_0000 != 0 {
            -i64::from(b & 0x7fff_ffff)
        } else {
            i64::from(b)
        }
    }
    ord(a_bits) - ord(b_bits)
}

/// Neumaier-compensated f64 accumulation (exact reference for the enclosure
/// checks; f64 products of f32 operands are exact).
#[inline]
fn neumaier_add(sum: &mut f64, comp: &mut f64, term: f64) {
    let t = *sum + term;
    *comp += if sum.abs() >= term.abs() {
        (*sum - t) + term
    } else {
        (term - t) + *sum
    };
    *sum = t;
}

/// Deterministic xorshift64* — reproducible operands, no dev-dependency
/// (idiom copied from the GEMM settling probe).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        (f64::from(bits) / f64::from(1u32 << 23) - 1.0) as f32
    }
    fn exp_scale(&mut self, lo: i32, hi: i32) -> f32 {
        let span = (hi - lo + 1) as u64;
        let e = lo + (self.next_u64() % span) as i32;
        f32::from_bits(((127 + e) as u32) << 23)
    }
}

// ===========================================================================
// §3 — GPU harness: production device, production loading seam, pass_simple
//       dispatch shape (mirrored locally; the lib helpers are pub(crate))
// ===========================================================================

static DEVICE: OnceLock<Result<WgpuDevice, String>> = OnceLock::new();
static SERIAL: Mutex<()> = Mutex::new(());

/// The PRODUCTION device: same adapter selection, features (PASSTHROUGH,
/// big-bindings), and limits as every scored run — the compiled-sequence
/// question is only meaningful on the configuration production runs.
fn require_device() -> &'static WgpuDevice {
    match DEVICE.get_or_init(|| WgpuDevice::new().map_err(|e| e.to_string())) {
        Ok(d) => d,
        Err(e) => panic!(
            "GPU required but not available: {e}. Run with --features gpu-tests \
             only when GPU hardware is present."
        ),
    }
}

/// Serialize the GPU tests in this binary (the in-lib `gpu_test_serial_guard`
/// is pub(crate); same discipline, local mutex).
fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Compile a WGSL module through the SAME seam semantics as production
/// (`shader_loading::create_compute_module`): use the live capability-resolved
/// profile stored on the exact production device, including the default AUTO
/// policy. Without this, an unset environment on a passthrough-capable adapter
/// would leave this probe attesting plain WGSL while production uses
/// DenormPreserve.
fn create_module(
    device: &wgpu::Device,
    denorm_preserve_enabled: bool,
    label: &str,
    wgsl: &str,
) -> wgpu::ShaderModule {
    if denorm_preserve_enabled {
        match ny_gpu_passthrough::create_denorm_preserving_module(device, label, wgsl) {
            Ok(m) => {
                println!("[u1-tree] '{label}' loaded via DenormPreserve passthrough");
                return m;
            }
            Err(reason) => {
                panic!(
                    "[u1-tree] passthrough refused for '{label}' ({reason}); \
                     refusing the mismatched test path (fail-closed)"
                );
            }
        }
    }
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
    })
}

/// Mirror of the lib's `create_simple_pipeline`: binding 0 = uniform params,
/// bindings 1..N = storage buffers with the given read_write flags, explicit
/// bind group layout (identical entries), entry point `main`.
fn build_pipeline(
    dev: &WgpuDevice,
    wgsl: &str,
    label: &str,
    rw: &[bool],
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let device = dev.device();
    let module = create_module(device, dev.denorm_preserve_enabled(), label, wgsl);
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    for (i, &is_rw) in rw.iter().enumerate() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: (i + 1) as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: !is_rw },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    (pipeline, layout)
}

/// One storage buffer's initial contents + whether the kernel writes it.
struct Buf<'a> {
    bytes: &'a [u8],
    rw: bool,
}

/// Dispatch mirroring `pass_simple` (binding 0 = params, 1.. = storages, own
/// compute pass, `dispatch_workgroups(wg_x.max(1), 1, 1)`) and read every
/// read_write buffer back as raw bit patterns (no float load that could
/// canonicalize a NaN). Panics on any error path — fail-closed.
fn dispatch_and_read(
    dev: &WgpuDevice,
    pipe: &(wgpu::ComputePipeline, wgpu::BindGroupLayout),
    params_bytes: &[u8],
    storages: &[Buf<'_>],
    wg_x: u32,
) -> Vec<Vec<u32>> {
    let device = dev.device();
    let queue = dev.queue();

    let p_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("u1_tree_params"),
        size: params_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&p_buf, 0, params_bytes);

    let mut bufs = Vec::with_capacity(storages.len());
    for s in storages {
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | if s.rw {
                wgpu::BufferUsages::COPY_SRC
            } else {
                wgpu::BufferUsages::empty()
            };
        let b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("u1_tree_storage"),
            size: s.bytes.len() as u64,
            usage,
            mapped_at_creation: false,
        });
        queue.write_buffer(&b, 0, s.bytes);
        bufs.push(b);
    }

    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: p_buf.as_entire_binding(),
    }];
    for (i, b) in bufs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: (i + 1) as u32,
            resource: b.as_entire_binding(),
        });
    }
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("u1_tree_bg"),
        layout: &pipe.1,
        entries: &entries,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("u1_tree_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("u1_tree_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe.0);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(wg_x.max(1), 1, 1);
    }

    // Stage every rw buffer for readback.
    let mut stages = Vec::new();
    for (i, s) in storages.iter().enumerate() {
        if s.rw {
            let stage = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("u1_tree_stage"),
                size: s.bytes.len() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&bufs[i], 0, &stage, 0, s.bytes.len() as u64);
            stages.push((stage, s.bytes.len() / 4));
        }
    }
    queue.submit(std::iter::once(encoder.finish()));

    let mut out = Vec::with_capacity(stages.len());
    for (stage, count) in &stages {
        let slice = stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .expect("u1_tree: map channel closed")
            .expect("u1_tree: buffer map failed");
        let data: Vec<u32> =
            bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range())[..*count].to_vec();
        stage.unmap();
        out.push(data);
    }
    out
}

fn as_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

// ===========================================================================
// §4 — Uniform structs (field-for-field mirrors of the production Pod structs)
// ===========================================================================

/// Mirror of `crown_backward_sound_resident::BiasParams`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BiasParamsT {
    num_specs: u32,
    k: u32,
    gamma_k: f32,
    additive: f32,
    slack: f32,
    eft_mode: u32,
    eft_r_slack: f32,
    _p: u32,
}

/// Mirror of `crown_backward_sound_resident::ActBiasParams`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ActBiasParamsT {
    num_specs: u32,
    num_neurons: u32,
    is_upper: u32,
    gamma_k: f32,
    additive: f32,
    slack: f32,
    num_specs_per_dom: u32,
    eft_mode: u32,
}

/// Mirror of `crown_concretize_sound::SoundConcretizeParams`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConcParamsT {
    num_specs: u32,
    input_dim: u32,
    gamma_n: f32,
    additive: f32,
    slack: f32,
    num_specs_per_dom: u32,
    eft_mode: u32,
    eft_r_slack: f32,
}

/// Mirror of the col2im `Params` uniform (14 u32 fields + vec2 padding).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Col2ImParamsT {
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
    _pad: [u32; 2],
}

// ===========================================================================
// §5 — CPU twins (line-for-line transcriptions) + hypothesis/control modes
// ===========================================================================

/// Which sequence a CPU twin executes. `Faithful` is the shipped one. The
/// `Drop*`/`ContiguousTaps` entries are NEGATIVE CONTROLS the compare must
/// REJECT; the `Tree*`/`Tail*`/`ChainBody*` entries are BENIGN-reassociation
/// HYPOTHESES used to classify any observed drift (each is RN-per-op inside
/// the (1+u)-per-op envelope — matching one is a named, bounded deviation;
/// matching none is a different sequence entirely).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TwinMode {
    /// Line-by-line transcription of the shipped WGSL.
    Faithful,
    /// Drops the strided chain's fma-barrier TwoSum residual `|es|` — the
    /// compiler folding the algebraically-zero compensation term. CONTROL.
    DropChainTwoSum,
    /// Drops the tree levels' `r0` residual — the 3-addend update degraded to
    /// the plain 2-addend reduction. CONTROL (this is exactly the term the
    /// U3 `+16` count exists for).
    DropTreeResidual,
    /// Tree residual update right-associated: `sr[t] + (sr[t+stride] + r0)`.
    /// HYPOTHESIS.
    TreeRightAssoc,
    /// Lane `t` consumes a CONTIGUOUS block instead of the strided class —
    /// a plausible-but-wrong partition. CONTROL.
    ContiguousTaps,
    /// Chain body `ev += a_err·|b|` contracted to one fma rounding. HYPOTHESIS.
    ChainBodyFusedEv,
    /// Tail `(sr[0]+rf)·r_slack + se[0]` contracted to one fma. HYPOTHESIS.
    TailFused,
    /// Tail outer sum right-associated: `err_pre + (round_up + flush)`.
    /// HYPOTHESIS.
    TailOuterRightAssoc,
    /// col2im only: drops the propagated `r_gemm` term from `rsum`. CONTROL.
    DropRGemm,
    /// col2im only: gathers the taps in reversed `(ki, kj)` order — detects
    /// whether the compare is sensitive to the serial chain ORDER. CONTROL.
    ReverseTaps,
}

/// What the faithful twin observed — discrimination evidence and the
/// preconditions the isolation tiers assert (fail-closed).
#[derive(Clone, Default)]
struct TreeStats {
    taps_total: u64,
    taps_nonzero_ep: u64,
    taps_nonzero_es: u64,
    taps_floor_charged: u64,
    /// Chain/tree intermediates that landed subnormal — MUST be 0 for the
    /// isolation tiers or the flush question contaminates the composition one.
    subnormal_intermediates: u64,
    /// Per spec row: Neumaier-f64 exact dot (`Σ a·b` resp. `Σ a·sel`), the
    /// reference the published radius must enclose.
    exact_dot: Vec<f64>,
    /// Per spec row: `Σ|a·b|` in f64 — the Higham envelope scale for the
    /// mode-0 (legacy) value chain, whose plain `v = v + a·b` accumulation the
    /// compiler is free to re-associate (unlike the EFT chain, whose `v` is
    /// consumed by the barriered TwoSum every tap).
    sum_abs_dot: Vec<f64>,
}

impl TreeStats {
    fn observe(&mut self, vals: &[f32]) {
        for &v in vals {
            if v != 0.0 && v.abs() < F32_MIN_NORMAL {
                self.subnormal_intermediates += 1;
            }
        }
    }
}

/// Shared workgroup lanes (the five `var<workgroup>` arrays).
struct Lanes {
    sv: [f32; 256],
    sa: [f32; 256],
    se: [f32; 256],
    sf: [f32; 256],
    sr: [f32; 256],
}

impl Lanes {
    fn new() -> Self {
        Lanes {
            sv: [0.0; 256],
            sa: [0.0; 256],
            se: [0.0; 256],
            sf: [1.0; 256], // `var fa: f32 = 1.0;` in EVERY thread
            sr: [0.0; 256],
        }
    }

    /// The shipped 8-level tree, `stride = 128 … 1`. Reading `[t+stride]`
    /// before writing `[t]` matches the barriered parallel semantics exactly
    /// (only indices `< stride` are written per level).
    fn tree_reduce(&mut self, mode: TwinMode, stats: &mut TreeStats) {
        let mut stride = 128usize;
        while stride > 0 {
            for t in 0..stride {
                let a0 = self.sv[t];
                let b0 = self.sv[t + stride];
                let (s0, es) = barriered_two_sum(a0, b0);
                let r0 = es.abs();
                self.sv[t] = s0;
                self.sa[t] += self.sa[t + stride];
                self.se[t] += self.se[t + stride];
                self.sf[t] += self.sf[t + stride];
                self.sr[t] = match mode {
                    TwinMode::DropTreeResidual => self.sr[t] + self.sr[t + stride],
                    TwinMode::TreeRightAssoc => self.sr[t] + (self.sr[t + stride] + r0),
                    _ => self.sr[t] + self.sr[t + stride] + r0,
                };
                stats.observe(&[s0, r0, self.sr[t]]);
            }
            stride >>= 1;
        }
    }
}

/// One tree-kernel case (bias or act_bias), uniforms included — the CPU twin
/// and the GPU dispatch consume the SAME struct so they cannot drift apart.
#[derive(Clone)]
struct TreeCase {
    name: String,
    num_specs: usize,
    /// `k` for the bias kernel, `num_neurons` for act_bias.
    k: usize,
    additive: f32,
    slack: f32,
    /// `eft_r_slack` (bias) / the r_slack carried in `gamma_k` (act_bias eft).
    r_slack: f32,
    /// mode-0 legacy γ_k (unused in EFT mode).
    gamma_k: f32,
    eft_mode: bool,
    /// act_bias only.
    is_upper: bool,
    /// act_bias/concretize batched-BaB partition; `== num_specs` single-domain.
    num_specs_per_dom: usize,
}

/// Inputs for a tree-kernel case.
struct TreeInputs {
    /// `a` — [num_specs * k].
    a: Vec<f32>,
    /// `a_err` (bias) / `err` (act_bias) — [num_specs * k].
    a_err: Vec<f32>,
    /// `bias` (bias kernel, [k]) / `lower_int` (act_bias, [n_domains * k]).
    b_lo: Vec<f32>,
    /// `upper_int` (act_bias only, [n_domains * k]); empty for the bias kernel.
    b_hi: Vec<f32>,
    /// `bias_out` preload — [num_specs].
    out_pre: Vec<f32>,
    /// `bias_err_out` preload — [num_specs].
    err_pre: Vec<f32>,
}

/// CPU twin of `CROWN_BIAS_ERR_ACCUMULATE_SHADER`. Returns
/// `(bias_out, bias_err_out, stats)`.
fn cpu_bias_twin(
    c: &TreeCase,
    inp: &TreeInputs,
    mode: TwinMode,
) -> (Vec<f32>, Vec<f32>, TreeStats) {
    let mut out_v = vec![0.0f32; c.num_specs];
    let mut out_r = vec![0.0f32; c.num_specs];
    let mut stats = TreeStats::default();
    let k = c.k;
    for s in 0..c.num_specs {
        let mut lanes = Lanes::new();
        let mut exact = 0.0f64;
        let mut comp = 0.0f64;
        let mut sum_abs = 0.0f64;
        for t in 0..256usize {
            let mut v = 0.0f32;
            let mut av = 0.0f32;
            let mut ev = 0.0f32;
            let mut fa = 1.0f32;
            let mut rv = 0.0f32;
            // Tap index stream: strided ascending (the shipped partition), or
            // the ContiguousTaps CONTROL partition.
            let taps: Vec<usize> = if mode == TwinMode::ContiguousTaps {
                let chunk = k.div_ceil(256);
                (t * chunk..((t + 1) * chunk).min(k)).collect()
            } else {
                (0..k).skip(t).step_by(256).collect()
            };
            for j in taps {
                let aj = inp.a[s * k + j];
                let bj = inp.b_lo[j];
                if !c.eft_mode {
                    v = v + aj * bj;
                    av = av + (aj * bj).abs();
                } else {
                    let prod = aj * bj;
                    let mut ep = aj.mul_add(bj, -prod).abs();
                    if aj != 0.0 && bj != 0.0 && prod.abs() < TWO_PROD_EXACT_FLOOR_F32 {
                        ep = F32_MIN_NORMAL;
                        stats.taps_floor_charged += 1;
                    }
                    let (s2, es) = barriered_two_sum(v, prod);
                    let charged_es = if mode == TwinMode::DropChainTwoSum {
                        0.0
                    } else {
                        es.abs()
                    };
                    rv = rv + ep + charged_es;
                    v = s2;
                    stats.taps_total += 1;
                    if ep != 0.0 {
                        stats.taps_nonzero_ep += 1;
                    }
                    if es != 0.0 {
                        stats.taps_nonzero_es += 1;
                    }
                    stats.observe(&[prod, ep, es, v, rv]);
                }
                let a_err = inp.a_err[s * k + j];
                if mode == TwinMode::ChainBodyFusedEv {
                    ev = a_err.mul_add(bj.abs(), ev);
                } else {
                    ev = ev + a_err * bj.abs();
                }
                fa = fa + aj.abs().max(bj.abs()).max(1.0);
                neumaier_add(&mut exact, &mut comp, f64::from(aj) * f64::from(bj));
                sum_abs += (f64::from(aj) * f64::from(bj)).abs();
            }
            lanes.sv[t] = v;
            lanes.sa[t] = av;
            lanes.se[t] = ev;
            lanes.sf[t] = fa;
            lanes.sr[t] = rv;
        }
        lanes.tree_reduce(mode, &mut stats);

        // Thread-0 tail.
        let old = inp.out_pre[s];
        let (sum, esf) = barriered_two_sum(old, lanes.sv[0]);
        out_v[s] = sum;
        let flush_scaled = round_up_pos(round_up_pos(lanes.sf[0] * c.slack) * F32_MIN_NORMAL);
        let flush = round_up_pos(c.additive + flush_scaled);
        let err_word = if !c.eft_mode {
            // The driver contracts `b·c + a` → fma (t4-measured on the eft
            // tail); the m0 γ term is the same shape. HYPOTHESIS arm.
            let reduced = match mode {
                TwinMode::TailFused => round_up_pos(c.gamma_k.mul_add(lanes.sa[0], lanes.se[0])),
                _ => round_up_pos(c.gamma_k * lanes.sa[0] + lanes.se[0]),
            };
            let local = round_up_pos(reduced * c.slack);
            match mode {
                TwinMode::TailOuterRightAssoc => round_up_pos(inp.err_pre[s] + (local + flush)),
                _ => round_up_pos(round_up_pos(inp.err_pre[s] + local) + flush),
            }
        } else {
            let rf = esf.abs();
            let residual = round_up_pos((lanes.sr[0] + rf) * c.r_slack);
            let propagated = round_up_pos(lanes.se[0] * c.slack);
            let local = round_up_pos(residual + propagated);
            match mode {
                TwinMode::TailOuterRightAssoc => round_up_pos(inp.err_pre[s] + (local + flush)),
                _ => round_up_pos(round_up_pos(inp.err_pre[s] + local) + flush),
            }
        };
        out_r[s] = err_word;
        stats.exact_dot.push(exact + comp);
        stats.sum_abs_dot.push(sum_abs);
    }
    (out_v, out_r, stats)
}

/// CPU twin of `CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER`. Returns
/// `(bias_out, bias_err_out, stats)`. In EFT mode the r_slack rides in the
/// `gamma_k` uniform field, exactly as production builds it.
fn cpu_act_bias_twin(
    c: &TreeCase,
    inp: &TreeInputs,
    mode: TwinMode,
) -> (Vec<f32>, Vec<f32>, TreeStats) {
    let mut out_v = vec![0.0f32; c.num_specs];
    let mut out_r = vec![0.0f32; c.num_specs];
    let mut stats = TreeStats::default();
    let nn = c.k;
    for s in 0..c.num_specs {
        // #batched-bab domain block: sbase = (s / max(nspd,1)) * nn.
        let sbase = (s / c.num_specs_per_dom.max(1)) * nn;
        let mut lanes = Lanes::new();
        let mut exact = 0.0f64;
        let mut comp = 0.0f64;
        let mut sum_abs = 0.0f64;
        for t in 0..256usize {
            let mut v = 0.0f32;
            let mut av = 0.0f32;
            let mut ev = 0.0f32;
            let mut fa = 1.0f32;
            let mut rv = 0.0f32;
            let taps: Vec<usize> = if mode == TwinMode::ContiguousTaps {
                let chunk = nn.div_ceil(256);
                (t * chunk..((t + 1) * chunk).min(nn)).collect()
            } else {
                (0..nn).skip(t).step_by(256).collect()
            };
            for j in taps {
                let idx = s * nn + j;
                let a_v = inp.a[idx];
                let li = inp.b_lo[sbase + j];
                let ui = inp.b_hi[sbase + j];
                // WGSL select(f, t, cond): is_upper==0 → a>=0 ? li : ui.
                let sel = if !c.is_upper {
                    if a_v >= 0.0 {
                        li
                    } else {
                        ui
                    }
                } else if a_v >= 0.0 {
                    ui
                } else {
                    li
                };
                if !c.eft_mode {
                    v = v + a_v * sel;
                    av = av + (a_v * sel).abs();
                    if mode == TwinMode::ChainBodyFusedEv {
                        ev = inp.a_err[idx].mul_add(li.abs() + ui.abs(), ev);
                    } else {
                        ev = ev + inp.a_err[idx] * (li.abs() + ui.abs());
                    }
                } else {
                    let prod = a_v * sel;
                    let mut ep = a_v.mul_add(sel, -prod).abs();
                    if a_v != 0.0 && sel != 0.0 && prod.abs() < TWO_PROD_EXACT_FLOOR_F32 {
                        ep = F32_MIN_NORMAL;
                        stats.taps_floor_charged += 1;
                    }
                    let (s2, es) = barriered_two_sum(v, prod);
                    let charged_es = if mode == TwinMode::DropChainTwoSum {
                        0.0
                    } else {
                        es.abs()
                    };
                    rv = rv + ep + charged_es;
                    v = s2;
                    // Lipschitz intercept propagation.
                    let e_in = inp.a_err[idx];
                    let prop = if a_v.abs() > e_in {
                        sel.abs()
                    } else {
                        li.abs().max(ui.abs())
                    };
                    if mode == TwinMode::ChainBodyFusedEv {
                        ev = e_in.mul_add(prop, ev);
                    } else {
                        ev = ev + e_in * prop;
                    }
                    stats.taps_total += 1;
                    if ep != 0.0 {
                        stats.taps_nonzero_ep += 1;
                    }
                    if es != 0.0 {
                        stats.taps_nonzero_es += 1;
                    }
                    stats.observe(&[prod, ep, es, v, rv]);
                }
                fa = fa + a_v.abs().max(li.abs().max(ui.abs())).max(1.0);
                neumaier_add(&mut exact, &mut comp, f64::from(a_v) * f64::from(sel));
                sum_abs += (f64::from(a_v) * f64::from(sel)).abs();
            }
            lanes.sv[t] = v;
            lanes.sa[t] = av;
            lanes.se[t] = ev;
            lanes.sf[t] = fa;
            lanes.sr[t] = rv;
        }
        lanes.tree_reduce(mode, &mut stats);

        let old = inp.out_pre[s];
        let (sum, esf) = barriered_two_sum(old, lanes.sv[0]);
        out_v[s] = sum;
        let flush_scaled = round_up_pos(round_up_pos(lanes.sf[0] * c.slack) * F32_MIN_NORMAL);
        let flush = round_up_pos(c.additive + flush_scaled);
        let err_word = if !c.eft_mode {
            // The driver contracts `b·c + a` → fma (t4-measured on the eft
            // tail); the m0 γ term is the same shape. HYPOTHESIS arm.
            let reduced = match mode {
                TwinMode::TailFused => round_up_pos(c.gamma_k.mul_add(lanes.sa[0], lanes.se[0])),
                _ => round_up_pos(c.gamma_k * lanes.sa[0] + lanes.se[0]),
            };
            let local = round_up_pos(reduced * c.slack);
            match mode {
                TwinMode::TailOuterRightAssoc => round_up_pos(inp.err_pre[s] + (local + flush)),
                _ => round_up_pos(round_up_pos(inp.err_pre[s] + local) + flush),
            }
        } else {
            // In EFT mode `gamma_k` carries r_slack (the γ term is unused).
            let rf = esf.abs();
            let residual = round_up_pos((lanes.sr[0] + rf) * c.r_slack);
            let propagated = round_up_pos(lanes.se[0] * c.slack);
            let local = round_up_pos(residual + propagated);
            match mode {
                TwinMode::TailOuterRightAssoc => round_up_pos(inp.err_pre[s] + (local + flush)),
                _ => round_up_pos(round_up_pos(inp.err_pre[s] + local) + flush),
            }
        };
        out_r[s] = err_word;
        stats.exact_dot.push(exact + comp);
        stats.sum_abs_dot.push(sum_abs);
    }
    (out_v, out_r, stats)
}

/// One concretize case.
#[derive(Clone)]
struct ConcCase {
    name: String,
    num_specs: usize,
    input_dim: usize,
    gamma_n: f32,
    additive: f32,
    slack: f32,
    num_specs_per_dom: usize,
    eft_mode: bool,
    eft_r_slack: f32,
}

/// Inputs for a concretize case (packed exactly as the shader binds them).
struct ConcInputs {
    a_lower: Vec<f32>,
    a_upper: Vec<f32>,
    /// [n_domains * input_dim] each.
    input_lower: Vec<f32>,
    input_upper: Vec<f32>,
    /// [lower | upper] — 2 * num_specs.
    bias: Vec<f32>,
    /// [lower_err | upper_err] — 2 * num_specs * input_dim.
    a_err: Vec<f32>,
}

/// Per-row f64 exact affine bounds (the enclosure reference).
struct ConcExact {
    lb: Vec<f64>,
    ub: Vec<f64>,
}

/// CPU twin of `CROWN_CONCRETIZE_SOUND_SHADER`. Returns
/// `(output_lower, output_upper, stats, exact)`.
fn cpu_concretize_twin(
    c: &ConcCase,
    inp: &ConcInputs,
    mode: TwinMode,
) -> (Vec<f32>, Vec<f32>, TreeStats, ConcExact) {
    let n = c.input_dim;
    let coeff = c.num_specs * n;
    let mut out_lo = vec![0.0f32; c.num_specs];
    let mut out_hi = vec![0.0f32; c.num_specs];
    let mut stats = TreeStats::default();
    let mut exact = ConcExact {
        lb: Vec::with_capacity(c.num_specs),
        ub: Vec::with_capacity(c.num_specs),
    };
    for spec_row in 0..c.num_specs {
        let dbase = (spec_row / c.num_specs_per_dom.max(1)) * n;
        // The seven `var<workgroup>` lanes.
        let mut sh_lb = [0.0f32; 256];
        let mut sh_ub = [0.0f32; 256];
        let mut sh_pl = [0.0f32; 256];
        let mut sh_pu = [0.0f32; 256];
        let mut sh_fa = [0.0f32; 256];
        let mut sh_rl = [0.0f32; 256];
        let mut sh_ru = [0.0f32; 256];
        let mut ex_lb = 0.0f64;
        let mut ex_lb_c = 0.0f64;
        let mut ex_ub = 0.0f64;
        let mut ex_ub_c = 0.0f64;
        for local_id in 0..256usize {
            let mut local_lb = 0.0f32;
            let mut local_ub = 0.0f32;
            let mut pen_l = 0.0f32;
            let mut pen_u = 0.0f32;
            let mut r_l = 0.0f32;
            let mut r_u = 0.0f32;
            let mut flushacc = 1.0f32;
            let mut degraded = false;
            let mut j = local_id;
            while j < n {
                let idx = spec_row * n + j;
                let a_l = inp.a_lower[idx];
                let a_u = inp.a_upper[idx];
                let e_l = inp.a_err[idx];
                let e_u = inp.a_err[coeff + idx];
                let x_l = inp.input_lower[dbase + j];
                let x_u = inp.input_upper[dbase + j];
                #[allow(clippy::eq_op)]
                let nan_l = a_l != a_l;
                #[allow(clippy::eq_op)]
                let nan_u = a_u != a_u;
                if nan_l || a_l.abs() >= FALLBACK_BOUND || nan_u || a_u.abs() >= FALLBACK_BOUND {
                    degraded = true;
                } else {
                    let a_l_pos = a_l.max(0.0);
                    let a_l_neg = a_l.min(0.0);
                    let a_u_pos = a_u.max(0.0);
                    let a_u_neg = a_u.min(0.0);
                    let xmax = x_l.abs().max(x_u.abs());
                    if !c.eft_mode {
                        local_lb = local_lb + a_l_pos * x_l + a_l_neg * x_u;
                        local_ub = local_ub + a_u_pos * x_u + a_u_neg * x_l;
                        pen_l = pen_l + (e_l + c.gamma_n * a_l.abs()) * xmax;
                        pen_u = pen_u + (e_u + c.gamma_n * a_u.abs()) * xmax;
                    } else {
                        // Four TwoProd/TwoSum chains, transcribed in shipped order.
                        let do_chain =
                            |acc: &mut f32,
                             r: &mut f32,
                             av: f32,
                             xv: f32,
                             stats: &mut TreeStats| {
                                let p = av * xv;
                                let mut ep = av.mul_add(xv, -p).abs();
                                if av != 0.0 && xv != 0.0 && p.abs() < TWO_PROD_EXACT_FLOOR_F32 {
                                    ep = F32_MIN_NORMAL;
                                    stats.taps_floor_charged += 1;
                                }
                                let (s2, es) = barriered_two_sum(*acc, p);
                                let charged_es = if mode == TwinMode::DropChainTwoSum {
                                    0.0
                                } else {
                                    es.abs()
                                };
                                *r = *r + ep + charged_es;
                                *acc = s2;
                                stats.taps_total += 1;
                                if ep != 0.0 {
                                    stats.taps_nonzero_ep += 1;
                                }
                                if es != 0.0 {
                                    stats.taps_nonzero_es += 1;
                                }
                                stats.observe(&[p, ep, es, *acc, *r]);
                            };
                        do_chain(&mut local_lb, &mut r_l, a_l_pos, x_l, &mut stats);
                        do_chain(&mut local_lb, &mut r_l, a_l_neg, x_u, &mut stats);
                        do_chain(&mut local_ub, &mut r_u, a_u_pos, x_u, &mut stats);
                        do_chain(&mut local_ub, &mut r_u, a_u_neg, x_l, &mut stats);
                        if mode == TwinMode::ChainBodyFusedEv {
                            pen_l = e_l.mul_add(xmax, pen_l);
                            pen_u = e_u.mul_add(xmax, pen_u);
                        } else {
                            pen_l = pen_l + e_l * xmax;
                            pen_u = pen_u + e_u * xmax;
                        }
                    }
                    flushacc = flushacc + a_l.abs().max(a_u.abs()).max(xmax).max(1.0);
                    // Exact references (true affine bounds over the box).
                    let fl = f64::from(a_l);
                    let fu = f64::from(a_u);
                    neumaier_add(
                        &mut ex_lb,
                        &mut ex_lb_c,
                        fl * if a_l >= 0.0 {
                            f64::from(x_l)
                        } else {
                            f64::from(x_u)
                        },
                    );
                    neumaier_add(
                        &mut ex_ub,
                        &mut ex_ub_c,
                        fu * if a_u >= 0.0 {
                            f64::from(x_u)
                        } else {
                            f64::from(x_l)
                        },
                    );
                }
                j += 256;
            }
            if degraded {
                local_lb = f32::NEG_INFINITY;
                local_ub = f32::INFINITY;
            }
            sh_lb[local_id] = local_lb;
            sh_ub[local_id] = local_ub;
            sh_pl[local_id] = pen_l;
            sh_pu[local_id] = pen_u;
            sh_fa[local_id] = flushacc;
            sh_rl[local_id] = r_l;
            sh_ru[local_id] = r_u;
        }

        // The shipped tree (value lanes with TwoSum residual capture, four
        // plain-add lanes, the shipped 3-addend residual update).
        let mut stride = 128usize;
        while stride > 0 {
            for t in 0..stride {
                let (sl0, esl) = barriered_two_sum(sh_lb[t], sh_lb[t + stride]);
                let rl_add = esl.abs();
                let (su0, esu) = barriered_two_sum(sh_ub[t], sh_ub[t + stride]);
                let ru_add = esu.abs();
                sh_lb[t] = sl0;
                sh_ub[t] = su0;
                sh_pl[t] += sh_pl[t + stride];
                sh_pu[t] += sh_pu[t + stride];
                sh_fa[t] += sh_fa[t + stride];
                sh_rl[t] = match mode {
                    TwinMode::DropTreeResidual => sh_rl[t] + sh_rl[t + stride],
                    TwinMode::TreeRightAssoc => sh_rl[t] + (sh_rl[t + stride] + rl_add),
                    _ => sh_rl[t] + sh_rl[t + stride] + rl_add,
                };
                sh_ru[t] = match mode {
                    TwinMode::DropTreeResidual => sh_ru[t] + sh_ru[t + stride],
                    TwinMode::TreeRightAssoc => sh_ru[t] + (sh_ru[t + stride] + ru_add),
                    _ => sh_ru[t] + sh_ru[t + stride] + ru_add,
                };
                stats.observe(&[sl0, su0, rl_add, ru_add]);
            }
            stride >>= 1;
        }

        // Thread-0 tail, transcribed in shipped order.
        let rs = if c.eft_mode { c.eft_r_slack } else { 0.0 };
        let flush_scaled = round_up_pos(round_up_pos(sh_fa[0] * c.slack) * F32_MIN_NORMAL);
        let flush = round_up_pos(c.additive + flush_scaled);
        let prop_l = round_up_pos(sh_pl[0] * c.slack);
        let prop_u = round_up_pos(sh_pu[0] * c.slack);
        let resid_l = round_up_pos(sh_rl[0] * rs);
        let resid_u = round_up_pos(sh_ru[0] * rs);
        let pen_l = round_up_pos(round_up_pos(prop_l + resid_l) + flush);
        let pen_u = round_up_pos(round_up_pos(prop_u + resid_u) + flush);
        let cl = next_down_f32_normal(sh_lb[0] + inp.bias[spec_row]);
        let cu = next_up_f32_normal(sh_ub[0] + inp.bias[c.num_specs + spec_row]);
        let mut lb = next_down_f32_normal(cl - pen_l);
        let mut ub = next_up_f32_normal(cu + pen_u);
        if is_non_finite(lb) {
            lb = -FALLBACK_BOUND;
        }
        if is_non_finite(ub) {
            ub = FALLBACK_BOUND;
        }
        if lb > ub {
            out_lo[spec_row] = -FALLBACK_BOUND;
            out_hi[spec_row] = FALLBACK_BOUND;
        } else {
            out_lo[spec_row] = lb;
            out_hi[spec_row] = ub;
        }
        exact
            .lb
            .push(ex_lb + ex_lb_c + f64::from(inp.bias[spec_row]));
        exact
            .ub
            .push(ex_ub + ex_ub_c + f64::from(inp.bias[c.num_specs + spec_row]));
    }
    (out_lo, out_hi, stats, exact)
}

/// One col2im case (conv geometry).
#[derive(Clone, Copy)]
struct Col2ImCase {
    name: &'static str,
    num_specs: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    pad: usize,
    out_h: usize,
    out_w: usize,
}

impl Col2ImCase {
    fn kernel_cols(&self) -> usize {
        self.in_channels * self.kernel_h * self.kernel_w
    }
    fn flat_input_dim(&self) -> usize {
        self.in_channels * self.in_h * self.in_w
    }
    fn gemm_len(&self) -> usize {
        self.num_specs * self.out_h * self.out_w * self.kernel_cols()
    }
    fn params(&self) -> Col2ImParamsT {
        Col2ImParamsT {
            num_specs: self.num_specs as u32,
            flat_input_dim: self.flat_input_dim() as u32,
            out_h: self.out_h as u32,
            out_w: self.out_w as u32,
            in_channels: self.in_channels as u32,
            in_h: self.in_h as u32,
            in_w: self.in_w as u32,
            kernel_h: self.kernel_h as u32,
            kernel_w: self.kernel_w as u32,
            stride_h: self.stride as u32,
            stride_w: self.stride as u32,
            pad_h: self.pad as u32,
            pad_w: self.pad as u32,
            kernel_cols: self.kernel_cols() as u32,
            _pad: [0; 2],
        }
    }
}

/// CPU twin of `CONV_COL2IM_EFT_TWIN_SHADER`. Returns `(v_dst, r_dst, stats,
/// exact_per_element)`.
fn cpu_col2im_twin(
    c: &Col2ImCase,
    v_gemm: &[f32],
    r_gemm: &[f32],
    mode: TwinMode,
) -> (Vec<f32>, Vec<f32>, TreeStats, Vec<f64>) {
    let flat = c.flat_input_dim();
    let total = c.num_specs * flat;
    let mut v_dst = vec![0.0f32; total];
    let mut r_dst = vec![0.0f32; total];
    let mut stats = TreeStats::default();
    let mut exact_out = Vec::with_capacity(total);
    let spatial = c.out_h * c.out_w;
    let in_hw = c.in_h * c.in_w;
    for thread_id in 0..total {
        let s = thread_id / flat;
        let flat_idx = thread_id % flat;
        let ic = flat_idx / in_hw;
        let rem = flat_idx % in_hw;
        let ih = rem / c.in_w;
        let iw = rem % c.in_w;
        let mut acc = 0.0f32;
        let mut rsum = 0.0f32;
        let mut exact = 0.0f64;
        let mut comp = 0.0f64;
        // Enumerate taps in the shipped (ki, kj) order; ReverseTaps CONTROL
        // reverses the order.
        let mut taps: Vec<usize> = Vec::new();
        for ki in 0..c.kernel_h {
            let ih_plus_ph = ih + c.pad;
            if ih_plus_ph < ki {
                continue;
            }
            let numerator_h = ih_plus_ph - ki;
            if numerator_h % c.stride != 0 {
                continue;
            }
            let gy = numerator_h / c.stride;
            if gy >= c.out_h {
                continue;
            }
            for kj in 0..c.kernel_w {
                let iw_plus_pw = iw + c.pad;
                if iw_plus_pw < kj {
                    continue;
                }
                let numerator_w = iw_plus_pw - kj;
                if numerator_w % c.stride != 0 {
                    continue;
                }
                let gx = numerator_w / c.stride;
                if gx >= c.out_w {
                    continue;
                }
                let gemm_row = s * spatial + gy * c.out_w + gx;
                let gemm_col = ic * c.kernel_h * c.kernel_w + ki * c.kernel_w + kj;
                taps.push(gemm_row * c.kernel_cols() + gemm_col);
            }
        }
        assert!(
            matches!(
                mode,
                TwinMode::Faithful
                    | TwinMode::DropChainTwoSum
                    | TwinMode::DropRGemm
                    | TwinMode::ReverseTaps
                    | TwinMode::TreeRightAssoc
            ),
            "cpu_col2im_twin: {mode:?} is not a col2im mode"
        );
        if mode == TwinMode::ReverseTaps {
            taps.reverse();
        }
        for &src in &taps {
            let v = v_gemm[src];
            let (s2, es) = barriered_two_sum(acc, v);
            let charged_es = if mode == TwinMode::DropChainTwoSum {
                0.0
            } else {
                es.abs()
            };
            let rg = if mode == TwinMode::DropRGemm {
                0.0
            } else {
                r_gemm[src]
            };
            rsum = match mode {
                // rsum + (r_gemm + |es|) — the plausible re-association.
                TwinMode::TreeRightAssoc => rsum + (rg + charged_es),
                _ => rsum + rg + charged_es,
            };
            acc = s2;
            stats.taps_total += 1;
            if es != 0.0 {
                stats.taps_nonzero_es += 1;
            }
            stats.observe(&[v, es, acc, rsum]);
            neumaier_add(&mut exact, &mut comp, f64::from(v));
        }
        v_dst[thread_id] = acc;
        r_dst[thread_id] = rsum;
        exact_out.push(exact + comp);
    }
    (v_dst, r_dst, stats, exact_out)
}

// ===========================================================================
// §6 — GPU runners + the settling verdict machinery
// ===========================================================================

/// The two observables of a tree-kernel dispatch (or lower/upper for
/// concretize, or the per-element v/r streams for col2im), as raw bits.
#[derive(Clone, PartialEq, Eq)]
struct RowWords {
    v: Vec<u32>,
    r: Vec<u32>,
}

fn words_from(v: &[f32], r: &[f32]) -> RowWords {
    RowWords {
        v: v.iter().map(|x| x.to_bits()).collect(),
        r: r.iter().map(|x| x.to_bits()).collect(),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TreeKernel {
    Bias,
    ActBias,
}

fn tree_pipeline(
    dev: &WgpuDevice,
    kernel: TreeKernel,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    match kernel {
        TreeKernel::Bias => build_pipeline(
            dev,
            BIAS_ERR_ACCUMULATE_WGSL,
            "u1_tree_bias",
            &[false, false, false, true, true],
        ),
        TreeKernel::ActBias => build_pipeline(
            dev,
            ACTIVATION_INTERCEPT_BIAS_WGSL,
            "u1_tree_act_bias",
            &[false, false, false, false, true, true],
        ),
    }
}

/// Dispatch a tree kernel exactly as production does: one workgroup per spec
/// row, params mirroring the production Pod structs (in EFT mode the act_bias
/// `gamma_k` field carries r_slack, as `mk_actbp` builds it).
fn gpu_run_tree(
    dev: &WgpuDevice,
    pipe: &(wgpu::ComputePipeline, wgpu::BindGroupLayout),
    kernel: TreeKernel,
    c: &TreeCase,
    inp: &TreeInputs,
) -> RowWords {
    let out = match kernel {
        TreeKernel::Bias => {
            let p = BiasParamsT {
                num_specs: c.num_specs as u32,
                k: c.k as u32,
                gamma_k: c.gamma_k,
                additive: c.additive,
                slack: c.slack,
                eft_mode: u32::from(c.eft_mode),
                eft_r_slack: c.r_slack,
                _p: 0,
            };
            dispatch_and_read(
                dev,
                pipe,
                bytemuck::bytes_of(&p),
                &[
                    Buf {
                        bytes: as_bytes(&inp.a),
                        rw: false,
                    },
                    Buf {
                        bytes: as_bytes(&inp.a_err),
                        rw: false,
                    },
                    Buf {
                        bytes: as_bytes(&inp.b_lo),
                        rw: false,
                    },
                    Buf {
                        bytes: as_bytes(&inp.out_pre),
                        rw: true,
                    },
                    Buf {
                        bytes: as_bytes(&inp.err_pre),
                        rw: true,
                    },
                ],
                c.num_specs as u32,
            )
        }
        TreeKernel::ActBias => {
            let p = ActBiasParamsT {
                num_specs: c.num_specs as u32,
                num_neurons: c.k as u32,
                is_upper: u32::from(c.is_upper),
                gamma_k: if c.eft_mode { c.r_slack } else { c.gamma_k },
                additive: c.additive,
                slack: c.slack,
                num_specs_per_dom: c.num_specs_per_dom as u32,
                eft_mode: u32::from(c.eft_mode),
            };
            dispatch_and_read(
                dev,
                pipe,
                bytemuck::bytes_of(&p),
                &[
                    Buf {
                        bytes: as_bytes(&inp.a),
                        rw: false,
                    },
                    Buf {
                        bytes: as_bytes(&inp.a_err),
                        rw: false,
                    },
                    Buf {
                        bytes: as_bytes(&inp.b_lo),
                        rw: false,
                    },
                    Buf {
                        bytes: as_bytes(&inp.b_hi),
                        rw: false,
                    },
                    Buf {
                        bytes: as_bytes(&inp.out_pre),
                        rw: true,
                    },
                    Buf {
                        bytes: as_bytes(&inp.err_pre),
                        rw: true,
                    },
                ],
                c.num_specs as u32,
            )
        }
    };
    RowWords {
        v: out[0].clone(),
        r: out[1].clone(),
    }
}

fn cpu_tree_twin(
    kernel: TreeKernel,
    c: &TreeCase,
    inp: &TreeInputs,
    mode: TwinMode,
) -> (Vec<f32>, Vec<f32>, TreeStats) {
    match kernel {
        TreeKernel::Bias => cpu_bias_twin(c, inp, mode),
        TreeKernel::ActBias => cpu_act_bias_twin(c, inp, mode),
    }
}

/// Drift pins per word: `Some(0)` = bit-exact HARD, `Some(n)` = ULP tripwire,
/// `None` = report-only (used where the legacy mode-0 value chain is
/// legitimately re-associable and the envelope check is the assertion instead).
#[derive(Copy, Clone)]
struct Pins {
    v_ulp: Option<i64>,
    r_ulp: Option<i64>,
}

/// The settling verdict for one case: stability, per-word drift vs the
/// faithful twin, hypothesis classification, and control rejection.
/// Returns per-control divergence counts (rows where the control's prediction
/// differs from faithful — the discriminating power the caller aggregates).
fn settle_words(
    label: &str,
    gpu: &RowWords,
    gpu2: &RowWords,
    faithful: &RowWords,
    hypotheses: &[(&'static str, RowWords)],
    controls: &[(&'static str, RowWords)],
    pins: Pins,
) -> Vec<(&'static str, usize)> {
    assert!(
        gpu == gpu2,
        "{label}: NOT run-to-run stable — a dropped/mis-placed workgroup \
         barrier or racy binding; composition verdict impossible"
    );
    let rows = gpu.v.len();
    assert_eq!(faithful.v.len(), rows, "{label}: faithful twin row count");

    let mut v_exact = 0usize;
    let mut r_exact = 0usize;
    let mut max_v_ulp = 0i64;
    let mut max_r_ulp = 0i64;
    let mut hypo_counts = vec![0usize; hypotheses.len()];
    let mut unexplained_r = 0usize;
    let mut r_explained = vec![false; rows];
    for i in 0..rows {
        if gpu.v[i] == faithful.v[i] {
            v_exact += 1;
        } else {
            let d = ulp_delta(gpu.v[i], faithful.v[i]).abs();
            max_v_ulp = max_v_ulp.max(d);
        }
        if gpu.r[i] == faithful.r[i] {
            r_exact += 1;
        } else {
            let d = ulp_delta(gpu.r[i], faithful.r[i]).abs();
            max_r_ulp = max_r_ulp.max(d);
            let mut explained = false;
            for (h, (_, hw)) in hypotheses.iter().enumerate() {
                if gpu.r[i] == hw.r[i] {
                    hypo_counts[h] += 1;
                    explained = true;
                    r_explained[i] = true;
                    break;
                }
            }
            if !explained {
                unexplained_r += 1;
            }
        }
    }

    // Controls: a GPU word matching a control WHERE THE CONTROL DIFFERS FROM
    // FAITHFUL is the smoking gun of a dropped/mis-partitioned term.
    let mut divergences: Vec<(&'static str, usize)> = Vec::new();
    let mut explained_collisions = 0usize;
    for &(name, ref cw) in controls {
        let mut divergent = 0usize;
        for i in 0..rows {
            let v_diff = cw.v[i] != faithful.v[i];
            let r_diff = cw.r[i] != faithful.r[i];
            if v_diff || r_diff {
                divergent += 1;
            }
            assert!(
                !(v_diff && gpu.v[i] == cw.v[i]),
                "{label}: GPU V word MATCHES the {name} control on row {i} \
                 (control diverges from the faithful sequence there) — the \
                 production kernel executes the degraded sequence"
            );
            // A collision on a row ALREADY explained by an accepted benign
            // hypothesis is not a smoking gun: at k ~ 2^8 a 1-ULP recombination
            // and a control partition often land on the same word. The
            // isolation tiers (T3/T4) pin the actual sequence directly.
            if r_diff && gpu.r[i] == cw.r[i] && r_explained[i] {
                explained_collisions += 1;
            }
            assert!(
                !(r_diff && gpu.r[i] == cw.r[i] && !r_explained[i]),
                "{label}: GPU R word MATCHES the {name} control on row {i} \
                 (control diverges from the faithful sequence there, and no \
                 accepted hypothesis explains the word) — the production \
                 kernel executes the degraded sequence"
            );
        }
        divergences.push((name, divergent));
    }

    print!(
        "[{label:<28}] rows={rows:<6} V exact {v_exact}/{rows} (max {max_v_ulp} ULP)  \
         R exact {r_exact}/{rows} (max {max_r_ulp} ULP)  unexplained-R {unexplained_r}\
         {}",
        if explained_collisions > 0 {
            format!(" ctl-collision-explained={explained_collisions}")
        } else {
            String::new()
        }
    );
    for (h, (name, _)) in hypotheses.iter().enumerate() {
        if hypo_counts[h] > 0 {
            print!("  {name}={}", hypo_counts[h]);
        }
    }
    for (name, d) in &divergences {
        print!("  ctl:{name}/{d}");
    }
    println!();

    if let Some(pin) = pins.v_ulp {
        assert!(
            max_v_ulp <= pin && (pin > 0 || v_exact == rows),
            "{label}: V drift {max_v_ulp} ULP exceeds the pin ({pin}) — the \
             value chain composition changed; this regression reopens U1 and \
             the channel must stay dark"
        );
    }
    if let Some(pin) = pins.r_ulp {
        assert!(
            max_r_ulp <= pin && (pin > 0 || r_exact == rows),
            "{label}: R drift {max_r_ulp} ULP exceeds the pin ({pin}) — \
             re-derive the composition before trusting the residual channel"
        );
    }
    divergences
}

/// f64 ENCLOSURE for the bias kernels: with zero `bias_err_out` preload, the
/// published word must bound `|old + exact_dot − bias_out|`. This is the
/// verdict-deciding semantic; in EFT mode it is exact (measured residuals ×
/// the U3-discharged r_slack) and asserted HARD.
fn check_tree_enclosure(
    label: &str,
    inp: &TreeInputs,
    stats: &TreeStats,
    gpu: &RowWords,
    hard: bool,
) {
    let mut violations = 0usize;
    let mut worst = f64::INFINITY;
    for s in 0..gpu.v.len() {
        if hard {
            assert_eq!(
                inp.err_pre[s], 0.0,
                "{label}: hard enclosure requires a zero bias_err_out preload"
            );
        }
        let v = f64::from(f32::from_bits(gpu.v[s]));
        let r = f64::from(f32::from_bits(gpu.r[s]));
        let truth = f64::from(inp.out_pre[s]) + stats.exact_dot[s];
        let lhs = (truth - v).abs();
        if lhs > 0.0 {
            let margin = r / lhs;
            if margin < worst {
                worst = margin;
            }
        }
        if lhs > r {
            violations += 1;
        }
    }
    println!(
        "                               ENCLOSURE |old+exact − V| <= R: \
         violations={violations} worst margin x{worst:.6}"
    );
    if hard {
        assert_eq!(
            violations, 0,
            "{label}: published radius FAILS to enclose the exact bias on \
             {violations} rows — the silent under-charge mode; hard fail"
        );
    }
}

/// Order-independent Higham envelope for the LEGACY (mode-0) value chain: the
/// plain `v = v + a·b` accumulation may be re-associated freely, but any
/// binary tree over the same term multiset satisfies
/// `|fl − exact| ≤ γ·Σ|a·b|`. Charged with a generous term count (chain taps +
/// tree adds + final add, doubled) so only a broken multiset can fail it.
fn check_mode0_value_envelope(
    label: &str,
    c: &TreeCase,
    inp: &TreeInputs,
    stats: &TreeStats,
    gpu: &RowWords,
) {
    let gamma = f64::from(gamma_k_local(2 * (c.k + 300)));
    for s in 0..gpu.v.len() {
        let v = f64::from(f32::from_bits(gpu.v[s]));
        let truth = f64::from(inp.out_pre[s]) + stats.exact_dot[s];
        let budget = gamma * (stats.sum_abs_dot[s] + truth.abs()) + 4.0 * U64 * truth.abs();
        assert!(
            (v - truth).abs() <= budget,
            "{label}: mode-0 value chain left the order-independent Higham \
             envelope on row {s} (|dev − exact| = {:.3e} > budget {:.3e}) — \
             the device is NOT summing the same term multiset",
            (v - truth).abs(),
            budget
        );
    }
}

/// Case-input generation for the composed (T1) matrix.
fn gen_tree_inputs(kernel: TreeKernel, c: &TreeCase, seed: u64, cancelling: bool) -> TreeInputs {
    let mut rng = Rng(seed | 1);
    let rows = c.num_specs;
    let k = c.k;
    let n_domains = (rows / c.num_specs_per_dom.max(1)).max(1);
    let mut a = vec![0.0f32; rows * k];
    if cancelling {
        for (i, x) in a.iter_mut().enumerate() {
            let sgn = if i % 2 == 0 { 1.0 } else { -1.0 };
            *x = sgn * (0.5 + 0.5 * rng.unit().abs()) * rng.exp_scale(-10, 8);
        }
    } else {
        for x in a.iter_mut() {
            *x = rng.unit() * rng.exp_scale(-6, 4);
        }
    }
    let mut a_err = vec![0.0f32; rows * k];
    for x in a_err.iter_mut() {
        *x = rng.unit().abs() * rng.exp_scale(-30, -20);
    }
    let blen = match kernel {
        TreeKernel::Bias => k,
        TreeKernel::ActBias => n_domains * k,
    };
    let mut b_lo = vec![0.0f32; blen];
    for x in b_lo.iter_mut() {
        *x = rng.unit() * rng.exp_scale(-8, 2);
    }
    let b_hi = match kernel {
        TreeKernel::Bias => Vec::new(),
        TreeKernel::ActBias => {
            let mut v = vec![0.0f32; blen];
            for x in v.iter_mut() {
                *x = rng.unit() * rng.exp_scale(-4, 2);
            }
            v
        }
    };
    let mut out_pre = vec![0.0f32; rows];
    for x in out_pre.iter_mut() {
        *x = rng.unit();
    }
    TreeInputs {
        a,
        a_err,
        b_lo,
        b_hi,
        out_pre,
        err_pre: vec![0.0f32; rows],
    }
}

/// Run one composed tree case end to end; returns per-control divergence
/// counts for the caller's aggregate discriminating-power assertion.
fn settle_tree_case(
    dev: &WgpuDevice,
    pipe: &(wgpu::ComputePipeline, wgpu::BindGroupLayout),
    kernel: TreeKernel,
    c: &TreeCase,
    inp: &TreeInputs,
) -> Vec<(&'static str, usize)> {
    let gpu = gpu_run_tree(dev, pipe, kernel, c, inp);
    let gpu2 = gpu_run_tree(dev, pipe, kernel, c, inp);
    let (fv, fr, stats) = cpu_tree_twin(kernel, c, inp, TwinMode::Faithful);
    let faithful = words_from(&fv, &fr);

    let (hypo_modes, control_modes): (&[(&'static str, TwinMode)], &[(&'static str, TwinMode)]) =
        if c.eft_mode {
            (
                &[
                    ("tree-right-assoc", TwinMode::TreeRightAssoc),
                    ("chain-ev-fused", TwinMode::ChainBodyFusedEv),
                    ("tail-fused", TwinMode::TailFused),
                    ("tail-outer-right", TwinMode::TailOuterRightAssoc),
                ],
                &[
                    ("drop-chain-twosum", TwinMode::DropChainTwoSum),
                    ("drop-tree-residual", TwinMode::DropTreeResidual),
                    ("contiguous-taps", TwinMode::ContiguousTaps),
                ],
            )
        } else {
            (
                &[
                    ("chain-ev-fused", TwinMode::ChainBodyFusedEv),
                    ("tail-fused", TwinMode::TailFused),
                    ("tail-outer-right", TwinMode::TailOuterRightAssoc),
                ],
                &[("contiguous-taps", TwinMode::ContiguousTaps)],
            )
        };
    let hypotheses: Vec<(&'static str, RowWords)> = hypo_modes
        .iter()
        .map(|&(n, m)| {
            let (v, r, _) = cpu_tree_twin(kernel, c, inp, m);
            (n, words_from(&v, &r))
        })
        .collect();
    let controls: Vec<(&'static str, RowWords)> = control_modes
        .iter()
        .map(|&(n, m)| {
            let (v, r, _) = cpu_tree_twin(kernel, c, inp, m);
            (n, words_from(&v, &r))
        })
        .collect();

    let pins = if c.eft_mode {
        // The EFT value chain is barriered (no reassociation freedom) — V is
        // bit-exact or the composition changed. R rides the GEMM twin's
        // measured 8-ULP ceiling.
        Pins {
            v_ulp: Some(0),
            r_ulp: Some(PINNED_MAX_ULP_DRIFT),
        }
    } else {
        // The legacy value chain is a plain reduction — the envelope check
        // below is the assertion; ULP pins would assert more than the shipped
        // WGSL promises.
        Pins {
            v_ulp: None,
            r_ulp: None,
        }
    };
    let divergences = settle_words(
        &c.name,
        &gpu,
        &gpu2,
        &faithful,
        &hypotheses,
        &controls,
        pins,
    );
    println!(
        "                               taps={t} ep!=0 {ep} es!=0 {es} \
         floor-charged={fc} subnormal-intermediates={sn}",
        t = stats.taps_total,
        ep = stats.taps_nonzero_ep,
        es = stats.taps_nonzero_es,
        fc = stats.taps_floor_charged,
        sn = stats.subnormal_intermediates,
    );

    if c.eft_mode {
        // Normal-range certification (canon: the GEMM probe's Family::Normal
        // discipline): on this box the plain WGSL path FLUSHES subnormals
        // (rung 3), so a subnormal intermediate would confound the flush
        // question with the composition question — and would void the V
        // bit-exact assertion for a reason that is NOT a composition failure.
        // The operand generators are ranged so this never fires; if it does,
        // fix the generator, not the assertion.
        assert_eq!(
            stats.subnormal_intermediates, 0,
            "{}: case is not normal-range; composition and flush are confounded",
            c.name
        );
        // Discriminating power: with random data every row must carry a
        // nonzero published radius, else the case proves nothing.
        for s in 0..c.num_specs {
            assert!(
                f32::from_bits(gpu.r[s]) > 0.0,
                "{}: row {s} published a zero radius — vacuous case",
                c.name
            );
        }
        check_tree_enclosure(&c.name, inp, &stats, &gpu, true);
    } else {
        check_mode0_value_envelope(&c.name, c, inp, &stats, &gpu);
        check_tree_enclosure(&c.name, inp, &stats, &gpu, false);
    }
    // The drop-chain-TwoSum control is only meaningful where the sequence HAS
    // TwoSum residuals (mirrors the GEMM probe's k=1 guard).
    if c.eft_mode && stats.taps_nonzero_es > 0 {
        let d = divergences
            .iter()
            .find(|(n, _)| *n == "drop-chain-twosum")
            .map_or(0, |(_, d)| *d);
        assert!(
            d > 0,
            "{}: drop-chain-twosum control is indistinguishable from the \
             faithful sequence despite {} nonzero TwoSum residuals — the \
             compare is not discriminating",
            c.name,
            stats.taps_nonzero_es
        );
    }
    divergences
}

// ---------------------------------------------------------------------------
// Concretize + col2im runners
// ---------------------------------------------------------------------------

fn concretize_pipeline(dev: &WgpuDevice) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    build_pipeline(
        dev,
        CONCRETIZE_SOUND_WGSL,
        "u1_tree_concretize",
        &[false, false, false, false, false, true, true, false],
    )
}

/// Dispatch the sound concretize exactly as production does (one workgroup per
/// spec row). Outputs are sentinel-prefilled: a silently no-op'd dispatch (the
/// async bind-group trap) reads back as a mismatch, never as agreement.
fn gpu_run_concretize(
    dev: &WgpuDevice,
    pipe: &(wgpu::ComputePipeline, wgpu::BindGroupLayout),
    c: &ConcCase,
    inp: &ConcInputs,
) -> RowWords {
    let sent = vec![UNWRITTEN_SENTINEL; c.num_specs];
    let p = ConcParamsT {
        num_specs: c.num_specs as u32,
        input_dim: c.input_dim as u32,
        gamma_n: c.gamma_n,
        additive: c.additive,
        slack: c.slack,
        num_specs_per_dom: c.num_specs_per_dom as u32,
        eft_mode: u32::from(c.eft_mode),
        eft_r_slack: c.eft_r_slack,
    };
    let out = dispatch_and_read(
        dev,
        pipe,
        bytemuck::bytes_of(&p),
        &[
            Buf {
                bytes: as_bytes(&inp.a_lower),
                rw: false,
            },
            Buf {
                bytes: as_bytes(&inp.a_upper),
                rw: false,
            },
            Buf {
                bytes: as_bytes(&inp.input_lower),
                rw: false,
            },
            Buf {
                bytes: as_bytes(&inp.input_upper),
                rw: false,
            },
            Buf {
                bytes: as_bytes(&inp.bias),
                rw: false,
            },
            Buf {
                bytes: bytemuck::cast_slice(&sent),
                rw: true,
            },
            Buf {
                bytes: bytemuck::cast_slice(&sent),
                rw: true,
            },
            Buf {
                bytes: as_bytes(&inp.a_err),
                rw: false,
            },
        ],
        c.num_specs as u32,
    );
    RowWords {
        v: out[0].clone(),
        r: out[1].clone(),
    }
}

fn assert_no_sentinel(label: &str, words: &RowWords) {
    for (i, w) in words.v.iter().chain(words.r.iter()).enumerate() {
        assert!(
            *w != UNWRITTEN_SENTINEL,
            "{label}: output word {i} still carries the unwritten sentinel — \
             the dispatch silently no-op'd (bind-group/limits trap); refusing \
             to treat it as agreement"
        );
    }
}

/// Enclosure for concretize: the published interval must contain the exact
/// affine bounds (`lb ≤ exact_lb`, `ub ≥ exact_ub`) — the verdict-facing claim.
fn check_concretize_enclosure(label: &str, gpu: &RowWords, exact: &ConcExact, hard: bool) {
    let mut violations = 0usize;
    let mut worst_slack = 0.0f64;
    for s in 0..gpu.v.len() {
        let lo = f64::from(f32::from_bits(gpu.v[s]));
        let hi = f64::from(f32::from_bits(gpu.r[s]));
        if lo > exact.lb[s] || hi < exact.ub[s] {
            violations += 1;
        }
        worst_slack = worst_slack.max(exact.lb[s] - lo).max(hi - exact.ub[s]);
    }
    println!(
        "                               ENCLOSURE lb<=exact<=ub: \
         violations={violations} worst outward slack {worst_slack:.3e}"
    );
    if hard {
        assert_eq!(
            violations, 0,
            "{label}: published interval FAILS to enclose the exact affine \
             bounds on {violations} rows — hard fail"
        );
    }
}

fn gen_conc_inputs(c: &ConcCase, seed: u64) -> ConcInputs {
    let mut rng = Rng(seed | 1);
    let n = c.input_dim;
    let rows = c.num_specs;
    let n_domains = (rows / c.num_specs_per_dom.max(1)).max(1);
    let mut a_lower = vec![0.0f32; rows * n];
    let mut a_upper = vec![0.0f32; rows * n];
    for x in a_lower.iter_mut() {
        *x = rng.unit() * rng.exp_scale(-6, 3);
    }
    for x in a_upper.iter_mut() {
        *x = rng.unit() * rng.exp_scale(-6, 3);
    }
    let mut a_err = vec![0.0f32; 2 * rows * n];
    for x in a_err.iter_mut() {
        *x = rng.unit().abs() * rng.exp_scale(-30, -22);
    }
    let mut input_lower = vec![0.0f32; n_domains * n];
    let mut input_upper = vec![0.0f32; n_domains * n];
    for j in 0..n_domains * n {
        let scale = rng.exp_scale(-3, 0);
        input_lower[j] = -(0.2 + rng.unit().abs()) * scale;
        input_upper[j] = (0.2 + rng.unit().abs()) * scale;
    }
    let mut bias = vec![0.0f32; 2 * rows];
    for x in bias.iter_mut() {
        *x = rng.unit();
    }
    ConcInputs {
        a_lower,
        a_upper,
        input_lower,
        input_upper,
        bias,
        a_err,
    }
}

/// Run one concretize case end to end (compare, controls, enclosure).
fn settle_concretize_case(
    dev: &WgpuDevice,
    pipe: &(wgpu::ComputePipeline, wgpu::BindGroupLayout),
    c: &ConcCase,
    inp: &ConcInputs,
    pins: Pins,
) {
    let gpu = gpu_run_concretize(dev, pipe, c, inp);
    let gpu2 = gpu_run_concretize(dev, pipe, c, inp);
    assert_no_sentinel(&c.name, &gpu);
    let (flo, fhi, stats, exact) = cpu_concretize_twin(c, inp, TwinMode::Faithful);
    let faithful = words_from(&flo, &fhi);

    let (hypo_modes, control_modes): (&[(&'static str, TwinMode)], &[(&'static str, TwinMode)]) =
        if c.eft_mode {
            (
                &[
                    ("tree-right-assoc", TwinMode::TreeRightAssoc),
                    ("chain-pen-fused", TwinMode::ChainBodyFusedEv),
                ],
                &[
                    ("drop-chain-twosum", TwinMode::DropChainTwoSum),
                    ("drop-tree-residual", TwinMode::DropTreeResidual),
                ],
            )
        } else {
            (&[], &[])
        };
    let hypotheses: Vec<(&'static str, RowWords)> = hypo_modes
        .iter()
        .map(|&(n2, m)| {
            let (lo, hi, _, _) = cpu_concretize_twin(c, inp, m);
            (n2, words_from(&lo, &hi))
        })
        .collect();
    let controls: Vec<(&'static str, RowWords)> = control_modes
        .iter()
        .map(|&(n2, m)| {
            let (lo, hi, _, _) = cpu_concretize_twin(c, inp, m);
            (n2, words_from(&lo, &hi))
        })
        .collect();

    let divergences = settle_words(
        &c.name,
        &gpu,
        &gpu2,
        &faithful,
        &hypotheses,
        &controls,
        pins,
    );
    if c.eft_mode {
        // Normal-range certification — see settle_tree_case: a subnormal
        // intermediate would confound this box's measured rung-3 flush with
        // the composition question this probe exists to settle.
        assert_eq!(
            stats.subnormal_intermediates, 0,
            "{}: case is not normal-range; composition and flush are confounded",
            c.name
        );
    }
    check_concretize_enclosure(&c.name, &gpu, &exact, true);

    if c.eft_mode && stats.taps_nonzero_es > 0 {
        let d = divergences
            .iter()
            .find(|(n2, _)| *n2 == "drop-chain-twosum")
            .map_or(0, |(_, d)| *d);
        assert!(
            d > 0,
            "{}: drop-chain-twosum control indistinguishable despite {} \
             nonzero TwoSum residuals — not discriminating",
            c.name,
            stats.taps_nonzero_es
        );
    }
}

// ===========================================================================
// §7 — CPU-only pins (helpers + twins), then the settling tests
// ===========================================================================

/// The local sound-consts transcriptions carry real weight (the enclosure
/// budgets); pin their defining properties.
#[test]
fn local_helpers_are_pinned() {
    // round_up_pos: transcription pins.
    assert_eq!(round_up_pos(0.0).to_bits(), 0.0f32.to_bits());
    assert_eq!(round_up_pos(-1.0).to_bits(), 0.0f32.to_bits());
    assert_eq!(round_up_pos(1.0).to_bits(), 1.0f32.to_bits() + 1);
    assert_eq!(
        round_up_pos(f32::from_bits(1)).to_bits(),
        F32_MIN_NORMAL.to_bits(),
        "positive subnormal floors to F32_MIN_NORMAL"
    );
    assert_eq!(
        round_up_pos(f32::INFINITY).to_bits(),
        f32::INFINITY.to_bits()
    );
    // next_down/up: transcription pins.
    assert_eq!(
        next_down_f32_normal(1.0).to_bits(),
        f32::from_bits(1.0f32.to_bits() - 1).to_bits()
    );
    assert_eq!(
        next_up_f32_normal(1.0).to_bits(),
        f32::from_bits(1.0f32.to_bits() + 1).to_bits()
    );
    assert_eq!(
        next_down_f32_normal(0.0).to_bits(),
        (-F32_MIN_NORMAL).to_bits()
    );
    assert_eq!(next_up_f32_normal(0.0).to_bits(), F32_MIN_NORMAL.to_bits());
    // up_f32_local: outward property.
    for x in [1.0000001f64, 1.0 / (1.0 - 1e-6), 3.5e-7, 1.9999999] {
        let y = up_f32_local(x);
        assert!(f64::from(y) >= x, "up_f32_local({x}) = {y} landed below");
    }
    // eft_r_slack_local: mirrors eft_r_slack_f32's structure — greater than
    // the pure inverse-gamma recovery, monotone in k, > 1.
    let mut prev = 1.0f32;
    for k in [1usize, 15, 256, 3072, 14400] {
        let s = eft_r_slack_local(k);
        let terms = (2 * k + 2 + TREE_REDUCTION_RESIDUAL_ADDS) as f64;
        let gamma = terms * U64 / (1.0 - terms * U64);
        assert!(
            f64::from(s) >= 1.0 / (1.0 - gamma),
            "k={k}: slack below the bare gamma recovery"
        );
        assert!(s >= prev, "slack must be monotone in k");
        prev = s;
    }
    // The host EFT primitives this file's twins rely on.
    ny_core::eft::eft_self_check().expect("host EFT reference must be sound");
    let (p, e) =
        ny_core::eft::two_prod_f32(f32::from_bits(0x3F80_0800), f32::from_bits(0x3F80_0800));
    assert_eq!(p.to_bits(), (1.0f32 + 2.0f32.powi(-11)).to_bits());
    assert_eq!(e.to_bits(), 0x3380_0000, "2^-24 (the GEMM probe's pin)");
}

/// The CPU twins must be faithful transcriptions: pinned scalar expectations,
/// independent of any GPU.
#[test]
fn cpu_twins_are_pinned_transcriptions() {
    ny_core::eft::eft_self_check().expect("host EFT reference must be sound");
    let val = f32::from_bits(0x3F80_0800); // 1 + 2^-12
    let rs = eft_r_slack_local(1);

    // Bias twin, k=1: V = TwoProd value. The local radius is
    // round_up_pos(|2^-24| * rs), then combined outward with the zero
    // propagated-error lane before the complete scalar publication is staged
    // outward (chain TwoSum residual and rf are exactly 0; flush zeroed).
    let c = TreeCase {
        name: "pin-bias-k1".into(),
        num_specs: 1,
        k: 1,
        additive: 0.0,
        slack: 0.0,
        r_slack: rs,
        gamma_k: 0.0,
        eft_mode: true,
        is_upper: false,
        num_specs_per_dom: 1,
    };
    let inp = TreeInputs {
        a: vec![val],
        a_err: vec![0.0],
        b_lo: vec![val],
        b_hi: Vec::new(),
        out_pre: vec![0.0],
        err_pre: vec![0.0],
    };
    let (v, r, stats) = cpu_bias_twin(&c, &inp, TwinMode::Faithful);
    let (p, e) = ny_core::eft::two_prod_f32(val, val);
    assert_eq!(v[0].to_bits(), p.to_bits());
    let local = round_up_pos(round_up_pos(e.abs() * rs));
    assert_eq!(
        r[0].to_bits(),
        outward_bias_update(0.0, local, 0.0).to_bits()
    );
    assert_eq!(stats.taps_floor_charged, 0);
    assert_eq!(stats.subnormal_intermediates, 0);

    // Act-bias twin with li = ui = val selects sel = val for either sign of a
    // and must agree with the bias twin numbers exactly.
    let inp_act = TreeInputs {
        a: vec![val],
        a_err: vec![0.0],
        b_lo: vec![val],
        b_hi: vec![val],
        out_pre: vec![0.0],
        err_pre: vec![0.0],
    };
    let (v2, r2, _) = cpu_act_bias_twin(&c, &inp_act, TwinMode::Faithful);
    assert_eq!(v2[0].to_bits(), v[0].to_bits());
    assert_eq!(r2[0].to_bits(), r[0].to_bits());

    // T4 tail algebra: with old = 1.0 and a single 2^-25 tap, rf = 2^-25 and
    // the recovered residual is round_up_pos(rf * rs), followed by an outward
    // combine with the zero propagated lane and staged scalar publication.
    let c4 = TreeCase {
        name: "pin-bias-tail".into(),
        num_specs: 1,
        k: 1,
        additive: 0.0,
        slack: 0.0,
        r_slack: rs,
        gamma_k: 0.0,
        eft_mode: true,
        is_upper: false,
        num_specs_per_dom: 1,
    };
    let rf = 2.0f32.powi(-25);
    let inp4 = TreeInputs {
        a: vec![rf],
        a_err: vec![0.0],
        b_lo: vec![1.0],
        b_hi: Vec::new(),
        out_pre: vec![1.0],
        err_pre: vec![0.0],
    };
    let (v4, r4, _) = cpu_bias_twin(&c4, &inp4, TwinMode::Faithful);
    assert_eq!(v4[0].to_bits(), 1.0f32.to_bits(), "fl(1 + 2^-25) = 1.0");
    let local4 = round_up_pos(round_up_pos(rf * rs));
    assert_eq!(
        r4[0].to_bits(),
        outward_bias_update(0.0, local4, 0.0).to_bits()
    );

    // col2im twin, single-tap geometry: v = fl(0 + v0) with zero residual,
    // r = r_gemm passthrough.
    let cc = Col2ImCase {
        name: "pin-col2im",
        num_specs: 1,
        in_channels: 1,
        in_h: 1,
        in_w: 1,
        kernel_h: 1,
        kernel_w: 1,
        stride: 1,
        pad: 0,
        out_h: 1,
        out_w: 1,
    };
    let (vc, rc, _, _) = cpu_col2im_twin(&cc, &[1.5], &[0.25], TwinMode::Faithful);
    assert_eq!(vc[0].to_bits(), 1.5f32.to_bits());
    assert_eq!(rc[0].to_bits(), 0.25f32.to_bits());

    // Padded/absent taps are inert: the drop-controls must equal Faithful
    // exactly when there is nothing for them to drop (k=1 exact single tap).
    let (vd, rd, _) = cpu_bias_twin(&c, &inp, TwinMode::DropChainTwoSum);
    assert_eq!(vd[0].to_bits(), v[0].to_bits());
    assert_eq!(rd[0].to_bits(), r[0].to_bits());
}

/// CROWN-shaped composed matrix for the tree kernels.
const T1_KS: &[usize] = &[1, 15, 255, 256, 257, 1000, 3072, 4096, 14400];
const T1_SPECS: &[usize] = &[1, 9, 32];

fn t1_case(kernel: TreeKernel, k: usize, specs: usize, eft: bool) -> TreeCase {
    TreeCase {
        name: format!(
            "{kernel:?}-{}-k{k}-s{specs}",
            if eft { "eft" } else { "m0" }
        ),
        num_specs: specs,
        k,
        additive: 1e-30,
        slack: 1.000001,
        r_slack: eft_r_slack_local(k),
        gamma_k: gamma_k_local(k),
        eft_mode: eft,
        is_upper: false,
        num_specs_per_dom: specs,
    }
}

fn run_t1_matrix(kernel: TreeKernel) {
    let _g = serial_guard();
    let dev = require_device();
    ny_core::eft::eft_self_check().expect("host EFT reference must be sound");
    if let Some(p) = WgpuDevice::probe_adapter() {
        println!(
            "\n=== #u1 tree settling ({kernel:?}), adapter: {} / {} ===",
            p.name, p.backend
        );
    }
    let pipe = tree_pipeline(dev, kernel);
    let mut agg_tree = 0usize;
    let mut agg_contig = 0usize;
    let mut seed = 0x5EED_0001u64;
    for &k in T1_KS {
        for &specs in T1_SPECS {
            let cancelling = matches!(k, 255 | 1000 | 4096 | 14400);
            let c = t1_case(kernel, k, specs, true);
            let inp = gen_tree_inputs(kernel, &c, seed, cancelling);
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let div = settle_tree_case(dev, &pipe, kernel, &c, &inp);
            for (n2, d) in div {
                if n2 == "drop-tree-residual" {
                    agg_tree += d;
                }
                if n2 == "contiguous-taps" {
                    agg_contig += d;
                }
            }
        }
    }
    // Batched-BaB partition (act_bias only: the bias kernel has no domain
    // indexing) — a wrong `sbase` read shows as a V/enclosure mismatch.
    if kernel == TreeKernel::ActBias {
        let mut c = t1_case(kernel, 3072, 32, true);
        c.num_specs_per_dom = 8;
        c.name = format!("{}-dom8", c.name);
        let inp = gen_tree_inputs(kernel, &c, seed, false);
        seed = seed.wrapping_add(0x9E37_79B9);
        settle_tree_case(dev, &pipe, kernel, &c, &inp);
        // And the upper-side selection.
        let mut cu = t1_case(kernel, 1000, 9, true);
        cu.is_upper = true;
        cu.name = format!("{}-upper", cu.name);
        let inp_u = gen_tree_inputs(kernel, &cu, seed, true);
        seed = seed.wrapping_add(0x9E37_79B9);
        settle_tree_case(dev, &pipe, kernel, &cu, &inp_u);
    }
    // Legacy mode-0 supplement (the LIVE arm): same tree, different R word;
    // the plain value chain is re-associable, so the envelope check inside
    // settle_tree_case carries the assertion instead of ULP pins.
    for &(k, specs) in &[(257usize, 9usize), (3072, 9)] {
        let c = t1_case(kernel, k, specs, false);
        let inp = gen_tree_inputs(kernel, &c, seed, false);
        seed = seed.wrapping_add(0x9E37_79B9);
        settle_tree_case(dev, &pipe, kernel, &c, &inp);
    }
    assert!(
        agg_tree > 0,
        "{kernel:?}: the drop-tree-residual control never diverged from the \
         faithful sequence across the whole matrix — the matrix cannot \
         discriminate the U3 +16 tree charge"
    );
    assert!(
        agg_contig > 0,
        "{kernel:?}: the contiguous-taps control never diverged — the matrix \
         cannot discriminate the strided partition"
    );
}

/// `#u1` T1: the bias tree kernel, composed, at CROWN shapes.
#[test]
fn u1_tree_bias_composed_settles_the_reduction() {
    run_t1_matrix(TreeKernel::Bias);
}

/// `#u1` T1: the activation-intercept tree kernel, composed, at CROWN shapes
/// (plus the batched-BaB domain partition and the upper-side selection).
#[test]
fn u1_tree_act_bias_composed_settles_the_reduction() {
    run_t1_matrix(TreeKernel::ActBias);
}

// ---------------------------------------------------------------------------
// T2 — chain isolation
// ---------------------------------------------------------------------------

/// One nonzero lane `t0`, chain length `len`, operands engineered so the ONLY
/// nonzero residuals are the chain's TwoProduct residuals — distinct powers of
/// two, so the R word is association-independent and bit-inequality is HARD.
///
/// Construction: taps `a = (1+2^-12)·4^i`, `b = (1+2^-12)`. Each product's RN
/// result is `(1+2^-11)·4^i` with residual exactly `2^(2i−24)`; every chain
/// partial sum needs ≤ 20 mantissa bits (exact ⇒ zero TwoSum residuals); the
/// residual accumulation sums ≤ 5 distinct powers of two spanning 8 bits
/// (exact under ANY association). All other lanes carry exact zeros.
fn t2_case(kernel: TreeKernel, t0: usize, chain_len: usize) -> (TreeCase, TreeInputs, f32) {
    let k = 256 * (chain_len - 1) + t0 + 1;
    let val = f32::from_bits(0x3F80_0800); // 1 + 2^-12
    let mut a = vec![0.0f32; k];
    let mut b = vec![1.0f32; k];
    let mut expected_rv = 0.0f32;
    for i in 0..chain_len {
        let j = t0 + 256 * i;
        a[j] = val * 2.0f32.powi(2 * i as i32);
        b[j] = val;
        expected_rv += 2.0f32.powi(2 * i as i32 - 24);
    }
    let c = TreeCase {
        name: format!("{kernel:?}-t2-lane{t0}-len{chain_len}"),
        num_specs: 1,
        k,
        additive: 0.0,
        slack: 0.0,
        r_slack: eft_r_slack_local(k),
        gamma_k: 0.0,
        eft_mode: true,
        is_upper: false,
        num_specs_per_dom: 1,
    };
    let b_hi = if kernel == TreeKernel::ActBias {
        b.clone()
    } else {
        Vec::new()
    };
    let inp = TreeInputs {
        a,
        a_err: vec![0.0; k],
        b_lo: b,
        b_hi,
        out_pre: vec![0.0],
        err_pre: vec![0.0],
    };
    (c, inp, expected_rv)
}

/// `#u1` T2: a mismatch here NAMES the strided chain (lane, length) — no
/// benign-reassociation escape exists because the residual multiset is
/// engineered association-independent.
#[test]
fn u1_tree_chain_isolation_names_the_strided_chain() {
    let _g = serial_guard();
    let dev = require_device();
    ny_core::eft::eft_self_check().expect("host EFT reference must be sound");
    for kernel in [TreeKernel::Bias, TreeKernel::ActBias] {
        let pipe = tree_pipeline(dev, kernel);
        for &t0 in &[0usize, 1, 17, 255] {
            for &len in &[1usize, 2, 5] {
                let (c, inp, expected_rv) = t2_case(kernel, t0, len);
                let (fv, fr, stats) = cpu_tree_twin(kernel, &c, &inp, TwinMode::Faithful);
                // Preconditions (fail closed if violated): the construction
                // must be exact — no floor charges, no subnormal traffic.
                assert_eq!(
                    stats.taps_floor_charged, 0,
                    "{}: T2 construction hit the exactness floor",
                    c.name
                );
                assert_eq!(
                    stats.subnormal_intermediates, 0,
                    "{}: T2 construction went subnormal",
                    c.name
                );
                // Construction pin: the residual stream is exactly the
                // engineered powers of two.
                assert_eq!(
                    fr[0].to_bits(),
                    outward_bias_update(
                        0.0,
                        round_up_pos(round_up_pos(expected_rv * c.r_slack)),
                        0.0,
                    )
                    .to_bits(),
                    "{}: CPU twin does not reproduce the engineered residual",
                    c.name
                );
                assert!(fr[0] > 0.0, "{}: vacuous T2 case", c.name);

                let gpu = gpu_run_tree(dev, &pipe, kernel, &c, &inp);
                let gpu2 = gpu_run_tree(dev, &pipe, kernel, &c, &inp);
                assert!(gpu == gpu2, "{}: not run-to-run stable", c.name);
                assert_eq!(
                    gpu.v[0],
                    fv[0].to_bits(),
                    "{}: VALUE mismatch — the strided chain for lane {t0} \
                     (length {len}) compiled to a different sequence",
                    c.name
                );
                assert_eq!(
                    gpu.r[0],
                    fr[0].to_bits(),
                    "{}: RESIDUAL mismatch on an association-independent \
                     stream — lane {t0}'s chain (length {len}) drops or \
                     alters a TwoProduct residual",
                    c.name
                );
            }
        }
        println!("[{kernel:?}-t2] 12/12 chain isolations bit-exact");
    }
}

// ---------------------------------------------------------------------------
// T3 — pair meeting-level isolation
// ---------------------------------------------------------------------------

/// `(stride, offset, use_value_set_B)`: all 15 distinct `(a, a+s)` pairs with
/// `a < s` at two offsets per level (stride 1 admits only (0,1); it runs twice
/// with different values). The msb rule makes level `stride` the unique
/// meeting point of the pair, so its TwoSum residual is the only nonzero
/// residual in the dispatch.
const T3_PAIRS: &[(usize, usize, bool)] = &[
    (128, 0, false),
    (128, 127, true),
    (64, 0, false),
    (64, 63, true),
    (32, 0, false),
    (32, 31, true),
    (16, 0, false),
    (16, 15, true),
    (8, 0, false),
    (8, 7, true),
    (4, 0, false),
    (4, 3, true),
    (2, 0, false),
    (2, 1, true),
    (1, 0, false),
    (1, 0, true),
];

/// Pair values: `x + y` rounds (y below half-ULP of x) with residual exactly
/// `y` — power-of-two operands keep every other op exact.
fn t3_row_values(set_b: bool) -> (f32, f32) {
    if set_b {
        (1.5, 1.5 * 2.0f32.powi(-25)) // residual 3·2^-26
    } else {
        (1.0, 2.0f32.powi(-25))
    }
}

/// `#u1` T3 for the two bias tree kernels: each row of one 16-row dispatch
/// isolates one tree level; a failing row NAMES the level.
#[test]
fn u1_tree_pair_isolation_names_the_reduction_level() {
    let _g = serial_guard();
    let dev = require_device();
    ny_core::eft::eft_self_check().expect("host EFT reference must be sound");
    let rows = T3_PAIRS.len();
    let k = 256usize;
    let rs = eft_r_slack_local(k);
    for kernel in [TreeKernel::Bias, TreeKernel::ActBias] {
        let pipe = tree_pipeline(dev, kernel);
        let mut a = vec![0.0f32; rows * k];
        for (r, &(stride, off, set_b)) in T3_PAIRS.iter().enumerate() {
            let (x, y) = t3_row_values(set_b);
            a[r * k + off] = x;
            a[r * k + off + stride] = y;
        }
        let b = vec![1.0f32; k];
        let c = TreeCase {
            name: format!("{kernel:?}-t3-pairs"),
            num_specs: rows,
            k,
            additive: 0.0,
            slack: 0.0,
            r_slack: rs,
            gamma_k: 0.0,
            eft_mode: true,
            is_upper: false,
            num_specs_per_dom: rows,
        };
        let inp = TreeInputs {
            a,
            a_err: vec![0.0; rows * k],
            b_lo: b.clone(),
            b_hi: if kernel == TreeKernel::ActBias {
                b.clone()
            } else {
                Vec::new()
            },
            out_pre: vec![0.0; rows],
            err_pre: vec![0.0; rows],
        };
        let (fv, fr, stats) = cpu_tree_twin(kernel, &c, &inp, TwinMode::Faithful);
        assert_eq!(stats.taps_floor_charged, 0, "{}: floor charged", c.name);
        assert_eq!(stats.subnormal_intermediates, 0, "{}: subnormal", c.name);
        // Construction pin per row: the only residual is the pair's r0 = y.
        for (r, &(stride, off, set_b)) in T3_PAIRS.iter().enumerate() {
            let (x, y) = t3_row_values(set_b);
            assert_eq!(
                fv[r].to_bits(),
                (x + y).to_bits(),
                "{}: row {r} (stride {stride}, offset {off}) value construction",
                c.name
            );
            assert_eq!(
                fr[r].to_bits(),
                outward_bias_update(0.0, round_up_pos(round_up_pos(y * rs)), 0.0).to_bits(),
                "{}: row {r} (stride {stride}, offset {off}) residual \
                 construction — the pair's r0 must be the only term",
                c.name
            );
        }
        let gpu = gpu_run_tree(dev, &pipe, kernel, &c, &inp);
        let gpu2 = gpu_run_tree(dev, &pipe, kernel, &c, &inp);
        assert!(gpu == gpu2, "{}: not run-to-run stable", c.name);
        for (r, &(stride, off, _)) in T3_PAIRS.iter().enumerate() {
            assert_eq!(
                gpu.v[r],
                fv[r].to_bits(),
                "{}: VALUE mismatch at tree level stride={stride} \
                 (pair ({off}, {})) — the value tree's TwoSum at that level \
                 compiled differently",
                c.name,
                off + stride
            );
            assert_eq!(
                gpu.r[r],
                fr[r].to_bits(),
                "{}: RESIDUAL mismatch at tree level stride={stride} \
                 (pair ({off}, {})) — the shipped 3-addend residual update at \
                 that level drops or alters r0",
                c.name,
                off + stride
            );
        }
        println!("[{kernel:?}-t3] 16/16 tree-level isolations bit-exact");
    }

    // Concretize: same pair patterns on the lower AND upper coefficient rows;
    // the words are deterministic (all uniforms zeroed, unit box, zero bias).
    let conc_pipe = concretize_pipeline(dev);
    let crs = eft_r_slack_local(2 * k);
    let mut a_low = vec![0.0f32; rows * k];
    for (r, &(stride, off, set_b)) in T3_PAIRS.iter().enumerate() {
        let (x, y) = t3_row_values(set_b);
        a_low[r * k + off] = x;
        a_low[r * k + off + stride] = y;
    }
    let cc = ConcCase {
        name: "Concretize-t3-pairs".into(),
        num_specs: rows,
        input_dim: k,
        gamma_n: 0.0,
        additive: 0.0,
        slack: 0.0,
        num_specs_per_dom: rows,
        eft_mode: true,
        eft_r_slack: crs,
    };
    let cinp = ConcInputs {
        a_lower: a_low.clone(),
        a_upper: a_low,
        input_lower: vec![1.0; k],
        input_upper: vec![1.0; k],
        bias: vec![0.0; 2 * rows],
        a_err: vec![0.0; 2 * rows * k],
    };
    let (flo, fhi, cstats, _) = cpu_concretize_twin(&cc, &cinp, TwinMode::Faithful);
    assert_eq!(cstats.taps_floor_charged, 0, "{}: floor charged", cc.name);
    // Construction pin: the residual path reduces to round_up_pos chains over
    // the pair's r0 alone.
    for (r, &(stride, off, set_b)) in T3_PAIRS.iter().enumerate() {
        let (x, y) = t3_row_values(set_b);
        // Tail algebra with zeroed uniforms: prop = flush = 0, so the penalty
        // reduces to two round_up_pos steps over the pair's r0 charge.
        let resid = round_up_pos(y * crs);
        let pen = round_up_pos(round_up_pos(0.0f32 + resid) + 0.0);
        let cl = next_down_f32_normal((x + y) + 0.0);
        let cu = next_up_f32_normal((x + y) + 0.0);
        let lb = next_down_f32_normal(cl - pen);
        let ub = next_up_f32_normal(cu + pen);
        assert_eq!(
            flo[r].to_bits(),
            lb.to_bits(),
            "{}: row {r} (stride {stride}, offset {off}) lower-word construction",
            cc.name
        );
        assert_eq!(
            fhi[r].to_bits(),
            ub.to_bits(),
            "{}: row {r} (stride {stride}, offset {off}) upper-word construction",
            cc.name
        );
    }
    let cgpu = gpu_run_concretize(dev, &conc_pipe, &cc, &cinp);
    let cgpu2 = gpu_run_concretize(dev, &conc_pipe, &cc, &cinp);
    assert!(cgpu == cgpu2, "{}: not run-to-run stable", cc.name);
    assert_no_sentinel(&cc.name, &cgpu);
    for (r, &(stride, off, _)) in T3_PAIRS.iter().enumerate() {
        assert_eq!(
            cgpu.v[r],
            flo[r].to_bits(),
            "{}: LOWER mismatch at tree level stride={stride} (offset {off})",
            cc.name
        );
        assert_eq!(
            cgpu.r[r],
            fhi[r].to_bits(),
            "{}: UPPER mismatch at tree level stride={stride} (offset {off})",
            cc.name
        );
    }
    println!("[Concretize-t3] 16/16 tree-level isolations bit-exact");
}

// ---------------------------------------------------------------------------
// T4 — thread-0 tail isolation + staged outward publication
// ---------------------------------------------------------------------------

/// `#u1` T4: `k = 1`, exact single product, nonzero preloaded `old` ⇒ `rf` is
/// the only residual. Follow-up cases pin the separately recovered propagated
/// lane and an outward floor beside an existing O(1) error.
#[test]
fn u1_tree_tail_isolation_pins_thread0_assembly() {
    let _g = serial_guard();
    let dev = require_device();
    ny_core::eft::eft_self_check().expect("host EFT reference must be sound");
    // rf enters as the TwoSum residual of `1.0 + a[0]` — any |x| < 2^-24
    // passes through EXACTLY. A full mantissa keeps the recovery multiply
    // nontrivial.
    let rf = f32::from_bits(((127 - 25) << 23) | 0x005A_9E37);
    assert!(
        rf < 2.0f32.powi(-24),
        "rf must pass through TwoSum(1.0, rf) exactly"
    );
    let rs = eft_r_slack_local(1);
    for kernel in [TreeKernel::Bias, TreeKernel::ActBias] {
        let pipe = tree_pipeline(dev, kernel);
        let mk_inputs = |a_err0: f32, e0: f32| TreeInputs {
            a: vec![rf],
            a_err: vec![a_err0],
            b_lo: vec![1.0],
            b_hi: if kernel == TreeKernel::ActBias {
                vec![1.0]
            } else {
                Vec::new()
            },
            out_pre: vec![1.0],
            err_pre: vec![e0],
        };
        let mk_case = |name: String, additive: f32| TreeCase {
            name,
            num_specs: 1,
            k: 1,
            additive,
            slack: 0.0,
            r_slack: rs,
            gamma_k: 0.0,
            eft_mode: true,
            is_upper: false,
            num_specs_per_dom: 1,
        };

        // (a) residual-only deterministic pin.
        let c = mk_case(format!("{kernel:?}-t4-rf-pin"), 0.0);
        let inp = mk_inputs(0.0, 0.0);
        let local = round_up_pos(round_up_pos(rf * rs));
        let expected = outward_bias_update(0.0, local, 0.0).to_bits();
        let (fv, fr, _) = cpu_tree_twin(kernel, &c, &inp, TwinMode::Faithful);
        assert_eq!(
            fr[0].to_bits(),
            expected,
            "{}: twin/algebra mismatch",
            c.name
        );
        let gpu = gpu_run_tree(dev, &pipe, kernel, &c, &inp);
        assert_eq!(gpu.v[0], fv[0].to_bits(), "{}: V mismatch", c.name);
        assert_eq!(
            gpu.r[0], expected,
            "{}: the thread-0 tail (rf + round_up_pos + ·r_slack) compiled to \
             a different sequence",
            c.name
        );

        // (b) propagated-error lane: p.slack must recover it separately from
        // the EFT residual lane. A large test slack makes omission observable.
        let mut c2 = mk_case(format!("{kernel:?}-t4-propagated-slack"), 0.0);
        c2.slack = 1.5;
        let inp2 = TreeInputs {
            a: vec![0.0],
            a_err: vec![1.0],
            b_lo: vec![1.0],
            b_hi: if kernel == TreeKernel::ActBias {
                vec![1.0]
            } else {
                Vec::new()
            },
            out_pre: vec![0.0],
            err_pre: vec![0.0],
        };
        let (_, fr2, _) = cpu_tree_twin(kernel, &c2, &inp2, TwinMode::Faithful);
        assert!(
            fr2[0] > 1.0,
            "{}: combine slack did not reach publication",
            c2.name
        );
        let gpu2 = gpu_run_tree(dev, &pipe, kernel, &c2, &inp2);
        assert_eq!(
            gpu2.r[0],
            fr2[0].to_bits(),
            "{}: propagated tail mismatch",
            c2.name
        );

        // (c) a positive floor beside O(1) prior error must survive staged
        // outward publication rather than being swallowed by ordinary RN.
        let c3 = mk_case(
            format!("{kernel:?}-t4-floor-publication"),
            f32::MIN_POSITIVE,
        );
        let inp3 = TreeInputs {
            a: vec![0.0],
            a_err: vec![0.0],
            b_lo: vec![1.0],
            b_hi: if kernel == TreeKernel::ActBias {
                vec![1.0]
            } else {
                Vec::new()
            },
            out_pre: vec![0.0],
            err_pre: vec![1.0],
        };
        let (_, fr3, _) = cpu_tree_twin(kernel, &c3, &inp3, TwinMode::Faithful);
        assert!(fr3[0] > 1.0, "{}: positive floor was swallowed", c3.name);
        let gpu3 = gpu_run_tree(dev, &pipe, kernel, &c3, &inp3);
        assert_eq!(
            gpu3.r[0],
            fr3[0].to_bits(),
            "{}: floor publication mismatch",
            c3.name
        );
    }
}

// ---------------------------------------------------------------------------
// Concretize composed cases
// ---------------------------------------------------------------------------

/// `#u1` concretize: V-isolation (deterministic words), the composed EFT case,
/// the batched-BaB partition (`num_specs_per_dom < num_specs`, per-domain
/// boxes), and the LIVE legacy mode — each with the hard interval enclosure.
#[test]
fn u1_concretize_settles_value_tree_and_residual_lanes() {
    let _g = serial_guard();
    let dev = require_device();
    ny_core::eft::eft_self_check().expect("host EFT reference must be sound");
    let pipe = concretize_pipeline(dev);

    // (a) V-isolation: power-of-two coefficients on a unit box make every
    // chain/tree/residual op EXACT, so both words are deterministic under any
    // association — bit-inequality is HARD and settles the value tree + tail.
    {
        let n = 3072usize;
        let rows = 9usize;
        let mut a_lower = vec![0.0f32; rows * n];
        let mut a_upper = vec![0.0f32; rows * n];
        for s in 0..rows {
            for j in 0..n {
                let sgn = if (j + s) % 3 == 0 { -1.0f32 } else { 1.0 };
                a_lower[s * n + j] = sgn * 2.0f32.powi(-((j % 8) as i32));
                let sgn2 = if (j + 2 * s) % 5 == 0 { -1.0f32 } else { 1.0 };
                a_upper[s * n + j] = sgn2 * 2.0f32.powi(-(((j + 3) % 8) as i32));
            }
        }
        let c = ConcCase {
            name: "Concretize-v-isolate".into(),
            num_specs: rows,
            input_dim: n,
            gamma_n: 0.0,
            additive: 0.0,
            slack: 0.0,
            num_specs_per_dom: rows,
            eft_mode: true,
            eft_r_slack: eft_r_slack_local(2 * n),
        };
        let inp = ConcInputs {
            a_lower,
            a_upper,
            input_lower: vec![1.0; n],
            input_upper: vec![1.0; n],
            bias: vec![0.0; 2 * rows],
            a_err: vec![0.0; 2 * rows * n],
        };
        let (_, _, stats, _) = cpu_concretize_twin(&c, &inp, TwinMode::Faithful);
        assert_eq!(
            stats.taps_nonzero_es, 0,
            "{}: V-isolation construction must be residual-free",
            c.name
        );
        settle_concretize_case(
            dev,
            &pipe,
            &c,
            &inp,
            Pins {
                v_ulp: Some(0),
                r_ulp: Some(0),
            },
        );
    }

    // (b) composed EFT at production shape and uniforms.
    {
        let c = ConcCase {
            name: "Concretize-eft-3072x96".into(),
            num_specs: 96,
            input_dim: 3072,
            gamma_n: 0.0,
            additive: 1e-30,
            slack: 1.000001,
            num_specs_per_dom: 96,
            eft_mode: true,
            // Production passes k = 2n to eft_r_slack_f32 (four chains ⇒ 4n
            // residual terms ≤ 2·(2n)+18).
            eft_r_slack: eft_r_slack_local(2 * 3072),
        };
        let inp = gen_conc_inputs(&c, 0xC0DE_0001);
        settle_concretize_case(
            dev,
            &pipe,
            &c,
            &inp,
            Pins {
                v_ulp: Some(PINNED_MAX_ULP_DRIFT),
                r_ulp: Some(PINNED_MAX_ULP_DRIFT),
            },
        );
    }

    // (c) batched-BaB partition: 4 domains × 8 rows, per-domain boxes —
    // concretizing against the WRONG box is the false-VERIFIED direction and
    // shows here as a value/enclosure mismatch vs the sbase-faithful twin.
    {
        let c = ConcCase {
            name: "Concretize-eft-dom8".into(),
            num_specs: 32,
            input_dim: 3072,
            gamma_n: 0.0,
            additive: 1e-30,
            slack: 1.000001,
            num_specs_per_dom: 8,
            eft_mode: true,
            eft_r_slack: eft_r_slack_local(2 * 3072),
        };
        let inp = gen_conc_inputs(&c, 0xC0DE_0002);
        settle_concretize_case(
            dev,
            &pipe,
            &c,
            &inp,
            Pins {
                v_ulp: Some(PINNED_MAX_ULP_DRIFT),
                r_ulp: Some(PINNED_MAX_ULP_DRIFT),
            },
        );
    }

    // (d) the LIVE legacy arm (eft_mode = 0): its plain value chain is
    // re-associable, so no ULP pin — the interval enclosure and stability are
    // the assertions (drift is reported by settle_words).
    {
        let c = ConcCase {
            name: "Concretize-m0-12288x9".into(),
            num_specs: 9,
            input_dim: 12288,
            gamma_n: gamma_k_local(2 * 12288),
            additive: 1e-30,
            slack: 1.000001,
            num_specs_per_dom: 9,
            eft_mode: false,
            eft_r_slack: 0.0,
        };
        let inp = gen_conc_inputs(&c, 0xC0DE_0003);
        settle_concretize_case(
            dev,
            &pipe,
            &c,
            &inp,
            Pins {
                v_ulp: None,
                r_ulp: None,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// col2im gather chain
// ---------------------------------------------------------------------------

fn gpu_run_col2im(
    dev: &WgpuDevice,
    pipe: &(wgpu::ComputePipeline, wgpu::BindGroupLayout),
    c: &Col2ImCase,
    v_gemm: &[f32],
    r_gemm: &[f32],
) -> RowWords {
    let total = c.num_specs * c.flat_input_dim();
    let sent = vec![UNWRITTEN_SENTINEL; total];
    let p = c.params();
    let out = dispatch_and_read(
        dev,
        pipe,
        bytemuck::bytes_of(&p),
        &[
            Buf {
                bytes: as_bytes(v_gemm),
                rw: false,
            },
            Buf {
                bytes: as_bytes(r_gemm),
                rw: false,
            },
            Buf {
                bytes: bytemuck::cast_slice(&sent),
                rw: true,
            },
            Buf {
                bytes: bytemuck::cast_slice(&sent),
                rw: true,
            },
        ],
        (total as u32).div_ceil(256),
    );
    RowWords {
        v: out[0].clone(),
        r: out[1].clone(),
    }
}

/// `#u1` col2im: the per-element serial gather chain (fma-barrier TwoSum value
/// re-accumulation + `rsum + r_gemm + |es|` residual stream), bit-compared and
/// enclosed. Case A (zero `r_gemm`) carries the hard enclosure; case B
/// (nonzero `r_gemm`, strided/padded geometry) carries the pass-through and
/// order controls.
#[test]
fn u1_col2im_settles_the_gather_chain() {
    let _g = serial_guard();
    let dev = require_device();
    ny_core::eft::eft_self_check().expect("host EFT reference must be sound");
    let pipe = build_pipeline(
        dev,
        CONV_COL2IM_EFT_TWIN_WGSL,
        "u1_col2im_eft_twin",
        &[false, false, true, true],
    );
    let cases = [
        (
            Col2ImCase {
                name: "col2im-3x8x8-k3s1p1",
                num_specs: 9,
                in_channels: 3,
                in_h: 8,
                in_w: 8,
                kernel_h: 3,
                kernel_w: 3,
                stride: 1,
                pad: 1,
                out_h: 8,
                out_w: 8,
            },
            false, // r_gemm zeroed → hard enclosure case
        ),
        (
            Col2ImCase {
                name: "col2im-8x16x16-k5s2p0",
                num_specs: 4,
                in_channels: 8,
                in_h: 16,
                in_w: 16,
                kernel_h: 5,
                kernel_w: 5,
                stride: 2,
                pad: 0,
                out_h: 6,
                out_w: 6,
            },
            true, // nonzero r_gemm → pass-through + order controls
        ),
    ];
    for (case, with_rgemm) in cases {
        let mut rng = Rng(0xC011_2130 ^ case.num_specs as u64);
        let mut v_gemm = vec![0.0f32; case.gemm_len()];
        for x in v_gemm.iter_mut() {
            *x = rng.unit() * rng.exp_scale(-6, 4);
        }
        let mut r_gemm = vec![0.0f32; case.gemm_len()];
        if with_rgemm {
            for x in r_gemm.iter_mut() {
                *x = rng.unit().abs() * rng.exp_scale(-30, -24);
            }
        }
        let gpu = gpu_run_col2im(dev, &pipe, &case, &v_gemm, &r_gemm);
        let gpu2 = gpu_run_col2im(dev, &pipe, &case, &v_gemm, &r_gemm);
        assert_no_sentinel(case.name, &gpu);
        let (fv, fr, stats, exact) = cpu_col2im_twin(&case, &v_gemm, &r_gemm, TwinMode::Faithful);
        let faithful = words_from(&fv, &fr);
        assert!(
            stats.taps_total > 0 && stats.taps_nonzero_es > 0,
            "{}: vacuous geometry (no taps / no residual traffic)",
            case.name
        );
        assert_eq!(
            stats.subnormal_intermediates, 0,
            "{}: case is not normal-range; composition and flush are confounded",
            case.name
        );

        let mut hypotheses: Vec<(&'static str, RowWords)> = Vec::new();
        {
            let (v, r, _, _) = cpu_col2im_twin(&case, &v_gemm, &r_gemm, TwinMode::TreeRightAssoc);
            hypotheses.push(("rsum-right-assoc", words_from(&v, &r)));
        }
        let mut controls: Vec<(&'static str, RowWords)> = Vec::new();
        {
            let (v, r, _, _) = cpu_col2im_twin(&case, &v_gemm, &r_gemm, TwinMode::DropChainTwoSum);
            controls.push(("drop-chain-twosum", words_from(&v, &r)));
        }
        if with_rgemm {
            let (v, r, _, _) = cpu_col2im_twin(&case, &v_gemm, &r_gemm, TwinMode::DropRGemm);
            controls.push(("drop-rgemm", words_from(&v, &r)));
            let (v2, r2, _, _) = cpu_col2im_twin(&case, &v_gemm, &r_gemm, TwinMode::ReverseTaps);
            controls.push(("reverse-taps", words_from(&v2, &r2)));
        }
        let divergences = settle_words(
            case.name,
            &gpu,
            &gpu2,
            &faithful,
            &hypotheses,
            &controls,
            Pins {
                v_ulp: Some(0),
                r_ulp: Some(PINNED_MAX_ULP_DRIFT),
            },
        );
        for (name, d) in &divergences {
            assert!(
                *d > 0,
                "{}: the {name} control never diverged — not discriminating",
                case.name
            );
        }

        if !with_rgemm {
            // Hard enclosure: |Σ_exact taps − v| ≤ r · r_slack(kh·kw) — the
            // telescoping claim the conv chain's host slack relies on.
            let rs = f64::from(eft_r_slack_local(case.kernel_h * case.kernel_w));
            let mut violations = 0usize;
            for i in 0..fv.len() {
                let v = f64::from(f32::from_bits(gpu.v[i]));
                let r = f64::from(f32::from_bits(gpu.r[i]));
                if (exact[i] - v).abs() > r * rs {
                    violations += 1;
                }
            }
            println!(
                "                               ENCLOSURE |exact − v| <= r·r_slack: \
                 violations={violations}"
            );
            assert_eq!(
                violations, 0,
                "{}: col2im residual stream fails to enclose — hard fail",
                case.name
            );
        }
    }
}
