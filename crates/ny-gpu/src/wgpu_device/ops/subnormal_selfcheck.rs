// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-adapter subnormal-conformance self-check for the EFT channel: does this
//! adapter preserve subnormal operands and the core add/multiply results that
//! the proof requires, and does its product-residual FMA publish at least the
//! exact residual? Residual lanes may conservatively over-charge; an exact-zero
//! flush is admitted only for a subnormal-expected residual because every
//! production EFT writer charges that loss through the
//! `rung3_flush_safe_additive` base, scaled outward by its downstream
//! residual-recovery multiplier where needed.
//! The GB10's FMA does DAZ-zero subnormal multiplicands even under
//! DenormPreserve, so production primary products deliberately use this
//! qualified core multiply; FMA remains only in residual calculations where
//! operand DAZ is conservative or covered by the per-operation floor.
//!
//! # Why this rung exists (it was the hole in the EFT gate)
//!
//! The CPU EFT reference refuses outright when subnormals are flushed
//! (`ny_core::eft::eft_self_check`, "f32 subnormals are flushed (FTZ/DAZ
//! active)") because BOTH arms of that module charge absolute underflow floors
//! derived from GRADUAL underflow: `η ≤ 2^-150` per rounding. Under FTZ the
//! loss is `2^-126` per rounding. The current GPU EFT writers separately charge
//! the narrowly admitted fma-residual loss at that larger scale; core add/mul
//! flushing is still refused by this probe.
//!
//! The WGSL twin gate (`ops/eft_selfcheck.rs`) had NO such lane. Its
//! `PROD_PAIRS`/`SUM_PAIRS` produce only NORMAL results (the smallest is
//! `2^-100`), so the device gate authorized the compensated channel without
//! ever establishing the subnormal-handling half of its own precondition. That
//! is the obligation this module discharges — or refuses under the exact policy
//! above.
//!
//! # `#u2b` — this rung is now COMPOSED INTO the EFT gate, not merely beside it
//!
//! Shipping this probe as an independent rung of `ops/sound_authority.rs` was
//! not enough. `verify_eft_primitives()` did not consult it, and the two
//! PRODUCTION consumers of the compensated channel
//! (`crown_backward_sound_resident.rs`, `crown_concretize_sound.rs`) gate on
//! `eft_primitives_cached()` under `NY_EFT_ERR=1` — a path that never touched
//! the authority ladder at all. On Apple M5 Max/Metal that combination was
//! measured granting the tightening on hardware this probe had already found to
//! flush. `verify_eft_primitives()` / `eft_primitives_cached()` therefore now
//! ENTAIL `verify_gradual_underflow()` / `gradual_underflow_cached()`.
//!
//! Two consequences to keep in mind when editing this file:
//!
//! * [`set_force_subnormal_selfcheck_fail`] and
//!   `NY_FORCE_GPU_SUBNORMAL_SELFCHECK_FAIL` now also close the EFT compensated
//!   channel. They still only ever force MORE closed.
//! * Anything that makes this probe WEAKER (a lane deleted, or an exception
//!   broader than the pinned floor/conservative residual policy) now silently widens
//!   what the compensated channel will authorize. `probe_operands_discriminate` and
//!   `discriminating_lanes_have_pinned_expectations` are the guards on that.
//!
//! # The failure mode being excluded
//!
//! The un-floored EFT decomposition is an IDENTITY:
//! `Σ a_i·w_i − acc_n = Σ ep_i + Σ es_i`. It holds only if every captured
//! residual is the EXACT residual of the executed op. If the adapter flushes,
//! a subnormal `ep`/`es` can be silently replaced by `0` and that identity no
//! longer holds. The production certificate remains an outward INEQUALITY only
//! when residual observations enclose the exact residual, or when an exact-zero
//! subnormal residual is covered by the always-on additive floor. Any other
//! mismatch would leave the smaller EFT arm unproved, so this probe must
//! discriminate and refuse it.
//!
//! # The probe
//!
//! Every operand is read from a storage buffer at runtime (no constant
//! folding). For each `(a, b)` pair the shader emits `fl(a+b)`, `fl(a·b)`, the
//! fma-barrier TwoSum residual `t`, and `|fma(a,b,-fl(a·b))|` — the SAME
//! operations the shipped EFT kernels use for primary products/additions and
//! product-residual recovery. Core lanes are bit-exact; residual lanes must
//! enclose their CPU reference under the policy below.
//! Dedicated amplified min/max-subnormal pairs require the core multiply to
//! produce signed normal results. The separate `denorm_preserve_probe`
//! characterizes the excluded `fma(a,b,0)` form and pins why it is not used.
//! `probe_operands_discriminate` pins that property so the table cannot be
//! weakened into a tautology.
//!
//! ANY under-charging mismatch outside the pinned floor-covered residual
//! exception, dispatch fault, or readback error ⇒ `false` (fail-closed).

use ny_core::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use super::super::WgpuDevice;
use super::ibp_forward::create_buffer;

/// Probe operand pairs `(a, b)` as f32 bit patterns, chosen so that DAZ
/// (subnormal operand flushed on read) and FTZ (subnormal result flushed on
/// write) are each observable on at least one emitted lane.
pub(crate) const PAIRS: [(u32, u32); 11] = [
    // 2^-149 (smallest subnormal) + 0.0 → must stay 2^-149. DAZ ⇒ 0.
    (0x0000_0001, 0x0000_0000),
    // 1.0 + 2^-149 → sum rounds to 1.0, but the TwoSum RESIDUAL is exactly
    // 2^-149, a SUBNORMAL result. An fma-only FTZ ⇒ residual 0, the precise
    // loss admitted and charged by the rung-3 additive floor.
    (0x3F80_0000, 0x0000_0001),
    // 2^-100 · 2^-40 = 2^-140: a subnormal PRODUCT from two normal operands.
    // FTZ ⇒ 0. This is the `ep` underflow band the channel must not lose.
    (0x0D80_0000, 0x2B80_0000),
    // largest subnormal (2^-126 − 2^-149) · 1.0 → itself. DAZ ⇒ 0.
    (0x007F_FFFF, 0x3F80_0000),
    // 2^-126 (smallest NORMAL) · 0.5 = 2^-127, a subnormal result. FTZ ⇒ 0.
    (0x0080_0000, 0x3F00_0000),
    // Amplified core-multiply DAZ discriminator: maxsub * 2^30 is NORMAL.
    (0x007F_FFFF, 0x4E80_0000),
    // Signed companion: maxsub * -2^30 = 0x8f7ffffe.
    (0x007F_FFFF, 0xCE80_0000),
    // Smallest-subnormal endpoints catch hardware that treats only the tiniest
    // denorms as zero: minsub * ±2^30 = ±2^-119, still NORMAL.
    (0x0000_0001, 0x4E80_0000),
    (0x0000_0001, 0xCE80_0000),
    // Product-residual FMA discriminators in BOTH operand slots. The core
    // product is 0x2f800001 and the exact residual magnitude is the NORMAL
    // 0x237ffffc. A measured operand-DAZ FMA returns -prod (conservative after
    // abs); zero or any smaller finite charge must refuse.
    (0x0040_0001, 0x6EFF_FFFF),
    (0x6EFF_FFFF, 0x0040_0001),
];
/// Four emitted lanes per pair: `a+b`, `a·b`, TwoSum residual `t`, and the
/// absolute product residual `|fma(a,b,-(a*b))|`.
pub(crate) const LANES_PER_PAIR: usize = 4;
pub(crate) const OUT_LEN: usize = PAIRS.len() * LANES_PER_PAIR;

/// 16-byte std140-clean uniform params. Layout MUST match WGSL `struct Params`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SubnormalSelfCheckParams {
    n_pairs: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// WGSL probe. `two_sum_fma_barrier` is byte-identical to the form in
/// `ops/eft_selfcheck.rs` and the shipped EFT kernels — the residual lane must
/// measure the SAME sequence the channel actually executes.
const SUBNORMAL_SELFCHECK_SHADER: &str = r#"
struct Params { n_pairs: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       inp:  array<u32>;
@group(0) @binding(2) var<storage, read_write>  outp: array<u32>;

// fma-barrier TwoSum: byte-identical to eft_selfcheck.rs / the shipped kernels.
fn two_sum_fma_barrier(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = fma(-1.0, a, s);   // s - a
    let sb = fma(-1.0, bb, s);  // s - bb
    let da = fma(-1.0, sb, a);  // a - (s - bb)
    let db = fma(-1.0, bb, b);  // b - bb
    return vec2<f32>(s, da + db);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    for (var i: u32 = 0u; i < params.n_pairs; i = i + 1u) {
        let a = bitcast<f32>(inp[2u * i]);
        let b = bitcast<f32>(inp[2u * i + 1u]);
        let ts = two_sum_fma_barrier(a, b);
        let prod = a * b;
        outp[4u * i]      = bitcast<u32>(a + b);
        outp[4u * i + 1u] = bitcast<u32>(prod);
        outp[4u * i + 2u] = bitcast<u32>(ts.y);
        outp[4u * i + 3u] = bitcast<u32>(abs(fma(a, b, -prod)));
    }
}
"#;

/// Test/operator hook: force [`WgpuDevice::verify_gradual_underflow`] to report
/// FAILURE. Read on every call — NOT cached — so a test can flip it without the
/// cached real result masking it.
static TEST_FORCE_FAIL: AtomicBool = AtomicBool::new(false);

fn env_forces_fail() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("NY_FORCE_GPU_SUBNORMAL_SELFCHECK_FAIL").is_some())
}

fn selfcheck_forced_to_fail() -> bool {
    TEST_FORCE_FAIL.load(Ordering::Relaxed) || env_forces_fail()
}

/// Test hook: force / release a subnormal self-check failure.
#[cfg(all(test, feature = "gpu-tests"))]
pub(crate) fn set_force_subnormal_selfcheck_fail(force: bool) {
    TEST_FORCE_FAIL.store(force, Ordering::Relaxed);
}

/// CPU reference for one pair: `(a+b, a·b, TwoSum residual,
/// |TwoProduct residual|)`, all bit patterns.
fn cpu_expect(a_bits: u32, b_bits: u32) -> [u32; LANES_PER_PAIR] {
    let a = f32::from_bits(a_bits);
    let b = f32::from_bits(b_bits);
    let (_, t) = ny_core::eft::two_sum_f32(a, b);
    let (_, ep) = ny_core::eft::two_prod_f32(a, b);
    [
        (a + b).to_bits(),
        (a * b).to_bits(),
        t.to_bits(),
        ep.abs().to_bits(),
    ]
}

// ---------------------------------------------------------------------------
// #flush-charge: PURE-FLUSH characterization (charged-Metal mode, lane M)
// ---------------------------------------------------------------------------

/// Adapter flush-behavior classification for the charged-flush authority path
/// (`ops/sound_authority.rs`).
///
/// This does NOT relax rung 3: `Conformant` is exactly the existing lane
/// policy, and anything that is not provably a pure subnormal flush refuses as
/// [`FlushClass::NonConformant`]. The ONLY new admission is `PureFlush`: every
/// lane the rung-3 policy rejects is bit-exactly the output of the pure
/// operand-DAZ + result-FTZ model (a ±0 in place of the exact value, never a
/// wrong nonzero) — the hardware class whose losses the exact-rational
/// `flush_charge_oracle` proves chargeable by additive widening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlushClass {
    /// Every lane satisfies the rung-3 policy: full qualification applies and
    /// the charged state is deliberately unreachable.
    Conformant,
    /// Rung 3 refuses, but every refused lane equals the pure DAZ+FTZ model
    /// prediction exactly (±0). Admissible for charged-flush authority ONLY —
    /// never for uncharged qualification.
    PureFlush,
    /// Any other behavior (a wrong nonzero, a dispatch fault, a forced-fail
    /// hook, a broken loading-path contract): refused outright.
    NonConformant,
}

/// Subnormal → same-signed zero, the DAZ/FTZ transfer of the modeled hardware.
/// Mirrors `flush_charge_oracle::Hw::flush_result`/`flush_operand` exactly (the
/// oracle pins the agreement so the two cannot drift apart).
fn flush_subnormal_to_zero(x: f32) -> f32 {
    if x != 0.0 && x.is_finite() && x.abs() < f32::MIN_POSITIVE {
        if x.is_sign_negative() {
            -0.0
        } else {
            0.0
        }
    } else {
        x
    }
}

fn pure_flush_add(a: f32, b: f32) -> f32 {
    flush_subnormal_to_zero(flush_subnormal_to_zero(a) + flush_subnormal_to_zero(b))
}

fn pure_flush_mul(a: f32, b: f32) -> f32 {
    flush_subnormal_to_zero(flush_subnormal_to_zero(a) * flush_subnormal_to_zero(b))
}

fn pure_flush_fma(a: f32, b: f32, c: f32) -> f32 {
    flush_subnormal_to_zero(
        flush_subnormal_to_zero(a).mul_add(flush_subnormal_to_zero(b), flush_subnormal_to_zero(c)),
    )
}

/// CPU twin of one probe pair under the pure operand-DAZ + result-FTZ model
/// (both core ops AND fma flush, matching the oracle's `METAL_CMP_DAZ`
/// hardware). Same op sequence as the WGSL probe, bit patterns out.
pub(crate) fn cpu_expect_pure_flush(a_bits: u32, b_bits: u32) -> [u32; LANES_PER_PAIR] {
    let a = f32::from_bits(a_bits);
    let b = f32::from_bits(b_bits);
    // two_sum_fma_barrier, flush-modeled per op
    let s = pure_flush_add(a, b);
    let bb = pure_flush_fma(-1.0, a, s);
    let sb = pure_flush_fma(-1.0, bb, s);
    let da = pure_flush_fma(-1.0, sb, a);
    let db = pure_flush_fma(-1.0, bb, b);
    let t = pure_flush_add(da, db);
    let prod = pure_flush_mul(a, b);
    let ep = pure_flush_fma(a, b, -prod).abs();
    [s.to_bits(), prod.to_bits(), t.to_bits(), ep.to_bits()]
}

/// Pure-flush model expectations for every emitted lane, in device output order.
pub(crate) fn pure_flush_expectations() -> Vec<u32> {
    let mut expected: Vec<u32> = Vec::with_capacity(OUT_LEN);
    for &(a, b) in &PAIRS {
        expected.extend_from_slice(&cpu_expect_pure_flush(a, b));
    }
    expected
}

/// Classify one complete raw lane table. Pure function so it is unit-testable
/// against pinned lane tables (including the measured Apple M5 Max table) with
/// no device.
pub(crate) fn classify_flush_lanes(lanes: &[u32]) -> FlushClass {
    let expected = cpu_expectations();
    if lanes.len() != expected.len() {
        return FlushClass::NonConformant;
    }
    if lanes
        .iter()
        .zip(expected.iter())
        .enumerate()
        .all(|(i, (&got, &exp))| subnormal_lane_accepted(i, got, exp))
    {
        return FlushClass::Conformant;
    }
    let flushed = pure_flush_expectations();
    for (i, (&got, &exp)) in lanes.iter().zip(expected.iter()).enumerate() {
        if subnormal_lane_accepted(i, got, exp) {
            continue;
        }
        // The ONLY new admission beyond the rung-3 policy: the device output is
        // bit-exactly the pure DAZ+FTZ model prediction AND is a ±0 (a flush of
        // the exact value, never a wrong nonzero).
        if got == flushed[i] && got & 0x7fff_ffff == 0 {
            continue;
        }
        return FlushClass::NonConformant;
    }
    FlushClass::PureFlush
}

/// Forced-fail visibility for the charged-flush authority read
/// (`charged_flush_authority_cached`): the hook must close BOTH the uncharged
/// and the charged mode. Only ever forces MORE closed.
pub(crate) fn probe_forced_to_fail() -> bool {
    selfcheck_forced_to_fail()
}

/// Pure rung-3 lane policy shared by the live gate and its diagnostic report.
/// Exact matches always pass. An exact `+0` or `-0` in either residual lane is
/// admitted only when its CPU expectation is nonzero subnormal; the rung-3
/// additive pays for that loss. The product-residual lane may instead publish
/// any finite nonnegative value at least as large as the exact residual, because
/// production takes its absolute value and a larger term only widens.
fn subnormal_lane_accepted(index: usize, got: u32, expected: u32) -> bool {
    if got == expected {
        return true;
    }
    let expected_magnitude = expected & 0x7fff_ffff;
    let expected_is_subnormal =
        expected_magnitude != 0 && expected_magnitude < f32::MIN_POSITIVE.to_bits();
    let got_is_zero = got & 0x7fff_ffff == 0;
    let lane = index % LANES_PER_PAIR;
    if (lane == 2 || lane == 3) && expected_is_subnormal && got_is_zero {
        return true;
    }
    if lane == 3 {
        let got_value = f32::from_bits(got);
        let expected_value = f32::from_bits(expected);
        return got_value.is_finite()
            && !got_value.is_sign_negative()
            && got_value >= expected_value;
    }
    false
}

impl WgpuDevice {
    /// One-time per-adapter authorization for NY's rung-3 subnormal policy:
    /// subnormal operands and core add/multiply results are preserved, and every
    /// product-residual FMA returns a finite charge enclosing the exact residual.
    /// An exact-zero subnormal residual is permitted only because every
    /// production EFT writer charges its loss explicitly. Primary EFT products
    /// use the qualified core multiply, not the measured direct-FMA DAZ form.
    ///
    /// `true` ⇒ the EFT residual decomposition plus its charged floor is an
    /// outward bound under the measured lane policy.
    /// `false` ⇒ REFUSED (fail-closed). Cached per device.
    pub(crate) fn verify_gradual_underflow(&self) -> bool {
        if selfcheck_forced_to_fail() || !self.denorm_preserve_contract_intact() {
            return false;
        }
        // The HOST reference must itself honour gradual underflow, else the
        // bit-comparison is meaningless. `eft_self_check` probes exactly that
        // (plus fused FMA and RN) and is rational-oracle pinned.
        if ny_core::eft::eft_self_check().is_err() {
            return false;
        }
        let passed = *self
            .subnormal_selfcheck
            .get_or_init(|| self.run_subnormal_selfcheck());
        // The probe module itself, or a concurrently/lazily created production
        // module, may have triggered the sticky passthrough fallback while the
        // cached probe was running. Re-check after reading/initializing it.
        passed && self.denorm_preserve_contract_intact()
    }

    /// Never-initializing cached read, for call sites INSIDE a GPU-checked
    /// section (running the probe there would self-deadlock on the enclosing
    /// lock). Uninitialized ⇒ `false` (fail-closed).
    pub(crate) fn gradual_underflow_cached(&self) -> bool {
        if selfcheck_forced_to_fail() || !self.denorm_preserve_contract_intact() {
            return false;
        }
        self.subnormal_selfcheck.get().copied().unwrap_or(false)
    }

    /// #flush-charge: characterize this adapter's subnormal behavior for the
    /// charged-flush authority admission predicate. Runs the SAME raw probe as
    /// rung 3 and classifies every lane through [`classify_flush_lanes`].
    ///
    /// Fail-closed: forced-fail hooks, a broken DenormPreserve loading-path
    /// contract, a non-conformant host reference, or any GPU error all return
    /// [`FlushClass::NonConformant`] — never a grant.
    pub(crate) fn characterize_flush_policy(&self) -> FlushClass {
        if selfcheck_forced_to_fail()
            || !self.denorm_preserve_contract_intact()
            || ny_core::eft::eft_self_check().is_err()
        {
            return FlushClass::NonConformant;
        }
        match self.run_gpu_checked("characterize_flush_policy", || {
            self.subnormal_selfcheck_raw()
        }) {
            Ok(lanes) => {
                let class = classify_flush_lanes(&lanes);
                tracing::info!(
                    target: "ny_gpu::wgpu",
                    adapter = %self.adapter_info.name,
                    backend = ?self.adapter_info.backend,
                    ?class,
                    "subnormal flush-policy characterization complete"
                );
                class
            }
            Err(e) => {
                tracing::warn!(
                    target: "ny_gpu::wgpu",
                    adapter = %self.adapter_info.name,
                    backend = ?self.adapter_info.backend,
                    error = %e,
                    "flush-policy characterization could not run: NON-CONFORMANT \
                     (fail-closed)"
                );
                FlushClass::NonConformant
            }
        }
    }

    /// Run (and log) the one-time probe. CONSERVATIVE: any mismatch outside the
    /// floor-covered/conservative residual policy OR any GPU error → `false`.
    fn run_subnormal_selfcheck(&self) -> bool {
        match self.run_gpu_checked("verify_gradual_underflow", || {
            self.subnormal_selfcheck_inner()
        }) {
            Ok(true) => true,
            Ok(false) => {
                tracing::warn!(
                    target: "ny_gpu::wgpu",
                    adapter = %self.adapter_info.name,
                    backend = ?self.adapter_info.backend,
                    "SUBNORMAL-CONFORMANCE self-check FAILED: this adapter \
                     mismatched a core add/multiply lane or under-charged a \
                     residual outside the exact-zero, subnormal-expected case \
                     covered by the scaled rung3 flush floor. REFUSING GPU verdict authority \
                     and the EFT compensated error channel; the a-priori Higham \
                     charge ships unchanged (fail-closed)"
                );
                false
            }
            Err(e) => {
                tracing::warn!(
                    target: "ny_gpu::wgpu",
                    adapter = %self.adapter_info.name,
                    backend = ?self.adapter_info.backend,
                    error = %e,
                    "SUBNORMAL-CONFORMANCE self-check could not run: REFUSING \
                     (fail-closed)"
                );
                false
            }
        }
    }

    /// Dispatch the probe and bit-compare every emitted lane against the CPU.
    ///
    /// #rung3-fma-floor refinement (GB10-measured 2026-08-10
    /// America/Los_Angeles): under an armed `DenormPreserve` execution mode the
    /// core add/mul lanes preserve subnormals bit-exactly but fma still flushes
    /// SUBNORMAL RESULTS. The residual lane uses the shipped kernels'
    /// fma-barrier form, so a subnormal-expected residual may read back as an
    /// exact ±0 flush. That loss is CHARGED unconditionally in every shipped
    /// EFT dispatch (the `sound_consts::rung3_flush_safe_additive` base,
    /// outward-scaled by the dispatch's residual recovery multiplier and
    /// folded into the kernels' `additive` flush term), so an exact-zero flush
    /// of a subnormal-EXPECTED residual is accepted here. Everything else stays
    /// bit-exact-required: the add/mul lanes on every pair (DAZ and plain-op
    /// FTZ still refuse). A product residual may be larger than the CPU value,
    /// but must be finite and never smaller. A fully-FTZ adapter still fails on
    /// lanes 0/1.
    fn subnormal_selfcheck_inner(&self) -> Result<bool> {
        let expected = cpu_expectations();
        let out = self.subnormal_selfcheck_raw()?;
        if out.len() != OUT_LEN {
            return Ok(false);
        }
        let mut floored_flushes = 0usize;
        let mut conservative_product_residuals = 0usize;
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            let accepted = subnormal_lane_accepted(i, got, exp);
            if got != exp && accepted {
                let expected_magnitude = exp & 0x7fff_ffff;
                let expected_is_subnormal =
                    expected_magnitude != 0 && expected_magnitude < f32::MIN_POSITIVE.to_bits();
                if got & 0x7fff_ffff == 0 && expected_is_subnormal {
                    floored_flushes += 1;
                } else {
                    conservative_product_residuals += 1;
                }
            }
            if !accepted {
                return Ok(false);
            }
        }
        if floored_flushes > 0 {
            tracing::info!(
                target: "ny_gpu::wgpu",
                adapter = %self.adapter_info.name,
                floored_flushes,
                "subnormal-conformance: {floored_flushes} subnormal FMA-derived \
                 residual lane(s) flushed to exact zero — ACCEPTED, the loss is \
                 charged by the always-on, residual-slack-scaled rung3 floor \
                 (core add/mul lanes are bit-exact; primary products use mul)"
            );
        }
        if conservative_product_residuals > 0 {
            tracing::info!(
                target: "ny_gpu::wgpu",
                adapter = %self.adapter_info.name,
                conservative_product_residuals,
                "subnormal-conformance: {conservative_product_residuals} product-residual \
                 lane(s) conservatively exceeded the exact residual — ACCEPTED \
                 because the shipped radius accumulates their absolute values"
            );
        }
        Ok(true)
    }

    /// Dispatch the probe and return the RAW device bits, one per emitted lane
    /// (`PAIRS.len() * LANES_PER_PAIR`, in `(a+b, a·b, TwoSum residual,
    /// product residual)` order per pair). Separated from the comparison so a failing adapter can be
    /// CHARACTERIZED lane by lane (which of DAZ / FTZ) rather than only refused.
    fn subnormal_selfcheck_raw(&self) -> Result<Vec<u32>> {
        let params = SubnormalSelfCheckParams {
            n_pairs: PAIRS.len() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_buffer(
            &self.device,
            "subnormal_selfcheck_params",
            size_of::<SubnormalSelfCheckParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

        let inp: Vec<u32> = PAIRS.iter().flat_map(|&(a, b)| [a, b]).collect();
        let inp_buf = create_buffer(
            &self.device,
            "subnormal_selfcheck_inp",
            (inp.len() * size_of::<u32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&inp_buf, 0, bytemuck::cast_slice(&inp));

        let out_bytes = (OUT_LEN * size_of::<u32>()) as u64;
        let out_buf = create_buffer(
            &self.device,
            "subnormal_selfcheck_out",
            out_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let (pipeline, layout) = self.create_simple_pipeline(
            SUBNORMAL_SELFCHECK_SHADER,
            "subnormal_selfcheck",
            &[false, true],
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("subnormal_selfcheck_bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("subnormal_selfcheck_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("subnormal_selfcheck_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let staging = create_buffer(
            &self.device,
            "subnormal_selfcheck_staging",
            out_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        WgpuDevice::read_u32_buffer(&self.device, &staging, OUT_LEN)
    }

    /// Diagnostic: run the probe once and return `(lane_label, device_bits,
    /// expected_bits)` for every emitted lane. Used by the measurement test to
    /// characterize a failing adapter (DAZ vs FTZ) instead of just refusing it.
    #[cfg(all(test, feature = "gpu-tests"))]
    pub(crate) fn subnormal_selfcheck_report(&self) -> Result<Vec<(&'static str, u32, u32)>> {
        let expected = cpu_expectations();
        let out = self.run_gpu_checked("subnormal_selfcheck_report", || {
            self.subnormal_selfcheck_raw()
        })?;
        Ok(LANE_LABELS
            .iter()
            .zip(out)
            .zip(expected)
            .map(|((&label, got), exp)| (label, got, exp))
            .collect())
    }
}

/// CPU expectations for every emitted lane, in device output order.
pub(crate) fn cpu_expectations() -> Vec<u32> {
    let mut expected: Vec<u32> = Vec::with_capacity(OUT_LEN);
    for &(a, b) in &PAIRS {
        expected.extend_from_slice(&cpu_expect(a, b));
    }
    expected
}

/// Human-readable label per emitted lane, in device output order. Length is
/// pinned to `OUT_LEN` by `lane_labels_cover_every_lane`.
#[cfg(test)]
const LANE_LABELS: [&str; OUT_LEN] = [
    "2^-149 + 0.0            [DAZ: operand]",
    "2^-149 * 0.0            [exact zero control]",
    "twosum(2^-149, 0.0).res [DAZ: operand]",
    "twoprod(2^-149,0).res   [exact zero control]",
    "1.0 + 2^-149            [DAZ: operand]",
    "1.0 * 2^-149            [FTZ: subnormal result]",
    "twosum(1.0, 2^-149).res [fma FTZ: FLOOR-covered residual]",
    "twoprod(1.0,2^-149).res [zero or conservative]",
    "2^-100 + 2^-40          [normal]",
    "2^-100 * 2^-40 = 2^-140 [FTZ: subnormal result from NORMAL operands]",
    "twosum(2^-100,2^-40).res[normal]",
    "twoprod(2^-100,2^-40).res[residual policy]",
    "maxsub + 1.0            [normal result]",
    "maxsub * 1.0 = maxsub   [DAZ: operand]",
    "twosum(maxsub,1.0).res  [DAZ: subnormal residual]",
    "twoprod(maxsub,1.0).res [zero or conservative]",
    "2^-126 + 0.5            [normal]",
    "2^-126 * 0.5 = 2^-127   [FTZ: subnormal result from NORMAL operands]",
    "twosum(2^-126,0.5).res  [normal]",
    "twoprod(2^-126,0.5).res [residual policy]",
    "maxsub + 2^30           [normal]",
    "maxsub * 2^30           [core DAZ discriminator]",
    "twosum(maxsub,2^30).res [FLOOR-covered residual]",
    "twoprod(maxsub,2^30).res[conservative FMA-DAZ allowed]",
    "maxsub + -2^30          [normal]",
    "maxsub * -2^30          [core DAZ signed discriminator]",
    "twosum(maxsub,-2^30).res[FLOOR-covered residual]",
    "twoprod(maxsub,-2^30).res[conservative FMA-DAZ allowed]",
    "minsub + 2^30           [normal]",
    "minsub * 2^30 = 2^-119  [core tiny-DAZ discriminator]",
    "twosum(minsub,2^30).res [FLOOR-covered residual]",
    "twoprod(minsub,2^30).res[conservative FMA-DAZ allowed]",
    "minsub + -2^30          [normal]",
    "minsub * -2^30=-2^-119  [core tiny-DAZ signed discriminator]",
    "twosum(minsub,-2^30).res[FLOOR-covered residual]",
    "twoprod(minsub,-2^30).res[conservative FMA-DAZ allowed]",
    "subA + hugeB             [product-residual pair]",
    "subA * hugeB             [normal core product]",
    "twosum(subA,hugeB).res  [residual policy]",
    "twoprod(subA,hugeB).res [normal enclosure discriminator]",
    "hugeB + subA             [swapped product-residual pair]",
    "hugeB * subA             [normal core product]",
    "twosum(hugeB,subA).res  [residual policy]",
    "twoprod(hugeB,subA).res [swapped-slot enclosure discriminator]",
];

#[cfg(test)]
mod cpu_tests {
    use super::*;

    #[test]
    fn lane_labels_cover_every_lane() {
        assert_eq!(LANE_LABELS.len(), OUT_LEN);
        assert!(
            LANE_LABELS.iter().all(|label| !label.trim().is_empty()),
            "every diagnostic lane must have a nonempty label"
        );
    }

    #[test]
    fn lane_policy_accepts_only_floor_covered_fma_result_zero() {
        let positive_subnormal = 0x0000_0001;
        let negative_subnormal = 0x8000_0001;
        let residual_lane = 2;

        assert!(subnormal_lane_accepted(
            residual_lane,
            0x0000_0000,
            positive_subnormal
        ));
        assert!(subnormal_lane_accepted(
            residual_lane,
            0x8000_0000,
            positive_subnormal
        ));
        assert!(subnormal_lane_accepted(
            residual_lane,
            0x0000_0000,
            negative_subnormal
        ));
        assert!(subnormal_lane_accepted(
            residual_lane,
            0x8000_0000,
            negative_subnormal
        ));

        assert!(
            !subnormal_lane_accepted(0, 0x0000_0000, positive_subnormal),
            "an add-lane flush must refuse"
        );
        assert!(
            !subnormal_lane_accepted(1, 0x0000_0000, positive_subnormal),
            "a multiply-lane flush must refuse"
        );
        assert!(
            !subnormal_lane_accepted(residual_lane, 0x0000_0002, positive_subnormal),
            "a wrong nonzero residual must refuse"
        );
        assert!(
            !subnormal_lane_accepted(residual_lane, 0x0000_0000, 1.0f32.to_bits()),
            "a zero in place of a normal residual must refuse"
        );
    }

    #[test]
    fn product_residual_policy_accepts_only_finite_enclosing_charges() {
        let product_lane = 3;
        let exact_normal = 0x237f_fffcu32;
        assert!(subnormal_lane_accepted(
            product_lane,
            exact_normal,
            exact_normal
        ));
        assert!(subnormal_lane_accepted(
            product_lane,
            exact_normal + 1,
            exact_normal
        ));
        assert!(
            !subnormal_lane_accepted(product_lane, exact_normal - 1, exact_normal),
            "a finite product-residual undercharge must refuse"
        );
        assert!(
            !subnormal_lane_accepted(product_lane, 0, exact_normal),
            "a zero in place of a normal exact residual must refuse"
        );
        assert!(!subnormal_lane_accepted(
            product_lane,
            f32::INFINITY.to_bits(),
            exact_normal
        ));
        assert!(!subnormal_lane_accepted(
            product_lane,
            (-1.0f32).to_bits(),
            exact_normal
        ));
        assert!(
            subnormal_lane_accepted(product_lane, 0, 0x0000_0001),
            "an exact-zero subnormal residual remains floor-covered"
        );
    }

    /// The table must DISCRIMINATE: a DAZ adapter and an FTZ adapter must each
    /// mismatch on at least one emitted lane. We prove this by construction —
    /// at least one lane has a subnormal OPERAND whose correct output is
    /// nonzero (catches DAZ) and at least one lane has a subnormal RESULT from
    /// NORMAL operands (catches FTZ, which DAZ-only hardware would pass).
    #[test]
    fn probe_operands_discriminate() {
        let subnormal = |x: f32| x != 0.0 && x.abs() < f32::MIN_POSITIVE;

        // (1) DAZ lane: a subnormal operand that must produce a NONZERO result.
        let daz = PAIRS.iter().any(|&(a, b)| {
            let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
            (subnormal(fa) || subnormal(fb))
                && cpu_expect(a, b)[..2]
                    .iter()
                    .any(|&bits| f32::from_bits(bits) != 0.0)
        });
        assert!(daz, "table must contain a DAZ-discriminating lane");

        // (2) FTZ lane: NORMAL (or zero) operands producing a SUBNORMAL result.
        // DAZ-only hardware passes (1) here, so this lane is independent.
        let ftz = PAIRS.iter().any(|&(a, b)| {
            let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
            !subnormal(fa)
                && !subnormal(fb)
                && cpu_expect(a, b)[..2]
                    .iter()
                    .any(|&bits| subnormal(f32::from_bits(bits)))
        });
        assert!(
            ftz,
            "table must contain an FTZ-discriminating lane (subnormal RESULT \
             from normal operands)"
        );

        // (3) Amplified core-multiply DAZ: a subnormal multiplicand whose
        // product is NORMAL and nonzero. Require both signs and both subnormal
        // endpoints; primary EFT products use this qualified operation instead
        // of the measured FMA-DAZ form.
        let mut mul_positive = false;
        let mut mul_negative = false;
        for &(a, b) in &PAIRS {
            let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
            if !(subnormal(fa) || subnormal(fb)) {
                continue;
            }
            let expected = f32::from_bits(cpu_expect(a, b)[1]);
            if expected.is_normal() {
                mul_positive |= expected.is_sign_positive();
                mul_negative |= expected.is_sign_negative();
            }
        }
        assert!(
            mul_positive && mul_negative,
            "table must contain positive and negative normal-result multiply \
             lanes with a subnormal multiplicand to discriminate core DAZ"
        );
        for pair in [
            (0x007F_FFFF, 0x4E80_0000),
            (0x007F_FFFF, 0xCE80_0000),
            (0x0000_0001, 0x4E80_0000),
            (0x0000_0001, 0xCE80_0000),
        ] {
            assert!(
                PAIRS.contains(&pair),
                "amplified core-multiply endpoint/sign lane {pair:?} disappeared"
            );
        }
        for pair in [(0x0040_0001, 0x6EFF_FFFF), (0x6EFF_FFFF, 0x0040_0001)] {
            assert!(
                PAIRS.contains(&pair),
                "normal-residual product-FMA slot discriminator {pair:?} disappeared"
            );
        }
    }

    /// #flush-charge: the 44-lane table MEASURED on Apple M5 Max / Metal
    /// (laneM capture, 2026-08-12; 13 MISS + 8 FLOOR lanes). This is the
    /// hardware the charged-flush mode exists for, so it is pinned here as a
    /// regression oracle: (1) it classifies PURE-FLUSH, (2) it is bit-exactly
    /// what the pure DAZ+FTZ model predicts on every lane, so driver/toolchain
    /// drift that changes the flush behavior stops classifying and REFUSES
    /// instead of silently mis-charging.
    pub(super) const MEASURED_M5_MAX_LANES: [u32; OUT_LEN] = [
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000, // (2^-149, 0.0)
        0x3f80_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000, // (1.0, 2^-149)
        0x2b80_0000,
        0x0000_0000,
        0x0d80_0000,
        0x0000_0000, // (2^-100, 2^-40)
        0x3f80_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000, // (maxsub, 1.0)
        0x3f00_0000,
        0x0000_0000,
        0x0080_0000,
        0x0000_0000, // (2^-126, 0.5)
        0x4e80_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000, // (maxsub, 2^30)
        0xce80_0000,
        0x8000_0000,
        0x0000_0000,
        0x0000_0000, // (maxsub, -2^30)
        0x4e80_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000, // (minsub, 2^30)
        0xce80_0000,
        0x8000_0000,
        0x0000_0000,
        0x0000_0000, // (minsub, -2^30)
        0x6eff_ffff,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000, // (subA, hugeB)
        0x6eff_ffff,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000, // (hugeB, subA)
    ];

    #[test]
    fn measured_m5_table_is_pure_flush_and_matches_the_model_exactly() {
        assert_eq!(
            classify_flush_lanes(&MEASURED_M5_MAX_LANES),
            FlushClass::PureFlush,
            "the measured Apple M5 Max lane table must classify PURE-FLUSH"
        );
        // Bit-exact model agreement on EVERY lane, not only the refused ones:
        // the charge model in flush_charge_oracle is derived for exactly this
        // hardware class, so any divergence must refuse.
        let model = pure_flush_expectations();
        for (i, (&got, &predicted)) in MEASURED_M5_MAX_LANES.iter().zip(model.iter()).enumerate() {
            assert_eq!(
                got, predicted,
                "lane {i}: measured 0x{got:08x} != pure-flush model 0x{predicted:08x}"
            );
        }
        // Non-vacuity: the table really does refuse rung 3 (13 MISS lanes).
        let expected = cpu_expectations();
        let refused = MEASURED_M5_MAX_LANES
            .iter()
            .zip(expected.iter())
            .enumerate()
            .filter(|(i, (&got, &exp))| !subnormal_lane_accepted(*i, got, exp))
            .count();
        assert_eq!(refused, 13, "the measured MISS-lane count drifted");
    }

    #[test]
    fn classifier_is_fail_closed_on_anything_but_a_pure_flush() {
        // A fully conformant adapter classifies Conformant.
        assert_eq!(
            classify_flush_lanes(&cpu_expectations()),
            FlushClass::Conformant
        );
        // A wrong NONZERO in any refused lane is NonConformant, even when every
        // other lane matches the measured flush table.
        let mut wrong_nonzero = MEASURED_M5_MAX_LANES;
        // lane 21 is `maxsub * 2^30` (measured flush 0x0); make it a wrong nonzero
        wrong_nonzero[21] = 0x0f7f_fffd; // one ULP BELOW the exact 0x0f7ffffe
        assert_eq!(
            classify_flush_lanes(&wrong_nonzero),
            FlushClass::NonConformant,
            "a wrong-nonzero core lane must refuse outright"
        );
        // A flush the model does NOT predict (zero where the model preserves)
        // is NonConformant: lane 8 is the normal add 2^-100 + 2^-40.
        let mut unmodeled_flush = MEASURED_M5_MAX_LANES;
        unmodeled_flush[8] = 0x0000_0000;
        assert_eq!(
            classify_flush_lanes(&unmodeled_flush),
            FlushClass::NonConformant,
            "a zero in place of a NORMAL result is not a subnormal flush"
        );
        // A truncated/oversized table refuses.
        assert_eq!(
            classify_flush_lanes(&MEASURED_M5_MAX_LANES[..OUT_LEN - 1]),
            FlushClass::NonConformant
        );
    }

    /// Pin the exact expectations that make the lanes discriminating, so a
    /// future edit to `PAIRS` cannot silently turn a probe lane into a
    /// tautology (e.g. an all-zero expectation that FTZ also produces).
    #[test]
    fn discriminating_lanes_have_pinned_expectations() {
        // 1.0 + 2^-149: residual lane is exactly the smallest subnormal. This
        // pins the sole mismatch class the shared lane policy may floor-cover.
        assert_eq!(
            cpu_expect(0x3F80_0000, 0x0000_0001)[2],
            0x0000_0001,
            "TwoSum residual of 1.0 + 2^-149 must be 2^-149 (fma FTZ ⇒ \
             floor-covered 0)"
        );
        // 2^-100 · 2^-40 = 2^-140 = 512 · 2^-149.
        assert_eq!(
            cpu_expect(0x0D80_0000, 0x2B80_0000)[1],
            0x0000_0200,
            "2^-100 · 2^-40 must be the subnormal 2^-140 (FTZ ⇒ 0)"
        );
        // 2^-126 · 0.5 = 2^-127 (subnormal).
        assert_eq!(
            cpu_expect(0x0080_0000, 0x3F00_0000)[1],
            0x0040_0000,
            "2^-126 · 0.5 must be the subnormal 2^-127 (FTZ ⇒ 0)"
        );
        // Largest subnormal · 1.0 must round-trip (DAZ ⇒ 0).
        assert_eq!(
            cpu_expect(0x007F_FFFF, 0x3F80_0000)[1],
            0x007F_FFFF,
            "largest subnormal · 1.0 must be itself (DAZ ⇒ 0)"
        );
        assert_eq!(
            cpu_expect(0x007F_FFFF, 0x4E80_0000)[1],
            0x0F7F_FFFE,
            "core multiply must preserve maxsub * 2^30 as a positive NORMAL result"
        );
        assert_eq!(
            cpu_expect(0x007F_FFFF, 0xCE80_0000)[1],
            0x8F7F_FFFE,
            "core multiply must preserve maxsub * -2^30 as a negative NORMAL result"
        );
        assert_eq!(
            cpu_expect(0x0000_0001, 0x4E80_0000)[1],
            0x0400_0000,
            "core multiply must preserve minsub * 2^30 as a positive NORMAL result"
        );
        assert_eq!(
            cpu_expect(0x0000_0001, 0xCE80_0000)[1],
            0x8400_0000,
            "core multiply must preserve minsub * -2^30 as a negative NORMAL result"
        );
        for (a, b) in [(0x0040_0001, 0x6EFF_FFFF), (0x6EFF_FFFF, 0x0040_0001)] {
            let expected = cpu_expect(a, b);
            assert_eq!(
                expected[1], 0x2F80_0001,
                "product-residual discriminator core product drifted"
            );
            assert_eq!(
                expected[3], 0x237F_FFFC,
                "product-residual discriminator must have a nonzero NORMAL residual"
            );
        }
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};

    /// MEASUREMENT (not an assertion of the outcome): report per-lane whether
    /// this adapter satisfies the exact rung-3 lane policy, then pin the
    /// fail-closed hook.
    ///
    /// This test does NOT assert that the adapter passes — its precise core-lane
    /// and fma-residual behavior is a hardware fact. It asserts only that the
    /// GATE agrees with the shared lane policy and loading-path contract, and
    /// that forcing failure refuses.
    #[test]
    fn gradual_underflow_measured_and_forced_fail_refuses() {
        let _serial = gpu_test_serial_guard();
        let device = require_device();

        // Characterize first; the later gate read must also observe any sticky
        // DenormPreserve fallback this diagnostic module creation caused.
        let report = device
            .subnormal_selfcheck_report()
            .expect("probe dispatch/readback");
        let verdict = device.verify_gradual_underflow();
        println!(
            "[subnormal/rung-3 probe] adapter={} backend={:?} => {}",
            device.adapter_info.name,
            device.adapter_info.backend,
            if verdict {
                "PASS (core add/mul subnormal lanes preserved; primary EFT \
                 products use mul and any displayed FLOOR residual is covered by the \
                 residual-slack-scaled rung3 floor)"
            } else {
                "FAIL (disallowed subnormal-lane mismatch or DenormPreserve \
                 loading-path fallback)"
            }
        );
        // Characterize lane by lane so a refusal is diagnosable, not just a bit.
        let mut disallowed_mismatches = 0usize;
        let mut floored_residuals = 0usize;
        let mut conservative_product_residuals = 0usize;
        for (i, (label, got, exp)) in report.iter().enumerate() {
            let accepted = subnormal_lane_accepted(i, *got, *exp);
            let flag = if got == exp {
                "ok   "
            } else if accepted {
                let expected_magnitude = exp & 0x7fff_ffff;
                let expected_is_subnormal =
                    expected_magnitude != 0 && expected_magnitude < f32::MIN_POSITIVE.to_bits();
                if got & 0x7fff_ffff == 0 && expected_is_subnormal {
                    floored_residuals += 1;
                    "FLOOR"
                } else {
                    conservative_product_residuals += 1;
                    "OVER "
                }
            } else {
                disallowed_mismatches += 1;
                "MISS "
            };
            println!(
                "  {flag} {label:<52} device=0x{got:08x} cpu=0x{exp:08x} \
                 (device={:e} cpu={:e})",
                f32::from_bits(*got),
                f32::from_bits(*exp)
            );
        }
        println!(
            "  => {disallowed_mismatches}/{} lanes disallowed; \
             {floored_residuals} floor-covered residual flush(es); \
             {conservative_product_residuals} conservative product residual(s)",
            report.len()
        );
        let loading_contract_intact = device.denorm_preserve_contract_intact();
        assert_eq!(
            verdict,
            disallowed_mismatches == 0 && loading_contract_intact,
            "the gate's verdict must agree with the shared lane policy and \
             DenormPreserve loading-path contract"
        );

        // Fail-closed hook must refuse regardless of the hardware verdict.
        set_force_subnormal_selfcheck_fail(true);
        assert!(
            !device.verify_gradual_underflow(),
            "forced failure must refuse gradual-underflow authorization"
        );
        set_force_subnormal_selfcheck_fail(false);
        assert_eq!(
            device.verify_gradual_underflow(),
            verdict,
            "cache must survive the forced-fail hook"
        );
    }

    /// #flush-charge MEASUREMENT: characterize this adapter's flush class and
    /// pin its consistency with the rung-3 gate — `Conformant` iff the gate's
    /// lane policy passes, and the forced-fail hook always yields
    /// `NonConformant` (only ever MORE closed).
    #[test]
    fn flush_class_measured_and_consistent_with_the_gate() {
        let _serial = gpu_test_serial_guard();
        let device = require_device();
        let class = device.characterize_flush_policy();
        let verdict = device.verify_gradual_underflow();
        println!(
            "[flush-class probe] adapter={} backend={:?} => {class:?} (rung3 gate = {verdict})",
            device.adapter_info.name, device.adapter_info.backend,
        );
        if verdict {
            assert_eq!(
                class,
                FlushClass::Conformant,
                "a passing rung-3 gate must classify Conformant"
            );
        } else {
            assert_ne!(
                class,
                FlushClass::Conformant,
                "a refusing rung-3 gate cannot classify Conformant"
            );
        }
        // On the charged-mode target hardware the class must be exactly the
        // pinned measured table's class. Only assert when the raw lanes match
        // the pinned capture (so other adapters stay measurement-only).
        if let Ok(lanes) = device.run_gpu_checked("flush_class_lane_recheck", || {
            device.subnormal_selfcheck_raw()
        }) {
            if lanes[..] == cpu_tests::MEASURED_M5_MAX_LANES[..] {
                assert_eq!(class, FlushClass::PureFlush);
            }
        }

        set_force_subnormal_selfcheck_fail(true);
        assert_eq!(
            device.characterize_flush_policy(),
            FlushClass::NonConformant,
            "forced failure must refuse the flush characterization"
        );
        set_force_subnormal_selfcheck_fail(false);
    }
}
