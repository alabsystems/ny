// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! THE single predicate that decides whether a CUDA GEMM/CROWN result may carry
//! VERDICT authority — the CUDA twin of `ny-gpu`'s
//! `wgpu_device/ops/sound_authority.rs`, so that ONE measured ladder governs
//! BOTH backends and NY uses whatever hardware it is on.
//!
//! # THE FINDING THIS MODULE EXISTS TO CORRECT
//!
//! Before this module, `impl GpuCrownBackward for CudaGemmEngine` ended with
//!
//! ```text
//! fn provides_sound_gpu_crown(&self) -> bool {
//!     true
//! }
//! ```
//!
//! — a hardwired literal, exactly as symmetric to `WgpuDevice`'s old hardwired
//! `false` as it is possible to be, and just as disconnected from measurement.
//! `sound_gpu_gate.rs` filters EVERY verdict-carrying GPU CROWN route by that
//! predicate, so on CUDA the verdict path was authorized by an ASSERTION.
//!
//! The engine is not unmeasured — [`CudaGemmEngine::with_ordinal`] refuses
//! construction unless [`CudaGemmEngine::assert_ieee_bit_exact`] (four
//! known-answer Sgemm/Dgemm probes) and
//! [`CudaGemmEngine::assert_deadline_f64_transport_bit_exact`] pass. But those
//! probes and this predicate were never connected: `provides_sound_gpu_crown`
//! does not observe them, so any future refactor that demotes a construction
//! refusal to a warning, or any second construction route, silently keeps full
//! verdict authority. "Measured somewhere, asserted at the gate" is the exact
//! failure shape the wgpu ladder was built to remove.
//!
//! # WHAT THE CUDA PATH ASSERTS WITHOUT MEASURING
//!
//! CUDA has f64. f64 does NOT discharge the composition obligations:
//!
//! * **`#u1-cuda` — the production dispatch is not the probed dispatch.**
//!   `assert_ieee_bit_exact` probes `gemm_f32`/`gemm_f64`. The sound f64 CROWN
//!   (`sound_crown::linear_step_f64`, the only device work on the CUDA verdict
//!   path) calls `gemm_f64_triplet`, which on an ATS device with
//!   `NY_CUDA_DGEMM_TRIPLET=1` routes to `gemm_f64_triplet_ats` — a DIFFERENT
//!   dispatch: one lock, three `dgemm`s queued on borrowed pageable host
//!   pointers, one drain, its own quiescence proof. Nothing bit-checks it, and
//!   its three arms are `[a, |a|, a_err] × [w, |w|, |w|]` — the VALUE channel
//!   and the two MONOTONE ERROR channels. An arm swap or a short/aliased output
//!   in the error channels UNDER-CHARGES the certificate, which is the unsound
//!   direction. [`CudaGemmEngine::verify_triplet_composition`] is the settling
//!   test.
//! * **`#u3-cuda` — subnormal handling in the device GEMM is asserted.**
//!   `linear_step_f64`'s certificate is
//!   `err = (γ_k·s + prop)·(1+2γ_k) + 8·k·η`, where `s = fl(|a|@|w|)` and
//!   `prop = fl(a_err@|w|)` are computed BY THE DEVICE and are monotone
//!   non-negative. If cuBLAS flushed subnormals, `s` and `prop` come back
//!   SMALLER, so the charge is smaller — an under-charge, i.e. a bound that got
//!   tighter via an unproven path. NVIDIA f64 is IEEE-with-subnormals and
//!   `-ftz` applies only to f32 device code, but the CUDA path never MEASURED
//!   either. wgpu measures exactly this (rung 3) and this box FAILS it.
//!   [`CudaGemmEngine::verify_gemm_gradual_underflow`] measures it for CUDA.
//! * **`#u4-cuda` — sentinel taint. CUDA is STRICTLY WEAKER THAN wgpu here.**
//!   `ny_core` ships both `is_crown_coeff_safe` and `is_crown_coeff_safe_f64`;
//!   the CPU sound concretize applies the `CROWN_COEFF_MAX` degrade
//!   (`ny-propagate/src/bounds/concretize.rs`) and the wgpu sound concretize
//!   applies `abs(a) >= FALLBACK_BOUND ⇒ degrade`. Measured 2026-08-06:
//!
//!   ```text
//!   $ grep -c 'FALLBACK_BOUND\|CROWN_COEFF_MAX\|is_crown_coeff_safe' \
//!       crates/ny-cuda/src/{sound_crown,lib,joint_alpha,ieee_selfcheck}.rs
//!   0    0    0    0
//!   ```
//!
//!   The CUDA f64 CROWN has NO sentinel detection anywhere on its production
//!   path: a saturated f32 coefficient (`±FALLBACK_BOUND`, which STANDS FOR an
//!   unknown real of magnitude at least `1e10` and up to `~3.4e38`) enters
//!   `backward_f64_core` as an ordinary, confident `1e10` and is concretized
//!   into a finite published bound. [`sentinel_taint_sticky`] measures this —
//!   and, because the CUDA composition is HOST f64 with only the GEMM
//!   offloaded, it measures it **with no NVIDIA device attached**. This module
//!   is the only file in the crate that names those symbols, which is what
//!   keeps the grep above a live, re-runnable check.
//!
//! # THE LADDER (all must hold; short-circuit, fail-closed)
//!
//! | rung | name | where |
//! |------|------|-------|
//! | C0 | [`ladder_requested`] — `NY_CUDA_AUTHORITY_LADDER=1`, **DEFAULT OFF** | host |
//! | C1 | [`ny_core::dd_selfcheck::dd_selfcheck_ok`] — the host f64 reference is conformant | host |
//! | C2 | [`sentinel_taint_sticky`] (`#u4-cuda`) | host |
//! | C3 | [`CudaGemmEngine::verify_ieee_gemm_model`] — the known-answer probes, OBSERVED | device |
//! | C4 | [`CudaGemmEngine::verify_gemm_gradual_underflow`] (`#u3-cuda`) | device |
//! | C5 | [`CudaGemmEngine::verify_triplet_composition`] (`#u1-cuda`) | device |
//!
//! # WHY C0's DEFAULT-OFF MEANS `true` HERE AND `false` THERE
//!
//! Both gates default to *the shipped hardwire*, so both backends are
//! BYTE-IDENTICAL with the gate unset. wgpu's hardwire was `false`, so its gate
//! can only OPEN and arming is the risky direction. CUDA's hardwire is `true`,
//! so this gate can only CLOSE. That asymmetry is deliberate, and it is what
//! lets this land without weakening anything that is sound today and without
//! taking away CUDA acceleration from anyone on the strength of probes that no
//! NVIDIA device in this workflow could run.
//!
//! `NY_CUDA_AUTHORITY_LADDER=1` therefore requests *more* scrutiny, never less.
//! There is no value of any environment variable that grants CUDA verdict
//! authority which the pre-existing hardwire did not already grant.
//!
//! # WHAT IS DELIBERATELY NOT CHANGED
//!
//! [`GemmEngine::as_gpu_crown_backward`] still returns `Some(self)`
//! unconditionally. On wgpu the two seams move together because every
//! `as_gpu_crown_backward` consumer there is verdict-facing; on CUDA that
//! method is also the entry point for NON-verdict acceleration (α-steering
//! gradients, β-gradient gathers, the joint adjoint), which
//! `sound_gpu_gate.rs` reaches WITHOUT the `provides_sound_gpu_crown` filter.
//! Returning `None` would delete acceleration, not unsoundness. The verdict
//! chokepoint is `provides_sound_gpu_crown`, and that is what this ladder
//! governs.
//!
//! # HARDWARE
//!
//! Rungs C3–C5 dispatch on a real NVIDIA device when the test host advertises
//! the CUDA libraries and a visible device. Hardware-free CI exercises the
//! pure, fail-closed admission seam instead; there are no permanently ignored
//! tests. Rungs C0–C2 are host-side and always measured.
//!
//! # MEASURED VERDICT for rung C2 on Apple M5 Max, 2026-08-06, NO CUDA DEVICE
//!
//! `cargo test -p ny-cuda report_cuda_sentinel_taint_lanes -- --nocapture`:
//!
//! ```text
//! lane 0 cancel-add, slope 1      v=0      lin_err=8.88e-6  coeff=0      err=1.78e-5   FAIL
//! lane 1 cancel-add, slope 0      v=0      lin_err=8.88e-6  coeff=0      err=4.94e-324 PASS
//! lane 2 sentinel * 1e-20 weight  v=1e-10  lin_err=4.44e-26 coeff=1e-10  err=9.99e-26  FAIL
//! lane 3 sentinel * 1 weight      v=1e10   lin_err=4.44e-6  coeff=1e10   err=9.99e-6   PASS
//! lane 4 sentinel err, slope 0    v=1      lin_err=1e10     coeff=0      err=4.94e-324 PASS
//! lane 5 sentinel err * 1e-25     v=1      lin_err=1e10     coeff=1e-25  err=2e-15     FAIL
//! consumer: concretize_f64(coeff=1e10, box=[-1e-20,1e-20]) -> (-1e-10, 1e-10) degraded=false
//! => sentinel_taint_sticky() = false
//! ```
//!
//! Read against `ny-gpu`'s identical six-lane table on the same box, CUDA is
//! **strictly weaker than Metal**:
//!
//! * **lane 0 — wgpu PASSES, CUDA FAILS.** wgpu's `|A|@|W|` channel saturates
//!   to `FALLBACK_BOUND` in f32 and the AW combine's `s_prod >= FALLBACK_BOUND`
//!   arm fires the `1e30` degrade. CUDA computes `|a|@|w| = 2e10` in f64, which
//!   neither saturates nor is compared against anything, so the certified error
//!   comes out as an ordinary `γ_4·2e10 ≈ 1.8e-5`. **f64's wider range removed
//!   the saturation that was carrying the taint, and nothing replaced it.**
//! * **lanes 2 and 5 — both backends FAIL identically.** Downscaling launders a
//!   magnitude on any hardware; this is a kernel/algorithm hole, not an adapter
//!   property, exactly as `ny-gpu`'s module concluded.
//! * **lane 3 + the consumer line — the sharpest result.** The taint SURVIVES
//!   the CUDA composition (`coeff = 1e10 ≥ FALLBACK_BOUND`) and is then
//!   DISCARDED at the consumer: `concretize_f64` has no coefficient guard, so it
//!   publishes a confident `±1e-10` where the true coefficient is unknown and
//!   the true contribution reaches `~3.4e18`. On wgpu the corresponding
//!   concretize has `abs(a) >= FALLBACK_BOUND ⇒ degrade` and this case is
//!   caught.
//!
//! REACHABILITY IS NOT CLAIMED. This measures the MECHANISM. Whether a
//! sentinel-valued f32 coefficient can actually reach `backward_f64_core` (the
//! candidate routes are the `spec`/`GpuCrownSeed` coefficients, which the CUDA
//! path's own docs say are "treated as exact") is a separate question this arc
//! did NOT settle. The moat's rule for that state is to refuse, which is what
//! the rung does.
//!
//! # WHAT WOULD MAKE C2 PASS
//!
//! The two shapes `ny-gpu` names both port, plus one that is CUDA-specific and
//! cheap: call the already-shipped `ny_core::is_crown_coeff_safe_f64` on every
//! incoming f32→f64 coefficient at the `backward_f64_core` boundary and degrade
//! the row, which is exactly what the CPU reference concretize does. That
//! closes lane 3 and the consumer line; lanes 0/2/5 additionally need the taint
//! to stop being a magnitude.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use ny_core::{GemmEngine, NyError, Result};

use crate::sound_crown::{activation_step_f64, concretize_f64, linear_step_f64};
use crate::CudaGemmEngine;

/// Environment gate: `NY_CUDA_AUTHORITY_LADDER=1` opts this process in to
/// probe-governed CUDA verdict authority. **Default OFF**, and default-off is
/// byte-identical to the historic hardwired `provides_sound_gpu_crown() ==
/// true`.
///
/// Read once per process. Any value other than exactly `"1"` leaves the ladder
/// disengaged — the same strict parse the wgpu gate uses, so neither gate can
/// be moved by a typo.
pub(crate) fn ladder_requested() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| {
        ladder_requested_from(std::env::var("NY_CUDA_AUTHORITY_LADDER").ok().as_deref())
    })
}

fn ladder_requested_from(value: Option<&str>) -> bool {
    value == Some("1")
}

/// The composed authority POLICY, as a pure function of the gate and the five
/// rung verdicts.
///
/// Extracted so the load-bearing claim — *ladder disengaged ⇒ the predicate is
/// the historic hardwired `true`, whatever any probe says* — is testable with no
/// NVIDIA device attached, which is the only way it could be tested in this
/// workflow. [`CudaGemmEngine::sound_gpu_authority`] implements the same policy
/// with short-circuit evaluation so that a disengaged ladder also dispatches
/// nothing.
const fn ladder_grants(gate: bool, rungs: [bool; 5]) -> bool {
    if !gate {
        // Byte-identical to `fn provides_sound_gpu_crown(&self) -> bool { true }`.
        return true;
    }
    rungs[0] && rungs[1] && rungs[2] && rungs[3] && rungs[4]
}

/// Test/operator hooks. Each only ever forces a rung MORE closed.
static FORCE_SENTINEL_FAIL: AtomicBool = AtomicBool::new(false);

fn env_forces(var: &str) -> bool {
    std::env::var_os(var).is_some()
}

// ---------------------------------------------------------------------------
// C2 (`#u4-cuda`) — SENTINEL TAINT, measured on the HOST composition.
// ---------------------------------------------------------------------------

/// The finite overflow sentinel every f32 producer saturates to. Mirrors
/// `ny_core::FALLBACK_BOUND` / `ny_core::CROWN_COEFF_MAX` (they are equal by
/// construction — see `ny-core/src/gemm.rs`), lifted to f64 because the CUDA
/// chain carries coefficients in f64.
const FALLBACK: f64 = ny_core::FALLBACK_BOUND as f64;

/// Contraction length of each probe lane. Matches the wgpu `#u4` probe so the
/// two backends' lane tables are directly comparable.
const K: usize = 4;

/// What the shipped CUDA composition must do with the taint on a given lane.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Expect {
    /// The composition does NOT annihilate the sentinel in exact arithmetic, so
    /// the taint MUST still be detectable at the end of the chain. A lane that
    /// loses it publishes a small confident number in place of an unknown one.
    StaySticky,
    /// The composition multiplies by EXACT ZERO. `R·0 == 0` for every finite
    /// real `R`, and the sentinel always stands for a finite real (an f32
    /// overflow of finite operands), so dropping the taint is justified. The
    /// lane still pins that the value is EXACTLY `0.0`.
    AnnihilateExactly,
}

/// One probe lane, run through the SHIPPED `sound_crown` composition:
/// `linear_step_f64` then `activation_step_f64`.
struct Lane {
    label: &'static str,
    /// Incoming f64 coefficient row (`m = 1`, `k = K`).
    a: [f64; K],
    /// f32 weight column (`k = K`, `n = 1`) — weights are f32 on this path.
    w: [f32; K],
    /// Incoming certified f64 coefficient error.
    e: [f64; K],
    /// Activation slope (`ls == us`, so the shader-equivalent sign routing
    /// cannot make the lane ambiguous).
    slope: f32,
    expect: Expect,
    /// Why that expectation is the SOUND one, in one line.
    rationale: &'static str,
}

/// The lane table. Lanes 0/3 establish the mechanism CAN carry the taint (so a
/// failure elsewhere is a real hole and not a probe that never armed); lanes
/// 2/5 are the laundering adversaries; lanes 1/4 are the exact-zero
/// annihilations. Deliberately the SAME six lanes as
/// `ny-gpu`'s `ops/sentinel_taint_selfcheck.rs`, so the CUDA and Metal verdicts
/// can be read side by side.
const LANES: [Lane; 6] = [
    Lane {
        label: "0 cancel-add, slope 1        (a@w cancels; |a|@|w| must not)",
        a: [1e10, -1e10, 0.0, 0.0],
        w: [1.0, 1.0, 0.0, 0.0],
        e: [0.0; K],
        slope: 1.0,
        expect: Expect::StaySticky,
        rationale: "true coefficients are unknown reals |R| >= 1e10; R1 + R2 is \
                    NOT 0, so the cancelled value channel must be caught by the \
                    monotone |a|@|w| channel",
    },
    Lane {
        label: "1 cancel-add, slope 0 EXACT  (dead ReLU annihilation)",
        a: [1e10, -1e10, 0.0, 0.0],
        w: [1.0, 1.0, 0.0, 0.0],
        e: [0.0; K],
        slope: 0.0,
        expect: Expect::AnnihilateExactly,
        rationale: "(R1 + R2)*0 == 0 exactly for every finite R, so dropping the \
                    taint is justified — the CPU reference still degrades here",
    },
    Lane {
        label: "2 sentinel * 1e-20 weight    (DOWNSCALE LAUNDER, value channel)",
        a: [1e10, 0.0, 0.0, 0.0],
        w: [1e-20, 0.0, 0.0, 0.0],
        e: [0.0; K],
        slope: 1.0,
        expect: Expect::StaySticky,
        rationale: "stored 1e10 stands for a true |R| up to ~3.4e38; R*1e-20 is \
                    up to 3.4e18 while the chain publishes 1e-10",
    },
    Lane {
        label: "3 sentinel * 1 weight        (armed control: taint MUST survive)",
        a: [1e10, 0.0, 0.0, 0.0],
        w: [1.0, 0.0, 0.0, 0.0],
        e: [0.0; K],
        slope: 1.0,
        expect: Expect::StaySticky,
        rationale: "no scaling at all — if even this loses the taint the probe \
                    itself is misconfigured",
    },
    Lane {
        label: "4 sentinel-magnitude err, slope 0 (dead ReLU annihilation, err channel)",
        a: [1.0, 0.0, 0.0, 0.0],
        w: [1.0, 0.0, 0.0, 0.0],
        e: [FALLBACK, 0.0, 0.0, 0.0],
        slope: 0.0,
        expect: Expect::AnnihilateExactly,
        rationale: "the coefficient is annihilated exactly, so its error budget \
                    legitimately goes with it",
    },
    Lane {
        label: "5 sentinel-magnitude err * 1e-25  (DOWNSCALE LAUNDER, error channel)",
        a: [1.0, 0.0, 0.0, 0.0],
        w: [1.0, 0.0, 0.0, 0.0],
        e: [FALLBACK, 0.0, 0.0, 0.0],
        slope: 1e-25,
        expect: Expect::StaySticky,
        rationale: "an error at the overflow sentinel means the true error is \
                    UNKNOWN and at least 1e10, not exactly 1e10, so scaling it \
                    by a Lipschitz factor is not a valid transport",
    },
];

/// The end-of-chain taint predicate — the SAME magnitude test the CPU reference
/// (`ny_core::is_crown_coeff_safe_f64`) and the wgpu concretize preflight apply.
///
/// NOTE the asymmetry this module reports: on wgpu this predicate corresponds to
/// code that actually RUNS at concretize time. In `ny-cuda` **nothing consults
/// it** — see [`published_bound_is_degraded`].
fn taint_visible(coeff: f64, err: f64) -> bool {
    !coeff.is_finite() || !err.is_finite() || coeff.abs() >= FALLBACK || err >= FALLBACK
}

/// Per-lane measurement.
#[derive(Copy, Clone, Debug)]
pub struct LaneOutcome {
    /// Which lane.
    pub label: &'static str,
    /// Why the pinned expectation is the sound one.
    pub rationale: &'static str,
    /// Value channel out of `linear_step_f64`.
    pub value: f64,
    /// Certified error out of `linear_step_f64`.
    pub linear_err: f64,
    /// End-of-chain coefficient, after `activation_step_f64`.
    pub coeff: f64,
    /// End-of-chain certified error.
    pub err: f64,
    /// Would a magnitude guard still see the taint?
    pub tainted_at_end: bool,
    /// Did the lane meet its pinned expectation?
    pub ok: bool,
    /// `true` for lanes whose expectation is `StaySticky` (the taint must
    /// still be detectable), `false` for the exact-zero annihilation lanes.
    pub expects_sticky: bool,
}

/// Run one lane through the SHIPPED CUDA composition on `eng`.
///
/// `eng` supplies only the `gemm_f64_triplet` the linear step offloads; every
/// other operation is the same host f64 code the CUDA verdict path runs. That
/// is why this rung is measurable without an NVIDIA device: pass
/// `ny_core::NaiveCpuGemmEngine` and the ONLY difference from a live GB10 is
/// the reduction order of an exactly-representable product, which cannot change
/// any lane's verdict.
fn run_lane<E: GemmEngine + ?Sized>(eng: &E, lane: &Lane) -> Result<LaneOutcome> {
    let (v, e1) = linear_step_f64(eng, 1, K, 1, &lane.a, &lane.e, &lane.w)?;
    let (la, _ua, le, _ue) = activation_step_f64(
        1,
        1,
        &v,
        &v,
        &e1,
        &e1,
        std::slice::from_ref(&lane.slope),
        std::slice::from_ref(&lane.slope),
    )?;
    let coeff = la[0];
    let err = le[0];
    let tainted_at_end = taint_visible(coeff, err);
    let ok = match lane.expect {
        Expect::StaySticky => tainted_at_end,
        // Exact annihilation: the value must be EXACTLY zero (not merely
        // small), and no taint may be claimed.
        Expect::AnnihilateExactly => coeff == 0.0 && !tainted_at_end,
    };
    Ok(LaneOutcome {
        label: lane.label,
        rationale: lane.rationale,
        value: v[0],
        linear_err: e1[0],
        coeff,
        err,
        tainted_at_end,
        ok,
        expects_sticky: lane.expect == Expect::StaySticky,
    })
}

/// Every lane's outcome, in table order.
pub fn sentinel_taint_lanes<E: GemmEngine + ?Sized>(eng: &E) -> Result<Vec<LaneOutcome>> {
    LANES.iter().map(|lane| run_lane(eng, lane)).collect()
}

/// The CONSUMER arm: does the shipped `concretize_f64` DEGRADE when the
/// coefficient it is handed is the overflow sentinel?
///
/// A sentinel coefficient of `1e10` stands for an unknown real of magnitude at
/// least `1e10`. Over the input box `[-1e-20, 1e-20]` the true contribution is
/// therefore anywhere up to `~3.4e18`, and the only sound publication is a
/// degrade (`±inf`, or at minimum a bound at/above the sentinel). Returns
/// `(lo, hi, degraded)`.
pub fn published_bound_is_degraded() -> (f32, f32, bool) {
    // One spec, one input dimension, coefficient at the sentinel, zero error,
    // zero accumulated bias — the cleanest possible statement of the question.
    let (lo_v, hi_v) = concretize_f64(
        1,
        1,
        &[FALLBACK],
        &[FALLBACK],
        &[0.0],
        &[0.0],
        &[-1e-20],
        &[1e-20],
        &[0.0],
        &[0.0],
        &[0.0],
        &[0.0],
    );
    let (lo, hi) = (lo_v[0], hi_v[0]);
    let degraded = !lo.is_finite()
        || !hi.is_finite()
        || lo <= -ny_core::FALLBACK_BOUND
        || hi >= ny_core::FALLBACK_BOUND;
    (lo, hi, degraded)
}

/// `#u4-cuda` rung: does the finite `±FALLBACK_BOUND` overflow sentinel survive
/// the CUDA sound CROWN composition, and does anything downstream ACT on it?
///
/// `true` ⇒ every lane behaved as soundness requires AND the concretize
/// consumer degrades on a sentinel coefficient. `false` ⇒ REFUSED (fail-closed):
/// at least one lane laundered the taint into a small confident number, or the
/// consumer published a confident finite bound, or the probe could not run.
///
/// Host-side and backend-independent (the CUDA sound CROWN composes in host
/// f64 and offloads only the GEMM), so it is cached
/// process-globally rather than per engine.
pub fn sentinel_taint_sticky() -> bool {
    if FORCE_SENTINEL_FAIL.load(Ordering::Relaxed)
        || env_forces("NY_FORCE_CUDA_SENTINEL_TAINT_SELFCHECK_FAIL")
    {
        return false;
    }
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let Ok(lanes) = sentinel_taint_lanes(&ny_core::NaiveCpuGemmEngine) else {
            return false;
        };
        let lanes_ok = lanes.iter().all(|l| l.ok);
        let (_, _, degraded) = published_bound_is_degraded();
        if !lanes_ok || !degraded {
            tracing::warn!(
                target: "ny_cuda::authority",
                failing_lanes = ?lanes.iter().filter(|l| !l.ok).map(|l| l.label).collect::<Vec<_>>(),
                consumer_degrades = degraded,
                "#u4-cuda SENTINEL-TAINT self-check FAILED: the CUDA sound f64 \
                 CROWN does not carry the ±FALLBACK_BOUND overflow sentinel \
                 through its fused composition, and `sound_crown::concretize_f64` \
                 applies no `is_crown_coeff_safe_f64` / CROWN_COEFF_MAX degrade \
                 at all — a saturated f32 coefficient is published as a confident \
                 finite bound"
            );
        }
        lanes_ok && degraded
    })
}

/// Test hook: force / release a `#u4-cuda` self-check failure.
#[cfg(test)]
pub(crate) fn set_force_sentinel_taint_fail(force: bool) {
    FORCE_SENTINEL_FAIL.store(force, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// C3–C5 — the on-device rungs.
// ---------------------------------------------------------------------------

/// CROWN-shaped probe dimensions for [`CudaGemmEngine::verify_triplet_composition`].
///
/// `256³ = 16 777 216` MACs is exactly `sound_crown::MIN_RESIDENT_MACS` (`1 <<
/// 24`), the size gate below which the CUDA sound CROWN refuses the GPU and
/// falls back to the CPU. So this is the SMALLEST shape the production path
/// ever runs — deliberately, because cuBLAS heuristics pick kernels by shape and
/// a probe at a shape production never uses proves nothing about production.
const TRIPLET_M: usize = 256;
const TRIPLET_K: usize = 256;
const TRIPLET_N: usize = 256;

/// f64 subnormal `2^-1070` (well inside the subnormal range; `2^-1074` is the
/// smallest). Its product with `1.0` is itself, and with `2^52` is the NORMAL
/// value `2^-1018`.
const SUBNORMAL_F64: u64 = 0x0000_0000_0000_0010;
/// f32 subnormal `2^-140`. Product with `1.0` is itself; with `2^30` it is the
/// NORMAL `2^-110`.
const SUBNORMAL_F32: u32 = 0x0000_2000;

impl CudaGemmEngine {
    /// Rung C3: the known-answer IEEE probes, OBSERVED by the authority
    /// predicate instead of merely having gated construction.
    ///
    /// Cached per engine. Any error ⇒ `false` (fail-closed).
    pub fn verify_ieee_gemm_model(&self) -> bool {
        if env_forces("NY_FORCE_CUDA_IEEE_SELFCHECK_FAIL") {
            return false;
        }
        *self
            .ieee_gemm_model_rung
            .get_or_init(|| match self.assert_ieee_bit_exact() {
                Ok(()) => true,
                Err(e) => {
                    tracing::warn!(
                        target: "ny_cuda::authority",
                        error = %e,
                        "rung C3 verify_ieee_gemm_model REFUSED"
                    );
                    false
                }
            })
    }

    /// Rung C4 (`#u3-cuda`): the device GEMM must honour GRADUAL UNDERFLOW — no
    /// FTZ on results, no DAZ on operands, in BOTH f64 and f32.
    ///
    /// # Why this is a soundness rung and not a nicety
    ///
    /// `sound_crown::linear_step_f64` charges
    /// `err = (γ_k·s + prop)·(1 + 2γ_k) + 8·k·η` where `s = fl(|a|@|w|)` and
    /// `prop = fl(a_err@|w|)` are **computed by the device**. Both are monotone
    /// non-negative channels. Flushing makes them SMALLER, so the certified
    /// error gets SMALLER — a bound tightened by an unproven path, which the
    /// moat forbids outright. The `8·k·η` additive term is an underflow floor
    /// for the HOST arithmetic; it is not a licence for the device to flush.
    ///
    /// The f32 arm matters because `gemm_f32` backs the abs-sum `S` seam, which
    /// is monotone non-negative for the same reason.
    ///
    /// Cached per engine. Any error or any bit deviation ⇒ `false`.
    pub fn verify_gemm_gradual_underflow(&self) -> bool {
        if env_forces("NY_FORCE_CUDA_SUBNORMAL_SELFCHECK_FAIL") {
            return false;
        }
        *self
            .gradual_underflow_rung
            .get_or_init(|| match self.run_gradual_underflow_probe() {
                Ok(()) => true,
                Err(e) => {
                    tracing::warn!(
                        target: "ny_cuda::authority",
                        error = %e,
                        "rung C4 verify_gemm_gradual_underflow REFUSED: this device's \
                         GEMM flushes subnormals, so the monotone |a|@|w| and \
                         err@|w| error channels can come back SMALLER than the \
                         certificate assumes (an UNDER-CHARGE)"
                    );
                    false
                }
            })
    }

    /// The four gradual-underflow known answers. Every expectation is exact in
    /// the target type, so a conformant engine returns ONE bit pattern for any
    /// reduction order.
    fn run_gradual_underflow_probe(&self) -> Result<()> {
        // f64 FTZ: subnormal × 1 must come back as the same subnormal.
        let sub64 = f64::from_bits(SUBNORMAL_F64);
        let got = self.gemm_f64(1, 1, 1, &[sub64], &[1.0])?;
        expect_bits_f64("f64 FTZ (subnormal result)", &got, &[sub64])?;

        // f64 DAZ: subnormal × 2^52 is the NORMAL 2^-1018. A device that
        // zeroes the OPERAND returns 0 even though the result is normal, so
        // this arm is the one an FTZ-only check would miss.
        let scale64 = f64::from_bits(0x4330_0000_0000_0000); // 2^52
        let got = self.gemm_f64(1, 1, 1, &[sub64], &[scale64])?;
        expect_bits_f64("f64 DAZ (subnormal operand)", &got, &[sub64 * scale64])?;

        // f32 twins for the abs-sum `S` seam.
        let sub32 = f32::from_bits(SUBNORMAL_F32);
        let got = self.gemm_f32(1, 1, 1, &[sub32], &[1.0])?;
        expect_bits_f32("f32 FTZ (subnormal result)", &got, &[sub32])?;

        let scale32 = f32::from_bits(0x4E80_0000); // 2^30
        let got = self.gemm_f32(1, 1, 1, &[sub32], &[scale32])?;
        expect_bits_f32("f32 DAZ (subnormal operand)", &got, &[sub32 * scale32])?;
        Ok(())
    }

    /// Rung C5 (`#u1-cuda`): the PRODUCTION dispatch the sound CROWN actually
    /// calls — `gemm_f64_triplet` — must agree BIT-FOR-BIT with (a) the exact
    /// mathematical answer and (b) three separate `gemm_f64` calls, at a
    /// CROWN-shaped `(m, k, n)`.
    ///
    /// This is the CUDA analogue of the wgpu obligation U1. `assert_ieee_bit_exact`
    /// probes `gemm_f64`; the verdict path calls `gemm_f64_triplet`, which on an
    /// ATS device with `NY_CUDA_DGEMM_TRIPLET=1` is a wholly different dispatch
    /// (`gemm_f64_triplet_ats`: one lock, three queued `dgemm`s on borrowed
    /// pageable host pointers, one drain). The three arms are the VALUE channel
    /// and the two MONOTONE ERROR channels, so an arm swap, a stale output
    /// buffer, or a short write in arms 1–2 UNDER-CHARGES the certificate.
    ///
    /// The three arms carry DIFFERENT exact answers (`×1`, `×2`, `×4`) precisely
    /// so a permutation or an aliased output cannot pass.
    ///
    /// Cached per engine. Any error or deviation ⇒ `false`.
    pub fn verify_triplet_composition(&self) -> bool {
        if env_forces("NY_FORCE_CUDA_TRIPLET_SELFCHECK_FAIL") {
            return false;
        }
        *self
            .triplet_composition_rung
            .get_or_init(|| match self.run_triplet_composition_probe() {
                Ok(()) => true,
                Err(e) => {
                    tracing::warn!(
                        target: "ny_cuda::authority",
                        error = %e,
                        "rung C5 verify_triplet_composition REFUSED: the fused \
                         gemm_f64_triplet the sound CROWN calls does not reproduce \
                         the probed single-GEMM dispatch bit-for-bit"
                    );
                    false
                }
            })
    }

    fn run_triplet_composition_probe(&self) -> Result<()> {
        let (a, b_scales, wants) = triplet_probe_operands();
        let b0: Vec<f64> = b_scales[0].clone();
        let b1: Vec<f64> = b_scales[1].clone();
        let b2: Vec<f64> = b_scales[2].clone();

        let fused = self.gemm_f64_triplet(
            TRIPLET_M,
            TRIPLET_K,
            TRIPLET_N,
            [&a, &a, &a],
            [&b0, &b1, &b2],
        )?;

        // (a) exact known answer, per arm.
        for (arm, (got, want)) in fused.iter().zip(&wants).enumerate() {
            let expected = vec![*want; TRIPLET_M * TRIPLET_N];
            expect_bits_f64(&format!("triplet arm {arm} known answer"), got, &expected)?;
        }

        // (b) the fused transaction must equal the PROBED single dispatch,
        // element for element. This is the composed-sequence integrity claim.
        for (arm, b) in [&b0, &b1, &b2].into_iter().enumerate() {
            let single = self.gemm_f64(TRIPLET_M, TRIPLET_K, TRIPLET_N, &a, b)?;
            expect_bits_f64(
                &format!("triplet arm {arm} vs single gemm_f64"),
                &fused[arm],
                &single,
            )?;
        }
        Ok(())
    }

    /// THE authority predicate. `true` ⇒ a sound CUDA CROWN result from this
    /// engine may carry verdict authority.
    ///
    /// With `NY_CUDA_AUTHORITY_LADDER` unset this returns `true` unconditionally
    /// and dispatches NOTHING — byte-identical to the historic hardwire (see the
    /// module docs for why the two backends' gates have opposite polarity).
    /// With the ladder engaged, authority is the conjunction of every rung and
    /// every failure mode is a refusal.
    pub fn sound_gpu_authority(&self) -> bool {
        if !ladder_requested() {
            return true;
        }
        // C1 (cheapest, host-side): the host f64 reference must be conformant.
        // `sound_crown` already demands this at the resident-cut carrier; as a
        // rung, the WHOLE path observes it.
        if !ny_core::dd_selfcheck::dd_selfcheck_ok() {
            return false;
        }
        // C2 (#u4-cuda), host-side and backend-independent.
        if !sentinel_taint_sticky() {
            return false;
        }
        // C3: the known-answer probes, observed rather than assumed.
        if !self.verify_ieee_gemm_model() {
            return false;
        }
        // C4 (#u3-cuda): no FTZ/DAZ in the monotone error channels.
        if !self.verify_gemm_gradual_underflow() {
            return false;
        }
        // C5 (#u1-cuda): the production triplet dispatch composes correctly.
        if !self.verify_triplet_composition() {
            return false;
        }
        true
    }

    /// Full rung-by-rung report, for logging and for tests. `None` on a rung
    /// means "not evaluated" (the ladder is disengaged), which is exactly the
    /// state that reproduces the historic hardwire.
    pub fn authority_ladder_report(&self) -> CudaAuthorityLadder {
        let gate = ladder_requested();
        if !gate {
            return CudaAuthorityLadder {
                gate,
                granted: ladder_grants(false, [false; 5]),
                ..CudaAuthorityLadder::default()
            };
        }
        let host_f64_reference = ny_core::dd_selfcheck::dd_selfcheck_ok();
        let sentinel = sentinel_taint_sticky();
        let ieee = self.verify_ieee_gemm_model();
        let underflow = self.verify_gemm_gradual_underflow();
        let triplet = self.verify_triplet_composition();
        CudaAuthorityLadder {
            gate,
            host_f64_reference: Some(host_f64_reference),
            sentinel_taint_sticky: Some(sentinel),
            ieee_gemm_model: Some(ieee),
            gemm_gradual_underflow: Some(underflow),
            triplet_composition: Some(triplet),
            granted: ladder_grants(
                gate,
                [host_f64_reference, sentinel, ieee, underflow, triplet],
            ),
        }
    }

    /// Evaluate the ladder once at construction (when engaged) so later reads
    /// hit warm caches, and log the verdict. No-op when the ladder is
    /// disengaged, which is what keeps the default path free as well as
    /// byte-identical.
    pub(crate) fn prime_sound_gpu_authority(&self) {
        if !ladder_requested() {
            return;
        }
        let report = self.authority_ladder_report();
        if report.granted {
            tracing::warn!(
                target: "ny_cuda::authority",
                device = %self.device_name(),
                "NY_CUDA_AUTHORITY_LADDER=1 and every rung PASSED: this device is \
                 measured-qualified for sound CUDA CROWN verdict authority. \
                 Arming for production remains a HUMAN decision."
            );
        } else {
            tracing::warn!(
                target: "ny_cuda::authority",
                device = %self.device_name(),
                report = ?report,
                "NY_CUDA_AUTHORITY_LADDER=1 but the rung ladder did NOT pass: CUDA \
                 GPU verdict authority REFUSED (fail-closed); the CPU sound path \
                 carries the verdict. NOTE: with the ladder DISENGAGED (the \
                 default) authority would have been GRANTED by the historic \
                 hardwire without any of these measurements."
            );
        }
    }
}

/// Rung-by-rung ladder state. `None` ⇒ not evaluated (ladder disengaged).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CudaAuthorityLadder {
    /// C0 — `NY_CUDA_AUTHORITY_LADDER=1`.
    pub gate: bool,
    /// C1 — host f64 (double-double) reference conformance.
    pub host_f64_reference: Option<bool>,
    /// C2 — `#u4-cuda` overflow-sentinel taint stickiness.
    pub sentinel_taint_sticky: Option<bool>,
    /// C3 — known-answer IEEE Sgemm/Dgemm bit-exactness.
    pub ieee_gemm_model: Option<bool>,
    /// C4 — `#u3-cuda` gradual underflow in the device GEMM.
    pub gemm_gradual_underflow: Option<bool>,
    /// C5 — `#u1-cuda` production triplet-dispatch composition.
    pub triplet_composition: Option<bool>,
    /// The composed verdict `provides_sound_gpu_crown()` reports.
    pub granted: bool,
}

/// Operands for the CROWN-shaped triplet probe: `a` rows are
/// `[1 + 2^-52, 2^-52 × (k-1)]`, and the three `b` arms are all-`1`, all-`2`,
/// all-`4`. Every partial sum is `s·(1 + m·2^-52)` with `m ≤ 256` and
/// `s ∈ {1,2,4}` — at most 53 significand bits, hence exact, hence
/// reduction-order independent.
fn triplet_probe_operands() -> (Vec<f64>, [Vec<f64>; 3], [f64; 3]) {
    let mut a = vec![f64::EPSILON; TRIPLET_M * TRIPLET_K];
    for row in 0..TRIPLET_M {
        a[row * TRIPLET_K] = 1.0 + f64::EPSILON;
    }
    let mut base = 1.0f64 + f64::EPSILON;
    for _ in 1..TRIPLET_K {
        base += f64::EPSILON;
    }
    let scales = [1.0f64, 2.0, 4.0];
    let bs = [
        vec![scales[0]; TRIPLET_K * TRIPLET_N],
        vec![scales[1]; TRIPLET_K * TRIPLET_N],
        vec![scales[2]; TRIPLET_K * TRIPLET_N],
    ];
    (a, bs, [base, base * 2.0, base * 4.0])
}

fn expect_bits_f64(probe: &str, got: &[f64], want: &[f64]) -> Result<()> {
    if got.len() != want.len() {
        return Err(NyError::InternalError(format!(
            "cuda authority: {probe} returned {} elements, want {}",
            got.len(),
            want.len()
        )));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            return Err(NyError::InternalError(format!(
                "cuda authority: {probe} NOT bit-exact at [{i}]: got {g:e} ({:#018x}), \
                 want {w:e} ({:#018x})",
                g.to_bits(),
                w.to_bits()
            )));
        }
    }
    Ok(())
}

fn expect_bits_f32(probe: &str, got: &[f32], want: &[f32]) -> Result<()> {
    if got.len() != want.len() {
        return Err(NyError::InternalError(format!(
            "cuda authority: {probe} returned {} elements, want {}",
            got.len(),
            want.len()
        )));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            return Err(NyError::InternalError(format!(
                "cuda authority: {probe} NOT bit-exact at [{i}]: got {g:e} ({:#010x}), \
                 want {w:e} ({:#010x})",
                g.to_bits(),
                w.to_bits()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate must be OFF unless the environment says exactly `"1"` — the same
    /// strict parse the wgpu gate uses.
    #[test]
    fn only_exact_one_engages_the_ladder() {
        assert!(!ladder_requested_from(None), "unset must not engage");
        assert!(!ladder_requested_from(Some("")), "empty must not engage");
        assert!(!ladder_requested_from(Some("0")), "0 must not engage");
        assert!(!ladder_requested_from(Some("true")), "true must not engage");
        assert!(
            !ladder_requested_from(Some(" 1")),
            "whitespace must not engage"
        );
        assert!(ladder_requested_from(Some("1")), "exactly 1 engages");
    }

    /// DEFAULT-OFF PIN without depending on the test runner's process-wide
    /// environment. The previous version silently returned when an operator set
    /// the variable, leaving this policy untested in precisely the armed run.
    #[test]
    fn absent_ladder_value_is_off_and_reproduces_the_hardwire() {
        assert!(!ladder_requested_from(None));
        assert!(ladder_grants(false, [false; 5]));
    }

    /// ORACLE (a): with the ladder DISENGAGED the predicate is the old literal
    /// `true` for EVERY possible rung assignment — so the default build cannot
    /// differ from the pre-ladder build by any route, and no bound and no
    /// verdict can move. Exhaustive over all 32 rung assignments.
    #[test]
    fn gate_off_is_byte_identical_to_the_old_hardwire() {
        for mask in 0u8..32 {
            let rungs = [
                mask & 1 != 0,
                mask & 2 != 0,
                mask & 4 != 0,
                mask & 8 != 0,
                mask & 16 != 0,
            ];
            assert!(
                ladder_grants(false, rungs),
                "ladder disengaged must reproduce `true` for rungs {rungs:?}"
            );
        }
    }

    /// ORACLE (b): engaged, authority is the CONJUNCTION — any single failing
    /// rung refuses, and only the all-pass assignment grants.
    #[test]
    fn gate_on_refuses_unless_every_rung_passes() {
        assert!(ladder_grants(true, [true; 5]), "all rungs pass ⇒ granted");
        for i in 0..5 {
            let mut rungs = [true; 5];
            rungs[i] = false;
            assert!(
                !ladder_grants(true, rungs),
                "rung {i} failing must refuse the ladder"
            );
        }
        assert!(!ladder_grants(true, [false; 5]));
    }

    /// ORACLE (c): the engaged ladder can only ever be MORE closed than the
    /// disengaged one. This is the property that makes landing the CUDA gate
    /// safe by construction — there is no rung assignment and no environment
    /// under which engaging it grants authority the hardwire did not.
    #[test]
    fn engaging_the_ladder_can_only_close() {
        for mask in 0u8..32 {
            let rungs = [
                mask & 1 != 0,
                mask & 2 != 0,
                mask & 4 != 0,
                mask & 8 != 0,
                mask & 16 != 0,
            ];
            assert!(
                !ladder_grants(true, rungs) || ladder_grants(false, rungs),
                "engaged must imply disengaged (monotone-closing) for {rungs:?}"
            );
        }
    }

    /// The taint predicate is the CPU reference's own guard, so the two can
    /// never drift apart silently.
    #[test]
    fn taint_predicate_matches_the_core_coefficient_guard() {
        for v in [
            0.0,
            1.0,
            9.999e9,
            1e10,
            -1e10,
            1e30,
            f64::INFINITY,
            f64::NAN,
        ] {
            assert_eq!(
                taint_visible(v, 0.0),
                !ny_core::is_crown_coeff_safe_f64(v),
                "taint predicate must agree with is_crown_coeff_safe_f64 at {v:e}"
            );
        }
    }

    /// The probe's sentinel constant is the shipped one.
    #[test]
    fn sentinel_matches_core() {
        assert_eq!(FALLBACK, f64::from(ny_core::FALLBACK_BOUND));
        assert_eq!(
            ny_core::FALLBACK_BOUND,
            ny_core::CROWN_COEFF_MAX,
            "the overflow sentinel and the coefficient degrade threshold are the \
             same number by construction"
        );
    }

    /// The triplet probe's three arms have DISTINCT exact answers, so a
    /// permuted or aliased output cannot pass.
    #[test]
    fn triplet_probe_arms_are_distinguishable_and_exact() {
        let (a, bs, wants) = triplet_probe_operands();
        assert_eq!(a.len(), TRIPLET_M * TRIPLET_K);
        assert_eq!(bs[0].len(), TRIPLET_K * TRIPLET_N);
        assert_eq!(wants[0].to_bits(), 0x3FF0_0000_0000_0100, "1 + 256*2^-52");
        assert_ne!(wants[0].to_bits(), wants[1].to_bits());
        assert_ne!(wants[1].to_bits(), wants[2].to_bits());
        // (m,k,n) = 256^3 is exactly the sound-CROWN size gate, so the probe
        // shape is the smallest shape production ever dispatches.
        assert_eq!(
            (TRIPLET_M * TRIPLET_K * TRIPLET_N) as u128,
            1u128 << 24,
            "probe must sit at the MIN_RESIDENT_MACS production boundary"
        );
    }

    /// The bit comparators fail closed on any deviation.
    #[test]
    fn bit_comparators_fail_closed() {
        assert!(expect_bits_f64("t", &[1.0, 2.0], &[1.0, 2.0]).is_ok());
        assert!(expect_bits_f64("t", &[1.0], &[1.0, 2.0]).is_err());
        assert!(
            expect_bits_f64("t", &[f64::NAN], &[1.0]).is_err(),
            "a NaN can never match a finite expectation"
        );
        assert!(
            expect_bits_f64("t", &[1.0], &[f64::from_bits(1.0f64.to_bits() + 1)]).is_err(),
            "a 1-ULP deviation must refuse"
        );
        assert!(
            expect_bits_f64("t", &[0.0], &[-0.0]).is_err(),
            "bit compare must separate +0 from -0"
        );
        assert!(expect_bits_f32("t", &[1.0], &[1.0]).is_ok());
        assert!(expect_bits_f32("t", &[1.0], &[1.0, 2.0]).is_err());
    }

    /// MEASUREMENT — `#u4-cuda` on this box, with NO NVIDIA device.
    ///
    /// The CUDA sound CROWN composes in HOST f64 and offloads only the GEMM, so
    /// substituting `NaiveCpuGemmEngine` changes nothing a lane depends on: every
    /// lane product is exactly representable, so reduction order is irrelevant.
    /// This test therefore reports the REAL CUDA verdict for the sentinel rung.
    #[test]
    fn report_cuda_sentinel_taint_lanes() {
        let lanes =
            sentinel_taint_lanes(&ny_core::NaiveCpuGemmEngine).expect("lane probe must run");
        println!("[#u4-cuda sentinel taint, ny-cuda sound_crown composition]");
        for l in &lanes {
            println!(
                "  {:<66} v={:<12.6e} lin_err={:<12.6e} coeff={:<12.6e} err={:<12.6e} \
                 tainted={:<5} expect={:<11} {}",
                l.label,
                l.value,
                l.linear_err,
                l.coeff,
                l.err,
                l.tainted_at_end,
                if l.expects_sticky {
                    "StaySticky"
                } else {
                    "Annihilate"
                },
                if l.ok { "PASS" } else { "FAIL" }
            );
        }
        let (lo, hi, degraded) = published_bound_is_degraded();
        println!(
            "  consumer: concretize_f64(coeff=1e10 sentinel, box=[-1e-20,1e-20]) \
             -> ({lo:e}, {hi:e}) degraded={degraded}"
        );
        println!(
            "  => sentinel_taint_sticky() = {}",
            lanes.iter().all(|l| l.ok) && degraded
        );

        // INVARIANT that must hold on any build: the composed rung implies every
        // lane AND the consumer degrade.
        if sentinel_taint_sticky() {
            assert!(
                lanes.iter().all(|l| l.ok),
                "rung granted with a failing lane"
            );
            assert!(
                degraded,
                "rung granted while the consumer publishes confidently"
            );
        }
    }

    /// FAIL-CLOSED: forcing the rung must refuse it, and must refuse it even
    /// though the composed authority predicate is `true` by default (ladder
    /// disengaged). Asserting on the RUNG rather than on the composed predicate
    /// keeps the two questions from being conflated.
    #[test]
    fn forced_sentinel_failure_refuses_the_rung() {
        set_force_sentinel_taint_fail(true);
        assert!(
            !sentinel_taint_sticky(),
            "a forced sentinel-taint failure must refuse rung C2"
        );
        set_force_sentinel_taint_fail(false);
    }

    /// Check and report the whole ladder on a qualified NVIDIA device. A host
    /// without CUDA validates the shared unavailable-capability seam instead.
    #[test]
    fn cuda_authority_ladder_is_coherent_when_hardware_is_capable() {
        crate::with_capable_cuda(|engine| {
            let report = engine.authority_ladder_report();
            println!(
                "[CUDA sound-GPU authority ladder] device={}\n  \
                 C0 NY_CUDA_AUTHORITY_LADDER      = {}\n  \
                 C1 host dd_selfcheck             = {:?}\n  \
                 C2 sentinel_taint_sticky (#u4)   = {:?}\n  \
                 C3 verify_ieee_gemm_model        = {:?}\n  \
                 C4 verify_gemm_gradual_underflow = {:?}\n  \
                 C5 verify_triplet_composition    = {:?}\n  \
                 => provides_sound_gpu_crown()    = {}",
                engine.device_name(),
                report.gate,
                report.host_f64_reference,
                report.sentinel_taint_sticky,
                report.ieee_gemm_model,
                report.gemm_gradual_underflow,
                report.triplet_composition,
                <CudaGemmEngine as ny_core::GpuCrownBackward>::provides_sound_gpu_crown(engine),
            );
            assert_eq!(
                report.granted,
                <CudaGemmEngine as ny_core::GpuCrownBackward>::provides_sound_gpu_crown(engine),
                "the report and public authority seam must agree"
            );
            if report.gate && report.granted {
                assert!(
                    report.host_f64_reference == Some(true)
                        && report.sentinel_taint_sticky == Some(true)
                        && report.ieee_gemm_model == Some(true)
                        && report.gemm_gradual_underflow == Some(true)
                        && report.triplet_composition == Some(true),
                    "authority must imply EVERY rung"
                );
            }
        });
    }

    /// ON-DEVICE: the `#u1-cuda` settling test on its own, so a failure is
    /// attributable without reading the whole ladder.
    #[test]
    fn triplet_dispatch_matches_single_dispatch_when_hardware_is_capable() {
        crate::with_capable_cuda(|engine| {
            engine
                .run_triplet_composition_probe()
                .expect("gemm_f64_triplet must be bit-identical to gemm_f64 at CROWN shape");
        });
    }

    /// ON-DEVICE: the `#u3-cuda` gradual-underflow probe on its own.
    #[test]
    fn device_gemm_honours_gradual_underflow_when_hardware_is_capable() {
        crate::with_capable_cuda(|engine| {
            engine
                .run_gradual_underflow_probe()
                .expect("cuBLAS must not FTZ/DAZ the monotone error channels");
        });
    }
}
