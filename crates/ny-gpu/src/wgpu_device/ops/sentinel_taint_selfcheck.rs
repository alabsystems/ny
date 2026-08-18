// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#u4` — ON-DEVICE probe for OVERFLOW-SENTINEL TAINT STICKINESS across a
//! modeled fused resident CROWN-backward segment.
//!
//! # The obligation
//!
//! Every GPU value kernel saturates a finite overflow to the FINITE sentinel
//! `±FALLBACK_BOUND` (`1e10`, `ny_core::gemm::FALLBACK_BOUND`) rather than to
//! `±inf` — see `nan_safe_clamp` in [`GEMM_F32_SHADER`] and
//! [`GEMM_F32_SMALL_K_SHADER`]. In the historical magnitude-only value path,
//! that in-band sentinel was the ONLY carrier of the fact that a coefficient's
//! true magnitude is unknown and strictly larger than what the buffer holds.
//! Three in-band choke points act on it:
//!
//! * `CROWN_AW_ERROR_COMBINE_SHADER` — `s_prod >= FALLBACK_BOUND ||
//!   prop >= FALLBACK_BOUND ⇒ e = 1e30` (deliberate degrade),
//! * the optional `CROWN_EFT_MIN_COMBINE_SHADER` — `prop >= FALLBACK_BOUND ||
//!   s_prod >= FALLBACK_BOUND ⇒` refuse its error-lowering `min`, and
//! * `CROWN_CONCRETIZE_SOUND_SHADER` + its host preflight — `|a| >=
//!   FALLBACK_BOUND ⇒ degrade the row`.
//!
//! Their base implementations remain MAGNITUDE tests (the EFT one executes
//! only when its optional arm is enabled), and numbers are destroyed by
//! arithmetic. Since the 2026-08-11 UTC source review, the supported resident route
//! carries a separate taint word by default, C1/C2 consult it, and any admitted
//! route that cannot provide complete words refuses rather than silently
//! falling back to these magnitude checks. This module measures both the
//! historical defect and the selected word-channel remedy.
//!
//! # The CPU analogue being ported
//!
//! `ny-propagate/src/bounds/tests/batched_linear.rs`,
//! `test_batched_exact_transport_sentinel_degrades_and_stays_sticky`:
//!
//! ```text
//! guarded  = coefficient CROWN_COEFF_MAX (== FALLBACK_BOUND)
//! zero     = coefficient 0
//! composed = guarded.compose(&zero)
//! assert composed.lower_a == -inf   // "Exact zero composition must not
//! assert composed.upper_a == +inf   //  cancel the finite transport sentinel"
//! ```
//!
//! The CPU is STRICTLY STICKY: it promotes the finite sentinel to `±inf` at
//! composition time, so no later arithmetic can launder it. The GPU has no such
//! promotion — the sentinel stays `1e10` and rides through `A@W`, the
//! `|A|@|W|` / `err@|W|` error GEMMs, the AW combine and the resident
//! activation as an ordinary float.
//!
//! # The probe
//!
//! Six lanes, each a `(m,k,n) = (LANES, K, LANES)` problem, run through the
//! actual shipped VALUE shader sources for the modeled
//! GEMM → AW-combine → activation segment, in resident order:
//!
//! ```text
//!   V    = GEMM_F32_SHADER(A, W)                      // signed value channel
//!   |A|  = ABS_COPY_SHADER(A) ; |W| = ABS_COPY_SHADER(W)
//!   S    = GEMM_F32_SHADER(|A|, |W|)                  // s_prod
//!   P    = GEMM_F32_SHADER(E, |W|)                    // prop
//!   E'   = CROWN_AW_ERROR_COMBINE_SHADER(S, P)        // the 1e30 degrade
//!   A'',E'' = CROWN_ACTIVATION_RESIDENT_SHADER(V, E', ls, us)
//! ```
//!
//! Lane `i` reads the diagonal element `(i, i)`. Its baseline end-of-segment
//! magnitude predicate is `|A''| >= FALLBACK_BOUND || E'' >= FALLBACK_BOUND ||
//! either non-finite`; since the 2026-08-10 addendum below, the diagnostic
//! with-word predicate additionally reads the OUT-OF-BAND TAINT WORDS
//! (`|| taint_word_a || taint_word_e`). The value segment dispatches shipped
//! sources. The parallel word path dispatches the GEMM/activation twins and
//! deliberately host-models the combine word OR, as detailed below.
//!
//! This is not the complete production resident path: it omits the optional EFT
//! min-combine and the bias/intercept/conv/residual folds. The authored combine
//! twin is composed separately by `ops/taint_chain.rs`, C2 by
//! `ops/eft_min_combine_taint_probe.rs`, and C1/host folds by CPU tests.
//!
//! # What each lane pins
//!
//! * [`Expect::StaySticky`] — the sentinel entered and the composition does NOT
//!   annihilate it in exact arithmetic, so the taint MUST still be visible at
//!   the end of the chain. A lane that loses it is publishing a small, finite,
//!   confident number in place of an unknown one: UNSOUND.
//! * [`Expect::AnnihilateExactly`] — the composition multiplies by EXACT ZERO.
//!   In the reals `R·0 = 0` for every finite `R`, and the sentinel always
//!   stands for a finite real (an f32 overflow of finite operands), so losing
//!   the taint here is arithmetically justified. The lane still pins that the
//!   value is EXACTLY `0.0` (not merely small), that no taint is claimed, and
//!   (with-word verdict) that the taint WORDS are `0` — the twins' `!= 0`
//!   conjuncts guarantee it, and a stuck word here would be the `±inf`
//!   tightness collapse in disguise.
//!   NOTE: this is where NY's GPU is deliberately LESS conservative than its
//!   own CPU reference, which degrades even this case to `±inf`. The divergence
//!   is reported by [`gpu_tests::report_sentinel_taint_lanes`].
//!
//! ANY lane that misses its expectation, any dispatch fault, any readback error
//! and any uninitialized cache ⇒ `false` (fail-closed). This rung authorizes
//! nothing on its own; it can only CLOSE the ladder in
//! `ops/sound_authority.rs`.
//!
//! # MEASURED VERDICT — the 2026-08-06 baseline (magnitude channels only; Apple M5 Max / Metal)
//!
//! ```text
//! lane 0 cancel-add, slope 1     V=0        s_prod=1e10 E'=1e30  A''=0      E''=2.000002e30  PASS
//! lane 1 cancel-add, slope 0     V=0        s_prod=1e10 E'=1e30  A''=0      E''=7.5e-37      PASS (exact annihilation)
//! lane 2 sentinel * 1e-20        V=1e-10    s_prod=1e-10 E'=2.4e-17 A''=1e-10 E''=5.4e-17   FAIL
//! lane 3 sentinel * 1            V=1e10     s_prod=1e10 E'=1e30  A''=1e10   E''=2.000002e30  PASS
//! lane 4 err-taint, slope 0      V=1        prop=1e10   E'=1e30  A''=0      E''=7.5e-37      PASS (exact annihilation)
//! lane 5 err-taint * 1e-25       V=1        prop=1e10   E'=1e30  A''=1e-25  E''=2.0000019e5  FAIL
//! ```
//!
//! So: the taint IS carried correctly through cancelling additions (the signed
//! `A@W` cancels to exactly `0`, and the MONOTONE `|A|@|W|` channel saturates
//! and fires the combine's degrade — lane 0), and it IS carried through a
//! unit-scale composition (lane 3), and the `1e30` charge is itself transported
//! by being clamped to `FALLBACK_BOUND` in the `err@|W|` GEMM and re-detected by
//! the combine's `prop >= FALLBACK_BOUND` arm (lanes 4/5, `prop = 1e10`).
//!
//! But the taint is DESTROYED BY DOWNSCALING, in BOTH channels:
//!
//! * lane 2 — one weight of `1e-20` turns the value-channel sentinel into
//!   `1e-10` with a `5e-17` error budget. The stored `1e10` stands for a true
//!   coefficient anywhere up to `~3.4e38`, so the true product is up to
//!   `3.4e18`: the chain publishes a CONFIDENT number that is up to 28 orders
//!   of magnitude too small.
//! * lane 5 — one activation slope of `1e-25` turns the `1e30` DEGRADE MARKER
//!   into an ordinary `2.0e5` error charge, below every downstream guard. `1e30`
//!   is not a bound on anything (the combine writes it precisely because the
//!   true reduction is UNKNOWN and strictly larger), so multiplying it by a
//!   Lipschitz factor is not a valid transport of that unknown.
//!
//! **On the magnitude channels U4 is therefore NOT DISCHARGED, and the failure
//! is in NY's kernel design, not in the adapter.** Both guards are magnitude
//! comparisons against a finite constant, so any adapter — Metal or CUDA —
//! will launder the taint the same way. That is the right outcome for a rung:
//! it makes the refusal MEASURED, and this baseline is what the twin-chain
//! addendum below is measured against.
//!
//! # What would make this rung pass
//!
//! The taint has to stop being a magnitude. Two shapes both work and both are
//! strictly widening:
//!
//! 1. **Saturate to `±inf`, not to `±FALLBACK_BOUND`.** REFUTED for NY:
//!    `inf·0 = NaN`, so every dead ReLU (the most common event in a deep
//!    network) would degrade its whole row — a tightness collapse on the hot
//!    path traded for the laundering bug. See the [`sh::GEMM_F32_TAINT_SHADER`]
//!    doc block; never propose it.
//! 2. **A separate taint bitmask buffer**, which must be OR'd (never multiplied)
//!    through every fused op and consulted by the verdict guards alongside the
//!    magnitude test. Costs one extra `u32` buffer per resident coefficient
//!    tensor and one `|=` per kernel. **ARMED FOR ADMITTED ROUTES; UNSUPPORTED
//!    CONFIGURATIONS REFUSE; RAW CROWN SOURCE GATE NOW OPEN, PUBLIC CLI
//!    INTEGRATION STILL CLOSED** — see the addendum below.
//!
//! # 2026-08-10/11 addendum — built and armed word path
//!
//! Shape 2 exists on device for all three modeled hops:
//! [`sh::GEMM_F32_TAINT_SHADER`],
//! [`sh::CROWN_AW_ERROR_COMBINE_TAINT_SHADER`], and
//! [`sh::CROWN_ACTIVATION_RESIDENT_TAINT_SHADER`] carry a `u32` taint word
//! beside the values under the propagation rule
//!
//! ```text
//! taint_out = OR over inputs of
//!             (taint_in AND (partner_value != 0 OR partner_taint != 0))
//!          OR (this op itself saturated/degraded)
//! ```
//!
//! (10/10 per-op probes green on the GB10, `ops/taint_channel_probe.rs`;
//! `ops/taint_chain.rs` composes the three twins end to end). A fourth twin,
//! `CROWN_EFT_MIN_COMBINE_TAINT_SHADER`, and its device probe implement the C2
//! tightening refusal. The fail-closed C1 consult body and host taint-fold
//! helpers are also built. Since the 2026-08-11 UTC arming review, AUTO mode runs
//! the worded walk whenever the taint twins are available; `NY_GPU_TAINT_WORDS=0`
//! explicitly opts out and `=1` requires the twins. The resident walk dispatches
//! the applicable transport twins and, when optional Linear EFT min-tightening
//! is active, C2; it invokes its host folds, ORs complete per-spec-row words,
//! and supplies them to the armed C1 consult. ResNet segment composition carries
//! words across its seam, while unsupported worded shapes/routes typed-refuse.
//! This older six-lane selfcheck runs the GEMM and activation twins beside the
//! unchanged magnitude chain and deliberately host-models the combine
//! transport:
//!
//! * the words are host-seeded exactly where an operand IS a saturation
//!   artifact (`|x| >= FALLBACK_BOUND` — the `±1e10` sentinel and the `1e30`
//!   degrade marker both qualify), which is precisely the set an upstream
//!   taint-twin op would have self-seeded;
//! * `|A|` reuses `A`'s words — `abs` is magnitude-preserving, so it neither
//!   creates nor destroys "magnitude unknown";
//! * this selfcheck models the combine's word transport on the host as
//!   `word(E') = word(S) | word(P)`. That is EXACT for this op under the rule:
//!   both inputs enter `e = (γ_k·s_prod + prop)·slack + flush` with NONZERO
//!   compile-time coefficients, so no `!= 0` conjunct can clear, and every
//!   saturation the combine's own `>= FALLBACK_BOUND` degrade fires on was
//!   already self-seeded by the GEMM twin (whose outputs saturate to exactly
//!   `±FALLBACK_BOUND`). The separately composed diagnostic chain executes
//!   the authored combine twin on device; this older lane runner intentionally
//!   retains its independently checkable host OR;
//! * the twins recompute the VALUES too, byte-for-byte the same source
//!   arithmetic; each lane's diagonal is required to read back BIT-IDENTICAL
//!   to the base chain (`twin_value_matches`). A diverging twin (e.g. the
//!   backend contracting `a*b + c` differently across the two modules) proves
//!   nothing about the production values' words, so it fails the with-word
//!   verdict — fail-closed.
//!
//! With the words in the end predicate, lanes 2 and 5 MEASURE sticky (the
//! `!= 0` conjuncts keep the word through the `1e-20` weight and the `1e-25`
//! slope) while lanes 1 and 4 still annihilate (words `0` at a dead ReLU).
//!
//! [`PRODUCTION_GUARDS_CONSULT_TAINT_WORD`] is now source-armed, so the RUNG
//! selects the with-word verdict. U5/U6 and B0 were subsequently discharged and
//! `PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED` is now open. Arming this channel
//! alone still grants nothing: authority requires the typed explicit constructor
//! and every live rung on the same device. The public `ComputeDevice` exposes
//! only the resulting qualified CROWN accessor.
//! [`gpu_tests::report_sentinel_taint_lanes`] still prints BOTH verdicts
//! ("magnitude-only" and "with-word") so the original defect remains visible.
//!
//! # 2026-08-10 addendum 2 — the random-wide drift pin
//!
//! The lane chain compares twin values against the base chain on six diagonal
//! elements at `K = 4` only. The twins are slated to become the production
//! dispatches, so a compiler divergence between the two compiled modules
//! (e.g. a different fma-contraction choice) must be caught over a WIDE
//! operand distribution and across the GEMM 16x16 tiling boundary, not just
//! on the sentinel lanes. `gpu_tests::random_wide_twin_drift_pin` runs the
//! same dual chain (the extracted [`WgpuDevice::dual_chain_run`] core — the
//! lane probe is a thin wrapper over it, so the two callers cannot drift
//! apart) at `(nl=8, k=64)` and `(nl=4, k=257)` on deterministic LCG
//! operands — magnitudes `2^-30..2^8`, both signs, ~1% exact zeros, one
//! operand at exactly `FALLBACK_BOUND` — and requires EVERY element of
//! V, S, P, E-combined, A'', E'' BIT-IDENTICAL between the base and twin
//! chains.
//!
//! # 2026-08-13 addendum — the `#flush-charge` §H subnormal-multiplier lanes
//!
//! The six-lane table above never puts a SUBNORMAL multiplier against the
//! worded sentinel, so it cannot see the charged-Metal hazard: on a
//! pure-flush adapter the multiply DAZ-zeroes the multiplier and — if the DAZ
//! also reaches the `!= 0` annihilation compares — the word drops exactly as
//! at a dead ReLU. The §H section at the bottom of this file measures that
//! domain live (both compare slots of the GEMM twin, the activation twin's
//! `slopes_live`, and `TAINT_ROW_OR_SHADER`'s partner conjunct; both signs,
//! min/max subnormal, the ±2^-126 NORMAL boundary, unit/zero controls) and
//! feeds `verify_subnormal_mult_taint` — a CHARGED-authority conjunct only;
//! the uncharged five-rung ladder is untouched. Measured M5 Max verdict:
//! PURE-FLUSH + compare-DAZ — annihilation is real and confined to exactly
//! the strictly-subnormal multipliers the charged walk guard refuses.

use ny_core::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use super::super::shaders as sh;
use super::super::sound_consts::{combine_slack_f32, gamma_k_f32};
use super::super::WgpuDevice;
use super::ibp_forward::create_buffer;
use super::subnormal_selfcheck::FlushClass;

/// The finite overflow sentinel every value kernel saturates to. Mirrors
/// `ny_core::gemm::FALLBACK_BOUND` and the `FALLBACK_BOUND` constant compiled
/// into the shaders; pinned equal by [`cpu_tests::sentinel_matches_core`].
const FALLBACK: f32 = ny_core::FALLBACK_BOUND;

/// The deliberate degrade charge `CROWN_AW_ERROR_COMBINE_SHADER` writes when a
/// reduction saturates. It is a MAGNITUDE, not a flag — the whole question this
/// module measures is whether it survives being scaled.
const ERR_TAINT: f32 = 1e30;

/// Contraction length of each probe lane.
const K: usize = 4;

/// Hard gate: may the out-of-band TAINT WORD count toward the RUNG?
///
/// ARMED 2026-08-11 UTC (source-review event, evidence in the arming commit and
/// TAINT_GUARD_AUDIT.md §4). The preconditions the previous doc demanded are
/// now facts:
/// * the resident walk dispatches the taint twins, transports words
///   on-device, and hands per-spec-row words to the C1 consult — BY DEFAULT
///   (AUTO gate: on whenever the twins are available; `NY_GPU_TAINT_WORDS=0`
///   opts out, `=1` turns twin-unavailability into a typed refusal). Measured
///   tax after the on-device row-OR: 1.09x (`taint_gate_overhead_report`).
/// * the resnet segment composition carries words (seam OR, launder-proof
///   pinned by `taint_resnet_compose_sentinel_row_survives_composition`);
/// * the C1 preflight refuses fail-closed on ABSENT words (`None` ⇒ typed
///   refusal ⇒ ny-propagate's existing Err fallback to the CPU backward), so
///   no un-worded GPU chain can reach a verdict once authority opens;
/// * C2 (the EFT min-combine, the chain's only error-LOWERING op) consults
///   the words in-shader.
///
/// With this `true` the rung reads the WITH-WORD verdict: lanes 2/5 (the
/// measured launder shapes) pass because the word channel carries what
/// magnitude cannot, and a stuck word on a dead ReLU REFUSES (the `!= 0`
/// annihilation conjuncts). U5/U6 and the B0 review subsequently discharged,
/// and `PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED` (ops/sound_authority.rs) is
/// now open. This const remains only one conjunct: authority also requires the
/// typed request and every live probe on the same device; failures cannot be
/// promoted by ambient process input.
pub(super) const PRODUCTION_GUARDS_CONSULT_TAINT_WORD: bool = true;

/// What the fused chain must do with the taint on a given lane.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Expect {
    /// Taint must still be detectable by the selfcheck's selected end predicate.
    StaySticky,
    /// Exact-zero annihilation: value must be EXACTLY `0.0`, taint legitimately
    /// gone.
    AnnihilateExactly,
}

/// One probe lane: row `i` of `A`/`E`, column `i` of `W`, and the activation
/// slope of neuron `i`. `ls == us` on every lane so the shader's sign-dependent
/// `sel` choice cannot make the lane ambiguous.
struct Lane {
    label: &'static str,
    a: [f32; K],
    w: [f32; K],
    e: [f32; K],
    slope: f32,
    expect: Expect,
    /// Why that expectation is the SOUND one, in one line.
    rationale: &'static str,
}

/// The lane table. Lanes 0/3 establish that the mechanism CAN carry the taint
/// (so a failure elsewhere is a real hole, not a probe that never armed);
/// lanes 2/5 are the laundering adversaries; lanes 1/4 are the exact-zero
/// annihilations the CPU reference refuses and the GPU permits.
const LANES: [Lane; 6] = [
    Lane {
        label: "0 cancel-add, slope 1        (A@W cancels; |A|@|W| must not)",
        a: [1e10, -1e10, 0.0, 0.0],
        w: [1.0, 1.0, 0.0, 0.0],
        e: [0.0; K],
        slope: 1.0,
        expect: Expect::StaySticky,
        rationale: "true A rows are unknown reals |R|>=1e10; R1+R2 is NOT 0, so \
                    the cancelled value channel must be caught by the monotone \
                    |A|@|W| channel",
    },
    Lane {
        label: "1 cancel-add, slope 0 EXACT  (dead ReLU annihilation)",
        a: [1e10, -1e10, 0.0, 0.0],
        w: [1.0, 1.0, 0.0, 0.0],
        e: [0.0; K],
        slope: 0.0,
        expect: Expect::AnnihilateExactly,
        rationale: "(R1+R2)*0 == 0 exactly for every finite R, so dropping the \
                    taint is justified — but the CPU reference degrades here",
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
        label: "4 err-taint 1e30, slope 0    (dead ReLU annihilation, err channel)",
        a: [1.0, 0.0, 0.0, 0.0],
        w: [1.0, 0.0, 0.0, 0.0],
        e: [ERR_TAINT, 0.0, 0.0, 0.0],
        slope: 0.0,
        expect: Expect::AnnihilateExactly,
        rationale: "the coefficient is annihilated exactly, so its error budget \
                    legitimately goes with it",
    },
    Lane {
        label: "5 err-taint 1e30 * 1e-25     (DOWNSCALE LAUNDER, error channel)",
        a: [1.0, 0.0, 0.0, 0.0],
        w: [1.0, 0.0, 0.0, 0.0],
        e: [ERR_TAINT, 0.0, 0.0, 0.0],
        slope: 1e-25,
        expect: Expect::StaySticky,
        rationale: "1e30 is a DEGRADE MARKER (`true error unknown`), not a \
                    bound, so scaling it by the Lipschitz factor is not a valid \
                    transport of the unknown",
    },
];

const NL: usize = LANES.len();

/// `Params { m, k, n, _pad }` — [`sh::GEMM_F32_SHADER`] and its taint twin
/// [`sh::GEMM_F32_TAINT_SHADER`] (identical uniform layout).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmParams {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

/// `Params { n, _p0, _p1, _p2 }` — [`sh::ABS_COPY_SHADER`].
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AbsParams {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

/// `Params { n, slack, gamma_k, additive, k, out_cols, w_l1_max, _pad }` —
/// [`sh::CROWN_AW_ERROR_COMBINE_SHADER`].
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CombineParams {
    n: u32,
    slack: f32,
    gamma_k: f32,
    additive: f32,
    k: u32,
    out_cols: u32,
    w_l1_max: f32,
    _pad: u32,
}

/// `Params { num_specs, num_neurons, is_upper, additive, num_specs_per_dom,
/// eft_mode, _p1, _p2 }` — [`sh::CROWN_ACTIVATION_RESIDENT_SHADER`] and its
/// taint twin [`sh::CROWN_ACTIVATION_RESIDENT_TAINT_SHADER`] (identical
/// uniform layout).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ActParams {
    num_specs: u32,
    num_neurons: u32,
    is_upper: u32,
    additive: f32,
    num_specs_per_dom: u32,
    eft_mode: u32,
    _p1: u32,
    _p2: u32,
}

/// Test/operator hook: force [`WgpuDevice::verify_sentinel_taint_sticky`] to
/// report FAILURE. Read on every call — NOT cached — so a test can flip it
/// without the cached real result masking it. Only ever forces MORE closed.
static TEST_FORCE_FAIL: AtomicBool = AtomicBool::new(false);

fn env_forces_fail() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("NY_FORCE_GPU_SENTINEL_TAINT_SELFCHECK_FAIL").is_some())
}

fn selfcheck_forced_to_fail() -> bool {
    TEST_FORCE_FAIL.load(Ordering::Relaxed) || env_forces_fail()
}

/// #flush-charge: forced-fail visibility for the charged-flush authority read
/// (`charged_flush_authority_cached`) — the hook must close the charged mode
/// too. Only ever forces MORE closed.
pub(crate) fn probe_forced_to_fail() -> bool {
    selfcheck_forced_to_fail()
}

/// Test hook: force / release a sentinel-taint self-check failure.
#[cfg(all(test, feature = "gpu-tests"))]
pub(crate) fn set_force_sentinel_taint_selfcheck_fail(force: bool) {
    TEST_FORCE_FAIL.store(force, Ordering::Relaxed);
}

/// Per-lane measurement: the end-of-chain coefficient and error, the twin
/// chain's taint words, and the verdicts against the lane's pinned
/// expectation.
///
/// `label`, `ok` and the two per-channel verdicts are load-bearing in a
/// non-test build (the rung reads `ok` to decide; the failure log reads the
/// channel verdicts to attribute); the mid-chain fields exist so a failing
/// adapter or kernel-set can be CHARACTERIZED lane by lane rather than merely
/// refused.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(not(all(test, feature = "gpu-tests")), allow(dead_code))]
pub(crate) struct LaneOutcome {
    pub(crate) label: &'static str,
    pub(crate) rationale: &'static str,
    /// End-of-chain coefficient `A''[i][i]`.
    pub(crate) coeff: f32,
    /// End-of-chain certified error `E''[i][i]`.
    pub(crate) err: f32,
    /// Mid-chain `s_prod = fl(|A|@|W|)[i][i]` (the monotone channel).
    pub(crate) s_prod: f32,
    /// Mid-chain `prop = fl(E@|W|)[i][i]`.
    pub(crate) prop: f32,
    /// Mid-chain combine output `E'[i][i]` (`1e30` ⇒ the degrade fired).
    pub(crate) combined_err: f32,
    /// Signed value channel `V[i][i]` before the activation.
    pub(crate) value: f32,
    /// Mid-chain taint word of `V` (the twin GEMM's `taint_out`), entering the
    /// activation twin.
    pub(crate) taint_word_v: bool,
    /// Mid-chain taint word of `E'`. This older selfcheck host-models it as
    /// `word(S) | word(P)`; the authored device combine twin is exercised by
    /// `ops/taint_chain.rs`. See the module addendum for why this OR is exact.
    pub(crate) taint_word_ecomb: bool,
    /// End-of-chain VALUE-channel taint word (`taint_a_out[i][i]` of the
    /// activation twin).
    pub(crate) taint_word_a: bool,
    /// End-of-chain ERROR-channel taint word (`taint_e_out[i][i]`).
    pub(crate) taint_word_e: bool,
    /// Did the twin chain's VALUES read back bit-identical to the base chain
    /// on this lane's diagonal (V, S, P, A'', E'')? The twins copy the base
    /// arithmetic byte-for-byte, so a mismatch means the two compiled modules
    /// diverged (e.g. different fma contraction) and the words prove nothing
    /// about the production values — the with-word verdict then refuses.
    pub(crate) twin_value_matches: bool,
    /// Would the historical end-of-segment MAGNITUDE proxy still see the taint?
    pub(crate) tainted_magnitude: bool,
    /// The unified end-of-chain predicate:
    /// `tainted_magnitude || taint_word_a || taint_word_e`.
    pub(crate) tainted_at_end: bool,
    /// Verdict using ONLY the magnitude channels (the 2026-08-06 baseline).
    pub(crate) ok_magnitude_only: bool,
    /// Verdict with the taint words as an ADDITIONAL detection channel, and
    /// with the twin-identity + clean-annihilation-word obligations.
    pub(crate) ok_with_word: bool,
    /// THE rung verdict for this lane: `ok_with_word` under the currently armed
    /// [`PRODUCTION_GUARDS_CONSULT_TAINT_WORD`], or `ok_magnitude_only` if a
    /// future source review closes that gate. The classifier itself cannot
    /// establish production integration.
    pub(crate) ok: bool,
    /// `true` for lanes whose expectation is [`Expect::StaySticky`].
    pub(crate) expects_sticky: bool,
}

/// The selfcheck's end-of-modeled-segment MAGNITUDE proxy. It deliberately
/// preserves the 2026-08-06 lane classification: a coefficient/error sentinel
/// or any non-finite endpoint is visible. It is not a literal model of every
/// production consumer: C2 is mid-chain, and the host affine-radius preflight
/// scales coefficient error by the input-box magnitude.
fn taint_visible(coeff: f32, err: f32) -> bool {
    !coeff.is_finite() || !err.is_finite() || coeff.abs() >= FALLBACK || err >= FALLBACK
}

/// The pure per-lane verdicts, split out of the GPU plumbing so the gating
/// semantics are CPU-testable (see `cpu_tests`).
struct LaneVerdicts {
    tainted_magnitude: bool,
    tainted_at_end: bool,
    ok_magnitude_only: bool,
    ok_with_word: bool,
    ok: bool,
}

/// Classify one lane from its end-of-chain measurements.
///
/// * `ok_magnitude_only` is the 2026-08-06 baseline verdict, unchanged.
/// * `ok_with_word` adds the taint words as a detection channel for sticky
///   lanes, REQUIRES them clear on annihilation lanes (the `!= 0` conjuncts
///   guarantee it; a stuck word is a twin regression), and requires the twin
///   values bit-identical to the base chain (a diverged twin measures nothing).
/// * `ok` — the rung's lane verdict — follows the source-reviewed channel gate,
///   [`PRODUCTION_GUARDS_CONSULT_TAINT_WORD`]. Its current `true` value records
///   the completed route-coverage and C1 arming review; this runtime classifier
///   cannot open or close it.
fn classify_lane(
    expect: Expect,
    coeff: f32,
    err: f32,
    word_a: bool,
    word_e: bool,
    twin_value_matches: bool,
) -> LaneVerdicts {
    let tainted_magnitude = taint_visible(coeff, err);
    let tainted_at_end = tainted_magnitude || word_a || word_e;
    let ok_magnitude_only = match expect {
        Expect::StaySticky => tainted_magnitude,
        // Exactly zero, and no spurious taint claimed.
        Expect::AnnihilateExactly => coeff == 0.0 && !tainted_magnitude,
    };
    let ok_with_word = twin_value_matches
        && match expect {
            Expect::StaySticky => tainted_at_end,
            Expect::AnnihilateExactly => coeff == 0.0 && !tainted_at_end,
        };
    let ok = if PRODUCTION_GUARDS_CONSULT_TAINT_WORD {
        ok_with_word
    } else {
        ok_magnitude_only
    };
    LaneVerdicts {
        tainted_magnitude,
        tainted_at_end,
        ok_magnitude_only,
        ok_with_word,
        ok,
    }
}

/// Full-buffer readback of one dual-chain run ([`WgpuDevice::dual_chain_run`])
/// at shape `(nl, k)`: the SHIPPED base chain beside its taint-twin chain,
/// every vector `nl * nl` long in row-major `[spec row x neuron]` order, every
/// element the RAW `u32` BIT PATTERN of what the shader stored (the file's
/// rule: bits are compared, never float-loaded — NaN payloads and signed
/// zeros must count).
///
/// * Base values: `v = fl(A@W)`, `s = fl(|A|@|W|)`, `p = fl(E@|W|)`,
///   `ecomb = combine(s, p)`, and the resident activation's `a_out`/`e_out`.
/// * Twin values: the taint twins' byte-for-byte recomputation of the same
///   arithmetic. This older runner obtains `twin_ecomb` by re-dispatching the
///   shipped value combine on `twin_s`/`twin_p`; the separate diagnostic
///   `ops/taint_chain.rs` runner exercises the authored combine twin. The drift
///   pin compares `twin_ecomb` against `ecomb`, while the lane probe ignores it
///   (the activation twin consumes the BASE `ecomb`, keeping lane behavior
///   identical to the 2026-08-10 baseline).
/// * Words: the out-of-band taint channel. `word_ecomb` is the host-modeled
///   `word_s | word_p` (see the module addendum for why the OR is exact).
#[cfg_attr(not(all(test, feature = "gpu-tests")), allow(dead_code))]
struct DualChainOut {
    v: Vec<u32>,
    s: Vec<u32>,
    p: Vec<u32>,
    ecomb: Vec<u32>,
    a_out: Vec<u32>,
    e_out: Vec<u32>,
    twin_v: Vec<u32>,
    twin_s: Vec<u32>,
    twin_p: Vec<u32>,
    twin_ecomb: Vec<u32>,
    twin_a_out: Vec<u32>,
    twin_e_out: Vec<u32>,
    word_v: Vec<u32>,
    word_s: Vec<u32>,
    word_p: Vec<u32>,
    word_ecomb: Vec<u32>,
    word_a_out: Vec<u32>,
    word_e_out: Vec<u32>,
}

impl WgpuDevice {
    /// `#u4` rung: does the finite overflow sentinel SURVIVE the fused resident
    /// ops on this adapter?
    ///
    /// `true` ⇒ every probe lane behaved as soundness requires, measured on
    /// the channels the production verdict path consults (see
    /// [`PRODUCTION_GUARDS_CONSULT_TAINT_WORD`]). `false` ⇒ REFUSED
    /// (fail-closed): at least one lane laundered the taint into a small
    /// confident number, or the probe could not run. Cached per device.
    pub(crate) fn verify_sentinel_taint_sticky(&self) -> bool {
        if selfcheck_forced_to_fail() {
            return false;
        }
        *self
            .sentinel_taint_selfcheck
            .get_or_init(|| self.run_sentinel_taint_selfcheck())
    }

    /// Run (and log) the one-time probe. CONSERVATIVE: any lane miss OR GPU
    /// error → `false`.
    fn run_sentinel_taint_selfcheck(&self) -> bool {
        match self.run_gpu_checked("verify_sentinel_taint_sticky", || {
            self.sentinel_taint_lanes_inner()
        }) {
            Ok(lanes) => {
                let failed: Vec<&str> = lanes.iter().filter(|l| !l.ok).map(|l| l.label).collect();
                if failed.is_empty() {
                    true
                } else {
                    // Keep the channel aggregate in the log so a failure can
                    // be attributed to the selected word path rather than the
                    // intentionally failing historical magnitude baseline.
                    let with_word_pass = lanes.iter().all(|l| l.ok_with_word);
                    tracing::warn!(
                        target: "ny_gpu::wgpu",
                        adapter = %self.adapter_info.name,
                        backend = ?self.adapter_info.backend,
                        lanes = ?failed,
                        with_word_pass,
                        word_gate_open = PRODUCTION_GUARDS_CONSULT_TAINT_WORD,
                        "#u4 SENTINEL-TAINT STICKINESS self-check FAILED: the finite \
                         ±FALLBACK_BOUND overflow sentinel did not satisfy the selected \
                         WITH-WORD obligation across the modeled GEMM/AW-combine/activation \
                         segment. At least one taint word was lost on a non-annihilating \
                         path, remained stuck after exact-zero annihilation, or accompanied \
                         twin values that diverged from the base values. The historical \
                         magnitude-only chain remains vulnerable to downscaling by design; \
                         it is diagnostic and cannot substitute for the armed word channel. \
                         REFUSING GPU verdict authority (fail-closed)"
                    );
                    false
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "ny_gpu::wgpu",
                    adapter = %self.adapter_info.name,
                    backend = ?self.adapter_info.backend,
                    error = %e,
                    "#u4 SENTINEL-TAINT STICKINESS self-check could not run: REFUSING \
                     (fail-closed)"
                );
                false
            }
        }
    }

    /// Diagnostic: run the probe once and return every lane's measurement, so a
    /// failing adapter/kernel-set is CHARACTERIZED rather than merely refused.
    #[cfg(all(test, feature = "gpu-tests"))]
    pub(crate) fn sentinel_taint_report(&self) -> Result<Vec<LaneOutcome>> {
        self.run_gpu_checked("sentinel_taint_report", || {
            self.sentinel_taint_lanes_inner()
        })
    }

    /// Build the lane operands, run the dual chain at the probe's `(NL, K)`
    /// shape via [`Self::dual_chain_run`], read the lane DIAGONAL, and
    /// classify every lane.
    fn sentinel_taint_lanes_inner(&self) -> Result<Vec<LaneOutcome>> {
        // ---- host-side operands -------------------------------------------
        // A is [NL x K] (lane i is row i); W is [K x NL] (lane i is column i);
        // E is [NL x K] (the incoming coefficient-error, same layout as A).
        let mut a = vec![0.0f32; NL * K];
        let mut w = vec![0.0f32; K * NL];
        let mut e = vec![0.0f32; NL * K];
        let mut ls = vec![0.0f32; NL];
        for (i, lane) in LANES.iter().enumerate() {
            for kk in 0..K {
                a[i * K + kk] = lane.a[kk];
                e[i * K + kk] = lane.e[kk];
                w[kk * NL + i] = lane.w[kk];
            }
            ls[i] = lane.slope;
        }

        // #u4 twin-chain seeds (2026-08-10): production will carry the words
        // from op to op; the probe host-seeds them exactly where an operand IS
        // a saturation artifact — `|x| >= FALLBACK_BOUND` — which is precisely
        // the set an upstream taint-twin op would have self-seeded (its
        // outputs saturate to exactly ±FALLBACK_BOUND, and the 1e30 degrade
        // marker also clears the threshold). Weights are exact host data:
        // never tainted.
        let taint_a_words: Vec<u32> = a.iter().map(|&x| u32::from(x.abs() >= FALLBACK)).collect();
        let taint_e_words: Vec<u32> = e.iter().map(|&x| u32::from(x.abs() >= FALLBACK)).collect();

        // `ls == us` on every lane (the shader's sign-dependent `sel` choice
        // must not make a lane ambiguous — see [`Lane`]); β = 0 everywhere.
        let beta = vec![0.0f32; NL];
        let out = self.dual_chain_run(
            NL,
            K,
            &a,
            &w,
            &e,
            &ls,
            &ls,
            &beta,
            &taint_a_words,
            &taint_e_words,
        )?;

        // ---- classify -------------------------------------------------------
        // The chain arrives as RAW BITS: the diagonal is converted to f32 for
        // the magnitude verdicts, while the twin identity stays a comparison
        // on the bits themselves (never float-compare: NaN and signed zero
        // must count). A twin mismatch means the twins are not measuring the
        // shipped arithmetic, so their words prove nothing (fails
        // ok_with_word).
        let mut outcomes = Vec::with_capacity(NL);
        for (i, lane) in LANES.iter().enumerate() {
            let d = i * NL + i;
            let value = f32::from_bits(out.v[d]);
            let s_prod = f32::from_bits(out.s[d]);
            let prop = f32::from_bits(out.p[d]);
            let combined_err = f32::from_bits(out.ecomb[d]);
            let coeff = f32::from_bits(out.a_out[d]);
            let err = f32::from_bits(out.e_out[d]);
            let taint_word_v = out.word_v[d] != 0;
            let taint_word_ecomb = out.word_ecomb[d] != 0;
            let taint_word_a = out.word_a_out[d] != 0;
            let taint_word_e = out.word_e_out[d] != 0;
            let twin_value_matches = out.twin_v[d] == out.v[d]
                && out.twin_s[d] == out.s[d]
                && out.twin_p[d] == out.p[d]
                && out.twin_a_out[d] == out.a_out[d]
                && out.twin_e_out[d] == out.e_out[d];
            let verdicts = classify_lane(
                lane.expect,
                coeff,
                err,
                taint_word_a,
                taint_word_e,
                twin_value_matches,
            );
            outcomes.push(LaneOutcome {
                label: lane.label,
                rationale: lane.rationale,
                coeff,
                err,
                s_prod,
                prop,
                combined_err,
                value,
                taint_word_v,
                taint_word_ecomb,
                taint_word_a,
                taint_word_e,
                twin_value_matches,
                tainted_magnitude: verdicts.tainted_magnitude,
                tainted_at_end: verdicts.tainted_at_end,
                ok_magnitude_only: verdicts.ok_magnitude_only,
                ok_with_word: verdicts.ok_with_word,
                ok: verdicts.ok,
                expects_sticky: lane.expect == Expect::StaySticky,
            });
        }
        Ok(outcomes)
    }

    /// Operand-independent core of the `#u4` probe: allocate the buffers, run
    /// the SHIPPED base chain and its taint-twin chain at shape `(nl, k)` —
    /// `A`/`E` are `[nl x k]`, `W` is `[k x nl]`, `ls`/`us`/`beta` are `[nl]`,
    /// every output is `[nl x nl]` — and read EVERY output buffer back as raw
    /// `u32` bit patterns ([`DualChainOut`]).
    ///
    /// Two callers, one dispatch/readback path (so neither can drift from the
    /// other): [`Self::sentinel_taint_lanes_inner`] runs the six lanes at
    /// `(NL, K)` and reads the diagonal; the random-wide drift pin
    /// (`gpu_tests::random_wide_twin_drift_pin`) runs CROWN-ish shapes with
    /// `k` past the GEMM 16x16 tiling boundary and compares every element.
    /// The word seeds are caller-supplied so both callers pin the same
    /// host-seeding rule.
    #[allow(clippy::too_many_arguments)]
    fn dual_chain_run(
        &self,
        nl: usize,
        k: usize,
        a: &[f32],
        w: &[f32],
        e: &[f32],
        ls: &[f32],
        us: &[f32],
        beta: &[f32],
        taint_a_words: &[u32],
        taint_e_words: &[u32],
    ) -> Result<DualChainOut> {
        assert_eq!(a.len(), nl * k, "A must be [nl x k] row-major");
        assert_eq!(w.len(), k * nl, "W must be [k x nl] row-major");
        assert_eq!(e.len(), nl * k, "E must be [nl x k] row-major");
        assert_eq!(ls.len(), nl, "ls must be [nl]");
        assert_eq!(us.len(), nl, "us must be [nl]");
        assert_eq!(beta.len(), nl, "beta must be [nl]");
        assert_eq!(taint_a_words.len(), nl * k, "word(A) must mirror A");
        assert_eq!(taint_e_words.len(), nl * k, "word(E) must mirror E");

        // Host-computed uniform inputs, exactly as the resident driver derives
        // them: per-spec-row ‖a_i‖₁ and a scalar over-bound on max_j‖w_j‖₁.
        let row_abs_a: Vec<f32> = (0..nl)
            .map(|i| (0..k).map(|kk| a[i * k + kk].abs()).sum::<f32>())
            .collect();
        let w_l1_max = (0..nl)
            .map(|j| (0..k).map(|kk| w[kk * nl + j].abs()).sum::<f32>())
            .fold(0.0f32, f32::max);
        let gamma = gamma_k_f32(k)?;
        let slack = combine_slack_f32(k)?;
        let additive = ny_core::ftz_safe_underflow_floor(k as u32);

        // ---- device buffers ------------------------------------------------
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let storage_src = storage | wgpu::BufferUsages::COPY_SRC;
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let dev = &self.device;
        let f32_buf = |label: &'static str, data: &[f32], usage: wgpu::BufferUsages| {
            let b = create_buffer(dev, label, (data.len() * 4) as u64, usage);
            self.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
            b
        };
        let u32_buf = |label: &'static str, data: &[u32], usage: wgpu::BufferUsages| {
            let b = create_buffer(dev, label, (data.len() * 4) as u64, usage);
            self.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
            b
        };

        let a_buf = f32_buf("u4_a", a, storage);
        let w_buf = f32_buf("u4_w", w, storage);
        let e_buf = f32_buf("u4_e", e, storage);
        let ls_buf = f32_buf("u4_ls", ls, storage);
        let us_buf = f32_buf("u4_us", us, storage);
        let beta_buf = f32_buf("u4_beta", beta, storage);
        let row_abs_buf = f32_buf("u4_row_abs_a", &row_abs_a, storage);

        // `[nl x k]` and `[k x nl]` hold the same element count, so one zero
        // template covers both abs targets.
        let zeros_ak = vec![0.0f32; nl * k];
        let zeros_out = vec![0.0f32; nl * nl];
        let abs_a_buf = f32_buf("u4_abs_a", &zeros_ak, storage);
        let abs_w_buf = f32_buf("u4_abs_w", &zeros_ak, storage);
        let v_buf = f32_buf("u4_v", &zeros_out, storage_src);
        let s_buf = f32_buf("u4_s_prod", &zeros_out, storage_src);
        let p_buf = f32_buf("u4_prop", &zeros_out, storage_src);
        let ecomb_buf = f32_buf("u4_e_combined", &zeros_out, storage_src);
        let aout_buf = f32_buf("u4_a_out", &zeros_out, storage_src);
        let eout_buf = f32_buf("u4_e_out", &zeros_out, storage_src);

        // Twin-chain buffers. The twin VALUE outputs must read back
        // bit-identical to the base chain (checked by both callers); the word
        // buffers are the measurement.
        let zeros_out_u32 = vec![0u32; nl * nl];
        let zeros_ak_u32 = vec![0u32; k * nl];
        let ta_words_buf = u32_buf("u4_tw_a_in", taint_a_words, storage);
        let te_words_buf = u32_buf("u4_tw_e_in", taint_e_words, storage);
        let tw_w_zero_buf = u32_buf("u4_tw_w_zero", &zeros_ak_u32, storage);
        let twv_buf = u32_buf("u4_tw_v", &zeros_out_u32, storage_src);
        let tws_buf = u32_buf("u4_tw_s", &zeros_out_u32, storage_src);
        let twp_buf = u32_buf("u4_tw_p", &zeros_out_u32, storage_src);
        let twin_v_buf = f32_buf("u4_twin_v", &zeros_out, storage_src);
        let twin_s_buf = f32_buf("u4_twin_s", &zeros_out, storage_src);
        let twin_p_buf = f32_buf("u4_twin_p", &zeros_out, storage_src);

        let gemm_p = create_buffer(dev, "u4_gemm_p", 16, uniform);
        self.queue.write_buffer(
            &gemm_p,
            0,
            bytemuck::cast_slice(&[GemmParams {
                m: nl as u32,
                k: k as u32,
                n: nl as u32,
                _pad: 0,
            }]),
        );
        let abs_ak_p = create_buffer(dev, "u4_abs_ak_p", 16, uniform);
        self.queue.write_buffer(
            &abs_ak_p,
            0,
            bytemuck::cast_slice(&[AbsParams {
                n: (nl * k) as u32,
                _p0: 0,
                _p1: 0,
                _p2: 0,
            }]),
        );
        let combine_p = create_buffer(dev, "u4_combine_p", 32, uniform);
        self.queue.write_buffer(
            &combine_p,
            0,
            bytemuck::cast_slice(&[CombineParams {
                n: (nl * nl) as u32,
                slack,
                gamma_k: gamma,
                additive,
                k: k as u32,
                out_cols: nl as u32,
                w_l1_max,
                _pad: 0,
            }]),
        );
        let act_p = create_buffer(dev, "u4_act_p", 32, uniform);
        self.queue.write_buffer(
            &act_p,
            0,
            bytemuck::cast_slice(&[ActParams {
                num_specs: nl as u32,
                num_neurons: nl as u32,
                is_upper: 0,
                additive,
                num_specs_per_dom: nl as u32,
                eft_mode: 0,
                _p1: 0,
                _p2: 0,
            }]),
        );

        // ---- pipelines: the SHIPPED sources, with the SHIPPED rw layouts ----
        let gemm =
            self.create_simple_pipeline(sh::GEMM_F32_SHADER, "u4_gemm", &[false, false, true]);
        let abs = self.create_simple_pipeline(sh::ABS_COPY_SHADER, "u4_abs", &[false, true]);
        let combine = self.create_simple_pipeline(
            sh::CROWN_AW_ERROR_COMBINE_SHADER,
            "u4_combine",
            &[false, false, true, false],
        );
        let act = self.create_simple_pipeline(
            sh::CROWN_ACTIVATION_RESIDENT_SHADER,
            "u4_act",
            &[false, false, false, false, true, true, false],
        );
        // Taint twins: a, b, out, taint_a, taint_b, taint_out (6 storage).
        let gemm_taint = self.create_simple_pipeline(
            sh::GEMM_F32_TAINT_SHADER,
            "u4_gemm_taint",
            &[false, false, true, false, false, true],
        );
        // a_in, err_in, ls, us, a_out, err_out, beta, ta_in, te_in, ta_out,
        // te_out (11 storage — under the 12-binding ceiling).
        let act_taint = self.create_simple_pipeline(
            sh::CROWN_ACTIVATION_RESIDENT_TAINT_SHADER,
            "u4_act_taint",
            &[
                false, false, false, false, true, true, false, false, false, true, true,
            ],
        );

        // GEMM_F32_SHADER is workgroup_size(16,16) with gid.x = col, gid.y =
        // row; its output here is [nl x nl] (m == n == nl), so both grid axes
        // cover nl. `k` never appears in the dispatch — it only drives the
        // in-shader tile loop (`num_tiles = ceil(k/16)`), which is exactly the
        // seam the random-wide drift pin crosses with k = 257.
        let gemm_groups = nl.div_ceil(16) as u32;
        let ak_groups = (nl * k).div_ceil(256) as u32;
        let out_groups = (nl * nl).div_ceil(256) as u32;

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("u4_sentinel_taint_encoder"),
        });
        // V = A @ W
        self.pass_simple_2d(
            &mut encoder,
            &gemm,
            &gemm_p,
            &[&a_buf, &w_buf, &v_buf],
            gemm_groups,
            gemm_groups,
        );
        // |A|, |W|
        self.pass_simple(
            &mut encoder,
            &abs,
            &abs_ak_p,
            &[&a_buf, &abs_a_buf],
            ak_groups,
        );
        self.pass_simple(
            &mut encoder,
            &abs,
            &abs_ak_p,
            &[&w_buf, &abs_w_buf],
            ak_groups,
        );
        // S = |A| @ |W| ; P = E @ |W|
        self.pass_simple_2d(
            &mut encoder,
            &gemm,
            &gemm_p,
            &[&abs_a_buf, &abs_w_buf, &s_buf],
            gemm_groups,
            gemm_groups,
        );
        self.pass_simple_2d(
            &mut encoder,
            &gemm,
            &gemm_p,
            &[&e_buf, &abs_w_buf, &p_buf],
            gemm_groups,
            gemm_groups,
        );
        // E' = combine(S, P)
        self.pass_simple(
            &mut encoder,
            &combine,
            &combine_p,
            &[&s_buf, &p_buf, &ecomb_buf, &row_abs_buf],
            out_groups,
        );
        // (A'', E'') = activation(V, E', ls, us)
        self.pass_simple(
            &mut encoder,
            &act,
            &act_p,
            &[
                &v_buf, &ecomb_buf, &ls_buf, &us_buf, &aout_buf, &eout_buf, &beta_buf,
            ],
            out_groups,
        );

        // ---- twin chain, first hop: the three GEMM taint twins -------------
        // Same operand buffers, same params, same dispatch geometry; the value
        // outputs go to twin scratch (bit-compared against the base chain by
        // the callers — diagonal-only for the lanes, every element for the
        // drift pin) and the words go to the taint buffers.
        // V twin: signed value channel.
        self.pass_simple_2d(
            &mut encoder,
            &gemm_taint,
            &gemm_p,
            &[
                &a_buf,
                &w_buf,
                &twin_v_buf,
                &ta_words_buf,
                &tw_w_zero_buf,
                &twv_buf,
            ],
            gemm_groups,
            gemm_groups,
        );
        // S twin: |A|@|W|. `abs` is magnitude-preserving, so word(|A|) ==
        // word(A) — the words ride beside the abs-copied values unchanged.
        self.pass_simple_2d(
            &mut encoder,
            &gemm_taint,
            &gemm_p,
            &[
                &abs_a_buf,
                &abs_w_buf,
                &twin_s_buf,
                &ta_words_buf,
                &tw_w_zero_buf,
                &tws_buf,
            ],
            gemm_groups,
            gemm_groups,
        );
        // P twin: E@|W|.
        self.pass_simple_2d(
            &mut encoder,
            &gemm_taint,
            &gemm_p,
            &[
                &e_buf,
                &abs_w_buf,
                &twin_p_buf,
                &te_words_buf,
                &tw_w_zero_buf,
                &twp_buf,
            ],
            gemm_groups,
            gemm_groups,
        );

        // ---- readback (submission 1) ---------------------------------------
        // Every buffer is read back as RAW u32 BIT PATTERNS (`read_u32_buffer`
        // is a pure reinterpret of the mapped bytes): no float load sits
        // between the shader's stores and the bit-identity comparisons, so a
        // NaN payload can never be canonicalized on the way.
        let n_out = nl * nl;
        let bytes = (n_out * 4) as u64;
        let stage = |label: &'static str| {
            create_buffer(
                dev,
                label,
                bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )
        };
        let st_v = stage("u4_st_v");
        let st_s = stage("u4_st_s");
        let st_p = stage("u4_st_p");
        let st_ec = stage("u4_st_ec");
        let st_ao = stage("u4_st_ao");
        let st_eo = stage("u4_st_eo");
        let st_tv = stage("u4_st_twin_v");
        let st_ts = stage("u4_st_twin_s");
        let st_tp = stage("u4_st_twin_p");
        let st_wv = stage("u4_st_word_v");
        let st_ws = stage("u4_st_word_s");
        let st_wp = stage("u4_st_word_p");
        for (src, dst) in [
            (&v_buf, &st_v),
            (&s_buf, &st_s),
            (&p_buf, &st_p),
            (&ecomb_buf, &st_ec),
            (&aout_buf, &st_ao),
            (&eout_buf, &st_eo),
            (&twin_v_buf, &st_tv),
            (&twin_s_buf, &st_ts),
            (&twin_p_buf, &st_tp),
            (&twv_buf, &st_wv),
            (&tws_buf, &st_ws),
            (&twp_buf, &st_wp),
        ] {
            encoder.copy_buffer_to_buffer(src, 0, dst, 0, bytes);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        let v = WgpuDevice::read_u32_buffer(dev, &st_v, n_out)?;
        let s = WgpuDevice::read_u32_buffer(dev, &st_s, n_out)?;
        let p = WgpuDevice::read_u32_buffer(dev, &st_p, n_out)?;
        let ecomb = WgpuDevice::read_u32_buffer(dev, &st_ec, n_out)?;
        let a_out = WgpuDevice::read_u32_buffer(dev, &st_ao, n_out)?;
        let e_out = WgpuDevice::read_u32_buffer(dev, &st_eo, n_out)?;
        let twin_v = WgpuDevice::read_u32_buffer(dev, &st_tv, n_out)?;
        let twin_s = WgpuDevice::read_u32_buffer(dev, &st_ts, n_out)?;
        let twin_p = WgpuDevice::read_u32_buffer(dev, &st_tp, n_out)?;
        let word_v = WgpuDevice::read_u32_buffer(dev, &st_wv, n_out)?;
        let word_s = WgpuDevice::read_u32_buffer(dev, &st_ws, n_out)?;
        let word_p = WgpuDevice::read_u32_buffer(dev, &st_wp, n_out)?;

        // ---- twin chain, second hop: host-modeled word + activation twin ---
        // This older selfcheck models combine word transport on the host as
        // `word(E') = word(S) | word(P)`. EXACT for this op under the
        // propagation rule: both inputs enter
        // `e = (γ_k·s_prod + prop)·slack + flush` with NONZERO compile-time
        // coefficients, so neither `!= 0` conjunct can clear, and every
        // saturation the combine's own `>= FALLBACK_BOUND` degrade fires on
        // was already self-seeded as a word by the GEMM twin (its outputs
        // saturate to exactly ±FALLBACK_BOUND). The OR can neither invent nor
        // launder taint. The authored device combine twin is exercised by the
        // separate `ops/taint_chain.rs` diagnostic; this older runner retains
        // the host OR as an independent cross-check.
        let word_ecomb: Vec<u32> = word_s
            .iter()
            .zip(word_p.iter())
            .map(|(&sw, &pw)| sw | pw)
            .collect();
        let taint_ec_buf = u32_buf("u4_tw_ecomb", &word_ecomb, storage);
        let twin_ao_buf = f32_buf("u4_twin_a_out", &zeros_out, storage_src);
        let twin_eo_buf = f32_buf("u4_twin_e_out", &zeros_out, storage_src);
        let twin_ec_buf = f32_buf("u4_twin_e_combined", &zeros_out, storage_src);
        let wao_buf = u32_buf("u4_tw_a_out", &zeros_out_u32, storage_src);
        let weo_buf = u32_buf("u4_tw_e_out", &zeros_out_u32, storage_src);

        let mut encoder2 = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("u4_sentinel_taint_twin_act_encoder"),
        });
        // Twin E' = combine(twin S, twin P): the SAME shipped combine pipeline
        // re-dispatched on the twin GEMM outputs, so the drift pin can compare
        // a fully twin-COMPOSED E-combined against the base one. The lane
        // wrapper ignores this buffer, and the activation twin below still
        // consumes the BASE ecomb_buf — lane behavior is unchanged.
        self.pass_simple(
            &mut encoder2,
            &combine,
            &combine_p,
            &[&twin_s_buf, &twin_p_buf, &twin_ec_buf, &row_abs_buf],
            out_groups,
        );
        // Twin (A'', E'') + words: SAME value inputs as the base activation
        // (v_buf, ecomb_buf), so the twin's byte-for-byte arithmetic sees the
        // identical operands and its outputs must bit-match aout/eout.
        self.pass_simple(
            &mut encoder2,
            &act_taint,
            &act_p,
            &[
                &v_buf,
                &ecomb_buf,
                &ls_buf,
                &us_buf,
                &twin_ao_buf,
                &twin_eo_buf,
                &beta_buf,
                &twv_buf,
                &taint_ec_buf,
                &wao_buf,
                &weo_buf,
            ],
            out_groups,
        );
        let st_tec = stage("u4_st_twin_ec");
        let st_tao = stage("u4_st_twin_ao");
        let st_teo = stage("u4_st_twin_eo");
        let st_wao = stage("u4_st_word_ao");
        let st_weo = stage("u4_st_word_eo");
        for (src, dst) in [
            (&twin_ec_buf, &st_tec),
            (&twin_ao_buf, &st_tao),
            (&twin_eo_buf, &st_teo),
            (&wao_buf, &st_wao),
            (&weo_buf, &st_weo),
        ] {
            encoder2.copy_buffer_to_buffer(src, 0, dst, 0, bytes);
        }
        self.queue.submit(std::iter::once(encoder2.finish()));

        let twin_ecomb = WgpuDevice::read_u32_buffer(dev, &st_tec, n_out)?;
        let twin_a_out = WgpuDevice::read_u32_buffer(dev, &st_tao, n_out)?;
        let twin_e_out = WgpuDevice::read_u32_buffer(dev, &st_teo, n_out)?;
        let word_a_out = WgpuDevice::read_u32_buffer(dev, &st_wao, n_out)?;
        let word_e_out = WgpuDevice::read_u32_buffer(dev, &st_weo, n_out)?;

        Ok(DualChainOut {
            v,
            s,
            p,
            ecomb,
            a_out,
            e_out,
            twin_v,
            twin_s,
            twin_p,
            twin_ecomb,
            twin_a_out,
            twin_e_out,
            word_v,
            word_s,
            word_p,
            word_ecomb,
            word_a_out,
            word_e_out,
        })
    }

    /// Test entry for the parameterized dual chain (the random-wide drift
    /// pin), run under the device error scope exactly like
    /// [`Self::sentinel_taint_report`].
    #[cfg(all(test, feature = "gpu-tests"))]
    #[allow(clippy::too_many_arguments)]
    fn dual_chain_probe(
        &self,
        nl: usize,
        k: usize,
        a: &[f32],
        w: &[f32],
        e: &[f32],
        ls: &[f32],
        us: &[f32],
        beta: &[f32],
        taint_a_words: &[u32],
        taint_e_words: &[u32],
    ) -> Result<DualChainOut> {
        self.run_gpu_checked("dual_chain_probe", || {
            self.dual_chain_run(nl, k, a, w, e, ls, us, beta, taint_a_words, taint_e_words)
        })
    }
}

// ---------------------------------------------------------------------------
// #flush-charge §H — the RUNG-5 SUBNORMAL-MULTIPLIER probe lane
// ---------------------------------------------------------------------------
//
// The hazard this section measures: every annihilation conjunct in the taint
// twins is an f32 compare against zero —
//
//   * `bv != 0.0` / `av != 0.0`  (GEMM_F32_TAINT_SHADER, both operand slots),
//   * `lsv != 0.0 || usv != 0.0` (CROWN_ACTIVATION_RESIDENT_TAINT_SHADER),
//   * `partner[..] == 0.0`       (shaders_taint::TAINT_ROW_OR_SHADER).
//
// On a pure-flush adapter (Apple M5 Max class) a SUBNORMAL multiplier is
// DAZ-zeroed by the multiply, so the product is a clean ±0 — and if the DAZ
// also applies to the COMPARE, the `!= 0` conjunct reads false and the taint
// word is dropped exactly as if the partner had been a legitimate dead-ReLU
// zero. The sentinel stands for a true |R| ≥ 1e10 and R·maxsub is up to ~4.0,
// so an unrefused subnormal multiplier would publish a clean, untainted zero
// in place of an unknown O(1) value: UNSOUND. MSL leaves the compare's DAZ
// behavior unspecified (`flush_charge_oracle::Hw::daz_compare` models BOTH),
// so which of the two closures holds had to be MEASURED, not argued:
//
//   * word KEPT on a subnormal partner  ⇒ the taint path is structurally
//     immune (compare reads the unflushed register) — v1's subnormal refusals
//     are belt-and-suspenders;
//   * word DROPPED beside a flushed ±0  ⇒ annihilation is real and the
//     charged walk guard's refusals of subnormal weights/bias/slopes (plus
//     nonzero-intercept §E refusal and the concretize subnormal-input
//     refusal) are LOAD-BEARING: they are exactly what keeps every subnormal
//     multiplier out of the twins.
//
// Either way, the refusals only close the routes if the annihilation domain
// is CONFINED to the strictly-subnormal multipliers they refuse. That proviso
// is the part no existing rung measured: the six-lane rung-5 table has no
// subnormal multiplier, and the rung-3/flush-class probes measure add/mul/fma
// but never a compare. The lanes below therefore pin the 2^-126 NORMAL
// boundary on both sides: a word lost at/above the boundary (or beside a
// value that did NOT flush) is HAZARDOUS — the guard's `< 2^-126` refusal
// predicate would no longer cover the annihilation domain — and charged-flush
// authority must refuse on such an adapter. `verify_subnormal_mult_taint` is
// consulted by `charged_flush_authority_cached` (fail-closed, primed by the
// charged constructor); the UNCHARGED five-rung ladder is deliberately
// untouched.
//
// MEASURED VERDICT — Apple M5 Max / Metal (see `h_gpu_tests::
// report_subnormal_mult_taint_lanes` and the pinned tables in `h_cpu_tests`):
// class PURE-FLUSH, and every strictly-subnormal multiplier ANNIHILATES the
// word beside a clean flushed ±0 in all three predicate families (GEMM both
// slots, activation slopes, row-OR partner), while both ±2^-126 boundary
// lanes stay sticky bit-exactly. The compare DAZ-flushes with the multiply
// (`METAL_CMP_DAZ`), the annihilation domain equals the refusal domain, and
// v1's guard refusals are confirmed LOAD-BEARING and SUFFICIENT.

/// One §H multiplier partner: the f32 bits, its class, and a label.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum MultClass {
    /// Strictly subnormal: refused by the charged walk guard; the DAZ compare
    /// MAY annihilate the word here, but only beside a clean flushed ±0.
    Subnormal,
    /// `±2^-126`, the smallest NORMAL: ADMITTED by the guard, so the word
    /// must survive on every admissible adapter — a loss here is HAZARDOUS.
    Boundary,
    /// `1.0`: armed control — the word must survive on ANY hardware.
    Unit,
    /// Exact `0.0`: annihilation control — the word must clear on ANY
    /// hardware (a stuck word is the ±inf tightness collapse in disguise).
    Zero,
}

/// The §H partner table: both signs, min/max subnormal, the ±2^-126 normal
/// boundary, and the two controls. Pinned by
/// `h_cpu_tests::subnormal_mult_table_discriminates`.
pub(crate) const SUBNORMAL_MULTIPLIERS: [(u32, MultClass, &str); 8] = [
    (0x0000_0001, MultClass::Subnormal, "+minsub 2^-149"),
    (0x8000_0001, MultClass::Subnormal, "-minsub"),
    (0x007F_FFFF, MultClass::Subnormal, "+maxsub 2^-126-2^-149"),
    (0x807F_FFFF, MultClass::Subnormal, "-maxsub"),
    (0x0080_0000, MultClass::Boundary, "+2^-126 smallest NORMAL"),
    (0x8080_0000, MultClass::Boundary, "-2^-126"),
    (0x3F80_0000, MultClass::Unit, "1.0 armed control"),
    (0x0000_0000, MultClass::Zero, "0.0 annihilation control"),
];

/// Number of §H partner lanes.
pub(crate) const NM: usize = SUBNORMAL_MULTIPLIERS.len();

/// Round-to-nearest f32 product of two f32 via exact f64 (48-bit significand
/// holds any f32×f32 product exactly; the single narrowing is the one RN).
fn rn_prod(a: f32, b: f32) -> f32 {
    (f64::from(a) * f64::from(b)) as f32
}

/// Subnormal → same-signed zero (the DAZ/FTZ transfer of the modeled
/// hardware; mirrors `subnormal_selfcheck::flush_subnormal_to_zero` and
/// `flush_charge_oracle::Hw::flush_operand`).
fn h_daz(x: f32) -> f32 {
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

/// Expected GEMM-twin diagonal VALUE bits for `sentinel × mult` at `k = 1`:
/// operand-DAZ (when `flush`), one RN product, the accumulator add from `0.0`
/// (which canonicalizes `-0` to `+0`), then `nan_safe_clamp`. No intermediate
/// here is ever a subnormal RESULT (the product is either ±0 or ≥ ~1.17e-28),
/// so result-FTZ contributes nothing on either hardware class.
pub(crate) fn gemm_expected_bits(sentinel: f32, mult: f32, flush: bool) -> u32 {
    let m = if flush { h_daz(mult) } else { mult };
    let prod = rn_prod(sentinel, m);
    let sum = 0.0f32 + prod;
    sum.clamp(-FALLBACK, FALLBACK).to_bits()
}

/// Expected activation-twin diagonal COEFFICIENT bits for the chain lane
/// `V = 1e10`, `ls = us = slope`, `β = 0`: `coeff = (V · sel) − 0.0`. The
/// bare multiply preserves the flushed slope's signed zero and the `− 0.0`
/// keeps it (IEEE: `-0 − +0 = -0`).
pub(crate) fn chain_coeff_expected_bits(slope: f32, flush: bool) -> u32 {
    let sel = if flush { h_daz(slope) } else { slope };
    let base = rn_prod(FALLBACK, sel);
    (base - 0.0f32).to_bits()
}

/// Complete §H measurement: every lane of the three sub-probes, exactly as
/// read back from the device. Plain data so the classifier is CPU-testable
/// against synthetic and pinned tables with no adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubnormalMultMeasurement {
    /// GEMM twin diagonal values: lanes `0..NM` carry the word on the A side
    /// (`a = sentinel`, `b = mult` — the production shape), lanes `NM..2·NM`
    /// on the B side (`a = mult`, `b = sentinel` — the symmetric conjunct).
    pub(crate) gemm_value_bits: [u32; 2 * NM],
    /// GEMM twin diagonal words, same lane order.
    pub(crate) gemm_word: [bool; 2 * NM],
    /// Activation chain diagonal coefficients (`A''`), one lane per partner
    /// used as the SLOPE (`ls == us == mult`).
    pub(crate) chain_coeff_bits: [u32; NM],
    /// End-of-chain VALUE-channel words (`taint_a_out`).
    pub(crate) chain_word_a: [bool; NM],
    /// End-of-chain ERROR-channel words (`taint_e_out`).
    pub(crate) chain_word_e: [bool; NM],
    /// Twin/base bit-identity (V, A'', E'') on each chain lane's diagonal.
    pub(crate) chain_twin_match: [bool; NM],
    /// `TAINT_ROW_OR_SHADER` per-COLUMN-partner mode: did row `i`'s seeded
    /// word survive partner `mult[i]`?
    pub(crate) rowor_kept: [bool; NM],
}

/// The §H verdict for one adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubnormalMultVerdict {
    /// Every nonzero multiplier — subnormal included — kept its word: the
    /// compares read the unflushed value and the taint path is structurally
    /// immune. The guard's subnormal refusals are defense-in-depth.
    StructurallyImmune,
    /// At least one strictly-subnormal multiplier annihilated its word, and
    /// every such loss sat beside a clean flushed ±0 while every boundary /
    /// unit lane stayed sticky and every zero lane cleared: the annihilation
    /// domain is confined to the multipliers the charged walk guard refuses,
    /// so the v1 refusals are load-bearing AND sufficient.
    AnnihilatesWithinSubnormal,
    /// Anything else: a word lost at/above the 2^-126 boundary, a word lost
    /// beside a non-flushed value, a stuck word on the zero control, a value
    /// diverging from the class model, twin drift, or a non-conformant
    /// adapter. Charged-flush authority must refuse.
    Hazardous,
}

/// Classify one complete §H measurement under the adapter's measured flush
/// class. Pure function; fail-closed (`Hazardous`) on every unmodeled shape.
pub(crate) fn classify_subnormal_mult(
    class: FlushClass,
    m: &SubnormalMultMeasurement,
) -> SubnormalMultVerdict {
    let flush = match class {
        FlushClass::Conformant => false,
        FlushClass::PureFlush => true,
        FlushClass::NonConformant => return SubnormalMultVerdict::Hazardous,
    };
    let mut any_annihilated = false;

    // GEMM lanes, word on either side (the value model is symmetric).
    for side in 0..2 {
        for (j, &(mult_bits, mult_class, _)) in SUBNORMAL_MULTIPLIERS.iter().enumerate() {
            let i = side * NM + j;
            let mult = f32::from_bits(mult_bits);
            if m.gemm_value_bits[i] != gemm_expected_bits(FALLBACK, mult, flush) {
                return SubnormalMultVerdict::Hazardous;
            }
            let word = m.gemm_word[i];
            match mult_class {
                MultClass::Unit | MultClass::Boundary => {
                    if !word {
                        return SubnormalMultVerdict::Hazardous;
                    }
                }
                MultClass::Zero => {
                    if word {
                        return SubnormalMultVerdict::Hazardous;
                    }
                }
                MultClass::Subnormal => {
                    if !word {
                        // A dropped word is admissible ONLY beside a clean
                        // flushed ±0 (redundant with the bit check above,
                        // kept as an independent fail-closed conjunct).
                        if !flush || m.gemm_value_bits[i] & 0x7fff_ffff != 0 {
                            return SubnormalMultVerdict::Hazardous;
                        }
                        any_annihilated = true;
                    }
                }
            }
        }
    }

    // Activation chain lanes (the partner is the SLOPE).
    for (i, &(mult_bits, mult_class, _)) in SUBNORMAL_MULTIPLIERS.iter().enumerate() {
        if !m.chain_twin_match[i] {
            return SubnormalMultVerdict::Hazardous;
        }
        let mult = f32::from_bits(mult_bits);
        if m.chain_coeff_bits[i] != chain_coeff_expected_bits(mult, flush) {
            return SubnormalMultVerdict::Hazardous;
        }
        let (wa, we) = (m.chain_word_a[i], m.chain_word_e[i]);
        if wa != we {
            // The twin computes `taint_e_out = select(0,te,live) | ta_kept`
            // with `te = 0` seeded here, so the two words can only disagree
            // through a transport defect.
            return SubnormalMultVerdict::Hazardous;
        }
        match mult_class {
            MultClass::Unit | MultClass::Boundary => {
                if !wa {
                    return SubnormalMultVerdict::Hazardous;
                }
            }
            MultClass::Zero => {
                if wa {
                    return SubnormalMultVerdict::Hazardous;
                }
            }
            MultClass::Subnormal => {
                if !wa {
                    if !flush || m.chain_coeff_bits[i] & 0x7fff_ffff != 0 {
                        return SubnormalMultVerdict::Hazardous;
                    }
                    any_annihilated = true;
                }
            }
        }
    }

    // Row-OR lanes (the partner is the annihilation conjunct's operand; this
    // op has no value channel of its own).
    for (i, &(_, mult_class, _)) in SUBNORMAL_MULTIPLIERS.iter().enumerate() {
        let kept = m.rowor_kept[i];
        match mult_class {
            MultClass::Unit | MultClass::Boundary => {
                if !kept {
                    return SubnormalMultVerdict::Hazardous;
                }
            }
            MultClass::Zero => {
                if kept {
                    return SubnormalMultVerdict::Hazardous;
                }
            }
            MultClass::Subnormal => {
                if !kept {
                    if !flush {
                        return SubnormalMultVerdict::Hazardous;
                    }
                    any_annihilated = true;
                }
            }
        }
    }

    if any_annihilated {
        SubnormalMultVerdict::AnnihilatesWithinSubnormal
    } else {
        SubnormalMultVerdict::StructurallyImmune
    }
}

/// The complete modeled measurement for a hardware class: `flush` selects the
/// pure-flush value model, `subnormal_words_kept` the compare semantics
/// (`true` = compare reads the unflushed register, `false` = compare
/// DAZ-flushes with the multiply). Shared by the CPU classifier tests and the
/// live report's pinned-table comparison.
#[cfg(test)]
pub(crate) fn model_measurement(
    flush: bool,
    subnormal_words_kept: bool,
) -> SubnormalMultMeasurement {
    let mut gemm_value_bits = [0u32; 2 * NM];
    let mut gemm_word = [false; 2 * NM];
    let mut chain_coeff_bits = [0u32; NM];
    let mut chain_word_a = [false; NM];
    for side in 0..2 {
        for (j, &(mult_bits, mult_class, _)) in SUBNORMAL_MULTIPLIERS.iter().enumerate() {
            let i = side * NM + j;
            let mult = f32::from_bits(mult_bits);
            gemm_value_bits[i] = gemm_expected_bits(FALLBACK, mult, flush);
            gemm_word[i] = match mult_class {
                MultClass::Unit | MultClass::Boundary => true,
                MultClass::Zero => false,
                MultClass::Subnormal => !flush || subnormal_words_kept,
            };
        }
    }
    for (i, &(mult_bits, mult_class, _)) in SUBNORMAL_MULTIPLIERS.iter().enumerate() {
        let mult = f32::from_bits(mult_bits);
        chain_coeff_bits[i] = chain_coeff_expected_bits(mult, flush);
        chain_word_a[i] = match mult_class {
            MultClass::Unit | MultClass::Boundary => true,
            MultClass::Zero => false,
            MultClass::Subnormal => !flush || subnormal_words_kept,
        };
    }
    let rowor_kept = chain_word_a;
    SubnormalMultMeasurement {
        gemm_value_bits,
        gemm_word,
        chain_coeff_bits,
        chain_word_a,
        chain_word_e: chain_word_a,
        chain_twin_match: [true; NM],
        rowor_kept,
    }
}

/// `Params { rows, cols, use_partner, _pad }` —
/// [`sh::TAINT_ROW_OR_SHADER`].
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HRowOrParams {
    rows: u32,
    cols: u32,
    use_partner: u32,
    _pad: u32,
}

impl WgpuDevice {
    /// #flush-charge §H: is this adapter's taint-word annihilation domain
    /// confined to the strictly-subnormal multipliers the charged walk guard
    /// refuses (or the word path structurally immune)?
    ///
    /// `true` ⇒ the measured verdict is [`SubnormalMultVerdict::
    /// StructurallyImmune`] or [`SubnormalMultVerdict::
    /// AnnihilatesWithinSubnormal`]. `false` ⇒ HAZARDOUS or the probe could
    /// not run (fail-closed). Cached per device; consulted only by
    /// `charged_flush_authority_cached` — no uncharged rung reads it.
    pub(crate) fn verify_subnormal_mult_taint(&self) -> bool {
        if selfcheck_forced_to_fail() {
            return false;
        }
        *self
            .subnormal_mult_taint_selfcheck
            .get_or_init(|| self.run_subnormal_mult_taint_probe())
    }

    /// Run (and log) the one-time §H probe. CONSERVATIVE: a hazardous verdict
    /// OR any GPU error → `false`.
    fn run_subnormal_mult_taint_probe(&self) -> bool {
        // Characterize OUTSIDE our own checked section (both helpers take the
        // GPU serialization lock; nesting would self-deadlock).
        let class = self.characterize_flush_policy();
        match self.run_gpu_checked("verify_subnormal_mult_taint", || {
            self.subnormal_mult_lanes_inner()
        }) {
            Ok(meas) => {
                let verdict = classify_subnormal_mult(class, &meas);
                if verdict == SubnormalMultVerdict::Hazardous {
                    tracing::warn!(
                        target: "ny_gpu::wgpu",
                        adapter = %self.adapter_info.name,
                        backend = ?self.adapter_info.backend,
                        ?class,
                        "#flush-charge §H SUBNORMAL-MULTIPLIER probe measured a \
                         HAZARDOUS annihilation domain: a taint word was lost \
                         outside the strictly-subnormal multipliers the charged \
                         walk guard refuses (or beside a non-flushed value). \
                         REFUSING charged-flush authority (fail-closed)"
                    );
                    false
                } else {
                    tracing::info!(
                        target: "ny_gpu::wgpu",
                        adapter = %self.adapter_info.name,
                        backend = ?self.adapter_info.backend,
                        ?class,
                        ?verdict,
                        "#flush-charge §H subnormal-multiplier probe verdict"
                    );
                    true
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "ny_gpu::wgpu",
                    adapter = %self.adapter_info.name,
                    backend = ?self.adapter_info.backend,
                    error = %e,
                    "#flush-charge §H subnormal-multiplier probe could not run: \
                     REFUSING charged-flush authority (fail-closed)"
                );
                false
            }
        }
    }

    /// Diagnostic: the measured flush class, the raw §H measurement, and its
    /// verdict, so a refusing adapter is CHARACTERIZED lane by lane.
    #[cfg(all(test, feature = "gpu-tests"))]
    pub(crate) fn subnormal_mult_report(
        &self,
    ) -> Result<(FlushClass, SubnormalMultMeasurement, SubnormalMultVerdict)> {
        let class = self.characterize_flush_policy();
        let meas = self.run_gpu_checked("subnormal_mult_report", || {
            self.subnormal_mult_lanes_inner()
        })?;
        let verdict = classify_subnormal_mult(class, &meas);
        Ok((class, meas, verdict))
    }

    /// Dispatch the three §H sub-probes and read every lane back.
    ///
    /// 1. GEMM twin at `(m,k,n) = (2·NM, 1, 2·NM)`: diagonal lane `i < NM`
    ///    puts the worded sentinel in `A` against multiplier `i` in `W` (the
    ///    production shape); lane `NM + i` mirrors it (worded sentinel in
    ///    `W`, multiplier in `A`) to exercise the symmetric conjunct.
    /// 2. The full dual chain at `(NM, K)` with `V = 1e10` and
    ///    `ls = us = mult[i]` — the activation twin's `slopes_live` compare.
    /// 3. `TAINT_ROW_OR_SHADER` in per-COLUMN-partner mode over an `NM × NM`
    ///    diagonal word matrix with `partner = mult` — the shipped row-OR
    ///    annihilation conjunct.
    fn subnormal_mult_lanes_inner(&self) -> Result<SubnormalMultMeasurement> {
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let storage_src = storage | wgpu::BufferUsages::COPY_SRC;
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let dev = &self.device;
        let f32_buf = |label: &'static str, data: &[f32], usage: wgpu::BufferUsages| {
            let b = create_buffer(dev, label, (data.len() * 4) as u64, usage);
            self.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
            b
        };
        let u32_buf = |label: &'static str, data: &[u32], usage: wgpu::BufferUsages| {
            let b = create_buffer(dev, label, (data.len() * 4) as u64, usage);
            self.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
            b
        };

        // ---- sub-probe 1: GEMM twin, both word sides ----------------------
        let nl2 = 2 * NM;
        let mut a = vec![0.0f32; nl2]; // [nl2 x 1]
        let mut b = vec![0.0f32; nl2]; // [1 x nl2]
        let mut wa = vec![0u32; nl2];
        let mut wb = vec![0u32; nl2];
        for (j, &(mult_bits, _, _)) in SUBNORMAL_MULTIPLIERS.iter().enumerate() {
            let mult = f32::from_bits(mult_bits);
            // Word on A: a = sentinel (worded), b = multiplier (clean).
            a[j] = FALLBACK;
            wa[j] = 1;
            b[j] = mult;
            // Word on B: a = multiplier (clean), b = sentinel (worded).
            a[NM + j] = mult;
            b[NM + j] = FALLBACK;
            wb[NM + j] = 1;
        }
        let a_buf = f32_buf("u4h_a", &a, storage);
        let b_buf = f32_buf("u4h_b", &b, storage);
        let wa_buf = u32_buf("u4h_wa", &wa, storage);
        let wb_buf = u32_buf("u4h_wb", &wb, storage);
        let zeros_out = vec![0.0f32; nl2 * nl2];
        let zeros_out_u32 = vec![0u32; nl2 * nl2];
        let out_buf = f32_buf("u4h_out", &zeros_out, storage_src);
        let wout_buf = u32_buf("u4h_wout", &zeros_out_u32, storage_src);
        let gemm_p = create_buffer(dev, "u4h_gemm_p", 16, uniform);
        self.queue.write_buffer(
            &gemm_p,
            0,
            bytemuck::cast_slice(&[GemmParams {
                m: nl2 as u32,
                k: 1,
                n: nl2 as u32,
                _pad: 0,
            }]),
        );
        let gemm_taint = self.create_simple_pipeline(
            sh::GEMM_F32_TAINT_SHADER,
            "u4h_gemm_taint",
            &[false, false, true, false, false, true],
        );
        let gemm_groups = nl2.div_ceil(16) as u32;

        // ---- sub-probe 3 operands: row-OR over a diagonal word matrix -----
        let mut diag_words = vec![0u32; NM * NM];
        let mut partner = vec![0.0f32; NM];
        for (j, &(mult_bits, _, _)) in SUBNORMAL_MULTIPLIERS.iter().enumerate() {
            diag_words[j * NM + j] = 1;
            partner[j] = f32::from_bits(mult_bits);
        }
        let diag_words_buf = u32_buf("u4h_rowor_words", &diag_words, storage);
        let partner_buf = f32_buf("u4h_rowor_partner", &partner, storage);
        let rows_out_buf = u32_buf("u4h_rowor_rows", &[0u32; NM], storage_src);
        let rowor_p = create_buffer(dev, "u4h_rowor_p", 16, uniform);
        self.queue.write_buffer(
            &rowor_p,
            0,
            bytemuck::cast_slice(&[HRowOrParams {
                rows: NM as u32,
                cols: NM as u32,
                use_partner: 1,
                _pad: 0,
            }]),
        );
        let rowor = self.create_simple_pipeline(
            sh::TAINT_ROW_OR_SHADER,
            "u4h_rowor",
            &[false, false, true],
        );

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("u4h_subnormal_mult_encoder"),
        });
        self.pass_simple_2d(
            &mut encoder,
            &gemm_taint,
            &gemm_p,
            &[&a_buf, &b_buf, &out_buf, &wa_buf, &wb_buf, &wout_buf],
            gemm_groups,
            gemm_groups,
        );
        self.pass_simple(
            &mut encoder,
            &rowor,
            &rowor_p,
            &[&diag_words_buf, &partner_buf, &rows_out_buf],
            ((NM * NM) as u32).div_ceil(256),
        );

        let bytes_out = (nl2 * nl2 * 4) as u64;
        let stage = |label: &'static str, bytes: u64| {
            create_buffer(
                dev,
                label,
                bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )
        };
        let st_out = stage("u4h_st_out", bytes_out);
        let st_wout = stage("u4h_st_wout", bytes_out);
        let st_rows = stage("u4h_st_rows", (NM * 4) as u64);
        encoder.copy_buffer_to_buffer(&out_buf, 0, &st_out, 0, bytes_out);
        encoder.copy_buffer_to_buffer(&wout_buf, 0, &st_wout, 0, bytes_out);
        encoder.copy_buffer_to_buffer(&rows_out_buf, 0, &st_rows, 0, (NM * 4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let out = WgpuDevice::read_u32_buffer(dev, &st_out, nl2 * nl2)?;
        let wout = WgpuDevice::read_u32_buffer(dev, &st_wout, nl2 * nl2)?;
        let rows = WgpuDevice::read_u32_buffer(dev, &st_rows, NM)?;

        // ---- sub-probe 2: the dual chain with the partner as the SLOPE ----
        let mut ca = vec![0.0f32; NM * K];
        let mut cw = vec![0.0f32; K * NM];
        let ce = vec![0.0f32; NM * K];
        let mut ls = vec![0.0f32; NM];
        for (i, &(mult_bits, _, _)) in SUBNORMAL_MULTIPLIERS.iter().enumerate() {
            ca[i * K] = FALLBACK;
            cw[i] = 1.0; // W row 0, column i
            ls[i] = f32::from_bits(mult_bits);
        }
        let taint_a_words: Vec<u32> = ca.iter().map(|&x| u32::from(x.abs() >= FALLBACK)).collect();
        let taint_e_words = vec![0u32; NM * K];
        let beta = vec![0.0f32; NM];
        let chain = self.dual_chain_run(
            NM,
            K,
            &ca,
            &cw,
            &ce,
            &ls,
            &ls,
            &beta,
            &taint_a_words,
            &taint_e_words,
        )?;

        // ---- assemble -----------------------------------------------------
        let mut meas = SubnormalMultMeasurement {
            gemm_value_bits: [0; 2 * NM],
            gemm_word: [false; 2 * NM],
            chain_coeff_bits: [0; NM],
            chain_word_a: [false; NM],
            chain_word_e: [false; NM],
            chain_twin_match: [false; NM],
            rowor_kept: [false; NM],
        };
        for i in 0..nl2 {
            let d = i * nl2 + i;
            meas.gemm_value_bits[i] = out[d];
            meas.gemm_word[i] = wout[d] != 0;
        }
        for i in 0..NM {
            let d = i * NM + i;
            meas.chain_coeff_bits[i] = chain.a_out[d];
            meas.chain_word_a[i] = chain.word_a_out[d] != 0;
            meas.chain_word_e[i] = chain.word_e_out[d] != 0;
            meas.chain_twin_match[i] = chain.twin_v[d] == chain.v[d]
                && chain.twin_a_out[d] == chain.a_out[d]
                && chain.twin_e_out[d] == chain.e_out[d];
            meas.rowor_kept[i] = rows[i] != 0;
        }
        Ok(meas)
    }
}

#[cfg(test)]
mod cpu_tests {
    use super::*;

    /// The probe's sentinel constant must be the one the shaders and the CPU
    /// reference actually use. If `FALLBACK_BOUND` ever moves, this fails
    /// before the probe can silently stop probing anything.
    #[test]
    fn sentinel_matches_core() {
        assert_eq!(FALLBACK, ny_core::FALLBACK_BOUND);
        assert_eq!(FALLBACK, ny_core::CROWN_COEFF_MAX);
        assert!(sh::GEMM_F32_SHADER.contains("const FALLBACK_BOUND: f32 = 1e10;"));
        assert!(sh::CROWN_AW_ERROR_COMBINE_SHADER.contains("const FALLBACK_BOUND: f32 = 1e10;"));
        assert!(sh::CROWN_AW_ERROR_COMBINE_SHADER.contains(
            "if (s_prod[i] >= FALLBACK_BOUND || prop[i] >= FALLBACK_BOUND) { e = 1e30; }"
        ));
    }

    /// The taint twins this probe dispatches must still carry the canonical
    /// propagation rule: OR'd never multiplied, only clean exact-zero partners
    /// annihilate, saturation self-seeds. Pinned as source text so a twin
    /// edit that weakens a conjunct fails HERE before the probe silently
    /// measures a different channel.
    #[test]
    fn taint_twin_shaders_pin_the_propagation_rule() {
        assert!(sh::GEMM_F32_TAINT_SHADER.contains("const FALLBACK_BOUND: f32 = 1e10;"));
        assert!(sh::GEMM_F32_TAINT_SHADER
            .contains("if (taw != 0u && (bv != 0.0 || tbw != 0u)) { taint = 1u; }"));
        assert!(sh::GEMM_F32_TAINT_SHADER
            .contains("if (tbw != 0u && (av != 0.0 || taw != 0u)) { taint = 1u; }"));
        assert!(sh::GEMM_F32_TAINT_SHADER
            .contains("if (guarded != guarded || abs(guarded) >= FALLBACK_BOUND) { taint = 1u; }"));
        assert!(sh::CROWN_ACTIVATION_RESIDENT_TAINT_SHADER
            .contains("let slopes_live = lsv != 0.0 || usv != 0.0;"));
        assert!(sh::CROWN_ACTIVATION_RESIDENT_TAINT_SHADER
            .contains("let ta_kept = select(0u, ta, slopes_live);"));
        assert!(sh::CROWN_ACTIVATION_RESIDENT_TAINT_SHADER
            .contains("taint_e_out[idx] = select(0u, te, slopes_live) | ta_kept;"));
    }

    /// The lane table must DISCRIMINATE: it must contain at least one lane that
    /// can only pass if the taint survives a NON-annihilating composition, and
    /// at least one armed control whose composition does not scale at all. A
    /// table of only exact-zero lanes would be a tautology.
    #[test]
    fn lane_table_discriminates() {
        let sticky: Vec<&Lane> = LANES
            .iter()
            .filter(|l| l.expect == Expect::StaySticky)
            .collect();
        assert!(sticky.len() >= 3, "need several sticky lanes");
        // At least one sticky lane scales the sentinel DOWN by a large factor —
        // that is the laundering adversary.
        assert!(
            LANES.iter().any(|l| l.expect == Expect::StaySticky
                && (l.w.iter().any(|&x| x != 0.0 && x.abs() < 1e-10)
                    || (l.slope != 0.0 && l.slope.abs() < 1e-10))),
            "no downscaling adversary in the table — the probe would be a tautology"
        );
        // At least one armed control with unit scaling.
        assert!(
            LANES
                .iter()
                .any(|l| l.expect == Expect::StaySticky && l.slope == 1.0 && l.w.contains(&1.0)),
            "no unit-scaling control lane"
        );
        // At least one exact-zero annihilation lane.
        assert!(
            LANES
                .iter()
                .any(|l| l.expect == Expect::AnnihilateExactly && l.slope == 0.0),
            "no exact-zero annihilation lane"
        );
        // Every lane actually introduces a sentinel or a taint somewhere.
        for l in LANES.iter() {
            assert!(
                l.a.iter().any(|&x| x.abs() >= FALLBACK) || l.e.iter().any(|&x| x >= FALLBACK),
                "lane `{}` carries no taint at all",
                l.label
            );
        }
    }

    /// The end-of-chain MAGNITUDE proxy must retain the historical baseline's
    /// exact classification, independently of the now-selected word channel.
    #[test]
    fn taint_predicate_matches_downstream_guards() {
        assert!(taint_visible(FALLBACK, 0.0));
        assert!(taint_visible(-FALLBACK, 0.0));
        assert!(taint_visible(0.0, ERR_TAINT));
        assert!(taint_visible(f32::NAN, 0.0));
        assert!(taint_visible(0.0, f32::INFINITY));
        assert!(!taint_visible(1.0, 1.0));
        // The laundered magnitudes the adversary lanes produce are, by
        // construction, INVISIBLE to those guards.
        assert!(!taint_visible(1e-10, 1e-6));
        assert!(!taint_visible(1e-25, 2e5));
    }

    /// The word channel may move the RUNG only while the source-reviewed guard
    /// consult is armed. These pins remain written against the gate constant so
    /// a future source-level quarantine change stays fail-closed.
    #[test]
    fn word_channel_is_gated_until_production_guards_consult_it() {
        // Lane-2 shape: laundered magnitudes (measured 2026-08-06: A''=1e-10,
        // E''=5.4e-17), word carried by the twins. With-word passes; the rung
        // verdict must track the gate.
        let v = classify_lane(Expect::StaySticky, 1e-10, 5.4e-17, true, true, true);
        assert!(!v.tainted_magnitude);
        assert!(
            v.tainted_at_end,
            "the word is an ADDITIONAL detection channel"
        );
        assert!(!v.ok_magnitude_only);
        assert!(v.ok_with_word);
        assert_eq!(v.ok, PRODUCTION_GUARDS_CONSULT_TAINT_WORD);

        // Lane-5 shape: the error word alone (measured: A''=1e-25, E''=2.0e5).
        let v = classify_lane(Expect::StaySticky, 1e-25, 2.0e5, false, true, true);
        assert!(!v.ok_magnitude_only && v.ok_with_word);
        assert_eq!(v.ok, PRODUCTION_GUARDS_CONSULT_TAINT_WORD);

        // Magnitude detection alone still passes with-word (lanes 0/3 need no
        // words): the word channel is additive, never a new obligation for
        // lanes the baseline already catches.
        let v = classify_lane(Expect::StaySticky, 1e10, 2.000002e30, false, false, true);
        assert!(v.ok_magnitude_only && v.ok_with_word && v.ok);
    }

    /// Annihilation lanes must be clean in EVERY channel for the with-word
    /// verdict, and a twin that computed different values proves nothing.
    #[test]
    fn annihilation_requires_clean_words_and_twin_identity() {
        // Clean dead-ReLU annihilation (measured: A''=0, E''=7.5e-37) passes
        // both verdicts.
        let v = classify_lane(Expect::AnnihilateExactly, 0.0, 7.5e-37, false, false, true);
        assert!(v.ok_magnitude_only && v.ok_with_word && !v.tainted_at_end);

        // A stuck word on a dead ReLU breaks the `!= 0` conjuncts: the
        // with-word verdict refuses (that stuck word IS the ±inf tightness
        // collapse in disguise). The historical magnitude-only proxy still
        // passes, but the armed source gate selects the refusing word verdict.
        let v = classify_lane(Expect::AnnihilateExactly, 0.0, 7.5e-37, false, true, true);
        assert!(v.ok_magnitude_only && !v.ok_with_word);
        assert_eq!(v.ok, !PRODUCTION_GUARDS_CONSULT_TAINT_WORD);

        // Twin value divergence: even a word that LOOKS right counts for
        // nothing, because it was computed beside different values than the
        // shipped chain's (fail-closed).
        let v = classify_lane(Expect::StaySticky, 1e-10, 5.4e-17, true, true, false);
        assert!(!v.ok_with_word);
        let v = classify_lane(Expect::AnnihilateExactly, 0.0, 7.5e-37, false, false, false);
        assert!(!v.ok_with_word);
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};

    /// MEASUREMENT: report every lane's real behaviour on this adapter, so the
    /// U4 evidence is a test output rather than a claim in a document. Prints
    /// BOTH aggregate verdicts — "magnitude-only" (the 2026-08-06 baseline)
    /// and "with-word" (the now-selected twin chain) — so the original
    /// laundering defect remains measurable after the source gate is armed.
    ///
    /// Asserts only the invariants that must hold on ANY hardware: the armed
    /// control lane must carry the taint in BOTH channels (else the probe
    /// proves nothing), the annihilation lanes' words must be clear (the
    /// `!= 0` conjuncts), and the rung must imply every lane passing.
    #[test]
    fn report_sentinel_taint_lanes() {
        let _serial = gpu_test_serial_guard();
        let device = require_device();
        let lanes = device
            .sentinel_taint_report()
            .expect("the #u4 probe must be able to run");

        println!(
            "[#u4 sentinel-taint stickiness] adapter={} backend={:?}",
            device.adapter_info.name, device.adapter_info.backend
        );
        for l in &lanes {
            println!(
                "  {:<7} {}\n      V={:e} s_prod={:e} prop={:e} E'={:e} -> A''={:e} E''={:e}\n      \
                 words: V={} E'={} -> A''={} E''={} twin_match={} | magnitude_tainted={} \
                 unified_tainted={}\n      ok(magnitude-only)={} ok(with-word)={} \
                 (expects_sticky={})\n      why: {}",
                if l.ok { "[PASS]" } else { "[FAIL]" },
                l.label,
                l.value,
                l.s_prod,
                l.prop,
                l.combined_err,
                l.coeff,
                l.err,
                l.taint_word_v,
                l.taint_word_ecomb,
                l.taint_word_a,
                l.taint_word_e,
                l.twin_value_matches,
                l.tainted_magnitude,
                l.tainted_at_end,
                l.ok_magnitude_only,
                l.ok_with_word,
                l.expects_sticky,
                l.rationale,
            );
        }
        let magnitude_only_pass = lanes.iter().all(|l| l.ok_magnitude_only);
        let with_word_pass = lanes.iter().all(|l| l.ok_with_word);
        let armed = device.verify_sentinel_taint_sticky();
        println!(
            "  => magnitude-only: {}, with-word: {} \
             (PRODUCTION_GUARDS_CONSULT_TAINT_WORD = {})",
            if magnitude_only_pass { "PASS" } else { "FAIL" },
            if with_word_pass { "PASS" } else { "FAIL" },
            PRODUCTION_GUARDS_CONSULT_TAINT_WORD,
        );
        println!("  => verify_sentinel_taint_sticky() = {armed}");

        // NON-VACUITY, magnitude channel: the unit-scaling control must carry
        // the taint. If it does not, the probe is misconfigured and every
        // other lane's result is meaningless.
        let control = lanes
            .iter()
            .find(|l| l.label.starts_with("3 "))
            .expect("control lane present");
        assert!(
            control.tainted_at_end,
            "#u4 probe is MISCONFIGURED: the unit-scaling control lane lost the \
             sentinel, so nothing else this probe reports can be interpreted"
        );
        // NON-VACUITY, word channel: the twins must carry the control lane's
        // word through a unit-scale composition in BOTH channels (measured
        // 10/10 on the GB10). A silent bind/dispatch failure reads back zeros
        // and fails HERE, loudly, instead of masquerading as a clean channel.
        assert!(
            control.taint_word_a && control.taint_word_e,
            "#u4 twin chain is DARK: the unit-scaling control lane's taint \
             words did not arrive ({control:?})"
        );

        // The `!= 0` conjuncts: exact dead-ReLU annihilation must clear the
        // WORDS too, or every dead ReLU would poison its row (the ±inf
        // tightness collapse the out-of-band design exists to avoid).
        for l in lanes.iter().filter(|l| !l.expects_sticky) {
            assert!(
                !l.taint_word_a && !l.taint_word_e,
                "annihilation lane `{}` has a stuck taint word ({l:?})",
                l.label
            );
        }

        // The rung must imply every lane passing.
        if armed {
            assert!(
                lanes.iter().all(|l| l.ok),
                "the rung passed while a lane FAILED"
            );
        }
    }

    /// Fail-closed: the forced-fail hook drives the rung closed, and a closed
    /// rung closes the whole authority ladder.
    #[test]
    fn forced_failure_refuses_the_rung_and_the_ladder() {
        let _serial = gpu_test_serial_guard();
        let device = require_device();

        set_force_sentinel_taint_selfcheck_fail(true);
        assert!(
            !device.verify_sentinel_taint_sticky(),
            "forced #u4 failure must refuse the rung"
        );
        assert!(
            !device.sound_gpu_authority(),
            "an ordinary device must remain unarmed while a rung is failing"
        );
        set_force_sentinel_taint_selfcheck_fail(false);
    }

    /// Deterministic 64-bit LCG (Knuth MMIX constants). No external rng
    /// crate: the operand set must be bit-reproducible on every host, so a
    /// drift-pin failure is a DIVERGENCE, never a flake.
    struct Lcg(u64);

    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }

        /// One random operand: `sign · mantissa[1,2) · 2^e` with `e` uniform
        /// in `[-30, 8]` (the CROWN coefficient regime, from near the flush
        /// floors up past the activation magnitudes), built from exact bit
        /// constructions so every value is a clean f32; ~1% EXACT zeros feed
        /// the twins' `!= 0` annihilation conjuncts.
        fn next_operand(&mut self) -> f32 {
            let r = self.next_u32();
            if r.is_multiple_of(100) {
                return 0.0;
            }
            let exp = -30i32 + ((r >> 8) % 39) as i32;
            let scale = f32::from_bits(((127 + exp) as u32) << 23);
            let mantissa = 1.0 + (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
            let sign = if self.next_u32() & 1 == 0 {
                1.0f32
            } else {
                -1.0f32
            };
            sign * mantissa * scale
        }
    }

    /// Seed for [`random_wide_twin_drift_pin`], a const so any future
    /// divergence report is reproducible verbatim. Mixed per shape below.
    const RANDOM_WIDE_DRIFT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

    /// #u4 drift pin: twin == base BIT-IDENTITY over random WIDE operands at
    /// CROWN-ish shapes. The lane probe compares twin values on six diagonal
    /// elements at K=4 only; the twins are slated to become the production
    /// dispatches, so a compiler divergence between the two compiled modules
    /// (e.g. a different fma-contraction choice) must be caught over a wide
    /// operand distribution — and across the GEMM 16x16 tiling boundary,
    /// which is why (nl=4, k=257) runs 17 k-tiles with a 15-lane zero-padded
    /// tail while (nl=8, k=64) exercises the small-shape regime wide.
    #[test]
    fn random_wide_twin_drift_pin() {
        let _serial = gpu_test_serial_guard();
        let device = require_device();

        for (nl, k) in [(8usize, 64usize), (4, 257)] {
            let mut rng = Lcg(RANDOM_WIDE_DRIFT_SEED ^ ((nl as u64) << 32) ^ k as u64);
            let mut a: Vec<f32> = (0..nl * k).map(|_| rng.next_operand()).collect();
            let w: Vec<f32> = (0..k * nl).map(|_| rng.next_operand()).collect();
            // E is the incoming coefficient-error channel: non-negative by
            // construction in production, so feed magnitudes.
            let e: Vec<f32> = (0..nl * k).map(|_| rng.next_operand().abs()).collect();
            let ls: Vec<f32> = (0..nl).map(|_| rng.next_operand()).collect();
            let us: Vec<f32> = (0..nl).map(|_| rng.next_operand()).collect();
            let beta: Vec<f32> = (0..nl).map(|_| rng.next_operand()).collect();

            // One operand at EXACTLY the sentinel, on an interior row/column
            // so the saturated tap sits inside a middle k-tile.
            let srow = nl / 2;
            let scol = k / 2;
            a[srow * k + scol] = FALLBACK;
            assert!(
                (0..nl).any(|j| w[scol * nl + j] != 0.0),
                "degenerate seed: weight row {scol} is all-zero, the planted \
                 sentinel cannot reach any output — pick a new \
                 RANDOM_WIDE_DRIFT_SEED"
            );

            // Host-seed the words with the SAME rule the lane wrapper uses: a
            // word wherever an operand IS a saturation artifact.
            let taint_a_words: Vec<u32> =
                a.iter().map(|&x| u32::from(x.abs() >= FALLBACK)).collect();
            let taint_e_words: Vec<u32> =
                e.iter().map(|&x| u32::from(x.abs() >= FALLBACK)).collect();

            let out = device
                .dual_chain_probe(
                    nl,
                    k,
                    &a,
                    &w,
                    &e,
                    &ls,
                    &us,
                    &beta,
                    &taint_a_words,
                    &taint_e_words,
                )
                .expect("the #u4 dual chain must be able to run");

            // Non-vacuity: the planted sentinel's word must arrive in the
            // first-hop twins (a silent bind/dispatch failure reads back
            // zeros and would otherwise masquerade as a clean pass).
            assert!(
                out.word_v.iter().any(|&x| x != 0),
                "(nl={nl}, k={k}) the V twin carried no word: the probe is dark"
            );
            assert!(
                out.word_s.iter().any(|&x| x != 0),
                "(nl={nl}, k={k}) the S twin carried no word: the probe is dark"
            );
            // And the converse: E is clean (|e| < 512) and W words are zero,
            // so P = E@|W| can neither inherit nor self-seed (tops out far
            // below FALLBACK_BOUND) — the twin must not INVENT taint.
            assert!(
                out.word_p.iter().all(|&x| x == 0),
                "(nl={nl}, k={k}) the P twin invented a word"
            );

            // The pin: EVERY element of every stage bit-identical between the
            // base chain and the twin chain.
            for (name, base, twin) in [
                ("V", &out.v, &out.twin_v),
                ("S", &out.s, &out.twin_s),
                ("P", &out.p, &out.twin_p),
                ("E-combined", &out.ecomb, &out.twin_ecomb),
                ("A''", &out.a_out, &out.twin_a_out),
                ("E''", &out.e_out, &out.twin_e_out),
            ] {
                assert_eq!(base.len(), nl * nl, "buffer {name}: wrong base size");
                assert_eq!(twin.len(), nl * nl, "buffer {name}: wrong twin size");
                for (idx, (&base_bits, &twin_bits)) in base.iter().zip(twin.iter()).enumerate() {
                    assert_eq!(
                        base_bits,
                        twin_bits,
                        "[#u4 drift] (nl={nl}, k={k}) buffer {name} diverged at index \
                         {idx}: base_bits={base_bits:#010x} twin_bits={twin_bits:#010x} \
                         (base={}, twin={}) — the two compiled modules disagree (e.g. \
                         different fma contraction); the twins cannot become production \
                         dispatches on this adapter",
                        f32::from_bits(base_bits),
                        f32::from_bits(twin_bits),
                    );
                }
            }
            println!(
                "[#u4 drift pin] (nl={nl}, k={k}): {} elements x 6 buffers bit-identical",
                nl * nl
            );
        }
    }
}

/// #flush-charge §H CPU tests: the classifier's gating semantics, with no
/// device (oracle-test discipline: every admissible and every refusing shape
/// is pinned by name).
#[cfg(test)]
mod h_cpu_tests {
    use super::*;

    /// The partner table must DISCRIMINATE: every Subnormal entry is strictly
    /// subnormal in both signs and at both endpoints, the Boundary entries
    /// are EXACTLY ±2^-126 (the guard's refusal threshold), and every
    /// sentinel × nonzero-partner product is NORMAL under IEEE — so a
    /// conformant adapter's lane can never hide behind an underflow.
    #[test]
    fn subnormal_mult_table_discriminates() {
        let sub = |x: f32| x != 0.0 && x.abs() < f32::MIN_POSITIVE;
        let mut signs = (false, false);
        let mut endpoints = (false, false);
        for &(bits, class, label) in &SUBNORMAL_MULTIPLIERS {
            let v = f32::from_bits(bits);
            match class {
                MultClass::Subnormal => {
                    assert!(sub(v), "lane `{label}` is not subnormal");
                    signs.0 |= v.is_sign_positive();
                    signs.1 |= v.is_sign_negative();
                    endpoints.0 |= bits & 0x7fff_ffff == 0x0000_0001;
                    endpoints.1 |= bits & 0x7fff_ffff == 0x007f_ffff;
                }
                MultClass::Boundary => {
                    assert_eq!(v.abs(), f32::MIN_POSITIVE, "lane `{label}` drifted");
                }
                MultClass::Unit => assert_eq!(v, 1.0),
                MultClass::Zero => assert_eq!(bits, 0),
            }
            if v != 0.0 {
                let prod = f32::from_bits(gemm_expected_bits(FALLBACK, v, false));
                assert!(
                    prod.is_normal(),
                    "lane `{label}`: IEEE product {prod:e} is not normal — the \
                     lane would not discriminate"
                );
            }
        }
        assert!(signs.0 && signs.1, "need both subnormal signs");
        assert!(endpoints.0 && endpoints.1, "need min AND max subnormal");
        // Both boundary signs present.
        assert!(SUBNORMAL_MULTIPLIERS
            .iter()
            .any(|&(b, c, _)| c == MultClass::Boundary && b == 0x0080_0000));
        assert!(SUBNORMAL_MULTIPLIERS
            .iter()
            .any(|&(b, c, _)| c == MultClass::Boundary && b == 0x8080_0000));
    }

    /// The two admissible hardware shapes classify as themselves:
    /// a conformant adapter keeping every word is IMMUNE, and a pure-flush
    /// adapter dropping words exactly on the strictly-subnormal lanes is
    /// ANNIHILATES-WITHIN-SUBNORMAL.
    #[test]
    fn classifier_admits_the_two_modeled_shapes() {
        assert_eq!(
            classify_subnormal_mult(FlushClass::Conformant, &model_measurement(false, true)),
            SubnormalMultVerdict::StructurallyImmune
        );
        assert_eq!(
            classify_subnormal_mult(FlushClass::PureFlush, &model_measurement(true, false)),
            SubnormalMultVerdict::AnnihilatesWithinSubnormal
        );
        // A pure-flush adapter whose compares read the UNFLUSHED register
        // (METAL_CMP_EXACT) keeps every word: structurally immune.
        assert_eq!(
            classify_subnormal_mult(FlushClass::PureFlush, &model_measurement(true, true)),
            SubnormalMultVerdict::StructurallyImmune
        );
    }

    /// Fail-closed pins, one refusing shape per name:
    /// * a word lost at the ±2^-126 NORMAL boundary — the guard's `< 2^-126`
    ///   refusal would no longer cover the annihilation domain;
    /// * a word lost beside a NON-flushed value — the lane-2 launder revived;
    /// * a stuck word on the exact-zero control — the ±inf collapse;
    /// * value bits diverging from the class model;
    /// * twin drift on a chain lane;
    /// * a non-conformant flush class.
    #[test]
    fn classifier_is_fail_closed_on_any_normal_domain_word_loss() {
        let boundary_idx = SUBNORMAL_MULTIPLIERS
            .iter()
            .position(|&(_, c, _)| c == MultClass::Boundary)
            .unwrap();
        let zero_idx = SUBNORMAL_MULTIPLIERS
            .iter()
            .position(|&(_, c, _)| c == MultClass::Zero)
            .unwrap();
        let sub_idx = SUBNORMAL_MULTIPLIERS
            .iter()
            .position(|&(_, c, _)| c == MultClass::Subnormal)
            .unwrap();

        // Boundary word loss, each sub-probe family.
        let mut m = model_measurement(true, false);
        m.gemm_word[boundary_idx] = false;
        assert_eq!(
            classify_subnormal_mult(FlushClass::PureFlush, &m),
            SubnormalMultVerdict::Hazardous,
            "a GEMM word lost at the normal boundary must refuse"
        );
        let mut m = model_measurement(true, false);
        m.chain_word_a[boundary_idx] = false;
        m.chain_word_e[boundary_idx] = false;
        assert_eq!(
            classify_subnormal_mult(FlushClass::PureFlush, &m),
            SubnormalMultVerdict::Hazardous,
            "an activation word lost at the normal boundary must refuse"
        );
        let mut m = model_measurement(true, false);
        m.rowor_kept[boundary_idx] = false;
        assert_eq!(
            classify_subnormal_mult(FlushClass::PureFlush, &m),
            SubnormalMultVerdict::Hazardous,
            "a row-OR word lost at the normal boundary must refuse"
        );

        // Word lost beside a NON-flushed value (conformant hardware dropping
        // a subnormal-lane word): the compare annihilated what the multiply
        // preserved — the launder shape.
        let mut m = model_measurement(false, true);
        m.gemm_word[sub_idx] = false;
        assert_eq!(
            classify_subnormal_mult(FlushClass::Conformant, &m),
            SubnormalMultVerdict::Hazardous,
            "a word lost beside a preserved value must refuse"
        );

        // Stuck word on the exact-zero control.
        let mut m = model_measurement(true, false);
        m.gemm_word[zero_idx] = true;
        assert_eq!(
            classify_subnormal_mult(FlushClass::PureFlush, &m),
            SubnormalMultVerdict::Hazardous
        );

        // Value drift against the class model.
        let mut m = model_measurement(true, false);
        m.gemm_value_bits[sub_idx] = 1.0f32.to_bits();
        assert_eq!(
            classify_subnormal_mult(FlushClass::PureFlush, &m),
            SubnormalMultVerdict::Hazardous
        );

        // Twin drift.
        let mut m = model_measurement(true, false);
        m.chain_twin_match[0] = false;
        assert_eq!(
            classify_subnormal_mult(FlushClass::PureFlush, &m),
            SubnormalMultVerdict::Hazardous
        );

        // Disagreeing chain word channels.
        let mut m = model_measurement(true, false);
        m.chain_word_e[zero_idx] = true;
        assert_eq!(
            classify_subnormal_mult(FlushClass::PureFlush, &m),
            SubnormalMultVerdict::Hazardous
        );

        // Non-conformant class refuses outright, even on a perfect table.
        assert_eq!(
            classify_subnormal_mult(FlushClass::NonConformant, &model_measurement(false, true)),
            SubnormalMultVerdict::Hazardous
        );
    }

    /// The value models the classifier compares against, pinned at the
    /// interesting points: the sentinel × subnormal products are NORMAL under
    /// IEEE (up to ~1.18e-28) and clean `+0` under the pure-flush model (the
    /// accumulator add canonicalizes `-0`), while the activation coefficient
    /// keeps the flushed slope's SIGNED zero.
    #[test]
    fn value_models_are_pinned() {
        // IEEE: 1e10 · 2^-149 and 1e10 · (2^-126 − 2^-149) are normal.
        let p_min = f32::from_bits(gemm_expected_bits(FALLBACK, f32::from_bits(1), false));
        assert!(p_min.is_normal() && p_min > 0.0);
        let p_max = f32::from_bits(gemm_expected_bits(
            FALLBACK,
            f32::from_bits(0x007f_ffff),
            false,
        ));
        assert!(p_max.is_normal());
        // Pure flush: both signs land on +0 after the accumulator add.
        assert_eq!(gemm_expected_bits(FALLBACK, f32::from_bits(1), true), 0);
        assert_eq!(
            gemm_expected_bits(FALLBACK, f32::from_bits(0x8000_0001), true),
            0
        );
        // Boundary is NOT flushed: identical bits under both models.
        let b = f32::MIN_POSITIVE;
        assert_eq!(
            gemm_expected_bits(FALLBACK, b, true),
            gemm_expected_bits(FALLBACK, b, false)
        );
        // Unit lane saturates to exactly the sentinel under both models.
        assert_eq!(gemm_expected_bits(FALLBACK, 1.0, true), FALLBACK.to_bits());
        // Activation: the flushed slope's signed zero survives `− 0.0`.
        assert_eq!(
            chain_coeff_expected_bits(f32::from_bits(0x8000_0001), true),
            (-0.0f32).to_bits()
        );
        assert_eq!(chain_coeff_expected_bits(f32::from_bits(1), true), 0);
        assert_eq!(
            chain_coeff_expected_bits(1.0, true),
            FALLBACK.to_bits(),
            "unit slope carries the sentinel through unchanged"
        );
    }

    /// #flush-charge §H — the measurement CAPTURED LIVE on Apple M5 Max /
    /// Metal (2026-08-13, `report_subnormal_mult_taint_lanes`). This is the
    /// hardware the charged mode exists for, so it is pinned as a regression
    /// oracle exactly like `subnormal_selfcheck`'s `MEASURED_M5_MAX_LANES`:
    /// the capture must equal the pure-flush + compare-DAZ model bit-for-bit
    /// (`model_measurement(true, false)` — annihilation confined to the
    /// strictly-subnormal lanes, boundary sticky) and classify
    /// ANNIHILATES-WITHIN-SUBNORMAL, so driver/toolchain drift that moves the
    /// annihilation domain stops classifying and REFUSES instead of silently
    /// out-running the guard's refusal predicate.
    #[test]
    fn measured_m5_subnormal_mult_capture_matches_the_cmp_daz_model() {
        let pinned = model_measurement(true, false);
        assert_eq!(
            classify_subnormal_mult(FlushClass::PureFlush, &pinned),
            SubnormalMultVerdict::AnnihilatesWithinSubnormal,
            "the pinned M5 Max capture must classify ANNIHILATES-WITHIN-SUBNORMAL"
        );
        // Non-vacuity: the capture really does annihilate (all 4 subnormal
        // lanes, in all three families) and really does keep the boundary.
        for (i, &(_, class, label)) in SUBNORMAL_MULTIPLIERS.iter().enumerate() {
            match class {
                MultClass::Subnormal => {
                    assert!(!pinned.gemm_word[i], "gemm A-side `{label}` kept?");
                    assert!(!pinned.gemm_word[NM + i], "gemm B-side `{label}` kept?");
                    assert!(!pinned.chain_word_a[i], "chain `{label}` kept?");
                    assert!(!pinned.rowor_kept[i], "row-OR `{label}` kept?");
                }
                MultClass::Boundary | MultClass::Unit => {
                    assert!(pinned.gemm_word[i] && pinned.gemm_word[NM + i]);
                    assert!(pinned.chain_word_a[i] && pinned.rowor_kept[i]);
                }
                MultClass::Zero => {
                    assert!(!pinned.gemm_word[i] && !pinned.gemm_word[NM + i]);
                    assert!(!pinned.chain_word_a[i] && !pinned.rowor_kept[i]);
                }
            }
        }
    }
}

/// #flush-charge §H GPU tests: the LIVE measurement on this adapter.
#[cfg(all(test, feature = "gpu-tests"))]
mod h_gpu_tests {
    use super::*;
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};

    /// MEASUREMENT: run the §H probe live and print every lane, so the
    /// annihilation-domain evidence is a test output rather than a claim.
    ///
    /// Asserts only what must hold on ANY hardware: the unit controls carry
    /// their words (non-vacuity), the exact-zero controls clear them, the
    /// gate agrees with the classifier, and — when this box measures the
    /// pinned M5 Max pure-flush capture — the measurement equals the pinned
    /// compare-DAZ model bit-for-bit.
    #[test]
    fn report_subnormal_mult_taint_lanes() {
        let _serial = gpu_test_serial_guard();
        let device = require_device();
        let (class, meas, verdict) = device
            .subnormal_mult_report()
            .expect("the §H probe must be able to run");

        println!(
            "[#flush-charge §H subnormal-multiplier probe] adapter={} backend={:?} \
             flush_class={class:?}",
            device.adapter_info.name, device.adapter_info.backend
        );
        for (j, &(bits, mclass, label)) in SUBNORMAL_MULTIPLIERS.iter().enumerate() {
            println!(
                "  [{mclass:?}] {label:<28} mult={:e}\n      gemm word-on-A: value={:e} \
                 (bits 0x{:08x}) word={}\n      gemm word-on-B: value={:e} (bits 0x{:08x}) \
                 word={}\n      chain slope:    coeff={:e} (bits 0x{:08x}) word_a={} \
                 word_e={} twin_match={}\n      row-OR partner: kept={}",
                f32::from_bits(bits),
                f32::from_bits(meas.gemm_value_bits[j]),
                meas.gemm_value_bits[j],
                meas.gemm_word[j],
                f32::from_bits(meas.gemm_value_bits[NM + j]),
                meas.gemm_value_bits[NM + j],
                meas.gemm_word[NM + j],
                f32::from_bits(meas.chain_coeff_bits[j]),
                meas.chain_coeff_bits[j],
                meas.chain_word_a[j],
                meas.chain_word_e[j],
                meas.chain_twin_match[j],
                meas.rowor_kept[j],
            );
        }
        println!(
            "  => verdict = {verdict:?}; verify_subnormal_mult_taint() = {}",
            device.verify_subnormal_mult_taint()
        );

        // Non-vacuity on ANY hardware: unit controls sticky, zero controls clear.
        let unit = SUBNORMAL_MULTIPLIERS
            .iter()
            .position(|&(_, c, _)| c == MultClass::Unit)
            .unwrap();
        let zero = SUBNORMAL_MULTIPLIERS
            .iter()
            .position(|&(_, c, _)| c == MultClass::Zero)
            .unwrap();
        assert!(
            meas.gemm_word[unit]
                && meas.gemm_word[NM + unit]
                && meas.chain_word_a[unit]
                && meas.rowor_kept[unit],
            "§H probe is MISCONFIGURED: a unit-control word did not arrive"
        );
        assert!(
            !meas.gemm_word[zero]
                && !meas.gemm_word[NM + zero]
                && !meas.chain_word_a[zero]
                && !meas.rowor_kept[zero],
            "§H probe: an exact-zero control kept a word"
        );

        // The cached gate must agree with the classifier.
        assert_eq!(
            device.verify_subnormal_mult_taint(),
            verdict != SubnormalMultVerdict::Hazardous,
            "the §H gate must agree with the classifier"
        );

        // On the charged-mode target hardware the capture must be exactly the
        // pinned compare-DAZ model. Only asserted when this box measures
        // PURE-FLUSH and the annihilating shape (other adapters stay
        // measurement-only).
        if class == FlushClass::PureFlush
            && verdict == SubnormalMultVerdict::AnnihilatesWithinSubnormal
        {
            assert_eq!(
                meas,
                model_measurement(true, false),
                "a pure-flush adapter that annihilates must match the \
                 compare-DAZ model bit-for-bit (M5 Max pinned capture)"
            );
        }
    }

    /// Fail-closed: the shared rung-5 forced-fail hook closes the §H gate
    /// too, and with it charged-flush authority (which reads the same hook).
    #[test]
    fn forced_failure_refuses_the_subnormal_mult_probe() {
        let _serial = gpu_test_serial_guard();
        let device = require_device();
        // Prime the real result first so the hook check below cannot be
        // satisfied by an uninitialized cache.
        let real = device.verify_subnormal_mult_taint();
        set_force_sentinel_taint_selfcheck_fail(true);
        assert!(
            !device.verify_subnormal_mult_taint(),
            "forced rung-5 failure must close the §H gate"
        );
        set_force_sentinel_taint_selfcheck_fail(false);
        assert_eq!(
            device.verify_subnormal_mult_taint(),
            real,
            "cache must survive the forced-fail hook"
        );
    }
}
