// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! THE single predicate that decides whether a wgpu GPU result may carry
//! VERDICT authority. The reviewed source gate is open after the B0 review,
//! but authority still requires an explicit typed request and a passing
//! per-device diagnostic ladder. Authority is stored on the exact device that
//! ran the probes; process environment cannot arm an ordinary device.
//!
//! # What this replaces
//!
//! Authority used to be a hardwired `false` literal in
//! `impl GpuCrownBackward for WgpuDevice` (plus a second, independent `None` in
//! `impl GemmEngine`). The per-adapter probe machinery remains useful evidence,
//! but passing probes are necessary rather than sufficient: ambient process
//! state must never turn incomplete evidence into a production proof claim.
//! [`PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED`] was therefore held at
//! source-level `false` until U1/U3/U4/U5/U6 were discharged. The reviewed
//! 2026-08-11 UTC B0 source change deliberately opened it. There is no
//! `Backend::Metal` branch here, and there must never be one.
//!
//! # The diagnostic ladder (ALL must hold; the report evaluates every rung and
//! fails closed)
//!
//! 0. [`WgpuVerdictRequest`] — a typed, explicit request consumed by
//!    [`WgpuDevice::new_for_verdict`]. [`WgpuDevice::new`] never evaluates this
//!    ladder and never gains verdict authority.
//! 1. [`WgpuDevice::verify_ieee_f32_model`] (`ops/f32_selfcheck.rs`) — the base
//!    IEEE-754 f32 model: no covert reduced precision, bit-exact
//!    `bitcast`-based directed rounding, no gross fast-math deviation.
//! 2. [`WgpuDevice::verify_eft_primitives`] (`ops/eft_selfcheck.rs`) — fma
//!    TwoProduct and the fma-barrier TwoSum match the normal-range probe table
//!    bit-exactly. Together with rung 3's subnormal policy/floor, that makes the
//!    EFT residual channel a valid certified error term. **Rung 2 now ENTAILS
//!    rung 3** (see below); it is still listed and evaluated separately so that
//!    the ladder report stays diagnosable rung by rung.
//! 3. [`WgpuDevice::verify_gradual_underflow`] (`ops/subnormal_selfcheck.rs`) —
//!    core add/multiply subnormal operands/results are preserved. Primary EFT
//!    products use that qualified multiply because the measured direct-FMA
//!    form DAZ-zeroes subnormal inputs. An exact-zero flush of a subnormal
//!    FMA-derived residual is accepted only because
//!    every shipped EFT dispatch charges the covering
//!    `rung3_flush_safe_additive` base, scaled outward by the same downstream
//!    residual-recovery multiplier. This is a PRECONDITION of rung 2's
//!    residual identity that rung 2's own normal-range probe cannot establish.
//! 4. [`ny_core::eft::eft_available`] — the HOST reference is itself conformant.
//!    Without it the bit-comparisons in rungs 2-3 are meaningless.
//! 5. [`WgpuDevice::verify_sentinel_taint_sticky`]
//!    (`ops/sentinel_taint_selfcheck.rs`) — **`#u4`**: the finite
//!    `±FALLBACK_BOUND` overflow sentinel survives every fused resident op
//!    whose real multiplicative partner is nonzero, including cancelling
//!    additions. A clean exactly-zero partner may soundly annihilate the word
//!    because the sentinel represents a finite real. Unlike rungs 1-3 this is
//!    not purely an adapter-conformance question: it also depends on NY's own
//!    kernel design. The armed out-of-band word channel now makes this rung pass
//!    on the measured GB10/Vulkan + DenormPreserve path (see the module docs and
//!    `report_sentinel_taint_lanes`). It remains a rung, rather than a comment,
//!    so any transport or consultation regression closes the ladder.
//!
//! # `#u2b` — rungs 2 and 3 can no longer disagree in the unsafe direction
//!
//! Listing rung 3 next to rung 2 protected the LADDER but not the CHANNEL. The
//! compensated EFT channel has its own, separate entry point: the two
//! production sites in `crown_backward_sound_resident.rs` and
//! `crown_concretize_sound.rs` switch on `NY_EFT_ERR=1 &&
//! eft_primitives_cached()` and never consult this ladder. So on an adapter
//! with rung2 = `true`, rung3 = `false` — measured, on Apple M5 Max/Metal —
//! verdict qualification correctly refused the VERDICT while the compensated
//! TIGHTENING was still authorized and firing on hardware that silently zeroes
//! the residuals the tightening measures.
//!
//! `verify_eft_primitives()` (and `eft_primitives_cached()`) therefore now
//! entail `verify_gradual_underflow()` (`gradual_underflow_cached()`). The
//! unsafe state `rung2 ∧ ¬rung3` is unrepresentable, which is pinned by
//! [`gpu_tests::report_ladder_and_pin_conjunction`] on every run. The safe
//! disagreement `¬rung2 ∧ rung3` (the normal-range EFT probes fail even though
//! the subnormal policy passes) is still representable and still refuses.
//!
//! # What a PASSING ladder does and does not license
//!
//! Passing is NECESSARY, NOT SUFFICIENT: the caller must consume an explicit
//! typed request and the reviewed source gate must be open. The probes establish
//! adapter CONFORMANCE;
//! they do not by themselves establish that NY's production kernels compose
//! correctly. That separate proof was tracked in the obligation ledger below.
//! Every entry was discharged before the 2026-08-11 UTC B0 source opening:
//!
//! * **U1 — composed-sequence integrity in the PRODUCTION kernels.**
//!   DISCHARGED 2026-08-10 America/Los_Angeles (2026-08-11 UTC) for the tiled
//!   GEMM (0-ULP bit-compare over
//!   ~48M taps) and the tree kernels (`tests/u1_tree_settling.rs`, 10 device
//!   tests: composed per-element bit-compare of Bias/ActBias/Concretize/
//!   col2im `(V, R)` against a CPU twin at CROWN shapes, plus chain/pair/tail
//!   isolation tiers). Two measured GB10/Vulkan findings, both
//!   enclosure-sound and pinned as ACCEPTED hypotheses: the driver
//!   fma-contracts the thread-0 error tails (eft AND legacy modes) — fewer
//!   roundings than the slack was sized for. The settling suite is the
//!   regression oracle for any kernel or toolchain change.
//! * **U3 — `eft_r_slack_f32` term count vs the tree-reduction kernels.**
//!   DISCHARGED 2026-08-06: the slack now charges the exact
//!   `2k + 2 + TREE_REDUCTION_RESIDUAL_ADDS` term reduction
//!   (`sound_consts.rs`, the counted 2-adds-per-level tree residue).
//! * **U4 — finite overflow-sentinel transport and consultation.**
//!   DISCHARGED for the admitted routes by the 2026-08-11 UTC arming review.
//!   The default AUTO Linear/Activation resident route dispatches the taint
//!   twins, carries words on-device, folds them into per-row words, and supplies
//!   them to fail-closed C1. C2 consults the word before the only error-lowering
//!   EFT min-combine. Admitted host and host-per-segment Linear/Activation
//!   routes perform G13 sweeps and carry the row words through composition.
//!   Resident Conv is admitted through exact-value reshape / scheduled GEMM /
//!   col2im twins carrying the word through the complete fused interior. Host
//!   Conv and segment-resident device seed/keep streams refuse until they carry
//!   equivalent word state. Unsupported configurations typed-refuse instead of
//!   dropping state;
//!   `NY_GPU_TAINT_WORDS=0` is therefore not an authority escape because C1
//!   refuses absent words. The GB10 diagnostic ladder measured rung 5 passing.
//! * **U5 — the Lipschitz propagation swap bundled under `NY_EFT_ERR`** (an
//!   a-priori claim, not an EFT measurement: `|sel|` / `max(|ls|,|us|)`
//!   replaces `|ls|+|us|` for coefficients AND intercepts).
//!   DISCHARGED 2026-08-10 by dedicated adversarial worst-realization device
//!   oracles (`crown_backward_sound_resident::u5_activation_lipschitz::`
//!   `act_eft_err_encloses_worst_realization` +
//!   `act_intercept_bias_eft_err_encloses_worst_realization`): exact-f64 sup
//!   of `|g(a′) − coeff|` over every realization `a′ ∈ [a−err, a+err]` of the
//!   piecewise-linear map `g(t) = t·sel(t) ∓ β` (continuous kink at 0 ⇒ the
//!   sup is at an interval end or at 0), asserted `≤` the published error for
//!   BOTH `eft_mode` settings across mixed-sign 2^-30..2^8 bands, slopes in
//!   [0,1] incl. exact 0/1, the β≠0 arm, sign-uncertain err bands, and
//!   2^-126 subnormal edges; the VALUE lane is pinned bit-identical across
//!   modes (the swap touches only the error channel).
//! * **U6 — concretize is not value-neutral in EFT mode** (by design: the
//!   barrier-fma sequence + measured-residual charge replaces the one-shot
//!   fold + γ_n charge). DISCHARGED 2026-08-10
//!   (`crown_concretize_sound::tests::`
//!   `u6_concretize_eft_vs_legacy_enclose_and_fail_closed_identity`): both
//!   modes independently enclose the exact-f64 corner oracle (enclosure is
//!   the assert — equality is never asserted), the modes' bits differ when
//!   the lane is authorized, and the FAIL-CLOSED IDENTITY holds — with the
//!   lane refused (forced per-call primitive failure) `NY_EFT_ERR=1` output
//!   is BIT-IDENTICAL to legacy, and the armed C1 taint consult refuses a
//!   tainted row in EFT mode exactly as in legacy mode.
//!
//!   Reviewer-noted residues accepted by the B0 authority review, neither a
//!   U5/U6 defect: (a) the tree kernels' propagated `se = Σ err·factor` lane
//!   accumulates in plain f32 with no recovery slack of its own
//!   (~γ_depth ≈ 6.5e-7 relative under-report ceiling, masked in practice by
//!   the γ_k / strict-Lipschitz margins — the elementwise kernels' ×SLACK
//!   recovers theirs); (b) `CROWN_EFT_MIN_COMBINE_SHADER`'s min-of-two-sound-
//!   bounds claim rests on its own doc argument plus the C2 taint probes —
//!   it is not exercised by the U6 oracle (which validates the concretize
//!   shader's own EFT branch, as the obligation is worded).
//!
//! With every listed obligation discharged, the B0 review opened
//! [`PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED`]. The only production opt-in is
//! now [`WgpuVerdictRequest`]; it cannot bypass the source gate or any
//! per-device ladder rung.
//!
//! # Fail-closed discipline
//!
//! Every rung returns `false` on mismatch, dispatch fault, readback error, or
//! an uninitialized cache. There is no path by which an error becomes a grant.
//! The forced-fail hooks (`NY_FORCE_GPU_F32_SELFCHECK_FAIL`,
//! `NY_FORCE_GPU_EFT_SELFCHECK_FAIL`, `NY_FORCE_GPU_SUBNORMAL_SELFCHECK_FAIL`,
//! `NY_FORCE_GPU_SENTINEL_TAINT_SELFCHECK_FAIL`) only ever force MORE closed,
//! and each is honoured here because the rung functions consult them internally
//! on every call.

use super::super::WgpuDevice;
use super::subnormal_selfcheck::FlushClass;
use ny_core::NyError;

/// Reviewed production source gate. Environment variables and runtime probe
/// results cannot change this value.
///
/// The 2026-08-11 UTC B0 review opened the gate after discharging the module's
/// safety-case obligations. Opening or closing it is a source-review event.
pub(crate) const PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED: bool = true;

/// #flush-charge: reviewed source gate for the CHARGED-flush authority state
/// (`QualifiedWithFlushCharge`) — a pure-flush adapter (Apple M5 Max / Metal
/// class: rung 3 refuses, every refused lane is a modeled ±0 flush) whose
/// flush losses are paid by CERTIFIED ADDITIVE widening instead of refused.
///
/// OPEN (`true`) since the 2026-08-13 source review: every charge-policy
/// audit on this ledger landed in `flush_charge_oracle.rs` and the walk
/// guards before the flip (each row retains its landing evidence):
///
/// * §D bias-combine transcription — LANDED (`charged_bias_combine_*` oracle
///   tests derive `CHARGED_BIAS_COMBINE_SLACK_FACTOR`).
/// * §F concretize legacy-branch transcription — LANDED
///   (`charged_concretize_*` oracle tests derive
///   `CHARGED_CONCRETIZE_SLACK_FACTOR`).
/// * §E activation intercept-bias shader audit — LANDED (`charged_act_bias_*`
///   oracle tests derive `CHARGED_ACT_BIAS_SLACK_FACTOR`; nonzero NORMAL
///   intercepts are re-admitted under the charge, subnormal intercepts stay
///   refused with the negative pin
///   `charged_act_bias_cannot_cover_subnormal_intercepts`).
/// * §H rung-5 subnormal-multiplier taint decision — RESOLVED 2026-08-13
///   (probe-lane evidence: the §H section of `ops/sentinel_taint_selfcheck.rs`,
///   `report_subnormal_mult_taint_lanes` live +
///   `measured_m5_subnormal_mult_capture_matches_the_cmp_daz_model` pinned).
///   Measured on the Apple M5 Max target: every strictly-subnormal multiplier
///   ANNIHILATES the taint word beside a clean flushed ±0 in all three
///   `!= 0`-conjunct families (GEMM both slots, activation slopes, row-OR
///   partner) while both ±2^-126 boundary lanes stay sticky — the compare
///   DAZ-flushes with the multiply, the annihilation domain equals the guard's
///   refusal domain, and v1's subnormal-weight/bias/slope refusals are
///   load-bearing AND sufficient. The finding is enforced per-device:
///   `verify_subnormal_mult_taint` is a charged-authority conjunct
///   (`charged_flush_authority_cached` + the charged constructor), refusing
///   any adapter whose annihilation domain leaks past the strictly-subnormal
///   refusal predicate.
/// * end-to-end enclosure-parity acceptance run on the charged device —
///   LANDED (Lane B harness 2026-08-13; row CLOSED by the 2026-08-13
///   opening review):
///   `wgpu_device/flush_charge_acceptance_gpu_tests.rs` runs the charged walk
///   on a TEST-SCOPED device (production admission minus only this gate;
///   compiled out of production builds) and asserts the charged GPU enclosure
///   CONTAINS the CPU f64 same-fold reference elementwise, tolerance zero —
///   measured green on the Apple M5 Max target (210 seeded random fixtures +
///   24 admitted-boundary fixtures, 610 spec rows, 0 breaches) under the
///   DEFAULT env (re-measured 2026-08-13 after admission-config landed).
///   ADMISSION-CONFIG (resolved 2026-08-13): the charged
///   constructors now build their device with the DenormPreserve policy
///   FORCED to the plain-WGSL (flushing) loading path — the configuration
///   the oracle charges model — so the probes measure it and admission holds
///   on Metal under the DEFAULT env, no `NY_GPU_DENORM_PRESERVE=0` required
///   (the earlier finding that the AUTO passthrough fallback poisons the
///   process no longer reaches the charged device: its loading path is
///   threaded per-device and never requests passthrough). An explicit
///   `NY_GPU_DENORM_PRESERVE=1` pin typed-refuses (env wins).
///   The 2026-08-13 opening review closed this row on that evidence; any
///   wall-clock claim stays HONESTLY LABELED (the REAL metaroom net is
///   REFUSED by the v1 charge policy's subnormal-bias guard — only the
///   admissible variant is measured).
///
/// The 2026-08-13 source review OPENED the gate with the same discipline as
/// [`PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED`]: every ledger row above was
/// discharged (§D/§F/§E LANDED, §H RESOLVED 2026-08-13, enclosure-parity
/// acceptance CLOSED 2026-08-13), the const-block pin in `cpu_tests` was
/// inverted by the review, and the flip was validated live on the Apple
/// M5 Max target (charged admission: rungs 1/4/5 pass, rung 3 refused,
/// PURE-FLUSH, §H clean, on the forced plain-WGSL device). Closing it again
/// is itself a source-review event and must re-invert the pin. NO rung of
/// the uncharged ladder moved with this opening: rung 3's refusal of
/// UNcharged authority is byte-identical either way.
pub(crate) const PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED: bool = true;

/// #flush-charge: the per-op charge model a charged-flush device carries. Every
/// field is either a WIDENING factor (multiplies a certified additive/slack —
/// downgrade-only, can only loosen enclosures) or a REFUSAL (closes a channel
/// whose loss no uniform bounds). All constants are sourced from named
/// exact-rational oracle tests in `flush_charge_oracle.rs` — never guessed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlushChargePolicy {
    /// `‖w‖₁` multiplier for the AW-error combines' operand-flush cover
    /// (`#daz-flush-cover-v2`, `derived_cover_plus_subnormal_weight_refusal_encloses`).
    pub(crate) w_l1_factor: f32,
    /// Concretize legacy-branch `slack` multiplier
    /// (oracle: `charged_concretize_factor_covers_the_multi_channel_daz_demand`).
    pub(crate) concretize_slack_factor: f32,
    /// Bias-combine `slack` multiplier
    /// (oracle: `charged_bias_combine_factor_covers_the_double_daz_demand`).
    pub(crate) bias_combine_factor: f32,
    /// Activation intercept-bias `slack` multiplier (`ActBiasParams.slack`,
    /// oracle: `charged_act_bias_factor_covers_the_double_daz_demand`; the
    /// subnormal-intercept channel stays refused —
    /// `charged_act_bias_cannot_cover_subnormal_intercepts`).
    pub(crate) act_bias_slack_factor: f32,
    /// Refuse layers whose weight tensor contains a subnormal: the
    /// `μ·‖err_i‖₁` loss of the `prop = fl(err@|W|)` GEMM has no covering
    /// uniform (`shipped_flush_cover_does_not_enclose_under_daz`, channel 1).
    pub(crate) refuse_subnormal_weights: bool,
    /// Refuse subnormal bias entries: the `err·|b|` channel loses `err·2^-126`
    /// when `b` is DAZ-zeroed and nothing in the bias combine scales with err.
    pub(crate) refuse_subnormal_bias: bool,
    /// Refuse subnormal activation slopes: closes both the elementwise
    /// `μ·|a|` amplification channel and the rung-5 subnormal-multiplier
    /// annihilation hazard (`b != 0` under a DAZ compare) in one predicate.
    /// The §H probe measured that hazard REAL on the M5 Max (words annihilate
    /// beside flushed ±0) and CONFINED to the strictly-subnormal domain this
    /// predicate refuses (`verify_subnormal_mult_taint`, a charged-authority
    /// conjunct).
    pub(crate) refuse_subnormal_slopes: bool,
    /// Refuse subnormal input-box endpoints: a DAZ-zeroed `x` bound under a
    /// large accumulated coefficient error `e` loses `e·2^-126` in the
    /// concretize penalty dot, and no flushacc term scales with `e`.
    pub(crate) refuse_subnormal_inputs: bool,
    /// The EFT compensated channel is FORBIDDEN on a flushing adapter (it
    /// measures the residuals the hardware zeroes). `eft_primitives_cached()`
    /// is false there by composition; this pins the intent so a future edit
    /// cannot re-open the channel.
    pub(crate) eft_forbidden: bool,
}

impl FlushChargePolicy {
    /// The single production policy. Constants live in `sound_consts` next to
    /// the shipped cover they widen and are pinned by the oracle tests.
    #[must_use]
    pub(crate) const fn production() -> Self {
        Self {
            w_l1_factor: crate::wgpu_device::sound_consts::CHARGED_W_L1_FACTOR,
            concretize_slack_factor:
                crate::wgpu_device::sound_consts::CHARGED_CONCRETIZE_SLACK_FACTOR,
            bias_combine_factor:
                crate::wgpu_device::sound_consts::CHARGED_BIAS_COMBINE_SLACK_FACTOR,
            act_bias_slack_factor: crate::wgpu_device::sound_consts::CHARGED_ACT_BIAS_SLACK_FACTOR,
            refuse_subnormal_weights: true,
            refuse_subnormal_bias: true,
            refuse_subnormal_slopes: true,
            refuse_subnormal_inputs: true,
            eft_forbidden: true,
        }
    }
}

/// #flush-charge: the authority state a device's qualification attempt landed
/// in. Diagnostic surface only — the production predicates read the stored
/// report/policy directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WgpuVerdictAuthority {
    /// All five rungs passed: full uncharged authority.
    Qualified,
    /// Rung 3 refused but the adapter is PURE-FLUSH and the charged
    /// constructor armed the charge policy.
    QualifiedWithFlushCharge(FlushChargePolicy),
    /// No verdict authority of either kind.
    Refused,
}

/// Explicit capability request for constructing a verdict-qualified WGPU
/// device.
///
/// The private field prevents accidental struct-literal construction and the
/// absence of [`Default`] keeps the authority transition visible at the call
/// site. This value carries no authority by itself: it is consumed exactly once
/// by [`WgpuDevice::new_for_verdict`], which runs the live ladder on the device
/// it will return.
#[derive(Debug, PartialEq, Eq)]
pub struct WgpuVerdictRequest {
    _explicit: (),
}

// A `Default` implementation would erase the deliberate authority request at
// call sites; construction must remain visibly explicit.
#[allow(clippy::new_without_default)]
impl WgpuVerdictRequest {
    /// Make an explicit request to qualify one new WGPU device for verdict use.
    #[must_use]
    pub const fn new() -> Self {
        Self { _explicit: () }
    }
}

/// #flush-charge: explicit capability request for constructing a CHARGED-flush
/// verdict device. A SEPARATE type from [`WgpuVerdictRequest`] so no existing
/// call site can drift into charged mode: reaching the charged constructor
/// requires naming this type at the call site.
#[derive(Debug, PartialEq, Eq)]
pub struct WgpuChargedVerdictRequest {
    _explicit: (),
}

#[allow(clippy::new_without_default)]
impl WgpuChargedVerdictRequest {
    /// Make an explicit request to qualify one new WGPU device for
    /// charged-flush verdict use.
    #[must_use]
    pub const fn new() -> Self {
        Self { _explicit: () }
    }
}

/// One of the five independently reported WGPU verdict-qualification rungs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuVerdictRung {
    /// IEEE-754 f32 execution and directed-rounding primitives.
    IeeeF32Model,
    /// Error-free-transform primitives, including their underflow dependency.
    EftPrimitives,
    /// Gradual-underflow policy and the charged flush floor.
    GradualUnderflow,
    /// Conformance of the host EFT reference used by the probes.
    HostEftReference,
    /// End-to-end finite-sentinel taint-word transport and consultation.
    SentinelTaintSticky,
}

impl std::fmt::Display for WgpuVerdictRung {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::IeeeF32Model => "IEEE f32 model",
            Self::EftPrimitives => "EFT primitives",
            Self::GradualUnderflow => "gradual underflow",
            Self::HostEftReference => "host EFT reference",
            Self::SentinelTaintSticky => "sentinel-taint stickiness",
        })
    }
}

/// Outcome of one qualification rung.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuVerdictRungOutcome {
    /// The rung ran and passed.
    Passed,
    /// The rung ran and refused authority.
    Failed,
    /// Device initialization failed before this rung could run.
    NotRun,
}

impl WgpuVerdictRungOutcome {
    #[must_use]
    const fn from_bool(passed: bool) -> Self {
        if passed {
            Self::Passed
        } else {
            Self::Failed
        }
    }

    /// Whether this rung ran and passed.
    #[must_use]
    pub const fn passed(self) -> bool {
        matches!(self, Self::Passed)
    }
}

/// Immutable evidence report for one device's verdict qualification attempt.
///
/// Reports name all five rung outcomes even when initialization prevented the
/// ladder from running. A successful report is stored on the exact
/// [`WgpuDevice`] that produced it; it cannot be transferred to another device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuVerdictReport {
    adapter: Option<String>,
    source_gate_open: bool,
    ieee_f32_model: WgpuVerdictRungOutcome,
    eft_primitives: WgpuVerdictRungOutcome,
    gradual_underflow: WgpuVerdictRungOutcome,
    host_eft_reference: WgpuVerdictRungOutcome,
    sentinel_taint_sticky: WgpuVerdictRungOutcome,
    /// #flush-charge: measured flush class, recorded only when a charged
    /// qualification attempt ran the characterization. `None` on the ordinary
    /// path (which never pays the extra dispatch).
    flush_class: Option<FlushClass>,
}

impl WgpuVerdictReport {
    fn not_run() -> Self {
        Self {
            adapter: None,
            source_gate_open: PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED,
            ieee_f32_model: WgpuVerdictRungOutcome::NotRun,
            eft_primitives: WgpuVerdictRungOutcome::NotRun,
            gradual_underflow: WgpuVerdictRungOutcome::NotRun,
            host_eft_reference: WgpuVerdictRungOutcome::NotRun,
            sentinel_taint_sticky: WgpuVerdictRungOutcome::NotRun,
            flush_class: None,
        }
    }

    #[cfg(test)]
    fn from_outcomes(source_gate_open: bool, outcomes: [WgpuVerdictRungOutcome; 5]) -> Self {
        Self {
            adapter: Some("test-adapter".to_string()),
            source_gate_open,
            ieee_f32_model: outcomes[0],
            eft_primitives: outcomes[1],
            gradual_underflow: outcomes[2],
            host_eft_reference: outcomes[3],
            sentinel_taint_sticky: outcomes[4],
            flush_class: None,
        }
    }

    /// Stable adapter identity associated with this report, or `None` when
    /// device initialization failed before an adapter could be qualified.
    #[must_use]
    pub fn adapter(&self) -> Option<&str> {
        self.adapter.as_deref()
    }

    /// Whether the reviewed source-level authority gate was open.
    #[must_use]
    pub const fn source_gate_open(&self) -> bool {
        self.source_gate_open
    }

    /// #flush-charge: measured flush class, when a charged qualification
    /// attempt characterized this adapter. `None` on the ordinary path.
    /// (Diagnostic surface for the charged constructor's callers and tests;
    /// production predicates read the stored policy, never this.)
    #[must_use]
    #[allow(dead_code)]
    pub(crate) const fn flush_class(&self) -> Option<FlushClass> {
        self.flush_class
    }

    /// Return the recorded outcome for a named rung.
    #[must_use]
    pub const fn outcome(&self, rung: WgpuVerdictRung) -> WgpuVerdictRungOutcome {
        match rung {
            WgpuVerdictRung::IeeeF32Model => self.ieee_f32_model,
            WgpuVerdictRung::EftPrimitives => self.eft_primitives,
            WgpuVerdictRung::GradualUnderflow => self.gradual_underflow,
            WgpuVerdictRung::HostEftReference => self.host_eft_reference,
            WgpuVerdictRung::SentinelTaintSticky => self.sentinel_taint_sticky,
        }
    }

    /// `true` only when the source gate and all five live rungs passed.
    #[must_use]
    pub const fn qualified(&self) -> bool {
        self.source_gate_open
            && self.ieee_f32_model.passed()
            && self.eft_primitives.passed()
            && self.gradual_underflow.passed()
            && self.host_eft_reference.passed()
            && self.sentinel_taint_sticky.passed()
    }

    /// First rung that did not pass, in the reviewed ladder order.
    ///
    /// `NotRun` counts as non-passing after a device exists. A report whose
    /// adapter is absent represents device initialization failure rather than
    /// a failed probe rung and therefore returns `None`.
    #[must_use]
    pub const fn failed_rung(&self) -> Option<WgpuVerdictRung> {
        if self.adapter.is_none() {
            return None;
        }
        if !self.ieee_f32_model.passed() {
            Some(WgpuVerdictRung::IeeeF32Model)
        } else if !self.eft_primitives.passed() {
            Some(WgpuVerdictRung::EftPrimitives)
        } else if !self.gradual_underflow.passed() {
            Some(WgpuVerdictRung::GradualUnderflow)
        } else if !self.host_eft_reference.passed() {
            Some(WgpuVerdictRung::HostEftReference)
        } else if !self.sentinel_taint_sticky.passed() {
            Some(WgpuVerdictRung::SentinelTaintSticky)
        } else {
            None
        }
    }

    /// Deterministic summary suitable for routing diagnostics.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        if self.adapter.is_none() {
            return "WGPU device initialization did not complete";
        }
        if !self.source_gate_open {
            return "the reviewed WGPU verdict source gate is closed";
        }
        // #flush-charge: when the failure profile is EXACTLY the
        // charged-admissible one (rungs 1/4/5 pass, rung 3 fails — rung 2
        // failing alongside is the #u2b entailment, not extra evidence) and
        // the measured class is PURE-FLUSH, say so. This is the routing hint
        // the charged constructor's admission predicate acts on.
        let charged_shape = self.ieee_f32_model.passed()
            && self.host_eft_reference.passed()
            && self.sentinel_taint_sticky.passed()
            && !self.gradual_underflow.passed();
        if charged_shape && matches!(self.flush_class, Some(FlushClass::PureFlush)) {
            return "the gradual-underflow qualification rung did not pass; \
                    measured flush class PURE-FLUSH — admissible under \
                    charged-flush authority (not requested)";
        }
        match self.failed_rung() {
            Some(WgpuVerdictRung::IeeeF32Model) => {
                "the IEEE f32-model qualification rung did not pass"
            }
            Some(WgpuVerdictRung::EftPrimitives) => {
                "the EFT-primitives qualification rung did not pass"
            }
            Some(WgpuVerdictRung::GradualUnderflow) => {
                "the gradual-underflow qualification rung did not pass"
            }
            Some(WgpuVerdictRung::HostEftReference) => {
                "the host-EFT-reference qualification rung did not pass"
            }
            Some(WgpuVerdictRung::SentinelTaintSticky) => {
                "the sentinel-taint qualification rung did not pass"
            }
            None => "all WGPU verdict qualification requirements passed",
        }
    }
}

/// Typed, inspectable refusal from [`WgpuDevice::new_for_verdict`].
#[derive(Debug)]
pub struct WgpuVerdictQualificationError {
    report: WgpuVerdictReport,
    source: NyError,
}

impl WgpuVerdictQualificationError {
    fn initialization(source: NyError) -> Self {
        Self {
            report: WgpuVerdictReport::not_run(),
            source,
        }
    }

    fn rejected(report: WgpuVerdictReport) -> Self {
        let source = NyError::UnsupportedConfiguration(report.reason().to_string());
        Self { report, source }
    }

    /// #flush-charge: rejection whose reason is specific to the CHARGED
    /// admission predicate (the base report's reason alone cannot express it).
    fn rejected_charged(report: WgpuVerdictReport, reason: String) -> Self {
        let source = NyError::UnsupportedConfiguration(reason);
        Self { report, source }
    }

    /// Complete five-rung report for the failed attempt.
    #[must_use]
    pub const fn report(&self) -> &WgpuVerdictReport {
        &self.report
    }

    /// Original NY error. Device-initialization errors are preserved verbatim;
    /// probe refusals use a typed unsupported-configuration error whose reason
    /// matches the report.
    #[must_use]
    pub const fn source_error(&self) -> &NyError {
        &self.source
    }

    /// Consume this error and recover its complete report.
    #[must_use]
    pub fn into_report(self) -> WgpuVerdictReport {
        self.report
    }
}

impl std::fmt::Display for WgpuVerdictQualificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WGPU verdict qualification refused: {}: {}",
            self.report.reason(),
            self.source
        )
    }
}

impl std::error::Error for WgpuVerdictQualificationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl WgpuDevice {
    /// Construct and qualify one device for verdict-bearing GPU CROWN.
    ///
    /// Device creation and all five probes happen on this one context. Every
    /// rung is evaluated eagerly so a rejection carries a complete report. The
    /// report is stored only after the whole conjunction passes; every error
    /// leaves no armed device behind.
    pub fn new_for_verdict(
        _request: WgpuVerdictRequest,
    ) -> Result<Self, WgpuVerdictQualificationError> {
        let mut device = Self::new().map_err(WgpuVerdictQualificationError::initialization)?;
        device
            .materialize_verdict_critical_pipelines()
            .map_err(WgpuVerdictQualificationError::initialization)?;
        let report = device.evaluate_verdict_report();
        if !report.qualified() {
            tracing::info!(
                target: "ny_gpu::wgpu",
                adapter = report.adapter().unwrap_or("unknown"),
                reason = report.reason(),
                "explicit WGPU verdict qualification REFUSED (fail-closed)"
            );
            return Err(WgpuVerdictQualificationError::rejected(report));
        }

        // INFO, not WARN: this is the SUCCESS arm — the refusal above is the
        // warning. Logging a pass at WARN put it on stderr at default verbosity,
        // which broke `--json`'s empty-stderr contract (#395) on any machine
        // with a qualifying GPU, and it erodes the signal that a real WARN from
        // this target means the device was refused.
        tracing::info!(
            target: "ny_gpu::wgpu",
            adapter = report.adapter().unwrap_or("unknown"),
            "explicit WGPU verdict qualification PASSED: this device is armed for sound GPU CROWN"
        );
        device.verdict_report = Some(report);
        // Prewarm the resident cut-apply selfcheck so the observation-only
        // Cut-CROWN shadow capability is qualification-time evidence. Its
        // failure narrows ONLY `provides_resident_cut_shadow`; no verdict rung
        // reads it.
        let _ = device.verify_resident_cut_apply();
        Ok(device)
    }

    /// Compile every lazy pipeline that can participate in a verdict-bearing
    /// proof before the immutable qualification receipt is created. This keeps
    /// a requested DenormPreserve passthrough failure in constructor admission,
    /// rather than discovering it after the CLI has reported qualified WGPU
    /// execution. The in-walk authority recheck remains defense in depth.
    fn materialize_verdict_critical_pipelines(&self) -> ny_core::Result<()> {
        self.run_gpu_checked("materialize_verdict_critical_pipelines", || {
            let _ = self.resident_backward_pipelines();
            let _ = self.resident_strided_gather_pipeline();
            let _ = self.sound_concretize_pipeline();
            // The worded DAG route needs the same 11-storage-binding budget
            // as the resident taint twins. A smaller device may still qualify
            // for other sound routes; its intermediate sweep cleanly declines.
            if self.device.limits().max_storage_buffers_per_shader_stage >= 11 {
                let _ = self.intermediate_sweep_dag_pipelines();
            }
            let _ = self.ibp_sound_pipelines();
            // Joint-alpha pipelines steer optimization rather than establish
            // bounds, but compiling them here prevents a later loading-path
            // fallback from invalidating an already-emitted backend receipt.
            let _ = self.joint_adjoint_pipelines();
            Ok(())
        })?;
        if !self.denorm_preserve_contract_intact() {
            return Err(NyError::UnsupportedConfiguration(
                "WGPU DenormPreserve loading-path contract failed while materializing \
                 verdict-critical pipelines"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// #flush-charge: construct and qualify one device for CHARGED-flush
    /// verdict GPU CROWN.
    ///
    /// CONFIGURATION (admission-config): the device is constructed with the
    /// DenormPreserve policy FORCED to the plain-WGSL (flushing) loading path
    /// (`shader_loading::resolve_denorm_preserve_forced_disabled`), because
    /// that is the exact configuration the oracle flush charges model — the
    /// probes below measure it and production runs it, with no env required.
    /// The single override: an explicit `NY_GPU_DENORM_PRESERVE=1` pin refuses
    /// with a typed error (env wins, repo-wide precedence rule) — the user
    /// pinned a configuration the charges cannot cover. Unset/`auto`/`0` all
    /// admit the forced device.
    ///
    /// Admission (ALL required; fail-closed on anything else):
    /// * the reviewed charged source gate
    ///   [`PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED`] is open, AND the
    ///   base gate is open;
    /// * rungs 1 (IEEE f32 model), 4 (host EFT reference) and 5 (sentinel
    ///   taint) pass on this exact device;
    /// * rung 3 (gradual underflow) FAILED — a fully conformant adapter must
    ///   take [`WgpuDevice::new_for_verdict`] instead, so the charged state is
    ///   unreachable there;
    /// * the measured flush class is exactly [`FlushClass::PureFlush`]: every
    ///   refused lane is a modeled ±0 flush, never a wrong nonzero.
    ///
    /// Rung 2 (EFT primitives) is deliberately NOT consulted: it entails rung 3
    /// and the compensated channel is forbidden under the charge policy anyway.
    ///
    /// LADDER ECONOMY (the CLI chain's second ~0.6s): refusals that are
    /// implied WITHOUT any measurement are taken BEFORE constructing a device
    /// or running any rung — the env `=1` pin (a process fact) and the closed
    /// compile-time charged gate (a source fact). Rung REUSE from a preceding
    /// uncharged attempt is deliberately not offered: the charged device is a
    /// DIFFERENT device in a DIFFERENT shader-loading configuration, and rung
    /// outcomes are configuration facts (rung 3 provably differs between the
    /// passthrough and plain-WGSL configurations on DenormPreserve-capable
    /// adapters), so borrowed measurements would be dishonest evidence.
    ///
    /// A successful device stores BOTH the (non-qualified) report — so
    /// `sound_gpu_authority_cached()` stays `false` and the uncharged mode is
    /// untouched — and the [`FlushChargePolicy`] consumed by the charge sites.
    pub fn new_for_verdict_flush_charged(
        _request: WgpuChargedVerdictRequest,
    ) -> Result<Self, WgpuVerdictQualificationError> {
        // Env precedence first: an explicit passthrough pin is typed-refused
        // before any device work.
        if let Err(source) =
            crate::wgpu_device::shader_loading::resolve_denorm_preserve_forced_disabled()
        {
            return Err(WgpuVerdictQualificationError::initialization(source));
        }
        // Compile-time gate: while CLOSED the refusal is predetermined, so no
        // device is built and no qualification ladder runs (the report
        // carries no adapter — the structural pin that nothing was measured).
        if !PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED {
            return Err(WgpuVerdictQualificationError::rejected_charged(
                WgpuVerdictReport::not_run(),
                "the reviewed CHARGED-flush WGPU verdict source gate is closed \
                 (charge-policy audits incomplete)"
                    .to_string(),
            ));
        }
        let mut device = Self::new_with_denorm_preserve_override(Some(
            crate::wgpu_device::shader_loading::DenormPreservePolicy::ForcedDisabled,
        ))
        .map_err(WgpuVerdictQualificationError::initialization)?;
        let mut report = device.evaluate_verdict_report();
        report.flush_class = Some(device.characterize_flush_policy());
        // §H: prime the rung-5 subnormal-multiplier probe eagerly on this
        // device — `charged_flush_authority_cached` reads only the cache
        // (fail-closed when unprimed).
        let subnormal_mult_ok = device.verify_subnormal_mult_taint();

        if report.qualified() {
            return Err(WgpuVerdictQualificationError::rejected_charged(
                report,
                "this adapter passes the full five-rung ladder; charged-flush \
                 authority is deliberately unreachable — use new_for_verdict"
                    .to_string(),
            ));
        }
        let rungs_ok = report.ieee_f32_model.passed()
            && report.host_eft_reference.passed()
            && report.sentinel_taint_sticky.passed();
        let pure_flush = matches!(report.flush_class, Some(FlushClass::PureFlush));
        let admissible = PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED
            && report.source_gate_open
            && rungs_ok
            && !report.gradual_underflow.passed()
            && pure_flush
            && subnormal_mult_ok;
        if !admissible {
            let reason = if !PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED {
                "the reviewed CHARGED-flush WGPU verdict source gate is closed \
                 (charge-policy audits incomplete)"
                    .to_string()
            } else if !pure_flush {
                format!(
                    "charged-flush qualification refused: measured flush class \
                     {:?} is not PURE-FLUSH ({})",
                    report.flush_class,
                    report.reason()
                )
            } else if !subnormal_mult_ok {
                "charged-flush qualification refused: the §H rung-5 \
                 subnormal-multiplier probe measured a HAZARDOUS taint-word \
                 annihilation domain (word loss outside the strictly-subnormal \
                 multipliers the walk guard refuses)"
                    .to_string()
            } else {
                format!("charged-flush qualification refused: {}", report.reason())
            };
            tracing::info!(
                target: "ny_gpu::wgpu",
                adapter = report.adapter().unwrap_or("unknown"),
                reason = %reason,
                "explicit WGPU CHARGED-flush qualification REFUSED (fail-closed)"
            );
            return Err(WgpuVerdictQualificationError::rejected_charged(
                report, reason,
            ));
        }

        tracing::warn!(
            target: "ny_gpu::wgpu",
            adapter = report.adapter().unwrap_or("unknown"),
            "explicit WGPU CHARGED-flush qualification PASSED: this device is \
             armed for sound GPU CROWN with certified flush charges \
             (EFT forbidden; subnormal weights/bias/slopes/inputs refused)"
        );
        device.verdict_report = Some(report);
        device.charged_policy = Some(FlushChargePolicy::production());
        // Prewarm the resident cut-apply selfcheck (qualification-time
        // evidence for `provides_resident_cut_shadow`; capability-only, no
        // verdict rung reads it). Runs AFTER the charge policy arms so the
        // probe measures the exact armed device.
        let _ = device.verify_resident_cut_apply();
        Ok(device)
    }

    /// #flush-charge TEST-SCOPED ACCEPTANCE CONSTRUCTOR. Compiled out of every
    /// production build: `cfg(any(test, feature = "gpu-tests"))`, and the
    /// `gpu-tests` feature exists solely for real-adapter test invocations
    /// (`cargo test -p ny-gpu --features gpu-tests`); no production crate in
    /// the workspace enables it. There is deliberately NO production-reachable
    /// bypass of [`PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED`] — this
    /// symbol does not exist in a production compilation, so no production
    /// caller can name it, link it, or reach it.
    ///
    /// Purpose: acceptance evidence for the enclosure-parity row on the
    /// charged gate's audit ledger (supplied pre-flip; the row was CLOSED by
    /// the 2026-08-13 opening review, and the harness remains the regression
    /// oracle). It builds a device whose charged ARITHMETIC is byte-identical
    /// to the production charged device — `FlushChargePolicy::production()`
    /// armed, `charged_walk_guard` and every widened slack live — WITHOUT
    /// consulting the reviewed source gate and WITHOUT granting any
    /// production verdict authority (the CLI/proof router can only reach the
    /// production constructor, which applies the full admission including
    /// the reviewed gate).
    ///
    /// Admission is the production predicate MINUS ONLY the reviewed charged
    /// source-gate conjunct. Every LIVE conjunct is retained and measured on
    /// this exact device — INCLUDING the forced plain-WGSL construction
    /// (admission-config): like the production charged constructor, the
    /// device is built through the ForcedDisabled override, so the probes
    /// measure the flushing configuration the charges model WITHOUT any env,
    /// and an explicit `NY_GPU_DENORM_PRESERVE=1` pin typed-refuses:
    /// * base source gate open;
    /// * NOT fully qualified (a 5/5 adapter must use `new_for_verdict`);
    /// * rungs 1 (IEEE f32 model), 4 (host EFT reference), 5 (sentinel taint)
    ///   pass; rung 3 (gradual underflow) FAILS;
    /// * the measured flush class is exactly [`FlushClass::PureFlush`];
    /// * the §H rung-5 subnormal-multiplier probe measures a non-hazardous
    ///   annihilation domain (`verify_subnormal_mult_taint`).
    #[cfg(any(test, feature = "gpu-tests"))]
    pub fn test_only_new_flush_charged_for_acceptance_evidence(
    ) -> Result<Self, WgpuVerdictQualificationError> {
        let mut device = Self::new_with_denorm_preserve_override(Some(
            crate::wgpu_device::shader_loading::DenormPreservePolicy::ForcedDisabled,
        ))
        .map_err(WgpuVerdictQualificationError::initialization)?;
        let mut report = device.evaluate_verdict_report();
        report.flush_class = Some(device.characterize_flush_policy());
        // §H: prime the rung-5 subnormal-multiplier probe eagerly, exactly as
        // the production charged constructor does.
        let subnormal_mult_ok = device.verify_subnormal_mult_taint();

        if report.qualified() {
            return Err(WgpuVerdictQualificationError::rejected_charged(
                report,
                "this adapter passes the full five-rung ladder; charged-flush \
                 acceptance evidence is deliberately unreachable — use \
                 new_for_verdict"
                    .to_string(),
            ));
        }
        let rungs_ok = report.ieee_f32_model.passed()
            && report.host_eft_reference.passed()
            && report.sentinel_taint_sticky.passed();
        let pure_flush = matches!(report.flush_class, Some(FlushClass::PureFlush));
        // The production admission with the SINGLE compile-time source-gate
        // conjunct removed; every live, per-device conjunct is identical.
        let admissible = report.source_gate_open
            && rungs_ok
            && !report.gradual_underflow.passed()
            && pure_flush
            && subnormal_mult_ok;
        if !admissible {
            let reason = if !pure_flush {
                format!(
                    "test-scoped charged-flush acceptance refused: measured \
                     flush class {:?} is not PURE-FLUSH ({})",
                    report.flush_class,
                    report.reason()
                )
            } else if !subnormal_mult_ok {
                "test-scoped charged-flush acceptance refused: the §H rung-5 \
                 subnormal-multiplier probe measured a HAZARDOUS taint-word \
                 annihilation domain"
                    .to_string()
            } else {
                format!(
                    "test-scoped charged-flush acceptance refused: {}",
                    report.reason()
                )
            };
            return Err(WgpuVerdictQualificationError::rejected_charged(
                report, reason,
            ));
        }

        tracing::warn!(
            target: "ny_gpu::wgpu",
            adapter = report.adapter().unwrap_or("unknown"),
            "TEST-SCOPED charged-flush acceptance device armed (compiled out \
             of production builds; production admission still requires the \
             reviewed source gate plus the full live ladder on its own device)"
        );
        device.verdict_report = Some(report);
        device.charged_policy = Some(FlushChargePolicy::production());
        Ok(device)
    }

    /// Successful qualification evidence stored on this exact device.
    /// Ordinary [`WgpuDevice::new`] devices always return `None`.
    #[must_use]
    pub fn verdict_report(&self) -> Option<&WgpuVerdictReport> {
        self.verdict_report.as_ref()
    }

    /// Evaluate all five rungs without short-circuiting.
    fn evaluate_verdict_report(&self) -> WgpuVerdictReport {
        let ieee_f32_model = self.verify_ieee_f32_model();
        let eft_primitives = self.verify_eft_primitives();
        let gradual_underflow = self.verify_gradual_underflow();
        let host_eft_reference = ny_core::eft::eft_available();
        let sentinel_taint_sticky = self.verify_sentinel_taint_sticky();

        WgpuVerdictReport {
            adapter: Some(format!(
                "{} ({:?}, {:?}); denorm_preserve={} policy={}",
                self.adapter_info.name,
                self.adapter_info.device_type,
                self.adapter_info.backend,
                self.denorm_preserve_enabled(),
                self.denorm_preserve_policy_name(),
            )),
            source_gate_open: PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED,
            ieee_f32_model: WgpuVerdictRungOutcome::from_bool(ieee_f32_model),
            eft_primitives: WgpuVerdictRungOutcome::from_bool(eft_primitives),
            gradual_underflow: WgpuVerdictRungOutcome::from_bool(gradual_underflow),
            host_eft_reference: WgpuVerdictRungOutcome::from_bool(host_eft_reference),
            sentinel_taint_sticky: WgpuVerdictRungOutcome::from_bool(sentinel_taint_sticky),
            flush_class: None,
        }
    }

    /// Never-dispatching authority predicate used from production kernels.
    /// An ordinary device has no stored report and therefore refuses.
    #[cfg(all(test, feature = "gpu-tests"))]
    pub(crate) fn sound_gpu_authority(&self) -> bool {
        self.sound_gpu_authority_cached()
    }

    /// Never-dispatching cached authority read. The successful report is
    /// immutable, while the loading-path contract can only transition from
    /// intact to poisoned if a later lazy DenormPreserve module falls back.
    /// Compose that sticky state on every read so no cached grant survives a
    /// production module the qualification did not attest.
    pub(crate) fn sound_gpu_authority_cached(&self) -> bool {
        self.verdict_report
            .as_ref()
            .is_some_and(WgpuVerdictReport::qualified)
            && self.denorm_preserve_contract_intact()
    }

    /// #flush-charge: never-dispatching cached CHARGED-flush authority read.
    ///
    /// `Some(policy)` only on a device built through
    /// [`WgpuDevice::new_for_verdict_flush_charged`] whose admission passed.
    /// Every forced-fail hook and the DenormPreserve loading-path contract
    /// close this too (only ever MORE closed), mirroring the uncharged reads.
    pub(crate) fn charged_flush_authority_cached(&self) -> Option<&FlushChargePolicy> {
        let policy = self.charged_policy.as_ref()?;
        // Rung-1 cached read consults its own forced-fail hook internally.
        if !self.f32_model_cached() {
            return None;
        }
        // Rung-3/rung-5 forced-fail hooks close charged authority as well.
        if super::subnormal_selfcheck::probe_forced_to_fail()
            || super::sentinel_taint_selfcheck::probe_forced_to_fail()
        {
            return None;
        }
        // Rung 5 must have measured PASS on this device (primed eagerly by the
        // charged constructor's ladder; an unprimed cache reads false).
        if !self
            .sentinel_taint_selfcheck
            .get()
            .copied()
            .unwrap_or(false)
        {
            return None;
        }
        // §H: the rung-5 subnormal-multiplier probe must have measured a
        // non-hazardous annihilation domain on this device — i.e. taint-word
        // annihilation confined to the strictly-subnormal multipliers the
        // charged walk guard refuses (or structural immunity). Primed eagerly
        // by the charged constructor; an unprimed cache reads false.
        if !self
            .subnormal_mult_taint_selfcheck
            .get()
            .copied()
            .unwrap_or(false)
        {
            return None;
        }
        // A production module that fell back from the requested DenormPreserve
        // loading path executes a pipeline the characterization did not cover.
        if !self.denorm_preserve_contract_intact() {
            return None;
        }
        Some(policy)
    }

    /// #flush-charge: diagnostic authority state for reports/routing.
    #[must_use]
    pub fn verdict_authority(&self) -> WgpuVerdictAuthority {
        if self.sound_gpu_authority_cached() {
            WgpuVerdictAuthority::Qualified
        } else if let Some(policy) = self.charged_flush_authority_cached() {
            WgpuVerdictAuthority::QualifiedWithFlushCharge(*policy)
        } else {
            WgpuVerdictAuthority::Refused
        }
    }
}

#[cfg(test)]
mod cpu_tests {
    use super::*;

    #[test]
    fn complete_report_requires_source_gate_and_all_five_rungs() {
        let pass = WgpuVerdictRungOutcome::Passed;
        let fail = WgpuVerdictRungOutcome::Failed;
        let report = WgpuVerdictReport::from_outcomes(true, [pass; 5]);
        assert!(report.qualified());
        assert_eq!(report.failed_rung(), None);
        assert_eq!(
            report.reason(),
            "all WGPU verdict qualification requirements passed"
        );

        let closed = WgpuVerdictReport::from_outcomes(false, [pass; 5]);
        assert!(!closed.qualified());
        assert_eq!(closed.failed_rung(), None);
        assert_eq!(
            closed.reason(),
            "the reviewed WGPU verdict source gate is closed"
        );

        for (index, expected) in [
            WgpuVerdictRung::IeeeF32Model,
            WgpuVerdictRung::EftPrimitives,
            WgpuVerdictRung::GradualUnderflow,
            WgpuVerdictRung::HostEftReference,
            WgpuVerdictRung::SentinelTaintSticky,
        ]
        .into_iter()
        .enumerate()
        {
            let mut outcomes = [pass; 5];
            outcomes[index] = fail;
            let report = WgpuVerdictReport::from_outcomes(true, outcomes);
            assert!(!report.qualified());
            assert_eq!(report.failed_rung(), Some(expected));
            assert_eq!(report.outcome(expected), fail);
        }
    }

    /// P0 SOUNDNESS PIN: runtime input can never change the reviewed source-gate
    /// state, which is BUILD-pinned here. This is CPU-only so every build can
    /// check the policy without an adapter.
    ///
    /// 2026-08-11 (B0) review: the const-block sense INVERTED from `!ENABLED`
    /// to `ENABLED` — every ledger obligation is discharged (U1/U3/U4-chain/
    /// U5/U6, see the module intro), the ladder measures 5/5 on the reference
    /// GB10/Vulkan adapter, and the decided-row differential accompanied the
    /// flip. Closing authority again is itself a source-review event and must
    /// re-invert this pin. Authority still arms only through the typed explicit
    /// constructor and only when all five per-device probes pass.
    #[test]
    fn compile_time_authority_state_is_the_reviewed_2026_08_11_opening() {
        // `const` block, so a drift in the reviewed state fails the BUILD
        // rather than a test run.
        const {
            assert!(
                PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED,
                "the 2026-08-11 B0 review OPENED WGPU authority; closing it is a source-review event"
            )
        };
        assert!(WgpuVerdictReport::from_outcomes(
            PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED,
            [WgpuVerdictRungOutcome::Passed; 5],
        )
        .qualified());
    }

    /// #flush-charge P0 SOUNDNESS PIN: the CHARGED-flush source gate is
    /// BUILD-pinned OPEN as of the 2026-08-13 review — every audit row on the
    /// const's ledger was discharged first (§D/§F/§E LANDED, §H RESOLVED
    /// 2026-08-13, end-to-end enclosure parity CLOSED 2026-08-13). Closing it
    /// again is itself a source-review event and must re-invert this pin.
    /// Runtime input can never change it, and an open gate still grants
    /// nothing by itself: admission requires the typed request plus the full
    /// live pure-flush ladder on the exact returned device.
    #[test]
    fn compile_time_charged_authority_state_is_the_reviewed_2026_08_13_opening() {
        const {
            assert!(
                PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED,
                "the 2026-08-13 review OPENED CHARGED-flush WGPU authority; \
                 closing it is a source-review event"
            )
        };
    }

    /// #flush-charge: the production policy is all-refusals-on with the
    /// oracle-audited widening factors, and the report's reason names the
    /// charged admissibility exactly when the measured class is PURE-FLUSH.
    #[test]
    fn charged_policy_constants_and_reason_are_pinned() {
        let policy = FlushChargePolicy::production();
        assert_eq!(policy.w_l1_factor, 4.0);
        assert_eq!(policy.concretize_slack_factor, 8.0);
        assert_eq!(policy.bias_combine_factor, 4.0);
        assert_eq!(policy.act_bias_slack_factor, 4.0);
        assert!(policy.refuse_subnormal_weights);
        assert!(policy.refuse_subnormal_bias);
        assert!(policy.refuse_subnormal_slopes);
        assert!(policy.refuse_subnormal_inputs);
        assert!(policy.eft_forbidden);

        let pass = WgpuVerdictRungOutcome::Passed;
        let fail = WgpuVerdictRungOutcome::Failed;
        // The REAL charged-target shape (Apple M5 Max): rung 2 fails by the
        // #u2b entailment alongside rung 3; rungs 1/4/5 pass.
        let mut report = WgpuVerdictReport::from_outcomes(true, [pass, fail, fail, pass, pass]);
        assert_eq!(
            report.reason(),
            "the EFT-primitives qualification rung did not pass",
            "without a characterization the reason must not mention charging"
        );
        report.flush_class = Some(FlushClass::PureFlush);
        assert_eq!(
            report.reason(),
            "the gradual-underflow qualification rung did not pass; \
             measured flush class PURE-FLUSH — admissible under \
             charged-flush authority (not requested)"
        );
        report.flush_class = Some(FlushClass::NonConformant);
        assert_eq!(
            report.reason(),
            "the EFT-primitives qualification rung did not pass"
        );
        // A PURE-FLUSH class does NOT produce the admissibility hint when the
        // failure profile is not the charged shape (here: rung 1 also fails).
        let mut hard_fail = WgpuVerdictReport::from_outcomes(true, [fail, fail, fail, pass, pass]);
        hard_fail.flush_class = Some(FlushClass::PureFlush);
        assert_eq!(
            hard_fail.reason(),
            "the IEEE f32-model qualification rung did not pass"
        );
        // The charged state can never masquerade as full qualification.
        report.flush_class = Some(FlushClass::PureFlush);
        assert!(!report.qualified());
    }

    /// #flush-charge Fix 3 STRUCTURAL PIN, revisited by the 2026-08-13
    /// opening review (as the closed-gate pin's message demanded): with the
    /// reviewed charged source gate OPEN, the only refusal the production
    /// charged constructor takes BEFORE building a device is the explicit
    /// `NY_GPU_DENORM_PRESERVE=1` pin (a process fact, checked first). Every
    /// other path builds the forced plain-WGSL device and runs a GENUINE
    /// admission measurement — not redundant work, so no gate skip applies
    /// any more. This test stays CPU-only: the env-pin refusal shape is
    /// pinned through the pure resolver, never by mutating process env or
    /// constructing a device.
    #[test]
    fn open_gate_charged_pre_device_refusal_is_exactly_the_env_pin() {
        use crate::wgpu_device::shader_loading::{
            resolve_denorm_preserve_forced_disabled_from, DenormPreservePolicy,
        };

        const {
            assert!(
                PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED,
                "gate closed again: restore the closed-gate skip pin \
                 (closed_gate_charged_refusal_builds_no_device_and_runs_no_ladder)"
            )
        };
        // The single pre-device refusal: an explicit passthrough pin.
        let error = resolve_denorm_preserve_forced_disabled_from(Some(std::ffi::OsStr::new("1")))
            .expect_err("an explicit passthrough pin must refuse charged mode");
        let message = error.to_string();
        assert!(
            message.contains("NY_GPU_DENORM_PRESERVE"),
            "the pre-device refusal must name the env pin, got: {message}"
        );
        // Unset / auto / 0 all resolve to the forced plain-WGSL configuration
        // — i.e. they proceed to a real per-device admission measurement.
        for raw in [None, Some("auto"), Some("0")] {
            let (policy, enabled) =
                resolve_denorm_preserve_forced_disabled_from(raw.map(std::ffi::OsStr::new))
                    .expect("default env admits the forced configuration");
            assert_eq!(policy, DenormPreservePolicy::ForcedDisabled);
            assert!(!enabled, "a forced device never requests passthrough");
        }
    }

    #[test]
    fn initialization_failure_report_is_complete_and_fail_closed() {
        let report = WgpuVerdictReport::not_run();
        assert!(!report.qualified());
        assert_eq!(report.failed_rung(), None);
        assert_eq!(
            report.reason(),
            "WGPU device initialization did not complete"
        );
        for rung in [
            WgpuVerdictRung::IeeeF32Model,
            WgpuVerdictRung::EftPrimitives,
            WgpuVerdictRung::GradualUnderflow,
            WgpuVerdictRung::HostEftReference,
            WgpuVerdictRung::SentinelTaintSticky,
        ] {
            assert_eq!(report.outcome(rung), WgpuVerdictRungOutcome::NotRun);
        }
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};

    /// Ambient process state can never arm the ordinary constructor.
    #[test]
    fn ordinary_device_is_unconditionally_unarmed() {
        let _serial = gpu_test_serial_guard();
        let device = require_device();
        assert!(device.verdict_report().is_none());
        assert!(!device.sound_gpu_authority());
        assert!(!ny_core::GpuCrownBackward::provides_sound_gpu_crown(
            &*device
        ));
        assert!(!ny_core::GpuCrownBackward::provides_sound_gpu_bab_bound_phase(&*device));
        assert!(ny_core::GpuCrownBackward::gpu_bab_bound_numerical_tcb(&*device).is_none());
        assert!(ny_core::GemmEngine::as_gpu_crown_backward(&*device).is_none());
        assert!(crate::wgpu_device::test_support::sound_gpu_crown_quarantined(&device));
    }

    /// The explicit constructor either stores a fully passing report on the
    /// returned device or returns the complete rejected report. There is no
    /// partially armed state.
    #[test]
    fn explicit_constructor_is_typed_and_fail_closed() {
        let _serial = gpu_test_serial_guard();
        match WgpuDevice::new_for_verdict(WgpuVerdictRequest::new()) {
            Ok(device) => {
                let report = device
                    .verdict_report()
                    .expect("qualified device stores its report");
                assert!(report.qualified());
                assert_eq!(report.failed_rung(), None);
                assert!(device.sound_gpu_authority_cached());
                assert!(ny_core::GpuCrownBackward::provides_sound_gpu_crown(&device));
                assert!(!ny_core::GpuCrownBackward::provides_sound_gpu_bab_bound_phase(&device));
                assert!(ny_core::GpuCrownBackward::gpu_bab_bound_numerical_tcb(&device).is_none());
                assert!(ny_core::GemmEngine::as_gpu_crown_backward(&device).is_some());
            }
            Err(error) => {
                assert!(!error.report().qualified());
                assert!(error.report().failed_rung().is_some());
                assert!(!error.to_string().is_empty());
                assert!(!error.source_error().to_string().is_empty());
            }
        }
    }

    /// A failed early rung does not short-circuit the report: every later rung
    /// still records `Passed` or `Failed`, never `NotRun`.
    #[test]
    fn simulated_early_failure_still_reports_all_rungs() {
        use super::super::f32_selfcheck::set_force_f32_selfcheck_fail;

        let _serial = gpu_test_serial_guard();
        let device = require_device();
        set_force_f32_selfcheck_fail(true);
        let report = device.evaluate_verdict_report();
        set_force_f32_selfcheck_fail(false);

        assert_eq!(
            report.outcome(WgpuVerdictRung::IeeeF32Model),
            WgpuVerdictRungOutcome::Failed
        );
        assert_eq!(report.failed_rung(), Some(WgpuVerdictRung::IeeeF32Model));
        for rung in [
            WgpuVerdictRung::EftPrimitives,
            WgpuVerdictRung::GradualUnderflow,
            WgpuVerdictRung::HostEftReference,
            WgpuVerdictRung::SentinelTaintSticky,
        ] {
            assert_ne!(report.outcome(rung), WgpuVerdictRungOutcome::NotRun);
        }
    }

    /// MEASUREMENT: report each rung's real verdict on this adapter and pin the
    /// complete source-gate/ladder conjunction.
    #[test]
    fn report_ladder_and_pin_conjunction() {
        let _serial = gpu_test_serial_guard();
        let device = require_device();
        let report = device.evaluate_verdict_report();
        let f32_model = report.outcome(WgpuVerdictRung::IeeeF32Model).passed();
        let eft = report.outcome(WgpuVerdictRung::EftPrimitives).passed();
        let subnormal = report.outcome(WgpuVerdictRung::GradualUnderflow).passed();
        let host = report.outcome(WgpuVerdictRung::HostEftReference).passed();
        let sentinel = report
            .outcome(WgpuVerdictRung::SentinelTaintSticky)
            .passed();
        let raw_eft = device.eft_primitives_raw_probe();

        println!(
            "[sound-GPU authority ladder] adapter={} backend={:?}\n  \
             rung4 host eft_available        = {host}\n  \
             rung1 verify_ieee_f32_model     = {f32_model}\n  \
             rung2 verify_eft_primitives     = {eft}\n  \
             rung3 verify_gradual_underflow  = {subnormal}\n  \
             rung5 verify_sentinel_taint_sticky = {sentinel}\n  \
             compile-time authority enabled  = {}\n  \
             => report qualified             = {}\n  \
             [raw diagnostic only] EFT probe     = {raw_eft}",
            device.adapter_info.name,
            device.adapter_info.backend,
            PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED,
            report.qualified(),
        );
        assert_eq!(
            report.qualified(),
            PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED
                && host
                && f32_model
                && eft
                && subnormal
                && sentinel
        );
        assert!(
            device.verdict_report().is_none(),
            "measuring an ordinary device must not arm it"
        );

        // #u2b THE INVARIANT THIS FILE EXISTS TO PIN: rungs 2 and 3 may never
        // disagree in the UNSAFE direction. `rung2 ∧ ¬rung3` says "the
        // compensated channel is authorized on hardware that flushes the
        // residuals it measures" — on this adapter that was the live state
        // before the composition, and it must now be unrepresentable.
        //
        // The opposite disagreement (`¬rung2 ∧ rung3`) is SAFE and is
        // deliberately left representable: an adapter can honour gradual
        // underflow and still fail the fma-barrier TwoSum.
        assert!(
            !eft || subnormal,
            "#u2b VIOLATION: verify_eft_primitives()=true while \
             verify_gradual_underflow()=false — the EFT authorization has \
             stopped entailing its own underflow precondition"
        );
        // Non-vacuity: the pin above must be doing work, i.e. the raw probe is
        // what the composition is actually restraining. On an adapter where the
        // raw probe passes but the underflow rung fails (this box), the
        // composed gate MUST report false — that is the whole fix, measured.
        if raw_eft && !subnormal {
            assert!(
                !eft,
                "the raw EFT probe passes and the adapter FLUSHES, so the \
                 composed gate must refuse"
            );
            println!(
                "  => #u2b ACTIVE on this adapter: raw probe passes, adapter \
                flushes, composed gate correctly REFUSES the compensated channel"
            );
        }

        // #flush-charge: report the measured flush class and the CHARGED
        // conjunction next to the uncharged ladder.
        let flush_class = device.characterize_flush_policy();
        let charged_conjunction = PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED
            && PRODUCTION_WGPU_VERDICT_AUTHORITY_ENABLED
            && f32_model
            && host
            && sentinel
            && !subnormal
            && flush_class == FlushClass::PureFlush;
        let policy = FlushChargePolicy::production();
        println!(
            "[charged-flush authority] measured flush class = {flush_class:?}\n  \
             charged compile-time gate       = {}\n  \
             charged conjunction (1∧4∧5∧¬3∧pure-flush∧gate) = {charged_conjunction}\n  \
             policy: w_l1_factor={} concretize_slack_factor={} \
             bias_combine_factor={} act_bias_slack_factor={} \
             refusals(weights/bias/slopes/inputs)=on eft=forbidden",
            PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED,
            policy.w_l1_factor,
            policy.concretize_slack_factor,
            policy.bias_combine_factor,
            policy.act_bias_slack_factor,
        );
        // While the reviewed charged gate is CLOSED, the conjunction must be
        // false no matter what the hardware measures.
        assert!(
            !charged_conjunction || PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED,
            "charged conjunction cannot hold with the source gate closed"
        );
        assert!(
            device.charged_flush_authority_cached().is_none(),
            "measuring must never arm charged authority on an ordinary device"
        );
    }

    /// #flush-charge: the charged constructor is typed and fail-closed in
    /// BOTH arms. With the reviewed charged source gate OPEN (2026-08-13),
    /// an admission stores charged — never full — authority on the exact
    /// measured device; a refusal carries the complete typed report and
    /// leaves no armed device behind.
    #[test]
    fn charged_constructor_is_typed_and_fail_closed() {
        let _serial = gpu_test_serial_guard();
        match WgpuDevice::new_for_verdict_flush_charged(WgpuChargedVerdictRequest::new()) {
            Ok(device) => {
                // Only reachable after a source review opens the gate.
                assert!(
                    PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED
                        && device.charged_flush_authority_cached().is_some(),
                    "successful charged construction requires both the reviewed source gate \
                     and a cached live authority report"
                );
                assert!(
                    !device.sound_gpu_authority_cached(),
                    "charged authority must never masquerade as full qualification"
                );
                assert!(!ny_core::GpuCrownBackward::provides_sound_gpu_bab_bound_phase(&device));
                assert!(ny_core::GpuCrownBackward::gpu_bab_bound_numerical_tcb(&device).is_none());
            }
            Err(error) => {
                assert!(!error.report().qualified());
                assert!(!error.to_string().is_empty());
                if !PRODUCTION_WGPU_CHARGED_VERDICT_AUTHORITY_ENABLED
                    && error.report().adapter().is_some()
                {
                    assert!(
                        error.source_error().to_string().contains("CHARGED")
                            || error
                                .source_error()
                                .to_string()
                                .contains("five-rung ladder"),
                        "the refusal must name the charged gate or the \
                         full-qualification redirect, got: {}",
                        error.source_error()
                    );
                }
            }
        }
    }

    /// #flush-charge Fix 1 LIVE PIN (admission-config): under the DEFAULT env
    /// the TEST-SCOPED twin builds its device with the DenormPreserve policy
    /// forced to the plain-WGSL path and ARMS on this pure-flush box — no
    /// `NY_GPU_DENORM_PRESERVE=0` required, and the process-wide passthrough
    /// poison (from any env-AUTO device built earlier in this test process)
    /// cannot reach it.
    #[test]
    fn forced_disabled_twin_arms_and_measures_pure_flush_under_default_env() {
        let _serial = gpu_test_serial_guard();
        match WgpuDevice::test_only_new_flush_charged_for_acceptance_evidence() {
            Ok(device) => {
                assert!(
                    !device.denorm_preserve_enabled(),
                    "the charged device must never request passthrough"
                );
                assert_eq!(device.denorm_preserve_policy_name(), "forced-disabled");
                assert!(
                    device.denorm_preserve_contract_intact(),
                    "a forced plain-WGSL device is structurally immune to the \
                     process passthrough-fallback poison"
                );
                let report = device
                    .verdict_report()
                    .expect("armed device stores its report");
                assert_eq!(report.flush_class(), Some(FlushClass::PureFlush));
                assert!(!report.qualified());
                assert!(device.charged_flush_authority_cached().is_some());
                assert!(!device.sound_gpu_authority_cached());
            }
            Err(error) => {
                let message = error.source_error().to_string();
                assert!(
                    message.contains("NY_GPU_DENORM_PRESERVE")
                        || message.contains("five-rung ladder"),
                    "under the default env the forced-Disabled twin must arm on \
                     a pure-flush box, or name the explicit env pin / the \
                     full-qualification redirect; got: {message}"
                );
                println!("[forced-disabled twin] not armed in this environment: {message}");
            }
        }
    }

    /// #flush-charge Fix 2: NO §H HAZARDOUS warn on the charged path under the
    /// default env, captured live. (a) The production constructor (gate OPEN
    /// since 2026-08-13) measures its own forced plain-WGSL device — on this
    /// pure-flush box it ADMITS, and its arming warn names charges, never
    /// HAZARDOUS. (b) A freshly built forced-Disabled twin measures the same
    /// forced configuration, whose §H verdict on this box is non-hazardous.
    /// Genuine Hazardous adapters keep their warn — routed through
    /// `tracing::warn!` like every other rung refusal, never an unconditional
    /// eprintln.
    #[test]
    fn default_env_charged_path_emits_no_hazardous_warn() {
        use std::sync::{Arc, Mutex};

        let _serial = gpu_test_serial_guard();

        #[derive(Clone)]
        struct Buf(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Buf {
            fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(data);
                Ok(data.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let sink = Buf(Arc::new(Mutex::new(Vec::new())));
        let writer = sink.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_writer(move || writer.clone())
            .finish();
        let (production, twin) = tracing::subscriber::with_default(subscriber, || {
            (
                WgpuDevice::new_for_verdict_flush_charged(WgpuChargedVerdictRequest::new())
                    .map(|_| ()),
                WgpuDevice::test_only_new_flush_charged_for_acceptance_evidence().map(|_| ()),
            )
        });
        // With the gate OPEN the production outcome is adapter-dependent (an
        // admission on this pure-flush box; a typed refusal elsewhere). The
        // pin here is the LOG contract, not the outcome.
        drop(production);
        let logs = String::from_utf8_lossy(&sink.0.lock().unwrap()).into_owned();
        assert!(
            !logs.contains("HAZARDOUS"),
            "the default-env charged path must not emit the §H HAZARDOUS warn; \
             captured warn-level output:\n{logs}"
        );
        // The twin's outcome is pinned separately; here it only matters that
        // whatever it measured produced no hazardous classification.
        drop(twin);
    }
}
