// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Historical differential-qualification scaffolding from the rejected S1 WGPU
//! corpus gate (#s1-wgpu-unquarantine).
//!
//! # Current state: this corpus is inert; the typed production seam is live
//!
//! Nothing in this module grants production WGPU proof authority, and
//! [`corpus_is_green`](crate::wgpu_verdict::corpus_is_green) remains
//! unconditionally `false`. That no longer means all public WGPU CROWN is
//! quarantined. Production authority now lives in `ny-gpu`: an explicit
//! `WgpuVerdictRequest` consumed by `WgpuDevice::new_for_verdict` or
//! `ComputeDevice::new_for_proof` runs five live rungs on one exact context and
//! opens only that device's CROWN seam. The reviewed resident CROWN route admits
//! Conv, while host Conv and segment-resident streams still refuse; ordinary
//! devices and standalone GEMM, convolution, IBP, and DAG accessors stay closed.
//! The CLI conditionally uses the qualified device and reports a CPU fallback
//! when qualification refuses.
//!
//! Neither `NY_WGPU_VERDICT` nor `NY_WGPU_VERDICT_DISABLE` can grant that typed
//! production authority. They affect only the historical mode helpers retained
//! here for tests and diagnostics. This module also retains interval-union
//! arithmetic, independent CPU-reference helpers, and a monotone differential
//! ledger so the rejected corpus design remains reviewable. Its notes are not
//! the current production ledger; the U1/U3/U4/U5/U6+B0 authority case lives in
//! `crates/ny-gpu/src/wgpu_device/ops/sound_authority.rs`, summarized by
//! `docs/CURRENT_STATE_2026-08-10.md`.
//!
//! # Retained mode semantics
//!
//! [`wgpu_verdict_mode`](crate::wgpu_verdict::wgpu_verdict_mode) defaults to
//! [`WgpuVerdictMode::Off`](crate::wgpu_verdict::WgpuVerdictMode::Off).
//! `NY_WGPU_VERDICT_DISABLE`, when present, forces `Off` regardless of
//! `NY_WGPU_VERDICT`. The other parsed modes describe the rejected corpus
//! design; they do not control the live typed constructor:
//!
//! | `NY_WGPU_VERDICT` | parsed mode | historical intended behavior |
//! |---|---|---|
//! | unset / `off` / `0` | [`Off`](crate::wgpu_verdict::WgpuVerdictMode::Off) | CPU/reference bound only |
//! | `differential` | [`Differential`](crate::wgpu_verdict::WgpuVerdictMode::Differential) | compute both bounds and return their union |
//! | `on` | [`Enabled`](crate::wgpu_verdict::WgpuVerdictMode::Enabled) | request GPU-only authority after qualification |
//!
//! [`union_bounds`](crate::wgpu_verdict::union_bounds) is sound whenever its
//! reference input is sound: the elementwise union contains the reference even if
//! the GPU input is wrong. The ledger records whether GPU bounds enclose independent
//! references, with violations monotone and permanent. Neither property bypasses
//! this historical gate: [`corpus_is_green`](crate::wgpu_verdict::corpus_is_green)
//! stays fail-closed. The live public WGPU CROWN path instead uses its own exact
//! request plus five-rung qualification and never consults this corpus.

use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Mode the rejected historical S1 corpus design would have requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WgpuVerdictMode {
    /// DEFAULT for the historical corpus design. No corpus-mode result may
    /// decide a bound, and its legacy taint instrumentation is disarmed. This
    /// does not close an independently typed and live-qualified production
    /// CROWN device.
    #[default]
    Off,
    /// Historical qualification mode: GPU and reference bounds would both be
    /// computed and their [`union_bounds`] returned while recording comparisons.
    Differential,
    /// GPU-only bounds requested under the rejected corpus design. No production
    /// caller consumes this value; the live typed constructor uses its own
    /// five-rung report instead.
    Enabled,
}

/// Environment variable selecting the mode. Unset ⇒ [`WgpuVerdictMode::Off`].
pub const MODE_ENV: &str = "NY_WGPU_VERDICT";
/// Disable-flag kill switch: set to anything ⇒ [`WgpuVerdictMode::Off`], always.
pub const DISABLE_ENV: &str = "NY_WGPU_VERDICT_DISABLE";

/// Parse a historical mode string. Unrecognized spellings are **Off**
/// (fail-closed rather than approximating an authority request).
#[must_use]
pub fn parse_mode(raw: Option<&str>) -> WgpuVerdictMode {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("differential") | Some("diff") => WgpuVerdictMode::Differential,
        Some("on") | Some("1") | Some("true") => WgpuVerdictMode::Enabled,
        _ => WgpuVerdictMode::Off,
    }
}

/// Resolve the mode from the two environment variables, disable-flag first.
#[must_use]
fn mode_from_env(disable: Option<&str>, raw: Option<&str>) -> WgpuVerdictMode {
    if disable.is_some() {
        return WgpuVerdictMode::Off;
    }
    parse_mode(raw)
}

/// Test override, preserving the rejected hot-path design's one-load encoding.
/// `0` = no override; otherwise [`mode_from_code`]. Production qualification
/// does not read this value.
static MODE_OVERRIDE: AtomicU8 = AtomicU8::new(0);

const OVERRIDE_NONE: u8 = 0;
const OVERRIDE_OFF: u8 = 1;
const OVERRIDE_DIFFERENTIAL: u8 = 2;
const OVERRIDE_ENABLED: u8 = 3;

const fn mode_to_code(mode: WgpuVerdictMode) -> u8 {
    match mode {
        WgpuVerdictMode::Off => OVERRIDE_OFF,
        WgpuVerdictMode::Differential => OVERRIDE_DIFFERENTIAL,
        WgpuVerdictMode::Enabled => OVERRIDE_ENABLED,
    }
}

/// Decode an override code. Anything unrecognized is `Off` (fail-closed).
const fn mode_from_code(code: u8) -> WgpuVerdictMode {
    match code {
        OVERRIDE_DIFFERENTIAL => WgpuVerdictMode::Differential,
        OVERRIDE_ENABLED => WgpuVerdictMode::Enabled,
        _ => WgpuVerdictMode::Off,
    }
}

/// The process's historical WGPU corpus mode. The environment is read once and
/// cached, and the test override is a single relaxed atomic load.
#[must_use]
pub fn wgpu_verdict_mode() -> WgpuVerdictMode {
    let forced = MODE_OVERRIDE.load(Ordering::Relaxed);
    if forced != OVERRIDE_NONE {
        return mode_from_code(forced);
    }
    static ENV_MODE: OnceLock<WgpuVerdictMode> = OnceLock::new();
    *ENV_MODE.get_or_init(|| {
        let disable = std::env::var(DISABLE_ENV).ok();
        let raw = std::env::var(MODE_ENV).ok();
        mode_from_env(disable.as_deref(), raw.as_deref())
    })
}

/// RAII test override of the process mode; restores the previous value on drop.
///
/// Retained publicly for compatibility with historical qualification tests.
/// Tests that use it must hold their suite's serialization guard because the
/// override is process-global. It cannot affect the live typed constructor.
#[derive(Debug)]
pub struct ModeOverrideGuard {
    previous: u8,
}

impl ModeOverrideGuard {
    /// Force `mode` until the guard drops.
    #[must_use]
    pub fn force(mode: WgpuVerdictMode) -> Self {
        let previous = MODE_OVERRIDE.swap(mode_to_code(mode), Ordering::SeqCst);
        Self { previous }
    }
}

impl Drop for ModeOverrideGuard {
    fn drop(&mut self) {
        MODE_OVERRIDE.store(self.previous, Ordering::SeqCst);
    }
}

/// Whether the historical corpus design requested its taint instrumentation.
///
/// Armed in `Differential` and `Enabled`; disarmed in `Off`. Production WGPU
/// qualification and its resident taint-word channel do not consult this bit.
#[must_use]
pub fn taint_instrumentation_armed() -> bool {
    !matches!(wgpu_verdict_mode(), WgpuVerdictMode::Off)
}

/// Whether the historical mode requested GPU-only verdict authority.
///
/// This is never sufficient and has no production consumer. The live WGPU path
/// requires an explicit typed constructor plus its independent five-rung report.
#[must_use]
pub fn verdict_authority_requested() -> bool {
    matches!(wgpu_verdict_mode(), WgpuVerdictMode::Enabled)
}

/// Whether the historical mode requests a differential (GPU + reference union).
#[must_use]
pub fn differential_required() -> bool {
    matches!(wgpu_verdict_mode(), WgpuVerdictMode::Differential)
}

// ---------------------------------------------------------------------------
// Part 4: the differential gate.
// ---------------------------------------------------------------------------

/// Outcome of comparing one GPU bound against one reference bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnclosureVerdict {
    /// Every GPU endpoint is outside-or-equal the reference endpoint: the GPU
    /// interval contains the reference interval on every row.
    Encloses,
    /// At least one GPU endpoint is strictly INSIDE the reference interval, or is
    /// NaN, or the shapes disagree. Treated as a violation.
    Violates {
        /// Index of the first offending row.
        index: usize,
    },
}

impl EnclosureVerdict {
    /// `true` only for [`EnclosureVerdict::Encloses`].
    #[must_use]
    pub fn is_enclosing(self) -> bool {
        matches!(self, EnclosureVerdict::Encloses)
    }
}

/// Does the GPU interval contain the reference interval, row by row?
///
/// The test is **strict** — `gpu_lower <= ref_lower && gpu_upper >= ref_upper`, no
/// tolerance. That deliberately flags a GPU bound that is merely *one ULP tighter*
/// than the reference, even though such a bound may well be sound: a false alarm
/// costs only a fallback to the CPU path (weaker), while a tolerance band would be
/// exactly the place a genuinely-too-narrow bound could hide (wrong). There is no
/// tolerance constant here to tune, by design.
///
/// A NaN on either side, or a length mismatch, is a violation: an unordered
/// comparison must never be read as "enclosing".
#[must_use]
pub fn encloses(
    gpu_lower: &[f32],
    gpu_upper: &[f32],
    ref_lower: &[f32],
    ref_upper: &[f32],
) -> EnclosureVerdict {
    let n = gpu_lower.len();
    if gpu_upper.len() != n || ref_lower.len() != n || ref_upper.len() != n {
        return EnclosureVerdict::Violates { index: 0 };
    }
    for i in 0..n {
        let (gl, gu, rl, ru) = (gpu_lower[i], gpu_upper[i], ref_lower[i], ref_upper[i]);
        // NaN is checked EXPLICITLY rather than relying on a negated comparison:
        // an unordered pair must be a violation, never a silent "enclosing".
        if gl.is_nan() || gu.is_nan() || rl.is_nan() || ru.is_nan() {
            return EnclosureVerdict::Violates { index: i };
        }
        if gl > rl || gu < ru {
            return EnclosureVerdict::Violates { index: i };
        }
    }
    EnclosureVerdict::Encloses
}

/// Elementwise union `(min(lower), max(upper))` — the value `Differential` mode
/// returns.
///
/// SOUNDNESS: the union contains both operands, so it is an enclosure of the true
/// range whenever *either* operand is. Since the reference bound is the proven-sound
/// one, the union is sound no matter how wrong the GPU bound is. NaN is propagated
/// outward to the widest possible reading (`-inf` / `+inf` are not used; a NaN
/// endpoint is replaced by the other operand's endpoint only when the other is a
/// number, otherwise the NaN is preserved so downstream NaN guards still fire).
#[must_use]
pub fn union_bounds(
    gpu_lower: &[f32],
    gpu_upper: &[f32],
    ref_lower: &[f32],
    ref_upper: &[f32],
) -> Option<(Vec<f32>, Vec<f32>)> {
    let n = gpu_lower.len();
    if gpu_upper.len() != n || ref_lower.len() != n || ref_upper.len() != n {
        return None;
    }
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    for i in 0..n {
        let (gl, gu, rl, ru) = (gpu_lower[i], gpu_upper[i], ref_lower[i], ref_upper[i]);
        // NaN-preserving outward merge: if either side is NaN the merged endpoint
        // is NaN, which every downstream readback guard already rejects. Otherwise
        // take the outward endpoint.
        lower.push(if gl.is_nan() || rl.is_nan() {
            f32::NAN
        } else if gl < rl {
            gl
        } else {
            rl
        });
        upper.push(if gu.is_nan() || ru.is_nan() {
            f32::NAN
        } else if gu > ru {
            gu
        } else {
            ru
        });
    }
    Some((lower, upper))
}

/// Monotone differential corpus: comparisons only ever increase, violations only
/// ever increase, and there is no API that lowers either. The diagnostic
/// [`CorpusStats::is_green_for_one_key`] result can therefore flip from `false` to
/// `true` as evidence accumulates and back to `false` on the first violation, but
/// never return to `true` afterward. Historical [`corpus_is_green`] remains an
/// unconditional `false` regardless of this state and is not the live typed
/// production gate.
#[derive(Debug, Default)]
struct DifferentialCorpus {
    comparisons: AtomicU64,
    violations: AtomicU64,
    first_violation_index: AtomicUsize,
}

fn corpus() -> &'static DifferentialCorpus {
    static CORPUS: OnceLock<DifferentialCorpus> = OnceLock::new();
    CORPUS.get_or_init(DifferentialCorpus::default)
}

/// A snapshot of the differential corpus, for diagnostics and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorpusStats {
    /// Total GPU-vs-reference comparisons recorded in this process.
    pub comparisons: u64,
    /// How many of them were NOT enclosing.
    pub violations: u64,
    /// Row index of the first violation seen (meaningless when `violations == 0`).
    pub first_violation_index: usize,
}

impl CorpusStats {
    /// Green ⇔ at least one comparison and zero violations, FOR ONE
    /// (entry point, shape class) — never process-wide.
    ///
    /// # This predicate is deliberately not enough on its own
    ///
    /// It was reviewed as verdict-narrowing in exactly this form.
    /// `comparisons > 0` is a corpus of size ONE, and it used to be read
    /// process-wide: a single enclosing comparison — possibly on a tiny 3-layer
    /// linear subnetwork — permanently qualified the raw GPU bound for EVERY
    /// shape and opened NINE further entry points
    /// (`crown_backward_gpu_seeded_sound`, `..._resnet_sound{,_grad,_beta}`,
    /// `..._beta_batched{,_grad,_trajectory,_coeff}`, `..._beta_grad`) that had
    /// never been compared at all and that drive DIFFERENT resident kernels.
    ///
    /// So callers must hold one ledger per (entry point, shape class) and may
    /// never consult another's. [`QualificationKey`] exists to make that the
    /// only expressible thing; the predicate itself stays narrow, because
    /// widening it is the failure mode it was flagged for.
    ///
    /// The corpus SIZE required before a bound may travel alone is not settable
    /// here and is not a constant: it must come from a measurement on the target
    /// hardware. This requirement belongs to the historical S1 corpus-gate design,
    /// not the separate current production-authority ledger in `ny-gpu`.
    #[must_use]
    pub fn is_green_for_one_key(self) -> bool {
        self.comparisons > 0 && self.violations == 0
    }
}

/// The scope a [`CorpusStats`] verdict is valid for (#s1-wgpu-unquarantine).
///
/// Qualification does not generalize across entry points or shape classes: each
/// drives a different kernel, and evidence for one is not evidence for another.
/// Carrying the scope in the type is what stops a future caller from reaching
/// for a process-wide "is it green yet" flag, which is what the reviewed
/// implementation did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QualificationKey {
    /// The GPU entry point that produced the bound.
    pub entry_point: &'static str,
    /// A coarse shape class; bounds of different rank/extent exercise different
    /// kernels and must qualify separately.
    pub shape_class: (usize, usize),
}

/// Record one GPU-vs-reference comparison in the process corpus and return its
/// verdict. Call this from every site that has both bounds in hand.
/// Where the CPU-side arm of a differential comparison was actually computed
/// (#s1-wgpu-unquarantine).
///
/// This exists because the reviewed implementation's "reference" was NOT a
/// reference. `crown_backward_sound_host` calls `self.conv_transpose_2d`
/// unqualified from `impl WgpuDevice`, which resolves to the INHERENT method —
/// the real fused GEMM+col2im GPU kernel — not to the quarantined `GemmEngine`
/// stub. So the differential compared the GPU against itself: a shared kernel
/// defect appears in both arms and cancels, and the gate certifies an agreement
/// it has not earned.
///
/// Making the provenance an explicit argument turns that from a latent trap
/// into a state the API cannot enter silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceProvenance {
    /// Computed end to end on the CPU, sharing no kernel with the GPU arm.
    /// The only provenance a differential comparison may be counted under.
    CpuIndependent,
    /// Computed wholly or partly on the accelerator being checked. Accepted by
    /// the API only so it can be REFUSED loudly rather than mistaken for a
    /// reference.
    SameAccelerator,
}

/// Record a differential comparison whose reference provenance is stated.
///
/// A `SameAccelerator` reference is never counted: it returns
/// [`EnclosureVerdict::Violates`] at index 0 and records a violation, because a
/// self-comparison must not be able to qualify anything. That direction is
/// deliberate — violations are monotone and permanent, so an attempt to qualify
/// the gate with a GPU-computed reference poisons the corpus rather than
/// silently passing.
///
/// [`record_enclosure`] is retained for the existing call sites and is
/// equivalent to passing [`ReferenceProvenance::CpuIndependent`]; it will become
/// the compatibility shim once the CPU reference lands.
pub fn record_enclosure_with_provenance(
    gpu_lower: &[f32],
    gpu_upper: &[f32],
    ref_lower: &[f32],
    ref_upper: &[f32],
    provenance: ReferenceProvenance,
) -> EnclosureVerdict {
    if provenance == ReferenceProvenance::SameAccelerator {
        let c = corpus();
        c.comparisons.fetch_add(1, Ordering::SeqCst);
        c.first_violation_index.store(0, Ordering::SeqCst);
        c.violations.fetch_add(1, Ordering::SeqCst);
        return EnclosureVerdict::Violates { index: 0 };
    }
    record_enclosure(gpu_lower, gpu_upper, ref_lower, ref_upper)
}

pub fn record_enclosure(
    gpu_lower: &[f32],
    gpu_upper: &[f32],
    ref_lower: &[f32],
    ref_upper: &[f32],
) -> EnclosureVerdict {
    let verdict = encloses(gpu_lower, gpu_upper, ref_lower, ref_upper);
    let c = corpus();
    c.comparisons.fetch_add(1, Ordering::SeqCst);
    if let EnclosureVerdict::Violates { index } = verdict {
        // Record the index BEFORE the violation count, so any reader that sees a
        // nonzero count also sees a meaningful index.
        c.first_violation_index.store(index, Ordering::SeqCst);
        c.violations.fetch_add(1, Ordering::SeqCst);
        // No logging here: `ny-core` deliberately has no `tracing` dependency.
        // The caller (which knows the call site, the layer, and the adapter) is
        // the one that logs; `corpus_stats()` carries the machine-readable facts.
    }
    verdict
}

/// Snapshot the differential corpus.
#[must_use]
pub fn corpus_stats() -> CorpusStats {
    let c = corpus();
    CorpusStats {
        comparisons: c.comparisons.load(Ordering::SeqCst),
        violations: c.violations.load(Ordering::SeqCst),
        first_violation_index: c.first_violation_index.load(Ordering::SeqCst),
    }
}

/// Has this dormant differential corpus qualified its historical GPU-only path?
///
/// **Returns `false` unconditionally.** No environment variable, corpus state, or
/// caller can make it `true`. This helper is retained for diagnostics and tests; it
/// is not the production WGPU verdict-authority seam. Public CROWN authority is
/// separately request- and probe-qualified in
/// `crates/ny-gpu/src/wgpu_device/ops/sound_authority.rs`; the typed
/// `WgpuDevice` and `ComputeDevice` proof constructors consume that request,
/// while the CLI reports and falls back to CPU on refusal. None consult this
/// corpus.
///
/// The numbered list below is the preserved review record for the rejected S1
/// corpus-gate design. Its status language describes that historical checkpoint;
/// it is not the current U1/U3/U4/U5/U6 production-obligation ledger:
///
/// 1. **Scope.** The qualification ledger is per-(entry point, shape class) —
///    see [`QualificationKey`] and [`CorpusStats::is_green_for_one_key`]. The
///    reviewed version read this predicate process-wide, so one comparison on a
///    tiny linear subnetwork qualified nine other entry points driving different
///    resident kernels. The per-key corpus SIZE must come from a measurement on
///    the target hardware; it is not a constant to be picked here.
/// 2. ~~**The reference is dead code.**~~ **REFUTED 2026-08-02 — do not act on
///    this as written.** The review held that `crown_backward_sound_host`
///    dispatches through `self`'s `GemmEngine` impl, which is a quarantined
///    `Err`, making the reference composition unreachable. It is not.
///    `crown_backward_sound_host` lives in `impl WgpuDevice`
///    (`ops/crown_backward_sound_host.rs:122`) and calls `self.conv_transpose_2d`
///    UNQUALIFIED; `ops/conv_transpose.rs:34-39` defines an INHERENT
///    `pub(crate) fn conv_transpose_2d` on the same type. Rust resolves inherent
///    methods before trait methods, so the call binds to the real fused
///    GEMM+col2im GPU kernel, never to `ops/gemm.rs:436`'s quarantined trait
///    stub. The reference runs.
/// 3. ~~**Common-mode blindness.**~~ **ADDRESSED 2026-08-02, pending a GPU
///    dispatch to confirm agreement.** Two things now stand between the gate and
///    a self-comparison: [`ReferenceProvenance`] makes the reference's origin an
///    explicit argument and REFUSES a `SameAccelerator` arm (poisoning the
///    corpus rather than counting it), and
///    [`conv_transpose_2d_cpu_reference`] supplies a genuinely independent CPU
///    arm — naive nested scatters, no engine, no BLAS, no shared buffers, so no
///    kernel defect can appear in both arms and cancel. What remains is
///    observing the two agree on real hardware; that is the differential's job
///    and it has not been run. Original text, for the record: the reference the
///    differential compared against was ITSELF on the GPU — Because of the resolution above, the
///    "reference" the differential compares a GPU bound against is ITSELF
///    computed on the GPU: same fused conv_transpose kernel, and it also
///    terminates in the same `concretize_sound_gpu`. A shared kernel defect
///    therefore appears identically in both arms and CANCELS, so the gate would
///    certify agreement it has not earned. This is the gating blocker, and it is
///    statically fixable HERE — no GPU hardware and no CUDA target required: the
///    reference must be routed through a CPU conv backward (ny-gpu already
///    depends on ny-propagate) end to end, concretization included. Until then
///    the differential compares the GPU against itself.
/// 4. **Taint coverage is claimed, not held.** (Blocked on `taint.rs`, which
///    cannot land without the accessor flips — but see the note below on how
///    dangerous those actually are, which is less than first reported.) Nine entry points are asserted to
///    run inside a `run_gpu_checked` epoch; `crown_backward.rs` explicitly says
///    "do NOT wrap in `run_gpu_checked` here", and `encode_taint_scan` silently
///    no-ops with no epoch open — precisely the entry points cifar100 routes
///    through.
/// 5. ~~**Self-revocation.**~~ **FIXED 2026-08-02.**
///    [`conv_transpose_2d_cpu_reference_enclosure`] returns a sound INTERVAL
///    instead of a point: it charges the f64 accumulation error as a measured
///    `k · 2^-53 · Σ|terms|` Higham bound and rounds outward through the f64→f32
///    narrowing. A GPU bound that differs only by f32 rounding therefore no
///    longer registers as a violation. `encloses` is UNCHANGED — a tolerance
///    band there would blunt the one check that detects a bound narrower than
///    the truth, which is the false-proof direction and the only thing this gate
///    exists to catch. `rounding_noise_survives_but_a_narrower_bound_still_violates`
///    pins both halves.
/// 6. **Scan the right bytes.** The taint pass reads every bound storage buffer
///    as f32 over its full allocation, including integer metadata and the stale
///    tail of `BufferPool`-recycled oversized buffers, so it will taint nearly
///    every epoch and mask any real measurement.
///
/// At that checkpoint, nothing here had been dispatched on hardware:
/// `TAINT_SCAN_SHADER`'s
/// `atomicOr` over a runtime-sized storage array has never run, and this
/// predicate has never been `true` anywhere. The design's own acceptance test —
/// one oval21 row with GPU bounds enclosing CPU bounds at every node — has not
/// been attempted.
///
/// # Historical correction on how exposed the accessor flips were (2026-08-02)
///
/// S1 was refused partly on the finding that `gpu_crown_backward_route`
/// (`ny-propagate/src/sound_gpu_gate.rs:447-466`) hands back `(engine, false)`
/// for any engine whose `as_gpu_crown_backward()` is `Some` when the gate is
/// disengaged, after which `crown_partial_gpu.rs:159` calls the fast, UNSOUND
/// `crown_backward_gpu`. That is accurate, and I then described the lane as
/// "ungated". It is not, and the overstatement is worth correcting because it
/// changes whether S1 may be landed at all.
///
/// `DEFAULT_SOUND_GPU_CROWN` is `true`, so the gate is engaged by default and the
/// route takes a qualified sound backend or the CPU fallback. At the time of the
/// S1 review, a CLI flag could explicitly release that gate. That exposure no
/// longer exists: production CLI entry points require sound GPU CROWN
/// unconditionally; beta-crown's retired compatibility value is rejected before
/// model loading, and verify/VNN-COMP have no opt-out. Only an explicit library or
/// test caller can invoke the programmatic release seam.
///
/// The absence of WGSL f64 is not itself a soundness blocker. The WGPU sound lane
/// carries a pure-f32 Higham `γ_k·S` correction and an EFT/double-single residual.
/// As of the 2026-08-11 UTC B0 review, production-kernel integrity U1, EFT
/// tree-accounting U3, overflow-taint obligation U4, and U5/U6 are discharged.
/// The word route is
/// AUTO/default-on when its twins are available, ResNet segments compose the
/// row words, and the armed C1 consult refuses absent or tainted rows. The raw
/// source constant is open; an exact process request and all five live probes
/// remain mandatory. The public `ComputeDevice` CROWN-only constructor and CLI
/// routing now consume that exact ladder; ordinary devices and all non-CROWN
/// accessors remain closed. See `docs/CURRENT_STATE_2026-08-10.md` for the
/// authoritative current ledger.
#[must_use]
pub fn corpus_is_green() -> bool {
    false
}

/// The unqualified corpus verdict, for diagnostics and tests ONLY.
///
/// Deliberately not consulted by [`corpus_is_green`]: reading real corpus state
/// into the fast-lane decision is the defect that blocked S1. Kept so the ledger
/// remains observable while the blockers are worked.
#[must_use]
pub fn corpus_is_green_unqualified() -> bool {
    corpus_stats().is_green_for_one_key()
}

/// Has ANY enclosure violation been recorded in this process?
///
/// Deliberately separate from [`corpus_is_green`]: "not yet qualified" (empty
/// corpus) must not be confused with "disqualified" in the rejected design. A
/// violation permanently poisons only this historical corpus because the
/// counter never decreases; the live typed production ladder is independent.
#[must_use]
pub fn corpus_has_violation() -> bool {
    corpus().violations.load(Ordering::SeqCst) > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unset historical mode state stays Off. Live production authority is
    /// separately explicit and typed; it is not inferred from these variables.
    #[test]
    fn default_mode_is_off_and_grants_nothing() {
        assert_eq!(parse_mode(None), WgpuVerdictMode::Off);
        assert_eq!(mode_from_env(None, None), WgpuVerdictMode::Off);
        let off = mode_from_env(None, Some("off"));
        assert_eq!(off, WgpuVerdictMode::Off);
    }

    /// A typo must fail CLOSED, not fall through to some other mode.
    #[test]
    fn unrecognized_mode_spelling_is_off() {
        for raw in ["yes", "enable", "ON!", "differential-mode", "", "  "] {
            assert_eq!(
                parse_mode(Some(raw)),
                WgpuVerdictMode::Off,
                "unrecognized {raw:?} must be Off"
            );
        }
    }

    /// The disable flag is read first and cannot be overridden by the mode var.
    #[test]
    fn disable_flag_beats_every_mode_request() {
        assert_eq!(mode_from_env(Some("1"), Some("on")), WgpuVerdictMode::Off);
        assert_eq!(
            mode_from_env(Some(""), Some("differential")),
            WgpuVerdictMode::Off
        );
        assert_eq!(mode_from_env(None, Some("on")), WgpuVerdictMode::Enabled);
        assert_eq!(
            mode_from_env(None, Some("differential")),
            WgpuVerdictMode::Differential
        );
    }

    /// Strict enclosure: equality is fine, one ULP of narrowing is a violation.
    #[test]
    fn enclosure_is_strict_in_the_narrowing_direction() {
        let rl = [0.0f32, -1.0];
        let ru = [1.0f32, 2.0];
        assert!(
            encloses(&rl, &ru, &rl, &ru).is_enclosing(),
            "equality holds"
        );

        let wider_l = [-0.5f32, -1.5];
        let wider_u = [1.5f32, 2.5];
        assert!(encloses(&wider_l, &wider_u, &rl, &ru).is_enclosing());

        // Lower endpoint one ULP INSIDE the reference: violation.
        let narrow_l = [f32::from_bits(0.0f32.to_bits() + 1), -1.0];
        assert_eq!(
            encloses(&narrow_l, &ru, &rl, &ru),
            EnclosureVerdict::Violates { index: 0 }
        );

        // Upper endpoint one ULP INSIDE the reference: violation on row 1.
        let narrow_u = [1.0f32, f32::from_bits(2.0f32.to_bits() - 1)];
        assert_eq!(
            encloses(&rl, &narrow_u, &rl, &ru),
            EnclosureVerdict::Violates { index: 1 }
        );
    }

    /// NaN is never "enclosing", and a shape mismatch is never "enclosing".
    #[test]
    fn nan_and_shape_mismatch_are_violations() {
        let rl = [0.0f32];
        let ru = [1.0f32];
        assert!(!encloses(&[f32::NAN], &ru, &rl, &ru).is_enclosing());
        assert!(!encloses(&rl, &[f32::NAN], &rl, &ru).is_enclosing());
        assert!(!encloses(&rl, &ru, &[f32::NAN], &ru).is_enclosing());
        assert!(!encloses(&[], &ru, &rl, &ru).is_enclosing());
    }

    /// The union contains BOTH operands — the property that makes `Differential`
    /// mode sound no matter what the GPU returned.
    #[test]
    fn union_contains_both_operands() {
        let gl = [-3.0f32, 0.5];
        let gu = [0.25f32, 9.0];
        let rl = [-1.0f32, 0.0];
        let ru = [2.0f32, 1.0];
        let (ul, uu) = union_bounds(&gl, &gu, &rl, &ru).expect("same shape");
        assert!(encloses(&ul, &uu, &gl, &gu).is_enclosing(), "contains GPU");
        assert!(
            encloses(&ul, &uu, &rl, &ru).is_enclosing(),
            "contains reference"
        );
        assert_eq!(ul, vec![-3.0, 0.0]);
        assert_eq!(uu, vec![2.0, 9.0]);
    }

    /// Even a catastrophically-wrong GPU bound cannot narrow the union below the
    /// reference — the "one of them is sound ⇒ the union is sound" argument.
    #[test]
    fn union_cannot_be_narrower_than_the_reference() {
        let rl = [-1.0f32; 8];
        let ru = [1.0f32; 8];
        // Degenerate GPU "bound": a point, far inside the reference.
        let gl = [0.0f32; 8];
        let gu = [0.0f32; 8];
        let (ul, uu) = union_bounds(&gl, &gu, &rl, &ru).expect("same shape");
        assert_eq!(ul, rl.to_vec());
        assert_eq!(uu, ru.to_vec());
    }

    /// A NaN endpoint survives the union so downstream NaN guards still fire.
    #[test]
    fn union_preserves_nan_for_downstream_guards() {
        let (ul, uu) = union_bounds(&[f32::NAN], &[1.0], &[0.0], &[1.0]).expect("same shape");
        assert!(ul[0].is_nan());
        assert_eq!(uu[0], 1.0);
    }

    /// A shape mismatch yields no union at all (the caller must fail closed).
    #[test]
    fn union_refuses_shape_mismatch() {
        assert!(union_bounds(&[0.0], &[1.0], &[0.0, 0.0], &[1.0, 1.0]).is_none());
    }

    /// The corpus is monotone: recording never lowers either counter, and a green
    /// corpus cannot be restored by piling on more good comparisons after a bad one.
    #[test]
    fn corpus_is_monotone_and_violation_is_permanent() {
        let before = corpus_stats();
        let rl = [0.0f32];
        let ru = [1.0f32];
        assert!(record_enclosure(&rl, &ru, &rl, &ru).is_enclosing());
        let after_good = corpus_stats();
        assert_eq!(after_good.comparisons, before.comparisons + 1);
        assert_eq!(after_good.violations, before.violations);

        // A violation, then more good comparisons: violations never decrease.
        assert!(!record_enclosure(&[0.5], &ru, &rl, &ru).is_enclosing());
        let after_bad = corpus_stats();
        assert_eq!(after_bad.violations, before.violations + 1);
        assert!(record_enclosure(&rl, &ru, &rl, &ru).is_enclosing());
        assert_eq!(corpus_stats().violations, after_bad.violations);
        assert!(
            !corpus_is_green(),
            "one violation must not revive the historical corpus gate"
        );
    }

    /// The override encoding round-trips, and an unknown code decodes to `Off`
    /// (fail-closed) rather than to some other mode.
    #[test]
    fn override_codes_round_trip_and_decode_unknown_as_off() {
        for mode in [
            WgpuVerdictMode::Off,
            WgpuVerdictMode::Differential,
            WgpuVerdictMode::Enabled,
        ] {
            assert_eq!(mode_from_code(mode_to_code(mode)), mode);
            assert_ne!(
                mode_to_code(mode),
                OVERRIDE_NONE,
                "an override code must never collide with 'no override'"
            );
        }
        assert_eq!(mode_from_code(OVERRIDE_NONE), WgpuVerdictMode::Off);
        assert_eq!(mode_from_code(200), WgpuVerdictMode::Off);
    }

    /// The mode override guard restores whatever was there before.
    #[test]
    fn mode_override_guard_restores_previous() {
        let outer = wgpu_verdict_mode();
        {
            let _g = ModeOverrideGuard::force(WgpuVerdictMode::Differential);
            assert_eq!(wgpu_verdict_mode(), WgpuVerdictMode::Differential);
            assert!(taint_instrumentation_armed());
            assert!(differential_required());
            assert!(!verdict_authority_requested());
            {
                let _inner = ModeOverrideGuard::force(WgpuVerdictMode::Enabled);
                assert!(verdict_authority_requested());
                assert!(taint_instrumentation_armed());
            }
            assert_eq!(wgpu_verdict_mode(), WgpuVerdictMode::Differential);
        }
        assert_eq!(wgpu_verdict_mode(), outer);
    }
}

#[cfg(test)]
mod s1_quarantine_tripwire {
    use super::*;

    /// #s1-wgpu-unquarantine TRIPWIRE. The rejected S1 corpus gate must remain
    /// unconditionally `false`; it is diagnostic scaffolding, not the current
    /// production-authority predicate.
    ///
    /// This test exists to make arming S1 a DELIBERATE act. If you are here
    /// because this test failed, you did not "fix a stale test" — you changed
    /// the seam that admits GPU-produced bounds into a formal verifier's verdict
    /// path. Reviving it requires a fresh review of the complete historical design,
    /// independent of the current U1/U3/U4/U5/U6 production-authority ledger.
    ///
    /// Deliberately independent of corpus state: the corpus is a process-global
    /// whose violations are permanent, so any assertion about its CONTENT races
    /// sibling tests. That the gate ignores it entirely is the property worth
    /// pinning, and it is race-free precisely because it ignores it.
    /// A GPU-computed "reference" must never be able to qualify anything.
    ///
    /// This is the shape of the actual defect: the reviewed reference resolved
    /// to the inherent GPU `conv_transpose_2d`, so the differential compared the
    /// accelerator against itself. Even a PERFECTLY enclosing self-comparison
    /// must be refused, because agreement with yourself is not evidence.
    #[test]
    fn a_same_accelerator_reference_is_refused_even_when_it_encloses() {
        // GPU strictly wider than the "reference" on both sides — this would be
        // a clean `Encloses` if the provenance were independent.
        let v = record_enclosure_with_provenance(
            &[-1.0, -1.0],
            &[2.0, 2.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            ReferenceProvenance::SameAccelerator,
        );
        assert_eq!(
            v,
            EnclosureVerdict::Violates { index: 0 },
            "a self-comparison must be refused, not counted as evidence"
        );
        assert!(
            !corpus_is_green(),
            "and it must not open the historical corpus lane either"
        );
    }

    #[test]
    fn fast_lane_is_fail_closed_regardless_of_corpus_state() {
        assert!(
            !corpus_is_green(),
            "the historical S1 corpus lane must remain fail-closed"
        );

        // A genuinely enclosing comparison — GPU WIDER than the reference on
        // both sides, which is the direction `encloses` requires — must still
        // not open the lane. The reviewed defect was exactly that one clean
        // comparison qualified nine unrelated entry points process-wide.
        let v = record_enclosure(&[-1.0, -1.0], &[2.0, 2.0], &[0.0, 0.0], &[1.0, 1.0]);
        assert_eq!(
            v,
            EnclosureVerdict::Encloses,
            "fixture must enclose: GPU [-1,2] contains reference [0,1]"
        );
        assert!(
            !corpus_is_green(),
            "a clean comparison must NOT open the historical corpus lane: qualification is \
             per-(entry point, shape class) and its size is unmeasured"
        );
    }
}

/// An INDEPENDENT CPU computation of `conv_transpose_2d`, for the differential
/// gate's reference arm (#s1-wgpu-unquarantine).
///
/// # Why this exists rather than reusing the engine
///
/// The differential's whole content is that two *independent* computations
/// agree. The reviewed implementation had no independence: its "reference"
/// called `self.conv_transpose_2d` from `impl WgpuDevice`, which resolves to the
/// INHERENT fused GEMM+col2im GPU kernel, so a shared kernel defect appeared in
/// both arms and cancelled. Calling back into any `GemmEngine` would repeat that
/// mistake in a subtler form, because the production CPU path and the GPU path
/// share call sites and buffer plumbing.
///
/// So this is deliberately naive and self-contained: two nested scatters, no
/// BLAS, no engine, no shared buffers. Slowness is the point — it is a
/// reference, not a lane.
///
/// # Relationship to [`crate::NaiveCpuGemmEngine`]
///
/// That engine already implements the same contract on the CPU, and I wrote this
/// without checking — a duplication worth being explicit about. It is kept only
/// because the differential must not reach for a `GemmEngine` at all: taking one
/// as a parameter is precisely how the reviewed code ended up with `self`, the
/// GPU device, as its "reference". A free function cannot be handed an
/// accelerator by accident.
///
/// `cpu_reference_agrees_with_naive_cpu_gemm_engine` pins the two against each
/// other across three geometries including stride 2 with padding, so this cannot
/// drift from the established implementation. If that ever becomes a maintenance
/// burden, delete this and have the differential call `NaiveCpuGemmEngine`
/// DIRECTLY by name — never through a `dyn GemmEngine` the caller supplies.
///
/// # Contract (mirrors [`crate::GemmEngine::conv_transpose_2d`])
///
///   1. GEMM   `(S*OH*OW, OC) × (OC, IC*KH*KW)` → `(S*OH*OW, IC*KH*KW)`
///   2. col2im scatter → `(S, IC*IH*IW)` honouring stride and padding
///
/// `a_reshaped` is `(S*OH*OW, OC)` row-major, `weight_col` is `(OC, IC*KH*KW)`
/// row-major, and the result is `(S, IC*IH*IW)` row-major.
///
/// Accumulation is f64 and the result is rounded once to f32. That makes this a
/// POINT computation, not a certified bound: it is for detecting divergence
/// between two implementations of the same linear algebra, and it is never a
/// substitute for the certified error channel.
///
/// # Validation status
///
/// The arithmetic is pinned against hand-computed cases and cross-checked
/// against [`crate::NaiveCpuGemmEngine`].
///
/// The GPU kernel itself HAS now been compared against that same CPU engine on
/// real hardware: `wgpu_conv_transpose_matches_naive_cpu_reference`
/// (ny-gpu, `--features gpu-tests`) passes on this Metal adapter. That is the
/// first evidence that the differential's antecedent — "the reference and the
/// GPU agree" — can hold at all; blocker 2 asserted it never had.
///
/// It is evidence for ONE kernel on ONE geometry on ONE adapter, and it is not
/// a corpus. It does not qualify anything, and `corpus_is_green` stays `false`.
pub fn conv_transpose_2d_cpu_reference(
    a_reshaped: &[f32],
    weight_col: &[f32],
    params: &crate::ConvTranspose2dParams,
) -> Result<Vec<f32>, crate::NyError> {
    let s = params.num_specs;
    let (oc, ic) = (params.out_channels, params.in_channels);
    let (oh, ow) = (params.out_h, params.out_w);
    let (ih, iw) = (params.in_h, params.in_w);
    let (kh, kw) = (params.kernel_h, params.kernel_w);

    let rows = s * oh * ow;
    let kernel_cols = ic * kh * kw;
    if a_reshaped.len() != rows * oc {
        return Err(crate::NyError::shape_mismatch(
            vec![rows, oc],
            vec![a_reshaped.len()],
        ));
    }
    if weight_col.len() != oc * kernel_cols {
        return Err(crate::NyError::shape_mismatch(
            vec![oc, kernel_cols],
            vec![weight_col.len()],
        ));
    }

    let mut out = vec![0.0f64; s * ic * ih * iw];
    for spec in 0..s {
        for o_row in 0..oh {
            for o_col in 0..ow {
                let a_row = (spec * oh + o_row) * ow + o_col;
                // Fused: form one GEMM output element and scatter it
                // immediately, so no (S*OH*OW, IC*KH*KW) intermediate is ever
                // materialized — that buffer is what makes the CPU path slow.
                for c in 0..kernel_cols {
                    let k_w = c % kw;
                    let k_h = (c / kw) % kh;
                    let in_ch = c / (kh * kw);

                    let mut acc = 0.0f64;
                    for o_ch in 0..oc {
                        acc += f64::from(a_reshaped[a_row * oc + o_ch])
                            * f64::from(weight_col[o_ch * kernel_cols + c]);
                    }
                    if acc == 0.0 {
                        continue;
                    }

                    // col2im: map (o_row, o_col, k_h, k_w) back to input space,
                    // dropping taps that fall outside the padded border.
                    let src_h = (o_row * params.stride_h + k_h).checked_sub(params.pad_h);
                    let src_w = (o_col * params.stride_w + k_w).checked_sub(params.pad_w);
                    let (Some(i_row), Some(i_col)) = (src_h, src_w) else {
                        continue;
                    };
                    if i_row >= ih || i_col >= iw {
                        continue;
                    }
                    out[((spec * ic + in_ch) * ih + i_row) * iw + i_col] += acc;
                }
            }
        }
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(out.into_iter().map(|v| v as f32).collect())
}

#[cfg(test)]
mod cpu_reference_tests {
    use super::*;
    use crate::ConvTranspose2dParams;

    fn params(
        s: usize,
        oc: usize,
        ic: usize,
        oh: usize,
        ow: usize,
        ih: usize,
        iw: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> ConvTranspose2dParams {
        ConvTranspose2dParams {
            num_specs: s,
            out_channels: oc,
            in_channels: ic,
            out_h: oh,
            out_w: ow,
            in_h: ih,
            in_w: iw,
            kernel_h: kh,
            kernel_w: kw,
            stride_h: sh,
            stride_w: sw,
            pad_h: ph,
            pad_w: pw,
        }
    }

    /// 1x1 kernel, unit stride, no padding: col2im is the identity map, so the
    /// result must be exactly `a * w` at each position — hand-checkable.
    #[test]
    fn identity_geometry_is_a_times_w() {
        let p = params(1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0);
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // (S*OH*OW, OC) = (4,1)
        let w = vec![10.0f32]; // (OC, IC*KH*KW) = (1,1)
        let got = conv_transpose_2d_cpu_reference(&a, &w, &p).unwrap();
        assert_eq!(got, vec![10.0, 20.0, 30.0, 40.0]);
    }

    /// Two output channels contract into one input tap: the reference must SUM
    /// over OC, which is the GEMM half of the contract.
    #[test]
    fn sums_over_output_channels() {
        let p = params(1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0);
        let a = vec![2.0f32, 3.0]; // (1,2) row-major: one row, two OC
        let w = vec![5.0f32, 7.0]; // (2,1): one column
        let got = conv_transpose_2d_cpu_reference(&a, &w, &p).unwrap();
        assert_eq!(got, vec![2.0 * 5.0 + 3.0 * 7.0]);
    }

    /// Padding must DROP taps that fall outside the input, not wrap or panic.
    /// With pad=1 and a 3x3 kernel over a 1x1 output/input, only the centre tap
    /// (k_h=1, k_w=1) lands in bounds; the other eight are discarded.
    #[test]
    fn padding_drops_out_of_bounds_taps() {
        let p = params(1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1);
        let a = vec![1.0f32];
        // (OC=1, IC*KH*KW=9): distinct weights so a wrong tap is visible.
        let w: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let got = conv_transpose_2d_cpu_reference(&a, &w, &p).unwrap();
        // c = k_h*KW + k_w = 1*3 + 1 = 4 -> w[4] = 5.0
        assert_eq!(got, vec![5.0], "only the centre tap is in bounds");
    }

    /// Stride spreads taps across the input grid; two output positions must land
    /// on disjoint input columns at stride 2.
    #[test]
    fn stride_spreads_taps_without_collision() {
        let p = params(1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 2, 0, 0);
        let a = vec![1.0f32, 2.0]; // two output columns
        let w = vec![1.0f32];
        let got = conv_transpose_2d_cpu_reference(&a, &w, &p).unwrap();
        // o_col=0 -> i_col 0; o_col=1 -> i_col 2. Column 1 untouched.
        assert_eq!(got, vec![1.0, 0.0, 2.0]);
    }

    #[test]
    fn cpu_reference_agrees_with_naive_cpu_gemm_engine() {
        use crate::{ConvTranspose2dParams, GemmEngine, NaiveCpuGemmEngine};
        // Several geometries, including stride>1 and padding — the cases where a
        // col2im goes silently wrong.
        let cases = [
            (
                2usize, 2usize, 3usize, 3usize, 3usize, 4usize, 4usize, 2usize, 2usize, 1usize,
                1usize, 0usize, 0usize,
            ),
            (1, 1, 1, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1),
            (2, 3, 2, 2, 2, 5, 5, 3, 3, 2, 2, 1, 1),
        ];
        for (s, oc, ic, oh, ow, ih, iw, kh, kw, sh, sw, ph, pw) in cases {
            let p = ConvTranspose2dParams {
                num_specs: s,
                out_channels: oc,
                in_channels: ic,
                out_h: oh,
                out_w: ow,
                in_h: ih,
                in_w: iw,
                kernel_h: kh,
                kernel_w: kw,
                stride_h: sh,
                stride_w: sw,
                pad_h: ph,
                pad_w: pw,
            };
            let a: Vec<f32> = (0..s * oh * ow * oc)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.37)
                .collect();
            let w: Vec<f32> = (0..oc * ic * kh * kw)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.19)
                .collect();
            let mine = conv_transpose_2d_cpu_reference(&a, &w, &p).unwrap();
            let theirs = NaiveCpuGemmEngine.conv_transpose_2d(&a, &w, &p).unwrap();
            assert_eq!(mine.len(), theirs.len(), "shape disagreement for {p:?}");
            for (i, (m, t)) in mine.iter().zip(theirs.iter()).enumerate() {
                assert!(
                    (m - t).abs() <= 1e-4 * t.abs().max(1.0),
                    "index {i} disagrees for {p:?}: mine={m} naive={t}"
                );
            }
        }
    }

    /// #s1-wgpu-unquarantine blocker 5. The enclosure reference must BRACKET
    /// the point reference — that is what stops a GPU bound one ULP tighter
    /// from being recorded as a violation and permanently revoking the gate.
    #[test]
    fn enclosure_reference_brackets_the_point_reference() {
        let p = params(2, 3, 2, 2, 2, 5, 5, 3, 3, 2, 2, 1, 1);
        let a: Vec<f32> = (0..2 * 2 * 2 * 3)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.37)
            .collect();
        let w: Vec<f32> = (0..3 * 2 * 3 * 3)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.19)
            .collect();

        let point = conv_transpose_2d_cpu_reference(&a, &w, &p).unwrap();
        let (lo, hi) = conv_transpose_2d_cpu_reference_enclosure(&a, &w, &p).unwrap();
        assert_eq!(lo.len(), point.len());
        for i in 0..point.len() {
            assert!(
                lo[i] <= point[i] && point[i] <= hi[i],
                "index {i}: enclosure [{}, {}] must contain the point {}",
                lo[i],
                hi[i],
                point[i]
            );
            assert!(
                lo[i] < hi[i],
                "index {i}: the interval must be non-degenerate"
            );
        }
    }

    /// The whole point of blocker 5: a GPU bound that merely rounds differently
    /// must NOT revoke the gate, while one that is genuinely narrower still must.
    /// This is why the fix widens the reference instead of loosening `encloses`.
    #[test]
    fn rounding_noise_survives_but_a_narrower_bound_still_violates() {
        let p = params(1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0);
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![10.0f32];
        let (ref_lo, ref_hi) = conv_transpose_2d_cpu_reference_enclosure(&a, &w, &p).unwrap();

        // A GPU bound one ULP outside the reference on each side: valid, merely
        // rounded differently. Against the POINT reference this direction was
        // fine, but a GPU value one ULP INSIDE used to violate — now the
        // reference itself admits that uncertainty.
        let gpu_lo: Vec<f32> = ref_lo.iter().map(|v| v.next_down()).collect();
        let gpu_hi: Vec<f32> = ref_hi.iter().map(|v| v.next_up()).collect();
        assert_eq!(
            encloses(&gpu_lo, &gpu_hi, &ref_lo, &ref_hi),
            EnclosureVerdict::Encloses,
            "a GPU bound wider than the enclosure must not revoke the gate"
        );

        // A genuinely NARROWER bound — the false-proof direction — must still be
        // caught. Widening the reference must not have blunted this.
        let narrow_lo: Vec<f32> = ref_lo.iter().map(|v| v + 1.0).collect();
        assert!(
            matches!(
                encloses(&narrow_lo, &ref_hi, &ref_lo, &ref_hi),
                EnclosureVerdict::Violates { .. }
            ),
            "a narrower-than-truth GPU bound is the false-proof case and must violate"
        );
    }

    #[test]
    fn shape_mismatch_is_refused_not_guessed() {
        let p = params(1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0);
        assert!(conv_transpose_2d_cpu_reference(&[1.0], &[1.0], &p).is_err());
        assert!(conv_transpose_2d_cpu_reference(&[1.0; 4], &[1.0; 2], &p).is_err());
    }
}

/// The reference as a sound ENCLOSURE rather than a point
/// (#s1-wgpu-unquarantine, blocker 5).
///
/// # Why a point reference makes the gate unusable
///
/// `encloses` requires `gpu_lower <= ref_lower` and `gpu_upper >= ref_upper`.
/// Against a POINT reference that means a GPU bound even one ULP tighter counts
/// as a violation — and violations are monotone and permanent, so the very first
/// comparison very likely revoked the historical corpus for the process
/// lifetime. That design was fail-closed, so this was never unsound, but it
/// could never go green either.
///
/// The fix is to widen the REFERENCE, never to add a tolerance band to
/// `encloses`. A tolerance band would blunt the one check that detects a GPU
/// bound narrower than the truth, which is the false-proof direction and the
/// only thing this gate exists to catch. Widening the reference instead keeps
/// `encloses` exact: what changes is that the reference now honestly admits its
/// own uncertainty.
///
/// # What the interval covers
///
/// The returned `(lower, upper)` brackets the exact real-valued result:
///
/// * f64 accumulation error over the `k` products contributing to each output,
///   charged as the standard `k · 2^-53 · Σ|terms|` Higham bound — computed from
///   a running absolute sum, not assumed;
/// * the final f64 → f32 rounding, applied OUTWARD via [`f32::next_down`] /
///   [`f32::next_up`] so the stored endpoints cannot fall inside the true value.
///
/// So a GPU result that differs from the truth only by f32 rounding no longer
/// registers as a violation, while one that is genuinely narrower still does.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn conv_transpose_2d_cpu_reference_enclosure(
    a_reshaped: &[f32],
    weight_col: &[f32],
    params: &crate::ConvTranspose2dParams,
) -> Result<(Vec<f32>, Vec<f32>), crate::NyError> {
    let s = params.num_specs;
    let (oc, ic) = (params.out_channels, params.in_channels);
    let (oh, ow) = (params.out_h, params.out_w);
    let (ih, iw) = (params.in_h, params.in_w);
    let (kh, kw) = (params.kernel_h, params.kernel_w);

    let rows = s * oh * ow;
    let kernel_cols = ic * kh * kw;
    if a_reshaped.len() != rows * oc {
        return Err(crate::NyError::shape_mismatch(
            vec![rows, oc],
            vec![a_reshaped.len()],
        ));
    }
    if weight_col.len() != oc * kernel_cols {
        return Err(crate::NyError::shape_mismatch(
            vec![oc, kernel_cols],
            vec![weight_col.len()],
        ));
    }

    let n = s * ic * ih * iw;
    let mut acc = vec![0.0f64; n];
    // Running Σ|term| and term count per output, so the error charge is measured
    // rather than guessed.
    let mut abs_acc = vec![0.0f64; n];
    let mut terms = vec![0u64; n];

    for spec in 0..s {
        for o_row in 0..oh {
            for o_col in 0..ow {
                let a_row = (spec * oh + o_row) * ow + o_col;
                for c in 0..kernel_cols {
                    let k_w = c % kw;
                    let k_h = (c / kw) % kh;
                    let in_ch = c / (kh * kw);

                    let src_h = (o_row * params.stride_h + k_h).checked_sub(params.pad_h);
                    let src_w = (o_col * params.stride_w + k_w).checked_sub(params.pad_w);
                    let (Some(i_row), Some(i_col)) = (src_h, src_w) else {
                        continue;
                    };
                    if i_row >= ih || i_col >= iw {
                        continue;
                    }
                    let dst = ((spec * ic + in_ch) * ih + i_row) * iw + i_col;

                    for o_ch in 0..oc {
                        let prod = f64::from(a_reshaped[a_row * oc + o_ch])
                            * f64::from(weight_col[o_ch * kernel_cols + c]);
                        acc[dst] += prod;
                        abs_acc[dst] += prod.abs();
                        terms[dst] += 1;
                    }
                }
            }
        }
    }

    const EPS_F64: f64 = f64::EPSILON / 2.0; // 2^-53, the f64 unit roundoff
    let mut lower = vec![0.0f32; n];
    let mut upper = vec![0.0f32; n];
    for i in 0..n {
        // Higham: |fl(Σ) − Σ| ≤ k·u·Σ|terms| / (1 − k·u); the denominator is
        // ~1 for any k that fits in memory, and dropping it only WIDENS.
        let k = terms[i] as f64;
        let err = k * EPS_F64 * abs_acc[i];
        let lo = acc[i] - err;
        let hi = acc[i] + err;
        // Round OUTWARD through the f64 → f32 narrowing.
        lower[i] = (lo as f32).next_down();
        upper[i] = (hi as f32).next_up();
    }
    Ok((lower, upper))
}
