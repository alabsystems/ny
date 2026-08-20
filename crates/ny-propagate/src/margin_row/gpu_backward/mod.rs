// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified f32 GPU lane backward — EFT/double-single (#margin-row-gpu-eft).
//!
//! DESIGN + SKELETON delivery: the typed seam API, the ds32 host twin, and
//! the EFT kernel sources. The device transaction itself lands in a dedicated
//! future session (the moat risk demands it); until then [`run_pass_certified`]
//! refuses with [`Refusal::Unimplemented`] EVEN WHEN ARMED, so this module
//! cannot touch a bound or a verdict. Full design, failure diagnosis of the
//! predecessor seam, and the moat protocol:
//! `docs/GPU_CERTIFIED_LANE_BACKWARD_DESIGN_2026-08-19.md`.
//!
//! # Why this exists when `gpu_seam` already does (the non-repeat contract)
//!
//! The existing seam (`NY_MARGIN_ROW_GPU=1`, `gpu_seam.rs`) is measured
//! VERDICT-BREAKING: it turns banked-unsat `idx_6659` into `unknown`
//! regardless of sweep budget. Three diagnosed defects, each a hard design
//! requirement here (design section 2/3):
//!
//! 1. **Precision (the budget-independent killer)**: the device publishes an
//!    A-PRIORI f32 Higham radius (`gamma_k * sum|A||W|`), measured
//!    0.088–0.126 absolute vs ~1e-5 actual on the cifar100 fold — above the
//!    lane's own closing margins (~1e-2..1e-1), so every child bound loosens
//!    below zero and the tree stops closing at ANY budget. ⇒ R1: certified
//!    error is A-POSTERIORI (measured EFT residual) wherever the accumulation
//!    permits, and the value path is double-single (f64-comparable). Sound
//!    both ways; only this direction keeps the proofs.
//! 2. **Shared-fate device state**: the seam installs the caller's deadline
//!    on the PROCESS-GLOBAL backend (`gpu_seam.rs` dispatch,
//!    `GpuCrownBackendDeadlineScope::set`) while the internal verifier's
//!    sweep has fallible deadline checks AFTER queue submission
//!    (`intermediate_sweep.rs`), whose early return permanently poisons the
//!    device memory ledger ("exited after submission without a final drain").
//!    ⇒ R2/R3: submit→drain is one unconditional critical section; deadlines
//!    are host-side only; failure marks a LANE-LOCAL latch
//!    ([`mark_channel_dead`]) and nothing else.
//! 3. **Serialized-device stalls**: refusal paths must not block on the GPU
//!    mutex the sweep holds for seconds. ⇒ R4: admission is non-blocking;
//!    a busy device is [`Refusal::Busy`] and the caller runs the CPU pass.
//!
//! # Fail-closed rules
//!
//! * DARK unless `NY_MARGIN_ROW_GPU_EFT` is exactly `"1"` (default OFF).
//!   Promotion to a typed preset key is an implementation-session obligation
//!   (env levers cannot fire in competition).
//! * Any `Err` from any entry means: run the untouched CPU pass, bit-for-bit
//!   (the `engine::run_seamed` contract). No refusal repairs anything.
//! * Verdict authority is gated on the moat protocol (design section 7) AND
//!   enforced by the type system at runtime (adversarial-review item 2 — the
//!   M1 bit gate is BLOCKING, not advisory): [`run_transaction`], the only
//!   `PassOut` producer in this module, requires an
//!   [`authority::VerdictAuthority`], whose sole constructor consumes an
//!   [`authority::DeviceParityProof`] — mintable only by the once-per-process
//!   ON-DEVICE bit-parity self-check (which also requires the resolved
//!   DenormPreserve channel ON, item 3). On top of that: M2 shadow enclosure
//!   on the moat rows x3, M3 same-binary A/B, and the ported error-floor and
//!   realization-probe guards at every admitted pass.
//!
//! # Engagement telemetry (rule R9)
//!
//! Arming prints `[margin-row-gpu-eft] armed ...` once per process; every
//! DISTINCT refusal reason prints once per process. A measurement without the
//! armed line is not a measurement of this lever. The banner embeds
//! [`PROVENANCE_MARKER`] for the R8 binary-provenance check
//! (`strings <binary> | grep margin-row-gpu-eft`).

// SKELETON: the lane call-site wiring (`engine::run_seamed`-style) lands in
// the dedicated implementation session, so nothing outside the tests calls
// into this module yet. Drop this allow when the transaction lands.
#![allow(dead_code)]

use std::sync::OnceLock;
use std::time::Instant;

use super::engine::{BackwardEngine, DomainGates, LaneDir, PassOut, Seed};

use authority::VerdictAuthority;

pub(crate) mod authority;
pub(crate) mod ds;

#[cfg(test)]
mod tests;

/// R8 binary-provenance marker: lands in `.rodata` via the armed banner.
/// Bump the suffix on every landing that changes behavior.
/// v2: adversarial-review resolution — authority chain (M1 made blocking),
/// denorm-preserve admission, gate-transform intercept pair.
pub(crate) const PROVENANCE_MARKER: &str = "margin-row-gpu-eft-skeleton-v2";

/// Pre-registered M2/M3 KILL criterion (adversarial-review item 1): the
/// literal substring of the internal verifier sweep's post-submit poison log
/// (`ny-gpu/src/wgpu_device/ops/intermediate_sweep.rs:277`). The sweep checks
/// its OWN `request.deadline` AFTER queue submission (`intermediate_sweep.rs:845`,
/// `:865`; armed at `:803` via `CallLocalCrownDeadlineScope` — NOT the backend
/// lease the old seam installed), so ANY added GPU contention — including this
/// lane's dedicated-queue traffic — can push it past that deadline between
/// submit and drain, and one trip poisons the device memory ledger to
/// `usize::MAX` for the rest of the process. R3 alone therefore does NOT
/// remove the trigger. Until the implementation session applies R2 to the
/// sweep itself (hoist those checks before submission or drain
/// unconditionally — a REQUIRED precondition for M3, design section 9), the
/// moat harness MUST grep every M2 shadow / M3 A/B run log for this constant:
/// any hit = KILL, the measured section-2.2 failure recurring. `tests.rs`
/// pins it against the sweep source so a reworded log line cannot silently
/// blind the criterion.
pub(crate) const SWEEP_POST_SUBMIT_KILL_LINE: &str =
    "exited after submission without a final drain";

/// EFT/double-single primitives (fma TwoProduct, fma-barrier TwoSum, ds ops).
/// Concatenated before every consumer kernel at pipeline build. Host twin:
/// [`ds`] — the M1 moat gate is bit-identity between the two.
#[allow(dead_code)] // consumed by the implementation session's transaction
pub(crate) const DS_PRIMITIVES_WGSL: &str = include_str!("kernels/ds_primitives.wgsl");

/// ReLU gate backward transform (design section 5, kernel 2).
#[allow(dead_code)] // consumed by the implementation session's transaction
pub(crate) const GATE_TRANSFORM_WGSL: &str = include_str!("kernels/gate_transform.wgsl");

/// Transposed-conv gather backward (design section 5, kernel 3).
#[allow(dead_code)] // consumed by the implementation session's transaction
pub(crate) const CONV_BACKWARD_WGSL: &str = include_str!("kernels/conv_backward.wgsl");

/// Why the certified GPU backward declined this pass. Every variant means the
/// caller MUST run the exact CPU backward; none is recoverable in-lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// `NY_MARGIN_ROW_GPU_EFT` is not exactly `"1"`.
    Disabled,
    /// The on-device M1 self-check and the device transaction do not exist
    /// yet (skeleton delivery). Emitted by the AUTHORITY GATE
    /// ([`device_authority`]), which runs after the pure admission predicates
    /// (channel latch, mode/shape — no device access, no verdict math; kept
    /// first so refusal telemetry reports the real reason) and STRICTLY
    /// before anything that can produce a [`PassOut`]: [`run_transaction`]
    /// requires an [`authority::VerdictAuthority`], so a prematurely-wired
    /// call site cannot reach verdict math — a property of the types now, not
    /// of comment discipline (adversarial-review item 5).
    Unimplemented,
    /// A prior device failure latched this lane's channel dead for the rest
    /// of the process ([`mark_channel_dead`]). Lane-local by design (R3):
    /// nothing shared is poisoned, and there is deliberately no reset.
    ChannelDead,
    /// The device (or its exclusive token) is held elsewhere. Non-blocking by
    /// design (R4): the caller runs the CPU pass instead of waiting.
    Busy,
    /// The pass is not in the certified-outward mode.
    NotOutward,
    /// Zero rows, or a seed width disagreement.
    Rows,
    /// The op list / gates / seed cannot be expressed at the kernel boundary.
    Unmappable(&'static str),
    /// The lane's deadline expired before submission, or the result returned
    /// after it (checked HOST-SIDE only — R3: never installed on a backend).
    Deadline,
    /// The device returned an error. Also latches [`mark_channel_dead`].
    Device,
    /// The returned payload failed the structural/finite check.
    Payload,
    /// Certified-error floor guard trip (model-ball floor; ported from
    /// `gpu_seam::error_floor_ok` per design section 7).
    ErrorFloor,
    /// Realization-probe guard trip: a published bound excluded a realized
    /// value of the functional — an unsoundness signal, never rate-limited.
    Probe,
    /// The NaN/Inf firewall rejected the converted pass.
    NonFinite,
}

impl Refusal {
    /// Stable small tag for the once-per-reason diagnostic. NOT verdict input.
    fn tag(self) -> u32 {
        match self {
            Refusal::Disabled => 0,
            Refusal::Unimplemented => 1,
            Refusal::ChannelDead => 2,
            Refusal::Busy => 3,
            Refusal::NotOutward => 4,
            Refusal::Rows => 5,
            Refusal::Unmappable(_) => 6,
            Refusal::Deadline => 7,
            Refusal::Device => 8,
            Refusal::Payload => 9,
            Refusal::ErrorFloor => 10,
            Refusal::Probe => 11,
            Refusal::NonFinite => 12,
        }
    }
}

/// Pure arming predicate — the SPEC of the gate, unit-tested.
#[inline]
fn armed_from_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Latched raw `NY_MARGIN_ROW_GPU_EFT` string, read once through the ny-levers
/// chokepoint (latch the STRING once, derive the DECISION per call — the
/// `gpu_seam::env_raw` discipline). `read_raw` rather than `read` because
/// `armed_from_raw` is the unit-tested SPEC of this gate's arming rule and
/// must stay in the production path instead of being restated centrally.
fn env_raw() -> Option<&'static str> {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    RAW.get_or_init(|| ny_levers::read_raw(&ny_levers::decls::margin_row::MARGIN_ROW_GPU_EFT))
        .as_deref()
}

/// Is the certified GPU backward armed?
#[inline]
pub(crate) fn enabled() -> bool {
    armed_from_raw(env_raw())
}

/// Lane-local channel-death latch (design R3).
///
/// A device failure marks THIS lane dead for the rest of the process and
/// touches nothing else — expressly NOT the sweep's memory ledger, whose
/// `usize::MAX` poisoning is one of the defects this module exists to not
/// repeat. Fail-closed: there is no reset, and every subsequent admission
/// refuses with [`Refusal::ChannelDead`].
fn channel_latch() -> &'static OnceLock<&'static str> {
    static DEAD: OnceLock<&'static str> = OnceLock::new();
    &DEAD
}

/// Latch the lane's channel dead. Idempotent; the FIRST reason wins.
#[allow(dead_code)] // wired by the implementation session's device transaction
pub(crate) fn mark_channel_dead(reason: &'static str) {
    if channel_latch().set(reason).is_ok() {
        // Always printed: channel death is a soundness-adjacent signal, not a
        // profiling note. Once per process by construction (OnceLock).
        eprintln!("[margin-row-gpu-eft] channel dead: {reason}");
    }
}

/// The latched death reason, if any.
pub(crate) fn channel_dead() -> Option<&'static str> {
    channel_latch().get().copied()
}

/// Print the armed banner once per process (engagement telemetry, rule R9).
fn note_armed_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!("[margin-row-gpu-eft] armed ({PROVENANCE_MARKER}): all passes refuse Unimplemented until the moat session grants authority");
    });
}

/// Report each DISTINCT refusal reason once per process (rate-limited by a
/// tag bitmask — the `gpu_seam::note_refusal` lesson: a refusal counter with
/// no reason attached costs a measurement cycle).
fn note_refusal(refusal: Refusal) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    let bit = 1u32 << refusal.tag();
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        eprintln!("[margin-row-gpu-eft] refused: {refusal:?}");
    }
}

/// Everything the certified pass needs beyond the CPU lane's arguments.
///
/// Deliberately NOT `gpu_seam::SeamCtx`: the deadline here is a HOST-SIDE
/// value consulted only between transactions (before submit / after drain,
/// design section 6) and is never installed on any backend object — the type
/// distinction keeps that contract from silently eroding into the
/// `GpuCrownBackendDeadlineScope` pattern that poisoned the shared device.
#[derive(Default, Clone, Copy)]
pub(crate) struct LaneCtx<'y> {
    /// Per-head-neuron magnitude bound `max(|ly_j|, |uy_j|)` over the y-box
    /// the seed was built against (the `seed_penalty` contract, unchanged
    /// from `gpu_seam`).
    #[allow(dead_code)] // consumed once the transaction exists
    pub(crate) y_abs: Option<&'y [f64]>,
    /// The caller's real deadline. Host-side checks only (R3).
    #[allow(dead_code)] // consumed once the transaction exists
    pub(crate) deadline: Option<Instant>,
}

/// One certified pass, GPU-authoritative on `Ok`.
///
/// CONTRACT (identical shape to `engine::run_seamed`): on `Ok` the returned
/// [`PassOut`] is a drop-in replacement for
/// `BackwardEngine::run(seed, dom, dir, None, false)` — the caller
/// concretizes it with the lane's own unchanged f64 `concretize_*`. On ANY
/// `Err` the caller MUST run that exact CPU call; the refusal carries no
/// partial result and grants no authority.
///
/// SKELETON: admission checks are real; the authority gate then refuses
/// [`Refusal::Unimplemented`] (the on-device self-check dispatch does not
/// exist yet). The device transaction, plan building, guard ports and the
/// M2 shadow gate land in the dedicated implementation session (design
/// sections 5–7).
#[allow(dead_code)] // call-site wiring is the implementation session's
pub(crate) fn run_pass_certified(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    ctx: &LaneCtx<'_>,
) -> Result<PassOut, Refusal> {
    if !enabled() {
        return Err(Refusal::Disabled);
    }
    note_armed_once();
    let result = run_pass_armed(eng, seed, dom, dir, ctx);
    if let Err(refusal) = result.as_ref() {
        note_refusal(*refusal);
    }
    result
}

/// The armed path with the env latch already decided, so tests and the future
/// shadow mode can drive it without touching the process-wide `OnceLock`.
fn run_pass_armed(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    ctx: &LaneCtx<'_>,
) -> Result<PassOut, Refusal> {
    if channel_dead().is_some() {
        return Err(Refusal::ChannelDead);
    }
    admission_shape(
        eng.root.mode.outward(),
        seed.s.ncols(),
        seed.s.nrows(),
        eng.net.n_y,
    )?;
    let auth = device_authority()?;
    run_transaction(auth, eng, seed, dom, dir, ctx)
}

/// The authority gate (design section 6 "admit"): once per process, cached
/// like `ny_core::eft::eft_available`, the module must dispatch the
/// `double_single_probe` adversarial lanes through `ds_primitives.wgsl` on
/// the live device and let [`authority::DeviceParityProof::qualify`]
/// bit-compare the readback against the [`ds`] twin (also checking the
/// resolved denorm-preserve channel). Only that proof can mint the
/// [`VerdictAuthority`] every transaction requires — the M1 gate is BLOCKING
/// at runtime on the running device/driver, not a one-time qualification
/// session (adversarial-review item 2).
fn device_authority() -> Result<&'static VerdictAuthority, Refusal> {
    static AUTHORITY: OnceLock<Option<VerdictAuthority>> = OnceLock::new();
    match AUTHORITY.get_or_init(run_device_self_check) {
        Some(auth) => Ok(auth),
        None => Err(match channel_dead() {
            // A failed self-check latched the channel; report that, forever.
            Some(_) => Refusal::ChannelDead,
            None => Refusal::Unimplemented,
        }),
    }
}

/// SKELETON: the probe dispatch needs the drain-safe device transaction
/// machinery (implementation session), so no authority can exist yet and the
/// module stays dark by construction. The implementation session replaces
/// this body with: acquire the lane's device handle, dispatch the probe
/// lanes, read back, `DeviceParityProof::qualify(...)` (any mismatch latches
/// the channel dead in there), `VerdictAuthority::grant(...)`.
fn run_device_self_check() -> Option<VerdictAuthority> {
    None
}

/// The drain-safe transaction (design section 6): the ONLY function in this
/// module that can produce a [`PassOut`], and it is unreachable without a
/// [`VerdictAuthority`] — which only the on-device parity self-check can
/// mint. SKELETON: refuses; the encoder/submit/drain/publish state machine
/// lands in the implementation session.
fn run_transaction(
    authority: &VerdictAuthority,
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    ctx: &LaneCtx<'_>,
) -> Result<PassOut, Refusal> {
    // Deliberately unused until the transaction exists; named so the
    // signature — including the authority requirement — is already final.
    let _ = (authority.parity_lanes(), eng, seed, dom, dir, ctx);
    Err(Refusal::Unimplemented)
}

/// Pure admission predicate (unit-tested without an engine): certified-
/// outward mode only, and the seed must be a non-empty `(n_y, R)` matrix.
fn admission_shape(
    outward: bool,
    rows: usize,
    seed_rows: usize,
    n_y: usize,
) -> Result<(), Refusal> {
    if !outward {
        return Err(Refusal::NotOutward);
    }
    if rows == 0 || n_y == 0 || seed_rows != n_y {
        return Err(Refusal::Rows);
    }
    Ok(())
}
