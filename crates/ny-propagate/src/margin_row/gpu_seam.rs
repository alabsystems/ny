// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! AUTHORITATIVE, dark, fail-closed GPU seam for the margin-row twin-wall
//! lane (#twinwall, #margin-row-gpu).
//!
//! # What changed since the widen-only prototype
//!
//! The earlier seam (`16c8118b`, never on main) could only compute
//! `min(cpu, gpu)`, because the certified GPU entries returned CONCRETIZED
//! BOUNDS and [`GpuCrownLayer`] had nowhere to carry the lane's certified
//! BN-fold terms. Both holes are now closed at the trait boundary:
//!
//! * [`CertifiedWeightError`] rides on `GpuCrownLayer::{Linear, Conv2d}`, so
//!   the DEVICE charges `ConvOp::weight_rel_err` and `ConvOp::bias_err` — the
//!   exact terms the CPU lane charges at `engine.rs` (the conv arm's
//!   `g = gamma_n(k_bwd+2) + weight_rel_err + gamma_n(k_bwd+2)*weight_rel_err`
//!   composition, plus `(|L| + E) * bias_err` into `eb`);
//! * `crown_backward_gpu_seeded_sound_coeffs` returns
//!   [`CertifiedCoeffs`] — affine coefficients AND their certified error, NOT a
//!   concretized bound — so the lane keeps its own concretization
//!   (`BackwardEngine::concretize`, f64, directed-outward, over the lane's own
//!   f64 root box) completely unchanged.
//!
//! The seam therefore REPLACES the CPU backward walk for an admitted pass
//! instead of shadowing it. That is the whole point: the CPU walk is the thing
//! that costs the time.
//!
//! # The soundness argument (why an authoritative GPU bound is legitimate)
//!
//! The lane's contract is: the published lower bound is `<=` the exact real
//! value of the seeded functional for every input in the root box (and, for a
//! piece-fixed domain, for every input in that domain). It is NOT "the GPU
//! must agree with the CPU". The GPU pass satisfies the contract on its own:
//!
//! 1. **Every relaxation the device is handed is a valid over-approximation of
//!    the corresponding real op.** ReLU lower lines are `alpha*x` with
//!    `alpha in [0, 1]` — sound for any such alpha, and `alpha` is 0/1 here so
//!    the f32 downcast is EXACT. ReLU upper lines are repaired after the
//!    downcast by [`outward_gate_f32`]: relu is convex and the line is affine,
//!    so `line >= relu` on `[l, u]` iff it holds at `l` and at `u`; the repair
//!    bumps the intercept UP until both endpoint residuals are non-negative.
//!    A downcast upper line can therefore never dip below the true ReLU.
//! 2. **Every parameter perturbation is charged.** The f64->f32 downcast of a
//!    weight is a relative perturbation `<= 2^-24`; the BN fold is a relative
//!    perturbation `<= weight_rel_err`. Relative to the exact weight they
//!    compose to `R = rho + u32 + rho*u32`; the device contract is relative to
//!    the SUPPLIED f32 weight, so [`CertifiedWeightError::weight_rel_err`]
//!    carries `R/(1-R)`, rounded UP into f32 (and the seam refuses `R >= 1`).
//!    Biases carry
//!    `max_ch(bias_err) + u32*max_ch(|bias|)`, rounded up, as an absolute term.
//! 3. **The seed's own error never reaches the device** — it is CONCRETIZED at
//!    the head pre-activation instead (see [`seed_penalty`]): a seed
//!    coefficient discrepancy `d_j` changes the functional by at most
//!    `sum_j d_j * yabs_j` over the caller's certified y-box, so it becomes a
//!    pure additive constant in `eb`. This covers BOTH the lane's own
//!    `Seed::e` and the exact f32 downcast residual of the seed itself.
//! 4. **The input box grants no coefficient authority.** It is rounded OUTWARD
//!    for signature parity, but the coefficient egress MUST ignore it (and all
//!    abs-max tables): [`CertifiedCoeffs`] is a box-independent coefficient
//!    enclosure. The lane later concretizes over its own f64 root box.
//! 5. **The lane concretizes, not the device.** The returned coefficients and
//!    their certified error re-enter `PassOut` unchanged (f32->f64 is exact)
//!    and `concretize` applies the same `E . xabs + eb` penalty, `gamma_n`
//!    envelope and double outward rounding it always did.
//! 6. **A residual `Add` is decomposed, not approximated.** `out = F(z) + P(z)`
//!    is an EXACT identity whenever both branches are pure functions of one
//!    common ancestor `z`, which [`Builder::trail`] verifies before the block is
//!    emitted; the backend's fork/merge computes
//!    `A_in = backward_F(A) + backward_P(A)` with the incoming bias counted
//!    once, and charges the merge's own f32 addition into the certified error.
//!    A branch that is not such a function (a nested `Add`, no common ancestor,
//!    a width disagreement, `z + z`) is a refusal.
//!
//! # Residual nets (why this module exists at all)
//!
//! Every cifar100 / tinyimagenet net the lane must accelerate is a ResNet. The
//! first armed measurement of this seam reported `gpu_seam_ok=0
//! gpu_seam_refused=2` and an unchanged runtime for exactly that reason: the
//! flat `crown_backward_gpu_seeded_sound_coeffs` entry has no residual form, so
//! the plan builder refused at the first `Add` and the seam never ran. The
//! `GpuCrownBackward::crown_backward_gpu_resnet_sound_coeffs` egress publishes
//! the COMPOSED segment frontier instead, and [`build_plan`] maps the lane's
//! `Add` into the `GpuResnetSegment` shape it consumes.
//!
//! # The guard (authority still needs a falsifier)
//!
//! Being sound *by argument* is not the same as the device *being* correct, so
//! every admitted pass is checked by three cheap, fail-closed guards. A trip
//! never repairs anything: it discards the device result and the caller runs
//! the exact CPU path.
//!
//! * **Structural** — dimensions, finiteness, non-negative error lanes.
//! * **Certified-error floor** (`O(n_in * R)`, the same reduction
//!   `concretize` already performs). For a layer whose weights are only known
//!   to relative accuracy `rho`, ANY certification valid over that whole
//!   `rho`-ball must charge at least `rho * sum_j |w_ij| |a_j| >= rho *
//!   |a_out_i|` at that layer, and error propagates to the input through the
//!   ABS operator, which dominates the signed one — so the final penalty must
//!   satisfy `pen + eb >= rho * sum_i |a_i| * xabs_i`. The coefficient egress
//!   preserves this charge in its coefficient/bias radius channels; it does
//!   not discharge coefficient radii against a domain box before publication.
//!   Written with the published coefficients it becomes
//!   `P >= rho*T / (1 + rho)` — [`error_floor_ok`].
//! * **Realization probe** (one f64 forward pass, `O(net)` not `O(net*R)`).
//!   The box midpoint is a genuine member of the root box; the lane's own
//!   `forward_points` gives the exact `y` there, so `sum_j S[j,r] * y_j` is a
//!   REALIZED value of the very functional the pass bounds. A published lower
//!   bound above it is a proof of unsoundness. For a piece-fixed domain the
//!   probe first checks that the midpoint actually satisfies the fixes (via
//!   `pre_sel` pre-activations) and SKIPS if it does not — a non-member is not
//!   a counterexample.
//!
//! The floor guard costs a reduction the caller was going to do anyway; the
//! probe costs one forward pass against a backward pass over `R` columns. The
//! speedup survives both.
//!
//! # Domain batching
//!
//! The per-pass seam accelerates ONE domain's backward walk. The batching shape
//! intended to attack the measured ~32x workload gap on the deciding cifar100 /
//! tinyimagenet pools is a DOMAIN-BATCHED wide pass — see [`batch`], which
//! stacks frontier domains into chunked certified calls of at most 16 domains
//! behind `NY_MARGIN_ROW_GPU_BATCH=1`.
//!
//! # Fail-closed rules
//!
//! * DARK unless `NY_MARGIN_ROW_GPU=1` (exact string).
//! * Only [`RoundMode::Outward`](super::rounding::RoundMode) passes are seamed.
//! * Only the CERTIFIED coefficient entries are called (flat for a unary chain,
//!   segment for a residual net), and only through
//!   [`crate::sound_gpu_gate::gpu_crown_backward_route_with_deadline`] with a
//!   FINITE routing marker, so the seam never runs a cold device factory
//!   inside a verifier budget.
//! * Any refusal — no device, no authority, no coefficient egress, unmappable
//!   op, deadline, shape, guard trip — returns [`Refusal`] and the caller runs
//!   the untouched CPU pass.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use ndarray::Array2;
use ny_core::{
    CertifiedCoeffs, CertifiedWeightError, GpuCrownBackward, GpuCrownLayer, GpuCrownSeed,
    GpuResnetSegment,
};

use super::engine::{BackwardEngine, DomainGates, LaneDir, PassOut, Seed};
use super::net::{ConvOp, TwinOp};
use super::prof::{bump, bump_always, Counter};
use super::rounding::{certify_up, gamma_n, next_down, next_up, UNIT};
use crate::sound_gpu_gate::{
    gpu_crown_backend_honors_deadline, gpu_crown_backward_route_with_deadline,
    GpuCrownBackendDeadlineScope,
};

pub(crate) mod batch;

/// Hard bound on how many ops the plan builder will follow. A malformed op
/// list cannot make the builder loop: an overrun is a refusal, not a repair.
const MAX_PLAN_OPS: usize = 512;

/// Routing marker used when the caller has no deadline of its own.
///
/// [`gpu_crown_backward_route_with_deadline`] switches to the NON-initializing
/// resolver whenever it is handed `Some(_)`. The seam always hands it `Some`,
/// so it observes only an already-materialized process-global backend. Routing
/// only; the deadline installed on the backend is the caller's real one.
const ROUTE_ONLY_MARKER: Duration = Duration::from_hours(1);

/// f32 unit roundoff (2^-24): the relative ball of ONE round-to-nearest
/// binary32 conversion. Every parameter the seam downcasts is charged this on
/// top of its own certified fold error, and the downcast is then VERIFIED
/// against it element-wise (see [`to_f32_params_rel`]).
const U32_UNIT: f64 = 5.960_464_477_539_063e-8;

/// Relative slack allowed to the realization probe, so the probe's own f64
/// evaluation of the functional cannot false-trip a correct bound. Gross
/// unsoundness is orders of magnitude above this.
const PROBE_REL_SLACK: f64 = 1e-9;

/// Why the seam declined to produce an authoritative pass. Every variant means
/// the caller must run the exact CPU backward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// `NY_MARGIN_ROW_GPU` is not exactly `"1"`.
    Disabled,
    /// The pass is not in the certified-outward mode.
    NotOutward,
    /// Zero rows, or a seed/`y_abs` width disagreement.
    Rows,
    /// The lane's op list, gates or seed cannot be expressed at the
    /// `GpuCrownLayer` / `GpuResnetSegment` boundary. The payload names the
    /// exact blocker.
    Unmappable(&'static str),
    /// The seed carries a certified error (or is not f32-exact) and the caller
    /// supplied no head-pre-activation magnitude bound to concretize it into.
    SeedNeedsYAbs,
    /// No PREWARMED backend with verdict-grade sound GPU CROWN authority.
    NoAuthority,
    /// The backend has authority but does not implement coefficient egress.
    NoCoeffEgress,
    /// A live deadline exists and the backend does not advertise cooperative
    /// cancellation.
    DeadlineUnsupported,
    /// The device returned an error (including its own deadline expiry).
    Device,
    /// The payload failed the structural/finite/ordering check.
    Payload,
    /// The returned certified error is below the a-priori floor implied by the
    /// weight errors the device was told to charge.
    ErrorFloor,
    /// The published bound exceeded a REALIZED value of the functional.
    Probe,
    /// The lane's own NaN/Inf firewall rejected the converted pass.
    NonFinite,
}

impl Refusal {
    /// Stable small tag, for the once-per-reason diagnostic below. NOT a
    /// verdict input.
    fn tag(self) -> u32 {
        match self {
            Refusal::Disabled => 0,
            Refusal::NotOutward => 1,
            Refusal::Rows => 2,
            Refusal::Unmappable(_) => 3,
            Refusal::SeedNeedsYAbs => 4,
            Refusal::NoAuthority => 5,
            Refusal::NoCoeffEgress => 6,
            Refusal::DeadlineUnsupported => 7,
            Refusal::Device => 8,
            Refusal::Payload => 9,
            Refusal::ErrorFloor => 10,
            Refusal::Probe => 11,
            Refusal::NonFinite => 12,
        }
    }
}

/// Report each DISTINCT refusal reason ONCE per process, under
/// `NY_MARGIN_ROW_PROFILE=1`.
///
/// `gpu_seam_ok=0 gpu_seam_refused=2` with no reason attached is exactly what
/// cost a measurement cycle: the counters proved the seam never ran but not
/// WHY. The reason is one line, printed once, behind the profile lever, and
/// feeds nothing back into the pass.
fn note_refusal(refusal: Refusal) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    if !super::prof::enabled() {
        return;
    }
    let bit = 1u32 << refusal.tag();
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        eprintln!("[margin-row-gpu] seam refused: {refusal:?}");
    }
}

/// Pure arming predicate — the SPEC of the gate, unit-tested below.
#[inline]
fn armed_from_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Latched raw `NY_MARGIN_ROW_GPU` string, read once through the ny-levers
/// chokepoint (latch the STRING, derive the DECISION per call).
fn env_raw() -> Option<&'static str> {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    RAW.get_or_init(|| ny_levers::read_raw(&ny_levers::decls::sound_channel::MARGIN_ROW_GPU))
        .as_deref()
}

/// Is the seam armed?
#[inline]
pub(crate) fn enabled() -> bool {
    armed_from_raw(env_raw())
}

/// Everything the seam needs that the pure-CPU lane does not carry.
#[derive(Default, Clone, Copy)]
pub(crate) struct SeamCtx<'y> {
    /// Per-head-neuron magnitude bound `max(|ly_j|, |uy_j|)` over the y-box the
    /// seed was built against. Required whenever the seed carries a certified
    /// error or is not exactly representable in f32 (see [`seed_penalty`]).
    pub(crate) y_abs: Option<&'y [f64]>,
    /// The caller's real deadline, installed on the dispatched backend.
    pub(crate) deadline: Option<Instant>,
}

/// One admitted pass, GPU-authoritative.
///
/// On `Ok` the returned [`PassOut`] is a drop-in replacement for
/// `BackwardEngine::run(seed, dom, dir, None, false)`: same shapes, same
/// meaning, and the caller concretizes it with the lane's own unchanged
/// `concretize_lower` / `concretize_upper`.
pub(crate) fn run_pass(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    ctx: &SeamCtx<'_>,
) -> Result<PassOut, Refusal> {
    let (lower, upper) = run_pass_inner(eng, seed, dom, ctx, Some(dir))?;
    match dir {
        LaneDir::Lower => lower.ok_or(Refusal::Payload),
        LaneDir::Upper => upper.ok_or(Refusal::Payload),
    }
}

/// BOTH lanes from ONE device call — the `y_rows` refresh shape.
///
/// The device's `lower_*` / `upper_*` halves are exactly the lane's Lower and
/// Upper passes under this module's activation mapping, so the identity-seeded
/// refresh costs one dispatch instead of two.
pub(crate) fn run_pass_pair(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    ctx: &SeamCtx<'_>,
) -> Result<(PassOut, PassOut), Refusal> {
    let (lower, upper) = run_pass_inner(eng, seed, dom, ctx, None)?;
    Ok((
        lower.ok_or(Refusal::Payload)?,
        upper.ok_or(Refusal::Payload)?,
    ))
}

fn run_pass_inner(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    ctx: &SeamCtx<'_>,
    only: Option<LaneDir>,
) -> Result<(Option<PassOut>, Option<PassOut>), Refusal> {
    if !enabled() {
        return Err(Refusal::Disabled);
    }
    run_pass_armed_recorded(eng, seed, dom, ctx, only)
}

/// [`run_pass_armed`] plus the EXACT counter bookkeeping production performs.
///
/// Split out of [`run_pass_inner`] so a device test can drive the armed path
/// and still observe the very counters the integrator reads out of
/// `NY_MARGIN_ROW_PROFILE` (`gpu_seam_ok` / `gpu_seam_refused` /
/// `gpu_seam_guard_trip`) — the env latch is a process-wide `OnceLock`, so a
/// test cannot reach them through the gate.
pub(crate) fn run_pass_armed_recorded(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    ctx: &SeamCtx<'_>,
    only: Option<LaneDir>,
) -> Result<(Option<PassOut>, Option<PassOut>), Refusal> {
    match run_pass_armed(eng, seed, dom, ctx, only) {
        Ok(v) => {
            bump(Counter::GpuSeamOk, 1);
            Ok(v)
        }
        Err(refusal) => {
            bump(Counter::GpuSeamRefused, 1);
            note_refusal(refusal);
            if matches!(refusal, Refusal::ErrorFloor | Refusal::Probe) {
                // Always recorded: a guard trip is a soundness signal, not a
                // profiling note.
                bump_always(Counter::GpuSeamGuardTrip, 1);
            }
            Err(refusal)
        }
    }
}

/// [`run_pass_inner`] with the environment gate already decided, so the
/// oracles can drive the seam without touching the process-wide latch.
pub(crate) fn run_pass_armed(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    ctx: &SeamCtx<'_>,
    only: Option<LaneDir>,
) -> Result<(Option<PassOut>, Option<PassOut>), Refusal> {
    let net = eng.net;
    let root = eng.root;
    if !root.mode.outward() {
        return Err(Refusal::NotOutward);
    }
    let rows = seed.s.ncols();
    if rows == 0 || seed.s.nrows() != net.n_y || net.n_y == 0 {
        return Err(Refusal::Rows);
    }
    if seed
        .e
        .as_ref()
        .is_some_and(|e| e.raw_dim() != seed.s.raw_dim())
    {
        return Err(Refusal::Rows);
    }
    let (plan, node_abs, rho_star) = build_plan(eng, dom)?;
    let pen_seed = seed_penalty(seed, ctx.y_abs, net.n_y, rows)?;
    let gseed = build_seed(net.n_y, seed, rows)?;
    let (lo, hi) = build_box(root.lo.as_slice(), root.hi.as_slice(), net.n_in)?;
    let cc = dispatch(&plan, &node_abs, &gseed, &lo, &hi, ctx.deadline)?;

    let mut out: (Option<PassOut>, Option<PassOut>) = (None, None);
    for dir in [LaneDir::Lower, LaneDir::Upper] {
        if only.is_some_and(|d| d != dir) {
            continue;
        }
        let pass = convert_and_check(&cc, dir, net.n_in, rows, &pen_seed)?;
        if !error_floor_ok(&pass, root.xabs.as_slice(), rho_star) {
            return Err(Refusal::ErrorFloor);
        }
        probe_guard(eng, seed, dom, dir, &pass)?;
        match dir {
            LaneDir::Lower => out.0 = Some(pass),
            LaneDir::Upper => out.1 = Some(pass),
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Plan construction
// ---------------------------------------------------------------------------

/// The device-side description of one admitted pass.
///
/// A net with no residual `Add` is a unary [`Plan::Chain`] and goes to the flat
/// coefficient egress, EXACTLY as before this module learned about residuals
/// (bit-identical for every chain net). A net with a residual `Add` decomposes
/// into backward-order [`Plan::Segments`] and goes to the SEGMENT coefficient
/// egress. Every cifar100/tinyimagenet net is the second kind — before the
/// segment egress existed the seam refused 100% of them.
pub(crate) enum Plan {
    /// A unary layer chain, backward order.
    Chain(Vec<GpuCrownLayer>),
    /// A residual decomposition, backward order.
    Segments(Vec<GpuResnetSegment>),
}

/// Borrowed context for one plan build.
struct Builder<'a, 'b> {
    eng: &'a BackwardEngine<'b>,
    dom: Option<&'a DomainGates>,
}

impl Builder<'_, '_> {
    /// One trunk ReLU as a `GpuCrownLayer::Activation`, plus that ReLU's
    /// PRE-activation abs-max frontier `max(|l|, |u|)`.
    ///
    /// The lane's lower lane sends `v >= 0` through `alpha` (no intercept) and
    /// `v < 0` through `(s, c)`; the upper lane is the mirror. The device's
    /// `Activation` uses exactly that sign rule with
    /// `[lower_slope | upper_slope | lower_intercept | upper_intercept]`.
    ///
    /// # Why the host-side `node_abs` invariant exists
    ///
    /// `node_abs` is NOT an input to coefficient publication: at the resnet
    /// [`CertifiedCoeffs`] trait seam it exists only for signature parity and
    /// MUST be ignored. The batch builder nevertheless keeps this host-side
    /// list aligned with its ReLU provenance. [`retarget_plan`] recomputes it
    /// for each domain and requires equality, proving that domain retargeting
    /// changed only the gate triple rather than the root record. Two facts make
    /// the shared value a valid invariant:
    ///
    /// * `LayerGates::l`/`u` is the lane's own CERTIFIED outward pre-activation
    ///   box over the ROOT input box (`RootGates` is only built in
    ///   `RoundMode::Outward` here — [`run_pass_armed`] refuses otherwise);
    /// * a piece-fixed DOMAIN only ever restricts the region, so the ROOT box
    ///   remains a valid (conservative) frontier for every domain the lane
    ///   passes down. The gate override changes `alpha/s/c`, never `l`/`u`.
    ///
    /// The value is rounded UP into f32 before the host-side comparison. This
    /// list grants no authority to fold a coefficient radius into bias.
    fn activation(&self, li: usize) -> Result<(GpuCrownLayer, Vec<f32>), Refusal> {
        let rec = self
            .eng
            .root
            .layers
            .get(li)
            .ok_or(Refusal::Unmappable("relu layer index out of range"))?;
        let (alpha, s, c) = match self.dom.and_then(|d| d.layers.get(&li)) {
            Some(gv) => (&gv.alpha[..], &gv.s[..], &gv.c[..]),
            None => (&rec.alpha[..], &rec.s[..], &rec.c[..]),
        };
        if rec.n == 0
            || alpha.len() != rec.n
            || s.len() != rec.n
            || c.len() != rec.n
            || rec.l.len() != rec.n
            || rec.u.len() != rec.n
        {
            return Err(Refusal::Unmappable("relu gate width"));
        }
        let (a32, s32, c32) = outward_gate_f32(alpha, s, c, &rec.l, &rec.u)?;
        let mut node_abs = Vec::with_capacity(rec.n);
        for (l, u) in rec.l.iter().zip(&rec.u) {
            if !(l.is_finite() && u.is_finite()) {
                return Err(Refusal::Unmappable("non-finite pre-activation box"));
            }
            node_abs.push(f32_toward_pos_inf(l.abs().max(u.abs()))?);
        }
        Ok((
            GpuCrownLayer::Activation {
                lower_slope: a32,
                upper_slope: s32,
                lower_intercept: vec![0.0; rec.n],
                upper_intercept: c32,
                num_neurons: rec.n,
            },
            node_abs,
        ))
    }

    /// Collect the backward-order layers on the SINGLE-INPUT path from tensor
    /// `from` down to tensor `stop` (exclusive), appending each ReLU's
    /// host-only `node_abs` retargeting value in the SAME order and folding
    /// `rho*` outward.
    ///
    /// Residual BRANCHES are strictly unary: a nested `Add`, a `Gemm` or a
    /// `ChannelAffine` inside one is a refusal, not a repair.
    fn walk(
        &self,
        from: usize,
        stop: usize,
        out: &mut Vec<GpuCrownLayer>,
        node_abs: &mut Vec<Vec<f32>>,
        relus: &mut Vec<usize>,
        rho_star: &mut f64,
    ) -> Result<(), Refusal> {
        let mut tid = from;
        for _ in 0..MAX_PLAN_OPS {
            if tid == stop {
                return Ok(());
            }
            if tid == 0 {
                return Err(Refusal::Unmappable("branch reached the network input"));
            }
            let op = self
                .eng
                .net
                .ops
                .get(tid - 1)
                .ok_or(Refusal::Unmappable("tensor without a producer"))?;
            match op {
                TwinOp::Flatten { input } => tid = *input,
                TwinOp::Relu { input, layer } => {
                    let (act, abs) = self.activation(*layer)?;
                    out.push(act);
                    node_abs.push(abs);
                    relus.push(*layer);
                    tid = *input;
                }
                TwinOp::Conv(cv) => {
                    let (l, rho) = conv(cv)?;
                    *rho_star = rho_star.max(rho);
                    out.push(l);
                    tid = cv.input;
                }
                TwinOp::ChannelAffine { .. } => {
                    return Err(Refusal::Unmappable("ChannelAffine in a residual branch"))
                }
                TwinOp::Add { .. } => {
                    return Err(Refusal::Unmappable("nested Add in a residual branch"))
                }
                TwinOp::Gemm { .. } => {
                    return Err(Refusal::Unmappable("Gemm in a residual branch"))
                }
            }
        }
        Err(Refusal::Unmappable("residual branch exceeds the op bound"))
    }

    /// Tensor ids on the single-input path from `from` toward the input, in
    /// order, stopping at the network input or at the first `Add`.
    ///
    /// Used to find the residual block's COMMON ANCESTOR `z`: the device's
    /// fork/merge is a sound decomposition of `out = F(z) + P(z)` only when
    /// both branches are pure functions of ONE shared tensor. A branch that
    /// crosses an `Add` is not, so the trail stops there and the ancestor
    /// search fails closed.
    fn trail(&self, from: usize) -> Vec<usize> {
        let mut trail = vec![from];
        let mut tid = from;
        for _ in 0..MAX_PLAN_OPS {
            if tid == 0 {
                break;
            }
            let Some(op) = self.eng.net.ops.get(tid - 1) else {
                break;
            };
            let next = match op {
                TwinOp::Flatten { input }
                | TwinOp::Relu { input, .. }
                | TwinOp::ChannelAffine { input, .. }
                | TwinOp::Gemm { input, .. } => *input,
                TwinOp::Conv(cv) => cv.input,
                TwinOp::Add { .. } => break,
            };
            trail.push(next);
            tid = next;
        }
        trail
    }
}

/// Build the device-side program (backward order), the host-only per-Activation
/// `node_abs` retargeting invariant in FOLD ORDER, and the largest certified
/// relative weight error handed to any layer (the guard's `rho*`).
///
/// The lane seeds at the head pre-activation `y`, so the first backward layer
/// is the head `Gemm` itself (`L0[k] = sum_j W1[j,k]*S[j]` is exactly the
/// device's `A_new = A @ weight` for a row-major `(n_y, n_h)` weight), then the
/// trunk in reverse execution order.
///
/// # Fold order (host-side batch invariant)
///
/// The coefficient egress MUST ignore `node_abs`; it never consumes these
/// magnitudes to publish [`CertifiedCoeffs`]. The list is paired on the host
/// with the `k`-th `Activation` and ReLU provenance in backend fold order:
/// segments in order, layers within a segment in order, and — for a
/// `ResidualProj` — the whole `F` branch before the whole `P` branch. Emitting
/// all three from the same traversal lets [`retarget_plan`] prove that batching
/// changes only gate values. `plan_node_abs_is_in_backend_fold_order` pins that
/// host invariant.
fn build_plan(
    eng: &BackwardEngine<'_>,
    dom: Option<&DomainGates>,
) -> Result<(Plan, Vec<Vec<f32>>, f64), Refusal> {
    let (plan, node_abs, rho_star, _relus) = build_plan_full(eng, dom)?;
    Ok((plan, node_abs, rho_star))
}

/// [`build_plan`] plus the ROOT-GATE LAYER INDEX of every emitted `Activation`,
/// in the SAME fold order as `node_abs`.
///
/// The batched lane (`batch.rs`) needs it to re-gate the shared skeleton per
/// domain: for the `k`-th `Activation` it re-runs `Builder::activation` with
/// that domain's overrides on layer `relus[k]`. Deriving the list FROM THIS
/// TRAVERSAL — rather than re-walking the op list separately — is what makes a
/// drift between "the k-th Activation" and "the layer whose gates it carries"
/// unrepresentable. `plan_relu_layers_track_the_emitted_activations` pins it.
fn build_plan_full(
    eng: &BackwardEngine<'_>,
    dom: Option<&DomainGates>,
) -> Result<(Plan, Vec<Vec<f32>>, f64, Vec<usize>), Refusal> {
    let net = eng.net;
    let builder = Builder { eng, dom };
    let (w1, b1, (n_y, n_h)) = net.gemm1();
    if w1.len() != n_y * n_h || b1.len() != n_y || n_y == 0 || n_h == 0 {
        return Err(Refusal::Unmappable("head gemm shape"));
    }
    let mut rho_star = 0.0f64;
    // The head Gemm's weights are EXACT in the lane's own algebra (the gemm1
    // arm charges no weight_rel_err), so only the f32 downcast is charged.
    let head_err = weight_error(0.0, b1, &[])?;
    rho_star = rho_star.max(f64::from(head_err.weight_rel_err));
    let head = GpuCrownLayer::Linear {
        weight: Arc::from(to_f32_params_rel(w1, U32_UNIT)?),
        bias: Some(Arc::from(to_f32_params_rel(b1, U32_UNIT)?)),
        out_features: n_y,
        in_features: n_h,
        cert_err: head_err,
    };

    let Some(TwinOp::Gemm { input: g1_in, .. }) = net.ops.get(net.i_gemm1) else {
        return Err(Refusal::Unmappable("head gemm is not a Gemm"));
    };
    let mut tid = *g1_in;
    if net.tsize.get(tid).copied() != Some(n_h) {
        return Err(Refusal::Unmappable("head gemm input width"));
    }
    let mut segments: Vec<GpuResnetSegment> = Vec::new();
    let mut chain: Vec<GpuCrownLayer> = vec![head];
    let mut node_abs: Vec<Vec<f32>> = Vec::new();
    let mut relus: Vec<usize> = Vec::new();
    let mut saw_residual = false;
    let mut layer_count = 1usize;
    let mut steps = 0usize;
    while tid != 0 {
        steps += 1;
        if steps > MAX_PLAN_OPS {
            return Err(Refusal::Unmappable("trunk exceeds the op bound"));
        }
        let op = net
            .ops
            .get(tid - 1)
            .ok_or(Refusal::Unmappable("tensor without a producer"))?;
        match op {
            TwinOp::Flatten { input } => tid = *input,
            TwinOp::Relu { input, layer } => {
                let (act, abs) = builder.activation(*layer)?;
                chain.push(act);
                node_abs.push(abs);
                relus.push(*layer);
                layer_count += 1;
                tid = *input;
            }
            TwinOp::Conv(cv) => {
                let (l, rho) = conv(cv)?;
                rho_star = rho_star.max(rho);
                chain.push(l);
                layer_count += 1;
                tid = cv.input;
            }
            // A diagonal affine has no `GpuCrownLayer` form: expressing it as a
            // `Linear` would materialize an n x n identity-scaled matrix
            // (gigabytes at cifar100 trunk widths). Refuse.
            TwinOp::ChannelAffine { .. } => {
                return Err(Refusal::Unmappable("ChannelAffine has no GpuCrownLayer"))
            }
            // STRUCTURALLY UNREACHABLE on a validated `TwinNet`: `compile`
            // admits exactly two Gemms (`i_gemm2 == i_gemm1 + 2`) and both sit
            // AFTER the trunk, so the backward walk from `gemm1`'s input can
            // never meet one. Kept as a fail-closed guard, not a capability
            // gap — there is no trunk-Gemm shape for the segment egress to map.
            TwinOp::Gemm { .. } => return Err(Refusal::Unmappable("unexpected trunk Gemm")),
            TwinOp::Add { lhs, rhs } => {
                saw_residual = true;
                let (lhs, rhs) = (*lhs, *rhs);
                let width = net.tsize.get(tid).copied();
                if width.is_none()
                    || net.tsize.get(lhs).copied() != width
                    || net.tsize.get(rhs).copied() != width
                {
                    return Err(Refusal::Unmappable("residual add width"));
                }
                // Both branches must be pure functions of ONE common tensor
                // `z`; only then is `out = F(z) + P(z)` exact and the device's
                // fork/merge a sound decomposition of it.
                let right_trail = builder.trail(rhs);
                let z = builder
                    .trail(lhs)
                    .into_iter()
                    .find(|t| right_trail.contains(t))
                    .ok_or(Refusal::Unmappable("no common residual ancestor"))?;
                // FOLD ORDER: the layers already accumulated in `chain` precede
                // this block, and within the block the backend folds `F` before
                // `P`. Appending in exactly that sequence keeps `node_abs`
                // aligned with the emitted Activations.
                let mut f_branch = Vec::new();
                builder.walk(
                    lhs,
                    z,
                    &mut f_branch,
                    &mut node_abs,
                    &mut relus,
                    &mut rho_star,
                )?;
                let mut p_branch = Vec::new();
                builder.walk(
                    rhs,
                    z,
                    &mut p_branch,
                    &mut node_abs,
                    &mut relus,
                    &mut rho_star,
                )?;
                layer_count += f_branch.len() + p_branch.len();
                let segment = match (f_branch.is_empty(), p_branch.is_empty()) {
                    (true, true) => {
                        // `out = z + z`. Not a residual block; the device has no
                        // variant for it (and `Residual` would halve the value).
                        return Err(Refusal::Unmappable("degenerate residual add"));
                    }
                    (false, true) => {
                        if net.tsize.get(z).copied() != width {
                            return Err(Refusal::Unmappable("identity skip width"));
                        }
                        GpuResnetSegment::Residual(f_branch)
                    }
                    (true, false) => {
                        if net.tsize.get(z).copied() != width {
                            return Err(Refusal::Unmappable("identity skip width"));
                        }
                        GpuResnetSegment::Residual(p_branch)
                    }
                    (false, false) => GpuResnetSegment::ResidualProj(f_branch, p_branch),
                };
                if !chain.is_empty() {
                    segments.push(GpuResnetSegment::Chain(std::mem::take(&mut chain)));
                }
                segments.push(segment);
                tid = z;
            }
        }
    }
    if !chain.is_empty() {
        segments.push(GpuResnetSegment::Chain(std::mem::take(&mut chain)));
    }
    if segments.is_empty() || layer_count < 2 {
        return Err(Refusal::Unmappable("empty trunk"));
    }
    // The two lists are appended from the SAME traversal, so a divergence is a
    // construction bug, not an input condition. Fail closed rather than publish
    // a plan whose `node_abs` and gate provenance could be misaligned.
    if node_abs.len() != relus.len() {
        return Err(Refusal::Unmappable("node_abs / relu fold-order drift"));
    }
    if !saw_residual {
        let mut layers = Vec::with_capacity(layer_count);
        for segment in segments {
            match segment {
                GpuResnetSegment::Chain(v) => layers.extend(v),
                // Unreachable: `saw_residual` is the only producer of the other
                // variants. Fail closed rather than assert.
                _ => return Err(Refusal::Unmappable("unexpected segment in a unary plan")),
            }
        }
        return Ok((Plan::Chain(layers), node_abs, rho_star, relus));
    }
    Ok((Plan::Segments(segments), node_abs, rho_star, relus))
}

/// Re-gate a SHARED skeleton for one domain: clone `plan`, replacing the `k`-th
/// `Activation` (in the backend's fold order) with the gate triple `dom`
/// specifies for root-gate layer `relus[k]`.
///
/// # Why this, and not "build the plan again with `dom`"
///
/// Rebuilding would mint a FRESH `Arc<[f32]>` for every conv/gemm weight per
/// domain. Two consequences, both bad: the device's homogeneity gate would fall
/// back from `Arc::ptr_eq` to an O(weights) value compare per domain per layer,
/// and — worse — "the domains share weights" would become a runtime coincidence
/// instead of a structural fact. Cloning the reference plan keeps the shared
/// `Arc`s IDENTICAL by construction, so per-domain drift in anything except the
/// relaxation is not representable.
///
/// # What may differ, and what may not
///
/// Only `lower_slope` / `upper_slope` / `upper_intercept` may move. The
/// host-only `node_abs` retargeting invariant is recomputed here from the SAME
/// root record and checked against the shared one: it is derived from
/// `LayerGates::l`/`u` over the ROOT box, and a piece fix only ever RESTRICTS the
/// region (it rewrites `alpha/s/c`, never `l`/`u`), so it must come out
/// identical. A mismatch means that invariant has been broken somewhere and the
/// plan is refused; the values are not sent to coefficient egress.
fn retarget_plan(
    plan: &Plan,
    relus: &[usize],
    shared_node_abs: &[Vec<f32>],
    eng: &BackwardEngine<'_>,
    dom: Option<&DomainGates>,
) -> Result<Plan, Refusal> {
    if relus.len() != shared_node_abs.len() {
        return Err(Refusal::Unmappable("node_abs / relu fold-order drift"));
    }
    let builder = Builder { eng, dom };
    // A single cursor advanced in FOLD ORDER — segments in order, layers within
    // a segment in order, `F` before `P` — the very order `build_plan_full`
    // appended `relus` in.
    let mut next = 0usize;
    let mut regate = |layers: &[GpuCrownLayer]| -> Result<Vec<GpuCrownLayer>, Refusal> {
        let mut out = Vec::with_capacity(layers.len());
        for l in layers {
            match l {
                GpuCrownLayer::Activation { num_neurons, .. } => {
                    let li = *relus
                        .get(next)
                        .ok_or(Refusal::Unmappable("more Activations than relu records"))?;
                    let (act, abs) = builder.activation(li)?;
                    let GpuCrownLayer::Activation {
                        num_neurons: n2, ..
                    } = &act
                    else {
                        return Err(Refusal::Unmappable("re-gate produced a non-Activation"));
                    };
                    if n2 != num_neurons {
                        return Err(Refusal::Unmappable("re-gate changed the Activation width"));
                    }
                    if abs != shared_node_abs[next] {
                        return Err(Refusal::Unmappable("domain moved the node_abs frontier"));
                    }
                    out.push(act);
                    next += 1;
                }
                // Shared weights: clone keeps the SAME `Arc` allocation.
                other => out.push(other.clone()),
            }
        }
        Ok(out)
    };
    let out = match plan {
        Plan::Chain(layers) => Plan::Chain(regate(layers.as_slice())?),
        Plan::Segments(segs) => {
            let mut out = Vec::with_capacity(segs.len());
            for s in segs {
                out.push(match s {
                    GpuResnetSegment::Chain(l) => GpuResnetSegment::Chain(regate(l.as_slice())?),
                    GpuResnetSegment::Residual(l) => {
                        GpuResnetSegment::Residual(regate(l.as_slice())?)
                    }
                    GpuResnetSegment::ResidualProj(f, p) => {
                        // F before P — the backend's own fold order.
                        let f2 = regate(f.as_slice())?;
                        let p2 = regate(p.as_slice())?;
                        GpuResnetSegment::ResidualProj(f2, p2)
                    }
                });
            }
            Plan::Segments(out)
        }
    };
    if next != relus.len() {
        return Err(Refusal::Unmappable("fewer Activations than relu records"));
    }
    Ok(out)
}

/// Downcast a ReLU gate into f32 so the DEVICE's relaxation still encloses the
/// real ReLU on `[l, u]`.
///
/// * lower line `alpha*x`: sound for any `alpha in [0, 1]`. The lane's alphas
///   are 0/1 (root gates and piece-fixes alike), so the downcast is EXACT; a
///   non-representable alpha refuses rather than guesses.
/// * upper line `s*x + c`: relu is convex and the line affine, so
///   `line >= relu` on `[l, u]` IFF it holds at both endpoints. When the
///   downcast is inexact the intercept is bumped up by the worst endpoint
///   residual (with rounding headroom) and then rounded UP into f32. An exact
///   downcast — every stable gate and every piece-fix, which are `(1,1,0)` or
///   `(0,0,0)` — is left untouched, so a piece-fixed neuron is never widened
///   against the ROOT box it no longer lives in.
fn outward_gate_f32(
    alpha: &[f64],
    s: &[f64],
    c: &[f64],
    l: &[f64],
    u: &[f64],
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Refusal> {
    let n = alpha.len();
    let mut a_out = Vec::with_capacity(n);
    let mut s_out = Vec::with_capacity(n);
    let mut c_out = Vec::with_capacity(n);
    for j in 0..n {
        if !(0.0..=1.0).contains(&alpha[j]) {
            return Err(Refusal::Unmappable("lower slope outside [0, 1]"));
        }
        let a32 = finite_f32(alpha[j])?;
        if f64::from(a32) != alpha[j] {
            return Err(Refusal::Unmappable("lower slope is not f32-exact"));
        }
        let s32 = finite_f32(s[j])?;
        let c32 = finite_f32(c[j])?;
        let (sd, cd) = (f64::from(s32), f64::from(c32));
        let c_final = if sd == s[j] && cd == c[j] {
            c32
        } else {
            if !(l[j].is_finite() && u[j].is_finite()) {
                return Err(Refusal::Unmappable("non-finite pre-activation box"));
            }
            let need_l = l[j].max(0.0) - (sd * l[j] + cd);
            let need_u = u[j].max(0.0) - (sd * u[j] + cd);
            let bump = need_l.max(need_u).max(0.0);
            if bump > 0.0 {
                f32_toward_pos_inf(next_up(cd + bump * (1.0 + 8.0 * UNIT)))?
            } else {
                c32
            }
        };
        a_out.push(a32);
        s_out.push(s32);
        c_out.push(c_final);
    }
    Ok((a_out, s_out, c_out))
}

/// One compiled conv as a `GpuCrownLayer::Conv2d`, plus the composed relative
/// weight error the device is told to charge.
fn conv(cv: &ConvOp) -> Result<(GpuCrownLayer, f64), Refusal> {
    if cv.transposed {
        return Err(Refusal::Unmappable("ConvTranspose has no GpuCrownLayer"));
    }
    let (pad_top, pad_left, pad_bottom, pad_right) = cv.pads;
    if pad_top != pad_bottom || pad_left != pad_right {
        return Err(Refusal::Unmappable("asymmetric conv padding"));
    }
    let (out_channels, in_channels, kernel_h, kernel_w) = cv.kernel;
    let (och, out_h, out_w) = cv.oshape;
    let (ich, in_h, in_w) = cv.ishape;
    if och != out_channels || ich != in_channels {
        return Err(Refusal::Unmappable("conv channel disagreement"));
    }
    if cv.wmat.len() != out_channels * in_channels * kernel_h * kernel_w {
        return Err(Refusal::Unmappable("conv weight length"));
    }
    if cv.bias.len() != out_channels || cv.bias_err.len() != out_channels {
        return Err(Refusal::Unmappable("conv bias length"));
    }
    let cert_err = weight_error(cv.weight_rel_err, &cv.bias, &cv.bias_err)?;
    // `wmat` is `[cout][cin][kh][kw]` row-major, which IS the device's
    // `weight_col` layout `(out_c, in_c*kh*kw)`.
    let weight_col: Arc<[f32]> = Arc::from(to_f32_params_rel(&cv.wmat, U32_UNIT)?);
    // A bias vector is omitted ONLY when there is nothing to carry AND nothing
    // to charge. If `bias_err` is nonzero the layer keeps its (possibly
    // all-zero) bias so the device has a bias term to attach
    // `cert_err.bias_abs_err` to — dropping the vector could drop the charge.
    let bias_expanded: Option<Arc<[f32]>> =
        if cv.bias.iter().all(|b| *b == 0.0) && cv.bias_err.iter().all(|e| *e == 0.0) {
            None
        } else {
            let bias32 = to_f32_params_rel(&cv.bias, U32_UNIT)?;
            let mut expanded = Vec::with_capacity(out_channels * out_h * out_w);
            for &v in &bias32 {
                for _ in 0..(out_h * out_w) {
                    expanded.push(v);
                }
            }
            Some(Arc::from(expanded))
        };
    Ok((
        GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h: cv.stride.0,
            stride_w: cv.stride.1,
            pad_h: pad_top,
            pad_w: pad_left,
            out_h,
            out_w,
            in_h,
            in_w,
            cert_err,
        },
        f64::from(cert_err.weight_rel_err),
    ))
}

/// Compose the lane's certified fold error with the f64->f32 downcast into the
/// [`CertifiedWeightError`] the device will charge.
///
/// * weights: composing the fold and downcast first gives
///   `|w32 - w_exact| <= R * |w_exact|`,
///   `R = rho + u32 + rho*u32`. [`CertifiedWeightError`] is defined relative to
///   the SUPPLIED `|w32|`, so the value handed to the device is `R/(1-R)`.
///   This follows from `|w32| >= (1-R)|w_exact|` and requires `R < 1`; a ball
///   reaching unity refuses because no finite supplied-weight-relative radius
///   follows.
/// * bias: `|b32 - b_exact| <= max_ch(bias_err) + u32 * max_ch(|bias|)`, an
///   absolute per-output bound (the API's `max over outputs` convention).
///
/// Both are rounded UP into f32 so the value the device receives is never
/// smaller than the real bound.
fn weight_error(
    weight_rel_err: f64,
    bias: &[f64],
    bias_err: &[f64],
) -> Result<CertifiedWeightError, Refusal> {
    if !weight_rel_err.is_finite() || weight_rel_err < 0.0 {
        return Err(Refusal::Unmappable("non-finite weight_rel_err"));
    }
    // Directed upper bound on the exact-weight-relative composed radius. Step
    // each positive operation separately; one successor after the whole
    // expression would not in general cover every rounded intermediate.
    let fold_downcast_product = next_up(weight_rel_err * U32_UNIT);
    let exact_rel = next_up(next_up(weight_rel_err + U32_UNIT) + fold_downcast_product);
    if exact_rel >= 1.0 {
        return Err(Refusal::Unmappable(
            "weight error ball reaches unity before denominator conversion",
        ));
    }
    // `R/(1-R)` converts an exact-weight-relative ball to the denominator the
    // public contract requires (`|w32|`). Round the denominator DOWN and the
    // quotient UP so the stored radius cannot under-charge.
    let denominator = next_down(1.0 - exact_rel);
    if denominator <= 0.0 {
        return Err(Refusal::Unmappable(
            "weight error denominator is not positive",
        ));
    }
    let rel = next_up(exact_rel / denominator);
    let mut abs_bias = 0.0f64;
    for &b in bias {
        if !b.is_finite() {
            return Err(Refusal::Unmappable("non-finite bias"));
        }
        abs_bias = abs_bias.max(b.abs());
    }
    let mut err_bias = 0.0f64;
    for &e in bias_err {
        if !(e.is_finite() && e >= 0.0) {
            return Err(Refusal::Unmappable("non-finite bias_err"));
        }
        err_bias = err_bias.max(e);
    }
    Ok(CertifiedWeightError {
        weight_rel_err: f32_toward_pos_inf(rel)?,
        bias_abs_err: f32_toward_pos_inf(next_up(err_bias + U32_UNIT * abs_bias))?,
    })
}

/// Concretize the seed's certified error AND its f32 downcast residual into a
/// per-row additive penalty at the head pre-activation.
///
/// The pass's functional is `sum_j S[j,r] * y_j`. If the stored `S` differs
/// from the exact real seed by `d_j`, the functional moves by at most
/// `sum_j |d_j| * yabs_j` for any `y` in the caller's certified y-box — a pure
/// constant, so it belongs in `eb` and never has to reach the device (which
/// treats a seed as exact). `d_j` here is the SUM of the lane's own `Seed::e`
/// and the EXACT downcast residual `|f64(S as f32) - S|` (exact because the two
/// values are within a factor of two, so the subtraction is exact).
fn seed_penalty(
    seed: &Seed,
    y_abs: Option<&[f64]>,
    n_y: usize,
    rows: usize,
) -> Result<Vec<f64>, Refusal> {
    let s = seed
        .s
        .as_slice()
        .ok_or(Refusal::Unmappable("seed is not standard layout"))?;
    let e = match seed.e.as_ref() {
        Some(m) => Some(
            m.as_slice()
                .ok_or(Refusal::Unmappable("seed error is not standard layout"))?,
        ),
        None => None,
    };
    let mut pen = vec![0.0f64; rows];
    for j in 0..n_y {
        for r in 0..rows {
            let v = s[j * rows + r];
            if !v.is_finite() {
                return Err(Refusal::Unmappable("non-finite seed"));
            }
            #[allow(clippy::cast_possible_truncation)]
            let v32 = v as f32;
            if !v32.is_finite() {
                return Err(Refusal::Unmappable("seed overflows f32"));
            }
            let dc = (ny_core::f32_to_f64_exact(v32) - v).abs();
            let ev = e.map_or(0.0, |e| e[j * rows + r]);
            if !(ev.is_finite() && ev >= 0.0) {
                return Err(Refusal::Unmappable("non-finite seed error"));
            }
            if dc == 0.0 && ev == 0.0 {
                continue;
            }
            let ya = y_abs.ok_or(Refusal::SeedNeedsYAbs)?;
            if ya.len() != n_y {
                return Err(Refusal::Rows);
            }
            if !(ya[j].is_finite() && ya[j] >= 0.0) {
                return Err(Refusal::Unmappable("non-finite y magnitude bound"));
            }
            pen[r] += (dc + ev) * ya[j];
        }
    }
    let g = gamma_n(n_y + 8);
    for p in &mut pen {
        *p = certify_up(*p, g);
        if !p.is_finite() {
            return Err(Refusal::NonFinite);
        }
    }
    Ok(pen)
}

/// Transpose the lane's `(n_y, R)` column-per-row seed into the device's
/// `(num_specs, current_dim)` row-major form. The seed goes over as EXACT; its
/// discrepancy is already accounted for by [`seed_penalty`].
fn build_seed(n_y: usize, seed: &Seed, rows: usize) -> Result<GpuCrownSeed, Refusal> {
    let src = seed
        .s
        .as_slice()
        .ok_or(Refusal::Unmappable("seed is not standard layout"))?;
    let mut a = vec![0.0f32; rows * n_y];
    for (r, dst) in a.chunks_exact_mut(n_y).enumerate() {
        for (j, slot) in dst.iter_mut().enumerate() {
            *slot = finite_f32(src[j * rows + r])?;
        }
    }
    let a: Arc<[f32]> = Arc::from(a);
    let b: Arc<[f32]> = Arc::from(vec![0.0f32; rows]);
    Ok(GpuCrownSeed {
        lower_a: Arc::clone(&a),
        upper_a: a,
        lower_b: Arc::clone(&b),
        upper_b: b,
        num_specs: rows,
        current_dim: n_y,
    })
}

/// The root input box, rounded OUTWARD into f32 for trait-signature parity.
///
/// The coefficient egress MUST ignore this box: its result is a
/// box-independent [`CertifiedCoeffs`] enclosure. The lane retains its f64 box
/// and chooses the eventual concretization after the frontier returns.
fn build_box(lo: &[f64], hi: &[f64], n_in: usize) -> Result<(Vec<f32>, Vec<f32>), Refusal> {
    if lo.len() != n_in || hi.len() != n_in || n_in == 0 {
        return Err(Refusal::Unmappable("input box width"));
    }
    let mut out_lo = Vec::with_capacity(n_in);
    let mut out_hi = Vec::with_capacity(n_in);
    for (&l, &u) in lo.iter().zip(hi) {
        if !(l.is_finite() && u.is_finite()) || l > u {
            return Err(Refusal::Unmappable("non-finite or inverted input box"));
        }
        out_lo.push(f32_toward_neg_inf(l)?);
        out_hi.push(f32_toward_pos_inf(u)?);
    }
    Ok((out_lo, out_hi))
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn dispatch(
    plan: &Plan,
    node_abs: &[Vec<f32>],
    gseed: &GpuCrownSeed,
    lo: &[f32],
    hi: &[f32],
    deadline: Option<Instant>,
) -> Result<CertifiedCoeffs, Refusal> {
    // ROUTE with a finite marker so the resolver observes only an
    // already-materialized backend; never run a cold factory here.
    let route_deadline = deadline.or_else(|| Instant::now().checked_add(ROUTE_ONLY_MARKER));
    if route_deadline.is_none() {
        return Err(Refusal::NoAuthority);
    }
    let (gpu, use_sound) =
        gpu_crown_backward_route_with_deadline(None, route_deadline).ok_or(Refusal::NoAuthority)?;
    if !use_sound || !gpu.provides_sound_gpu_crown() {
        return Err(Refusal::NoAuthority);
    }
    if !gpu_crown_backend_honors_deadline(gpu, deadline) {
        return Err(Refusal::DeadlineUnsupported);
    }
    // Install the CALLER's real deadline (None installs nothing) on the exact
    // backend that receives the dispatch.
    let _scope = GpuCrownBackendDeadlineScope::set(gpu, deadline);
    dispatch_on(gpu, plan, node_abs, gseed, lo, hi)
}

/// The trait call itself, split out so tests can drive a recording backend.
///
/// A [`Plan::Chain`] goes to the flat coefficient egress exactly as before; a
/// [`Plan::Segments`] goes to the SEGMENT coefficient egress, which publishes
/// the frontier COMPOSED ACROSS every segment (chains folded, identity skips
/// added, projection branches merged) — the same frontier the resnet bounds
/// entry would have concretized.
///
/// The box and abs-max parameters are present only because the coefficient
/// entry mirrors the bounds-entry signature. The [`GpuCrownBackward`] contract
/// requires the backend to ignore all of them: [`CertifiedCoeffs`] certifies
/// coefficients independently of any domain, so folding an error radius
/// against `frontier_abs` or `node_abs` and publishing the result as bias error
/// would violate the return type. `frontier_abs` is empty here; the nonempty
/// `node_abs` argument is retained only for API parity and remains a host-side
/// plan/retargeting invariant. The lane itself chooses the eventual f64
/// concretization after publication.
fn dispatch_on(
    gpu: &dyn GpuCrownBackward,
    plan: &Plan,
    node_abs: &[Vec<f32>],
    gseed: &GpuCrownSeed,
    lo: &[f32],
    hi: &[f32],
) -> Result<CertifiedCoeffs, Refusal> {
    let published = match plan {
        Plan::Chain(layers) => gpu.crown_backward_gpu_seeded_sound_coeffs(layers, gseed, lo, hi),
        Plan::Segments(segments) => {
            gpu.crown_backward_gpu_resnet_sound_coeffs(segments, gseed, lo, hi, &[], node_abs)
        }
    };
    published
        .map_err(|_| Refusal::Device)?
        .ok_or(Refusal::NoCoeffEgress)
}

// ---------------------------------------------------------------------------
// Conversion + guards
// ---------------------------------------------------------------------------

/// Convert one lane of a [`CertifiedCoeffs`] payload into the lane's
/// coefficient + certified-error representation.
///
/// `f32 -> f64` is EXACT, so nothing is rounded here; the only added term is
/// the seed penalty folded into `eb`.
pub(crate) fn convert_and_check(
    cc: &CertifiedCoeffs,
    dir: LaneDir,
    n_in: usize,
    rows: usize,
    pen_seed: &[f64],
) -> Result<PassOut, Refusal> {
    if cc.num_specs != rows || cc.dim != n_in || pen_seed.len() != rows {
        return Err(Refusal::Payload);
    }
    let want_a = rows.checked_mul(n_in).ok_or(Refusal::Payload)?;
    let (a_src, ae_src, b_src, be_src) = match dir {
        LaneDir::Lower => (&cc.lower_a, &cc.lower_a_err, &cc.lower_b, &cc.lower_b_err),
        LaneDir::Upper => (&cc.upper_a, &cc.upper_a_err, &cc.upper_b, &cc.upper_b_err),
    };
    if a_src.len() != want_a
        || ae_src.len() != want_a
        || b_src.len() != rows
        || be_src.len() != rows
    {
        return Err(Refusal::Payload);
    }
    let mut a = Array2::<f64>::zeros((n_in, rows));
    let mut e = Array2::<f64>::zeros((n_in, rows));
    {
        let av = a.as_slice_mut().expect("standard layout");
        let ev = e.as_slice_mut().expect("standard layout");
        for r in 0..rows {
            for i in 0..n_in {
                let coeff = ny_core::f32_to_f64_exact(a_src[r * n_in + i]);
                let err = ny_core::f32_to_f64_exact(ae_src[r * n_in + i]);
                if !coeff.is_finite() || !err.is_finite() || err < 0.0 {
                    return Err(Refusal::Payload);
                }
                av[i * rows + r] = coeff;
                ev[i * rows + r] = err;
            }
        }
    }
    let mut b = Vec::with_capacity(rows);
    let mut eb = Vec::with_capacity(rows);
    for r in 0..rows {
        let bv = ny_core::f32_to_f64_exact(b_src[r]);
        let bev = ny_core::f32_to_f64_exact(be_src[r]);
        if !bv.is_finite() || !bev.is_finite() || bev < 0.0 {
            return Err(Refusal::Payload);
        }
        b.push(bv);
        eb.push(next_up(bev + pen_seed[r]));
    }
    let out = PassOut {
        a,
        e: Some(e),
        b,
        eb,
        coll: None,
        coll_rows: None,
    };
    // Same fail-closed firewall the CPU pass applies before any verdict math.
    if out.a.iter().any(|v| !v.is_finite())
        || out.b.iter().any(|v| !v.is_finite())
        || out
            .e
            .as_ref()
            .is_some_and(|e| e.iter().any(|v| !v.is_finite()))
        || out.eb.iter().any(|v| !v.is_finite())
    {
        return Err(Refusal::NonFinite);
    }
    Ok(out)
}

/// The certified-error floor guard.
///
/// For every layer whose weights the device was told are only known to
/// relative accuracy `rho`, a certification valid over that whole ball must
/// charge at least `rho * sum_j |w_ij| |a_j| >= rho * |a_out_i|` at that layer,
/// and error reaches the input through the ABS backward operator, which
/// dominates the signed one. So the total concretized penalty obeys
/// `P >= rho * sum_i |a_i^exact| * xabs_i`. Replacing the exact coefficients by
/// the PUBLISHED ones costs at most `P` itself, giving the checkable form
/// `P >= rho*T / (1 + rho)` with `T = sum_i |a_i| * xabs_i`.
///
/// The reduction is exactly the one `concretize` performs, so the guard is
/// free in the asymptotic sense; the floor is evaluated with `1 ulp` of
/// downward slack so the guard's own rounding cannot false-trip.
pub(crate) fn error_floor_ok(out: &PassOut, xabs: &[f64], rho: f64) -> bool {
    if !(rho.is_finite() && rho > 0.0) {
        return true;
    }
    let rows = out.a.ncols();
    let n_in = out.a.nrows();
    if xabs.len() != n_in || out.eb.len() != rows {
        return false;
    }
    let Some(err) = out.e.as_ref() else {
        return false;
    };
    let asl = out.a.as_slice().expect("standard layout");
    let esl = err.as_slice().expect("standard layout");
    let mut t = vec![0.0f64; rows];
    let mut p = out.eb.clone();
    for i in 0..n_in {
        let xa = xabs[i];
        let arow = &asl[i * rows..(i + 1) * rows];
        let erow = &esl[i * rows..(i + 1) * rows];
        for r in 0..rows {
            t[r] += arow[r].abs() * xa;
            p[r] += erow[r] * xa;
        }
    }
    let scale = rho / (1.0 + rho);
    (0..rows).all(|r| {
        let floor = next_down(t[r] * scale);
        p[r] >= floor
    })
}

/// The realization probe: a published bound that a REALIZED value of the same
/// functional violates is unsound, full stop.
///
/// One exact f64 forward pass at the box midpoint (always a member of the root
/// box) gives `y`, so `sum_j S[j,r] * y_j` is a value the pass must bracket.
/// For a piece-fixed domain the midpoint must also satisfy every fix — checked
/// with `forward_points`' `pre_sel` pre-activation collection — and the probe
/// SKIPS (never trips) when it does not, because a non-member proves nothing.
fn probe_guard(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    out: &PassOut,
) -> Result<(), Refusal> {
    let net = eng.net;
    let root = eng.root;
    let rows = seed.s.ncols();
    let mut x = Array2::<f64>::zeros((net.n_in, 1));
    for i in 0..net.n_in {
        x[[i, 0]] = root.mid[i];
    }
    let fixes = domain_fixes(eng, dom);
    let sel: BTreeMap<usize, Vec<usize>> = fixes
        .iter()
        .map(|(op, list)| (*op, list.iter().map(|(j, _)| *j).collect()))
        .collect();
    let Ok((y, pre)) = net.forward_points(&x, &sel) else {
        // A forward failure is not evidence of unsoundness; skip the probe
        // rather than discard an otherwise-validated pass.
        return Ok(());
    };
    for (op, list) in &fixes {
        let Some(vals) = pre.get(op) else {
            return Ok(());
        };
        for (row, (_, positive)) in list.iter().enumerate() {
            let v = vals[[row, 0]];
            if (*positive && v < 0.0) || (!*positive && v > 0.0) {
                // The probe point is outside this domain: not a counterexample.
                return Ok(());
            }
        }
    }
    let bound = match dir {
        LaneDir::Lower => eng.concretize_lower(out),
        LaneDir::Upper => eng.concretize_upper(out),
    };
    if bound.len() != rows {
        return Err(Refusal::Payload);
    }
    let ss = seed.s.as_slice().expect("standard layout");
    for r in 0..rows {
        let mut f = 0.0f64;
        for j in 0..net.n_y {
            f += ss[j * rows + r] * y[[j, 0]];
        }
        if !f.is_finite() {
            return Ok(());
        }
        let tol = PROBE_REL_SLACK * (1.0 + f.abs());
        let violated = match dir {
            LaneDir::Lower => bound[r] > f + tol,
            LaneDir::Upper => bound[r] < f - tol,
        };
        if violated {
            return Err(Refusal::Probe);
        }
    }
    Ok(())
}

/// The neurons a domain piece-fixes, grouped by RELU OP index, with `true` for
/// a positive (pass-through) fix.
///
/// A fix is detected as a gate that DIFFERS from the frozen root gate; the lane
/// only ever writes `(1,1,0)` or `(0,0,0)` there (`engine::domain_gates`).
fn domain_fixes(
    eng: &BackwardEngine<'_>,
    dom: Option<&DomainGates>,
) -> Vec<(usize, Vec<(usize, bool)>)> {
    let Some(dom) = dom else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (&li, gv) in &dom.layers {
        let Some(rec) = eng.root.layers.get(li) else {
            continue;
        };
        if gv.alpha.len() != rec.n || gv.s.len() != rec.n || gv.c.len() != rec.n {
            continue;
        }
        let mut list = Vec::new();
        for j in 0..rec.n {
            if gv.alpha[j] != rec.alpha[j] || gv.s[j] != rec.s[j] || gv.c[j] != rec.c[j] {
                list.push((j, gv.alpha[j] == 1.0 && gv.s[j] == 1.0));
            }
        }
        if !list.is_empty() {
            out.push((rec.op, list));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------

fn finite_f32(x: f64) -> Result<f32, Refusal> {
    if !x.is_finite() {
        return Err(Refusal::Unmappable("non-finite parameter"));
    }
    #[allow(clippy::cast_possible_truncation)]
    let y = x as f32;
    if !y.is_finite() {
        return Err(Refusal::Unmappable("parameter overflows f32"));
    }
    Ok(y)
}

/// Downcast PARAMETERS whose discrepancy is charged as a RELATIVE ball.
///
/// [`weight_error`] promises the device that every supplied weight is within
/// `u32` relative of the f64 value it was folded from. Round-to-nearest
/// normally guarantees exactly that, but a hardware conversion in DAZ/FTZ mode
/// can flush a binary32-subnormal result to zero — a 100% relative move that
/// the charge would NOT cover. So the promise is VERIFIED here, per element,
/// against the exact widening: a violation refuses the plan instead of
/// shipping an under-charged weight.
fn to_f32_params_rel(src: &[f64], rel_budget: f64) -> Result<Vec<f32>, Refusal> {
    let mut out = Vec::with_capacity(src.len());
    for &x in src {
        let y = finite_f32(x)?;
        let widened = ny_core::f32_to_f64_exact(y);
        if (widened - x).abs() > rel_budget * x.abs() {
            return Err(Refusal::Unmappable(
                "parameter downcast exceeds the charged relative ball",
            ));
        }
        out.push(y);
    }
    Ok(out)
}

/// Largest f32 that is `<= x` (DAZ/FTZ-safe; ny-core's certified helper).
fn f32_toward_neg_inf(x: f64) -> Result<f32, Refusal> {
    if !x.is_finite() {
        return Err(Refusal::Unmappable("non-finite value"));
    }
    let out = ny_core::f64_to_f32_down(x);
    if !out.is_finite() {
        return Err(Refusal::Unmappable("value underflows f32"));
    }
    Ok(out)
}

/// Smallest f32 that is `>= x` (DAZ/FTZ-safe; ny-core's certified helper).
fn f32_toward_pos_inf(x: f64) -> Result<f32, Refusal> {
    if !x.is_finite() {
        return Err(Refusal::Unmappable("non-finite value"));
    }
    let out = ny_core::f64_to_f32_up(x);
    if !out.is_finite() {
        return Err(Refusal::Unmappable("value overflows f32"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
