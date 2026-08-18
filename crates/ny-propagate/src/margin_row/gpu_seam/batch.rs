// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DOMAIN BATCHING for the margin-row GPU seam (#margin-row-gpu-batch).
//!
//! # Why this module exists
//!
//! `docs/VNNCOMP_CRITICAL_PATH_2026-08-12.md` measured the gap on the deciding
//! cifar100 / tinyimagenet pools at **~32x**, not a constant factor: 0 of 10
//! open rows convert even at a 4x budget. The lane that actually proves those
//! rows is this one — the concurrent margin-row certified CPU BaB — and it
//! processes BaB domains **one at a time**. The per-pass seam
//! ([`super::run_pass`]) accelerates ONE domain's backward walk. Stacking the
//! frontier domains into chunked certified GPU calls is the batching shape
//! intended to attack that measured workload gap. Each call carries at most
//! [`MAX_BATCH_DOMAINS`] domains and may be narrower after device preflight.
//!
//! # What varies per domain — the reason the batch is expressible
//!
//! Read [`super::run_pass_armed`] top-down for the y-refresh shape this module
//! batches (`build_pack` -> `y_rows_seamed` -> `run_pass_pair`) and every input
//! or host-side invariant is one of these:
//!
//! | input | source | varies per domain? |
//! |---|---|---|
//! | conv / head-gemm weights + `CertifiedWeightError` | `net`, `conv()`, `weight_error()` | **no** |
//! | ReLU gate triple `(alpha, s, c)` | `DomainGates::layers[li]`, else the frozen root record | **YES — the only one** |
//! | host-only per-ReLU `node_abs` invariant | `LayerGates::l`/`u` over the ROOT box | **no** (a piece fix rewrites gates, never `l`/`u`) |
//! | `rho*` (the error-floor guard's ball) | the same weights | **no** |
//! | seed | `BackwardEngine::identity_seed` | **no** (identity, shared) |
//! | seed penalty | the same seed, `y_abs` | **no** |
//! | signature-parity input box | `RootGates::lo`/`hi` — the ROOT box | **no** (the coefficient egress ignores it; the lane concretizes) |
//!
//! So the batch shape is: **ONE shared skeleton (same `Arc` weight
//! allocations), ONE shared seed, and `N` per-domain `Activation` blocks.** A
//! shared box is carried only for trait-signature parity, and `node_abs` stays
//! on the host to validate retargeting; the coefficient egress receives empty
//! abs-max tables and MUST ignore every domain box. This is the coefficient
//! form of [`GpuResnetBatchedDomainRef`]'s contract, and it means the device's
//! HOLE-7 homogeneity gate passes on `Arc::ptr_eq` rather than an O(weights)
//! compare — see [`super::retarget_plan`], which CLONES the reference plan
//! instead of rebuilding it so shared-weight identity is structural, not
//! coincidental.
//!
//! Note what is NOT batched here, deliberately:
//!
//! * the margin-seeded `eval_with_pack` pass — it requests `Collect::unst_abs`
//!   (the branching shortlist), which the coefficient egress does not produce;
//! * `score_candidates` — per-ROW gate exceptions have no `GpuCrownLayer` form
//!   (the seam refuses `exc` outright, and the lane's own cross-domain row
//!   stack `run_domain_stacked` already covers that axis on the CPU);
//! * the y-pack's CPU tail (`YBox::from_rows`, `row_dots`) — cheap, and keeping
//!   it per-domain makes a batched pack BIT-IDENTICAL to a serial one given the
//!   same `(al, au)`.
//!
//! # The slot mapping (the killer defect)
//!
//! Publishing, for domain A, the bound computed for domain B's gates would be
//! an unsound verdict that no guard reliably catches — B's enclosure need not
//! bound A's functional at all once their piece fixes diverge. The mapping is
//! therefore made structural at every hop:
//!
//! 1. `domains[d]` is built from `gates[d]` in ONE `map` — no index arithmetic;
//! 2. the device publishes DOMAIN-MAJOR rows and splits them with
//!    `split_batched_certified_coeffs`, which length-checks before slicing and
//!    walks with `chunks_exact` (pinned:
//!    `batched_split_is_domain_major_and_contiguous`);
//! 3. this module re-pairs them with a length-checked `zip` of the SAME `gates`
//!    slice — again no index — so a payload of the wrong length refuses
//!    instead of shifting the association;
//! 4. every domain's certified-error FLOOR guard and REALIZATION PROBE are then
//!    evaluated against **its own** gates, so a permutation that survived 1-3
//!    still has to get past a probe built from the wrong piece fixes.
//!
//! `batched_slot_permutation_is_detectable` falsifies a deliberately permuted
//! device payload at the lane-side re-pairing boundary. A separate device test
//! pins the backend's domain-major split.
//!
//! # Fail-closed rules
//!
//! * DARK unless BOTH `NY_MARGIN_ROW_GPU=1` and `NY_MARGIN_ROW_GPU_BATCH=1`
//!   (exact strings). This gate never enables the seam on its own.
//! * Only `RoundMode::Outward` passes are batched.
//! * Only a `Plan::Segments` plan is batched; a unary chain has no batched
//!   coefficient egress and stays on the per-pass seam.
//! * Only a typed pre-dispatch capacity refusal narrows and retries. Any other
//!   refusal — gate, chain plan, unmappable layer, heterogeneous skeleton,
//!   authority, deadline, device fault, shape, either guard — aborts the whole
//!   optional wave and falls back to the per-pass seam and then the exact CPU
//!   walk. No partial result is published.

use std::sync::OnceLock;
use std::time::Instant;

use ny_core::{CertifiedCoeffs, GpuCrownBackward, GpuCrownSeed, GpuResnetBatchedDomainRef};

use super::super::engine::{BackwardEngine, DomainGates, LaneDir, PassOut, Seed};
use super::super::prof::{bump, bump_always, Counter};
use super::{
    build_box, build_plan_full, build_seed, convert_and_check, error_floor_ok, probe_guard,
    retarget_plan, seed_penalty, Plan, Refusal, SeamCtx, ROUTE_ONLY_MARKER,
};
use crate::sound_gpu_gate::{
    gpu_crown_backend_honors_deadline, gpu_crown_backward_route_with_deadline,
    GpuCrownBackendDeadlineScope,
};

/// Largest domain count the lane will offer to ONE batched dispatch.
///
/// The device applies its own, tighter, `max_compute_workgroups_per_dimension`
/// cap and DECLINES a batch that would overrun it (the overrun is a latent
/// false-VERIFY hole, not merely a crash risk). Since that limit depends on the
/// net's widest 1-D dispatch and on the seed's row count, the lane cannot
/// compute it here — so [`run_batch`] HALVES only on the backend's typed
/// pre-dispatch capacity refusal, which finds the device's own admissible width
/// in at most a few host-side retries. The expensive reference plan is built
/// once and reused.
pub(crate) const MAX_BATCH_DOMAINS: usize = 16;

/// Below this width there is nothing to gain over the per-pass seam, so the
/// halving retry stops here rather than degenerating into it.
const MIN_BATCH_DOMAINS: usize = 2;

/// Failure classification local to the domain-batched seam.
///
/// Only `Capacity` may be retried at a narrower width. Its producer is the
/// backend's typed, pre-dispatch `GpuBatchCapacityExceeded` refusal. Keeping it
/// separate from `Refusal::Device` makes it impossible for a deadline,
/// firewall, validation, or device-execution error to enter the width ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchError {
    Capacity,
    DeadlineExpired,
    Lane(Refusal),
}

impl From<Refusal> for BatchError {
    fn from(value: Refusal) -> Self {
        Self::Lane(value)
    }
}

#[inline]
fn check_deadline_at(deadline: Option<Instant>, now: Instant) -> Result<(), BatchError> {
    if deadline.is_some_and(|limit| now >= limit) {
        Err(BatchError::DeadlineExpired)
    } else {
        Ok(())
    }
}

#[inline]
fn check_deadline(deadline: Option<Instant>) -> Result<(), BatchError> {
    check_deadline_at(deadline, Instant::now())
}

/// Pure arming predicate — the SPEC of the gate, unit-tested below.
#[inline]
fn armed_from_raw(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Latched raw `NY_MARGIN_ROW_GPU_BATCH` string, read once through the
/// ny-levers chokepoint (latch the STRING, derive the DECISION per call).
fn env_raw() -> Option<&'static str> {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    RAW.get_or_init(|| ny_levers::read_raw(&ny_levers::decls::sound_channel::MARGIN_ROW_GPU_BATCH))
        .as_deref()
}

/// Is domain batching armed? SUBORDINATE: the per-pass seam gate must be armed
/// too, so this can never be the thing that first sends a verdict-bearing bound
/// to the device.
#[inline]
pub(crate) fn enabled() -> bool {
    super::enabled() && armed_from_raw(env_raw())
}

/// Report each DISTINCT batched-refusal reason ONCE per process, under
/// `NY_MARGIN_ROW_PROFILE=1`. Same rationale as [`super::note_refusal`]: a
/// counter that proves the lane never ran but not WHY costs a measurement
/// cycle. Feeds nothing back into the pass.
fn note_refusal(refusal: Refusal) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    if !super::super::prof::enabled() {
        return;
    }
    let bit = 1u32 << refusal.tag();
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        eprintln!("[margin-row-gpu-batch] batch refused: {refusal:?}");
    }
}

fn note_batch_error(error: BatchError) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);

    match error {
        BatchError::Lane(refusal) => note_refusal(refusal),
        BatchError::Capacity | BatchError::DeadlineExpired => {
            if !super::super::prof::enabled() {
                return;
            }
            let bit = match error {
                BatchError::Capacity => 1,
                BatchError::DeadlineExpired => 2,
                BatchError::Lane(_) => unreachable!("lane error handled above"),
            };
            if SEEN.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
                eprintln!("[margin-row-gpu-batch] batch refused: {error:?}");
            }
        }
    }
}

/// Everything a batch of domains SHARES, built once per frontier batch.
///
/// Building this is the only O(weights) step in the lane's GPU path (the f64 ->
/// f32 parameter downcast and its per-element relative-ball verification), so
/// it is hoisted out of the per-domain loop AND out of the halving retry.
pub(crate) struct BatchPlan {
    /// The reference skeleton, gated at the FROZEN ROOT. Every domain's plan is
    /// this one with its `Activation`s replaced, so all of them share these
    /// exact weight `Arc`s.
    plan: Plan,
    /// Host-only per-`Activation` PRE-activation abs-max invariant, in backend
    /// fold order. Domain-independent (derived from the root `l`/`u`), shared
    /// by every domain and re-verified in [`retarget_plan`]; never sent to the
    /// coefficient egress.
    node_abs: Vec<Vec<f32>>,
    /// Root-gate layer index of each `Activation`, same fold order.
    relus: Vec<usize>,
    /// Largest certified relative weight error handed to any layer.
    rho_star: f64,
    /// The shared identity seed (the y-row refresh shape).
    seed: Seed,
    /// The device form of `seed`.
    gseed: GpuCrownSeed,
    /// Per-row additive penalty concretizing the seed's own error.
    pen_seed: Vec<f64>,
    /// The signature-parity ROOT box, rounded OUTWARD into f32. The coefficient
    /// egress ignores these fields.
    lo: Vec<f32>,
    hi: Vec<f32>,
    /// Spec rows per domain (`n_y` for the identity y-refresh seed).
    rows: usize,
}

/// Build everything the batch shares. Refuses for exactly the reasons the
/// per-pass seam refuses, plus one: a unary CHAIN plan, which has no batched
/// coefficient egress and belongs on the per-pass seam.
pub(crate) fn prepare(eng: &BackwardEngine<'_>, ctx: &SeamCtx<'_>) -> Result<BatchPlan, Refusal> {
    let net = eng.net;
    if !eng.root.mode.outward() {
        return Err(Refusal::NotOutward);
    }
    let seed = eng.identity_seed();
    let rows = seed.s.ncols();
    if rows == 0 || seed.s.nrows() != net.n_y || net.n_y == 0 {
        return Err(Refusal::Rows);
    }
    // The reference skeleton is built at the FROZEN ROOT (`dom = None`): its
    // Activations are replaced per domain anyway, and everything that is NOT
    // replaced — weights, the host-only `node_abs` invariant, `rho*` — is
    // domain-independent, which is exactly the claim this construction encodes.
    let (plan, node_abs, rho_star, relus) = build_plan_full(eng, None)?;
    if matches!(plan, Plan::Chain(_)) {
        return Err(Refusal::Unmappable(
            "unary chain plan has no batched coefficient egress",
        ));
    }
    let pen_seed = seed_penalty(&seed, ctx.y_abs, net.n_y, rows)?;
    let gseed = build_seed(net.n_y, &seed, rows)?;
    let (lo, hi) = build_box(eng.root.lo.as_slice(), eng.root.hi.as_slice(), net.n_in)?;
    Ok(BatchPlan {
        plan,
        node_abs,
        relus,
        rho_star,
        seed,
        gseed,
        pen_seed,
        lo,
        hi,
        rows,
    })
}

/// Fold `gates.len()` domains' y-row refresh in ONE certified GPU call and
/// return each domain's `(lower, upper)` pass, in `gates` order.
///
/// On `Ok` every returned pair is a drop-in replacement for
/// `BackwardEngine::y_rows(Some(gates[d]))`: same shapes, same meaning, and the
/// caller concretizes it with the lane's own unchanged f64 `concretize_*`.
///
/// This is the ALL-OR-NOTHING unit. A guard trip on ANY domain discards the
/// WHOLE chunk: a partial result would need a second, differently-shaped
/// bookkeeping path, and the domains that failed nothing lose only the time of
/// one CPU rebuild.
pub(crate) fn run_chunk(
    bp: &BatchPlan,
    eng: &BackwardEngine<'_>,
    gates: &[&DomainGates],
    deadline: Option<Instant>,
) -> Result<Vec<(PassOut, PassOut)>, BatchError> {
    check_deadline(deadline)?;
    if gates.is_empty() {
        return Err(Refusal::Rows.into());
    }
    // One re-gated skeleton per domain. `retarget_plan` clones the shared
    // weight `Arc`s and re-verifies that the domain did not move `node_abs`.
    let plans: Vec<Plan> = gates
        .iter()
        .map(|&g| retarget_plan(&bp.plan, &bp.relus, &bp.node_abs, eng, Some(g)))
        .collect::<Result<_, Refusal>>()
        .map_err(BatchError::Lane)?;
    let segments: Vec<&[ny_core::GpuResnetSegment]> = plans
        .iter()
        .map(|p| match p {
            Plan::Segments(s) => Ok(s.as_slice()),
            // Unreachable: `prepare` refused a chain plan and `retarget_plan`
            // preserves the variant. Fail closed rather than assert.
            Plan::Chain(_) => Err(Refusal::Unmappable("chain plan in a batch")),
        })
        .collect::<Result<_, Refusal>>()
        .map_err(BatchError::Lane)?;
    // THE BATCH SHAPE: per-domain segments (relaxation only), everything else
    // shared. Coefficient egress intentionally receives EMPTY beta/abs tables:
    // moving coefficient error into a domain-concretized bias would destroy
    // the coefficient-wise enclosure this API publishes. `bp.node_abs` above
    // remains a host-side retargeting invariant only; it grants no egress
    // authority and is not sent to the backend.
    let refs: Vec<GpuResnetBatchedDomainRef<'_>> = segments
        .iter()
        .map(|&s| GpuResnetBatchedDomainRef {
            segments: s,
            input_lower: &bp.lo,
            input_upper: &bp.hi,
            beta_signed: &[],
            frontier_abs: &[],
            node_abs: &[],
        })
        .collect();
    let published = dispatch_batch(&refs, &bp.gseed, deadline)?;
    finish_chunk(bp, eng, gates, &published).map_err(BatchError::Lane)
}

/// THE SLOT MAP, lane side: re-pair the device's per-domain payloads with the
/// domains that asked for them, and run each domain's own guards.
///
/// Split out of [`run_chunk`] so it is falsifiable WITHOUT a device: the tests
/// feed it deliberately permuted payloads and assert the published bounds move.
///
/// Length is checked BEFORE anything is paired, and the pairing is a `zip` of
/// the SAME `gates` slice the request was built from — there is no index to get
/// wrong. A wrong-length payload REFUSES; it is never truncated to fit, because
/// truncation would silently shift every later domain's association.
pub(crate) fn finish_chunk(
    bp: &BatchPlan,
    eng: &BackwardEngine<'_>,
    gates: &[&DomainGates],
    published: &[CertifiedCoeffs],
) -> Result<Vec<(PassOut, PassOut)>, Refusal> {
    if published.len() != gates.len() {
        return Err(Refusal::Payload);
    }
    let n_in = eng.net.n_in;
    let mut out = Vec::with_capacity(gates.len());
    for (dom, cc) in gates.iter().zip(published) {
        out.push(convert_pair(bp, eng, dom, cc, n_in)?);
    }
    Ok(out)
}

/// One domain's payload -> its `(lower, upper)` pass, with THAT domain's own
/// guards.
///
/// The guards are the whole reason a slot error is not merely "caught by
/// review": the realization probe re-derives this domain's piece fixes from
/// `dom` and checks the published bound against a forward pass restricted to
/// them, so a bound computed for a sibling's gates has to survive a falsifier
/// built from the RIGHT ones.
fn convert_pair(
    bp: &BatchPlan,
    eng: &BackwardEngine<'_>,
    dom: &DomainGates,
    cc: &CertifiedCoeffs,
    n_in: usize,
) -> Result<(PassOut, PassOut), Refusal> {
    let mut lower = None;
    let mut upper = None;
    for dir in [LaneDir::Lower, LaneDir::Upper] {
        let pass = convert_and_check(cc, dir, n_in, bp.rows, &bp.pen_seed)?;
        if !error_floor_ok(&pass, eng.root.xabs.as_slice(), bp.rho_star) {
            return Err(Refusal::ErrorFloor);
        }
        probe_guard(eng, &bp.seed, Some(dom), dir, &pass)?;
        match dir {
            LaneDir::Lower => lower = Some(pass),
            LaneDir::Upper => upper = Some(pass),
        }
    }
    Ok((
        lower.ok_or(Refusal::Payload)?,
        upper.ok_or(Refusal::Payload)?,
    ))
}

/// Route to the prewarmed sound backend and issue the ONE batched call.
///
/// Byte-for-byte the same routing discipline as [`super::dispatch`]: a FINITE
/// routing marker so the resolver only ever observes an already-materialized
/// process-global backend (never a cold factory inside a verifier budget), an
/// explicit authority check, an explicit cooperative-cancellation check, and
/// the caller's REAL deadline installed on the exact backend that receives the
/// dispatch.
fn dispatch_batch(
    refs: &[GpuResnetBatchedDomainRef<'_>],
    gseed: &GpuCrownSeed,
    deadline: Option<Instant>,
) -> Result<Vec<CertifiedCoeffs>, BatchError> {
    check_deadline(deadline)?;
    let route_deadline = deadline.or_else(|| Instant::now().checked_add(ROUTE_ONLY_MARKER));
    if route_deadline.is_none() {
        return Err(Refusal::NoAuthority.into());
    }
    let (gpu, use_sound) = gpu_crown_backward_route_with_deadline(None, route_deadline)
        .ok_or(BatchError::Lane(Refusal::NoAuthority))?;
    if !use_sound || !gpu.provides_sound_gpu_crown() {
        return Err(Refusal::NoAuthority.into());
    }
    if !gpu_crown_backend_honors_deadline(gpu, deadline) {
        return Err(Refusal::DeadlineUnsupported.into());
    }
    let _scope = GpuCrownBackendDeadlineScope::set(gpu, deadline);
    dispatch_batch_on(gpu, refs, gseed)
}

/// The trait call itself, split out so tests can drive a recording backend.
fn dispatch_batch_on(
    gpu: &dyn GpuCrownBackward,
    refs: &[GpuResnetBatchedDomainRef<'_>],
    gseed: &GpuCrownSeed,
) -> Result<Vec<CertifiedCoeffs>, BatchError> {
    // Source-specific denominator. Do not feed the graph/BaB wide-lane tally:
    // this margin-row coefficient egress has its own gate and publication
    // guards, so mixing it into that lane's attempts/published ratio makes both
    // receipts false.
    bump(Counter::GpuBatchAttempts, 1);
    match gpu.crown_backward_gpu_resnet_sound_batched_coeffs(refs, gseed) {
        Ok(Some(published)) => Ok(published),
        Ok(None) => Err(Refusal::NoCoeffEgress.into()),
        Err(error) if error.is_gpu_batch_capacity_exceeded() => Err(BatchError::Capacity),
        Err(error) if error.is_deadline_exceeded() => Err(BatchError::DeadlineExpired),
        Err(_) => Err(Refusal::Device.into()),
    }
}

/// The production entry: fold `gates` in fixed-width chunks at a
/// device-admitted width, recording the counters the integrator reads out of
/// `NY_MARGIN_ROW_PROFILE`.
///
/// Returns `None` (never a partial answer) if any chunk refuses, so the caller
/// simply does nothing and the established one-at-a-time path rebuilds every
/// pack.
///
/// Chunk width starts at [`MAX_BATCH_DOMAINS`] and HALVES down to
/// [`MIN_BATCH_DOMAINS`] only on the backend's typed, pre-dispatch capacity
/// refusal. `Ok(None)` (unsupported/declined), deadline expiry, a guard trip, a
/// device fault, and a shape/firewall error are all terminal for this optional
/// wave. Narrowing any of those would change the arithmetic after a hard
/// fail-closed signal and could re-issue GPU work from an earlier chunk.
pub(crate) fn run_batch(
    eng: &BackwardEngine<'_>,
    gates: &[&DomainGates],
    deadline: Option<Instant>,
) -> Option<Vec<(PassOut, PassOut)>> {
    if !enabled() {
        return None;
    }
    run_batch_armed_recorded(eng, gates, deadline)
}

/// [`run_batch`] with the environment gate already decided, plus the EXACT
/// counter bookkeeping production performs.
///
/// Split out for the same reason [`super::run_pass_armed_recorded`] is: the env
/// latch is a process-wide `OnceLock`, so a device test cannot reach the armed
/// path — or the counters the integrator reads out of `NY_MARGIN_ROW_PROFILE`
/// — through the gate.
pub(crate) fn run_batch_armed_recorded(
    eng: &BackwardEngine<'_>,
    gates: &[&DomainGates],
    deadline: Option<Instant>,
) -> Option<Vec<(PassOut, PassOut)>> {
    if gates.len() < MIN_BATCH_DOMAINS {
        return None;
    }
    if let Err(error) = check_deadline(deadline) {
        record_batch_error(error);
        return None;
    }
    let ctx = SeamCtx {
        y_abs: None,
        deadline,
    };
    let bp = match prepare(eng, &ctx) {
        Ok(bp) => bp,
        Err(refusal) => {
            record_batch_error(refusal.into());
            return None;
        }
    };
    let initial_width = MAX_BATCH_DOMAINS.min(gates.len());
    let (out, chunks) = run_width_ladder_with_clock(
        initial_width,
        deadline,
        |width| run_all_chunks(&bp, eng, gates, width, deadline),
        Instant::now,
    )
    .ok()?;
    // Counted only on a PUBLISHED, still-in-budget batch, so `gpu_batch_ok`
    // cannot overstate the lane: a wave computed and then discarded is a
    // refusal, not a firing.
    bump(Counter::GpuBatchOk, chunks as u64);
    bump(Counter::GpuBatchDomains, out.len() as u64);
    Some(out)
}

/// Run the strictly typed width ladder. The injected clock makes all deadline
/// boundaries deterministic in host-only tests.
fn run_width_ladder_with_clock<T>(
    initial_width: usize,
    deadline: Option<Instant>,
    mut attempt: impl FnMut(usize) -> Result<T, BatchError>,
    mut now: impl FnMut() -> Instant,
) -> Result<T, BatchError> {
    let mut width = initial_width.max(MIN_BATCH_DOMAINS);
    loop {
        if let Err(error) = check_deadline_at(deadline, now()) {
            record_batch_error(error);
            return Err(error);
        }
        match attempt(width) {
            Ok(value) => {
                // Reject a result that completed after the caller's deadline;
                // it must never be cached as a published y-pack.
                if let Err(error) = check_deadline_at(deadline, now()) {
                    record_batch_error(error);
                    return Err(error);
                }
                return Ok(value);
            }
            Err(error) => {
                record_batch_error(error);
                if error != BatchError::Capacity || width <= MIN_BATCH_DOMAINS {
                    return Err(error);
                }
                width = (width / 2).max(MIN_BATCH_DOMAINS);
            }
        }
    }
}

/// Fold every chunk at one width, all-or-nothing. Results are appended in
/// chunk order, and chunks are visited in `gates` order, so output slot `d` is
/// `gates[d]`'s own pass. Returns the results and the number of wide calls made.
fn run_all_chunks(
    bp: &BatchPlan,
    eng: &BackwardEngine<'_>,
    gates: &[&DomainGates],
    width: usize,
    deadline: Option<Instant>,
) -> Result<(Vec<(PassOut, PassOut)>, usize), BatchError> {
    run_chunks_with(gates, width, deadline, |chunk| {
        run_chunk(bp, eng, chunk, deadline)
    })
}

/// Generic all-or-nothing chunk walker, split out so host tests can inject a
/// failure after an earlier chunk has completed. The outer width ladder must
/// never restart those earlier chunks unless the failure is the backend's
/// typed, pre-dispatch capacity refusal.
fn run_chunks_with<T, U>(
    inputs: &[T],
    width: usize,
    deadline: Option<Instant>,
    mut run: impl FnMut(&[T]) -> Result<Vec<U>, BatchError>,
) -> Result<(Vec<U>, usize), BatchError> {
    let mut out = Vec::with_capacity(inputs.len());
    let mut chunks = 0usize;
    for chunk in inputs.chunks(width.max(1)) {
        check_deadline(deadline)?;
        let got = run(chunk)?;
        if got.len() != chunk.len() {
            return Err(Refusal::Payload.into());
        }
        chunks += 1;
        out.extend(got);
    }
    // Defensive: the caller re-pairs these with `gates` positionally.
    if out.len() != inputs.len() {
        return Err(Refusal::Payload.into());
    }
    Ok((out, chunks))
}

fn record_batch_error(error: BatchError) {
    bump(Counter::GpuBatchRefused, 1);
    note_batch_error(error);
    if matches!(
        error,
        BatchError::Lane(Refusal::ErrorFloor | Refusal::Probe)
    ) {
        // Always recorded: a guard trip is a soundness signal, not a
        // profiling note.
        bump_always(Counter::GpuBatchGuardTrip, 1);
    }
}

#[cfg(test)]
mod tests;
