// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #attack-concurrent — the verifier's upfront PGD on a sibling thread of the
//! proof (typed key `attack.concurrent`, `BetaCrownConfig::attack_concurrent`,
//! default off).
//!
//! # Why
//!
//! The conjunctive / single-clause upfront PGD inside the relational verifier
//! (`verify/graph.rs` and `verify/sequential.rs`) is handed the window
//! `start .. ledger.upfront_pgd_deadline()` (`upfront_pgd_fraction` of the
//! verifier's wall: 0.20 by default, 0.4 on acasxu) and, on every row it does
//! not falsify, runs that whole window BEFORE the first proof step. The cost
//! census (`review/cost-census` 4fb15b19d, re-derived from flight records)
//! measured attack time as 0.614 of the wall on unsat rows, at least 0.85 of it
//! provably serial; dist_shift spends 57.0 of 57.5 s in it.
//!
//! # What
//!
//! [`run_race`] runs the IDENTICAL attack call on a scoped sibling thread
//! inside a private Rayon pool `attack-concurrent-{i}` while the proof — the
//! unchanged remainder of the lane — runs on the calling thread with the
//! global pool:
//!
//! * the attack's window is stretched to `W = slice * (N + k_a) / k_a`
//!   ([`stretched_window`]: `N` the proof's global pool, `k_a` the private
//!   pool), so while both sides are busy Linux CFS still gives the attack at
//!   least the CPU-seconds the serial slice gave it; the site is admitted only
//!   when the UNCAPPED stretch fits before `overall_deadline - GATE_RESERVE`;
//! * a proof that DECIDES (`Verified`, or a proof-side `Violated`) or panics
//!   cancels the attack through the `PgdConfig::cancel` token its deadline
//!   polls already read;
//! * an UNDECIDED proof (`Timeout`/`Unknown`/`Err`) does not cut the attack
//!   short of its serial CPU-seconds: the attack keeps running to
//!   [`undecided_attack_deadline`] — `slice + p * N / (N + k_a)` after the
//!   site start (`p` the proof's wall), capped at `W` — and is cancelled
//!   there, so a proof that gives up EARLY no longer holds the lanes after
//!   the race (MIP escalation, the post-BaB attack) for the whole stretched
//!   window: they start no later than the serial order would start them;
//! * an attack candidate goes through the scored path's unchanged sat
//!   pipeline: the `publish` hook PREPARES it on the attack thread (the
//!   dispatch's deadline gate, JSON rendering, and ENTRY INTO THE ATTACK
//!   DOOR — `vnncomp::prepare_concurrent_attack_sat`, so the publication is
//!   registered in flight before the race can return) and hands back an owned
//!   adjudication job — `vnncomp::publish_concurrent_attack_sat`: the
//!   trusted-ORT gate, VNN-LIB 2.0 normalization, and ONE critical section
//!   under the #sat-precedence lock that claims terminal publication and
//!   writes RESULTS_FILE — which runs on a DETACHED `attack-concurrent-gate`
//!   thread. An upheld `sat` ends the process (`_exit`), which is the proof's
//!   cancellation: no engine plumbing, the margin-row early exit's precedent.
//!   A downgraded candidate leaves the proof's verdict in charge — a
//!   certified verdict is never discarded by the attack;
//! * THE JOIN IS BOUNDED once the proof decided: the attack thread waits for
//!   its in-flight adjudication only until [`decided_join_deadline`] —
//!   [`GATE_RESERVE`] after the decision, never past
//!   `overall_deadline - JOIN_RESERVE` — so a slow gate (a cold ORT session on
//!   a large model) can no longer push the certified verdict past the
//!   dispatch's authoritative deadline gate, which would read it as
//!   `Timeout`. An abandoned adjudication keeps running detached, inside the
//!   ATTACK DOOR the main thread's publication synchronizes with. A proof
//!   PANIC cancels the attack but does NOT bound that wait: the panic ends
//!   the process with no main-thread verdict, so the in-flight candidate is
//!   the only verdict left ([`CancelOnUnwind`]).
//!
//! Every admission refusal runs today's serial code verbatim ([`admit`]).
//!
//! # Soundness
//!
//! Scheduling only. The proof is the unchanged code under the unchanged
//! absolute deadlines (it merely starts `slice` earlier). A `sat` is
//! published only after the unchanged `gate_sat_with_trusted_oracle`, whose
//! confirmation latch the publisher additionally REQUIRES inside its critical
//! section. First verdict wins: one terminal-publication claim, and the
//! constructive witness outranks a bound-based `unsat` (#sat-precedence).
//!
//! An adjudication may OUTLIVE the race (the bounded join), so the main
//! thread's door no longer relies on the join: on a run where a race spawned
//! it is `vnncomp::publish_main_verdict_in`, which WAITS while a concurrent-
//! attack publication is inside the attack door (until its own hard
//! deadline), then CLOSES the door (no attack publication may start or write
//! after it) and runs the #sat-precedence filter, the terminal claim and the
//! write in ONE critical section of the precedence lock. So exactly one
//! verdict reaches RESULTS_FILE, an upheld witness that arrives in time wins,
//! a confirmation latched before the door turns its `unsat` into `unknown`,
//! and one that arrives after it can no longer write.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use anyhow::Result;
use ndarray::ArrayD;
use ny_propagate::{BabVerificationStatus, BetaCrownResult, PgdCancelToken};

/// Wall kept between the end of the stretched attack window and the
/// verifier's overall deadline, so a candidate found at the very end of the
/// window can still pass the trusted gate and be published. The post-BaB
/// attack's publication margin (`vnncomp::POSTBAB_ATTACK_SAFETY_MARGIN`).
pub(crate) const GATE_RESERVE: Duration = Duration::from_secs(5);

/// Wall a DECIDED proof's bounded join keeps before the verifier's overall
/// deadline — the dispatch's authoritative deadline gate
/// (`gate_result_at_deadline`, which reads the clock AFTER the lane returns):
/// the race stops waiting for an in-flight adjudication by
/// `overall_deadline - JOIN_RESERVE` at the latest, so the certified verdict
/// reaches that gate while it still admits it. The lane's tail after the join
/// is microseconds; one second is the margin.
pub(crate) const JOIN_RESERVE: Duration = Duration::from_secs(1);

/// Stack reserved for the attack thread and every worker of its private pool:
/// the Linux main thread's 8 MiB, where the serial site ran the attack body
/// and the trusted gate. Rust's 2 MiB default would be a NEW stack-depth
/// limit for that code (a stack overflow aborts the process, verdict and
/// all). A virtual reservation only; pages are committed on demand.
pub(crate) const ATTACK_THREAD_STACK_BYTES: usize = 8 << 20;

/// The two v1 race sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RaceSite {
    /// `verify/graph.rs`: the graph conjunctive / single-clause upfront PGD.
    GraphUpfront,
    /// `verify/sequential.rs`: the sequential conjunctive upfront PGD.
    SeqUpfront,
    /// The shared-input, original dual-network falsifier.
    RelationalShared,
}

impl RaceSite {
    /// Stable token for logs and records.
    pub(crate) const fn token(self) -> &'static str {
        match self {
            Self::GraphUpfront => "graph_upfront",
            Self::SeqUpfront => "seq_upfront",
            Self::RelationalShared => "relational_shared",
        }
    }
}

/// Facts only the VNN-COMP wrapper knows, handed down through
/// `BetaCrownInstanceOverrides` and `DispatchContext`. `None` (interactive
/// `ny beta-crown`, test faces) refuses the race as `publisher_unarmed`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ConcurrentAttackWrapperFacts {
    /// The concurrent margin-row worker is live for this instance (CR-X: at
    /// most two concurrent heavy CPU lanes; that lane owns the concurrency).
    pub(crate) margin_row_lane_live: bool,
}

/// Why a race site did, or did not, run concurrently. Checked in this order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AdmitRoute {
    /// No scored publication target (interactive call, short-grace route).
    PublisherUnarmed,
    /// The concurrent margin-row worker owns the concurrency (CR-X).
    MarginRowOwnsConcurrency,
    /// The attack and the proof would both run on an accelerator (the
    /// attack-steering engine or the proof's own GEMM handle on one side, the
    /// proof's GEMM handle or the GPU BaB on the other): one device, two heavy
    /// users, so concurrency buys nothing and the lane stands down (vgg's
    /// wgpu proof device). An accelerated attack beside a CPU proof is NOT
    /// shared — that is the most complementary pair this lane serves.
    SharedDevice,
    /// A `Sign` net: the BNN lanes' order stays serial.
    BnnSerial,
    /// The QDQ report-only latch would withhold the verdict anyway.
    QdqReportOnly,
    /// No upfront slice (an unbounded run, or the slice already expired).
    NoSlice,
    /// `site_start + W` does not fit before `overall_deadline - GATE_RESERVE`.
    StretchDoesNotFit,
    /// The memory floor / R4 lease refused the lane.
    LeaseDenied,
    /// The private Rayon pool could not be built.
    PoolBuildFailed,
    /// Every check passed: the attack runs concurrently.
    Spawned,
}

impl AdmitRoute {
    /// Stable token for the `[attack-concurrent] admit` line.
    pub(crate) const fn token(self) -> &'static str {
        match self {
            Self::PublisherUnarmed => "publisher_unarmed",
            Self::MarginRowOwnsConcurrency => "margin_row_owns_concurrency",
            Self::SharedDevice => "shared_device",
            Self::BnnSerial => "bnn_serial",
            Self::QdqReportOnly => "qdq_report_only",
            Self::NoSlice => "no_slice",
            Self::StretchDoesNotFit => "stretch_does_not_fit",
            Self::LeaseDenied => "lease_denied",
            Self::PoolBuildFailed => "pool_build_failed",
            Self::Spawned => "spawned",
        }
    }
}

/// The admission facts known at a site before any side effect.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AdmissionFacts {
    /// A scored publication target is armed AND the wrapper handed its facts.
    pub(crate) publisher_armed: bool,
    /// The concurrent margin-row worker is live.
    pub(crate) margin_row_lane_live: bool,
    /// The attack and the proof would share an accelerator
    /// ([`shared_device`]).
    pub(crate) shared_device: bool,
    /// The model has a `Sign` layer.
    pub(crate) sign_net: bool,
    /// The QDQ report-only latch is engaged.
    pub(crate) qdq_report_only: bool,
}

/// The admitted attack window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RaceWindow {
    /// Today's slice: `upfront_pgd_deadline - site_start`.
    pub(crate) slice: Duration,
    /// The stretched window `W`.
    pub(crate) window: Duration,
    /// `N`: the proof's (global) Rayon pool width.
    pub(crate) host_threads: usize,
    /// `k_a`: the attack's private pool width.
    pub(crate) width_pool: usize,
}

impl RaceWindow {
    /// `W / slice` (`(N + k_a) / k_a` up to rounding).
    pub(crate) fn stretch(&self) -> f64 {
        if self.slice.is_zero() {
            1.0
        } else {
            self.window.as_secs_f64() / self.slice.as_secs_f64()
        }
    }
}

/// `W = slice * (N + k_a) / k_a`, in integer nanoseconds, saturating. With
/// both sides busy the attack's CFS share is `k_a / (N + k_a)`, so `W` buys it
/// `slice * k_a` thread-seconds' worth of share-weighted time: at least what
/// the serial slice gave an attack of width `<= k_a`.
pub(crate) fn stretched_window(
    slice: Duration,
    host_threads: usize,
    width_pool: usize,
) -> Duration {
    let width = u128::try_from(width_pool.max(1)).unwrap_or(1);
    let host = u128::try_from(host_threads).unwrap_or(0);
    let nanos = slice.as_nanos().saturating_mul(host.saturating_add(width)) / width;
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}

/// The window a site would race with, or `Err(route)` when the stretch does
/// not fit. `slice_deadline` is the site's `upfront_pgd_deadline()`;
/// `overall_deadline` the verifier ledger's overall deadline.
pub(crate) fn plan_window(
    site_start: Instant,
    slice_deadline: Option<Instant>,
    overall_deadline: Option<Instant>,
    host_threads: usize,
    width_pool: usize,
) -> std::result::Result<RaceWindow, AdmitRoute> {
    let Some(slice_deadline) = slice_deadline else {
        return Err(AdmitRoute::NoSlice);
    };
    let slice = slice_deadline.saturating_duration_since(site_start);
    if slice.is_zero() {
        return Err(AdmitRoute::NoSlice);
    }
    let window = stretched_window(slice, host_threads, width_pool);
    let fits = match (
        overall_deadline.and_then(|deadline| deadline.checked_sub(GATE_RESERVE)),
        site_start.checked_add(window),
    ) {
        (Some(latest_end), Some(end)) => end <= latest_end,
        _ => false,
    };
    if !fits {
        return Err(AdmitRoute::StretchDoesNotFit);
    }
    Ok(RaceWindow {
        slice,
        window,
        host_threads,
        width_pool,
    })
}

/// When a DECIDED proof stops waiting for the attack thread's in-flight
/// adjudication (follow-up (a)): the designed gate budget [`GATE_RESERVE`]
/// after the decision, never later than `overall_deadline - JOIN_RESERVE`,
/// never before the decision itself.
///
/// The window `W <= overall_deadline - GATE_RESERVE` bounds when a candidate
/// can ARRIVE, not how long its gate takes. Without this bound a gate that
/// outlives the reserve holds the join, and the dispatch's deadline gate —
/// which reads the clock after the lane returns — turns a `Verified` decided
/// in time into `Timeout`.
pub(crate) fn decided_join_deadline(
    decided_at: Instant,
    overall_deadline: Option<Instant>,
) -> Instant {
    let budget_end = decided_at.checked_add(GATE_RESERVE).unwrap_or(decided_at);
    match overall_deadline.and_then(|deadline| deadline.checked_sub(JOIN_RESERVE)) {
        Some(cap) => budget_end.min(cap).max(decided_at),
        None => budget_end,
    }
}

/// The attack's deadline after an UNDECIDED proof returned at `returned_at`
/// (follow-up (c)): `site_start + slice + p * N / (N + k_a)` with
/// `p = returned_at - site_start`, capped at the stretched end.
///
/// While both sides ran, the attack's CFS share was at worst
/// `k_a / (N + k_a)` of its pool (both pools saturated — the model
/// [`stretched_window`] is sized by); alone afterwards it runs at its full
/// rate. It has therefore had the serial slice's CPU-seconds exactly when
/// `p * k_a / (N + k_a) + (D - returned_at) = slice`, i.e. at `D` above. And
/// `D <= slice_deadline + p`: the lanes after the race never start later than
/// the serial order (the attack's slice, then the proof) would start them.
/// For `p >= W` the cap applies and nothing is shrunk.
pub(crate) fn undecided_attack_deadline(
    site_start: Instant,
    window: &RaceWindow,
    returned_at: Instant,
    stretched_end: Instant,
) -> Instant {
    let proof_wall = returned_at.saturating_duration_since(site_start);
    let width = u128::try_from(window.width_pool.max(1)).unwrap_or(1);
    let host = u128::try_from(window.host_threads).unwrap_or(0);
    let share_lost = proof_wall.as_nanos().saturating_mul(host) / host.saturating_add(width);
    let share_lost = Duration::from_nanos(u64::try_from(share_lost).unwrap_or(u64::MAX));
    site_start
        .checked_add(window.slice)
        .and_then(|serial_end| serial_end.checked_add(share_lost))
        .map_or(stretched_end, |deadline| deadline.min(stretched_end))
}

/// The pure admission chain (everything but the memory lease and the pool),
/// in the order of [`AdmitRoute`].
pub(crate) fn admit(
    facts: &AdmissionFacts,
    window: std::result::Result<RaceWindow, AdmitRoute>,
) -> std::result::Result<RaceWindow, AdmitRoute> {
    if !facts.publisher_armed {
        return Err(AdmitRoute::PublisherUnarmed);
    }
    if facts.margin_row_lane_live {
        return Err(AdmitRoute::MarginRowOwnsConcurrency);
    }
    if facts.shared_device {
        return Err(AdmitRoute::SharedDevice);
    }
    if facts.sign_net {
        return Err(AdmitRoute::BnnSerial);
    }
    if facts.qdq_report_only {
        return Err(AdmitRoute::QdqReportOnly);
    }
    window
}

/// What the scored publisher answered when it RETURNED (an upheld,
/// published `sat` never returns: the process ends).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ConcurrentSatPublication {
    /// No scored publication target is armed (unreachable after admission).
    NotArmed,
    /// The dispatch's deadline gate turned the candidate into `Timeout`.
    SuppressedAtDeadline,
    /// The QDQ report-only latch withheld it (as the main path would).
    Withheld,
    /// The trusted-ORT gate or the VNN-LIB 2.0 normalization downgraded it.
    Downgraded,
    /// The single terminal-publication claim was already taken: a CONFLICT
    /// (unreachable by admission — logged as an emergency, never written).
    ClaimTaken,
    /// The publisher refused a `sat` the gate never confirmed.
    Unconfirmed,
    /// The write failed after the claim; the claim was given back.
    WriteFailed(String),
    /// The main thread had already CLOSED the attack door — its own verdict
    /// is committed (follow-up (b)) — so the candidate was not published.
    DoorClosed,
}

impl ConcurrentSatPublication {
    /// Whether the candidate must travel back to the main thread as the
    /// lane's `Violated` result — exactly today's serial hand-off — because
    /// nothing adjudicated it here (`NotArmed`) or the adjudicated `sat`
    /// could not be written (`WriteFailed`).
    pub(crate) const fn returns_candidate_to_main(&self) -> bool {
        matches!(self, Self::NotArmed | Self::WriteFailed(_))
    }

    /// The `done` line's outcome token for this publication.
    pub(crate) const fn outcome(&self) -> AttackOutcome {
        match self {
            Self::NotArmed => AttackOutcome::CandidateUnpublished,
            Self::SuppressedAtDeadline => AttackOutcome::CandidateSuppressed,
            Self::Withheld => AttackOutcome::CandidateWithheld,
            Self::Downgraded | Self::Unconfirmed => AttackOutcome::CandidateDowngraded,
            Self::ClaimTaken => AttackOutcome::Conflict,
            Self::WriteFailed(_) => AttackOutcome::CandidateWriteFailed,
            Self::DoorClosed => AttackOutcome::CandidateDoorClosed,
        }
    }
}

/// Timings the publisher prints on its `done` line before it ends the
/// process.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConcurrentSatReport {
    /// The race site.
    pub(crate) site: RaceSite,
    /// Attack wall until the candidate.
    pub(crate) attack_wall: Duration,
    /// Attack CPU (summed over the private pool's workers) until then.
    pub(crate) attack_cpu: Option<Duration>,
    /// `k_a`.
    pub(crate) width_pool: usize,
}

impl ConcurrentSatReport {
    /// The `[attack-concurrent] done ... outcome=sat_published` line.
    pub(crate) fn published_line(&self) -> String {
        format!(
            "[attack-concurrent] done site={} outcome={} attack_wall_ms={} attack_cpu_ms={} \
             width_pool={}",
            self.site.token(),
            AttackOutcome::SatPublished.token(),
            self.attack_wall.as_millis(),
            ms_token(self.attack_cpu),
            self.width_pool,
        )
    }
}

/// A candidate's adjudication — the trusted gate and the terminal
/// publication — as an OWNED job, so it can run on a detached thread that
/// may outlive the race scope (follow-up (a)). Returns only when the
/// publication was declined; an upheld, published `sat` ends the process.
pub(crate) type AdjudicationJob = Box<dyn FnOnce() -> ConcurrentSatPublication + Send + 'static>;

/// What the `publish` hook makes of a candidate on the attack thread.
pub(crate) enum PreparedSat {
    /// Declined before any adjudication (the dispatch's deadline gate, a
    /// render failure, no publisher).
    Declined(ConcurrentSatPublication),
    /// The candidate's adjudication, to run off the race scope.
    Adjudicate(AdjudicationJob),
}

/// The scored sat pipeline, as the race sites see it: PREPARE a candidate on
/// the attack thread (borrowing the dispatch's context) into a declined
/// answer or an owned adjudication job.
pub(crate) type PublishFn<'a> =
    &'a (dyn Fn(&BetaCrownResult, &ConcurrentSatReport) -> PreparedSat + Sync);

/// The publisher a site falls back to when it has no context. Admission
/// refuses a context-less site (`publisher_unarmed`), so a race never reaches
/// this; it exists so the type is total, and it answers `NotArmed`, which
/// hands any candidate back to the main thread exactly as the serial site.
pub(crate) fn unarmed_publish(
    _result: &BetaCrownResult,
    _report: &ConcurrentSatReport,
) -> PreparedSat {
    PreparedSat::Declined(ConcurrentSatPublication::NotArmed)
}

/// What a race site receives from the dispatch.
pub(crate) struct ConcurrentAttackCtx<'a> {
    /// The wrapper's facts (`None`: interactive / test faces).
    pub(crate) wrapper: Option<ConcurrentAttackWrapperFacts>,
    /// A scored publication target is armed.
    pub(crate) publisher_armed: bool,
    /// The scored sat pipeline.
    pub(crate) publish: PublishFn<'a>,
}

impl ConcurrentAttackCtx<'_> {
    /// The context's share of [`AdmissionFacts`].
    pub(crate) fn admission_facts(
        ctx: Option<&Self>,
        shared_device: bool,
        sign_net: bool,
    ) -> AdmissionFacts {
        let wrapper = ctx.and_then(|ctx| ctx.wrapper);
        AdmissionFacts {
            publisher_armed: ctx.is_some_and(|ctx| ctx.publisher_armed) && wrapper.is_some(),
            margin_row_lane_live: wrapper.is_some_and(|facts| facts.margin_row_lane_live),
            shared_device,
            sign_net,
            qdq_report_only: super::qdq_probe::qdq_relaxation_engaged(),
        }
    }
}

/// How the attack side of a race ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttackOutcome {
    /// An upheld `sat` was published (printed by the publisher, which exits).
    SatPublished,
    /// The gate (or normalization) downgraded the candidate.
    CandidateDowngraded,
    /// The deadline gate suppressed the candidate.
    CandidateSuppressed,
    /// The QDQ report-only latch withheld the candidate.
    CandidateWithheld,
    /// No publisher: the candidate went back to the main thread.
    CandidateUnpublished,
    /// The adjudicated `sat` could not be written: back to the main thread.
    CandidateWriteFailed,
    /// The terminal claim was already taken (emergency).
    Conflict,
    /// The main thread had closed the attack door: its verdict is committed.
    CandidateDoorClosed,
    /// The proof decided, and the candidate's adjudication outlived the
    /// bounded join: it continues detached, inside the attack door.
    AdjudicationAbandoned,
    /// The proof decided and cancelled the attack.
    CancelledByProof,
    /// The proof gave up early; the attack was cut at its serial-equivalent
    /// deadline ([`undecided_attack_deadline`]).
    WindowShrunk,
    /// The stretched window ran out.
    Deadline,
    /// The attack finished its restarts inside the window.
    Exhausted,
    /// The attack returned an error (treated as "no candidate").
    Error,
    /// The attack panicked (treated as "no candidate").
    Panic,
}

impl AttackOutcome {
    /// Stable token for the `done` line.
    pub(crate) const fn token(self) -> &'static str {
        match self {
            Self::SatPublished => "sat_published",
            Self::CandidateDowngraded => "candidate_downgraded",
            Self::CandidateSuppressed => "candidate_suppressed",
            Self::CandidateWithheld => "candidate_withheld",
            Self::CandidateUnpublished => "candidate_unpublished",
            Self::CandidateWriteFailed => "candidate_write_failed",
            Self::Conflict => "conflict",
            Self::CandidateDoorClosed => "candidate_door_closed",
            Self::AdjudicationAbandoned => "adjudication_abandoned",
            Self::CancelledByProof => "cancelled_by_proof",
            Self::WindowShrunk => "window_shrunk",
            Self::Deadline => "deadline",
            Self::Exhausted => "exhausted",
            Self::Error => "error",
            Self::Panic => "panic",
        }
    }
}

/// An attack's hit: the counterexample input and its internal output.
pub(crate) type AttackHit = Option<(ArrayD<f32>, ArrayD<f32>)>;

/// Everything [`run_race`] needs besides the two closures.
pub(crate) struct RacePlan<'a> {
    /// The site.
    pub(crate) site: RaceSite,
    /// The admitted window.
    pub(crate) window: RaceWindow,
    /// When the site started (the attack window's origin).
    pub(crate) site_start: Instant,
    /// Today's (unstretched) attack deadline: the serial fallback's.
    pub(crate) slice_deadline: Instant,
    /// The verifier ledger's overall deadline — the dispatch's authoritative
    /// deadline gate. Caps a decided proof's bounded join
    /// ([`decided_join_deadline`]).
    pub(crate) overall_deadline: Option<Instant>,
    /// The lane's start, for a returned `Violated`'s `time_elapsed`.
    pub(crate) lane_start: Instant,
    /// The scored sat pipeline.
    pub(crate) publish: PublishFn<'a>,
}

/// Concurrent attack threads alive in this process (debug evidence that the
/// scope joined them before the main thread publishes).
static LIVE_ATTACK_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Whether any race site spawned in this process.
static RACE_ENGAGED: AtomicBool = AtomicBool::new(false);

/// `Some(live attack threads)` when a race ran in this process; `None` on
/// every dark run (nothing to report, nothing recorded).
pub(crate) fn live_threads_if_engaged() -> Option<usize> {
    RACE_ENGAGED
        .load(Ordering::SeqCst)
        .then(|| LIVE_ATTACK_THREADS.load(Ordering::SeqCst))
}

struct LiveThread;

impl LiveThread {
    fn enter() -> Self {
        LIVE_ATTACK_THREADS.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

impl Drop for LiveThread {
    fn drop(&mut self) {
        LIVE_ATTACK_THREADS.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Why the race cancelled the attack's token (the first reason wins).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CancelReason {
    /// Not cancelled.
    None,
    /// The proof decided.
    ProofDecided,
    /// The proof panicked.
    ProofPanicked,
    /// The proof gave up early; the attack reached its serial-equivalent
    /// deadline ([`undecided_attack_deadline`]).
    WindowShrunk,
}

impl CancelReason {
    /// Stable token for the `done` line.
    pub(crate) const fn token(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ProofDecided => "proof_decided",
            Self::ProofPanicked => "proof_panicked",
            Self::WindowShrunk => "window_shrunk",
        }
    }

    const fn code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::ProofDecided => 1,
            Self::ProofPanicked => 2,
            Self::WindowShrunk => 3,
        }
    }

    const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::ProofDecided,
            2 => Self::ProofPanicked,
            3 => Self::WindowShrunk,
            _ => Self::None,
        }
    }
}

/// How the attack's candidate (if any) was adjudicated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Adjudication {
    /// No candidate.
    None,
    /// The `publish` hook declined it before any adjudication.
    Declined,
    /// The adjudication answered within the join.
    Answered,
    /// The proof decided and the adjudication outlived the bounded join.
    Abandoned,
    /// The adjudication panicked (treated as "no candidate").
    Panicked,
}

impl Adjudication {
    /// Stable token for the `done` line.
    pub(crate) const fn token(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Declined => "declined",
            Self::Answered => "answered",
            Self::Abandoned => "abandoned",
            Self::Panicked => "panicked",
        }
    }
}

/// A detached adjudication's answer: `Err(())` when it panicked.
type AdjudicationAnswer = std::result::Result<ConcurrentSatPublication, ()>;

#[derive(Debug, Default)]
struct RaceState {
    /// Set once the proof DECIDED: the attack thread stops waiting for an
    /// in-flight adjudication at this instant (never on a proof panic).
    join_deadline: Option<Instant>,
    /// The attack thread finished (its report is ready to join).
    attack_done: bool,
    /// A detached adjudication's answer, until the attack thread takes it.
    answer: Option<AdjudicationAnswer>,
    /// The attack thread stopped waiting for the answer (the join deadline
    /// passed first).
    abandoned: bool,
}

/// State the main thread, the attack thread and a detached adjudication
/// share: the bounded join (a), the attack-done signal for the undecided
/// shrink (c), and the adjudication's answer.
#[derive(Debug, Default)]
struct RaceShared {
    state: Mutex<RaceState>,
    changed: Condvar,
    cancel_reason: AtomicU8,
}

impl RaceShared {
    fn lock(&self) -> MutexGuard<'_, RaceState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wait_for<'a>(
        &self,
        guard: MutexGuard<'a, RaceState>,
        timeout: Option<Duration>,
    ) -> MutexGuard<'a, RaceState> {
        match timeout {
            Some(timeout) => {
                self.changed
                    .wait_timeout(guard, timeout)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0
            }
            None => self
                .changed
                .wait(guard)
                .unwrap_or_else(PoisonError::into_inner),
        }
    }

    /// Cancel the attack's token, recording WHY (the first reason wins; the
    /// reason is stored before the token, so a reader that sees the
    /// cancellation also sees its reason).
    fn cancel(&self, reason: CancelReason, token: &PgdCancelToken) {
        let _ = self.cancel_reason.compare_exchange(
            CancelReason::None.code(),
            reason.code(),
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
        token.cancel();
    }

    fn cancel_reason(&self) -> CancelReason {
        CancelReason::from_code(self.cancel_reason.load(Ordering::SeqCst))
    }

    /// Bound the attack thread's wait for an in-flight adjudication (the
    /// earliest deadline wins).
    fn set_join_deadline(&self, at: Instant) {
        {
            let mut state = self.lock();
            state.join_deadline = Some(state.join_deadline.map_or(at, |current| current.min(at)));
        }
        self.changed.notify_all();
    }

    fn join_deadline(&self) -> Option<Instant> {
        self.lock().join_deadline
    }

    fn mark_attack_done(&self) {
        self.lock().attack_done = true;
        self.changed.notify_all();
    }

    /// Wait until the attack thread finished or `until`; `true` = finished.
    fn wait_attack_done_until(&self, until: Instant) -> bool {
        let mut state = self.lock();
        loop {
            if state.attack_done {
                return true;
            }
            let now = Instant::now();
            if now >= until {
                return false;
            }
            state = self.wait_for(state, Some(until - now));
        }
    }

    /// Post a detached adjudication's answer; `false` when nobody listens any
    /// more (the attack thread abandoned it at the join deadline).
    fn post_answer(&self, answer: AdjudicationAnswer) -> bool {
        let listened = {
            let mut state = self.lock();
            state.answer = Some(answer);
            !state.abandoned
        };
        self.changed.notify_all();
        listened
    }

    /// The attack thread's wait for its adjudication: unbounded while no
    /// join deadline is set (the proof is still running, or gave up), until
    /// the join deadline once one is. `None`: abandoned.
    fn await_answer(&self) -> Option<AdjudicationAnswer> {
        let mut state = self.lock();
        loop {
            if let Some(answer) = state.answer.take() {
                return Some(answer);
            }
            let timeout = match state.join_deadline {
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        state.abandoned = true;
                        return None;
                    }
                    Some(deadline - now)
                }
                None => None,
            };
            state = self.wait_for(state, timeout);
        }
    }
}

/// Marks the attack thread finished when it ends (panics included), so the
/// main thread's shrink wait (c) wakes at once.
struct AttackDone<'a>(&'a RaceShared);

impl Drop for AttackDone<'_> {
    fn drop(&mut self) {
        self.0.mark_attack_done();
    }
}

/// Cancels the token when dropped unless disarmed: a proof that PANICS
/// cancels the attack, so the scope's join (and the panic's propagation) is
/// not held up by a full attack window.
///
/// It deliberately does NOT bound the wait for an in-flight adjudication
/// (review(attack-conc-followups)): nothing on the scored path catches a
/// proof panic, so it ends the PROCESS with the `unknown` placeholder on
/// disk and no main-thread verdict follows. A candidate already inside the
/// trusted gate is then the only verdict this instance can still reach —
/// abandoning it (as an `Instant::now()` join deadline here did) loses an
/// upheld `sat` the serial order and the reviewed base both published, and
/// the detached gate dies with the process. The gate is bounded by its own
/// instance-deadline logic and the watchdogs; waiting for it costs nothing.
struct CancelOnUnwind<'a> {
    shared: &'a RaceShared,
    token: &'a PgdCancelToken,
    armed: bool,
}

impl Drop for CancelOnUnwind<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.shared.cancel(CancelReason::ProofPanicked, self.token);
        }
    }
}

/// What the attack thread hands back at the join.
struct AttackThreadReport {
    outcome: AttackOutcome,
    adjudication: Adjudication,
    wall: Duration,
    cpu: Option<Duration>,
    /// A candidate that must travel to the main thread as the lane's result.
    candidate: Option<BetaCrownResult>,
}

/// What a race observed: the `done` line's decision facts, returned by
/// [`run_race_observed`] so tests can assert them (production reads the
/// `done` line; [`run_race`] drops the observation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct RaceObservation {
    /// How the attack side ended.
    pub(crate) outcome: AttackOutcome,
    /// How its candidate was adjudicated.
    pub(crate) adjudication: Adjudication,
    /// Whether the proof decided.
    pub(crate) proof_decided: bool,
    /// Why the token was cancelled, if it was.
    pub(crate) cancel: CancelReason,
    /// The attack's effective deadline: `W`'s end, or the undecided shrink.
    pub(crate) attack_deadline: Instant,
    /// A decided proof's bounded join deadline.
    pub(crate) join_deadline: Option<Instant>,
}

/// Whether a proof result is DECIDED (cancels the attack).
pub(crate) fn proof_decided(result: &Result<BetaCrownResult>) -> bool {
    matches!(
        result,
        Ok(BetaCrownResult {
            result: BabVerificationStatus::Verified | BabVerificationStatus::Violated { .. },
            ..
        })
    )
}

/// The lane's `Violated` result for an attack hit (the shape the serial
/// sites build).
pub(crate) fn violated_result(
    counterexample: &ArrayD<f32>,
    output: &ArrayD<f32>,
    lane_start: Instant,
) -> BetaCrownResult {
    BetaCrownResult {
        result: BabVerificationStatus::Violated {
            counterexample: counterexample.iter().copied().collect(),
            output: output.iter().copied().collect(),
        },
        domains_explored: 0,
        domains_verified: 0,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed: lane_start.elapsed(),
        output_bounds: None,
    }
}

fn ms_token(duration: Option<Duration>) -> String {
    duration.map_or_else(|| "na".to_owned(), |d| d.as_millis().to_string())
}

/// CPU nanoseconds consumed so far by the CALLING thread
/// (`/proc/thread-self/schedstat`, first field), `None` when unavailable.
#[cfg(target_os = "linux")]
fn thread_cpu_ns() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
    text.split_ascii_whitespace()
        .next()?
        .parse::<u64>()
        .ok()
        .filter(|&ns| ns > 0)
}

#[cfg(not(target_os = "linux"))]
fn thread_cpu_ns() -> Option<u64> {
    None
}

/// CPU consumed by every worker of `pool` (the attack's private pool runs
/// nothing else). `None` when any worker cannot read its clock.
fn pool_cpu(pool: &rayon::ThreadPool) -> Option<Duration> {
    pool.broadcast(|_| thread_cpu_ns())
        .into_iter()
        .try_fold(0u64, |sum, ns| ns.map(|ns| sum.saturating_add(ns)))
        .map(Duration::from_nanos)
}

/// Build the attack's private pool (`attack-concurrent-{i}`).
pub(crate) fn build_pool(width: usize) -> Option<rayon::ThreadPool> {
    match rayon::ThreadPoolBuilder::new()
        .num_threads(width.max(1))
        .thread_name(|i| format!("attack-concurrent-{i}"))
        .stack_size(ATTACK_THREAD_STACK_BYTES)
        .build()
    {
        Ok(pool) => Some(pool),
        Err(error) => {
            eprintln!("[attack-concurrent] private {width}-thread pool REFUSED: {error}");
            None
        }
    }
}

/// `k_a` with the router off: every core (the static split, `s = 1/2` when the
/// proof saturates its own pool).
pub(crate) fn static_width() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// `N`: the width of the pool the proof runs on (the global pool, read from
/// the calling thread).
pub(crate) fn proof_pool_width() -> usize {
    rayon::current_num_threads()
}

/// Print the site's admit line.
pub(crate) fn print_admit_line(
    site: RaceSite,
    route: AdmitRoute,
    window: Option<&RaceWindow>,
    lease: &str,
) {
    eprintln!(
        "[attack-concurrent] admit site={} route={} width_pool={} host_threads={} slice_ms={} \
         window_ms={} stretch={} lease={lease}",
        site.token(),
        route.token(),
        window.map_or(0, |w| w.width_pool),
        window.map_or_else(proof_pool_width, |w| w.host_threads),
        window.map_or(0, |w| w.slice.as_millis()),
        window.map_or(0, |w| w.window.as_millis()),
        window.map_or_else(|| "na".to_owned(), |w| format!("{:.3}", w.stretch())),
    );
}

/// Memory admission: the live optional-heavy floor, and the R4 lease when one
/// is armed. Returns `(admitted, token)`.
pub(crate) fn admit_memory() -> (bool, &'static str) {
    use ny_core::router::lease::LeaseHolder;
    let armed = ny_core::memory_lease::global().is_armed();
    let admitted = crate::router::lease::admit(
        LeaseHolder::AttackConcurrent,
        crate::commands::vnncomp::optional_heavy_memory_allowed_now,
    );
    let token = match (armed, admitted) {
        (true, true) => "granted",
        (true, false) => "denied",
        (false, true) => "unarmed",
        (false, false) => "floor_denied",
    };
    (admitted, token)
}

/// A site's admitted race: its window and its private pool.
pub(crate) struct AdmittedRace {
    /// The admitted window.
    pub(crate) window: RaceWindow,
    /// The attack's private pool.
    pub(crate) pool: rayon::ThreadPool,
}

/// A site's full admission, run only when `attack.concurrent` is armed: the
/// pure chain ([`admit`] over [`plan_window`]), then memory, then the pool.
/// Prints exactly one `[attack-concurrent] admit` line and notes it in the
/// flight record; `None` means the site runs today's serial code verbatim.
#[allow(clippy::too_many_arguments)]
pub(crate) fn admit_site(
    site: RaceSite,
    ctx: Option<&ConcurrentAttackCtx<'_>>,
    shared_device: bool,
    sign_net: bool,
    site_start: Instant,
    slice_deadline: Option<Instant>,
    overall_deadline: Option<Instant>,
) -> Option<AdmittedRace> {
    let planned = plan_window(
        site_start,
        slice_deadline,
        overall_deadline,
        proof_pool_width(),
        static_width(),
    );
    let facts = ConcurrentAttackCtx::admission_facts(ctx, shared_device, sign_net);
    let refuse = |route: AdmitRoute, window: Option<&RaceWindow>, lease: &str| {
        print_admit_line(site, route, window, lease);
        crate::flight::note(
            "attack_concurrent_admit",
            crate::flight::FlightStatus::Skipped,
            Some(format!("site={} route={}", site.token(), route.token())),
        );
    };
    let window = match admit(&facts, planned) {
        Ok(window) => window,
        Err(route) => {
            refuse(route, planned.as_ref().ok(), "na");
            return None;
        }
    };
    let (memory_ok, lease) = admit_memory();
    if !memory_ok {
        refuse(AdmitRoute::LeaseDenied, Some(&window), lease);
        return None;
    }
    let Some(pool) = build_pool(window.width_pool) else {
        crate::router::lease::release(ny_core::router::lease::LeaseHolder::AttackConcurrent);
        refuse(AdmitRoute::PoolBuildFailed, Some(&window), lease);
        return None;
    };
    print_admit_line(site, AdmitRoute::Spawned, Some(&window), lease);
    crate::flight::note(
        "attack_concurrent_admit",
        crate::flight::FlightStatus::Ran,
        Some(format!(
            "site={} route=spawned width_pool={} host_threads={} slice_ms={} window_ms={} \
             lease={lease}",
            site.token(),
            window.width_pool,
            window.host_threads,
            window.slice.as_millis(),
            window.window.as_millis(),
        )),
    );
    Some(AdmittedRace { window, pool })
}

/// Whether an engine is an ACCELERATOR: every engine whose provenance does
/// not name a CPU implementation (`compute-device-cpu`, `faer-cpu`,
/// `naive-cpu`, `accelerate-veclib-cpu`). An unknown provenance counts as an
/// accelerator, so [`shared_device`] errs towards standing down.
pub(crate) fn engine_is_accelerator(engine: &dyn ny_core::GemmEngine) -> bool {
    !engine.backend_provenance().contains("cpu")
}

/// Whether the attack and the proof would SHARE an accelerator: the attack's
/// engine is one AND the proof reaches one (its own GEMM handle, or the GPU
/// BaB). A GPU-steered attack beside a CPU proof (the attack-steering wgpu
/// device on a CPU-proof category) is not shared and is admitted; the
/// attack's CPU side then runs in its private pool as everywhere else. The
/// CPU proof device's own cuBLAS offload of very large GEMMs is not a handle
/// the site can see and is not counted.
pub(crate) fn shared_device(
    attack_engine: Option<&dyn ny_core::GemmEngine>,
    proof_engine: Option<&dyn ny_core::GemmEngine>,
    proof_gpu_bab: bool,
) -> bool {
    attack_engine.is_some_and(engine_is_accelerator)
        && (proof_gpu_bab || proof_engine.is_some_and(engine_is_accelerator))
}

/// Whether a graph model has a `Sign` layer (the BNN lanes stay serial).
pub(crate) fn graph_has_sign(graph: &ny_propagate::GraphNetwork) -> bool {
    graph.node_names().iter().any(|name| {
        graph
            .node(name)
            .is_some_and(|node| matches!(node.layer(), ny_propagate::Layer::Sign(_)))
    })
}

/// Whether a sequential model has a `Sign` layer.
pub(crate) fn network_has_sign(network: &ny_propagate::Network) -> bool {
    network
        .layers()
        .iter()
        .any(|layer| matches!(layer, ny_propagate::Layer::Sign(_)))
}

/// Returns the lane's memory lease when the race ends (a no-op unarmed). Held
/// through an `Arc` by the race AND by a detached adjudication, so the lease
/// (which models the gate's ORT session) is returned when the LAST of them
/// ends.
struct LeaseRelease;

impl Drop for LeaseRelease {
    fn drop(&mut self) {
        crate::router::lease::release(ny_core::router::lease::LeaseHolder::AttackConcurrent);
    }
}

/// Run the attack and the proof concurrently; see the module docs.
///
/// `attack(token, deadline)` must run the site's IDENTICAL attack call with
/// `PgdConfig { deadline: Some(deadline), cancel: Some(token), .. }`; it is
/// `Fn` so the spawn-failure fallback can still run it serially. `proof()` is
/// the lane's unchanged remainder.
pub(crate) fn run_race<A, P>(
    plan: RacePlan<'_>,
    pool: rayon::ThreadPool,
    attack: A,
    proof: P,
) -> Result<BetaCrownResult>
where
    A: Fn(PgdCancelToken, Instant) -> Result<AttackHit> + Sync,
    P: FnOnce() -> Result<BetaCrownResult>,
{
    run_race_observed(plan, pool, attack, proof).0
}

/// [`run_race`], also returning what the race observed (`None` when the
/// attack thread could not be spawned and the serial order ran instead).
pub(crate) fn run_race_observed<A, P>(
    plan: RacePlan<'_>,
    pool: rayon::ThreadPool,
    attack: A,
    proof: P,
) -> (Result<BetaCrownResult>, Option<RaceObservation>)
where
    A: Fn(PgdCancelToken, Instant) -> Result<AttackHit> + Sync,
    P: FnOnce() -> Result<BetaCrownResult>,
{
    let lease = Arc::new(LeaseRelease);
    RACE_ENGAGED.store(true, Ordering::SeqCst);
    let attack_deadline = plan
        .site_start
        .checked_add(plan.window.window)
        .unwrap_or(plan.slice_deadline);
    let token = PgdCancelToken::new();
    let shared = Arc::new(RaceShared::default());
    let site = plan.site;
    let window = plan.window;
    let width_pool = window.width_pool;
    let publish = plan.publish;
    let lane_start = plan.lane_start;
    let attack_ref = &attack;
    let pool_ref = &pool;
    let shared_ref = &shared;
    let lease_ref = &lease;

    std::thread::scope(|scope| {
        let thread_token = token.clone();
        let spawned = std::thread::Builder::new()
            .name("attack-concurrent".to_owned())
            .stack_size(ATTACK_THREAD_STACK_BYTES)
            .spawn_scoped(scope, move || {
                let _live = LiveThread::enter();
                let _done = AttackDone(shared_ref);
                attack_thread(
                    AttackThreadPlan {
                        site,
                        token: &thread_token,
                        attack_deadline,
                        publish,
                        lane_start,
                        width_pool,
                    },
                    pool_ref,
                    attack_ref,
                    shared_ref,
                    lease_ref,
                )
            });
        let handle = match spawned {
            Ok(handle) => handle,
            Err(error) => {
                // Today's serial order, on this thread, with today's slice.
                eprintln!(
                    "[attack-concurrent] spawn REFUSED ({error}); site={} runs the serial order",
                    site.token()
                );
                let serial = match attack(PgdCancelToken::new(), plan.slice_deadline) {
                    Ok(Some((counterexample, output))) => {
                        Ok(violated_result(&counterexample, &output, lane_start))
                    }
                    Ok(None) => proof(),
                    Err(error) => Err(error),
                };
                return (serial, None);
            }
        };

        let mut guard = CancelOnUnwind {
            shared: shared_ref,
            token: &token,
            armed: true,
        };
        let proof_started = Instant::now();
        let proof_result = proof();
        let returned_at = Instant::now();
        let proof_wall = returned_at.saturating_duration_since(proof_started);
        guard.armed = false;
        drop(guard);
        let decided = proof_decided(&proof_result);
        let mut effective_deadline = attack_deadline;
        let mut cancelled_at = None;
        if decided {
            // (a) cancel the attack AND bound the wait for an in-flight
            // adjudication, so a slow gate cannot carry the certified verdict
            // past the dispatch's deadline gate.
            shared.cancel(CancelReason::ProofDecided, &token);
            cancelled_at = Some(Instant::now());
            shared.set_join_deadline(decided_join_deadline(returned_at, plan.overall_deadline));
        } else {
            // (c) an undecided proof keeps the attack to its serial-
            // equivalent deadline, not the stretched window.
            let shrunk =
                undecided_attack_deadline(plan.site_start, &window, returned_at, attack_deadline);
            if shrunk < attack_deadline {
                effective_deadline = shrunk;
                if !shared.wait_attack_done_until(shrunk) {
                    shared.cancel(CancelReason::WindowShrunk, &token);
                    cancelled_at = Some(Instant::now());
                }
            }
        }

        let join_started = Instant::now();
        let report = handle.join().unwrap_or(AttackThreadReport {
            outcome: AttackOutcome::Panic,
            adjudication: Adjudication::None,
            wall: Duration::ZERO,
            cpu: None,
            candidate: None,
        });
        let join_wait = join_started.elapsed();
        let s_eff = match report.cpu {
            Some(cpu) if !report.wall.is_zero() => {
                format!("{:.3}", cpu.as_secs_f64() / report.wall.as_secs_f64())
            }
            _ => "na".to_owned(),
        };
        let cancel = shared.cancel_reason();
        let join_deadline = shared.join_deadline();
        let line = format!(
            "[attack-concurrent] done site={} outcome={} attack_wall_ms={} attack_cpu_ms={} \
             proof_wall_during_ms={} s_eff={s_eff} cancel_to_join_ms={} join_wait_ms={} \
             proof_decided={} width_pool={width_pool} adjudication={} cancel={} \
             attack_deadline_ms={}",
            site.token(),
            report.outcome.token(),
            report.wall.as_millis(),
            ms_token(report.cpu),
            proof_wall.as_millis(),
            ms_token(cancelled_at.map(|at| at.elapsed())),
            join_wait.as_millis(),
            u8::from(decided),
            report.adjudication.token(),
            cancel.token(),
            effective_deadline
                .saturating_duration_since(plan.site_start)
                .as_millis(),
        );
        eprintln!("{line}");
        crate::flight::note(
            "attack_concurrent",
            crate::flight::FlightStatus::Ran,
            Some(line.trim_start_matches("[attack-concurrent] ").to_owned()),
        );
        let observation = RaceObservation {
            outcome: report.outcome,
            adjudication: report.adjudication,
            proof_decided: decided,
            cancel,
            attack_deadline: effective_deadline,
            join_deadline,
        };
        let result = match report.candidate {
            // Nothing adjudicated the candidate here (or its write failed):
            // hand it to the main thread exactly as the serial site does.
            Some(candidate) => Ok(candidate),
            None => proof_result,
        };
        (result, Some(observation))
    })
}

/// The attack thread's fixed inputs.
struct AttackThreadPlan<'a> {
    site: RaceSite,
    token: &'a PgdCancelToken,
    attack_deadline: Instant,
    publish: PublishFn<'a>,
    lane_start: Instant,
    width_pool: usize,
}

/// The attack thread: the identical attack inside the private pool, then —
/// outside the pool — the candidate's preparation and its (detached)
/// adjudication.
fn attack_thread<A>(
    plan: AttackThreadPlan<'_>,
    pool: &rayon::ThreadPool,
    attack: &A,
    shared: &Arc<RaceShared>,
    lease: &Arc<LeaseRelease>,
) -> AttackThreadReport
where
    A: Fn(PgdCancelToken, Instant) -> Result<AttackHit> + Sync,
{
    let AttackThreadPlan {
        site,
        token,
        attack_deadline,
        publish,
        lane_start,
        width_pool,
    } = plan;
    let started = Instant::now();
    let raw = pool.install(|| {
        let _faer = ny_propagate::faer_parallelism::nested_faer_par_scope();
        catch_unwind(AssertUnwindSafe(|| attack(token.clone(), attack_deadline)))
    });
    let wall = started.elapsed();
    let cpu = pool_cpu(pool);
    let report = |outcome, adjudication, candidate| AttackThreadReport {
        outcome,
        adjudication,
        wall,
        cpu,
        candidate,
    };
    match raw {
        Err(_) => report(AttackOutcome::Panic, Adjudication::None, None),
        Ok(Err(error)) => {
            eprintln!(
                "[attack-concurrent] attack error site={} (treated as no candidate): {error:#}",
                site.token()
            );
            report(AttackOutcome::Error, Adjudication::None, None)
        }
        Ok(Ok(None)) => report(
            if token.is_cancelled() {
                match shared.cancel_reason() {
                    CancelReason::WindowShrunk => AttackOutcome::WindowShrunk,
                    _ => AttackOutcome::CancelledByProof,
                }
            } else if Instant::now() >= attack_deadline {
                AttackOutcome::Deadline
            } else {
                AttackOutcome::Exhausted
            },
            Adjudication::None,
            None,
        ),
        Ok(Ok(Some((counterexample, output)))) => {
            let result = violated_result(&counterexample, &output, lane_start);
            eprintln!(
                "[attack-concurrent] candidate site={} attack_wall_ms={} attack_cpu_ms={} \
                 proof_cancelled={}",
                site.token(),
                wall.as_millis(),
                ms_token(cpu),
                u8::from(token.is_cancelled()),
            );
            let prepared = publish(
                &result,
                &ConcurrentSatReport {
                    site,
                    attack_wall: wall,
                    attack_cpu: cpu,
                    width_pool,
                },
            );
            let (answer, adjudication) = match prepared {
                PreparedSat::Declined(declined) => (Some(Ok(declined)), Adjudication::Declined),
                PreparedSat::Adjudicate(job) => match adjudicate(site, job, shared, lease) {
                    Some(Ok(answer)) => (Some(Ok(answer)), Adjudication::Answered),
                    Some(Err(())) => (Some(Err(())), Adjudication::Panicked),
                    None => (None, Adjudication::Abandoned),
                },
            };
            let publication = match answer {
                Some(Ok(publication)) => publication,
                // A panicking publisher is "no candidate", as an attack panic.
                Some(Err(())) => return report(AttackOutcome::Panic, adjudication, None),
                // The proof decided and the gate outlived the bounded join:
                // the proof's certified verdict travels; the adjudication
                // stays inside the attack door, which the main thread's
                // publication synchronizes with.
                None => return report(AttackOutcome::AdjudicationAbandoned, adjudication, None),
            };
            if publication.outcome() == AttackOutcome::CandidateDowngraded {
                // The serial order hands a downgraded candidate to the post-BaB
                // attack as its first seed (the ORT-divergence class: a genuine
                // violation often sits nearby). Here the proof's result travels
                // instead, so park the point in the best-margin tracker the
                // post-BaB lane already reads (margin 0 = an internal
                // counterexample). Guidance only: every candidate that lane
                // produces still passes the unchanged gate.
                super::best_margin_export::record_best_margin_candidate(0.0, &counterexample);
            }
            let candidate = publication.returns_candidate_to_main().then_some(result);
            report(publication.outcome(), adjudication, candidate)
        }
    }
}

/// Run a candidate's adjudication on a DETACHED `attack-concurrent-gate`
/// thread and wait for its answer: unboundedly while the proof is still
/// running or gave up, until the race's join deadline once the proof DECIDED
/// (follow-up (a)). `None`: abandoned — the adjudication keeps running
/// detached (it owns everything it reads, and it holds the lease), inside the
/// attack door the main thread's publication waits on or closes. When the
/// thread cannot be spawned the job runs here, as before the follow-up.
fn adjudicate(
    site: RaceSite,
    job: AdjudicationJob,
    shared: &Arc<RaceShared>,
    lease: &Arc<LeaseRelease>,
) -> Option<AdjudicationAnswer> {
    let slot = Arc::new(Mutex::new(Some(job)));
    let take = |slot: &Mutex<Option<AdjudicationJob>>| {
        slot.lock().unwrap_or_else(PoisonError::into_inner).take()
    };
    let detached_slot = Arc::clone(&slot);
    let detached_shared = Arc::clone(shared);
    let detached_lease = Arc::clone(lease);
    let spawned = std::thread::Builder::new()
        .name("attack-concurrent-gate".to_owned())
        .stack_size(ATTACK_THREAD_STACK_BYTES)
        .spawn(move || {
            let _lease = detached_lease;
            let Some(job) = take(&detached_slot) else {
                return;
            };
            let started = Instant::now();
            let answer = catch_unwind(AssertUnwindSafe(job)).map_err(|_| ());
            let outcome = match &answer {
                Ok(publication) => publication.outcome().token(),
                Err(()) => AttackOutcome::Panic.token(),
            };
            if !detached_shared.post_answer(answer) {
                eprintln!(
                    "[attack-concurrent] abandoned adjudication answered site={} outcome={outcome} \
                     gate_ms={}",
                    site.token(),
                    started.elapsed().as_millis(),
                );
            }
        });
    match spawned {
        Ok(_detached) => shared.await_answer(),
        Err(error) => {
            eprintln!(
                "[attack-concurrent] gate thread REFUSED ({error}); site={} adjudicates on the \
                 attack thread",
                site.token()
            );
            let job = take(&slot)?;
            Some(catch_unwind(AssertUnwindSafe(job)).map_err(|_| ()))
        }
    }
}

#[cfg(test)]
#[path = "attack_concurrent_tests.rs"]
mod tests;
