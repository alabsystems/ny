// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Runtime budget ledger for β-CROWN verification phases.
//!
//! All timeout consumers derive from one ledger instead of recomputing
//! fractions directly from raw `timeout`. Later phases consume the
//! **remaining** time, not the original timeout.
//!
//! Part of #2206. Design: designs/2026-03-16-issue-2206-adaptive-phase-budgeting.md

use ny_propagate::PhaseBudgetConfig;
use std::time::{Duration, Instant};

/// A representable engine-local horizon for a logically unbounded CLI run.
///
/// Several low-level BaB APIs still carry `Duration` rather than
/// `Option<Instant>`. Give them a decades-long operational horizon instead of
/// `u64::MAX`, which panics when those APIs add it to `Instant::now()`.
pub(in crate::commands::beta_crown) fn operational_unbounded_timeout() -> Duration {
    const FIFTY_YEARS_SECS: u64 = 50 * 365 * 24 * 60 * 60;
    let now = Instant::now();
    let mut seconds = FIFTY_YEARS_SECS;
    while seconds > 0 {
        let candidate = Duration::from_secs(seconds);
        if now.checked_add(candidate).is_some() {
            return candidate;
        }
        seconds /= 2;
    }
    Duration::ZERO
}

/// Budgets at or below this total get the small-budget attack cap.
const SMALL_BUDGET_TOTAL: Duration = Duration::from_secs(30);

/// Attack-phase ceiling (fraction of total) on small budgets.
const SMALL_BUDGET_ATTACK_FRACTION: f32 = 0.15;

/// Whether the short-budget MIP grant should retain a tail for the independent
/// outer VNN-COMP post-BaB falsifier.
///
/// `None` is the interactive/legacy route and preserves the historical
/// reserve. VNN-COMP supplies an explicit typed decision from the same preset
/// snapshot that controls the wrapper.
fn small_budget_postbab_tail_armed(wrapper_attack_enabled: Option<bool>) -> bool {
    wrapper_attack_enabled.unwrap_or(true)
}

const SMALL_BUDGET_POSTBAB_MIP_TAIL: Duration = Duration::from_secs(3);
const SMALL_BUDGET_POSTBAB_MIP_MAX_TOTAL: Duration = Duration::from_secs(25);

fn small_budget_postbab_mip_reserve(
    total: Duration,
    wrapper_attack_enabled: Option<bool>,
) -> Duration {
    if total <= SMALL_BUDGET_POSTBAB_MIP_MAX_TOTAL
        && small_budget_postbab_tail_armed(wrapper_attack_enabled)
    {
        SMALL_BUDGET_POSTBAB_MIP_TAIL
    } else {
        Duration::ZERO
    }
}

/// Whether the small-budget attack time cap is disabled (`NY_NO_PGD_TIME_CAP=1`).
fn pgd_time_cap_disabled() -> bool {
    crate::plan_resolver::env_value_is_exact_one(
        std::env::var_os(crate::plan_resolver::PGD_TIME_CAP_DISABLE_ENV).as_deref(),
    )
}

/// Share the existing `NY_PHASE_TELEMETRY=1` diagnostic gate with the
/// propagate-side phase/frontier markers, resolved through the ny-levers
/// chokepoint at call time (lever-debt batch B1 preparation). Emission happens
/// a handful of times per verify run, so a per-call env read costs nothing;
/// the chokepoint's exact-`"1"` `Bool` parse preserves the historical arming
/// rule. This remains live process state until Phase 2 injects a per-run
/// `LeverSet`.
fn phase_telemetry_enabled() -> bool {
    ny_levers::read(&ny_levers::decls::telemetry::PHASE_TELEMETRY)
        .value
        .as_bool()
}

/// Runtime budget ledger for β-CROWN verification.
///
/// Tracks wall-clock elapsed time and derives per-phase deadlines from
/// a single [`PhaseBudgetConfig`]. Replaces the scattered local fraction
/// arithmetic in `attack_budget.rs`, `sequential.rs`, `graph.rs`,
/// `disjunctive.rs`, and `mod.rs`.
#[derive(Clone)]
pub(in crate::commands::beta_crown) struct PhaseBudgetLedger {
    start: Instant,
    total: Option<Duration>,
    policy: PhaseBudgetConfig,
    /// FIX 2 (default-on Graph-MIP + feature `mip`): when the Graph-MIP whole-net
    /// escalation is armed and the phase policy requests a nonzero reservation,
    /// the BaB deadline RESERVES the MIP slice inside the
    /// scored budget (so the escalation is not reaped by the vnncomp watchdog)
    /// and `mip_timeout` never overdraws the actual remaining wall (measured:
    /// the un-capped floor granted 27 s with 0 s remaining at a 120 s budget
    /// → watchdog kill mid-escalation). `NY_GRAPH_MIP=0` or a non-MIP build
    /// keeps every formula byte-identical. A zero-reservation category policy
    /// also leaves the full BaB slice available.
    mip_reservation_armed: bool,
    /// Independently resolved outer VNN-COMP attack route. `None` retains the
    /// historical interactive small-budget reserve.
    post_bab_wrapper_attack_enabled: Option<bool>,
    /// MIP-handoff enforcement (#mip-handoff). TWO independent arming sources
    /// reach this one budget treatment:
    ///
    /// 1. the default-dark SafeNLP shared-prefix solver experiment
    ///    (`NY_MIP_SAFENLP_SHARED_PREFIX=1`), which needs a reachable MIP
    ///    handoff: keep the policy-sized slice even when the already-loaded
    ///    graph is statically unsupported, because dispatch can still reload an
    ///    encodable sequential network before the AY solve;
    /// 2. the typed preset opt-in `phase_budget.enforce_mip_handoff`, which
    ///    touches NOTHING but this schedule.
    ///
    /// This flag never arms MIP on its own; explicit `bab`, a non-MIP build,
    /// `NY_GRAPH_MIP=0`, and zero-reservation policies remain disarmed.
    safenlp_shared_prefix_budget_repair: bool,
}

impl PhaseBudgetLedger {
    /// Create a new ledger.
    ///
    /// - `timeout_secs`: 0 means unbounded (no deadline).
    /// - `policy`: phase budget fractions from `BetaCrownConfig.phase_budget`.
    #[cfg(test)]
    pub(in crate::commands::beta_crown) fn new(
        timeout_secs: u64,
        policy: PhaseBudgetConfig,
    ) -> Self {
        Self::new_duration(
            (timeout_secs > 0).then(|| Duration::from_secs(timeout_secs)),
            policy,
        )
    }

    /// Create a ledger from an exact duration.
    ///
    /// `None` means unbounded; `Some(Duration::ZERO)` means the bounded budget
    /// is already exhausted. This is used when an earlier verification phase
    /// consumed part or all of the CLI timeout.
    #[cfg(test)]
    pub(in crate::commands::beta_crown) fn new_duration(
        total: Option<Duration>,
        policy: PhaseBudgetConfig,
    ) -> Self {
        Self::from_start_and_total(Instant::now(), total, policy)
    }

    /// Create a ledger that preserves an existing absolute deadline.
    ///
    /// This prevents an earlier verification phase's remaining duration from
    /// being re-anchored to a later clock read. `None` remains unbounded.
    pub(in crate::commands::beta_crown) fn from_deadline(
        deadline: Option<Instant>,
        policy: PhaseBudgetConfig,
    ) -> Self {
        let now = Instant::now();
        match deadline {
            Some(deadline) if deadline <= now => {
                // Preserve an already-expired absolute deadline exactly. Using
                // `now + saturating_duration_since(now)` would silently move it
                // forward to this constructor's clock read.
                Self::from_start_and_total(deadline, Some(Duration::ZERO), policy)
            }
            Some(deadline) => {
                Self::from_start_and_total(now, Some(deadline.duration_since(now)), policy)
            }
            None => Self::from_start_and_total(now, None, policy),
        }
    }

    fn from_start_and_total(
        start: Instant,
        total: Option<Duration>,
        policy: PhaseBudgetConfig,
    ) -> Self {
        #[cfg(feature = "mip")]
        let mip_reservation_armed =
            super::super::graph_mip::graph_mip_enabled() && policy.requests_mip_reservation();
        #[cfg(not(feature = "mip"))]
        let mip_reservation_armed = false;
        // #mip-handoff: the typed preset opt-in is part of the POLICY, so it
        // must hold on every ledger construction path, not only the one that
        // threads the environment experiment through
        // `with_safenlp_shared_prefix_budget_repair`.
        let safenlp_shared_prefix_budget_repair =
            mip_reservation_armed && policy.enforce_mip_handoff;
        Self {
            start,
            total,
            policy,
            mip_reservation_armed,
            post_bab_wrapper_attack_enabled: None,
            safenlp_shared_prefix_budget_repair,
        }
    }

    /// Thread the outer VNN-COMP attack route into the small-budget MIP ledger.
    ///
    /// This is independent of the engine's `post_bab_pgd_fraction`: changing
    /// either policy must not silently rewrite the other's allocation.
    pub(in crate::commands::beta_crown) fn with_post_bab_wrapper_attack(
        mut self,
        enabled: Option<bool>,
    ) -> Self {
        self.post_bab_wrapper_attack_enabled = enabled;
        self
    }

    /// Arm the MIP reservation only when the selected complete-verifier policy
    /// can actually escalate after BaB. Explicit `bab` runs must retain that
    /// otherwise-unused slice.
    pub(in crate::commands::beta_crown) fn with_mip_escalation_allowed(
        mut self,
        allowed: bool,
    ) -> Self {
        if !allowed {
            self.mip_reservation_armed = false;
            // #mip-handoff: an unarmed ledger must not retain the handoff
            // treatment either — explicit `bab` keeps its exact historical
            // schedule even when a preset opted the category in.
            self.safenlp_shared_prefix_budget_repair = false;
        }
        self
    }

    /// Derive a phase-local ledger without ever extending this ledger's
    /// authoritative wall-clock deadline.
    ///
    /// Relational verification uses this for adaptive clause/group slices.
    /// The child starts now, but its duration is capped by the parent's exact
    /// remaining duration and it inherits the parent's MIP-reservation policy.
    /// Thus an explicit `bab` run cannot accidentally re-arm MIP in a nested
    /// lane, and no nested lane can rebase the competition timeout.
    pub(in crate::commands::beta_crown) fn child_with_timeout_secs(
        &self,
        timeout_secs: u64,
    ) -> Self {
        let start = Instant::now();
        let parent_remaining = self
            .overall_deadline()
            .map(|deadline| deadline.saturating_duration_since(start));
        let requested = (timeout_secs > 0).then(|| Duration::from_secs(timeout_secs));
        let total = match (parent_remaining, requested) {
            (Some(parent_remaining), Some(requested)) => Some(parent_remaining.min(requested)),
            (Some(parent_remaining), None) => Some(parent_remaining),
            (None, Some(requested)) => Some(requested),
            (None, None) => None,
        };
        Self {
            start,
            total,
            policy: self.policy.clone(),
            mip_reservation_armed: self.mip_reservation_armed,
            post_bab_wrapper_attack_enabled: self.post_bab_wrapper_attack_enabled,
            safenlp_shared_prefix_budget_repair: self.safenlp_shared_prefix_budget_repair,
        }
    }

    /// Latch the existing SafeNLP shared-prefix experiment into the budget
    /// policy selected by production dispatch.
    ///
    /// The experiment does not grant authority and cannot arm an escalation:
    /// it only changes how an already-armed whole-net MIP reservation is
    /// protected and sized. Keeping this as an explicit dispatch input makes
    /// gate-off construction and every non-dispatch ledger byte-for-byte
    /// identical to the historical path.
    pub(in crate::commands::beta_crown) fn with_safenlp_shared_prefix_budget_repair(
        mut self,
        enabled: bool,
    ) -> Self {
        self.safenlp_shared_prefix_budget_repair =
            (enabled || self.policy.enforce_mip_handoff) && self.mip_reservation_armed;
        self
    }

    /// Test-only override for the arming flag (env-free, race-free tests).
    ///
    /// `safenlp_shared_prefix_budget_repair` is DERIVED from
    /// `mip_reservation_armed && policy.enforce_mip_handoff`. The constructor and
    /// `with_safenlp_shared_prefix_budget_repair` both re-derive it; this
    /// override did not, so flipping the arming flag left the derived field
    /// stale at whatever the constructor computed. Under `cargo test -p ny-cli
    /// --bin ny` the `mip` feature is off (ny-cli's defaults are
    /// pytorch/coreml/gguf), so the constructor takes the
    /// `#[cfg(not(feature = "mip"))]` arm, derives `false`, and the override
    /// then armed the ledger without ever setting the flag the preset path
    /// asserts.
    #[cfg(test)]
    pub(in crate::commands::beta_crown) fn with_mip_reservation(mut self, armed: bool) -> Self {
        self.mip_reservation_armed = armed;
        self.safenlp_shared_prefix_budget_repair = armed && self.policy.enforce_mip_handoff;
        self
    }

    /// #deadlane — DISARM the Graph-MIP reservation when a purely STATIC scan of
    /// the model proves the escalation can never fire.
    ///
    /// `ineligible == true` means the already-loaded graph cannot reach the
    /// Graph-MIP encoder (see `dispatch::graph_mip_layer_set_supported`), so
    /// the slice [`Self::bab_deadline`] carves out of the SCORED budget can only
    /// be dead time on the historical route. Measured on vit_2023: 23 s of
    /// BaB's 95 s internal tier reserved for an escalation whose own log line
    /// says "NOT eligible (unsupported layer …)" on every one of the 200 rows.
    ///
    /// The default-dark SafeNLP shared-prefix treatment is the narrow
    /// exception: its tiny graph can be reloaded as an encodable sequential
    /// network by production dispatch, so graph-static ineligibility does not
    /// prove that its later whole-net MIP path is dead.
    ///
    /// This does NOT weaken deadline discipline: it moves the BaB deadline later
    /// INSIDE the same scored budget (never past `overall_deadline`), so the
    /// out-of-process watchdog still bounds the run. `ineligible == false` leaves
    /// every formula byte-identical.
    pub(in crate::commands::beta_crown) fn with_static_mip_ineligibility(
        mut self,
        ineligible: bool,
    ) -> Self {
        if ineligible && !self.safenlp_shared_prefix_budget_repair {
            self.mip_reservation_armed = false;
        }
        self
    }

    /// The MIP slice the armed reservation carves out of the scored budget:
    /// `total * mip_min_fraction` clamped to `[mip_min_secs, mip_max_secs]` —
    /// the SAME formula as [`Self::mip_timeout`]'s floor, so the reservation
    /// and the grant agree.
    fn mip_reserved_slice(&self) -> Option<Duration> {
        let total = self.total?;
        let pb = &self.policy;
        if !pb.requests_mip_reservation() {
            return Some(Duration::ZERO);
        }
        let fraction = if pb.mip_min_fraction.is_finite() {
            pb.mip_min_fraction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let secs = total
            .mul_f32(fraction)
            .as_secs()
            .clamp(pb.mip_min_secs, pb.mip_max_secs);
        Some(Duration::from_secs(secs))
    }

    /// Overall deadline (`start + total`), or `None` if unbounded.
    pub(in crate::commands::beta_crown) fn overall_deadline(&self) -> Option<Instant> {
        self.total.map(|d| self.start + d)
    }

    /// Deadline for a phase identified by a fractional budget.
    ///
    /// Returns `start + total * fraction`, or `None` if unbounded.
    pub(in crate::commands::beta_crown) fn phase_deadline(&self, fraction: f32) -> Option<Instant> {
        self.total
            .map(|d| self.start + d.mul_f32(fraction.clamp(0.0, 1.0)))
    }

    /// Phase deadline computed from **now** (not the ledger start):
    /// `now + total * fraction`, capped at the overall deadline.
    ///
    /// Use for phases that must still get their slice even when earlier phases
    /// ran long. The CROWN precheck previously computed `start + total * 0.20`,
    /// which was already in the past after a 50% attack phase — the root CROWN
    /// pass never ran and root bounds silently fell back to IBP
    /// (lsnc_relu: roots [-30, 30] vs thresholds ~0.4). #four-walls
    pub(in crate::commands::beta_crown) fn phase_deadline_from_now(
        &self,
        fraction: f32,
    ) -> Option<Instant> {
        let total = self.total?;
        let slice = total.mul_f32(fraction.clamp(0.0, 1.0));
        let overall = self.start + total;
        let candidate = Instant::now().checked_add(slice).unwrap_or(overall);
        Some(candidate.min(overall))
    }

    /// Disjunctive CROWN/alpha-precheck deadline: the from-now fraction slice
    /// ([`Self::phase_deadline_from_now`]), additionally clamped to the
    /// optional ABSOLUTE ceiling `policy.disjunctive_precheck_max_secs`
    /// (#precheck-abs-cap).
    ///
    /// A FRACTION bounds nothing here: the slice is `now + total*frac`, so it
    /// is granted in full no matter how much of the budget the attack phase
    /// already spent, and the phase it sizes (the CROWN-IBP root collection)
    /// grows with the model. `disjunctive_precheck_max_secs` bounds the phase
    /// in absolute terms — the precheck keeps a slice big enough for the work
    /// it needs and every second it does not spend is reclaimed by BaB, which
    /// re-bases on [`Self::remaining`]. `None` keeps the pure-fraction slice,
    /// so every category that does not set the knob is byte-identical.
    ///
    /// The cap is measured from NOW (the phase start), matching the from-now
    /// slice it clamps — unlike `disjunctive_pgd_max_secs`, whose phase is
    /// anchored at the ledger start.
    pub(in crate::commands::beta_crown) fn disjunctive_precheck_deadline(
        &self,
        fraction: f32,
    ) -> Option<Instant> {
        let abs_cap = self.policy.disjunctive_precheck_max_secs.map(|max_secs| {
            let capped = Instant::now() + Duration::from_secs(max_secs);
            match self.overall_deadline() {
                Some(overall) => capped.min(overall),
                None => capped,
            }
        });
        match (self.phase_deadline_from_now(fraction), abs_cap) {
            (Some(frac_deadline), Some(cap)) => Some(frac_deadline.min(cap)),
            (Some(frac_deadline), None) => Some(frac_deadline),
            // Unbounded total: the absolute ceiling still bounds the phase.
            (None, cap) => cap,
        }
    }

    /// Attack-phase deadline with a hard cap on tiny budgets.
    ///
    /// Fraction-based attack slices (upfront 0.20, disjunctive 0.50) are sized
    /// for 100s+ budgets. On tiny budgets (total <= 30s) they starve
    /// CROWN + BaB: lsnc_relu (20s ny budget) burned 10.0s in the disjunctive
    /// attack before any bound ran, while the preset intent was ~3s. Cap the
    /// attack slice at 15% of total there — by TIME, not by restart count,
    /// since per-eval cost varies by orders of magnitude across models.
    /// Presets that set an even smaller fraction keep their smaller slice.
    /// Disable the cap with `NY_NO_PGD_TIME_CAP=1`. #four-walls
    pub(in crate::commands::beta_crown) fn attack_phase_deadline(
        &self,
        fraction: f32,
    ) -> Option<Instant> {
        self.attack_phase_deadline_with_cap_policy(fraction, pgd_time_cap_disabled())
    }

    /// Environment-free core used by the disjunctive ledger and differential
    /// policy tests. `pgd_time_cap_disabled` is captured by the public boundary.
    fn attack_phase_deadline_with_cap_policy(
        &self,
        fraction: f32,
        pgd_time_cap_disabled: bool,
    ) -> Option<Instant> {
        let total = self.total?;
        let capped_fraction = if total <= SMALL_BUDGET_TOTAL && !pgd_time_cap_disabled {
            fraction.min(SMALL_BUDGET_ATTACK_FRACTION)
        } else {
            fraction
        };
        self.phase_deadline(capped_fraction)
    }

    /// Disjunctive global-PGD deadline: the fraction slice
    /// ([`Self::attack_phase_deadline`] with `disjunctive_pgd_fraction`, so the
    /// tiny-budget cap still applies), additionally clamped to the optional
    /// ABSOLUTE ceiling `policy.disjunctive_pgd_max_secs`.
    ///
    /// The fraction is anchored at the ledger start (`start + total*frac`), so on
    /// a hard UNSAT instance a large budget hands the falsifier a huge slice it
    /// wastes (it never finds a counterexample). `disjunctive_pgd_max_secs`
    /// bounds that in absolute terms; every second not spent here is reclaimed by
    /// the fast-path BaB, which re-bases on [`Self::remaining`]. `None` leaves the
    /// pure-fraction behavior unchanged.
    ///
    /// #attack-floor: the optional `policy.disjunctive_pgd_min_secs` is applied
    /// LAST, as an absolute FLOOR that outranks the tiny-budget cap. It exists
    /// for categories where BaB provably cannot decide, so the 15 % cap is pure
    /// loss (measured on lsnc_relu — see the field docs). The floor is itself
    /// clamped to HALF the scored total, so BaB always keeps at least half the
    /// budget and the phase can never pass the overall deadline. `None`
    /// (default) is byte-identical to before the knob existed.
    ///
    /// #attack-anchor: `policy.disjunctive_pgd_from_phase_start` moves the
    /// anchor from the LEDGER START to the PHASE START (`now`), for both the
    /// fraction slice and `disjunctive_pgd_max_secs`. The ledger-start anchor
    /// charges everything that ran before the falsifier — model load, graph
    /// build, VNN-LIB parse — against the falsifier's own slice, and on a big
    /// model that consumes all of it: MEASURED on cifar100_2024
    /// `CIFAR100_resnet_large` at the official 100 s budget with the shipped
    /// `0.05` fraction, the batched exact-VJP lane got **0.1 s of its 5 s
    /// slice and took ZERO steps**. The overall deadline still clamps the
    /// phase, and every second the falsifier does not spend is reclaimed by
    /// BaB (which re-bases on [`Self::remaining`]).
    pub(in crate::commands::beta_crown) fn disjunctive_pgd_deadline(&self) -> Option<Instant> {
        self.disjunctive_pgd_deadline_with_cap_policy(pgd_time_cap_disabled())
    }

    /// Environment-free deadline core. The public method captures the exact
    /// diagnostic input once; tests inject it directly without mutating global
    /// process state.
    fn disjunctive_pgd_deadline_with_cap_policy(
        &self,
        pgd_time_cap_disabled: bool,
    ) -> Option<Instant> {
        let total = self.total?;
        let from_phase_start = self.policy.disjunctive_pgd_from_phase_start;
        let frac_deadline = if from_phase_start {
            self.attack_phase_deadline_from_now(
                self.policy.disjunctive_pgd_fraction,
                pgd_time_cap_disabled,
            )?
        } else {
            self.attack_phase_deadline_with_cap_policy(
                self.policy.disjunctive_pgd_fraction,
                pgd_time_cap_disabled,
            )?
        };
        let overall = self.overall_deadline();
        let capped = match self.policy.disjunctive_pgd_max_secs {
            Some(max_secs) => {
                let anchor = if from_phase_start {
                    Instant::now()
                } else {
                    self.start
                };
                // `max_secs` is an authored u64. A diagnostic or malformed
                // preset may legitimately contain `u64::MAX`; adding that
                // duration to an Instant panics on supported platforms. A cap
                // larger than the whole ledger is semantically inert anyway,
                // so clamp before checked addition and fall back to the
                // already-authoritative overall/fraction deadline if the host
                // Instant range is narrower still.
                let cap_duration = Duration::from_secs(max_secs).min(total);
                let abs_cap = anchor
                    .checked_add(cap_duration)
                    .or(overall)
                    .unwrap_or(frac_deadline);
                // The overall deadline still binds: a from-phase-start cap can
                // never push the falsifier past the scored budget.
                let abs_cap = overall.map_or(abs_cap, |o| abs_cap.min(o));
                frac_deadline.min(abs_cap)
            }
            None => frac_deadline,
        };
        let floored = self
            .disjunctive_pgd_floor()
            .map_or(capped, |f| capped.max(f));
        // The floor is the last word on the slice, but never on the budget: a
        // phase-start-anchored floor is clamped back to the scored deadline.
        Some(overall.map_or(floored, |o| floored.min(o)))
    }

    /// [`Self::attack_phase_deadline`] measured from NOW (#attack-anchor).
    /// Applies the same tiny-budget attack cap, then defers to
    /// [`Self::phase_deadline_from_now`], which already clamps to the overall
    /// deadline.
    fn attack_phase_deadline_from_now(
        &self,
        fraction: f32,
        pgd_time_cap_disabled: bool,
    ) -> Option<Instant> {
        let total = self.total?;
        let capped_fraction = if total <= SMALL_BUDGET_TOTAL && !pgd_time_cap_disabled {
            fraction.min(SMALL_BUDGET_ATTACK_FRACTION)
        } else {
            fraction
        };
        self.phase_deadline_from_now(capped_fraction)
    }

    /// #attack-floor: `anchor + min(disjunctive_pgd_min_secs, total/2)`, or
    /// `None` when the knob is unset/zero or the ledger is unbounded.
    ///
    /// The anchor is the ledger start, or — under #attack-anchor
    /// (`disjunctive_pgd_from_phase_start`) — the PHASE start, so that "give the
    /// falsifier N seconds" means N seconds of falsification rather than N
    /// seconds minus whatever model load and spec parsing already cost. The
    /// half-budget clamp is unchanged, and the caller clamps the result to the
    /// overall deadline, so BaB can never be starved past the scored budget.
    fn disjunctive_pgd_floor(&self) -> Option<Instant> {
        let total = self.total?;
        let min_secs = self.policy.disjunctive_pgd_min_secs?;
        if min_secs == 0 {
            return None;
        }
        let anchor = if self.policy.disjunctive_pgd_from_phase_start {
            Instant::now()
        } else {
            self.start
        };
        let requested = Duration::from_secs(min_secs);
        Some(anchor + requested.min(total.mul_f32(0.5)))
    }

    /// Wall-clock time remaining, or `None` if unbounded.
    pub(in crate::commands::beta_crown) fn remaining(&self) -> Option<Duration> {
        self.total.map(|d| d.saturating_sub(self.start.elapsed()))
    }

    /// Remaining duration for legacy engine APIs that cannot express an
    /// optional deadline.
    pub(in crate::commands::beta_crown) fn remaining_for_engine(&self) -> Duration {
        self.remaining()
            .unwrap_or_else(operational_unbounded_timeout)
    }

    /// Remaining wall-clock seconds, clamped to `[0, u64::MAX]`.
    ///
    /// Returns `u64::MAX` if the timeout is unbounded.
    pub(in crate::commands::beta_crown) fn remaining_secs_clamped(&self) -> u64 {
        match self.remaining() {
            Some(d) => d.as_secs(),
            None => u64::MAX,
        }
    }

    /// Upfront PGD deadline: `start + total * upfront_pgd_fraction`,
    /// time-capped on tiny budgets (see [`Self::attack_phase_deadline`]).
    pub(in crate::commands::beta_crown) fn upfront_pgd_deadline(&self) -> Option<Instant> {
        self.attack_phase_deadline(self.policy.upfront_pgd_fraction)
    }

    /// BaB deadline: `start + total * (1 - post_bab_pgd_fraction)`.
    ///
    /// Used by CLI paths to thread the wall-clock deadline into BaB engine
    /// entry points, so the engine derives its phase budgets from remaining
    /// time rather than the original configured timeout (#4321).
    pub(in crate::commands::beta_crown) fn bab_deadline(&self) -> Option<Instant> {
        let total = self.total?;
        let frac = self.policy.post_bab_pgd_fraction.clamp(0.0, 0.5);
        let mut bab = total.mul_f32(1.0 - frac);
        // FIX 2: with the Graph-MIP escalation armed, reserve its slice INSIDE
        // the scored budget so the escalation runs before the watchdog, not
        // after it. BaB keeps at least half the total (a pathological policy
        // cannot starve it). The runtime ledger uses the same default-on gate
        // as dispatch; the exact-zero kill switch and category policies that
        // request a zero MIP slice leave this unarmed.
        if self.mip_reservation_armed {
            if let Some(reserved) = self.mip_reserved_slice() {
                bab = bab.saturating_sub(reserved);
                if !self.safenlp_shared_prefix_budget_repair {
                    bab = bab.max(total.mul_f32(0.5));
                }
            }
        }
        Some(self.start + bab)
    }

    /// BaB deadline for the default-dark SafeNLP shared-prefix experiment.
    ///
    /// Same-LHS reduced verification historically receives no absolute
    /// deadline and intentionally owns the full remaining wall clock (needed
    /// by ACAS-Xu). Only the exact shared-prefix treatment may interrupt that
    /// historical lane early enough to reach its reserved MIP slice.
    pub(in crate::commands::beta_crown) fn safenlp_shared_prefix_bab_deadline(
        &self,
    ) -> Option<Instant> {
        self.safenlp_shared_prefix_budget_repair
            .then(|| self.bab_deadline())
            .flatten()
    }

    /// Access the underlying policy for callers that need specific fractions.
    pub(in crate::commands::beta_crown) fn policy(&self) -> &PhaseBudgetConfig {
        &self.policy
    }

    /// Emit the complete phase-ledger allocation under the shared dark
    /// `NY_PHASE_TELEMETRY=1` gate.
    ///
    /// This is deliberately print-only: deadlines and policy values are read
    /// after they have already been computed, and no returned value can affect
    /// scheduling or proof authority. Planned deadline fields are offsets from
    /// this ledger's own start; `elapsed_s`/`remaining_s` expose nested-ledger
    /// rebasing (notably disjunctive PGD -> graph multi-objective BaB).
    pub(in crate::commands::beta_crown) fn emit_telemetry(&self, label: &str) {
        if !phase_telemetry_enabled() {
            return;
        }
        eprintln!("{}", self.telemetry_line(label, Instant::now()));
    }

    /// Pure formatter used by [`Self::emit_telemetry`] and deterministic tests.
    fn telemetry_line(&self, label: &str, now: Instant) -> String {
        fn duration_field(value: Option<Duration>) -> String {
            value.map_or_else(
                || "none".to_string(),
                |duration| format!("{:.3}", duration.as_secs_f64()),
            )
        }

        let elapsed = now.saturating_duration_since(self.start);
        let remaining = self.total.map(|total| total.saturating_sub(elapsed));
        let offset = |deadline: Option<Instant>| {
            deadline.map(|instant| instant.saturating_duration_since(self.start))
        };
        let pgd_cap = self
            .policy
            .disjunctive_pgd_max_secs
            .map(Duration::from_secs);
        let mip_reserved = if self.mip_reservation_armed {
            self.mip_reserved_slice()
        } else {
            None
        };

        format!(
            "[budget] {label} total_s={} elapsed_s={} remaining_s={} \
             disj_pgd_end_s={} bab_end_s={} overall_end_s={} \
             initial_bounds_fraction={:.3} post_bab_pgd_fraction={:.3} \
             disjunctive_pgd_fraction={:.3} disjunctive_pgd_cap_s={} \
             mip_reservation_armed={} mip_reserved_s={} \
             safenlp_shared_prefix_budget_repair={}",
            duration_field(self.total),
            duration_field(Some(elapsed)),
            duration_field(remaining),
            duration_field(offset(self.disjunctive_pgd_deadline())),
            duration_field(offset(self.bab_deadline())),
            duration_field(offset(self.overall_deadline())),
            self.policy.initial_bounds_fraction,
            self.policy.post_bab_pgd_fraction,
            self.policy.disjunctive_pgd_fraction,
            duration_field(pgd_cap),
            self.mip_reservation_armed,
            duration_field(mip_reserved),
            self.safenlp_shared_prefix_budget_repair,
        )
    }

    /// Per-clause timeout for constraint iteration.
    ///
    /// - **Conjunctive** (SAFE if ANY violated): returns full remaining time
    ///   since we early-exit on first success.
    /// - **Disjunctive** (SAFE if ALL violated): splits remaining time evenly
    ///   across `remaining_clauses`, with a floor of `min_clause_secs`.
    ///
    /// Returns `None` if the timeout is unbounded.
    pub(in crate::commands::beta_crown) fn per_clause_timeout(
        &self,
        remaining_clauses: usize,
        disjunctive: bool,
    ) -> Option<Duration> {
        let remaining = self.remaining()?;
        if !disjunctive || remaining_clauses == 0 {
            return Some(remaining);
        }
        let n = remaining_clauses.max(1) as u32;
        let per_clause = remaining / n;
        let min = Duration::from_secs(self.policy.min_clause_secs);
        Some(per_clause.max(min))
    }

    /// MIP fallback timeout: actual remaining wall-clock seconds.
    ///
    /// `mip_min_fraction` / `mip_min_secs` size the reservation made by
    /// [`Self::bab_deadline`]; they must never inflate the solver grant after
    /// that deadline has already been consumed.  In particular, an unreserved
    /// auto-MIP fallback used to turn `0s` remaining into a 25--30s timeout and
    /// was then killed by the outer VNN-COMP watchdog before it could report
    /// the already-sound BaB timeout.
    ///
    /// Returns `None` if the timeout is unbounded.
    // Only the mip-feature escalation path consumes this at runtime; the unit
    // test below still exercises it, so the allow only applies to non-mip
    // non-test builds.
    #[cfg_attr(all(not(feature = "mip"), not(test)), allow(dead_code))]
    pub(in crate::commands::beta_crown) fn mip_timeout(&self) -> Option<u64> {
        let total = self.total?;
        let remaining_secs = self.remaining()?.as_secs();
        // #postbab-small-budget: on small internal budgets (scored 20-30s
        // instances, internal <= 25s) the auto-MIP fallback taking the FULL
        // remaining wall ran the verify to its internal deadline, so the
        // scored-budget leftover shrank below the post-BaB falsification
        // lane's minimum and the lane never started — losing razor-thin SAT
        // rows (safenlp class) that the same binary wins whenever the lane
        // runs. Reserve 3s of the final MIP grant there so the verify returns
        // early enough for the attack window. Measured (2026-07-20,
        // hyperrectangle_1997): the reserved seconds came from an ABANDONED
        // final phase-split slice (15/16 subproblems, no verdict), so the
        // reservation costs nothing on that class; a genuinely-late MIP proof
        // loses at most 3s of its final slice. Larger budgets are unchanged.
        let small_budget_attack_reserve = if self.safenlp_shared_prefix_budget_repair {
            0
        } else {
            small_budget_postbab_mip_reserve(total, self.post_bab_wrapper_attack_enabled).as_secs()
        };
        // Hard invariant: a phase-local timeout may not exceed the enclosing
        // wall-clock budget.  Reservation (armed or not) only determines how
        // much time BaB leaves; it does not authorize borrowing beyond the
        // overall deadline.
        Some(remaining_secs.saturating_sub(small_budget_attack_reserve))
    }

    /// Absolute deadline for a MIP escalation started now.
    ///
    /// This is the live [`Self::mip_timeout`] grant capped at the enclosing
    /// overall deadline. On small budgets it therefore preserves the final
    /// falsification reserve instead of handing the solver the entire
    /// remaining wall clock.
    #[cfg_attr(all(not(feature = "mip"), not(test)), allow(dead_code))]
    pub(in crate::commands::beta_crown) fn mip_deadline(&self) -> Option<Instant> {
        let overall = self.overall_deadline()?;
        let grant = Duration::from_secs(self.mip_timeout()?);
        let candidate = Instant::now().checked_add(grant).unwrap_or(overall);
        Some(candidate.min(overall))
    }

    /// Deterministic production-schedule probe used by dispatch regressions:
    /// how many whole seconds would MIP receive if BaB returned exactly at its
    /// planned deadline? This mirrors [`Self::mip_timeout`] without sleeping or
    /// rebasing either absolute deadline.
    #[cfg(test)]
    pub(in crate::commands::beta_crown) fn planned_mip_timeout_at_bab_deadline(
        &self,
    ) -> Option<u64> {
        let total = self.total?;
        let bab_deadline = self.bab_deadline()?;
        let remaining_secs = self
            .overall_deadline()?
            .saturating_duration_since(bab_deadline)
            .as_secs();
        let small_budget_attack_reserve = if self.safenlp_shared_prefix_budget_repair {
            0
        } else {
            small_budget_postbab_mip_reserve(total, self.post_bab_wrapper_attack_enabled).as_secs()
        };
        Some(remaining_secs.saturating_sub(small_budget_attack_reserve))
    }

    // The only consumer is `dispatch`'s `#[cfg(all(test, feature = "mip"))]`
    // module, so a non-mip test build legitimately sees this accessor as dead.
    // Keep it compiled in both builds instead of narrowing the `cfg` — a
    // non-mip regression must be able to read the arming flag without first
    // re-widening the gate.
    #[cfg(test)]
    #[cfg_attr(not(feature = "mip"), allow(dead_code))]
    pub(in crate::commands::beta_crown) fn mip_reservation_armed_for_test(&self) -> bool {
        self.mip_reservation_armed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_budget_postbab_tail_uses_independent_wrapper_policy() {
        assert!(small_budget_postbab_tail_armed(None));
        assert!(small_budget_postbab_tail_armed(Some(true)));
        assert!(!small_budget_postbab_tail_armed(Some(false)));
        assert_eq!(
            small_budget_postbab_mip_reserve(Duration::from_secs(25), Some(true)),
            Duration::from_secs(3),
            "the inclusive small-budget boundary retains the attack tail"
        );
        assert_eq!(
            small_budget_postbab_mip_reserve(Duration::from_secs(26), Some(true)),
            Duration::ZERO,
            "larger totals leave MIP untouched"
        );
        assert_eq!(
            small_budget_postbab_mip_reserve(Duration::from_secs(25), Some(false)),
            Duration::ZERO,
            "an explicit wrapper opt-out has no outer attack consumer"
        );
    }

    #[test]
    fn ledger_unbounded_timeout_returns_none_deadlines() {
        let ledger = PhaseBudgetLedger::new(0, PhaseBudgetConfig::default());
        assert!(ledger.overall_deadline().is_none());
        assert!(ledger.upfront_pgd_deadline().is_none());
        assert!(ledger.remaining().is_none());
        assert_eq!(ledger.remaining_secs_clamped(), u64::MAX);
        let engine_timeout = ledger.remaining_for_engine();
        assert!(engine_timeout >= Duration::from_hours(24));
        assert!(Instant::now().checked_add(engine_timeout).is_some());
    }

    #[test]
    fn exact_zero_duration_is_an_exhausted_bounded_budget() {
        let ledger =
            PhaseBudgetLedger::new_duration(Some(Duration::ZERO), PhaseBudgetConfig::default());
        let deadline = ledger.overall_deadline().expect("bounded");
        assert!(deadline <= Instant::now());
        assert_eq!(ledger.remaining(), Some(Duration::ZERO));
    }

    #[test]
    fn existing_deadline_is_not_reanchored() {
        let deadline = Instant::now() + Duration::from_secs(10);
        let ledger = PhaseBudgetLedger::from_deadline(Some(deadline), PhaseBudgetConfig::default());
        assert_eq!(ledger.overall_deadline(), Some(deadline));
    }

    #[test]
    fn expired_deadline_is_not_reanchored() {
        let deadline = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("one second before now is representable");
        let ledger = PhaseBudgetLedger::from_deadline(Some(deadline), PhaseBudgetConfig::default());
        assert_eq!(ledger.overall_deadline(), Some(deadline));
        assert_eq!(ledger.remaining(), Some(Duration::ZERO));
    }

    #[test]
    fn mip_deadline_never_exceeds_overall_deadline() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        assert!(ledger.mip_deadline().expect("bounded") <= ledger.overall_deadline().unwrap());
    }

    #[test]
    fn small_budget_mip_deadline_preserves_attack_tail() {
        // Assert the POLICY, not the clock.
        //
        // This used to compare `overall_deadline()` against `mip_deadline()`.
        // The ledger anchors `overall` at construction, but `mip_deadline()`
        // reads `Instant::now()` a SECOND time and adds the grant to it, so the
        // measured gap is `total - grant - (elapsed between the two reads)`.
        // With the tail sized at exactly three seconds, any scheduling delay at
        // all pushes it under and the test fails — which is why it only ever
        // failed inside a loaded parallel workspace run and passed 5/5 in
        // isolation.
        //
        // `mip_timeout()` is the quantity the reserve actually governs, and the
        // property that matters is that the grant leaves the tail inside the
        // budget. That is clock-independent.
        let total_secs = 20;
        let ledger = PhaseBudgetLedger::new(total_secs, PhaseBudgetConfig::default());
        let grant = ledger.mip_timeout().expect("bounded");
        assert!(
            total_secs - grant >= 3,
            "small-budget MIP grant must leave the configured three-second tail: \
             total={total_secs}s grant={grant}s"
        );
        // And the absolute deadline still respects the enclosing budget.
        assert!(
            ledger.mip_deadline().expect("bounded") <= ledger.overall_deadline().expect("bounded"),
            "the MIP deadline may never exceed the overall deadline"
        );
    }

    #[test]
    fn ledger_bounded_timeout_produces_deadlines() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        assert!(ledger.overall_deadline().is_some());
        assert!(ledger.upfront_pgd_deadline().is_some());
        assert!(ledger.remaining().is_some());
        assert!(ledger.remaining_secs_clamped() <= 100);
    }

    #[test]
    fn budget_telemetry_exposes_planned_offsets_and_nested_elapsed_time() {
        let policy = PhaseBudgetConfig {
            initial_bounds_fraction: 0.15,
            disjunctive_pgd_fraction: 0.40,
            disjunctive_pgd_max_secs: Some(30),
            post_bab_pgd_fraction: 0.10,
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(100, policy).with_mip_reservation(false);
        let line = ledger.telemetry_line(
            "disjunctive-graph-handoff",
            ledger.start + Duration::from_secs(30),
        );
        assert!(line.starts_with("[budget] disjunctive-graph-handoff "));
        assert!(line.contains("total_s=100.000"));
        assert!(line.contains("elapsed_s=30.000"));
        assert!(line.contains("remaining_s=70.000"));
        assert!(line.contains("disj_pgd_end_s=30.000"));
        assert!(line.contains("bab_end_s=90.000"));
        assert!(line.contains("overall_end_s=100.000"));
        assert!(line.contains("initial_bounds_fraction=0.150"));
        assert!(line.contains("post_bab_pgd_fraction=0.100"));
        assert!(line.contains("disjunctive_pgd_fraction=0.400"));
        assert!(line.contains("disjunctive_pgd_cap_s=30.000"));
        assert!(line.contains("mip_reservation_armed=false"));
        assert!(line.contains("mip_reserved_s=none"));
        assert!(line.contains("safenlp_shared_prefix_budget_repair=false"));
    }

    #[test]
    fn upfront_pgd_deadline_matches_current_attack_budget_behavior() {
        // Current behavior: attack_budget.rs line 12-17 → timeout / 5 = 0.20
        let ledger = PhaseBudgetLedger::new(150, PhaseBudgetConfig::default());
        let deadline = ledger.upfront_pgd_deadline().expect("bounded");
        let budget = deadline.duration_since(ledger.start);
        // 150 * 0.20 = 30 seconds
        assert_eq!(budget, Duration::from_secs(30));
    }

    #[test]
    fn phase_deadline_fraction_zero_returns_start() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let deadline = ledger.phase_deadline(0.0).expect("bounded");
        // fraction=0 → deadline = start
        let budget = deadline.duration_since(ledger.start);
        assert_eq!(budget, Duration::ZERO);
    }

    #[test]
    fn phase_deadline_fraction_one_returns_overall() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let phase = ledger.phase_deadline(1.0).expect("bounded");
        let overall = ledger.overall_deadline().expect("bounded");
        assert_eq!(phase, overall);
    }

    #[test]
    fn per_clause_timeout_conjunctive_returns_remaining() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let per_clause = ledger.per_clause_timeout(5, false).expect("bounded");
        // Conjunctive: full remaining time, not divided by 5.
        assert!(per_clause.as_secs() > 10);
    }

    #[test]
    fn per_clause_timeout_disjunctive_splits_evenly() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let per_clause = ledger.per_clause_timeout(10, true).expect("bounded");
        // Disjunctive: ~100s / 10 = ~10s, floor is min_clause_secs (1).
        assert!(per_clause.as_secs() >= 1);
        assert!(per_clause.as_secs() <= 15);
    }

    #[test]
    fn per_clause_timeout_unbounded_returns_none() {
        let ledger = PhaseBudgetLedger::new(0, PhaseBudgetConfig::default());
        assert!(ledger.per_clause_timeout(5, true).is_none());
    }

    #[test]
    fn mip_timeout_returns_actual_remaining_budget() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let mip = ledger.mip_timeout().expect("bounded");
        assert!(mip <= 100);
        assert!(mip >= 99);
    }

    #[test]
    fn mip_timeout_small_budget_reserves_attack_window() {
        // #postbab-small-budget: internal totals <= 25s reserve 3s of the MIP
        // grant so the post-BaB falsification lane keeps a scored-budget
        // window (safenlp-class 20s instances → internal 15s → grant <= 12s).
        let ledger = PhaseBudgetLedger::new(15, PhaseBudgetConfig::default());
        let mip = ledger.mip_timeout().expect("bounded");
        assert!(mip <= 12, "small-budget grant must reserve 3s (got {mip}s)");
        // Larger budgets are unchanged (no reserve).
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let mip = ledger.mip_timeout().expect("bounded");
        assert!(
            mip >= 99,
            "large-budget grant must be untouched (got {mip}s)"
        );

        // Engine and wrapper policies are independent. Exercise the complete
        // 2x2 matrix: only the outer wrapper decision controls this reserve.
        for (engine_fraction, wrapper_enabled, expect_reserve) in [
            (0.0, true, true),
            (0.0, false, false),
            (0.1, true, true),
            (0.1, false, false),
        ] {
            let ledger = PhaseBudgetLedger::new(
                15,
                PhaseBudgetConfig {
                    post_bab_pgd_fraction: engine_fraction,
                    ..Default::default()
                },
            )
            .with_post_bab_wrapper_attack(Some(wrapper_enabled));
            let mip = ledger.mip_timeout().expect("bounded");
            if expect_reserve {
                assert!(
                    mip <= 12,
                    "wrapper-on fraction={engine_fraction} must reserve 3s (got {mip}s)"
                );
            } else {
                assert!(
                    mip >= 14,
                    "wrapper-off fraction={engine_fraction} must reclaim the tail (got {mip}s)"
                );
            }
        }
    }

    #[test]
    fn mip_timeout_unbounded_returns_none() {
        let ledger = PhaseBudgetLedger::new(0, PhaseBudgetConfig::default());
        assert!(ledger.mip_timeout().is_none());
    }

    #[test]
    fn bab_deadline_with_zero_reservation_matches_overall() {
        // With post_bab_pgd_fraction = 0.0 → no PGD reservation, so the BaB
        // deadline equals the overall deadline. (The *default* fraction is 0.10
        // — see PhaseBudgetConfig::default and its config tests — so this case
        // must set the fraction explicitly rather than rely on the default.)
        let policy = PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.0,
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(100, policy).with_mip_reservation(false);
        let bab = ledger.bab_deadline().expect("bounded");
        let overall = ledger.overall_deadline().expect("bounded");
        assert_eq!(bab, overall);
    }

    #[test]
    fn bab_deadline_with_reservation_is_earlier_than_overall() {
        // Reserve 10% for PGD → bab_deadline = start + 100 * 0.90 = 90s.
        let policy = PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.10,
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(100, policy).with_mip_reservation(false);
        let bab = ledger.bab_deadline().expect("bounded");
        let overall = ledger.overall_deadline().expect("bounded");
        assert!(bab < overall);
        // Verify the gap is ~10s (the PGD reservation).
        let gap = overall.duration_since(bab);
        assert!(gap.as_secs() >= 9 && gap.as_secs() <= 11);
    }

    #[test]
    fn bab_deadline_unbounded_returns_none() {
        let policy = PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.10,
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(0, policy);
        assert!(ledger.bab_deadline().is_none());
    }

    #[test]
    fn attack_phase_deadline_caps_small_budgets_at_fifteen_percent() {
        // lsnc_relu class: 20s ny budget, disjunctive fraction 0.50 would give
        // 10s — the cap holds the attack slice to 0.15 * 20 = 3s.
        let ledger = PhaseBudgetLedger::new(20, PhaseBudgetConfig::default());
        let deadline = ledger.attack_phase_deadline(0.50).expect("bounded");
        let slice = deadline.duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(20).mul_f32(0.15));
    }

    /// #attack-anchor — the defect, reproduced without a clock read: with the
    /// LEDGER-START anchor, work that ran before the falsifier is charged
    /// against the falsifier's own slice, and enough of it leaves the phase
    /// with nothing. cifar100 shape: 100s budget, `disjunctive_pgd_fraction`
    /// 0.05 (a 5s slice), 4.9s already spent on model load / graph build /
    /// VNN-LIB parse.
    #[test]
    fn ledger_start_anchor_charges_setup_time_to_the_falsifier() {
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_fraction: 0.05,
            disjunctive_pgd_max_secs: Some(30),
            ..Default::default()
        };
        // The ledger started 4.9s ago; only 0.1s of the 5s slice is left.
        let start = Instant::now()
            .checked_sub(Duration::from_millis(4_900))
            .expect("4.9s before now is representable");
        let ledger =
            PhaseBudgetLedger::from_start_and_total(start, Some(Duration::from_secs(100)), policy);
        let left = ledger
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .saturating_duration_since(Instant::now());
        assert!(
            left < Duration::from_millis(400),
            "ledger-start anchor leaves the falsifier ~0.1s, got {left:?}"
        );
    }

    /// #attack-anchor — the fix: the same 4.9s of setup, and the phase gets the
    /// 5% the preset actually asked for, measured from the phase start.
    #[test]
    fn phase_start_anchor_delivers_the_fraction_the_preset_asked_for() {
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_fraction: 0.05,
            disjunctive_pgd_max_secs: Some(30),
            disjunctive_pgd_from_phase_start: true,
            ..Default::default()
        };
        let start = Instant::now()
            .checked_sub(Duration::from_millis(4_900))
            .expect("4.9s before now is representable");
        let ledger =
            PhaseBudgetLedger::from_start_and_total(start, Some(Duration::from_secs(100)), policy);
        let left = ledger
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .saturating_duration_since(Instant::now());
        assert!(
            left > Duration::from_millis(4_500) && left <= Duration::from_secs(5),
            "phase-start anchor must grant ~5s, got {left:?}"
        );
    }

    /// #attack-anchor — the phase-start anchor can never push the falsifier
    /// past the scored budget, however large the fraction or the cap.
    #[test]
    fn phase_start_anchor_never_passes_the_overall_deadline() {
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_fraction: 0.90,
            disjunctive_pgd_max_secs: Some(300),
            disjunctive_pgd_from_phase_start: true,
            ..Default::default()
        };
        let start = Instant::now()
            .checked_sub(Duration::from_secs(90))
            .expect("90s before now is representable");
        let ledger =
            PhaseBudgetLedger::from_start_and_total(start, Some(Duration::from_secs(100)), policy);
        let deadline = ledger.disjunctive_pgd_deadline().expect("bounded");
        assert!(
            deadline <= ledger.overall_deadline().expect("bounded"),
            "phase-start anchor must stay inside the scored budget"
        );
    }

    /// #attack-anchor — default `false` is byte-identical to before the knob.
    #[test]
    fn phase_start_anchor_is_inert_by_default() {
        for max_secs in [None, Some(30)] {
            let base = PhaseBudgetConfig {
                disjunctive_pgd_fraction: 0.05,
                disjunctive_pgd_max_secs: max_secs,
                ..Default::default()
            };
            let start = Instant::now()
                .checked_sub(Duration::from_secs(2))
                .expect("two seconds before now is representable");
            let baseline = PhaseBudgetLedger::from_start_and_total(
                start,
                Some(Duration::from_secs(100)),
                base.clone(),
            );
            let explicit = PhaseBudgetLedger::from_start_and_total(
                start,
                Some(Duration::from_secs(100)),
                PhaseBudgetConfig {
                    disjunctive_pgd_from_phase_start: false,
                    ..base
                },
            );
            assert_eq!(
                baseline
                    .disjunctive_pgd_deadline()
                    .expect("bounded")
                    .duration_since(start),
                explicit
                    .disjunctive_pgd_deadline()
                    .expect("bounded")
                    .duration_since(start),
                "max_secs={max_secs:?}"
            );
        }
    }

    /// #attack-floor — the opt-in absolute floor outranks the tiny-budget cap.
    /// lsnc_relu at its measured 20s internal budget: the cap alone gives the
    /// disjunctive attack 3.0s and the ORT-confirmed `quadrotor2d_state_34`
    /// counterexample lands ~3.6s in (measured), so the row times out; the 5s
    /// floor restores it.
    #[test]
    fn disjunctive_pgd_floor_outranks_the_small_budget_cap() {
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_min_secs: Some(5),
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(20, policy);
        let slice = ledger
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(5), "floor must beat the 3s cap");
    }

    /// The floor NEVER takes more than half the scored budget, so BaB keeps at
    /// least half however large the requested floor is.
    #[test]
    fn disjunctive_pgd_floor_is_clamped_to_half_the_budget() {
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_min_secs: Some(999),
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(20, policy);
        let slice = ledger
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(10));
    }

    /// Unset (and explicit `0`) leave the phase byte-identical to the pure
    /// cap/ceiling behavior — no other category can be perturbed by this knob.
    #[test]
    fn disjunctive_pgd_floor_unset_or_zero_is_byte_identical() {
        let baseline = PhaseBudgetLedger::new(20, PhaseBudgetConfig::default());
        let baseline_slice = baseline
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .duration_since(baseline.start);
        for min_secs in [None, Some(0)] {
            let policy = PhaseBudgetConfig {
                disjunctive_pgd_min_secs: min_secs,
                ..Default::default()
            };
            let ledger = PhaseBudgetLedger::new(20, policy);
            let slice = ledger
                .disjunctive_pgd_deadline()
                .expect("bounded")
                .duration_since(ledger.start);
            assert_eq!(slice, baseline_slice, "min_secs={min_secs:?} must be inert");
        }
    }

    /// A floor BELOW the already-granted slice cannot shrink the phase (it is a
    /// floor, not a cap): a large budget keeps its full fraction.
    #[test]
    fn disjunctive_pgd_floor_never_shrinks_a_larger_slice() {
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_min_secs: Some(5),
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(200, policy);
        let slice = ledger
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(200).mul_f32(0.50));
    }

    #[test]
    fn attack_phase_deadline_keeps_smaller_preset_fractions_on_small_budgets() {
        // A preset that already asks for less than the cap keeps its slice.
        let ledger = PhaseBudgetLedger::new(20, PhaseBudgetConfig::default());
        let deadline = ledger.attack_phase_deadline(0.10).expect("bounded");
        let slice = deadline.duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(20).mul_f32(0.10));
    }

    #[test]
    fn attack_phase_deadline_uncapped_on_large_budgets() {
        // 180s budget: fraction-based slice unchanged (traffic_signs class).
        let ledger = PhaseBudgetLedger::new(180, PhaseBudgetConfig::default());
        let deadline = ledger.attack_phase_deadline(0.50).expect("bounded");
        let slice = deadline.duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(90));
    }

    #[test]
    fn attack_phase_deadline_unbounded_returns_none() {
        let ledger = PhaseBudgetLedger::new(0, PhaseBudgetConfig::default());
        assert!(ledger.attack_phase_deadline(0.50).is_none());
    }

    #[test]
    fn disjunctive_pgd_deadline_none_cap_matches_fraction() {
        // With no absolute cap, the disjunctive deadline equals the plain
        // fraction slice (0.20 * 200s = 40s).
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_fraction: 0.20,
            disjunctive_pgd_max_secs: None,
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(200, policy);
        let slice = ledger
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(40));
    }

    #[test]
    fn disjunctive_pgd_deadline_absolute_cap_bounds_large_budget() {
        // 0.20 * 300s = 60s by fraction, but the 30s absolute cap clamps it —
        // the reclaim/robustness fix for hold-heavy conv benchmarks.
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_fraction: 0.20,
            disjunctive_pgd_max_secs: Some(30),
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(300, policy);
        let slice = ledger
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(30));
    }

    #[test]
    fn disjunctive_pgd_deadline_u64_max_cap_is_inert_and_cannot_overflow() {
        // The authored cap is an unconstrained u64. It is a ceiling, so a cap
        // larger than the whole ledger must behave like no cap rather than
        // panicking while constructing `start + Duration::MAX`.
        for from_phase_start in [false, true] {
            let policy = PhaseBudgetConfig {
                disjunctive_pgd_fraction: 0.20,
                disjunctive_pgd_max_secs: Some(u64::MAX),
                disjunctive_pgd_from_phase_start: from_phase_start,
                ..Default::default()
            };
            let ledger = PhaseBudgetLedger::new(100, policy);
            let deadline = ledger.disjunctive_pgd_deadline().expect("bounded");
            if !from_phase_start {
                assert_eq!(
                    deadline.duration_since(ledger.start),
                    Duration::from_secs(20)
                );
            }
            assert!(deadline <= ledger.overall_deadline().expect("bounded"));
        }
    }

    #[test]
    fn collins_runtime_slice_matches_the_nominal_plan_at_tiers_and_boundary() {
        for from_phase_start in [false, true] {
            for min_secs in [None, Some(20)] {
                let policy = PhaseBudgetConfig {
                    disjunctive_pgd_fraction: 0.50,
                    disjunctive_pgd_max_secs: Some(15),
                    disjunctive_pgd_min_secs: min_secs,
                    disjunctive_pgd_from_phase_start: from_phase_start,
                    ..Default::default()
                };
                for pgd_time_cap_disabled in [false, true] {
                    for total_secs in [25_u64, 30, 31, 285, 570, 1140] {
                        let ledger = PhaseBudgetLedger::new(total_secs, policy.clone());
                        let before = Instant::now();
                        let deadline = ledger
                            .disjunctive_pgd_deadline_with_cap_policy(pgd_time_cap_disabled)
                            .expect("bounded");
                        let after = Instant::now();
                        let nominal_slice =
                            crate::plan_resolver::planned_disjunctive_pgd_slice_secs(
                                total_secs,
                                0.50,
                                Some(15),
                                min_secs,
                                pgd_time_cap_disabled,
                            );
                        if from_phase_start {
                            // The runtime samples its anchor between these two
                            // clocks. Bracket the exact nominal duration without
                            // assuming a zero-cost call.
                            let lower = deadline.saturating_duration_since(after).as_secs_f64();
                            let upper = deadline.saturating_duration_since(before).as_secs_f64();
                            assert!(
                                lower <= nominal_slice && nominal_slice <= upper,
                                "total={total_secs}, min={min_secs:?}, cap_disabled={pgd_time_cap_disabled}: \
                                 nominal={nominal_slice}, bracket=[{lower},{upper}]"
                            );
                        } else {
                            let runtime_slice = deadline.duration_since(ledger.start).as_secs_f64();
                            assert_eq!(
                                runtime_slice, nominal_slice,
                                "total={total_secs}, min={min_secs:?}, \
                                 cap_disabled={pgd_time_cap_disabled}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn disjunctive_pgd_deadline_cap_inactive_when_fraction_smaller() {
        // At a small budget the fraction slice is already under the cap, so the
        // cap is a no-op (0.20 * 100s = 20s < 30s cap).
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_fraction: 0.20,
            disjunctive_pgd_max_secs: Some(30),
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(100, policy);
        let slice = ledger
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(20));
    }

    #[test]
    fn disjunctive_pgd_deadline_unbounded_returns_none() {
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_max_secs: Some(30),
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(0, policy);
        assert!(ledger.disjunctive_pgd_deadline().is_none());
    }

    /// TEETH for #reclaim-pgd at relusplitter's REAL official budgets.
    ///
    /// relusplitter runs four budgets — 30/60/90/180 s — against the sealed default
    /// `disjunctive_pgd_fraction: 0.50`. This pins exactly what
    /// `configs/vnncomp25/relusplitter.yaml`'s `disjunctive_pgd_max_secs: 5` does at
    /// each one, INCLUDING the budget where it is deliberately inert. Changing the
    /// preset value without re-measuring moves these numbers.
    #[test]
    fn relusplitter_pgd_cap_reclaims_at_official_budgets() {
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_max_secs: Some(5),
            ..Default::default()
        };
        assert_eq!(
            policy.disjunctive_pgd_fraction, 0.50,
            "relusplitter does not override the fraction — this arithmetic assumes \
             the sealed default"
        );

        // 30 s (mnist_fc rows): the pre-existing small-budget attack cap already
        // holds the slice to 0.15 * 30 = 4.5 s, which is BELOW the 5 s ceiling, so
        // the ceiling is INERT here. This is the `min` direction that matters: an
        // absolute CAP must never LENGTHEN a phase, or it would steal budget from
        // the bound phases that actually produce verdicts.
        let ledger = PhaseBudgetLedger::new(30, policy.clone());
        let slice = ledger
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .duration_since(ledger.start);
        assert_eq!(
            slice,
            Duration::from_secs(30).mul_f32(0.15),
            "the 5s ceiling must not lengthen the already-smaller 30s attack slice",
        );

        // 60 / 90 / 180 s: the fraction grants 30 / 45 / 90 s and the ceiling holds
        // each to 5 s. The reclaimed remainder flows to BaB, which re-bases on
        // `remaining()`.
        for (total_secs, uncapped_secs) in [(60_u64, 30_u64), (90, 45), (180, 90)] {
            let baseline = PhaseBudgetLedger::new(total_secs, PhaseBudgetConfig::default());
            let baseline_slice = baseline
                .disjunctive_pgd_deadline()
                .expect("bounded")
                .duration_since(baseline.start);
            assert_eq!(
                baseline_slice,
                Duration::from_secs(uncapped_secs),
                "uncapped {total_secs}s budget grants the plain 0.50 fraction",
            );

            let capped_ledger = PhaseBudgetLedger::new(total_secs, policy.clone());
            let capped_slice = capped_ledger
                .disjunctive_pgd_deadline()
                .expect("bounded")
                .duration_since(capped_ledger.start);
            assert_eq!(
                capped_slice,
                Duration::from_secs(5),
                "the absolute ceiling must bind at a {total_secs}s budget",
            );
            assert!(
                capped_slice < baseline_slice,
                "the ceiling must actually reclaim time at {total_secs}s: \
                 capped={capped_slice:?} baseline={baseline_slice:?}",
            );
            assert!(
                capped_slice < Duration::from_secs(total_secs),
                "an attack slice must stay strictly inside the overall budget",
            );
        }
    }

    /// The ceiling bounds an ATTACK phase and NOTHING else. Every bound-producing
    /// phase — initial bounds, reduced verification, MIP, post-BaB reservation —
    /// must be byte-identical with and without it.
    ///
    /// This is the machine-checkable half of the soundness argument for
    /// #reclaim-pgd: a knob that cannot shorten a bound phase cannot change which
    /// verdicts are provable, so it cannot turn an `unknown` into an `unsat`.
    #[test]
    fn disjunctive_pgd_cap_does_not_move_any_bound_phase() {
        let baseline = PhaseBudgetLedger::new(180, PhaseBudgetConfig::default());
        let capped = PhaseBudgetLedger::new(
            180,
            PhaseBudgetConfig {
                disjunctive_pgd_max_secs: Some(5),
                ..Default::default()
            },
        );

        for fraction in [0.05_f32, 0.20, 0.40, 0.50, 1.0] {
            assert_eq!(
                baseline
                    .phase_deadline(fraction)
                    .map(|d| d.duration_since(baseline.start)),
                capped
                    .phase_deadline(fraction)
                    .map(|d| d.duration_since(capped.start)),
                "phase_deadline({fraction}) must be unaffected by the attack ceiling",
            );
        }
    }

    #[test]
    fn upfront_pgd_deadline_capped_on_small_budgets() {
        // 25s budget with default upfront fraction 0.20 → 5s uncapped;
        // small-budget cap holds it to 0.15 * 25 = 3.75s.
        let ledger = PhaseBudgetLedger::new(25, PhaseBudgetConfig::default());
        let deadline = ledger.upfront_pgd_deadline().expect("bounded");
        let slice = deadline.duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(25).mul_f32(0.15));
    }

    #[test]
    fn phase_deadline_from_now_grants_fresh_slice() {
        // Even immediately after ledger creation, the from-now deadline is at
        // least the fraction slice into the future (up to the overall cap).
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let before = Instant::now();
        let deadline = ledger.phase_deadline_from_now(0.20).expect("bounded");
        // Slice ≈ 20s from now; allow generous slack below for scheduling.
        let slice = deadline.duration_since(before);
        assert!(slice >= Duration::from_secs(19));
        assert!(slice <= Duration::from_secs(21));
    }

    #[test]
    fn phase_deadline_from_now_capped_at_overall() {
        // fraction 1.0 from now would exceed the overall deadline; the result
        // must be clamped to it.
        let ledger = PhaseBudgetLedger::new(10, PhaseBudgetConfig::default());
        let deadline = ledger.phase_deadline_from_now(1.0).expect("bounded");
        let overall = ledger.overall_deadline().expect("bounded");
        assert!(deadline <= overall);
    }

    #[test]
    fn phase_deadline_from_now_unbounded_returns_none() {
        let ledger = PhaseBudgetLedger::new(0, PhaseBudgetConfig::default());
        assert!(ledger.phase_deadline_from_now(0.20).is_none());
    }

    /// #precheck-abs-cap: with no absolute ceiling the precheck deadline is
    /// exactly the from-now fraction slice — every category that does not set
    /// the knob is byte-identical.
    #[test]
    fn disjunctive_precheck_deadline_without_cap_is_the_fraction_slice() {
        let ledger = PhaseBudgetLedger::new(900, PhaseBudgetConfig::default());
        assert!(ledger.policy().disjunctive_precheck_max_secs.is_none());
        let before = Instant::now();
        let deadline = ledger.disjunctive_precheck_deadline(0.85).expect("bounded");
        let slice = deadline.duration_since(before);
        // 0.85 * 900 = 765s from now (clamped only by the overall deadline).
        assert!(slice >= Duration::from_secs(764), "slice={slice:?}");
        assert!(slice <= Duration::from_secs(766), "slice={slice:?}");
    }

    /// #precheck-abs-cap: the absolute ceiling wins over a ballooned fraction.
    #[test]
    fn disjunctive_precheck_deadline_respects_the_absolute_ceiling() {
        let policy = PhaseBudgetConfig {
            disjunctive_precheck_max_secs: Some(250),
            ..PhaseBudgetConfig::default()
        };
        let ledger = PhaseBudgetLedger::new(900, policy);
        let before = Instant::now();
        let deadline = ledger.disjunctive_precheck_deadline(0.85).expect("bounded");
        let slice = deadline.duration_since(before);
        assert!(slice <= Duration::from_secs(251), "slice={slice:?}");
        assert!(slice >= Duration::from_secs(249), "slice={slice:?}");
    }

    /// #precheck-abs-cap: a ceiling LARGER than the fraction slice never
    /// extends the phase, and the overall deadline still dominates.
    #[test]
    fn disjunctive_precheck_deadline_ceiling_never_extends_the_slice() {
        let policy = PhaseBudgetConfig {
            disjunctive_precheck_max_secs: Some(10_000),
            ..PhaseBudgetConfig::default()
        };
        let ledger = PhaseBudgetLedger::new(100, policy);
        let deadline = ledger.disjunctive_precheck_deadline(0.20).expect("bounded");
        let overall = ledger.overall_deadline().expect("bounded");
        assert!(deadline <= overall);
        let slice = deadline.saturating_duration_since(Instant::now());
        assert!(slice <= Duration::from_secs(21), "slice={slice:?}");
    }

    /// #precheck-abs-cap: on an UNBOUNDED ledger the fraction yields nothing,
    /// so the absolute ceiling is the only bound the phase has.
    #[test]
    fn disjunctive_precheck_deadline_unbounded_uses_only_the_ceiling() {
        let unbounded = PhaseBudgetLedger::new(0, PhaseBudgetConfig::default());
        assert!(unbounded.disjunctive_precheck_deadline(0.85).is_none());
        let policy = PhaseBudgetConfig {
            disjunctive_precheck_max_secs: Some(30),
            ..PhaseBudgetConfig::default()
        };
        let capped = PhaseBudgetLedger::new(0, policy);
        let deadline = capped.disjunctive_precheck_deadline(0.85).expect("capped");
        let slice = deadline.saturating_duration_since(Instant::now());
        assert!(slice <= Duration::from_secs(31), "slice={slice:?}");
    }

    /// FIX 2 — armed reservation math: the BaB deadline carves the MIP slice
    /// (`total*mip_min_fraction` clamped `[mip_min_secs, mip_max_secs]`) out of
    /// the scored budget, floored at half the total.
    #[test]
    fn armed_bab_deadline_reserves_the_mip_slice() {
        let policy = PhaseBudgetConfig::default();
        let unarmed = PhaseBudgetLedger::new(120, policy.clone()).with_mip_reservation(false);
        let armed = PhaseBudgetLedger::new(120, policy.clone()).with_mip_reservation(true);
        let reserved = armed.mip_reserved_slice().expect("bounded slice");
        // Default: 120*0.25 = 30 clamped [mip_min_secs, mip_max_secs].
        let expected = Duration::from_secs((120f32 * policy.mip_min_fraction) as u64)
            .min(Duration::from_secs(policy.mip_max_secs))
            .max(Duration::from_secs(policy.mip_min_secs));
        assert_eq!(
            reserved, expected,
            "reservation = mip_timeout's floor formula"
        );

        let unarmed_dl = unarmed.bab_deadline().expect("bounded");
        let armed_dl = armed.bab_deadline().expect("bounded");
        // The armed BaB deadline is EARLIER by ~the reserved slice (start times
        // differ by nanoseconds between the two ledgers; allow 1 s slack).
        let delta = unarmed_dl.saturating_duration_since(armed_dl);
        assert!(
            delta + Duration::from_secs(1) >= reserved && delta <= reserved,
            "armed BaB deadline must reserve the MIP slice (delta={delta:?}, reserved={reserved:?})"
        );
    }

    /// #deadlane — a STATICALLY ineligible Graph-MIP escalation must not reserve
    /// any of the scored budget: BaB gets the slice back (measured 23 s of
    /// vit_2023's 95 s internal tier), while an ELIGIBLE net keeps the
    /// reservation byte-identically. The reclaimed time stays INSIDE the scored
    /// budget — `overall_deadline` is untouched, so the watchdog still bounds it.
    #[test]
    fn static_mip_ineligibility_returns_the_reserved_slice_to_bab() {
        let policy = PhaseBudgetConfig::default();
        let armed = PhaseBudgetLedger::new(120, policy.clone()).with_mip_reservation(true);
        let mut reclaimed = PhaseBudgetLedger::new(120, policy.clone())
            .with_mip_reservation(true)
            .with_static_mip_ineligibility(true);
        let mut still_armed = PhaseBudgetLedger::new(120, policy)
            .with_mip_reservation(true)
            .with_static_mip_ineligibility(false);
        // Remove constructor-time skew so the arithmetic compares exactly.
        reclaimed.start = armed.start;
        still_armed.start = armed.start;

        assert!(
            reclaimed.bab_deadline().expect("bounded") > armed.bab_deadline().expect("bounded"),
            "a statically-impossible escalation must not shorten BaB's deadline"
        );
        assert_eq!(
            still_armed.bab_deadline(),
            armed.bab_deadline(),
            "an ELIGIBLE net keeps the reservation byte-identically"
        );
        // The reclaim never pushes BaB past the scored deadline.
        assert!(
            reclaimed.bab_deadline().expect("bounded")
                <= reclaimed.overall_deadline().expect("bounded"),
            "the reclaimed BaB deadline must stay inside the scored budget"
        );
    }

    /// The SafeNLP treatment keeps the full policy-sized reservation through a
    /// graph-static decline because production can still reload a sequential
    /// network. On the official 15-second internal tier this also bypasses the
    /// historical half-budget clamp and three-second attack tail, leaving a
    /// dispatch-admissible AY-MIP grant.
    #[test]
    fn safenlp_shared_prefix_repair_makes_the_policy_slice_reachable() {
        let policy = PhaseBudgetConfig {
            mip_min_fraction: 0.65,
            mip_min_secs: 8,
            mip_max_secs: 30,
            post_bab_pgd_fraction: 0.10,
            ..PhaseBudgetConfig::default()
        };
        let historical = PhaseBudgetLedger::new(15, policy.clone())
            .with_mip_reservation(true)
            .with_static_mip_ineligibility(true);
        assert!(!historical.mip_reservation_armed);
        assert_eq!(
            historical.planned_mip_timeout_at_bab_deadline(),
            Some(0),
            "13.5s BaB end leaves one whole second, then the 3s tail starves MIP"
        );
        assert_eq!(
            historical.safenlp_shared_prefix_bab_deadline(),
            None,
            "gate-off same-LHS verification must retain its full historical wall clock"
        );

        let repaired = PhaseBudgetLedger::new(15, policy)
            .with_mip_reservation(true)
            .with_safenlp_shared_prefix_budget_repair(true)
            .with_static_mip_ineligibility(true);
        assert!(repaired.mip_reservation_armed);
        assert_eq!(repaired.mip_reserved_slice(), Some(Duration::from_secs(9)));
        assert_eq!(
            repaired.safenlp_shared_prefix_bab_deadline(),
            repaired.bab_deadline(),
            "gate-on same-LHS verification must stop at the reserved MIP handoff"
        );
        assert_eq!(
            repaired.planned_mip_timeout_at_bab_deadline(),
            Some(10),
            "the 9s policy slice plus the existing 1.5s post-BaB share must reach MIP"
        );
        assert!(
            repaired
                .planned_mip_timeout_at_bab_deadline()
                .expect("bounded")
                >= 5,
            "production dispatch admits only grants of at least five seconds"
        );
    }

    /// #mip-handoff — the TYPED PRESET opt-in must reach the same schedule the
    /// environment experiment reaches, without any environment read.
    ///
    /// This is the regression guard for the defect measured on safenlp_2024
    /// (2026-08-06): `mip_min_fraction: 0.65` carved a 9 s slice that the
    /// same-LHS BaB never yielded, so `mip_timeout` was 0 s and the escalation
    /// gate (`>= 5`) refused on every row of the category.
    #[test]
    fn preset_enforce_mip_handoff_arms_the_same_schedule_as_the_environment_gate() {
        let policy = PhaseBudgetConfig {
            mip_min_fraction: 0.65,
            mip_min_secs: 8,
            mip_max_secs: 30,
            post_bab_pgd_fraction: 0.10,
            enforce_mip_handoff: true,
            ..PhaseBudgetConfig::default()
        };
        // No `with_safenlp_shared_prefix_budget_repair(true)` anywhere: the
        // policy alone must arm it.
        let by_preset = PhaseBudgetLedger::new(15, policy.clone()).with_mip_reservation(true);
        assert!(by_preset.safenlp_shared_prefix_budget_repair);
        assert_eq!(
            by_preset.safenlp_shared_prefix_bab_deadline(),
            by_preset.bab_deadline(),
            "the preset opt-in must hand the same-LHS lane an absolute deadline"
        );
        assert_eq!(
            by_preset.planned_mip_timeout_at_bab_deadline(),
            Some(10),
            "the escalation must receive the slice the preset declares"
        );

        // Identical to the environment-gated arm.
        let by_env = PhaseBudgetLedger::new(
            15,
            PhaseBudgetConfig {
                enforce_mip_handoff: false,
                ..policy
            },
        )
        .with_mip_reservation(true)
        .with_safenlp_shared_prefix_budget_repair(true);
        assert_eq!(
            by_preset.planned_mip_timeout_at_bab_deadline(),
            by_env.planned_mip_timeout_at_bab_deadline()
        );

        // OFF is byte-identical to the historical schedule, and starves MIP.
        let historical = PhaseBudgetLedger::new(
            15,
            PhaseBudgetConfig {
                enforce_mip_handoff: false,
                ..policy
            },
        )
        .with_mip_reservation(true);
        assert!(!historical.safenlp_shared_prefix_budget_repair);
        assert_eq!(historical.safenlp_shared_prefix_bab_deadline(), None);
        // The historical schedule cannot reach the gate even in the best case
        // where BaB stops exactly at its planned deadline: the half-budget floor
        // pushes the handoff to 7.5s and the 3s attack tail then leaves 4s < 5s.
        // In production it is worse still — with no absolute deadline the
        // same-LHS lane runs to the internal deadline and the grant is 0s.
        assert_eq!(historical.planned_mip_timeout_at_bab_deadline(), Some(4));
    }

    /// The preset opt-in must never survive an explicit `--complete-verifier bab`.
    #[test]
    fn preset_enforce_mip_handoff_yields_to_explicit_bab() {
        let policy = PhaseBudgetConfig {
            mip_min_fraction: 0.65,
            mip_min_secs: 8,
            enforce_mip_handoff: true,
            ..PhaseBudgetConfig::default()
        };
        let historical = PhaseBudgetLedger::new(
            15,
            PhaseBudgetConfig {
                enforce_mip_handoff: false,
                ..policy
            },
        )
        .with_mip_reservation(true)
        .with_mip_escalation_allowed(false);
        let mut bab_only = PhaseBudgetLedger::new(15, policy)
            .with_mip_reservation(true)
            .with_mip_escalation_allowed(false);
        bab_only.start = historical.start;
        assert!(!bab_only.safenlp_shared_prefix_budget_repair);
        assert_eq!(bab_only.safenlp_shared_prefix_bab_deadline(), None);
        assert_eq!(bab_only.bab_deadline(), historical.bab_deadline());
    }

    #[test]
    fn safenlp_shared_prefix_repair_does_not_rearm_explicit_bab() {
        let base = PhaseBudgetLedger::new(
            15,
            PhaseBudgetConfig {
                mip_min_fraction: 0.65,
                mip_min_secs: 8,
                ..PhaseBudgetConfig::default()
            },
        )
        .with_mip_reservation(true)
        .with_mip_escalation_allowed(false);
        let historical_bab_only = base.clone();
        let repaired_bab_only = base
            .with_safenlp_shared_prefix_budget_repair(true)
            .with_static_mip_ineligibility(true);
        assert!(
            !repaired_bab_only.mip_reservation_armed,
            "the experiment must never override explicit complete-verifier=bab"
        );
        assert!(
            !repaired_bab_only.safenlp_shared_prefix_budget_repair,
            "an unarmed ledger must not latch the budget treatment"
        );
        assert_eq!(
            repaired_bab_only.safenlp_shared_prefix_bab_deadline(),
            None,
            "explicit BaB must not acquire a shared-prefix handoff deadline"
        );
        assert_eq!(
            repaired_bab_only.bab_deadline(),
            historical_bab_only.bab_deadline(),
            "explicit BaB must retain its exact historical deadline"
        );
        assert_eq!(
            repaired_bab_only.planned_mip_timeout_at_bab_deadline(),
            historical_bab_only.planned_mip_timeout_at_bab_deadline(),
            "explicit BaB must retain the historical (unused) grant formula"
        );
    }

    #[test]
    fn safenlp_shared_prefix_repair_is_inert_on_every_unarmed_ledger() {
        let zero_policy = PhaseBudgetConfig {
            mip_min_fraction: 0.0,
            mip_min_secs: 0,
            post_bab_pgd_fraction: 0.10,
            ..PhaseBudgetConfig::default()
        };
        let zero_base = PhaseBudgetLedger::new(15, zero_policy).with_mip_reservation(false);
        let zero_historical = zero_base.clone();
        let zero_treatment = zero_base.with_safenlp_shared_prefix_budget_repair(true);
        assert!(!zero_treatment.mip_reservation_armed);
        assert!(!zero_treatment.safenlp_shared_prefix_budget_repair);
        assert_eq!(
            zero_treatment.bab_deadline(),
            zero_historical.bab_deadline(),
            "zero-reservation policy must retain its historical BaB deadline"
        );
        assert_eq!(
            zero_treatment.planned_mip_timeout_at_bab_deadline(),
            zero_historical.planned_mip_timeout_at_bab_deadline(),
            "zero-reservation policy must keep the three-second attack tail"
        );

        let unarmed_base =
            PhaseBudgetLedger::new(15, PhaseBudgetConfig::default()).with_mip_reservation(false);
        let unarmed_historical = unarmed_base.clone();
        let unarmed_treatment = unarmed_base.with_safenlp_shared_prefix_budget_repair(true);
        assert!(!unarmed_treatment.safenlp_shared_prefix_budget_repair);
        assert_eq!(
            unarmed_treatment.bab_deadline(),
            unarmed_historical.bab_deadline()
        );
        assert_eq!(
            unarmed_treatment.planned_mip_timeout_at_bab_deadline(),
            unarmed_historical.planned_mip_timeout_at_bab_deadline()
        );
    }

    #[test]
    fn explicit_bab_policy_disarms_unusable_mip_reservation() {
        let policy = PhaseBudgetConfig::default();
        let armed = PhaseBudgetLedger::new(120, policy.clone()).with_mip_reservation(true);
        let mut bab_only = PhaseBudgetLedger::new(120, policy)
            .with_mip_reservation(true)
            .with_mip_escalation_allowed(false);
        bab_only.start = armed.start;

        assert!(
            bab_only.bab_deadline() > armed.bab_deadline(),
            "explicit BaB must receive the slice that cannot be used for MIP"
        );
        let expected = bab_only.start
            + Duration::from_mins(2)
                .mul_f32(1.0 - bab_only.policy.post_bab_pgd_fraction.clamp(0.0, 0.5));
        assert_eq!(bab_only.bab_deadline(), Some(expected));
    }

    #[test]
    fn relational_child_ledger_preserves_disarm_and_parent_wall() {
        let policy = PhaseBudgetConfig::default();
        let parent = PhaseBudgetLedger::new(120, policy)
            .with_mip_reservation(true)
            .with_mip_escalation_allowed(false);
        let child = parent.child_with_timeout_secs(300);

        assert!(
            !child.mip_reservation_armed,
            "nested relational lanes must not re-arm MIP for explicit BaB"
        );
        assert!(
            child.overall_deadline() <= parent.overall_deadline(),
            "a nested relational timeout must be capped by the authoritative parent wall"
        );
        let expected = child.start
            + child
                .total
                .expect("bounded child")
                .mul_f32(1.0 - child.policy.post_bab_pgd_fraction.clamp(0.0, 0.5));
        assert_eq!(child.bab_deadline(), Some(expected));
    }

    #[test]
    fn relational_dispatch_uses_the_same_authoritative_ledger() {
        let dispatch = include_str!("../dispatch.rs");
        let production = dispatch
            .split("#[cfg(all(test")
            .next()
            .expect("production dispatch source");
        let call_start = production
            .rfind("verify_relational_constraints_with_ledger(")
            .expect("relational ledger-aware dispatch call");
        let call = &production[call_start..];
        let call_end = call.find(")?").expect("end of relational dispatch call");
        assert!(
            call[..call_end].contains("&bab_ledger"),
            "relational verification must borrow the same disarmed/started ledger"
        );

        for source in [
            include_str!("graph.rs"),
            include_str!("sequential.rs"),
            include_str!("disjunctive.rs"),
        ] {
            assert!(
                !source.contains("PhaseBudgetLedger::new(timeout"),
                "a relational lane must not reset the authoritative timeout or MIP policy"
            );
        }
    }

    #[test]
    fn zero_mip_policy_disarms_and_preserves_the_exact_bab_deadline() {
        let policy = PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.10,
            mip_min_fraction: 0.0,
            mip_min_secs: 0,
            ..Default::default()
        };
        assert!(!policy.requests_mip_reservation());

        let unarmed = PhaseBudgetLedger::new(64, policy.clone()).with_mip_reservation(false);
        let mut forced_armed = PhaseBudgetLedger::new(64, policy).with_mip_reservation(true);
        // Compare the policy arithmetic exactly, without constructor-time skew.
        forced_armed.start = unarmed.start;

        assert_eq!(
            forced_armed.mip_reserved_slice(),
            Some(Duration::ZERO),
            "a zero policy must carve out no hidden minimum"
        );
        assert_eq!(forced_armed.bab_deadline(), unarmed.bab_deadline());
        assert_eq!(
            forced_armed.bab_deadline(),
            Some(unarmed.start + Duration::from_secs(64).mul_f32(0.90))
        );

        let runtime = PhaseBudgetLedger::new(
            64,
            PhaseBudgetConfig {
                mip_min_fraction: 0.0,
                mip_min_secs: 0,
                ..Default::default()
            },
        );
        assert!(
            !runtime.mip_reservation_armed,
            "the runtime constructor must share the zero-policy admission predicate"
        );
    }

    #[test]
    fn unordered_mip_reservation_clamp_declines_without_panicking() {
        let policy = PhaseBudgetConfig {
            mip_min_fraction: 0.25,
            mip_min_secs: 30,
            mip_max_secs: 5,
            ..Default::default()
        };
        assert!(!policy.requests_mip_reservation());

        let ledger = PhaseBudgetLedger::new(64, policy).with_mip_reservation(true);
        assert_eq!(ledger.mip_reserved_slice(), Some(Duration::ZERO));
    }

    #[cfg(feature = "mip")]
    #[test]
    fn default_on_graph_mip_arms_the_runtime_ledger_when_unset() {
        assert!(
            std::env::var_os("NY_GRAPH_MIP").is_none(),
            "run this regression with NY_GRAPH_MIP unset"
        );
        let ledger = PhaseBudgetLedger::new(120, PhaseBudgetConfig::default());
        assert!(
            ledger.mip_reservation_armed,
            "the runtime ledger must reserve a slice under the default-on gate"
        );
    }

    /// FIX 2 — the armed BaB deadline never drops below half the total budget
    /// (a pathological reservation cannot starve BaB).
    #[test]
    fn armed_bab_deadline_keeps_half_the_budget() {
        let policy = PhaseBudgetConfig {
            mip_min_fraction: 1.0, // pathological: reserve everything
            mip_min_secs: 0,
            mip_max_secs: u64::MAX,
            ..Default::default()
        };
        let armed = PhaseBudgetLedger::new(100, policy).with_mip_reservation(true);
        let dl = armed.bab_deadline().expect("bounded");
        let bab = dl.saturating_duration_since(armed.start);
        assert!(
            bab >= Duration::from_secs(50),
            "BaB must keep at least half the total (got {bab:?})"
        );
    }

    /// Every `mip_timeout` is capped at the actual remaining wall.  Reservation
    /// changes when BaB stops, not whether the fallback may overdraw the outer
    /// deadline.
    #[test]
    fn mip_timeout_capped_at_remaining_with_or_without_reservation() {
        let policy = PhaseBudgetConfig::default();
        let mut armed = PhaseBudgetLedger::new(120, policy.clone()).with_mip_reservation(true);
        // Simulate 118 s elapsed: remaining ≈ 2 s.
        armed.start = Instant::now()
            .checked_sub(Duration::from_secs(118))
            .unwrap();
        let granted = armed.mip_timeout().expect("bounded");
        assert!(
            granted <= 2,
            "armed grant must not exceed remaining (got {granted}s)"
        );

        // An explicitly unarmed ledger is subject to the same hard wall.
        let mut unarmed = PhaseBudgetLedger::new(120, policy).with_mip_reservation(false);
        unarmed.start = Instant::now()
            .checked_sub(Duration::from_secs(118))
            .unwrap();
        let granted = unarmed.mip_timeout().expect("bounded");
        assert!(
            granted <= 2,
            "unarmed grant must not exceed remaining (got {granted}s)"
        );
    }

    /// An explicitly unarmed BaB retains its legacy deadline, while the
    /// subsequent solver grant remains bounded by the actual wall clock.
    #[test]
    fn unarmed_bab_deadline_is_legacy_but_mip_grant_is_wall_clamped() {
        let policy = PhaseBudgetConfig::default();
        let ledger = PhaseBudgetLedger::new(100, policy.clone()).with_mip_reservation(false);
        let frac = policy.post_bab_pgd_fraction.clamp(0.0, 0.5);
        let expected = ledger.start + Duration::from_secs(100).mul_f32(1.0 - frac);
        assert_eq!(ledger.bab_deadline(), Some(expected), "legacy BaB deadline");
        let granted = ledger.mip_timeout().expect("bounded");
        assert!(granted <= 100, "MIP grant cannot exceed total wall");
        assert!(granted >= 99, "fresh ledger should retain nearly all wall");
    }
}
