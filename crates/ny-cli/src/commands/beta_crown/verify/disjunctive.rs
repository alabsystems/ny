// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-clause disjunction verification.
//!
//! Handles VNN-LIB specs with `(assert (or ...))` that produce multiple
//! independent clauses, each verified separately with timeout budgeting.

use anyhow::Result;
use ny_core::GemmEngine;
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BetaCrownVerifier, GraphNetwork,
    Layer, SharedGraphBootstrap,
};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use super::attack_budget::disjunctive_sampling_budget;
use super::disjunctive_pgd::{beta_crown_pgd_config, try_disjunctive_sampling_attack_with_config};
use super::disjunctive_precheck::{alpha_crown_precheck_clauses, crown_precheck_clauses};
use super::disjunctive_unified::{
    supports_grouped_disjunctive_contract, try_no_branchable_neuron_pgd_fallback,
    try_prechecked_unified_input_split_disjunctive_bab,
};
use super::phase_budget::PhaseBudgetLedger;
use super::BetaCrownModel;
use crate::commands::beta_crown::attack_arming::{AttackEngineSource, ResolvedAttackEngine};
use crate::commands::beta_crown::constraint_plan::build_grouped_disjunctive_objectives;

fn disable_clause_invprop(config: &mut BetaCrownConfig) {
    config.alpha_config.output_constraints = None;
    config.alpha_config.invprop.enabled = false;
    config.alpha_config.invprop.optimize_gammas = false;
}

/// Rebind INVPROP's assume-violation region to one extracted disjunct.
///
/// Top-level setup deliberately refuses to dualize a disjunction as one
/// conjunction.  Once the serial disjunctive lane has isolated a clause,
/// however, that clause is exactly the conjunctive region whose emptiness this
/// invocation must prove.  Keep the rebinding local so learned gamma state and
/// constraint matrices can never leak across clauses.
pub(super) fn config_for_clause_invprop(
    config: &BetaCrownConfig,
    clause_spec: &VnnLibSpec,
) -> BetaCrownConfig {
    let observe_rebind = config.alpha_config.invprop.enabled;
    if observe_rebind {
        ny_propagate::execution_telemetry::record_invprop_clause_rebind_attempt();
    }
    let mut clause_config = config.clone();
    // Clear first: a malformed/programmatic clause must fall back to ordinary
    // proof with INVPROP inert, never inherit another clause's conditional
    // matrix.  Refusing this optimization must not abort the whole instance.
    clause_config.alpha_config.output_constraints = None;

    let refusal = if clause_spec.is_disjunction {
        Some("clause is still marked as a disjunction")
    } else if !clause_spec.output_constraint_clauses.is_empty() {
        Some("residual clause grouping was not cleared")
    } else if clause_spec.output_constraints.is_empty() {
        Some("clause has no output constraints")
    } else {
        None
    };
    if let Some(reason) = refusal {
        warn!(reason, "Skipping clause-local INVPROP rebinding");
        if observe_rebind {
            ny_propagate::execution_telemetry::record_invprop_clause_rebind_refused();
        }
        disable_clause_invprop(&mut clause_config);
        return clause_config;
    }

    // Parsed specs already carry this guarantee, but programmatic callers can
    // construct VnnLibSpec directly.  Validate before matrix conversion so a
    // malformed output index fails closed instead of indexing ndarray storage.
    if let Err(error) = clause_spec.validate_output_indices() {
        warn!(%error, "Skipping clause-local INVPROP rebinding");
        if observe_rebind {
            ny_propagate::execution_telemetry::record_invprop_clause_rebind_refused();
        }
        disable_clause_invprop(&mut clause_config);
        return clause_config;
    }

    let output_constraints = match clause_spec.to_output_constraints() {
        Ok(constraints) if constraints.is_conjunction && constraints.clause_indices.is_none() => {
            constraints
        }
        Ok(_) => {
            // Structurally unreachable after the checks above; keep this guard
            // as defense in depth.  The propagation layer independently no-ops
            // on non-conjunctive matrices as its final soundness boundary.
            warn!("Skipping clause-local INVPROP rebinding: conversion was not conjunctive");
            if observe_rebind {
                ny_propagate::execution_telemetry::record_invprop_clause_rebind_refused();
            }
            disable_clause_invprop(&mut clause_config);
            return clause_config;
        }
        Err(error) => {
            warn!(%error, "Skipping clause-local INVPROP rebinding");
            if observe_rebind {
                ny_propagate::execution_telemetry::record_invprop_clause_rebind_refused();
            }
            disable_clause_invprop(&mut clause_config);
            return clause_config;
        }
    };

    // Overwrite unconditionally: a programmatic caller may carry a matrix for
    // another property even though the CLI top-level disjunction leaves None.
    clause_config.alpha_config.output_constraints = Some(output_constraints);
    if observe_rebind {
        ny_propagate::execution_telemetry::record_invprop_clause_rebind_accepted();
    }
    clause_config
}

pub(super) fn finalize_disjunctive_result(
    aggregated: BetaCrownResult,
    overall_start: Instant,
    final_status: BabVerificationStatus,
) -> BetaCrownResult {
    let mut aggregated = super::normalize_result_wall_time(aggregated, overall_start);
    aggregated.result = final_status;
    aggregated
}
fn graph_has_conv2d_layers(model_net: &BetaCrownModel) -> bool {
    let BetaCrownModel::Graph(graph) = model_net else {
        return false;
    };

    graph.node_names().iter().any(|name| {
        graph.node(name).is_some_and(|node| {
            matches!(node.layer(), Layer::Conv2d(_) | Layer::ConvTranspose2d(_))
        })
    })
}

fn supports_graph_multi_objective_clause_batch(
    model_net: &BetaCrownModel,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    use_relu_split: bool,
    gpu_bab: bool,
) -> bool {
    graph_has_conv2d_layers(model_net)
        && use_relu_split
        && !gpu_bab
        && clauses.iter().all(|clause| clause.len() == 1)
        && per_clause_input_bounds
            .iter()
            .all(|bounds| bounds.is_empty())
}

/// Run the attack while optionally warming the forward-linear cache in a
/// scoped worker.  Keeping the optionality outside `thread::scope` makes the
/// short-budget contract testable: a refused warmer is never spawned, so there
/// is no worker for scope teardown to join.
pub(super) fn run_with_optional_forward_linear_warmer<W, A, R>(warmer: Option<W>, attack: A) -> R
where
    W: FnOnce() + Send,
    A: FnOnce() -> R,
{
    std::thread::scope(|scope| {
        if let Some(warmer) = warmer {
            scope.spawn(warmer);
        }
        attack()
    })
}

/// #budget-abandonment FIX B: may the disjunctive loop make another pass over
/// the clauses it failed to decide?
///
/// FOUR conditions, ALL required, so the loop terminates by construction:
///
/// * `pass < retry_passes` — the typed key is the hard cap.
/// * `undecided_out > 0` — nothing left to retry otherwise.
/// * `undecided_out < undecided_in` — the pass just run SHRANK the undecided
///   set. A pass that decided nothing would only repeat itself, and this is
///   also what keeps the total number of passes bounded by the clause count.
/// * `budget_live` — the overall deadline is still live and the ledger still
///   has time. A retry that ran past the publication margin could lose a row
///   that was already decided, which is the one real hazard in this fix.
pub(super) const fn another_clause_retry_pass(
    pass: usize,
    retry_passes: usize,
    undecided_in: usize,
    undecided_out: usize,
    budget_live: bool,
) -> bool {
    pass < retry_passes && undecided_out > 0 && undecided_out < undecided_in && budget_live
}

/// Per-clause verdicts carried ACROSS retry passes.
///
/// SOUNDNESS, and the reason this is a type rather than two `Vec`s: there is
/// NO method that clears a recorded potential violation. HEAD's `saw_potential`
/// was a monotone OR over a single pass; once retries exist, a per-clause flag
/// that a later pass can reset lets a clause that signalled
/// `PotentialViolation` on pass 0 come back `Verified` on pass 1 and leave the
/// aggregate `Verified` -- a row that is `unknown` today published as `unsat`
/// because a SCHEDULING key was armed. `begin_visit` therefore clears the
/// diagnostic reason only, and a clause that has signalled is `is_terminal`
/// and never revisited.
#[derive(Debug, Default)]
struct ClauseVerdicts {
    /// Monotone. Set by `record_potential`, cleared by nothing.
    potential: Vec<bool>,
    /// Diagnostic only; recomputed on every visit.
    reason: Vec<Option<String>>,
}

impl ClauseVerdicts {
    fn new(clauses: usize) -> Self {
        Self {
            potential: vec![false; clauses],
            reason: vec![None; clauses],
        }
    }

    /// Start a visit to `idx`: the DIAGNOSTIC reason is about to be
    /// recomputed. The potential-violation flag is deliberately untouched.
    fn begin_visit(&mut self, idx: usize) {
        self.reason[idx] = None;
    }

    fn record_potential(&mut self, idx: usize) {
        self.potential[idx] = true;
    }

    fn record_unknown(&mut self, idx: usize, reason: String) {
        self.reason[idx] = Some(reason);
    }

    /// A clause whose verdict no further pass can change.
    fn is_terminal(&self, idx: usize) -> bool {
        self.potential[idx]
    }

    fn potential_count(&self) -> usize {
        self.potential.iter().filter(|&&flag| flag).count()
    }

    /// Same three-way precedence as HEAD: any potential violation outranks any
    /// unknown, and only a clean sweep is `Verified`.
    fn final_status(&self) -> BabVerificationStatus {
        if self.potential.iter().any(|&flag| flag) {
            BabVerificationStatus::potential_violation()
        } else if let Some(reason) = self.reason.iter().flatten().next() {
            BabVerificationStatus::Unknown {
                reason: reason.clone(),
            }
        } else {
            BabVerificationStatus::Verified
        }
    }
}

/// Per-clause counters, kept OUT of the running aggregate so a clause visited
/// on more than one retry pass contributes exactly once.
#[derive(Debug, Clone, Copy)]
struct ClauseStats {
    domains_explored: usize,
    domains_verified: usize,
    cuts_generated: usize,
    time_elapsed: Duration,
}

/// Fold every clause's LAST recorded counters into the aggregate, once.
fn fold_clause_stats(aggregated: &mut BetaCrownResult, stats: &[Option<ClauseStats>]) {
    for entry in stats.iter().flatten() {
        aggregated.domains_explored += entry.domains_explored;
        aggregated.domains_verified += entry.domains_verified;
        aggregated.cuts_generated += entry.cuts_generated;
        aggregated.time_elapsed += entry.time_elapsed;
    }
}

/// Budget a clause pass may allocate from.
///
/// Pass 0 is HEAD verbatim -- `ledger.remaining()`, the FULL authority. Every
/// RETRY pass is additionally clamped to `deadline` (the caller passes
/// `bab_deadline()`), so a retry cannot allocate itself the wall the Graph-MIP
/// escalation and the post-BaB confirmation are holding in reserve.
fn pass_remaining(ledger: &PhaseBudgetLedger, pass: usize, deadline: Option<Instant>) -> Duration {
    let remaining = ledger.remaining().unwrap_or(Duration::ZERO);
    if pass == 0 {
        return remaining;
    }
    match deadline {
        Some(dl) => remaining.min(dl.saturating_duration_since(Instant::now())),
        None => remaining,
    }
}

// Justification: Multi-clause disjunction requires the full verification context
// (model, bounds, spec, config, verifier, flags, engine) to check each clause.
#[allow(clippy::too_many_arguments)]
/// #disjunctive-shared-bootstrap: the one bootstrap the serial clause loop
/// lends to its clauses, plus the counters its observable line reports.
struct SharedBootstrapCache {
    bootstrap: Option<SharedGraphBootstrap>,
    armed: bool,
    route: &'static str,
    /// Why `bootstrap` is `None` (`ok` when it is `Some`). Every value is a
    /// distinct precondition so the observable says WHICH gate declined.
    reason: &'static str,
    /// Clauses that consumed the shared bootstrap.
    reused: usize,
    /// Clauses that carried their own input box and re-bootstrapped on it.
    fallback_box: usize,
}

impl SharedBootstrapCache {
    fn unshared(armed: bool, route: &'static str, reason: &'static str) -> Self {
        Self {
            bootstrap: None,
            armed,
            route,
            reason,
            reused: 0,
            fallback_box: 0,
        }
    }

    /// The per-clause view: `None` when nothing is shared OR the clause has
    /// its own input box (`per_clause_input_bounds[idx]` non-empty -- the same
    /// predicate the loop uses to tighten `effective_input`), because a
    /// bootstrap over the PROPERTY box is not an enclosure for a clause box
    /// that is not that box. Such a clause takes the exact historical path.
    fn for_clause(&mut self, vnnlib: &VnnLibSpec, idx: usize) -> Option<&SharedGraphBootstrap> {
        let shared = self.bootstrap.as_ref()?;
        let clause_has_own_box = vnnlib
            .per_clause_input_bounds
            .get(idx)
            .is_some_and(|bounds| !bounds.is_empty());
        if clause_has_own_box {
            self.fallback_box += 1;
            None
        } else {
            self.reused += 1;
            Some(shared)
        }
    }

    /// Unconditional, once per disjunction, at default verbosity: an ABSENT
    /// line means the binary has no lever, so a null measured on it is VOID;
    /// `armed=0` is a yaml that never set the key; `computed=0 reason=<gate>`
    /// names the precondition that declined.
    fn emit(&self, clauses: usize) {
        let (computed, secs, alpha, fallback) = match &self.bootstrap {
            Some(shared) => (
                1,
                shared.bootstrap_secs,
                u8::from(shared.has_root_alpha()),
                u8::from(shared.warmup_fallback_fired),
            ),
            None => (0, 0.0, 0, 0),
        };
        let line = format!(
            "[shared-bootstrap] armed={} route={} computed={computed} reason={} \
bootstrap_s={secs:.1} alpha={alpha} fallback={fallback} reused={} fallback_box={} clauses={clauses}",
            u8::from(self.armed),
            self.route,
            self.reason,
            self.reused,
            self.fallback_box,
        );
        eprintln!("{line}");
        info!("{line}");
    }
}

/// Every OUTPUT index any pending clause reads, sorted and deduplicated -- the
/// margin-subset seed the shared bootstrap publishes (the multi-objective root
/// does the same union; a superset of rows tightened is sound).
///
/// `OutputConstraint` is `#[non_exhaustive]`, and an atom this function cannot
/// read is an atom whose output indices it does not know. Publishing a seed
/// that omits them would leave THAT clause's rows on the untightened IBP box
/// while its own per-clause bootstrap would have tightened them -- a lost
/// proof, silently. So an unknown variant returns the EMPTY publication, which
/// disengages subset seeding entirely and takes the full-width path: the most
/// expensive answer and the one that can never be tighter for one clause than
/// for another.
fn referenced_output_indices(clauses: &[Vec<OutputConstraint>], pending: &[usize]) -> Vec<usize> {
    let mut referenced: Vec<usize> = Vec::new();
    for &clause_idx in pending {
        for atom in &clauses[clause_idx] {
            match *atom {
                OutputConstraint::LessEq(i, j)
                | OutputConstraint::GreaterEq(i, j)
                | OutputConstraint::LessThan(i, j)
                | OutputConstraint::GreaterThan(i, j) => {
                    referenced.push(i);
                    referenced.push(j);
                }
                OutputConstraint::LessEqConst(i, _)
                | OutputConstraint::GreaterEqConst(i, _)
                | OutputConstraint::LessThanConst(i, _)
                | OutputConstraint::GreaterThanConst(i, _) => referenced.push(i),
                _ => return Vec::new(),
            }
        }
    }
    referenced.sort_unstable();
    referenced.dedup();
    referenced
}

/// #disjunctive-shared-bootstrap: compute the one bootstrap the serial loop
/// lends to its clauses, or say why not.
///
/// Preconditions, each its own `reason`: the key is armed; the model is a
/// `Graph` (the bootstrap is the graph engine's); the route is ReLU-split
/// without gpu_bab (the only route that reads the precomputed slot); INVPROP is
/// off (it conditions node bounds on the clause's OWN output constraint, so a
/// shared map would be wrong for every other clause); at least two clauses
/// are pending (one clause shares nothing); the BaB deadline is live.
///
/// Deadlines mirror one clause's (`bab_loop.rs`): the node sweep gets the BaB
/// boundary, the iterative root pass `initial_bounds_fraction` of what remains
/// -- applied ONCE to the whole property instead of once per clause slice.
/// A decline or an error here is never a verdict: the loop below then runs
/// the per-clause bootstrap exactly as it always has.
#[allow(clippy::too_many_arguments)]
fn build_disjunctive_shared_bootstrap(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    pending: &[usize],
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    use_relu_split: bool,
    gpu_bab: bool,
    ledger: &PhaseBudgetLedger,
) -> SharedBootstrapCache {
    let armed = config.disjunctive_shared_bootstrap;
    let route = match model_net {
        BetaCrownModel::Sequential(_) => "sequential",
        BetaCrownModel::Graph(_) if gpu_bab => "gpu_bab",
        BetaCrownModel::Graph(_) if use_relu_split => "relu_split",
        BetaCrownModel::Graph(_) => "input_split",
    };
    if !armed {
        return SharedBootstrapCache::unshared(armed, route, "dark");
    }
    let BetaCrownModel::Graph(graph) = model_net else {
        return SharedBootstrapCache::unshared(armed, route, "sequential");
    };
    if route != "relu_split" {
        return SharedBootstrapCache::unshared(armed, route, "route");
    }
    if config.alpha_config.invprop.enabled {
        return SharedBootstrapCache::unshared(armed, route, "invprop");
    }
    if pending.len() < 2 {
        return SharedBootstrapCache::unshared(armed, route, "single");
    }
    let now = Instant::now();
    let bab_deadline = ledger.bab_deadline().or_else(|| ledger.overall_deadline());
    if bab_deadline.is_some_and(|deadline| now >= deadline) {
        return SharedBootstrapCache::unshared(armed, route, "expired");
    }
    let iterative_deadline = bab_deadline.map(|deadline| {
        let frac = config.phase_budget.initial_bounds_fraction.clamp(0.0, 1.0);
        let slice = deadline.saturating_duration_since(now).mul_f32(frac);
        now.checked_add(slice).unwrap_or(now).min(deadline)
    });
    // Same config the clauses derive theirs from (modulo the clause matrix,
    // refused above, and the clause timeout, which the deadlines carry).
    let shared_verifier = verifier.with_config_from(config.clone());
    match shared_verifier.compute_shared_graph_bootstrap(
        graph,
        input,
        referenced_output_indices(clauses, pending),
        bab_deadline,
        iterative_deadline,
    ) {
        Ok(Some(bootstrap)) => SharedBootstrapCache {
            bootstrap: Some(bootstrap),
            armed,
            route,
            reason: "ok",
            reused: 0,
            fallback_box: 0,
        },
        Ok(None) => SharedBootstrapCache::unshared(armed, route, "declined"),
        Err(error) => {
            // Fail OPEN to the historical path: the lever must not add a
            // failure mode the per-clause bootstrap would not also hit.
            warn!(%error, "#disjunctive-shared-bootstrap: bootstrap failed; clauses re-bootstrap");
            SharedBootstrapCache::unshared(armed, route, "error")
        }
    }
}

pub(super) fn verify_multi_clause_disjunction(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    use_relu_split: bool,
    gpu_bab: bool,
    pgd_attack: bool,
    pgd_restarts: usize,
    pgd_steps: usize,
    timeout: u64,
    gemm_engine: Option<&dyn GemmEngine>,
    attack_engine_source: AttackEngineSource<'_>,
    json: bool,
    ledger: &PhaseBudgetLedger,
) -> Result<BetaCrownResult> {
    // #attack-steering-unquarantine: falsification lanes take the live
    // steering accelerator when one exists, else whatever the proof channel
    // carries (CPU / None). Bound-bearing consumers below keep `gemm_engine`.
    // #wallhugger-arming-cost: the take is NON-BLOCKING — while background
    // arming is still in flight the attack runs un-steered (best-effort by
    // nature); later take-points below re-take for late pickup.
    let attack_take = attack_engine_source.take();
    let attack_engine = attack_take
        .as_ref()
        .map(ResolvedAttackEngine::as_gemm)
        .or(gemm_engine);
    // LEVER 1: fresh IMB early-attempt flag per verify (scopes the in-lane suppression
    // to this instance; no-op today since ny runs one instance per process).
    ny_propagate::imb::reset_early_attempted();

    let clauses = if vnnlib.output_constraint_clauses.is_empty() {
        vec![vnnlib.output_constraints.clone()]
    } else {
        vnnlib.output_constraint_clauses.clone()
    };

    let num_clauses = clauses.len() as u64;

    // Keep all nested work on the caller's authoritative ledger; rebasing it
    // here would extend the wall-clock deadline and could re-arm an explicitly
    // disabled MIP reservation.
    ledger.emit_telemetry("disjunctive-enter");
    let overall_start = Instant::now();
    let overall_deadline = ledger.overall_deadline();
    let timeout_result = || BetaCrownResult {
        result: BabVerificationStatus::Timeout,
        domains_explored: 0,
        time_elapsed: overall_start.elapsed(),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    };

    // #fit-100s (dark, `NY_SKIP_DISJ_PGD=1`, default OFF = byte-identical):
    // skip the disjunctive global PGD phase AND its #attack-extend retry
    // entirely — measured on cifar100_2024 prop7641 @100s the phase burned
    // 39.3s (30s PGD cap + 9.7s extension on a "promising" −0.022 margin)
    // finding nothing on a hold instance. The forward-linear reference map
    // warm (#w5-bab-throughput, ~22s, REQUIRED by the bootstrap + root margin
    // pass) still runs — synchronously, bounded by the overall deadline — so
    // only the dead falsification time is reclaimed for BaB. Measurement
    // lever for hold-heavy sweeps; competition keeps PGD (a missed sat is a
    // lost +10 — or worse, a wrong-verdict risk taken by BaB instead).
    let skip_disj_pgd_env = matches!(std::env::var("NY_SKIP_DISJ_PGD").ok().as_deref(), Some("1"));
    // #disj-demand (typed preset key `bab.phase_budget.demand_sized_disjunctive_pgd`,
    // default OFF): zero the disjunctive attack slice when the ledger's static
    // probe proved the net is outside the exact-gradient fragment, so the phase
    // could only spend the slice on SPSA over point-IBP forwards. MEASURED on
    // vit_2023 at the official 100 s budget: 4.75 s of a 95 s tier per row,
    // zero counterexamples ever. The skip takes the SAME path as
    // NY_SKIP_DISJ_PGD below (warm still runs, #attack-extend retry skipped),
    // and the seconds flow to BaB because the fast path re-bases on
    // `ledger.remaining()`. Attack-scheduling only; the eprintln is the
    // lever's own observable.
    let disj_pgd_demand_zeroed = ledger.disjunctive_pgd_demand_zeroed();
    if pgd_attack && disj_pgd_demand_zeroed && !skip_disj_pgd_env {
        eprintln!(
            "[disj-pgd] demand-zeroed: net is statically outside the exact-gradient fragment \
             and the preset armed demand_sized_disjunctive_pgd; the disjunctive attack slice \
             goes to BaB (#disj-demand)"
        );
    }
    let skip_disj_pgd = skip_disj_pgd_env || disj_pgd_demand_zeroed;
    if pgd_attack && skip_disj_pgd {
        if let BetaCrownModel::Graph(g) = model_net {
            if graph_has_conv2d_layers(model_net)
                && GraphNetwork::forward_linear_reference_enabled()
            {
                let warm_start = Instant::now();
                match g.collect_forward_linear_bounds_dag_cached(
                    input,
                    gemm_engine,
                    overall_deadline,
                ) {
                    Ok(_) => info!(
                        elapsed_s = warm_start.elapsed().as_secs_f64(),
                        "Forward-linear reference map warmed; disjunctive PGD skipped (NY_SKIP_DISJ_PGD, #fit-100s)"
                    ),
                    Err(e) => debug!(
                        error = %e,
                        "Forward-linear warmer refused under NY_SKIP_DISJ_PGD (fail-closed; verify phase recomputes)"
                    ),
                }
            }
        }
    }

    // Global disjunctive PGD attack before per-clause BaB.
    // Reserve (1 - disjunctive_pgd_fraction) of timeout for CROWN precheck + BaB.
    // Source: PhaseBudgetConfig.disjunctive_pgd_fraction (default 0.50).
    if pgd_attack && !skip_disj_pgd {
        let (attack_restarts, attack_steps) = disjunctive_sampling_budget(pgd_restarts, pgd_steps);
        // Time-capped on tiny budgets so the attack cannot starve the CROWN
        // precheck + BaB phases (#four-walls; lsnc_relu burned 50% of a 20s
        // budget here before any bound computation ran), and clamped to the
        // optional absolute ceiling `disjunctive_pgd_max_secs` so a LARGE budget
        // cannot hand a hard-UNSAT (hold) falsifier a huge dead slice before BaB
        // (#reclaim-pgd; the fast-path BaB re-bases on `ledger.remaining()`, so
        // the reclaimed seconds flow straight to BaB).
        let pgd_deadline = ledger.disjunctive_pgd_deadline();
        // #attack-setup-regrant: the declared search slice measured from this
        // phase's start, capped by the overall deadline. The batched exact-VJP
        // driver applies it ONCE, and only when its setup already consumed
        // `pgd_deadline` before its first search step (a run that would
        // otherwise do zero steps); every other run keeps `pgd_deadline`.
        let setup_regrant = ledger.disjunctive_pgd_setup_regrant();
        // #w5-bab-throughput: warm the certified forward-linear reference map
        // CONCURRENTLY with the attack. On conv DAGs (cifar100-class), the
        // verify phase's bootstrap, spec-propagation setup, and root-margin
        // composition all consume this input-keyed map (~25s one-off on
        // cifar100), and it depends only on (graph, root input box) — both
        // known now. Computing it during the attack phase converts dead serial
        // time into BaB budget; `configured_graph_for_crown` carries the cache
        // into the verify clone. Sound: the warmer only populates a cache of
        // certified bounds through the exact machinery the verify phase would
        // run itself; on deadline refusal nothing is cached (status quo).
        let warm_graph: Option<&GraphNetwork> = match model_net {
            BetaCrownModel::Graph(g)
                if graph_has_conv2d_layers(model_net)
                    && GraphNetwork::forward_linear_reference_enabled() =>
            {
                Some(g)
            }
            _ => None,
        };
        const WARM_GRACE: Duration = Duration::from_secs(10);
        let warm_deadline = match (pgd_deadline, ledger.overall_deadline()) {
            (Some(p), Some(o)) => Some((p + WARM_GRACE).min(o)),
            (Some(p), None) => Some(p + WARM_GRACE),
            (None, o) => o,
        };
        let warm_graph = warm_graph.filter(|_| {
            let admitted = GraphNetwork::forward_linear_cold_build_admitted(warm_deadline);
            if !admitted {
                info!(
                    "Skipping optional forward-linear attack-phase warmer: insufficient cold-build headroom"
                );
            }
            admitted
        });
        let mut attack_feedback = super::disjunctive_pgd::DisjunctiveAttackFeedback::default();
        let warmer = warm_graph.map(|g| {
            move || {
                    let warm_start = Instant::now();
                    match g.collect_forward_linear_bounds_dag_cached(
                        input,
                        gemm_engine,
                        warm_deadline,
                    ) {
                        Ok(_) => info!(
                            elapsed_s = warm_start.elapsed().as_secs_f64(),
                            "Forward-linear reference map warmed during attack phase (#w5-bab-throughput)"
                        ),
                        Err(e) => debug!(
                            error = %e,
                            "Forward-linear attack-phase warmer refused (fail-closed; verify phase recomputes)"
                        ),
                    }
                }
        });
        // #attack-stall (design S4): the adaptive per-INSTANCE cutoff for this
        // phase. Default-inert — no shipped preset sets the knob — so this is a
        // no-op until a category arms it behind its own A/B. When armed it can
        // only end candidate GENERATION early; the reclaimed seconds flow to
        // BaB automatically because the fast-path re-bases on
        // `ledger.remaining()`.
        let stall = super::attack_stall::AttackStallPolicy::from_phase_policy(ledger.policy());
        if stall.is_armed() {
            info!(
                "Adaptive attack stall cutoff ARMED for the disjunctive PGD phase (#attack-stall); \
                 NY_ATTACK_STALL_CUT=0 disables"
            );
        }
        let attack_outcome = run_with_optional_forward_linear_warmer(warmer, || {
            try_disjunctive_sampling_attack_with_config(
                model_net,
                input,
                &clauses,
                &vnnlib.per_clause_input_bounds,
                ny_propagate::PgdConfig {
                    setup_regrant,
                    ..beta_crown_pgd_config(config, attack_restarts, attack_steps, pgd_deadline)
                },
                attack_engine,
                json,
                stall,
                Some(&mut attack_feedback),
            )
        });
        match attack_outcome {
            Ok(Some(result)) => return Ok(result),
            Ok(None) => {
                // #attack-extend: ONE bounded attack extension when the phase
                // ended without a counterexample but the closest-to-violation
                // margin was PROMISING (a hair under 0 ⇒ the ascent was cut by
                // the preset attack cap, and BaB can never prove a sat
                // instance). Hopeless margins hand off to BaB immediately, so
                // near-wall UNSAT proofs keep their slice. The retry CONTINUES
                // the restart seed sequence where the cap cut it (replaying the
                // cut restart from scratch), reproducing the measured longer
                // continuous run. Attack-only; NY_ATTACK_EXTEND=0 disables.
                if let Some(ext_slice) = super::attack_extension::attack_extension_slice(
                    ledger.remaining(),
                    &attack_feedback,
                    ledger.policy().attack_extension_fraction,
                ) {
                    let ext_deadline = {
                        let d = Instant::now() + ext_slice;
                        overall_deadline.map_or(d, |o| d.min(o))
                    };
                    let best_margin = attack_feedback.best_margin.unwrap_or(f32::NEG_INFINITY);
                    info!(
                        best_margin,
                        hit_deadline = attack_feedback.hit_deadline,
                        restarts_started = attack_feedback.restarts_started,
                        steps_taken = attack_feedback.steps_taken,
                        extension_s = ext_slice.as_secs_f64(),
                        "Attack extension granted: promising margin, one bounded retry (#attack-extend)"
                    );
                    if !json {
                        println!(
                            "  Attack extension: best margin {best_margin:.5} is promising (>= 0 is a counterexample) — one bounded retry (+{:.1}s, continuing the restart sequence)...",
                            ext_slice.as_secs_f64()
                        );
                    }
                    let ext_config = ny_propagate::PgdConfig {
                        // Continue the restart seed sequence at the restart the
                        // first run's cap cut (seed 42 + k ⇒ restart index k;
                        // see graph_pgd.rs restart_seed_offset).
                        seed:
                            42u64.wrapping_add(
                                attack_feedback.restarts_started.saturating_sub(1) as u64
                            ),
                        // The retry is deadline-bound; keep the preset's step
                        // depth but never let the restart count bind first.
                        num_restarts: attack_restarts.max(4096),
                        ..beta_crown_pgd_config(
                            config,
                            attack_restarts,
                            attack_steps,
                            Some(ext_deadline),
                        )
                    };
                    // #wallhugger-arming-cost late pickup: arming may have
                    // finished during the first attack wave; re-take so the
                    // extension retry is steered when the engine is now ready.
                    let ext_take = attack_engine_source.take();
                    let ext_attack_engine = ext_take
                        .as_ref()
                        .map(ResolvedAttackEngine::as_gemm)
                        .or(attack_engine);
                    match try_disjunctive_sampling_attack_with_config(
                        model_net,
                        input,
                        &clauses,
                        &vnnlib.per_clause_input_bounds,
                        ext_config,
                        ext_attack_engine,
                        json,
                        // The extension retry keeps the measured #attack-extend
                        // behavior exactly: it is granted only to an ascent the
                        // margin gate already called promising, and it is
                        // reached only when no stall cut fired.
                        super::attack_stall::AttackStallPolicy::disabled(),
                        None,
                    ) {
                        Ok(Some(result)) => return Ok(result),
                        Ok(None) => {
                            info!("Attack extension exhausted without a counterexample; handing off to BaB");
                        }
                        Err(e) => {
                            debug!(error = %e, "Attack extension failed, continuing to BaB");
                        }
                    }
                } else if attack_feedback.stalled_out {
                    // Flight note only. "The attack stopped improving" is a
                    // statement about the FALSIFIER, never about the property:
                    // the bound/BaB phases below run exactly as they would have
                    // after a finished or deadline-cut attack, over the same
                    // certified bounds, and remain the only source of a verdict.
                    info!(
                        best_margin = attack_feedback.best_margin.unwrap_or(f32::NEG_INFINITY),
                        restarts_started = attack_feedback.restarts_started,
                        steps_taken = attack_feedback.steps_taken,
                        remaining_s = ledger
                            .remaining()
                            .map_or(f64::INFINITY, |d| d.as_secs_f64()),
                        "Attack stall cutoff fired: margin plateaued, remaining attack budget handed to the bound/BaB phases (#attack-stall)"
                    );
                } else if attack_feedback.best_margin.is_some() {
                    debug!(
                        best_margin = attack_feedback.best_margin.unwrap_or(f32::NEG_INFINITY),
                        "No attack extension: margin not promising or budget too small (#attack-extend)"
                    );
                }
            }
            Err(e) => {
                debug!(error = %e, "Disjunctive sampling attack failed, continuing");
            }
        }
    }

    // Print-only discriminator for the scored-100 vs 200 ResNet pathology:
    // this lands after the PGD worker AND the scoped forward-linear warmer
    // have joined, so its elapsed field prices their complete serial wall tax.
    ledger.emit_telemetry("disjunctive-after-pgd-and-warmer");

    if overall_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return Ok(timeout_result());
    }

    // #attack-open-disjuncts (typed `attack.open_disjunct_restrict`, DARK by
    // default — one bool test when unset): the root-pruned, per-disjunct,
    // many-restart attack stage. It reads the margin-row lane's certified root
    // (bounded wait), attacks only the disjuncts that root leaves open, lowest
    // root bound first, and hands a candidate up ONLY through the unchanged
    // `re_evaluate_and_confirm` funnel; vnncomp's trusted-ORT gate still
    // re-confirms any `sat`. Attack-only: it moves seconds from BaB to the
    // falsifier (bounded by its keys and a tail reserve), never a bound.
    if config.open_disjunct_attack.restrict {
        if pgd_attack && !disj_pgd_demand_zeroed {
            // Late pickup, as the #attack-extend retry does: arming may have
            // finished during the joint phase.
            let stage_take = attack_engine_source.take();
            let stage_engine = stage_take
                .as_ref()
                .map(ResolvedAttackEngine::as_gemm)
                .or(attack_engine);
            let stage_inputs = super::attack_open_disjuncts::StageInputs {
                model_net,
                input,
                clauses: &clauses,
                per_clause_input_bounds: &vnnlib.per_clause_input_bounds,
                config,
                pgd_steps,
                attack_engine: stage_engine,
                overall_deadline,
            };
            match super::attack_open_disjuncts::run_open_disjunct_attack_stage(&stage_inputs) {
                Ok(Some(result)) => return Ok(result),
                Ok(None) => {}
                Err(e) => {
                    debug!(error = %e, "Open-disjunct attack stage failed; continuing to BaB");
                }
            }
            if overall_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Ok(timeout_result());
            }
        } else {
            // Armed but gated off here: say so, so the row is not read as one
            // an earlier lane decided (an absence is not evidence).
            super::attack_open_disjuncts::note_stage_declined(
                config.open_disjunct_attack,
                if pgd_attack {
                    "demand_zeroed"
                } else {
                    "pgd_disabled"
                },
            );
        }
    }

    // Per-disjunct alpha fast-path.
    if config.optimize_disjuncts_separately && use_relu_split {
        return super::disjunctive_per_disjunct::try_per_disjunct_multi_objective(
            model_net,
            config,
            verifier,
            input,
            vnnlib,
            &clauses,
            gemm_engine,
            pgd_attack,
            pgd_restarts,
            pgd_steps,
            json,
            timeout,
            overall_start,
            ledger,
        );
    }

    // Conv2d-heavy graph fast-path — skip precheck, route to multi-
    // objective BaB directly (precheck is redundant, BaB recomputes anyway).
    if supports_graph_multi_objective_clause_batch(
        model_net,
        &clauses,
        &vnnlib.per_clause_input_bounds,
        use_relu_split,
        gpu_bab,
    ) {
        let remaining_timeout = ledger.remaining_for_engine();
        if timeout > 0 && remaining_timeout.is_zero() {
            return Ok(timeout_result());
        }

        let remaining_timeout_secs = ledger.remaining_secs_clamped();
        let remaining_timeout_secs = if timeout == 0 {
            0
        } else {
            remaining_timeout_secs.max(1)
        };
        let remaining_config = BetaCrownConfig {
            timeout: remaining_timeout,
            ..config.clone()
        };
        let remaining_verifier = verifier.with_config_from(remaining_config.clone());

        debug!(
            clauses = clauses.len(),
            timeout_s = remaining_timeout_secs,
            "Graph multi-objective fast-path: skipping precheck, routing all clauses to BaB"
        );
        ledger.emit_telemetry("disjunctive-graph-handoff");

        let nested_ledger = ledger.child_with_timeout_secs(remaining_timeout_secs);
        let mut result = super::verify_relational_constraints_impl(
            model_net,
            input,
            vnnlib,
            &remaining_config,
            &remaining_verifier,
            use_relu_split,
            gpu_bab,
            false, // global disjunctive PGD already ran above
            pgd_restarts,
            pgd_steps,
            remaining_timeout_secs,
            gemm_engine,
            attack_engine_source,
            json,
            &nested_ledger,
            None, // the multi-objective lane bootstraps once itself
            None, // #attack-concurrent: no upfront PGD on this nested call
        )?;
        result.time_elapsed = overall_start.elapsed();

        // Fallback PGD for no-branchable-neuron BaB results.
        // #wallhugger-arming-cost late pickup: this runs after BaB, so arming
        // has long since settled; re-take rather than reuse the phase-start
        // resolution.
        let fallback_take = attack_engine_source.take();
        let fallback_attack_engine = fallback_take
            .as_ref()
            .map(ResolvedAttackEngine::as_gemm)
            .or(attack_engine);
        if let Some(sat) = try_no_branchable_neuron_pgd_fallback(
            &result,
            pgd_attack,
            model_net,
            input,
            &clauses,
            &vnnlib.per_clause_input_bounds,
            config,
            pgd_restarts,
            pgd_steps,
            fallback_attack_engine,
            json,
            ledger,
        ) {
            return Ok(sat);
        }

        result.time_elapsed = overall_start.elapsed();
        return Ok(result);
    }

    // LEVER 1 — IMB EARLY FAST-PATH, AHEAD of the ~365 s CROWN-IBP per-output precheck.
    // The IMB certificate neither needs nor consumes that precheck, so running it here
    // — AFTER the PGD attack (so SAT instances are still falsified first) but BEFORE the
    // precheck — hands it the full remaining budget instead of the ~115 s that were left
    // downstream. On a clear ⇒ return `unsat` immediately; on a miss/disarmed/error ⇒
    // `None` and the UNCHANGED precheck + BaB run exactly as today (the downstream
    // in-lane `#imb-early` block is suppressed via `imb::early_attempted`, so the IMB
    // never repeats). Gated `NY_IMB=1 && NY_IMB_WIRE=1`; a no-op for every other model.
    if let BetaCrownModel::Graph(g) = model_net {
        if supports_grouped_disjunctive_contract(vnnlib, use_relu_split) {
            if let Ok(plan) = build_grouped_disjunctive_objectives(vnnlib) {
                if let Some(res) = verifier.try_imb_early_disjunctive(
                    g,
                    input,
                    &plan.objectives,
                    &plan.thresholds,
                    &plan.clause_sizes,
                    gemm_engine,
                    overall_deadline,
                ) {
                    return Ok(finalize_disjunctive_result(
                        res,
                        overall_start,
                        BabVerificationStatus::Verified,
                    ));
                }
            }
        }
    }

    // CROWN pre-check: screen clauses with a single pass before per-clause BaB.
    // Cap precheck budget to disjunctive_precheck_fraction of total timeout.
    // The slice is computed from NOW (not the ledger start): with the start
    // base, any attack phase longer than the precheck fraction left this
    // deadline already expired, so the root CROWN pass silently degraded to
    // IBP-only bounds (#four-walls; `phase_deadline_from_now` is capped at the
    // overall deadline, so this only reclaims the slice the precheck was
    // always meant to have).
    // Per-clause-box disjunctions (nn4sys mscn/lindex bands) get a LARGER
    // slice: for that shape the precheck is the batched box-refinement engine
    // — the only lane measured to make progress (999/1000 clauses on mscn
    // cardinality_0_500 in <1s), while the downstream serial per-clause BaB
    // was measured to burn its whole slice without converging (shared-root
    // intermediate bounds; 100k domains/87s, zero clauses closed). The
    // refinement returns as soon as every clause is decided, so the bigger
    // slice only spends time that the serial loop would otherwise waste;
    // the remaining fraction still reaches the loop for any stragglers.
    // 0.95 (was 0.75, then 0.9): the mscn `_dual` band instances are decided
    // ONLY by the box-refinement screen's f64 escalation — measured on
    // cardinality_1_240_128_dual, the downstream serial per-clause BaB closes
    // ZERO clauses (0.9s/clause of f32 input-split BaB against ±1e-5 margins
    // below the f32 rounding floor), so every second reserved for it after an
    // unfinished screen is wasted. The screen still returns the moment all
    // clauses are decided, so fast instances (lindex, non-dual mscn, <1s) are
    // unaffected, and the small reserve still reaches the downstream lanes
    // for shapes the screen cannot close.
    // The bump is GRAPH-ONLY and 0.95: it funds the batched box-refinement
    // screen, the only lane measured to progress on the nn4sys Mul-heavy DAGs
    // (per-clause BaB closes zero clauses there), and with the batched
    // multi-box screen (#f64-batch-boxes) cardinality_1_1_2048_dual's full
    // refinement tree costs ~12.8s at its 20s official budget — a 0.9 slice
    // finished the tree at the deadline's edge (verified=0 by <0.5s) while
    // the reserve went unused (per-clause BaB closes zero, MIP ineligible for
    // graph models). On SEQUENTIAL models with per-clause boxes (acasxu
    // prop_6-class input-box disjunctions) the screen is one cheap CROWN pass
    // per clause and the real solver is the per-clause input-split BaB loop
    // below — a bumped slice starved it, so sequential keeps the policy value.
    // Saturation-Escape Branching engagement (M1; preset key
    // `bab.branching.input_split.sat_escape_branch`, env `NY_SAT_ESCAPE_BRANCH`
    // overriding either way — default OFF ⇒ this whole block is byte-identical).
    // On the mscn `_dual`
    // per-clause-box graphs the box-refinement screen structurally CANNOT close
    // the near-closing multi-dim boxes (f64 enclosure below the f32 margin
    // floor), so the 0.95 precheck fraction burns the whole budget on a doomed
    // screen and the serial per-clause input-split BaB loop below is reached with
    // ~0 budget — it returns `unknown` WITHOUT ever branching the inconclusive
    // disjuncts (the measured engagement gap). With the gate on, cap the screen
    // at the default fraction so it still discharges the easy clauses but RESERVES
    // budget for the per-clause BaB loop, where the SEB scorer targets the
    // de-saturating dims. Soundness is unchanged: the serial loop still proves
    // HOLD only when every inconclusive disjunct is refuted on its exact box
    // partition (`multi_dim_split_boxes` is a complete cover), identical to today.
    // `sat_escape_branch_armed` is the ONE resolution point shared with the
    // ny-propagate SEB scorer gate, so this budget cap and the brancher it
    // funds can never disagree (#nn4sys-seb-dark).
    let seb = config.sat_escape_branch_armed();
    let has_per_clause_boxes = vnnlib.per_clause_input_bounds.iter().any(|b| !b.is_empty());
    let precheck_fraction = if has_per_clause_boxes && matches!(model_net, BetaCrownModel::Graph(_))
    {
        if seb {
            ledger.policy().disjunctive_precheck_fraction.min(0.5)
        } else {
            ledger.policy().disjunctive_precheck_fraction.max(0.95)
        }
    } else {
        ledger.policy().disjunctive_precheck_fraction
    };
    // #precheck-abs-cap: the fraction slice, additionally clamped by the
    // optional ABSOLUTE ceiling `disjunctive_precheck_max_secs`. A FRACTION of
    // TOTAL granted from NOW bounds nothing relative to what is left (the
    // attack phase's spend is not deducted) and the work it sizes grows with
    // the model, so the fraction alone degenerates into a BaB-starvation lever
    // on large budgets. `None` (every category that does not set the knob)
    // keeps the pure-fraction deadline byte-identical.
    let precheck_deadline = {
        let phase = ledger.disjunctive_precheck_deadline(precheck_fraction);
        match (phase, overall_deadline) {
            (Some(p), Some(o)) => Some(p.min(o)),
            (p, o) => p.or(o),
        }
    };
    let t_crown = Instant::now();
    // #precheck-flight (class (c) instrument). The pre-BaB CROWN precheck used
    // to announce itself ONLY through the INFO line below, and the flight
    // stream had NO pre-BaB phase boundary at all — between `margin_row_reserve`
    // at ~0.02 s and `result_publish` at the wall there was nothing. So
    // "the precheck ate the budget" (MEASURED at 717.9 s of 900 on cgan2026
    // row 10, `cGAN_imgSz64_nCh_1` prop_2) is undetectable from any banked
    // artifact: `ny benchmarks run` forces `RUST_LOG=error`, discarding the
    // INFO line, and the sidecar carried no substitute. The board-wide stop
    // census (`reports/measured-2026/stop_census_2026-08-30/` §4) scores that
    // class 1 row and says plainly that the 1 is a blind spot, not a
    // measurement.
    //
    // These two notes make the phase COMPUTABLE for every future bank at zero
    // runtime cost: the pair brackets the phase, so a consumer differences
    // their `at_secs` — no new field, no new file, and the start note lands
    // even if the phase is later cut off by the overall deadline.
    //
    // ORDINAL, not a latch. `verify_multi_clause_disjunction` recurses for
    // box-grouped sub-problems, so a run can contain several prechecks. Each
    // pair is numbered instead of only the first being recorded, because an
    // absent note and a suppressed note are indistinguishable downstream — the
    // trap this instrument exists to close. **Ordinal 0 is the OUTERMOST
    // precheck**, the one `disjunctive_precheck_fraction` sizes and the one the
    // census measured; a consumer computing class (c) uses that pair.
    let precheck_ordinal = {
        static PRECHECKS_NOTED: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        PRECHECKS_NOTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    };
    crate::flight::note(
        "crown_precheck_begin",
        crate::flight::FlightStatus::Ran,
        Some(format!(
            "ordinal={precheck_ordinal} clauses={} precheck_fraction={:.3}",
            clauses.len(),
            precheck_fraction
        )),
    );
    // #fwd-root-screen (DARK, typed `bab.forward_root_screen`, V4 route (b)):
    // ONE certified correlation-preserving zonotope forward pass over the root
    // box, BEFORE the CROWN precheck. Its refutations are UNIONED with the
    // precheck's below — it never replaces IBP/C1 — and when it alone refutes
    // every clause the row is verified here without spending the precheck.
    // Dark: `None`, no work, no output, the historical path byte-for-byte.
    // The screen may spend at most HALF of what is left of the precheck slice:
    // the CROWN/IBP precheck below is what banks the IBP- and C1-refuted rows
    // (the most fragile unsat set), and a screen that ate the whole slice would
    // hand it an expired deadline. A screen cut by this cap declines
    // (`reason=deadline`), which refutes nothing.
    let fwd_root_refuted: Option<Vec<bool>> = match model_net {
        BetaCrownModel::Graph(graph) if config.forward_root_screen => {
            let screen_deadline = precheck_deadline.map(|end| {
                let now = Instant::now();
                now + end.saturating_duration_since(now) / 2
            });
            Some(super::disjunctive_precheck::forward_root_screen_clauses(
                graph,
                input,
                &clauses,
                config.forward_root_screen_max_mb,
                screen_deadline,
            ))
        }
        _ => None,
    };
    if let Some(screen) = fwd_root_refuted.as_ref() {
        if !clauses.is_empty() && screen.len() == clauses.len() && screen.iter().all(|&r| r) {
            eprintln!(
                "[fwd-root] union screen_refuted={total} precheck_refuted=skipped \
                 union_refuted={total} new_by_screen={total} total={total}",
                total = clauses.len()
            );
            crate::flight::note(
                "crown_precheck_end",
                crate::flight::FlightStatus::Ran,
                Some(format!(
                    "ordinal={precheck_ordinal} verified={total} total={total} elapsed_s={:.3} \
                     by=fwd_root_screen",
                    t_crown.elapsed().as_secs_f64(),
                    total = clauses.len()
                )),
            );
            return Ok(BetaCrownResult {
                result: BabVerificationStatus::Verified,
                domains_explored: 0,
                time_elapsed: overall_start.elapsed(),
                max_depth_reached: 0,
                output_bounds: None,
                cuts_generated: 0,
                domains_verified: 0,
            });
        }
    }
    // #lowdim-fold (typed opt-in `bab.branching.input_split.lowdim_fold`): on a
    // graph model with per-clause input boxes, prove clauses on their FREE
    // coordinates first (proofs only), within `lowdim_prover_share` of the
    // precheck slice; the screen below then sees only the clauses it left.
    let lowdim_fold_verified: Option<Vec<bool>> = match model_net {
        BetaCrownModel::Graph(graph) if config.input_split_lowdim_fold && has_per_clause_boxes => {
            let now = Instant::now();
            let end = precheck_deadline.unwrap_or(now + Duration::from_secs(60));
            let lane_end = now
                + end
                    .saturating_duration_since(now)
                    .mul_f32(config.lowdim_prover_share());
            eprintln!("[lowdim-fold] armed=1 clauses={}", clauses.len());
            Some(super::disjunctive_precheck::lowdim_fold_clauses(
                graph,
                input,
                &clauses,
                &vnnlib.per_clause_input_bounds,
                vnnlib.num_outputs,
                &config.lowdim_config(),
                lane_end,
            ))
        }
        _ => None,
    };
    let pre_verified = match lowdim_fold_verified {
        Some(done) if done.len() == clauses.len() && done.iter().any(|&v| v) => {
            let keep: Vec<usize> = (0..clauses.len()).filter(|&i| !done[i]).collect();
            let mut merged = done;
            if !keep.is_empty() {
                let sub_clauses: Vec<Vec<OutputConstraint>> =
                    keep.iter().map(|&i| clauses[i].clone()).collect();
                let sub_boxes: Vec<std::collections::BTreeMap<usize, (f64, f64)>> = keep
                    .iter()
                    .map(|&i| {
                        vnnlib
                            .per_clause_input_bounds
                            .get(i)
                            .cloned()
                            .unwrap_or_default()
                    })
                    .collect();
                let sub = crown_precheck_clauses(
                    model_net,
                    input,
                    &sub_clauses,
                    &sub_boxes,
                    gemm_engine,
                    precheck_deadline,
                );
                for (k, &i) in keep.iter().enumerate() {
                    merged[i] = sub.get(k).copied().unwrap_or(false);
                }
            }
            merged
        }
        _ => crown_precheck_clauses(
            model_net,
            input,
            &clauses,
            &vnnlib.per_clause_input_bounds,
            gemm_engine,
            precheck_deadline,
        ),
    };
    // #fwd-root-screen: UNION with the screen's certified refutations (a no-op
    // when the key is dark: `fwd_root_refuted` is `None`).
    let pre_verified = match fwd_root_refuted.as_deref() {
        Some(screen) => {
            let merged =
                super::disjunctive_precheck::union_refutations(&pre_verified, Some(screen));
            let count = |v: &[bool]| v.iter().filter(|&&r| r).count();
            let new_by_screen = merged
                .iter()
                .zip(pre_verified.iter())
                .filter(|(u, p)| **u && !**p)
                .count();
            eprintln!(
                "[fwd-root] union screen_refuted={} precheck_refuted={} union_refuted={} \
                 new_by_screen={new_by_screen} total={}",
                count(screen),
                count(pre_verified.as_slice()),
                count(merged.as_slice()),
                clauses.len()
            );
            merged
        }
        None => pre_verified,
    };
    let pre_verified_count = pre_verified.iter().filter(|&&v| v).count();
    let precheck_elapsed_s = t_crown.elapsed().as_secs_f64();
    crate::flight::note(
        "crown_precheck_end",
        crate::flight::FlightStatus::Ran,
        Some(format!(
            "ordinal={precheck_ordinal} verified={pre_verified_count} total={} \
             elapsed_s={precheck_elapsed_s:.3}",
            clauses.len()
        )),
    );
    info!(
        verified = pre_verified_count,
        total = clauses.len(),
        elapsed_s = precheck_elapsed_s,
        "CROWN precheck phase complete"
    );

    if overall_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return Ok(timeout_result());
    }

    if pre_verified_count > 0 {
        debug!(
            pre_verified = pre_verified_count,
            total = num_clauses,
            elapsed_s = overall_start.elapsed().as_secs_f64(),
            "CROWN pre-check verified clauses"
        );
    }

    // Fast path: if all clauses verified by CROWN pre-check, skip BaB entirely.
    if pre_verified_count == clauses.len() {
        return Ok(BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored: 0,
            time_elapsed: overall_start.elapsed(),
            max_depth_reached: 0,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 0,
        });
    }

    // Alpha-CROWN pre-check. Gated on use_alpha_crown.
    // Slice computed from NOW for the same wrong-base reason as the CROWN
    // precheck above (#four-walls).
    let pre_verified = if config.use_alpha_crown {
        let deadline = {
            // #precheck-abs-cap: same absolute ceiling as the CROWN precheck
            // above. This phase is a SECOND full fraction slice, so without an
            // absolute bound the two prechecks together can spend
            // `2 * fraction * total` after the attack phase already ran.
            let phase =
                ledger.disjunctive_precheck_deadline(ledger.policy().disjunctive_precheck_fraction);
            match (phase, overall_deadline) {
                (Some(p), Some(o)) => Some(p.min(o)),
                (p, o) => p.or(o),
            }
        };
        let t0 = Instant::now();
        let result = alpha_crown_precheck_clauses(
            model_net,
            input,
            &clauses,
            &pre_verified,
            &vnnlib.per_clause_input_bounds,
            gemm_engine,
            deadline,
        );
        info!(
            verified = result.iter().filter(|&&v| v).count(),
            total = clauses.len(),
            elapsed_s = t0.elapsed().as_secs_f64(),
            total_elapsed_s = overall_start.elapsed().as_secs_f64(),
            "Alpha-CROWN precheck phase complete"
        );
        result
    } else {
        debug!("Skipping alpha-CROWN precheck: use_alpha_crown=false");
        pre_verified
    };
    let pre_verified_count = pre_verified.iter().filter(|&&v| v).count();

    if overall_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        return Ok(timeout_result());
    }

    if pre_verified_count == clauses.len() {
        debug!(
            elapsed_s = overall_start.elapsed().as_secs_f64(),
            "Alpha-CROWN pre-check verified ALL clauses — skipping BaB"
        );
        return Ok(BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored: 0,
            time_elapsed: overall_start.elapsed(),
            max_depth_reached: 0,
            output_bounds: None,
            cuts_generated: 0,
            domains_verified: 0,
        });
    }

    // Box-grouped decomposition for input-box disjunctions (acasxu prop_6-class:
    // `(or <box_1> <box_2>) AND (or <out_1> ... <out_m>)` normalizes to K·m
    // clauses over K distinct boxes). Solving per CLAUSE re-pays the BaB tree m
    // times per box (measured on ACASXU_run2a_1_1/prop_6: 58s per clause — 8
    // clauses can never fit 111s), while ONE grouped multi-clause tree over a
    // box amortizes all m objectives (measured: 58s for all 4 — the same cost
    // as one). Group the unresolved clauses by bit-identical box and solve each
    // box as an independent sub-problem via the grouped unified lane.
    // Aggregation is EXACT: UNSAT iff every box verifies; SAT if any box yields
    // a confirmed counterexample (a witness for `box_k ∧ clause` is a witness
    // for the full property); anything else stays unknown/timeout.
    if let Some(result) = try_box_grouped_disjunctive(
        model_net,
        input,
        vnnlib,
        &pre_verified,
        &clauses,
        config,
        verifier,
        use_relu_split,
        gpu_bab,
        pgd_restarts,
        pgd_steps,
        timeout,
        gemm_engine,
        attack_engine_source,
        json,
        ledger,
        overall_start,
    )? {
        return Ok(result);
    }

    // Unified input-split disjunctive BaB.
    if let Some(result) = try_prechecked_unified_input_split_disjunctive_bab(
        model_net,
        input,
        vnnlib,
        &pre_verified,
        config,
        verifier,
        &clauses,
        use_relu_split,
        gpu_bab,
        pgd_attack,
        pgd_restarts,
        pgd_steps,
        timeout,
        gemm_engine,
        json,
        ledger,
        overall_start,
    )? {
        return Ok(result);
    }

    let mut aggregated = BetaCrownResult {
        result: BabVerificationStatus::Verified,
        domains_explored: 0,
        time_elapsed: Duration::from_secs(0),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    };
    // #budget-abandonment FIX B: retry passes over the clauses one pass could
    // not decide.
    //
    // The allocator below is ONE PASS and back-loads the budget: each clause
    // gets `remaining / <undecided from here on>`, so the FIRST clauses get the
    // SMALLEST slice and the failures concentrate exactly there. Measured at
    // HEAD on 14_ieee_ml4acopf prop1 (56 OR-clauses): clause 1 received 10 s and
    // died, clause 56 received 442 s and used 2.1 s, and the row ended with
    // 470 s of its 600 s budget unspent. Nothing anywhere in this function ever
    // looked at an undecided clause twice.
    //
    // `disjunctive_clause_retry_passes == 0` is today's single pass and is
    // byte-identical: the loop below runs once and breaks.
    let retry_passes = config.disjunctive_clause_retry_passes;
    // BLAST-RADIUS FIX: pass 0 owns the FULL ledger, exactly as it always has.
    // Every RETRY pass is bounded by `bab_deadline()` instead, which already
    // subtracts the post-BaB PGD slice and the Graph-MIP escalation's reserved
    // wall. Without this a retry eats the residual the MIP escalation and
    // `confirm_potential_violation` consume, and a row decided today by
    // MIP-after-inconclusive-BaB becomes `unknown`.
    let retry_deadline = match (ledger.bab_deadline(), overall_deadline) {
        (Some(bab), Some(overall)) => Some(bab.min(overall)),
        (Some(bab), None) => Some(bab),
        (None, overall) => overall,
    };
    // Per-clause verdicts carried ACROSS passes. A clause a later pass does not
    // reach keeps the status it last had, which is what makes a deadline-cut
    // retry pass safe.
    let mut verdicts = ClauseVerdicts::new(clauses.len());
    let mut clause_stats: Vec<Option<ClauseStats>> = vec![None; clauses.len()];
    // Pass 0 visits exactly the clauses the CROWN pre-check did not verify, in
    // index order -- the same set, in the same order, as the historical
    // `for (idx, clause) in clauses.iter().enumerate()` with its
    // `if pre_verified[idx] { continue; }`.
    let mut pending: Vec<usize> = (0..clauses.len()).filter(|&i| !pre_verified[i]).collect();
    let mut pass: usize = 0;
    // #disjunctive-shared-bootstrap: computed ONCE here, before the first
    // pass, and lent to every clause on every pass (same box, same network).
    // Dark (`reason=dark`) leaves the per-clause bootstrap byte-identical.
    let mut shared_bootstrap = build_disjunctive_shared_bootstrap(
        model_net,
        input,
        &clauses,
        &pending,
        config,
        verifier,
        use_relu_split,
        gpu_bab,
        ledger,
    );

    loop {
        let undecided_in = pending.len();
        let mut still_undecided: Vec<usize> = Vec::new();
        let mut budget_each_s: u64 = 0;
        let pass_deadline = if pass == 0 {
            overall_deadline
        } else {
            retry_deadline
        };

        for (position, &idx) in pending.iter().enumerate() {
            let clause = &clauses[idx];
            // Clears the DIAGNOSTIC reason only. `ClauseVerdicts` exposes no
            // way to clear a recorded potential violation -- see its doc
            // comment for why that is a soundness property and not a style
            // choice.
            verdicts.begin_visit(idx);

            // Re-checked before EVERY clause, on every pass: a retry must never
            // eat the publication margin. Losing an already-decided row to a
            // retry is the one real hazard in this fix. Pass 0 checks the FULL
            // ledger (byte-identical to HEAD); retries check the MIP-reserving
            // `bab_deadline`.
            if pass_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                verdicts.record_unknown(
                    idx,
                    format!("Timeout before clause {} verification", idx + 1),
                );
                still_undecided.extend_from_slice(&pending[position..]);
                break;
            }

            // Adaptive timeout: remaining time / remaining BaB clauses.
            // Easy clauses (verified by CROWN pre-check) are excluded from the
            // count so hard clauses get the full remaining budget.
            // Minimum 1s per clause to avoid integer truncation when remaining
            // seconds < clause count (e.g., CIFAR-100: 99 clauses, 7s remaining
            // → 7/99=0 in integer division → instant "timeout before clause 1").
            let clause_timeout = if timeout == 0 {
                0
            } else {
                // Pass 0: `ledger.remaining()`, unchanged. Retries: additionally
                // clamped to the MIP-reserving bound so a retry cannot allocate
                // itself wall the escalation is holding.
                let remaining = pass_remaining(ledger, pass, pass_deadline);
                if remaining.is_zero() {
                    verdicts.record_unknown(
                        idx,
                        format!("Timeout before clause {} verification", idx + 1),
                    );
                    still_undecided.extend_from_slice(&pending[position..]);
                    break;
                }
                // Count remaining UNDECIDED clauses from this position onward.
                // On pass 0 `pending` is exactly the non-pre-verified indices in
                // order, so this is the same integer as the historical
                // `pre_verified[idx..].iter().filter(|&&v| !v).count()`.
                let remaining_bab = (pending.len() - position).max(1) as u64;
                // Use ceiling division to ensure at least 1s per clause when
                // there is any remaining time.
                let remaining_secs = remaining.as_secs().max(1);
                let adaptive = remaining_secs / remaining_bab;
                adaptive.max(1)
            };

            if position == 0 {
                budget_each_s = clause_timeout;
            }

            debug!(
                clause = idx + 1,
                total = num_clauses,
                timeout_s = clause_timeout,
                elapsed_s = overall_start.elapsed().as_secs(),
                "Disjunctive clause verification"
            );

            let mut clause_spec = vnnlib.clone();
            clause_spec.output_constraints = clause.clone();
            clause_spec.output_constraint_clauses = Vec::new();
            clause_spec.is_disjunction = false;
            clause_spec.per_clause_input_bounds = Vec::new();

            // Apply per-clause input bounds if available (e.g., nn4sys lindex).
            // Tighten the input BoundedTensor to this clause's specific domain.
            let clause_input;
            let effective_input =
                if let Some(clause_bounds) = vnnlib.per_clause_input_bounds.get(idx) {
                    if !clause_bounds.is_empty() {
                        // Tighten spec's input bounds for this clause
                        for (&var_idx, &(lower, upper)) in clause_bounds {
                            if var_idx < clause_spec.input_bounds.len() {
                                clause_spec.input_bounds[var_idx] = (lower, upper);
                            }
                        }
                        // Create tightened BoundedTensor for this clause's input domain
                        let flat_len = input.lower().len();
                        let mut new_lower = input.lower().clone();
                        let mut new_upper = input.upper().clone();
                        if let (Some(lo_slice), Some(up_slice)) =
                            (new_lower.as_slice_mut(), new_upper.as_slice_mut())
                        {
                            for (&var_idx, &(lb, ub)) in clause_bounds {
                                if var_idx < flat_len {
                                    // Directed f64→f32 rounding: only widen when the
                                    // conversion is inexact (see f64_to_f32_floor —
                                    // unconditional ULP widening pushed tight nn4sys
                                    // band margins below the provable floor).
                                    let lb_f32 = super::disjunctive_precheck::f64_to_f32_floor(lb);
                                    let ub_f32 = super::disjunctive_precheck::f64_to_f32_ceil(ub);
                                    lo_slice[var_idx] = lo_slice[var_idx].max(lb_f32);
                                    up_slice[var_idx] = up_slice[var_idx].min(ub_f32);
                                }
                            }
                        }
                        clause_input = BoundedTensor::new(new_lower, new_upper)
                            .unwrap_or_else(|_| input.clone());
                        &clause_input
                    } else {
                        input
                    }
                } else {
                    input
                };

            // Disable per-clause PGD attacks: each clause's verify_sequential_relational
            // runs sampling+SPSA+PGD+reduced-PGD, taking ~7s per clause on MNIST FC models.
            // With N clauses (typically 9 for MNIST 10-class), PGD alone exceeds the timeout
            // budget (9 × 7s = 63s vs 30s timeout). Per-clause PGD is also redundant — each
            // clause checks "can class k beat the true class" individually, but a single global
            // PGD could check all classes at once. Disabling per-clause PGD lets each clause
            // use its full adaptive timeout budget for CROWN + BaB verification.
            let clause_ledger = ledger.child_with_timeout_secs(clause_timeout);
            let mut clause_config = config_for_clause_invprop(config, &clause_spec);
            // Anchor every engine-local deadline to the exact child budget.  In
            // particular, do not give later clauses a fresh copy of the full
            // property timeout when their adaptive slice is much smaller.
            clause_config.timeout = clause_config
                .timeout
                .min(clause_ledger.remaining_for_engine());
            // Keep the config/verifier pair coherent at this dispatch boundary.
            // Current proof-bearing descendants rebuild from `clause_config`, and
            // this matching parent also makes future direct verifier reads safe.
            let clause_verifier = verifier.with_config_from(clause_config.clone());
            // `None` for a clause with its own input box -- see `for_clause`.
            let clause_shared = shared_bootstrap.for_clause(vnnlib, idx);
            let clause_result = super::verify_relational_constraints_impl(
                model_net,
                effective_input,
                &clause_spec,
                &clause_config,
                &clause_verifier,
                use_relu_split,
                gpu_bab,
                false, // pgd_attack disabled — per-clause PGD is too expensive
                pgd_restarts,
                pgd_steps,
                clause_timeout,
                gemm_engine,
                attack_engine_source,
                json,
                &clause_ledger,
                clause_shared,
                None, // #attack-concurrent: no upfront PGD on a clause call
            )?;

            // Counters are kept PER CLAUSE and folded once after the pass
            // loop. Accumulating here would add a retried clause's counters
            // again on every pass, overstating the published
            // `domains_explored` / `domains_verified` / `cuts_generated` /
            // `time_elapsed` on armed rows. `max_depth_reached` is a max and
            // is safe to take eagerly.
            clause_stats[idx] = Some(ClauseStats {
                domains_explored: clause_result.domains_explored,
                domains_verified: clause_result.domains_verified,
                cuts_generated: clause_result.cuts_generated,
                time_elapsed: clause_result.time_elapsed,
            });
            aggregated.max_depth_reached = aggregated
                .max_depth_reached
                .max(clause_result.max_depth_reached);

            match clause_result.result {
                BabVerificationStatus::Verified => {}
                violation @ BabVerificationStatus::Violated { .. } => {
                    fold_clause_stats(&mut aggregated, &clause_stats);
                    return Ok(finalize_disjunctive_result(
                        aggregated,
                        overall_start,
                        violation,
                    ));
                }
                BabVerificationStatus::PotentialViolation { .. } => {
                    // Sticky and TERMINAL. It is still counted in
                    // `undecided_out` (so the observable keeps HEAD's meaning),
                    // but `ClauseVerdicts::is_terminal` filters it out of the
                    // next pass below: the aggregate is `PotentialViolation`
                    // regardless of what a retry would say, so retrying it can
                    // only burn budget or erase the finding.
                    verdicts.record_potential(idx);
                    still_undecided.push(idx);
                }
                BabVerificationStatus::Unknown { reason } => {
                    verdicts.record_unknown(idx, format!("Clause {}: {}", idx + 1, reason));
                    still_undecided.push(idx);
                }
                BabVerificationStatus::Timeout => {
                    verdicts.record_unknown(idx, format!("Clause {}: timeout", idx + 1));
                    still_undecided.push(idx);
                }
            }
        }

        let undecided_out = still_undecided.len();
        // `pass=0` is emitted ALWAYS, so an absent line means the binary has no
        // lever and a null measured on it is VOID rather than negative.
        // `armed=` carries the RESOLVED key value, so an unarmed yaml
        // (`armed=0`) and a yaml that never reached this cell are no longer
        // byte-identical lines. Without it a null on the armed arm cannot be
        // told from a dark route -- the inference the project's rules forbid.
        eprintln!(
            "[clause-retry] pass={pass} armed={retry_passes} undecided_in={undecided_in} \
undecided_out={undecided_out} potential={} budget_each_s={budget_each_s} remaining_s={} \
retry_bound_s={}",
            verdicts.potential_count(),
            ledger.remaining().unwrap_or(Duration::ZERO).as_secs(),
            retry_deadline.map_or(0, |dl| dl
                .saturating_duration_since(Instant::now())
                .as_secs()),
        );
        // Terminal clauses (a signalled potential violation) never go round
        // again; `undecided_out` above still counts them.
        pending = still_undecided
            .into_iter()
            .filter(|&idx| !verdicts.is_terminal(idx))
            .collect();
        let budget_live = retry_deadline.is_none_or(|deadline| Instant::now() < deadline)
            && (timeout == 0 || !pass_remaining(ledger, pass + 1, retry_deadline).is_zero());
        if pending.is_empty()
            || !another_clause_retry_pass(
                pass,
                retry_passes,
                undecided_in,
                undecided_out,
                budget_live,
            )
        {
            break;
        }
        pass += 1;
    }
    fold_clause_stats(&mut aggregated, &clause_stats);
    shared_bootstrap.emit(clauses.len());

    let final_status = verdicts.final_status();

    Ok(finalize_disjunctive_result(
        aggregated,
        overall_start,
        final_status,
    ))
}

/// Maximum number of DISTINCT per-clause input boxes the box-grouped
/// decomposition will take on. Input-box disjunctions (acasxu prop_6-class)
/// have a handful of boxes; the nn4sys lindex/mscn band families have
/// hundreds-to-thousands of distinct boxes and are served by the batched
/// box-refinement screen instead — this cap keeps them there.
const MAX_BOX_GROUPS: usize = 8;

/// Bit-exact grouping key for a per-clause input box (f64 bounds have no Eq;
/// bit-identical grouping mirrors the box-refinement engine's box grouping).
fn clause_box_key(map: &std::collections::BTreeMap<usize, (f64, f64)>) -> Vec<(usize, u64, u64)> {
    map.iter()
        .map(|(&idx, &(lo, hi))| (idx, lo.to_bits(), hi.to_bits()))
        .collect()
}

/// Box-grouped decomposition of a per-clause-box disjunction: solve each
/// DISTINCT input box as one independent multi-clause disjunctive sub-problem
/// (its clauses over its box, boxes cleared), recursing into
/// [`verify_multi_clause_disjunction`] so the sub-problem routes through the
/// grouped unified input-split lane (one shared BaB tree amortizing all of the
/// box's objectives). Recursion terminates because the sub-spec has NO
/// per-clause boxes (this gate requires one).
///
/// Returns `Ok(None)` (decline — caller falls through to the legacy serial
/// per-clause loop) unless the spec is a disjunction whose clauses group into
/// `1..=MAX_BOX_GROUPS` distinct boxes.
///
/// SOUNDNESS / aggregation exactness: the property's unsafe region is
/// `∃k: x ∈ box_k ∧ clause_k(y)`. Grouping preserves it exactly — the group of
/// box B carries ALL clauses whose box is B, each sub-problem verifies its
/// disjunction over B tightened with directed rounding (`tighten_input_to_box`
/// encloses the real clause domain), and the verdict is Verified only when
/// EVERY group (and every pre-verified clause) is discharged. A group
/// counterexample satisfies `box_k ∧ clause_k` and therefore the full
/// property; it is still re-confirmed downstream (confirm_potential_violation
/// + the trusted-ORT vnncomp gate) before any `sat` is scored.
// Justification: forwards the full verification context of the caller plus the
// precheck results and budget ledger.
#[allow(clippy::too_many_arguments)]
fn try_box_grouped_disjunctive(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    pre_verified: &[bool],
    clauses: &[Vec<OutputConstraint>],
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    use_relu_split: bool,
    gpu_bab: bool,
    pgd_restarts: usize,
    pgd_steps: usize,
    timeout: u64,
    gemm_engine: Option<&dyn GemmEngine>,
    attack_engine_source: AttackEngineSource<'_>,
    json: bool,
    ledger: &PhaseBudgetLedger,
    overall_start: Instant,
) -> Result<Option<BetaCrownResult>> {
    if !vnnlib.is_disjunction
        || vnnlib.per_clause_input_bounds.len() != clauses.len()
        || pre_verified.len() != clauses.len()
        || !vnnlib
            .per_clause_input_bounds
            .iter()
            .any(|bounds| !bounds.is_empty())
    {
        return Ok(None);
    }

    // Group UNRESOLVED clause indices by bit-identical box (order-preserving).
    // A clause with an EMPTY box map ranges over the global input box; it forms
    // its own (untightened) group, keeping the union semantics exact.
    let mut groups: Vec<(Vec<(usize, u64, u64)>, Vec<usize>)> = Vec::new();
    let mut unresolved = 0usize;
    for (idx, bounds) in vnnlib.per_clause_input_bounds.iter().enumerate() {
        if pre_verified[idx] {
            continue;
        }
        unresolved += 1;
        let key = clause_box_key(bounds);
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, indices)) => indices.push(idx),
            None => groups.push((key, vec![idx])),
        }
    }
    // Fire only when grouping actually amortizes (strictly more clauses than
    // boxes). One-clause-per-box shapes (nn4sys mscn `_dual` singletons) keep
    // their measured route: box-refinement screen + serial per-clause loop.
    if groups.is_empty() || groups.len() > MAX_BOX_GROUPS || unresolved <= groups.len() {
        return Ok(None);
    }

    info!(
        groups = groups.len(),
        clauses = clauses.len(),
        pre_verified = pre_verified.iter().filter(|&&v| v).count(),
        "Box-grouped disjunctive decomposition: one grouped sub-problem per distinct clause box"
    );

    let mut aggregated = BetaCrownResult {
        result: BabVerificationStatus::Verified,
        domains_explored: 0,
        time_elapsed: Duration::from_secs(0),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    };
    let mut saw_potential = false;
    let mut unknown_reason: Option<String> = None;

    // DFS work stack of RE-ROOTED sub-problems: (box map, clause indices,
    // re-root depth). When a node's grouped BaB comes back undecided
    // (timeout / domain cap), its box is bisected on the widest effective axis
    // and both halves are re-solved with FRESH ROOTS. Measured on
    // ACASXU_run2a_1_1/prop_6 box 2: the engine's own tree reuses ROOT
    // intermediate bounds, so 100k in-tree domains fail to close
    // what a fresh-rooted half-box discharges in ~1.8k domains. Bisection
    // children exactly cover the parent (`[lo, mid] ∪ [mid, hi]`), so the
    // partition preserves the unsafe-region union and Verified still means
    // EVERY leaf of every group's box partition is discharged.
    let mut stack: Vec<(
        std::collections::BTreeMap<usize, (f64, f64)>,
        Vec<usize>,
        u16,
    )> = groups
        .iter()
        .rev()
        .map(|(_, clause_indices)| {
            (
                vnnlib.per_clause_input_bounds[clause_indices[0]].clone(),
                clause_indices.clone(),
                0u16,
            )
        })
        .collect();
    let mut nodes_processed = 0usize;

    while let Some((clause_bounds, clause_indices, depth)) = stack.pop() {
        let remaining = ledger.remaining_for_engine();
        if timeout > 0 && remaining.is_zero() {
            unknown_reason
                .get_or_insert_with(|| "Timeout before box sub-problem verification".to_string());
            break;
        }
        nodes_processed += 1;
        // Adaptive budget split: remaining time / open sub-problems (>= 1s
        // each, mirroring the serial per-clause loop's slice arithmetic).
        let group_timeout = if timeout == 0 {
            0
        } else {
            let open = (stack.len() + 1) as u64;
            (remaining.as_secs().max(1) / open).max(1)
        };

        // Sub-spec: this box's clauses over the box, per-clause boxes CLEARED
        // (every clause in the group shares the box, now enforced by the
        // tightened input tensor), disjunctive aggregation preserved.
        let mut sub_spec = vnnlib.clone();
        sub_spec.output_constraint_clauses = clause_indices
            .iter()
            .map(|&idx| clauses[idx].clone())
            .collect();
        sub_spec.output_constraints = sub_spec
            .output_constraint_clauses
            .iter()
            .flatten()
            .cloned()
            .collect();
        sub_spec.is_disjunction = true;
        sub_spec.per_clause_input_bounds = Vec::new();

        let group_input = if clause_bounds.is_empty() {
            input.clone()
        } else {
            // Mirror the box restriction into the sub-spec's own input bounds
            // (attack-box construction and witness gating read them).
            for (&var_idx, &(lb, ub)) in &clause_bounds {
                if var_idx < sub_spec.input_bounds.len() {
                    let global = sub_spec.input_bounds[var_idx];
                    sub_spec.input_bounds[var_idx] = (global.0.max(lb), global.1.min(ub));
                }
            }
            super::disjunctive_precheck::tighten_input_to_box(input, &clause_bounds)
        };

        // Fail-fast domain cap while re-rooting is still available: an
        // undecided verdict at REROOT_DOMAIN_CAP arrives in seconds and the
        // fresh-rooted halves converge orders of magnitude faster than more
        // in-tree domains would (root-bound staleness). The final depth gets
        // the caller's full cap as a last chance. SOUND: a smaller domain cap
        // can only turn would-be-verified into unknown, never the reverse.
        let can_reroot =
            depth < MAX_REROOT_DEPTH && nodes_processed + stack.len() < MAX_REROOT_NODES;
        let node_config = if can_reroot && config.max_domains > REROOT_DOMAIN_CAP {
            let mut c = config.clone();
            c.max_domains = REROOT_DOMAIN_CAP;
            std::borrow::Cow::Owned(c)
        } else {
            std::borrow::Cow::Borrowed(config)
        };

        debug!(
            depth,
            open = stack.len() + 1,
            group_clauses = clause_indices.len(),
            timeout_s = group_timeout,
            elapsed_s = overall_start.elapsed().as_secs_f64(),
            "Box-grouped disjunctive sub-problem"
        );

        // Recurse: the sub-spec has no per-clause boxes, so this call routes to
        // the grouped unified input-split lane (or the serial per-clause loop
        // for shapes the unified gate declines). The global disjunctive PGD
        // already ran over the full hull, so per-group attack stays off.
        let group_ledger = ledger.child_with_timeout_secs(group_timeout);
        let group_result = verify_multi_clause_disjunction(
            model_net,
            &group_input,
            &sub_spec,
            &node_config,
            verifier,
            use_relu_split,
            gpu_bab,
            false, // pgd_attack: the global attack phase already ran
            pgd_restarts,
            pgd_steps,
            group_timeout,
            gemm_engine,
            attack_engine_source,
            json,
            &group_ledger,
        )?;

        aggregated.domains_explored += group_result.domains_explored;
        aggregated.domains_verified += group_result.domains_verified;
        aggregated.cuts_generated += group_result.cuts_generated;
        aggregated.max_depth_reached = aggregated
            .max_depth_reached
            .max(group_result.max_depth_reached);

        match group_result.result {
            BabVerificationStatus::Verified => {}
            violation @ BabVerificationStatus::Violated { .. } => {
                return Ok(Some(finalize_disjunctive_result(
                    aggregated,
                    overall_start,
                    violation,
                )));
            }
            BabVerificationStatus::PotentialViolation { .. } => {
                saw_potential = true;
            }
            BabVerificationStatus::Unknown { .. } | BabVerificationStatus::Timeout => {
                let reason = match &group_result.result {
                    BabVerificationStatus::Unknown { reason } => reason.clone(),
                    _ => "timeout".to_string(),
                };
                // Re-root: bisect this box on its widest effective axis and
                // re-solve both halves with fresh roots (budget permitting).
                let bisected = can_reroot
                    && (timeout == 0 || remaining.as_secs() >= 2)
                    && match bisect_effective_box(&group_input, &clause_bounds) {
                        Some((left, right)) => {
                            stack.push((right, clause_indices.clone(), depth + 1));
                            stack.push((left, clause_indices, depth + 1));
                            true
                        }
                        None => false,
                    };
                if !bisected {
                    unknown_reason.get_or_insert_with(|| {
                        format!("Box sub-problem (depth {}): {}", depth, reason)
                    });
                }
            }
        }
    }

    let final_status = if saw_potential {
        BabVerificationStatus::potential_violation()
    } else if let Some(reason) = unknown_reason {
        BabVerificationStatus::Unknown { reason }
    } else {
        BabVerificationStatus::Verified
    };

    Ok(Some(finalize_disjunctive_result(
        aggregated,
        overall_start,
        final_status,
    )))
}

/// Maximum re-root bisection depth per box group (leaf width = axis width /
/// 2^16 in the worst case; the measured acasxu prop_6 hard slivers verify
/// within a handful of levels).
const MAX_REROOT_DEPTH: u16 = 16;

/// Hard cap on re-rooted sub-problems per property (bounds the fixed
/// per-sub-problem root-bound cost).
const MAX_REROOT_NODES: usize = 64;

/// Fail-fast domain cap for re-root-eligible sub-problems: verified boxes on
/// the measured acasxu prop_6 profile need <= ~5k domains; a box still open at
/// this many in-tree domains converges faster after a fresh-rooted bisection
/// than by exploring further against stale root intermediate bounds.
const REROOT_DOMAIN_CAP: usize = 25_000;

/// Bisect the EFFECTIVE box of a re-rooted sub-problem on its widest axis.
///
/// The effective box is the tightened input tensor (clause box ∩ global box);
/// the returned halves are expressed as clause-box maps whose split axis is
/// overridden with `[lo, mid]` / `[mid, hi]` (f64 midpoint of the effective
/// f32 bounds). The halves exactly cover the parent (mid shared — closed
/// intervals over-cover, which is sound). Returns `None` when no axis is
/// splittable (degenerate box), so the caller keeps the parent's undecided
/// verdict instead of looping.
fn bisect_effective_box(
    group_input: &BoundedTensor,
    clause_bounds: &std::collections::BTreeMap<usize, (f64, f64)>,
) -> Option<(
    std::collections::BTreeMap<usize, (f64, f64)>,
    std::collections::BTreeMap<usize, (f64, f64)>,
)> {
    let lower = group_input.lower();
    let upper = group_input.upper();
    let (lo, hi) = (lower.as_slice()?, upper.as_slice()?);
    let mut best: Option<(usize, f64, f64, f64)> = None;
    for (idx, (&l, &u)) in lo.iter().zip(hi.iter()).enumerate() {
        let (l, u) = (f64::from(l), f64::from(u));
        let width = u - l;
        if width.is_finite() && best.as_ref().is_none_or(|(_, _, _, w)| width > *w) {
            best = Some((idx, l, u, width));
        }
    }
    let (axis, l, u, _) = best?;
    let mid = l + (u - l) / 2.0;
    if !(mid > l && mid < u) {
        return None; // degenerate: no representable split point
    }
    let mut left = clause_bounds.clone();
    let mut right = clause_bounds.clone();
    left.insert(axis, (l, mid));
    right.insert(axis, (mid, u));
    Some((left, right))
}

#[cfg(test)]
mod clause_verdict_tests {
    use super::*;

    /// THE soundness pin for `bab.disjunctive_clause_retry_passes`: a retry
    /// pass must not be able to erase a potential violation and publish
    /// `Verified` (which the caller scores as `unsat`).
    #[test]
    fn a_retry_pass_can_never_erase_a_potential_violation() {
        let mut verdicts = ClauseVerdicts::new(2);

        // Pass 0: clause 0 signals a potential violation, clause 1 is unknown.
        verdicts.begin_visit(0);
        verdicts.record_potential(0);
        verdicts.begin_visit(1);
        verdicts.record_unknown(1, "Clause 2: timeout".to_string());
        assert!(matches!(
            verdicts.final_status(),
            BabVerificationStatus::PotentialViolation { .. }
        ));
        assert!(verdicts.is_terminal(0), "a signalled clause is terminal");
        assert!(!verdicts.is_terminal(1));

        // Pass 1 revisits clause 0 (it must not, but even if the pending filter
        // were wrong the flag survives) and this time proves nothing at all.
        verdicts.begin_visit(0);
        assert!(
            verdicts.is_terminal(0),
            "begin_visit must not clear the potential-violation flag"
        );
        assert!(
            matches!(
                verdicts.final_status(),
                BabVerificationStatus::PotentialViolation { .. }
            ),
            "a revisit that records nothing must not upgrade the aggregate"
        );

        // And a pass that VERIFIES the same clause still cannot upgrade it:
        // `Verified` records nothing, so the sticky flag decides.
        verdicts.begin_visit(0);
        verdicts.begin_visit(1);
        assert!(matches!(
            verdicts.final_status(),
            BabVerificationStatus::PotentialViolation { .. }
        ));
    }

    /// The clean-sweep and unknown-precedence arms are HEAD's, unchanged.
    #[test]
    fn aggregate_precedence_matches_head() {
        let mut verdicts = ClauseVerdicts::new(3);
        assert!(matches!(
            verdicts.final_status(),
            BabVerificationStatus::Verified
        ));

        verdicts.record_unknown(2, "Clause 3: timeout".to_string());
        let BabVerificationStatus::Unknown { reason } = verdicts.final_status() else {
            panic!("an unknown clause must make the aggregate Unknown");
        };
        assert_eq!(reason, "Clause 3: timeout");

        verdicts.record_potential(0);
        assert!(
            matches!(
                verdicts.final_status(),
                BabVerificationStatus::PotentialViolation { .. }
            ),
            "a potential violation outranks an unknown"
        );
        assert_eq!(verdicts.potential_count(), 1);
    }

    /// A clause visited on more than one pass contributes its counters ONCE.
    #[test]
    fn clause_counters_fold_once_across_passes() {
        let mut aggregated = BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored: 0,
            domains_verified: 0,
            cuts_generated: 0,
            max_depth_reached: 0,
            time_elapsed: Duration::ZERO,
            output_bounds: None,
        };
        let mut stats: Vec<Option<ClauseStats>> = vec![None; 2];
        // Clause 0 visited twice (pass 0 then pass 1); clause 1 once.
        stats[0] = Some(ClauseStats {
            domains_explored: 7,
            domains_verified: 1,
            cuts_generated: 0,
            time_elapsed: Duration::from_secs(3),
        });
        stats[0] = Some(ClauseStats {
            domains_explored: 40,
            domains_verified: 2,
            cuts_generated: 1,
            time_elapsed: Duration::from_secs(11),
        });
        stats[1] = Some(ClauseStats {
            domains_explored: 5,
            domains_verified: 1,
            cuts_generated: 0,
            time_elapsed: Duration::from_secs(2),
        });
        fold_clause_stats(&mut aggregated, &stats);
        assert_eq!(
            aggregated.domains_explored, 45,
            "last visit only, not 7+40+5"
        );
        assert_eq!(aggregated.domains_verified, 3);
        assert_eq!(aggregated.cuts_generated, 1);
        assert_eq!(aggregated.time_elapsed, Duration::from_secs(13));
    }
}
