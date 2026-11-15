// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #hull-root (wave-6 C4, phase 1) — the sequential-lane adapter for the
//! certified single-hidden-layer per-neuron hull root bound
//! (`ny_mip::single_hidden_hull`; diagnosis `scratchpad/vbs/diag6/safenlp.md`).
//!
//! # Why
//!
//! On safenlp the sequential relational lane's input-split BaB owns the whole
//! deadline and cannot close the 115 bank-timeout rows: its planet root sits a
//! median -3.93 below zero, while the per-neuron IDEAL hull root is `> 0` on
//! 30/115 rows with one ~85 ms LP (MEASURED offline, float). This adapter runs
//! that hull once, at the root, after the upfront attack and before the
//! same-LHS reduction, and publishes `Verified` only on a strictly positive
//! CERTIFIED value.
//!
//! # Gate (every condition fails closed to "declined", the historical lane)
//!
//! * the typed key `bab.single_hidden_hull` resolves to `root` or `bab`
//!   (default `off`: this module is never entered and prints nothing);
//! * a conjunctive, single-network, global-box spec with exactly ONE
//!   relational unsafe row (`Y_i <= Y_j`, `<`, `>=`, `>`);
//! * the network canonicalizes — through the SAME authority the MIP encoder
//!   uses, [`canonicalize_mip_feedforward_network`] (Conv unfold, shape strip,
//!   exact constant fold) — to exactly `Linear -> ReLU -> Linear`;
//! * dimensions agree with the input box and the spec.
//!
//! # Soundness
//!
//! Weights are the canonical network's f32 values (exact in f64); biases are
//! the fold's EXACT f64 sidecar (`FoldedMipNetwork::exact_biases`), never the
//! rounded structural copy; the box is the lane's own outward-rounded input
//! tensor. The certificate itself is `ny_mip::single_hidden_hull`'s closed
//! form, valid for ARBITRARY multipliers — the AY float LP only proposes them.
//! A strictly positive certified lower bound on `Y_i - Y_j` (resp.
//! `Y_j - Y_i`) over the box proves the non-strict AND the strict unsafe row
//! empty. Nothing here can publish `sat` or override a counterexample: the
//! upfront attack has already run, and a declined/undecided root simply
//! continues into the unchanged historical lane.
//!
//! # Observable
//!
//! Printed UNCONDITIONALLY on every run where the key is armed, with a fixed
//! schema (every field always present, `na` when not computed), so "never
//! engaged" is distinguishable from "engaged and undecided":
//!
//! ```text
//! [hull-root] armed mode=root engaged=<0|1> verdict=<verified|undecided|declined> reason=<ok|...>
//!   unstable=<U|na> neg=<N|na> pos=<P|na> lp_lb=<v|na> cert_lb=<v|na>
//!   cert_route=<lp|subgradient|triangle|none> lp_status=<...> sg_iters=<k> t_ms=<ms>
//! ```
//!
//! (one line). The same payload is recorded as a `hull_root` flight note so it
//! survives `ny benchmarks run`, which discards stderr.
//!
//! # Phase 2: `bab` (#hull-bab)
//!
//! With the key at `bab`, the same gate admits the row and
//! `ny_mip::single_hidden_hull::bab::hull_bab` runs a deadline-bounded,
//! best-first ReLU-split tree over the unstable concave (`c_j < 0`) neurons,
//! every node certified by the closed form plus one weak-duality term per
//! split halfspace. `Verified` is published ONLY when the frontier emptied
//! (every leaf certified `> 0`); every other stop (deadline, node cap, an
//! exhausted leaf, a worker failure) falls through to the historical lane
//! with the wall the tree did not spend. The tree owns
//! `min(0.6 * remaining, 60 s)`. Its observable, printed unconditionally on
//! every `bab` run with a fixed schema, is
//!
//! ```text
//! [hull-bab] armed mode=bab engaged=<0|1> verdict=<verified|undecided|declined>
//!   reason=<ok|...> stop=<proved|deadline|node_cap|exhausted|worker_failure|na>
//!   unstable= neg= pos= root_lb= root_route=<lp|triangle|none> root_lp_lb=
//!   root_lp_status= nodes= closed= open= worst= depth= lp_fail= workers= t_ms=
//! ```
//!
//! (one line), mirrored into a `hull_bab` flight note.

use std::time::{Duration, Instant};

use ny_mip::single_hidden_hull::bab::{
    hull_bab, HullBabOptions, HullBabReport, HULL_BAB_MAX_WORKERS,
};
use ny_mip::single_hidden_hull::{
    hull_root_bound_for_goal, HullBoundGoal, HullMarginProblem, HullRootReport,
};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{BabVerificationStatus, BetaCrownResult, Layer, Network, SingleHiddenHullMode};
use ny_tensor::BoundedTensor;

use super::mip_highs::canonicalize_mip_feedforward_network;
use super::mip_single_hidden::is_single_hidden_linear_relu_linear;

/// Share of the remaining wall the hull root may spend (LP + fallback).
pub(super) const HULL_ROOT_MAX_WALL_SHARE: f64 = 0.25;
/// Absolute cap on the hull root's slice, in seconds.
pub(super) const HULL_ROOT_MAX_SECS: f64 = 4.0;

/// The hull root's absolute deadline: `now + min(share * remaining, cap)`
/// (`cap` alone when the ledger is unbounded). Whatever the hull does not
/// spend stays with the reduction, which re-derives its budget from the
/// ledger's absolute deadline.
pub(super) fn hull_root_deadline(now: Instant, remaining: Option<Duration>) -> Instant {
    let cap = Duration::from_secs_f64(HULL_ROOT_MAX_SECS);
    let slice = remaining.map_or(cap, |left| left.mul_f64(HULL_ROOT_MAX_WALL_SHARE).min(cap));
    now.checked_add(slice).unwrap_or(now)
}

/// Share of the remaining wall the hull TREE may spend (#hull-bab). The rest
/// stays with the historical input-split lane (safenlp.md Plan 2: "fall back
/// to input-split after a 60% slice").
pub(super) const HULL_BAB_MAX_WALL_SHARE: f64 = 0.6;
/// Absolute cap on the hull tree's slice, in seconds.
pub(super) const HULL_BAB_MAX_SECS: f64 = 60.0;

/// The hull tree's absolute deadline: `now + min(0.6 * remaining, 60 s)`
/// (`60 s` alone when the ledger is unbounded).
pub(super) fn hull_bab_deadline(now: Instant, remaining: Option<Duration>) -> Instant {
    let cap = Duration::from_secs_f64(HULL_BAB_MAX_SECS);
    let slice = remaining.map_or(cap, |left| left.mul_f64(HULL_BAB_MAX_WALL_SHARE).min(cap));
    now.checked_add(slice).unwrap_or(now)
}

/// Node workers for the hull tree: the host's parallelism, capped.
fn hull_bab_workers() -> usize {
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .clamp(1, HULL_BAB_MAX_WORKERS)
}

/// The margin objective `(output, coefficient)` whose strict positivity over
/// the box excludes the ONE relational unsafe row, or `None`.
///
/// Unsafe `Y_i <= Y_j` (or `<`) is empty iff `Y_i - Y_j > 0` everywhere;
/// unsafe `Y_i >= Y_j` (or `>`) iff `Y_j - Y_i > 0`. Strict and non-strict
/// rows share the same sufficient strict-positive complement (as in
/// `certified_shared_tree_pairwise_indices`).
fn single_relational_margin(vnnlib: &VnnLibSpec) -> Option<[(usize, f64); 2]> {
    let [constraint] = vnnlib.output_constraints.as_slice() else {
        return None;
    };
    match vnnlib.output_constraint_clauses.as_slice() {
        [] => {}
        [clause] if clause.as_slice() == std::slice::from_ref(constraint) => {}
        _ => return None,
    }
    let (positive, negative) = match constraint {
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => (*i, *j),
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => (*j, *i),
        _ => return None,
    };
    if positive == negative || positive >= vnnlib.num_outputs || negative >= vnnlib.num_outputs {
        return None;
    }
    Some([(positive, 1.0), (negative, -1.0)])
}

/// The admission gate. `Err` carries the stable decline reason printed by the
/// observable.
fn admit(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
) -> Result<HullMarginProblem, &'static str> {
    if vnnlib.is_disjunction {
        return Err("disjunctive_spec");
    }
    if vnnlib.dual_network.is_some() {
        return Err("dual_network_spec");
    }
    if vnnlib
        .per_clause_input_bounds
        .iter()
        .any(|clause| !clause.is_empty())
    {
        return Err("per_clause_input_box");
    }
    let objective = single_relational_margin(vnnlib).ok_or("not_one_relational_row")?;
    let folded = canonicalize_mip_feedforward_network(network, input.shape())
        .map_err(|_| "canonicalize_refused")?;
    if !is_single_hidden_linear_relu_linear(&folded) {
        return Err("not_single_hidden");
    }
    let [Layer::Linear(first), Layer::ReLU(_), Layer::Linear(last)] = folded.layers() else {
        return Err("not_single_hidden");
    };
    let [first_bias, last_bias] = folded.exact_biases() else {
        return Err("bias_sidecar_mismatch");
    };
    let (hidden_dim, input_dim) = first.weight().dim();
    let (output_dim, last_input_dim) = last.weight().dim();
    if last_input_dim != hidden_dim
        || first_bias.len() != hidden_dim
        || last_bias.len() != output_dim
    {
        return Err("shape_mismatch");
    }
    if input.lower().len() != input_dim
        || vnnlib.num_inputs != input_dim
        || vnnlib.num_outputs != output_dim
    {
        return Err("dimension_mismatch");
    }
    let w1: Vec<f64> = first.weight().iter().map(|&v| f64::from(v)).collect();
    let w2: Vec<f64> = last.weight().iter().map(|&v| f64::from(v)).collect();
    let lower: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
    let upper: Vec<f64> = input.upper().iter().map(|&v| f64::from(v)).collect();
    HullMarginProblem::new(
        w1,
        first_bias.clone(),
        input_dim,
        &w2,
        last_bias,
        output_dim,
        &objective,
        0.0,
        lower,
        upper,
    )
    .map_err(|decline| decline.label())
}

fn fmt_value(value: Option<f64>) -> String {
    value.map_or_else(|| "na".to_string(), |v| format!("{v:.6e}"))
}

fn fmt_count(value: Option<usize>) -> String {
    value.map_or_else(|| "na".to_string(), |v| v.to_string())
}

/// The fixed-schema observable payload (everything after `[hull-root] `).
fn observable_payload(
    mode: SingleHiddenHullMode,
    report: Option<&HullRootReport>,
    reason: &str,
    elapsed: Duration,
) -> String {
    let verdict = match report {
        None => "declined",
        Some(r) if r.verified() => "verified",
        Some(_) => "undecided",
    };
    format!(
        "armed mode={} engaged={} verdict={verdict} reason={reason} unstable={} neg={} pos={} \
         lp_lb={} cert_lb={} cert_route={} lp_status={} sg_iters={} t_ms={}",
        mode.label(),
        u8::from(report.is_some()),
        fmt_count(report.map(|r| r.unstable)),
        fmt_count(report.map(|r| r.neg_unstable)),
        fmt_count(report.map(|r| r.pos_unstable)),
        fmt_value(report.and_then(|r| r.lp_value)),
        fmt_value(report.and_then(|r| r.certified)),
        report
            .and_then(|r| r.route)
            .map_or("none", |route| route.label()),
        report.map_or("na", |r| r.lp_status.label()),
        report.map_or(0, |r| r.subgradient_iterations),
        elapsed.as_millis(),
    )
}

/// Run the hull root when armed. `Some(Verified)` only on a strictly positive
/// certified margin lower bound; `None` in every other case (off, declined,
/// undecided, deadline), leaving the historical lane untouched.
pub(super) fn try_single_hidden_hull_root(
    mode: SingleHiddenHullMode,
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    deadline: Instant,
    start_time: Instant,
) -> Option<BetaCrownResult> {
    if !mode.is_armed() {
        return None;
    }
    let began = Instant::now();
    let (report, reason) = match admit(network, input, vnnlib) {
        Err(reason) => (None, reason),
        Ok(problem) => (
            Some(hull_root_bound_for_goal(
                &problem,
                deadline,
                HullBoundGoal::StrictPositive,
            )),
            "ok",
        ),
    };
    let payload = observable_payload(mode, report.as_ref(), reason, began.elapsed());
    eprintln!("[hull-root] {payload}");
    let status = if report.is_some() {
        crate::flight::FlightStatus::Ran
    } else {
        crate::flight::FlightStatus::Skipped
    };
    crate::flight::note("hull_root", status, Some(payload));
    report
        .filter(|report| {
            if !report.verified() || ny_core::require_f64_interval_proof_environment().is_err() {
                return false;
            }
            let publication_deadline = ny_core::instance_budget::deadline()
                .map_or(deadline, |instance| instance.min(deadline));
            Instant::now() < publication_deadline
        })
        .map(|_| BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored: 1,
            domains_verified: 1,
            cuts_generated: 0,
            max_depth_reached: 0,
            time_elapsed: start_time.elapsed(),
            output_bounds: None,
        })
}

/// The fixed-schema `[hull-bab]` observable payload (everything after
/// `[hull-bab] `). Every field is always present (`na` when not computed).
fn bab_observable_payload(
    report: Option<&HullBabReport>,
    reason: &str,
    workers: usize,
    elapsed: Duration,
) -> String {
    let verdict = match report {
        None => "declined",
        Some(r) if r.verified() => "verified",
        Some(_) => "undecided",
    };
    format!(
        "armed mode=bab engaged={} verdict={verdict} reason={reason} stop={} unstable={} \
         neg={} pos={} root_lb={} root_route={} root_lp_lb={} root_lp_status={} nodes={} \
         closed={} open={} worst={} depth={} lp_fail={} workers={workers} t_ms={}",
        u8::from(report.is_some()),
        report.map_or("na", |r| r.stop.label()),
        fmt_count(report.map(|r| r.unstable)),
        fmt_count(report.map(|r| r.neg_unstable)),
        fmt_count(report.map(|r| r.pos_unstable)),
        fmt_value(report.and_then(|r| r.root_certified)),
        report
            .and_then(|r| r.root_route)
            .map_or("none", |route| route.label()),
        fmt_value(report.and_then(|r| r.root_lp_value)),
        report.map_or("na", |r| r.root_lp_status.label()),
        fmt_count(report.map(|r| r.nodes)),
        fmt_count(report.map(|r| r.closed)),
        fmt_count(report.map(|r| r.open)),
        fmt_value(report.and_then(|r| r.worst)),
        fmt_count(report.map(|r| r.max_depth)),
        fmt_count(report.map(|r| r.lp_failures)),
        elapsed.as_millis(),
    )
}

/// Run the hull TREE (#hull-bab). `Some(Verified)` only when every leaf was
/// certified `> 0`; `None` in every other case (declined, deadline, node cap,
/// exhausted leaf, worker failure), leaving the historical lane untouched.
/// Never publishes `sat`.
pub(super) fn try_single_hidden_hull_bab(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    deadline: Instant,
    start_time: Instant,
    workers: usize,
) -> Option<BetaCrownResult> {
    let began = Instant::now();
    let workers = workers.clamp(1, HULL_BAB_MAX_WORKERS);
    let (report, reason) = match admit(network, input, vnnlib) {
        Err(reason) => (None, reason),
        Ok(problem) => (
            Some(hull_bab(
                &problem,
                deadline,
                HullBabOptions {
                    workers,
                    ..HullBabOptions::default()
                },
            )),
            "ok",
        ),
    };
    let payload = bab_observable_payload(report.as_ref(), reason, workers, began.elapsed());
    eprintln!("[hull-bab] {payload}");
    let status = if report.is_some() {
        crate::flight::FlightStatus::Ran
    } else {
        crate::flight::FlightStatus::Skipped
    };
    crate::flight::note("hull_bab", status, Some(payload));
    report
        .filter(HullBabReport::verified)
        .map(|report| BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored: report.nodes,
            domains_verified: report.closed,
            cuts_generated: 0,
            max_depth_reached: report.max_depth,
            time_elapsed: start_time.elapsed(),
            output_bounds: None,
        })
}

/// The sequential lane's single entry point for the typed mode: `off` does
/// nothing (no work, no line), `root` runs the phase-1 root bound on its
/// `min(0.25 * remaining, 4 s)` slice exactly as before, `bab` runs the tree
/// on its `min(0.6 * remaining, 60 s)` slice. `remaining` is the ledger's
/// remaining wall at the call.
pub(super) fn try_single_hidden_hull(
    mode: SingleHiddenHullMode,
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    remaining: Option<Duration>,
    start_time: Instant,
) -> Option<BetaCrownResult> {
    match mode {
        SingleHiddenHullMode::Off => None,
        SingleHiddenHullMode::Root => try_single_hidden_hull_root(
            mode,
            network,
            input,
            vnnlib,
            hull_root_deadline(Instant::now(), remaining),
            start_time,
        ),
        SingleHiddenHullMode::Bab => try_single_hidden_hull_bab(
            network,
            input,
            vnnlib,
            hull_bab_deadline(Instant::now(), remaining),
            start_time,
            hull_bab_workers(),
        ),
    }
}

#[cfg(test)]
#[path = "single_hidden_hull_tests.rs"]
mod tests;

// Explicit existing-ledger bridge; does not change the typed default modes.
#[path = "single_hidden_hull_resource.rs"]
pub(super) mod resource;
