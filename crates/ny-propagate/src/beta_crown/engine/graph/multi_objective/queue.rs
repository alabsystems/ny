// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Queue helpers for multi-objective graph BaB verification.
//!
//! Extracts batch pop, prefilter, and batched-result folding so `verify.rs`
//! only coordinates the existing root/sequential/batched services.

use std::collections::BinaryHeap;
use std::time::Instant;

use ny_core::Result;
use tracing::{debug, info, warn};

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::conflict_clauses_graph::GraphClauseStore;
use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
use crate::beta_crown::engine::domain_results::MultiObjectiveGraphDomainResult;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::graph_mip_leaf::{
    GraphMipLeafOracle, GraphMipLeafRequest, GraphMipLeafVerdict, LeafSplit,
};
use crate::beta_crown::result::BabVerificationStatus;
use crate::GraphNetwork;

use super::shared::violation_drop_is_certified;

/// Everything the UNDECIDED-child requeue needs to consult the Graph-MIP leaf
/// oracle (increment 6, `docs/GRAPH_MIP_LEAF_SOLVER.md`). A `None` oracle
/// keeps the requeue byte-identical (`NY_GRAPH_MIP_LEAF=0` at the CLI).
pub(super) struct LeafOracleCtx<'a> {
    pub(super) oracle: &'a dyn GraphMipLeafOracle,
    pub(super) graph: &'a GraphNetwork,
    pub(super) objectives: &'a [Vec<f32>],
    pub(super) thresholds: &'a [f32],
    pub(super) deadline: Option<Instant>,
}

/// Consult the leaf oracle for an UNDECIDED child that is about to be
/// requeued. Builds the request from the child's split history (the premises)
/// and its UNVERIFIED objective rows. Returns `Undecided` when no oracle is
/// attached or there is nothing to solve — the caller then pushes the child
/// exactly as before (strictly additive; see the trait's soundness contract).
pub(super) fn consult_leaf_oracle(
    ctx: Option<&LeafOracleCtx<'_>>,
    child: &MultiObjectiveGraphBabDomain,
) -> GraphMipLeafVerdict {
    consult_leaf_oracle_with_clock(ctx, child, Instant::now)
}

fn consult_leaf_oracle_with_clock<N>(
    ctx: Option<&LeafOracleCtx<'_>>,
    child: &MultiObjectiveGraphBabDomain,
    mut now: N,
) -> GraphMipLeafVerdict
where
    N: FnMut() -> Instant,
{
    let Some(ctx) = ctx else {
        return GraphMipLeafVerdict::Undecided;
    };
    if ctx.deadline.is_some_and(|deadline| now() >= deadline) {
        debug!(
            depth = child.depth,
            "Graph-MIP leaf oracle deadline expired before the call; result declined"
        );
        return GraphMipLeafVerdict::Undecided;
    }
    // The premises: one LeafSplit per applied ReLU constraint.
    let splits: Vec<LeafSplit> = child
        .history
        .constraints
        .iter()
        .map(|c| LeafSplit {
            relu_node: c.node_name.clone(),
            neuron_idx: c.neuron_idx,
            is_active: c.is_active,
        })
        .collect();
    // Only the still-unverified rows need solving (a verified row's bound
    // already cleared its threshold in this subdomain).
    let rows: Vec<(Vec<f32>, f32)> = ctx
        .objectives
        .iter()
        .zip(ctx.thresholds.iter())
        .zip(child.verified.iter())
        .filter(|(_, verified)| !**verified)
        .map(|((obj, thr), _)| (obj.clone(), *thr))
        .collect();
    if rows.is_empty() {
        // Defensive: an all-verified child never reaches the requeue arm.
        return GraphMipLeafVerdict::Undecided;
    }
    // Preserve the established public oracle request without exposing the
    // tracked table. This leaf-only compatibility copy shares every tensor
    // allocation through Arc and lives exactly through the synchronous call.
    let leaf_node_bounds = child.node_bounds.to_shared_hash_map();
    let req = GraphMipLeafRequest {
        graph: ctx.graph,
        input_bounds: &child.input_bounds,
        node_bounds: &leaf_node_bounds,
        splits,
        rows,
        depth: child.depth,
        deadline: ctx.deadline,
    };
    let verdict = ctx.oracle.solve_leaf(&req);
    // The oracle owns an internal budget, but the caller retains final
    // authority over the enclosing BaB deadline. Revoke even a sound verdict
    // that arrives at/after that deadline, exactly like the input-split leaf
    // consumers do, so terminal tail work cannot turn a timeout into Verified.
    if ctx.deadline.is_some_and(|deadline| now() >= deadline) {
        debug!(
            depth = child.depth,
            "Graph-MIP leaf oracle completed at or after the BaB deadline; result declined"
        );
        GraphMipLeafVerdict::Undecided
    } else {
        verdict
    }
}

/// Master gate for the leaf oracle's SAT RETURN (#mip-leaf-witness).
///
/// DEFAULT-ON: a graph-forward-confirmed witness that violates every objective
/// row is published as the run's verdict candidate. `NY_GRAPH_MIP_LEAF_SAT=0`
/// restores the pre-change ADVISORY behaviour byte-for-byte (log loudly,
/// requeue the child, return no verdict) — the control arm for A/B measurement
/// and the kill switch if a category ever needs the old shape. Only the exact
/// string `"0"` disarms; unset, empty and malformed keep the default, matching
/// `NY_GRAPH_MIP_LEAF`'s own parse.
pub(super) fn leaf_sat_return_enabled() -> bool {
    ny_levers::read(&ny_levers::decls::graph_mip::GRAPH_MIP_LEAF_SAT)
        .value
        .as_bool()
}

/// Does the leaf oracle's graph-forward-confirmed witness violate EVERY
/// objective row at once?
///
/// SOUNDNESS — why every row, and not merely the row the oracle solved. The
/// leaf solver emits one decision MIP PER undecided row and returns on the
/// FIRST row whose MIP is SAT (`ny-cli .../graph_mip_leaf.rs`, the
/// `MipResult::Sat` arm), so its witness is revalidated against exactly ONE
/// row. What a single satisfied row buys depends on the property's CLAUSE
/// LAYOUT, and this lane cannot see that layout: the CLI's
/// `build_multi_objectives` FLATTENS `VnnLibSpec::output_constraint_clauses`
/// (each clause a conjunction; the unsafe region is the OR of the clauses)
/// into a bare `objectives`/`thresholds` pair, discarding the clause
/// boundaries. On an OR-of-AND spec one satisfied row is NOT a counterexample.
///
/// SCOPE — state this precisely, because a soundness comment that overstates
/// gets copied into the next change. Requiring every row to hold at the
/// concrete point is:
/// * EXACT for conjunctive specs (unsafe = AND of all rows): the conjunction
///   holding at the point IS the property's violation predicate;
/// * SUFFICIENT for disjunctions WITHOUT per-clause input boxes — OR of
///   single-row clauses (some row holds, so the OR holds) and OR of AND-clauses
///   over one shared input box (every clause's rows hold, so every clause
///   holds);
/// * NOT SUFFICIENT when the spec carries PER-CLAUSE INPUT BOXES
///   (`VnnLibSpec::per_clause_input_bounds`, each clause being
///   `(and <that clause's input box> <its output rows>)`). Every output row
///   holding at a point drawn from the HULL implies nothing about any clause,
///   because the point need not lie in any clause's own input box.
///
/// That last case is exactly why publication is ALSO gated on
/// [`GraphMipLeafOracle::may_publish_violation_witness`], which is fail-closed
/// by default and answered by the CLI, the only layer that can see the clause
/// layout. This predicate is the in-lane obligation; that flag is the
/// layout obligation; and `gate_sat_with_trusted_oracle`'s
/// `property_violated_f64` (the nn4sys 71-false-sat fix) remains the final
/// authority over both.
///
/// It is deliberately conservative on OR-of-single-row properties: a genuine
/// one-row counterexample stays advisory and the child keeps searching. That
/// can cost a `sat` we would otherwise have had to GUESS at; it cannot cost a
/// verdict.
///
/// Strictness is deliberately NOT decided here. The normalized objective rows
/// do not retain whether the authored atom was strict or non-strict, so equality
/// must remain a candidate (`<=` below): it is a real violation for a non-strict
/// atom and not one for a strict atom. Every public witness therefore still
/// passes the exact VNN-LIB publication gate, which preserves that distinction;
/// this row screen alone never authorizes a standalone SAT claim.
///
/// Re-evaluated in f64 from the oracle's graph-forward output. Any shape
/// mismatch or non-finite value fails CLOSED (no sat).
pub(super) fn witness_violates_every_objective_row(
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    output: &[f32],
) -> bool {
    if objectives.is_empty() || objectives.len() != thresholds.len() || output.is_empty() {
        return false;
    }
    if output.iter().any(|y| !y.is_finite()) {
        return false;
    }
    objectives
        .iter()
        .zip(thresholds.iter())
        .all(|(coeffs, &threshold)| {
            if coeffs.len() != output.len()
                || !threshold.is_finite()
                || coeffs.iter().any(|c| !c.is_finite())
            {
                return false;
            }
            let margin: f64 = coeffs
                .iter()
                .zip(output.iter())
                .map(|(c, y)| f64::from(*c) * f64::from(*y))
                .sum();
            margin.is_finite() && margin <= f64::from(threshold)
        })
}

/// Fold a leaf-oracle verdict into the queue/lifecycle. `Undecided` pushes the
/// child (the pre-oracle behavior); `VerifiedAllRows` counts it verified (and
/// feeds the verified-domain cut pool exactly like a BaB-verified child).
///
/// `Violated` carries a witness the oracle has ALREADY confirmed by re-running
/// the ORIGINAL graph forward at an in-box point (the contract at
/// `ny-cli .../graph_mip_leaf.rs::revalidate_leaf_witness`). When that witness
/// covers EVERY objective row ([`witness_violates_every_objective_row`]) it is
/// published into `leaf_violation` as a typed
/// [`BabVerificationStatus::Violated`] — the carrier the verifier returns and
/// the CLI renders as `counterexample_vnnlib`, where the UNCHANGED trusted
/// ONNX-Runtime gate (`vnncomp.rs::gate_sat_with_trusted_oracle`) stays the
/// sole verdict authority and downgrades anything it cannot re-confirm to a
/// sound `unknown`. A witness that does not cover every row stays ADVISORY,
/// exactly as before.
///
/// THE QUEUE IS NEVER DRAINED. The child is pushed back onto the heap first
/// and UNCONDITIONALLY on both violated paths, so this arm changes only what
/// the verifier may RETURN, never what the frontier contains. That leaves the
/// #violdrop/prop1498 protection fully intact: what the vit_2023 measurement
/// forbade is ABANDONING a child's sub-region on an uncertified
/// `upper < threshold` interval reading (see
/// `shared::violation_drop_is_certified`) — a β-derived bound carrying no
/// certificate for that child's region. This path abandons nothing and reads
/// no interval: its evidence is a concrete point that a real forward pass puts
/// inside the unsafe set.
pub(super) fn apply_leaf_verdict(
    verdict: GraphMipLeafVerdict,
    child: MultiObjectiveGraphBabDomain,
    queue: &mut BinaryHeap<MultiObjectiveGraphBabDomain>,
    cut_pool: &mut GraphCutPool,
    lifecycle: &mut GraphBabLifecycle,
    clause_store: &mut GraphClauseStore,
    enable_cuts: bool,
    leaf_ctx: Option<&LeafOracleCtx<'_>>,
    leaf_violation: &mut Option<BabVerificationStatus>,
) -> Result<()> {
    match verdict {
        GraphMipLeafVerdict::VerifiedAllRows => {
            lifecycle.domains_verified += 1;
            // This exact close has the same region-wide authority as a bound
            // close. Publish its pure ReLU history once, at the sole terminal
            // disposition point for the leaf verdict.
            clause_store.record_verified_close(&child.history);
            info!(
                depth = child.depth,
                "Graph-MIP leaf oracle: subdomain certified-UNSAT on all rows — verified"
            );
            if enable_cuts && cut_pool.add_from_verified_domain(&child.history)? {
                cut_pool.merge_cuts();
            }
        }
        GraphMipLeafVerdict::Violated { witness, output } => {
            let depth = child.depth;
            // Ask the coverage question BEFORE the child moves, so the answer
            // cannot depend on queue state.
            // Publication needs BOTH obligations. `may_publish_violation_witness`
            // is the LAYOUT obligation: fail-closed by default, and answered
            // `true` only by a caller that can see the clause structure and has
            // ruled out per-clause input boxes — the case where all-rows is NOT
            // sufficient (see `witness_violates_every_objective_row`'s SCOPE).
            let covers_property = leaf_sat_return_enabled()
                && leaf_ctx.is_some_and(|ctx| {
                    ctx.oracle.may_publish_violation_witness()
                        && witness_violates_every_objective_row(
                            ctx.objectives,
                            ctx.thresholds,
                            &output,
                        )
                });
            // NEVER DRAIN (#violdrop/prop1498): the child returns to the
            // frontier unconditionally, on the sat path as well as the
            // advisory one.
            queue.push(child);
            if covers_property && leaf_violation.is_none() {
                warn!(
                    depth,
                    witness_len = witness.len(),
                    output_len = output.len(),
                    rows = leaf_ctx.map_or(0, |ctx| ctx.objectives.len()),
                    "Graph-MIP leaf oracle: CONFIRMED in-box counterexample violating EVERY \
                     objective row — published as the run's sat candidate (child stays queued; \
                     the trusted ONNX-Runtime gate remains the verdict authority)"
                );
                *leaf_violation = Some(BabVerificationStatus::Violated {
                    counterexample: witness,
                    output,
                });
            } else {
                warn!(
                    depth,
                    witness_len = witness.len(),
                    output_len = output.len(),
                    "Graph-MIP leaf oracle: CONFIRMED in-box counterexample on a subdomain \
                     (advisory — the witness does not cover every objective row, so it is not \
                     a counterexample to the whole property; child requeued)"
                );
            }
        }
        GraphMipLeafVerdict::Undecided => {
            queue.push(child);
        }
    }
    Ok(())
}

pub(super) fn pop_domain_batch(
    queue: &mut BinaryHeap<MultiObjectiveGraphBabDomain>,
    batch_size: usize,
) -> Vec<MultiObjectiveGraphBabDomain> {
    // The queue is an exact upper bound on this wave. Do not honor an
    // adversarially large configured batch size as an allocation request when
    // only a small frontier exists.
    let mut batch = Vec::with_capacity(batch_size.min(queue.len()));
    while batch.len() < batch_size {
        let Some(domain) = queue.pop() else {
            break;
        };
        batch.push(domain);
    }
    batch
}

pub(super) fn prefilter_batch(
    batch: Vec<MultiObjectiveGraphBabDomain>,
    thresholds: &[f32],
    conjunctive: bool,
    max_depth: usize,
    enable_cuts: bool,
    lifecycle: &mut GraphBabLifecycle,
    cut_pool: &mut GraphCutPool,
    clause_store: &mut GraphClauseStore,
) -> Result<Vec<MultiObjectiveGraphBabDomain>> {
    let mut domains_to_process = Vec::new();

    for domain in batch {
        lifecycle.domains_explored += 1;
        lifecycle.max_depth_reached = lifecycle.max_depth_reached.max(domain.depth);

        // Conflict-clause prune at the pop (NY_BAB_CLAUSE_LEARN=1, default
        // off): a recorded clause proves its region safe under THIS run's
        // objective semantics — in the conjunctive lane a verified close means
        // some objective cleared on the whole region, so the conjunction is
        // impossible everywhere on it; in the disjunctive lane ALL objectives
        // cleared. A superset pure-ReLU literal set covers a subregion of that
        // certified region, so it is safe under the same semantics and may be
        // closed verified WITHOUT bound work. Fails closed for impure
        // (GenBaB/norm) histories. Ordering mirrors the v1 sequential
        // prefilter: prune-check precedes the domain's own verified check; a
        // pruned domain deliberately skips cut generation (its history is a
        // superset of a stored clause the cut machinery already saw).
        if clause_store.should_prune(&domain.history) {
            lifecycle.domains_verified += 1;
            continue;
        }

        let domain_verified = if conjunctive {
            domain.any_verified()
        } else {
            domain.all_verified()
        };
        if domain_verified {
            lifecycle.domains_verified += 1;
            // Verified close under this run's semantics: record the literal
            // set (no-op unless gated on AND the history is pure ReLU-at-0).
            clause_store.record_verified_close(&domain.history);
            if enable_cuts && cut_pool.add_from_verified_domain(&domain.history)? {
                let merged_len = cut_pool.merge_cuts();
                debug!("Merged verified-domain graph cuts (pool_len={merged_len})");
            }
            continue;
        }

        // #violdrop drop site 4 of 5 (see `shared::violdrop_site_probe` for the
        // full list). A popped BaB CHILD carries β-derived objective bounds whose
        // UPPER end is not certified for its sub-region, so `upper < threshold`
        // cannot prove a violation there — only the ROOT (`depth == 0`,
        // unaugmented α-CROWN interval) may be dropped on that reading. Every
        // site must be gated: each one alone re-creates the same queue collapse.
        let domain_dropped = if conjunctive {
            domain.all_violated(thresholds, false)
        } else {
            domain.any_violated(thresholds, false)
        };
        if domain_dropped {
            super::shared::violdrop_site_probe("prefilter_batch", domain.depth);
        }
        if domain_dropped && violation_drop_is_certified(domain.depth) {
            lifecycle.unresolved_due_to_violated_drop = true;
            continue;
        }

        if domain.depth >= max_depth {
            lifecycle.unresolved_due_to_depth = true;
            continue;
        }

        domains_to_process.push(domain);
    }

    Ok(domains_to_process)
}

/// Terminal status observed while folding one shared-executor batch.
///
/// A GPU admission refusal is a budget outcome, not a numerical propagation
/// failure. Thread it explicitly to the outer verifier so it can return
/// `Timeout` immediately after accounting for any completed siblings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "DeadlineExpired must be converted to a verifier Timeout"]
pub(super) enum MultiObjectiveBatchApplyStatus {
    Completed,
    DeadlineExpired,
}

#[cfg(test)]
mod violdrop_tests {
    use std::collections::BinaryHeap;
    use std::time::Instant;

    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    use super::{apply_batched_results, MultiObjectiveBatchApplyStatus};
    use crate::beta_crown::bab_cuts::GraphCutPool;
    use crate::beta_crown::conflict_clauses_graph::GraphClauseStore;
    use crate::beta_crown::domain::MultiObjectiveGraphBabDomain;
    use crate::beta_crown::engine::domain_results::MultiObjectiveGraphDomainResult;
    use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;

    /// A minimal unverified frontier domain: one objective, interval strictly
    /// BELOW its threshold in the lower direction, so it is neither verified
    /// (`lower > t` false) nor all-verified.
    fn unverified_domain() -> MultiObjectiveGraphBabDomain {
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .expect("input box");
        MultiObjectiveGraphBabDomain::root(
            std::collections::HashMap::new(),
            vec![(-2.0, -0.5)],
            &input,
            &[0.0],
            false,
        )
        .expect("root domain")
    }

    /// #violdrop — a dropped sibling must NOT take the surviving children with
    /// it. Before the fix the batched lane replaced the whole `Children` result
    /// with `Violation`, so the survivors never reached the heap and the queue
    /// emptied after a single root split (measured: vit_2023 ibp_3_3_8_3005,
    /// `explored=1 queue=0 max_depth=0` at 2.58 s of a 90.25 s grant).
    #[test]
    fn children_with_violated_drop_enqueues_survivors_and_flags_the_drop() {
        let mut queue: BinaryHeap<MultiObjectiveGraphBabDomain> = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::new(8);
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, _leaf_violation) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::ChildrenWithViolatedDrop(
                vec![(unverified_domain(), false)],
            )],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            None,
            &mut clause_store,
        )
        .expect("folding a violated-drop result must not error");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::Completed);
        assert_eq!(
            queue.len(),
            1,
            "the SURVIVING sibling must still be enqueued (this is the whole bug)"
        );
        assert!(
            lifecycle.unresolved_due_to_violated_drop,
            "the abandoned sibling's sub-region must still be recorded as unresolved"
        );
    }

    /// The legacy `Violation` result (every child dropped, no survivor) keeps its
    /// exact behavior: nothing enqueued, drop recorded.
    #[test]
    fn violation_without_survivors_is_unchanged() {
        let mut queue: BinaryHeap<MultiObjectiveGraphBabDomain> = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::new(8);
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, _leaf_violation) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::Violation],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            None,
            &mut clause_store,
        )
        .expect("folding a violation result must not error");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::Completed);
        assert!(queue.is_empty());
        assert!(lifecycle.unresolved_due_to_violated_drop);
    }

    /// A plain `Children` result must NOT raise the drop flag.
    #[test]
    fn plain_children_do_not_flag_a_drop() {
        let mut queue: BinaryHeap<MultiObjectiveGraphBabDomain> = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::new(8);
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, _leaf_violation) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::Children(vec![(
                unverified_domain(),
                false,
            )])],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            None,
            &mut clause_store,
        )
        .expect("folding children must not error");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::Completed);
        assert_eq!(queue.len(), 1);
        assert!(!lifecycle.unresolved_due_to_violated_drop);
    }
}

/// Leaf-oracle cost accumulator for ONE fold (`[phase] mo-leaf-oracle`).
///
/// The consult runs once per undecided child, so a per-child `eprintln!` would
/// itself perturb the loop it is measuring. The elapsed time is summed here
/// instead and reported once by [`apply_batched_results`]. Both fields stay at
/// zero unless phase telemetry is armed — nothing is timed and nothing is
/// printed on the ordinary path.
struct LeafOracleProbe {
    /// Latched once per fold. The gate is a `OnceLock` acquire plus a compare;
    /// re-reading it per undecided child inside the loop is allocation-free but
    /// still work in a hot path, and every other marker in this change hoists
    /// the read. Keep them consistent.
    armed: bool,
    consults: usize,
    secs: f64,
}

impl LeafOracleProbe {
    fn new() -> Self {
        Self {
            armed: crate::phase_telemetry::phase_telemetry_enabled(),
            consults: 0,
            secs: 0.0,
        }
    }
}

/// Enqueue / close one parent's surviving children (#violdrop extraction).
///
/// Shared verbatim by the `Children` and `ChildrenWithViolatedDrop` arms of
/// [`apply_batched_results`] so a dropped sibling cannot change how the
/// survivors are handled.
fn apply_batched_children(
    children: Vec<(MultiObjectiveGraphBabDomain, bool)>,
    queue: &mut BinaryHeap<MultiObjectiveGraphBabDomain>,
    cut_pool: &mut GraphCutPool,
    lifecycle: &mut GraphBabLifecycle,
    enable_cuts: bool,
    leaf_ctx: Option<&LeafOracleCtx<'_>>,
    clause_store: &mut GraphClauseStore,
    leaf_violation: &mut Option<BabVerificationStatus>,
    leaf_probe: &mut LeafOracleProbe,
) -> Result<()> {
    for (child, all_verified) in children {
        if all_verified {
            lifecycle.domains_verified += 1;
            // `all_verified` = EVERY objective row cleared on the child's
            // region — a verified close under both lane semantics
            // (disjunctive requires exactly this; conjunctive requires only
            // ANY, and ALL implies ANY), so recording is sound regardless of
            // `conjunctive`.
            clause_store.record_verified_close(&child.history);
            if enable_cuts && cut_pool.add_from_verified_domain(&child.history)? {
                cut_pool.merge_cuts();
            }
        } else {
            // Graph-MIP LEAF escalation (increment 6): before the undecided
            // child re-enters the heap, let the attached oracle (if any) try to
            // decide the subdomain exactly. `Undecided` (and the default
            // no-oracle path) pushes the child unchanged — strictly additive.
            // Print-only accounting for that escalation. A `None` ctx returns
            // immediately, so a fold that reports many consults and ~0s is
            // itself the evidence that no oracle is attached.
            let consult_start = leaf_probe.armed.then(Instant::now);
            let verdict = consult_leaf_oracle(leaf_ctx, &child);
            if let Some(t) = consult_start {
                leaf_probe.consults += 1;
                leaf_probe.secs += t.elapsed().as_secs_f64();
            }
            apply_leaf_verdict(
                verdict,
                child,
                queue,
                cut_pool,
                lifecycle,
                clause_store,
                enable_cuts,
                leaf_ctx,
                leaf_violation,
            )?;
        }
    }
    Ok(())
}

/// Fold one shared-executor wave into the queue/lifecycle.
///
/// The second element of the returned pair is the leaf oracle's sat candidate
/// (see [`apply_leaf_verdict`]): `Some` only when a graph-forward-confirmed
/// in-box witness violates EVERY objective row. It is a RETURN channel only —
/// the fold itself, including every queue push and every lifecycle counter, is
/// byte-identical whether or not a candidate is produced, and the candidate's
/// own child is still enqueued. The first candidate of a wave wins; later ones
/// are logged advisory so the fold cannot depend on sibling order.
pub(super) fn apply_batched_results(
    results: Vec<MultiObjectiveGraphDomainResult>,
    queue: &mut BinaryHeap<MultiObjectiveGraphBabDomain>,
    cut_pool: &mut GraphCutPool,
    lifecycle: &mut GraphBabLifecycle,
    enable_cuts: bool,
    leaf_ctx: Option<&LeafOracleCtx<'_>>,
    clause_store: &mut GraphClauseStore,
) -> Result<(
    MultiObjectiveBatchApplyStatus,
    Option<BabVerificationStatus>,
)> {
    let mut leaf_violation: Option<BabVerificationStatus> = None;
    let mut leaf_probe = LeafOracleProbe::new();
    // Pre-scan before any optional terminal-tail work. In particular, a
    // completed sibling must not launch Graph-MIP leaf solving or cut
    // generation and consume the reserve after another parent already
    // reported deadline admission failure. Ordinary result accounting still
    // runs below.
    let status = if results
        .iter()
        .any(|result| matches!(result, MultiObjectiveGraphDomainResult::DeadlineExpired))
    {
        MultiObjectiveBatchApplyStatus::DeadlineExpired
    } else {
        MultiObjectiveBatchApplyStatus::Completed
    };
    let terminal_deadline = status == MultiObjectiveBatchApplyStatus::DeadlineExpired;
    let enable_cuts = enable_cuts && !terminal_deadline;
    let leaf_ctx = if terminal_deadline { None } else { leaf_ctx };

    for result in results {
        match result {
            MultiObjectiveGraphDomainResult::AlreadyVerified => {
                // No history is carried by this variant — nothing to record
                // (fail-safe: strictly less pruning power, never unsound).
                lifecycle.domains_verified += 1;
            }
            MultiObjectiveGraphDomainResult::Violation => {
                lifecycle.unresolved_due_to_violated_drop = true;
            }
            MultiObjectiveGraphDomainResult::Children(children) => {
                apply_batched_children(
                    children,
                    queue,
                    cut_pool,
                    lifecycle,
                    enable_cuts,
                    leaf_ctx,
                    clause_store,
                    &mut leaf_violation,
                    &mut leaf_probe,
                )?;
            }
            MultiObjectiveGraphDomainResult::ChildrenWithViolatedDrop(children) => {
                // #violdrop: record the ABANDONED sibling's sub-region (so the
                // verdict cannot claim `Verified` for it) and STILL enqueue every
                // surviving child. The batched lane used to replace the whole
                // result with `Violation`, discarding the survivors too.
                lifecycle.unresolved_due_to_violated_drop = true;
                apply_batched_children(
                    children,
                    queue,
                    cut_pool,
                    lifecycle,
                    enable_cuts,
                    leaf_ctx,
                    clause_store,
                    &mut leaf_violation,
                    &mut leaf_probe,
                )?;
            }
            MultiObjectiveGraphDomainResult::NoUnstable {
                all_verified,
                any_violated,
            } => {
                if all_verified {
                    lifecycle.domains_verified += 1;
                } else if any_violated {
                    lifecycle.unresolved_due_to_violated_drop = true;
                } else {
                    lifecycle.unresolved_due_to_no_branch = true;
                }
            }
            MultiObjectiveGraphDomainResult::PropagationFailure => {
                lifecycle.unresolved_due_to_propagation_failure = true;
            }
            MultiObjectiveGraphDomainResult::DeadlineExpired => {
                // The pre-scan above already latched the typed batch status.
            }
        }
    }

    // One line per fold, and only when armed: `consults` is non-zero only if
    // the loop above timed at least one child.
    if leaf_probe.consults > 0 {
        eprintln!(
            "[phase] mo-leaf-oracle consults={} secs={:.2}",
            leaf_probe.consults, leaf_probe.secs
        );
    }

    Ok((status, leaf_violation))
}

#[cfg(test)]
mod deadline_status_tests {
    use std::collections::BinaryHeap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    use super::*;
    use crate::beta_crown::branching::GraphNeuronConstraint;
    use crate::beta_crown::conflict_clauses_graph::{
        reset_test_record_attempts, reset_test_store_mutations, test_record_attempts,
        test_store_mutations,
    };
    use crate::beta_crown::graph_mip_leaf::{
        GraphMipLeafOracle, GraphMipLeafRequest, GraphMipLeafVerdict,
    };

    struct CountingLeafOracle {
        calls: AtomicUsize,
    }

    impl CountingLeafOracle {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl GraphMipLeafOracle for CountingLeafOracle {
        fn solve_leaf(&self, _req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
            self.calls.fetch_add(1, Ordering::SeqCst);
            GraphMipLeafVerdict::Undecided
        }
    }

    struct VerifyingLeafOracle {
        calls: AtomicUsize,
    }

    impl VerifyingLeafOracle {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl GraphMipLeafOracle for VerifyingLeafOracle {
        fn solve_leaf(&self, _req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
            self.calls.fetch_add(1, Ordering::SeqCst);
            GraphMipLeafVerdict::VerifiedAllRows
        }
    }

    fn undecided_child() -> MultiObjectiveGraphBabDomain {
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("test input bounds");
        MultiObjectiveGraphBabDomain::root(
            std::collections::HashMap::new(),
            vec![(-1.0, 1.0)],
            &input,
            &[0.0],
            false,
        )
        .expect("test child")
    }

    fn verified_child_with_cut_history() -> MultiObjectiveGraphBabDomain {
        let mut child = undecided_child();
        child.history.add_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 1.0,
        });
        child.history.add_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 1,
            is_active: false,
            score: 1.0,
        });
        child
    }

    #[test]
    fn post_call_deadline_revokes_late_verified_leaf_result() {
        let oracle = VerifyingLeafOracle::new();
        let graph = GraphNetwork::new();
        let objectives = vec![vec![1.0_f32]];
        let thresholds = vec![0.0_f32];
        let before_deadline = Instant::now();
        let deadline = before_deadline + Duration::from_secs(1);
        let leaf_ctx = LeafOracleCtx {
            oracle: &oracle,
            graph: &graph,
            objectives: &objectives,
            thresholds: &thresholds,
            deadline: Some(deadline),
        };
        let child = undecided_child();
        let mut clock = [before_deadline, deadline].into_iter();

        let verdict = consult_leaf_oracle_with_clock(Some(&leaf_ctx), &child, || {
            clock
                .next()
                .expect("one pre-call and one post-call reading")
        });

        assert!(matches!(verdict, GraphMipLeafVerdict::Undecided));
        assert_eq!(
            oracle.calls.load(Ordering::SeqCst),
            1,
            "the on-time call ran, but its late result must carry no authority"
        );
        assert!(
            clock.next().is_none(),
            "deadline must be checked after solving"
        );
    }

    #[test]
    fn exact_leaf_verified_close_records_one_conflict_clause() {
        reset_test_record_attempts();
        reset_test_store_mutations();
        let oracle = VerifyingLeafOracle::new();
        let graph = GraphNetwork::new();
        let objectives = vec![vec![1.0_f32]];
        let thresholds = vec![0.0_f32];
        let leaf_ctx = LeafOracleCtx {
            oracle: &oracle,
            graph: &graph,
            objectives: &objectives,
            thresholds: &thresholds,
            deadline: None,
        };
        let child = verified_child_with_cut_history();
        let history = child.history.clone();
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::with_capacity(true, 16);

        let (status, _leaf_violation) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::Children(vec![(
                child, false,
            )])],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            Some(&leaf_ctx),
            &mut clause_store,
        )
        .expect("exact verified close is infallible without cuts");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::Completed);
        assert_eq!(oracle.calls.load(Ordering::SeqCst), 1);
        assert_eq!(lifecycle.domains_verified, 1);
        assert!(queue.is_empty());
        assert_eq!(clause_store.len(), 1);
        assert_eq!(
            test_record_attempts(),
            1,
            "the batched exact close must invoke the record boundary exactly once"
        );
        assert_eq!(
            test_store_mutations(),
            1,
            "the exact close must be published at exactly one disposition point"
        );
        assert!(
            clause_store.should_prune(&history),
            "the exact leaf's pure ReLU region must be reusable"
        );
    }

    #[test]
    fn deadline_result_stays_typed_and_does_not_become_propagation_failure() {
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, _leaf_violation) = apply_batched_results(
            vec![
                MultiObjectiveGraphDomainResult::AlreadyVerified,
                MultiObjectiveGraphDomainResult::DeadlineExpired,
            ],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            None,
            &mut clause_store,
        )
        .expect("deadline folding is infallible");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::DeadlineExpired);
        assert_eq!(
            lifecycle.domains_verified, 1,
            "completed sibling accounting must survive terminal deadline folding"
        );
        assert!(
            !lifecycle.unresolved_due_to_propagation_failure,
            "deadline admission must not be relabelled as numerical failure"
        );
    }

    #[test]
    fn propagation_failure_does_not_impersonate_deadline_expiry() {
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, _leaf_violation) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::PropagationFailure],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            None,
            &mut clause_store,
        )
        .expect("failure folding is infallible");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::Completed);
        assert!(lifecycle.unresolved_due_to_propagation_failure);
    }

    #[test]
    fn terminal_deadline_suppresses_leaf_oracle_but_conservatively_queues_child() {
        let oracle = CountingLeafOracle::new();
        let graph = GraphNetwork::new();
        let objectives = vec![vec![1.0_f32]];
        let thresholds = vec![0.0_f32];
        let leaf_ctx = LeafOracleCtx {
            oracle: &oracle,
            graph: &graph,
            objectives: &objectives,
            thresholds: &thresholds,
            deadline: None,
        };
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, _leaf_violation) = apply_batched_results(
            vec![
                MultiObjectiveGraphDomainResult::Children(vec![(undecided_child(), false)]),
                MultiObjectiveGraphDomainResult::DeadlineExpired,
            ],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            true,
            Some(&leaf_ctx),
            &mut clause_store,
        )
        .expect("deadline folding is infallible");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::DeadlineExpired);
        assert_eq!(oracle.calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            queue.len(),
            1,
            "unresolved completed sibling must remain covered"
        );
    }

    #[test]
    fn terminal_deadline_preserves_violated_drop_survivor_without_tail_work() {
        let oracle = CountingLeafOracle::new();
        let graph = GraphNetwork::new();
        let objectives = vec![vec![1.0_f32]];
        let thresholds = vec![0.0_f32];
        let leaf_ctx = LeafOracleCtx {
            oracle: &oracle,
            graph: &graph,
            objectives: &objectives,
            thresholds: &thresholds,
            deadline: None,
        };
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, _leaf_violation) = apply_batched_results(
            vec![
                MultiObjectiveGraphDomainResult::ChildrenWithViolatedDrop(vec![(
                    undecided_child(),
                    false,
                )]),
                MultiObjectiveGraphDomainResult::DeadlineExpired,
            ],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            true,
            Some(&leaf_ctx),
            &mut clause_store,
        )
        .expect("terminal deadline/drop folding is infallible");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::DeadlineExpired);
        assert_eq!(oracle.calls.load(Ordering::SeqCst), 0);
        assert_eq!(cut_pool.total_generated, 0);
        assert_eq!(queue.len(), 1, "the surviving sibling must remain covered");
        assert!(
            lifecycle.unresolved_due_to_violated_drop,
            "the abandoned sibling must remain recorded as unresolved"
        );
    }

    #[test]
    fn completed_batch_still_consults_leaf_oracle() {
        let oracle = CountingLeafOracle::new();
        let graph = GraphNetwork::new();
        let objectives = vec![vec![1.0_f32]];
        let thresholds = vec![0.0_f32];
        let leaf_ctx = LeafOracleCtx {
            oracle: &oracle,
            graph: &graph,
            objectives: &objectives,
            thresholds: &thresholds,
            deadline: None,
        };
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, _leaf_violation) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::Children(vec![(
                undecided_child(),
                false,
            )])],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            Some(&leaf_ctx),
            &mut clause_store,
        )
        .expect("completed folding is infallible");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::Completed);
        assert_eq!(oracle.calls.load(Ordering::SeqCst), 1);
        assert_eq!(queue.len(), 1);
    }

    /// A leaf oracle that hands back a CONFIRMED in-box witness plus the
    /// graph-forward output at that point (what the real ny-cli solver returns
    /// from its `revalidate_leaf_witness` gate).
    struct ViolatingLeafOracle {
        witness: Vec<f32>,
        output: Vec<f32>,
        /// Stands in for the CLI's clause-layout classification. `true` models
        /// a spec with no per-clause input boxes (where the lane's all-rows
        /// check is sufficient); `false` models both the per-clause-box case
        /// and the fail-closed default.
        may_publish: bool,
    }

    impl GraphMipLeafOracle for ViolatingLeafOracle {
        fn solve_leaf(&self, _req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
            GraphMipLeafVerdict::Violated {
                witness: self.witness.clone(),
                output: self.output.clone(),
            }
        }

        fn may_publish_violation_witness(&self) -> bool {
            self.may_publish
        }
    }

    /// An oracle that confirms a witness but does NOT override the trait's
    /// fail-closed publication default — exactly what a future oracle written
    /// without thinking about clause layout would be.
    struct DefaultPublicationLeafOracle {
        witness: Vec<f32>,
        output: Vec<f32>,
    }

    impl GraphMipLeafOracle for DefaultPublicationLeafOracle {
        fn solve_leaf(&self, _req: &GraphMipLeafRequest<'_>) -> GraphMipLeafVerdict {
            GraphMipLeafVerdict::Violated {
                witness: self.witness.clone(),
                output: self.output.clone(),
            }
        }
    }

    /// Drive `apply_leaf_verdict` end-to-end through the real batched fold: an
    /// oracle-confirmed witness that violates EVERY objective row must come
    /// back as the typed `BabVerificationStatus::Violated` carrier the CLI
    /// renders into `counterexample_vnnlib` (and hence into
    /// `gate_sat_with_trusted_oracle`) — and the child must STILL be on the
    /// queue.
    #[test]
    fn leaf_witness_covering_every_row_returns_the_typed_sat_and_keeps_the_child() {
        // Two rows, both violated at the witness output: row 0 is `1*y <= 0`
        // and row 1 is `2*y <= 0`, and the graph forward gives y = -1.5.
        let objectives = vec![vec![1.0_f32], vec![2.0_f32]];
        let thresholds = vec![0.0_f32, 0.0_f32];
        let oracle = ViolatingLeafOracle {
            witness: vec![0.25_f32, -0.5],
            output: vec![-1.5_f32],
            may_publish: true,
        };
        let graph = GraphNetwork::new();
        let leaf_ctx = LeafOracleCtx {
            oracle: &oracle,
            graph: &graph,
            objectives: &objectives,
            thresholds: &thresholds,
            deadline: None,
        };
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, leaf_violation) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::Children(vec![(
                undecided_child(),
                false,
            )])],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            Some(&leaf_ctx),
            &mut clause_store,
        )
        .expect("folding a confirmed leaf witness is infallible");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::Completed);
        assert_eq!(
            leaf_violation,
            Some(BabVerificationStatus::Violated {
                counterexample: vec![0.25_f32, -0.5],
                output: vec![-1.5_f32],
            }),
            "the confirmed witness must reach the verifier through the existing carrier"
        );
        assert_eq!(
            queue.len(),
            1,
            "#violdrop/prop1498: returning a sat must NOT drain the queue"
        );
        assert_eq!(
            lifecycle.domains_verified, 0,
            "a violated leaf is not a verified close"
        );
        assert!(
            !lifecycle.has_unresolved(),
            "publishing a sat candidate must not raise an unresolved flag"
        );
    }

    /// The LAYOUT obligation, negative case: an oracle that confirms the same
    /// witness but does NOT declare the clause layout admissible must stay
    /// ADVISORY. This is the per-clause-input-box case
    /// (`VnnLibSpec::per_clause_input_bounds` non-empty), where every output
    /// row holding at a hull point implies nothing about any clause — and it is
    /// also what any oracle that never thought about layout inherits, because
    /// `may_publish_violation_witness` defaults to false.
    #[test]
    fn a_witness_whose_layout_is_not_cleared_stays_advisory() {
        let objectives = vec![vec![1.0_f32], vec![2.0_f32]];
        let thresholds = vec![0.0_f32, 0.0_f32];
        let oracle = ViolatingLeafOracle {
            witness: vec![0.25_f32, -0.5],
            output: vec![-1.5_f32],
            may_publish: false,
        };
        let graph = GraphNetwork::new();
        let leaf_ctx = LeafOracleCtx {
            oracle: &oracle,
            graph: &graph,
            objectives: &objectives,
            thresholds: &thresholds,
            deadline: None,
        };
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, leaf_violation) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::Children(vec![(
                undecided_child(),
                false,
            )])],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            Some(&leaf_ctx),
            &mut clause_store,
        )
        .expect("folding a confirmed leaf witness is infallible");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::Completed);
        assert_eq!(
            leaf_violation, None,
            "a witness whose clause layout was not cleared must NOT be published"
        );
        assert_eq!(
            queue.len(),
            1,
            "the advisory path must requeue the child exactly as before"
        );
    }

    /// The same guarantee for an oracle that simply never overrides the trait
    /// default — the shape a future oracle author would write. Publication must
    /// require an AFFIRMATIVE statement, never silence.
    #[test]
    fn an_oracle_using_the_trait_default_does_not_publish() {
        let objectives = vec![vec![1.0_f32], vec![2.0_f32]];
        let thresholds = vec![0.0_f32, 0.0_f32];
        let oracle = DefaultPublicationLeafOracle {
            witness: vec![0.25_f32, -0.5],
            output: vec![-1.5_f32],
        };
        let graph = GraphNetwork::new();
        let leaf_ctx = LeafOracleCtx {
            oracle: &oracle,
            graph: &graph,
            objectives: &objectives,
            thresholds: &thresholds,
            deadline: None,
        };
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (_status, leaf_violation) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::Children(vec![(
                undecided_child(),
                false,
            )])],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            Some(&leaf_ctx),
            &mut clause_store,
        )
        .expect("folding a confirmed leaf witness is infallible");

        assert_eq!(
            leaf_violation, None,
            "the trait default is fail-closed: silence must not publish a sat"
        );
        assert_eq!(queue.len(), 1, "the child is still requeued");
    }

    /// The coverage obligation is real: a witness that satisfies only SOME rows
    /// is not a counterexample to the whole property, so it stays advisory and
    /// the child is requeued exactly as before this change.
    #[test]
    fn leaf_witness_covering_only_one_row_stays_advisory_and_requeues() {
        // Row 0 (`1*y <= 0`) holds at y = -1.5; row 1 (`-1*y <= -10`, i.e.
        // y >= 10) does not.
        let objectives = vec![vec![1.0_f32], vec![-1.0_f32]];
        let thresholds = vec![0.0_f32, -10.0_f32];
        let oracle = ViolatingLeafOracle {
            witness: vec![0.25_f32, -0.5],
            output: vec![-1.5_f32],
            may_publish: true,
        };
        let graph = GraphNetwork::new();
        let leaf_ctx = LeafOracleCtx {
            oracle: &oracle,
            graph: &graph,
            objectives: &objectives,
            thresholds: &thresholds,
            deadline: None,
        };
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, leaf_violation) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::Children(vec![(
                undecided_child(),
                false,
            )])],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            Some(&leaf_ctx),
            &mut clause_store,
        )
        .expect("folding an uncovered leaf witness is infallible");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::Completed);
        assert_eq!(
            leaf_violation, None,
            "one satisfied row is not a counterexample to an OR-of-AND property"
        );
        assert_eq!(queue.len(), 1, "the child is requeued, exactly as before");
    }

    /// The pre-change contract for every non-oracle run: no oracle attached =>
    /// no sat channel, byte-identical fold.
    #[test]
    fn no_oracle_never_produces_a_sat_candidate() {
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, leaf_violation) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::Children(vec![(
                undecided_child(),
                false,
            )])],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            None,
            &mut clause_store,
        )
        .expect("no-oracle folding is infallible");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::Completed);
        assert_eq!(leaf_violation, None);
        assert_eq!(queue.len(), 1);
    }

    /// The terminal-deadline suppression already blocks the oracle call, so it
    /// must also block any sat candidate: no verdict, no witness, no channel.
    #[test]
    fn terminal_deadline_suppresses_the_leaf_sat_candidate() {
        let objectives = vec![vec![1.0_f32]];
        let thresholds = vec![0.0_f32];
        let oracle = ViolatingLeafOracle {
            witness: vec![0.25_f32],
            output: vec![-1.5_f32],
            may_publish: true,
        };
        let graph = GraphNetwork::new();
        let leaf_ctx = LeafOracleCtx {
            oracle: &oracle,
            graph: &graph,
            objectives: &objectives,
            thresholds: &thresholds,
            deadline: None,
        };
        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();

        let (status, leaf_violation) = apply_batched_results(
            vec![
                MultiObjectiveGraphDomainResult::Children(vec![(undecided_child(), false)]),
                MultiObjectiveGraphDomainResult::DeadlineExpired,
            ],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            false,
            Some(&leaf_ctx),
            &mut clause_store,
        )
        .expect("terminal deadline folding is infallible");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::DeadlineExpired);
        assert_eq!(
            leaf_violation, None,
            "terminal-deadline tail work must not mint a verdict"
        );
        assert_eq!(queue.len(), 1);
    }

    /// The coverage predicate itself: the boundary and the fail-closed cases.
    #[test]
    fn witness_coverage_predicate_boundaries_and_fail_closed_cases() {
        // Equality counts as a violation (`obj · y <= threshold`).
        assert!(witness_violates_every_objective_row(
            &[vec![1.0], vec![1.0]],
            &[0.0, 2.0],
            &[0.0]
        ));
        // One row above its threshold breaks coverage.
        assert!(!witness_violates_every_objective_row(
            &[vec![1.0], vec![1.0]],
            &[0.0, -2.0],
            &[0.0]
        ));
        // Fail closed: no rows at all.
        assert!(!witness_violates_every_objective_row(&[], &[], &[1.0]));
        // Fail closed: rows/thresholds length mismatch.
        assert!(!witness_violates_every_objective_row(
            &[vec![1.0], vec![1.0]],
            &[0.0],
            &[-1.0]
        ));
        // Fail closed: coefficient row does not match the output width.
        assert!(!witness_violates_every_objective_row(
            &[vec![1.0, 1.0]],
            &[0.0],
            &[-1.0]
        ));
        // Fail closed: non-finite output, threshold, coefficient.
        assert!(!witness_violates_every_objective_row(
            &[vec![1.0]],
            &[0.0],
            &[f32::NAN]
        ));
        assert!(!witness_violates_every_objective_row(
            &[vec![1.0]],
            &[f32::NAN],
            &[-1.0]
        ));
        assert!(!witness_violates_every_objective_row(
            &[vec![f32::INFINITY]],
            &[0.0],
            &[-1.0]
        ));
        // Fail closed: empty output vector.
        assert!(!witness_violates_every_objective_row(
            &[vec![1.0]],
            &[0.0],
            &[]
        ));
    }

    #[test]
    fn terminal_deadline_suppresses_verified_sibling_cut_generation() {
        // Control first: this history is deep/non-trivial enough to generate a
        // verified-domain cut when the batch is not terminal.
        let mut control_queue = BinaryHeap::new();
        let mut control_pool = GraphCutPool::default();
        let mut control_lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut control_clauses = GraphClauseStore::disabled();
        let (control_status, _) = apply_batched_results(
            vec![MultiObjectiveGraphDomainResult::Children(vec![(
                verified_child_with_cut_history(),
                true,
            )])],
            &mut control_queue,
            &mut control_pool,
            &mut control_lifecycle,
            true,
            None,
            &mut control_clauses,
        )
        .expect("control cut folding");
        assert_eq!(control_status, MultiObjectiveBatchApplyStatus::Completed);
        assert_eq!(control_pool.total_generated, 1);

        let mut queue = BinaryHeap::new();
        let mut cut_pool = GraphCutPool::default();
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        let mut clause_store = GraphClauseStore::disabled();
        let (status, _leaf_violation) = apply_batched_results(
            vec![
                MultiObjectiveGraphDomainResult::Children(vec![(
                    verified_child_with_cut_history(),
                    true,
                )]),
                MultiObjectiveGraphDomainResult::DeadlineExpired,
            ],
            &mut queue,
            &mut cut_pool,
            &mut lifecycle,
            true,
            None,
            &mut clause_store,
        )
        .expect("terminal deadline cut folding");

        assert_eq!(status, MultiObjectiveBatchApplyStatus::DeadlineExpired);
        assert_eq!(
            cut_pool.total_generated, 0,
            "completed siblings are accounted but must not generate terminal-tail cuts"
        );
        assert_eq!(lifecycle.domains_verified, 1);
    }
}
