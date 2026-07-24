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
use crate::GraphNetwork;

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
    let Some(ctx) = ctx else {
        return GraphMipLeafVerdict::Undecided;
    };
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
    let req = GraphMipLeafRequest {
        graph: ctx.graph,
        input_bounds: &child.input_bounds,
        node_bounds: &child.node_bounds,
        splits,
        rows,
        depth: child.depth,
        deadline: ctx.deadline,
    };
    ctx.oracle.solve_leaf(&req)
}

/// Fold a leaf-oracle verdict into the queue/lifecycle. `Undecided` pushes the
/// child (the pre-oracle behavior); `VerifiedAllRows` counts it verified (and
/// feeds the verified-domain cut pool exactly like a BaB-verified child);
/// `Violated` is ADVISORY: the witness (already graph-forward-confirmed by the
/// oracle) is logged loudly, but the child is REQUEUED — never dropped — so a
/// leaf verdict can only ever convert "requeue" into "verified", and no oracle
/// outcome can drain the queue / end the run early (the measured prop1498
/// failure mode). Sat-side reporting stays with the PGD/ORT lanes until the
/// BaB lanes grow witness plumbing; a well-behaved oracle latches itself off
/// after a confirmed SAT so this path does not repeat.
pub(super) fn apply_leaf_verdict(
    verdict: GraphMipLeafVerdict,
    child: MultiObjectiveGraphBabDomain,
    queue: &mut BinaryHeap<MultiObjectiveGraphBabDomain>,
    cut_pool: &mut GraphCutPool,
    lifecycle: &mut GraphBabLifecycle,
    enable_cuts: bool,
) -> Result<()> {
    match verdict {
        GraphMipLeafVerdict::VerifiedAllRows => {
            lifecycle.domains_verified += 1;
            info!(
                depth = child.depth,
                "Graph-MIP leaf oracle: subdomain certified-UNSAT on all rows — verified"
            );
            if enable_cuts && cut_pool.add_from_verified_domain(&child.history)? {
                cut_pool.merge_cuts();
            }
        }
        GraphMipLeafVerdict::Violated { witness, output } => {
            warn!(
                depth = child.depth,
                witness_len = witness.len(),
                output_len = output.len(),
                "Graph-MIP leaf oracle: CONFIRMED in-box counterexample on a subdomain \
                 (advisory — child requeued; sat reporting stays with the attack lanes)"
            );
            queue.push(child);
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
    let mut batch = Vec::with_capacity(batch_size);
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

        let domain_dropped = if conjunctive {
            domain.all_violated(thresholds, false)
        } else {
            domain.any_violated(thresholds, false)
        };
        if domain_dropped {
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

pub(super) fn apply_batched_results(
    results: Vec<MultiObjectiveGraphDomainResult>,
    queue: &mut BinaryHeap<MultiObjectiveGraphBabDomain>,
    cut_pool: &mut GraphCutPool,
    lifecycle: &mut GraphBabLifecycle,
    enable_cuts: bool,
    leaf_ctx: Option<&LeafOracleCtx<'_>>,
    clause_store: &mut GraphClauseStore,
) -> Result<()> {
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
                for (child, all_verified) in children {
                    if all_verified {
                        lifecycle.domains_verified += 1;
                        // `all_verified` = EVERY objective row cleared on the
                        // child's region — a verified close under both lane
                        // semantics (disjunctive requires exactly this;
                        // conjunctive requires only ANY, and ALL implies ANY),
                        // so recording is sound regardless of `conjunctive`.
                        clause_store.record_verified_close(&child.history);
                        if enable_cuts && cut_pool.add_from_verified_domain(&child.history)? {
                            cut_pool.merge_cuts();
                        }
                    } else {
                        // Graph-MIP LEAF escalation (increment 6): before the
                        // undecided child re-enters the heap, let the attached
                        // oracle (if any) try to decide the subdomain exactly.
                        // `Undecided` (and the default no-oracle path) pushes
                        // the child unchanged — strictly additive.
                        let verdict = consult_leaf_oracle(leaf_ctx, &child);
                        apply_leaf_verdict(
                            verdict,
                            child,
                            queue,
                            cut_pool,
                            lifecycle,
                            enable_cuts,
                        )?;
                    }
                }
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
        }
    }

    Ok(())
}
