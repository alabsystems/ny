// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verification dispatch for β-CROWN constraint verification.
//!
//! Splits verification by concern:
//! - `graph` — Graph model per-constraint iteration
//! - `sequential` — Sequential model verification (reduced + per-constraint)
//! - `disjunctive` — Multi-clause disjunction handling
//! - `pgd` — PGD attack functions for counterexample search

mod attack_budget;
mod attack_extension;
mod constraint_iter;
mod disjunctive;
/// Batched box-refinement screen for per-clause-input-box disjunctions
/// (nn4sys mscn/lindex band properties).
mod disjunctive_box_refine;
mod disjunctive_per_disjunct;
mod disjunctive_pgd;
mod disjunctive_precheck;
/// Unified input-split disjunctive BaB lane (Packet B of #3740).
mod disjunctive_unified;
mod graph;
mod graph_pgd;
mod graph_pgd_batched;
mod graph_pgd_exact;
mod graph_pgd_init;
mod graph_pgd_vjp_batched;
mod graph_pgd_vjp_batched_disj;
pub(super) mod ort_attack;
mod pgd;
mod pgd_precheck;
mod pgd_sampling;
pub(super) mod phase_budget;
mod potential_violation;
mod sequential;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_conjunctive_routing;
#[cfg(test)]
mod tests_disjunctive_timing;
#[cfg(test)]
mod tests_graph_disjunctive;
#[cfg(test)]
mod tests_graph_pgd;
#[cfg(test)]
mod tests_precheck;

use anyhow::Result;
use ndarray::ArrayD;
use ny_core::GemmEngine;
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BetaCrownVerifier, GraphNetwork,
};
use ny_tensor::BoundedTensor;
use std::time::Instant;

// Re-exports for submodules — these items are used across multiple split files.
pub(crate) use super::constraint_eval::augment_network_with_spec;
pub(crate) use super::constraint_plan::{
    build_constraint_objective, build_multi_objectives, classify_constraints, AggregationMode,
};
pub(crate) use super::engine_dispatch::dispatch_graph_constraint;
pub(in crate::commands::beta_crown) use pgd_precheck::{try_pgd_before_mip, PgdMipPrecheck};
pub(in crate::commands::beta_crown) use potential_violation::confirm_potential_violation;
// Private use — accessible from child modules (disjunctive, tests, etc.)
use super::BetaCrownModel;

pub(super) fn normalize_result_wall_time(
    mut result: BetaCrownResult,
    overall_start: Instant,
) -> BetaCrownResult {
    result.time_elapsed = result.time_elapsed.max(overall_start.elapsed());
    result
}

/// Verify with relational constraints (Y_i <= Y_j, Y_i >= Y_j, etc.)
// Justification: Verification dispatch forwards CLI-specified parameters (model, input,
// spec, config, verifier, flags, engine) that are assembled by different callers.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_relational_constraints(
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
    json: bool,
) -> Result<BetaCrownResult> {
    // Per-clause-box disjunctions route to the disjunctive lane even with a
    // SINGLE clause (nn4sys mscn `_dual` cardinality_1_1): the clause's own
    // input box + band constraint can only be decided by the box-refinement
    // screen (+ f64 leaf escalation), which lives behind this dispatch. For
    // one clause the disjunctive aggregation (UNSAT ⇔ ALL clauses impossible)
    // degenerates to exactly the conjunctive reading of that clause, so the
    // verdict semantics are unchanged.
    if vnnlib.has_multi_constraint_disjunction() || vnnlib.has_boxed_clause_disjunction() {
        return disjunctive::verify_multi_clause_disjunction(
            model_net,
            input,
            vnnlib,
            config,
            verifier,
            use_relu_split,
            gpu_bab,
            pgd_attack,
            pgd_restarts,
            pgd_steps,
            timeout,
            gemm_engine,
            json,
        );
    }

    verify_relational_constraints_impl(
        model_net,
        input,
        vnnlib,
        config,
        verifier,
        use_relu_split,
        gpu_bab,
        pgd_attack,
        pgd_restarts,
        pgd_steps,
        timeout,
        gemm_engine,
        json,
    )
}

// Justification: Core verification implementation — parameters represent independent
// verification context (model, bounds, spec, config, engine, output flags).
#[allow(clippy::too_many_arguments)]
fn verify_relational_constraints_impl(
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
    json: bool,
) -> Result<BetaCrownResult> {
    // For conjunctive properties with >=2 constraints on Sequential models, convert
    // to Graph and use the multi-objective conjunctive BaB path. Joint BaB is
    // strictly more powerful than per-constraint decomposition: it proves per-subdomain
    // falsity of SOME constraint (different constraints in different subdomains), while
    // per-constraint requires each conjunct to be universally false.
    //
    // The multi-objective conjunctive BaB is only implemented for Graph models
    // (shared CROWN backward pass across all specs). Sequential feedforward networks
    // (e.g., sat_relu, lsnc_relu) are trivially convertible via from_sequential.
    //
    // Reference: designs/2026-03-05-joint-conjunctive-bab.md Phase 4.
    if let BetaCrownModel::Sequential(network) = model_net {
        let classification = classify_constraints(vnnlib);
        let is_conjunctive = classification.aggregation == AggregationMode::Conjunctive;
        let total_constraints = vnnlib.output_constraints.len();

        // Upgrade Sequential → Graph for multi-objective conjunctive BaB when using
        // ReLU splitting, OR when input splitting and the same-LHS max-diff reduction
        // does NOT apply.
        //
        // With input splitting, the routing is decided by the property SHAPE
        // (MEASURED 2026-07-10, contention-matched A/B):
        //
        // - Same-LHS relational conjunctions (acasxu prop_2/3/4: Y_0 vs Y_1..Y_4):
        //   stay on the sequential pipeline, whose reduced max-diff phase encodes
        //   min over x of max_j(violation_j) via MaxPool. CROWN's max relaxation can
        //   pick convex combinations that dominate EVERY single row, so it decides
        //   joint-witness conjunctions (the verifying conjunct varies across the box)
        //   per-domain, where the multi-objective any-row check needs to split until
        //   one row dominates each box: 4_2/prop_3 = 672 max-diff domains (~4s BaB)
        //   vs any-row TIMEOUT at 116s (>100k domains); 1_5/prop_2 any-row explored
        //   981k domains (depth 33, 494s BaB) without converging.
        //
        // - Everything else (constant/mixed conjunctions, no max-diff reduction:
        //   sat_relu Y_0>=1 AND Y_1<=0, lsnc band constraints): upgrade to the graph
        //   multi-objective conjunctive input-split lane — the sequential pipeline
        //   has no joint engine for these (reduced verification bails to the weaker
        //   per-constraint decomposition). MEASURED: sat_relu unsat_v30_c38
        //   unsat 36.4s → 2s; lsnc quadrotor2d_state_0 unsat 29.9s → 28s (contended).
        //
        // Ref: #1923 — the conjunctive upgrade previously hardcoded use_relu_split=true,
        // routing input-split models through ReLU-split BaB and causing ACAS-Xu 4_x
        // timeout (19583 domains, 0 verified). That regression does not reapply:
        // `use_relu_split` is threaded through unchanged, so input-split models that
        // upgrade here run `verify_graph_input_split_multi_objective_conjunctive`,
        // not ReLU-split BaB; and acasxu-shaped properties don't upgrade at all.
        let same_lhs_reducible = sequential::normalize_same_lhs_reduction(vnnlib).is_some();
        if is_conjunctive && total_constraints >= 2 && (use_relu_split || !same_lhs_reducible) {
            let mut graph = GraphNetwork::from_sequential(network)?;
            graph.set_use_patches_mode(config.use_patches());
            if !json {
                println!(
                    "\n  Conjunctive property ({} constraints): upgrading to Graph multi-objective BaB",
                    total_constraints
                );
            }
            return graph::verify_graph_relational(
                &graph,
                input,
                vnnlib,
                config,
                verifier,
                use_relu_split,
                gpu_bab,
                timeout,
                gemm_engine,
                json,
            );
        }
    }

    if let BetaCrownModel::Graph(graph) = model_net {
        graph::verify_graph_relational(
            graph,
            input,
            vnnlib,
            config,
            verifier,
            use_relu_split,
            gpu_bab,
            timeout,
            gemm_engine,
            json,
        )
    } else {
        let network = match model_net {
            BetaCrownModel::Sequential(network) => network,
            BetaCrownModel::Graph(_) => {
                return Err(anyhow::anyhow!(
                    "Graph model variant reached sequential path (should have been handled above)"
                ))
            }
        };

        sequential::verify_sequential_relational(
            network,
            input,
            vnnlib,
            config,
            verifier,
            pgd_attack,
            pgd_restarts,
            pgd_steps,
            timeout,
            gemm_engine,
            json,
        )
    }
}

/// Whether a per-constraint result proves that the unsafe constraint is impossible.
///
/// Reference (alpha-beta-CROWN): `complete_verifier_func.py` treats any non-safe
/// batch status (`unsafe_bab`, `unknown`) as non-safe; counterexamples are never
/// counted as safety proofs.
fn constraint_is_safety_proof(status: &BabVerificationStatus) -> bool {
    matches!(status, BabVerificationStatus::Verified)
}

/// Convert a non-verified disjunctive constraint result into a final verdict.
///
/// In disjunctive mode, safety requires every constraint to be provably violated.
/// A concrete counterexample for any single constraint proves the property violated.
fn disjunctive_failure_to_final_status(
    status: &BabVerificationStatus,
    constraint_desc: &str,
) -> BabVerificationStatus {
    debug_assert!(
        !constraint_is_safety_proof(status),
        "verified status should not be routed to disjunctive failure handling"
    );

    match status {
        BabVerificationStatus::Violated {
            counterexample,
            output,
        } => BabVerificationStatus::Violated {
            counterexample: counterexample.clone(),
            output: output.clone(),
        },
        BabVerificationStatus::PotentialViolation => BabVerificationStatus::Unknown {
            reason: format!(
                "Constraint {} may hold; disjunctive safety requires all constraints to be provably violated",
                constraint_desc
            ),
        },
        BabVerificationStatus::Unknown { reason } => BabVerificationStatus::Unknown {
            reason: format!("Constraint {} unknown: {}", constraint_desc, reason),
        },
        BabVerificationStatus::Timeout => BabVerificationStatus::Unknown {
            reason: format!(
                "Constraint {} timed out before proving violation",
                constraint_desc
            ),
        },
        BabVerificationStatus::Verified => unreachable!(
            "verified status should not be routed to disjunctive failure handling"
        ),
    }
}

/// Compute final relational status from aggregation mode and safety-proof count.
fn finalize_relational_status(
    aggregation: AggregationMode,
    proved_violated_count: usize,
    total_constraint_count: usize,
    checked_constraints: usize,
) -> BabVerificationStatus {
    match aggregation {
        AggregationMode::Disjunctive => {
            if proved_violated_count == total_constraint_count && total_constraint_count > 0 {
                BabVerificationStatus::Verified
            } else {
                BabVerificationStatus::Unknown {
                    reason: format!(
                        "Only {}/{} constraints provably violated (disjunction requires all)",
                        proved_violated_count, total_constraint_count
                    ),
                }
            }
        }
        AggregationMode::Conjunctive => {
            if proved_violated_count > 0 {
                BabVerificationStatus::Verified
            } else {
                BabVerificationStatus::Unknown {
                    reason: format!(
                        "No constraint was provably violated ({} constraints checked)",
                        checked_constraints
                    ),
                }
            }
        }
    }
}

/// Check if a concrete output satisfies ALL conjunctive constraints (is unsafe).
///
/// For conjunctive unsafe properties, ALL constraints must hold for the input to
/// be a valid counterexample to safety. Returns true when ALL constraints are
/// satisfied at the output, meaning this is a genuine unsafe input.
///
/// Used by both graph and sequential paths for cross-validating per-constraint
/// BaB counterexamples against the full property. Part of #3209.
pub(super) fn check_unsafe_counterexample(
    output: &ArrayD<f32>,
    constraints: &[OutputConstraint],
) -> bool {
    /// Safely retrieve an output value by index. Returns `None` on OOB,
    /// preventing silent 0.0 substitution. Part of #4375.
    fn out_at(output: &ArrayD<f32>, i: usize) -> Option<f32> {
        output.iter().nth(i).copied()
    }
    for constraint in constraints {
        let holds = match constraint {
            OutputConstraint::LessEq(i, j) => {
                matches!((out_at(output, *i), out_at(output, *j)), (Some(yi), Some(yj)) if yi <= yj)
            }
            OutputConstraint::LessThan(i, j) => {
                matches!((out_at(output, *i), out_at(output, *j)), (Some(yi), Some(yj)) if yi < yj)
            }
            OutputConstraint::GreaterEq(i, j) => {
                matches!((out_at(output, *i), out_at(output, *j)), (Some(yi), Some(yj)) if yi >= yj)
            }
            OutputConstraint::GreaterThan(i, j) => {
                matches!((out_at(output, *i), out_at(output, *j)), (Some(yi), Some(yj)) if yi > yj)
            }
            OutputConstraint::LessEqConst(i, c) => {
                out_at(output, *i).is_some_and(|y| y <= *c as f32)
            }
            OutputConstraint::LessThanConst(i, c) => {
                out_at(output, *i).is_some_and(|y| y < *c as f32)
            }
            OutputConstraint::GreaterEqConst(i, c) => {
                out_at(output, *i).is_some_and(|y| y >= *c as f32)
            }
            OutputConstraint::GreaterThanConst(i, c) => {
                out_at(output, *i).is_some_and(|y| y > *c as f32)
            }
            _ => false, // unknown constraint variants cannot confirm unsafe
        };
        if !holds {
            return false;
        }
    }
    // All constraints hold → this is a genuine unsafe counterexample
    !constraints.is_empty()
}

/// Verify standard (non-relational) constraints.
// Justification: Standard verification entry point forwards CLI parameters (model, bounds,
// spec, config, verifier, BaB flags, attack config, engine) to inner dispatch.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_standard(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    effective_threshold: f32,
    const_output_idx: Option<usize>,
    output_dim: usize,
    use_relu_split: bool,
    gpu_bab: bool,
    verifier: &BetaCrownVerifier,
    gemm_engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Result<BetaCrownResult> {
    match model_net {
        BetaCrownModel::Sequential(network) => {
            if let Some(output_idx) = const_output_idx {
                if output_idx >= output_dim {
                    anyhow::bail!(
                        "VNNLIB output index {} out of range (model outputs={})",
                        output_idx,
                        output_dim
                    );
                }
                // Build augmented network via shared eval helper (#1881 Step 4)
                let mut coeffs = vec![0.0f32; output_dim];
                coeffs[output_idx] = 1.0;
                let augmented = augment_network_with_spec(network, coeffs)?;
                Ok(verifier.verify_with_engine(
                    &augmented,
                    input,
                    effective_threshold,
                    gemm_engine,
                    deadline,
                )?)
            } else {
                Ok(verifier.verify_with_engine(
                    network,
                    input,
                    effective_threshold,
                    gemm_engine,
                    deadline,
                )?)
            }
        }
        BetaCrownModel::Graph(graph) => {
            let output_idx = const_output_idx.ok_or_else(|| {
                anyhow::anyhow!(
                    "GraphNetwork β-CROWN requires a property with an explicit output index"
                )
            })?;
            if output_idx >= output_dim {
                anyhow::bail!(
                    "VNNLIB output index {} out of range (model outputs={})",
                    output_idx,
                    output_dim
                );
            }
            let mut objective = vec![0.0f32; output_dim];
            objective[output_idx] = 1.0;
            // Dispatch via shared engine adapter (#1881)
            Ok(dispatch_graph_constraint(
                verifier,
                graph,
                input,
                &objective,
                effective_threshold,
                use_relu_split,
                gpu_bab,
                None,
                gemm_engine,
                deadline,
            )?)
        }
    }
}
