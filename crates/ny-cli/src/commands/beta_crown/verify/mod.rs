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
/// Adaptive attack stall cutoff (#attack-stall, design S4): stop a disjunctive
/// PGD phase whose margin has plateaued and let the bound/BaB phases have the
/// rest. Default-inert; attack-budget only.
mod attack_stall;
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
/// Default-off exact adaptive complete cover for authenticated NN4SYS clause
/// boxes with two or three genuinely ranged input coordinates.
mod nn4sys_nd_cover;
/// Default-off exact dyadic complete-cover prepass for NN4SYS per-clause
/// boxes with one genuinely ranged input coordinate. A result is authoritative
/// only after all leaves in the eligible subset close.
mod nn4sys_scalar_cover;
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
mod tests_objective_direction;
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

use self::phase_budget::PhaseBudgetLedger;

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

/// Pin `verify_upper_bound` to the objective ENCODING this subtree uses.
///
/// `BetaCrownConfig::verify_upper_bound` does not describe the property — it
/// describes how the objective handed to the engine is written down. The
/// engine's stop test is `BetaCrownConfig::domain_is_verified_for_mode`:
/// `upper < threshold` when the flag is set, `lower > threshold` when it is not.
///
/// This CLI has exactly two encodings and only ONE of them is flag-driven:
///
/// * [`verify_standard`] passes the RAW one-hot objective `+Y_idx` together with
///   the spec's own constant, so the direction genuinely varies per constraint
///   (`Y_i >= c` unsafe ⇒ prove `upper(Y_i) < c`). `prepare_inputs` /
///   `constraint_plan::extract_constant_params` computes exactly that flag, and
///   that path keeps it.
///
/// * everything reachable from [`verify_relational_constraints_with_ledger`] builds its
///   objectives with `constraint_plan::build_constraint_objective`, which
///   SIGN-NORMALIZES every row into violation semantics: `Y_i >= c` becomes
///   `(-1·Y_i, -c)` and `Y_i <= c` becomes `(+1·Y_i, +c)`. Proving a row's
///   constraint impossible is then ALWAYS `lower(spec·Y) > threshold`, for
///   every row, regardless of the original comparator — i.e.
///   `verify_upper_bound == false`, unconditionally.
///
/// The two conventions used to be crossed. The flag was derived from the FIRST
/// constant constraint of the WHOLE spec and then handed, unchanged, to the
/// sign-normalized subtree. On the ml4acopf disjunction
/// `(or (>= Y_159 0.01) (<= Y_159 -0.01))` the first constraint is `>=`, so the
/// flag came out `true` and the per-clause lane tested `upper(-Y_159) < -0.01`,
/// i.e. `lower(Y_159) > 0.01` — the exact NEGATION of the clause it was asked to
/// refute (measured: both clauses logged `threshold: -0.01, verify_upper=true`).
///
/// That is not merely incomplete. On a box where the unsafe clause holds
/// everywhere the inverted test SUCCEEDS, reporting `Verified` for a property
/// that is actually violated — an unsound `unsat`.
///
/// The graph multi-objective engine is lower-bound-only: its domains pass a hard
/// `false` (`MultiObjectiveGraphBabDomain::update_bounds(.., false)`), and its
/// public ingress now rejects `verify_upper_bound=true` before any root or child
/// verdict can acquire authority. This helper states the same contract at the
/// entry to the sign-normalized CLI subtree, so the single-objective
/// per-constraint / per-clause / grouped-disjunctive lanes agree with it.
pub(super) fn config_for_normalized_objectives(config: &BetaCrownConfig) -> BetaCrownConfig {
    BetaCrownConfig {
        verify_upper_bound: false,
        ..config.clone()
    }
}

/// Verify with relational constraints (Y_i <= Y_j, Y_i >= Y_j, etc.)
// Justification: Verification dispatch forwards CLI-specified parameters (model, input,
// spec, config, verifier, flags, engine) that are assembled by different callers.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
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
    let ledger = PhaseBudgetLedger::new(timeout, config.phase_budget.clone())
        .with_static_mip_ineligibility(super::dispatch::graph_mip_statically_ineligible(model_net));
    verify_relational_constraints_with_ledger(
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
        // Test face: mirror the pre-quarantine world where the single engine
        // handle also reached the attack lanes (pre-resolved, no arming).
        super::attack_arming::AttackEngineSource::Static(gemm_engine),
        json,
        &ledger,
    )
}

/// Dispatch relational verification under an already-started authoritative
/// phase ledger. The CLI BaB entry uses this face so IBP/model setup time is
/// charged once and explicit-`bab` MIP policy reaches every nested lane.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_relational_constraints_with_ledger(
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
    attack_engine_source: super::attack_arming::AttackEngineSource<'_>,
    json: bool,
    ledger: &PhaseBudgetLedger,
) -> Result<BetaCrownResult> {
    // ENTRY to the sign-normalized objective subtree: every lane below builds
    // its rows with `build_constraint_objective`, whose violation-semantics
    // encoding is decided by `lower(spec·Y) > threshold` for EVERY row. Pin the
    // engine's direction flag to that encoding here — see
    // `config_for_normalized_objectives` for why the incoming flag (computed
    // from the raw spec for `verify_standard`) is the wrong one to carry in.
    // The verifier is rebuilt from the same config so the lanes that read
    // `verifier.config.verify_upper_bound` directly (grouped-disjunctive child
    // batches, relaxed clip, input-split scoring) agree with the stop test.
    let config = &config_for_normalized_objectives(config);
    let normalized_verifier = verifier.with_config_from(config.clone());
    let verifier = &normalized_verifier;

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
            attack_engine_source,
            json,
            ledger,
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
        attack_engine_source,
        json,
        ledger,
    )
}

/// Whether a sequential relational property needs the shared multi-row graph lane.
///
/// For an unsafe conjunction, proving any one normalized row impossible is enough
/// to discharge a subdomain, and the any-row certificate is never weaker than the
/// sequential max-diff one GIVEN THE SAME INTERMEDIATES (dense MaxPool lowers via
/// a single argmax row, `layers/pooling/max.rs:455-505`, so `lb(d_i*) <= max_j
/// lb(d_j)`).
///
/// But the intermediates are NOT the same. An input-split same-LHS property that
/// is reducible stays on the sequential engine precisely because that engine
/// recomputes CROWN-IBP intermediate bounds PER DOMAIN
/// (`engine/input_split.rs:484-501`), while the graph input-split engine
/// structurally forbids CROWN-IBP (`beta_crown/mod.rs:1544-1554`, "disabling
/// unsupported CROWN-IBP") and falls back to a forward-linear root map plus
/// per-domain IBP. Looser intermediates give looser ReLU triangles, so each input
/// split discharges less and the frontier outgrows the search.
///
/// Measured: dropping this exclusion (faa66c38) sent official acasxu prop_3/prop_4
/// from `unsat` in ~45-51s to `timeout` at 116s with ~71k domains explored; the
/// commit's own removed comment recorded 4_2/prop_3 at 672 domains / ~4s on the
/// sequential lane versus >100k domains / timeout on the graph lane. Restoring the
/// exclusion trades one SOUND certificate for another — faa66c38 added no verdict
/// authority (its comment: "adds no new verdict authority") — so this is a routing
/// choice, never a soundness one.
pub(super) fn should_upgrade_sequential_conjunction_to_graph(
    vnnlib: &VnnLibSpec,
    use_relu_split: bool,
) -> bool {
    if classify_constraints(vnnlib).aggregation != AggregationMode::Conjunctive
        || vnnlib.output_constraints.len() < 2
    {
        return false;
    }
    // Reducible same-LHS input-split properties (acasxu prop_2/3/4) keep the
    // sequential lane and its per-domain CROWN-IBP intermediates. ReLU splitting
    // and non-reducible conjunctions still upgrade.
    use_relu_split || sequential::normalize_same_lhs_reduction(vnnlib).is_none()
}

/// Adapt the bound bootstrap when a same-LHS Sequential conjunction is routed
/// through the Graph input-split engine.
///
/// Top-level capability resolution still sees a Sequential model, so an ACAS
/// invocation can arrive here with `use_crown_ibp=true` (as the proof-only trace
/// did). Carrying that bit literally across this later representation boundary
/// has a very different cost: with no precomputed map, graph spec-CROWN runs its
/// O(N^2) per-target DAG CROWN-IBP collector. On CPU the converted ACAS MLP can
/// spend the complete property deadline there before exploring a domain. A
/// forward-linear root map is a certified, bounded-cost bootstrap for this small
/// unary graph. Per-domain IBP enhancement then refreshes the references after
/// every input split, retaining plain-CROWN behavior instead of freezing root
/// intermediates.
///
/// Keep this adaptation restricted to the newly upgraded same-LHS input-split
/// route. Native Graph models, ReLU splitting, non-reducible conjunctions, and
/// explicitly selected alpha-CROWN retain their incoming configuration.
pub(super) fn config_for_sequential_conjunction_graph(
    config: &BetaCrownConfig,
    vnnlib: &VnnLibSpec,
    use_relu_split: bool,
) -> BetaCrownConfig {
    let mut graph_config = config.clone();
    if !use_relu_split
        && !config.use_alpha_crown
        && sequential::normalize_same_lhs_reduction(vnnlib).is_some()
    {
        graph_config.use_forward_bounds = true;
        graph_config.use_crown_ibp = false;
        graph_config.input_split_ibp_enhancement = true;
    }
    graph_config
}

/// Resolve the exact config used after the late Sequential-to-Graph
/// representation boundary. Keeping this as one planning seam lets both the
/// verifier and the effective-treatment evidence describe the same route.
pub(super) fn planned_sequential_conjunction_graph_config(
    config: &BetaCrownConfig,
    vnnlib: &VnnLibSpec,
    use_relu_split: bool,
) -> Option<BetaCrownConfig> {
    should_upgrade_sequential_conjunction_to_graph(vnnlib, use_relu_split)
        .then(|| config_for_sequential_conjunction_graph(config, vnnlib, use_relu_split))
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
    // #attack-steering-conjunctive: the falsification accelerator channel. Kept
    // SEPARATE from `gemm_engine` (the quarantined proof handle) all the way to
    // the upfront-PGD call site — see `graph::verify_graph_relational`.
    attack_engine_source: super::attack_arming::AttackEngineSource<'_>,
    json: bool,
    ledger: &PhaseBudgetLedger,
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
    // Same-LHS relational conjunctions belong on this path too. The sequential
    // reduction appends a synthetic MaxPool to the signed-difference rows. For an
    // unstable MaxPool window its certified lower relaxation routes through one
    // `argmax(lower)` row, discarding the other per-row CROWN certificates. The
    // graph lane retains every signed-difference row and closes a box only when
    // ANY row has a certified `lower > threshold`. This is the reference
    // `stop_criterion_batch_any` semantics and adds no new verdict authority.
    //
    // `use_relu_split` is threaded through unchanged below: input-split callers
    // still enter the graph input-split engine, not ReLU-split BaB.
    // Reference: designs/2026-03-05-joint-conjunctive-bab.md Phase 4.
    if let BetaCrownModel::Sequential(network) = model_net {
        if let Some(mut graph_config) =
            planned_sequential_conjunction_graph_config(config, vnnlib, use_relu_split)
        {
            // This is a late representation boundary: carry the adapted
            // config into a matching verifier and anchor it to the ledger's
            // actual remaining budget.  Passing the incoming verifier left
            // its forward/CROWN-IBP policy inconsistent with graph_config.
            graph_config.timeout = graph_config.timeout.min(ledger.remaining_for_engine());
            let graph_verifier = verifier.with_config_from(graph_config.clone());
            let total_constraints = vnnlib.output_constraints.len();
            let mut graph = GraphNetwork::from_sequential(network)?;
            graph.set_use_patches_mode(graph_config.use_patches());
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
                &graph_config,
                &graph_verifier,
                use_relu_split,
                gpu_bab,
                pgd_attack,
                timeout,
                gemm_engine,
                attack_engine_source,
                json,
                ledger,
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
            pgd_attack,
            timeout,
            gemm_engine,
            attack_engine_source,
            json,
            ledger,
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
            ledger,
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
        BabVerificationStatus::PotentialViolation { .. } => BabVerificationStatus::Unknown {
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
    // This is the final SAT authority used by attacks and exact-cell routes.
    // Non-finite network outputs are numerical failures, not witnesses; IEEE
    // comparisons such as `inf <= inf` must not confirm a counterexample.
    if output.is_empty() || output.iter().any(|value| !value.is_finite()) {
        return false;
    }

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
                c.is_finite() && out_at(output, *i).is_some_and(|y| f64::from(y) <= *c)
            }
            OutputConstraint::LessThanConst(i, c) => {
                c.is_finite() && out_at(output, *i).is_some_and(|y| f64::from(y) < *c)
            }
            OutputConstraint::GreaterEqConst(i, c) => {
                c.is_finite() && out_at(output, *i).is_some_and(|y| f64::from(y) >= *c)
            }
            OutputConstraint::GreaterThanConst(i, c) => {
                c.is_finite() && out_at(output, *i).is_some_and(|y| f64::from(y) > *c)
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
