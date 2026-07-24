// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential model verification paths.
//!
//! Handles Network (non-graph) verification including:
//! - Reduced max-difference verification for targeted robustness
//! - Per-constraint iteration with timeout budgeting
//! - PGD attack integration

use anyhow::Result;
use ndarray::Array2;
use ny_core::GemmEngine;
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{
    layers::{LinearLayer, MaxPool2dLayer, ReshapeLayer},
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BetaCrownVerifier, Layer as PropLayer,
    Network,
};
use ny_tensor::BoundedTensor;

use super::constraint_iter::{iterate_constraints, ConstraintIterConfig};
use super::disjunctive_pgd::beta_crown_pgd_config;
use super::phase_budget::PhaseBudgetLedger;
use super::{augment_network_with_spec, classify_constraints, AggregationMode};

/// Detect the same-LHS relational reduction pattern used by reduced max-diff
/// verification, and return its normalized form.
///
/// Normalizes all relational constraints to (lhs, rhs) pairs where the LHS index
/// is consistent across all constraints. LessEq(i, j) is equivalent to
/// GreaterEq(j, i), so when the literal same-LHS check fails, everything is
/// re-normalized to the "ge" direction (this handles ACAS Xu prop_2:
/// Y_1<=Y_0, Y_2<=Y_0, ... → Y_0>=Y_1, Y_0>=Y_2, ...).
///
/// Returns `Some((family, lhs_idx, sorted-deduped rhs_indices))` when the
/// reduction applies, `None` otherwise (e.g. any constant constraint present).
///
/// Also used by the conjunctive dispatch in `verify_relational_constraints_impl`:
/// when this pattern applies, the reduced max-diff BaB (min over x of
/// max_j(violation_j) > 0) decides joint-witness conjunctions — where the
/// verifying conjunct varies across the input box — far more efficiently than
/// the graph multi-objective any-row lane (MEASURED 2026-07-10: 4_2/prop_3
/// verifies in 672 max-diff domains vs >100k any-row domains/timeout).
pub(super) fn normalize_same_lhs_reduction(
    vnnlib: &VnnLibSpec,
) -> Option<(&'static str, usize, Vec<usize>)> {
    let mut rel_family: Option<&'static str> = None; // "ge" or "le"
    let mut lhs_idx: Option<usize> = None;
    let mut rhs_indices: Vec<usize> = Vec::new();

    // First pass: try same-LHS directly
    let mut same_lhs_ok = true;
    for c in &vnnlib.output_constraints {
        match c {
            OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                if rel_family.is_none() {
                    rel_family = Some("ge");
                }
                if rel_family != Some("ge") {
                    same_lhs_ok = false;
                    break;
                }
                if lhs_idx.is_none() {
                    lhs_idx = Some(*i);
                }
                if lhs_idx != Some(*i) {
                    same_lhs_ok = false;
                    break;
                }
                rhs_indices.push(*j);
            }
            OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                if rel_family.is_none() {
                    rel_family = Some("le");
                }
                if rel_family != Some("le") {
                    same_lhs_ok = false;
                    break;
                }
                if lhs_idx.is_none() {
                    lhs_idx = Some(*i);
                }
                if lhs_idx != Some(*i) {
                    same_lhs_ok = false;
                    break;
                }
                rhs_indices.push(*j);
            }
            _ => {
                same_lhs_ok = false;
                break;
            }
        }
    }

    // Second pass: if same-LHS failed, try normalizing LessEq(i,j) → GreaterEq(j,i)
    if !same_lhs_ok {
        rel_family = None;
        lhs_idx = None;
        rhs_indices.clear();
        for c in &vnnlib.output_constraints {
            // Normalize: flip LessEq/LessThan to GreaterEq/GreaterThan
            let (normalized_family, i, j) = match c {
                OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                    ("ge", *i, *j)
                }
                OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                    ("ge", *j, *i) // Flip: LessEq(i,j) → GreaterEq(j,i)
                }
                _ => return None,
            };
            if rel_family.is_none() {
                rel_family = Some(normalized_family);
            }
            if rel_family != Some(normalized_family) {
                return None;
            }
            if lhs_idx.is_none() {
                lhs_idx = Some(i);
            }
            if lhs_idx != Some(i) {
                return None;
            }
            rhs_indices.push(j);
        }
    }

    rhs_indices.sort_unstable();
    rhs_indices.dedup();
    let (Some(family), Some(lhs)) = (rel_family, lhs_idx) else {
        return None;
    };
    if rhs_indices.is_empty() {
        return None;
    }
    Some((family, lhs, rhs_indices))
}

/// Verify sequential model with relational constraints.
// Justification: Sequential verification forwards the same parameter set as graph
// verification — network, bounds, constraints, config, BaB/attack flags, engine.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_sequential_relational(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    pgd_attack: bool,
    pgd_restarts: usize,
    pgd_steps: usize,
    timeout: u64,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<BetaCrownResult> {
    let aggregation = classify_constraints(vnnlib).aggregation;
    let is_disjunction = aggregation == AggregationMode::Disjunctive;
    let ledger = PhaseBudgetLedger::new(timeout, config.phase_budget.clone());
    let start_time = std::time::Instant::now();
    let pgd_deadline = ledger.upfront_pgd_deadline();

    // Early PGD attack before any CROWN computation. For categories where most
    // instances are violated (e.g., soundnessbench), PGD finds counterexamples in
    // seconds, avoiding the expensive 40% reduced-verification CROWN pass on
    // Conv-heavy networks. This does NOT affect soundness: PGD only reports
    // violations, verified by an independent forward pass.
    //
    // Single-clause disjunctions (output_constraint_clauses.len() <= 1) are
    // semantically conjunctive — all constraints in the single clause must hold
    // simultaneously. PGD on the conjunctive interpretation is correct. (#3309)
    //
    // Match the graph path's #3781 budget split: reserve 80% of the total timeout
    // for reduced verification + BaB so expensive sequential PGD cannot consume
    // the whole wall-clock budget before verification begins.
    let is_single_clause = vnnlib.output_constraint_clauses.len() <= 1;
    if pgd_attack && (!is_disjunction || is_single_clause) {
        let early_pgd = super::pgd::try_conjunctive_pgd_attack_upfront_with_config(
            network,
            input,
            vnnlib,
            beta_crown_pgd_config(config, pgd_restarts, pgd_steps, pgd_deadline),
            gemm_engine,
            json,
        )?;
        if let Some((counterexample, output)) = early_pgd {
            return Ok(BetaCrownResult {
                result: BabVerificationStatus::Violated {
                    counterexample: counterexample.iter().copied().collect(),
                    output: output.iter().copied().collect(),
                },
                domains_explored: 0,
                domains_verified: 0,
                cuts_generated: 0,
                max_depth_reached: 0,
                time_elapsed: start_time.elapsed(),
                output_bounds: None,
            });
        }
    }

    // Reduced max-difference verification is only sound for conjunctive aggregation.
    if !is_disjunction {
        // Allocate reduced_verification_fraction of total timeout for reduced
        // verification. The max-diff augmented network (with MaxPool2d) can be
        // harder to verify than individual constraints. If inconclusive, we need
        // remaining time for per-constraint fallback. Without this limit, reduced
        // verification consumes the entire timeout.
        // Source: PhaseBudgetConfig.reduced_verification_fraction (default 0.40).
        //
        // EXCEPTION (MEASURED 2026-07-10): when the same-LHS reduction pattern
        // applies, the max-diff objective semantically SUBSUMES every
        // per-constraint check — per-constraint-i success means min_x m_i(x) > 0,
        // which implies min_x max_j m_j(x) > 0, the max-diff success criterion —
        // so reserving budget for the per-constraint fallback cannot recover
        // anything the max-diff BaB misses. Give the max-diff BaB the full
        // remaining wall-clock instead. This is what acasxu prop_2 needs: its
        // verifying conjunct varies across the input box, so per-constraint
        // decomposition is provably useless there (each conjunct individually
        // falsifiable — PGD found Y_1-Y_0 = -3.7e-4 < 0 on 1_5), while the 0.4
        // sub-budget starved the one engine that can decide it (26400 domains
        // at timeout under the old split).
        let reduction_applies = normalize_same_lhs_reduction(vnnlib).is_some();
        let reduced_timeout = if reduction_applies {
            ledger
                .remaining()
                .unwrap_or(std::time::Duration::from_secs(timeout))
        } else {
            std::time::Duration::from_secs(timeout).mul_f32(
                ledger
                    .policy()
                    .reduced_verification_fraction
                    .clamp(0.0, 1.0),
            )
        };
        let mut reduced_config = config.clone();
        reduced_config.timeout = reduced_timeout;
        let reduced_verifier = verifier.with_config_from(reduced_config);

        let reduced_result = try_reduced_verification(
            network,
            input,
            vnnlib,
            config,
            &reduced_verifier,
            pgd_attack,
            pgd_restarts,
            pgd_steps,
            pgd_deadline,
            gemm_engine,
            json,
        )?;

        if let Some(result) = reduced_result {
            match &result.result {
                BabVerificationStatus::Verified | BabVerificationStatus::Violated { .. } => {
                    return Ok(result);
                }
                BabVerificationStatus::Unknown { .. } | BabVerificationStatus::Timeout
                    if reduction_applies =>
                {
                    // Same-LHS reduction: per-constraint verification is subsumed
                    // by the max-diff objective (see the budget note above), so
                    // running it after an inconclusive full-budget max-diff BaB
                    // only burns wall-clock on checks that cannot succeed where
                    // max-diff failed. MEASURED 2026-07-10 (1_6/prop_2): the
                    // fall-through leaked ~25s into 3 useless per-constraint
                    // rounds after the max-diff phase. Return directly.
                    // PotentialViolation still falls through: it carries a
                    // concrete counterexample lead worth chasing in the
                    // per-constraint attack phases.
                    if !json {
                        println!(
                            "\nReduced (max-diff) verification inconclusive; skipping per-constraint fallback (subsumed by max-diff for same-LHS conjunctions)."
                        );
                    }
                    return Ok(result);
                }
                _ => {
                    // Reduced verification returned Unknown/PotentialViolation —
                    // fall through to per-constraint path which may succeed with
                    // tighter per-output bounds.
                    if !json {
                        println!(
                            "\nReduced verification inconclusive, falling back to per-constraint verification."
                        );
                    }
                }
            }
        }
    } else if !json {
        println!(
            "\nSkipping reduced relational verification for disjunctive property (requires all constraints provably violated)."
        );
    }

    // Fall back to per-constraint verification with remaining time budget.
    // Check actual Duration (not truncated as_secs()) to avoid false timeout
    // when sub-second time remains (e.g., 0.999s → as_secs()=0 → false exhaust).
    let remaining_duration = ledger.remaining().unwrap_or(std::time::Duration::ZERO);
    if timeout > 0 && remaining_duration.is_zero() {
        return Ok(BetaCrownResult {
            result: BabVerificationStatus::Timeout,
            domains_explored: 0,
            domains_verified: 0,
            cuts_generated: 0,
            max_depth_reached: 0,
            time_elapsed: start_time.elapsed(),
            output_bounds: None,
        });
    }
    let remaining_timeout = if timeout == 0 {
        0 // unbounded
    } else {
        remaining_duration.as_secs().max(1)
    };

    verify_sequential_per_constraint(
        network,
        input,
        vnnlib,
        aggregation,
        config,
        verifier,
        pgd_attack,
        pgd_restarts,
        pgd_steps,
        remaining_timeout,
        gemm_engine,
        json,
    )
}

/// Try reduced verification for common targeted-robustness properties.
// Justification: Reduced verification optimization requires the full verification
// context to fall back to standard verification if the reduction doesn't apply.
#[allow(clippy::too_many_arguments)]
fn try_reduced_verification(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    pgd_attack: bool,
    pgd_restarts: usize,
    pgd_steps: usize,
    pgd_deadline: Option<std::time::Instant>,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<Option<BetaCrownResult>> {
    let Some((family, lhs, rhs_indices)) = normalize_same_lhs_reduction(vnnlib) else {
        return Ok(None);
    };

    let num_outputs = vnnlib.num_outputs;
    let k = rhs_indices.len();

    if !json {
        let constraint_desc = match family {
            "ge" => format!("Y_{} >= Y_j for j in {:?}", lhs, rhs_indices),
            "le" => format!("Y_{} <= Y_j for j in {:?}", lhs, rhs_indices),
            _ => "unknown".to_string(),
        };
        println!(
            "\nRelational constraint reduction detected: {}",
            constraint_desc
        );
        println!(
            "Reducing to single objective: maxdiff = max_j(signed_diff_j), verify maxdiff > 0"
        );
    }

    // Build Linear layer computing signed differences, one per RHS:
    // - For unsafe (Y_lhs >= Y_rhs): signed_diff = Y_rhs - Y_lhs
    // - For unsafe (Y_lhs <= Y_rhs): signed_diff = Y_lhs - Y_rhs
    let mut weights = vec![0.0f32; k * num_outputs];
    for (row, &rhs) in rhs_indices.iter().enumerate() {
        let row_start = row * num_outputs;
        match family {
            "ge" => {
                weights[row_start + rhs] = 1.0;
                weights[row_start + lhs] = -1.0;
            }
            "le" => {
                weights[row_start + lhs] = 1.0;
                weights[row_start + rhs] = -1.0;
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unknown relational family '{}' (expected 'ge' or 'le')",
                    family
                ))
            }
        }
    }

    let spec_weight = Array2::from_shape_vec((k, num_outputs), weights).map_err(|e| {
        anyhow::anyhow!(
            "failed to build relational spec weight with shape ({k}, {num_outputs}): {e}"
        )
    })?;

    // Signed-difference sub-network: [orig] + Linear(k signed diffs), WITHOUT the
    // maxpool tail. Its k outputs ARE the per-conjunct signed diffs whose
    // pointwise max is the max-diff objective — the joint-margin closer runs one
    // CROWN-with-linear pass on it per domain (see JointMarginCloser).
    let mut truncated = network.clone();
    truncated.add_layer(PropLayer::Linear(LinearLayer::new(spec_weight, None)?));

    // Full max-diff net = signed-diff sub-net + [Reshape, MaxPool(1,k), Reshape].
    let mut augmented = truncated.clone();
    augmented.add_layer(PropLayer::Reshape(ReshapeLayer::new(vec![1, 1, k as i64])));
    augmented.add_layer(PropLayer::MaxPool2d(MaxPool2dLayer::new(
        (1, k),
        (1, k),
        (0, 0),
    )));
    augmented.add_layer(PropLayer::Reshape(ReshapeLayer::new(vec![1])));

    // JOINT-MARGIN closer (task #40): CROWN's per-domain bound routes the MaxPool
    // lower relaxation through a single conjunct, which diverges on acasxu prop_2
    // (diag c7126554). The closer certifies a tighter JOINT lower bound over ALL
    // conjuncts per domain (min_x max_j g_j via a sound dual certificate). Sound
    // (only ever raises a domain's lower bound).
    //
    // OPT-IN (NY_JOINT_MARGIN_LP=1), default-OFF — MEASURED NEGATIVE (task #40,
    // 1_5/prop_2, contention-matched A/B): the joint over the per-conjunct
    // CROWN-*backward affine* bounds NEVER beats the MaxPool baseline
    // (computed=20000/20000 domains, raw_improved=0, flipped=0, and up to 3.466
    // LOOSER). Root cause: the baseline scalar bound is rescued by
    // `tighten_crown_output` intersecting with the IBP-*forward* bound
    // `max_j(IBP_lower_j)`, which for acasxu is far tighter than the raw
    // CROWN-backward affine bounds the closer extracts (plain-IBP intermediates,
    // no alpha → loose ReLU relaxations). The joint aggregation needs tight
    // per-conjunct AFFINE bounds to express the x-coupling; here only the IBP
    // CONSTANTS are tight, and constants cannot couple. So the bottleneck is
    // per-conjunct affine looseness, not the aggregation — closing it needs
    // alpha / crown-ibp per-conjunct bounds (contraindicated for acasxu, #3453),
    // not this lever. Default-OFF makes the pipeline byte-identical to baseline;
    // the sound machinery + `NY_JOINT_MARGIN_DIAG` accounting are kept for
    // reproducibility and a future tight-affine integration.
    let joint_lp_enabled = std::env::var("NY_JOINT_MARGIN_LP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("on"))
        .unwrap_or(false);
    let reduced_result = if joint_lp_enabled && k >= 2 {
        let closer = ny_propagate::JointMarginCloser::new(std::sync::Arc::new(truncated), k);
        verifier
            .with_config_from(verifier.config.clone())
            .with_joint_margin_closer(std::sync::Arc::new(closer))
            .verify_with_engine(&augmented, input, 0.0, gemm_engine, None)?
    } else {
        verifier.verify_with_engine(&augmented, input, 0.0, gemm_engine, None)?
    };

    // If reduced verification couldn't prove property and PGD is enabled,
    // try conjunctive PGD attack to find counterexample
    if matches!(
        reduced_result.result,
        BabVerificationStatus::PotentialViolation
            | BabVerificationStatus::Unknown { .. }
            | BabVerificationStatus::Timeout
    ) && pgd_attack
    {
        if let Some(result) = super::pgd::try_conjunctive_pgd_attack_with_config(
            network,
            input,
            vnnlib,
            &reduced_result,
            beta_crown_pgd_config(config, pgd_restarts, pgd_steps, pgd_deadline),
            gemm_engine,
            json,
        )? {
            return Ok(Some(result));
        }
    }

    Ok(Some(reduced_result))
}

/// Verify sequential model with per-constraint verification.
// Justification: Per-constraint verification iterates each constraint with the full
// verification context — grouping into a struct would duplicate BetaCrownConfig.
#[allow(clippy::too_many_arguments)]
fn verify_sequential_per_constraint(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    aggregation: AggregationMode,
    config: &BetaCrownConfig,
    verifier: &BetaCrownVerifier,
    _pgd_attack: bool,
    _pgd_restarts: usize,
    _pgd_steps: usize,
    timeout: u64,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<BetaCrownResult> {
    // Use total constraint count (not just relational) since constants are now included (#1888)
    let total_constraint_count = vnnlib.output_constraints.len();
    let is_disjunction = aggregation == AggregationMode::Disjunctive;

    // Budget timeout across constraints:
    // - Conjunctive (SAFE if ANY violated): each constraint gets the full remaining time.
    //   Only ONE needs to succeed, so don't split the budget — try each with full time
    //   and stop on first success. Splitting penalizes the easy constraint that could
    //   be solved with more time (e.g., ACAS Xu prop_2 with 4 constraints: 116s/5 = 23s
    //   per constraint is often insufficient, but 116s for the easiest one may suffice).
    // - Disjunctive (SAFE if ALL violated): split evenly since all must be proved.
    //   The minimum loop threshold is 10ms (not 100ms) to handle specs with many
    //   constraints (e.g., lindex 400 constraints at 30s = 74ms each, sufficient
    //   for quick BaB/PGD counterexample search).
    let per_constraint_timeout = if !is_disjunction {
        // Conjunctive: use full timeout for each (early exit on first success)
        std::time::Duration::from_secs(timeout)
    } else if total_constraint_count > 0 {
        // Disjunctive: split evenly with buffer
        let n = total_constraint_count.min(u32::MAX as usize - 1) as u32;
        std::time::Duration::from_secs(timeout) / (n + 1)
    } else {
        std::time::Duration::from_secs(timeout)
    };

    if !json {
        println!("\nRunning β-CROWN with constraints...");
        if is_disjunction {
            println!("Disjunctive property: SAFE if ALL constraints are provably violated.");
            println!(
                "Timeout budget: {:.1}s per constraint ({} constraints)",
                per_constraint_timeout.as_secs_f64(),
                total_constraint_count
            );
        } else {
            println!("Conjunctive property: SAFE if ANY constraint is provably violated.");
            println!(
                "Early-exit strategy: full timeout ({:.1}s) per constraint, stop on first success ({} constraints)",
                per_constraint_timeout.as_secs_f64(),
                total_constraint_count
            );
        }
    }

    // Note: conjunctive PGD attack was already run in verify_sequential_relational
    // (line 52-75) before reduced verification. The upfront attack now includes
    // a generic sampling+SPSA phase that checks random start points and intermediate
    // steps against ALL constraints (#3209). Running it again here would duplicate
    // work with the same seed, so we skip straight to per-constraint BaB.

    let iter_config = ConstraintIterConfig {
        aggregation,
        overall_timeout: std::time::Duration::from_secs(timeout),
        per_constraint_timeout,
        min_timeout_ms: 10,
        total_constraint_count,
        num_outputs: vnnlib.num_outputs,
        base_config: config.clone(),
        parent_verifier: Some(verifier),
        engine: verifier.engine_arc(),
        json,
    };

    // Cross-validation closure: evaluate original (un-augmented) network at a
    // concrete counterexample input. When per-constraint BaB finds a counterexample
    // for constraint i, this checks if that same input satisfies ALL constraints —
    // making it a joint counterexample proving the property VIOLATED. Part of #3209.
    let eval_original = |cx_input: &[f32]| -> Result<ndarray::ArrayD<f32>> {
        let point = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(input.lower().shape()),
            cx_input.to_vec(),
        )?;
        super::pgd::evaluate_network(network, &point, gemm_engine)
    };

    iterate_constraints(
        &vnnlib.output_constraints,
        &iter_config,
        |dispatch| {
            let augmented = augment_network_with_spec(network, dispatch.spec_coeffs.to_vec())?;
            Ok(dispatch.verifier.verify_with_engine(
                &augmented,
                input,
                dispatch.threshold,
                gemm_engine,
                None, // TODO(#4321): thread CLI deadline for sequential per-constraint
            )?)
        },
        Some(&eval_original),
    )
}
