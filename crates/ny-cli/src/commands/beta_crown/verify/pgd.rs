// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PGD attacks for constraint verification, both after reduced verification and
//! upfront before per-constraint iteration.

use anyhow::Result;
use ndarray::ArrayD;
use ny_core::GemmEngine;
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{
    BabVerificationStatus, BetaCrownResult, Network, PgdAttacker, PgdConfig, PgdResult,
};
#[cfg(test)]
use ny_propagate::{PgdAlphaMode, PgdInitialization, PgdOptimizer};
use ny_tensor::BoundedTensor;
#[cfg(test)]
use std::time::Instant;

use super::attack_budget::upfront_conjunctive_sampling_budget;

/// Classification of conjunctive constraint attack type.
enum AttackKind {
    /// target <= all comparisons (for example, `Y_0 <= Y_1 && Y_0 <= Y_2`)
    LessEq {
        target: usize,
        comparisons: Vec<usize>,
    },
    /// target >= all comparisons (for example, `Y_0 >= Y_1 && Y_0 >= Y_2`)
    GreaterEq {
        target: usize,
        comparisons: Vec<usize>,
    },
}

/// Normalize a relational constraint to (lhs, rhs, is_less_eq).
/// Returns None for strict or non-relational constraints.
fn normalize_relational(c: &OutputConstraint) -> Option<(usize, usize, bool)> {
    match c {
        OutputConstraint::LessEq(i, j) => Some((*i, *j, true)),
        OutputConstraint::GreaterEq(i, j) => Some((*i, *j, false)),
        _ => None,
    }
}

/// Check if all (lhs, rhs) pairs share a common LHS, returning (shared, rhs_list).
fn try_shared_lhs(pairs: &[(usize, usize)]) -> Option<(usize, Vec<usize>)> {
    let target = pairs.first()?.0;
    if pairs.iter().all(|(a, _)| *a == target) {
        let comps: Vec<usize> = pairs.iter().map(|(_, b)| *b).collect();
        Some((target, comps))
    } else {
        None
    }
}

/// Check if all (lhs, rhs) pairs share a common RHS, returning (shared, lhs_list).
fn try_shared_rhs(pairs: &[(usize, usize)]) -> Option<(usize, Vec<usize>)> {
    let target = pairs.first()?.1;
    if pairs.iter().all(|(_, b)| *b == target) {
        let comps: Vec<usize> = pairs.iter().map(|(a, _)| *a).collect();
        Some((target, comps))
    } else {
        None
    }
}

/// Classify conjunctive constraints into an attack kind, handling both shared-LHS
/// and shared-RHS patterns for `<=` and `>=` relations.
fn classify_conjunctive_attack(constraints: &[OutputConstraint]) -> Option<AttackKind> {
    if constraints.is_empty() {
        return None;
    }
    // Normalize all to (lhs, rhs, is_le); bail on non-relational constraints.
    let triples: Vec<(usize, usize, bool)> = constraints
        .iter()
        .map(normalize_relational)
        .collect::<Option<_>>()?;
    let all_le = triples.iter().all(|t| t.2);
    let all_ge = triples.iter().all(|t| !t.2);
    let pairs: Vec<(usize, usize)> = triples.iter().map(|t| (t.0, t.1)).collect();
    if all_le {
        // Same-LHS: Y_target <= Y_j for all j
        if let Some((t, c)) = try_shared_lhs(&pairs) {
            return Some(AttackKind::LessEq {
                target: t,
                comparisons: c,
            });
        }
        // Same-RHS: Y_i <= Y_target → Y_target >= Y_i
        if let Some((t, c)) = try_shared_rhs(&pairs) {
            return Some(AttackKind::GreaterEq {
                target: t,
                comparisons: c,
            });
        }
    } else if all_ge {
        // Same-LHS: Y_target >= Y_j for all j
        if let Some((t, c)) = try_shared_lhs(&pairs) {
            return Some(AttackKind::GreaterEq {
                target: t,
                comparisons: c,
            });
        }
        // Same-RHS: Y_i >= Y_target → Y_target <= Y_i
        if let Some((t, c)) = try_shared_rhs(&pairs) {
            return Some(AttackKind::LessEq {
                target: t,
                comparisons: c,
            });
        }
    }
    None
}

fn confirm_pgd_candidate(
    network: &Network,
    result: PgdResult,
    constraints: &[OutputConstraint],
    attack_desc: &str,
) -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
    let PgdResult {
        found_counterexample,
        counterexample,
        output,
        best_output_value,
        ..
    } = result;

    if !found_counterexample {
        return Ok(None);
    }

    let counterexample = counterexample.ok_or_else(|| {
        anyhow::anyhow!(
            "PGD reported found_counterexample=true but did not return a counterexample tensor"
        )
    })?;
    let output = output.ok_or_else(|| {
        anyhow::anyhow!(
            "PGD reported found_counterexample=true but did not return an output tensor"
        )
    })?;

    // Match alpha-beta-CROWN's `check_and_save_cex()` / `test_conditions()` flow:
    // optimization proposes candidates, but final verdicts still require a
    // full-spec check on the concrete output.
    if !super::check_unsafe_counterexample(&output, constraints) {
        tracing::warn!(
            "Rejected sequential PGD candidate after full-spec confirmation: attack={} best_value={}",
            attack_desc,
            best_output_value
        );
        return Ok(None);
    }

    // Part of #4419: independent re-validation via CPU-only forward pass.
    // The PGD evaluator used the same engine for optimization and output
    // computation. Re-evaluate with engine=None to confirm independently.
    let revalidated = evaluate_network(network, &counterexample, None)?;
    if super::check_unsafe_counterexample(&revalidated, constraints) {
        return Ok(Some((counterexample, revalidated)));
    }

    tracing::warn!(
        "Sequential PGD candidate passed full-spec but failed independent re-validation: \
         attack={} best_value={}",
        attack_desc,
        best_output_value
    );
    Ok(None)
}

#[cfg(test)]
fn legacy_pgd_config(
    num_restarts: usize,
    num_steps: usize,
    initialization: PgdInitialization,
    osi_steps: usize,
    deadline: Option<Instant>,
) -> PgdConfig {
    PgdConfig {
        num_restarts,
        num_steps,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: true,
        deadline,
        initialization,
        osi_steps,
        optimizer: PgdOptimizer::SignedGradient,
        alpha_mode: PgdAlphaMode::Scalar(0.01),
        ..Default::default()
    }
}

/// Dispatch a classified PGD attack shared by the upfront and pre-MIP helpers.
// Justification: keep the caller-controlled verification inputs explicit.
fn dispatch_classified_attack(
    network: &Network,
    input: &BoundedTensor,
    kind: &AttackKind,
    pgd_config: PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> (String, ny_core::Result<PgdResult>) {
    // CPU-routed instances arrive with gemm_engine=None; substitute the shared
    // CPU-variant engine so PGD still batches restarts (engine presence selects
    // the batched attack), and the batched GEMMs above the MACs gate reach the
    // cuBLAS f32 seam. Below the gate the engine errs per call and the
    // per-layer CPU fallbacks keep the previous numerics. Attack results are
    // never verdict-deciding without a concrete re-check, so this is
    // speed-only.
    let gemm_engine = gemm_engine.or_else(|| Some(ny_gpu::shared_cpu_engine() as &dyn GemmEngine));
    let attacker = PgdAttacker::new_with_optional_engine(pgd_config, gemm_engine);
    match kind {
        AttackKind::LessEq {
            target,
            comparisons,
        } => {
            if !json {
                println!(
                    "\n  Running conjunctive PGD attack: find x where Y_{} <= Y_j for all j in {:?}",
                    target, comparisons
                );
            }
            let desc = format!("max(Y_{} - Y_j)", target);
            let r = attacker.attack_conjunctive_less_eq(network, input, *target, comparisons);
            (desc, r)
        }
        AttackKind::GreaterEq {
            target,
            comparisons,
        } => {
            if !json {
                println!(
                    "\n  Running conjunctive PGD attack: find x where Y_{} >= Y_j for all j in {:?}",
                    target, comparisons
                );
            }
            let desc = format!("max(Y_j - Y_{})", target);
            let r = attacker.attack_conjunctive_greater_eq(network, input, *target, comparisons);
            (desc, r)
        }
    }
}

/// Try conjunctive PGD attack after reduced verification fails.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn try_conjunctive_pgd_attack(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    reduced_result: &BetaCrownResult,
    pgd_restarts: usize,
    pgd_steps: usize,
    initialization: PgdInitialization,
    osi_steps: usize,
    deadline: Option<Instant>,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<Option<BetaCrownResult>> {
    try_conjunctive_pgd_attack_with_config(
        network,
        input,
        vnnlib,
        reduced_result,
        legacy_pgd_config(pgd_restarts, pgd_steps, initialization, osi_steps, deadline),
        gemm_engine,
        json,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_conjunctive_pgd_attack_with_config(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    reduced_result: &BetaCrownResult,
    pgd_config: PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<Option<BetaCrownResult>> {
    let attack_kind = classify_conjunctive_attack(&vnnlib.output_constraints);

    if let Some(kind) = attack_kind {
        let (attack_desc, result) =
            dispatch_classified_attack(network, input, &kind, pgd_config, gemm_engine, json);
        match result {
            Ok(result) => {
                let best_output_value = result.best_output_value;
                if let Some((counterexample, output)) = confirm_pgd_candidate(
                    network,
                    result,
                    &vnnlib.output_constraints,
                    &attack_desc,
                )? {
                    if !json {
                        println!(
                            "    PGD found counterexample! {} passed full-spec confirmation",
                            attack_desc
                        );
                    }
                    return Ok(Some(BetaCrownResult {
                        result: BabVerificationStatus::Violated {
                            counterexample: counterexample.iter().copied().collect(),
                            output: output.iter().copied().collect(),
                        },
                        domains_explored: reduced_result.domains_explored,
                        domains_verified: reduced_result.domains_verified,
                        cuts_generated: reduced_result.cuts_generated,
                        max_depth_reached: reduced_result.max_depth_reached,
                        time_elapsed: reduced_result.time_elapsed,
                        output_bounds: reduced_result.output_bounds.clone(),
                    }));
                }
                if !json {
                    println!(
                        "    PGD: no confirmed counterexample found. Best {}: {}",
                        attack_desc, best_output_value
                    );
                }
            }
            Err(e) => {
                tracing::warn!("PGD error (conjunctive attack): {e}");
            }
        }
    }

    Ok(None)
}

/// Try conjunctive PGD upfront before per-constraint verification via a generic
/// sampling/SPSA phase followed by the classified `PgdAttacker` path.
// Justification: deadline, engine, and search budget stay explicit at the callsite.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn try_conjunctive_pgd_attack_upfront(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    pgd_restarts: usize,
    pgd_steps: usize,
    initialization: PgdInitialization,
    osi_steps: usize,
    deadline: Option<Instant>,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
    try_conjunctive_pgd_attack_upfront_with_config(
        network,
        input,
        vnnlib,
        legacy_pgd_config(pgd_restarts, pgd_steps, initialization, osi_steps, deadline),
        gemm_engine,
        json,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_conjunctive_pgd_attack_upfront_with_config(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    pgd_config: PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
    // Phase 1: Generic sampling+SPSA attack.
    // Keep the SPSA step floor, but respect explicitly tuned restart counts.
    let (sampling_restarts, sampling_steps) =
        upfront_conjunctive_sampling_budget(pgd_config.num_restarts, pgd_config.num_steps);
    let sampling_config = PgdConfig {
        num_restarts: sampling_restarts,
        num_steps: sampling_steps,
        ..pgd_config
    };
    if let Some(result) = super::pgd_sampling::try_conjunctive_sampling_attack(
        network,
        input,
        &vnnlib.output_constraints,
        &sampling_config,
        gemm_engine,
        json,
    )? {
        return Ok(Some(result));
    }

    // Phase 2: Classified PgdAttacker approach (original logic).
    if let Some(kind) = classify_conjunctive_attack(&vnnlib.output_constraints) {
        let (attack_desc, result) =
            dispatch_classified_attack(network, input, &kind, pgd_config, gemm_engine, json);
        match result {
            Ok(result) => {
                let best_output_value = result.best_output_value;
                if let Some((counterexample, output)) = confirm_pgd_candidate(
                    network,
                    result,
                    &vnnlib.output_constraints,
                    &attack_desc,
                )? {
                    if !json {
                        println!(
                            "  Conjunctive PGD found counterexample! {} passed full-spec confirmation",
                            attack_desc
                        );
                    }
                    return Ok(Some((counterexample, output)));
                }
                if !json {
                    println!(
                        "  Conjunctive PGD: no confirmed counterexample found. Best {}: {}",
                        attack_desc, best_output_value
                    );
                }
            }
            Err(e) => {
                tracing::warn!("Conjunctive PGD error (upfront attack): {e}");
            }
        }
        return Ok(None);
    }

    // Fallback: try constant-bound random sampling attack.
    if super::pgd_sampling::has_constant_bound_constraints(&vnnlib.output_constraints) {
        return super::pgd_sampling::try_constant_bound_attack(
            network,
            input,
            &vnnlib.output_constraints,
            pgd_config.num_restarts,
            gemm_engine,
            json,
        );
    }

    Ok(None)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_pgd_before_mip_with_candidate_with_config(
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    pgd_config: PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<super::PgdMipPrecheck> {
    // Phase 1: Generic sampling+SPSA attack.
    let (sampling_restarts, sampling_steps) =
        upfront_conjunctive_sampling_budget(pgd_config.num_restarts, pgd_config.num_steps);
    let sampling_config = PgdConfig {
        num_restarts: sampling_restarts,
        num_steps: sampling_steps,
        ..pgd_config
    };
    if let Some(result) = super::pgd_sampling::try_conjunctive_sampling_attack(
        network,
        input,
        &vnnlib.output_constraints,
        &sampling_config,
        gemm_engine,
        json,
    )? {
        return Ok(super::PgdMipPrecheck {
            confirmed_counterexample: Some(result),
            warm_start_candidate: None,
        });
    }

    // Phase 2: Classified PgdAttacker approach.
    if let Some(kind) = classify_conjunctive_attack(&vnnlib.output_constraints) {
        let (attack_desc, result) =
            dispatch_classified_attack(network, input, &kind, pgd_config, gemm_engine, json);
        match result {
            Ok(pgd_result) => {
                let best_output_value = pgd_result.best_output_value;
                // Preserve the best candidate input for warm-starting MIP even when
                // PGD does not prove a full counterexample (#3865).
                let warm_start_input = pgd_result.counterexample.clone();
                if let Some((counterexample, output)) = confirm_pgd_candidate(
                    network,
                    pgd_result,
                    &vnnlib.output_constraints,
                    &attack_desc,
                )? {
                    if !json {
                        println!(
                            "  Conjunctive PGD found counterexample! {} passed full-spec confirmation",
                            attack_desc
                        );
                    }
                    return Ok(super::PgdMipPrecheck {
                        confirmed_counterexample: Some((counterexample, output)),
                        warm_start_candidate: None,
                    });
                }
                if !json {
                    println!(
                        "  Conjunctive PGD: no confirmed counterexample found. Best {}: {}",
                        attack_desc, best_output_value
                    );
                }
                return Ok(super::PgdMipPrecheck {
                    confirmed_counterexample: None,
                    warm_start_candidate: warm_start_input,
                });
            }
            Err(e) => {
                tracing::warn!("Conjunctive PGD error (upfront attack): {e}");
            }
        }
        return Ok(super::PgdMipPrecheck::default());
    }

    // Fallback: try constant-bound random sampling attack.
    if super::pgd_sampling::has_constant_bound_constraints(&vnnlib.output_constraints) {
        if let Some(result) = super::pgd_sampling::try_constant_bound_attack(
            network,
            input,
            &vnnlib.output_constraints,
            pgd_config.num_restarts,
            gemm_engine,
            json,
        )? {
            return Ok(super::PgdMipPrecheck {
                confirmed_counterexample: Some(result),
                warm_start_candidate: None,
            });
        }
    }

    Ok(super::PgdMipPrecheck::default())
}

/// Evaluate network at a concrete point via the exact concrete forward.
/// Completes bd68815 — IBP `.lower()` of a point box is not the network value
/// (see disjunctive_pgd::evaluate_model).
pub(super) fn evaluate_network(
    network: &Network,
    point: &ArrayD<f32>,
    gemm_engine: Option<&dyn GemmEngine>,
) -> Result<ArrayD<f32>> {
    let input_bounds = BoundedTensor::concrete(point.clone())?;
    let output = network.propagate_concrete_point(&input_bounds, gemm_engine)?;
    Ok(output.center())
}
