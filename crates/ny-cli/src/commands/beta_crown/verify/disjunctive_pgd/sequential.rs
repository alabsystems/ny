// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential-model fast path for disjunctive PGD.
//!
//! Classification-style disjunctions often share a single target label. For
//! those shapes, a single PGD objective `max_j margin_j` is cheaper and more
//! effective than clause-by-clause SPSA over the generic disjunctive surface.

use anyhow::Result;
use ndarray::ArrayD;
use ny_core::GemmEngine;
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::{BetaCrownResult, Network, PgdAttacker, PgdConfig};
#[cfg(test)]
use ny_propagate::{PgdAlphaMode, PgdInitialization, PgdOptimizer};
use ny_tensor::BoundedTensor;
#[cfg(test)]
use std::time::Instant;

use super::{
    candidate_in_attack_box, candidate_output_is_finite, constraint_margin, find_satisfied_clause,
    make_violated_result,
};

/// A confirmed PGD counterexample: `(counterexample_input, model_output, satisfied_clause_index)`.
type PgdCounterexample = (ArrayD<f32>, ArrayD<f32>, usize);

/// Sequential multi-clause classification specs can be attacked directly as a
/// shared-target "one-vs-rest" objective instead of clause-by-clause SPSA.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DisjunctiveAttackKind {
    AnyComparisonGeTarget {
        target: usize,
        comparisons: Vec<usize>,
    },
    TargetGeAnyComparison {
        target: usize,
        comparisons: Vec<usize>,
    },
}

fn normalize_single_relational_clause(clause: &[OutputConstraint]) -> Option<(usize, usize, bool)> {
    match clause {
        [OutputConstraint::LessEq(i, j)] => Some((*i, *j, true)),
        [OutputConstraint::GreaterEq(i, j)] => Some((*i, *j, false)),
        _ => None,
    }
}

pub(crate) fn classify_disjunctive_attack(
    clauses: &[Vec<OutputConstraint>],
) -> Option<DisjunctiveAttackKind> {
    if clauses.is_empty() {
        return None;
    }

    let triples: Vec<(usize, usize, bool)> = clauses
        .iter()
        .map(|clause| normalize_single_relational_clause(clause))
        .collect::<Option<_>>()?;

    let all_le = triples.iter().all(|triple| triple.2);
    let all_ge = triples.iter().all(|triple| !triple.2);
    if !all_le && !all_ge {
        return None;
    }

    let pairs: Vec<(usize, usize)> = triples.iter().map(|triple| (triple.0, triple.1)).collect();
    let same_lhs = pairs.iter().all(|(lhs, _)| *lhs == pairs[0].0);
    let same_rhs = pairs.iter().all(|(_, rhs)| *rhs == pairs[0].1);

    if all_ge {
        if same_rhs {
            return Some(DisjunctiveAttackKind::AnyComparisonGeTarget {
                target: pairs[0].1,
                comparisons: pairs.iter().map(|(lhs, _)| *lhs).collect(),
            });
        }
        if same_lhs {
            return Some(DisjunctiveAttackKind::TargetGeAnyComparison {
                target: pairs[0].0,
                comparisons: pairs.iter().map(|(_, rhs)| *rhs).collect(),
            });
        }
    } else if same_lhs {
        return Some(DisjunctiveAttackKind::AnyComparisonGeTarget {
            target: pairs[0].0,
            comparisons: pairs.iter().map(|(_, rhs)| *rhs).collect(),
        });
    } else if same_rhs {
        return Some(DisjunctiveAttackKind::TargetGeAnyComparison {
            target: pairs[0].1,
            comparisons: pairs.iter().map(|(lhs, _)| *lhs).collect(),
        });
    }

    None
}

fn confirm_disjunctive_pgd_candidate(
    found_counterexample: bool,
    counterexample: Option<ArrayD<f32>>,
    output: Option<ArrayD<f32>>,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    attack_desc: &str,
    best_output_value: f32,
) -> Result<Option<PgdCounterexample>> {
    if !found_counterexample {
        return Ok(None);
    }

    let counterexample = counterexample.ok_or_else(|| {
        anyhow::anyhow!(
            "disjunctive PGD reported found_counterexample=true but did not return a counterexample tensor"
        )
    })?;
    let output = output.ok_or_else(|| {
        anyhow::anyhow!(
            "disjunctive PGD reported found_counterexample=true but did not return an output tensor"
        )
    })?;

    // This direct sequential lane owns its own confirmation funnel instead of
    // calling the generic `re_evaluate_and_confirm`, so it must apply the same
    // wrong-buffer/shape/non-finite input gate explicitly.
    if !candidate_in_attack_box(&counterexample, input) {
        tracing::warn!(
            "Rejected sequential disjunctive PGD candidate outside the attack box \
             (wrong-buffer guard, #witness-box-audit): attack={}",
            attack_desc,
        );
        return Ok(None);
    }
    // A constraint may ignore some model outputs. Rust's f32::max semantics can
    // otherwise hide a lone NaN while the serialized Y assignment remains
    // organizer-invalid. Refuse the entire output vector fail-closed.
    if !candidate_output_is_finite(&output) {
        tracing::warn!(
            "Rejected sequential disjunctive PGD candidate with non-finite model output: attack={}",
            attack_desc,
        );
        return Ok(None);
    }

    // Epsilon margin guard: reject borderline counterexamples where f32
    // accumulation-order divergence could cause sign flips between our
    // forward pass and the VNN-COMP reference evaluator. Part of #4375;
    // noise-scaled (fixed 1e-5 was below the measured ~3e-5 ny<->ORT
    // deviation on cora cifar10 — see noise_scaled_margin).
    let epsilon = super::noise_scaled_margin(&output);
    // Per-clause input-box gate (mirrors `re_evaluate_and_confirm`): a clause
    // stripped of its own input box (per-clause-box disjunctions — nn4sys
    // mscn/lindex bands, acasxu prop_6 disjunct boxes) is satisfied output-only
    // by witnesses OUTSIDE its box; the true unsafe region requires the input
    // to lie in the SAME clause's box. Without this, the classification
    // fast-path returned hull-interior false CEs that the trusted ORT gate then
    // rejected — downgrading a provable-unsat run to unknown before BaB ran.
    let confirmed_idx = clauses.iter().enumerate().find_map(|(idx, clause)| {
        let in_box = per_clause_input_bounds
            .get(idx)
            .map_or(true, |b| super::point_in_clause_box(&counterexample, b));
        (in_box
            && !clause.is_empty()
            && clause
                .iter()
                .all(|c| constraint_margin(c, &output) >= epsilon))
        .then_some(idx)
    });
    if let Some(clause_idx) = confirmed_idx {
        return Ok(Some((counterexample, output, clause_idx)));
    }
    if find_satisfied_clause(&output, clauses).is_some() {
        tracing::warn!(
            "Rejected disjunctive PGD candidate (margin < {:.0e} or witness outside the satisfied clause's input box): attack={}",
            epsilon,
            attack_desc,
        );
    } else {
        tracing::warn!(
            "Rejected disjunctive PGD candidate after full-spec confirmation: attack={} best_value={}",
            attack_desc,
            best_output_value
        );
    }
    Ok(None)
}

#[cfg(test)]
pub(super) fn disjunctive_pgd_config(
    num_restarts: usize,
    num_steps: usize,
    initialization: PgdInitialization,
    osi_steps: usize,
    deadline: Option<Instant>,
    restart_when_stuck: bool,
) -> PgdConfig {
    PgdConfig {
        num_restarts,
        num_steps,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: true,
        deadline,
        restart_when_stuck,
        initialization,
        osi_steps,
        optimizer: PgdOptimizer::SignedGradient,
        alpha_mode: PgdAlphaMode::Scalar(0.01),
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)] // Attack context + per-clause box gate for confirmation
pub(super) fn try_sequential_disjunctive_pgd_attack_with_config(
    network: &Network,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    pgd_config: PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<Option<BetaCrownResult>> {
    let Some(attack_kind) = classify_disjunctive_attack(clauses) else {
        return Ok(None);
    };

    // See verify/pgd.rs: substitute the shared CPU-variant engine on CPU-routed
    // instances so PGD batches restarts and large batched GEMMs reach the
    // cuBLAS f32 seam; sub-gate calls err and fall back per layer (speed-only).
    let gemm_engine = gemm_engine.or_else(|| Some(ny_gpu::shared_cpu_engine() as &dyn GemmEngine));
    let attacker = PgdAttacker::new_with_optional_engine(pgd_config, gemm_engine);

    let (attack_desc, result): (String, _) = match attack_kind {
        DisjunctiveAttackKind::AnyComparisonGeTarget {
            target,
            comparisons,
        } => {
            if !json {
                tracing::info!(
                    "\n  Running disjunctive classification PGD: find x where any Y_j >= Y_{} for j in {:?}",
                    target, comparisons
                );
            }
            let desc = format!("max(Y_j - Y_{})", target);
            let result =
                attacker.attack_disjunctive_greater_eq(network, input, target, &comparisons)?;
            (desc, result)
        }
        DisjunctiveAttackKind::TargetGeAnyComparison {
            target,
            comparisons,
        } => {
            if !json {
                tracing::info!(
                    "\n  Running disjunctive classification PGD: find x where any Y_{} >= Y_j for j in {:?}",
                    target, comparisons
                );
            }
            let desc = format!("max(Y_{} - Y_j)", target);
            let result =
                attacker.attack_disjunctive_less_eq(network, input, target, &comparisons)?;
            (desc, result)
        }
    };

    let best_output_value = result.best_output_value;
    let found_counterexample = result.found_counterexample;
    let counterexample = result.counterexample;
    let output = result.output;

    if let Some((counterexample, output, clause_idx)) = confirm_disjunctive_pgd_candidate(
        found_counterexample,
        counterexample,
        output,
        input,
        clauses,
        per_clause_input_bounds,
        &attack_desc,
        best_output_value,
    )? {
        if !json {
            tracing::info!(
                "  Disjunctive classification PGD found counterexample! {} satisfied clause {}",
                attack_desc.as_str(),
                clause_idx + 1
            );
        }
        return Ok(Some(make_violated_result(counterexample, output)));
    }

    if !json {
        tracing::info!(
            "  Disjunctive classification PGD: no confirmed counterexample found. Best {}: {}",
            attack_desc,
            best_output_value
        );
    }
    Ok(None)
}

#[cfg(test)]
mod confirmation_tests {
    use super::*;
    use ndarray::arr1;

    fn unit_box() -> BoundedTensor {
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid unit box")
    }

    fn always_satisfied_clause() -> Vec<Vec<OutputConstraint>> {
        vec![vec![OutputConstraint::GreaterEq(1, 0)]]
    }

    #[test]
    fn sequential_confirmation_rejects_out_of_attack_box_candidate() {
        let confirmed = confirm_disjunctive_pgd_candidate(
            true,
            Some(arr1(&[5.0_f32]).into_dyn()),
            Some(arr1(&[0.0_f32, 1.0_f32]).into_dyn()),
            &unit_box(),
            &always_satisfied_clause(),
            &[],
            "unit-test",
            1.0,
        )
        .expect("confirmation is fail-closed, not fallible");
        assert!(
            confirmed.is_none(),
            "the sequential fast path must not bypass the shared attack-box gate"
        );
    }

    #[test]
    fn sequential_confirmation_rejects_unused_non_finite_output() {
        let confirmed = confirm_disjunctive_pgd_candidate(
            true,
            Some(arr1(&[0.5_f32]).into_dyn()),
            // The first two outputs strongly satisfy the only clause; Y_2 is
            // deliberately unused so this pins the whole-vector finite gate.
            Some(arr1(&[0.0_f32, 1.0_f32, f32::NAN]).into_dyn()),
            &unit_box(),
            &always_satisfied_clause(),
            &[],
            "unit-test",
            1.0,
        )
        .expect("confirmation is fail-closed, not fallible");
        assert!(
            confirmed.is_none(),
            "an organizer-invalid NaN Y assignment must never be published"
        );
    }

    #[test]
    fn sequential_confirmation_preserves_genuine_in_box_candidate() {
        let candidate = arr1(&[0.5_f32]).into_dyn();
        let confirmed = confirm_disjunctive_pgd_candidate(
            true,
            Some(candidate.clone()),
            Some(arr1(&[0.0_f32, 1.0_f32]).into_dyn()),
            &unit_box(),
            &always_satisfied_clause(),
            &[],
            "unit-test",
            1.0,
        )
        .expect("confirmation succeeds")
        .expect("genuine candidate remains accepted");
        assert_eq!(
            confirmed.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            candidate.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(confirmed.2, 0);
    }
}
