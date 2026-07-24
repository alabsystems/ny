// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Disjunctive PGD random-sampling + SPSA attack for multi-clause properties.
//! Mirrors alpha-beta-CROWN's upfront attack on the full disjunction before clause-wise BaB.

mod config;
mod rng;
mod sequential;

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::{
    project_to_bounds_in_place, BabVerificationStatus, BetaCrownResult, GraphNetwork, PgdConfig,
    PgdInitialization, PgdStepState,
};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};

use super::attack_budget::direct_disjunctive_classification_budget;
use super::BetaCrownModel;
pub(in crate::commands::beta_crown::verify) use config::beta_crown_pgd_config;
use rng::SimpleRng;
#[cfg(test)]
use sequential::disjunctive_pgd_config;
use sequential::try_sequential_disjunctive_pgd_attack_with_config;
#[cfg(test)]
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

#[cfg(test)]
pub(crate) fn classify_disjunctive_attack(
    clauses: &[Vec<OutputConstraint>],
) -> Option<DisjunctiveAttackKind> {
    sequential::classify_disjunctive_attack(clauses).map(|kind| match kind {
        sequential::DisjunctiveAttackKind::AnyComparisonGeTarget {
            target,
            comparisons,
        } => DisjunctiveAttackKind::AnyComparisonGeTarget {
            target,
            comparisons,
        },
        sequential::DisjunctiveAttackKind::TargetGeAnyComparison {
            target,
            comparisons,
        } => DisjunctiveAttackKind::TargetGeAnyComparison {
            target,
            comparisons,
        },
    })
}

/// Evaluate a model at a concrete input point via the exact concrete forward.
///
/// Completes bd68815: the IBP `.lower()` of a point box is NOT the network
/// value — sound IBP widens even a degenerate input (Graph models with
/// LayerNorm/Softmax widen substantially; the sound Linear path widens by
/// ULPs), so `.lower()` fabricated false counterexamples that the trusted-ORT
/// gate then rejected, burning the instance (vit_2023, cgan_2023 class).
fn evaluate_model(
    model_net: &BetaCrownModel,
    point: &ArrayD<f32>,
    gemm_engine: Option<&dyn GemmEngine>,
) -> Result<ArrayD<f32>> {
    // ORT-routed candidate scoring (#four-walls): the trusted-runtime forward
    // is milliseconds where the internal per-layer walk costs 45ms+ on conv
    // nets, making the attack phase actually productive. Falls back to the
    // internal point forward when the oracle is unavailable. Sound: any
    // violation claim is re-confirmed by the independent vnncomp ORT gate
    // before a `sat` is scored. NY_ORT_ATTACK=0 disables.
    if let Some(output) = super::ort_attack::ort_forward_point(point) {
        return Ok(output);
    }
    let input_bounds = BoundedTensor::concrete(point.clone())?;
    let output = match model_net {
        BetaCrownModel::Sequential(network) => {
            network.propagate_concrete_point(&input_bounds, gemm_engine)?
        }
        BetaCrownModel::Graph(graph) => {
            graph.propagate_concrete_point(&input_bounds, gemm_engine, None)?
        }
    };
    Ok(output.center())
}

/// Minimum accepted counterexample margin, scaled to cross-implementation
/// forward noise. ny's exact f32 forward and ONNX Runtime disagree by up to
/// ~3e-5 absolute on ±14 logits (measured, cora_2024 cifar10-point) from
/// accumulation-order alone; a fixed 1e-5 guard is BELOW that noise, so PGD —
/// which adversarially maximizes ny's margin — lands borderline-robust
/// instances in [1e-5, ~6e-5], where the trusted-ORT gate reads them SAFE and
/// the whole instance burns on a false-counterexample downgrade. Higham-style:
/// scale at 1e-5 relative (5-50x over the ~2e-6 measured relative noise; 1e-4 relative would swallow engineered tight bands on large-magnitude outputs, e.g. nn4sys cardinalities), floored at 1e-4. A sat below this tolerance
/// could not survive the ORT gate anyway (and a false sat scores -150 vs +10).
pub(super) fn noise_scaled_margin(output: &ArrayD<f32>) -> f32 {
    let max_abs = output.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    (1e-4_f32).max(1e-5 * max_abs)
}

/// Retrieve an output value by index, failing closed on OOB coordinates (#4375).
fn output_at(output: &ArrayD<f32>, i: usize) -> Option<f32> {
    output.iter().nth(i).copied()
}

/// Compute a single-constraint satisfaction margin.
/// OOB coordinates map to `f32::NEG_INFINITY` so invalid clauses fail closed (#4375).
fn constraint_margin(constraint: &OutputConstraint, output: &ArrayD<f32>) -> f32 {
    match constraint {
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            match (output_at(output, *i), output_at(output, *j)) {
                (Some(yi), Some(yj)) => yi - yj,
                _ => f32::NEG_INFINITY,
            }
        }
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
            match (output_at(output, *i), output_at(output, *j)) {
                (Some(yi), Some(yj)) => yj - yi,
                _ => f32::NEG_INFINITY,
            }
        }
        OutputConstraint::GreaterEqConst(i, c) | OutputConstraint::GreaterThanConst(i, c) => {
            output_at(output, *i).map_or(f32::NEG_INFINITY, |y| y - *c as f32)
        }
        OutputConstraint::LessEqConst(i, c) | OutputConstraint::LessThanConst(i, c) => {
            output_at(output, *i).map_or(f32::NEG_INFINITY, |y| *c as f32 - y)
        }
        _ => 0.0, // Unknown variant: neutral margin
    }
}

/// Check if ALL constraints in a single clause are satisfied.
fn clause_satisfied(clause: &[OutputConstraint], output: &ArrayD<f32>) -> bool {
    !clause.is_empty() && clause.iter().all(|c| constraint_margin(c, output) >= 0.0)
}

/// Find the first clause in the disjunction that is satisfied at the given output.
fn find_satisfied_clause(output: &ArrayD<f32>, clauses: &[Vec<OutputConstraint>]) -> Option<usize> {
    clauses
        .iter()
        .position(|clause| clause_satisfied(clause, output))
}

/// Compute the minimum constraint margin for a clause (all constraints must hold).
fn clause_margin(clause: &[OutputConstraint], output: &ArrayD<f32>) -> f32 {
    clause
        .iter()
        .map(|c| constraint_margin(c, output))
        .fold(f32::INFINITY, f32::min)
}

/// Find the clause closest to being satisfied and its worst constraint for SPSA.
fn find_best_clause_target<'a>(
    output: &ArrayD<f32>,
    clauses: &'a [Vec<OutputConstraint>],
) -> Option<&'a OutputConstraint> {
    let mut best_idx = 0;
    let mut best_margin = f32::NEG_INFINITY;
    for (idx, clause) in clauses.iter().enumerate() {
        let margin = clause_margin(clause, output);
        if margin > best_margin {
            best_margin = margin;
            best_idx = idx;
        }
    }
    // Return the worst constraint in the best clause (the bottleneck for SPSA)
    clauses[best_idx]
        .iter()
        .min_by(|a, b| constraint_margin(a, output).total_cmp(&constraint_margin(b, output)))
}

/// Whether a concrete point lies in a clause's per-clause INPUT box.
///
/// The VNN-LIB normalizer strips per-clause input constraints out of the (output)
/// clauses and into `per_clause_input_bounds`, so a clause checked output-only is
/// satisfied by ANY witness whose output matches its band — even one whose input is
/// in a DIFFERENT clause's box. For nn4sys lindex/mscn (≈150 disjoint input boxes ×
/// 2 output bands) that yields a false counterexample for almost any sampled point,
/// which the trusted oracle then rejects → `unknown` instead of the provable `unsat`.
/// Re-pairing the witness with its clause's input box here removes those false CEs.
///
/// An empty box (no per-clause input restriction, e.g. classification benchmarks
/// with a single global input box) is vacuously satisfied, so this is a no-op there.
fn point_in_clause_box(
    point: &ArrayD<f32>,
    clause_box: &std::collections::BTreeMap<usize, (f64, f64)>,
) -> bool {
    clause_box.iter().all(|(&idx, &(lo, hi))| {
        point.iter().nth(idx).is_some_and(|&v| {
            let v = f64::from(v);
            // Tolerance: the witness may be projected onto the box boundary.
            v >= lo - 1e-6 && v <= hi + 1e-6
        })
    })
}

/// Independent re-evaluation: run the model fresh on the counterexample and
/// re-check constraints with an epsilon margin guard, requiring the witness to lie
/// in a satisfied clause's per-clause input box. Rejects borderline counterexamples
/// (f32 accumulation-order sign flips) and input-box-mismatched false CEs (#4375,
/// nn4sys per-clause-box fix). Wrong answers carry a VNN-COMP penalty; timeouts do
/// not — so rejecting borderline/mismatched SAT claims is the safe choice.
fn re_evaluate_and_confirm(
    model_net: &BetaCrownModel,
    counterexample: &ArrayD<f32>,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    gemm_engine: Option<&dyn GemmEngine>,
) -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
    let output = evaluate_model(model_net, counterexample, gemm_engine)?;
    // Reject borderline counterexamples: require margin >= epsilon on ALL
    // constraints in at least one clause AND the witness to lie in that clause's
    // per-clause input box (so a stripped output band is paired with its own input
    // sub-box, not some other clause's — the nn4sys lindex/mscn false-CE fix).
    // Noise-scaled: fixed 1e-5 was below the measured ny<->ORT forward deviation.
    let epsilon = noise_scaled_margin(&output);
    let confirmed = clauses.iter().enumerate().any(|(idx, clause)| {
        let in_box = per_clause_input_bounds
            .get(idx)
            .map_or(true, |b| point_in_clause_box(counterexample, b));
        in_box
            && !clause.is_empty()
            && clause
                .iter()
                .all(|c| constraint_margin(c, &output) >= epsilon)
    });
    if confirmed {
        Ok(Some((counterexample.clone(), output)))
    } else {
        Ok(None)
    }
}

/// Default-ON kill switch for the sequential exact-gradient disjunctive PGD lane.
/// `NY_DISJ_EXACT_PGD=0` disables (falls back to the legacy SPSA/sampling paths).
/// Attack-only: gates candidate GENERATION, never confirmation or verdicts.
fn sequential_exact_disjunctive_pgd_enabled() -> bool {
    std::env::var("NY_DISJ_EXACT_PGD").ok().as_deref() != Some("0")
}

/// Margin/budget telemetry from the disjunctive attack lanes that support it
/// (currently the graph / graph-lowered exact-gradient PGD). Consumed by the
/// adaptive attack-extension decision in `disjunctive.rs` (#attack-extend).
///
/// `best_margin: None` means no lane reported margin telemetry (e.g. the
/// legacy sampling/SPSA fallbacks) — the extension then never fires, i.e.
/// status-quo handoff to BaB. ATTACK-ONLY: this steers attack budget, never
/// bounds or verdicts.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct DisjunctiveAttackFeedback {
    /// Closest-to-violation clause margin observed (`>= 0` ⇒ candidate).
    pub best_margin: Option<f32>,
    /// True when an attack lane stopped on its phase deadline (budget-bound).
    pub hit_deadline: bool,
    /// Extension-fire diagnostics: restarts started / PGD steps taken.
    pub restarts_started: usize,
    pub steps_taken: usize,
}

impl DisjunctiveAttackFeedback {
    fn absorb(&mut self, outcome: &super::graph_pgd::GraphDisjunctiveAttackOutcome) {
        if outcome.best_margin > f32::NEG_INFINITY {
            self.best_margin = Some(
                self.best_margin
                    .map_or(outcome.best_margin, |b| b.max(outcome.best_margin)),
            );
        }
        self.hit_deadline |= outcome.hit_deadline;
        self.restarts_started += outcome.restarts_started;
        self.steps_taken += outcome.steps_taken;
    }
}

fn make_violated_result(counterexample: ArrayD<f32>, output: ArrayD<f32>) -> BetaCrownResult {
    BetaCrownResult {
        result: BabVerificationStatus::Violated {
            counterexample: counterexample.iter().copied().collect(),
            output: output.iter().copied().collect(),
        },
        domains_explored: 0,
        time_elapsed: Duration::from_secs(0),
        max_depth_reached: 0,
        output_bounds: None,
        cuts_generated: 0,
        domains_verified: 0,
    }
}

fn sample_uniform_point(
    shape: &[usize],
    lower: &[f32],
    upper: &[f32],
    rng: &mut SimpleRng,
) -> Result<ArrayD<f32>> {
    let vals: Vec<f32> = lower
        .iter()
        .zip(upper.iter())
        .map(|(&lo, &hi)| lo + rng.next_f32() * (hi - lo))
        .collect();
    Ok(ArrayD::from_shape_vec(IxDyn(shape), vals)?)
}

fn initialize_attack_point(
    pgd_config: &PgdConfig,
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    lower: &[f32],
    upper: &[f32],
    shape: &[usize],
    rng: &mut SimpleRng,
    gemm_engine: Option<&dyn GemmEngine>,
) -> Result<ArrayD<f32>> {
    let mut x = sample_uniform_point(shape, lower, upper, rng)?;
    if !matches!(pgd_config.initialization, PgdInitialization::Osi) || pgd_config.osi_steps == 0 {
        return Ok(x);
    }

    let probe_output = evaluate_model(model_net, &x, gemm_engine)?;
    let output_dim = probe_output.len();
    let w: Vec<f32> = (0..output_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let spsa_delta = pgd_config
        .suggested_spsa_delta(input)
        .max(pgd_config.spsa_delta);
    let mut step_state =
        PgdStepState::new_signed_gradient(pgd_config.alpha_mode, pgd_config.step_size, input);

    for _step in 0..pgd_config.osi_steps {
        let delta =
            ArrayD::from_shape_fn(IxDyn(shape), |_| if rng.next_bool() { 1.0 } else { -1.0 });
        let mut x_plus = &x + &(&delta * spsa_delta);
        let mut x_minus = &x - &(&delta * spsa_delta);
        project_to_bounds_in_place(&mut x_plus, input.lower(), input.upper());
        project_to_bounds_in_place(&mut x_minus, input.lower(), input.upper());

        let out_plus = evaluate_model(model_net, &x_plus, gemm_engine)?;
        let out_minus = evaluate_model(model_net, &x_minus, gemm_engine)?;
        let f_plus: f32 = out_plus.iter().zip(w.iter()).map(|(&o, &wi)| o * wi).sum();
        let f_minus: f32 = out_minus.iter().zip(w.iter()).map(|(&o, &wi)| o * wi).sum();
        if !f_plus.is_finite() || !f_minus.is_finite() {
            continue;
        }

        let sign_diff = if f_plus > f_minus {
            1.0_f32
        } else if f_plus < f_minus {
            -1.0_f32
        } else {
            0.0_f32
        };
        let pseudo_gradient = &delta * sign_diff;
        x = step_state.step(&pseudo_gradient, &x, input, true);
    }

    Ok(x)
}

/// Run a global sampling+SPSA attack on the disjunctive property.
///
/// For disjunctive properties like `(OR (Y_0 >= Y_1) (Y_2 >= Y_1) ...)`,
/// we need to find ANY input where ANY clause is satisfied. This is equivalent
/// to finding an adversarial example (misclassification).
///
/// Two-phase strategy:
/// - Phase 1: Pure random sampling (1 forward pass per sample). For large-epsilon
///   classification problems (e.g., traffic_signs eps=15), random perturbations
///   almost certainly cause misclassification. Maximizing random evaluations is
///   far more efficient than SPSA's 3 forward passes per gradient step.
/// - Phase 2: SPSA gradient optimization with the shared PGD config surface.
///
/// Reference: alpha-beta-CROWN complete_verifier.py runs PGD attack on the
/// full property before per-clause BaB dispatch.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn try_disjunctive_sampling_attack(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    num_restarts: usize,
    num_steps: usize,
    initialization: PgdInitialization,
    osi_steps: usize,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
    deadline: Option<Instant>,
    restart_when_stuck: bool,
) -> Result<Option<BetaCrownResult>> {
    try_disjunctive_sampling_attack_with_config(
        model_net,
        input,
        clauses,
        &[],
        disjunctive_pgd_config(
            num_restarts,
            num_steps,
            initialization,
            osi_steps,
            deadline,
            restart_when_stuck,
        ),
        gemm_engine,
        json,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_disjunctive_sampling_attack_with_config(
    model_net: &BetaCrownModel,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    per_clause_input_bounds: &[std::collections::BTreeMap<usize, (f64, f64)>],
    pgd_config: PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
    mut feedback: Option<&mut DisjunctiveAttackFeedback>,
) -> Result<Option<BetaCrownResult>> {
    if clauses.is_empty() {
        return Ok(None);
    }

    // Sequential exact-gradient disjunctive PGD (the cora_2024-class `sat` lever;
    // default ON, kill-switch `NY_DISJ_EXACT_PGD=0`). The sequential disjunctive
    // path previously had NO true-gradient attack: the classification fast path
    // below is SPSA-driven and the final fallback is random sampling + SPSA —
    // zero-order methods that miss classic small-eps MLP counterexamples
    // (probe-reproduced: 59 cora_2024 gt=sat instances stayed unknown while every
    // winning tool lands plain PGD counterexamples in 6-12s). Lower the layer
    // chain into a GraphNetwork and reuse the graph disjunctive PGD, which drives
    // the easiest disjunct with the exact point-Jacobian VJP
    // (`attack_point_gradient`). Only engages when the lowered graph is inside the
    // exact-gradient fragment — otherwise the legacy paths below run unchanged.
    // Attack-only: any candidate must still pass `re_evaluate_and_confirm` here
    // AND the trusted-ORT vnncomp gate before a `sat` is emitted.
    if let BetaCrownModel::Sequential(network) = model_net {
        if sequential_exact_disjunctive_pgd_enabled() {
            match GraphNetwork::from_sequential(network) {
                Ok(graph) if super::graph_pgd::graph_supports_exact_gradients(&graph) => {
                    let outcome = super::graph_pgd::try_graph_disjunctive_pgd_attack(
                        &graph,
                        input,
                        clauses,
                        &pgd_config,
                        gemm_engine,
                        json,
                    )?;
                    if let Some(fb) = feedback.as_deref_mut() {
                        fb.absorb(&outcome);
                    }
                    if let Some(candidate) = outcome.candidate {
                        if let Some((cx, out)) = re_evaluate_and_confirm(
                            model_net,
                            &candidate,
                            clauses,
                            per_clause_input_bounds,
                            gemm_engine,
                        )? {
                            if !json {
                                println!(
                                    "  Disjunctive attack: counterexample confirmed by sequential exact-gradient PGD!"
                                );
                            }
                            return Ok(Some(make_violated_result(cx, out)));
                        }
                    }
                }
                // Outside the exact-gradient fragment: skip (no SPSA duplication);
                // the legacy classification/sampling/SPSA paths below run unchanged.
                Ok(_) => {}
                Err(e) => {
                    tracing::debug!(
                        "sequential-to-graph lowering for exact disjunctive PGD failed: {e}"
                    );
                }
            }
        }
    }

    if let BetaCrownModel::Sequential(network) = model_net {
        let (attack_restarts, attack_steps) =
            direct_disjunctive_classification_budget(pgd_config.num_restarts, pgd_config.num_steps);
        if let Some(result) = try_sequential_disjunctive_pgd_attack_with_config(
            network,
            input,
            clauses,
            per_clause_input_bounds,
            PgdConfig {
                num_restarts: attack_restarts,
                num_steps: attack_steps,
                ..pgd_config
            },
            gemm_engine,
            json,
        )? {
            return Ok(Some(result));
        }
    }

    // Graph (conv-resnet) gradient disjunctive PGD — the cifar100/tinyimagenet `sat`
    // lever. Without this, Graph models fall through to random sampling below, which
    // cannot find adversarial counterexamples at small `eps`. Any candidate is
    // re-evaluated + confirmed (and the verdict re-checked against the full
    // property), so this is sound regardless of attack internals (#vnncomp-sat).
    if let BetaCrownModel::Graph(graph) = model_net {
        let outcome = super::graph_pgd::try_graph_disjunctive_pgd_attack(
            graph,
            input,
            clauses,
            &pgd_config,
            gemm_engine,
            json,
        )?;
        if let Some(fb) = feedback {
            fb.absorb(&outcome);
        }
        if let Some(candidate) = outcome.candidate {
            if let Some((cx, out)) = re_evaluate_and_confirm(
                model_net,
                &candidate,
                clauses,
                per_clause_input_bounds,
                gemm_engine,
            )? {
                if !json {
                    println!(
                        "  Disjunctive attack: counterexample confirmed by graph gradient PGD!"
                    );
                }
                return Ok(Some(make_violated_result(cx, out)));
            }
        }
    }

    let lo = input.lower();
    let hi = input.upper();
    let lo_owned: Vec<f32>;
    let lo_slice: &[f32] = match lo.as_slice() {
        Some(s) => s,
        None => {
            lo_owned = lo.iter().copied().collect();
            &lo_owned
        }
    };
    let hi_owned: Vec<f32>;
    let hi_slice: &[f32] = match hi.as_slice() {
        Some(s) => s,
        None => {
            hi_owned = hi.iter().copied().collect();
            &hi_owned
        }
    };

    let random_budget = pgd_config
        .num_restarts
        .saturating_mul(pgd_config.num_steps.saturating_mul(3).saturating_add(1));
    if !json {
        println!(
            "\n  Running disjunctive attack ({} random samples + {} SPSA restarts × {} steps, {} clauses)...",
            random_budget,
            pgd_config.num_restarts,
            pgd_config.num_steps,
            clauses.len()
        );
    }

    let mut rng = SimpleRng::new(42);
    for sample in 0..random_budget {
        if pgd_config.deadline.is_some_and(|d| Instant::now() >= d) {
            if !json {
                println!(
                    "  Random sampling: deadline at sample {}/{}",
                    sample, random_budget
                );
            }
            return Ok(None);
        }

        let x = sample_uniform_point(lo.shape(), lo_slice, hi_slice, &mut rng)?;
        let output = evaluate_model(model_net, &x, gemm_engine)?;
        if find_satisfied_clause(&output, clauses).is_some() {
            if let Some((cx, out)) = re_evaluate_and_confirm(
                model_net,
                &x,
                clauses,
                per_clause_input_bounds,
                gemm_engine,
            )? {
                if !json {
                    println!(
                        "  Disjunctive attack: counterexample confirmed at random sample {}!",
                        sample,
                    );
                }
                return Ok(Some(make_violated_result(cx, out)));
            }
        }
    }

    let spsa_delta = pgd_config
        .suggested_spsa_delta(input)
        .max(pgd_config.spsa_delta);
    for restart in 0..pgd_config.num_restarts {
        if pgd_config.deadline.is_some_and(|d| Instant::now() >= d) {
            if !json {
                println!(
                    "  Disjunctive SPSA: deadline at restart {}/{}",
                    restart, pgd_config.num_restarts
                );
            }
            break;
        }

        let mut step_state = PgdStepState::from_config(
            pgd_config.optimizer,
            pgd_config.alpha_mode,
            pgd_config.step_size,
            pgd_config.adam,
            input,
            input.shape(),
        );
        let mut x = initialize_attack_point(
            &pgd_config,
            model_net,
            input,
            lo_slice,
            hi_slice,
            lo.shape(),
            &mut rng,
            gemm_engine,
        )?;
        let mut output = evaluate_model(model_net, &x, gemm_engine)?;
        if find_satisfied_clause(&output, clauses).is_some() {
            if let Some((cx, out)) = re_evaluate_and_confirm(
                model_net,
                &x,
                clauses,
                per_clause_input_bounds,
                gemm_engine,
            )? {
                if !json {
                    println!(
                        "  Disjunctive attack: counterexample confirmed at SPSA restart {} (random)!",
                        restart,
                    );
                }
                return Ok(Some(make_violated_result(cx, out)));
            }
        }

        for step in 0..pgd_config.num_steps {
            if pgd_config.deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            let Some(target_constraint) = find_best_clause_target(&output, clauses) else {
                break;
            };

            let pert_vals: Vec<f32> = (0..lo_slice.len())
                .map(|_| if rng.next_bool() { 1.0_f32 } else { -1.0_f32 })
                .collect();
            let perturbation = ArrayD::from_shape_vec(IxDyn(x.shape()), pert_vals)?;
            let mut x_plus = &x + &perturbation * spsa_delta;
            let mut x_minus = &x - &perturbation * spsa_delta;
            project_to_bounds_in_place(&mut x_plus, input.lower(), input.upper());
            project_to_bounds_in_place(&mut x_minus, input.lower(), input.upper());

            let out_plus = evaluate_model(model_net, &x_plus, gemm_engine)?;
            let out_minus = evaluate_model(model_net, &x_minus, gemm_engine)?;
            let margin_plus = constraint_margin(target_constraint, &out_plus);
            let margin_minus = constraint_margin(target_constraint, &out_minus);
            if margin_plus.is_nan() || margin_minus.is_nan() {
                continue;
            }

            let grad = &perturbation * ((margin_plus - margin_minus) / (2.0 * spsa_delta));
            let previous_x = if pgd_config.restart_when_stuck {
                Some(x.clone())
            } else {
                None
            };
            x = step_state.step(&grad, &x, input, true);
            if let Some(ref previous_x) = previous_x {
                if previous_x.iter().zip(x.iter()).all(|(&a, &b)| a == b) {
                    x = sample_uniform_point(lo.shape(), lo_slice, hi_slice, &mut rng)?;
                    step_state.reset();
                }
            }

            output = evaluate_model(model_net, &x, gemm_engine)?;
            if find_satisfied_clause(&output, clauses).is_some() {
                if let Some((cx, out)) = re_evaluate_and_confirm(
                    model_net,
                    &x,
                    clauses,
                    per_clause_input_bounds,
                    gemm_engine,
                )? {
                    if !json {
                        println!(
                            "  Disjunctive attack: counterexample confirmed at SPSA restart {}, step {}!",
                            restart, step,
                        );
                    }
                    return Ok(Some(make_violated_result(cx, out)));
                }
            }
        }
    }

    if !json {
        println!("  Disjunctive attack: no counterexample found.");
    }
    Ok(None)
}
