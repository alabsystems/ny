// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Random sampling + SPSA conjunctive PGD attack.
//!
//! Generic approach that handles ANY constraint type, not just same-target
//! relational patterns. Checks ALL constraints jointly after each step.
//! This addresses #3209 where per-constraint PGD finds individual violations
//! but never checks if any single input satisfies ALL constraints simultaneously.
//!
//! Mirrors the graph path's `try_graph_pgd_upfront` approach for sequential
//! networks (uses `Network` + IBP evaluation instead of `GraphNetwork`).

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::{project_to_bounds_in_place, Network, PgdConfig, PgdStepState};
use ny_tensor::BoundedTensor;
use std::time::Instant;

/// Satisfaction margin target for SPSA gradient optimization.
///
/// Positive margin = constraint satisfied, negative = violated.
enum MarginTarget {
    /// margin = Y_a - Y_b (positive when Y_a >= Y_b)
    Relational(usize, usize),
    /// margin = Y_i - c (positive when Y_i >= c)
    ConstLower(usize, f32),
    /// margin = c - Y_i (positive when Y_i <= c)
    ConstUpper(usize, f32),
}

impl MarginTarget {
    fn margin(&self, output: &ArrayD<f32>) -> f32 {
        match self {
            MarginTarget::Relational(a, b) => {
                let ya = output.iter().nth(*a).copied().unwrap_or(0.0);
                let yb = output.iter().nth(*b).copied().unwrap_or(0.0);
                ya - yb
            }
            MarginTarget::ConstLower(i, c) => output.iter().nth(*i).copied().unwrap_or(0.0) - c,
            MarginTarget::ConstUpper(i, c) => c - output.iter().nth(*i).copied().unwrap_or(0.0),
        }
    }
}

/// Find the constraint with the smallest satisfaction margin (most-violated or
/// least-satisfied). Returns the SPSA optimization target for that constraint.
fn find_worst_constraint_target(
    output: &ArrayD<f32>,
    constraints: &[OutputConstraint],
) -> Option<MarginTarget> {
    let mut min_margin = f32::INFINITY;
    let mut worst: Option<MarginTarget> = None;
    for c in constraints {
        let (target, margin) = match c {
            OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                let yi = output.iter().nth(*i).copied().unwrap_or(0.0);
                let yj = output.iter().nth(*j).copied().unwrap_or(0.0);
                (MarginTarget::Relational(*j, *i), yj - yi)
            }
            OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                let yi = output.iter().nth(*i).copied().unwrap_or(0.0);
                let yj = output.iter().nth(*j).copied().unwrap_or(0.0);
                (MarginTarget::Relational(*i, *j), yi - yj)
            }
            OutputConstraint::LessEqConst(i, c) | OutputConstraint::LessThanConst(i, c) => {
                let y = output.iter().nth(*i).copied().unwrap_or(0.0);
                (MarginTarget::ConstUpper(*i, *c as f32), *c as f32 - y)
            }
            OutputConstraint::GreaterEqConst(i, c) | OutputConstraint::GreaterThanConst(i, c) => {
                let y = output.iter().nth(*i).copied().unwrap_or(0.0);
                (MarginTarget::ConstLower(*i, *c as f32), y - *c as f32)
            }
            _ => continue, // skip unknown constraint variants
        };
        if margin < min_margin {
            min_margin = margin;
            worst = Some(target);
        }
    }
    worst
}

/// Simple xorshift64 RNG (avoids `rand` dependency).
struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 & 0xFFFF_FFFF) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    fn next_bool(&mut self) -> bool {
        self.next_u32() & 1 == 0
    }
}

/// Check whether the PGD deadline has been exceeded.
///
/// Previously checked only every 50 steps (`step % 50 == 49`), which caused
/// PGD to overshoot its 20% budget by 100+ seconds on expensive graph models
/// (soundnessbench: 58s actual vs 30s budget, cifar100: 176s vs 20s budget).
/// Each graph SPSA step involves expensive forward+backward passes that can
/// take seconds per step, so coarse deadline polling is insufficient.
///
/// Now checks every step. `Instant::now()` costs ~20ns, negligible vs step cost.
///
/// Part of #2206 (Packet D: tighten deadline enforcement).
pub(super) fn spsa_step_deadline_exceeded(_step: usize, deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|limit| Instant::now() >= limit)
}

/// Random sampling + SPSA conjunctive attack for sequential networks.
///
/// Generic approach that handles ANY constraint type:
/// 1. Samples random points and checks ALL constraints immediately
/// 2. Uses SPSA to optimize toward satisfying the worst constraint
/// 3. Checks ALL constraints after each step (early termination)
///
/// This addresses #3209 where the classified PgdAttacker approach:
/// - Never checks random start points (many violation regions are large)
/// - Never checks intermediate SPSA steps (only the final point)
/// - Uses two separate SPSA perturbations per step (higher variance)
// Justification: the sampling attack forwards verification context directly so
// the caller controls budget, deadline, and engine selection explicitly.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_conjunctive_sampling_attack(
    network: &Network,
    input: &BoundedTensor,
    constraints: &[OutputConstraint],
    pgd_config: &PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
    if constraints.is_empty() {
        return Ok(None);
    }

    let num_restarts = pgd_config.num_restarts;
    let num_steps = pgd_config.num_steps;
    let deadline = pgd_config.deadline;
    let spsa_delta = pgd_config
        .suggested_spsa_delta(input)
        .max(pgd_config.spsa_delta);
    let n = input.lower().len();
    let lo = input.lower();
    let hi = input.upper();
    // Materialize bounds to contiguous slices. When lower/upper are non-contiguous
    // (e.g., from reshape/slice/transpose), as_slice() returns None. Previously this
    // silently fell back to &[], causing PGD to search [0,1]^n. (#3263 F3)
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

    if !json {
        println!(
            "\n  Running conjunctive sampling+SPSA attack ({} restarts, {} steps)...",
            num_restarts, num_steps
        );
    }

    let mut rng = SimpleRng::new(42);

    for restart in 0..num_restarts {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            tracing::info!("Sampling PGD: deadline exceeded at restart {restart}/{num_restarts}");
            break;
        }
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let l = lo_slice[i];
                let h = hi_slice[i];
                l + rng.next_f32() * (h - l)
            })
            .collect();
        let mut x = ArrayD::from_shape_vec(IxDyn(lo.shape()), vals)?;
        let mut step_state = PgdStepState::from_config(
            pgd_config.optimizer,
            pgd_config.alpha_mode,
            pgd_config.step_size,
            pgd_config.adam,
            input,
            input.shape(),
        );

        let mut output = super::pgd::evaluate_network(network, &x, gemm_engine)?;
        if super::check_unsafe_counterexample(&output, constraints) {
            if !json {
                println!(
                    "  Conjunctive sampling: counterexample at restart {} (random)!",
                    restart
                );
            }
            return Ok(Some((x, output)));
        }

        for step in 0..num_steps {
            // Intra-step deadline check every 50 steps (#3781). This mirrors the
            // graph PGD guard and prevents one expensive restart from consuming
            // the entire sequential verification budget.
            if spsa_step_deadline_exceeded(step, deadline) {
                tracing::info!(
                    "Sampling PGD: deadline exceeded at restart {restart}, step {step}/{num_steps}"
                );
                return Ok(None);
            }
            let Some(target) = find_worst_constraint_target(&output, constraints) else {
                break;
            };

            let pert_vals: Vec<f32> = (0..n)
                .map(|_| if rng.next_bool() { 1.0_f32 } else { -1.0_f32 })
                .collect();
            let perturbation = ArrayD::from_shape_vec(IxDyn(x.shape()), pert_vals)?;

            let mut x_plus = &x + &perturbation * spsa_delta;
            let mut x_minus = &x - &perturbation * spsa_delta;
            project_to_bounds_in_place(&mut x_plus, lo, hi);
            project_to_bounds_in_place(&mut x_minus, lo, hi);
            let out_plus = super::pgd::evaluate_network(network, &x_plus, gemm_engine)?;
            let out_minus = super::pgd::evaluate_network(network, &x_minus, gemm_engine)?;

            let margin_plus = target.margin(&out_plus);
            let margin_minus = target.margin(&out_minus);

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
                    let vals: Vec<f32> = (0..n)
                        .map(|i| {
                            let l = lo_slice[i];
                            let h = hi_slice[i];
                            l + rng.next_f32() * (h - l)
                        })
                        .collect();
                    x = ArrayD::from_shape_vec(IxDyn(lo.shape()), vals)?;
                    step_state.reset();
                }
            }

            output = super::pgd::evaluate_network(network, &x, gemm_engine)?;
            if super::check_unsafe_counterexample(&output, constraints) {
                if !json {
                    println!(
                        "  Conjunctive sampling: counterexample at restart {}, step {}!",
                        restart, step
                    );
                }
                return Ok(Some((x, output)));
            }
        }
    }

    if !json {
        println!("  Conjunctive sampling+SPSA: no counterexample found.");
    }
    Ok(None)
}

/// Check if constraints contain any constant-bound constraints.
pub(super) fn has_constant_bound_constraints(constraints: &[OutputConstraint]) -> bool {
    constraints.iter().any(|c| {
        matches!(
            c,
            OutputConstraint::LessEqConst(..)
                | OutputConstraint::GreaterEqConst(..)
                | OutputConstraint::LessThanConst(..)
                | OutputConstraint::GreaterThanConst(..)
        )
    })
}

/// Count how many constant-bound constraints are satisfied by the output.
fn count_satisfied_const_bounds(output: &ArrayD<f32>, constraints: &[OutputConstraint]) -> usize {
    // Materialize to contiguous slice (#3263 F3).
    let out_owned: Vec<f32>;
    let out_flat: &[f32] = match output.as_slice() {
        Some(s) => s,
        None => {
            out_owned = output.iter().copied().collect();
            &out_owned
        }
    };
    constraints
        .iter()
        .filter(|c| match c {
            OutputConstraint::LessEqConst(idx, val) => {
                out_flat.get(*idx).is_some_and(|&y| y <= *val as f32)
            }
            OutputConstraint::GreaterEqConst(idx, val) => {
                out_flat.get(*idx).is_some_and(|&y| y >= *val as f32)
            }
            OutputConstraint::LessThanConst(idx, val) => {
                out_flat.get(*idx).is_some_and(|&y| y < *val as f32)
            }
            OutputConstraint::GreaterThanConst(idx, val) => {
                out_flat.get(*idx).is_some_and(|&y| y > *val as f32)
            }
            _ => true,
        })
        .count()
}

/// Try targeted sampling attack for constant-bound constraints.
///
/// For properties like soundnessbench where ALL constraints are Y_i <= c or Y_i >= c,
/// the relational PGD classifier returns None. This function handles that case by:
/// 1. Trying the center point (most likely to violate for soundness benchmarks)
/// 2. Trying lower/upper bound corner points
/// 3. Sampling a few random points
///
/// Limited to ~20 evaluations since IBP through Conv networks takes ~3-6s each.
pub(super) fn try_constant_bound_attack(
    network: &Network,
    input: &BoundedTensor,
    constraints: &[OutputConstraint],
    num_samples: usize,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
    let lower = input.lower();
    let upper = input.upper();
    let n = lower.len();
    // Cap at 20 evaluations — IBP through Conv is ~3-6s each
    let effective_samples = num_samples.min(20);

    if !json {
        println!(
            "\n  Running constant-bound sampling attack ({} points, {} constraints)...",
            effective_samples,
            constraints.len()
        );
    }

    // Materialize bounds to contiguous slices (#3263 F3).
    let lo_owned: Vec<f32>;
    let lo_slice: &[f32] = match lower.as_slice() {
        Some(s) => s,
        None => {
            lo_owned = lower.iter().copied().collect();
            &lo_owned
        }
    };
    let hi_owned: Vec<f32>;
    let hi_slice: &[f32] = match upper.as_slice() {
        Some(s) => s,
        None => {
            hi_owned = upper.iter().copied().collect();
            &hi_owned
        }
    };
    let mut best_satisfied_count = 0usize;
    let mut sample_idx = 0usize;

    // Helper: evaluate and check
    let mut try_point =
        |point: ArrayD<f32>, label: &str| -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
            let output = super::pgd::evaluate_network(network, &point, gemm_engine)?;
            // Use the unified constraint checker (#3269 F3) which checks ALL
            // constraint types (relational + constant-bound), not just constants.
            if super::check_unsafe_counterexample(&output, constraints) {
                if !json {
                    println!(
                        "  Constant-bound attack: found counterexample at {}!",
                        label
                    );
                }
                return Ok(Some((point, output)));
            }
            let satisfied = count_satisfied_const_bounds(&output, constraints);
            if satisfied > best_satisfied_count {
                best_satisfied_count = satisfied;
            }
            if !json && satisfied > 0 {
                println!(
                    "    {}: {}/{} constraints satisfied",
                    label,
                    satisfied,
                    constraints.len()
                );
            }
            Ok(None)
        };

    // Point 1: Center of input region
    let center_vec: Vec<f32> = (0..n)
        .map(|i| f32::midpoint(lo_slice[i], hi_slice[i]))
        .collect();
    let center = ArrayD::from_shape_vec(IxDyn(lower.shape()), center_vec)?;
    if let Some(result) = try_point(center, "center")? {
        return Ok(Some(result));
    }
    sample_idx += 1;

    // Point 2: Lower bound
    if sample_idx < effective_samples {
        if let Some(result) = try_point(lower.clone(), "lower-bound")? {
            return Ok(Some(result));
        }
        sample_idx += 1;
    }

    // Point 3: Upper bound
    if sample_idx < effective_samples {
        if let Some(result) = try_point(upper.clone(), "upper-bound")? {
            return Ok(Some(result));
        }
        sample_idx += 1;
    }

    // Remaining: random points
    let mut rng = SimpleRng::new(42);
    while sample_idx < effective_samples {
        let point_vec: Vec<f32> = (0..n)
            .map(|i| lo_slice[i] + rng.next_f32() * (hi_slice[i] - lo_slice[i]))
            .collect();
        let point = ArrayD::from_shape_vec(IxDyn(lower.shape()), point_vec)?;
        if let Some(result) = try_point(point, &format!("random-{}", sample_idx - 2))? {
            return Ok(Some(result));
        }
        sample_idx += 1;
    }

    if !json {
        println!(
            "  Constant-bound attack: no counterexample found. Best: {}/{} constraints satisfied.",
            best_satisfied_count,
            constraints.len()
        );
    }

    Ok(None)
}
