// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph PGD attack with exact-gradient and SPSA gradient estimation.
//!
//! Extracted from `graph.rs` to stay under file size limit.
//! Uses exact reverse-mode gradients via concrete CROWN linear extraction
//! on a narrow ResNet-style DAG whitelist, falling back to SPSA on
//! unsupported graphs or when the CROWN relaxation is not locally exact.
//! Reference: alpha-beta-CROWN `general_spec_attack.py:312-337`.

use anyhow::Result;
use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::gama::{
    gama_guidance, gama_guidance_cotangent, gama_lambda_at, gama_lin_steps, gama_softmax,
};
use ny_propagate::layers::AddConstantLayer;
use ny_propagate::{
    project_to_bounds_in_place, GraphNetwork, GraphNode, Layer, PgdConfig, PgdStepState,
};
use ny_tensor::BoundedTensor;
use std::time::Instant;

#[cfg(test)]
pub(super) use super::graph_pgd_init::independent_graph_forward;
pub(super) use super::graph_pgd_init::{
    evaluate_graph, initialize_graph_point, revalidate_graph_counterexample, sample_center_point,
    sample_uniform_point, SimpleRng,
};
use super::pgd_sampling::spsa_step_deadline_exceeded;
#[cfg(test)]
use ny_propagate::{PgdAlphaMode, PgdInitialization, PgdOptimizer};

/// PGD target: which output margin to optimize (renamed from SpsaTarget for #4274).
#[derive(Clone, Copy)]
pub(super) enum GraphPgdTarget {
    Relational(usize, usize),
    Constant(usize, f32),
    NegConstant(usize, f32),
}

impl GraphPgdTarget {
    pub(super) fn margin(&self, output: &ArrayD<f32>) -> f32 {
        match self {
            Self::Relational(a, b) => {
                let ya = output.iter().nth(*a).copied().unwrap_or(0.0);
                let yb = output.iter().nth(*b).copied().unwrap_or(0.0);
                ya - yb
            }
            Self::Constant(i, c) => output.iter().nth(*i).copied().unwrap_or(0.0) - c,
            Self::NegConstant(i, c) => c - output.iter().nth(*i).copied().unwrap_or(0.0),
        }
    }

    /// Build a single-row specification matrix for this target.
    /// The row encodes `C @ y` such that the margin equals `C @ y` (possibly shifted by a constant).
    /// `num_outputs` is the total number of output dimensions.
    pub(super) fn to_spec_row(self, num_outputs: usize) -> Array2<f32> {
        let mut row = Array2::zeros((1, num_outputs));
        match self {
            Self::Relational(a, b) => {
                if a < num_outputs {
                    row[[0, a]] = 1.0;
                }
                if b < num_outputs {
                    row[[0, b]] = -1.0;
                }
            }
            Self::Constant(i, _) => {
                if i < num_outputs {
                    row[[0, i]] = 1.0;
                }
            }
            Self::NegConstant(i, _) => {
                if i < num_outputs {
                    row[[0, i]] = -1.0;
                }
            }
        }
        row
    }
}

pub(super) fn emit_graph_pgd_status(json: bool, message: std::fmt::Arguments<'_>) {
    if !json {
        println!("{message}");
    }
}

// Re-export from extracted module for sibling and test access.
pub(super) use super::graph_pgd_exact::{
    exact_graph_margin_gradient, exact_graph_spec_gradient, graph_supports_exact_gradients,
};

// ---------------------------------------------------------------------------
// Restart-batching whitelist (broader, for batched SPSA)
// ---------------------------------------------------------------------------

fn layer_supports_restart_batching(layer: &Layer) -> bool {
    matches!(
        layer,
        // Keep this whitelist in lockstep with the direct
        // `graph_pgd_preserve_leading_axis_matches_sequential_*` regressions.
        // Untested families stay on the sequential fallback path.
        //
        // Transpose is tentatively admitted (#4094): ONNX-style batch-preserving
        // perms (e.g. [0,2,1]) resolve correctly under a prepended restart axis.
        // Incompatible manual perms (e.g. [1,0] on rank-3) produce ShapeMismatch
        // at runtime and fall back to sequential via FallbackToSequential.
        Layer::Linear(_)
            | Layer::Conv1d(_)
            | Layer::Conv2d(_)
            | Layer::ConvTranspose1d(_)
            | Layer::AveragePool(_)
            | Layer::MaxPool2d(_)
            | Layer::ReLU(_)
            | Layer::LeakyReLU(_)
            | Layer::Exp(_)
            | Layer::Log(_)
            | Layer::BatchNorm(_)
            | Layer::MulBinary(_)
            | Layer::Add(_)
            | Layer::Sub(_)
            | Layer::Div(_)
            | Layer::AddConstant(_)
            | Layer::Reshape(_)
            | Layer::Flatten(_)
            | Layer::Transpose(_)
            | Layer::MulConstant(_)
            | Layer::Abs(_)
            | Layer::Sqrt(_)
            | Layer::DivConstant(_)
            | Layer::SubConstant(_)
            | Layer::PowConstant(_)
            | Layer::Tanh(_)
            | Layer::Sigmoid(_)
            | Layer::Softmax(_)
            | Layer::LogSoftmax(_)
    )
}

fn graph_supports_restart_batching(graph: &GraphNetwork) -> bool {
    graph.node_names().iter().all(|node_name| {
        graph
            .node(node_name)
            .is_some_and(|node| layer_supports_restart_batching(node.layer()))
    })
}

#[cfg(test)]
fn legacy_pgd_config(
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
        ..PgdConfig::default()
    }
}

/// Try upfront PGD-style random sampling attack on a GraphNetwork.
// Justification: graph PGD takes the full attack budget and execution context
// directly so callers can thread deadlines and engine selection explicitly.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn try_graph_pgd_upfront(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    num_restarts: usize,
    num_steps: usize,
    initialization: PgdInitialization,
    osi_steps: usize,
    deadline: Option<Instant>,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
    restart_when_stuck: bool,
) -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
    try_graph_pgd_upfront_with_config(
        graph,
        input,
        vnnlib,
        legacy_pgd_config(
            num_restarts,
            num_steps,
            initialization,
            osi_steps,
            deadline,
            restart_when_stuck,
        ),
        gemm_engine,
        json,
    )
}

pub(super) fn try_graph_pgd_upfront_with_config(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    pgd_config: PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<Option<(ArrayD<f32>, ArrayD<f32>)>> {
    let pgd_start = Instant::now();
    let engine_label = if gemm_engine.is_some() { "GPU" } else { "CPU" };
    let exact_grad_eligible = graph_supports_exact_gradients(graph);
    if !json {
        println!(
            "\n  Running graph PGD attack ({} restarts, {} steps, engine={}, exact_grad={})...",
            pgd_config.num_restarts, pgd_config.num_steps, engine_label, exact_grad_eligible
        );
    }

    // #soundnessbench routing (DEFAULT ON; `NY_PGD_EXACT_BATCHED=0` restores the
    // old unconditional batched dispatch): when the graph is in the exact-gradient
    // fragment, SKIP the generic batched attack — it is SPSA-ONLY (one-sample
    // finite-difference gradient of the single worst conjunct by default, or the
    // joint hinge plus guidance when GAMA is configured) and probe-proven unable
    // to land soundnessbench's adversarially-hidden counterexamples, while
    // the sequential loop below ascends the JOINT AND-clause margin with the exact
    // point-Jacobian (the exact lever α,β-CROWN uses to solve all 50 in 30-125s).
    // General by construction: the gate is gradient ELIGIBILITY, not a benchmark
    // check. Ineligible graphs keep the batched SPSA dispatch unchanged, so
    // categories that win via batched SPSA are unaffected. Attack-only either
    // way — every candidate still flows through revalidate_graph_counterexample.
    let route_exact_over_batched =
        exact_grad_eligible && std::env::var("NY_PGD_EXACT_BATCHED").ok().as_deref() != Some("0");

    if gemm_engine.is_some()
        && pgd_config.num_restarts > 1
        && graph_supports_restart_batching(graph)
        && !route_exact_over_batched
    {
        match super::graph_pgd_batched::try_graph_pgd_upfront_batched_with_config(
            graph,
            input,
            vnnlib,
            pgd_config.clone(),
            gemm_engine,
            json,
        )? {
            super::graph_pgd_batched::BatchedGraphPgdOutcome::Completed(result) => {
                return Ok(result.map(|b| *b));
            }
            super::graph_pgd_batched::BatchedGraphPgdOutcome::FallbackToSequential => {}
        }
    } else if route_exact_over_batched && gemm_engine.is_some() && pgd_config.num_restarts > 1 {
        emit_graph_pgd_status(
            json,
            format_args!(
                "  Graph PGD: joint exact-gradient attack ({} conjuncts; batched SPSA bypassed, NY_PGD_EXACT_BATCHED=0 restores)",
                vnnlib.output_constraints.len()
            ),
        );
        // #batched-vjp: run K deep restarts per step from ONE wide GPU pass
        // (exact joint-margin gradients). DEFAULT ON when the wide plan builds
        // for this graph shape; kill switch NY_PGD_VJP_BATCH=0. Any capability
        // miss / GPU failure falls through to the sequential loop below,
        // byte-identical to the pre-batched behavior.
        match super::graph_pgd_vjp_batched::try_graph_pgd_vjp_batched(
            graph,
            input,
            vnnlib,
            &pgd_config,
            gemm_engine,
            json,
        )? {
            super::graph_pgd_vjp_batched::VjpBatchedOutcome::Completed(result) => {
                return Ok(result);
            }
            super::graph_pgd_vjp_batched::VjpBatchedOutcome::FallbackToSequential => {}
        }
    }

    let constraints = &vnnlib.output_constraints;
    let spsa_delta = pgd_config
        .suggested_spsa_delta(input)
        .max(pgd_config.spsa_delta);

    // JOINT AND-clause objective (#soundnessbench): the modeled conjuncts of the
    // property, built once. Each step ascends Σ_c min(margin_c, 0) over ALL
    // unsatisfied conjuncts simultaneously (the α,β-CROWN joint-margin attack)
    // instead of only the single worst conjunct, which stalls when pushing one
    // margin up pushes another down. With exact gradients this costs ONE backward
    // pass per step: the objective's gradient is the SUM of the active conjuncts'
    // spec rows, and the network Jacobian is linear in the spec row.
    let conjunct_targets: Vec<GraphPgdTarget> =
        constraints.iter().filter_map(constraint_target).collect();
    // Unlike the disjunctive classification path, a one-clause VNNLIB property
    // reaches this upfront conjunctive path. Resolve GAMA here so constant
    // thresholds receive guidance without replacing their raw satisfaction
    // margins or changing the candidate screen.
    let gama_lambda0 = gama_lambda_init(&pgd_config);
    let gama_lin = gama_lin_steps(pgd_config.num_steps);

    // Determine output dimension count from the first evaluation.
    // Updated after first forward pass in each restart.
    let mut num_outputs: Option<usize> = None;

    // #soundnessbench diag (`NY_PGD_DIAG=1`): per-restart throughput + best joint
    // hinge loss (0 ⇒ every modeled conjunct satisfied). Lets us tell "attack
    // lands just short" from "gradient goes astray" without touching the hot loop.
    let diag = std::env::var("NY_PGD_DIAG").ok().as_deref() == Some("1");
    let mut best_joint_seen = f32::NEG_INFINITY;

    // #soundnessbench restart time-slicing (opt-in, default OFF): one exact-gradient
    // step on a deep conv-resnet is ~140ms, so a single `pgd_steps=1000` restart
    // consumes the whole attack budget. Slicing would trade that DEPTH for restart
    // DIVERSITY — but measurement shows these adversarially-hidden CEs need the
    // depth (a sliced 37-step restart stalls at joint≈-0.86; the full ~800-step
    // restart reaches the near-boundary point the ORT refinement then finishes).
    // So slicing stays OFF unless `NY_PGD_EXACT_SLICE_S` is set, where it caps each
    // restart to that many wall-seconds for experimentation. The step / overall
    // deadline still bound everything.
    let restart_slice: Option<std::time::Duration> = std::env::var("NY_PGD_EXACT_SLICE_S")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(std::time::Duration::from_secs_f64);
    // Minimum steps a sliced restart must take before the slice can cut it, so a
    // momentarily-slow first step never aborts a restart after 1-2 updates.
    const MIN_SLICE_STEPS: usize = 12;

    for restart in 0..pgd_config.num_restarts {
        if pgd_config.deadline.is_some_and(|d| Instant::now() >= d) {
            if !json {
                println!(
                    "  Graph PGD: deadline at restart {}/{} ({:.2}s, engine={})",
                    restart,
                    pgd_config.num_restarts,
                    pgd_start.elapsed().as_secs_f64(),
                    engine_label
                );
            }
            tracing::info!(
                "Graph PGD: deadline exceeded at restart {}/{}, returning",
                restart,
                pgd_config.num_restarts
            );
            break;
        }

        let restart_start = Instant::now();
        let mut rng = SimpleRng::new(42 + restart as u64);
        let mut step_state = PgdStepState::from_config(
            pgd_config.optimizer,
            pgd_config.alpha_mode,
            pgd_config.step_size,
            pgd_config.adam,
            input,
            input.shape(),
        );
        let mut x = initialize_graph_point(&pgd_config, graph, input, &mut rng, gemm_engine)?;
        let mut output = evaluate_graph(graph, &x, gemm_engine)?;
        let mut gama_p_ref = gama_lambda0.and_then(|_| gama_reference_softmax(&output));
        let mut gama_step = 0usize;
        if num_outputs.is_none() {
            num_outputs = Some(output.len());
        }
        // #postbab-seed: export the closest-to-violation point seen (guidance-only;
        // see `best_margin_export`). Cost is one O(conjuncts) pass on the output
        // vector — noise next to the network forward that produced it.
        if !conjunct_targets.is_empty() {
            let joint = joint_hinge_loss(&conjunct_targets, &output);
            super::super::best_margin_export::record_best_margin_candidate(joint, &x);
            if joint > best_joint_seen {
                best_joint_seen = joint;
            }
        }
        if super::check_unsafe_counterexample(&output, constraints) {
            let ctx = format!("restart {restart} (random sample)");
            if let Some(pair) =
                revalidate_graph_counterexample(graph, x.clone(), constraints, &ctx)?
            {
                emit_graph_pgd_status(
                    json,
                    format_args!("  Graph PGD found counterexample at {ctx}!"),
                );
                return Ok(Some(pair));
            }
        }

        let mut exact_grad_used = 0_usize;
        let mut spsa_used = 0_usize;

        for step in 0..pgd_config.num_steps {
            if spsa_step_deadline_exceeded(step, pgd_config.deadline) {
                if !json {
                    println!(
                        "  Graph PGD: deadline at restart {}, step {}/{} ({:.2}s, engine={})",
                        restart,
                        step,
                        pgd_config.num_steps,
                        pgd_start.elapsed().as_secs_f64(),
                        engine_label
                    );
                }
                tracing::info!(
                    "Graph PGD: deadline exceeded at restart {}, step {}/{}",
                    restart,
                    step,
                    pgd_config.num_steps
                );
                return Ok(None);
            }

            // Per-restart time slice: move to the next seed once this restart has
            // spent its slice (after a minimum number of steps). Diversity over
            // depth on the exact-routed path (see `restart_slice`).
            if step >= MIN_SLICE_STEPS
                && restart_slice.is_some_and(|slice| restart_start.elapsed() >= slice)
            {
                break;
            }

            // Joint objective: sum of hinge losses over ALL unsatisfied conjuncts.
            let Some(raw_spec_row) = joint_unsat_spec_row(
                &conjunct_targets,
                &output,
                num_outputs.unwrap_or(output.len()),
            ) else {
                break;
            };
            let gama_lambda = gama_lambda0
                .map(|lambda0| gama_lambda_at(lambda0, gama_step, gama_lin))
                .unwrap_or(0.0);
            let spec_row = if let Some(p_ref) = gama_p_ref.as_deref() {
                add_gama_guidance_to_spec_row(raw_spec_row, &output, p_ref, gama_lambda)
            } else {
                raw_spec_row
            };

            // Try exact gradient first, fall back to SPSA (#4274). The joint
            // objective's gradient is one backward pass on the summed spec row.
            let grad = if exact_grad_eligible {
                match exact_graph_spec_gradient(
                    graph,
                    &x,
                    &output,
                    &spec_row,
                    gemm_engine,
                    pgd_config.deadline,
                ) {
                    Ok(Some(g)) => {
                        exact_grad_used += 1;
                        g
                    }
                    Ok(None) | Err(_) => {
                        spsa_used += 1;
                        joint_spsa_gradient(
                            graph,
                            input,
                            &x,
                            &conjunct_targets,
                            gemm_engine,
                            &mut rng,
                            spsa_delta,
                            gama_p_ref.as_deref().map(|p_ref| (p_ref, gama_lambda)),
                        )?
                    }
                }
            } else {
                spsa_used += 1;
                joint_spsa_gradient(
                    graph,
                    input,
                    &x,
                    &conjunct_targets,
                    gemm_engine,
                    &mut rng,
                    spsa_delta,
                    gama_p_ref.as_deref().map(|p_ref| (p_ref, gama_lambda)),
                )?
            };

            // Gradient ascent on satisfaction margin: maximize margin to push toward
            // the unsafe region. In VNN-LIB, constraints describe the UNSAFE region —
            // satisfying the margin means the point IS a counterexample to safety.
            let previous_x = if pgd_config.restart_when_stuck {
                Some(x.clone())
            } else {
                None
            };
            x = step_state.step(&grad, &x, input, true);
            gama_step = gama_step.saturating_add(1);

            // Restart-when-stuck (#4278): if projection maps back to the same point,
            // resample and re-evaluate instead of continuing with a dead restart.
            if let Some(ref prev) = previous_x {
                if prev.iter().zip(x.iter()).all(|(&a, &b)| a == b) {
                    x = sample_uniform_point(input, &mut rng);
                    step_state.reset();
                    output = evaluate_graph(graph, &x, gemm_engine)?;
                    reset_gama_lane(&mut gama_p_ref, &mut gama_step);
                    gama_p_ref = gama_lambda0.and_then(|_| gama_reference_softmax(&output));
                    if super::check_unsafe_counterexample(&output, constraints) {
                        if let Some(pair) = revalidate_graph_counterexample(
                            graph,
                            x.clone(),
                            constraints,
                            "restart-when-stuck",
                        )? {
                            return Ok(Some(pair));
                        }
                    }
                    continue;
                }
            }

            output = evaluate_graph(graph, &x, gemm_engine)?;
            // #postbab-seed: track/export the best-margin point (guidance-only).
            if !conjunct_targets.is_empty() {
                let joint = joint_hinge_loss(&conjunct_targets, &output);
                super::super::best_margin_export::record_best_margin_candidate(joint, &x);
                if joint > best_joint_seen {
                    best_joint_seen = joint;
                }
            }
            if super::check_unsafe_counterexample(&output, constraints) {
                let ctx = format!("restart {restart}, step {step}");
                if let Some(pair) =
                    revalidate_graph_counterexample(graph, x.clone(), constraints, &ctx)?
                {
                    if diag {
                        eprintln!(
                            "[pgd-diag] CE FOUND at restart {restart} step {step} elapsed={:.1}s",
                            pgd_start.elapsed().as_secs_f64()
                        );
                    }
                    emit_graph_pgd_status(
                        json,
                        format_args!("  Graph PGD found counterexample at {ctx}!"),
                    );
                    return Ok(Some(pair));
                }
            }
        }

        if diag {
            eprintln!(
                "[pgd-diag] restart {restart}: exact_grad={exact_grad_used} spsa={spsa_used} elapsed={:.1}s best_joint={best_joint_seen:.5} (0 => CE)",
                pgd_start.elapsed().as_secs_f64()
            );
        }
        if exact_grad_used > 0 || spsa_used > 0 {
            tracing::info!(
                "Graph PGD restart {}: exact_grad={}, spsa={}",
                restart,
                exact_grad_used,
                spsa_used
            );
        }

        let final_output = evaluate_graph(graph, &x, gemm_engine)?;
        if super::check_unsafe_counterexample(&final_output, constraints) {
            let ctx = format!("restart {restart} (final check)");
            if let Some(pair) = revalidate_graph_counterexample(graph, x, constraints, &ctx)? {
                emit_graph_pgd_status(
                    json,
                    format_args!("  Graph PGD found counterexample at {ctx}!"),
                );
                return Ok(Some(pair));
            }
        }
    }

    if !json {
        println!(
            "  Graph PGD: no counterexample found. ({:.2}s, engine={})",
            pgd_start.elapsed().as_secs_f64(),
            engine_label
        );
    }
    Ok(None)
}

/// The PGD target whose margin-ascent drives the point toward SATISFYING
/// `constraint` (i.e. toward the UNSAFE region): `target.margin(output) >= 0`
/// iff the constraint holds at `output`. Returns `None` for constraint
/// variants the graph attack does not model.
pub(super) fn constraint_target(constraint: &OutputConstraint) -> Option<GraphPgdTarget> {
    match constraint {
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
            // unsafe when y_i <= y_j → margin = y_j - y_i, target maximizes y_j - y_i.
            Some(GraphPgdTarget::Relational(*j, *i))
        }
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            Some(GraphPgdTarget::Relational(*i, *j))
        }
        OutputConstraint::GreaterEqConst(i, c) | OutputConstraint::GreaterThanConst(i, c) => {
            Some(GraphPgdTarget::Constant(*i, *c as f32))
        }
        OutputConstraint::LessEqConst(i, c) | OutputConstraint::LessThanConst(i, c) => {
            Some(GraphPgdTarget::NegConstant(*i, *c as f32))
        }
        _ => None,
    }
}

/// The satisfaction margin of one constraint and the PGD target that, when its
/// margin is ascended, drives the point toward satisfying that constraint (i.e.
/// toward the UNSAFE region). Margin ≥ 0 means the constraint already holds.
/// Returns `None` for constraint variants the graph attack does not model.
fn constraint_margin_target(
    output: &ArrayD<f32>,
    constraint: &OutputConstraint,
) -> Option<(f32, GraphPgdTarget)> {
    constraint_target(constraint).map(|t| (t.margin(output), t))
}

/// Joint AND-clause hinge loss `Σ_c min(margin_c, 0)` over the modeled
/// conjuncts: strictly negative until EVERY modeled conjunct holds, and its
/// ascent pushes all unsatisfied conjuncts toward satisfaction at once.
/// When the hinge saturates (all modeled margins ≥ 0) it returns the MINIMUM
/// margin instead, so the SPSA estimate keeps signal exactly where
/// [`joint_unsat_spec_row`] falls back to the worst-conjunct row (strict
/// inequalities / unmodeled variants can leave the full CE check failing at a
/// saturated hinge). Continuous at the crossover (both pieces → 0).
pub(super) fn joint_hinge_loss(targets: &[GraphPgdTarget], output: &ArrayD<f32>) -> f32 {
    let mut hinge_sum = 0.0f32;
    let mut min_margin = f32::INFINITY;
    for target in targets {
        let margin = target.margin(output);
        min_margin = min_margin.min(margin);
        if margin < 0.0 {
            hinge_sum += margin;
        }
    }
    if min_margin < 0.0 {
        hinge_sum
    } else {
        min_margin
    }
}

/// Gradient spec row of the joint AND-clause objective at `output`: the SUM of
/// the margin rows of the hinge-ACTIVE (unsatisfied, margin < 0) conjuncts —
/// the hinge constants drop out of the derivative, so this single row fed to
/// one backward pass yields the exact joint gradient. When every modeled
/// conjunct is already satisfied (the full CE check may still fail on
/// unmodeled variants) it falls back to the single worst (min-margin) conjunct
/// row, matching the previous worst-target behavior. `None` only when there is
/// no modeled conjunct at all.
pub(super) fn joint_unsat_spec_row(
    targets: &[GraphPgdTarget],
    output: &ArrayD<f32>,
    num_outputs: usize,
) -> Option<Array2<f32>> {
    let mut row = Array2::<f32>::zeros((1, num_outputs));
    let mut active = 0usize;
    let mut worst: Option<(f32, &GraphPgdTarget)> = None;
    for target in targets {
        let margin = target.margin(output);
        if worst.is_none_or(|(m, _)| margin < m) {
            worst = Some((margin, target));
        }
        if margin < 0.0 {
            row += &target.to_spec_row(num_outputs);
            active += 1;
        }
    }
    if active == 0 {
        return worst.map(|(_, t)| t.to_spec_row(num_outputs));
    }
    Some(row)
}

/// Capture a finite softmax reference at the start of a GAMA-guided lane.
/// Invalid network outputs disable guidance for that lane; the raw attack
/// objective remains available.
pub(super) fn gama_reference_softmax(output: &ArrayD<f32>) -> Option<Vec<f32>> {
    if output.is_empty() || output.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let p_ref = gama_softmax(output);
    p_ref.iter().all(|v| v.is_finite()).then_some(p_ref)
}

/// Reset guidance state whenever a lane starts a genuinely new trajectory.
/// The next finite output becomes the new fixed reference and annealing starts
/// again at lambda_0.
pub(super) fn reset_gama_lane(p_ref: &mut Option<Vec<f32>>, step: &mut usize) {
    *p_ref = None;
    *step = 0;
}

// Above this point a unit-scale raw specification coefficient can no longer be
// represented alongside the guidance term in f32. Treat such configuration as
// numerically invalid and retain the raw objective.
const MAX_NUMERIC_GAMA_LAMBDA: f32 = 1.0 / f32::EPSILON;

fn valid_gama_lambda(lambda: f32) -> bool {
    lambda.is_finite() && lambda > 0.0 && lambda <= MAX_NUMERIC_GAMA_LAMBDA
}

/// Add only the GAMA guidance cotangent to an arbitrary raw specification row.
///
/// Keeping the raw row intact is essential for constant-threshold properties:
/// `Y_i >= c` has raw satisfaction margin `y_i-c`, while `Y_i <= c` has
/// `c-y_i`. Softmax is not threshold preserving for those atoms, so the GAMA
/// term may steer the attack but must never replace the raw joint-slack row.
/// Invalid guidance fails open to the byte-identical raw row.
pub(super) fn add_gama_guidance_to_spec_row(
    mut raw_row: Array2<f32>,
    output: &ArrayD<f32>,
    p_ref: &[f32],
    lambda: f32,
) -> Array2<f32> {
    if !valid_gama_lambda(lambda)
        || raw_row.nrows() != 1
        || raw_row.ncols() != output.len()
        || output.iter().any(|v| !v.is_finite())
    {
        return raw_row;
    }
    let q = gama_softmax(output);
    let Some(guidance) = gama_guidance_cotangent(&q, p_ref) else {
        return raw_row;
    };
    let additions: Vec<f32> = guidance.iter().map(|v| lambda * v).collect();
    if additions.iter().any(|v| !v.is_finite()) {
        return raw_row;
    }
    if raw_row
        .row(0)
        .iter()
        .zip(&additions)
        .any(|(&dst, &add)| !(dst + add).is_finite())
    {
        return raw_row;
    }
    for (dst, add) in raw_row.row_mut(0).iter_mut().zip(additions) {
        *dst += add;
    }
    raw_row
}

/// Scalar companion of [`add_gama_guidance_to_spec_row`] for the SPSA fallback.
/// Raw joint slack remains the authority and invalid guidance reduces exactly
/// to the historical objective.
pub(super) fn joint_gama_objective(
    targets: &[GraphPgdTarget],
    output: &ArrayD<f32>,
    p_ref: &[f32],
    lambda: f32,
) -> f32 {
    let raw = joint_hinge_loss(targets, output);
    if !valid_gama_lambda(lambda)
        || output.len() != p_ref.len()
        || output.iter().chain(p_ref).any(|v| !v.is_finite())
    {
        return raw;
    }
    let guided = raw + lambda * gama_guidance(&gama_softmax(output), p_ref);
    if guided.is_finite() {
        guided
    } else {
        raw
    }
}

/// Select the PGD target for a DISJUNCTIVE property `(or clause_1 ... clause_k)`.
///
/// A clause is a conjunction; it holds when ALL its constraints hold, so its
/// "satisfiability margin" is the MIN over its constraints (the bottleneck). The
/// disjunction is satisfied as soon as ANY clause holds, so the most promising
/// direction is the clause whose bottleneck margin is LARGEST (closest to/above 0);
/// we ascend that clause's bottleneck constraint. (Contrast the conjunctive attack,
/// which ascends the global MIN-margin constraint.)
pub(super) fn best_disjunctive_target(
    output: &ArrayD<f32>,
    clauses: &[Vec<OutputConstraint>],
) -> Option<GraphPgdTarget> {
    let mut best_clause_margin = f32::NEG_INFINITY;
    let mut best_target: Option<GraphPgdTarget> = None;
    for clause in clauses {
        // Bottleneck (min-margin) constraint of this clause.
        let mut clause_margin = f32::INFINITY;
        let mut clause_target: Option<GraphPgdTarget> = None;
        for c in clause {
            if let Some((m, t)) = constraint_margin_target(output, c) {
                if m < clause_margin {
                    clause_margin = m;
                    clause_target = Some(t);
                }
            }
        }
        if let Some(t) = clause_target {
            if clause_margin > best_clause_margin {
                best_clause_margin = clause_margin;
                best_target = Some(t);
            }
        }
    }
    best_target
}

/// All clause targets ranked best-first (largest clause margin = closest to
/// violation). Used for TARGETED PGD: each restart drives one FIXED target class
/// to its box corner instead of re-selecting the closest class every step —
/// which, across a large classification disjunction (e.g. cifar100's 99 clauses),
/// thrashes between classes and never commits to any single class's violating
/// vertex. Mirrors α,β-CROWN's per-target attack. Attack-only.
pub(super) fn ranked_disjunctive_targets(
    output: &ArrayD<f32>,
    clauses: &[Vec<OutputConstraint>],
) -> Vec<GraphPgdTarget> {
    let mut scored: Vec<(f32, GraphPgdTarget)> = Vec::new();
    for clause in clauses {
        let mut clause_margin = f32::INFINITY;
        let mut clause_target: Option<GraphPgdTarget> = None;
        for c in clause {
            if let Some((m, t)) = constraint_margin_target(output, c) {
                if m < clause_margin {
                    clause_margin = m;
                    clause_target = Some(t);
                }
            }
        }
        if let Some(t) = clause_target {
            scored.push((clause_margin, t));
        }
    }
    // Best-first: the closest-to-violation clause leads.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().map(|(_, t)| t).collect()
}

/// Outcome of [`try_graph_disjunctive_pgd_attack`]: the candidate (if any) plus
/// margin/budget telemetry for the adaptive attack-extension decision
/// (#attack-extend). ATTACK-ONLY data — it steers where the remaining attack
/// budget goes, never a verdict.
pub(super) struct GraphDisjunctiveAttackOutcome {
    /// Candidate counterexample input (a point where some clause held on the
    /// attack forward). The caller re-evaluates + confirms before any `sat`.
    pub candidate: Option<ArrayD<f32>>,
    /// Closest-to-violation clause bottleneck margin observed across EVERY
    /// evaluated point (`>= 0` ⇒ candidate). `NEG_INFINITY` when no point with
    /// a modeled constraint was evaluated.
    pub best_margin: f32,
    /// True when the attack stopped on its phase deadline (budget-bound)
    /// rather than exhausting its configured restarts (work-bound).
    pub hit_deadline: bool,
    /// Restarts started and PGD steps taken — extension-fire diagnostics.
    pub restarts_started: usize,
    pub steps_taken: usize,
}

impl GraphDisjunctiveAttackOutcome {
    fn empty() -> Self {
        Self {
            candidate: None,
            best_margin: f32::NEG_INFINITY,
            hit_deadline: false,
            restarts_started: 0,
            steps_taken: 0,
        }
    }
}

/// The satisfiability margin of the CLOSEST clause at `output`: max over
/// clauses of the clause's bottleneck (min-constraint) margin. `>= 0` means
/// some clause holds (counterexample). `None` when no clause has a modeled
/// constraint.
pub(super) fn best_clause_bottleneck_margin(
    output: &ArrayD<f32>,
    clauses: &[Vec<OutputConstraint>],
) -> Option<f32> {
    let mut best: Option<f32> = None;
    for clause in clauses {
        let mut clause_margin = f32::INFINITY;
        let mut modeled = false;
        for c in clause {
            if let Some((m, _)) = constraint_margin_target(output, c) {
                clause_margin = clause_margin.min(m);
                modeled = true;
            }
        }
        if modeled {
            best = Some(best.map_or(clause_margin, |b| b.max(clause_margin)));
        }
    }
    best
}

/// Gradient-based **disjunctive** PGD attack on a `GraphNetwork` (the conv-resnet
/// counterpart of [`try_sequential_disjunctive_pgd_attack_with_config`], which is
/// Sequential-only). The graph disjunctive path otherwise falls back to random
/// sampling, which cannot find adversarial counterexamples in a high-dimensional
/// conv net at small `eps` — this targets the easiest disjunct with exact/SPSA
/// gradients, exactly the lever α,β-CROWN uses to find cifar100/tinyimagenet `sat`
/// instances in seconds.
///
/// Returns the candidate counterexample input on success (a point where some clause
/// holds). The caller re-evaluates and confirms it (and the verdict is ultimately
/// re-checked against the full property), so this is sound regardless of attack
/// internals — it can only produce candidates, never decide `sat`.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_graph_disjunctive_pgd_attack(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    clauses: &[Vec<OutputConstraint>],
    pgd_config: &PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<GraphDisjunctiveAttackOutcome> {
    let mut outcome = GraphDisjunctiveAttackOutcome::empty();
    if clauses.is_empty() {
        return Ok(outcome);
    }
    let exact_grad_eligible = graph_supports_exact_gradients(graph);
    let spsa_delta = pgd_config
        .suggested_spsa_delta(input)
        .max(pgd_config.spsa_delta);
    // #vnncomp-gama (attack-only, default OFF): when enabled — via the preset
    // `attack_mode: diversed_GAMA_PGD` or the `NY_PGD_GAMA=1` override —
    // RELATIONAL-target steps of the sequential loop ascend the GAMA guidance
    // loss instead of the raw margin. Phase order: the batched raw-margin
    // lane still runs first (unchanged); GAMA is a follow-on straggler phase.
    let gama_lambda0 = gama_lambda_init(pgd_config);
    let gama_lin = gama_lin_steps(pgd_config.num_steps);
    // #surrogate-sign (attack-only, preset `attack: surrogate_sign_gradient`):
    // STE surrogate graph for SPSA probes on Sign (BNN) nets. None when the
    // knob is off or the graph has no Sign layer — probes then run unchanged.
    let ste_surrogate: Option<GraphNetwork> = pgd_config
        .surrogate_sign_gradient
        .then(|| sign_ste_surrogate_graph(graph))
        .flatten();
    let mut num_outputs: Option<usize> = None;
    let satisfied = |output: &ArrayD<f32>| {
        clauses
            .iter()
            .any(|clause| super::check_unsafe_counterexample(output, clause))
    };

    if !json {
        println!(
            "\n  Running graph disjunctive PGD attack ({} restarts × {} steps, {} clauses, exact_grad={})...",
            pgd_config.num_restarts, pgd_config.num_steps, clauses.len(), exact_grad_eligible
        );
    }

    // #attack-extend increment 2: batched exact-VJP lane — K parallel restarts
    // per wide GPU pass, so the preset restart budget fits INSIDE the attack
    // cap (~0.3s/step sequential on metaroom 6cnn ⇒ 2 restarts per 20s slice;
    // batched ⇒ all 30). DEFAULT ON when the wide plan builds and a GPU CROWN
    // engine is present; `NY_PGD_VJP_BATCH=0` disables. The experimental
    // sequential-lane levers bypass it so they keep working unchanged. On any
    // capability miss the sequential loop below runs byte-identically.
    //
    // GAMA (#1449) does NOT bypass this driver: the batched raw-margin lane
    // runs FIRST, unchanged — it is the measured winner of the easy `sat`
    // pool (cifar100 26/29, tinyimagenet 33/38) and must never regress. When
    // it completes WITHOUT a candidate and GAMA is enabled, the sequential
    // loop below runs as a FOLLOW-ON phase with the GAMA guidance objective —
    // extra budget spent only on the stragglers the raw attack already missed
    // (deadline-capped, so a near-wall instance's BaB slice stays protected).
    let experimental_lane_active = graph_pgd_targeted_enabled() || graph_pgd_clean_init_enabled();
    if gemm_engine.is_some() && pgd_config.num_restarts > 1 && !experimental_lane_active {
        match super::graph_pgd_vjp_batched_disj::try_graph_disjunctive_pgd_vjp_batched(
            graph,
            input,
            clauses,
            pgd_config,
            gemm_engine,
            json,
            &mut outcome,
        )? {
            super::graph_pgd_vjp_batched_disj::DisjVjpBatchedOutcome::Completed => {
                let gama_follow_on = gama_lambda0.is_some()
                    && outcome.candidate.is_none()
                    && !outcome.hit_deadline
                    && pgd_config.deadline.is_none_or(|d| Instant::now() < d);
                if !gama_follow_on {
                    return Ok(outcome);
                }
                if !json {
                    println!("  Batched PGD found no candidate; entering GAMA follow-on phase...");
                }
            }
            super::graph_pgd_vjp_batched_disj::DisjVjpBatchedOutcome::FallbackToSequential => {}
        }
    }

    // #vnncomp-targeted (attack-only, `NY_PGD_TARGETED=1`, default OFF): rank the
    // candidate target classes ONCE at the clean image, so each restart commits to
    // a distinct top-k class (restart r → r-th closest competitor) and drives it
    // from clean to its violating corner — instead of re-picking the closest class
    // every step and thrashing across the disjunction. Empty ⇒ feature off.
    let targeted_ranked: Vec<GraphPgdTarget> = if graph_pgd_targeted_enabled() {
        let clean_out = evaluate_graph(graph, &sample_center_point(input), gemm_engine)?;
        ranked_disjunctive_targets(&clean_out, clauses)
    } else {
        Vec::new()
    };

    // Closest-to-violation margin tracking (#attack-extend): always on — it is
    // a handful of output-vector reads per step, negligible next to the forward
    // pass — and returned to the caller so the adaptive extension can tell
    // "attack lands just short" (near 0) from "gradient goes astray" (very
    // negative). margin >= 0 ⇒ counterexample. `NY_PGD_DIAG=1` prints it.
    let diag = std::env::var("NY_PGD_DIAG").ok().as_deref() == Some("1");

    // Restart-index seed offset (#attack-extend): byte-identical to the
    // historical `1009 + restart` for the default `PgdConfig.seed` (42). The
    // adaptive attack extension passes `seed = 42 + k` to CONTINUE the restart
    // seed sequence at index `k` — the restarts the first run's phase cap cut
    // off — rather than replaying already-explored trajectories or gambling on
    // a fresh randomization: the continuous longer run is the measured
    // reference that lands the counterexample (metaroom spec_idx_129/148).
    let restart_seed_offset = pgd_config.seed.wrapping_sub(42);

    for restart in 0..pgd_config.num_restarts {
        if pgd_config.deadline.is_some_and(|d| Instant::now() >= d) {
            outcome.hit_deadline = true;
            if diag {
                eprintln!(
                    "[pgd-diag] restart-deadline: best disjunctive margin seen = {:.5} (>=0 => CE)",
                    outcome.best_margin
                );
            }
            return Ok(outcome);
        }
        outcome.restarts_started += 1;
        let mut rng = SimpleRng::new(1009 + restart_seed_offset.wrapping_add(restart as u64));
        let mut step_state = PgdStepState::from_config(
            pgd_config.optimizer,
            pgd_config.alpha_mode,
            pgd_config.step_size,
            pgd_config.adam,
            input,
            input.shape(),
        );
        // #vnncomp-clean-init (attack-only, `NY_PGD_CLEAN_INIT=1`, default OFF):
        // seed restart 0 from the box center (the clean image) instead of a
        // uniform-random corner. For robust nets the CE is a small push from
        // clean; random far-corner inits rarely reach it. When off, byte-
        // identical to the prior all-random behavior.
        let mut x = if !targeted_ranked.is_empty() {
            // Targeted mode: every restart starts from the clean image and is
            // differentiated by its fixed target class (below), driving that
            // class from clean toward its violating box corner.
            sample_center_point(input)
        } else if restart == 0 && graph_pgd_clean_init_enabled() {
            sample_center_point(input)
        } else {
            initialize_graph_point(pgd_config, graph, input, &mut rng, gemm_engine)?
        };
        let mut output = evaluate_graph(graph, &x, gemm_engine)?;
        num_outputs.get_or_insert(output.len());
        if let Some(m) = best_clause_bottleneck_margin(&output, clauses) {
            outcome.best_margin = outcome.best_margin.max(m);
        }
        if satisfied(&output) {
            outcome.candidate = Some(x);
            return Ok(outcome);
        }
        // #vnncomp-gama: fixed reference softmax `P` at this restart's init point.
        let gama_p_ref: Option<Vec<f32>> = gama_lambda0.map(|_| gama_softmax(&output));

        // #vnncomp-targeted: this restart's FIXED target = the r-th closest class
        // ranked at the clean image, driven for all steps (no per-step re-pick).
        let fixed_target: Option<GraphPgdTarget> =
            (!targeted_ranked.is_empty()).then(|| targeted_ranked[restart % targeted_ranked.len()]);

        for step in 0..pgd_config.num_steps {
            if spsa_step_deadline_exceeded(step, pgd_config.deadline) {
                outcome.hit_deadline = true;
                if diag {
                    eprintln!(
                        "[pgd-diag] step-deadline r{restart} s{step}: best margin = {:.5}",
                        outcome.best_margin
                    );
                }
                return Ok(outcome);
            }
            let Some(target) = fixed_target.or_else(|| best_disjunctive_target(&output, clauses))
            else {
                break;
            };
            // #vnncomp-gama: on a RELATIONAL classification disjunct with GAMA on,
            // ascend the annealed GAMA guidance loss. Non-relational targets
            // (const thresholds — softmax meaningless) keep the raw path.
            let use_gama = gama_lambda0.is_some()
                && gama_p_ref.is_some()
                && matches!(target, GraphPgdTarget::Relational(_, _));
            let grad = if use_gama {
                let lambda = gama_lambda_at(gama_lambda0.unwrap(), step, gama_lin);
                // EXACT GAMA gradient via the point-Jacobian VJP: feed the analytic
                // GAMA cotangent dL/dz to attack_point_gradient. Only falls back to
                // noisy SPSA-of-GAMA if the fast VJP is unavailable (non-whitelist).
                let q = gama_softmax(&output);
                let exact_gama =
                    match gama_cotangent(&q, gama_p_ref.as_ref().unwrap(), &target, lambda) {
                        Some(cot) => graph.attack_point_gradient(
                            &x,
                            &cot,
                            gemm_engine,
                            pgd_config.deadline,
                        )?,
                        None => None,
                    };
                match exact_gama {
                    Some(g) => g,
                    None => spsa_gama_gradient(
                        graph,
                        input,
                        &x,
                        &target,
                        gama_p_ref.as_ref().unwrap(),
                        lambda,
                        gemm_engine,
                        &mut rng,
                        spsa_delta,
                        ste_surrogate.as_ref(),
                    )?,
                }
            } else if exact_grad_eligible {
                match exact_graph_margin_gradient(
                    graph,
                    &x,
                    &output,
                    &target,
                    num_outputs.unwrap_or(output.len()),
                    gemm_engine,
                    pgd_config.deadline,
                ) {
                    Ok(Some(g)) => g,
                    Ok(None) | Err(_) => spsa_gradient(
                        graph,
                        input,
                        &x,
                        &target,
                        gemm_engine,
                        &mut rng,
                        spsa_delta,
                        ste_surrogate.as_ref(),
                    )?,
                }
            } else {
                spsa_gradient(
                    graph,
                    input,
                    &x,
                    &target,
                    gemm_engine,
                    &mut rng,
                    spsa_delta,
                    ste_surrogate.as_ref(),
                )?
            };
            // Ascend the easiest-disjunct margin → toward the unsafe region.
            x = step_state.step(&grad, &x, input, true);
            output = evaluate_graph(graph, &x, gemm_engine)?;
            outcome.steps_taken += 1;
            if let Some(m) = best_clause_bottleneck_margin(&output, clauses) {
                outcome.best_margin = outcome.best_margin.max(m);
            }
            if satisfied(&output) {
                if !json {
                    println!(
                        "  Graph disjunctive PGD found candidate at restart {restart}, step {step}!"
                    );
                }
                outcome.candidate = Some(x);
                return Ok(outcome);
            }
        }
    }
    Ok(outcome)
}

/// #surrogate-sign (attack-only, `attack: surrogate_sign_gradient`): STE
/// surrogate graph for SPSA probes — every `Layer::Sign` replaced by the
/// identity (`AddConstant(0)`), so probe finite-differences see
/// `d/dx sign(x) = 1` at any activation scale. Through the discrete Sign (or
/// its tanh smoothing) a BNN's SPSA differences are exactly zero away from
/// the flip surfaces, which is why traffic_signs-class attacks go nowhere.
/// Built ONCE per attack. PROBES ONLY: violation checks, restart re-evals,
/// and candidate confirmation all stay on the TRUE graph, and every candidate
/// is re-validated downstream, so this can never affect a sound verdict.
fn sign_ste_surrogate_graph(graph: &GraphNetwork) -> Option<GraphNetwork> {
    let has_sign = graph.node_names().iter().any(|name| {
        graph
            .node(name)
            .is_some_and(|node| matches!(node.layer(), Layer::Sign(_)))
    });
    if !has_sign {
        return None;
    }
    let mut surrogate = GraphNetwork::new();
    for name in graph.node_names() {
        let node = graph.node(name)?;
        let layer = match node.layer() {
            Layer::Sign(_) => Layer::AddConstant(AddConstantLayer::new(ArrayD::zeros(IxDyn(&[1])))),
            other => other.clone(),
        };
        surrogate.add_node(GraphNode::new(
            node.name().to_string(),
            layer,
            node.inputs().to_vec(),
        ));
    }
    surrogate.set_output(graph.output_name());
    Some(surrogate)
}

/// One SPSA two-point gradient estimate of an arbitrary scalar objective.
///
/// `surrogate` (#surrogate-sign): when set, probes are evaluated on that
/// graph through the internal point forward — NOT through `evaluate_graph`,
/// whose ORT scoring route always runs the ORIGINAL model and would silently
/// undo the surrogate.
fn spsa_gradient_of(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    x: &ArrayD<f32>,
    gemm_engine: Option<&dyn GemmEngine>,
    rng: &mut SimpleRng,
    spsa_delta: f32,
    surrogate: Option<&GraphNetwork>,
    objective: impl Fn(&ArrayD<f32>) -> f32,
) -> Result<ArrayD<f32>> {
    let n = x.len();
    let perturbation = {
        let vals: Vec<f32> = (0..n)
            .map(|_| if rng.next_bool() { 1.0 } else { -1.0 })
            .collect();
        ArrayD::from_shape_vec(IxDyn(x.shape()), vals)?
    };

    let mut x_plus = x + &perturbation * spsa_delta;
    let mut x_minus = x - &perturbation * spsa_delta;
    project_to_bounds_in_place(&mut x_plus, input.lower(), input.upper());
    project_to_bounds_in_place(&mut x_minus, input.lower(), input.upper());
    let eval_probe = |point: &ArrayD<f32>| -> Result<ArrayD<f32>> {
        match surrogate {
            Some(surrogate_graph) => {
                let point_bounds = BoundedTensor::concrete(point.clone())?;
                Ok(surrogate_graph
                    .propagate_concrete_point(&point_bounds, gemm_engine, None)?
                    .center())
            }
            None => evaluate_graph(graph, point, gemm_engine),
        }
    };
    let out_plus = eval_probe(&x_plus)?;
    let out_minus = eval_probe(&x_minus)?;

    let obj_plus = objective(&out_plus);
    let obj_minus = objective(&out_minus);
    if obj_plus.is_nan() || obj_minus.is_nan() {
        return Ok(ArrayD::zeros(IxDyn(x.shape())));
    }

    Ok(&perturbation * ((obj_plus - obj_minus) / (2.0 * spsa_delta)))
}

/// Compute SPSA gradient estimate for one step (single-target margin).
fn spsa_gradient(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    x: &ArrayD<f32>,
    target: &GraphPgdTarget,
    gemm_engine: Option<&dyn GemmEngine>,
    rng: &mut SimpleRng,
    spsa_delta: f32,
    surrogate: Option<&GraphNetwork>,
) -> Result<ArrayD<f32>> {
    spsa_gradient_of(
        graph,
        input,
        x,
        gemm_engine,
        rng,
        spsa_delta,
        surrogate,
        |out| target.margin(out),
    )
}

/// SPSA gradient estimate of the JOINT AND-clause hinge objective
/// (`Σ_c min(margin_c, 0)`) — the fallback used by the conjunctive graph
/// attack when the exact joint gradient is unavailable.
fn joint_spsa_gradient(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    x: &ArrayD<f32>,
    targets: &[GraphPgdTarget],
    gemm_engine: Option<&dyn GemmEngine>,
    rng: &mut SimpleRng,
    spsa_delta: f32,
    gama: Option<(&[f32], f32)>,
) -> Result<ArrayD<f32>> {
    spsa_gradient_of(graph, input, x, gemm_engine, rng, spsa_delta, None, |out| {
        if let Some((p_ref, lambda)) = gama {
            joint_gama_objective(targets, out, p_ref, lambda)
        } else {
            joint_hinge_loss(targets, out)
        }
    })
}

// ---------------------------------------------------------------------------
// #vnncomp-gama: GAMA-PGD (Guided Adversarial Margin Attack) — attack-only lever.
//
// Opt-in (`NY_PGD_GAMA=1`, default OFF) guidance adds a squared-L2 term between
// the current softmax and a FIXED reference softmax `P` (the restart's init
// point), annealed via `λ` from `NY_PGD_GAMA_LAMBDA` (default 50) linearly to 0
// over `round(NY_PGD_GAMA_LIN_FRAC · steps)` steps (default 0.25). Relational
// classification targets retain their established softmax-margin objective.
// Single-clause conjunctive properties — including constant thresholds such as
// `Y_i >= c` and `Y_i <= c` — instead retain the exact raw joint-slack row and
// add only the guidance derivative, because softmax does not preserve a constant
// threshold. Formula verified vs val-iisc GAMA_PGD.py.
//
// SOUNDNESS: attack-only, ZERO false-VERIFIED risk. GAMA changes ONLY the gradient
// direction; the candidate is still accepted only when `satisfied()` holds on the
// RAW logits and is re-confirmed by the independent model forward. In particular,
// a positive guidance-augmented objective is never treated as property success.
// Opt-in (`NY_PGD_CLEAN_INIT=1`, default OFF): start restart 0 of the graph
// disjunctive PGD from the box center (clean image) rather than a uniform-random
// point. α,β-CROWN's first PGD restart is clean-init. Attack-only ⇒ can only
// find counterexamples, never affects a sound/unsat verdict.
fn graph_pgd_clean_init_enabled() -> bool {
    std::env::var("NY_PGD_CLEAN_INIT").ok().as_deref() == Some("1")
}

// Opt-in (`NY_PGD_TARGETED=1`, default OFF): each restart of the graph disjunctive
// PGD fixes ONE target class (ranked at its init point, restart r → r-th closest)
// and drives it for all steps, instead of re-picking the closest class every step.
// Prevents target-thrashing across large classification disjunctions. Attack-only.
fn graph_pgd_targeted_enabled() -> bool {
    std::env::var("NY_PGD_TARGETED").ok().as_deref() == Some("1")
}

/// Resolve the GAMA guidance weight λ₀ (#1449): config-first, env override.
///
/// - `NY_PGD_GAMA=0` — force OFF (even when the preset enabled it);
/// - `NY_PGD_GAMA=1` — force ON, λ₀ from `NY_PGD_GAMA_LAMBDA`
///   (missing means [`ny_propagate::GAMA_LAMBDA_DEFAULT`], while a present
///   invalid value disables GAMA);
/// - unset — `pgd_config.gama_lambda`, i.e. the preset's
///   `attack_mode: diversed_GAMA_PGD` wiring.
/// - non-finite, non-positive, or numerically destructive values disable GAMA.
pub(super) fn gama_lambda_init(pgd_config: &PgdConfig) -> Option<f32> {
    let env_lambda = || match std::env::var("NY_PGD_GAMA_LAMBDA") {
        Ok(value) => value
            .parse()
            .ok()
            .filter(|&lambda| valid_gama_lambda(lambda)),
        Err(std::env::VarError::NotPresent) => Some(ny_propagate::GAMA_LAMBDA_DEFAULT),
        Err(std::env::VarError::NotUnicode(_)) => None,
    };
    let resolved = match std::env::var("NY_PGD_GAMA").ok().as_deref() {
        Some("0") => None,
        Some("1") => env_lambda(),
        _ => pgd_config.gama_lambda,
    };
    resolved.filter(|&lambda| valid_gama_lambda(lambda))
}

#[cfg(test)]
pub(super) static GAMA_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Exact cotangent `dL/dz` (1×C row) of the GAMA loss
/// `L = (q_{j*} − q_t) + λ·Σ_c (P_c − q_c)²`, `q = softmax(z)`, for a
/// `Relational(j*, t)` target. Feeding this to `attack_point_gradient` as the
/// spec row yields the EXACT `dL/dinput` (softmax Jacobian composed with the
/// network point-Jacobian) — replacing the noisy SPSA-of-GAMA. Derivation:
/// d(q_a)/dz_k = q_a(1{a=k} − q_k); d(guidance)/dz_k = 2·q_k·[(q_k − P_k) − s]
/// with s = ⟨q,q⟩ − ⟨P,q⟩. Returns None for non-relational targets.
fn gama_cotangent(
    q: &[f32],
    p_ref: &[f32],
    target: &GraphPgdTarget,
    lambda: f32,
) -> Option<Array2<f32>> {
    let (jstar, t) = match target {
        GraphPgdTarget::Relational(a, b) => (*a, *b),
        _ => return None,
    };
    let c = q.len();
    if jstar >= c || t >= c || p_ref.len() != c {
        return None;
    }
    let guidance = gama_guidance_cotangent(q, p_ref)?;
    let qj = q[jstar];
    let qt = q[t];
    let mut row = Array2::<f32>::zeros((1, c));
    for k in 0..c {
        let ind_j = if k == jstar { 1.0 } else { 0.0 };
        let ind_t = if k == t { 1.0 } else { 0.0 };
        let dmargin = qj * (ind_j - q[k]) - qt * (ind_t - q[k]);
        row[[0, k]] = dmargin + lambda * guidance[k];
    }
    Some(row)
}

/// GAMA loss `L = softmax_margin(target) + λ·Σ_c (P_c − q_c)²`, `q = softmax(out)`.
fn gama_loss(out: &ArrayD<f32>, target: &GraphPgdTarget, p_ref: &[f32], lambda: f32) -> f32 {
    let q = gama_softmax(out);
    let q_arr = ArrayD::from_shape_vec(IxDyn(out.shape()), q.clone())
        .unwrap_or_else(|_| ArrayD::zeros(IxDyn(out.shape())));
    // softmax margin: the SAME linear margin the target uses, on the softmax vector
    // (order-preserving, so it shares the raw margin's zero-crossing / accept gate).
    let margin = target.margin(&q_arr);
    margin + lambda * gama_guidance(&q, p_ref)
}

/// SPSA of the GAMA loss (increment 1) — `spsa_gradient` with the scalar objective
/// swapped for [`gama_loss`]. Exact-cotangent GAMA is a later increment.
#[allow(clippy::too_many_arguments)] // GAMA objective + surrogate probe routing
fn spsa_gama_gradient(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    x: &ArrayD<f32>,
    target: &GraphPgdTarget,
    p_ref: &[f32],
    lambda: f32,
    gemm_engine: Option<&dyn GemmEngine>,
    rng: &mut SimpleRng,
    spsa_delta: f32,
    surrogate: Option<&GraphNetwork>,
) -> Result<ArrayD<f32>> {
    spsa_gradient_of(
        graph,
        input,
        x,
        gemm_engine,
        rng,
        spsa_delta,
        surrogate,
        |out| gama_loss(out, target, p_ref, lambda),
    )
}

#[cfg(test)]
mod gama_cotangent_tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    // The analytic GAMA cotangent dL/dz must match a central finite-difference of
    // gama_loss w.r.t. the logits z, coordinate-by-coordinate.
    #[test]
    fn gama_cotangent_matches_finite_difference_of_loss() {
        let c = 6usize;
        let z = vec![0.5f32, -1.2, 2.0, 0.1, -0.7, 1.3];
        let z0 = vec![-0.3f32, 0.8, 1.1, -1.0, 0.4, 0.2];
        let target = GraphPgdTarget::Relational(2, 4); // j*=2, t=4
        let lambda = 3.0f32;
        let p_ref = gama_softmax(&ArrayD::from_shape_vec(IxDyn(&[c]), z0).unwrap());
        let q = gama_softmax(&ArrayD::from_shape_vec(IxDyn(&[c]), z.clone()).unwrap());
        let cot = gama_cotangent(&q, &p_ref, &target, lambda).expect("relational cotangent");
        let delta = 1e-3f32;
        for k in 0..c {
            let mut zp = z.clone();
            zp[k] += delta;
            let mut zm = z.clone();
            zm[k] -= delta;
            let lp = gama_loss(
                &ArrayD::from_shape_vec(IxDyn(&[c]), zp).unwrap(),
                &target,
                &p_ref,
                lambda,
            );
            let lm = gama_loss(
                &ArrayD::from_shape_vec(IxDyn(&[c]), zm).unwrap(),
                &target,
                &p_ref,
                lambda,
            );
            let fd = (lp - lm) / (2.0 * delta);
            let an = cot[[0, k]];
            assert!((fd - an).abs() < 5e-3, "coord {k}: fd={fd} analytic={an}");
        }
    }

    #[test]
    fn gama_cotangent_rejects_non_relational() {
        let q = vec![0.2f32, 0.3, 0.5];
        assert!(gama_cotangent(&q, &q, &GraphPgdTarget::Constant(0, 1.0), 1.0).is_none());
    }

    #[test]
    fn conjunctive_constant_gama_row_matches_finite_difference() {
        let z = vec![0.1f32, -0.3, 0.7, -0.2];
        let z_ref = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.5, 0.6, 0.2, 1.0]).unwrap();
        let p_ref = gama_reference_softmax(&z_ref).expect("finite reference");
        let targets = vec![
            GraphPgdTarget::Constant(0, 0.8),
            GraphPgdTarget::NegConstant(2, 0.0),
            GraphPgdTarget::Constant(3, 0.4),
        ];
        let output = ArrayD::from_shape_vec(IxDyn(&[4]), z.clone()).unwrap();
        let raw = joint_unsat_spec_row(&targets, &output, 4).expect("modeled targets");
        let lambda = 2.75f32;
        let guided = add_gama_guidance_to_spec_row(raw, &output, &p_ref, lambda);

        let delta = 1e-3f32;
        for k in 0..z.len() {
            let mut zp = z.clone();
            zp[k] += delta;
            let mut zm = z.clone();
            zm[k] -= delta;
            let lp = joint_gama_objective(
                &targets,
                &ArrayD::from_shape_vec(IxDyn(&[4]), zp).unwrap(),
                &p_ref,
                lambda,
            );
            let lm = joint_gama_objective(
                &targets,
                &ArrayD::from_shape_vec(IxDyn(&[4]), zm).unwrap(),
                &p_ref,
                lambda,
            );
            let fd = (lp - lm) / (2.0 * delta);
            let analytic = guided[[0, k]];
            assert!(
                (fd - analytic).abs() < 5e-3,
                "coord {k}: fd={fd} analytic={analytic}"
            );
        }
    }

    #[test]
    fn conjunctive_gama_invalid_guidance_falls_back_exactly_to_raw_row() {
        let output = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.2f32, -0.4, 0.7]).unwrap();
        let raw = Array2::from_shape_vec((1, 3), vec![1.0f32, -1.0, 0.0]).unwrap();
        let wrong_row_shape = Array2::from_shape_vec((1, 2), vec![1.0f32, -1.0]).unwrap();
        assert_eq!(
            add_gama_guidance_to_spec_row(wrong_row_shape.clone(), &output, &[0.2, 0.3, 0.5], 1.0,),
            wrong_row_shape,
            "specification shape mismatch must preserve the raw row"
        );
        assert_eq!(
            add_gama_guidance_to_spec_row(raw.clone(), &output, &[0.5, 0.5], 1.0),
            raw,
            "reference shape mismatch must preserve the raw row"
        );
        let nonfinite = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.2f32, f32::NAN, 0.7]).unwrap();
        assert_eq!(
            add_gama_guidance_to_spec_row(raw.clone(), &nonfinite, &[0.2, 0.3, 0.5], 1.0,),
            raw,
            "non-finite output must preserve the raw row"
        );
        assert_eq!(
            add_gama_guidance_to_spec_row(raw.clone(), &output, &[0.2, 0.3, 0.5], f32::MAX,),
            raw,
            "numerically destructive lambda must preserve the raw row"
        );
        assert_eq!(
            add_gama_guidance_to_spec_row(raw.clone(), &output, &[0.2, 0.3, 0.5], 0.0),
            raw,
            "lambda zero must be byte-identical to the raw path"
        );
    }

    #[test]
    fn conjunctive_gama_reset_clears_reference_and_schedule() {
        let mut p_ref = Some(vec![0.2f32, 0.3, 0.5]);
        let mut step = 17usize;
        reset_gama_lane(&mut p_ref, &mut step);
        assert!(p_ref.is_none());
        assert_eq!(step, 0);
    }

    #[test]
    fn guided_objective_cannot_authorize_constant_threshold_success() {
        let output = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0f32, 0.0]).unwrap();
        let target = GraphPgdTarget::Constant(0, 2.0);
        let targets = [target];
        let p_ref = [1.0f32, 0.0];
        let guided = joint_gama_objective(&targets, &output, &p_ref, 10.0);
        assert!(guided > 0.0, "the guidance term should dominate this probe");
        assert!(
            joint_hinge_loss(&targets, &output) < 0.0,
            "raw property margin remains unsatisfied"
        );
        let constraints = [OutputConstraint::GreaterEqConst(0, 2.0)];
        assert!(
            !super::super::check_unsafe_counterexample(&output, &constraints),
            "only the raw VNNLIB constraint can authorize a candidate"
        );
    }

    // #1449: λ₀ resolution is config-first with env override. All three
    // precedence cases live in ONE test so the env mutation window stays
    // sequential (no parallel test observes a half-set NY_PGD_GAMA).
    #[test]
    fn gama_lambda_init_config_first_env_override() {
        // GAMA_ENV_LOCK serializes against the other GAMA env tests; the
        // mutation itself routes through the blessed env choke point (clippy
        // env wall), which restores pre-test state on exit.
        let _lock = GAMA_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        ny_test_utils::env::with_env_edits(|env| {
            let cfg_on = PgdConfig {
                gama_lambda: Some(7.0),
                ..PgdConfig::default()
            };
            let cfg_off = PgdConfig::default();

            // Unset env → config passthrough.
            env.remove("NY_PGD_GAMA");
            assert_eq!(gama_lambda_init(&cfg_on), Some(7.0));
            assert_eq!(gama_lambda_init(&cfg_off), None);

            let cfg_invalid = PgdConfig {
                gama_lambda: Some(f32::NAN),
                ..PgdConfig::default()
            };
            assert_eq!(
                gama_lambda_init(&cfg_invalid),
                None,
                "invalid configured guidance must preserve the raw route"
            );

            // NY_PGD_GAMA=0 → force off, even when the preset enabled it.
            env.set("NY_PGD_GAMA", "0");
            assert_eq!(gama_lambda_init(&cfg_on), None);

            // NY_PGD_GAMA=1 → force on with the default λ₀ when the config is off.
            env.set("NY_PGD_GAMA", "1");
            env.remove("NY_PGD_GAMA_LAMBDA");
            assert_eq!(
                gama_lambda_init(&cfg_off),
                Some(ny_propagate::GAMA_LAMBDA_DEFAULT)
            );
            env.set("NY_PGD_GAMA_LAMBDA", "7.5");
            assert_eq!(gama_lambda_init(&cfg_off), Some(7.5));
            for invalid in ["NaN", "-1", "0", "1e20", "not-a-number"] {
                env.set("NY_PGD_GAMA_LAMBDA", invalid);
                assert_eq!(
                    gama_lambda_init(&cfg_off),
                    None,
                    "present invalid lambda {invalid:?} must disable GAMA"
                );
            }
        });
    }
}
