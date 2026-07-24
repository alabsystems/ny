// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Engine-backed batched graph PGD helpers.
//!
//! The generic restart-batched SPSA route keeps its historical single-worst
//! raw objective when GAMA is off. When configured, it uses the same raw joint
//! hinge plus annealed softmax guidance as the sequential conjunctive route,
//! evaluated from the existing stacked SPSA outputs without extra forwards.

mod batching;
mod fallback;
#[cfg(test)]
mod tests;

use anyhow::Result;
use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::GemmEngine;
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::gama::{gama_lambda_at, gama_lin_steps};
use ny_propagate::{
    project_to_bounds_in_place, GraphNetwork, PgdConfig, PgdInitialization, PgdStepState,
};
#[cfg(test)]
use ny_propagate::{PgdAlphaMode, PgdOptimizer};
use ny_tensor::BoundedTensor;
use std::time::Instant;

use super::graph_pgd::{
    constraint_target, gama_lambda_init, gama_reference_softmax, joint_gama_objective,
    reset_gama_lane, GraphPgdTarget,
};
use super::pgd_sampling::spsa_step_deadline_exceeded;
use batching::{
    assign_batched_item, batched_item, fill_restart_perturbation, sample_uniform_batch,
    sample_uniform_point, RestartBatchLayout, SimpleRng,
};
use fallback::{
    graph_pgd_batching_error_should_skip, should_retry_with_folded_batch,
    skip_incompatible_batched_graph_pgd,
};

enum SpsaTarget {
    Relational(usize, usize),
    Constant(usize, f32),
    NegConstant(usize, f32),
}

impl SpsaTarget {
    fn margin(&self, output: &ArrayD<f32>) -> f32 {
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
}

#[cfg(test)]
pub(super) static GENERIC_BATCHED_GAMA_OBJECTIVE_EVALS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Scalar objective for one existing SPSA probe output. The `None` branch is
/// intentionally the historical single-target margin, preserving default-off
/// floating-point behavior. A configured lane instead uses the conjunctive raw
/// hinge plus guidance; candidate acceptance never consults this value.
fn batched_spsa_objective(
    legacy_target: &SpsaTarget,
    conjunct_targets: &[GraphPgdTarget],
    output: &ArrayD<f32>,
    gama: Option<(&[f32], f32)>,
) -> f32 {
    if let Some((p_ref, lambda)) = gama {
        joint_gama_objective(conjunct_targets, output, p_ref, lambda)
    } else {
        legacy_target.margin(output)
    }
}

/// Element budget per internal-forward chunk. Sized so a chunk stays small
/// enough that the deadline check between chunks is meaningful even on
/// vggnet16-class inputs (150528 elems/item → 1 item per chunk) while tiny
/// models (lsnc: 6 elems/item) still evaluate the whole batch in one dispatch.
const CHUNK_TARGET_ELEMS: usize = 262_144;

/// Outcome of a (chunked, deadline-checked) batched attack forward.
// Transient return value whose LARGE variant is the common case: boxing it
// would cost an allocation per batched forward for no storage win.
#[allow(clippy::large_enum_variant)]
enum BatchForward {
    Output(ArrayD<f32>),
    DeadlineExceeded,
}

/// Batched attack forward with deadline checks between chunks (#four-walls).
///
/// The previous implementation issued ONE un-preemptible
/// `propagate_concrete_point_preserve_leading_axis` for the whole restart
/// batch — 180s+ through the internal conv-GEMM path on vggnet16, past any
/// deadline and the vnncomp watchdog. This version:
///
/// 1. routes per-item forwards through the ORT attack oracle when available
///    (milliseconds per item, deadline-checked per item), and otherwise
/// 2. splits the batch into `CHUNK_TARGET_ELEMS`-sized chunks of the internal
///    TRUE point forward, checking the deadline between chunks.
///
/// The internal path remains the TRUE point forward (center-collapse per
/// node) preserving the restart axis. The whole-box `.lower()` is NOT the
/// network value for a point input (per-node soundness widening amplified by
/// the deep DAG) and fabricates false counterexamples that ORT rejects
/// (cgan_2023 unknown-downgrade). #cgan-eval.
fn evaluate_graph_batch(
    graph: &GraphNetwork,
    inputs: &ArrayD<f32>,
    gemm_engine: Option<&dyn GemmEngine>,
    num_items: usize,
    layout: RestartBatchLayout,
    deadline: Option<Instant>,
) -> Result<BatchForward> {
    let deadline_hit = || deadline.is_some_and(|d| Instant::now() >= d);
    if num_items == 0 || inputs.is_empty() {
        let input_bounds = BoundedTensor::concrete(inputs.clone())?;
        let output_bounds =
            graph.propagate_concrete_point_preserve_leading_axis(&input_bounds, gemm_engine)?;
        return Ok(BatchForward::Output(output_bounds.center()));
    }
    let item_elems = inputs.len() / num_items;

    // ORT-routed per-item scoring: only for the prepend-axis layout, where an
    // item is exactly one model input (the folded layout's item/batch
    // semantics are graph-specific and stay on the internal path).
    if matches!(layout, RestartBatchLayout::PrependAxis)
        && super::ort_attack::ort_attack_registered_for_len(item_elems)
    {
        if let Some(output) = evaluate_batch_via_ort(inputs, num_items, deadline)? {
            return Ok(output);
        }
        // Oracle became unavailable mid-batch — fall through to the internal
        // chunked path below.
    }

    let rows_per_item = layout.leading_extent(1)?;
    let chunk_items = (CHUNK_TARGET_ELEMS / item_elems.max(1)).clamp(1, num_items);
    if chunk_items >= num_items {
        // Single dispatch: cheap enough that chunking would only add overhead.
        let input_bounds = BoundedTensor::concrete(inputs.clone())?;
        let output_bounds =
            graph.propagate_concrete_point_preserve_leading_axis(&input_bounds, gemm_engine)?;
        return Ok(BatchForward::Output(output_bounds.center()));
    }

    let mut chunk_outputs: Vec<ArrayD<f32>> = Vec::new();
    let mut item_start = 0usize;
    while item_start < num_items {
        if deadline_hit() {
            return Ok(BatchForward::DeadlineExceeded);
        }
        let item_end = (item_start + chunk_items).min(num_items);
        let chunk = inputs
            .slice_axis(
                Axis(0),
                ndarray::Slice::from(item_start * rows_per_item..item_end * rows_per_item),
            )
            .to_owned()
            .into_dyn();
        let input_bounds = BoundedTensor::concrete(chunk)?;
        let output_bounds =
            graph.propagate_concrete_point_preserve_leading_axis(&input_bounds, gemm_engine)?;
        chunk_outputs.push(output_bounds.center());
        item_start = item_end;
    }
    let views: Vec<_> = chunk_outputs.iter().map(|o| o.view()).collect();
    Ok(BatchForward::Output(ndarray::concatenate(Axis(0), &views)?))
}

/// Per-item ORT forwards over a prepend-axis batch, deadline-checked per item.
///
/// Returns `Ok(None)` when the oracle stops answering (callers fall back to
/// the internal chunked forward).
fn evaluate_batch_via_ort(
    inputs: &ArrayD<f32>,
    num_items: usize,
    deadline: Option<Instant>,
) -> Result<Option<BatchForward>> {
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(num_items);
    let mut out_len: Option<usize> = None;
    for item in 0..num_items {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Ok(Some(BatchForward::DeadlineExceeded));
        }
        let point = inputs.index_axis(Axis(0), item);
        let flat: Vec<f32> = point.iter().copied().collect();
        let Some(output) = super::ort_attack::ort_forward_flat(&flat) else {
            return Ok(None);
        };
        if *out_len.get_or_insert(output.len()) != output.len() {
            return Ok(None);
        }
        rows.push(output);
    }
    let out_len = out_len.unwrap_or(0);
    let mut data = Vec::with_capacity(num_items * out_len);
    for row in rows {
        data.extend_from_slice(&row);
    }
    let output = ArrayD::from_shape_vec(IxDyn(&[num_items, out_len]), data)?;
    Ok(Some(BatchForward::Output(output)))
}

pub(super) enum BatchedGraphPgdOutcome {
    Completed(Option<Box<(ArrayD<f32>, ArrayD<f32>)>>),
    FallbackToSequential,
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

// Justification: the batched graph PGD helper mirrors the non-batched graph
// PGD entrypoint and forwards the same independent runtime inputs.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn try_graph_pgd_upfront_batched(
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
) -> Result<BatchedGraphPgdOutcome> {
    try_graph_pgd_upfront_batched_with_config(
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

pub(super) fn try_graph_pgd_upfront_batched_with_config(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    pgd_config: PgdConfig,
    gemm_engine: Option<&dyn GemmEngine>,
    json: bool,
) -> Result<BatchedGraphPgdOutcome> {
    let pgd_start = Instant::now();
    let constraints = &vnnlib.output_constraints;
    let requested_gama_lambda = gama_lambda_init(&pgd_config);
    let conjunct_targets: Vec<GraphPgdTarget> = if requested_gama_lambda.is_some() {
        constraints.iter().filter_map(constraint_target).collect()
    } else {
        Vec::new()
    };
    let gama_lambda0 = requested_gama_lambda.filter(|_| !conjunct_targets.is_empty());
    let gama_lin = gama_lambda0
        .map(|_| gama_lin_steps(pgd_config.num_steps))
        .unwrap_or(1);
    let spsa_delta = pgd_config
        .suggested_spsa_delta(input)
        .max(pgd_config.spsa_delta);
    let engine_label = if gemm_engine.is_some() { "GPU" } else { "CPU" };
    let deadline = pgd_config.deadline;
    let emit_batch_deadline = |context: &str| {
        super::graph_pgd::emit_graph_pgd_status(
            json,
            format_args!(
                "  Graph PGD: deadline during batched forward ({context}, {:.2}s, engine={})",
                pgd_start.elapsed().as_secs_f64(),
                engine_label
            ),
        );
        tracing::info!(
            "Graph PGD: deadline exceeded during batched forward ({})",
            context
        );
    };
    // Three-way outcome so the shape-mismatch fallback (sequential attack) and
    // the mid-batch deadline (attack phase over) stay distinguishable.
    // Transient return value whose LARGE variant is the common case: boxing
    // it would cost an allocation per batched forward for no storage win.
    #[allow(clippy::large_enum_variant)]
    enum BatchEval {
        Output(ArrayD<f32>),
        DeadlineExceeded,
        Incompatible,
    }
    let evaluate_or_skip =
        |inputs: &ArrayD<f32>, num_items: usize, layout: RestartBatchLayout| -> Result<BatchEval> {
            match evaluate_graph_batch(graph, inputs, gemm_engine, num_items, layout, deadline) {
                Ok(BatchForward::Output(output_batch)) => Ok(BatchEval::Output(output_batch)),
                Ok(BatchForward::DeadlineExceeded) => Ok(BatchEval::DeadlineExceeded),
                Err(err) if graph_pgd_batching_error_should_skip(&err) => {
                    skip_incompatible_batched_graph_pgd(&err, json, pgd_start, engine_label);
                    Ok(BatchEval::Incompatible)
                }
                Err(err) => Err(err),
            }
        };

    if pgd_config.deadline.is_some_and(|d| Instant::now() >= d) {
        super::graph_pgd::emit_graph_pgd_status(
            json,
            format_args!(
                "  Graph PGD: deadline at restart 0/{} ({:.2}s, engine={})",
                pgd_config.num_restarts,
                pgd_start.elapsed().as_secs_f64(),
                engine_label
            ),
        );
        tracing::info!(
            "Graph PGD: deadline exceeded before batched restart 0/{}",
            pgd_config.num_restarts
        );
        return Ok(BatchedGraphPgdOutcome::Completed(None));
    }

    let rngs: Vec<SimpleRng> = (0..pgd_config.num_restarts)
        .map(|restart| SimpleRng::new(42 + restart as u64))
        .collect();
    let mut layout = RestartBatchLayout::PrependAxis;
    let mut sampling_rngs = rngs.clone();
    let mut x_batch = sample_uniform_batch(input, &mut sampling_rngs, layout)?;
    let mut output_batch = match evaluate_graph_batch(
        graph,
        &x_batch,
        gemm_engine,
        pgd_config.num_restarts,
        layout,
        deadline,
    ) {
        Ok(BatchForward::Output(output_batch)) => output_batch,
        Ok(BatchForward::DeadlineExceeded) => {
            emit_batch_deadline("initial sample");
            return Ok(BatchedGraphPgdOutcome::Completed(None));
        }
        Err(err) if should_retry_with_folded_batch(input.shape(), &err) => {
            layout = RestartBatchLayout::FoldLeadingAxis {
                chunk: input.shape()[0],
            };
            sampling_rngs = rngs;
            x_batch = sample_uniform_batch(input, &mut sampling_rngs, layout)?;
            match evaluate_or_skip(&x_batch, pgd_config.num_restarts, layout)? {
                BatchEval::Output(output_batch) => output_batch,
                BatchEval::DeadlineExceeded => {
                    emit_batch_deadline("initial sample (folded)");
                    return Ok(BatchedGraphPgdOutcome::Completed(None));
                }
                BatchEval::Incompatible => {
                    return Ok(BatchedGraphPgdOutcome::FallbackToSequential);
                }
            }
        }
        Err(err) if graph_pgd_batching_error_should_skip(&err) => {
            skip_incompatible_batched_graph_pgd(&err, json, pgd_start, engine_label);
            return Ok(BatchedGraphPgdOutcome::FallbackToSequential);
        }
        Err(err) => return Err(err),
    };
    let mut rngs = sampling_rngs;
    let mut step_states: Vec<PgdStepState> = (0..pgd_config.num_restarts)
        .map(|_| {
            PgdStepState::from_config(
                pgd_config.optimizer,
                pgd_config.alpha_mode,
                pgd_config.step_size,
                pgd_config.adam,
                input,
                input.shape(),
            )
        })
        .collect();

    // Apply OSI initialization per-restart if configured (#1449).
    // OSI requires sequential forward passes per restart to push seeds toward
    // diverse output-space regions before the batched PGD loop.
    if matches!(pgd_config.initialization, PgdInitialization::Osi) && pgd_config.osi_steps > 0 {
        for restart in 0..pgd_config.num_restarts {
            // OSI runs sequential per-restart forwards — keep it preemptible
            // on expensive models (#four-walls).
            if deadline.is_some_and(|d| Instant::now() >= d) {
                emit_batch_deadline("OSI initialization");
                return Ok(BatchedGraphPgdOutcome::Completed(None));
            }
            let mut rng = super::graph_pgd::SimpleRng::new(42 + restart as u64);
            let osi_point = super::graph_pgd::initialize_graph_point(
                &pgd_config,
                graph,
                input,
                &mut rng,
                gemm_engine,
            )?;
            assign_batched_item(&mut x_batch, restart, layout, &osi_point);
        }
        // Recompute output_batch after OSI initialization.
        output_batch = match evaluate_or_skip(&x_batch, pgd_config.num_restarts, layout)? {
            BatchEval::Output(recomputed) => recomputed,
            BatchEval::DeadlineExceeded => {
                emit_batch_deadline("OSI re-evaluation");
                return Ok(BatchedGraphPgdOutcome::Completed(None));
            }
            BatchEval::Incompatible => {
                return Ok(BatchedGraphPgdOutcome::FallbackToSequential);
            }
        };
    }

    // Capture each lane's fixed reference from the already-computed point after
    // uniform/OSI initialization. Empty vectors when GAMA is off keep the hot
    // objective branch and all floating-point operations byte-identical.
    let mut gama_p_refs: Vec<Option<Vec<f32>>> = if gama_lambda0.is_some() {
        (0..pgd_config.num_restarts)
            .map(|restart| gama_reference_softmax(&batched_item(&output_batch, restart, layout)))
            .collect()
    } else {
        Vec::new()
    };
    let mut gama_steps = if gama_lambda0.is_some() {
        vec![0usize; pgd_config.num_restarts]
    } else {
        Vec::new()
    };
    if let Some(lambda0) = gama_lambda0 {
        super::graph_pgd::emit_graph_pgd_status(
            json,
            format_args!(
                "  Graph PGD: generic batched SPSA uses GAMA-guided joint objective (lambda0={lambda0}, {} modeled conjuncts; raw constraints + independent revalidation remain authoritative)",
                conjunct_targets.len()
            ),
        );
    }

    for restart in 0..pgd_config.num_restarts {
        let output = batched_item(&output_batch, restart, layout);
        if super::check_unsafe_counterexample(&output, constraints) {
            let candidate = batched_item(&x_batch, restart, layout);
            let ctx = format!("batched restart {restart} (random sample)");
            if let Some(pair) = super::graph_pgd::revalidate_graph_counterexample(
                graph,
                candidate,
                constraints,
                &ctx,
            )? {
                return Ok(BatchedGraphPgdOutcome::Completed(Some(Box::new(pair))));
            }
        }
    }

    for step in 0..pgd_config.num_steps {
        if spsa_step_deadline_exceeded(step, pgd_config.deadline) {
            super::graph_pgd::emit_graph_pgd_status(
                json,
                format_args!(
                    "  Graph PGD: deadline at step {}/{} ({:.2}s, engine={})",
                    step,
                    pgd_config.num_steps,
                    pgd_start.elapsed().as_secs_f64(),
                    engine_label
                ),
            );
            tracing::info!(
                "Graph PGD: deadline exceeded in batched path at step {}/{}",
                step,
                pgd_config.num_steps
            );
            return Ok(BatchedGraphPgdOutcome::Completed(None));
        }

        let mut perturbation_batch = ArrayD::zeros(IxDyn(
            &layout.batch_shape(input.shape(), pgd_config.num_restarts)?,
        ));
        let mut targets = Vec::with_capacity(pgd_config.num_restarts);
        let mut has_target = false;

        for (restart, rng) in rngs.iter_mut().enumerate() {
            let output = batched_item(&output_batch, restart, layout);
            let mut min_margin = f32::INFINITY;
            let mut worst_target: Option<SpsaTarget> = None;

            for constraint in constraints.iter() {
                match constraint {
                    OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => {
                        let yi = output.iter().nth(*i).copied().unwrap_or(0.0);
                        let yj = output.iter().nth(*j).copied().unwrap_or(0.0);
                        let margin = yj - yi;
                        if margin < min_margin {
                            min_margin = margin;
                            worst_target = Some(SpsaTarget::Relational(*j, *i));
                        }
                    }
                    OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
                        let yi = output.iter().nth(*i).copied().unwrap_or(0.0);
                        let yj = output.iter().nth(*j).copied().unwrap_or(0.0);
                        let margin = yi - yj;
                        if margin < min_margin {
                            min_margin = margin;
                            worst_target = Some(SpsaTarget::Relational(*i, *j));
                        }
                    }
                    OutputConstraint::GreaterEqConst(i, c)
                    | OutputConstraint::GreaterThanConst(i, c) => {
                        let y = output.iter().nth(*i).copied().unwrap_or(0.0);
                        let margin = y - *c as f32;
                        if margin < min_margin {
                            min_margin = margin;
                            worst_target = Some(SpsaTarget::Constant(*i, *c as f32));
                        }
                    }
                    OutputConstraint::LessEqConst(i, c) | OutputConstraint::LessThanConst(i, c) => {
                        let y = output.iter().nth(*i).copied().unwrap_or(0.0);
                        let margin = *c as f32 - y;
                        if margin < min_margin {
                            min_margin = margin;
                            worst_target = Some(SpsaTarget::NegConstant(*i, *c as f32));
                        }
                    }
                    _ => {}
                }
            }

            if worst_target.is_some() {
                has_target = true;
            }
            targets.push(worst_target);
            fill_restart_perturbation(&mut perturbation_batch, restart, layout, rng);
        }

        if !has_target {
            break;
        }

        let mut x_plus_batch = &x_batch + &(&perturbation_batch * spsa_delta);
        let mut x_minus_batch = &x_batch - &(&perturbation_batch * spsa_delta);
        for restart in 0..pgd_config.num_restarts {
            let mut x_plus = batched_item(&x_plus_batch, restart, layout);
            let mut x_minus = batched_item(&x_minus_batch, restart, layout);
            project_to_bounds_in_place(&mut x_plus, input.lower(), input.upper());
            project_to_bounds_in_place(&mut x_minus, input.lower(), input.upper());
            assign_batched_item(&mut x_plus_batch, restart, layout, &x_plus);
            assign_batched_item(&mut x_minus_batch, restart, layout, &x_minus);
        }

        let stacked = ndarray::concatenate(Axis(0), &[x_plus_batch.view(), x_minus_batch.view()])?;
        let outputs = match evaluate_or_skip(&stacked, pgd_config.num_restarts * 2, layout)? {
            BatchEval::Output(outputs) => outputs,
            BatchEval::DeadlineExceeded => {
                emit_batch_deadline("SPSA probe");
                return Ok(BatchedGraphPgdOutcome::Completed(None));
            }
            BatchEval::Incompatible => {
                return Ok(BatchedGraphPgdOutcome::FallbackToSequential);
            }
        };
        let (output_plus, output_minus) = outputs
            .view()
            .split_at(Axis(0), layout.leading_extent(pgd_config.num_restarts)?);

        for (restart, target) in targets.iter().enumerate() {
            let Some(target) = target else {
                continue;
            };

            let out_plus = batched_item(&output_plus, restart, layout);
            let out_minus = batched_item(&output_minus, restart, layout);
            let gama = gama_lambda0.and_then(|lambda0| {
                gama_p_refs[restart].as_deref().map(|p_ref| {
                    (
                        p_ref,
                        gama_lambda_at(lambda0, gama_steps[restart], gama_lin),
                    )
                })
            });
            #[cfg(test)]
            if gama.is_some() {
                GENERIC_BATCHED_GAMA_OBJECTIVE_EVALS
                    .fetch_add(2, std::sync::atomic::Ordering::Relaxed);
            }
            let obj_plus = batched_spsa_objective(target, &conjunct_targets, &out_plus, gama);
            let obj_minus = batched_spsa_objective(target, &conjunct_targets, &out_minus, gama);

            if obj_plus.is_nan() || obj_minus.is_nan() {
                continue;
            }

            let grad = batched_item(&perturbation_batch, restart, layout)
                * ((obj_plus - obj_minus) / (2.0 * spsa_delta));
            let previous_x = if pgd_config.restart_when_stuck {
                Some(batched_item(&x_batch, restart, layout))
            } else {
                None
            };
            let mut x = batched_item(&x_batch, restart, layout);
            x = step_states[restart].step(&grad, &x, input, true);
            if gama_lambda0.is_some() && gama_p_refs[restart].is_some() {
                gama_steps[restart] = gama_steps[restart].saturating_add(1);
            }

            // Restart-when-stuck (#4278): resample if projected step is a no-op.
            if let Some(ref prev) = previous_x {
                if prev.iter().zip(x.iter()).all(|(&a, &b)| a == b) {
                    x = sample_uniform_point(input, &mut rngs[restart]);
                    step_states[restart].reset();
                    if gama_lambda0.is_some() {
                        reset_gama_lane(&mut gama_p_refs[restart], &mut gama_steps[restart]);
                    }
                }
            }
            assign_batched_item(&mut x_batch, restart, layout, &x);
        }

        output_batch = match evaluate_or_skip(&x_batch, pgd_config.num_restarts, layout)? {
            BatchEval::Output(recomputed) => recomputed,
            BatchEval::DeadlineExceeded => {
                emit_batch_deadline("step re-evaluation");
                return Ok(BatchedGraphPgdOutcome::Completed(None));
            }
            BatchEval::Incompatible => {
                return Ok(BatchedGraphPgdOutcome::FallbackToSequential);
            }
        };
        for restart in 0..pgd_config.num_restarts {
            let output = batched_item(&output_batch, restart, layout);
            // A stuck-lane resample starts a fresh trajectory. Its first
            // already-computed output becomes the new fixed GAMA reference;
            // no additional model evaluation is issued.
            if gama_lambda0.is_some() && gama_p_refs[restart].is_none() {
                gama_p_refs[restart] = gama_reference_softmax(&output);
            }
            if super::check_unsafe_counterexample(&output, constraints) {
                let candidate = batched_item(&x_batch, restart, layout);
                let ctx = format!("batched restart {restart}, step {step}");
                if let Some(pair) = super::graph_pgd::revalidate_graph_counterexample(
                    graph,
                    candidate,
                    constraints,
                    &ctx,
                )? {
                    return Ok(BatchedGraphPgdOutcome::Completed(Some(Box::new(pair))));
                }
            }
        }
    }

    // `output_batch` is already consistent with the current `x_batch`: the
    // per-step recompute (line ~385) refreshes it after every completed step,
    // and on the no-target/early-break or num_steps==0 paths `x_batch` was never
    // mutated past the initial/OSI evaluation that seeded `output_batch`. Reuse
    // it for the final counterexample sweep instead of issuing a redundant batched
    // forward pass — this keeps the batched GEMM-dispatch count at two per step
    // (#3955).
    for restart in 0..pgd_config.num_restarts {
        let output = batched_item(&output_batch, restart, layout);
        if super::check_unsafe_counterexample(&output, constraints) {
            let candidate = batched_item(&x_batch, restart, layout);
            let ctx = format!("batched restart {restart} (final check)");
            if let Some(pair) = super::graph_pgd::revalidate_graph_counterexample(
                graph,
                candidate,
                constraints,
                &ctx,
            )? {
                return Ok(BatchedGraphPgdOutcome::Completed(Some(Box::new(pair))));
            }
        }
    }

    super::graph_pgd::emit_graph_pgd_status(
        json,
        format_args!(
            "  Graph PGD: no counterexample found. ({:.2}s, engine={})",
            pgd_start.elapsed().as_secs_f64(),
            engine_label
        ),
    );
    Ok(BatchedGraphPgdOutcome::Completed(None))
}
