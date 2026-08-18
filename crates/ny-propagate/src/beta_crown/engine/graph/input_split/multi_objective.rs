// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-objective conjunctive input-split BaB verifier.

mod affine_conic;
mod screen_child;
mod selective_direct;

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::Array2;
use ny_core::phase_yield::{from_result as phase_yield_from_result, PhaseYield};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{info, trace};

use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::bounds::GraphAlphaState;
use crate::{ConjunctiveProofObjectives, GraphNetwork};

use self::screen_child::screen_multi_obj_child;
use super::super::shared::state::GraphBabLifecycle;
use super::batching::{
    bound_deferred_multi_obj_domains_batch, input_split_loop_batch_size,
    pop_multi_obj_input_domain_batch, root_map_spec_obj_bounds,
};
use super::build_batches::compute_crown_or_ibp_bounds_in_build_batches;
use super::loop_batch_size::InputSplitLoopBatchDecision;
use super::mul_binary::maybe_optimize_mul_binary_alphas;
use super::root_bounds::collect_input_split_root_node_bounds;
use super::shared::{
    build_child_input_owned, compute_crown_or_ibp_bounds_with_node_bounds, extract_obj_bounds,
    multi_dim_split_boxes, multi_obj_domain_priority, multi_obj_domain_verified,
    MultiObjInputDomain,
};
use super::shared_specs::compute_crown_or_ibp_bounds_batched_specs;
use crate::beta_crown::engine::BetaCrownVerifier;

struct SelectiveDirectPlan {
    spec_matrix: Array2<f32>,
    thresholds: [f32; 1],
}

#[derive(Default)]
struct SelectiveDirectTelemetry {
    attempted_rows: usize,
    completed_rows: usize,
    microbatches: usize,
    late_discarded: usize,
    phase_declines: usize,
    errors: usize,
    elapsed: Duration,
    root_attempted: bool,
    root_completed: bool,
    root_slice_exhausted: bool,
    budget_exhausted: bool,
    disabled: bool,
}

impl SelectiveDirectTelemetry {
    fn log_summary(
        &self,
        outcome: &'static str,
        quota: &selective_direct::SelectiveDirectQuota,
        closures: usize,
        explored: usize,
    ) {
        if !self.root_attempted && self.attempted_rows == 0 && self.errors == 0 {
            return;
        }
        info!(
            outcome,
            closures,
            root_attempted = self.root_attempted,
            root_completed = self.root_completed,
            root_slice_exhausted = self.root_slice_exhausted,
            budget_exhausted = self.budget_exhausted,
            scheduled_nonroot_rows = quota.selected(),
            attempted_nonroot_rows = self.attempted_rows,
            completed_nonroot_rows = self.completed_rows,
            microbatches = self.microbatches,
            candidates_seen = quota.candidates_seen(),
            late_discarded = self.late_discarded,
            phase_declines = self.phase_declines,
            errors = self.errors,
            disabled = self.disabled,
            elapsed_s = self.elapsed.as_secs_f64(),
            explored,
            "[multi-obj] authenticated selective direct-conic summary"
        );
    }
}

#[inline]
pub(super) fn multi_objective_loop_batch_decision(
    requested_batch_size: usize,
    input_elems: usize,
    affine_conic_closure: bool,
    conic_queue_refresh_batch_size: usize,
) -> Result<InputSplitLoopBatchDecision> {
    let decision = input_split_loop_batch_size(requested_batch_size, input_elems)?;
    if affine_conic_closure {
        decision.with_conic_queue_refresh_cap(conic_queue_refresh_batch_size)
    } else {
        Ok(decision)
    }
}

/// Resolve the BaB wall slice for conjunctive multi-objective input splitting.
///
/// A caller-supplied deadline is already an authoritative phase boundary (the
/// CLI ledger has reserved its post-BaB slice). Only the convenience path with
/// no deadline derives that reservation from the configured timeout.
pub(super) fn multi_objective_bab_timeout(
    configured_timeout: Duration,
    post_bab_pgd_fraction: f32,
    deadline: Option<Instant>,
    now: Instant,
) -> Duration {
    match deadline {
        Some(dl) => dl.saturating_duration_since(now),
        None => configured_timeout.mul_f32(1.0 - post_bab_pgd_fraction.clamp(0.0, 0.5)),
    }
}

impl BetaCrownVerifier {
    /// Verify an authenticated proof-only objective plan for conjunctive graph
    /// input splitting.
    ///
    /// This is the typed verifier boundary for derived objectives. It checks
    /// the explicit config gate, refuses certificate-export authority (the
    /// external format cannot encode this provenance yet), and revalidates the
    /// sealed plan before delegating to the shared multi-objective
    /// implementation.
    /// The CLI separately authenticates the exact source-property AST as a
    /// narrow rollout policy; sound programmatic callers may construct the same
    /// sealed plan directly from matching source constraints.
    pub fn verify_graph_input_split_conjunctive_proof_objectives(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        proof_objectives: &ConjunctiveProofObjectives,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        if !self.config.input_split_conic_objective_eligible() {
            return Err(NyError::InvalidSpec(
                "Conjunctive proof objectives require an enabled conic gate and verdict-only artifact authority"
                    .to_string(),
            ));
        }
        if !proof_objectives.has_valid_provenance() {
            return Err(NyError::InvalidSpec(
                "Conjunctive proof objective provenance failed revalidation".to_string(),
            ));
        }

        // Keep the hot CROWN carrier at the two source rows. The sealed third
        // row remains available to the implementation for bounded selective
        // direct propagation, which preserves shared nonlinear cancellation
        // without perturbing every domain's alpha, priority, split, and clip
        // state.
        self.verify_graph_input_split_multi_objective_conjunctive_impl(
            graph,
            input,
            &proof_objectives.objectives()[..2],
            &proof_objectives.thresholds()[..2],
            engine,
            deadline,
            Some((
                proof_objectives.objectives()[2].as_slice(),
                proof_objectives.thresholds()[2],
            )),
        )
    }

    /// Multi-objective conjunctive input-split verification.
    ///
    /// Uses a multi-row spec_matrix to compute bounds on all objectives simultaneously
    /// in a single CROWN backward pass (preserving output correlations). A subdomain is
    /// verified when ANY objective has `lower > threshold` (conjunctive: if any conjunct
    /// is provably impossible, the conjunction cannot hold).
    ///
    /// Reference: alpha-beta-CROWN `stop_criterion_batch_any` in
    /// `auto_LiRPA/utils.py:107-113`, `multi_spec_keep_func_all` in
    /// `auto_LiRPA/utils.py:143-144`.
    /// `deadline`: If `Some`, the BaB engine derives its phase budgets from
    /// remaining wall-clock time instead of `self.config.timeout` (#4321).
    pub fn verify_graph_input_split_multi_objective_conjunctive(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        self.verify_graph_input_split_multi_objective_conjunctive_impl(
            graph, input, objectives, thresholds, engine, deadline, None,
        )
    }

    /// Shared implementation. `authenticated_direct` is reachable only through
    /// the proof-objective boundary above; raw callers cannot acquire synthetic
    /// proof authority by setting a boolean or reconstructing a row.
    #[allow(clippy::too_many_arguments)]
    fn verify_graph_input_split_multi_objective_conjunctive_impl(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objectives: &[Vec<f32>],
        thresholds: &[f32],
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        authenticated_direct: Option<(&[f32], f32)>,
    ) -> Result<BetaCrownResult> {
        self.config.validate()?;
        // This engine's root, queue, and child stopping rules all authorize a
        // verdict from `lower > threshold`. CLI callers sign-normalize upper
        // constraints before reaching this API; direct callers must do the
        // same. Accepting the scalar engine's direction switch here would make
        // the multi-objective stopping rule unsound.
        if self.config.verify_upper_bound {
            return Err(NyError::InvalidSpec(
                "graph input-split multi-objective verification requires sign-normalized \
                 lower-bound objectives (verify_upper_bound=false)"
                    .to_string(),
            ));
        }
        let engine = self.resolve_engine(engine);
        if objectives.is_empty() || objectives.len() != thresholds.len() {
            return Err(NyError::InvalidSpec(format!(
                "Multi-objective: {} objectives vs {} thresholds",
                objectives.len(),
                thresholds.len()
            )));
        }

        // Reject non-finite thresholds early. NaN thresholds make IEEE 754 comparisons
        // silently false, causing BaB to run to exhaustion. Part of #3646.
        for (i, &t) in thresholds.iter().enumerate() {
            if !t.is_finite() {
                return Err(NyError::NumericalInstability(format!(
                    "Multi-objective threshold[{}] is non-finite ({}); \
                     BaB cannot make progress with NaN/Inf thresholds",
                    i, t
                )));
            }
        }

        let graph = self.configured_graph_for_crown(graph);
        let graph = &graph;

        let num_specs = objectives.len();
        let spec_dim = objectives[0].len();
        let affine_conic_closure = authenticated_direct.is_some();
        let affine_conic_source_thresholds = if affine_conic_closure {
            Some(thresholds.get(..2).ok_or_else(|| {
                NyError::InvalidSpec(
                    "Authenticated affine-conic plan is missing its two source thresholds"
                        .to_string(),
                )
            })?)
        } else {
            None
        };
        let selective_direct_plan =
            if let Some((direct_row, direct_threshold)) = authenticated_direct {
                if num_specs != 2
                    || spec_dim != 2
                    || direct_row.len() != spec_dim
                    || direct_row.iter().any(|value| !value.is_finite())
                    || !direct_threshold.is_finite()
                {
                    return Err(NyError::InvalidSpec(
                        "Authenticated selective direct-conic plan has an invalid row layout"
                            .to_string(),
                    ));
                }
                Some(SelectiveDirectPlan {
                    spec_matrix: Array2::from_shape_vec((1, spec_dim), direct_row.to_vec())
                        .map_err(|err| NyError::InvalidSpec(format!("direct conic spec: {err}")))?,
                    thresholds: [direct_threshold],
                })
            } else {
                None
            };

        // Build multi-row spec matrix: each row is one C-matrix row (objective).
        let mut spec_data = Vec::with_capacity(num_specs * spec_dim);
        for obj in objectives {
            if obj.len() != spec_dim {
                return Err(NyError::InvalidSpec(format!(
                    "Objective dimension mismatch: {} vs {}",
                    obj.len(),
                    spec_dim
                )));
            }
            spec_data.extend_from_slice(obj);
        }
        let spec_matrix = Array2::from_shape_vec((num_specs, spec_dim), spec_data)
            .map_err(|e| NyError::InvalidSpec(format!("spec matrix: {}", e)))?;

        let now = Instant::now();
        let mut lifecycle = GraphBabLifecycle::new(now);
        // Warmup cap (#2206 Packet C, #4095): initial bounds get at most
        // `initial_bounds_fraction` of the BaB timeout. Mirrors core.rs pattern.
        //
        // When a wall-clock deadline is provided (#4321), derive the effective
        // timeout from remaining time instead of the configured timeout.
        let pgd_frac = self
            .config
            .phase_budget
            .post_bab_pgd_fraction
            .clamp(0.0, 0.5);
        let bab_timeout = multi_objective_bab_timeout(self.config.timeout, pgd_frac, deadline, now);
        let selective_direct_budget =
            bab_timeout.mul_f32(selective_direct::SELECTIVE_DIRECT_BAB_FRACTION);
        let selective_direct_root_budget = bab_timeout
            .mul_f32(selective_direct::SELECTIVE_DIRECT_ROOT_BAB_FRACTION)
            .min(selective_direct_budget);
        let initial_deadline = {
            let frac = self
                .config
                .phase_budget
                .initial_bounds_fraction
                .clamp(0.0, 1.0);
            Some(GraphBabLifecycle::fail_closed_deadline(
                now,
                bab_timeout.mul_f32(frac),
            ))
        };
        // Per-domain deadline for CROWN backward passes in the BaB loop.
        let crown_deadline = Some(GraphBabLifecycle::fail_closed_deadline(now, bab_timeout));
        let mut domains_verified_by_clip = 0usize;
        let mut domains_verified_by_affine_conic = 0usize;
        let mut domains_verified_by_selective_direct = 0usize;
        let mut selective_direct_telemetry = SelectiveDirectTelemetry::default();

        let (root_node_bounds, root_alpha_state): (
            Option<HashMap<String, BoundedTensor>>,
            Option<GraphAlphaState>,
        ) = collect_input_split_root_node_bounds(
            graph,
            input,
            &self.config,
            engine,
            initial_deadline,
            "multi-objective input splitting",
            None,
        )?;

        // Retest the already-collected, certified output box before paying for
        // MulBinary optimization or a fresh spec-CROWN backward. Projection is
        // outward-rounded; a missing output, shape mismatch, non-finite bound,
        // or non-decisive result fails open to the historical path below.
        if let Some(root_map_obj_bounds) = root_node_bounds
            .as_ref()
            .and_then(|root_map| root_map_spec_obj_bounds(graph, root_map, &spec_matrix))
        {
            if multi_obj_domain_verified(&root_map_obj_bounds, thresholds) {
                info!(
                    "[multi-obj] certified root-map output box verifies the root; skipping fresh spec-CROWN and child bounding"
                );
                lifecycle.domains_explored = 1;
                lifecycle.domains_verified = 1;
                return Ok(lifecycle.build_result(BabVerificationStatus::Verified));
            }
        }

        // Phase 4 (#3439): MulBinary SPSA alpha optimization.
        // Uses initial_deadline so warmup phases respect the cap (#4095).
        let mul_binary_alphas_multi = maybe_optimize_mul_binary_alphas(
            graph,
            input,
            &spec_matrix,
            engine,
            initial_deadline,
            self.config.crown_backward_layers,
            "Graph input split (multi-objective)",
        )?;

        // Per-domain bound computation: CROWN with IBP fallback, returning per-objective
        // bounds and optional LinearBounds for split dimension scoring.
        let crown_bkwd = self.config.crown_backward_layers;
        // The compact row deliberately uses shared/default alpha only. A
        // one-row carrier must never reinterpret source-row-0 spec deltas as
        // belonging to the derived row. MulBinary alphas are also omitted from
        // this optional proof pass so every unsupported or specialized graph
        // can fail open to the already-computed source search unchanged.
        let compute_selective_direct_bounds =
            |input_bounds_batch: &[&BoundedTensor],
             direct_plan: &SelectiveDirectPlan,
             direct_deadline: Option<Instant>| {
                compute_crown_or_ibp_bounds_batched_specs(
                    graph,
                    input_bounds_batch,
                    &direct_plan.spec_matrix,
                    engine,
                    root_node_bounds.as_ref(),
                    None,
                    None,
                    direct_deadline,
                    crown_bkwd,
                    self.config.input_split_ibp_enhancement,
                    false,
                )
            };
        let compute_bounds = |input_bounds: &BoundedTensor,
                              node_bounds: Option<&HashMap<String, BoundedTensor>>|
         -> Result<super::shared::MultiObjBounds> {
            let (bounds, linear) = compute_crown_or_ibp_bounds_with_node_bounds(
                graph,
                input_bounds,
                &spec_matrix,
                engine,
                root_node_bounds.as_ref(),
                node_bounds,
                root_alpha_state.as_ref(),
                mul_binary_alphas_multi.as_ref(),
                crown_deadline,
                crown_bkwd,
                self.config.input_split_ibp_enhancement,
            )?;
            Ok((extract_obj_bounds(&bounds, num_specs)?, linear))
        };

        // Per-sub-domain α refinement closure (alpha-beta-CROWN
        // input_split/bounding.py:90-179), ported from the single-objective loop
        // (single_objective.rs). Built only when the knob is enabled AND α-CROWN
        // is in use AND a root α state exists to seed children from. When `None`,
        // the per-child path uses the frozen `compute_bounds` above — i.e. ny's
        // historical single frozen-alpha pass (the no-regression default).
        let warm_alpha_enabled = self.config.input_split_alpha_iteration > 0
            && self.config.use_alpha_crown
            && root_alpha_state.is_some();
        let warm_compute_bounds = |input_bounds: &BoundedTensor,
                                   node_bounds: Option<&HashMap<String, BoundedTensor>>,
                                   parent_alpha: &GraphAlphaState|
         -> Result<screen_child::WarmMultiObjBoundsResult> {
            let (bounds, linear, refined_alpha) =
                super::shared::compute_warm_start_crown_bounds_with_refined_alpha(
                    graph,
                    input_bounds,
                    &spec_matrix,
                    engine,
                    node_bounds,
                    parent_alpha,
                    mul_binary_alphas_multi.as_ref(),
                    crown_deadline,
                    crown_bkwd,
                    &self.config,
                )?;
            Ok((
                extract_obj_bounds(&bounds, num_specs)?,
                linear,
                refined_alpha,
            ))
        };
        let warm_compute_bounds_opt: Option<&screen_child::WarmMultiObjComputeBoundsFn<'_>> =
            if warm_alpha_enabled {
                Some(&warm_compute_bounds)
            } else {
                None
            };

        // Root domain bounds via shared CROWN-or-IBP dispatch (#3453).
        let (root_bounds, root_linear) = compute_crown_or_ibp_bounds_in_build_batches(
            graph,
            input,
            &spec_matrix,
            self.config.build_batch_size,
            engine,
            root_node_bounds.as_ref(),
            root_alpha_state.as_ref(),
            mul_binary_alphas_multi.as_ref(),
            initial_deadline,
            crown_bkwd,
            self.config.input_split_ibp_enhancement,
        )?;
        let root_obj_bounds = extract_obj_bounds(&root_bounds, num_specs)?;

        info!(
            "[multi-obj] {} objectives, root bounds (alpha={}, forward_bounds={}): {}",
            num_specs,
            self.config.use_alpha_crown,
            self.config.use_forward_bounds,
            root_obj_bounds
                .iter()
                .zip(thresholds.iter())
                .map(|((l, u), &t)| format!("[{:.6}, {:.6}] thr={:.6}", l, u, t))
                .collect::<Vec<_>>()
                .join(", ")
        );

        if multi_obj_domain_verified(&root_obj_bounds, thresholds) {
            lifecycle.domains_explored = 1;
            lifecycle.domains_verified = 1;
            return Ok(lifecycle.build_result(BabVerificationStatus::Verified));
        }
        if let Some(source_thresholds) = affine_conic_source_thresholds {
            if let Some(evaluation) = root_linear.as_ref().and_then(|linear| {
                affine_conic::evaluate_affine_conic_closure(linear, input, source_thresholds)
            }) {
                info!(
                    lower_bound = evaluation.lower_bound,
                    threshold_upper = evaluation.threshold_upper,
                    gap = evaluation.gap(),
                    lhs_weight = evaluation.lhs_weight,
                    rhs_weight = evaluation.rhs_weight,
                    verified = evaluation.verifies(),
                    "[multi-obj] authenticated affine conic root evaluation"
                );
                if evaluation.verifies() {
                    lifecycle.domains_explored = 1;
                    lifecycle.domains_verified = 1;
                    return Ok(lifecycle.build_result(BabVerificationStatus::Verified));
                }
            }
        }
        let active_root_direct_plan = if !selective_direct_telemetry.disabled
            && crown_deadline.is_none_or(|limit| Instant::now() < limit)
        {
            selective_direct_plan.as_ref()
        } else {
            None
        };
        if let Some(direct_plan) = active_root_direct_plan {
            let direct_started = Instant::now();
            if let Some(direct_deadline) = selective_direct::call_deadline(
                direct_started,
                crown_deadline,
                selective_direct_budget,
                selective_direct_telemetry.elapsed,
                selective_direct_root_budget,
            ) {
                selective_direct_telemetry.root_attempted = true;
                let root_batch = [input];
                let direct_result = compute_selective_direct_bounds(
                    &root_batch,
                    direct_plan,
                    Some(direct_deadline),
                );
                let direct_finished = Instant::now();
                selective_direct_telemetry.elapsed +=
                    direct_finished.duration_since(direct_started);
                let global_expired = crown_deadline.is_some_and(|limit| direct_finished >= limit);
                if direct_finished >= direct_deadline && !global_expired {
                    selective_direct_telemetry.root_slice_exhausted = true;
                }
                if selective_direct_telemetry.elapsed >= selective_direct_budget {
                    selective_direct_telemetry.budget_exhausted = true;
                    selective_direct_telemetry.disabled = true;
                }
                let direct_result = phase_yield_from_result(
                    direct_result,
                    direct_finished,
                    Some(direct_deadline),
                    crown_deadline,
                    NyError::is_deadline_exceeded,
                );
                let usable_result = match direct_result {
                    Ok(PhaseYield::Complete(_) | PhaseYield::Partial(_)) if global_expired => {
                        selective_direct_telemetry.late_discarded += 1;
                        info!(
                            root_attempted = selective_direct_telemetry.root_attempted,
                            late_discarded = selective_direct_telemetry.late_discarded,
                            elapsed_s = selective_direct_telemetry.elapsed.as_secs_f64(),
                            "[multi-obj] discarded selective direct-conic root result after global deadline"
                        );
                        None
                    }
                    Ok(PhaseYield::Complete(result) | PhaseYield::Partial(result)) => Some(result),
                    Ok(PhaseYield::Declined(reason)) => {
                        selective_direct_telemetry.phase_declines += 1;
                        info!(
                            ?reason,
                            root_slice_exhausted = selective_direct_telemetry.root_slice_exhausted,
                            elapsed_s = selective_direct_telemetry.elapsed.as_secs_f64(),
                            "[multi-obj] selective direct-conic root phase declined"
                        );
                        None
                    }
                    Err(err) if err.is_deadline_exceeded() => {
                        info!(
                            error = %err,
                            elapsed_s = selective_direct_telemetry.elapsed.as_secs_f64(),
                            "[multi-obj] selective direct-conic root reached the global deadline"
                        );
                        None
                    }
                    Err(err) => {
                        selective_direct_telemetry.errors += 1;
                        selective_direct_telemetry.disabled = true;
                        info!(
                            error = %err,
                            errors = selective_direct_telemetry.errors,
                            "[multi-obj] disabling optional selective direct-conic pass after root error"
                        );
                        None
                    }
                };
                if let Some(result) = usable_result {
                    if result.bounds.len() != 1 {
                        selective_direct_telemetry.errors += 1;
                        selective_direct_telemetry.disabled = true;
                        info!(
                            expected_bounds = 1,
                            actual_bounds = result.bounds.len(),
                            errors = selective_direct_telemetry.errors,
                            "[multi-obj] disabling optional selective direct-conic pass after root cardinality mismatch"
                        );
                    } else {
                        let root_direct_bounds = match extract_obj_bounds(&result.bounds[0], 1) {
                            Ok(bounds) => bounds,
                            Err(err) => {
                                selective_direct_telemetry.errors += 1;
                                selective_direct_telemetry.disabled = true;
                                info!(
                                    error = %err,
                                    errors = selective_direct_telemetry.errors,
                                    "[multi-obj] disabling optional selective direct-conic pass after malformed root bound"
                                );
                                Vec::new()
                            }
                        };
                        selective_direct_telemetry.root_completed = !root_direct_bounds.is_empty();
                        let root_direct_verified = selective_direct_telemetry.root_completed
                            && multi_obj_domain_verified(
                                &root_direct_bounds,
                                &direct_plan.thresholds,
                            );
                        info!(
                            root_attempted = selective_direct_telemetry.root_attempted,
                            root_completed = selective_direct_telemetry.root_completed,
                            root_slice_exhausted = selective_direct_telemetry.root_slice_exhausted,
                            candidate_verified = root_direct_verified,
                            elapsed_s = selective_direct_telemetry.elapsed.as_secs_f64(),
                            "[multi-obj] authenticated selective direct-conic root evaluation"
                        );
                        if root_direct_verified {
                            if crown_deadline.is_some_and(|limit| Instant::now() >= limit) {
                                selective_direct_telemetry.late_discarded += 1;
                                info!(
                                    late_discarded = selective_direct_telemetry.late_discarded,
                                    "[multi-obj] discarded selective direct-conic root verdict after global deadline"
                                );
                            } else {
                                lifecycle.domains_explored = 1;
                                lifecycle.domains_verified = 1;
                                return Ok(lifecycle.build_result(BabVerificationStatus::Verified));
                            }
                        }
                    }
                }
            } else {
                selective_direct_telemetry.budget_exhausted = true;
                selective_direct_telemetry.disabled = true;
            }
        }
        // Use bab_timeout so post-BaB PGD reservation is respected (#4095).
        if lifecycle.start_time.elapsed() > bab_timeout {
            return Ok(lifecycle.timeout_result());
        }

        let root_priority = multi_obj_domain_priority(&root_obj_bounds, thresholds);
        // Seed the root domain with the root-optimized α state so its children can
        // warm-start from it (per-sub-domain refinement). Only populated when the
        // warm path is enabled; otherwise `None` keeps the frozen-default behavior.
        let root_inherited_alpha = if warm_alpha_enabled {
            root_alpha_state.clone().map(Arc::new)
        } else {
            None
        };
        let mut queue: BinaryHeap<MultiObjInputDomain> = BinaryHeap::new();
        queue.push(MultiObjInputDomain {
            input_bounds: Arc::new(input.clone()),
            obj_bounds: root_obj_bounds,
            linear_bounds: root_linear,
            depth: 0,
            priority: root_priority,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: root_inherited_alpha,
        });

        let loop_batch_size = if self.config.reorder_bab {
            let decision = multi_objective_loop_batch_decision(
                self.config.batch_size,
                input.len(),
                affine_conic_closure,
                self.config.input_split_conic_queue_refresh_batch_size,
            )?;
            info!(
                requested_batch_size = decision.requested_batch_size,
                effective_batch_size = decision.effective_batch_size,
                clamp_reason = decision.clamp_reason.as_str(),
                conic_queue_refresh_batch_size = affine_conic_closure
                    .then_some(self.config.input_split_conic_queue_refresh_batch_size),
                input_elems = input.len(),
                "[multi-obj] using reordered BaB (bound -> filter -> split -> clip)"
            );
            decision.effective_batch_size
        } else {
            1
        };
        let mut selective_direct_quota =
            selective_direct::SelectiveDirectQuota::new(if selective_direct_plan.is_some() {
                selective_direct::SELECTIVE_DIRECT_STRIDE
            } else {
                0
            });
        // Truncated CROWN uses the rayon fallback, so keep its optional direct
        // dispatches serial to bound simultaneously live reference-map clones.
        let selective_direct_microbatch_cap = if crown_bkwd.is_some() {
            1
        } else {
            selective_direct::SELECTIVE_DIRECT_MICROBATCH_CAP
        };
        let mut batch_index = 0usize;

        while !queue.is_empty() {
            if lifecycle.start_time.elapsed() > bab_timeout {
                selective_direct_telemetry.log_summary(
                    "timeout",
                    &selective_direct_quota,
                    domains_verified_by_selective_direct,
                    lifecycle.domains_explored,
                );
                return Ok(lifecycle.timeout_result());
            }
            if lifecycle.domains_explored >= self.config.max_domains {
                selective_direct_telemetry.log_summary(
                    "domain_limit",
                    &selective_direct_quota,
                    domains_verified_by_selective_direct,
                    lifecycle.domains_explored,
                );
                return Ok(lifecycle.build_result(BabVerificationStatus::Unknown {
                    reason: format!(
                        "Domain limit {}: {}/{} verified",
                        self.config.max_domains,
                        lifecycle.domains_verified,
                        lifecycle.domains_explored
                    ),
                }));
            }

            let mut domains = pop_multi_obj_input_domain_batch(&mut queue, loop_batch_size);
            bound_deferred_multi_obj_domains_batch(
                &mut domains,
                graph,
                &spec_matrix,
                thresholds,
                engine,
                root_node_bounds.as_ref(),
                root_alpha_state.as_ref(),
                mul_binary_alphas_multi.as_ref(),
                crown_deadline,
                crown_bkwd,
                &self.config,
                self.graph_domain_batch_metrics_sink(),
                batch_index,
            )?;
            batch_index += 1;

            let mut selective_direct_verified = vec![false; domains.len()];
            let active_direct_plan = if !selective_direct_telemetry.disabled
                && crown_deadline.is_none_or(|limit| Instant::now() < limit)
            {
                selective_direct_plan.as_ref()
            } else {
                None
            };
            if let Some(direct_plan) = active_direct_plan {
                let remaining_domains = self
                    .config
                    .max_domains
                    .saturating_sub(lifecycle.domains_explored)
                    .min(domains.len());
                let candidates: Vec<_> = domains
                    .iter()
                    .take(remaining_domains)
                    .enumerate()
                    .filter_map(|(domain_index, domain)| {
                        if multi_obj_domain_verified(&domain.obj_bounds, thresholds) {
                            return None;
                        }
                        let affine = affine_conic_source_thresholds.and_then(|source_thresholds| {
                            domain.linear_bounds.as_ref().and_then(|linear| {
                                affine_conic::evaluate_affine_conic_closure(
                                    linear,
                                    domain.input_bounds.as_ref(),
                                    source_thresholds,
                                )
                            })
                        });
                        if affine.is_some_and(affine_conic::ConicEvaluation::verifies) {
                            return None;
                        }
                        Some(selective_direct::SelectiveDirectCandidate::new(
                            domain_index,
                            affine.map(|evaluation| {
                                (
                                    evaluation.gap(),
                                    evaluation.lhs_weight,
                                    evaluation.rhs_weight,
                                )
                            }),
                            domain.priority,
                        ))
                    })
                    .collect();
                let selected = selective_direct_quota.select(&candidates);
                let mut verified_this_refresh = 0usize;
                for chunk in selective_direct::selective_direct_chunks(
                    &selected,
                    selective_direct_microbatch_cap,
                ) {
                    if crown_deadline.is_some_and(|limit| Instant::now() >= limit) {
                        break;
                    }
                    let direct_started = Instant::now();
                    let Some(direct_deadline) = selective_direct::call_deadline(
                        direct_started,
                        crown_deadline,
                        selective_direct_budget,
                        selective_direct_telemetry.elapsed,
                        Duration::MAX,
                    ) else {
                        selective_direct_telemetry.budget_exhausted = true;
                        selective_direct_telemetry.disabled = true;
                        info!(
                            batch_index,
                            elapsed_s = selective_direct_telemetry.elapsed.as_secs_f64(),
                            "[multi-obj] exhausted selective direct-conic wall-time pool"
                        );
                        break;
                    };
                    let inputs: Vec<_> = chunk
                        .iter()
                        .map(|&domain_index| domains[domain_index].input_bounds.as_ref())
                        .collect();
                    selective_direct_telemetry.attempted_rows += inputs.len();
                    selective_direct_telemetry.microbatches += 1;
                    let direct_result = compute_selective_direct_bounds(
                        &inputs,
                        direct_plan,
                        Some(direct_deadline),
                    );
                    let direct_finished = Instant::now();
                    selective_direct_telemetry.elapsed +=
                        direct_finished.duration_since(direct_started);
                    let global_expired =
                        crown_deadline.is_some_and(|limit| direct_finished >= limit);
                    let phase_expired = direct_finished >= direct_deadline && !global_expired;
                    if phase_expired
                        || selective_direct_telemetry.elapsed >= selective_direct_budget
                    {
                        selective_direct_telemetry.budget_exhausted = true;
                        selective_direct_telemetry.disabled = true;
                    }
                    let direct_result = phase_yield_from_result(
                        direct_result,
                        direct_finished,
                        Some(direct_deadline),
                        crown_deadline,
                        NyError::is_deadline_exceeded,
                    );
                    let usable_result = match direct_result {
                        Ok(PhaseYield::Complete(_) | PhaseYield::Partial(_)) if global_expired => {
                            selective_direct_telemetry.late_discarded += inputs.len();
                            info!(
                                batch_index,
                                discarded_rows = inputs.len(),
                                late_discarded = selective_direct_telemetry.late_discarded,
                                elapsed_s = selective_direct_telemetry.elapsed.as_secs_f64(),
                                "[multi-obj] discarded selective direct-conic microbatch after global deadline"
                            );
                            None
                        }
                        Ok(PhaseYield::Complete(result) | PhaseYield::Partial(result)) => {
                            Some(result)
                        }
                        Ok(PhaseYield::Declined(reason)) => {
                            selective_direct_telemetry.phase_declines += 1;
                            selective_direct_telemetry.disabled = true;
                            info!(
                                batch_index,
                                ?reason,
                                budget_exhausted = selective_direct_telemetry.budget_exhausted,
                                elapsed_s = selective_direct_telemetry.elapsed.as_secs_f64(),
                                "[multi-obj] disabling selective direct-conic after a phase decline"
                            );
                            None
                        }
                        Err(err) if err.is_deadline_exceeded() => {
                            info!(
                                batch_index,
                                error = %err,
                                elapsed_s = selective_direct_telemetry.elapsed.as_secs_f64(),
                                "[multi-obj] selective direct-conic microbatch reached the global deadline"
                            );
                            None
                        }
                        Err(err) => {
                            selective_direct_telemetry.errors += 1;
                            selective_direct_telemetry.disabled = true;
                            info!(
                                error = %err,
                                errors = selective_direct_telemetry.errors,
                                "[multi-obj] disabling optional selective direct-conic pass after rebound error"
                            );
                            None
                        }
                    };
                    let Some(result) = usable_result else {
                        break;
                    };
                    let verdicts: Result<Vec<bool>> = result
                        .bounds
                        .iter()
                        .map(|bounds| {
                            Ok(multi_obj_domain_verified(
                                &extract_obj_bounds(bounds, 1)?,
                                &direct_plan.thresholds,
                            ))
                        })
                        .collect();
                    let verdicts = match verdicts {
                        Ok(verdicts) if verdicts.len() == chunk.len() => verdicts,
                        Ok(verdicts) => {
                            selective_direct_telemetry.errors += 1;
                            selective_direct_telemetry.disabled = true;
                            info!(
                                batch_index,
                                expected_bounds = chunk.len(),
                                actual_bounds = verdicts.len(),
                                errors = selective_direct_telemetry.errors,
                                "[multi-obj] disabling optional selective direct-conic pass after result cardinality mismatch"
                            );
                            break;
                        }
                        Err(err) => {
                            selective_direct_telemetry.errors += 1;
                            selective_direct_telemetry.disabled = true;
                            info!(
                                error = %err,
                                errors = selective_direct_telemetry.errors,
                                "[multi-obj] disabling optional selective direct-conic pass after malformed result"
                            );
                            break;
                        }
                    };
                    selective_direct_telemetry.completed_rows += verdicts.len();
                    for (&domain_index, verified) in chunk.iter().zip(verdicts) {
                        if verified {
                            selective_direct_verified[domain_index] = true;
                            verified_this_refresh += 1;
                        }
                    }
                    if selective_direct_telemetry.disabled {
                        break;
                    }
                }
                if !selected.is_empty()
                    && (batch_index <= 4
                        || batch_index.is_power_of_two()
                        || batch_index.is_multiple_of(16))
                {
                    info!(
                        batch_index,
                        candidates = candidates.len(),
                        selected = selected.len(),
                        verified = verified_this_refresh,
                        candidates_seen = selective_direct_quota.candidates_seen(),
                        scheduled_nonroot_rows = selective_direct_quota.selected(),
                        attempted_nonroot_rows = selective_direct_telemetry.attempted_rows,
                        completed_nonroot_rows = selective_direct_telemetry.completed_rows,
                        microbatches = selective_direct_telemetry.microbatches,
                        microbatch_cap = selective_direct_microbatch_cap,
                        budget_exhausted = selective_direct_telemetry.budget_exhausted,
                        late_discarded = selective_direct_telemetry.late_discarded,
                        phase_declines = selective_direct_telemetry.phase_declines,
                        errors = selective_direct_telemetry.errors,
                        elapsed_s = selective_direct_telemetry.elapsed.as_secs_f64(),
                        "[multi-obj] authenticated selective direct-conic refresh"
                    );
                }
            }

            for (domain_index, domain) in domains.into_iter().enumerate() {
                if lifecycle.start_time.elapsed() > bab_timeout {
                    selective_direct_telemetry.log_summary(
                        "timeout",
                        &selective_direct_quota,
                        domains_verified_by_selective_direct,
                        lifecycle.domains_explored,
                    );
                    return Ok(lifecycle.timeout_result());
                }
                if lifecycle.domains_explored >= self.config.max_domains {
                    selective_direct_telemetry.log_summary(
                        "domain_limit",
                        &selective_direct_quota,
                        domains_verified_by_selective_direct,
                        lifecycle.domains_explored,
                    );
                    return Ok(lifecycle.build_result(BabVerificationStatus::Unknown {
                        reason: format!(
                            "Domain limit {}: {}/{} verified",
                            self.config.max_domains,
                            lifecycle.domains_verified,
                            lifecycle.domains_explored
                        ),
                    }));
                }

                lifecycle.domains_explored += 1;
                lifecycle.max_depth_reached = lifecycle.max_depth_reached.max(domain.depth);

                if selective_direct_verified[domain_index] {
                    domains_verified_by_selective_direct += 1;
                    lifecycle.domains_verified += 1;
                    if domains_verified_by_selective_direct <= 4
                        || domains_verified_by_selective_direct.is_power_of_two()
                    {
                        info!(
                            closures = domains_verified_by_selective_direct,
                            explored = lifecycle.domains_explored,
                            depth = domain.depth,
                            attempted_nonroot_rows = selective_direct_telemetry.attempted_rows,
                            "[multi-obj] authenticated selective direct-conic closure"
                        );
                    }
                    continue;
                }

                if lifecycle.domains_explored.is_multiple_of(1000)
                    || lifecycle.domains_explored <= 5
                {
                    trace!(
                        "[multi-obj] explored={} verified={} clipped={} depth={} queue={} pri={:.4}",
                        lifecycle.domains_explored,
                        lifecycle.domains_verified,
                        domains_verified_by_clip,
                        domain.depth,
                        queue.len(),
                        domain.priority,
                    );
                }

                if multi_obj_domain_verified(&domain.obj_bounds, thresholds) {
                    lifecycle.domains_verified += 1;
                    continue;
                }
                if let Some(source_thresholds) = affine_conic_source_thresholds {
                    if let Some(evaluation) = domain.linear_bounds.as_ref().and_then(|linear| {
                        affine_conic::evaluate_affine_conic_closure(
                            linear,
                            domain.input_bounds.as_ref(),
                            source_thresholds,
                        )
                    }) {
                        if lifecycle.domains_explored <= 5
                            || lifecycle.domains_explored.is_multiple_of(1000)
                        {
                            info!(
                                explored = lifecycle.domains_explored,
                                closures = domains_verified_by_affine_conic,
                                depth = domain.depth,
                                lower_bound = evaluation.lower_bound,
                                threshold_upper = evaluation.threshold_upper,
                                gap = evaluation.gap(),
                                lhs_weight = evaluation.lhs_weight,
                                rhs_weight = evaluation.rhs_weight,
                                verified = evaluation.verifies(),
                                "[multi-obj] authenticated affine conic evaluation"
                            );
                        }
                        if evaluation.verifies() {
                            domains_verified_by_affine_conic += 1;
                            lifecycle.domains_verified += 1;
                            if domains_verified_by_affine_conic <= 4
                                || domains_verified_by_affine_conic.is_power_of_two()
                            {
                                info!(
                                    closures = domains_verified_by_affine_conic,
                                    explored = lifecycle.domains_explored,
                                    depth = domain.depth,
                                    gap = evaluation.gap(),
                                    lhs_weight = evaluation.lhs_weight,
                                    rhs_weight = evaluation.rhs_weight,
                                    "[multi-obj] authenticated affine conic closure"
                                );
                            }
                            continue;
                        }
                    }
                }
                if domain.depth >= self.config.max_depth {
                    lifecycle.unresolved_due_to_depth = true;
                    continue;
                }

                // Select split dimension using CROWN linear coefficients when available.
                let domain_bounds: Vec<f32> = domain
                    .obj_bounds
                    .iter()
                    .map(|(lower, upper)| {
                        if self.config.verify_upper_bound {
                            *upper
                        } else {
                            *lower
                        }
                    })
                    .collect();
                // Multi-dimensional input split: select the top `input_split_depth`
                // dims by SB score and midpoint-split each, producing up to
                // 2^depth children that EXACTLY COVER the parent (completeness
                // preserved). At depth 1 this is exactly the original left/right
                // pair. Mirrors the single-objective loop (process_batch.rs) so
                // the preset's `input_split.depth` knob now applies to the joint
                // conjunctive lane too (acasxu prop_2/3/4 route here).
                let split_dims = self.select_input_dimensions_sb(
                    domain.input_bounds.as_ref(),
                    domain.linear_bounds.as_ref(),
                    Some(domain_bounds.as_slice()),
                    Some(thresholds),
                );
                let shape = domain.input_bounds.as_ref().lower().shape().to_vec();
                let (flat_lower, flat_upper) = {
                    let flat = domain.input_bounds.as_ref().flatten();
                    (flat.lower().clone(), flat.upper().clone())
                };
                let child_boxes = multi_dim_split_boxes(flat_lower, flat_upper, &split_dims);

                if child_boxes.len() <= 1 {
                    lifecycle.unresolved_due_to_unsplittable = true;
                    continue;
                }

                for (child_lower, child_upper) in child_boxes {
                    let child_input = build_child_input_owned(child_lower, child_upper, &shape)?;
                    screen_multi_obj_child(
                        self,
                        graph,
                        child_input,
                        &spec_matrix,
                        thresholds,
                        engine,
                        &compute_bounds,
                        warm_compute_bounds_opt,
                        &domain,
                        &mut queue,
                        &mut lifecycle,
                        &mut domains_verified_by_clip,
                    )?;
                }
            }
        }

        if domains_verified_by_clip > 0 {
            info!(
                "[multi-obj] domains_verified_by_clip={} out of {} verified ({} explored)",
                domains_verified_by_clip, lifecycle.domains_verified, lifecycle.domains_explored
            );
        }
        if domains_verified_by_affine_conic > 0 {
            info!(
                closures = domains_verified_by_affine_conic,
                explored = lifecycle.domains_explored,
                "[multi-obj] authenticated affine conic closure summary"
            );
        }
        selective_direct_telemetry.log_summary(
            "queue_exhausted",
            &selective_direct_quota,
            domains_verified_by_selective_direct,
            lifecycle.domains_explored,
        );

        Ok(lifecycle.build_final_result())
    }
}
