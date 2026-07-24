// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-objective conjunctive input-split BaB verifier.

mod screen_child;

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{info, trace};

use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::bounds::GraphAlphaState;
use crate::GraphNetwork;

use self::screen_child::screen_multi_obj_child;
use super::super::shared::state::GraphBabLifecycle;
use super::batching::{
    bound_deferred_multi_obj_domains_batch, input_split_loop_batch_size,
    pop_multi_obj_input_domain_batch,
};
use super::build_batches::compute_crown_or_ibp_bounds_in_build_batches;
use super::mul_binary::maybe_optimize_mul_binary_alphas;
use super::root_bounds::collect_input_split_root_node_bounds;
use super::shared::{
    build_child_input_owned, compute_crown_or_ibp_bounds_with_node_bounds, extract_obj_bounds,
    multi_dim_split_boxes, multi_obj_domain_priority, multi_obj_domain_verified,
    MultiObjInputDomain,
};
use crate::beta_crown::engine::BetaCrownVerifier;

impl BetaCrownVerifier {
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
        self.config.validate()?;
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
        let effective_total = match deadline {
            Some(dl) => dl.saturating_duration_since(now),
            None => self.config.timeout,
        };
        let bab_timeout = effective_total.mul_f32(1.0 - pgd_frac);
        let initial_deadline = {
            let frac = self
                .config
                .phase_budget
                .initial_bounds_fraction
                .clamp(0.0, 1.0);
            Some(now + bab_timeout.mul_f32(frac))
        };
        // Per-domain deadline for CROWN backward passes in the BaB loop.
        let crown_deadline = Some(now + bab_timeout);
        let mut domains_verified_by_clip = 0usize;

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

        if self.config.reorder_bab {
            let loop_batch = input_split_loop_batch_size(self.config.batch_size, input.len())?;
            info!(
                requested_batch_size = loop_batch.requested_batch_size,
                effective_batch_size = loop_batch.effective_batch_size,
                clamp_reason = loop_batch.clamp_reason.as_str(),
                input_elems = input.len(),
                "[multi-obj] using reordered BaB (bound -> filter -> split -> clip)"
            );
        }

        let loop_batch_size = if self.config.reorder_bab {
            input_split_loop_batch_size(self.config.batch_size, input.len())?.effective_batch_size
        } else {
            1
        };
        let mut batch_index = 0usize;

        while !queue.is_empty() {
            if lifecycle.start_time.elapsed() > bab_timeout {
                return Ok(lifecycle.timeout_result());
            }
            if lifecycle.domains_explored >= self.config.max_domains {
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

            for domain in domains {
                if lifecycle.start_time.elapsed() > bab_timeout {
                    return Ok(lifecycle.timeout_result());
                }
                if lifecycle.domains_explored >= self.config.max_domains {
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

        Ok(lifecycle.build_final_result())
    }
}
