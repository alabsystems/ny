// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Single-objective graph input-split BaB verifier.

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::info;

use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::bounds::LinearBounds;
use crate::GraphNetwork;

use super::super::shared::state::GraphBabLifecycle;
use super::batching::{
    bound_deferred_domains_batch_with_metrics, input_split_loop_batch_size, pop_input_domain_batch,
    root_map_spec_obj_bounds, should_run_adv_check_on_batch, try_adv_check_on_batch,
};
use super::mul_binary::maybe_optimize_mul_binary_alphas;
use super::root_bounds::collect_input_split_root_node_bounds;
use super::shared::{
    compute_crown_or_ibp_bounds, compute_crown_or_ibp_bounds_with_node_bounds,
    graph_output_bounds_are_finite, graph_spec_ibp_root_screen_with_deadline, scalar_output_bounds,
    GraphInputDomain,
};
use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::beta_crown::engine::BetaCrownVerifier;

mod process_batch;
mod screen_child;

use self::process_batch::process_single_objective_domain_batch;

#[cfg(test)]
thread_local! {
    static ROOT_SPEC_CROWN_ENTRY_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn reset_root_spec_crown_entry_count() {
    ROOT_SPEC_CROWN_ENTRY_COUNT.set(0);
}

#[cfg(test)]
pub(super) fn root_spec_crown_entry_count() -> usize {
    ROOT_SPEC_CROWN_ENTRY_COUNT.get()
}

impl BetaCrownVerifier {
    pub fn verify_graph_input_split(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
    ) -> Result<BetaCrownResult> {
        self.verify_graph_input_split_impl(graph, input, objective, threshold, self.engine(), None)
    }

    /// Verify GraphNetwork with input splitting and optional GPU acceleration.
    ///
    /// `deadline`: If `Some`, the BaB engine derives its phase budgets from
    /// remaining wall-clock time instead of `self.config.timeout` (#4321).
    pub fn verify_graph_input_split_with_engine(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        let engine = self.resolve_engine(engine);
        self.verify_graph_input_split_impl(graph, input, objective, threshold, engine, deadline)
    }

    /// Internal implementation with optional GemmEngine for GPU acceleration.
    pub(crate) fn verify_graph_input_split_impl(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        objective: &[f32],
        threshold: f32,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<BetaCrownResult> {
        self.config.validate()?;
        let graph = self.configured_graph_for_crown(graph);
        let graph = &graph;
        if self.config.use_crown_ibp {
            return Err(NyError::UnsupportedConfiguration(
                "GraphNetwork β-CROWN input splitting does not support --crown-ibp".to_string(),
            ));
        }
        if self.config.enable_cuts {
            return Err(NyError::UnsupportedConfiguration(
                "GraphNetwork β-CROWN input splitting does not support cutting planes".to_string(),
            ));
        }

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
            Some(GraphBabLifecycle::fail_closed_deadline(
                now,
                bab_timeout.mul_f32(frac),
            ))
        };
        // Per-domain deadline for CROWN backward passes in the BaB loop.
        let crown_deadline = Some(GraphBabLifecycle::fail_closed_deadline(now, bab_timeout));
        let mut domains_verified_by_ibp = 0usize;
        let mut domains_screened_by_crown = 0usize;
        let warmup_start = Instant::now();

        let spec_matrix = Array2::from_shape_vec((1, objective.len()), objective.to_vec())
            .map_err(|e| NyError::InvalidSpec(format!("spec matrix: {}", e)))?;

        let (root_node_bounds, root_alpha_state) = collect_input_split_root_node_bounds(
            graph,
            input,
            &self.config,
            engine,
            initial_deadline,
            "input splitting",
            None,
        )?;
        if root_node_bounds.is_some() {
            info!(
                "Graph input split warmup: root intermediate bounds finished in {:.3}s",
                warmup_start.elapsed().as_secs_f64()
            );
        }

        // The root collector has already paid for a certified output box. A
        // direct outward-rounded projection can be decisive even when another
        // spec-CROWN backward would be looser. Missing/malformed/non-decisive
        // maps simply retain the historical warmup below.
        if let Some((root_lower, root_upper)) = root_node_bounds
            .as_ref()
            .and_then(|root_map| root_map_spec_obj_bounds(graph, root_map, &spec_matrix))
            .and_then(|bounds| bounds.into_iter().next())
        {
            if self
                .config
                .domain_is_verified(root_lower, root_upper, threshold)
            {
                info!(
                    "Graph input split: certified root-map output box verifies the root; skipping fresh spec-CROWN and child bounding"
                );
                lifecycle.domains_explored = 1;
                lifecycle.domains_verified = 1;
                return Ok(lifecycle.build_result_with_bounds(
                    BabVerificationStatus::Verified,
                    scalar_output_bounds(root_lower, root_upper)?,
                ));
            }
            if self
                .config
                .domain_is_violation(root_lower, root_upper, threshold)
            {
                info!(
                    "Graph input split: certified root-map output box proves a root violation; skipping fresh spec-CROWN and child bounding"
                );
                lifecycle.domains_explored = 1;
                return Ok(lifecycle.build_result_with_bounds(
                    BabVerificationStatus::potential_violation(),
                    scalar_output_bounds(root_lower, root_upper)?,
                ));
            }
        }

        // Phase 4 (#3439): MulBinary SPSA alpha optimization.
        // Uses initial_deadline so warmup phases respect the cap (#4095).
        let mul_binary_alphas = maybe_optimize_mul_binary_alphas(
            graph,
            input,
            &spec_matrix,
            engine,
            initial_deadline,
            self.config.crown_backward_layers,
            "Graph input split",
        )?;
        info!(
            "Graph input split warmup: MulBinary stage finished in {:.3}s total",
            warmup_start.elapsed().as_secs_f64()
        );

        // Per-domain bound computation via shared CROWN-or-IBP helper. Part of #3453.
        let crown_bkwd = self.config.crown_backward_layers;
        let compute_bounds = |input_bounds: &BoundedTensor,
                              node_bounds: Option<&HashMap<String, BoundedTensor>>|
         -> Result<(f32, f32, Option<LinearBounds>)> {
            let (bounds, linear) = compute_crown_or_ibp_bounds_with_node_bounds(
                graph,
                input_bounds,
                &spec_matrix,
                engine,
                root_node_bounds.as_ref(),
                node_bounds,
                root_alpha_state.as_ref(),
                mul_binary_alphas.as_ref(),
                crown_deadline,
                crown_bkwd,
                self.config.input_split_ibp_enhancement,
            )?;
            Ok((bounds.lower_scalar(), bounds.upper_scalar(), linear))
        };

        // Per-sub-domain α refinement closure (alpha-beta-CROWN
        // input_split/bounding.py:90-179). Built only when the knob is enabled AND
        // α-CROWN is in use AND a root α state exists to seed children from. When
        // `None`, the per-child path uses the frozen `compute_bounds` above — i.e.
        // ny's historical single frozen-alpha pass (the no-regression default).
        let warm_alpha_enabled = self.config.input_split_alpha_iteration > 0
            && self.config.use_alpha_crown
            && root_alpha_state.is_some();
        let warm_compute_bounds = |input_bounds: &BoundedTensor,
                                   node_bounds: Option<&HashMap<String, BoundedTensor>>,
                                   parent_alpha: &crate::bounds::GraphAlphaState|
         -> Result<(
            f32,
            f32,
            Option<LinearBounds>,
            crate::bounds::GraphAlphaState,
        )> {
            let (bounds, linear, refined_alpha) =
                super::shared::compute_warm_start_crown_bounds_with_refined_alpha(
                    graph,
                    input_bounds,
                    &spec_matrix,
                    engine,
                    node_bounds,
                    parent_alpha,
                    mul_binary_alphas.as_ref(),
                    crown_deadline,
                    crown_bkwd,
                    &self.config,
                )?;
            Ok((
                bounds.lower_scalar(),
                bounds.upper_scalar(),
                linear,
                refined_alpha,
            ))
        };
        let warm_compute_bounds_opt: Option<&screen_child::WarmComputeBoundsFn<'_>> =
            if warm_alpha_enabled {
                Some(&warm_compute_bounds)
            } else {
                None
            };

        // IBP enhancement: screen root domain before CROWN. Part of #3813.
        if self.config.input_split_ibp_enhancement {
            match graph_spec_ibp_root_screen_with_deadline(
                graph,
                input,
                &spec_matrix,
                engine,
                initial_deadline,
            ) {
                Ok((ibp_bounds, _)) => {
                    if !graph_output_bounds_are_finite(&ibp_bounds) {
                        info!(
                            "Graph input split warmup: skipping IBP root screen after {:.3}s \
                             because enhancement-only IBP produced non-finite bounds",
                            warmup_start.elapsed().as_secs_f64()
                        );
                    } else {
                        let ibp_lower = ibp_bounds.lower_scalar();
                        let ibp_upper = ibp_bounds.upper_scalar();
                        info!(
                            "Graph input split warmup: IBP root screen finished in {:.3}s total",
                            warmup_start.elapsed().as_secs_f64()
                        );
                        if self
                            .config
                            .domain_is_verified(ibp_lower, ibp_upper, threshold)
                        {
                            info!(
                                "Graph input split: root domain verified by IBP alone \
                                 (lower={}, upper={}, threshold={})",
                                ibp_lower, ibp_upper, threshold
                            );
                            lifecycle.domains_explored = 1;
                            lifecycle.domains_verified = 1;
                            return Ok(lifecycle.build_result_with_bounds(
                                BabVerificationStatus::Verified,
                                scalar_output_bounds(ibp_lower, ibp_upper)?,
                            ));
                        }
                    }
                }
                Err(NyError::DeadlineExceeded(_)) => {
                    info!(
                        "Graph input split warmup: skipping IBP root screen after {:.3}s \
                         because the warmup deadline expired",
                        warmup_start.elapsed().as_secs_f64()
                    );
                }
                Err(err) => {
                    info!(
                        "Graph input split warmup: skipping IBP root screen after {:.3}s \
                         due to enhancement-only error: {}",
                        warmup_start.elapsed().as_secs_f64(),
                        err
                    );
                }
            }
        }

        // Root domain bounds via shared CROWN-or-IBP dispatch (#3453).
        #[cfg(test)]
        ROOT_SPEC_CROWN_ENTRY_COUNT.set(ROOT_SPEC_CROWN_ENTRY_COUNT.get() + 1);
        let (root_lower, root_upper, root_linear) = {
            let (bounds, linear) = compute_crown_or_ibp_bounds(
                graph,
                input,
                &spec_matrix,
                engine,
                root_node_bounds.as_ref(),
                root_alpha_state.as_ref(),
                mul_binary_alphas.as_ref(),
                initial_deadline,
                crown_bkwd,
                self.config.input_split_ibp_enhancement,
            )?;
            if self.config.input_split_ibp_enhancement && !graph_output_bounds_are_finite(&bounds) {
                info!(
                    "Graph input split warmup: enhanced root bounds produced non-finite output; retrying plain input-split bounds"
                );
                let (plain_bounds, plain_linear) = compute_crown_or_ibp_bounds(
                    graph,
                    input,
                    &spec_matrix,
                    engine,
                    root_node_bounds.as_ref(),
                    root_alpha_state.as_ref(),
                    mul_binary_alphas.as_ref(),
                    initial_deadline,
                    crown_bkwd,
                    false,
                )?;
                (
                    plain_bounds.lower_scalar(),
                    plain_bounds.upper_scalar(),
                    plain_linear,
                )
            } else {
                (bounds.lower_scalar(), bounds.upper_scalar(), linear)
            }
        };
        info!(
            "Graph input split warmup: root CROWN finished in {:.3}s total",
            warmup_start.elapsed().as_secs_f64()
        );

        info!(
            "Graph β-CROWN (input split) initial bounds: [{}, {}], threshold: {}, alpha={}, forward_bounds={}",
            root_lower,
            root_upper,
            threshold,
            self.config.use_alpha_crown,
            self.config.use_forward_bounds
        );

        // Use bab_timeout so post-BaB PGD reservation is respected (#4095).
        if lifecycle.start_time.elapsed() > bab_timeout {
            return Ok(lifecycle.timeout_result());
        }
        if self
            .config
            .domain_is_verified(root_lower, root_upper, threshold)
        {
            lifecycle.domains_explored = 1;
            lifecycle.domains_verified = 1;
            return Ok(lifecycle.build_result_with_bounds(
                BabVerificationStatus::Verified,
                scalar_output_bounds(root_lower, root_upper)?,
            ));
        }
        if self
            .config
            .domain_is_violation(root_lower, root_upper, threshold)
        {
            lifecycle.domains_explored = 1;
            return Ok(lifecycle.build_result_with_bounds(
                BabVerificationStatus::potential_violation(),
                scalar_output_bounds(root_lower, root_upper)?,
            ));
        }

        // Non-finite root bounds cannot enter the BaB queue (a ±inf/NaN
        // priority would create zombie domains, #2982/#3125). Degrade to the
        // honest Unknown — same unresolved flag the loop sets when a popped
        // domain cannot be split — instead of hard-erroring the whole run
        // (mirrors the loop's unsplittable handling and the GPU path's
        // graceful degradation, #1860).
        let priority = match self.config.domain_priority(root_lower, root_upper) {
            Ok(priority) => priority,
            Err(NyError::NumericalInstability(_)) => {
                lifecycle.domains_explored = 1;
                lifecycle.unresolved_due_to_unsplittable = true;
                return Ok(lifecycle.build_final_result());
            }
            Err(err) => return Err(err),
        };
        // Seed the root domain with the root-optimized α state so its children can
        // warm-start from it (per-sub-domain refinement). Only populated when the
        // warm path is enabled; otherwise `None` keeps the frozen-default behavior.
        let root_inherited_alpha = if warm_alpha_enabled {
            root_alpha_state.clone().map(Arc::new)
        } else {
            None
        };
        let mut queue: BinaryHeap<GraphInputDomain> = BinaryHeap::new();
        queue.push(GraphInputDomain {
            input_bounds: Arc::new(input.clone()),
            lower_bound: root_lower,
            upper_bound: root_upper,
            depth: 0,
            priority,
            linear_bounds: root_linear,
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
                "Graph input split: using reordered BaB (bound → filter → split → clip)"
            );
        }

        let loop_batch_size = if self.config.reorder_bab {
            input_split_loop_batch_size(self.config.batch_size, input.len())?.effective_batch_size
        } else {
            1
        };
        let mut batch_index = 0usize;

        while !queue.is_empty() {
            if let Some(termination) =
                lifecycle.check_termination(bab_timeout, self.config.max_domains)
            {
                return Ok(termination);
            }

            let mut domains = pop_input_domain_batch(&mut queue, loop_batch_size);
            bound_deferred_domains_batch_with_metrics(
                &mut domains,
                graph,
                &spec_matrix,
                engine,
                root_node_bounds.as_ref(),
                root_alpha_state.as_ref(),
                mul_binary_alphas.as_ref(),
                crown_deadline,
                crown_bkwd,
                &self.config,
                self.graph_domain_batch_metrics_sink(),
                batch_index,
            )?;
            batch_index += 1;

            // adv_check: PGD probe for early SAT detection. Ref:
            // batch_branch_and_bound.py:81-90 and attack_in_input_split.py
            //
            // #advcheck-witness: the probe ran a TRUE concrete forward on a
            // point of the current SUB-box and found a genuine violation.
            // Carry that point through the PotentialViolation so the post-BaB
            // confirmer verifies IT, rather than re-searching the whole ROOT
            // box for a point we already held (which routinely failed and
            // downgraded a validated candidate to Unknown). The witness is a
            // candidate only -- the confirmer re-evaluates it and the trusted
            // ONNX-Runtime gate still decides every scored `sat`.
            if should_run_adv_check_on_batch(lifecycle.domains_explored, self.config.adv_check) {
                if let Some(witness) = try_adv_check_on_batch(
                    graph,
                    &domains,
                    objective,
                    threshold,
                    self.config.verify_upper_bound,
                    crown_deadline,
                    lifecycle.domains_explored as u64,
                    engine,
                )? {
                    info!(
                        "adv_check: PGD found counterexample from picked batch at domain {} \
                         (carrying the concrete witness to the confirmer)",
                        lifecycle.domains_explored,
                    );
                    return Ok(lifecycle
                        .build_result(BabVerificationStatus::potential_violation_with(witness)));
                }
            }

            if let Some(result) = process_single_objective_domain_batch(
                self,
                graph,
                domains,
                objective,
                threshold,
                &spec_matrix,
                engine,
                &compute_bounds,
                warm_compute_bounds_opt,
                bab_timeout,
                &mut queue,
                &mut lifecycle,
                &mut domains_verified_by_ibp,
                &mut domains_screened_by_crown,
            )? {
                return Ok(result);
            }
        }

        if self.config.input_split_ibp_enhancement {
            let total = domains_verified_by_ibp + domains_screened_by_crown;
            let rate = if total > 0 {
                100.0 * domains_verified_by_ibp as f64 / total as f64
            } else {
                0.0
            };
            info!(
                "IBP pre-screen rate: {}/{} = {:.1}% (CROWN: {})",
                domains_verified_by_ibp, total, rate, domains_screened_by_crown
            );
        }

        Ok(lifecycle.build_final_result())
    }
}
