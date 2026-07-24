// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Warm-start alpha-CROWN bound collection for per-domain BaB.
//!
//! Split from `alpha.rs` for file size compliance.

use super::*;
use crate::network::alpha_crown_loop::finite_lower_sum;

impl GraphNetwork {
    /// Collect α-CROWN bounds with warm-start from a parent domain's alpha state.
    ///
    /// For per-domain alpha-CROWN in Branch-and-Bound: copies optimized slopes
    /// from the root domain's alpha state, then runs `config.iterations` rounds
    /// of SPSA gradient ascent to refine the alphas for the sub-domain. This
    /// produces tighter intermediate bounds than inheriting root alphas without
    /// optimization, at the cost of ~5-50ms per domain (vs ~0.1ms for plain CROWN).
    ///
    /// The warm-start dramatically reduces iterations needed: 10 warm-started
    /// iterations achieve similar tightness to 20+ from-scratch iterations because
    /// the root's optimized slopes are already good approximations for sub-domains.
    ///
    /// Reference: Xu et al., "Fast and Complete: Enabling Complete Neural Network
    /// Verification with Rapid and Massively Parallel Bound Propagation" (ICLR 2021),
    /// Section 3.2.
    /// Part of #3453, #3439.
    pub fn collect_alpha_crown_bounds_dag_warm(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        warm_start: &GraphAlphaState,
    ) -> Result<(
        std::collections::HashMap<String, BoundedTensor>,
        GraphAlphaState,
    )> {
        self.collect_alpha_crown_bounds_dag_warm_with_engine(input, config, warm_start, None)
    }

    /// Collect α-CROWN bounds with warm-start and optional GEMM acceleration.
    pub fn collect_alpha_crown_bounds_dag_warm_with_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        warm_start: &GraphAlphaState,
        engine: Option<&dyn ny_core::GemmEngine>,
    ) -> Result<(
        std::collections::HashMap<String, BoundedTensor>,
        GraphAlphaState,
    )> {
        let exec_order = self.exec_order()?;

        // Step 1: Compute the configured reference bounds for the sub-domain.
        // In particular, `fix_interm_bounds=true` must honor the same certified
        // forward-linear/IBP policy as the non-warm DAG alpha path instead of
        // paying for a fresh O(N^2) CROWN-IBP collection on every BaB child.
        let reference_bounds =
            self.collect_alpha_reference_bounds_with_engine(input, config, engine, exec_order)?;

        // Step 2: Initialize alpha state from sub-domain bounds, then warm-start.
        let relu_nodes: Vec<String> = exec_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::ReLU(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        // Create fresh alpha state (sets correct unstable masks for sub-domain).
        let mut alpha_state = GraphAlphaState::new();
        for relu_name in &relu_nodes {
            let pre_act = self.relu_preactivation_bounds(
                relu_name,
                input,
                &reference_bounds,
                "warm-start-init",
            )?;
            alpha_state.add_relu_node(relu_name, pre_act, !config.full_conv_alpha)?;
        }
        let s_shaped_nodes: Vec<String> = exec_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::Sigmoid(_) | Layer::Tanh(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        for node_name in &s_shaped_nodes {
            let node = self.nodes.get(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("S-shaped node {} not found", node_name))
            })?;
            let pre_activation = reference_bounds.get(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Pre-activation bounds for monotone node '{}' not found",
                    node_name
                ))
            })?;
            match &node.layer {
                Layer::Sigmoid(_) => alpha_state.add_sigmoid_node(node_name, pre_activation)?,
                Layer::Tanh(_) => alpha_state.add_tanh_node(node_name, pre_activation)?,
                _ => {}
            }
        }
        let sqrt_nodes: Vec<String> = exec_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::Sqrt(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        for node_name in &sqrt_nodes {
            let node = self.nodes.get(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("Sqrt node {} not found", node_name))
            })?;
            let input_name = node.require_unary_input()?;
            let pre_activation = if input_name == NETWORK_INPUT {
                input
            } else {
                reference_bounds.get(input_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Pre-activation bounds for sqrt node '{}' not found",
                        node_name
                    ))
                })?
            };
            if pre_activation
                .lower()
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
            {
                alpha_state.add_sqrt_node(node_name, pre_activation)?;
            }
        }
        let reciprocal_nodes: Vec<String> = exec_order
            .iter()
            .filter(|name| {
                self.nodes
                    .get(*name)
                    .map(|n| matches!(n.layer, Layer::Reciprocal(_)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        for node_name in &reciprocal_nodes {
            let node = self.nodes.get(node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!("Reciprocal node {} not found", node_name))
            })?;
            let input_name = node.require_unary_input()?;
            let pre_activation = if input_name == NETWORK_INPUT {
                input
            } else {
                reference_bounds.get(input_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Pre-activation bounds for reciprocal node '{}' not found",
                        node_name
                    ))
                })?
            };
            let all_positive = pre_activation
                .lower()
                .iter()
                .all(|v| v.is_finite() && *v > 0.0);
            let all_negative = pre_activation
                .upper()
                .iter()
                .all(|v| v.is_finite() && *v < 0.0);
            if all_positive || all_negative {
                alpha_state.add_reciprocal_node(node_name, pre_activation)?;
            }
        }

        // Warm-start: copy optimized alpha values from parent where neurons
        // remain unstable. Optimizer state (velocity, adam_m, adam_v) is reset
        // since the sub-domain has different geometry.
        for relu_name in &relu_nodes {
            // Lower alphas
            if let Some(warm_alpha) = warm_start.alphas.get(relu_name) {
                if let (Some(new_alpha), Some(mask)) = (
                    alpha_state.alphas.get_mut(relu_name),
                    alpha_state.unstable_mask.get(relu_name),
                ) {
                    let len = new_alpha.len().min(warm_alpha.len());
                    for i in 0..len {
                        if mask[i] && warm_alpha[i].is_finite() {
                            new_alpha[i] = warm_alpha[i].clamp(0.0, 1.0);
                        }
                    }
                }
            }
            // Upper alphas (#3393 dual alpha)
            if let Some(warm_upper) = warm_start.alphas_upper.get(relu_name) {
                if let (Some(new_upper), Some(mask)) = (
                    alpha_state.alphas_upper.get_mut(relu_name),
                    alpha_state.unstable_mask.get(relu_name),
                ) {
                    let len = new_upper.len().min(warm_upper.len());
                    for i in 0..len {
                        if mask[i] && warm_upper[i].is_finite() {
                            new_upper[i] = warm_upper[i].clamp(0.0, 1.0);
                        }
                    }
                }
            }
        }
        for node_name in &s_shaped_nodes {
            if let (Some(child_alpha), Some(parent_alpha)) = (
                alpha_state.monotone_s_shaped_alpha_mut(node_name),
                warm_start.monotone_s_shaped_alpha(node_name),
            ) {
                child_alpha.warm_start_from(parent_alpha);
            }
        }
        for node_name in &sqrt_nodes {
            if let (Some(child_alpha), Some(parent_alpha)) = (
                alpha_state.sqrt_alpha_mut(node_name),
                warm_start.sqrt_alpha(node_name),
            ) {
                child_alpha.warm_start_from(parent_alpha);
            }
        }
        for node_name in &reciprocal_nodes {
            if let (Some(child_alpha), Some(parent_alpha)) = (
                alpha_state.reciprocal_alpha_mut(node_name),
                warm_start.reciprocal_alpha(node_name),
            ) {
                child_alpha.warm_start_from(parent_alpha);
            }
        }

        let num_unstable = alpha_state.num_unstable();
        let num_monotone = alpha_state.monotone_alpha_names().count();
        let num_sqrt = alpha_state.sqrt_alpha_names().count();
        let num_reciprocal = alpha_state.reciprocal_alpha_names().count();
        if num_unstable == 0 && num_monotone == 0 && num_sqrt == 0 && num_reciprocal == 0 {
            debug!(
                "GraphNetwork α-CROWN warm-start: No optimizable activation state, using reference bounds"
            );
            return Ok((reference_bounds, alpha_state));
        }

        // Step 3: Iterative alpha optimization on the sub-domain.
        // Refine warm-started alphas via SPSA gradient ascent targeting the output
        // lower bound. This closes the "warm-start iteration cap gap" (R1-1864):
        // without optimization, inherited root alphas may be suboptimal for the
        // sub-domain's tighter geometry after input splitting.
        //
        // The loop structure mirrors collect_alpha_crown_bounds_dag (alpha.rs)
        // but starts from warm-started slopes instead of the default 0.5.
        // Output node + element-wise best output bounds across iterations (#2251) — parity
        // with the main alpha-CROWN path (alpha.rs:285-307 + merge 497-543). The scalar
        // `best_output_lower` tracked in the loop is only a sum for logging; per-dimension
        // best recovers tightness the scalar comparison discards. Hoisted to function scope
        // so the post-loop merge (Step 5) can intersect them into the output bounds. SOUND:
        // each iteration's output_bounds is a valid CROWN over-approximation, so element-wise
        // max-of-lowers / min-of-uppers (an intersection of sound intervals) still contains
        // the true output — never tighter than the true reachable range.
        let output_node = if self.output_node.is_empty() {
            exec_order.last().cloned().ok_or_else(|| {
                NyError::InvalidSpec(
                    "α-CROWN warm-start: output_node empty and exec_order empty".to_string(),
                )
            })?
        } else {
            self.output_node.clone()
        };
        let mut best_output_lower_arr: Option<ArrayD<f32>> = None;
        let mut best_output_upper_arr: Option<ArrayD<f32>> = None;

        if config.iterations > 0 {
            let mut lr = config.learning_rate;
            let eps = 1e-3_f32;
            let mut best_output_lower = f32::NEG_INFINITY;
            let mut sparse_mask: Option<std::collections::BTreeMap<String, Array1<bool>>> = None;
            let use_sparse = config.sparse_ratio < 1.0 && config.sparse_ratio > 0.0;

            for iter in 0..config.iterations {
                if config.past_deadline() {
                    debug!(
                        "α-CROWN warm-start: deadline at iter {}/{}, returning current best",
                        iter, config.iterations
                    );
                    break;
                }

                // Forward pass: compute output bounds with current alphas.
                let output_bounds = match self.propagate_crown_to_node_with_alpha(
                    input,
                    &output_node,
                    &std::collections::HashMap::new(),
                    &reference_bounds,
                    &alpha_state,
                    engine,
                    config.deadline,
                ) {
                    Ok(bounds) => bounds,
                    // CpuMemoryExceeded: Conv2d backward memory-cap backstop
                    // (#conv-crown-oom); skip optimization (warm-start keeps the
                    // sound pre-warm bounds), same as the other degradation paths.
                    Err(
                        NyError::UnsupportedOp(_)
                        | NyError::UnsupportedConfiguration(_)
                        | NyError::CpuMemoryExceeded { .. }
                        | NyError::DeadlineExceeded(_),
                    ) => {
                        debug!(
                            "α-CROWN warm-start: unsupported op in backward to '{}', skipping optimization",
                            output_node
                        );
                        break;
                    }
                    Err(e) => return Err(e),
                };

                // Intersect the raw backward output bound with the always-available
                // reference output bound before it drives the objective / elementwise-best. SOUND:
                // both enclose the output node's reachable set, so the per-element
                // intersection (union on disjoint) still encloses it — never looser; on
                // NaN/shape-mismatch (None) keep the CROWN bound unchanged (sound;
                // NaN-guarded downstream). Mirrors the post-loop reference-bound intersection.
                let output_bounds = match reference_bounds.get(&output_node) {
                    Some(reference_out) if reference_out.shape() == output_bounds.shape() => {
                        output_bounds
                            .intersection_per_element(reference_out)
                            .map(|(t, _)| t)
                            .unwrap_or(output_bounds)
                    }
                    _ => output_bounds,
                };

                let output_lower = finite_lower_sum(output_bounds.lower());
                if !output_lower.is_finite() || output_bounds.lower().iter().any(|v| v.is_nan()) {
                    // NaN guard: skip update but continue (LR decay still applies).
                    debug!("α-CROWN warm-start iter {}: NaN in output, skipping", iter);
                } else {
                    if output_lower > best_output_lower {
                        best_output_lower = output_lower;
                    }
                    // Element-wise best across iterations (mirrors alpha.rs:290-307): skip the
                    // warmup window via should_save_best to avoid locking in noisy early bounds.
                    let is_last_iter = iter == config.iterations - 1;
                    match (&mut best_output_lower_arr, &mut best_output_upper_arr) {
                        (Some(ref mut best_l), Some(ref mut best_u)) => {
                            if config.should_save_best(iter, is_last_iter) {
                                crate::network::graph_alpha::propagate_helpers::update_elementwise_best_bounds(
                                    best_l,
                                    best_u,
                                    &output_bounds,
                                    iter,
                                )?;
                            }
                        }
                        _ => {
                            best_output_lower_arr = Some(output_bounds.lower().clone());
                            best_output_upper_arr = Some(output_bounds.upper().clone());
                        }
                    }
                }

                // Skip gradient on last iteration.
                if iter == config.iterations - 1 {
                    break;
                }

                // SPSA gradient computation.
                let gradients = self.compute_spsa_gradients_dag_for_output_sparse(
                    input,
                    &reference_bounds,
                    &alpha_state,
                    &output_node,
                    eps,
                    config.spsa_samples,
                    sparse_mask.as_ref(),
                    engine,
                )?;

                // Sparse mask selection after first iteration.
                if iter == 0
                    && use_sparse
                    && !gradients
                        .relu
                        .values()
                        .any(|g| g.iter().any(|v| !v.is_finite()))
                {
                    sparse_mask = Some(Self::select_top_alphas(
                        &gradients.relu,
                        config.sparse_ratio,
                    ));
                }

                // Alpha update via Adam (gradient ascent: negate for maximize).
                let adam_params = config.adam_params(lr, iter + 1);
                for relu_name in &relu_nodes {
                    if let Some(grad) = gradients.relu.get(relu_name) {
                        if grad.iter().any(|v| !v.is_finite()) {
                            continue; // Per-ReLU NaN guard (#2867)
                        }
                        let mask = sparse_mask.as_ref().and_then(|m| m.get(relu_name));
                        let neg_grad = if let Some(mask_arr) = mask {
                            let masked: Vec<f32> = grad
                                .iter()
                                .zip(mask_arr.iter())
                                .map(|(&g, &active)| if active { -g } else { 0.0 })
                                .collect();
                            Array1::from_vec(masked)
                        } else {
                            -grad
                        };
                        // Channel-only alpha reduction (#4404)
                        let neg_grad = alpha_state.reduce_gradient(relu_name, &neg_grad);
                        match config.optimizer {
                            Optimizer::Adam => {
                                alpha_state.update_adam(relu_name, &neg_grad, &adam_params);
                                alpha_state.update_adam_upper(relu_name, &neg_grad, &adam_params);
                            }
                            Optimizer::Sgd => {
                                alpha_state.update(relu_name, &neg_grad, lr, config.momentum);
                                alpha_state.update_upper(relu_name, &neg_grad, lr, config.momentum);
                            }
                        }
                    }
                }
                for node_name in &s_shaped_nodes {
                    if let Some(grad) = gradients.monotone.get(node_name) {
                        if grad.any_non_finite() {
                            continue;
                        }
                        let neg_grad = grad.negate();
                        if let Some(alpha) = alpha_state.monotone_s_shaped_alpha_mut(node_name) {
                            match config.optimizer {
                                Optimizer::Adam => alpha.update_adam(&neg_grad, &adam_params),
                                Optimizer::Sgd => alpha.update_sgd(&neg_grad, lr, config.momentum),
                            }
                        }
                    }
                }
                for node_name in &sqrt_nodes {
                    if let Some(grad) = gradients.sqrt.get(node_name) {
                        if grad.any_non_finite() {
                            continue;
                        }
                        let neg_grad = grad.negate();
                        if let Some(alpha) = alpha_state.sqrt_alpha_mut(node_name) {
                            match config.optimizer {
                                Optimizer::Adam => alpha.update_adam(&neg_grad, &adam_params),
                                Optimizer::Sgd => alpha.update_sgd(&neg_grad, lr, config.momentum),
                            }
                        }
                    }
                }
                for node_name in &reciprocal_nodes {
                    if let Some(grad) = gradients.reciprocal.get(node_name) {
                        if grad.any_non_finite() {
                            continue;
                        }
                        let neg_grad = grad.negate();
                        if let Some(alpha) = alpha_state.reciprocal_alpha_mut(node_name) {
                            match config.optimizer {
                                Optimizer::Adam => alpha.update_adam(&neg_grad, &adam_params),
                                Optimizer::Sgd => alpha.update_sgd(&neg_grad, lr, config.momentum),
                            }
                        }
                    }
                }

                lr *= config.lr_decay;
            }

            debug!(
                "α-CROWN warm-start: {} iters, best output lower = {:.6}, {} unstable ReLU, {} monotone nodes, {} sqrt nodes",
                config.iterations, best_output_lower, num_unstable, num_monotone, num_sqrt
            );
        }

        // Step 4+5: with optimized alphas, compute intermediate CROWN bounds and intersect with
        // the reference bounds for soundness — UNLESS config.fix_interm_bounds is set, in which
        // case skip the expensive O(N²) post-optimization CROWN pass and use the sound reference
        // bounds directly
        // (mirrors the main alpha-CROWN path, alpha.rs:451-453). fix_interm_bounds defaults true, so
        // this avoids a per-domain CROWN pass in the BaB hot path. SOUND either way: skipping CROWN
        // keeps a certified reference over-approximation — never tighter than the true range.
        let mut best_bounds = reference_bounds.clone();
        if !config.fix_interm_bounds {
            let crown_bounds = self.collect_crown_bounds_with_alpha(
                input,
                &reference_bounds,
                &alpha_state,
                engine,
                config.deadline,
            )?;
            for (name, crown_bound) in &crown_bounds {
                if let Some(reference_bound) = reference_bounds.get(name) {
                    if crown_bound.shape() == reference_bound.shape() {
                        let (tightened, disjoint) = reference_bound
                            .intersection_per_element(crown_bound)
                            .unwrap_or_else(|| {
                                debug!(
                                    "α-CROWN warm-start: {} intersection failed (NaN/shape), using reference bound",
                                    name
                                );
                                (reference_bound.clone(), 0)
                            });
                        if disjoint > 0 {
                            debug!(
                                "α-CROWN warm-start: {} intersection: {} disjoint, union fallback",
                                name, disjoint
                            );
                        }
                        best_bounds.insert(name.clone(), tightened);
                    }
                }
            }
        }

        // Merge the element-wise best output bounds (across optimization iterations) into the
        // output node's bounds (#2251), mirroring alpha.rs:497-543. Intersect-only with the
        // existing reference∩CROWN bound, so the result stays a sound over-approximation; inversions
        // and disjoint elements fall back safely (clamp to ±inf / union). Without this the
        // scalar-sum comparison can discard per-dimension improvements the warm-start found.
        if let (Some(mut ew_lower), Some(mut ew_upper)) =
            (best_output_lower_arr, best_output_upper_arr)
        {
            crate::network::graph_alpha::propagate_helpers::clamp_inverted_best_bounds(
                &mut ew_lower,
                &mut ew_upper,
                "α-CROWN warm-start output",
            );
            match BoundedTensor::new_allow_infinite(ew_lower, ew_upper) {
                Ok(ew_bounds) => match best_bounds.get(&output_node) {
                    Some(existing) if existing.shape() == ew_bounds.shape() => {
                        let (tightened, disjoint) = existing
                            .intersection_per_element(&ew_bounds)
                            .unwrap_or_else(|| (existing.clone(), 0));
                        if disjoint > 0 {
                            debug!(
                                "α-CROWN warm-start: elementwise best: {} disjoint, union fallback",
                                disjoint
                            );
                        }
                        best_bounds.insert(output_node.clone(), tightened);
                    }
                    Some(_) => {}
                    None => {
                        best_bounds.insert(output_node.clone(), ew_bounds);
                    }
                },
                Err(e) => {
                    warn!(
                        "α-CROWN warm-start: new_allow_infinite failed for output {output_node:?}, per-dim improvement dropped: {e}"
                    );
                }
            }
        }

        debug!(
                "GraphNetwork α-CROWN warm-start: {} unstable ReLU neurons, {} monotone nodes, {} sqrt nodes, {} bound sets",
                num_unstable,
                num_monotone,
                num_sqrt,
                best_bounds.len()
            );

        Ok((best_bounds, alpha_state))
    }
}
