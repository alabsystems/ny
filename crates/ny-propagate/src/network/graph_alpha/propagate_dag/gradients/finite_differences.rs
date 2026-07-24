// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Finite differences gradient computation for DAG α-CROWN.
//!
//! Extracted from `gradients.rs` as part of #4187.

use super::super::super::runtime_state::DagAlphaRuntimeState;
use super::super::DagAlphaLoopContext;
use super::{alpha_refs, zeros_like_gradients};
use crate::network::alpha_crown_loop::finite_lower_sum;
use crate::network::core::GraphNetwork;
use ndarray::{Array1, Array2, Array4};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;

impl GraphNetwork {
    /// Finite differences gradient computation for DAG α-CROWN.
    ///
    /// Computes per-neuron gradients by perturbing each alpha value ±eps
    /// and measuring the change in the lower bound sum.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::network::graph_alpha::propagate_dag) fn compute_fd_gradients(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &mut DagAlphaRuntimeState,
        bilinear_alphas: &HashMap<String, Array4<f32>>,
        mul_binary_alphas: &HashMap<String, Array2<f32>>,
        gradients: &[Array1<f32>],
        num_relus: usize,
        eps: f32,
    ) -> Result<Vec<Array1<f32>>> {
        // Pass current bilinear/mul_binary alphas through backward calls (#3287, #3439).
        let (bl_fd, mb_fd) = alpha_refs(ctx, bilinear_alphas, mul_binary_alphas);

        let mut scratch = zeros_like_gradients(gradients);
        let mut scratch_upper = zeros_like_gradients(gradients);
        let original_graph = runtime.snapshot_graph();

        // Wrap in closure so `?` returns to the closure, not the
        // outer function. Ensures alpha restoration on error (#2554).
        let result = (|| -> Result<Vec<Array1<f32>>> {
            let mut grads = Vec::with_capacity(num_relus);

            for relu_idx in 0..num_relus {
                let num_neurons = runtime.relu_len(relu_idx).ok_or_else(|| {
                    NyError::InternalError(format!(
                        "missing DAG ReLU alpha length for index {relu_idx}"
                    ))
                })?;
                let mut grad = Array1::<f32>::zeros(num_neurons);

                for neuron_idx in 0..num_neurons {
                    let mask = runtime.relu_unstable_mask(relu_idx).ok_or_else(|| {
                        NyError::InternalError(format!(
                            "missing DAG ReLU unstable mask for index {relu_idx}"
                        ))
                    })?;
                    if !mask[neuron_idx] {
                        continue;
                    }

                    let (orig_alpha, orig_alpha_upper) =
                        runtime.relu_alpha_entry(relu_idx, neuron_idx)?;

                    // f(alpha + eps) — perturb both paths (#3393)
                    runtime.set_relu_alpha_entry(
                        relu_idx,
                        neuron_idx,
                        orig_alpha + eps,
                        orig_alpha_upper + eps,
                    )?;
                    scratch.iter_mut().for_each(|g| g.fill(0.0));
                    scratch_upper.iter_mut().for_each(|g| g.fill(0.0));
                    // Propagate CROWN errors (#2065, #1941).
                    let bounds_plus = self.dag_alpha_backward_pass_with_engine(
                        ctx.input,
                        node_bounds,
                        ctx.exec_order,
                        ctx.output_dim,
                        ctx.input_dim,
                        runtime.relu_name_to_idx(),
                        runtime.graph(),
                        runtime.invprop(),
                        &mut scratch,
                        &mut scratch_upper,
                        ctx.engine,
                        bl_fd,
                        mb_fd,
                        ctx.config.deadline,
                    )?;
                    // Filter -Inf to prevent NaN gradient from (-Inf)-(-Inf) (#3272, #2857).
                    let lower_plus: f32 = finite_lower_sum(bounds_plus.lower());

                    // f(alpha - eps) — perturb both paths (#3393)
                    runtime.set_relu_alpha_entry(
                        relu_idx,
                        neuron_idx,
                        orig_alpha - eps,
                        orig_alpha_upper - eps,
                    )?;
                    scratch.iter_mut().for_each(|g| g.fill(0.0));
                    scratch_upper.iter_mut().for_each(|g| g.fill(0.0));
                    // Propagate CROWN errors (#2065, #1941).
                    let bounds_minus = self.dag_alpha_backward_pass_with_engine(
                        ctx.input,
                        node_bounds,
                        ctx.exec_order,
                        ctx.output_dim,
                        ctx.input_dim,
                        runtime.relu_name_to_idx(),
                        runtime.graph(),
                        runtime.invprop(),
                        &mut scratch,
                        &mut scratch_upper,
                        ctx.engine,
                        bl_fd,
                        mb_fd,
                        ctx.config.deadline,
                    )?;
                    // Filter -Inf to prevent NaN gradient from (-Inf)-(-Inf) (#3272, #2857).
                    let lower_minus: f32 = finite_lower_sum(bounds_minus.lower());

                    // Per-neuron restore (inside closure)
                    runtime.set_relu_alpha_entry(
                        relu_idx,
                        neuron_idx,
                        orig_alpha,
                        orig_alpha_upper,
                    )?;
                    grad[neuron_idx] = (lower_plus - lower_minus) / (2.0 * eps);
                }

                grads.push(grad);
            }

            Ok(grads)
        })();

        // Always restore alpha state, including early error
        // returns from `?` inside the closure (#2554, #3393).
        runtime.restore_graph(&original_graph);
        result
    }
}
