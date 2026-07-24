// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alpha-CROWN backward pass helpers for `GraphNetwork`.

mod gpu_suffix;
mod gradients;
mod nonlinear;

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::bounds::{GraphAlphaCrownIntermediate, GraphAlphaState, LinearBounds};
use crate::invprop::InvpropState;
use crate::layers::Layer;
use crate::network::backward_dispatch::{dispatch_backward_layer, DispatchContext};
use crate::network::core::{crown_backward_step_patches, CrownStepResult};
use crate::network::graph_alpha::invprop_backward::augment_bounds_with_constraints;
use crate::network::CrownMergeAccumulator;
use crate::MulBinaryRelaxationMode;

use ndarray::Array1;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

#[cfg(test)]
use nonlinear::retry_monotone_shape_mismatch_with_fixed_slope;
use nonlinear::{handle_nonlinear_node, DagAlphaNodeContext, NonlinearNodeResult};

use crate::network::core::{
    apply_dense_backward_dispatch_result, try_dense_spatial_patches_reentry, GraphNetwork,
    NETWORK_INPUT,
};

impl GraphNetwork {
    fn apply_invprop_constraints(
        node_name: &str,
        bounds: LinearBounds,
        invprop_state: Option<&InvpropState>,
    ) -> LinearBounds {
        let state = match invprop_state {
            Some(state) => state,
            None => return bounds,
        };

        let gammas = match state.layer_gammas(node_name) {
            Some(gammas) if gammas.active => gammas,
            _ => return bounds,
        };

        let gammas_lower = gammas.lower_gammas().to_owned();
        let gammas_upper = gammas.upper_gammas().to_owned();

        augment_bounds_with_constraints(&bounds, &state.constraints, &gammas_lower, &gammas_upper)
    }

    /// Helper method to run a single DAG backward pass with alpha values and optional engine.
    /// Returns concrete bounds and populates gradients for alpha optimization.
    // Justification: DAG backward pass needs input, node bounds, execution order, output dim,
    // alpha state, objective coefficients, engine, and constraints — from different graph sources.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dag_alpha_backward_pass_with_engine(
        &self,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        relu_name_to_idx: &HashMap<String, usize>,
        alpha_state: &GraphAlphaState,
        invprop_state: Option<&InvpropState>,
        gradients: &mut [Array1<f32>],
        gradients_upper: &mut [Array1<f32>],
        engine: Option<&dyn GemmEngine>,
        bilinear_alphas: Option<&HashMap<String, ndarray::Array4<f32>>>,
        mul_binary_alphas: Option<&HashMap<String, ndarray::Array2<f32>>>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let (bounds, _) = self.dag_alpha_backward_pass_core(
            input,
            node_bounds,
            exec_order,
            output_dim,
            input_dim,
            relu_name_to_idx,
            alpha_state,
            invprop_state,
            gradients,
            gradients_upper,
            true,
            engine,
            None,
            bilinear_alphas,
            mul_binary_alphas,
            deadline,
        )?;
        Ok(bounds)
    }

    /// Certified DAG alpha backward without producing local ReLU gradients.
    ///
    /// This is reserved for an optimization loop's terminal evaluated state:
    /// the bound arithmetic is identical to
    /// [`Self::dag_alpha_backward_pass_with_engine`], but no gradient buffer is
    /// allocated or filled because no later bound can consume an update.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dag_alpha_bound_pass_with_engine(
        &self,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        relu_name_to_idx: &HashMap<String, usize>,
        alpha_state: &GraphAlphaState,
        invprop_state: Option<&InvpropState>,
        engine: Option<&dyn GemmEngine>,
        bilinear_alphas: Option<&HashMap<String, ndarray::Array4<f32>>>,
        mul_binary_alphas: Option<&HashMap<String, ndarray::Array2<f32>>>,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let mut gradients = Vec::<Array1<f32>>::new();
        let mut gradients_upper = Vec::<Array1<f32>>::new();
        let (bounds, _) = self.dag_alpha_backward_pass_core(
            input,
            node_bounds,
            exec_order,
            output_dim,
            input_dim,
            relu_name_to_idx,
            alpha_state,
            invprop_state,
            &mut gradients,
            &mut gradients_upper,
            false,
            engine,
            None,
            bilinear_alphas,
            mul_binary_alphas,
            deadline,
        )?;
        Ok(bounds)
    }

    /// DAG backward pass that stores intermediate A matrices for chain-rule gradient computation.
    ///
    /// This is similar to `dag_alpha_backward_pass_with_engine` but also captures the A matrix
    /// at each ReLU node BEFORE the ReLU is applied, enabling true chain-rule gradients.
    // Justification: Same as dag_alpha_backward_pass_with_engine plus intermediate A-matrix
    // capture — parameters are the full graph alpha-CROWN context.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dag_alpha_backward_pass_with_intermediates(
        &self,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        relu_name_to_idx: &HashMap<String, usize>,
        alpha_state: &GraphAlphaState,
        invprop_state: Option<&InvpropState>,
        gradients: &mut [Array1<f32>],
        gradients_upper: &mut [Array1<f32>],
        engine: Option<&dyn GemmEngine>,
        bilinear_alphas: Option<&HashMap<String, ndarray::Array4<f32>>>,
        mul_binary_alphas: Option<&HashMap<String, ndarray::Array2<f32>>>,
        deadline: Option<Instant>,
    ) -> Result<(BoundedTensor, GraphAlphaCrownIntermediate)> {
        let mut intermediate = GraphAlphaCrownIntermediate::new();
        let (bounds, _) = self.dag_alpha_backward_pass_core(
            input,
            node_bounds,
            exec_order,
            output_dim,
            input_dim,
            relu_name_to_idx,
            alpha_state,
            invprop_state,
            gradients,
            gradients_upper,
            true,
            engine,
            Some(&mut intermediate),
            bilinear_alphas,
            mul_binary_alphas,
            deadline,
        )?;
        Ok((bounds, intermediate))
    }

    /// Shared backward pass core. When `intermediate` is `Some`, captures A-matrices and
    /// pre-ReLU bounds at each ReLU node for chain-rule gradient computation.
    ///
    /// Returns `(concretized_bounds, final_linear_bounds)` where `final_linear_bounds` is
    /// the pre-concretization `LinearBounds` at the input node.
    // Justification: DAG backward pass needs input, node bounds, execution order, output dim,
    // alpha state, objective coefficients, engine, constraints, and optional intermediate
    // storage — from different graph sources.
    #[allow(clippy::too_many_arguments)]
    fn dag_alpha_backward_pass_core(
        &self,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        relu_name_to_idx: &HashMap<String, usize>,
        alpha_state: &GraphAlphaState,
        invprop_state: Option<&InvpropState>,
        gradients: &mut [Array1<f32>],
        gradients_upper: &mut [Array1<f32>],
        track_gradients: bool,
        engine: Option<&dyn GemmEngine>,
        intermediate: Option<&mut GraphAlphaCrownIntermediate>,
        bilinear_alphas: Option<&HashMap<String, ndarray::Array4<f32>>>,
        mul_binary_alphas: Option<&HashMap<String, ndarray::Array2<f32>>>,
        deadline: Option<Instant>,
    ) -> Result<(BoundedTensor, Option<LinearBounds>)> {
        // Determine output node
        let output_node_name = if self.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
        } else {
            &self.output_node
        };

        // Pre-build indexed lookups to avoid HashMap probes in the backward hot loop.
        let mut name_to_idx: HashMap<&str, usize> = HashMap::with_capacity(exec_order.len() + 1);
        for (i, name) in exec_order.iter().enumerate() {
            name_to_idx.insert(name.as_str(), i);
        }
        name_to_idx.insert(NETWORK_INPUT, exec_order.len());
        let nodes_by_idx: Vec<&_> = exec_order
            .iter()
            .map(|name| {
                self.nodes
                    .get(name)
                    .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", name)))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut bounds_by_idx: Vec<Option<&BoundedTensor>> = exec_order
            .iter()
            .map(|name| node_bounds.get(name.as_str()))
            .collect();
        bounds_by_idx.push(Some(input)); // NETWORK_INPUT

        // Phase 1: CrownBounds-aware initialization (#3293).
        // Patches mode for 3D spatial output with Conv2d; matrix mode with cuts
        // (reference: abcrown.py:228-231).
        let mut node_crown_bounds = CrownMergeAccumulator::new_indexed(exec_order);

        let output_idx = *name_to_idx.get(output_node_name.as_str()).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Output node {} not in exec_order",
                output_node_name
            ))
        })?;
        let output_bounds = bounds_by_idx[output_idx].ok_or_else(|| {
            NyError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;
        let output_shape = output_bounds.shape();

        let has_conv2d = nodes_by_idx
            .iter()
            .any(|n| matches!(n.layer, Layer::Conv2d(_)));
        let use_patches_seed = output_shape.len() == 3 && has_conv2d && self.use_patches_mode;
        // #margin-subset-alpha: when the initial-bounds scope published the
        // spec-referenced OUTPUT indices (single-margin specs on wide heads,
        // e.g. vggnet16 `(>= Y_200 Y_177)` over 1000 outputs), seed ONLY the k
        // referenced identity rows. SOUND by row-independence: the backward
        // walk, the per-row CROWN error term, and the per-row concretize are
        // all row-local, so the k computed rows are bit-identical to their
        // full-width counterparts; the concretized k rows are scattered over
        // the output node's sound reference bounds at the exits below, so
        // every returned row remains a valid enclosure. This also scopes the
        // per-ReLU alpha GRADIENTS to the k rows the objective actually reads
        // — full-width the alpha phase materialized `[1000 x 401408]` conv
        // buffers per iteration (kernel-OOM at 119 GB on vggnet16 spec1) and
        // never finished iteration 0. Unpublished scope => `None` =>
        // byte-identical full-width behavior.
        //
        // INVPROP note: with a k-row seed the output-seed gamma augment's
        // `is_output_identity_seed` gate fails closed (no augment). That only
        // forgoes an optional tightening — the shipped default keeps gammas at
        // zero (`optimize_gammas` off), where the augment is a no-op anyway.
        let margin_subset = if use_patches_seed {
            None
        } else {
            crate::output_margin_seed::margin_subset_indices(output_dim)
        };
        let initial_crown_bounds = if use_patches_seed {
            let (oc, oh, ow) = (output_shape[0], output_shape[1], output_shape[2]);
            debug!(
                "DAG α-CROWN: Initializing Patches mode (output {}x{}x{})",
                oc, oh, ow
            );
            CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
                (oc, oh, ow),
                (oc, oh, ow),
            )))
        } else if let Some(indices) = margin_subset.as_deref() {
            debug!(
                "DAG α-CROWN: margin-subset OUTPUT seed engaged (k={} of {} rows)",
                indices.len(),
                output_dim
            );
            CrownBounds::Dense(LinearBounds::identity_rows(output_dim, indices))
        } else {
            CrownBounds::Dense(LinearBounds::identity(output_dim))
        };
        node_crown_bounds.insert(output_node_name.clone(), initial_crown_bounds);

        // Every downstream use of `output_dim` in this walk denotes the SEED
        // ROW COUNT (accumulator hints, zero-coefficient bias blocks) — shadow
        // it with the actual row count. Full-width seeds keep it unchanged.
        let output_dim = margin_subset.as_deref().map_or(output_dim, <[usize]>::len);

        // Track if we've accumulated bounds to the input
        let mut input_accumulated = false;

        // We need a mutable reference to the intermediate that persists across the loop,
        // so rebind it here.
        let mut intermediate = intermediate;

        // Backward pass through nodes in reverse order.
        // Use enumeration to get direct Vec indices, avoiding HashMap lookups.
        for (rev_pos, node_name) in exec_order.iter().rev().enumerate() {
            let idx = exec_order.len() - 1 - rev_pos;
            let node = nodes_by_idx[idx];

            // Get this node's accumulated CrownBounds via direct index.
            let mut node_cb = match node_crown_bounds.take_by_idx(idx)? {
                Some(cb) => cb,
                None => {
                    // Node has no consumers (not output, not used by anyone)
                    continue;
                }
            };

            // #3813 Slice 5: Dense→Patches re-entry at Conv2d boundaries.
            // Mirror of propagation.rs:455-490 for alpha-CROWN.
            // When classifier-head logits reach a Conv2d node through Dense rows
            // (after Linear/Flatten), convert to Patches so the Conv2d/ReLU Patches
            // path handles the CNN trunk with alpha optimization preserved.
            // Gated by use_patches_mode: matrix mode skips re-entry (abcrown.py:228-231).
            try_dense_spatial_patches_reentry(
                &mut node_cb,
                node,
                node_name,
                node_bounds,
                self.use_patches_mode,
                "DAG α-CROWN",
            );

            // Get pre-activation bounds for this node.
            // Use first input (not require_unary_input) because multi-input nodes
            // like MulBinary, BilinearCrown, Div, and Where resolve their full
            // input sets in their specific handlers or the shared dispatch core.
            // Fix: #4113 (regression from #4097 tightening require_unary_input).
            let first_input = node.inputs.first().map(String::as_str).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Node '{}' ({}) has no inputs",
                    node_name,
                    node.layer.layer_type()
                ))
            })?;
            let first_input_idx = name_to_idx.get(first_input).copied();
            let pre_activation = match first_input_idx {
                Some(fi) => bounds_by_idx[fi].ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        first_input
                    ))
                })?,
                None => {
                    return Err(NyError::InvalidSpec(format!(
                        "Pre-activation input {} not in exec_order or NETWORK_INPUT",
                        first_input
                    )));
                }
            };

            node_cb = match handle_nonlinear_node(
                self,
                node_name,
                node,
                first_input,
                node_cb,
                pre_activation,
                DagAlphaNodeContext {
                    input,
                    relu_name_to_idx,
                    alpha_state,
                    invprop_state,
                    gradients,
                    gradients_upper,
                    track_gradients,
                    node_crown_bounds: &mut node_crown_bounds,
                    intermediate: intermediate.as_deref_mut(),
                    output_dim,
                    input_dim,
                    input_accumulated: &mut input_accumulated,
                    engine,
                    deadline,
                },
            )? {
                NonlinearNodeResult::NotHandled(node_cb) => *node_cb,
                NonlinearNodeResult::Continue => continue,
                NonlinearNodeResult::ReturnBounds(bounds) => return Ok((bounds, None)),
            };

            let is_patches = matches!(&node_cb, CrownBounds::Patches(_));

            // === Non-ReLU Patches fast-path (#3293) ===
            if is_patches && node.inputs.len() == 1 {
                match crown_backward_step_patches(
                    &node.layer,
                    &mut node_cb,
                    pre_activation,
                    engine,
                    0,
                    "DAG-α-CROWN",
                    deadline,
                ) {
                    Ok(CrownStepResult::Continue) => {
                        self.accumulate_crown_bounds_to_input(
                            first_input,
                            node_cb,
                            &mut node_crown_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                        continue;
                    }
                    Ok(CrownStepResult::IbpFallback(fallback)) => {
                        if fallback.reason
                            == crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded
                        {
                            debug!(
                                "DAG α-CROWN: Patches dispatch hit memory budget at {}: {}; falling back to CROWN",
                                node_name, fallback.details
                            );
                            let bounds = self
                                .propagate_crown_with_engine_and_deadline(input, engine, deadline)?
                                .bounds;
                            return Ok((bounds, None));
                        }
                        // Fall through to Dense dispatch below
                    }
                    Err(_) => {
                        // Fall through to Dense dispatch below
                    }
                }
                // Ensure Dense for below.
                if matches!(&node_cb, CrownBounds::Patches(_)) {
                    match node_cb.ensure_dense() {
                        Ok(_) => {}
                        Err(e) => {
                            debug!(
                                "DAG α-CROWN: ensure_dense failed at {}: {}, CROWN fallback",
                                node_name, e
                            );
                            let bounds = self
                                .propagate_crown_with_engine_and_deadline(input, engine, deadline)?
                                .bounds;
                            return Ok((bounds, None));
                        }
                    }
                }
            }

            // === Patches residual passthrough for Add/Sub (#4382) ===
            if crate::network::core::graph::backward_helpers::try_apply_patches_residual_passthrough(
                self,
                node,
                &node_cb,
                node_bounds,
                &mut node_crown_bounds,
                output_dim,
                input_dim,
                &mut input_accumulated,
                "DAG-α-CROWN",
            )? {
                continue;
            }

            // === Dense dispatch ===
            let node_lb = node_cb.into_dense()?;
            let node_lb = Self::apply_invprop_constraints(node_name, node_lb, invprop_state);

            // Try GPU suffix: if the remaining backward chain from this node to
            // NETWORK_INPUT is a GPU-extractable unary chain, offload the entire
            // suffix to GPU and return directly.
            //
            // Skip when `intermediate.is_some()` (#GPU-suffix-alpha-freeze): the
            // AnalyticChain gradient pass captures pre-ReLU A-matrices at each
            // ReLU node to compute true chain-rule alpha gradients. The GPU suffix
            // returns concretized bounds and `None` for intermediates, so taking
            // it here would leave the intermediate store empty → zero ReLU
            // gradients → alpha frozen at its initial value (slope 1.0). The CPU
            // backward then never runs, so the optimized (tighter) alpha is never
            // found and the GPU path returns a SOUND-but-LOOSER bound than CPU.
            // The bounds-only pass (`intermediate.is_none()`) still uses the GPU
            // suffix, now reading the alpha that the CPU gradient pass optimized.
            let gpu_suffix_bounds = if intermediate.is_none() {
                self.try_alpha_backward_gpu_suffix(
                    input,
                    &node_lb,
                    node_name,
                    node_bounds,
                    alpha_state,
                    engine,
                )?
            } else {
                None
            };
            if let Some(concrete_bounds) = gpu_suffix_bounds {
                // #margin-subset-alpha: the GPU suffix concretized the k-row
                // seed chain — scatter over the output node's sound reference
                // bounds (full-width no-op).
                let concrete_bounds = match margin_subset.as_deref() {
                    Some(indices) => crate::output_margin_seed::scatter_subset_bounds_over_base(
                        output_bounds,
                        indices,
                        &concrete_bounds,
                    )?,
                    None => concrete_bounds,
                };
                return Ok((concrete_bounds, None));
            }

            let ctx = DispatchContext {
                node_name,
                layer: &node.layer,
                inputs: &node.inputs,
                pre_activation,
                network_input: input,
                node_bounds: node_bounds.into(),
                engine,
                deadline,
                bilinear_alphas,
                mul_binary_relaxation: MulBinaryRelaxationMode::default(),
                mul_binary_alphas,
                norm_inv_rms_override: None,
            };

            // #3813: Catch ShapeMismatch from Dense Conv2d backward when graph
            // restructuring (e.g., RSPLITTER) changes intermediate dimensions.
            // Fall back to plain CROWN (same as Unsupported path below).
            let result = match dispatch_backward_layer(&ctx, &node_lb) {
                Ok(r) => r,
                Err(
                    e @ NyError::ShapeMismatch { .. }
                    | e @ NyError::UnsupportedOp(_)
                    | e @ NyError::UnsupportedConfiguration(_)
                    | e @ NyError::NumericalInstability(_)
                    | e @ NyError::DeadlineExceeded(_),
                ) => {
                    debug!(
                        "DAG α-CROWN: dispatch error at {} ({}): {}, falling back to CROWN",
                        node_name,
                        node.layer.layer_type(),
                        e,
                    );
                    let bounds = self
                        .propagate_crown_with_engine_and_deadline(input, engine, deadline)?
                        .bounds;
                    return Ok((bounds, None));
                }
                Err(e) => return Err(e),
            };

            match apply_dense_backward_dispatch_result(
                self,
                node,
                first_input,
                &node_lb,
                result,
                &mut node_crown_bounds,
                output_dim,
                input_dim,
                &mut input_accumulated,
                "Alpha dispatch",
            ) {
                Ok(()) => {}
                Err(NyError::UnsupportedOp(reason)) => {
                    debug!(
                        "DAG α-CROWN: Unsupported layer {} ({}): {}, falling back to CROWN",
                        node_name,
                        node.layer.layer_type(),
                        reason,
                    );
                    let bounds = self
                        .propagate_crown_with_engine_and_deadline(input, engine, deadline)?
                        .bounds;
                    return Ok((bounds, None));
                }
                Err(e) => return Err(e),
            }
        }

        // Concretize final bounds.
        let final_cb = node_crown_bounds.take(NETWORK_INPUT)?.ok_or_else(|| {
            if input_accumulated {
                NyError::InvalidSpec(
                    "DAG α-CROWN: Input bounds accumulated but not found".to_string(),
                )
            } else {
                NyError::InvalidSpec("DAG α-CROWN: No path from output to input".to_string())
            }
        })?;
        let input_lb = final_cb.into_dense()?;

        if let Some(ref mut inter) = intermediate {
            inter.final_bounds = input_lb.clone();
        }
        let final_lb = input_lb.clone();
        let concrete = input_lb.concretize_sound(input);
        // #margin-subset-alpha: scatter the k concretized rows over the output
        // node's sound reference bounds so callers always see the full output
        // width. Unreferenced rows keep the reference enclosure (valid, merely
        // looser); the elementwise best-bounds merge and the spec early-exit
        // projection only ever read the k referenced rows.
        let concrete = match margin_subset.as_deref() {
            Some(indices) => crate::output_margin_seed::scatter_subset_bounds_over_base(
                output_bounds,
                indices,
                &concrete,
            )?,
            None => concrete,
        };
        Ok((concrete, Some(final_lb)))
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod true_grad_oracle_tests;
