// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Alpha-CROWN backward pass helpers for `GraphNetwork`.

mod gpu_suffix;
pub(crate) mod gradients;
mod nonlinear;

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds, PatchesMaterializationPurpose};
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

use nonlinear::{handle_nonlinear_node, AlphaRowScope, DagAlphaNodeContext, NonlinearNodeResult};
#[cfg(test)]
use nonlinear::{
    record_dense_relu_intermediate, record_patches_relu_intermediate,
    retry_monotone_shape_mismatch_with_fixed_slope,
};

use crate::network::core::{
    apply_dense_backward_dispatch_result_with_deadline,
    try_dense_spatial_patches_reentry_with_deadline, GraphNetwork, NETWORK_INPUT,
};

/// Execute the historical full-output CROWN fallback unless this is the
/// compact exact-seed face.
///
/// A fallback seeded with the raw output identity can allocate O(output width)
/// rows before the exact-seed wrapper notices that intermediates were lost.
/// Refusing before invoking the callback is therefore part of that face's OOM
/// contract, not merely an optimization.
fn full_output_crown_fallback_or_refuse<F>(
    forbid_full_output_fallback: bool,
    fallback: F,
) -> Result<BoundedTensor>
where
    F: FnOnce() -> Result<BoundedTensor>,
{
    if forbid_full_output_fallback {
        return Err(NyError::UnsupportedConfiguration(
            "DAG alpha-CROWN compact exact seed refused a full-output CROWN fallback".to_string(),
        ));
    }
    fallback()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PatchesResourceErrorDisposition {
    /// Preserve the deadline authority and return the error immediately.
    AtomicReturn,
    /// Skip densification and take the existing full-CROWN memory fallback.
    FullCrownFallback,
    /// Preserve the historical semantic-error Dense retry.
    DenseRetry,
}

/// Classify resource-bounded Patches failures before any Dense materialization.
///
/// A deadline remains authoritative and propagates to the outer per-node
/// fallback classifier. A memory refusal takes the same full-CROWN fallback as
/// `CrownStepResult::IbpFallback(MemoryBudgetExceeded)`; returning the raw error
/// here would abort public alpha propagation instead of degrading safely.
fn classify_patches_resource_error(error: &NyError) -> PatchesResourceErrorDisposition {
    match error {
        NyError::DeadlineExceeded(_) => PatchesResourceErrorDisposition::AtomicReturn,
        NyError::CpuMemoryExceeded { .. } => PatchesResourceErrorDisposition::FullCrownFallback,
        _ => PatchesResourceErrorDisposition::DenseRetry,
    }
}

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

        let (gammas_lower, gammas_upper) = match state.layer_gammas(node_name) {
            Some(gammas) if gammas.active => match gammas.checked_bound_gammas() {
                Some(pair) => pair,
                None => return bounds,
            },
            _ => return bounds,
        };

        augment_bounds_with_constraints(
            &bounds,
            &state.constraints,
            &gammas_lower.to_owned(),
            &gammas_upper.to_owned(),
        )
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
        let (bounds, _, _, _, _) = self.dag_alpha_backward_pass_core(
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
            None,
        )?;
        Ok(bounds)
    }

    /// Authoritative DAG backward result, including certified infeasibility
    /// provenance captured before conservative bound repair.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dag_alpha_backward_pass_with_engine_and_infeasibility(
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
    ) -> Result<(BoundedTensor, bool, Option<f64>, Vec<Option<f64>>, bool)> {
        let (
            bounds,
            final_linear_bounds,
            certified_finite_inversion,
            max_finite_gap,
            row_finite_gaps,
        ) = self.dag_alpha_backward_pass_core(
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
            None,
        )?;
        Ok((
            bounds,
            certified_finite_inversion,
            max_finite_gap,
            row_finite_gaps,
            final_linear_bounds.is_some(),
        ))
    }

    /// Certified DAG alpha backward without producing local ReLU gradients.
    ///
    /// This is reserved for an optimization loop's terminal evaluated state:
    /// the bound arithmetic is identical to
    /// [`Self::dag_alpha_backward_pass_with_engine`], but no gradient buffer is
    /// allocated or filled because no later bound can consume an update.
    #[cfg(test)]
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
        let (bounds, _, _, _, _) = self.dag_alpha_backward_pass_core(
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
            None,
        )?;
        Ok(bounds)
    }

    /// Bound-only authoritative DAG backward with pre-repair infeasibility
    /// provenance.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dag_alpha_bound_pass_with_engine_and_infeasibility(
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
    ) -> Result<(BoundedTensor, bool, Option<f64>, Vec<Option<f64>>, bool)> {
        let mut gradients = Vec::<Array1<f32>>::new();
        let mut gradients_upper = Vec::<Array1<f32>>::new();
        let (
            bounds,
            final_linear_bounds,
            certified_finite_inversion,
            max_finite_gap,
            row_finite_gaps,
        ) = self.dag_alpha_backward_pass_core(
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
            None,
        )?;
        Ok((
            bounds,
            certified_finite_inversion,
            max_finite_gap,
            row_finite_gaps,
            final_linear_bounds.is_some(),
        ))
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
        let (bounds, _, _, _, _) = self.dag_alpha_backward_pass_core(
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
            None,
        )?;
        Ok((bounds, intermediate))
    }

    /// DAG alpha backward with one caller-supplied exact dense output seed.
    ///
    /// This is the bounded-allocation face used by gap attribution: a compact
    /// `k x output_dim` property matrix seeds the original graph directly, so
    /// no synthetic-head `GraphNetwork::clone()` can duplicate model weights
    /// and caches. The seed rows are shared-alpha objectives, not raw output
    /// spec slots.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dag_alpha_backward_pass_with_intermediates_and_exact_seed(
        &self,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        exec_order: &[String],
        output_dim: usize,
        input_dim: usize,
        relu_name_to_idx: &HashMap<String, usize>,
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn GemmEngine>,
        deadline: Instant,
        seed: &ndarray::Array2<f32>,
    ) -> Result<(BoundedTensor, GraphAlphaCrownIntermediate)> {
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "DAG alpha-CROWN exact seed deadline expired before validation".to_string(),
            ));
        }
        if alpha_state.has_spec_deltas() {
            return Err(NyError::InvalidSpec(
                "DAG alpha-CROWN exact seed requires a shared-alpha state without spec deltas"
                    .to_string(),
            ));
        }
        if seed.nrows() == 0 || seed.nrows() > 3 || seed.ncols() != output_dim {
            return Err(NyError::InvalidSpec(format!(
                "DAG alpha-CROWN exact seed must be 1..=3 x {output_dim}, got {}x{}",
                seed.nrows(),
                seed.ncols()
            )));
        }
        let seed = LinearBounds::from_spec_matrix(seed.clone())?;
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "DAG alpha-CROWN exact seed deadline expired during validation".to_string(),
            ));
        }
        let mut gradients = Vec::<Array1<f32>>::new();
        let mut gradients_upper = Vec::<Array1<f32>>::new();
        let mut intermediate = GraphAlphaCrownIntermediate::new();
        let (bounds, final_linear_bounds, _, _, _) = self.dag_alpha_backward_pass_core(
            input,
            node_bounds,
            exec_order,
            output_dim,
            input_dim,
            relu_name_to_idx,
            alpha_state,
            None,
            &mut gradients,
            &mut gradients_upper,
            false,
            engine,
            Some(&mut intermediate),
            None,
            None,
            Some(deadline),
            Some(seed),
        )?;
        if Instant::now() >= deadline {
            return Err(NyError::DeadlineExceeded(
                "DAG alpha-CROWN exact seed deadline expired during backward".to_string(),
            ));
        }
        if final_linear_bounds.is_none() {
            return Err(NyError::InvalidSpec(
                "DAG alpha-CROWN exact seed degraded before producing final linear bounds"
                    .to_string(),
            ));
        }
        Ok((bounds, intermediate))
    }

    /// Shared backward pass core. When `intermediate` is `Some`, captures A-matrices and
    /// pre-ReLU bounds at each ReLU node for chain-rule gradient computation.
    ///
    /// Returns `(concretized_bounds, final_linear_bounds,
    /// certified_finite_inversion, max_finite_gap, row_finite_gaps)`, preserving
    /// pre-repair proof/optimizer provenance while `concretized_bounds` remains
    /// a valid non-inverted `BoundedTensor`.
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
        exact_output_seed: Option<LinearBounds>,
    ) -> Result<(
        BoundedTensor,
        Option<LinearBounds>,
        bool,
        Option<f64>,
        Vec<Option<f64>>,
    )> {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(
                "DAG alpha-CROWN deadline expired before backward admission".to_string(),
            ));
        }
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
        let exact_seed_rows = if let Some(seed) = exact_output_seed.as_ref() {
            if seed.num_outputs() == 0
                || seed.num_inputs() != output_dim
                || output_bounds.len() != output_dim
            {
                return Err(NyError::ShapeMismatch {
                    expected: vec![output_dim],
                    got: vec![seed.num_outputs(), seed.num_inputs(), output_bounds.len()],
                });
            }
            Some(seed.num_outputs())
        } else {
            None
        };
        let use_patches_seed = exact_seed_rows.is_none()
            && output_shape.len() == 3
            && has_conv2d
            && self.use_patches_mode;
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
        let margin_subset = if exact_seed_rows.is_some() || use_patches_seed {
            None
        } else {
            crate::output_margin_seed::margin_subset_indices(output_dim)
        };
        let initial_crown_bounds = if let Some(seed) = exact_output_seed {
            debug!(
                "DAG alpha-CROWN: exact dense output seed engaged (k={} of {} columns)",
                seed.num_outputs(),
                seed.num_inputs()
            );
            CrownBounds::Dense(seed)
        } else if use_patches_seed {
            let (oc, oh, ow) = (output_shape[0], output_shape[1], output_shape[2]);
            debug!(
                "DAG α-CROWN: Initializing Patches mode (output {}x{}x{})",
                oc, oh, ow
            );
            let shape = (oc, oh, ow);
            CrownBounds::Patches(Box::new(PatchesLinearBounds::try_identity_with_deadline(
                shape, shape, deadline, 0,
            )?))
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
        let output_dim = exact_seed_rows
            .unwrap_or_else(|| margin_subset.as_deref().map_or(output_dim, <[usize]>::len));

        // #spec-axis-alpha: this walk's rows ARE output specs — dense seeds
        // carry the subset map so carrier row j resolves to its ORIGINAL
        // output row (a k-row subset seed makes j a compact index, never a
        // spec id). The Patches seed keeps δ off until slice 3 wires the 7-D
        // arm; refresh walks never construct this context at all.
        let row_scope = if exact_seed_rows.is_some() || use_patches_seed {
            AlphaRowScope::Shared
        } else {
            AlphaRowScope::OutputSpecs {
                subset: margin_subset
                    .clone()
                    .map(|indices| std::sync::Arc::from(&indices[..])),
            }
        };

        // Track if we've accumulated bounds to the input
        let mut input_accumulated = false;

        // We need a mutable reference to the intermediate that persists across the loop,
        // so rebind it here.
        let mut intermediate = intermediate;

        // #iter0-alpha-parity (dark, NY_ITER0_PARITY_TRACE=1, print-only):
        // claim one walk id per backward walk so this fold's per-node lines
        // separate from the baseline fold's in an interleaved log.
        let parity_trace = crate::iter0_parity_trace::iter0_parity_trace_enabled()
            .then(crate::iter0_parity_trace::next_walk_id);
        // #patches-drop (dark, NY_PATCHES_CARRIER_TRACE=1, print-only): publish
        // this walk's position so a `[patches-drop]` line emitted deep inside
        // the materializer names the node whose carrier densified.
        let carrier_trace = crate::patches_carrier_trace::enabled();

        // Backward pass through nodes in reverse order.
        // Use enumeration to get direct Vec indices, avoiding HashMap lookups.
        for (rev_pos, node_name) in exec_order.iter().rev().enumerate() {
            let idx = exec_order.len() - 1 - rev_pos;
            let node = nodes_by_idx[idx];
            // A malformed/error suffix receipt poisons GPU acceleration for
            // this alpha state. Later nodes must use a genuine CPU engine path,
            // not the general methods of the backend that just refused.
            let effective_engine = if gpu_suffix::gpu_suffix_runtime_refused(alpha_state) {
                None
            } else {
                engine
            };

            // Get this node's accumulated CrownBounds via direct index.
            let mut node_cb = match node_crown_bounds.take_by_idx_with_deadline(idx, deadline)? {
                Some(cb) => cb,
                None => {
                    // Node has no consumers (not output, not used by anyone)
                    continue;
                }
            };
            if let Some(walk) = parity_trace {
                crate::iter0_parity_trace::trace_node(
                    walk,
                    "dag-alpha",
                    node_name,
                    node.layer.layer_type(),
                    &node_cb,
                );
            }
            if carrier_trace {
                crate::patches_carrier_trace::enter_node("dag-alpha", node_name);
            }

            // #3813 Slice 5: Dense→Patches re-entry at Conv2d boundaries.
            // Mirror of propagation.rs:455-490 for alpha-CROWN.
            // When classifier-head logits reach a Conv2d node through Dense rows
            // (after Linear/Flatten), convert to Patches so the Conv2d/ReLU Patches
            // path handles the CNN trunk with alpha optimization preserved.
            // Gated by use_patches_mode: matrix mode skips re-entry (abcrown.py:228-231).
            try_dense_spatial_patches_reentry_with_deadline(
                &mut node_cb,
                node,
                node_name,
                node_bounds,
                self.use_patches_mode,
                "DAG α-CROWN",
                deadline,
            );
            // A verifier deadline on an ordinary Dense seed retains the
            // historical entry/post-checked route. Strict finite closure is
            // authorized only when this node really entered through Patches
            // and would otherwise cross back into legacy Dense work.
            let finite_structured_boundary =
                matches!(&node_cb, CrownBounds::Patches(_)) && deadline.is_some();

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

            let nonlinear_result = match handle_nonlinear_node(
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
                    engine: effective_engine,
                    deadline,
                    finite_structured_boundary,
                    forbid_full_output_fallback: exact_seed_rows.is_some(),
                    row_scope: &row_scope,
                },
            ) {
                Ok(result) => result,
                Err(error) => match classify_patches_resource_error(&error) {
                    PatchesResourceErrorDisposition::FullCrownFallback => {
                        debug!(
                            "DAG α-CROWN: nonlinear accumulation/materialization hit the \
                             memory budget at {}: {}; falling back to CROWN",
                            node_name, error
                        );
                        let bounds = full_output_crown_fallback_or_refuse(
                            exact_seed_rows.is_some(),
                            || {
                                self.propagate_crown_with_engine_and_deadline(
                                    input,
                                    effective_engine,
                                    deadline,
                                )
                                .map(|result| result.bounds)
                            },
                        )?;
                        return Ok((bounds, None, false, None, Vec::new()));
                    }
                    PatchesResourceErrorDisposition::AtomicReturn
                    | PatchesResourceErrorDisposition::DenseRetry => return Err(error),
                },
            };
            node_cb = match nonlinear_result {
                NonlinearNodeResult::NotHandled(node_cb) => *node_cb,
                NonlinearNodeResult::Continue => continue,
                NonlinearNodeResult::ReturnBounds(bounds) => {
                    return Ok((bounds, None, false, None, Vec::new()));
                }
            };

            let is_patches = matches!(&node_cb, CrownBounds::Patches(_));

            // === Non-ReLU Patches fast-path (#3293) ===
            if is_patches && node.inputs.len() == 1 {
                let memory_fallback_details = match crown_backward_step_patches(
                    &node.layer,
                    &mut node_cb,
                    pre_activation,
                    effective_engine,
                    0,
                    "DAG-α-CROWN",
                    deadline,
                ) {
                    Ok(CrownStepResult::Continue) => {
                        match self.accumulate_crown_bounds_to_input_with_deadline(
                            first_input,
                            node_cb,
                            &mut node_crown_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                            deadline,
                        ) {
                            Ok(()) => continue,
                            Err(error) => match classify_patches_resource_error(&error) {
                                PatchesResourceErrorDisposition::AtomicReturn
                                | PatchesResourceErrorDisposition::DenseRetry => {
                                    return Err(error);
                                }
                                PatchesResourceErrorDisposition::FullCrownFallback => {
                                    debug!(
                                        "DAG α-CROWN: Patches accumulation hit the memory \
                                         budget at {}: {}; falling back to CROWN",
                                        node_name, error
                                    );
                                    let bounds = full_output_crown_fallback_or_refuse(
                                        exact_seed_rows.is_some(),
                                        || {
                                            self.propagate_crown_with_engine_and_deadline(
                                                input,
                                                effective_engine,
                                                deadline,
                                            )
                                            .map(|result| result.bounds)
                                        },
                                    )?;
                                    return Ok((bounds, None, false, None, Vec::new()));
                                }
                            },
                        }
                    }
                    Ok(CrownStepResult::IbpFallback(fallback)) => {
                        if fallback.reason
                            == crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded
                        {
                            Some(fallback.details)
                        } else {
                            // Fall through to Dense dispatch below.
                            None
                        }
                    }
                    Err(error) => match classify_patches_resource_error(&error) {
                        PatchesResourceErrorDisposition::AtomicReturn => return Err(error),
                        PatchesResourceErrorDisposition::FullCrownFallback => {
                            Some(error.to_string())
                        }
                        PatchesResourceErrorDisposition::DenseRetry => None,
                    },
                };
                if let Some(details) = memory_fallback_details {
                    debug!(
                        "DAG α-CROWN: Patches dispatch hit memory budget at {}: {}; falling back to CROWN",
                        node_name, details
                    );
                    let bounds =
                        full_output_crown_fallback_or_refuse(exact_seed_rows.is_some(), || {
                            self.propagate_crown_with_engine_and_deadline(
                                input,
                                effective_engine,
                                deadline,
                            )
                            .map(|result| result.bounds)
                        })?;
                    return Ok((bounds, None, false, None, Vec::new()));
                }
                // Ensure Dense for below.
                if matches!(&node_cb, CrownBounds::Patches(_)) {
                    match node_cb.ensure_dense_with_deadline(deadline) {
                        Ok(_) => {}
                        Err(error @ NyError::DeadlineExceeded(_)) => return Err(error),
                        Err(e) => {
                            debug!(
                                "DAG α-CROWN: ensure_dense failed at {}: {}, CROWN fallback",
                                node_name, e
                            );
                            let bounds = full_output_crown_fallback_or_refuse(
                                exact_seed_rows.is_some(),
                                || {
                                    self.propagate_crown_with_engine_and_deadline(
                                        input,
                                        effective_engine,
                                        deadline,
                                    )
                                    .map(|result| result.bounds)
                                },
                            )?;
                            return Ok((bounds, None, false, None, Vec::new()));
                        }
                    }
                }
            }

            // === Patches residual passthrough for Add/Sub (#4382) ===
            match crate::network::core::graph::backward_helpers::try_apply_patches_residual_passthrough_with_deadline(
                self,
                node,
                &mut node_cb,
                node_bounds,
                &mut node_crown_bounds,
                output_dim,
                input_dim,
                &mut input_accumulated,
                "DAG-α-CROWN",
                deadline,
            ) {
                Ok(true) => continue,
                Ok(false) => {}
                Err(error) => match classify_patches_resource_error(&error) {
                    PatchesResourceErrorDisposition::FullCrownFallback => {
                        debug!(
                            "DAG α-CROWN: Patches residual merge hit the memory budget at {}: \
                             {}; falling back to CROWN",
                            node_name, error
                        );
                        let bounds = full_output_crown_fallback_or_refuse(
                            exact_seed_rows.is_some(),
                            || {
                                self.propagate_crown_with_engine_and_deadline(
                                    input,
                                    effective_engine,
                                    deadline,
                                )
                                .map(|result| result.bounds)
                            },
                        )?;
                        return Ok((bounds, None, false, None, Vec::new()));
                    }
                    PatchesResourceErrorDisposition::AtomicReturn
                    | PatchesResourceErrorDisposition::DenseRetry => return Err(error),
                },
            }

            // Every cooperative finite Patches operator above either
            // accumulated and continued or returned a typed fallback. The
            // remaining Dense/INVPROP/shared-dispatch path still contains
            // unpollable transforms, so do not densify or scan under the same
            // absolute deadline.
            if finite_structured_boundary {
                let bounds =
                    full_output_crown_fallback_or_refuse(exact_seed_rows.is_some(), || {
                        self.propagate_crown_with_engine_and_deadline(
                            input,
                            effective_engine,
                            deadline,
                        )
                        .map(|result| result.bounds)
                    })?;
                return Ok((bounds, None, false, None, Vec::new()));
            }

            // === Dense dispatch ===
            if let Err(error) = node_cb.ensure_dense_with_deadline(deadline) {
                match classify_patches_resource_error(&error) {
                    PatchesResourceErrorDisposition::FullCrownFallback => {
                        debug!(
                            "DAG α-CROWN: Dense boundary hit the memory budget at {}: {}; \
                             falling back to CROWN",
                            node_name, error
                        );
                        let bounds = full_output_crown_fallback_or_refuse(
                            exact_seed_rows.is_some(),
                            || {
                                self.propagate_crown_with_engine_and_deadline(
                                    input,
                                    effective_engine,
                                    deadline,
                                )
                                .map(|result| result.bounds)
                            },
                        )?;
                        return Ok((bounds, None, false, None, Vec::new()));
                    }
                    PatchesResourceErrorDisposition::AtomicReturn
                    | PatchesResourceErrorDisposition::DenseRetry => return Err(error),
                }
            }
            let CrownBounds::Dense(node_lb) = node_cb else {
                unreachable!("successful Dense-boundary preparation must publish Dense")
            };
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
            // INVPROP disables it for both treatment arms: this suffix returns
            // only repaired concrete bounds, so it cannot preserve a certified
            // pre-repair inversion or distinguish a completed conditioned fold
            // from a fallback. Keeping OFF/ON on the same CPU route also avoids
            // a hardware-path confound in treatment evidence.
            let gpu_suffix_bounds = if intermediate.is_none() && invprop_state.is_none() {
                self.try_alpha_backward_gpu_suffix(
                    input,
                    &node_lb,
                    node_name,
                    node_bounds,
                    alpha_state,
                    effective_engine,
                    deadline,
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
                return Ok((concrete_bounds, None, false, None, Vec::new()));
            }

            // The attempt above may have poisoned the route for this same
            // node. Re-read before constructing the CPU continuation context.
            let effective_engine = if gpu_suffix::gpu_suffix_runtime_refused(alpha_state) {
                None
            } else {
                effective_engine
            };

            let ctx = DispatchContext {
                node_name,
                layer: &node.layer,
                inputs: &node.inputs,
                pre_activation,
                network_input: input,
                node_bounds: node_bounds.into(),
                engine: effective_engine,
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
                Err(e @ NyError::DeadlineExceeded(_)) => return Err(e),
                Err(
                    e @ NyError::ShapeMismatch { .. }
                    | e @ NyError::UnsupportedOp(_)
                    | e @ NyError::UnsupportedConfiguration(_)
                    | e @ NyError::NumericalInstability(_)
                    | e @ NyError::CpuMemoryExceeded { .. },
                ) => {
                    debug!(
                        "DAG α-CROWN: dispatch error at {} ({}): {}, falling back to CROWN",
                        node_name,
                        node.layer.layer_type(),
                        e,
                    );
                    let bounds =
                        full_output_crown_fallback_or_refuse(exact_seed_rows.is_some(), || {
                            self.propagate_crown_with_engine_and_deadline(
                                input,
                                effective_engine,
                                deadline,
                            )
                            .map(|result| result.bounds)
                        })?;
                    return Ok((bounds, None, false, None, Vec::new()));
                }
                Err(e) => return Err(e),
            };

            match apply_dense_backward_dispatch_result_with_deadline(
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
                deadline,
            ) {
                Ok(()) => {}
                Err(NyError::UnsupportedOp(reason)) => {
                    debug!(
                        "DAG α-CROWN: Unsupported layer {} ({}): {}, falling back to CROWN",
                        node_name,
                        node.layer.layer_type(),
                        reason,
                    );
                    let bounds =
                        full_output_crown_fallback_or_refuse(exact_seed_rows.is_some(), || {
                            self.propagate_crown_with_engine_and_deadline(
                                input,
                                effective_engine,
                                deadline,
                            )
                            .map(|result| result.bounds)
                        })?;
                    return Ok((bounds, None, false, None, Vec::new()));
                }
                Err(error @ NyError::CpuMemoryExceeded { .. }) => {
                    debug!(
                        "DAG α-CROWN: dispatch merge hit the memory budget at {} ({}): {}; \
                         falling back to CROWN",
                        node_name,
                        node.layer.layer_type(),
                        error
                    );
                    let bounds =
                        full_output_crown_fallback_or_refuse(exact_seed_rows.is_some(), || {
                            self.propagate_crown_with_engine_and_deadline(
                                input,
                                effective_engine,
                                deadline,
                            )
                            .map(|result| result.bounds)
                        })?;
                    return Ok((bounds, None, false, None, Vec::new()));
                }
                Err(e) => return Err(e),
            }
        }

        // Concretize final bounds.
        let final_effective_engine = if gpu_suffix::gpu_suffix_runtime_refused(alpha_state) {
            None
        } else {
            engine
        };
        let mut final_cb = node_crown_bounds
            .take_with_deadline(NETWORK_INPUT, deadline)?
            .ok_or_else(|| {
                if input_accumulated {
                    NyError::InvalidSpec(
                        "DAG α-CROWN: Input bounds accumulated but not found".to_string(),
                    )
                } else {
                    NyError::InvalidSpec("DAG α-CROWN: No path from output to input".to_string())
                }
            })?;
        if let Err(error) = final_cb.ensure_dense_with_deadline_for_purpose(
            deadline,
            PatchesMaterializationPurpose::NetworkInputTerminal,
        ) {
            match classify_patches_resource_error(&error) {
                PatchesResourceErrorDisposition::FullCrownFallback => {
                    debug!(
                        "DAG α-CROWN: final Patches materialization hit the memory budget: {}; \
                         falling back to CROWN",
                        error
                    );
                    let bounds =
                        full_output_crown_fallback_or_refuse(exact_seed_rows.is_some(), || {
                            self.propagate_crown_with_engine_and_deadline(
                                input,
                                final_effective_engine,
                                deadline,
                            )
                            .map(|result| result.bounds)
                        })?;
                    return Ok((bounds, None, false, None, Vec::new()));
                }
                PatchesResourceErrorDisposition::AtomicReturn
                | PatchesResourceErrorDisposition::DenseRetry => return Err(error),
            }
        }
        let CrownBounds::Dense(input_lb) = final_cb else {
            unreachable!("successful final-bound preparation must publish Dense")
        };

        // Typed pre-repair row provenance allocates/scans one metadata entry
        // per output. It is necessary only when a caller supplied an active
        // INVPROP seed that can authorize a proof or steer gamma. Ordinary,
        // OFF-control, and zero-gamma recovery folds keep the allocation-free
        // public concretization hot path.
        let retained_intermediate_bytes = intermediate
            .as_deref()
            .map_or(0, GraphAlphaCrownIntermediate::logical_memory_bytes);
        let concretization = if invprop_state.is_some() {
            input_lb
                .concretize_sound_with_infeasibility_deadline_and_resident(
                    input,
                    deadline,
                    retained_intermediate_bytes,
                )
                .map(|concretized| {
                    (
                        concretized.bounds,
                        concretized.certified_finite_inversion,
                        concretized.max_finite_gap,
                        concretized.row_finite_gaps,
                    )
                })
        } else {
            input_lb
                .concretize_sound_with_deadline_and_resident(
                    input,
                    deadline,
                    retained_intermediate_bytes,
                )
                .map(|bounds| (bounds, false, None, Vec::new()))
        };
        let (concrete, certified_finite_inversion, max_finite_gap, row_finite_gaps) =
            match concretization {
                Ok(result) => result,
                Err(error @ NyError::CpuMemoryExceeded { .. }) => {
                    debug!(
                        "DAG α-CROWN: final concretization hit the memory budget: {}; \
                         falling back to CROWN",
                        error
                    );
                    let bounds =
                        full_output_crown_fallback_or_refuse(exact_seed_rows.is_some(), || {
                            self.propagate_crown_with_engine_and_deadline(
                                input,
                                final_effective_engine,
                                deadline,
                            )
                            .map(|result| result.bounds)
                        })?;
                    return Ok((bounds, None, false, None, Vec::new()));
                }
                Err(error) => return Err(error),
            };
        if let Some(ref mut inter) = intermediate {
            let retained_bytes = inter.logical_memory_bytes();
            inter.final_bounds = match input_lb.try_clone_with_deadline(deadline, retained_bytes) {
                Ok(bounds) => bounds,
                Err(error @ NyError::DeadlineExceeded(_)) => return Err(error),
                Err(error @ NyError::CpuMemoryExceeded { .. }) => {
                    debug!(
                        "DAG α-CROWN: final intermediate retention hit the memory budget: {}; \
                         falling back to CROWN",
                        error
                    );
                    let bounds =
                        full_output_crown_fallback_or_refuse(exact_seed_rows.is_some(), || {
                            self.propagate_crown_with_engine_and_deadline(
                                input,
                                final_effective_engine,
                                deadline,
                            )
                            .map(|result| result.bounds)
                        })?;
                    return Ok((bounds, None, false, None, Vec::new()));
                }
                Err(error) => return Err(error),
            };
        }
        // Concretization borrowed the relation; return that exact owned carrier
        // instead of making a second unbounded clone.
        let final_lb = input_lb;
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
        Ok((
            concrete,
            Some(final_lb),
            certified_finite_inversion,
            max_finite_gap,
            row_finite_gaps,
        ))
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod true_grad_oracle_tests;
