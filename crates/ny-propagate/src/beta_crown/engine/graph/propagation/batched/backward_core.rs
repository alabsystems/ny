// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared backward traversal core for batched CROWN propagation.
//!
//! This module contains the layer-dispatch match logic that is shared between
//! the standard backward pass and the lA-capture variant. Extracting this
//! eliminates ~400 LOC of duplication between the two kernels.
//!
//! # Reference
//! - alpha-beta-CROWN: `auto_LiRPA/bound_general.py` (backward pass)
//! - Design: designs/2026-02-09-code-structure-wave2-graph-engine-split.md (Step 4.3)

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::Array2;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::layers::common::BoundPropagation;
use crate::network::backward_dispatch::{
    dispatch_backward_layer, BackwardDispatchResult, DispatchContext,
};
use crate::network::{backward_div_to_numerator, DivBackwardResult, GraphNetwork};
use crate::{Layer, LinearBounds, MulBinaryRelaxationMode, NETWORK_INPUT};

use super::backward_stack;
use super::indexed_pending::IndexedPendingLinearBounds;

/// Runtime gate for the #lsnc-relu STEP 2 DOMAIN-batched ReLU backward.
///
/// `-1` = uninitialized (read the env once, then cache); `1` = ON; `0` = OFF.
///
/// Default is OFF (the SAFE per-domain loop). The batched path is proven BIT-IDENTICAL
/// to the per-domain loop (see `propagate_linear_multi_domain_relu` and the parity
/// tests in `tests_soundness.rs`, which assert exact f32 equality of coefficients,
/// biases, AND certified error), so enabling it can never change a certified bound —
/// but its measured END-TO-END win is small (~2%: the ReLU op is ~1.1–1.25x faster in
/// isolation, yet is only a minor slice of the input-split backward; the per-domain
/// f64 certified-error Linear backward `aw_f64_with_abssum` remains the ~75% hotspot).
/// A ~2% win does not justify flipping the global ReLU path on a SOUNDNESS-CRITICAL
/// verifier, so it is opt-in: set `NY_INPUT_SPLIT_BATCHED_RELU=1` to enable (the A/B
/// override, mirroring `NY_INPUT_SPLIT_SHARED_FWD` / `NY_BATCHED_NAIVE_ENGINE`).
static BATCHED_RELU_MODE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// Whether the domain-batched ReLU backward is enabled (see [`BATCHED_RELU_MODE`]).
pub(super) fn input_split_batched_relu_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match BATCHED_RELU_MODE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = matches!(
                std::env::var("NY_INPUT_SPLIT_BATCHED_RELU").ok().as_deref(),
                Some("1")
            );
            BATCHED_RELU_MODE.store(i8::from(on), Ordering::Relaxed);
            on
        }
    }
}

/// Test-only runtime override for the batched-ReLU gate: `Some(true|false)` forces
/// ON/OFF, `None` restores the env-derived default. Lets the parity + throughput
/// micro-benchmark A/B the exact same pipeline without mutating process-global env.
#[cfg(test)]
pub(crate) fn force_batched_relu(mode: Option<bool>) {
    use std::sync::atomic::Ordering;
    let v = match mode {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    };
    BATCHED_RELU_MODE.store(v, Ordering::Relaxed);
}

/// Process a single node's backward pass for all domains in the batch.
///
/// This is the shared layer-dispatch core extracted from both
/// `propagate_crown_batched_backward_internal` and
/// `propagate_crown_batched_backward_internal_with_la`.
///
/// # Arguments
/// * `node_name` - Name of the current node being processed
/// * `node` - The graph node reference
/// * `node_lbs` - Per-domain linear bounds at this node (consumed)
/// * `constrained_inputs` - Per-domain constrained input bounds
/// * `bounds_caches` - Per-domain IBP bounds caches
/// * `beta_states` - Per-domain β parameters
/// * `alpha_states` - Per-domain α parameters
/// * `node_linear_bounds` - Indexed pending bounds carrier (updated in place)
/// * `n_domains` - Batch size
/// * `engine` - GPU compute engine
/// * `deadline` - Per-node deadline for intra-kernel timeout (#3795)
/// * `stack_domains` - Domain-stack conv/BN backwards into one dispatch call
///   (#cgan-batched-stack; `false` = historical per-domain loop, bit-identical)
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_node_backward(
    node_name: &str,
    node: &crate::GraphNode,
    node_lbs: Vec<Option<LinearBounds>>,
    constrained_inputs: &[BoundedTensor],
    // #lsnc-shared-fwd: borrowed per-domain node-bounds caches (slice of refs).
    // In the input-split lane every element aliases ONE shared warmup map, so
    // there is no per-domain HashMap deep clone. Read-only throughout.
    bounds_caches: &[&HashMap<String, Arc<BoundedTensor>>],
    beta_states: &[Option<&GraphBetaState>],
    alpha_states: &[Option<&GraphDomainAlphaState>],
    node_linear_bounds: &mut IndexedPendingLinearBounds,
    n_domains: usize,
    network_input_dim: usize,
    engine: &dyn GemmEngine,
    deadline: Option<std::time::Instant>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    stack_domains: bool,
) -> Result<()> {
    // Structural invariant: all parallel arrays must match n_domains. (#2824, #2637)
    // Runtime check — debug_assert alone is compiled out in release builds,
    // leaving 15+ index sites unguarded against length mismatches.
    if constrained_inputs.len() != n_domains
        || bounds_caches.len() != n_domains
        || beta_states.len() != n_domains
        || alpha_states.len() != n_domains
        || node_lbs.len() != n_domains
    {
        return Err(NyError::InternalError(format!(
            "dispatch_node_backward: parallel array length mismatch (n_domains={n_domains}): \
             constrained_inputs={}, bounds_caches={}, beta_states={}, alpha_states={}, node_lbs={}",
            constrained_inputs.len(),
            bounds_caches.len(),
            beta_states.len(),
            alpha_states.len(),
            node_lbs.len(),
        )));
    }

    // Helper: validate minimum input count for the current layer.
    let require_inputs = |min: usize| -> Result<()> {
        if node.inputs.len() < min {
            Err(NyError::InvalidSpec(format!(
                "Node '{}' ({}) requires at least {} input(s), got {}",
                node_name,
                node.layer.layer_type(),
                min,
                node.inputs.len(),
            )))
        } else {
            Ok(())
        }
    };
    let require_unary_input = || -> Result<&str> {
        node.require_unary_input().map_err(|_| {
            NyError::InvalidSpec(format!(
                "Node '{}' ({}) has no inputs for CROWN backward propagation",
                node_name,
                node.layer.layer_type()
            ))
        })
    };
    let require_binary_input_names = || -> Result<(&str, &str)> {
        node.require_binary_inputs().map_err(|_| {
            NyError::InvalidSpec(format!(
                "Node '{}' ({}) requires 2 inputs for CROWN backward propagation but has {}",
                node_name,
                node.layer.layer_type(),
                node.inputs.len()
            ))
        })
    };

    if let Layer::Linear(l) = &node.layer {
        require_inputs(1)?;
        let first_input = require_unary_input()?;

        // Collect all domains that have bounds at this node
        let mut active_indices: Vec<usize> = Vec::with_capacity(n_domains);
        let mut active_bounds: Vec<&LinearBounds> = Vec::with_capacity(n_domains);

        for (i, lb_opt) in node_lbs.iter().enumerate() {
            if let Some(lb) = lb_opt {
                active_indices.push(i);
                active_bounds.push(lb);
            }
        }

        if active_bounds.is_empty() {
            return Ok(());
        }

        // BATCHED GPU GEMM: Process all domains in one kernel call
        let new_bounds = l.propagate_linear_batched_with_engine(&active_bounds, engine)?;

        // Distribute results back to domains
        for (result_idx, &domain_idx) in active_indices.iter().enumerate() {
            node_linear_bounds.accumulate_name(
                first_input,
                new_bounds[result_idx].clone(),
                domain_idx,
            )?;
        }
        return Ok(());
    }

    if let Layer::ReLU(r) = &node.layer {
        require_inputs(1)?;
        let first_input = require_unary_input()?;

        // #lsnc-relu STEP 2: DOMAIN-batched ReLU backward. Vectorizes the per-domain
        // loop below into one batched pass (each sub-box carries its own box-dependent
        // triangle slope, so a per-domain `[num_neurons]` relaxation is applied to that
        // domain's `[num_outputs, num_neurons]` coefficient block). BIT-IDENTICAL to the
        // scalar loop — `propagate_linear_multi_domain_relu` returns `None` (decline) for
        // any domain outside its clean fast-path class, and the scalar loop then runs.
        if input_split_batched_relu_enabled() {
            let mut active_idx: Vec<usize> = Vec::with_capacity(n_domains);
            let mut active_bounds: Vec<&LinearBounds> = Vec::with_capacity(n_domains);
            let mut active_pre: Vec<&BoundedTensor> = Vec::with_capacity(n_domains);
            let mut active_alpha: Vec<Option<(ndarray::Array1<f32>, ndarray::Array1<f32>)>> =
                Vec::with_capacity(n_domains);
            for (domain_idx, lb_opt) in node_lbs.iter().enumerate() {
                let Some(lb) = lb_opt else { continue };
                let pre_activation = resolve_pre_activation(
                    &node.inputs,
                    &constrained_inputs[domain_idx],
                    bounds_caches[domain_idx],
                )?;
                // Same α source as the scalar arm: optimized α when present and
                // non-empty (#1841), else `None` = heuristic (α = 1 if u > -l, else 0).
                let alpha = match alpha_states[domain_idx] {
                    Some(alpha_state) if !alpha_state.is_empty() => {
                        let lower = alpha_state.build_alpha_array(node_name, pre_activation);
                        let upper = alpha_state.build_alpha_upper_array(node_name, pre_activation);
                        Some((lower, upper))
                    }
                    _ => None,
                };
                active_idx.push(domain_idx);
                active_bounds.push(lb);
                active_pre.push(pre_activation);
                active_alpha.push(alpha);
            }

            if let Some(results) =
                r.propagate_linear_multi_domain_relu(&active_bounds, &active_pre, &active_alpha)?
            {
                for (k, mut new_lb) in results.into_iter().enumerate() {
                    let domain_idx = active_idx[k];
                    // Same post-processing as the scalar arm, per domain: eager per-row
                    // discharge of the carried coefficient error over the pre-activation
                    // cut, then the β contribution.
                    new_lb.fold_coeff_err_over_box_eager(active_pre[k]);
                    apply_beta_contribution(node_name, beta_states[domain_idx], &mut new_lb);
                    node_linear_bounds.accumulate_name(first_input, new_lb, domain_idx)?;
                }
                return Ok(());
            }
            // DECLINED (non-contiguous / non-finite / shape-mismatched pre-activation):
            // fall through to the byte-identical scalar loop.
        }

        // Process ReLU for each domain (different pre-activation bounds).
        // Uses optimized alpha values when available (#1841), falling back
        // to the heuristic (alpha = 1 if u > -l, else 0).
        for (domain_idx, lb_opt) in node_lbs.into_iter().enumerate() {
            let Some(lb) = lb_opt else { continue };

            let pre_activation = resolve_pre_activation(
                &node.inputs,
                &constrained_inputs[domain_idx],
                bounds_caches[domain_idx],
            )?;

            let new_lb = relu_backward_domain(
                node_name,
                r,
                lb,
                pre_activation,
                alpha_states[domain_idx],
                beta_states[domain_idx],
            )?;

            node_linear_bounds.accumulate_name(first_input, new_lb, domain_idx)?;
        }
        return Ok(());
    }

    // alpha-beta-CROWN normalizes Div to reciprocal-plus-multiply during graph
    // optimization, so the batched graph-CROWN path reuses the reciprocal
    // scaling helper instead of routing Div through the unary shared dispatch.
    if matches!(&node.layer, Layer::Div(_)) {
        require_inputs(2)?;
        let (input_a_name, input_b_name) = require_binary_input_names()?;

        for (domain_idx, lb_opt) in node_lbs.into_iter().enumerate() {
            let Some(lb) = lb_opt else { continue };
            div_backward_domain(
                node_name,
                input_a_name,
                input_b_name,
                lb,
                &constrained_inputs[domain_idx],
                bounds_caches[domain_idx],
                network_input_dim,
                &mut |input_name, bounds| {
                    node_linear_bounds.accumulate_name(input_name, bounds, domain_idx)
                },
            )?;
        }
        return Ok(());
    }

    // #cgan-batched-stack: domain-stack the conv/BN backward into ONE dispatch
    // call across all active domains (row-independent linear operators; hulled
    // boxes for the sound-widening bias folds — see backward_stack module docs).
    // Fail-closed: `Ok(false)` means nothing was accumulated and the per-domain
    // loop below runs unchanged.
    if stack_domains && backward_stack::layer_supports_domain_stacking(&node.layer) {
        let handled = backward_stack::try_stacked_dispatch(
            node_name,
            node,
            &node_lbs,
            constrained_inputs,
            bounds_caches,
            node_linear_bounds,
            engine,
            deadline,
            mul_binary_alphas,
        )?;
        if handled {
            return Ok(());
        }
    }

    // All non-Linear/ReLU layers now route through the shared dispatch core.
    // Input count is validated by each dispatch result arm (Binary requires 2,
    // Nary validates per-input, Single requires at least 1 at accumulation).
    for (domain_idx, lb_opt) in node_lbs.into_iter().enumerate() {
        let Some(lb) = lb_opt else { continue };
        generic_backward_domain(
            node_name,
            node,
            lb,
            &constrained_inputs[domain_idx],
            bounds_caches[domain_idx],
            network_input_dim,
            engine,
            deadline,
            mul_binary_alphas,
            &mut |input_name, bounds| {
                node_linear_bounds.accumulate_name(input_name, bounds, domain_idx)
            },
        )?;
    }

    Ok(())
}

/// One domain of the scalar ReLU backward arm of [`dispatch_node_backward`]:
/// optimized-α selection (#1841) or heuristic kernel, the eager per-row
/// coefficient-error discharge over the per-domain pre-activation cut
/// (#cgan-conv-err-compose), then the β contribution. Extracted verbatim so the
/// SoA batched backward (#lsnc-batched-bwd, `batched_bwd.rs`) runs the
/// IDENTICAL per-domain reference code under coarse rayon chunks — bit-parity
/// by construction.
pub(super) fn relu_backward_domain(
    node_name: &str,
    r: &crate::layers::ReLULayer,
    lb: LinearBounds,
    pre_activation: &BoundedTensor,
    alpha_state: Option<&GraphDomainAlphaState>,
    beta_state: Option<&GraphBetaState>,
) -> Result<LinearBounds> {
    // Use optimized alpha when available, else heuristic
    let mut new_lb = if let Some(alpha_state) = alpha_state {
        if !alpha_state.is_empty() {
            let alphas = alpha_state.build_alpha_array(node_name, pre_activation);
            let alphas_upper = alpha_state.build_alpha_upper_array(node_name, pre_activation);
            let (lb_result, _grad, _grad_upper) =
                r.propagate_linear_with_alpha(&lb, pre_activation, &alphas, Some(&alphas_upper))?;
            lb_result
        } else {
            r.propagate_linear_with_bounds(&lb, pre_activation)?
        }
    } else {
        r.propagate_linear_with_bounds(&lb, pre_activation)?
    };

    // Eager per-row discharge of the carried coefficient error over the
    // pre-activation cut, matching the scalar spec-CROWN driver
    // (#cgan-conv-err-compose, see LinearBounds::fold_coeff_err_over_box_eager
    // — keeps batched↔direct parity bit-consistent).
    new_lb.fold_coeff_err_over_box_eager(pre_activation);

    // Add β contribution if present
    apply_beta_contribution(node_name, beta_state, &mut new_lb);

    Ok(new_lb)
}

/// One domain of the Div backward arm of [`dispatch_node_backward`]. The
/// contribution target is genuinely PER-DOMAIN (numerator propagation vs a
/// concretized bias carrier on `NETWORK_INPUT`, decided by this domain's
/// denominator box), so results flow through the `sink` callback. Extracted
/// verbatim for the SoA batched backward (#lsnc-batched-bwd) — bit-parity by
/// construction.
#[allow(clippy::too_many_arguments)]
pub(super) fn div_backward_domain(
    node_name: &str,
    input_a_name: &str,
    input_b_name: &str,
    lb: LinearBounds,
    constrained_input: &BoundedTensor,
    bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
    network_input_dim: usize,
    sink: &mut dyn FnMut(&str, LinearBounds) -> Result<()>,
) -> Result<()> {
    let input_a_bounds = if input_a_name == NETWORK_INPUT {
        constrained_input
    } else {
        bounds_cache.get(input_a_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Div input A '{input_a_name}' not found at node '{node_name}'",
            ))
        })?
    };
    let input_b_bounds = if input_b_name == NETWORK_INPUT {
        constrained_input
    } else {
        bounds_cache.get(input_b_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Div input B '{input_b_name}' not found at node '{node_name}'",
            ))
        })?
    };
    let node_output_bounds = bounds_cache.get(node_name).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "Div output bounds for '{node_name}' not found during batched CROWN",
        ))
    })?;

    match backward_div_to_numerator(&lb, input_a_bounds, input_b_bounds, node_output_bounds)? {
        DivBackwardResult::PropagateNumerator(bounds) => {
            sink(input_a_name, *bounds)?;
        }
        DivBackwardResult::ConcretizeCurrentNode(bias) => {
            let bias_lb = LinearBounds::new_or_conservative(
                Array2::zeros((bias.lower.len(), network_input_dim)),
                *bias.lower,
                Array2::zeros((bias.upper.len(), network_input_dim)),
                *bias.upper,
            )?;
            sink(NETWORK_INPUT, bias_lb)?;
        }
    }
    Ok(())
}

/// One domain of the shared-dispatch (non-Linear/ReLU/Div) backward arm of
/// [`dispatch_node_backward`]: pre-activation resolution, `DispatchContext`
/// construction, `dispatch_backward_layer`, and every result/error arm.
/// Contributions flow through the `sink` callback. Extracted verbatim for the
/// SoA batched backward (#lsnc-batched-bwd) — bit-parity by construction.
#[allow(clippy::too_many_arguments)]
pub(super) fn generic_backward_domain(
    node_name: &str,
    node: &crate::GraphNode,
    lb: LinearBounds,
    constrained_input: &BoundedTensor,
    bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
    network_input_dim: usize,
    engine: &dyn GemmEngine,
    deadline: Option<std::time::Instant>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    sink: &mut dyn FnMut(&str, LinearBounds) -> Result<()>,
) -> Result<()> {
    let require_inputs = |min: usize| -> Result<()> {
        if node.inputs.len() < min {
            Err(NyError::InvalidSpec(format!(
                "Node '{}' ({}) requires at least {} input(s), got {}",
                node_name,
                node.layer.layer_type(),
                min,
                node.inputs.len(),
            )))
        } else {
            Ok(())
        }
    };
    let require_binary_input_names = || -> Result<(&str, &str)> {
        node.require_binary_inputs().map_err(|_| {
            NyError::InvalidSpec(format!(
                "Node '{}' ({}) requires 2 inputs for CROWN backward propagation but has {}",
                node_name,
                node.layer.layer_type(),
                node.inputs.len()
            ))
        })
    };

    // Only resolve cached pre-activation bounds for layers that actually
    // consume them. Some shared-dispatch cases (e.g., Concat with constant
    // first input) do not need this and may not have a cache entry.
    let pre_activation = if layer_needs_pre_activation_lookup(&node.layer) {
        resolve_pre_activation(&node.inputs, constrained_input, bounds_cache)?
    } else {
        constrained_input
    };

    let ctx = DispatchContext {
        node_name,
        layer: &node.layer,
        inputs: &node.inputs,
        pre_activation,
        network_input: constrained_input,
        node_bounds: bounds_cache.into(),
        engine: Some(engine),
        deadline, // #3795: thread deadline into dispatch
        bilinear_alphas: None,
        mul_binary_relaxation: MulBinaryRelaxationMode::default(),
        mul_binary_alphas, // #4284: thread shared root-level MulBinary alphas
        norm_inv_rms_override: None,
    };

    match dispatch_backward_layer(&ctx, &lb) {
        Ok(result) => match result {
            BackwardDispatchResult::Single(new_lb) => {
                let first_input = node.inputs.first().ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Node '{}' ({}) has no inputs for CROWN backward propagation",
                        node_name,
                        node.layer.layer_type()
                    ))
                })?;
                sink(first_input, *new_lb)?;
            }
            BackwardDispatchResult::Binary {
                bounds_a,
                bounds_b,
                bias_lower,
                bias_upper,
            } => {
                require_inputs(2)?;
                let (input_a_name, input_b_name) = require_binary_input_names()?;
                // Accumulate bias exactly once (#2617)
                // Migrated from from_parts_unchecked: bias from dispatch
                // could carry NaN. Zero A-matrices are always safe. See #3438.
                // Match the flat-path helper and alpha-beta-CROWN's network-input
                // carrier width: the bias channel targets NETWORK_INPUT, not the
                // current branch-local intermediate width.
                let bias_lb = LinearBounds::new_or_conservative(
                    Array2::zeros((bias_lower.len(), network_input_dim)),
                    bias_lower,
                    Array2::zeros((bias_upper.len(), network_input_dim)),
                    bias_upper,
                )?;
                sink(NETWORK_INPUT, bias_lb)?;
                GraphNetwork::verify_split_path_bias_zero(
                    &bounds_a,
                    "Batched dispatch binary lhs split path",
                )?;
                GraphNetwork::verify_split_path_bias_zero(
                    &bounds_b,
                    "Batched dispatch binary rhs split path",
                )?;
                sink(input_a_name, *bounds_a)?;
                sink(input_b_name, *bounds_b)?;
            }
            BackwardDispatchResult::Nary {
                bounds,
                bias_lower,
                bias_upper,
            } => {
                // Accumulate bias exactly once (#2617)
                // Migrated from from_parts_unchecked: bias from dispatch
                // could carry NaN. Zero A-matrices are always safe. See #3438.
                // Concat/split-path bias still lands on NETWORK_INPUT, so use the
                // full network-input width instead of the first branch width.
                let bias_lb = LinearBounds::new_or_conservative(
                    Array2::zeros((bias_lower.len(), network_input_dim)),
                    bias_lower,
                    Array2::zeros((bias_upper.len(), network_input_dim)),
                    bias_upper,
                )?;
                sink(NETWORK_INPUT, bias_lb)?;
                for (input_name, maybe_lb) in node.inputs.iter().zip(bounds) {
                    if let Some(new_lb) = maybe_lb {
                        GraphNetwork::verify_split_path_bias_zero(
                            &new_lb,
                            "Batched dispatch n-ary split path",
                        )?;
                        sink(input_name, new_lb)?;
                    }
                }
            }
            BackwardDispatchResult::PassThrough => {
                let first_input = node.inputs.first().ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Node '{}' ({}) has no inputs for CROWN backward propagation",
                        node_name,
                        node.layer.layer_type()
                    ))
                })?;
                sink(first_input, lb)?;
            }
            BackwardDispatchResult::Unsupported(reason) => {
                return Err(NyError::UnsupportedOp(format!(
                    "Batched CROWN: layer '{}' ({}) unsupported: {}",
                    node_name,
                    node.layer.layer_type(),
                    reason,
                )));
            }
        },
        // #3166: Catch UnsupportedOp and UnsupportedConfiguration.
        // #2888: NumericalInstability also triggers fallback for consistency,
        // though dispatch_backward_layer already converts it to Unsupported.
        Err(
            NyError::UnsupportedOp(msg)
            | NyError::UnsupportedConfiguration(msg)
            | NyError::NumericalInstability(msg),
        ) => {
            return Err(NyError::UnsupportedOp(format!(
                "Batched CROWN backward at node '{}' ({}): {}",
                node_name,
                node.layer.layer_type(),
                msg,
            )));
        }
        Err(err) => {
            return Err(NyError::InvalidSpec(format!(
                "Batched CROWN failed at node '{}' ({}): {}",
                node_name,
                node.layer.layer_type(),
                err
            )));
        }
    }

    Ok(())
}

/// Returns true when shared backward dispatch needs a real pre-activation tensor.
///
/// Most linear dispatch paths do not consume `ctx.pre_activation`, but layers
/// that call `set_input_shape(ctx.pre_activation.shape())` during dispatch
/// need the actual predecessor output shape, not the network input shape.
fn layer_needs_pre_activation_lookup(layer: &Layer) -> bool {
    layer.requires_pre_activation_bounds()
        || matches!(
            layer,
            Layer::Transpose(_)
                | Layer::Tile(_)
                | Layer::Slice(_)
                | Layer::Gather(_)
                | Layer::Conv1d(_)
                | Layer::ConvTranspose1d(_)
                | Layer::Conv2d(_)
                | Layer::ConvTranspose2d(_)
        )
}

/// Resolve pre-activation bounds for a node's first input.
///
/// Returns the appropriate `BoundedTensor` for computing CROWN relaxation:
/// either the constrained input bounds (if the input is `_input`) or the
/// cached IBP bounds from the forward pass.
pub(super) fn resolve_pre_activation<'a>(
    node_inputs: &[String],
    constrained_input: &'a BoundedTensor,
    bounds_cache: &'a HashMap<String, Arc<BoundedTensor>>,
) -> Result<&'a BoundedTensor> {
    let pre_activation_name = node_inputs.first().ok_or_else(|| {
        NyError::InvalidSpec(
            "Batched CROWN pre-activation lookup requires at least one input".to_string(),
        )
    })?;
    if pre_activation_name == NETWORK_INPUT {
        Ok(constrained_input)
    } else {
        bounds_cache
            .get(pre_activation_name)
            .map(|a| a.as_ref())
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Pre-activation bounds for {} not found",
                    pre_activation_name
                ))
            })
    }
}

/// Apply β contribution to linear bounds for Lagrangian optimization.
///
/// When β state is present for the current node, adjusts the lower/upper
/// coefficient matrices by subtracting/adding signed β for each constrained
/// neuron.
fn apply_beta_contribution(
    node_name: &str,
    beta_state: Option<&GraphBetaState>,
    new_lb: &mut LinearBounds,
) {
    let Some(beta_state) = beta_state else {
        return;
    };
    // Part of #2936: use indexed per-node iteration instead of O(B) full scan.
    for entry in beta_state.entries_for_node(node_name) {
        let j = entry.neuron_idx;
        if j >= new_lb.num_inputs() {
            continue;
        }
        let signed_beta = entry.signed_value();
        // #2415/#2826: non-finite beta must be rejected before abs-threshold filtering.
        if !signed_beta.is_finite() {
            tracing::warn!(
                node_name,
                neuron_idx = j,
                signed_beta,
                "Skipping non-finite beta contribution in batched graph backward"
            );
            continue;
        }
        if signed_beta.abs() < 1e-10 {
            continue;
        }

        // #vnncomp-aw-soundness: fold the f32 rounding of the β split mutation
        // (a - β / a + β) into the certified coefficient error when present,
        // so the certificate cannot under-count |stored_f32 - true_coeff|
        // (false-proof risk; mirrors the conv f32/err fix in becc501).
        new_lb.apply_beta_split_to_column(j, signed_beta);
    }
}

/// Accumulate CROWN linear bounds during backward propagation.
///
/// When multiple predecessor nodes contribute bounds to the same input node,
/// they are accumulated (summed) per the CROWN framework's linear relaxation.
/// Uses NaN-safe addition to prevent INF-cancellation NaN from corrupting
/// the accumulation chain.
///
/// # Reference
/// - CROWN (Zhang et al., 2018): linear bound accumulation at merge points
#[cfg(test)]
pub(super) fn accumulate_crown_bounds_batched(
    input_name: &str,
    new_lb: LinearBounds,
    node_linear_bounds: &mut HashMap<String, Vec<Option<LinearBounds>>>,
    input_accumulated: &mut bool,
    domain_idx: usize,
    n_domains: usize,
) {
    if input_name == NETWORK_INPUT {
        *input_accumulated = true;
    }

    let entry = node_linear_bounds
        .entry(input_name.to_string())
        .or_insert_with(|| vec![None; n_domains]);

    if let Some(existing) = &mut entry[domain_idx] {
        // Accumulate: add the new bounds to existing (NaN-safe)
        let new_la = GraphNetwork::safe_add(existing.lower_a(), new_lb.lower_a(), true);
        let new_lb_val = GraphNetwork::safe_add(existing.lower_b(), new_lb.lower_b(), true);
        let new_ua = GraphNetwork::safe_add(existing.upper_a(), new_lb.upper_a(), false);
        let new_ub = GraphNetwork::safe_add(existing.upper_b(), new_lb.upper_b(), false);
        *existing.lower_a_mut() = new_la;
        *existing.lower_b_mut() = new_lb_val;
        *existing.upper_a_mut() = new_ua;
        *existing.upper_b_mut() = new_ub;
    } else {
        entry[domain_idx] = Some(new_lb);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use ndarray::{arr1, arr2, Array1};

    use crate::layers::{Layer, LinearLayer};

    use super::super::indexed_pending::IndexedPendingLinearBounds;
    use super::{apply_beta_contribution, resolve_pre_activation};
    use crate::beta_crown::state::{GraphBetaEntry, GraphBetaState};
    use crate::{BoundedTensor, GraphNetwork, GraphNode, LinearBounds};

    fn make_pending(n_domains: usize) -> IndexedPendingLinearBounds {
        let mut graph = GraphNetwork::new();
        graph
            .try_add_node(GraphNode::from_input(
                "relu1",
                Layer::Linear(
                    LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32])))
                        .expect("linear layer"),
                ),
            ))
            .expect("node should add");
        graph.set_output("relu1");
        let plan = graph.dispatch_plan().expect("dispatch plan should build");
        IndexedPendingLinearBounds::new(plan, n_domains)
    }

    #[test]
    fn test_resolve_pre_activation_rejects_empty_inputs_2112() {
        let constrained_input =
            BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
                .expect("valid constrained input");
        let bounds_cache: HashMap<String, Arc<BoundedTensor>> = HashMap::new();

        let err = resolve_pre_activation(&[], &constrained_input, &bounds_cache)
            .expect_err("empty input list must return InvalidSpec");
        let msg = err.to_string();

        assert!(
            msg.contains("requires at least one input"),
            "expected explicit empty-input diagnostic, got: {msg}"
        );
    }

    #[test]
    fn test_apply_beta_contribution_skips_non_finite_signed_beta_2826() {
        for sign in [1.0_f32, -1.0_f32] {
            let beta_state = GraphBetaState {
                entries: vec![GraphBetaEntry {
                    node_name: "relu1".to_string(),
                    neuron_idx: 1,
                    split_point: 0.0,
                    value: f32::INFINITY,
                    sign,
                    grad: 0.0,
                    m: 0.0,
                    v: 0.0,
                    v_max: 0.0,
                }],
                ..GraphBetaState::empty()
            };

            let mut bounds = LinearBounds {
                lower_a: arr2(&[[1.0, -2.0]]),
                lower_b: Array1::zeros(1),
                upper_a: arr2(&[[3.0, 4.0]]),
                upper_b: Array1::zeros(1),
                lower_a_err: None,
                upper_a_err: None,
            };
            let baseline = bounds.clone();

            apply_beta_contribution("relu1", Some(&beta_state), &mut bounds);

            assert_eq!(
                bounds.lower_a, baseline.lower_a,
                "lower_a should remain unchanged for non-finite signed beta with sign={sign}"
            );
            assert_eq!(
                bounds.upper_a, baseline.upper_a,
                "upper_a should remain unchanged for non-finite signed beta with sign={sign}"
            );
        }
    }

    /// Regression test: dispatch_node_backward returns Err on parallel array
    /// length mismatch instead of panicking at unchecked indexing sites (#2824).
    #[test]
    fn test_dispatch_node_backward_rejects_length_mismatch_2824() {
        use crate::{GraphNode, Layer, ReLULayer};
        use ndarray::arr1;
        use ny_core::GemmEngine;

        struct StubEngine;
        impl GemmEngine for StubEngine {
            fn gemm_f32(
                &self,
                m: usize,
                _k: usize,
                n: usize,
                _a: &[f32],
                _b: &[f32],
            ) -> ny_core::Result<Vec<f32>> {
                Ok(vec![0.0; m * n])
            }
        }

        let node = GraphNode {
            name: "relu1".to_string(),
            layer: Layer::ReLU(ReLULayer::new()),
            inputs: vec!["_input".to_string()],
        };

        // n_domains=2 but constrained_inputs has only 1 entry → mismatch
        let bt = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

        let node_lbs = vec![None, None];
        let owned_caches = [HashMap::new(), HashMap::new()];
        let bounds_caches: Vec<&HashMap<String, Arc<BoundedTensor>>> =
            owned_caches.iter().collect();
        let beta_states: Vec<Option<&GraphBetaState>> = vec![None, None];
        let alpha_states: Vec<Option<&crate::beta_crown::state::GraphDomainAlphaState>> =
            vec![None, None];
        let mut node_linear_bounds = make_pending(2);

        let engine = StubEngine;

        // constrained_inputs has length 1, but n_domains=2 → should Err
        let err = super::dispatch_node_backward(
            "relu1",
            &node,
            node_lbs,
            &[bt], // length 1 vs n_domains=2
            &bounds_caches,
            &beta_states,
            &alpha_states,
            &mut node_linear_bounds,
            2, // n_domains
            1, // network_input_dim
            &engine,
            None,
            None,  // mul_binary_alphas
            false, // stack_domains
        )
        .expect_err("mismatched parallel array lengths must return Err");

        let msg = err.to_string();
        assert!(
            msg.contains("parallel array length mismatch"),
            "expected descriptive mismatch error, got: {msg}"
        );
    }
}
