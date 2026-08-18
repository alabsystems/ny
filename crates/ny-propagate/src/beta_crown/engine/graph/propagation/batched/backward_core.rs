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

use crate::beta_crown::engine::graph::adaptive_microbatch::MicrobatchRefusalReason;
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
    if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
        return Err(NyError::DeadlineExceeded(format!(
            "batched CROWN backward exceeded its deadline before node '{node_name}'"
        )));
    }

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

        // A stacked batched GEMM has no intra-call cancellation seam. Preserve
        // that fast path only for unscored work; deadline-scored traversal uses
        // the row-chunked single-domain API and can stop between chunks/domains.
        let new_bounds = if deadline.is_some() {
            let mut propagated = Vec::with_capacity(active_bounds.len());
            for bounds in active_bounds {
                let bounds = l
                    .propagate_linear_with_engine_and_deadline(bounds, Some(engine), deadline)?
                    .into_owned();
                if deadline.is_some_and(|value| std::time::Instant::now() >= value) {
                    return Err(NyError::DeadlineExceeded(format!(
                        "batched CROWN backward exceeded its deadline at Linear node '{node_name}'"
                    )));
                }
                propagated.push(bounds);
            }
            propagated
        } else {
            // BATCHED GPU GEMM: Process all domains in one kernel call.
            l.propagate_linear_batched_with_engine(&active_bounds, engine)?
        };

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
    } else if layer_prefers_pre_activation_lookup(&node.layer) {
        // #lsnc-subconst: SubConstant's broadcast backward (a671e609) reads the
        // PREDECESSOR shape to decide elementwise-vs-broadcast. Handing it the
        // network input made every elementwise `Sub` whose layer dim differs
        // from the input dim look like a broadcast, and the broadcast probe then
        // hard-errored `ShapeMismatch`. On lsnc_relu that killed the batched
        // CROWN backward at `/Sub` (input [6] vs layer [3]) and aborted BaB on
        // its FIRST batch, so all 80 instances returned `unknown` in ~3.4 s.
        // Resolve the real predecessor when it is cached; degrade to the
        // historical network input otherwise (byte-identical fallback), since
        // this lookup is an accuracy aid, never a soundness precondition.
        resolve_pre_activation(&node.inputs, constrained_input, bounds_cache)
            .unwrap_or(constrained_input)
    } else {
        constrained_input
    };

    if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(format!(
            "Batched CROWN deadline expired before Dense dispatch at '{node_name}'"
        )));
    }
    let ctx = DispatchContext {
        node_name,
        layer: &node.layer,
        inputs: &node.inputs,
        pre_activation,
        network_input: constrained_input,
        node_bounds: bounds_cache.into(),
        engine: Some(engine),
        // This helper already owns a Dense carrier, so ordinary dispatch keeps
        // the historical operator route while threading authority into the
        // deadline-aware Linear/Conv kernels. The surrounding checks retain
        // publication bracketing for legacy indivisible operators.
        deadline,
        bilinear_alphas: None,
        mul_binary_relaxation: MulBinaryRelaxationMode::default(),
        mul_binary_alphas, // #4284: thread shared root-level MulBinary alphas
        norm_inv_rms_override: None,
    };

    let dispatched = dispatch_backward_layer(&ctx, &lb);
    if deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
        return Err(NyError::DeadlineExceeded(format!(
            "Batched CROWN deadline expired after Dense dispatch at '{node_name}'"
        )));
    }
    match dispatched {
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
        Err(err) => {
            return Err(contextualize_generic_backward_error(
                node_name,
                node.layer.layer_type(),
                engine.forbids_unbounded_cpu_fallback(),
                err,
            ));
        }
    }

    Ok(())
}

/// Map a failed `dispatch_backward_layer` call onto the error the batched
/// CROWN backward surfaces, adding node context WITHOUT erasing the typed
/// variants that upstream classifiers depend on.
///
/// Extracted from the tail of [`generic_backward_domain`] so the mapping is
/// directly unit-testable: the GPU refusals it must preserve are produced by
/// the wgpu batched kernels at runtime, and a mock `GemmEngine` cannot inject
/// them here because the layer-local GPU->CPU fallbacks swallow engine
/// refusals before they reach this frame.
fn contextualize_generic_backward_error(
    node_name: &str,
    layer_type: &str,
    forbids_unbounded_cpu_fallback: bool,
    error: NyError,
) -> NyError {
    match error {
        // #3166: Catch UnsupportedOp and UnsupportedConfiguration.
        // #2888: NumericalInstability also triggers fallback for consistency,
        // though dispatch_backward_layer already converts it to Unsupported.
        NyError::UnsupportedOp(msg)
        | NyError::UnsupportedConfiguration(msg)
        | NyError::NumericalInstability(msg) => NyError::UnsupportedOp(format!(
            "Batched CROWN backward at node '{node_name}' ({layer_type}): {msg}",
        )),
        // Deadline authority is already contextualized by the layer-local
        // pollable path and must remain structurally visible to the VNN-COMP
        // caller. Wrapping it in InvalidSpec turns an ordinary verifier timeout
        // into a fatal "verification produced no verdict" result.
        error @ NyError::DeadlineExceeded(_) => error,
        // A bounded host facade's memory refusal is likewise authoritative:
        // wrapping it would make the adapter classify it as an ordinary
        // decline and retry through an uncapped scalar allocation. It stays
        // BARE - `is_cpu_memory_exceeded()` does not recurse through wrappers,
        // so even a LayerError shell would hide it from that adapter.
        error @ NyError::CpuMemoryExceeded { .. } if forbids_unbounded_cpu_fallback => error,
        // #oom-shrink-retry: a memory/device refusal recognized by the
        // adaptive-microbatch classifier must keep its typed variant, or the
        // opted-in controller lanes degrade from the designed shrink-retry to
        // the non-shrinking fallback (input-split lane: a fatal error). Wrap
        // in LayerError - `MicrobatchRefusalReason::from_error` recurses
        // through it - so node context survives without laundering the type
        // into InvalidSpec. Delegating the guard to `from_error` itself pins
        // the preserved set to exactly what the classifier acts on; anything
        // it declines still takes the historical InvalidSpec arm below.
        error if MicrobatchRefusalReason::from_error(&error).is_some() => NyError::LayerError {
            // Graph nodes carry no sequential layer index; the node context
            // lives in `layer_type`.
            layer_index: 0,
            layer_type: format!("{layer_type} at node '{node_name}'"),
            source: Box::new(error),
        },
        error => NyError::InvalidSpec(format!(
            "Batched CROWN failed at node '{node_name}' ({layer_type}): {error}"
        )),
    }
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

/// Returns true when shared backward dispatch READS the pre-activation shape but
/// still has a sound behavior without it.
///
/// Distinct from [`layer_needs_pre_activation_lookup`], whose layers hard-require
/// the tensor: these resolve it opportunistically and fall back to the network
/// input when the predecessor is not cached. See the `#lsnc-subconst` note at the
/// call site.
fn layer_prefers_pre_activation_lookup(layer: &Layer) -> bool {
    matches!(layer, Layer::SubConstant(_))
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

    use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};
    use ny_core::{GemmEngine, NaiveCpuGemmEngine, NyError, Result};
    use ny_test_utils::CountingGemmEngine;

    use crate::layers::{Conv2dLayer, ConvTranspose2dLayer, Layer, LinearLayer};

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

    fn conv_transpose_dispatch_fixture() -> (GraphNode, BoundedTensor) {
        let kernel = ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0_f32);
        let conv = ConvTranspose2dLayer::new(kernel, None, (1, 1), (0, 0))
            .expect("valid ConvTranspose2d layer");
        let node = GraphNode::from_input("ConvTranspose_7", Layer::ConvTranspose2d(conv));
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 1]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1, 1, 1]), 1.0_f32),
        )
        .expect("valid bounded input");
        (node, input)
    }

    struct BoundedMemoryEngine;

    impl GemmEngine for BoundedMemoryEngine {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            panic!("bounded Conv2d must force the certified f64 route")
        }

        fn gemm_f64(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f64],
            _b: &[f64],
        ) -> Result<Vec<f64>> {
            Err(NyError::CpuMemoryExceeded {
                required_bytes: 2,
                budget_bytes: 1,
                site: "bounded generic backward test",
            })
        }

        fn forbids_unbounded_cpu_fallback(&self) -> bool {
            true
        }

        fn provides_deadline_pollable_host_gemm(&self) -> bool {
            true
        }
    }

    #[test]
    fn generic_backward_preserves_bounded_engine_memory_refusal() {
        // Even with the legacy kill switch set to zero, bounded Conv2d forces
        // the certified f64 route. Its structured host-memory refusal must
        // remain terminal instead of entering any local/global fallback.
        crate::tests::with_serialized_env_vars(&[("NY_CONV_SKIP_DEAD_F32", "0")], || {
            let kernel = ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0_f32);
            let conv =
                Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).expect("valid Conv2d test layer");
            let node = GraphNode::from_input("Conv_0", Layer::Conv2d(conv));
            let input = BoundedTensor::new(
                ArrayD::from_elem(IxDyn(&[1, 1, 1]), -1.0_f32),
                ArrayD::from_elem(IxDyn(&[1, 1, 1]), 1.0_f32),
            )
            .expect("valid bounded input");
            let mut sink = |_name: &str, _bounds: LinearBounds| Ok(());
            let error = super::generic_backward_domain(
                node.name(),
                &node,
                LinearBounds::identity(1),
                &input,
                &HashMap::new(),
                input.len(),
                &BoundedMemoryEngine,
                None,
                None,
                &mut sink,
            )
            .expect_err("bounded memory refusal must remain structured");
            assert!(error.is_cpu_memory_exceeded(), "wrong error: {error}");
        });
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

    #[test]
    fn deadline_scored_linear_dispatch_bypasses_opaque_engine() {
        let node = GraphNode {
            name: "relu1".to_string(),
            layer: Layer::Linear(LinearLayer::new(arr2(&[[2.0_f32]]), None).expect("linear layer")),
            inputs: vec![crate::NETWORK_INPUT.to_string()],
        };
        let coeff = Array2::<f32>::ones((65, 1));
        let bounds = LinearBounds::new(coeff.clone(), Array1::zeros(65), coeff, Array1::zeros(65))
            .expect("linear bounds");
        let constrained_inputs = vec![
            BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
                .expect("first input"),
            BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn())
                .expect("second input"),
        ];
        let owned_caches = [HashMap::new(), HashMap::new()];
        let bounds_caches: Vec<&HashMap<String, Arc<BoundedTensor>>> =
            owned_caches.iter().collect();
        let beta_states: Vec<Option<&GraphBetaState>> = vec![None, None];
        let alpha_states: Vec<Option<&crate::beta_crown::state::GraphDomainAlphaState>> =
            vec![None, None];
        let mut pending = make_pending(2);
        let engine = CountingGemmEngine::new();

        super::dispatch_node_backward(
            "relu1",
            &node,
            vec![Some(bounds.clone()), Some(bounds)],
            &constrained_inputs,
            &bounds_caches,
            &beta_states,
            &alpha_states,
            &mut pending,
            2,
            1,
            &engine,
            Some(std::time::Instant::now() + std::time::Duration::from_secs(5)),
            None,
            false,
        )
        .expect("finite deadline should complete");

        assert_eq!(
            engine.gemm_calls(),
            0,
            "finite-deadline Linear propagation must use the pollable CPU path"
        );
    }

    #[test]
    fn expired_linear_dispatch_refuses_before_engine_launch() {
        let node = GraphNode {
            name: "relu1".to_string(),
            layer: Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("linear layer")),
            inputs: vec![crate::NETWORK_INPUT.to_string()],
        };
        let constrained_input =
            BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).expect("input");
        let owned_cache = HashMap::new();
        let bounds_caches = vec![&owned_cache];
        let beta_states: Vec<Option<&GraphBetaState>> = vec![None];
        let alpha_states: Vec<Option<&crate::beta_crown::state::GraphDomainAlphaState>> =
            vec![None];
        let mut pending = make_pending(1);
        let engine = CountingGemmEngine::new();

        let error = super::dispatch_node_backward(
            "relu1",
            &node,
            vec![Some(LinearBounds::identity(1))],
            &[constrained_input],
            &bounds_caches,
            &beta_states,
            &alpha_states,
            &mut pending,
            1,
            1,
            &engine,
            Some(
                std::time::Instant::now()
                    .checked_sub(std::time::Duration::from_millis(1))
                    .expect("one millisecond fits before the current instant"),
            ),
            None,
            false,
        )
        .expect_err("expired deadline must refuse");

        assert!(error.is_deadline_exceeded());
        assert_eq!(engine.gemm_calls(), 0);
    }

    /// Regression for the cGAN VNN-COMP canary: ConvTranspose's pollable
    /// backward correctly returned DeadlineExceeded, but the generic batched
    /// wrapper erased the variant by rebuilding it as InvalidSpec.
    #[test]
    fn generic_backward_preserves_conv_transpose_deadline_provenance() {
        let (node, constrained_input) = conv_transpose_dispatch_fixture();
        let bounds_cache = HashMap::new();
        let mut sink_calls = 0usize;
        let mut sink = |_input_name: &str, _bounds: LinearBounds| -> Result<()> {
            sink_calls += 1;
            Ok(())
        };
        let expired = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_millis(1))
            .expect("one millisecond fits before the current instant");

        let error = super::generic_backward_domain(
            "ConvTranspose_7",
            &node,
            LinearBounds::identity(1),
            &constrained_input,
            &bounds_cache,
            constrained_input.len(),
            &NaiveCpuGemmEngine,
            Some(expired),
            None,
            &mut sink,
        )
        .expect_err("expired ConvTranspose dispatch must remain a typed deadline");

        assert!(
            matches!(error, NyError::DeadlineExceeded(_)),
            "deadline provenance must survive batched dispatch: {error:?}"
        );
        assert_eq!(
            sink_calls, 0,
            "a failed dispatch must not publish partial predecessor bounds"
        );
    }

    /// Companion control: non-deadline failures retain the batched node/layer
    /// context that makes malformed models diagnosable.
    #[test]
    fn generic_backward_contextualizes_non_deadline_dispatch_failure() {
        let (node, constrained_input) = conv_transpose_dispatch_fixture();
        let bounds_cache = HashMap::new();
        let mut sink = |_input_name: &str, _bounds: LinearBounds| -> Result<()> { Ok(()) };

        let error = super::generic_backward_domain(
            "ConvTranspose_7",
            &node,
            LinearBounds::identity(2),
            &constrained_input,
            &bounds_cache,
            constrained_input.len(),
            &NaiveCpuGemmEngine,
            None,
            None,
            &mut sink,
        )
        .expect_err("wrong ConvTranspose objective width must fail");

        let NyError::InvalidSpec(message) = error else {
            panic!("non-deadline dispatch failure should be contextualized as InvalidSpec");
        };
        assert!(
            message.contains("Batched CROWN failed at node 'ConvTranspose_7' (ConvTranspose2d)"),
            "missing batched node/layer context: {message}"
        );
        assert!(
            message.contains("Shape mismatch"),
            "missing original dispatch diagnostic: {message}"
        );
    }

    /// #oom-shrink-retry: the terminal catch-all used to stringify
    /// GpuMemoryExceeded / wgpu-OOM InternalError into InvalidSpec, so
    /// `MicrobatchRefusalReason::from_error` one frame up returned None and
    /// the opted-in controller lanes never took the designed shrink-retry.
    /// This box has no GPU and the layer-local GPU->CPU fallbacks swallow
    /// mock-engine refusals before this frame, so a runtime OOM cannot be
    /// exercised end-to-end here; type preservation through the mapping the
    /// production Err path calls is what is testable.
    #[test]
    fn backward_dispatch_error_keeps_memory_refusals_classifiable() {
        use crate::beta_crown::engine::graph::adaptive_microbatch::MicrobatchRefusalReason;

        // GPU allocation refusal -> DeviceAllocation, wrapped but classifiable.
        let error = super::contextualize_generic_backward_error(
            "Conv_0",
            "Conv2d",
            false,
            NyError::GpuMemoryExceeded {
                required_bytes: 2,
                budget_bytes: 1,
            },
        );
        assert_eq!(
            MicrobatchRefusalReason::from_error(&error),
            Some(MicrobatchRefusalReason::DeviceAllocation),
            "GpuMemoryExceeded must reach the shrink-retry classifier intact: {error:?}"
        );
        let NyError::LayerError {
            layer_type, source, ..
        } = error
        else {
            panic!("preserved refusal should carry node context via LayerError");
        };
        assert!(
            layer_type.contains("node 'Conv_0'"),
            "missing node context: {layer_type}"
        );
        assert!(matches!(*source, NyError::GpuMemoryExceeded { .. }));

        // wgpu runtime OOM surfaces as a structured InternalError prefix.
        let error = super::contextualize_generic_backward_error(
            "Conv_0",
            "Conv2d",
            false,
            NyError::InternalError("wgpu out-of-memory in batched backward".into()),
        );
        assert_eq!(
            MicrobatchRefusalReason::from_error(&error),
            Some(MicrobatchRefusalReason::DeviceAllocation),
            "wgpu OOM InternalError must stay classifiable: {error:?}"
        );

        // Unbounded-CPU lanes: the host refusal becomes classifiable
        // (HostAllocation) instead of the historical InvalidSpec laundering.
        let error = super::contextualize_generic_backward_error(
            "Conv_0",
            "Conv2d",
            false,
            NyError::CpuMemoryExceeded {
                required_bytes: 2,
                budget_bytes: 1,
                site: "test",
            },
        );
        assert_eq!(
            MicrobatchRefusalReason::from_error(&error),
            Some(MicrobatchRefusalReason::HostAllocation),
        );

        // Bounded facade: the host refusal must stay BARE (its adapter checks
        // `is_cpu_memory_exceeded()` without recursing through wrappers).
        let error = super::contextualize_generic_backward_error(
            "Conv_0",
            "Conv2d",
            true,
            NyError::CpuMemoryExceeded {
                required_bytes: 2,
                budget_bytes: 1,
                site: "test",
            },
        );
        assert!(
            error.is_cpu_memory_exceeded(),
            "bounded-facade refusal must remain bare: {error:?}"
        );
    }

    /// Companion control for #oom-shrink-retry: errors the classifier does not
    /// act on must keep the historical contextualized-InvalidSpec mapping.
    #[test]
    fn backward_dispatch_error_still_contextualizes_unrelated_errors() {
        use crate::beta_crown::engine::graph::adaptive_microbatch::MicrobatchRefusalReason;

        let error = super::contextualize_generic_backward_error(
            "Conv_0",
            "Conv2d",
            false,
            NyError::shape_mismatch(vec![1], vec![2]),
        );
        assert!(
            MicrobatchRefusalReason::from_error(&error).is_none(),
            "unrelated errors must not become retryable refusals"
        );
        let NyError::InvalidSpec(message) = error else {
            panic!("unrelated dispatch failures must still contextualize as InvalidSpec");
        };
        assert!(
            message.contains("Batched CROWN failed at node 'Conv_0' (Conv2d)"),
            "missing batched node/layer context: {message}"
        );
        assert!(
            message.contains("Shape mismatch"),
            "missing original diagnostic: {message}"
        );

        // A validation-class wgpu InternalError (no locally-owned prefix) is a
        // bug, deliberately not retried, and keeps the InvalidSpec mapping.
        let error = super::contextualize_generic_backward_error(
            "Conv_0",
            "Conv2d",
            false,
            NyError::InternalError("wgpu validation error: bad bind group".into()),
        );
        assert!(matches!(error, NyError::InvalidSpec(_)));
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
            ) -> Result<Vec<f32>> {
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
