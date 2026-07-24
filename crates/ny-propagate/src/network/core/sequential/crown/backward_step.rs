// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dense CROWN backward-step helpers and materialization budget guards.

use crate::bounds::patches::CrownBounds;
use crate::bounds::LinearBounds;
use crate::layers::Layer;
use crate::network::backward_dispatch::{
    dispatch_backward_layer, BackwardDispatchResult, DispatchContext,
};
use crate::network::crown_memory::{cpu_crown_dense_budget_bytes, DenseMaterializationEstimate};
use crate::types::{CrownIbpFallbackReason, MulBinaryRelaxationMode};
use crate::BoundPropagation;
use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::{CrownStepFallback, CrownStepResult};

pub(super) fn dense_identity_budget_estimate(
    site: &'static str,
    dim: usize,
) -> Option<DenseMaterializationEstimate> {
    let estimate = DenseMaterializationEstimate {
        site,
        rows: dim,
        cols: dim,
        required_bytes: LinearBounds::identity_pair_bytes(dim).unwrap_or(usize::MAX),
    };
    estimate
        .exceeds_budget(cpu_crown_dense_budget_bytes())
        .then_some(estimate)
}

pub(super) fn dense_materialization_budget_estimate(
    crown_bounds: &CrownBounds,
    site: &'static str,
) -> Result<Option<DenseMaterializationEstimate>> {
    match crown_bounds {
        CrownBounds::Dense(_) => Ok(None),
        CrownBounds::Patches(pb) => {
            let (rows, cols) = pb.dense_pair_shape()?;
            let estimate = DenseMaterializationEstimate::new(site, rows, cols);
            Ok(estimate
                .exceeds_budget(cpu_crown_dense_budget_bytes())
                .then_some(estimate))
        }
    }
}

pub(super) fn log_dense_materialization_budget_fallback(
    label: &str,
    estimate: DenseMaterializationEstimate,
    layer_idx: Option<usize>,
    layer_type: Option<&str>,
) {
    let details = estimate.budget_exceeded_details(cpu_crown_dense_budget_bytes());
    match (layer_idx, layer_type) {
        (Some(idx), Some(kind)) => {
            info!("{label}: layer {idx} ({kind}) {details}; falling back to IBP")
        }
        _ => info!("{label}: {details}; falling back to IBP"),
    }
}

fn memory_budget_step_fallback(
    estimate: DenseMaterializationEstimate,
    layer_idx: usize,
    layer_type: &str,
) -> CrownStepFallback {
    CrownStepFallback {
        reason: CrownIbpFallbackReason::MemoryBudgetExceeded,
        details: format!(
            "layer {layer_idx} ({layer_type}) {}",
            estimate.budget_exceeded_details(cpu_crown_dense_budget_bytes())
        ),
    }
}

pub(super) fn guard_dense_materialization_budget(
    crown_bounds: &CrownBounds,
    site: &'static str,
    layer: &Layer,
    layer_idx: usize,
    label: &str,
) -> Result<Option<CrownStepResult>> {
    let Some(estimate) = dense_materialization_budget_estimate(crown_bounds, site)? else {
        return Ok(None);
    };
    log_dense_materialization_budget_fallback(
        label,
        estimate,
        Some(layer_idx),
        Some(layer.layer_type()),
    );
    Ok(Some(CrownStepResult::IbpFallback(
        memory_budget_step_fallback(estimate, layer_idx, layer.layer_type()),
    )))
}

/// Execute one CROWN backward propagation step for a single layer (Dense only).
///
/// Called directly by `propagate_crown_fast` and `propagate_crown_with_linear`.
/// Called indirectly by `propagate_crown_with_engine` (and `propagate_crown_ibp`)
/// via [`super::crown_backward_step_patches`], which handles Conv2d Patches dispatch
/// and delegates all other layers to this function.
///
/// The only variation across callers was whether a
/// GEMM engine is passed for `Layer::Linear`; this is captured by the
/// `engine` parameter.
///
/// # Returns
///
/// - `Ok(CrownStepResult::Continue)` — propagation succeeded, `linear_bounds`
///   updated in-place.
/// - `Ok(CrownStepResult::IbpFallback(_))` — caller should fall back to IBP.
/// - `Err(e)` — propagation error.
///
/// # Design reference
///
/// See `designs/2026-03-13-crown-backward-step-dispatch-bridge.md` for rationale.
/// Delegates per-layer shape setup and trait dispatch to `dispatch_backward_layer`
/// (backward_dispatch module), with pre-filters for ReLU and multi-input ops.
pub(super) fn crown_backward_step(
    layer: &Layer,
    linear_bounds: &mut LinearBounds,
    pre_activation: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    layer_idx: usize,
    label: &str,
) -> Result<CrownStepResult> {
    // === Pre-filter 1: ReLU ===
    // dispatch_backward_layer returns Unsupported for ReLU because graph-CROWN
    // needs alpha/beta state. Sequential CROWN uses the simpler
    // propagate_crown_backward trait path (no alpha optimization).
    if matches!(layer, Layer::ReLU(_)) {
        return crown_backward_relu(layer, linear_bounds, pre_activation, layer_idx, label);
    }

    // === Pre-filter 1b: embedded-constant Where ===
    // In a sequential chain the only well-formed Where is the embedded-constant
    // form: Where(cond, const_true, const_false) where `cond` is the chain input
    // (= pre_activation). Both branches are constants, so the output is a constant
    // vector w.r.t. the chain input. `embedded_constant_select_output` returns the
    // EXACT per-element select when `cond` is bound-independent (lower == upper)
    // and the sound IBP union otherwise — strictly no looser than the generic IBP
    // fallback that follows. Concretize the accumulated CROWN bounds against that
    // (zero-A, constant-bias result), tightening UNSAT for cctsdb/nn4sys/vit
    // masks. A non-embedded sequential Where (3 named edges) cannot exist, so it
    // still drops to the multi-input IBP fallback below.
    if let Layer::Where(where_layer) = layer {
        if where_layer.has_embedded_constants() {
            match where_layer.embedded_constant_select_output(pre_activation) {
                Ok(select) => {
                    let concretized = linear_bounds.concretize_sound(&select);
                    let concretized_flat = concretized.flatten();
                    let num_outputs = concretized_flat.len();
                    // `len()` == `flatten().len()` (flatten preserves element
                    // count) with no allocation.
                    let num_inputs = pre_activation.len();
                    *linear_bounds = LinearBounds::new_or_conservative(
                        Array2::zeros((num_outputs, num_inputs)),
                        Array1::from_vec(concretized_flat.lower().iter().copied().collect()),
                        Array2::zeros((num_outputs, num_inputs)),
                        Array1::from_vec(concretized_flat.upper().iter().copied().collect()),
                    )?;
                    return Ok(CrownStepResult::Continue);
                }
                // Constants failed to broadcast / produced NaN: keep the sound
                // generic IBP fallback below by falling through.
                Err(e) => {
                    debug!(
                        "{}: embedded-constant Where at layer {} fell back to IBP: {}",
                        label, layer_idx, e
                    );
                }
            }
        }
    }

    // === Pre-filter 2: Multi-input ops ===
    // dispatch_backward_layer handles binary/n-ary ops by reading ctx.inputs[0..N]
    // and resolving node_bounds. Sequential networks pass inputs=&[] (no named graph
    // edges), so binary dispatch would fail with InvalidSpec. Pre-filter these to
    // the same IbpFallback that the inline code used.
    if matches!(
        layer,
        Layer::MatMul(_)
            | Layer::MulBinary(_)
            | Layer::Add(_)
            | Layer::Concat(_)
            | Layer::Sub(_)
            | Layer::Div(_)
            | Layer::Atan2(_)
            | Layer::BilinearCrown(_)
            | Layer::MinBinary(_)
            | Layer::MaxBinary(_)
            | Layer::ExpandLikeLastAxis(_)
            | Layer::ScatterAdd(_)
            | Layer::IndexAdd(_)
            | Layer::ScatterNd(_)
            | Layer::Where(_)
            | Layer::NonZero(_)
            | Layer::SelfAttention(_)
    ) {
        debug!(
            "{}: layer {} ({}) not supported in sequential network, using IBP",
            label,
            layer_idx,
            layer.layer_type()
        );
        return Ok(CrownStepResult::IbpFallback(CrownStepFallback {
            reason: CrownIbpFallbackReason::CrownPropagationError,
            details: format!(
                "layer {} ({}) not supported in sequential network",
                layer_idx,
                layer.layer_type()
            ),
        }));
    }

    // === Pre-filter 3: SkipMerge pass-through ===
    // SkipMerge is identity in sequential CROWN — bounds pass through unchanged.
    // dispatch_backward_layer requires ctx.inputs.len() == 1 (graph edge),
    // but sequential networks pass inputs=&[] (no named edges). Handle directly.
    if matches!(layer, Layer::SkipMerge(_)) {
        return Ok(CrownStepResult::Continue);
    }

    // Build minimal context for unary dispatch. After pre-filtering ReLU,
    // multi-input ops, and SkipMerge, only unary shape-setup and
    // trait-dispatch layers remain.
    let empty_bounds: HashMap<String, BoundedTensor> = HashMap::new();
    let ctx = DispatchContext {
        node_name: label,
        layer,
        inputs: &[],
        pre_activation,
        network_input: pre_activation,
        node_bounds: (&empty_bounds).into(),
        engine,
        deadline: None,
        bilinear_alphas: None,
        mul_binary_relaxation: MulBinaryRelaxationMode::default(),
        mul_binary_alphas: None,
        norm_inv_rms_override: None,
    };

    // dispatch_backward_layer may raise UnsupportedOp (e.g., Conv1d with
    // shape < 2 dims), NumericalInstability, ShapeMismatch (#3813: Dense
    // Conv2d backward when graph restructuring changes dimensions), or
    // DeadlineExceeded (#3795). In sequential CROWN, these are graceful
    // IBP fallbacks via per-layer IBP concretization.
    let dispatch_result = match dispatch_backward_layer(&ctx, linear_bounds) {
        Ok(result) => result,
        Err(
            NyError::UnsupportedOp(ref msg)
            | NyError::UnsupportedConfiguration(ref msg)
            | NyError::NumericalInstability(ref msg)
            | NyError::DeadlineExceeded(ref msg),
        ) => {
            return crown_backward_ibp_concretize(
                layer,
                linear_bounds,
                pre_activation,
                layer_idx,
                label,
                msg,
            );
        }
        Err(NyError::ShapeMismatch {
            ref expected,
            ref got,
        }) => {
            let msg = format!("shape mismatch: expected {:?}, got {:?}", expected, got);
            return crown_backward_ibp_concretize(
                layer,
                linear_bounds,
                pre_activation,
                layer_idx,
                label,
                &msg,
            );
        }
        Err(e) => return Err(e),
    };

    match dispatch_result {
        BackwardDispatchResult::Single(new_lb) => {
            *linear_bounds = *new_lb;
            Ok(CrownStepResult::Continue)
        }
        BackwardDispatchResult::PassThrough => Ok(CrownStepResult::Continue),
        BackwardDispatchResult::Unsupported(msg) => {
            // Per-layer IBP concretization: when dispatch returns Unsupported,
            // concretize accumulated CROWN bounds through IBP at this layer
            // instead of discarding ALL CROWN work with whole-network IBP fallback.
            crown_backward_ibp_concretize(
                layer,
                linear_bounds,
                pre_activation,
                layer_idx,
                label,
                &msg,
            )
        }
        // Binary/Nary: unreachable after pre-filter, but handle defensively.
        BackwardDispatchResult::Binary { .. } | BackwardDispatchResult::Nary { .. } => {
            Ok(CrownStepResult::IbpFallback(CrownStepFallback {
                reason: CrownIbpFallbackReason::CrownPropagationError,
                details: format!(
                    "unexpected multi-input result for {} in sequential CROWN",
                    layer.layer_type()
                ),
            }))
        }
    }
}

/// ReLU backward via trait dispatch (no alpha optimization in sequential CROWN).
///
/// dispatch_backward_layer returns Unsupported for ReLU because graph-CROWN
/// needs alpha/beta state that varies per call site. Sequential CROWN uses
/// the simpler propagate_crown_backward trait path.
fn crown_backward_relu(
    layer: &Layer,
    linear_bounds: &mut LinearBounds,
    pre_activation: &BoundedTensor,
    layer_idx: usize,
    label: &str,
) -> Result<CrownStepResult> {
    match layer.propagate_crown_backward(linear_bounds, Some(pre_activation)) {
        Ok(mut next) => {
            // Eager per-row discharge of the carried coefficient error over the
            // pre-activation cut (#cgan-conv-err-compose, see
            // LinearBounds::fold_coeff_err_over_box_eager). Rows with a
            // non-finite penalty keep carrying (prior behavior).
            next.fold_coeff_err_over_box_eager(pre_activation);
            *linear_bounds = next;
            Ok(CrownStepResult::Continue)
        }
        Err(
            NyError::UnsupportedOp(ref msg)
            | NyError::UnsupportedConfiguration(ref msg)
            | NyError::NumericalInstability(ref msg),
        ) => crown_backward_ibp_concretize(
            layer,
            linear_bounds,
            pre_activation,
            layer_idx,
            label,
            msg,
        ),
        // #3813: ShapeMismatch triggers per-layer IBP concretization (same as
        // crown_backward_step). RSPLITTER models change intermediate dimensions,
        // causing shape mismatches in ReLU backward propagation. IBP fallback
        // is always sound.
        Err(NyError::ShapeMismatch {
            ref expected,
            ref got,
        }) => {
            let msg = format!(
                "ReLU shape mismatch: expected {:?}, got {:?}",
                expected, got
            );
            crown_backward_ibp_concretize(
                layer,
                linear_bounds,
                pre_activation,
                layer_idx,
                label,
                &msg,
            )
        }
        Err(e) => Err(e),
    }
}

/// Per-layer IBP concretization fallback.
///
/// When CROWN backward fails for a single layer, concretize accumulated CROWN
/// bounds through IBP at that layer instead of discarding ALL CROWN work with
/// whole-network IBP fallback.
/// Pattern: beta_crown/engine/backward/layer_dispatch.rs:beta_crown_ibp_fallback.
fn crown_backward_ibp_concretize(
    layer: &Layer,
    linear_bounds: &mut LinearBounds,
    pre_activation: &BoundedTensor,
    layer_idx: usize,
    label: &str,
    reason: &str,
) -> Result<CrownStepResult> {
    warn!(
        "{}: layer {} ({}) unsupported/unstable, \
         concretizing to per-layer IBP bounds: {}",
        label,
        layer_idx,
        layer.layer_type(),
        reason,
    );
    let post_bounds = layer.propagate_ibp(pre_activation)?;
    let concretized = linear_bounds.concretize_sound(&post_bounds);
    let concretized_flat = concretized.flatten();
    let num_outputs = concretized_flat.len();
    // `len()` == `flatten().len()` (flatten preserves element count) with no allocation.
    let num_inputs = pre_activation.len();
    *linear_bounds = LinearBounds::new_or_conservative(
        Array2::zeros((num_outputs, num_inputs)),
        Array1::from_vec(concretized_flat.lower().iter().copied().collect()),
        Array2::zeros((num_outputs, num_inputs)),
        Array1::from_vec(concretized_flat.upper().iter().copied().collect()),
    )?;
    Ok(CrownStepResult::Continue)
}

#[cfg(test)]
#[path = "backward_step_tests.rs"]
mod tests;
