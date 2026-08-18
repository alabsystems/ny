// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dispatch helpers for special-case batched CROWN graph nodes.

use std::collections::HashMap;
use std::time::Instant;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::bounds::patches_batched::BatchedCrownBounds;
use crate::bounds::BatchedLinearBounds;
use crate::layers::Layer;
use crate::network::core::graph::batched_accumulator::BatchedCrownAccumulator;
use crate::network::GraphNode;
use crate::types::{BoundsProvenance, CrownBackwardResult};
use crate::MulBinaryRelaxationMode;

use super::binary_ops::{resolve_binary_input_bounds, AttentionCompositionRuntime};
use super::GraphNetwork;

/// Result of the Conv2d Patches fast-path attempt in the batched CROWN backward loop.
///
/// The backward loop tries Patches-mode propagation for Conv2d nodes before
/// falling through to the standard Dense dispatch.
pub(super) enum PatchesDispatchResult {
    /// Patches backward handled this node. Caller should `continue` the loop.
    Handled,
    /// Partial CROWN fallback was triggered. Caller should return this result.
    PartialFallback(Box<CrownBackwardResult>),
    /// Patches not applicable or already converted to Dense. Proceed with standard dispatch.
    DenseBounds(Box<BatchedLinearBounds>),
}

/// Result of dispatching a special-case binary or graph-structured batched operator.
#[derive(Debug)]
pub(super) enum SpecialBatchedDispatchResult {
    NotHandled,
    Handled,
    PartialFallback(Box<CrownBackwardResult>),
}

/// Whether a batched CROWN backward error should trigger partial IBP fallback.
///
/// Unsupported operations/configurations and explicit numerical-instability
/// guards degrade to IBP because the fallback remains sound. Shape mismatch is
/// intentionally excluded for the base path because silently widening previously
/// supported layers can mask regressions (#4219).
pub(super) fn is_fallback_eligible_error(error: &NyError) -> bool {
    matches!(
        error,
        NyError::UnsupportedOp(_)
            | NyError::UnsupportedConfiguration(_)
            | NyError::NumericalInstability(_)
    )
}

/// Bilinear batched fallback extends the base fallback set with shape mismatch.
///
/// Bilinear alpha-parameterized relaxation can hit legitimate shape-mismatch
/// fallback cases during partial composition, so the dispatch path keeps that
/// variant in the eligible set.
pub(super) fn is_bilinear_fallback_eligible_error(error: &NyError) -> bool {
    is_fallback_eligible_error(error) || matches!(error, NyError::ShapeMismatch { .. })
}

impl GraphNetwork {
    /// Handle the binary and graph-structured batched operators that do not route
    /// through the standard unary `Layer::propagate_crown_backward_batched` path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_special_batched_operator(
        &self,
        node: &GraphNode,
        node_name: &str,
        node_lb: &BatchedLinearBounds,
        input: &BoundedTensor,
        node_bounds: &HashMap<String, BoundedTensor>,
        output_shape: &[usize],
        mul_binary_relaxation: MulBinaryRelaxationMode,
        bilinear_alphas: Option<&HashMap<String, ndarray::Array4<f32>>>,
        attention_runtime: &mut AttentionCompositionRuntime<'_>,
        node_linear_bounds: &mut BatchedCrownAccumulator,
    ) -> Result<SpecialBatchedDispatchResult> {
        match &node.layer {
            Layer::ExpandLikeLastAxis(expand) => {
                let binary_inputs = resolve_binary_input_bounds(
                    node,
                    input,
                    node_bounds,
                    "IBP bounds for ExpandLikeLastAxis source",
                    "IBP bounds for ExpandLikeLastAxis reference",
                )?;
                let (lb_a, lb_b) = expand
                    .propagate_linear_batched_binary(
                        node_lb,
                        binary_inputs.input_a_bounds,
                        binary_inputs.input_b_bounds,
                    )
                    .map_err(|e| {
                        NyError::InvalidSpec(format!(
                            "Batched CROWN failed at node '{}' (ExpandLikeLastAxis): {}",
                            node_name, e
                        ))
                    })?;
                Self::accumulate_binary_bounds(
                    binary_inputs.input_a_name,
                    lb_a,
                    binary_inputs.input_b_name,
                    lb_b,
                    node_linear_bounds,
                )?;
                Ok(SpecialBatchedDispatchResult::Handled)
            }
            Layer::Add(add) => {
                let (add_a, add_b) = node.require_binary_inputs()?;
                let (lb_a, lb_b) = add.propagate_linear_batched_binary(node_lb).map_err(|e| {
                    NyError::InvalidSpec(format!(
                        "Batched CROWN failed at node '{}' (Add): {}",
                        node_name, e
                    ))
                })?;
                Self::accumulate_binary_bounds(add_a, lb_a, add_b, lb_b, node_linear_bounds)?;
                Ok(SpecialBatchedDispatchResult::Handled)
            }
            Layer::MatMul(matmul) => {
                let binary_inputs = resolve_binary_input_bounds(
                    node,
                    input,
                    node_bounds,
                    "IBP bounds for MatMul input A",
                    "IBP bounds for MatMul input B",
                )?;
                match self.handle_matmul_batched(
                    matmul,
                    node_name,
                    node_lb,
                    node_bounds,
                    output_shape,
                    binary_inputs,
                    attention_runtime,
                    node_linear_bounds,
                )? {
                    Some(fallback) => Ok(SpecialBatchedDispatchResult::PartialFallback(Box::new(
                        fallback,
                    ))),
                    None => Ok(SpecialBatchedDispatchResult::Handled),
                }
            }
            Layer::MulBinary(mul) => {
                let binary_inputs = resolve_binary_input_bounds(
                    node,
                    input,
                    node_bounds,
                    "IBP bounds for MulBinary input A",
                    "IBP bounds for MulBinary input B",
                )?;
                match mul.propagate_linear_batched_binary(
                    node_lb,
                    binary_inputs.input_a_bounds,
                    binary_inputs.input_b_bounds,
                    mul_binary_relaxation,
                ) {
                    Ok((lb_a, lb_b)) => {
                        let label = format!("MulBinary {:?}", mul_binary_relaxation);
                        if let Some(fallback) = Self::validate_binary_bounds_and_accumulate(
                            lb_a,
                            lb_b,
                            binary_inputs.input_a_name,
                            binary_inputs.input_b_name,
                            node_name,
                            &label,
                            node_lb,
                            node_bounds,
                            output_shape,
                            node_linear_bounds,
                        )? {
                            return Ok(SpecialBatchedDispatchResult::PartialFallback(Box::new(
                                fallback,
                            )));
                        }
                        Ok(SpecialBatchedDispatchResult::Handled)
                    }
                    Err(e) if is_fallback_eligible_error(&e) => {
                        debug!(
                            "GraphNetwork batched CROWN: MulBinary '{}' {:?} failed ({}), using partial CROWN",
                            node_name,
                            mul_binary_relaxation,
                            e
                        );
                        Self::partial_crown_ibp_fallback(
                            node_lb,
                            node_name,
                            node_bounds,
                            output_shape,
                        )
                        .map(Box::new)
                        .map(SpecialBatchedDispatchResult::PartialFallback)
                    }
                    Err(e) => Err(e),
                }
            }
            Layer::BilinearCrown(bilinear) => {
                let binary_inputs = resolve_binary_input_bounds(
                    node,
                    input,
                    node_bounds,
                    "IBP bounds for BilinearCrown input A",
                    "IBP bounds for BilinearCrown input B",
                )?;
                let node_alpha = bilinear_alphas.and_then(|m| m.get(node_name));
                match bilinear.propagate_linear_batched_binary_with_alpha(
                    node_lb,
                    binary_inputs.input_a_bounds,
                    binary_inputs.input_b_bounds,
                    node_alpha,
                ) {
                    Ok((lb_a, lb_b)) => {
                        if let Some(fallback) = Self::validate_binary_bounds_and_accumulate(
                            lb_a,
                            lb_b,
                            binary_inputs.input_a_name,
                            binary_inputs.input_b_name,
                            node_name,
                            "BilinearCrown",
                            node_lb,
                            node_bounds,
                            output_shape,
                            node_linear_bounds,
                        )? {
                            return Ok(SpecialBatchedDispatchResult::PartialFallback(Box::new(
                                fallback,
                            )));
                        }
                        Ok(SpecialBatchedDispatchResult::Handled)
                    }
                    Err(e) if is_bilinear_fallback_eligible_error(&e) => {
                        debug!(
                            "BilinearCrown '{}' failed ({}), using partial CROWN",
                            node_name, e
                        );
                        Self::partial_crown_ibp_fallback(
                            node_lb,
                            node_name,
                            node_bounds,
                            output_shape,
                        )
                        .map(Box::new)
                        .map(SpecialBatchedDispatchResult::PartialFallback)
                    }
                    Err(e) => Err(e),
                }
            }
            _ => Ok(SpecialBatchedDispatchResult::NotHandled),
        }
    }

    /// Dispatch a unary layer through the standard batched CROWN backward path.
    ///
    /// Returns `Ok(None)` on successful accumulation (continue the loop).
    /// Returns `Ok(Some(bt))` when partial CROWN fallback is triggered (caller
    /// should return this from the main method).
    ///
    /// Extracted to avoid duplication between the main dispatch and the
    /// Conv2d Patches fallback path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_batched_unary(
        layer: &Layer,
        node_name: &str,
        node_lb: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        first_input: &str,
        node_bounds: &HashMap<String, BoundedTensor>,
        output_shape: &[usize],
        node_linear_bounds: &mut BatchedCrownAccumulator,
    ) -> Result<Option<BoundedTensor>> {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(format!(
                "batched CROWN: deadline exceeded before unary dispatch at '{node_name}'"
            )));
        }
        let propagated =
            layer.propagate_crown_backward_batched(node_lb, Some(pre_activation), engine);
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(format!(
                "batched CROWN: deadline exceeded after unary dispatch at '{node_name}'"
            )));
        }
        match propagated {
            Ok(new_lb) => {
                node_linear_bounds.accumulate_dense(first_input, new_lb)?;
                Ok(None)
            }
            // #3131: Catch UnsupportedOp and UnsupportedConfiguration for partial CROWN
            // fallback. #2888: NumericalInstability also triggers fallback — non-finite
            // pre-activation bounds should degrade to IBP, not abort verification.
            // #4146: ShapeMismatch also triggers fallback — defense-in-depth for layers
            // that still emit ShapeMismatch from batched CROWN (e.g., Resize, Pad).
            Err(
                e @ NyError::UnsupportedOp(_)
                | e @ NyError::UnsupportedConfiguration(_)
                | e @ NyError::ShapeMismatch { .. }
                | e @ NyError::NumericalInstability(_),
            ) => {
                // Check for other pending paths (from upstream binary op splits).
                // If other intermediate nodes or network-input bounds have been
                // accumulated from a different path, partial CROWN fallback at this
                // node would only capture this one path's contribution — producing
                // unsound bounds. Propagate the error so the caller can fall back
                // to IBP entirely.  Part of #2072.
                let has_other_pending = !node_linear_bounds.is_empty();
                if has_other_pending {
                    debug!(
                        "GraphNetwork batched CROWN: unsupported layer '{}' ({}: {}) \
                         with other pending paths — propagating error",
                        node_name,
                        layer.layer_type(),
                        e
                    );
                    return Err(e);
                }
                debug!(
                    "GraphNetwork batched CROWN: unsupported layer '{}' ({}: {}), using partial CROWN",
                    node_name,
                    layer.layer_type(),
                    e
                );
                let node_ibp_bounds = node_bounds.get(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!("IBP bounds for node '{}' not found", node_name))
                })?;
                let fallback =
                    Self::partial_crown_fallback(node_lb, node_ibp_bounds, output_shape)?;
                Ok(Some(fallback))
            }
            // #3107: Preserve error type — don't wrap in InvalidSpec.
            Err(e) => Err(e),
        }
    }

    /// Try Conv2d Patches fast-path for a single-input Conv2d node, or convert to Dense.
    ///
    /// Phase 4 (#2613): For single-input Conv2d nodes in Patches mode, uses the unbatched
    /// Patches backward to compose strides/padding without materializing the full dense
    /// A-matrix. This provides the same 222x memory reduction as the unbatched CROWN path.
    ///
    /// Returns `Handled` if Patches succeeded, `PartialFallback` if Dense fallback
    /// triggered partial CROWN, or `DenseBounds` with the converted linear bounds
    /// for standard binary/unary dispatch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_conv2d_patches_or_dense(
        node_bcb: BatchedCrownBounds,
        layer: &Layer,
        num_inputs: usize,
        node_name: &str,
        pre_activation: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
        first_input: &str,
        node_bounds: &HashMap<String, BoundedTensor>,
        output_shape: &[usize],
        node_linear_bounds: &mut BatchedCrownAccumulator,
        peak_memory_bytes: &mut usize,
    ) -> Result<PatchesDispatchResult> {
        let is_patches = matches!(&node_bcb, BatchedCrownBounds::Patches(_));
        if is_patches && num_inputs == 1 {
            if let Layer::Conv2d(conv) = layer {
                if let BatchedCrownBounds::Patches(pb) = node_bcb {
                    let input_shape_dims = pre_activation.shape();
                    if input_shape_dims.len() >= 3 {
                        let (in_h, in_w) = (
                            input_shape_dims[input_shape_dims.len() - 2],
                            input_shape_dims[input_shape_dims.len() - 1],
                        );
                        let mut conv_with_shape = conv.clone();
                        conv_with_shape.set_input_shape(in_h, in_w);
                        match conv_with_shape
                            .propagate_patches_engine_and_deadline(&pb, engine, deadline)
                        {
                            Ok(result) => {
                                let new_bcb = if deadline.is_some() {
                                    BatchedCrownBounds::from_crown_bounds_with_deadline(
                                        result, deadline,
                                    )?
                                } else {
                                    BatchedCrownBounds::from_crown_bounds(result)?
                                };
                                let node_mem = new_bcb.memory_bytes();
                                *peak_memory_bytes = (*peak_memory_bytes).max(node_mem);
                                debug!(
                                    "GraphNetwork batched CROWN: Conv2d '{}' Patches backward [{}] ({:.1} MB)",
                                    node_name,
                                    if new_bcb.is_patches() { "Patches" } else { "Dense" },
                                    node_mem as f64 / 1_048_576.0
                                );
                                node_linear_bounds.accumulate(first_input, new_bcb)?;
                                return Ok(PatchesDispatchResult::Handled);
                            }
                            Err(e) if e.is_deadline_exceeded() => return Err(e),
                            Err(e) => {
                                debug!(
                                    "GraphNetwork batched CROWN: Conv2d '{}' Patches failed ({}), falling back to Dense",
                                    node_name, e
                                );
                            }
                        }
                    }
                    let fallback_label = if input_shape_dims.len() >= 3 {
                        "crown_batched:conv2d_patches_fallback"
                    } else {
                        "crown_batched:conv2d_small_input"
                    };
                    let node_lb = BatchedCrownBounds::Patches(pb)
                        .into_batched_dense_checked_with_deadline(fallback_label, deadline)?;
                    if let Some(fallback) = Self::dispatch_batched_unary(
                        layer,
                        node_name,
                        &node_lb,
                        pre_activation,
                        engine,
                        deadline,
                        first_input,
                        node_bounds,
                        output_shape,
                        node_linear_bounds,
                    )? {
                        return Ok(PatchesDispatchResult::PartialFallback(Box::new(
                            CrownBackwardResult {
                                bounds: fallback,
                                provenance: BoundsProvenance::Crown,
                            },
                        )));
                    }
                    return Ok(PatchesDispatchResult::Handled);
                }
            }
        }
        Ok(PatchesDispatchResult::DenseBounds(Box::new(
            node_bcb.into_batched_dense_checked_with_deadline(
                "crown_batched:dense_dispatch",
                deadline,
            )?,
        )))
    }
}

#[cfg(test)]
mod tests {
    use ny_core::NyError;

    use super::{is_bilinear_fallback_eligible_error, is_fallback_eligible_error};

    #[test]
    fn test_unsupported_op_is_fallback_eligible() {
        assert!(is_fallback_eligible_error(&NyError::UnsupportedOp(
            "test".to_string()
        )));
    }

    #[test]
    fn test_shape_mismatch_is_not_base_fallback_eligible() {
        assert!(!is_fallback_eligible_error(&NyError::ShapeMismatch {
            expected: vec![1, 2],
            got: vec![2, 1],
        }));
    }

    #[test]
    fn test_shape_mismatch_is_bilinear_fallback_eligible() {
        assert!(is_bilinear_fallback_eligible_error(
            &NyError::ShapeMismatch {
                expected: vec![1, 2],
                got: vec![2, 1],
            }
        ));
    }

    #[test]
    fn test_invalid_spec_is_not_fallback_eligible() {
        assert!(!is_fallback_eligible_error(&NyError::InvalidSpec(
            "bad node".to_string()
        )));
        assert!(!is_bilinear_fallback_eligible_error(&NyError::InvalidSpec(
            "bad node".to_string()
        )));
    }

    #[test]
    fn test_internal_error_is_not_fallback_eligible() {
        assert!(!is_fallback_eligible_error(&NyError::InternalError(
            "broken invariant".to_string()
        )));
        assert!(!is_bilinear_fallback_eligible_error(
            &NyError::InternalError("broken invariant".to_string())
        ));
    }
}
