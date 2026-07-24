// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Binary-op helpers for graph batched CROWN dispatch.

use std::collections::HashMap;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::bounds::BatchedLinearBounds;
use crate::layers::MatMulLayer;
use crate::network::core::graph::batched_accumulator::BatchedCrownAccumulator;
use crate::network::GraphNode;
use crate::types::{BoundsProvenance, CrownBackwardResult};

use super::{resolve_node_bounds, GraphNetwork};

pub(super) struct BinaryInputBounds<'a> {
    pub(super) input_a_name: &'a str,
    pub(super) input_b_name: &'a str,
    pub(super) input_a_bounds: &'a BoundedTensor,
    pub(super) input_b_bounds: &'a BoundedTensor,
}

/// Resolve both input names and IBP bounds for a binary graph node.
pub(super) fn resolve_binary_input_bounds<'a>(
    node: &'a GraphNode,
    input: &'a BoundedTensor,
    node_bounds: &'a HashMap<String, BoundedTensor>,
    input_a_context: &str,
    input_b_context: &str,
) -> Result<BinaryInputBounds<'a>> {
    let (input_a_name, input_b_name) = node.require_binary_inputs()?;
    let input_a_bounds = resolve_node_bounds(input_a_name, input, node_bounds, input_a_context)?;
    let input_b_bounds = resolve_node_bounds(input_b_name, input, node_bounds, input_b_context)?;
    Ok(BinaryInputBounds {
        input_a_name,
        input_b_name,
        input_a_bounds,
        input_b_bounds,
    })
}

/// Controls whether the attention-shaped MatMul retry path accumulates bounds
/// and continues backward (full composition) or discards them and falls back
/// to partial CROWN (production default).
///
/// The production default is `PartialFallback` because McCormick bilinear
/// relaxation treats Q and K as independent, making full CROWN bounds looser
/// than IBP-at-attention when Q and K share inputs (correlated paths).
///
/// `FullComposition` is experimental — it gives CROWN propagation through the
/// attention MatMul, useful as a tighter seed when Q and K are uncorrelated
/// or perturbation radii are small.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum AttentionCompositionStrategy {
    /// Discard the attention-identity retry result and concretize with IBP
    /// at the attention boundary (current production behavior).
    #[default]
    PartialFallback,
    /// Accumulate the attention-identity retry result and continue backward
    /// through the graph (experimental full-composition path for #318).
    FullComposition,
}

pub(super) struct AttentionCompositionRuntime<'a> {
    strategy: AttentionCompositionStrategy,
    full_composition_used: Option<&'a mut bool>,
}

impl AttentionCompositionRuntime<'_> {
    pub(super) fn production() -> Self {
        Self {
            strategy: AttentionCompositionStrategy::default(),
            full_composition_used: None,
        }
    }

    pub(super) fn full_composition() -> Self {
        Self {
            strategy: AttentionCompositionStrategy::FullComposition,
            full_composition_used: None,
        }
    }

    fn should_retry_attention_identity(&self) -> bool {
        self.strategy == AttentionCompositionStrategy::FullComposition
    }
}

#[cfg(test)]
impl<'a> AttentionCompositionRuntime<'a> {
    pub(super) fn full_composition_with_diagnostic(full_composition_used: &'a mut bool) -> Self {
        Self {
            strategy: AttentionCompositionStrategy::FullComposition,
            full_composition_used: Some(full_composition_used),
        }
    }
}

impl GraphNetwork {
    /// Accumulate batched CROWN bounds for both outputs of a binary op.
    ///
    /// Deduplicates the 5× repeated pair of `accumulate_dense_batched_bounds_to_input`
    /// calls in the MatMul, MulBinary, BilinearCrown, and ExpandLikeLastAxis branches.
    pub(super) fn accumulate_binary_bounds(
        input_a_name: &str,
        lb_a: BatchedLinearBounds,
        input_b_name: &str,
        lb_b: BatchedLinearBounds,
        node_linear_bounds: &mut BatchedCrownAccumulator,
    ) -> Result<()> {
        node_linear_bounds.accumulate_dense(input_a_name, lb_a)?;
        node_linear_bounds.accumulate_dense(input_b_name, lb_b)?;
        Ok(())
    }

    /// Partial CROWN fallback: look up IBP bounds for a node and concretize.
    ///
    /// Combines the repeated pattern of `node_bounds.get(node_name)` + `partial_crown_fallback`
    /// + `CrownBackwardResult::Crown` wrapping used ~5× in the binary-op dispatch.
    pub(super) fn partial_crown_ibp_fallback(
        node_lb: &BatchedLinearBounds,
        node_name: &str,
        node_bounds: &HashMap<String, BoundedTensor>,
        output_shape: &[usize],
    ) -> Result<CrownBackwardResult> {
        let ibp_bounds = node_bounds.get(node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!("IBP bounds for '{}' not found", node_name))
        })?;
        Self::partial_crown_fallback(node_lb, ibp_bounds, output_shape).map(|bounds| {
            CrownBackwardResult {
                bounds,
                provenance: BoundsProvenance::Crown,
            }
        })
    }

    /// Run the MatMul-specific batched CROWN handler, including the attention
    /// identity retry path used by the experimental full-composition mode.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn handle_matmul_batched(
        &self,
        matmul: &MatMulLayer,
        node_name: &str,
        node_lb: &BatchedLinearBounds,
        node_bounds: &HashMap<String, BoundedTensor>,
        output_shape: &[usize],
        binary_inputs: BinaryInputBounds<'_>,
        attention_runtime: &mut AttentionCompositionRuntime<'_>,
        node_linear_bounds: &mut BatchedCrownAccumulator,
    ) -> Result<Option<CrownBackwardResult>> {
        let BinaryInputBounds {
            input_a_name,
            input_b_name,
            input_a_bounds,
            input_b_bounds,
        } = binary_inputs;
        let (lb_a, lb_b) = match matmul.propagate_linear_batched_binary(
            node_lb,
            input_a_bounds,
            input_b_bounds,
        ) {
            Ok(bounds) => bounds,
            Err(e) => {
                if !attention_runtime.should_retry_attention_identity() {
                    debug!(
                        "GraphNetwork batched CROWN: MatMul '{}' falling back to partial CROWN without attention retry ({})",
                        node_name, e
                    );
                    return Self::partial_crown_ibp_fallback(
                        node_lb,
                        node_name,
                        node_bounds,
                        output_shape,
                    )
                    .map(Some);
                }

                // First attempt failed. Check if this is an attention-shaped MatMul
                // where we can try using identity_for_attention for tighter bounds.
                let matmul_ibp_bounds = node_bounds.get(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "IBP bounds for MatMul node '{}' not found",
                        node_name
                    ))
                })?;
                let matmul_output_shape = matmul_ibp_bounds.shape();

                // Try attention-specific CROWN for attention-shaped outputs
                // [batch, heads, seq, seq] with seq <= 64 (memory limit from
                // identity_for_attention).
                if let Some(attention_identity) =
                    BatchedLinearBounds::identity_for_attention(matmul_output_shape)
                {
                    debug!(
                        "GraphNetwork batched CROWN: MatMul '{}' trying attention identity ({})",
                        node_name, e
                    );

                    // Retry with batched attention identity bounds.
                    // The identity carries [batch, heads] batch dims; BilinearCrownLayer
                    // tiles its flat McCormick coefficients to match via tile_to_batch,
                    // enabling full composition through the attention MatMul.
                    match matmul.propagate_linear_batched_binary(
                        &attention_identity,
                        input_a_bounds,
                        input_b_bounds,
                    ) {
                        Ok((lb_a_attn, lb_b_attn)) => {
                            // Experimental (#318): accumulate attention-identity
                            // bounds and continue backward. This gives CROWN
                            // propagation through the attention MatMul instead
                            // of concretizing with IBP at this boundary.
                            if let Some(used) = attention_runtime.full_composition_used.as_mut() {
                                **used = true;
                            }
                            debug!(
                                "GraphNetwork batched CROWN: MatMul '{}' attention full composition — accumulating bounds and continuing backward",
                                node_name
                            );
                            Self::accumulate_binary_bounds(
                                input_a_name,
                                lb_a_attn,
                                input_b_name,
                                lb_b_attn,
                                node_linear_bounds,
                            )?;
                            return Ok(None);
                        }
                        Err(e2) => {
                            debug!(
                                "GraphNetwork batched CROWN: MatMul '{}' attention CROWN also failed ({}), using partial CROWN",
                                node_name, e2
                            );
                        }
                    }
                } else {
                    debug!(
                        "GraphNetwork batched CROWN: MatMul '{}' not supported ({}), using partial CROWN",
                        node_name, e
                    );
                }

                // Partial CROWN: concretize the accumulated bounds using the
                // IBP bounds. partial_crown_fallback checks both IBP and node_lb
                // for Inf/NaN.
                return Self::partial_crown_ibp_fallback(
                    node_lb,
                    node_name,
                    node_bounds,
                    output_shape,
                )
                .map(Some);
            }
        };

        Self::accumulate_binary_bounds(input_a_name, lb_a, input_b_name, lb_b, node_linear_bounds)?;
        Ok(None)
    }

    /// Validate binary-op linear bounds for non-finite coefficients, then accumulate
    /// on success or fall back to partial CROWN.
    ///
    /// Returns `Ok(None)` on successful accumulation (caller continues the loop).
    /// Returns `Ok(Some(result))` on partial CROWN fallback (caller returns early).
    ///
    /// Deduplicates the MulBinary and BilinearCrown success-path validation pattern.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn validate_binary_bounds_and_accumulate(
        lb_a: BatchedLinearBounds,
        lb_b: BatchedLinearBounds,
        input_a_name: &str,
        input_b_name: &str,
        node_name: &str,
        label: &str,
        node_lb: &BatchedLinearBounds,
        node_bounds: &HashMap<String, BoundedTensor>,
        output_shape: &[usize],
        node_linear_bounds: &mut BatchedCrownAccumulator,
    ) -> Result<Option<CrownBackwardResult>> {
        let has_bad =
            Self::has_non_finite_coefficients(&lb_a) || Self::has_non_finite_coefficients(&lb_b);
        if has_bad {
            debug!(
                "GraphNetwork batched CROWN: {} '{}' produced inf/NaN, using partial CROWN",
                label, node_name
            );
            return Self::partial_crown_ibp_fallback(node_lb, node_name, node_bounds, output_shape)
                .map(Some);
        }
        debug!(
            "GraphNetwork batched CROWN: {} '{}' CROWN succeeded",
            label, node_name
        );
        Self::accumulate_binary_bounds(input_a_name, lb_a, input_b_name, lb_b, node_linear_bounds)?;
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::AttentionCompositionRuntime;

    #[test]
    fn test_attention_identity_retry_only_for_full_composition() {
        let production = AttentionCompositionRuntime::production();
        assert!(
            !production.should_retry_attention_identity(),
            "production batched MatMul fallback should skip the attention retry"
        );

        let full_composition = AttentionCompositionRuntime::full_composition();
        assert!(
            full_composition.should_retry_attention_identity(),
            "experimental full composition must keep the attention retry"
        );

        let mut used_attention_full_composition = false;
        let diagnostic = AttentionCompositionRuntime::full_composition_with_diagnostic(
            &mut used_attention_full_composition,
        );
        assert!(
            diagnostic.should_retry_attention_identity(),
            "diagnostic full composition must keep the attention retry"
        );
    }
}
