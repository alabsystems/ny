// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward propagation within a single transformer block.
//!
//! Starts with identity bounds at the block's output and propagates backward
//! through the block's nodes. Binary ops (MatMul, MulBinary) trigger partial
//! CROWN fallback. LayerNorm and RmsNorm dispatch to decomposed backward.
//!
//! Part of #3221, #3447.

use std::collections::HashMap;
use std::time::Instant;

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::bounds::patches_batched::BatchedCrownBounds;
use crate::bounds::BatchedLinearBounds;
use crate::layers::normalization::decomposed::{
    decomposed_instance_norm_crown_backward_channel_batched, decomposed_norm_crown_backward,
    decomposed_rms_norm_crown_backward,
};
use crate::layers::Layer;
use crate::network::crown_memory::check_batched_identity_budget;
use crate::network::tighten_crown_output_with_provenance_and_deadline;
use crate::types::{BoundsProvenance, CrownBackwardResult, CrownIbpFallbackReason};

use super::{BlockAlphaState, LayerNormValidationStats};
use crate::network::core::graph::GraphNetwork;

fn has_non_finite_block_bounds_with_deadline(
    bounds: &BatchedLinearBounds,
    deadline: Option<Instant>,
) -> Result<bool> {
    let Some(limit) = deadline else {
        return Ok(GraphNetwork::has_non_finite_coefficients(bounds));
    };
    if Instant::now() >= limit {
        return Err(NyError::DeadlineExceeded(
            "block CROWN: deadline exceeded before final coefficient scan".into(),
        ));
    }
    let mut work = 0usize;
    for values in [
        bounds.lower_a(),
        bounds.lower_b(),
        bounds.upper_a(),
        bounds.upper_b(),
    ] {
        for &value in values {
            if !value.is_finite() {
                if Instant::now() >= limit {
                    return Err(NyError::DeadlineExceeded(
                        "block CROWN: deadline exceeded during final coefficient scan".into(),
                    ));
                }
                return Ok(true);
            }
            work += 1;
            if work >= 4096 {
                work = 0;
                if Instant::now() >= limit {
                    return Err(NyError::DeadlineExceeded(
                        "block CROWN: deadline exceeded during final coefficient scan".into(),
                    ));
                }
            }
        }
    }
    if Instant::now() >= limit {
        return Err(NyError::DeadlineExceeded(
            "block CROWN: deadline exceeded after final coefficient scan".into(),
        ));
    }
    Ok(false)
}

/// Constant context for the block-wise CROWN backward loop.
///
/// Groups parameters that are invariant across all nodes in a single
/// block backward pass, keeping `apply_ibp_fallback_at_node` within
/// clippy's argument limit. Part of #3812.
struct BlockBackwardCtx<'a> {
    label: &'a str,
    block_node_bounds: &'a HashMap<String, BoundedTensor>,
    block_input: &'a BoundedTensor,
    output_shape: &'a [usize],
    block_input_idx: usize,
}

/// Block-local indexed pending bounds for the block-wise backward loop.
///
/// Packet D (#4298) only needs Dense propagation within a block, but the
/// carrier stores `BatchedCrownBounds` so merges can reuse the existing
/// checked Dense merge path without reimplementing the f64 sidecar logic.
struct BlockIndexedPendingBounds {
    bounds_by_idx: Vec<Option<BatchedCrownBounds>>,
    deadline: Option<Instant>,
}

impl BlockIndexedPendingBounds {
    #[cfg(test)]
    fn new(slot_count: usize) -> Self {
        Self::new_with_deadline(slot_count, None)
    }

    fn new_with_deadline(slot_count: usize, deadline: Option<Instant>) -> Self {
        Self {
            bounds_by_idx: std::iter::repeat_with(|| None).take(slot_count).collect(),
            deadline,
        }
    }

    fn insert(&mut self, idx: usize, bounds: BatchedCrownBounds) {
        self.bounds_by_idx[idx] = Some(bounds);
    }

    fn take(&mut self, idx: usize) -> Option<BatchedCrownBounds> {
        self.bounds_by_idx[idx].take()
    }

    fn accumulate_dense(
        &mut self,
        idx: usize,
        new_bounds: impl Into<BatchedLinearBounds>,
    ) -> Result<()> {
        self.accumulate(idx, BatchedCrownBounds::Dense(new_bounds.into()))
    }

    fn accumulate(&mut self, idx: usize, new_bounds: BatchedCrownBounds) -> Result<()> {
        if let Some(existing) = self.bounds_by_idx[idx].as_mut() {
            let new_blb = new_bounds.into_batched_dense_checked_with_deadline(
                "crown_block_wise:indexed_pending:new",
                self.deadline,
            )?;
            existing.merge_dense_checked_with_deadline(
                new_blb,
                "crown_block_wise:indexed_pending:existing",
                self.deadline,
            )?;
        } else {
            self.bounds_by_idx[idx] = Some(new_bounds);
        }
        Ok(())
    }
}

impl GraphNetwork {
    fn resolve_block_input_idx(
        inputs: &[String],
        index: usize,
        block_name_to_idx: &HashMap<String, usize>,
        block_input_idx: usize,
    ) -> Result<usize> {
        let input_name = inputs.get(index).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Node has {} inputs, expected at least {}",
                inputs.len(),
                index + 1
            ))
        })?;
        Ok(block_name_to_idx
            .get(input_name)
            .copied()
            .unwrap_or(block_input_idx))
    }

    fn finalize_block_crown_output(
        final_lb: BatchedLinearBounds,
        block_input: &BoundedTensor,
        output_bounds: &BoundedTensor,
        label: &str,
        deadline: Option<Instant>,
    ) -> Result<CrownBackwardResult> {
        let deadline_fallback = || CrownBackwardResult {
            bounds: output_bounds.clone(),
            provenance: BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded),
        };
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Ok(deadline_fallback());
        }
        let has_non_finite = match has_non_finite_block_bounds_with_deadline(&final_lb, deadline) {
            Err(error) if error.is_deadline_exceeded() => return Ok(deadline_fallback()),
            result => result?,
        };
        if has_non_finite {
            debug!(
                "Per-block {}: non-finite coefficients in final bounds, using IBP fallback",
                label
            );
            return Ok(CrownBackwardResult {
                bounds: output_bounds.clone(),
                provenance: BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::CrownPropagationError,
                ),
            });
        }
        let result = match final_lb.concretize_sound_with_deadline(block_input, deadline) {
            Err(error) if error.is_deadline_exceeded() => return Ok(deadline_fallback()),
            result => result?,
        };
        // #4242: upgrade to provenance-aware tightening for consistency with all
        // other CROWN orchestrators. Detects inverted bounds in addition to NaN.
        let (tightened, provenance) = match tighten_crown_output_with_provenance_and_deadline(
            result,
            output_bounds,
            label,
            deadline,
        ) {
            Err(error) if error.is_deadline_exceeded() => return Ok(deadline_fallback()),
            result => result?,
        };
        debug!(
            "Per-block {}: tightening provenance = {:?}",
            label, provenance
        );
        Ok(CrownBackwardResult {
            bounds: tightened,
            provenance,
        })
    }

    /// Run CROWN backward propagation within a single block (test-only convenience).
    ///
    /// Production code should use `crown_backward_within_block_with_engine` directly
    /// so the engine parameter is not silently discarded (#3772).
    #[cfg(test)]
    pub(crate) fn crown_backward_within_block(
        &self,
        block_nodes: &[String],
        block_node_bounds: &HashMap<String, BoundedTensor>,
        block_input: &BoundedTensor,
    ) -> Result<(
        BoundedTensor,
        Vec<LayerNormValidationStats>,
        BoundsProvenance,
    )> {
        self.crown_backward_within_block_with_engine(
            block_nodes,
            block_node_bounds,
            block_input,
            None,
            None,
            None,
        )
    }

    /// Engine-aware variant of `crown_backward_within_block` (#3597).
    ///
    /// When `alpha_state` is `Some`, GELU layers dispatch through per-neuron
    /// alpha parameters for tighter lower bounds (alpha-CROWN). When `None`,
    /// GELU uses standard CROWN relaxation. Part of #3447.
    pub(crate) fn crown_backward_within_block_with_engine(
        &self,
        block_nodes: &[String],
        block_node_bounds: &HashMap<String, BoundedTensor>,
        block_input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        alpha_state: Option<&BlockAlphaState>,
        deadline: Option<Instant>,
    ) -> Result<(
        BoundedTensor,
        Vec<LayerNormValidationStats>,
        BoundsProvenance,
    )> {
        let label = if alpha_state.is_some() {
            "alpha-CROWN"
        } else {
            "CROWN"
        };
        let budget_label = if alpha_state.is_some() {
            "GraphNetwork::crown_backward_within_block(alpha-CROWN)"
        } else {
            "GraphNetwork::crown_backward_within_block"
        };
        let dispatch_label = if alpha_state.is_some() {
            "crown_block_wise(alpha-CROWN):node_dispatch"
        } else {
            "crown_block_wise:node_dispatch"
        };
        let final_label = if alpha_state.is_some() {
            "crown_block_wise(alpha-CROWN):final_concretization"
        } else {
            "crown_block_wise:final_concretization"
        };

        if block_nodes.is_empty() {
            return Err(NyError::InvalidSpec(format!(
                "Per-block {}: empty block",
                label
            )));
        }

        let block_name_to_idx: HashMap<String, usize> = block_nodes
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, node_name)| (node_name, idx))
            .collect();
        let block_input_idx = block_nodes.len();
        let nodes_by_idx = block_nodes
            .iter()
            .map(|node_name| {
                self.nodes
                    .get(node_name)
                    .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", node_name)))
            })
            .collect::<Result<Vec<_>>>()?;
        let bounds_by_idx = block_nodes
            .iter()
            .map(|node_name| {
                block_node_bounds.get(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Per-block {}: IBP bounds not found for node '{}'",
                        label, node_name
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Output node of the block.
        let output_node_name = block_nodes
            .last()
            .ok_or_else(|| NyError::InvalidSpec("Empty block".to_string()))?;
        let output_bounds = block_node_bounds.get(output_node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Per-block {}: IBP bounds not found for output node '{}'",
                label, output_node_name
            ))
        })?;
        let output_shape = output_bounds.shape().to_vec();

        let ctx = BlockBackwardCtx {
            label,
            block_node_bounds,
            block_input,
            output_shape: &output_shape,
            block_input_idx,
        };

        // Initialize identity bounds at block output.
        // Guard: check CPU dense budget before allocating (#3550).
        check_batched_identity_budget(budget_label, &output_shape)?;
        let mut linear_bounds =
            BlockIndexedPendingBounds::new_with_deadline(block_input_idx + 1, deadline);
        let output_node_idx = *block_name_to_idx.get(output_node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Per-block {}: output node '{}' missing from block-local index",
                label, output_node_name
            ))
        })?;
        linear_bounds.insert(
            output_node_idx,
            BatchedCrownBounds::Dense(BatchedLinearBounds::identity(&output_shape)?),
        );

        let mut norm_stats: Vec<LayerNormValidationStats> = Vec::new();

        // Backward through block nodes in reverse topological order.
        for node_idx in (0..block_nodes.len()).rev() {
            let node_name = block_nodes[node_idx].as_str();
            if deadline.is_some_and(|d| Instant::now() >= d) {
                debug!(
                    "Per-block {}: deadline exceeded at node '{}', using IBP fallback",
                    label, node_name
                );
                let provenance =
                    BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded);
                return Ok((output_bounds.clone(), norm_stats, provenance));
            }
            let node_bcb = match linear_bounds.take(node_idx) {
                Some(bcb) => bcb,
                None => continue, // No accumulated bounds for this node.
            };

            let node = nodes_by_idx[node_idx];

            // Convert to Dense for dispatch.
            let node_lb =
                node_bcb.into_batched_dense_checked_with_deadline(dispatch_label, deadline)?;

            // Get first input name, mapping outside-block references to the sentinel.
            let first_input_idx = Self::resolve_block_input_idx(
                &node.inputs,
                0,
                &block_name_to_idx,
                block_input_idx,
            )?;
            let pre_activation = if first_input_idx == block_input_idx {
                block_input
            } else {
                bounds_by_idx[first_input_idx]
            };

            match &node.layer {
                // Add: backward is identity to both inputs (d(a+b)/da = I, d(a+b)/db = I).
                Layer::Add(add) => {
                    let (lb_a, lb_b) =
                        add.propagate_linear_batched_binary(&node_lb).map_err(|e| {
                            NyError::InternalError(format!(
                                "Per-block {}: Add backward failed for '{}': {}",
                                label, node_name, e
                            ))
                        })?;

                    let add_a_idx = Self::resolve_block_input_idx(
                        &node.inputs,
                        0,
                        &block_name_to_idx,
                        block_input_idx,
                    )?;
                    let add_b_idx = Self::resolve_block_input_idx(
                        &node.inputs,
                        1,
                        &block_name_to_idx,
                        block_input_idx,
                    )?;

                    linear_bounds.accumulate_dense(add_a_idx, lb_a)?;
                    linear_bounds.accumulate_dense(add_b_idx, lb_b)?;
                }

                // LayerNorm: attempt decomposed CROWN backward through primitives.
                // Decomposes into x→mean→d=x-mean→d²→var→sqrt→1/std→d*inv_std→γ·norm+β
                // and propagates CROWN backward through each. If the decomposition
                // fails or produces non-finite coefficients, fall back to
                // interval-derived bias-only bounds. Part of #318.
                //
                // Reference: alpha-beta-CROWN auto_LiRPA/operators/normalization.py:303-331
                Layer::LayerNorm(ln) => {
                    match decomposed_norm_crown_backward(
                        &node_lb,
                        &ln.ny,
                        &ln.beta,
                        ln.eps,
                        pre_activation,
                        ln.forward_mode,
                    ) {
                        Ok(result) if !Self::has_non_finite_coefficients(&result.bounds) => {
                            debug!(
                                "Per-block {}: decomposed norm '{}' succeeded",
                                label, node_name
                            );
                            norm_stats.push(LayerNormValidationStats {
                                node_name: node_name.to_string(),
                                fallback_rows: result.validation.fallback_rows,
                                total_rows: result.validation.total_rows,
                            });
                            linear_bounds.accumulate_dense(first_input_idx, result.bounds)?;
                        }
                        other => {
                            match &other {
                                Ok(_) => {
                                    debug!(
                                        "Per-block {}: decomposed norm '{}' produced non-finite, falling back to IBP",
                                        label, node_name
                                    );
                                }
                                Err(e) => {
                                    debug!(
                                        "Per-block {}: decomposed norm '{}' failed ({}), falling back to IBP",
                                        label, node_name, e
                                    );
                                }
                            }
                            self.apply_ibp_fallback_at_node(
                                node_name,
                                &node_lb,
                                &ctx,
                                &mut linear_bounds,
                            )?;
                        }
                    }
                }

                // RmsNorm: decomposed CROWN backward through primitives.
                // Decomposes into x→x²→mean(x²)→sqrt→1/rms→x*inv_rms→γ·norm
                // (simpler than LayerNorm: no mean subtraction, no beta).
                // Part of #3387.
                Layer::RmsNorm(rn) => {
                    match decomposed_rms_norm_crown_backward(
                        &node_lb,
                        &rn.ny,
                        rn.eps,
                        pre_activation,
                    ) {
                        Ok(result) if !Self::has_non_finite_coefficients(&result.bounds) => {
                            debug!(
                                "Per-block {}: decomposed RmsNorm '{}' succeeded",
                                label, node_name
                            );
                            norm_stats.push(LayerNormValidationStats {
                                node_name: node_name.to_string(),
                                fallback_rows: result.validation.fallback_rows,
                                total_rows: result.validation.total_rows,
                            });
                            linear_bounds.accumulate_dense(first_input_idx, result.bounds)?;
                        }
                        other => {
                            match &other {
                                Ok(_) => debug!(
                                    "Per-block {}: decomposed RmsNorm '{}' produced \
                                     non-finite, falling back to IBP",
                                    label, node_name
                                ),
                                Err(e) => debug!(
                                    "Per-block {}: decomposed RmsNorm '{}' failed \
                                     ({}), falling back to IBP",
                                    label, node_name, e
                                ),
                            }
                            self.apply_ibp_fallback_at_node(
                                node_name,
                                &node_lb,
                                &ctx,
                                &mut linear_bounds,
                            )?;
                        }
                    }
                }

                // InstanceNorm1d: decomposed CROWN backward through per-channel
                // primitives. Calls decomposed_norm_crown_backward once per channel
                // since InstanceNorm is LayerNorm applied per-channel. Part of #3830.
                Layer::InstanceNorm1d(inst) => {
                    match decomposed_instance_norm_crown_backward_channel_batched(
                        &node_lb,
                        &inst.ny,
                        &inst.beta,
                        inst.eps,
                        pre_activation,
                        inst.forward_mode,
                        inst.num_channels(),
                    ) {
                        Ok(result) if !Self::has_non_finite_coefficients(&result.bounds) => {
                            debug!(
                                "Per-block {}: decomposed InstanceNorm '{}' succeeded",
                                label, node_name
                            );
                            norm_stats.push(LayerNormValidationStats {
                                node_name: node_name.to_string(),
                                fallback_rows: result.validation.fallback_rows,
                                total_rows: result.validation.total_rows,
                            });
                            linear_bounds.accumulate_dense(first_input_idx, result.bounds)?;
                        }
                        other => {
                            match &other {
                                Ok(_) => debug!(
                                    "Per-block {}: decomposed InstanceNorm '{}' produced \
                                     non-finite, falling back to IBP",
                                    label, node_name
                                ),
                                Err(e) => debug!(
                                    "Per-block {}: decomposed InstanceNorm '{}' failed \
                                     ({}), falling back to IBP",
                                    label, node_name, e
                                ),
                            }
                            self.apply_ibp_fallback_at_node(
                                node_name,
                                &node_lb,
                                &ctx,
                                &mut linear_bounds,
                            )?;
                        }
                    }
                }

                // AdaIN1d: fixed-style InstanceNorm1d with effective affine
                // parameters. Reuse the same decomposed per-channel helper as
                // InstanceNorm instead of partial fallback. Part of #3912.
                Layer::AdaIN1d(adain) => {
                    match adain.effective_instance_norm().and_then(|effective| {
                        decomposed_instance_norm_crown_backward_channel_batched(
                            &node_lb,
                            &effective.ny,
                            &effective.beta,
                            effective.eps,
                            pre_activation,
                            effective.forward_mode,
                            effective.num_channels(),
                        )
                    }) {
                        Ok(result) if !Self::has_non_finite_coefficients(&result.bounds) => {
                            debug!(
                                "Per-block {}: decomposed AdaIN '{}' succeeded",
                                label, node_name
                            );
                            norm_stats.push(LayerNormValidationStats {
                                node_name: node_name.to_string(),
                                fallback_rows: result.validation.fallback_rows,
                                total_rows: result.validation.total_rows,
                            });
                            linear_bounds.accumulate_dense(first_input_idx, result.bounds)?;
                        }
                        other => {
                            match &other {
                                Ok(_) => debug!(
                                    "Per-block {}: decomposed AdaIN '{}' produced \
                                     non-finite, falling back to IBP",
                                    label, node_name
                                ),
                                Err(e) => debug!(
                                    "Per-block {}: decomposed AdaIN '{}' failed \
                                     ({}), falling back to IBP",
                                    label, node_name, e
                                ),
                            }
                            self.apply_ibp_fallback_at_node(
                                node_name,
                                &node_lb,
                                &ctx,
                                &mut linear_bounds,
                            )?;
                        }
                    }
                }

                // Other normalization layers and binary ops: partial CROWN fallback.
                // IMPORTANT: We do NOT return immediately. Other paths (e.g.,
                // a residual Add connection) may have already accumulated linear
                // bounds at the block-input accumulator. Returning here would lose those
                // contributions. Instead, we concretize this sub-path to a constant
                // interval and accumulate it as a bias-only contribution.
                Layer::BatchNorm(_)
                | Layer::GroupNorm(_)
                | Layer::MatMul(_)
                | Layer::MulBinary(_)
                | Layer::ExpandLikeLastAxis(_)
                | Layer::BilinearCrown(_) => {
                    debug!(
                        "Per-block {}: partial fallback at '{}' ({})",
                        label,
                        node_name,
                        node.layer.layer_type()
                    );
                    self.apply_ibp_fallback_at_node(node_name, &node_lb, &ctx, &mut linear_bounds)?;
                }

                // GELU: when alpha_state is provided, use per-neuron alpha
                // parameters for tighter lower bounds. Otherwise fall through
                // to standard CROWN backward. Part of #3447.
                Layer::GELU(gelu) if alpha_state.is_some() => {
                    // Match guard guarantees alpha_state is Some; use safe
                    // destructuring instead of expect() (#3812).
                    let state = match alpha_state {
                        Some(s) => s,
                        None => {
                            return Err(NyError::InternalError(
                                "alpha_state match guard violated".to_string(),
                            ));
                        }
                    };
                    let new_lb = if let Some(alphas) = state.gelu_alphas.get(node_name) {
                        gelu.propagate_linear_batched_with_bounds_and_alpha(
                            &node_lb,
                            pre_activation,
                            alphas,
                        )?
                    } else {
                        node.layer.propagate_crown_backward_batched(
                            &node_lb,
                            Some(pre_activation),
                            engine,
                        )?
                    };
                    linear_bounds.accumulate_dense(first_input_idx, new_lb)?;
                }

                // Unary layers (including GELU without alpha): standard CROWN backward.
                _ => {
                    match node.layer.propagate_crown_backward_batched(
                        &node_lb,
                        Some(pre_activation),
                        engine,
                    ) {
                        Ok(new_lb) => {
                            linear_bounds.accumulate_dense(first_input_idx, new_lb)?;
                        }
                        // #2888: NumericalInstability also triggers partial CROWN fallback.
                        // SoundnessRefusal must still propagate — see error.rs:109-110 (#3106).
                        //
                        // ShapeMismatch is also recoverable here: it arises when an
                        // upstream layer hands this layer bounds in an incompatible
                        // linear-coefficient representation (e.g. the channel-batched
                        // InstanceNorm decomposed backward emits a per-channel [out, T]
                        // matrix, but a Conv1d backward expects the flat [out, C*T]
                        // representation). Concretizing to the layer's sound IBP
                        // interval is conservative (it only widens bounds), so falling
                        // back is sound — it never loosens a real soundness check.
                        Err(
                            NyError::UnsupportedOp(reason)
                            | NyError::UnsupportedConfiguration(reason)
                            | NyError::NumericalInstability(reason),
                        ) => {
                            debug!(
                                "Per-block {}: unsupported/unstable layer '{}' ({}): {}, partial fallback",
                                label, node_name,
                                node.layer.layer_type(),
                                reason
                            );
                            self.apply_ibp_fallback_at_node(
                                node_name,
                                &node_lb,
                                &ctx,
                                &mut linear_bounds,
                            )?;
                        }
                        Err(e @ NyError::ShapeMismatch { .. }) => {
                            debug!(
                                "Per-block {}: representation-incompatible bounds at layer '{}' ({}): {}, partial fallback",
                                label,
                                node_name,
                                node.layer.layer_type(),
                                e
                            );
                            self.apply_ibp_fallback_at_node(
                                node_name,
                                &node_lb,
                                &ctx,
                                &mut linear_bounds,
                            )?;
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }

        // Concretize accumulated bounds at block input.
        let final_bcb = linear_bounds.take(block_input_idx).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Per-block {}: no accumulated bounds at block input sentinel",
                label
            ))
        })?;

        let final_lb = final_bcb.into_batched_dense_checked_with_deadline(final_label, deadline)?;

        let result = Self::finalize_block_crown_output(
            final_lb,
            block_input,
            output_bounds,
            label,
            deadline,
        )?;
        Ok((result.bounds, norm_stats, result.provenance))
    }

    /// Apply IBP fallback for a node: concretize to interval and accumulate
    /// as bias-only bounds at the block input sentinel.
    ///
    /// Used when a node cannot be propagated through analytically (unsupported
    /// layer, non-finite decomposed result, etc.). Deduplicates the 4
    /// identical fallback blocks in `crown_backward_within_block_with_engine`.
    /// Part of #3812.
    fn apply_ibp_fallback_at_node(
        &self,
        node_name: &str,
        node_lb: &BatchedLinearBounds,
        ctx: &BlockBackwardCtx<'_>,
        linear_bounds: &mut BlockIndexedPendingBounds,
    ) -> Result<()> {
        let node_ibp = ctx.block_node_bounds.get(node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Per-block {}: IBP bounds not found for '{}'",
                ctx.label, node_name
            ))
        })?;
        let fallback_bt = Self::partial_crown_fallback(node_lb, node_ibp, ctx.output_shape)?;
        let bias_only = Self::bias_only_bounds_from_interval(
            &fallback_bt,
            ctx.block_input.shape(),
            ctx.output_shape,
        )?;
        linear_bounds.accumulate_dense(ctx.block_input_idx, bias_only)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, array, ArrayD, IxDyn};
    use ny_tensor::{next_down_f32, BoundedTensor};
    use std::time::{Duration, Instant};

    use crate::bounds::BatchedLinearBounds;
    use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
    use crate::GraphNetwork;

    use super::BlockIndexedPendingBounds;

    fn scalar_batched_linear_bounds(value: f32) -> BatchedLinearBounds {
        BatchedLinearBounds::from_parts_unchecked(
            array![[value]].into_dyn(),
            array![value].into_dyn(),
            array![[value]].into_dyn(),
            array![value].into_dyn(),
            vec![1],
            vec![1],
        )
    }

    #[test]
    fn test_finalize_block_crown_output_nonfinite_coefficients_use_forward_fallback_4256() {
        let final_lb = BatchedLinearBounds::new(
            ArrayD::from_elem(IxDyn(&[1, 1]), f32::INFINITY),
            ArrayD::zeros(IxDyn(&[1])),
            ArrayD::from_elem(IxDyn(&[1, 1]), f32::INFINITY),
            ArrayD::zeros(IxDyn(&[1])),
            vec![1],
            vec![1],
        )
        .expect("infinite coefficients should remain constructible");
        let block_input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("block input should construct");
        let output_bounds =
            BoundedTensor::new(arr1(&[-0.25_f32]).into_dyn(), arr1(&[0.75_f32]).into_dyn())
                .expect("output bounds should construct");

        let result = GraphNetwork::finalize_block_crown_output(
            final_lb,
            &block_input,
            &output_bounds,
            "CROWN",
            None,
        )
        .expect("non-finite final coefficients should fall back, not error");

        assert_eq!(
            result.provenance,
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::CrownPropagationError),
            "non-finite final coefficients should surface CrownPropagationError fallback"
        );
        assert_eq!(
            result.bounds.lower(),
            output_bounds.lower(),
            "fallback lower bounds should reuse the provided forward output"
        );
        assert_eq!(
            result.bounds.upper(),
            output_bounds.upper(),
            "fallback upper bounds should reuse the provided forward output"
        );
    }

    #[test]
    fn test_finalize_block_crown_output_expired_deadline_is_terminal_forward_fallback() {
        let final_lb = scalar_batched_linear_bounds(1.0);
        let block_input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("block input should construct");
        let output_bounds =
            BoundedTensor::new(arr1(&[-0.25_f32]).into_dyn(), arr1(&[0.75_f32]).into_dyn())
                .expect("output bounds should construct");
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("one-second deadline subtraction");

        let result = GraphNetwork::finalize_block_crown_output(
            final_lb,
            &block_input,
            &output_bounds,
            "CROWN",
            Some(expired),
        )
        .expect("expired block finalization should use established forward fallback");

        assert_eq!(
            result.provenance,
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded)
        );
        assert_eq!(result.bounds.lower(), output_bounds.lower());
        assert_eq!(result.bounds.upper(), output_bounds.upper());
    }

    #[test]
    fn test_block_indexed_pending_bounds_merges_sentinel_slot_4298() {
        let mut pending = BlockIndexedPendingBounds::new(3);
        pending.insert(
            1,
            crate::bounds::patches_batched::BatchedCrownBounds::Dense(
                scalar_batched_linear_bounds(1.0),
            ),
        );

        pending
            .accumulate_dense(2, scalar_batched_linear_bounds(2.0))
            .expect("first sentinel insert should succeed");
        pending
            .accumulate_dense(2, scalar_batched_linear_bounds(3.0))
            .expect("sentinel merge should succeed");

        let sentinel = pending
            .take(2)
            .expect("sentinel slot should exist")
            .into_batched_dense_checked("test_block_indexed_pending_bounds_merges_sentinel_slot")
            .expect("sentinel slot should stay dense");
        let residual = pending
            .take(1)
            .expect("residual slot should exist")
            .into_batched_dense_checked("test_block_indexed_pending_bounds_merges_sentinel_slot")
            .expect("residual slot should stay dense");

        assert_eq!(
            sentinel.lower_a()[[0, 0]],
            next_down_f32(5.0),
            "indexed sentinel slot should merge lower A contributions"
        );
        assert_eq!(
            sentinel.lower_b()[0],
            next_down_f32(5.0),
            "indexed sentinel slot should merge lower bias contributions"
        );
        assert_eq!(
            residual.lower_a()[[0, 0]],
            1.0,
            "non-sentinel slots should remain independently addressable"
        );
    }
}
