// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN propagation for GraphNetwork.
//!
//! This module contains the large `propagate_crown_batched` method that preserves
//! tensor shape structure throughout propagation. Essential for transformer models
//! where operations like attention have cross-position interactions.
//!
//! Phase 4 (#2613): Uses `BatchedCrownBounds` to support Patches mode for CNN DAGs.
//! Conv2d layers stay in sparse Patches representation; other layers ensure Dense.

use std::borrow::Cow;
use std::collections::HashMap;
use std::time::Instant;

use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, info};

use crate::bounds::patches::PatchesLinearBounds;
use crate::bounds::patches_batched::BatchedCrownBounds;
use crate::bounds::BatchedLinearBounds;
use crate::layers::Layer;
use crate::network::core::graph::batched_accumulator::BatchedCrownAccumulator;
use crate::network::crown_memory::check_batched_identity_budget;
use crate::network::tighten_crown_output_with_provenance;
use crate::types::{BoundsProvenance, CrownBackwardResult, CrownIbpFallbackReason};
use crate::MulBinaryRelaxationMode;

use super::{GraphNetwork, NETWORK_INPUT};

#[path = "crown_batched/binary_ops.rs"]
mod binary_ops;
#[path = "crown_batched/dispatch.rs"]
mod dispatch;
#[cfg(test)]
#[path = "crown_batched/dispatch_special_tests.rs"]
mod dispatch_special_tests;
#[cfg(test)]
#[path = "crown_batched/dispatch_tests.rs"]
mod dispatch_tests;
#[path = "crown_batched/entrypoints.rs"]
mod entrypoints;

use binary_ops::AttentionCompositionRuntime;
use dispatch::{PatchesDispatchResult, SpecialBatchedDispatchResult};

/// Resolve a node name to its IBP bounds, using the network input for `NETWORK_INPUT`.
///
/// Deduplicates the 9× repeated pattern in `propagate_crown_batched_inner`:
/// ```ignore
/// let bounds = if name == NETWORK_INPUT { input } else { node_bounds.get(name)? };
/// ```
fn resolve_node_bounds<'a>(
    name: &str,
    input: &'a BoundedTensor,
    node_bounds: &'a HashMap<String, BoundedTensor>,
    context: &str,
) -> Result<&'a BoundedTensor> {
    if name == NETWORK_INPUT {
        Ok(input)
    } else {
        node_bounds
            .get(name)
            .ok_or_else(|| NyError::InvalidSpec(format!("{} '{}' not found", context, name)))
    }
}

impl GraphNetwork {
    /// Inner implementation of batched CROWN propagation.
    ///
    /// Shared by `propagate_crown_batched_with_relaxation`,
    /// `propagate_crown_batched_with_relaxation_and_deadline`, and
    /// `propagate_crown_batched_with_engine_relaxation_and_deadline`, and
    /// `propagate_crown_batched_with_bilinear_alphas`.
    ///
    /// The `engine` parameter threads GPU GEMM acceleration into the engine-aware
    /// batched backward dispatch. Layers opt in incrementally; today the graph
    /// loop can use it for Linear and Conv2d batched CROWN paths.
    ///
    /// Phase 3 (#4297): Uses `CrownDispatchPlan` for Vec-indexed node access
    /// and `BatchedCrownAccumulator` for Vec-indexed bounds storage, eliminating
    /// HashMap lookups from the hot backward loop.
    fn propagate_crown_batched_inner(
        &self,
        input: &BoundedTensor,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        bilinear_alphas: Option<&HashMap<String, ndarray::Array4<f32>>>,
        deadline: Option<Instant>,
        engine: Option<&dyn GemmEngine>,
        mut attention_runtime: AttentionCompositionRuntime<'_>,
    ) -> Result<CrownBackwardResult> {
        if self.nodes.is_empty() {
            return Ok(CrownBackwardResult {
                bounds: input.clone(),
                provenance: BoundsProvenance::Crown,
            });
        }

        // Get dispatch plan for indexed access (#4297).
        let plan = self.dispatch_plan()?;

        if deadline.is_some_and(|d| Instant::now() >= d) {
            info!("Batched CROWN: deadline exceeded before intermediate bounds collection, falling back to IBP");
            return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                bounds,
                provenance: BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::DeadlineExceeded,
                ),
            });
        }

        // Fast-fail before the (potentially O(N²) per-node CROWN-IBP) intermediate
        // collection when the graph contains a `SelfAttention` node.
        //
        // `SelfAttention`'s batched CROWN backward is categorically unimplemented:
        // `Layer::propagate_crown_backward_batched` returns `UnsupportedOp` for it
        // unconditionally (layer_enum/dispatch.rs). In any transformer-shaped graph
        // the attention node is reached with other pending backward paths (the
        // residual `Add` always creates ≥2 paths), so `dispatch_batched_unary`
        // propagates that `UnsupportedOp` rather than taking the single-path partial
        // -CROWN fallback — i.e. the batched backward is *guaranteed* to return
        // `UnsupportedOp`. Today that error surfaces only AFTER the full intermediate
        // -bound collection (measured ~7s on the 4-block SVTR encoder), which is then
        // discarded when the caller (alpha-/fixed-slope CROWN fallback chain) re-runs
        // the bound on the dense `crown_backward_with_relaxation` path that DOES handle
        // `SelfAttention` (via IBP concretization at the attention boundary).
        //
        // Returning the SAME `UnsupportedOp` here, before the collection, is a pure
        // fast-fail: the caller's fallback is unchanged and the produced bound is
        // bit-identical — only the wasted intermediate collection is skipped. The
        // experimental full-composition lane represents attention as `MatMul` nodes
        // (handled by the binary dispatch), not as a fused `SelfAttention` layer, so
        // it is unaffected by this guard.
        if self
            .nodes
            .values()
            .any(|node| matches!(node.layer, Layer::SelfAttention(_)))
        {
            return Err(NyError::UnsupportedOp(
                "SelfAttention batched CROWN backward not implemented".to_string(),
            ));
        }

        // Step 1: Collect bounds at each node for nonlinear relaxations.
        //
        // Batched CROWN is primarily used for transformer-style graphs; avoid CROWN-IBP
        // intermediate tightening unless the graph is CNN-style and supported.
        let use_crown_ibp = self.should_use_crown_ibp_intermediates();
        let use_per_node_crown_ibp = self.should_collect_per_node_crown_ibp_intermediates();
        let node_bounds = if use_per_node_crown_ibp {
            self.collect_crown_ibp_bounds_dag_with_deadline_and_engine(input, deadline, engine)?
        } else {
            if use_crown_ibp {
                info!(
                    "GraphNetwork batched CROWN: {} nodes exceeds per-node CROWN-IBP threshold {}, using IBP intermediates for final backward pass",
                    self.nodes.len(),
                    crate::network::core::graph::CROWN_IBP_PER_NODE_THRESHOLD
                );
            }
            self.collect_node_bounds_with_engine(input, engine)?
        };

        let has_conv2d = plan.has_conv2d;

        // Determine output node and shape via dispatch plan (#4297).
        let output_node_name = plan.name_of(plan.output_node_idx);

        let output_bounds = node_bounds.get(output_node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;
        let output_shape = output_bounds.shape().to_vec();

        // Compute flattened output shape for Dense fallback and partial_crown_fallback.
        let batched_output_shape = if has_conv2d {
            if output_shape.len() < 3 {
                output_shape.clone()
            } else {
                let mut flat_shape = output_shape[..output_shape.len() - 3].to_vec();
                let flat_dim = checked_shape_product(&output_shape[output_shape.len() - 3..])
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "Graph batched CROWN: Conv2d output spatial dims overflow: {:?}",
                            &output_shape[output_shape.len() - 3..]
                        ))
                    })?;
                flat_shape.push(flat_dim);
                flat_shape
            }
        } else {
            output_shape.clone()
        };

        debug!(
            "GraphNetwork batched CROWN: Starting backward propagation from {:?}",
            batched_output_shape
        );

        // Step 2: Initialize batched linear bounds per node.
        // Phase 4 (#2613): Use BatchedCrownBounds to support Patches mode for CNN DAGs.
        // When the output is 3D spatial with Conv2d layers, start in Patches mode.
        // Phase 3 (#4297): Use BatchedCrownAccumulator for Vec-indexed storage.
        let mut node_linear_bounds = BatchedCrownAccumulator::new(plan);

        let initial_bounds = if output_shape.len() == 3 && has_conv2d {
            let (oc, oh, ow) = (output_shape[0], output_shape[1], output_shape[2]);
            debug!(
                "GraphNetwork batched CROWN: Initializing Patches mode (output {}x{}x{})",
                oc, oh, ow
            );
            BatchedCrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
                (oc, oh, ow),
                (oc, oh, ow),
            )))
        } else {
            // Guard: check CPU dense budget before allocating batched identity (#3550).
            check_batched_identity_budget(
                "GraphNetwork::propagate_crown_batched",
                &batched_output_shape,
            )?;
            BatchedCrownBounds::Dense(BatchedLinearBounds::identity(&batched_output_shape)?)
        };
        let initial_bounds_mem = initial_bounds.memory_bytes();
        node_linear_bounds.insert(output_node_name, initial_bounds);

        // Track peak memory across all live bounds for Patches diagnostics (#2613).
        let mut peak_memory_bytes: usize = initial_bounds_mem;

        // Pre-build node lookup vector (#4297) — no HashMap access in hot loop.
        let nodes_by_idx: Vec<&_> = plan
            .exec_order
            .iter()
            .map(|&idx| {
                self.nodes.get(plan.name_of(idx)).ok_or_else(|| {
                    NyError::InvalidSpec(format!("Node not found: {}", plan.name_of(idx)))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Step 3: Propagate backward through nodes in reverse order (#4297: index-based).
        for &idx in plan.reverse_order.iter() {
            let node_name = plan.name_of(idx);

            // Deadline check at each node in the backward loop (#3398).
            // If exceeded, fall back to IBP which is always sound.
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    info!(
                        "Batched CROWN: deadline exceeded at node '{}', falling back to IBP",
                        node_name
                    );
                    return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                        bounds,
                        provenance: BoundsProvenance::ForwardFallback(
                            CrownIbpFallbackReason::DeadlineExceeded,
                        ),
                    });
                }
            }

            // Direct Vec-indexed node lookup (#4297) — no HashMap access.
            let node = nodes_by_idx[idx];

            // Get this node's accumulated bounds via direct index (#4297).
            // We can move it out because reverse-topological traversal guarantees
            // all consumers have already contributed their bounds.
            let node_bcb = match node_linear_bounds.take_idx(idx) {
                Some(bcb) => bcb,
                None => {
                    // Node has no consumers (not output, not used by anyone)
                    continue;
                }
            };

            // Get pre-activation bounds for this node.
            // Use first input (not require_unary_input) because multi-input nodes
            // like Concat, Div, Sub, and Where resolve their full input sets in
            // their specific handlers below. The pre-activation from the first
            // input is used for Patches dispatch and the shared core; multi-input
            // nodes skip Patches (guarded by node.inputs.len() == 1 in
            // try_conv2d_patches_or_dense) and their handlers resolve their own
            // inputs. Fix: #4136 (regression from #4097 tightening
            // require_unary_input to reject multi-input). Mirrors #4113 fix in
            // DAG-CROWN (graph_crown/propagation.rs:360-385).
            let first_input_idx = plan.first_input_idx(idx);
            let first_input = plan.name_of(first_input_idx);
            let pre_activation = if plan.is_network_input(first_input_idx) {
                input
            } else {
                node_bounds.get(first_input).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Pre-activation bounds for '{}' not found",
                        first_input
                    ))
                })?
            };

            // Phase 4 Conv2d Patches fast-path (#2613) or Dense conversion.
            let node_lb = match Self::try_conv2d_patches_or_dense(
                node_bcb,
                &node.layer,
                node.inputs.len(),
                node_name,
                pre_activation,
                engine,
                first_input,
                &node_bounds,
                &output_shape,
                &mut node_linear_bounds,
                &mut peak_memory_bytes,
            )? {
                PatchesDispatchResult::Handled => continue,
                PatchesDispatchResult::PartialFallback(result) => return Ok(*result),
                PatchesDispatchResult::DenseBounds(lb) => *lb,
            };

            // Keep binary/special graph operators local; delegate all unary dispatch to Layer.
            match self.dispatch_special_batched_operator(
                node,
                node_name,
                &node_lb,
                input,
                &node_bounds,
                &output_shape,
                mul_binary_relaxation,
                bilinear_alphas,
                &mut attention_runtime,
                &mut node_linear_bounds,
            )? {
                SpecialBatchedDispatchResult::Handled => continue,
                SpecialBatchedDispatchResult::PartialFallback(fallback) => return Ok(*fallback),
                SpecialBatchedDispatchResult::NotHandled => {}
            }

            // All non-binary layers: delegate to dispatch_batched_unary for
            // unary trait dispatch. Exhaustive — no catch-all (#3424).
            //
            // Elementwise activations:
            match &node.layer {
                Layer::ReLU(_) | Layer::GELU(_) | Layer::SiLU(_) | Layer::Tanh(_)
                | Layer::Sigmoid(_) | Layer::Exp(_) | Layer::Log(_) | Layer::Sqrt(_)
                | Layer::Reciprocal(_) | Layer::Softplus(_) | Layer::HardSwish(_)
                | Layer::Mish(_) | Layer::Selu(_) | Layer::Softsign(_) | Layer::Arctan(_)
                | Layer::Tan(_) | Layer::Sin(_) | Layer::Cos(_) | Layer::Elu(_)
                | Layer::Celu(_) | Layer::LeakyReLU(_) | Layer::HardSigmoid(_)
                | Layer::Clip(_) | Layer::ThresholdedRelu(_) | Layer::Abs(_)
                | Layer::PowConstant(_) | Layer::Floor(_) | Layer::Ceil(_)
                | Layer::Round(_) | Layer::Trunc(_) | Layer::Sign(_) | Layer::PRelu(_) | Layer::Shrink(_)
                | Layer::Snake(_) | Layer::Compare(_)
                // Softmax family:
                | Layer::Softmax(_) | Layer::CausalSoftmax(_) | Layer::LogSoftmax(_)
                | Layer::LogSumExp(_)
                // Normalization:
                | Layer::LayerNorm(_) | Layer::RmsNorm(_) | Layer::InstanceNorm1d(_)
                | Layer::GroupNorm(_) | Layer::AdaIN1d(_) | Layer::BatchNorm(_)
                // Constant arithmetic:
                | Layer::AddConstant(_) | Layer::MulConstant(_) | Layer::DivConstant(_)
                | Layer::SubConstant(_)
                // Reductions:
                | Layer::ReduceMean(_) | Layer::ReduceSum(_) | Layer::CumSum(_)
                | Layer::ReduceMax(_) | Layer::ReduceMin(_)
                | Layer::Topk(_) | Layer::ArgMax(_) | Layer::ArgMin(_) | Layer::ArgSort(_)
                // Linear / convolutions:
                | Layer::Linear(_) | Layer::Conv1d(_) | Layer::Conv2d(_)
                | Layer::ConvTranspose1d(_) | Layer::ConvTranspose2d(_)
                // Shape transforms:
                | Layer::Flatten(_) | Layer::Reshape(_) | Layer::Transpose(_)
                | Layer::Tile(_) | Layer::Gather(_) | Layer::ScatterAdd(_) | Layer::IndexAdd(_)
                | Layer::ScatterNd(_) | Layer::Pad(_)
                | Layer::Resize(_) | Layer::Slice(_)
                | Layer::Squeeze(_)
                | Layer::Unsqueeze(_)
                | Layer::QdqPerturbation(_)
                // Pooling:
                | Layer::AveragePool(_) | Layer::MaxPool2d(_)
                // Positional encoding:
                | Layer::RoPE(_)
                // Binary ops without explicit batched handler (dispatch_batched_unary
                // handles via propagate_crown_backward_batched → partial CROWN fallback):
                | Layer::Sub(_) | Layer::Div(_) | Layer::Atan2(_) | Layer::Concat(_)
                | Layer::MinBinary(_)
                | Layer::MaxBinary(_) | Layer::Where(_)
                // Comparison:
                | Layer::CompareTensor(_)
                // Special / data-dependent:
                | Layer::NonZero(_) | Layer::SelfAttention(_) | Layer::SkipMerge(_)
                | Layer::OpaqueSkip(_) => {
                    let layer = &node.layer;
                    if let Some(fallback) = Self::dispatch_batched_unary(
                        layer,
                        node_name,
                        &node_lb,
                        pre_activation,
                        engine,
                        first_input,
                        &node_bounds,
                        &output_shape,
                        &mut node_linear_bounds,
                    )? {
                        return Ok(CrownBackwardResult {
                            bounds: fallback,
                            provenance: BoundsProvenance::Crown,
                        });
                    }
                }
                Layer::ExpandLikeLastAxis(_)
                | Layer::Add(_)
                | Layer::MatMul(_)
                | Layer::MulBinary(_)
                | Layer::BilinearCrown(_) => unreachable!(
                    "special batched dispatch should have handled {}",
                    node.layer.layer_type()
                ),
            }
        }

        // Compute total live memory at end of backward pass.
        let final_live_memory: usize = node_linear_bounds.total_memory_bytes();
        peak_memory_bytes = peak_memory_bytes.max(final_live_memory);
        debug!(
            "GraphNetwork batched CROWN: backward complete — peak memory: {:.1} MB",
            peak_memory_bytes as f64 / 1_048_576.0
        );

        // Step 4: Concretize using input bounds.
        // Convert BatchedCrownBounds to BatchedLinearBounds for concretization.
        let final_bcb = node_linear_bounds
            .take(NETWORK_INPUT)
            .ok_or_else(|| NyError::InvalidSpec("No path to network input found".to_string()))?;
        let final_bounds =
            final_bcb.into_batched_dense_checked("crown_batched:final_concretization")?;

        // Check if final bounds coefficients contain inf/NaN
        if Self::has_non_finite_coefficients(&final_bounds) {
            debug!(
                "GraphNetwork batched CROWN: final linear bounds contain inf/NaN, falling back to IBP"
            );
            let ibp = self.propagate_ibp(input)?;
            return Ok(CrownBackwardResult {
                bounds: Self::sanitize_bounds_for_fallback(&ibp),
                provenance: BoundsProvenance::ForwardFallback(
                    CrownIbpFallbackReason::CrownPropagationError,
                ),
            });
        }

        debug!(
            "GraphNetwork batched CROWN: Concretizing with input shape {:?}, output shape {:?}",
            final_bounds.input_shape, final_bounds.output_shape
        );

        let input_for_concretize = if input.shape() == final_bounds.input_shape.as_slice() {
            Cow::Borrowed(input)
        } else {
            Cow::Owned(input.reshape(&final_bounds.input_shape)?)
        };
        // concretize_sound() guarantees no NaN/inversion (#2287).
        let crown_output = final_bounds.concretize_sound(input_for_concretize.as_ref())?;
        // Post-concretization tightening with provenance — matches graph CROWN (#4240, #4242).
        let (tightened_output, provenance) =
            tighten_crown_output_with_provenance(crown_output, output_bounds, "Batched CROWN")?;

        // Ensure output shape matches expected
        if tightened_output.shape() != output_shape.as_slice() {
            tightened_output
                .reshape(&output_shape)
                .map(|bounds| CrownBackwardResult { bounds, provenance })
        } else {
            Ok(CrownBackwardResult {
                bounds: tightened_output,
                provenance,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, array};
    use ny_tensor::BoundedTensor;

    use super::GraphNetwork;
    use crate::layers::{AddLayer, Layer, LinearLayer};
    use crate::network::core::graph::GraphNode;

    #[test]
    fn test_batched_crown_indexed_pending_bounds_matches_fixed_slope_add_merge_4297() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "branch_a",
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[1.0_f32])))
                    .expect("branch_a linear should construct"),
            ),
        ));
        graph.add_node(GraphNode::from_input(
            "branch_b",
            Layer::Linear(
                LinearLayer::new(arr2(&[[2.0_f32]]), Some(arr1(&[-0.5_f32])))
                    .expect("branch_b linear should construct"),
            ),
        ));
        graph.add_node(GraphNode::binary(
            "sum",
            Layer::Add(AddLayer),
            "branch_a",
            "branch_b",
        ));
        graph.set_output("sum");

        let input = BoundedTensor::new(array![-1.0_f32].into_dyn(), array![2.0_f32].into_dyn())
            .expect("input bounds should construct");

        let batched = graph
            .propagate_crown_batched(&input)
            .expect("batched CROWN should succeed");
        let fixed_slope = graph
            .propagate_crown_fixed_slope(&input)
            .expect("fixed-slope CROWN should succeed");

        assert_eq!(
            batched.lower(),
            fixed_slope.lower(),
            "#4297 regression: indexed pending bounds must preserve lower bounds across Add merges"
        );
        assert_eq!(
            batched.upper(),
            fixed_slope.upper(),
            "#4297 regression: indexed pending bounds must preserve upper bounds across Add merges"
        );
        assert!(
            (batched.lower()[[0]] - (-2.5)).abs() < 1e-5,
            "branch_a + branch_b lower should be ~-2.5, got {}",
            batched.lower()[[0]]
        );
        assert!(
            (batched.upper()[[0]] - 6.5).abs() < 1e-5,
            "branch_a + branch_b upper should be ~6.5, got {}",
            batched.upper()[[0]]
        );
    }
}
