// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::bounds::LinearBounds;
use crate::layers::Layer;
use crate::network::core::graph::backward_helpers::{
    mask_linear_bounds_columns, where_constant_mask,
};
use crate::network::core::{
    apply_dense_backward_dispatch_result, crown_backward_step_patches, CrownStepResult,
    NETWORK_INPUT,
};
use crate::network::tighten_crown_output_with_provenance;
use crate::types::{BoundsProvenance, CrownBackwardResult, CrownIbpFallbackReason};
use crate::MulBinaryRelaxationMode;

use ndarray::Array2;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::{debug, info};

use super::super::core::GraphNetwork;
use super::helpers::is_softmax_decomposition_mul;
use super::spec_propagation::SpecCrownRequest;
use crate::network::CrownMergeAccumulator;

/// Extension trait for CROWN backward propagation on graph networks.
pub(crate) trait GraphNetworkCrownExt {
    fn crown_backward_with_relaxation(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<BoundedTensor>;

    /// CROWN backward propagation with explicit provenance metadata.
    ///
    /// Returns a [`CrownBackwardResult`] that indicates whether the output bounds
    /// came from actual CROWN backward propagation or were silently replaced with
    /// forward bounds due to invalid CROWN output (NaN/Inf or inverted intervals).
    fn crown_backward_with_relaxation_and_provenance(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<CrownBackwardResult>;

    /// CROWN backward propagation with deadline enforcement (#3398).
    ///
    /// When `deadline` is `Some`, checks elapsed time at each node in the backward
    /// loop. If the deadline is exceeded, falls back to IBP (always sound, cheap).
    /// This prevents graph CROWN backward from exceeding the verification timeout
    /// for large models (e.g., 900s+ overruns on VNN-COMP categories).
    fn crown_backward_with_relaxation_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
    ) -> Result<CrownBackwardResult>;

    fn crown_backward_with_relaxation_and_deadline_and_truncation(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
        crown_backward_layers: Option<usize>,
    ) -> Result<CrownBackwardResult>;

    /// Like [`Self::crown_backward_with_relaxation_and_deadline_and_truncation`]
    /// but reuses caller-precollected intermediate node bounds instead of
    /// running the internal Step-1 collection (#dedup-root-collections Fix B).
    ///
    /// `precollected_node_bounds` must be a valid enclosure map for the SAME
    /// input box, covering every graph node (any CROWN-IBP / forward-linear /
    /// IBP collection over `input` qualifies; an extra `NETWORK_INPUT` entry
    /// is ignored). When `Some`, the pre-collection deadline gate is also
    /// skipped — the bounds are already paid for, so falling back to vacuous
    /// IBP before even starting the backward pass would discard them. The
    /// per-node deadline checks inside the backward loop remain in force
    /// (sound IBP fallback on true budget exhaustion). Passing `None` is
    /// byte-for-byte the legacy behavior.
    fn crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
        crown_backward_layers: Option<usize>,
        precollected_node_bounds: Option<&std::collections::HashMap<String, BoundedTensor>>,
    ) -> Result<CrownBackwardResult>;

    fn crown_backward_specs_with_relaxation(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<BoundedTensor>;

    #[cfg(test)]
    fn crown_backward_specs_linear_with_relaxation(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<(BoundedTensor, Option<LinearBounds>)>;
}

impl GraphNetworkCrownExt for GraphNetwork {
    fn crown_backward_with_relaxation(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<BoundedTensor> {
        self.crown_backward_with_relaxation_and_deadline_and_truncation(
            input,
            engine,
            mul_binary_relaxation,
            None,
            None,
        )
        .map(|result| result.bounds)
    }

    fn crown_backward_with_relaxation_and_provenance(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<CrownBackwardResult> {
        self.crown_backward_with_relaxation_and_deadline_and_truncation(
            input,
            engine,
            mul_binary_relaxation,
            None,
            None,
        )
    }

    fn crown_backward_with_relaxation_and_deadline(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
    ) -> Result<CrownBackwardResult> {
        self.crown_backward_with_relaxation_and_deadline_and_truncation(
            input,
            engine,
            mul_binary_relaxation,
            deadline,
            None,
        )
    }

    fn crown_backward_with_relaxation_and_deadline_and_truncation(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
        crown_backward_layers: Option<usize>,
    ) -> Result<CrownBackwardResult> {
        self.crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
            input,
            engine,
            mul_binary_relaxation,
            deadline,
            crown_backward_layers,
            None,
        )
    }

    fn crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
        &self,
        input: &BoundedTensor,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
        deadline: Option<Instant>,
        crown_backward_layers: Option<usize>,
        precollected_node_bounds: Option<&std::collections::HashMap<String, BoundedTensor>>,
    ) -> Result<CrownBackwardResult> {
        // Disable the L2/Cauchy–Schwarz lever for the entire fixed-slope CROWN
        // backward scope (this is the single chokepoint all the public
        // `crown_backward_with_relaxation*` variants funnel through, and the
        // deadline/empty fallbacks below call `propagate_ibp` from inside it).
        // The CROWN-IBP intermediate forward passes collected here skip the
        // per-pass lever work. Sound (lever only tightens); restored on drop.
        let _l2_lever_off = crate::l2_lever_gate::L2LeverGuard::disabled();
        if self.nodes.is_empty() {
            return Ok(CrownBackwardResult {
                bounds: input.clone(),
                provenance: BoundsProvenance::Crown,
            });
        }

        // Get execution order
        let exec_order = self.exec_order()?;
        let plan = self.dispatch_plan()?;

        // Whether this graph family qualifies for CROWN-IBP intermediates —
        // pure function of the graph; also names the provenance label below.
        let use_crown_ibp = self.should_use_crown_ibp_intermediates();

        // Step 1: Bounds at each node for nonlinear relaxations.
        //
        // #dedup-root-collections Fix B: when the caller already holds a valid
        // same-box enclosure map (e.g., the DAG alpha init reference bounds —
        // previously this function re-collected the IDENTICAL map, ~73 s of
        // dead work per root episode on vggnet16_2022), reuse it and skip both
        // the internal collection and the pre-collection deadline gate (that
        // gate only protects the collection cost; discarding paid-for bounds
        // for vacuous IBP would lose the root objective). The per-node
        // deadline checks in the backward loop below remain in force.
        let collected_node_bounds;
        let node_bounds: &std::collections::HashMap<String, BoundedTensor> = if let Some(bounds) =
            precollected_node_bounds
        {
            bounds
        } else {
            // Deadline check before expensive CROWN-IBP collection (#3398).
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    info!("GraphNetwork DAG-CROWN: deadline exceeded before CROWN-IBP collection, falling back to IBP");
                    return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                        bounds,
                        provenance: BoundsProvenance::ForwardFallback(
                            CrownIbpFallbackReason::DeadlineExceeded,
                        ),
                    });
                }
            }

            // - CNN-style DAGs: use expensive CROWN-IBP intermediates for much tighter ReLU relaxations.
            // - Transformer-style graphs: use IBP forward bounds (includes transformer-specific tightening).
            let use_per_node_crown_ibp = self.should_collect_per_node_crown_ibp_intermediates();
            // Conv-DAG forward-linear intermediates (#vnncomp-image-forward-linear):
            // same policy/flag as the alpha reference collection and spec setup.
            // Every alpha iteration's CROWN pass re-entered the O(L²) per-node
            // CROWN-IBP repair here (measured 42s on cifar100 — alpha finished 0/20
            // iterations); the cached certified forward pass is free after its
            // first computation. Fail-closed to the existing selection.
            let forward_linear_bounds = {
                let conv_dag = self.has_conv_layers()
                    && self
                        .exec_order()
                        .map(|order| !self.is_sequential_graph(order))
                        .unwrap_or(false);
                if conv_dag && GraphNetwork::forward_linear_reference_enabled() {
                    match self.collect_forward_linear_bounds_dag_cached(input, engine, deadline) {
                        Ok(bounds) => {
                            info!(
                            "GraphNetwork DAG-CROWN: forward-linear intermediates (conv DAG, cached)"
                        );
                            Some((*bounds).clone())
                        }
                        Err(
                            error @ (NyError::UnsupportedOp(_)
                            | NyError::UnsupportedConfiguration(_)
                            | NyError::DeadlineExceeded(_)
                            | NyError::ShapeMismatch { .. }
                            | NyError::CpuMemoryExceeded { .. }),
                        ) => {
                            info!(
                                "GraphNetwork DAG-CROWN: forward-linear intermediates unavailable \
                             ({error}); falling back (fail-closed)"
                            );
                            None
                        }
                        Err(error) => return Err(error),
                    }
                } else {
                    None
                }
            };
            collected_node_bounds = if let Some(bounds) = forward_linear_bounds {
                bounds
            } else if use_per_node_crown_ibp {
                // Pass deadline to CROWN-IBP collection so the O(N²) per-node backward
                // passes respect the overall verification timeout. Without this, large
                // CNN DAGs (e.g., metaroom 6cnn_ry_49_8, 49 layers) can spend 13+
                // minutes in CROWN-IBP despite a 210s timeout. Fixed in #3397.
                self.collect_crown_ibp_bounds_dag_with_status_and_deadline(input, deadline, engine)?
                    .bounds
            } else {
                if use_crown_ibp {
                    info!(
                    "GraphNetwork DAG-CROWN: {} nodes exceeds per-node CROWN-IBP threshold {}, using IBP intermediates for final backward pass",
                    self.nodes.len(),
                    crate::network::core::graph::CROWN_IBP_PER_NODE_THRESHOLD
                );
                }
                // Keep DAG CROWN relaxation intermediates on the scalar IBP path.
                // `collect_node_bounds_with_engine` feeds Linear IBP through the GEMM
                // engine, which accumulates in f32 instead of the scalar path's f64
                // dot products (`layers/linear/ibp.rs`). That precision loss is enough
                // to flip the short-seq talker CROWN canary from Verified to
                // Unknown(BoundsTooLoose) (#4219).
                self.collect_node_bounds(input)?
            };
            &collected_node_bounds
        };

        // Determine output node and dimension
        let output_node_name = plan.name_of(plan.output_node_idx);
        debug_assert_eq!(plan.index_of(output_node_name), Some(plan.output_node_idx));

        let output_bounds = node_bounds.get(output_node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!("Output node {} not found", output_node_name))
        })?;
        let output_dim = output_bounds.len();
        let output_shape = output_bounds.shape().to_vec();

        debug!(
            "GraphNetwork DAG-CROWN: Starting backward propagation from {} outputs",
            output_dim
        );

        // Pre-build node lookup vector (eliminates self.nodes HashMap lookups in hot loop).
        // Pattern from DAG alpha-CROWN backward/mod.rs:193-200.
        let nodes_by_idx: Vec<&_> = plan
            .exec_order
            .iter()
            .map(|&idx| {
                self.nodes
                    .get(plan.name_of(idx))
                    .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", idx)))
            })
            .collect::<Result<Vec<_>>>()?;

        // Step 2: Initialize linear bounds per node
        // Each node tracks the accumulated linear bounds from all its consumers.
        // Phase 1b (#2613): Use CrownBounds to support Patches mode for CNN DAGs.
        // When the output is 3D spatial with Conv2d layers, start in Patches mode.
        // Accumulation at merge points converts to Dense via ensure_dense().
        let mut node_linear_bounds = CrownMergeAccumulator::new_indexed(exec_order);

        // Output node starts with identity bounds — Patches when spatial + Conv2d
        // and use_patches_mode is enabled. Matrix mode (use_patches_mode=false) forces
        // Dense throughout, matching the reference conv_mode='matrix' policy.
        // Reference: abcrown.py:228-231 — matrix when cuts enabled.
        let has_conv2d = plan.has_conv2d;
        let use_patches_seed = output_shape.len() == 3 && has_conv2d && self.use_patches_mode;
        // #margin-subset-seed (#margin-subset-alpha): when the initial-bounds
        // scope published the spec-referenced OUTPUT indices (single-margin
        // specs on wide heads, e.g. vggnet16 `(>= Y_200 Y_177)` over 1000
        // outputs), seed ONLY the k referenced identity rows. Each row is
        // bit-identical in semantics to its full-width counterpart
        // (row-independence: backward walk, per-row error term, and per-row
        // concretize are all row-local); the k concretized rows are scattered
        // over the output node's sound forward bounds at the exits below.
        // On vggnet16 the full-width seed materializes `[1000 x 401408]` conv
        // coefficient buffers (measured 119 GB anon-RSS, kernel-OOM) for 998
        // rows the verdict never reads; the k-row seed is ~500x smaller.
        // Unpublished scope (every caller outside the single-margin
        // initial-bounds computation) => `None` => byte-identical behavior.
        let margin_subset = if use_patches_seed {
            None
        } else {
            crate::output_margin_seed::margin_subset_indices(output_dim)
        };
        let initial_crown_bounds = if use_patches_seed {
            let (oc, oh, ow) = (output_shape[0], output_shape[1], output_shape[2]);
            debug!(
                "GraphNetwork DAG-CROWN: Initializing Patches mode (output {}x{}x{})",
                oc, oh, ow
            );
            CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
                (oc, oh, ow),
                (oc, oh, ow),
            )))
        } else if let Some(indices) = margin_subset.as_deref() {
            info!(
                "GraphNetwork DAG-CROWN: margin-subset OUTPUT seed engaged (k={} of {} rows)",
                indices.len(),
                output_dim
            );
            CrownBounds::Dense(LinearBounds::identity_rows(output_dim, indices))
        } else {
            CrownBounds::Dense(LinearBounds::identity(output_dim))
        };
        node_linear_bounds.insert(output_node_name.to_string(), initial_crown_bounds);

        // Number of rows the seed carries. Every downstream use of the former
        // `output_dim` in this walk denotes the SEED ROW COUNT (zero-coefficient
        // bias blocks, frontier concretization, accumulator hints) — never the
        // output node's width — so shadow it with the row count. Full-width
        // seeds keep `seed_rows == output_dim` (byte-identical).
        let seed_rows = margin_subset.as_deref().map_or(output_dim, <[usize]>::len);
        let output_dim = seed_rows;

        let input_dim = input.len();
        let mut input_accumulated = false;

        // Step 3: Propagate backward through nodes in reverse order.
        // Phase 1b (#2613): CrownBounds-aware dispatch. Single-input nodes in Patches
        // mode use crown_backward_step_patches for Conv2d/BN/activation/pool Patches
        // dispatch. Multi-input nodes and Patches-unsupported layers convert to Dense
        // via ensure_dense(). Accumulation at merge points is always Dense.

        // Per-node deadline budgeting (#3795): give each backward step a fraction of
        // the remaining budget instead of the full global deadline. Without this, a
        // single large Conv2d backward can consume the entire timeout, leaving zero
        // time for BaB domain exploration.
        //
        // Budget policy (matches crown_tighten.rs constants):
        //   per_node = max(remaining / nodes_remaining, remaining * 0.25)
        //   minimum floor = 2.0s (below this, bail to IBP immediately)
        const OUTPUT_CROWN_MAX_BUDGET_FRACTION: f64 = 0.25;
        const OUTPUT_CROWN_MIN_NODE_BUDGET_SECS: f64 = 2.0;
        let total_backward_nodes = plan.node_count();
        let mut backward_steps = 0usize;

        for (rev_pos, &idx) in plan.reverse_order.iter().enumerate() {
            let node_name = plan.name_of(idx);
            // Deadline enforcement: check before each node's backward pass (#3398).
            // For large models (e.g., relusplitter with 1094s+ overruns), a single
            // backward pass can exceed the entire verification budget. Checking at
            // each node gives O(node_count) granularity. Falls back to IBP which
            // is always sound (just looser). Matches spec_propagation.rs:188-194.
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    info!(
                        "GraphNetwork DAG-CROWN: deadline exceeded at node '{}', falling back to IBP",
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

            // Compute per-node deadline for this backward step (#3795).
            let node_deadline = super::backward_node_dispatch::compute_node_deadline(
                deadline,
                rev_pos,
                total_backward_nodes,
                OUTPUT_CROWN_MAX_BUDGET_FRACTION,
                OUTPUT_CROWN_MIN_NODE_BUDGET_SECS,
            );

            // If the overall deadline expires during budget calculation, bail to IBP for
            // the remaining backward pass. Sub-floor node shares keep the global deadline
            // so CROWN LinearBounds are preserved on short-budget tiny graphs (#3881).
            if deadline.is_some() && node_deadline.is_none() {
                info!(
                    "GraphNetwork DAG-CROWN: deadline expired while budgeting '{}' ({}/{} nodes), falling back to IBP",
                    node_name,
                    rev_pos + 1,
                    total_backward_nodes,
                );
                return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                    bounds,
                    provenance: BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::DeadlineExceeded,
                    ),
                });
            }

            if crown_backward_layers.is_some_and(|max_layers| backward_steps >= max_layers) {
                info!(
                    "GraphNetwork DAG-CROWN: truncating backward after {} nodes at frontier '{}'",
                    backward_steps, node_name
                );
                let final_bounds = self.concretize_crown_frontier_to_network_input(
                    &mut node_linear_bounds,
                    node_bounds,
                    output_dim,
                    input_dim,
                    &mut input_accumulated,
                )?;
                let crown_output = final_bounds.concretize_sound(input);
                // #margin-subset-seed: scatter the k computed rows over the
                // output node's sound forward bounds (full-width no-op).
                let crown_output = match margin_subset.as_deref() {
                    Some(indices) => crate::output_margin_seed::scatter_subset_bounds_over_base(
                        output_bounds,
                        indices,
                        &crown_output,
                    )?,
                    None => crown_output,
                };
                let crown_output = crown_output.reshape(&output_shape)?;
                let label = if use_crown_ibp {
                    "GraphNetwork DAG-CROWN (CROWN-IBP)"
                } else {
                    "GraphNetwork DAG-CROWN"
                };
                let (tightened, provenance) =
                    tighten_crown_output_with_provenance(crown_output, output_bounds, label)?;
                return Ok(CrownBackwardResult {
                    bounds: tightened,
                    provenance,
                });
            }

            // Direct Vec-indexed node lookup (#4296) — no HashMap access in hot loop.
            let node = nodes_by_idx[idx];

            // Get this node's accumulated CrownBounds via direct index (#4296).
            // We can move it out because reverse-topological traversal guarantees
            // all consumers have already contributed their bounds.
            let mut node_cb = match node_linear_bounds.take_by_idx(idx)? {
                Some(cb) => cb,
                None => {
                    debug!(
                        "GraphNetwork DAG-CROWN: node {} has no consumers, skipping",
                        node_name
                    );
                    continue;
                }
            };
            backward_steps += 1;

            // Use the plan's first-input route for shared pre-activation logic.
            let first_input_idx = plan.first_input_idx(idx);
            let first_input = plan.name_of(first_input_idx);
            let pre_activation = if plan.is_network_input(first_input_idx) {
                input
            } else {
                node_bounds.get(first_input).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        first_input
                    ))
                })?
            };

            // #3813: Dense→Patches re-entry at unary Conv2d boundaries.
            super::backward_node_dispatch::try_patches_reentry(
                &mut node_cb,
                node,
                node_bounds,
                node_name,
                self.use_patches_mode,
                "GraphNetwork DAG-CROWN",
            );

            let is_patches = matches!(&node_cb, CrownBounds::Patches(_));
            debug!(
                "GraphNetwork DAG-CROWN: backward through {} ({}) [{}]",
                node_name,
                node.layer.layer_type(),
                if is_patches { "Patches" } else { "Dense" }
            );

            // === Phase 1b Patches fast-path (#2613) ===
            // For single-input nodes in Patches mode, use the sequential Patches-aware
            // dispatch. This handles Conv2d, BatchNorm, activations (30 types), AvgPool,
            // MaxPool natively in Patches, and terminates to Dense at Linear/Flatten/Reshape.
            // Multi-input layers (MulBinary, Where, Add, etc.) always use Dense because
            // Patches accumulation at merge points is not yet supported (Phase 4).
            if is_patches && node.inputs.len() == 1 {
                match crown_backward_step_patches(
                    &node.layer,
                    &mut node_cb,
                    pre_activation,
                    engine,
                    0, // layer_idx not meaningful in graph
                    "DAG-CROWN",
                    deadline,
                ) {
                    Ok(CrownStepResult::Continue) => {
                        self.accumulate_crown_bounds_to_input(
                            first_input,
                            node_cb,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                        continue;
                    }
                    Ok(CrownStepResult::IbpFallback(fallback)) => {
                        if fallback.reason == CrownIbpFallbackReason::MemoryBudgetExceeded {
                            debug!(
                                "GraphNetwork DAG-CROWN: Patches dispatch for {} ({}) hit memory budget guard: {}; using IBP",
                                node_name,
                                node.layer.layer_type(),
                                fallback.details
                            );
                            return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                                bounds,
                                provenance: BoundsProvenance::ForwardFallback(fallback.reason),
                            });
                        }
                        debug!(
                            "GraphNetwork DAG-CROWN: Patches dispatch for {} ({}) \
                             requested IBP fallback, trying Dense",
                            node_name,
                            node.layer.layer_type()
                        );
                        // Fall through to Dense dispatch below
                    }
                    Err(error @ NyError::CpuMemoryExceeded { .. }) => {
                        debug!(
                            "GraphNetwork DAG-CROWN: Patches dispatch for {} ({}) hit memory budget guard: {}; using IBP",
                            node_name,
                            node.layer.layer_type(),
                            error
                        );
                        return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                            bounds,
                            provenance: BoundsProvenance::ForwardFallback(
                                CrownIbpFallbackReason::MemoryBudgetExceeded,
                            ),
                        });
                    }
                    Err(e) => {
                        debug!(
                            "GraphNetwork DAG-CROWN: Patches dispatch for {} ({}) failed: {}, \
                             falling back to Dense",
                            node_name,
                            node.layer.layer_type(),
                            e
                        );
                        // Fall through to Dense dispatch below
                    }
                }
                // Patches dispatch didn't handle this layer — ensure Dense for below.
                if matches!(&node_cb, CrownBounds::Patches(_)) {
                    match node_cb.ensure_dense() {
                        Ok(_) => {}
                        Err(e) => {
                            debug!(
                                "GraphNetwork DAG-CROWN: ensure_dense failed at {}: {}, IBP fallback",
                                node_name, e
                            );
                            return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                                bounds,
                                provenance: BoundsProvenance::ForwardFallback(
                                    CrownIbpFallbackReason::CrownPropagationError,
                                ),
                            });
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
                &mut node_linear_bounds,
                output_dim,
                input_dim,
                &mut input_accumulated,
                "DAG-CROWN",
            )? {
                continue;
            }

            // === Dense dispatch ===
            // Convert CrownBounds to LinearBounds for existing dispatch logic.
            // For multi-input nodes or after Patches fallback, this is the main path.
            let node_lb = node_cb.into_dense()?;

            // Handle site-specific layers first (MulBinary, Where, Linear dimension
            // check), then route all other layers through the shared dispatch core
            // (#1949 Step B). This eliminates ~400 LOC of duplicated match arms.

            // === Linear: pre-dispatch dimension check with IBP fallback (#2817) ===
            if super::backward_node_dispatch::linear_dimension_mismatch(node, &node_lb) {
                return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                    bounds,
                    provenance: BoundsProvenance::ForwardFallback(
                        CrownIbpFallbackReason::ShapeMismatch,
                    ),
                });
            }

            // === ReLU: heuristic relaxation via shared dispatch (#3935) ===
            if matches!(&node.layer, Layer::ReLU(_)) {
                use super::backward_node_dispatch::{dispatch_relu_backward, NodeDispatchResult};
                match dispatch_relu_backward(
                    self.cut_fold_scope(),
                    node,
                    &node_lb,
                    pre_activation,
                    node_name,
                    "GraphNetwork DAG-CROWN",
                    None,
                    None,
                )? {
                    NodeDispatchResult::SingleDense(bounds) => {
                        self.accumulate_dense_bounds_to_input(
                            first_input,
                            *bounds,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                    }
                    NodeDispatchResult::IbpFallback(reason) => {
                        return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                            bounds,
                            provenance: BoundsProvenance::ForwardFallback(reason),
                        });
                    }
                }
                continue;
            }

            // === MulBinary: site-specific (relaxation mode, softmax decomposition, IBP fallback) ===
            if matches!(&node.layer, Layer::MulBinary(_)) {
                use super::backward_node_dispatch::{
                    dispatch_mul_binary_backward, MulBinaryDispatchCtx, MulBinaryDispatchResult,
                };

                let (input_a_name, input_b_name) = node.require_binary_inputs()?;
                let input_a_bounds = self.bounds_ref(input_a_name, input, node_bounds)?;
                let input_b_bounds = self.bounds_ref(input_b_name, input, node_bounds)?;

                let dispatch_ctx = MulBinaryDispatchCtx {
                    node,
                    node_name,
                    node_lb: &node_lb,
                    input_a_bounds,
                    input_b_bounds,
                    mul_binary_relaxation,
                    mul_binary_alpha: None,
                    softmax_decomposition: is_softmax_decomposition_mul(self, node),
                    label: "GraphNetwork DAG-CROWN",
                };
                match dispatch_mul_binary_backward(&dispatch_ctx)? {
                    MulBinaryDispatchResult::BinaryDense {
                        bounds_a,
                        bounds_b,
                        bias_lower,
                        bias_upper,
                    } => {
                        Self::accumulate_bias_to_network_input_crown(
                            &bias_lower,
                            &bias_upper,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                        );
                        self.accumulate_dense_bounds_to_input(
                            input_a_name,
                            *bounds_a,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                        self.accumulate_dense_bounds_to_input(
                            input_b_name,
                            *bounds_b,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                    }
                    MulBinaryDispatchResult::SoftmaxNonFinite => {
                        return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                            bounds,
                            provenance: BoundsProvenance::ForwardFallback(
                                CrownIbpFallbackReason::CrownPropagationError,
                            ),
                        });
                    }
                    MulBinaryDispatchResult::RecoverableError(err) => {
                        debug!(
                            "GraphNetwork DAG-CROWN: MulBinary '{}' {:?} failed ({}), falling back to IBP",
                            node_name, mul_binary_relaxation, err,
                        );
                        return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                            bounds,
                            provenance: BoundsProvenance::ForwardFallback(
                                CrownIbpFallbackReason::CrownPropagationError,
                            ),
                        });
                    }
                }
                continue;
            }

            // === Div: site-specific (positive-denominator reciprocal scaling) ===
            if matches!(&node.layer, Layer::Div(_)) {
                use super::backward_node_dispatch::{backward_div_to_numerator, DivBackwardResult};

                let (input_a_name, input_b_name) = node.require_binary_inputs()?;
                let input_a_bounds = self.bounds_ref(input_a_name, input, node_bounds)?;
                let input_b_bounds = self.bounds_ref(input_b_name, input, node_bounds)?;
                let node_output_bounds = node_bounds.get(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Div output bounds for {} not found during DAG-CROWN",
                        node_name
                    ))
                })?;

                match backward_div_to_numerator(
                    &node_lb,
                    input_a_bounds,
                    input_b_bounds,
                    node_output_bounds,
                )? {
                    DivBackwardResult::PropagateNumerator(bounds) => {
                        self.accumulate_dense_bounds_to_input(
                            input_a_name,
                            *bounds,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                    }
                    DivBackwardResult::ConcretizeCurrentNode(bias) => {
                        Self::accumulate_bias_to_network_input_crown(
                            &bias.lower,
                            &bias.upper,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                        );
                    }
                }
                continue;
            }

            // === Where: site-specific (ternary conditional with concretization) ===
            if let Layer::Where(where_layer) = &node.layer {
                let where_bounds = node_bounds.get(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Where output bounds for {} not found during DAG-CROWN",
                        node_name
                    ))
                })?;

                // === Embedded-constant Where (single `cond` input; both branches
                // are constants). The output is a constant vector w.r.t. the network
                // input — no linear dependence on `cond` — so the EXACT CROWN backward
                // folds the entire output into the bias and routes zero to `cond`.
                // `embedded_constant_select_output` returns the exact per-element
                // select when `cond` is constant (tighter than IBP) and the sound
                // IBP union otherwise. require_ternary_inputs would error here because
                // the node has only 1 input.
                if where_layer.has_embedded_constants() {
                    let cond_input = node.require_unary_input().map_err(|_| {
                        NyError::InvalidSpec(format!(
                            "Where node {} with embedded constants requires 1 input (condition)",
                            node_name
                        ))
                    })?;
                    let cond_bounds = self.bounds_ref(cond_input, input, node_bounds)?;
                    let select = where_layer.embedded_constant_select_output(cond_bounds)?;
                    let concrete = node_lb.concretize_checked(&select)?;
                    let (lower_b, upper_b) =
                        concrete.flatten_to_ix1("graph-crown embedded-constant Where")?;
                    let zeros = Array2::<f32>::zeros((output_dim, input_dim));
                    let const_lb =
                        LinearBounds::new_or_conservative(zeros.clone(), lower_b, zeros, upper_b)?;
                    self.accumulate_dense_bounds_to_input(
                        NETWORK_INPUT,
                        const_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    )?;
                    continue;
                }

                let (cond_input, true_input, false_input) = node.require_ternary_inputs()?;
                let cond_bounds = self.bounds_ref(cond_input, input, node_bounds)?;
                let cond_all_true = cond_bounds.lower().iter().all(|&v| v >= 0.5);
                let cond_all_false = cond_bounds.upper().iter().all(|&v| v <= 0.5);

                if cond_all_true {
                    self.accumulate_dense_bounds_to_input(
                        true_input,
                        node_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    )?;
                    continue;
                } else if cond_all_false {
                    self.accumulate_dense_bounds_to_input(
                        false_input,
                        node_lb,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                    )?;
                    continue;
                }

                // === Exact per-element select for a bound-independent (constant)
                // condition mask (#Where-const-cond). When the condition is fixed
                // (lower == upper elementwise), Where degenerates to a fixed 0/1
                // mask: output[i] = true_input[i] if mask[i] else false_input[i].
                // This is an EXACT linear transform — route each output column to
                // the correct branch by zeroing the other branch's columns. The
                // generic mixed fallback below would instead concretize the whole
                // tensor (loose IBP), so we prefer this exact split.
                if let Some(mask) = where_constant_mask(cond_bounds) {
                    debug_assert_eq!(mask.len(), node_lb.num_inputs());
                    if mask.len() == node_lb.num_inputs() {
                        let true_lb = mask_linear_bounds_columns(&node_lb, &mask, true);
                        let false_lb = mask_linear_bounds_columns(&node_lb, &mask, false);
                        self.accumulate_dense_bounds_to_input(
                            true_input,
                            true_lb,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                        self.accumulate_dense_bounds_to_input(
                            false_input,
                            false_lb,
                            &mut node_linear_bounds,
                            output_dim,
                            input_dim,
                            &mut input_accumulated,
                        )?;
                        continue;
                    }
                }

                let concrete = node_lb.concretize_checked(where_bounds)?;
                let (lower_b, upper_b) =
                    concrete.flatten_to_ix1("graph-crown Where mixed fallback")?;

                let zeros = Array2::<f32>::zeros((output_dim, input_dim));
                let const_lb =
                    LinearBounds::new_or_conservative(zeros.clone(), lower_b, zeros, upper_b)?;

                self.accumulate_dense_bounds_to_input(
                    NETWORK_INPUT,
                    const_lb,
                    &mut node_linear_bounds,
                    output_dim,
                    input_dim,
                    &mut input_accumulated,
                )?;
                continue;
            }

            // === All other layers: shared dispatch core (#1949 Step B, #3935) ===
            use super::backward_node_dispatch::{
                dispatch_shared_core, SharedDispatchCtx, SharedDispatchResult,
            };
            let shared_ctx = SharedDispatchCtx {
                node,
                node_name,
                node_lb: &node_lb,
                pre_activation,
                network_input: input,
                node_bounds,
                engine,
                node_deadline,
                mul_binary_relaxation,
                label: "GraphNetwork DAG-CROWN",
            };
            let shared_result = match dispatch_shared_core(&shared_ctx) {
                Ok(result) => result,
                Err(error @ NyError::CpuMemoryExceeded { .. }) => {
                    debug!(
                        "GraphNetwork DAG-CROWN: dense dispatch for {} ({}) hit memory budget guard: {}; using IBP",
                        node_name,
                        node.layer.layer_type(),
                        error
                    );
                    return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                        bounds,
                        provenance: BoundsProvenance::ForwardFallback(
                            CrownIbpFallbackReason::MemoryBudgetExceeded,
                        ),
                    });
                }
                Err(error) => return Err(error),
            };
            match shared_result {
                SharedDispatchResult::Dispatch(result) => {
                    apply_dense_backward_dispatch_result(
                        self,
                        node,
                        first_input,
                        &node_lb,
                        *result,
                        &mut node_linear_bounds,
                        output_dim,
                        input_dim,
                        &mut input_accumulated,
                        "Dispatch",
                    )?;
                }
                SharedDispatchResult::IbpFallback(reason) => {
                    return self.propagate_ibp(input).map(|bounds| CrownBackwardResult {
                        bounds,
                        provenance: BoundsProvenance::ForwardFallback(reason),
                    });
                }
            }
        }

        // Step 4: Concretize final bounds.
        // Convert CrownBounds to Dense for concretization.
        let final_cb = node_linear_bounds
            .take(NETWORK_INPUT)?
            .ok_or_else(|| NyError::InvalidSpec("No path to network input found".to_string()))?;
        let final_bounds = final_cb.into_dense()?;

        debug!(
            "GraphNetwork DAG-CROWN: Concretizing {} outputs from {} inputs",
            final_bounds.num_outputs(),
            final_bounds.num_inputs()
        );
        let crown_output = final_bounds.concretize_sound(input);
        // #margin-subset-seed: scatter the k computed rows over the output
        // node's sound forward bounds (full-width no-op). Every row of the
        // scattered result is a valid enclosure; the tighten below intersects
        // with the forward bounds exactly as for a full-width map.
        let crown_output = match margin_subset.as_deref() {
            Some(indices) => crate::output_margin_seed::scatter_subset_bounds_over_base(
                output_bounds,
                indices,
                &crown_output,
            )?,
            None => crown_output,
        };
        let crown_output = crown_output.reshape(&output_shape)?;

        // Post-concretization tightening with provenance — shared with all CROWN paths (#3043).
        let label = if use_crown_ibp {
            "GraphNetwork DAG-CROWN (CROWN-IBP)"
        } else {
            "GraphNetwork DAG-CROWN"
        };
        let (tightened, provenance) =
            tighten_crown_output_with_provenance(crown_output, output_bounds, label)?;

        Ok(CrownBackwardResult {
            bounds: tightened,
            provenance,
        })
    }

    fn crown_backward_specs_with_relaxation(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<BoundedTensor> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .mul_binary_relaxation(mul_binary_relaxation)
            .run()
    }

    #[cfg(test)]
    fn crown_backward_specs_linear_with_relaxation(
        &self,
        input: &BoundedTensor,
        spec_matrix: &Array2<f32>,
        engine: Option<&dyn GemmEngine>,
        mul_binary_relaxation: MulBinaryRelaxationMode,
    ) -> Result<(BoundedTensor, Option<LinearBounds>)> {
        SpecCrownRequest::new(self, input, spec_matrix, engine)
            .mul_binary_relaxation(mul_binary_relaxation)
            .run_with_linear()
    }
}
