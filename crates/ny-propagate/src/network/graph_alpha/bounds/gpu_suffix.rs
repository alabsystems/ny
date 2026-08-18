// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Seeded GPU suffix acceleration for per-target graph alpha/CROWN backward.

use super::target_backward_patches::resolve_preactivation;
use crate::bounds::patches::PatchesMaterializationPurpose;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::layers::Layer;
use crate::network::core::{
    extract_relu_gpu_layer_with_alpha, try_extract_single_gpu_layer, GraphNetwork,
    GraphTargetShapeContract, NETWORK_INPUT,
};
use crate::network::CrownMergeAccumulator;
use ndarray::{ArrayD, IxDyn};
use ny_core::{GemmEngine, GpuCrownLayer, GpuCrownSeed, NyError, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

// ---------------------------------------------------------------------------
// Precomputed GPU suffix plan (#4340)
// ---------------------------------------------------------------------------

/// One entry in the precomputed GPU suffix plan. Stores the extracted
/// `GpuCrownLayer` for this node and the name of its unary input (or `None`
/// when the input is `NETWORK_INPUT`, i.e. the chain terminates).
struct GpuSuffixPlanNode {
    next_input: Option<String>,
    layer: GpuCrownLayer,
}

/// A target-local plan that records which nodes form GPU-eligible unary tails.
///
/// Built once per `propagate_crown_to_node_core` call in O(N) topological
/// time. The hot backward loop then consults the plan for O(1) eligibility
/// checks instead of re-walking the suffix at every step.
pub(super) struct GpuSuffixPlan {
    entries: HashMap<String, GpuSuffixPlanNode>,
}

pub(super) fn take_only_gpu_layer(mut gpu_layers: Vec<GpuCrownLayer>) -> Option<GpuCrownLayer> {
    if gpu_layers.len() == 1 {
        gpu_layers.pop()
    } else {
        None
    }
}

impl GpuSuffixPlan {
    /// Build the plan by walking `relevant_nodes` in topological (forward)
    /// order. A node is eligible if:
    /// 1. it is unary (exactly one input)
    /// 2. it has no active non-ReLU alpha (S-shaped/Sqrt)
    /// 3. its layer can be extracted to a `GpuCrownLayer`
    /// 4. its unary input is either `NETWORK_INPUT` or already in the plan
    pub(super) fn build(
        relevant_nodes: &[String],
        graph: &GraphNetwork,
        input: &BoundedTensor,
        crown_bounds: &HashMap<String, BoundedTensor>,
        ibp_bounds: &HashMap<String, BoundedTensor>,
        alpha_state: Option<&GraphAlphaState>,
    ) -> Self {
        let mut entries = HashMap::new();

        for node_name in relevant_nodes {
            // Guard 1: no S-shaped/sqrt alpha on GPU
            if suffix_has_active_non_relu_alpha(node_name, alpha_state) {
                continue;
            }

            let Some(node) = graph.nodes.get(node_name) else {
                continue;
            };

            // Guard 2: must be unary
            if node.inputs().len() != 1 {
                continue;
            }

            let Ok(input_name) = node.require_unary_input() else {
                continue;
            };

            // Guard 3: input must be NETWORK_INPUT or already eligible
            let terminates = input_name == NETWORK_INPUT;
            if !terminates && !entries.contains_key(input_name) {
                continue;
            }

            // Guard 4: pre-activation resolution must succeed
            let Ok(pre_activation) =
                resolve_preactivation(input, input_name, crown_bounds, ibp_bounds)
            else {
                continue;
            };

            // Guard 5: GPU layer extraction must succeed
            let mut gpu_layers = Vec::with_capacity(1);
            if try_extract_single_gpu_layer_with_alpha(
                &node.layer,
                pre_activation,
                alpha_state,
                node_name,
                &mut gpu_layers,
            )
            .is_none()
                || gpu_layers.is_empty()
            {
                continue;
            }

            let layer_count = gpu_layers.len();
            let Some(layer) = take_only_gpu_layer(gpu_layers) else {
                debug!(
                    node_name = node_name,
                    layer_count,
                    "Graph alpha GPU suffix skipped: extraction produced unexpected GPU layer count"
                );
                continue;
            };
            entries.insert(
                node_name.clone(),
                GpuSuffixPlanNode {
                    next_input: if terminates {
                        None
                    } else {
                        Some(input_name.to_string())
                    },
                    layer,
                },
            );
        }

        GpuSuffixPlan { entries }
    }

    /// Check if `node_name` is in the plan (O(1)).
    pub(super) fn contains(&self, node_name: &str) -> bool {
        self.entries.contains_key(node_name)
    }

    /// Materialize the full suffix layer vector by following `next_input`
    /// links from `start_node` to the terminal node (whose input is
    /// `NETWORK_INPUT`). Returns `None` if `start_node` is not in the plan.
    pub(super) fn materialize_suffix(&self, start_node: &str) -> Option<Vec<GpuCrownLayer>> {
        if !self.entries.contains_key(start_node) {
            return None;
        }
        let mut layers = Vec::new();
        let mut current = start_node;
        loop {
            let entry = self.entries.get(current)?;
            layers.push(entry.layer.clone());
            match &entry.next_input {
                None => break,
                Some(next) => current = next,
            }
        }
        Some(layers)
    }
}

/// Check if a node has active NON-ReLU alpha (S-shaped or Sqrt).
/// ReLU alpha is handled by the alpha-aware GPU extraction; only S-shaped and
/// Sqrt alpha require CPU fallback (not yet ported to GPU).
fn suffix_has_active_non_relu_alpha(
    node_name: &str,
    alpha_state: Option<&GraphAlphaState>,
) -> bool {
    let Some(alpha_state) = alpha_state else {
        return false;
    };
    alpha_state.monotone_s_shaped_alpha(node_name).is_some()
        || alpha_state.sqrt_alpha(node_name).is_some()
        || alpha_state.reciprocal_alpha(node_name).is_some()
}

/// Extract a GPU layer descriptor for a single node, using alpha-aware ReLU
/// slopes when alpha state is present.
///
/// For ReLU nodes with active alpha, builds the `Activation` descriptor from
/// optimized alpha values rather than the fixed heuristic. All other layer
/// types delegate to [`try_extract_single_gpu_layer`].
fn try_extract_single_gpu_layer_with_alpha(
    layer: &Layer,
    pre_activation: &BoundedTensor,
    alpha_state: Option<&GraphAlphaState>,
    node_name: &str,
    gpu_layers: &mut Vec<GpuCrownLayer>,
) -> Option<()> {
    if let Layer::ReLU(_) = layer {
        if let Some(alpha_state) = alpha_state {
            if let Some((alpha_lower, alpha_upper)) = alpha_state.relu_alpha_pair(node_name) {
                if let Some(unstable_mask) = alpha_state.relu_unstable_mask(node_name) {
                    let pre_l = pre_activation.lower().as_slice()?;
                    let pre_u = pre_activation.upper().as_slice()?;
                    // #4404: expand channel-only alpha to full spatial for GPU extraction.
                    let al_expanded = alpha_state.expand_alpha(node_name, alpha_lower);
                    let au_expanded = alpha_state.expand_alpha(node_name, alpha_upper);
                    let mask_expanded = if alpha_state.spatial_shape(node_name).is_some() {
                        alpha_state.expand_mask(node_name, unstable_mask)
                    } else {
                        unstable_mask.clone()
                    };
                    let gpu_layer = extract_relu_gpu_layer_with_alpha(
                        pre_l,
                        pre_u,
                        al_expanded.as_slice()?,
                        au_expanded.as_slice()?,
                        mask_expanded.as_slice()?,
                    );
                    gpu_layers.push(gpu_layer);
                    return Some(());
                }
            }
        }
    }
    // Not a ReLU with active alpha — fall back to standard extraction.
    try_extract_single_gpu_layer(layer, pre_activation, gpu_layers)
}

/// Finish a per-target graph backward pass on GPU using the precomputed
/// suffix plan (#4340). Looks up `node_name` in the plan in O(1) and
/// materializes the cached layer vector if eligible, then runs the existing
/// seeded GPU dispatch path unchanged.
fn try_finish_target_gpu_suffix_with_plan(
    input: &BoundedTensor,
    node_name: &str,
    node_lb: &LinearBounds,
    plan: &GpuSuffixPlan,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    target_contract: &GraphTargetShapeContract,
) -> Result<Option<BoundedTensor>> {
    // Building the host seed below scans and copies every coefficient and both
    // input endpoints before the GPU call.  That legacy preparation is neither
    // fallible nor cooperatively pollable, so a backend's own deadline support
    // is insufficient to authorize this shortcut.  Keep finite requests on the
    // CPU path, whose materialization/reduction kernels observe the same
    // absolute deadline.  The ordinary no-deadline GPU suffix is unchanged.
    // #gpu-suffix-expiry set-mate: default-off, byte-identical unarmed.
    if let Some(limit) = deadline {
        if Instant::now() >= limit {
            return Err(NyError::DeadlineExceeded(
                "Graph alpha GPU suffix deadline expired before host seed preparation".into(),
            ));
        }
        if crate::sound_gpu_gate::gpu_suffix_declines_under_finite_authority(limit) {
            return Ok(None);
        }
    }

    // Soundness gate (#vnncomp-gpu-crown-soundness): under the gate, route to the
    // SOUND seeded GPU-resident backward when available; else CPU sound fallback.
    let Some((gpu, use_sound)) =
        crate::sound_gpu_gate::gpu_crown_backward_route_with_deadline(engine, deadline)
    else {
        return Ok(None);
    };

    // O(1) plan lookup — ineligible nodes bail immediately.
    if !plan.contains(node_name) {
        return Ok(None);
    }

    if node_lb.lower_a().iter().any(|value| !value.is_finite())
        || node_lb.upper_a().iter().any(|value| !value.is_finite())
        || node_lb.lower_b().iter().any(|value| !value.is_finite())
        || node_lb.upper_b().iter().any(|value| !value.is_finite())
    {
        debug!(
            node_name = node_name,
            "Graph alpha GPU suffix skipped: non-finite seed coefficients or bias"
        );
        return Ok(None);
    }

    let Some(gpu_layers) = plan.materialize_suffix(node_name) else {
        return Ok(None);
    };

    let seed = GpuCrownSeed {
        lower_a: node_lb.lower_a().iter().copied().collect::<Vec<_>>().into(),
        upper_a: node_lb.upper_a().iter().copied().collect::<Vec<_>>().into(),
        lower_b: node_lb.lower_b().iter().copied().collect::<Vec<_>>().into(),
        upper_b: node_lb.upper_b().iter().copied().collect::<Vec<_>>().into(),
        num_specs: node_lb.num_outputs(),
        current_dim: node_lb.num_inputs(),
    };
    let input_lower: Vec<f32> = input.lower().iter().copied().collect();
    let input_upper: Vec<f32> = input.upper().iter().copied().collect();

    let seeded = if use_sound {
        gpu.crown_backward_gpu_seeded_sound(&gpu_layers, &seed, &input_lower, &input_upper)
    } else {
        gpu.crown_backward_gpu_seeded(&gpu_layers, &seed, &input_lower, &input_upper)
    };
    let gpu_result = match seeded {
        Ok(result) => result,
        Err(error) => {
            debug!(
                node_name = node_name,
                error = %error,
                "Graph alpha GPU suffix failed; falling back to CPU backward"
            );
            return Ok(None);
        }
    };

    if !crate::sound_gpu_gate::gpu_crown_result_is_publishable(&gpu_result, seed.num_specs) {
        debug!(
            node_name = node_name,
            "Graph alpha GPU suffix produced malformed bounds; falling back to CPU backward"
        );
        return Ok(None);
    }

    let (Ok(lower), Ok(upper)) = (
        ArrayD::from_shape_vec(IxDyn(&[seed.num_specs]), gpu_result.lower_bounds),
        ArrayD::from_shape_vec(IxDyn(&[seed.num_specs]), gpu_result.upper_bounds),
    ) else {
        return Ok(None);
    };
    let Some(bounds) = BoundedTensor::new(lower, upper).ok() else {
        return Ok(None);
    };
    let Some(bounds) = target_contract
        .restore_concrete(bounds, "Graph alpha-CROWN GPU suffix restore")
        .ok()
    else {
        return Ok(None);
    };
    Ok(Some(bounds))
}

#[allow(clippy::too_many_arguments)] // extends try_finish_target_gpu_suffix with merge accumulator
pub(super) fn try_finish_target_gpu_suffix_with_pending_input(
    input: &BoundedTensor,
    node_name: &str,
    node_lb: &LinearBounds,
    plan: &GpuSuffixPlan,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    target_contract: &GraphTargetShapeContract,
    node_crown_bounds: &mut CrownMergeAccumulator,
) -> Result<Option<BoundedTensor>> {
    let has_only_pending_input_contribution = node_crown_bounds.has_only_key(NETWORK_INPUT);
    if !node_crown_bounds.is_empty() && !has_only_pending_input_contribution {
        return Ok(None);
    }

    let Some(mut bounds) = try_finish_target_gpu_suffix_with_plan(
        input,
        node_name,
        node_lb,
        plan,
        engine,
        deadline,
        target_contract,
    )?
    else {
        return Ok(None);
    };

    if has_only_pending_input_contribution {
        let pending_input_contribution =
            node_crown_bounds.take(NETWORK_INPUT)?.ok_or_else(|| {
                NyError::InvalidSpec(
                    "Graph alpha GPU suffix expected pending input contribution".to_string(),
                )
            })?;
        let pending_input_contribution = target_contract.restore_concrete(
            pending_input_contribution
                .into_dense_with_deadline_for_purpose(
                    deadline,
                    PatchesMaterializationPurpose::NetworkInputTerminal,
                )?
                .concretize_sound(input),
            "Graph alpha-CROWN pending input contribution restore",
        )?;
        bounds = add_concrete_bounds(
            bounds,
            &pending_input_contribution,
            "Graph alpha-CROWN pending input contribution merge",
        )?;
    }

    Ok(Some(bounds))
}

pub(super) fn add_concrete_bounds(
    lhs: BoundedTensor,
    rhs: &BoundedTensor,
    context: &'static str,
) -> Result<BoundedTensor> {
    if lhs.shape() != rhs.shape() {
        return Err(NyError::shape_mismatch(
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ));
    }

    let lower_sum = lhs.lower() + rhs.lower();
    let upper_sum = lhs.upper() + rhs.upper();

    // Both operands can legitimately contain ±Inf (GPU suffix uses Widen strategy,
    // concretize_sound uses new_allow_infinite). The sum can produce:
    // - Inf + finite = Inf (valid conservative bound)
    // - Inf + (-Inf) = NaN (proves nothing, must be repaired)
    // Conservative strategy repairs NaN → -inf (lower) / +inf (upper) and passes
    // ±Inf endpoints and finite values through unchanged. A non-finite endpoint
    // carries no proven bound in that direction, and any finite substitute
    // (FALLBACK_BOUND included) is a strict subset of that unbounded interval —
    // an unsound tightening — so the repair must widen, never clamp.
    let non_finite_count = lower_sum.iter().filter(|v| !v.is_finite()).count()
        + upper_sum.iter().filter(|v| !v.is_finite()).count();
    if non_finite_count > 0 {
        debug!(
            non_finite_count,
            context,
            "add_concrete_bounds: {non_finite_count} non-finite sum endpoints (NaN widened to ±inf, ±Inf preserved)"
        );
    }
    BoundedTensor::new_repaired(lower_sum, upper_sum, RepairStrategy::Conservative)
}
