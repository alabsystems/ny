// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU suffix extraction for the graph alpha backward pass.
//!
//! When the remaining backward chain from a node to NETWORK_INPUT is a
//! GPU-extractable unary chain, this module offloads the entire suffix
//! to a single GPU dispatch via `GpuCrownBackward::crown_backward_gpu_seeded`.

use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::layers::Layer;
use crate::network::core::{
    extract_relu_gpu_layer_with_alpha, try_extract_single_gpu_layer, GraphNetwork, NETWORK_INPUT,
};

use ndarray::{ArrayD, IxDyn};
use ny_core::{GemmEngine, GpuCrownSeed, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

/// Per-alpha-state fail-closed marker. Once a selected GPU backend returns an
/// error or malformed receipt, the remainder of this solve uses CPU suffixes
/// instead of probing the same untrusted backend again at a later graph node.
const GPU_SUFFIX_RUNTIME_REFUSED: &str = "\0gpu-suffix-runtime-refused";

pub(super) fn gpu_suffix_runtime_refused(alpha_state: &GraphAlphaState) -> bool {
    alpha_state
        .gpu_suffix_ineligible
        .read()
        .is_ok_and(|set| set.contains(GPU_SUFFIX_RUNTIME_REFUSED))
}

impl GraphNetwork {
    /// Try to offload the remaining backward suffix to GPU.
    ///
    /// Starting from `node_name`, walks backward through single-input nodes
    /// toward `NETWORK_INPUT`, extracting GPU layers. ReLU nodes with active
    /// alpha use alpha-aware extraction (#4312); only S-shaped/Sqrt alpha
    /// forces CPU fallback.
    ///
    /// Returns `Some(concrete_bounds)` on success, `None` if fallback to CPU
    /// is needed (non-GPU layer, multi-input node, non-ReLU alpha, no GPU engine).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_alpha_backward_gpu_suffix(
        &self,
        input: &BoundedTensor,
        node_lb: &LinearBounds,
        node_name: &str,
        node_bounds: &HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<Option<BoundedTensor>> {
        // The legacy GPU shortcut builds a host seed, input endpoint copies,
        // and an extracted layer Vec with infallible O(N) operations. A GPU
        // deadline receipt cannot authorize work that happens before dispatch.
        // Keep finite requests on the cooperatively pollable CPU relation path;
        // no-deadline GPU behavior remains unchanged.
        // #gpu-suffix-expiry: default-off, byte-identical unarmed. See
        // `sound_gpu_gate::gpu_suffix_declines_under_finite_authority` — armed, a
        // live deadline with headroom no longer forces the CPU relation path.
        if let Some(limit) = deadline {
            if Instant::now() >= limit {
                return Err(ny_core::NyError::DeadlineExceeded(
                    "alpha-CROWN GPU suffix deadline expired before host seed preparation".into(),
                ));
            }
            if crate::sound_gpu_gate::gpu_suffix_declines_under_finite_authority(limit) {
                return Ok(None);
            }
        }

        // Soundness gate (#vnncomp-gpu-crown-soundness): under the gate, route to
        // the SOUND seeded GPU-resident backward (certified) when the engine
        // provides it; otherwise fall back to the CPU sound suffix path.
        let (gpu, use_sound) =
            match crate::sound_gpu_gate::gpu_crown_backward_route_with_deadline(engine, deadline) {
                Some(route) => route,
                None => return Ok(None),
            };

        // Negative cache: suffix extractability from this node is a property of
        // the graph structure, unchanged across alpha iterations. On
        // suffix-ineligible graphs (vit attention) every backward pass paid a
        // full extraction walk per node per iteration for nothing.
        if alpha_state
            .gpu_suffix_ineligible
            .read()
            .is_ok_and(|set| set.contains(node_name) || set.contains(GPU_SUFFIX_RUNTIME_REFUSED))
        {
            return Ok(None);
        }

        // Reject non-finite seed coefficients
        if node_lb.lower_a().iter().any(|v| !v.is_finite())
            || node_lb.upper_a().iter().any(|v| !v.is_finite())
            || node_lb.lower_b().iter().any(|v| !v.is_finite())
            || node_lb.upper_b().iter().any(|v| !v.is_finite())
        {
            return Ok(None);
        }

        let gpu_layers =
            match self.extract_gpu_suffix_layers(input, node_name, node_bounds, alpha_state) {
                Some(layers) => layers,
                None => {
                    // The unary suffix bailed — typically on a residual `Add`
                    // (multi-input). Try the resnet decomposition onto the sound
                    // GPU-resident resnet backward (#vnncomp-resnet); any failure
                    // returns Ok(None) → proven-sound CPU fallback (moat preserved).
                    let resnet =
                        crate::network::graph_alpha::resnet_decompose::try_resnet_gpu_suffix(
                            self,
                            input,
                            node_name,
                            node_bounds,
                            node_bounds,
                            Some(alpha_state),
                            engine,
                            deadline,
                            node_lb,
                        );
                    if matches!(resnet, Ok(None)) {
                        // Both structural routes declined: remember, so later
                        // iterations skip the extraction walk (perf only —
                        // structure never changes; seed-dependent rejections
                        // above are deliberately NOT cached).
                        if let Ok(mut set) = alpha_state.gpu_suffix_ineligible.write() {
                            set.insert(node_name.to_string());
                        }
                    }
                    return resnet;
                }
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
                if let Ok(mut set) = alpha_state.gpu_suffix_ineligible.write() {
                    set.insert(GPU_SUFFIX_RUNTIME_REFUSED.to_string());
                }
                debug!(
                    node_name = node_name,
                    error = %error,
                    "Alpha-CROWN GPU suffix failed; falling back to CPU backward"
                );
                return Ok(None);
            }
        };

        // A device result is atomic. Shape, finiteness, and interval ordering
        // must all validate before any endpoint can affect the host proof.
        if !crate::sound_gpu_gate::gpu_crown_result_is_publishable(&gpu_result, seed.num_specs) {
            if let Ok(mut set) = alpha_state.gpu_suffix_ineligible.write() {
                set.insert(GPU_SUFFIX_RUNTIME_REFUSED.to_string());
            }
            debug!(
                node_name = node_name,
                "Alpha-CROWN GPU suffix produced malformed bounds; falling back to CPU backward"
            );
            return Ok(None);
        }

        let (Ok(lower), Ok(upper)) = (
            ArrayD::from_shape_vec(IxDyn(&[seed.num_specs]), gpu_result.lower_bounds),
            ArrayD::from_shape_vec(IxDyn(&[seed.num_specs]), gpu_result.upper_bounds),
        ) else {
            return Ok(None);
        };
        Ok(BoundedTensor::new(lower, upper).ok())
    }

    /// Walk backward from `node_name` extracting GPU layers until NETWORK_INPUT.
    ///
    /// Returns `None` if any node in the chain is non-extractable.
    fn extract_gpu_suffix_layers(
        &self,
        input: &BoundedTensor,
        node_name: &str,
        node_bounds: &HashMap<String, BoundedTensor>,
        alpha_state: &GraphAlphaState,
    ) -> Option<Vec<ny_core::GpuCrownLayer>> {
        let mut gpu_layers = Vec::new();
        let mut current_name = node_name.to_string();

        loop {
            // Only bail for S-shaped/sqrt alpha (not yet supported on GPU).
            // ReLU alpha is handled by alpha-aware extraction below (#4312).
            let has_non_relu_alpha = alpha_state.monotone_s_shaped_alpha(&current_name).is_some()
                || alpha_state.sqrt_alpha(&current_name).is_some();
            if has_non_relu_alpha {
                return None;
            }

            let node = self.nodes.get(&current_name)?;
            if node.inputs.len() != 1 {
                return None;
            }

            let input_name = node.require_unary_input().ok()?;
            let pre_activation = if input_name == NETWORK_INPUT {
                input
            } else {
                node_bounds.get(input_name)?
            };

            // Use alpha-aware extraction for ReLU nodes with active alpha.
            let extracted = if let Layer::ReLU(_) = &node.layer {
                try_extract_relu_with_alpha(
                    &current_name,
                    pre_activation,
                    alpha_state,
                    &node.layer,
                    &mut gpu_layers,
                )
            } else {
                try_extract_single_gpu_layer(&node.layer, pre_activation, &mut gpu_layers)
            };
            extracted?;

            if input_name == NETWORK_INPUT {
                break;
            }
            current_name = input_name.to_string();
        }

        Some(gpu_layers)
    }
}

/// Try alpha-aware ReLU GPU extraction, falling back to standard extraction.
fn try_extract_relu_with_alpha(
    node_name: &str,
    pre_activation: &BoundedTensor,
    alpha_state: &GraphAlphaState,
    layer: &Layer,
    gpu_layers: &mut Vec<ny_core::GpuCrownLayer>,
) -> Option<()> {
    if let Some((al, au)) = alpha_state.relu_alpha_pair(node_name) {
        if let Some(mask) = alpha_state.relu_unstable_mask(node_name) {
            if let (Some(pre_l), Some(pre_u)) = (
                pre_activation.lower().as_slice(),
                pre_activation.upper().as_slice(),
            ) {
                // #4404: expand channel-only alpha to full spatial for GPU extraction.
                let al_expanded = alpha_state.expand_alpha(node_name, al);
                let au_expanded = alpha_state.expand_alpha(node_name, au);
                let mask_expanded = if alpha_state.spatial_shape(node_name).is_some() {
                    // Expand per-channel mask to per-neuron
                    alpha_state.expand_mask(node_name, mask)
                } else {
                    mask.clone()
                };
                if let (Some(al_s), Some(au_s), Some(mask_s)) = (
                    al_expanded.as_slice(),
                    au_expanded.as_slice(),
                    mask_expanded.as_slice(),
                ) {
                    let gpu_layer =
                        extract_relu_gpu_layer_with_alpha(pre_l, pre_u, al_s, au_s, mask_s);
                    gpu_layers.push(gpu_layer);
                    return Some(());
                }
            }
        }
    }
    try_extract_single_gpu_layer(layer, pre_activation, gpu_layers)
}
