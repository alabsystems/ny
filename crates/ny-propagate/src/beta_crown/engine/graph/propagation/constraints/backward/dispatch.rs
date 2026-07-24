// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Node-step dispatch coordinator for constrained backward CROWN.
//!
//! Contains:
//! - `ConstrainedNodeContext` — per-node context carrier
//! - `capture_constrained_linear_bounds` — linear bounds capture for lA caching
//! - `apply_constrained_backward_dispatch_result` — n-ary dispatch accumulation
//! - `try_finish_constrained_gpu_suffix` — seeded GPU suffix fast-exit
//! - `process_constrained_backward_node` — per-node router (Linear/ReLU/generic)
//! - `process_shared_dispatch_node` — generic operator backward dispatch
//!
//! Part of #4293 (directory-module split from former backward.rs monolith).

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{GpuCrownSeed, NyError, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};
use tracing::debug;

use crate::bounds::patches::CrownBounds;
use crate::network::backward_dispatch::{
    dispatch_backward_layer, BackwardDispatchResult, DispatchContext,
};
use crate::network::{
    apply_dense_backward_dispatch_result, try_extract_single_gpu_layer, GraphNode,
};
use crate::{GraphNetwork, Layer, LinearBounds, MulBinaryRelaxationMode, NETWORK_INPUT};

use super::super::super::super::super::BetaCrownVerifier;
use super::super::patches::try_patches_step;
use super::{
    resolve_pre_activation, BackwardCrownResult, BackwardParams, ConstrainedBackwardSetup,
};

pub(super) struct ConstrainedNodeContext<'a> {
    pub node_name: &'a str,
    pub node: &'a GraphNode,
    pub first_input: &'a str,
}

fn capture_constrained_linear_bounds(
    current: &ConstrainedNodeContext<'_>,
    node_cb: &CrownBounds,
    captured_linear_bounds: &mut Option<HashMap<String, LinearBounds>>,
) -> Result<()> {
    let Some(linear_bounds_map) = captured_linear_bounds.as_mut() else {
        return Ok(());
    };
    let captured_lb = match node_cb {
        CrownBounds::Dense(lb) => lb.clone(),
        CrownBounds::Patches(_) => node_cb.clone().into_dense()?,
    };
    linear_bounds_map.insert(current.node_name.to_string(), captured_lb);
    Ok(())
}

fn apply_constrained_backward_dispatch_result(
    params: &BackwardParams<'_>,
    current: &ConstrainedNodeContext<'_>,
    pass_through_bounds: &LinearBounds,
    result: BackwardDispatchResult,
    setup: &mut ConstrainedBackwardSetup<'_, '_>,
) -> Result<()> {
    if let BackwardDispatchResult::Nary {
        bounds,
        bias_lower,
        bias_upper,
    } = result
    {
        // Preserve constrained backward's existing bias width contract: the zero-A
        // bias wrapper follows the first concrete split path rather than the caller's
        // network-input width. The shared helper defaults to input_dim instead.
        let a_cols = bounds
            .iter()
            .find_map(|bound| bound.as_ref())
            .map(|bound| bound.lower_a().ncols())
            .unwrap_or(params.constrained_input.len());
        let bias_lb = LinearBounds::new_or_conservative(
            Array2::zeros((bias_lower.len(), a_cols)),
            bias_lower,
            Array2::zeros((bias_upper.len(), a_cols)),
            bias_upper,
        )?;
        params.graph.accumulate_dense_bounds_to_input(
            NETWORK_INPUT,
            bias_lb,
            &mut setup.state.node_crown_bounds,
            setup.output_dim,
            setup.input_dim,
            &mut setup.state.input_accumulated,
        )?;
        for (input_name, bound_opt) in current.node.inputs.iter().zip(bounds) {
            if let Some(bound) = bound_opt {
                GraphNetwork::verify_split_path_bias_zero(
                    &bound,
                    "Constrained dispatch n-ary split path",
                )?;
                params.graph.accumulate_dense_bounds_to_input(
                    input_name,
                    bound,
                    &mut setup.state.node_crown_bounds,
                    setup.output_dim,
                    setup.input_dim,
                    &mut setup.state.input_accumulated,
                )?;
            }
        }
        return Ok(());
    }

    apply_dense_backward_dispatch_result(
        params.graph,
        current.node,
        current.first_input,
        pass_through_bounds,
        result,
        &mut setup.state.node_crown_bounds,
        setup.output_dim,
        setup.input_dim,
        &mut setup.state.input_accumulated,
        "Constrained dispatch",
    )
    .map_err(|error| match error {
        NyError::UnsupportedOp(reason) => NyError::UnsupportedOp(format!(
            "Constrained CROWN: layer '{}' ({}) unsupported: {}",
            current.node_name,
            current.node.layer.layer_type(),
            reason
        )),
        other => other,
    })
}

impl BetaCrownVerifier {
    pub(super) fn try_finish_constrained_gpu_suffix(
        &self,
        params: &BackwardParams<'_>,
        node_name: &str,
        node_lb: &LinearBounds,
        bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
        output_shape: &[usize],
    ) -> Result<Option<BoundedTensor>> {
        // Soundness gate (#vnncomp-gpu-crown-soundness): masked when soundness is
        // required, forcing the CPU sound constrained-suffix path. See
        // `sound_gpu_gate`.
        let Some((gpu, use_sound)) =
            crate::sound_gpu_gate::gpu_crown_backward_route(params.context.engine)
        else {
            return Ok(None);
        };

        if node_lb.lower_a().iter().any(|value| !value.is_finite())
            || node_lb.upper_a().iter().any(|value| !value.is_finite())
            || node_lb.lower_b().iter().any(|value| !value.is_finite())
            || node_lb.upper_b().iter().any(|value| !value.is_finite())
        {
            debug!(
                node_name,
                "Constrained GPU suffix skipped: non-finite seed coefficients or bias"
            );
            return Ok(None);
        }

        let mut gpu_layers = Vec::new();
        let mut current_name = node_name;
        loop {
            let node =
                params.graph.nodes.get(current_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!("Node not found: {}", current_name))
                })?;
            if node.inputs().len() != 1 {
                return Ok(None);
            }

            let input_name = node.require_unary_input()?;
            let pre_activation = if input_name == NETWORK_INPUT {
                params.constrained_input
            } else {
                bounds_cache.get(input_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Pre-activation bounds for {} not found",
                        input_name
                    ))
                })?
            };

            if try_extract_single_gpu_layer(&node.layer, pre_activation, &mut gpu_layers).is_none()
            {
                return Ok(None);
            }

            if input_name == NETWORK_INPUT {
                break;
            }
            current_name = input_name;
        }

        let seed = GpuCrownSeed {
            lower_a: node_lb.lower_a().iter().copied().collect::<Vec<_>>().into(),
            upper_a: node_lb.upper_a().iter().copied().collect::<Vec<_>>().into(),
            lower_b: node_lb.lower_b().iter().copied().collect::<Vec<_>>().into(),
            upper_b: node_lb.upper_b().iter().copied().collect::<Vec<_>>().into(),
            num_specs: node_lb.num_outputs(),
            current_dim: node_lb.num_inputs(),
        };
        let input_lower: Vec<f32> = params.constrained_input.lower().iter().copied().collect();
        let input_upper: Vec<f32> = params.constrained_input.upper().iter().copied().collect();

        let seeded = if use_sound {
            gpu.crown_backward_gpu_seeded_sound(&gpu_layers, &seed, &input_lower, &input_upper)
        } else {
            gpu.crown_backward_gpu_seeded(&gpu_layers, &seed, &input_lower, &input_upper)
        };
        let gpu_result = match seeded {
            Ok(result) => result,
            Err(error) => {
                debug!(
                    node_name,
                    error = %error,
                    "Constrained GPU suffix failed; falling back to CPU backward"
                );
                return Ok(None);
            }
        };

        if gpu_result
            .lower_bounds
            .iter()
            .chain(gpu_result.upper_bounds.iter())
            .any(|value| value.is_nan())
        {
            debug!(
                node_name,
                "Constrained GPU suffix produced NaN bounds; falling back to CPU backward"
            );
            return Ok(None);
        }

        let lower = ArrayD::from_shape_vec(IxDyn(output_shape), gpu_result.lower_bounds).map_err(
            |error| NyError::InvalidSpec(format!("Constrained GPU suffix lower reshape: {error}")),
        )?;
        let upper = ArrayD::from_shape_vec(IxDyn(output_shape), gpu_result.upper_bounds).map_err(
            |error| NyError::InvalidSpec(format!("Constrained GPU suffix upper reshape: {error}")),
        )?;
        Ok(Some(BoundedTensor::new_repaired(
            lower,
            upper,
            RepairStrategy::Widen,
        )?))
    }

    pub(super) fn process_constrained_backward_node(
        &self,
        params: &BackwardParams<'_>,
        is_standard: bool,
        node_name: &str,
        bounds_cache_mut: &mut HashMap<String, Arc<BoundedTensor>>,
        setup: &mut ConstrainedBackwardSetup<'_, '_>,
    ) -> Result<Option<BackwardCrownResult>> {
        let node = params
            .graph
            .nodes
            .get(node_name)
            .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", node_name)))?;
        let first_input = if node.layer.is_binary() || node.layer.is_ternary() {
            node.inputs.first().map(String::as_str).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Node '{}' ({}) has no inputs",
                    node_name,
                    node.layer.layer_type()
                ))
            })?
        } else {
            node.require_unary_input()?
        };
        let current = ConstrainedNodeContext {
            node_name,
            node,
            first_input,
        };
        let node_cb = match setup.state.node_crown_bounds.take(node_name)? {
            Some(bounds) => bounds,
            None => return Ok(None),
        };

        capture_constrained_linear_bounds(
            &current,
            &node_cb,
            &mut setup.state.captured_linear_bounds,
        )?;

        if is_standard && params.context.history.constraints.len() >= 12 {
            if let CrownBounds::Dense(lb) = &node_cb {
                debug!(
                    "[#1817 node] {} lower_a={:?} upper_a={:?}",
                    current.node_name,
                    lb.lower_a()
                        .as_slice()
                        .map(|slice| &slice[..slice.len().min(8)]),
                    lb.upper_a()
                        .as_slice()
                        .map(|slice| &slice[..slice.len().min(8)]),
                );
            } else {
                debug!(
                    "[#1817 node] {} carrier=patches memory_bytes={}",
                    current.node_name,
                    node_cb.memory_bytes()
                );
            }
        }

        if matches!(&current.node.layer, Layer::Linear(_)) {
            self.process_linear_override(
                params,
                &current,
                node_cb,
                &*bounds_cache_mut,
                setup,
                is_standard,
            )?;
            return Ok(None);
        }

        if matches!(&current.node.layer, Layer::ReLU(_)) {
            let pre_activation = resolve_pre_activation(
                current.first_input,
                params.constrained_input,
                &*bounds_cache_mut,
            )?;
            self.process_relu_override(
                params,
                &current,
                pre_activation,
                node_cb,
                setup,
                is_standard,
            )?;
            return Ok(None);
        }

        self.process_shared_dispatch_node(
            params,
            &current,
            node_cb,
            bounds_cache_mut,
            setup,
            is_standard,
        )
    }

    fn process_shared_dispatch_node(
        &self,
        params: &BackwardParams<'_>,
        current: &ConstrainedNodeContext<'_>,
        mut node_cb: CrownBounds,
        bounds_cache_mut: &mut HashMap<String, Arc<BoundedTensor>>,
        setup: &mut ConstrainedBackwardSetup<'_, '_>,
        is_standard: bool,
    ) -> Result<Option<BackwardCrownResult>> {
        let pre_activation = resolve_pre_activation(
            current.first_input,
            params.constrained_input,
            &*bounds_cache_mut,
        )?;
        // GenBaB inv_rms norm branching (#norm-genbab): collect per-node inv_rms
        // window overrides from this domain's history, so the RmsNorm CROWN
        // backward narrows its certified inv_rms interval to the requesting
        // child subdomain.
        let norm_inv_rms_map = params.context.history.norm_inv_rms_overrides();
        let ctx = DispatchContext {
            node_name: current.node_name,
            layer: &current.node.layer,
            inputs: &current.node.inputs,
            pre_activation,
            network_input: params.constrained_input,
            node_bounds: (&*bounds_cache_mut).into(),
            engine: params.context.engine,
            deadline: params.deadline,
            bilinear_alphas: None,
            mul_binary_relaxation: MulBinaryRelaxationMode::default(),
            mul_binary_alphas: None,
            norm_inv_rms_override: norm_inv_rms_map.as_ref(),
        };

        if try_patches_step(
            params.graph,
            "Constrained CROWN",
            current.node_name,
            current.node,
            &mut node_cb,
            current.first_input,
            pre_activation,
            &*bounds_cache_mut,
            &mut setup.state.node_crown_bounds,
            &mut setup.state.input_accumulated,
            params.context.engine,
            params.deadline,
            params.patches_policy,
        )? {
            return Ok(None);
        }

        let node_lb = node_cb.into_dense()?;
        if is_standard && !params.capture_linear_bounds && setup.state.node_crown_bounds.is_empty()
        {
            if let Some(output_bounds) = self.try_finish_constrained_gpu_suffix(
                params,
                current.node_name,
                &node_lb,
                &*bounds_cache_mut,
                &setup.output_shape,
            )? {
                let output_bounds = self.apply_graph_cut_contribution_if_needed(
                    params,
                    bounds_cache_mut,
                    output_bounds,
                )?;
                return Ok(Some(BackwardCrownResult {
                    output_bounds,
                    intermediate: None,
                    captured_la: None,
                }));
            }
        }

        let result = dispatch_backward_layer(&ctx, &node_lb)?;
        apply_constrained_backward_dispatch_result(params, current, &node_lb, result, setup)?;
        Ok(None)
    }
}
