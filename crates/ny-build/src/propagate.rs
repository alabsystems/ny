// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential propagation network builder.
//!
//! Converts a list of [`LayerSpec`]s into a [`ny_propagate::Network`] for
//! IBP/CROWN bound propagation. This is the construction-side counterpart of
//! the ONNX parsing in `ny-onnx`.

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_propagate::layers::OpaqueSkipLayer;
use ny_propagate::Layer;
use ny_propagate::Network as PropNetwork;
use std::collections::HashMap;
use tracing::debug;

use crate::{is_multi_output_split, ConvertContext, LayerSpec};

/// Options controlling sequential network conversion.
#[derive(Clone, Copy, Debug, Default)]
pub struct PropagateNetworkOptions {
    /// If true, skip Reshape layers whose target shape is not statically known.
    /// This is a best-effort mode and may be unsound for shape-sensitive models.
    pub allow_dynamic_reshape: bool,
}

impl PropagateNetworkOptions {
    /// Allow skipping dynamic Reshape layers during sequential conversion.
    pub fn permissive() -> Self {
        Self {
            allow_dynamic_reshape: true,
        }
    }
}

/// Build a sequential [`PropNetwork`] from a list of layer specifications.
///
/// Iterates `layers`, converting each via [`ConvertContext::convert_layer`].
/// Dynamic-shape Reshape layers are either skipped (if `options.allow_dynamic_reshape`)
/// or produce an error pointing the caller to [`PropagateNetworkOptions::permissive`].
pub fn build_propagate_network(
    layers: &[LayerSpec],
    ctx: &ConvertContext<'_>,
    options: &PropagateNetworkOptions,
) -> Result<PropNetwork> {
    Ok(build_propagate_network_indexed(layers, ctx, options)?.0)
}

/// Like [`build_propagate_network`], but also return, for each input spec,
/// the index of the network layer it produced (`None` for specs that were
/// skipped as pre-evaluated constants or constant shape-computation chains).
///
/// Callers that slice the resulting sequential network by spec ranges (e.g.
/// per-block extraction) need this map to stay consistent with the builder's
/// skip decisions; deriving it independently drifts as the builder evolves.
pub fn build_propagate_network_indexed(
    layers: &[LayerSpec],
    ctx: &ConvertContext<'_>,
    options: &PropagateNetworkOptions,
) -> Result<(PropNetwork, Vec<Option<usize>>)> {
    let mut network = PropNetwork::new();
    let mut index_map: Vec<Option<usize>> = Vec::with_capacity(layers.len());
    let mut evaluated_constants: HashMap<String, ArrayD<f32>> = HashMap::new();

    for layer_spec in layers {
        let all_inputs_constant = !layer_spec.inputs.is_empty()
            && layer_spec
                .inputs
                .iter()
                .all(|inp| ctx.is_constant(inp) || evaluated_constants.contains_key(inp));
        let is_shape_with_static_input = layer_spec.layer_type == ny_core::LayerType::Shape
            && layer_spec
                .inputs
                .first()
                .is_some_and(|input| ctx.tensor_shapes.contains_key(input));

        if all_inputs_constant || layer_spec.inputs.is_empty() || is_shape_with_static_input {
            if layer_spec.outputs.len() != 1 {
                continue;
            }
            let eval_ctx = ConvertContext::with_evaluated_constants(
                ctx.weights,
                ctx.tensor_shapes,
                ctx.constant_tensors,
                &evaluated_constants,
            )
            .with_model_unbatched(ctx.model_unbatched);
            if let Some(value) = eval_ctx.evaluate_constant_layer(layer_spec, &evaluated_constants)
            {
                let output = &layer_spec.outputs[0];
                debug!(
                    "Pre-evaluated constant {} -> shape {:?} in sequential network",
                    output,
                    value.shape()
                );
                evaluated_constants.insert(output.clone(), value);
            } else {
                debug!(
                    "Could not pre-evaluate {} ({:?}) with inputs {:?}",
                    layer_spec.name, layer_spec.layer_type, layer_spec.inputs
                );
            }
        }
    }

    let ctx = ConvertContext::with_evaluated_constants(
        ctx.weights,
        ctx.tensor_shapes,
        ctx.constant_tensors,
        &evaluated_constants,
    )
    .with_model_unbatched(ctx.model_unbatched);

    for layer_spec in layers {
        if is_multi_output_split(layer_spec) {
            return Err(NyError::ModelLoad(format!(
                "Split '{}' has {} outputs and requires graph network construction",
                layer_spec.name,
                layer_spec.outputs.len()
            )));
        }

        // Skip nodes whose inputs are all constants — these are shape-computation
        // chains (Shape→Gather→Unsqueeze→Concat) that produce constant outputs.
        // Their results are already in the weights store from const-folding.
        // Consistent with GraphNetwork builder (builder.rs:200-226). Part of #3312.
        let all_inputs_constant = !layer_spec.inputs.is_empty()
            && layer_spec.inputs.iter().all(|inp| ctx.is_constant(inp));
        if all_inputs_constant {
            debug!(
                "Skipping layer {} (all inputs constant) in sequential network",
                layer_spec.name
            );
            index_map.push(None);
            continue;
        }
        if !layer_spec.outputs.is_empty()
            && layer_spec
                .outputs
                .iter()
                .all(|output| evaluated_constants.contains_key(output))
        {
            debug!(
                "Skipping layer {} (output pre-evaluated) in sequential network",
                layer_spec.name
            );
            index_map.push(None);
            continue;
        }

        let layer = match ctx.convert_layer(layer_spec) {
            Ok(l) => l,
            Err(NyError::UnsupportedOp(msg))
                if msg.contains("dynamic shape") && options.allow_dynamic_reshape =>
            {
                debug!(
                    "Skipping Reshape {} with dynamic shape in sequential network",
                    layer_spec.name
                );
                index_map.push(None);
                continue;
            }
            Err(NyError::UnsupportedOp(msg)) if msg.contains("dynamic shape") => {
                return Err(NyError::UnsupportedOp(format!(
                    "{}; use PropagateNetworkOptions::permissive() to allow skipping",
                    msg
                )));
            }
            Err(NyError::UnsupportedOp(msg))
                if msg.contains("needs constant folding")
                    || msg.starts_with("Shape op ")
                    || msg.contains("targets batch dimension which does not exist") =>
            {
                debug!(
                    "Replacing unsupported layer {} with conservative OpaqueSkip in sequential network: {}",
                    layer_spec.name, msg
                );
                Layer::OpaqueSkip(OpaqueSkipLayer::new())
            }
            Err(e) => return Err(e),
        };
        index_map.push(Some(network.num_layers()));
        network.add_layer(layer);
    }

    Ok((network, index_map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AttributeValue, WeightStore};
    use ny_core::LayerType;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn sequential_builder_refuses_multi_output_split_before_conversion() {
        let spec = LayerSpec {
            name: "split".to_string(),
            layer_type: LayerType::Slice,
            inputs: vec!["x".to_string()],
            outputs: vec!["a".to_string(), "b".to_string()],
            weights: None,
            attributes: HashMap::from([
                ("axis".to_string(), AttributeValue::Int(1)),
                ("split".to_string(), AttributeValue::Ints(vec![1, 1])),
            ]),
        };
        let weights = WeightStore::new();
        let tensor_shapes = HashMap::new();
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);

        let err = build_propagate_network(&[spec], &ctx, &PropagateNetworkOptions::default())
            .expect_err("multi-output Split requires graph construction");
        assert!(
            err.to_string()
                .contains("requires graph network construction"),
            "expected graph-construction error, got: {err}"
        );
    }
}
