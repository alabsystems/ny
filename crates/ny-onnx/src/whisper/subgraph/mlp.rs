// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::LayerSpec;
use ny_core::{LayerType, NyError, Result};
use ny_propagate::{GraphNetwork, GraphNode};
use tracing::info;

use super::super::loader::block_index::parse_block_index;
use super::super::model::WhisperModel;

impl WhisperModel {
    /// Extract the MLP subgraph (without the residual Add).
    ///
    /// This extracts: mlp_ln → Linear → GELU → Linear → bias Add
    /// Output is the MLP delta to be added to the residual.
    pub fn mlp_subgraph(&self, index: usize) -> Result<GraphNetwork> {
        if index >= self.encoder_layers {
            return Err(NyError::InvalidSpec(format!(
                "Encoder layer {} out of range (max {})",
                index, self.encoder_layers
            )));
        }

        let block_layers = self.block_layers_for_index(index)?;
        let weights = &self.model.weights;
        let constants = &self.model.constant_tensors;
        let mlp_ln_tokens = ["final_layer_norm", "mlp_ln"];

        let find_weight_name = |tokens: &[&str], suffix: &str| -> Option<String> {
            weights
                .keys()
                .find(|name| {
                    parse_block_index(name) == Some(index)
                        && name.contains(suffix)
                        && tokens.iter().any(|token| name.contains(token))
                })
                .map(|s| s.to_string())
        };

        let mlp_ln_weight = find_weight_name(&mlp_ln_tokens, "weight");
        let mlp_ln_spec = mlp_ln_weight
            .as_ref()
            .and_then(|weight| {
                block_layers.iter().copied().find(|spec| {
                    spec.layer_type == LayerType::LayerNorm
                        && spec.inputs.iter().any(|input| input == weight)
                })
            })
            .or_else(|| Self::find_layernorm_by_tokens(&block_layers, &mlp_ln_tokens))
            .ok_or_else(|| {
                let detail = if mlp_ln_weight.is_some() {
                    "MLP LayerNorm node not found"
                } else {
                    "MLP LayerNorm weight/node not found"
                };
                NyError::InvalidSpec(format!("{} for block {}", detail, index))
            })?;

        let fc1_bias = find_weight_name(&["fc1", "mlp.0", "mlp/0"], "bias");
        let fc2_bias = find_weight_name(&["fc2", "mlp.2", "mlp/2"], "bias");

        let mut output_to_spec: std::collections::HashMap<String, &LayerSpec> =
            std::collections::HashMap::new();
        for spec in block_layers.iter() {
            for output in &spec.outputs {
                if !output.is_empty() {
                    output_to_spec.insert(output.clone(), spec);
                }
            }
        }

        let trace_activation_source = |tensor: &str| -> String {
            let mut current = tensor.to_string();
            let mut visited = std::collections::HashSet::new();
            while let Some(spec) = output_to_spec.get(&current) {
                if !visited.insert(spec.name.clone()) {
                    break;
                }
                let passthrough = match spec.layer_type {
                    LayerType::Reshape | LayerType::Transpose => true,
                    LayerType::Mul | LayerType::Div => {
                        let activation_inputs = spec.inputs.iter().filter(|name| {
                            !weights.contains_key(name) && !constants.contains(*name)
                        });
                        activation_inputs.count() == 1
                    }
                    _ => false,
                };
                if !passthrough {
                    break;
                }
                let next = spec
                    .inputs
                    .iter()
                    .find(|name| !weights.contains_key(name) && !constants.contains(*name));
                if let Some(next) = next {
                    current = next.clone();
                } else {
                    break;
                }
            }
            current
        };

        let trace_weight_source = |tensor: &str| -> Option<String> {
            if weights.contains_key(tensor) {
                return Some(tensor.to_string());
            }
            let mut current = tensor.to_string();
            let mut visited = std::collections::HashSet::new();
            loop {
                let spec = output_to_spec.get(&current)?;
                if !visited.insert(spec.name.clone()) {
                    return None;
                }
                let next = match spec.layer_type {
                    LayerType::Transpose => spec.inputs.first(),
                    LayerType::Reshape => spec
                        .inputs
                        .iter()
                        .find(|name| weights.contains_key(name))
                        .or_else(|| spec.inputs.iter().find(|name| !constants.contains(*name))),
                    LayerType::Mul | LayerType::Div => {
                        spec.inputs.iter().find(|name| !constants.contains(*name))
                    }
                    _ => None,
                };
                let next = next?;
                if weights.contains_key(next) {
                    return Some(next.clone());
                }
                current = next.clone();
            }
        };

        let find_add_by_bias = |bias: &str| -> Option<&LayerSpec> {
            block_layers.iter().copied().find(|spec| {
                spec.layer_type == LayerType::Add && spec.inputs.iter().any(|input| input == bias)
            })
        };

        let find_matmul_from_add = |add_spec: &LayerSpec| -> Option<&LayerSpec> {
            let activation_input = add_spec
                .inputs
                .iter()
                .find(|name| !weights.contains_key(name) && !constants.contains(*name))?;
            let origin = trace_activation_source(activation_input);
            let producer = output_to_spec.get(&origin)?;
            if matches!(producer.layer_type, LayerType::MatMul | LayerType::Linear) {
                Some(*producer)
            } else {
                None
            }
        };

        let weight_matches_tokens = |name: &str, tokens: &[&str]| -> bool {
            parse_block_index(name) == Some(index)
                && name.contains("weight")
                && tokens.iter().any(|token| name.contains(token))
        };

        let find_matmul_by_weight_tokens = |tokens: &[&str]| -> Option<&LayerSpec> {
            block_layers.iter().copied().find(|spec| {
                if !matches!(spec.layer_type, LayerType::MatMul | LayerType::Linear) {
                    return false;
                }
                if let Some(weights_ref) = &spec.weights {
                    if weight_matches_tokens(&weights_ref.name, tokens) {
                        return true;
                    }
                }
                spec.inputs.iter().any(|input| {
                    trace_weight_source(input)
                        .is_some_and(|name| weight_matches_tokens(&name, tokens))
                })
            })
        };

        let fc1_add_spec = fc1_bias.as_ref().and_then(|bias| find_add_by_bias(bias));
        let fc2_add_spec = fc2_bias.as_ref().and_then(|bias| find_add_by_bias(bias));

        let fc1_matmul_spec = fc1_add_spec
            .and_then(&find_matmul_from_add)
            .or_else(|| find_matmul_by_weight_tokens(&["fc1", "mlp.0", "mlp/0"]))
            .ok_or_else(|| {
                NyError::InvalidSpec(format!("MLP fc1 MatMul not found for block {}", index))
            })?;
        let fc2_matmul_spec = fc2_add_spec
            .and_then(find_matmul_from_add)
            .or_else(|| find_matmul_by_weight_tokens(&["fc2", "mlp.2", "mlp/2"]))
            .ok_or_else(|| {
                NyError::InvalidSpec(format!("MLP fc2 MatMul not found for block {}", index))
            })?;

        let fc1_src_spec = fc1_add_spec.unwrap_or(fc1_matmul_spec);
        let fc1_src_output = fc1_src_spec.outputs.first().ok_or_else(|| {
            NyError::InvalidSpec(format!("MLP fc1 output missing for block {}", index))
        })?;

        let gelu_spec = block_layers
            .iter()
            .copied()
            .filter(|spec| spec.layer_type == LayerType::GELU)
            .find(|spec| {
                let activation_input = spec
                    .inputs
                    .iter()
                    .find(|name| !weights.contains_key(name) && !constants.contains(*name));
                activation_input
                    .is_some_and(|input| trace_activation_source(input) == *fc1_src_output)
            })
            .ok_or_else(|| {
                NyError::InvalidSpec(format!("MLP GELU not found for block {}", index))
            })?;

        let mlp_layer_names: std::collections::HashSet<String> = [
            &mlp_ln_spec.name,
            &fc1_matmul_spec.name,
            &gelu_spec.name,
            &fc2_matmul_spec.name,
        ]
        .iter()
        .map(|s| (*s).clone())
        .chain(fc1_add_spec.map(|spec| spec.name.clone()))
        .chain(fc2_add_spec.map(|spec| spec.name.clone()))
        .collect();

        let mlp_output = if let Some(fc2_add) = fc2_add_spec {
            fc2_add.name.clone()
        } else {
            fc2_matmul_spec.name.clone()
        };

        let mut graph = GraphNetwork::new();
        let mut prev_node: Option<String> = None;

        for &spec in &block_layers {
            // Only include MLP layers
            if !mlp_layer_names.contains(&spec.name) {
                continue;
            }

            let layer = self.model.convert_layer(spec)?;

            // Sequential input
            let inputs = match &prev_node {
                Some(name) => vec![name.clone()],
                None => vec!["_input".to_string()],
            };

            graph.try_add_node(GraphNode::new(spec.name.clone(), layer, inputs))?;
            prev_node = Some(spec.name.clone());
        }

        let output_node = if graph.contains_node(&mlp_output) {
            mlp_output
        } else if let Some(last) = prev_node {
            last
        } else {
            return Err(NyError::InvalidSpec(format!(
                "MLP subgraph for block {} produced no nodes",
                index
            )));
        };

        graph.set_output(output_node);

        info!(
            "Built MLP subgraph for block {} with {} nodes",
            index,
            graph.num_nodes()
        );

        Ok(graph)
    }
}
