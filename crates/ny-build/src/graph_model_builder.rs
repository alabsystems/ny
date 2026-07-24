// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Builder for the owned traced-producer graph handoff surface.
//!
//! External traced producers should target [`GraphModel`](crate::GraphModel) /
//! [`GraphModel::build_graph_network`](crate::GraphModel::build_graph_network)
//! rather than constructing `ny_propagate` graphs directly. Multi-input
//! traced models should use [`GraphModelBuilder::frozen_input`] for auxiliary
//! inputs that are constant with respect to the verification query.

use crate::{DataType, GraphModel, LayerSpec, Network, TensorSpec, WeightStore};
use ndarray::ArrayD;
use std::collections::{HashMap, HashSet};

/// Builder-style helper for assembling [`GraphModel`] values.
#[must_use]
#[derive(Debug, Clone)]
pub struct GraphModelBuilder {
    name: String,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    layers: Vec<LayerSpec>,
    param_count: usize,
    weights: WeightStore,
    tensor_producer: HashMap<String, String>,
    constant_tensors: HashSet<String>,
    tensor_shapes: HashMap<String, Vec<i64>>,
}

impl GraphModelBuilder {
    /// Start a new builder for the named graph model.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            layers: Vec::new(),
            param_count: 0,
            weights: WeightStore::new(),
            tensor_producer: HashMap::new(),
            constant_tensors: HashSet::new(),
            tensor_shapes: HashMap::new(),
        }
    }

    /// Add a network input tensor.
    pub fn input(mut self, name: impl Into<String>, shape: &[i64], dtype: DataType) -> Self {
        self.inputs.push(TensorSpec {
            name: name.into(),
            shape: shape.to_vec(),
            dtype,
        });
        self
    }

    /// Add a network output tensor.
    pub fn output(mut self, name: impl Into<String>, shape: &[i64], dtype: DataType) -> Self {
        self.outputs.push(TensorSpec {
            name: name.into(),
            shape: shape.to_vec(),
            dtype,
        });
        self
    }

    /// Add a layer specification in topological order.
    pub fn layer(mut self, layer: LayerSpec) -> Self {
        self.layers.push(layer);
        self
    }

    /// Insert a weight tensor into the weight store.
    pub fn weight(mut self, name: impl Into<String>, weights: ArrayD<f32>) -> Self {
        self.weights.insert(name.into(), weights);
        self
    }

    /// Record a frozen auxiliary input for a traced multi-input model.
    ///
    /// `declared_shape` should match the producer-visible input shape, while
    /// `value` is already in ny's unbatched propagation layout after
    /// any stripped batch axis has been removed. The helper stores the tensor in
    /// `weights`, marks it in `constant_tensors`, records the original shape in
    /// `tensor_shapes`, and deliberately leaves `network.inputs` unchanged so
    /// only live activation inputs remain bounded during verification.
    pub fn frozen_input(
        mut self,
        name: impl Into<String>,
        declared_shape: &[i64],
        value: ArrayD<f32>,
    ) -> Self {
        let name = name.into();
        self.weights.insert(name.clone(), value);
        self.constant_tensors.insert(name.clone());
        self.tensor_shapes.insert(name, declared_shape.to_vec());
        self
    }

    /// Record tensor producer metadata for traced structural ops.
    pub fn tensor_producer(
        mut self,
        tensor_name: impl Into<String>,
        producer_name: impl Into<String>,
    ) -> Self {
        self.tensor_producer
            .insert(tensor_name.into(), producer_name.into());
        self
    }

    /// Mark a tensor as constant with respect to activation inputs.
    pub fn constant_tensor(mut self, tensor_name: impl Into<String>) -> Self {
        self.constant_tensors.insert(tensor_name.into());
        self
    }

    /// Record a known tensor shape for conversion-time reasoning.
    pub fn tensor_shape(mut self, tensor_name: impl Into<String>, shape: &[i64]) -> Self {
        self.tensor_shapes
            .insert(tensor_name.into(), shape.to_vec());
        self
    }

    /// Set the trainable parameter count reported by the model.
    pub fn param_count(mut self, param_count: usize) -> Self {
        self.param_count = param_count;
        self
    }

    /// Assemble the owned graph handoff contract.
    #[must_use]
    pub fn build(self) -> GraphModel {
        let network = Network {
            name: self.name,
            inputs: self.inputs,
            outputs: self.outputs,
            layers: self.layers,
            param_count: self.param_count,
        };

        GraphModel::new(network, self.weights)
            .with_tensor_producer(self.tensor_producer)
            .with_constant_tensors(self.constant_tensors)
            .with_tensor_shapes(self.tensor_shapes)
    }
}

#[cfg(test)]
mod tests {
    use super::GraphModelBuilder;
    use crate::{DataType, GraphModel, LayerSpec};
    use ndarray::{ArrayD, IxDyn};
    use ny_core::LayerType;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn builder_assembles_owned_graph_model_metadata() {
        let graph_model = GraphModelBuilder::new("builder-contract")
            .input("input", &[1, 2], DataType::Float32)
            .output("relu_out", &[1, 2], DataType::Float32)
            .weight(
                "bias",
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).expect("valid bias tensor"),
            )
            .layer(LayerSpec {
                name: "relu".to_string(),
                layer_type: LayerType::ReLU,
                inputs: vec!["input".to_string()],
                outputs: vec!["relu_out".to_string()],
                weights: None,
                attributes: HashMap::new(),
            })
            .tensor_producer("relu_out", "input")
            .constant_tensor("bias")
            .tensor_shape("input", &[1, 2])
            .tensor_shape("relu_out", &[1, 2])
            .param_count(2)
            .build();

        assert_eq!(graph_model.network.name, "builder-contract");
        assert_eq!(graph_model.network.inputs[0].name, "input");
        assert_eq!(graph_model.network.outputs[0].name, "relu_out");
        assert_eq!(graph_model.network.param_count, 2);
        assert_eq!(graph_model.network.layers.len(), 1);
        assert!(graph_model.weights.contains_key("bias"));
        assert_eq!(
            graph_model
                .tensor_producer
                .get("relu_out")
                .map(String::as_str),
            Some("input")
        );
        assert!(graph_model.constant_tensors.contains("bias"));
        assert_eq!(graph_model.tensor_shapes.get("input"), Some(&vec![1, 2]));
        assert_eq!(graph_model.tensor_shapes.get("relu_out"), Some(&vec![1, 2]));
    }

    #[test]
    fn frozen_input_records_contract_without_adding_network_input() {
        let graph_model = GraphModelBuilder::new("frozen-input-contract")
            .input("activation", &[1, 1, 2], DataType::Float32)
            .output("out", &[1, 1, 2], DataType::Float32)
            .frozen_input(
                "style",
                &[1, 2],
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 20.0])
                    .expect("valid frozen style tensor"),
            )
            .build();

        assert_eq!(
            graph_model
                .network
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            vec!["activation"],
            "frozen auxiliary inputs should not become bounded network inputs"
        );
        assert!(
            graph_model.weights.contains_key("style"),
            "frozen auxiliary input should be stored in weights"
        );
        assert!(
            graph_model.constant_tensors.contains("style"),
            "frozen auxiliary input should be marked constant"
        );
        assert_eq!(
            graph_model.tensor_shapes.get("style"),
            Some(&vec![1, 2]),
            "frozen auxiliary input should retain the producer-declared shape"
        );
    }

    fn multi_aux_frozen_graph_model() -> GraphModel {
        GraphModelBuilder::new("frozen-input-multi-aux-contract")
            .input("hidden_states", &[1, 4, 2], DataType::Float32)
            .output("out", &[1, 4, 2], DataType::Float32)
            .frozen_input(
                "cos",
                &[1, 4, 2],
                ArrayD::from_elem(IxDyn(&[4, 2]), 1.0_f32),
            )
            .frozen_input(
                "sin",
                &[1, 4, 2],
                ArrayD::from_elem(IxDyn(&[4, 2]), 2.0_f32),
            )
            .frozen_input(
                "mask",
                &[1, 4, 4],
                ArrayD::from_shape_vec(
                    IxDyn(&[4, 4]),
                    vec![
                        1.0_f32, 1.0, 9.0, 9.0, //
                        1.0, 1.0, 9.0, 9.0, //
                        1.0, 1.0, 9.0, 9.0, //
                        1.0, 1.0, 9.0, 9.0,
                    ],
                )
                .expect("valid frozen mask tensor"),
            )
            .build()
    }

    fn assert_multi_aux_frozen_contract(graph_model: &GraphModel) {
        assert_eq!(
            graph_model
                .network
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            vec!["hidden_states"],
            "multi-frozen auxiliary inputs should not become bounded network inputs"
        );
        assert_eq!(
            graph_model
                .constant_tensors
                .iter()
                .map(String::as_str)
                .collect::<HashSet<_>>(),
            HashSet::from(["cos", "sin", "mask"]),
            "multi-frozen auxiliary inputs should be recorded exactly in constant_tensors"
        );
        assert_eq!(
            graph_model.tensor_shapes.get("cos"),
            Some(&vec![1, 4, 2]),
            "cos should retain the producer-declared batched shape"
        );
        assert_eq!(
            graph_model.tensor_shapes.get("sin"),
            Some(&vec![1, 4, 2]),
            "sin should retain the producer-declared batched shape"
        );
        assert_eq!(
            graph_model.tensor_shapes.get("mask"),
            Some(&vec![1, 4, 4]),
            "mask should retain the producer-declared batched shape"
        );
        assert_eq!(
            graph_model
                .weights
                .get("cos")
                .expect("cos should be stored in weights")
                .shape(),
            &[4_usize, 2_usize],
            "cos should be stored unbatched in propagation layout"
        );
        assert_eq!(
            graph_model
                .weights
                .get("sin")
                .expect("sin should be stored in weights")
                .shape(),
            &[4_usize, 2_usize],
            "sin should be stored unbatched in propagation layout"
        );
        assert_eq!(
            graph_model
                .weights
                .get("mask")
                .expect("mask should be stored in weights")
                .shape(),
            &[4_usize, 4_usize],
            "mask should be stored unbatched in propagation layout"
        );
    }

    #[test]
    fn frozen_input_records_multi_aux_contract_without_adding_network_inputs_3924() {
        let graph_model = multi_aux_frozen_graph_model();
        assert_multi_aux_frozen_contract(&graph_model);
    }
}
