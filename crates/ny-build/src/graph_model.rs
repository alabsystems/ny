// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Owned graph-build contract for external graph producers.
//!
//! `GraphModel` packages the model specification, weights, and auxiliary graph
//! metadata required by [`build_graph_network`](crate::build_graph_network).
//! External integrations that trace imperative model code (for example a
//! dynamic-tensor tracing path) can populate this contract directly instead of
//! reverse-engineering `GraphBuildInputs`.
//! External traced producers should target this surface (or
//! [`crate::GraphModelBuilder`]) and
//! [`GraphModel::build_graph_network`], not direct `ny_propagate` graph
//! construction. For traced multi-input models, frozen auxiliary inputs should
//! be stored as unbatched weight tensors while `tensor_shapes` retains the
//! original producer-declared shape used during conversion-time reasoning.

use crate::{
    build_graph_network, GraphBuildInputs, GraphNetworkOptions, MixedPrecisionPolicy, Network,
    WeightStore,
};
use ny_core::Result;
use ny_propagate::GraphNetwork;
use std::collections::{HashMap, HashSet};

/// Owned model + metadata needed to build a verification graph.
#[derive(Debug, Clone)]
pub struct GraphModel {
    /// Model specification in the neutral ny-build format.
    pub network: Network,
    /// Weight tensors referenced by the model specification.
    pub weights: WeightStore,
    /// Tensor name -> producer tensor mapping for tracing through structural ops.
    pub tensor_producer: HashMap<String, String>,
    /// Tensor names that are constant with respect to activation inputs,
    /// including frozen auxiliary traced inputs.
    pub constant_tensors: HashSet<String>,
    /// Known tensor shapes keyed by tensor name, retaining producer-declared
    /// shapes even when frozen traced inputs are stored unbatched in `weights`.
    pub tensor_shapes: HashMap<String, Vec<i64>>,
    /// Optional mixed-precision policy for verifying at the deployed precision (P8).
    ///
    /// ADDITIVE + OPT-IN: `None` (the default from every existing constructor)
    /// means today's pure-f32 idealization with no widening. `Some(policy)`
    /// requests SOUND widening to the policy's compute/accumulate precisions.
    pub mixed_precision: Option<MixedPrecisionPolicy>,
}

impl GraphModel {
    /// Create a graph-build contract with empty graph metadata.
    #[must_use]
    pub fn new(network: Network, weights: WeightStore) -> Self {
        Self {
            network,
            weights,
            tensor_producer: HashMap::new(),
            constant_tensors: HashSet::new(),
            tensor_shapes: HashMap::new(),
            mixed_precision: None,
        }
    }

    /// Set tensor producer metadata used to trace through structural ops.
    #[must_use]
    pub fn with_tensor_producer(mut self, tensor_producer: HashMap<String, String>) -> Self {
        self.tensor_producer = tensor_producer;
        self
    }

    /// Set the constant tensor set used during graph conversion.
    #[must_use]
    pub fn with_constant_tensors(mut self, constant_tensors: HashSet<String>) -> Self {
        self.constant_tensors = constant_tensors;
        self
    }

    /// Set known tensor shapes for conversion-time shape reasoning.
    #[must_use]
    pub fn with_tensor_shapes(mut self, tensor_shapes: HashMap<String, Vec<i64>>) -> Self {
        self.tensor_shapes = tensor_shapes;
        self
    }

    /// Opt in to a mixed-precision verification policy (P8).
    ///
    /// ADDITIVE: leaving this unset preserves the default `None` (pure-f32
    /// idealization). Setting a non-F32 policy requests SOUND widening to the
    /// deployed precision.
    #[must_use]
    pub fn with_mixed_precision(mut self, policy: MixedPrecisionPolicy) -> Self {
        self.mixed_precision = Some(policy);
        self
    }

    /// Borrow this owned contract as the low-level graph builder inputs.
    #[must_use]
    pub fn graph_build_inputs(&self) -> GraphBuildInputs<'_> {
        GraphBuildInputs {
            layers: &self.network.layers,
            inputs: &self.network.inputs,
            outputs: &self.network.outputs,
            weights: &self.weights,
            tensor_producer: &self.tensor_producer,
            constant_tensors: &self.constant_tensors,
            tensor_shapes: &self.tensor_shapes,
        }
    }

    /// Build a verification [`GraphNetwork`] from this contract.
    pub fn build_graph_network(&self, options: GraphNetworkOptions) -> Result<GraphNetwork> {
        build_graph_network(&self.graph_build_inputs(), options)
    }
}

#[cfg(test)]
mod tests {
    use super::GraphModel;
    use crate::{DataType, LayerSpec, MixedPrecisionPolicy, Network, TensorSpec, WeightStore};
    use ny_core::{FloatPrecision, LayerType};
    use std::collections::{HashMap, HashSet};

    fn tensor_spec(name: &str, shape: &[i64]) -> TensorSpec {
        TensorSpec {
            name: name.to_string(),
            shape: shape.to_vec(),
            dtype: DataType::Float32,
        }
    }

    fn relu_layer(name: &str, input: &str, output: &str) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::ReLU,
            inputs: vec![input.to_string()],
            outputs: vec![output.to_string()],
            weights: None,
            attributes: HashMap::new(),
        }
    }

    #[test]
    fn graph_model_builds_graph_network_from_owned_contract() {
        let network = Network {
            name: "relu".to_string(),
            inputs: vec![tensor_spec("input", &[1, 2])],
            outputs: vec![tensor_spec("relu_out", &[1, 2])],
            layers: vec![relu_layer("relu", "input", "relu_out")],
            param_count: 0,
        };
        let graph_model = GraphModel::new(network, WeightStore::new())
            .with_tensor_producer(HashMap::from([(
                "relu_out".to_string(),
                "input".to_string(),
            )]))
            .with_constant_tensors(HashSet::new())
            .with_tensor_shapes(HashMap::from([
                ("input".to_string(), vec![1, 2]),
                ("relu_out".to_string(), vec![1, 2]),
            ]));

        let graph = graph_model
            .build_graph_network(crate::GraphNetworkOptions::default())
            .expect("owned graph contract should build");
        let relu = graph.node("relu").expect("relu node should exist");
        assert_eq!(
            relu.inputs(),
            &["_input".to_string()],
            "declared input tensors should still route through the graph input sentinel"
        );
    }

    fn tiny_network() -> Network {
        Network {
            name: "relu".to_string(),
            inputs: vec![tensor_spec("input", &[1, 2])],
            outputs: vec![tensor_spec("relu_out", &[1, 2])],
            layers: vec![relu_layer("relu", "input", "relu_out")],
            param_count: 0,
        }
    }

    #[test]
    fn graph_model_default_has_no_mixed_precision_policy() {
        // ADDITIVE: every existing constructor path must yield today's behavior,
        // i.e. no mixed-precision policy (pure-f32 idealization).
        let model = GraphModel::new(tiny_network(), WeightStore::new());
        assert!(
            model.mixed_precision.is_none(),
            "default GraphModel must not carry a mixed-precision policy"
        );
    }

    #[test]
    fn graph_model_default_policy_is_f32_idealization() {
        // When a caller materializes the implicit policy, it must equal all-f32.
        let model = GraphModel::new(tiny_network(), WeightStore::new());
        let effective = model.mixed_precision.unwrap_or_default();
        assert!(
            effective.is_idealized_f32(),
            "implicit policy must be the f32 idealization"
        );
        assert_eq!(effective.compute, FloatPrecision::F32);
        assert_eq!(effective.accumulate, FloatPrecision::F32);
    }

    #[test]
    fn with_mixed_precision_round_trips_through_builder() {
        let policy = MixedPrecisionPolicy::new(FloatPrecision::F16, FloatPrecision::F32);
        let model =
            GraphModel::new(tiny_network(), WeightStore::new()).with_mixed_precision(policy);
        assert_eq!(
            model.mixed_precision,
            Some(policy),
            "builder must store and return the exact policy that was set"
        );
        // Other builder methods must not clobber the policy.
        let model = model.with_tensor_shapes(HashMap::from([("input".to_string(), vec![1, 2])]));
        assert_eq!(model.mixed_precision, Some(policy));
    }

    #[test]
    fn opting_into_mixed_precision_still_builds_the_same_graph() {
        // Setting a policy is purely additive metadata; graph construction is
        // unchanged (widening is applied downstream, not at build time here).
        let model = GraphModel::new(tiny_network(), WeightStore::new())
            .with_tensor_producer(HashMap::from([(
                "relu_out".to_string(),
                "input".to_string(),
            )]))
            .with_constant_tensors(HashSet::new())
            .with_tensor_shapes(HashMap::from([
                ("input".to_string(), vec![1, 2]),
                ("relu_out".to_string(), vec![1, 2]),
            ]))
            .with_mixed_precision(MixedPrecisionPolicy::uniform(FloatPrecision::Bf16));
        let graph = model
            .build_graph_network(crate::GraphNetworkOptions::default())
            .expect("policy-tagged graph contract should still build");
        assert!(graph.node("relu").is_some());
    }
}
