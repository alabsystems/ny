// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph node representation.

use ny_core::{NyError, Result};

use crate::layers::Layer;

/// Sentinel node name representing the network's external input.
///
/// All graph entry points use this as their input name. Graph traversal and
/// CROWN backward propagation terminate when they reach a node whose input
/// is `NETWORK_INPUT`.
///
/// Source: α,β-CROWN Python baseline uses `"/input"` in `BoundedModule`.
pub const NETWORK_INPUT: &str = "_input";

/// A node in a computation graph.
///
/// Each node represents a single operation that takes one or more inputs
/// and produces a single output. Nodes can reference other nodes' outputs
/// or the network input.
///
/// Fields are `pub(crate)` to prevent external crates from constructing
/// invalid nodes or mutating them after validation. Use the constructors
/// (`new`, `from_input`, `binary`) and accessor methods (`name`, `layer`,
/// `inputs`) instead.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier for this node.
    pub(crate) name: String,
    /// The layer/operation to apply.
    pub(crate) layer: Layer,
    /// Names of input nodes. For unary ops, this has 1 element.
    /// For binary ops (MatMul, Add), this has 2 elements.
    /// Special value: [`NETWORK_INPUT`] refers to the network input.
    pub(crate) inputs: Vec<String>,
}

impl GraphNode {
    // -- Accessors --

    /// Returns the node's unique name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the node's layer/operation.
    #[inline]
    pub fn layer(&self) -> &Layer {
        &self.layer
    }

    /// Returns the names of this node's inputs.
    #[inline]
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    // -- Constructors --

    /// Create a new graph node.
    ///
    /// Asserts that the input count meets the layer's minimum arity
    /// requirement (`Layer::min_inputs`). This catches construction errors early
    /// instead of deferring to propagation-time `require_*_inputs()` checks (#2481).
    /// Validated in both debug and release builds (#2686).
    pub fn new(name: impl Into<String>, layer: Layer, inputs: Vec<String>) -> Self {
        let name = name.into();
        assert!(
            inputs.len() >= layer.min_inputs(),
            "GraphNode '{name}' ({}) requires at least {} input(s) but got {}",
            layer.layer_type(),
            layer.min_inputs(),
            inputs.len(),
        );
        Self {
            name,
            layer,
            inputs,
        }
    }

    /// Fallible constructor: returns `InvalidSpec` if input count is below
    /// the layer's minimum arity, instead of panicking.
    ///
    /// Use this in tests that intentionally construct malformed nodes to verify
    /// that propagation returns structured errors (#2099, #2991).
    pub fn try_new(name: impl Into<String>, layer: Layer, inputs: Vec<String>) -> Result<Self> {
        let name = name.into();
        if inputs.len() < layer.min_inputs() {
            return Err(NyError::InvalidSpec(format!(
                "GraphNode '{name}' ({}) requires at least {} input(s) but got {}",
                layer.layer_type(),
                layer.min_inputs(),
                inputs.len(),
            )));
        }
        Ok(Self {
            name,
            layer,
            inputs,
        })
    }

    /// Create a node that takes network input.
    pub fn from_input(name: impl Into<String>, layer: Layer) -> Self {
        Self::new(name, layer, vec![NETWORK_INPUT.to_string()])
    }

    /// Create a binary operation node.
    pub fn binary(
        name: impl Into<String>,
        layer: Layer,
        input_a: impl Into<String>,
        input_b: impl Into<String>,
    ) -> Self {
        Self::new(name, layer, vec![input_a.into(), input_b.into()])
    }

    /// Returns the first input name, or `InvalidSpec` unless the node has exactly 1 input.
    ///
    /// Use this instead of direct `node.inputs[0]` to avoid panics on malformed
    /// graph nodes with empty input lists (#2099). Rejects extra inputs that
    /// would otherwise be silently truncated by unary graph helpers.
    #[inline]
    pub fn require_unary_input(&self) -> Result<&str> {
        if self.inputs.len() != 1 {
            return Err(NyError::InvalidSpec(format!(
                "Node '{}' ({}) requires exactly 1 input but has {}",
                self.name,
                self.layer.layer_type(),
                self.inputs.len()
            )));
        }
        Ok(self.inputs[0].as_str())
    }

    /// Returns the first two input names, or `InvalidSpec` unless the node has exactly 2 inputs.
    ///
    /// Use this instead of direct `node.inputs[0]`/`node.inputs[1]` to avoid panics
    /// on malformed graph nodes (#2099). Rejects extra inputs that would be silently
    /// dropped by fixed-arity operators (#2666).
    #[inline]
    pub fn require_binary_inputs(&self) -> Result<(&str, &str)> {
        if self.inputs.len() != 2 {
            return Err(NyError::InvalidSpec(format!(
                "Node '{}' ({}) requires exactly 2 inputs but has {}",
                self.name,
                self.layer.layer_type(),
                self.inputs.len()
            )));
        }
        Ok((self.inputs[0].as_str(), self.inputs[1].as_str()))
    }

    /// Returns the first three input names, or `InvalidSpec` unless the node has exactly 3 inputs.
    ///
    /// Use this for ternary operators (for example `Where`) so malformed graph
    /// edges fail with a structured error instead of falling through a fallback path.
    #[inline]
    pub fn require_ternary_inputs(&self) -> Result<(&str, &str, &str)> {
        if self.inputs.len() != 3 {
            return Err(NyError::InvalidSpec(format!(
                "Node '{}' ({}) requires exactly 3 inputs but has {}",
                self.name,
                self.layer.layer_type(),
                self.inputs.len()
            )));
        }
        Ok((
            self.inputs[0].as_str(),
            self.inputs[1].as_str(),
            self.inputs[2].as_str(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{AddLayer, Layer};

    /// Regression test for #2666: require_binary_inputs rejects extra inputs.
    #[test]
    fn require_binary_inputs_rejects_extra_inputs_2666() {
        let node = GraphNode {
            name: "add_malformed".to_string(),
            layer: Layer::Add(AddLayer),
            inputs: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        };
        let error = node
            .require_binary_inputs()
            .expect_err("3-input binary node should be rejected");
        assert!(
            matches!(
                &error,
                NyError::InvalidSpec(msg) if msg.contains("exactly 2 inputs")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #2666: require_binary_inputs still works for correct arity.
    #[test]
    fn require_binary_inputs_accepts_exact_arity() {
        let node = GraphNode {
            name: "add_ok".to_string(),
            layer: Layer::Add(AddLayer),
            inputs: vec!["a".to_string(), "b".to_string()],
        };
        let (a, b) = node
            .require_binary_inputs()
            .expect("2-input binary node should succeed");
        assert_eq!(a, "a");
        assert_eq!(b, "b");
    }

    /// Regression test for #2666: require_binary_inputs still rejects insufficient inputs.
    #[test]
    fn require_binary_inputs_rejects_insufficient_inputs() {
        let node = GraphNode {
            name: "add_short".to_string(),
            layer: Layer::Add(AddLayer),
            inputs: vec!["a".to_string()],
        };
        let error = node
            .require_binary_inputs()
            .expect_err("1-input binary node should be rejected");
        assert!(
            matches!(
                &error,
                NyError::InvalidSpec(msg) if msg.contains("exactly 2 inputs")
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Regression test for #4097: unary helpers must reject malformed extra inputs
    /// instead of silently truncating to the first edge.
    #[test]
    fn require_unary_input_rejects_extra_inputs_4097() {
        let node = GraphNode {
            name: "relu_malformed".to_string(),
            layer: Layer::ReLU(crate::layers::ReLULayer),
            inputs: vec!["a".to_string(), "b".to_string()],
        };
        let error = node
            .require_unary_input()
            .expect_err("2-input unary node should be rejected");
        assert!(
            matches!(
                &error,
                NyError::InvalidSpec(msg) if msg.contains("exactly 1 input")
            ),
            "unexpected error: {error:?}"
        );
    }
}
