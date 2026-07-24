// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Options for DAG-based graph network construction.

/// Policy for builder-local compound-node rewrites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompoundNodePolicy {
    /// Preserve the original model inventory.
    Preserve,
    /// Rewrite supported normalization nodes (LayerNorm, RmsNorm, InstanceNorm)
    /// into primitive graph ops for CROWN backward propagation.
    DecomposeNormalization,
}

/// Options controlling DAG graph network conversion.
#[derive(Clone, Copy, Debug)]
pub struct GraphNetworkOptions {
    /// If true, skip Reshape layers whose target shape is not statically known.
    /// This is a best-effort mode and may be unsound for shape-sensitive models.
    pub allow_dynamic_reshape: bool,
    /// Optional override to select a specific ONNX output by index.
    pub output_index: Option<usize>,
    /// Behavior when declared ONNX outputs cannot be resolved to graph nodes.
    pub missing_output_policy: MissingOutputPolicy,
    /// Policy for builder-local compound-node rewrites.
    pub compound_node_policy: CompoundNodePolicy,
}

/// Behavior when declared ONNX outputs cannot be resolved to graph nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingOutputPolicy {
    /// Treat missing outputs as conversion errors.
    Error,
    /// Emit a warning and fall back to the last added node.
    WarnAndFallback,
}

impl Default for GraphNetworkOptions {
    fn default() -> Self {
        Self {
            allow_dynamic_reshape: false,
            output_index: None,
            missing_output_policy: MissingOutputPolicy::Error,
            compound_node_policy: CompoundNodePolicy::Preserve,
        }
    }
}

impl GraphNetworkOptions {
    /// Allow skipping dynamic Reshape layers during graph conversion.
    pub fn permissive() -> Self {
        Self {
            allow_dynamic_reshape: true,
            missing_output_policy: MissingOutputPolicy::WarnAndFallback,
            ..Self::default()
        }
    }
}
