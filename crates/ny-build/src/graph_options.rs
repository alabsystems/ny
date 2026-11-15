// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Options for DAG-based graph network construction.

use ny_propagate::layers::PointLinearBuildPolicy;

/// Construction-time affine storage selection. Original point storage is a
/// separate representation, never eager fields filled with empty placeholders.
#[derive(Clone, Copy, Debug, Default)]
pub enum LinearStoragePolicy<'a> {
    #[default]
    EagerBounds,
    OriginalPoint(PointLinearBuildPolicy<'a>),
}

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
    /// Dark multi-input mechanism (#multi-input-propagation): when true and the
    /// model declares MORE THAN ONE graph input, the builder no longer aliases
    /// every input tensor to the single `_input` node (which is only sound when
    /// there is exactly one input; with several it wires every branch to the
    /// FIRST input's box and fails closed downstream on the shape mismatch).
    /// Instead the network input becomes the flat concatenation of all declared
    /// inputs in DECLARED ORDER, and each input tensor is carved back out by an
    /// exact Slice(+Reshape) view node, so bounds and gradients route to the
    /// right coordinate slices. CONTAINMENT: the joint box is exactly the
    /// product of the per-input boxes — Slice/Reshape are coordinate
    /// selections, no widening. Single-input models take the identical legacy
    /// path regardless of this flag. Default OFF: multi-input models keep
    /// today's fail-closed refusal byte-identical.
    pub multi_input_concat: bool,
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
            multi_input_concat: false,
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
