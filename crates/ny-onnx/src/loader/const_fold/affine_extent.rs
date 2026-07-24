// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Affine-extent symbolic shape deduction for variable-start slices
//! (#cctsdb B2).
//!
//! Windowed-mask graphs (cctsdb_yolo_2023) slice constant tensors with
//! DATA-DEPENDENT starts/ends built from the same scalar: `Slice(data,
//! starts=[x], ends=[x+w])`. ORT shape inference reports the output extent as
//! dynamic, so `Shape`-of-slice chains never const-fold and the whole
//! downstream shape cluster (ConstantOfShape/Expand/Reshape) degrades to
//! OpaqueSkip. But the EXTENT `ends - starts = w` is a compile-time constant
//! whenever both endpoints are affine forms `coeff * s + offset` of the SAME
//! symbol `s` with equal coefficients.
//!
//! This pass deduces such static output shapes and records them into
//! `inferred_shapes`, letting the existing const-fold cascade
//! (`try_fold_shape_node` -> ConstantOfShape/Expand/Where/Reshape) run.
//!
//! Clamp caveat (design B2, patch-3): ONNX clamps `starts`/`ends` to
//! `[0, dim]`, so a window touching the right edge yields FEWER elements at
//! runtime than the static extent recorded here (x=62, w=3, dim=64 -> true
//! extent 2, recorded 3). We deliberately record the UNCLAMPED maximum
//! extent: the internal representation is static-shape, and the consumers of
//! these windows (zero-filled updates, scatter index coordinates) tolerate
//! the extra positions — the bounded-index ScatterND (B4) rejects the
//! out-of-range sentinel rows, matching exactly the writes the true clipped
//! graph performs.

use std::collections::HashMap;

use crate::onnx_proto;
use crate::WeightStore;
use tracing::debug;

use super::common::read_tensor_i64s;

/// An affine form `coeff * root + offset` over a single opaque symbol.
///
/// `root` is a tensor name: equality of names implies equality of runtime
/// values (same tensor), which is the only property the extent computation
/// relies on. A pure constant has `root == ""` and `coeff == 0`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AffineForm {
    root: String,
    coeff: i64,
    offset: i64,
}

impl AffineForm {
    fn constant(offset: i64) -> Self {
        Self {
            root: String::new(),
            coeff: 0,
            offset,
        }
    }

    fn symbol(root: &str) -> Self {
        Self {
            root: root.to_string(),
            coeff: 1,
            offset: 0,
        }
    }

    fn is_constant(&self) -> bool {
        self.root.is_empty()
    }

    /// `self + sign * other`, when the symbolic parts are compatible
    /// (same root, or one side constant).
    fn combine(&self, other: &AffineForm, sign: i64) -> Option<AffineForm> {
        let root = if self.is_constant() {
            other.root.clone()
        } else if other.is_constant() || other.root == self.root {
            self.root.clone()
        } else {
            return None; // different symbols: not affine in one variable
        };
        Some(AffineForm {
            root,
            coeff: self.coeff.checked_add(sign.checked_mul(other.coeff)?)?,
            offset: self.offset.checked_add(sign.checked_mul(other.offset)?)?,
        })
    }
}

const MAX_TRACE_DEPTH: usize = 24;

/// Trace the affine form of a single-element integer tensor.
///
/// Every step is VALUE-exact for 1-element tensors:
/// - constants read their concrete value;
/// - `Unsqueeze`/`Squeeze`/`Identity`/`Reshape`/`Flatten` preserve the value;
/// - `Add`/`Sub` combine affine forms (constant or same-symbol sides);
/// - anything else (Gather-of-input, Cast, ...) becomes an OPAQUE SYMBOL —
///   its runtime value is unknown but self-identical, which is all the
///   extent difference needs. (Cast is deliberately NOT traced through:
///   float->int truncation is not affine; the cast OUTPUT is the symbol.)
///
/// The starting tensor is a Slice starts/ends operand of a single-axis slice,
/// hence 1-element; all traced ops preserve the element count backward.
fn affine_scalar_form(
    name: &str,
    node_by_output: &HashMap<&str, &onnx_proto::NodeProto>,
    weights: &WeightStore,
    depth: usize,
) -> Option<AffineForm> {
    if depth == 0 || name.is_empty() {
        return None;
    }
    if let Some(values) = read_tensor_i64s(weights, name) {
        return (values.len() == 1).then(|| AffineForm::constant(values[0]));
    }
    let Some(node) = node_by_output.get(name) else {
        return Some(AffineForm::symbol(name));
    };
    match node.op_type.as_str() {
        "Unsqueeze" | "Squeeze" | "Identity" | "Reshape" | "Flatten" => {
            affine_scalar_form(node.input.first()?, node_by_output, weights, depth - 1)
        }
        "Add" | "Sub" if node.input.len() >= 2 => {
            let lhs = affine_scalar_form(&node.input[0], node_by_output, weights, depth - 1)?;
            let rhs = affine_scalar_form(&node.input[1], node_by_output, weights, depth - 1)?;
            let sign = if node.op_type == "Sub" { -1 } else { 1 };
            lhs.combine(&rhs, sign)
        }
        _ => Some(AffineForm::symbol(name)),
    }
}

/// Deduce static output shapes for variable-start slices whose extent is
/// affine-constant, recording them into `inferred_shapes`.
///
/// Runs to an internal fixpoint (chained slices feed each other's data
/// shapes). Returns `true` if any new shape was recorded — the caller should
/// re-run constant folding so `Shape`-of-slice chains cascade.
pub(super) fn augment_inferred_shapes_with_affine_slice_extents(
    graph: &onnx_proto::GraphProto,
    weights: &WeightStore,
    inferred_shapes: &mut HashMap<String, Vec<i64>>,
) -> bool {
    let mut node_by_output: HashMap<&str, &onnx_proto::NodeProto> = HashMap::new();
    for node in &graph.node {
        for output in &node.output {
            if !output.is_empty() {
                node_by_output.insert(output.as_str(), node);
            }
        }
    }

    let mut any_added = false;
    let mut changed = true;
    while changed {
        changed = false;
        for node in &graph.node {
            if node.op_type != "Slice" {
                continue;
            }
            let Some(output_name) = node.output.first().filter(|name| !name.is_empty()) else {
                continue;
            };
            let known_static = inferred_shapes
                .get(output_name.as_str())
                .is_some_and(|shape| shape.iter().all(|&dim| dim > 0));
            if known_static {
                continue;
            }
            let Some(shape) =
                deduce_affine_slice_shape(node, &node_by_output, weights, inferred_shapes)
            else {
                continue;
            };
            debug!(
                "Affine-extent slice shape: {} (node {}) -> {:?}",
                output_name, node.name, shape
            );
            inferred_shapes.insert(output_name.clone(), shape);
            any_added = true;
            changed = true;
        }
    }
    any_added
}

/// The static (max-extent) output shape of a single-axis Slice with affine
/// starts/ends over the same symbol, or `None`.
fn deduce_affine_slice_shape(
    node: &onnx_proto::NodeProto,
    node_by_output: &HashMap<&str, &onnx_proto::NodeProto>,
    weights: &WeightStore,
    inferred_shapes: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    let data_name = node.input.first().filter(|name| !name.is_empty())?;
    let starts_name = node.input.get(1).filter(|name| !name.is_empty())?;
    let ends_name = node.input.get(2).filter(|name| !name.is_empty())?;

    // Data shape: prior inference (ORT or this pass) or a constant's shape.
    let data_shape: Vec<i64> = inferred_shapes
        .get(data_name.as_str())
        .cloned()
        .or_else(|| {
            weights
                .get(data_name)
                .map(|w| w.shape().iter().map(|&d| d as i64).collect())
        })?;
    if data_shape.iter().any(|&dim| dim <= 0) {
        return None;
    }

    // Single sliced axis only: starts/ends are then 1-element tensors, the
    // shape the affine tracer is exact for.
    let axes: Vec<i64> = match node.input.get(3).filter(|name| !name.is_empty()) {
        Some(name) => read_tensor_i64s(weights, name)?,
        None => vec![0],
    };
    if axes.len() != 1 {
        return None;
    }
    // Steps must be 1 (or absent): other steps change the extent formula.
    if let Some(steps_name) = node.input.get(4).filter(|name| !name.is_empty()) {
        let steps = read_tensor_i64s(weights, steps_name)?;
        if steps != vec![1] {
            return None;
        }
    }
    let rank = data_shape.len() as i64;
    let axis = if axes[0] < 0 { axes[0] + rank } else { axes[0] };
    if axis < 0 || axis >= rank {
        return None;
    }
    let axis = axis as usize;

    let starts = affine_scalar_form(starts_name, node_by_output, weights, MAX_TRACE_DEPTH)?;
    let ends = affine_scalar_form(ends_name, node_by_output, weights, MAX_TRACE_DEPTH)?;
    if starts.is_constant() && ends.is_constant() {
        // Fully constant: the existing const-fold/shape-inference machinery
        // owns exact (clamped) semantics; do not duplicate them here.
        return None;
    }
    if starts.root != ends.root || starts.coeff != ends.coeff {
        return None;
    }

    // Constant extent, recorded as the UNCLAMPED maximum (see module docs for
    // the right-edge caveat), capped to the axis length and floored at 0.
    let extent = ends
        .offset
        .checked_sub(starts.offset)?
        .clamp(0, data_shape[axis]);
    let mut output_shape = data_shape;
    output_shape[axis] = extent;
    Some(output_shape)
}

#[cfg(test)]
#[path = "affine_extent_tests.rs"]
mod tests;
