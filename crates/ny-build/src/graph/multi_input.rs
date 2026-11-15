// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-input graph propagation: the concat-view adapter (#multi-input-propagation).
//!
//! The propagation stack has exactly ONE input terminal, the `_input` node
//! (`NETWORK_INPUT`), and that identity is load-bearing across ny-propagate
//! (CROWN backward terminates on it, soundness/equivalence/imb all key on it).
//! A model with several declared graph inputs therefore cannot be represented
//! directly. This module implements the standard treatment instead:
//!
//! - the network input becomes the FLAT CONCATENATION of every declared input,
//!   in DECLARED ORDER (the same order the VNN-LIB `X_i` variables enumerate
//!   multi-input models: input 0 flattened row-major, then input 1, ...);
//! - each declared input tensor is carved back out of that flat vector by an
//!   exact `Slice` view node (plus a `Reshape` when the input is not rank-1),
//!   and every consumer of the input tensor is wired to its view node.
//!
//! CONTAINMENT: a box over the concatenated vector, restricted to a
//! coordinate slice, is EXACTLY the per-input box — `Slice` and `Reshape` are
//! coordinate selections (0/1 linear maps), so IBP intervals and CROWN linear
//! bounds route to the right slices with zero widening. The joint box is
//! exactly the product of the per-input boxes.
//!
//! FAIL-CLOSED: the layout is verified to be an exact partition (offset 0,
//! contiguous, covering, declared order) before any node is emitted; any
//! dynamic non-batch dimension refuses. The alternative design — a true
//! multi-node input layer inside ny-propagate — was rejected as the larger
//! change: it would re-thread the `NETWORK_INPUT` sentinel through every
//! backward/soundness/equivalence path in the propagation stack, a wide-blast
//! bound-path rewrite, where this adapter reuses two existing exactly-bounded
//! layers at the boundary and leaves the propagation stack untouched.

use super::INPUT_NODE_NAME;
use crate::TensorSpec;
use ny_core::{NyError, Result};
use ny_propagate::layers::{ReshapeLayer, SliceLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use std::collections::HashMap;

/// One declared input's slot inside the flat concatenated network input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatInputSlot {
    /// Declared input tensor name.
    pub name: String,
    /// Start coordinate (inclusive) in the flat concatenated input.
    pub offset: usize,
    /// Flattened element count of this input.
    pub len: usize,
    /// Internal (batch-squeezed) shape this input's consumers expect.
    pub internal_shape: Vec<usize>,
}

/// Declared-order layout of every graph input inside the flat concatenated
/// network input. Construct via [`concat_input_layout`]; `verify_partition`
/// re-checks the invariants and is the fail-closed chokepoint every consumer
/// runs before trusting a layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatInputLayout {
    slots: Vec<ConcatInputSlot>,
    total_len: usize,
}

impl ConcatInputLayout {
    /// Per-input slots in declared input order.
    pub fn slots(&self) -> &[ConcatInputSlot] {
        &self.slots
    }

    /// Total flattened element count of the concatenated network input.
    pub fn total_len(&self) -> usize {
        self.total_len
    }

    /// Fail-closed containment check: the slots must form an EXACT partition
    /// of `[0, total_len)` — first offset 0, each slot contiguous with the
    /// previous, lengths matching the slot's internal shape product, and the
    /// final slot ending at `total_len`. Any violation is a hard error; a
    /// layout that fails here must never reach the graph.
    pub fn verify_partition(&self) -> Result<()> {
        let mut cursor = 0usize;
        for (index, slot) in self.slots.iter().enumerate() {
            if slot.offset != cursor {
                return Err(NyError::ModelLoad(format!(
                    "multi-input concat layout violation: input {index} ('{}') starts at \
                     offset {} but the previous slot ends at {} — the joint box would not \
                     be the exact product of the per-input boxes",
                    slot.name, slot.offset, cursor
                )));
            }
            let shape_len: usize = slot.internal_shape.iter().product();
            if slot.len == 0 || shape_len != slot.len {
                return Err(NyError::ModelLoad(format!(
                    "multi-input concat layout violation: input {index} ('{}') declares \
                     {} elements but its internal shape {:?} holds {}",
                    slot.name, slot.len, slot.internal_shape, shape_len
                )));
            }
            cursor = cursor
                .checked_add(slot.len)
                .ok_or_else(|| NyError::ModelLoad("multi-input layout overflow".to_string()))?;
        }
        if cursor != self.total_len {
            return Err(NyError::ModelLoad(format!(
                "multi-input concat layout violation: slots cover {} elements but the \
                 layout claims {}",
                cursor, self.total_len
            )));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn corrupt_slot_offset_for_test(&mut self, index: usize, delta: isize) {
        let slot = &mut self.slots[index];
        slot.offset = slot.offset.checked_add_signed(delta).expect("test offset");
    }
}

/// Resolve one declared input's internal (batch-squeezed) shape, failing
/// closed on anything the layout cannot prove:
///
/// - a non-positive (dynamic) dimension anywhere but a leading batch axis
///   refuses (the flat offset arithmetic would be unprovable);
/// - a leading batch axis must be 1 (or dynamic, resolved to 1) when the model
///   is batched — mirroring `internal_shape_from_onnx_shape`, whose squeeze
///   convention every downstream axis adjustment assumes.
fn internal_input_shape(spec: &TensorSpec, model_unbatched: bool) -> Result<Vec<usize>> {
    let mut dims: Vec<i64> = spec.shape.clone();
    if !model_unbatched && dims.len() >= 2 {
        // Batched model: the leading axis is the batch axis; a dynamic batch
        // resolves to 1 exactly as `resolve_dynamic_shape(shape, 1)` does.
        if dims[0] <= 0 {
            dims[0] = 1;
        }
        if dims[0] != 1 {
            return Err(NyError::ModelLoad(format!(
                "multi-input concat: input '{}' has batch dimension {} (shape {:?}); \
                 only batch 1 is supported",
                spec.name, dims[0], spec.shape
            )));
        }
        dims.remove(0);
    }
    if dims.iter().any(|&dim| dim <= 0) {
        return Err(NyError::ModelLoad(format!(
            "multi-input concat: input '{}' has a dynamic non-batch dimension \
             (shape {:?}); the flat input layout cannot be proven",
            spec.name, spec.shape
        )));
    }
    Ok(dims.iter().map(|&dim| dim as usize).collect())
}

/// Build the declared-order concat layout for a multi-input model.
///
/// The result is already `verify_partition`-checked. Duplicate input names
/// refuse (a duplicate would make `tensor_to_node` routing ambiguous).
pub fn concat_input_layout(inputs: &[TensorSpec]) -> Result<ConcatInputLayout> {
    if inputs.len() < 2 {
        return Err(NyError::ModelLoad(format!(
            "multi-input concat layout requested for {} input(s); the single-input \
             path must stay on the legacy `_input` route",
            inputs.len()
        )));
    }
    let model_unbatched = crate::convert::model_is_unbatched(inputs);
    let mut slots = Vec::with_capacity(inputs.len());
    let mut offset = 0usize;
    for (index, spec) in inputs.iter().enumerate() {
        if inputs[..index].iter().any(|prior| prior.name == spec.name) {
            return Err(NyError::ModelLoad(format!(
                "multi-input concat: duplicate declared input name '{}'",
                spec.name
            )));
        }
        let internal_shape = internal_input_shape(spec, model_unbatched)?;
        let len: usize = internal_shape.iter().product();
        slots.push(ConcatInputSlot {
            name: spec.name.clone(),
            offset,
            len,
            internal_shape,
        });
        offset = offset
            .checked_add(len)
            .ok_or_else(|| NyError::ModelLoad("multi-input layout overflow".to_string()))?;
    }
    let layout = ConcatInputLayout {
        slots,
        total_len: offset,
    };
    layout.verify_partition()?;
    Ok(layout)
}

/// Emit the per-input view nodes into `graph` and route each declared input
/// tensor name to its view node in `tensor_to_node`.
///
/// Every view is `Slice(axis 0, offset..offset+len)` over the flat
/// `[total_len]` network input, followed by a `Reshape` to the input's
/// internal shape when that shape is not already `[len]`. Both are exact
/// coordinate selections; bound propagation through them is width-preserving
/// per coordinate.
pub(super) fn insert_input_view_nodes(
    graph: &mut GraphNetwork,
    tensor_to_node: &mut HashMap<String, String>,
    layout: &ConcatInputLayout,
) -> Result<()> {
    // Fail-closed: never emit from an unverified layout.
    layout.verify_partition()?;
    for (index, slot) in layout.slots().iter().enumerate() {
        let slice_name = format!("_input_view_{index}");
        let mut slice = SliceLayer::new(0, slot.offset, slot.offset + slot.len);
        slice.set_input_shape(vec![layout.total_len()]);
        graph.try_add_node(GraphNode::new(
            slice_name.clone(),
            Layer::Slice(slice),
            vec![INPUT_NODE_NAME.to_string()],
        ))?;
        graph.set_declared_shape(slice_name.clone(), vec![slot.len]);
        let view_name = if slot.internal_shape == [slot.len] {
            slice_name
        } else {
            let reshape_name = format!("_input_view_{index}_reshape");
            let target: Vec<i64> = slot.internal_shape.iter().map(|&dim| dim as i64).collect();
            graph.try_add_node(GraphNode::new(
                reshape_name.clone(),
                Layer::Reshape(ReshapeLayer::new(target)),
                vec![slice_name],
            ))?;
            graph.set_declared_shape(reshape_name.clone(), slot.internal_shape.clone());
            reshape_name
        };
        tensor_to_node.insert(slot.name.clone(), view_name);
    }
    Ok(())
}

#[cfg(test)]
#[path = "multi_input_tests.rs"]
mod tests;
