// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph-DAG GPU-resident IBP plan lowering (#4276, #4318).
//!
//! Lowers a `GraphNetwork` into a [`GpuDagIbpPlanDesc`] when every node uses a
//! supported layer type. Returns `None` on any unsupported node (fail-closed).
//!
//! Supported subset: `Linear`, `Conv2d(groups=1)`, `ReLU`, `Add`, `Flatten`,
//! `Reshape`. This is lowering-only — no WGPU execution in this module.
//!
//! Reference: designs/2026-03-21-issue-4276-dag-ibp-child-packets.md §Packet A

use std::collections::HashMap;

use ny_core::{checked_shape_product, GpuDagIbpOp, GpuDagIbpPlanDesc, NETWORK_INPUT_IDX};

use crate::layers::Layer;

use super::super::{GraphNetwork, NETWORK_INPUT};

/// Try to lower a graph network into a DAG-resident IBP plan descriptor.
///
/// Returns `Some(plan)` only if every node in topological order uses a
/// supported layer. Returns `None` for any unsupported layer, grouped Conv2d,
/// or shape mismatch — the caller falls back to the CPU graph IBP loop.
pub(crate) fn try_lower_graph_dag(
    graph: &GraphNetwork,
    input_shape: &[usize],
) -> Option<GpuDagIbpPlanDesc> {
    let exec_order = graph.exec_order().ok()?;
    if exec_order.is_empty() {
        return None;
    }

    // Map node name → op index for resolving input references.
    let mut name_to_idx: HashMap<&str, usize> = HashMap::with_capacity(exec_order.len());

    // Track output shape of each op for shape validation.
    let mut op_shapes: Vec<Vec<usize>> = Vec::with_capacity(exec_order.len());
    let mut ops: Vec<GpuDagIbpOp> = Vec::with_capacity(exec_order.len());

    for (op_idx, node_name) in exec_order.iter().enumerate() {
        let node = graph.nodes.get(node_name.as_str())?;
        let inputs = node.inputs();

        let (op, output_shape) = match &node.layer {
            Layer::Linear(linear) => {
                let input_idx = resolve_input_idx(inputs.first()?, &name_to_idx)?;
                let input_shape = resolve_shape(input_idx, input_shape, &op_shapes);
                let weight_slice = linear.weight.as_slice()?;
                let (out_features, in_features) = linear.weight.dim();
                let bias = match linear.bias.as_ref() {
                    Some(b) => Some(b.as_slice()?.to_vec().into()),
                    None => None,
                };
                let last_dim = *input_shape.last()?;
                if last_dim != in_features {
                    return None;
                }
                let mut out_shape = input_shape.to_vec();
                *out_shape.last_mut()? = out_features;
                let op = GpuDagIbpOp::Linear {
                    weight: weight_slice.to_vec().into(),
                    bias,
                    out_features,
                    in_features,
                    input_idx,
                };
                (op, out_shape)
            }
            Layer::Conv2d(conv) => {
                if conv.groups != 1 {
                    return None;
                }
                let input_idx = resolve_input_idx(inputs.first()?, &name_to_idx)?;
                let in_shape = resolve_shape(input_idx, input_shape, &op_shapes);
                let (batch_size, input_channels, input_h, input_w) = match in_shape {
                    [c, h, w] => (None, *c, *h, *w),
                    [b, c, h, w] => (Some(*b), *c, *h, *w),
                    _ => return None,
                };
                if input_channels != conv.in_channels() {
                    return None;
                }
                let (out_h, out_w) = conv.output_size(input_h, input_w).ok()?;
                let weight_slice = conv.kernel.as_slice()?;
                let bias = match conv.bias.as_ref() {
                    Some(b) => Some(b.as_slice()?.to_vec().into()),
                    None => None,
                };
                let out_shape = match batch_size {
                    Some(b) => vec![b, conv.out_channels(), out_h, out_w],
                    None => vec![conv.out_channels(), out_h, out_w],
                };
                let op = GpuDagIbpOp::Conv2d {
                    weight: weight_slice.to_vec().into(),
                    bias,
                    out_channels: conv.out_channels(),
                    in_channels: conv.in_channels(),
                    kernel_h: conv.kernel_size().0,
                    kernel_w: conv.kernel_size().1,
                    stride_h: conv.stride.0,
                    stride_w: conv.stride.1,
                    pad_h: conv.padding.0,
                    pad_w: conv.padding.1,
                    groups: conv.groups,
                    input_h,
                    input_w,
                    input_idx,
                };
                (op, out_shape)
            }
            Layer::ReLU(_) => {
                let input_idx = resolve_input_idx(inputs.first()?, &name_to_idx)?;
                let in_shape = resolve_shape(input_idx, input_shape, &op_shapes);
                let num_elements = checked_shape_product(in_shape)?;
                let op = GpuDagIbpOp::ReLU {
                    num_elements,
                    input_idx,
                };
                (op, in_shape.to_vec())
            }
            Layer::Add(_) => {
                if inputs.len() < 2 {
                    return None;
                }
                let input_a_idx = resolve_input_idx(&inputs[0], &name_to_idx)?;
                let input_b_idx = resolve_input_idx(&inputs[1], &name_to_idx)?;
                let shape_a = resolve_shape(input_a_idx, input_shape, &op_shapes);
                let shape_b = resolve_shape(input_b_idx, input_shape, &op_shapes);
                if shape_a != shape_b {
                    return None;
                }
                let num_elements = checked_shape_product(shape_a)?;
                let op = GpuDagIbpOp::Add {
                    num_elements,
                    input_a_idx,
                    input_b_idx,
                };
                (op, shape_a.to_vec())
            }
            Layer::Flatten(f) => {
                let input_idx = resolve_input_idx(inputs.first()?, &name_to_idx)?;
                let in_shape = resolve_shape(input_idx, input_shape, &op_shapes);
                let output_shape = f.compute_output_shape(in_shape).ok()?;
                let op = GpuDagIbpOp::View {
                    output_shape: output_shape.clone().into(),
                    input_idx,
                };
                (op, output_shape)
            }
            Layer::Reshape(r) => {
                let input_idx = resolve_input_idx(inputs.first()?, &name_to_idx)?;
                let in_shape = resolve_shape(input_idx, input_shape, &op_shapes);
                let output_shape = r.compute_output_shape(in_shape).ok()?;
                let op = GpuDagIbpOp::View {
                    output_shape: output_shape.clone().into(),
                    input_idx,
                };
                (op, output_shape)
            }
            Layer::AveragePool(pool) => {
                let input_idx = resolve_input_idx(inputs.first()?, &name_to_idx)?;
                let in_shape = resolve_shape(input_idx, input_shape, &op_shapes);
                let (batch_size, channels, in_h, in_w) = match in_shape {
                    [c, h, w] => (None, *c, *h, *w),
                    [b, c, h, w] => (Some(*b), *c, *h, *w),
                    _ => return None,
                };
                let (out_h, out_w) = pool.output_size(in_h, in_w).ok()?;
                let (kernel_h, kernel_w) = if pool.is_global() {
                    (in_h, in_w)
                } else {
                    pool.kernel_size
                };
                let out_shape = match batch_size {
                    Some(b) => vec![b, channels, out_h, out_w],
                    None => vec![channels, out_h, out_w],
                };
                let num_elements = checked_shape_product(&out_shape)?;
                let op = GpuDagIbpOp::AveragePool {
                    channels,
                    input_h: in_h,
                    input_w: in_w,
                    output_h: out_h,
                    output_w: out_w,
                    kernel_h,
                    kernel_w,
                    stride_h: if pool.is_global() { 1 } else { pool.stride.0 },
                    stride_w: if pool.is_global() { 1 } else { pool.stride.1 },
                    pad_h: if pool.is_global() { 0 } else { pool.padding.0 },
                    pad_w: if pool.is_global() { 0 } else { pool.padding.1 },
                    count_include_pad: pool.count_include_pad,
                    is_global: pool.is_global(),
                    num_elements,
                    input_idx,
                };
                (op, out_shape)
            }
            // Unsupported layer: fail closed.
            _ => return None,
        };

        name_to_idx.insert(node_name.as_str(), op_idx);
        ops.push(op);
        op_shapes.push(output_shape);
    }

    // Look up the designated output node's index in the plan.
    let output_op_idx = *name_to_idx.get(graph.output_name())?;
    Some(GpuDagIbpPlanDesc {
        ops,
        input_shape: input_shape.to_vec(),
        output_op_idx,
    })
}

/// Resolve a node input name to an op index in the plan.
///
/// Returns `NETWORK_INPUT_IDX` for the network input sentinel, or the op
/// index from the name-to-index map.
fn resolve_input_idx(input_name: &str, name_to_idx: &HashMap<&str, usize>) -> Option<usize> {
    if input_name == NETWORK_INPUT {
        Some(NETWORK_INPUT_IDX)
    } else {
        name_to_idx.get(input_name).copied()
    }
}

/// Resolve the output shape of an op (or the network input shape).
fn resolve_shape<'a>(
    idx: usize,
    input_shape: &'a [usize],
    op_shapes: &'a [Vec<usize>],
) -> &'a [usize] {
    if idx == NETWORK_INPUT_IDX {
        input_shape
    } else {
        &op_shapes[idx]
    }
}

#[cfg(test)]
mod tests;
