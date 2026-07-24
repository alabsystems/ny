// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pooling family: `AvgPool1d`, `AvgPool2d`, `MaxPool1d`, `MaxPool2d`,
//! `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveMaxPool2d`.
//!
//! Ported from NN's `trace_to_graph_layerspec_pool.rs`, which lowers exactly
//! four of the seven family ops:
//!
//! - `AvgPool2d` → `LayerType::AveragePool` with ONNX-style `kernel_shape` /
//!   `strides` / `pads` (symmetric `[ph, pw, ph, pw]`) and
//!   `count_include_pad = 1` (PyTorch `nn.AvgPool2d` default).
//! - `MaxPool2d` → `LayerType::MaxPool` with the same geometry attributes
//!   (no `count_include_pad`; max-pool padding never raises the max).
//! - `MaxPool1d` → `LayerType::MaxPool` with 1-element `kernel_shape` /
//!   `strides` and 2-element `pads = [p, p]`.
//! - `AdaptiveAvgPool2d` → `LayerType::AveragePool` with kernel = stride =
//!   `input_dim / output_dim`, refused unless the input spatial dims divide
//!   evenly by the output dims (the common global-pool `output_size = [1, 1]`
//!   case always does).
//!
//! `AvgPool1d`, `AdaptiveAvgPool1d`, and `AdaptiveMaxPool2d` have **no**
//! lowering in NN's translator (they fall through to its catch-all), so they
//! keep the sound-by-construction `UnsupportedOp` refusal here — inventing a
//! lowering NN never emits would break emission parity.

use std::collections::HashMap;

use ny_build::AttributeValue;
use ny_core::{LayerType, NyError, Result};

use crate::schema::{TraceNode, TraceOp};

use super::{dim_as_i64, op_name, simple_spec, Ctx, NodeOutput};

/// Translate a pooling-family op (`AvgPool1d`, `AvgPool2d`, `MaxPool1d`, `MaxPool2d`, `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveMaxPool2d`) node.
///
/// `AvgPool2d`, `MaxPool2d`, `MaxPool1d`, and `AdaptiveAvgPool2d` are ported
/// from NN's emission (see the module docs for the exact LayerSpec shapes).
/// The remaining ops refuse with the exact [`NyError::UnsupportedOp`] error
/// the pre-split catch-all arm produced, matching NN, which has no lowering
/// for them either.
pub(super) fn translate_pooling(
    node: &TraceNode,
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    _node_names: &HashMap<u64, String>,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    match &node.op {
        TraceOp::AvgPool2d {
            kernel_size,
            stride,
            padding,
        } => translate_avg_pool2d(
            name,
            kernel_size,
            stride,
            padding,
            input_tensors,
            output_tensor,
        ),
        TraceOp::MaxPool2d {
            kernel_size,
            stride,
            padding,
        } => translate_max_pool2d(
            name,
            kernel_size,
            stride,
            padding,
            input_tensors,
            output_tensor,
        ),
        TraceOp::MaxPool1d {
            kernel_size,
            stride,
            padding,
        } => translate_max_pool1d(
            name,
            *kernel_size,
            *stride,
            *padding,
            input_tensors,
            output_tensor,
        ),
        TraceOp::AdaptiveAvgPool2d { output_size } => {
            translate_adaptive_avg_pool2d(name, output_size, input_tensors, output_tensor, ctx)
        }

        // Not ported: NN's trace-to-graph translator has no lowering for these
        // (they fall through to its catch-all). Fail-closed refusal, same
        // message shape as the pre-split catch-all arm.
        TraceOp::AvgPool1d { .. }
        | TraceOp::AdaptiveAvgPool1d { .. }
        | TraceOp::AdaptiveMaxPool2d { .. } => Err(NyError::UnsupportedOp(format!(
            "{} not supported in NY trace translation",
            op_name(&node.op)
        ))),

        // Dispatch in mod.rs only routes the seven family ops here; anything
        // else is a routing bug. Fail closed rather than emit a wrong layer.
        other => Err(NyError::InternalError(format!(
            "translate_pooling dispatched on non-pooling op {}",
            op_name(other)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Average pooling
// ---------------------------------------------------------------------------

/// Translate `TraceOp::AvgPool2d` to an `AveragePool` LayerSpec.
///
/// Mirrors NN's `translate_avg_pool2d`: ONNX-style attributes with symmetric
/// pads `[ph, pw, ph, pw]` and `count_include_pad = 1` — PyTorch
/// `nn.AvgPool2d`'s default and the only semantics the graph-build pooling
/// layer's `Pool2dParams`-equivalent path expresses. NN validates the param
/// slices are length 2; the schema's `[usize; 2]` arrays guarantee that
/// statically.
fn translate_avg_pool2d(
    name: &str,
    kernel_size: &[usize; 2],
    stride: &[usize; 2],
    padding: &[usize; 2],
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    if input_tensors.is_empty() {
        return Err(NyError::UnsupportedOp(
            "AvgPool2d has no inputs (expected 1)".to_string(),
        ));
    }
    let kh = dim_as_i64(kernel_size[0], "AvgPool2d kernel_h")?;
    let kw = dim_as_i64(kernel_size[1], "AvgPool2d kernel_w")?;
    let sh = dim_as_i64(stride[0], "AvgPool2d stride_h")?;
    let sw = dim_as_i64(stride[1], "AvgPool2d stride_w")?;
    let ph = dim_as_i64(padding[0], "AvgPool2d pad_h")?;
    let pw = dim_as_i64(padding[1], "AvgPool2d pad_w")?;

    let mut attrs = HashMap::new();
    attrs.insert(
        "kernel_shape".to_string(),
        AttributeValue::Ints(vec![kh, kw]),
    );
    attrs.insert("strides".to_string(), AttributeValue::Ints(vec![sh, sw]));
    attrs.insert(
        "pads".to_string(),
        AttributeValue::Ints(vec![ph, pw, ph, pw]),
    );
    attrs.insert("count_include_pad".to_string(), AttributeValue::Int(1));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::AveragePool,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    )))
}

// ---------------------------------------------------------------------------
// Max pooling
// ---------------------------------------------------------------------------

/// Translate `TraceOp::MaxPool2d` to a `MaxPool` LayerSpec.
///
/// Mirrors NN's `translate_max_pool2d`: same geometry attributes as
/// `AvgPool2d` but no `count_include_pad` — max-pool padding contributes
/// `-inf` and never raises the max, so the attribute does not apply.
fn translate_max_pool2d(
    name: &str,
    kernel_size: &[usize; 2],
    stride: &[usize; 2],
    padding: &[usize; 2],
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    if input_tensors.is_empty() {
        return Err(NyError::UnsupportedOp(
            "MaxPool2d has no inputs (expected 1)".to_string(),
        ));
    }
    let kh = dim_as_i64(kernel_size[0], "MaxPool2d kernel_h")?;
    let kw = dim_as_i64(kernel_size[1], "MaxPool2d kernel_w")?;
    let sh = dim_as_i64(stride[0], "MaxPool2d stride_h")?;
    let sw = dim_as_i64(stride[1], "MaxPool2d stride_w")?;
    let ph = dim_as_i64(padding[0], "MaxPool2d pad_h")?;
    let pw = dim_as_i64(padding[1], "MaxPool2d pad_w")?;

    let mut attrs = HashMap::new();
    attrs.insert(
        "kernel_shape".to_string(),
        AttributeValue::Ints(vec![kh, kw]),
    );
    attrs.insert("strides".to_string(), AttributeValue::Ints(vec![sh, sw]));
    attrs.insert(
        "pads".to_string(),
        AttributeValue::Ints(vec![ph, pw, ph, pw]),
    );
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::MaxPool,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    )))
}

/// Translate `TraceOp::MaxPool1d` to a `MaxPool` LayerSpec.
///
/// Mirrors NN's `translate_max_pool1d`: 1-element `kernel_shape` / `strides`
/// and 2-element symmetric `pads = [p, p]` (the graph-build pooling converter
/// broadcasts 1-element geometry across both spatial dims, exactly as it does
/// for NN's emission).
fn translate_max_pool1d(
    name: &str,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    if input_tensors.is_empty() {
        return Err(NyError::UnsupportedOp(
            "MaxPool1d has no inputs (expected 1)".to_string(),
        ));
    }
    let k = dim_as_i64(kernel_size, "MaxPool1d kernel_size")?;
    let s = dim_as_i64(stride, "MaxPool1d stride")?;
    let p = dim_as_i64(padding, "MaxPool1d padding")?;

    let mut attrs = HashMap::new();
    attrs.insert("kernel_shape".to_string(), AttributeValue::Ints(vec![k]));
    attrs.insert("strides".to_string(), AttributeValue::Ints(vec![s]));
    attrs.insert("pads".to_string(), AttributeValue::Ints(vec![p, p]));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::MaxPool,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    )))
}

// ---------------------------------------------------------------------------
// Adaptive average pooling
// ---------------------------------------------------------------------------

/// Translate `TraceOp::AdaptiveAvgPool2d` by computing kernel/stride from
/// input/output shapes.
///
/// Mirrors NN's `translate_adaptive_avg_pool2d`: only supported when the input
/// spatial dims divide evenly by the output dims (producing a regular,
/// non-overlapping pooling pattern with kernel = stride = `in / out`). The
/// common case `output_size = [1, 1]` (global average pooling) always
/// satisfies this. Anything else is refused — approximating the uneven-window
/// PyTorch semantics with a regular kernel would change which inputs are
/// averaged.
fn translate_adaptive_avg_pool2d(
    name: &str,
    output_size: &[usize; 2],
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &Ctx,
) -> Result<NodeOutput> {
    if input_tensors.is_empty() {
        return Err(NyError::UnsupportedOp(
            "AdaptiveAvgPool2d has no inputs (expected 1)".to_string(),
        ));
    }
    let [out_h, out_w] = *output_size;
    if out_h == 0 || out_w == 0 {
        return Err(NyError::UnsupportedOp(
            "AdaptiveAvgPool2d output_size must be > 0".to_string(),
        ));
    }

    // Look up input shape from context (recorded by previous node translation).
    let input_shape = ctx.tensor_shapes.get(&input_tensors[0]).ok_or_else(|| {
        NyError::InternalError(format!(
            "AdaptiveAvgPool2d: input shape for '{}' not found in context",
            input_tensors[0]
        ))
    })?;

    // Input is [batch, channels, in_h, in_w].
    if input_shape.len() < 4 {
        return Err(NyError::UnsupportedOp(format!(
            "AdaptiveAvgPool2d: expected 4D input, got {}D",
            input_shape.len()
        )));
    }
    let in_h = checked_i64_to_usize(input_shape[2], "AdaptiveAvgPool2d input height")?;
    let in_w = checked_i64_to_usize(input_shape[3], "AdaptiveAvgPool2d input width")?;

    // Only translate when input divides evenly (regular pooling pattern).
    if in_h % out_h != 0 || in_w % out_w != 0 {
        return Err(NyError::UnsupportedOp(format!(
            "AdaptiveAvgPool2d: input [{in_h},{in_w}] not evenly divisible \
             by output [{out_h},{out_w}]"
        )));
    }

    let kernel_h = in_h / out_h;
    let kernel_w = in_w / out_w;

    let kh = dim_as_i64(kernel_h, "AdaptiveAvgPool2d kernel_h")?;
    let kw = dim_as_i64(kernel_w, "AdaptiveAvgPool2d kernel_w")?;
    let sh = kh; // stride == kernel for non-overlapping adaptive pool
    let sw = kw;

    let mut attrs = HashMap::new();
    attrs.insert(
        "kernel_shape".to_string(),
        AttributeValue::Ints(vec![kh, kw]),
    );
    attrs.insert("strides".to_string(), AttributeValue::Ints(vec![sh, sw]));
    attrs.insert("pads".to_string(), AttributeValue::Ints(vec![0, 0, 0, 0]));
    attrs.insert("count_include_pad".to_string(), AttributeValue::Int(1));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::AveragePool,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    )))
}

/// Checked `i64` → `usize` conversion rejecting negative values.
///
/// Local copy of NN's `graph_tensor::checked_i64_to_usize` — `mod.rs` has no
/// shared helper for this direction (only `dim_as_i64`); dedupe later if
/// another family needs it.
fn checked_i64_to_usize(val: i64, context: &str) -> Result<usize> {
    usize::try_from(val).map_err(|_| {
        NyError::InternalError(format!(
            "{context}: i64 value {val} is negative (cannot convert to usize)"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::super::translate;
    use crate::schema::{ComputationGraph, DType, NodeId, TraceNode, TraceOp};
    use ny_build::{AttributeValue, GraphModel, LayerSpec};
    use ny_core::{LayerType, NyError};

    fn node(id: u64, name: &str, op: TraceOp, inputs: &[u64], shape: &[usize]) -> TraceNode {
        TraceNode::new(
            NodeId(id),
            name,
            op,
            inputs.iter().map(|&i| NodeId(i)).collect(),
            shape.to_vec(),
            DType::F32,
        )
    }

    fn find_layer<'m>(model: &'m GraphModel, lt: &LayerType) -> &'m LayerSpec {
        model
            .network
            .layers
            .iter()
            .find(|l| &l.layer_type == lt)
            .unwrap_or_else(|| panic!("no {lt:?} layer in translated model"))
    }

    fn assert_builds(model: &GraphModel, what: &str) {
        model
            .build_graph_network(ny_build::GraphNetworkOptions::default())
            .unwrap_or_else(|e| panic!("{what} GraphModel builds a graph network: {e}"));
    }

    /// AvgPool2d emits AveragePool with NN's exact attributes
    /// (kernel_shape/strides/pads/count_include_pad=1).
    #[test]
    fn avg_pool2d_maps_with_attrs() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 4, 4]),
            node(
                1,
                "pool",
                TraceOp::AvgPool2d {
                    kernel_size: [2, 2],
                    stride: [2, 2],
                    padding: [0, 0],
                },
                &[0],
                &[1, 1, 2, 2],
            ),
        ]);
        let model = translate(&graph).expect("AvgPool2d translates");
        let pool = find_layer(&model, &LayerType::AveragePool);
        assert_eq!(
            pool.attributes.get("kernel_shape"),
            Some(&AttributeValue::Ints(vec![2, 2]))
        );
        assert_eq!(
            pool.attributes.get("strides"),
            Some(&AttributeValue::Ints(vec![2, 2]))
        );
        assert_eq!(
            pool.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![0, 0, 0, 0]))
        );
        assert_eq!(
            pool.attributes.get("count_include_pad"),
            Some(&AttributeValue::Int(1)),
            "PyTorch nn.AvgPool2d default count_include_pad=true"
        );
        assert_builds(&model, "AvgPool2d");
    }

    /// AvgPool2d with padding emits symmetric 4-element pads [ph, pw, ph, pw].
    #[test]
    fn avg_pool2d_padding_is_symmetric() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 4, 4]),
            node(
                1,
                "pool",
                TraceOp::AvgPool2d {
                    kernel_size: [3, 3],
                    stride: [1, 1],
                    padding: [1, 1],
                },
                &[0],
                &[1, 1, 4, 4],
            ),
        ]);
        let model = translate(&graph).expect("AvgPool2d+pad translates");
        let pool = find_layer(&model, &LayerType::AveragePool);
        assert_eq!(
            pool.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![1, 1, 1, 1]))
        );
        assert_builds(&model, "AvgPool2d+pad");
    }

    /// MaxPool2d emits MaxPool with NN's attributes and NO count_include_pad.
    #[test]
    fn max_pool2d_maps_with_attrs() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2, 6, 6]),
            node(
                1,
                "pool",
                TraceOp::MaxPool2d {
                    kernel_size: [3, 3],
                    stride: [2, 2],
                    padding: [1, 1],
                },
                &[0],
                &[1, 2, 3, 3],
            ),
        ]);
        let model = translate(&graph).expect("MaxPool2d translates");
        let pool = find_layer(&model, &LayerType::MaxPool);
        assert_eq!(
            pool.attributes.get("kernel_shape"),
            Some(&AttributeValue::Ints(vec![3, 3]))
        );
        assert_eq!(
            pool.attributes.get("strides"),
            Some(&AttributeValue::Ints(vec![2, 2]))
        );
        assert_eq!(
            pool.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![1, 1, 1, 1]))
        );
        assert!(
            !pool.attributes.contains_key("count_include_pad"),
            "MaxPool has no count_include_pad (padding never raises the max)"
        );
        assert_builds(&model, "MaxPool2d");
    }

    /// MaxPool1d emits MaxPool with 1-element kernel/strides and pads [p, p].
    #[test]
    fn max_pool1d_maps_with_attrs() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 8]),
            node(
                1,
                "pool",
                TraceOp::MaxPool1d {
                    kernel_size: 2,
                    stride: 2,
                    padding: 0,
                },
                &[0],
                &[1, 1, 4],
            ),
        ]);
        let model = translate(&graph).expect("MaxPool1d translates");
        let pool = find_layer(&model, &LayerType::MaxPool);
        assert_eq!(
            pool.attributes.get("kernel_shape"),
            Some(&AttributeValue::Ints(vec![2]))
        );
        assert_eq!(
            pool.attributes.get("strides"),
            Some(&AttributeValue::Ints(vec![2]))
        );
        assert_eq!(
            pool.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![0, 0]))
        );
        assert_builds(&model, "MaxPool1d");
    }

    /// AdaptiveAvgPool2d with evenly-dividing dims emits AveragePool with
    /// kernel = stride = in/out and zero pads.
    #[test]
    fn adaptive_avg_pool2d_derives_kernel_from_shapes() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 3, 4, 4]),
            node(
                1,
                "pool",
                TraceOp::AdaptiveAvgPool2d {
                    output_size: [2, 2],
                },
                &[0],
                &[1, 3, 2, 2],
            ),
        ]);
        let model = translate(&graph).expect("AdaptiveAvgPool2d translates");
        let pool = find_layer(&model, &LayerType::AveragePool);
        assert_eq!(
            pool.attributes.get("kernel_shape"),
            Some(&AttributeValue::Ints(vec![2, 2])),
            "kernel = in/out = 4/2"
        );
        assert_eq!(
            pool.attributes.get("strides"),
            Some(&AttributeValue::Ints(vec![2, 2])),
            "stride == kernel for non-overlapping adaptive pool"
        );
        assert_eq!(
            pool.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![0, 0, 0, 0]))
        );
        assert_eq!(
            pool.attributes.get("count_include_pad"),
            Some(&AttributeValue::Int(1))
        );
        assert_builds(&model, "AdaptiveAvgPool2d");
    }

    /// Global average pooling (output_size = [1, 1]) always divides evenly.
    #[test]
    fn adaptive_avg_pool2d_global_pool() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2, 6, 6]),
            node(
                1,
                "pool",
                TraceOp::AdaptiveAvgPool2d {
                    output_size: [1, 1],
                },
                &[0],
                &[1, 2, 1, 1],
            ),
        ]);
        let model = translate(&graph).expect("global AdaptiveAvgPool2d translates");
        let pool = find_layer(&model, &LayerType::AveragePool);
        assert_eq!(
            pool.attributes.get("kernel_shape"),
            Some(&AttributeValue::Ints(vec![6, 6]))
        );
        assert_eq!(
            pool.attributes.get("strides"),
            Some(&AttributeValue::Ints(vec![6, 6]))
        );
        assert_builds(&model, "global AdaptiveAvgPool2d");
    }

    /// AdaptiveAvgPool2d refuses non-evenly-dividing geometry (no sound
    /// regular-kernel equivalent of PyTorch's uneven windows).
    #[test]
    fn adaptive_avg_pool2d_uneven_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 4, 4]),
            node(
                1,
                "pool",
                TraceOp::AdaptiveAvgPool2d {
                    output_size: [3, 3],
                },
                &[0],
                &[1, 1, 3, 3],
            ),
        ]);
        let err = translate(&graph).expect_err("uneven adaptive pool refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("not evenly divisible")),
            "expected UnsupportedOp mentioning divisibility, got: {err:?}"
        );
    }

    /// AdaptiveAvgPool2d refuses non-4D input (mirrors NN's error path; NN's
    /// `test_adaptive_avg_pool2d_rejected` feeds a 2D input).
    #[test]
    fn adaptive_avg_pool2d_non_4d_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 4]),
            node(
                1,
                "pool",
                TraceOp::AdaptiveAvgPool2d {
                    output_size: [1, 1],
                },
                &[0],
                &[1, 3, 1, 1],
            ),
        ]);
        let err = translate(&graph).expect_err("2D input refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("AdaptiveAvgPool2d") && m.contains("expected 4D input")),
            "expected UnsupportedOp naming AdaptiveAvgPool2d + 4D, got: {err:?}"
        );
    }

    /// The three ops NN never lowers stay refused with the catch-all message.
    #[test]
    fn unported_pooling_ops_stay_refused() {
        let cases: Vec<(TraceOp, &str, Vec<usize>, Vec<usize>)> = vec![
            (
                TraceOp::AvgPool1d {
                    kernel_size: 2,
                    stride: 2,
                    padding: 0,
                },
                "AvgPool1d",
                vec![1, 1, 8],
                vec![1, 1, 4],
            ),
            (
                TraceOp::AdaptiveAvgPool1d { output_size: 1 },
                "AdaptiveAvgPool1d",
                vec![1, 1, 8],
                vec![1, 1, 1],
            ),
            (
                TraceOp::AdaptiveMaxPool2d {
                    output_size: [1, 1],
                },
                "AdaptiveMaxPool2d",
                vec![1, 1, 4, 4],
                vec![1, 1, 1, 1],
            ),
        ];
        for (op, opname, in_shape, out_shape) in cases {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &in_shape),
                node(1, "pool", op, &[0], &out_shape),
            ]);
            let err = translate(&graph).expect_err("unported pooling op refused");
            match err {
                NyError::UnsupportedOp(msg) => {
                    assert_eq!(
                        msg,
                        format!("{opname} not supported in NY trace translation"),
                        "exact catch-all refusal message for {opname}"
                    );
                }
                other => panic!("expected UnsupportedOp for {opname}, got {other:?}"),
            }
        }
    }
}
