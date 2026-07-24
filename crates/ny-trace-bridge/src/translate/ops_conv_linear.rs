// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Extended conv / linear / embedding family: `Conv3d`, `ConvTranspose1d`,
//! `ConvTranspose2d`, `QLinear`, `Embedding`, `Narrow`, `Expand`.
//!
//! Ported from NN's LayerSpec translators:
//!
//! - `ConvTranspose1d` / `ConvTranspose2d` mirror
//!   `trace_to_graph_layerspec_conv_transpose.rs`, including the
//!   `output_padding != 0` decomposition for the 1-D case
//!   (`ConvTranspose1d(output_padding=0)` + identity-matrix Linear zero-pad)
//!   and the fail-closed refusals (`dilation != 1` for both; 2-D
//!   `output_padding != 0`).
//! - `QLinear` routes to the shared Linear translator, exactly as NN's
//!   `translate_conv_dispatch` does (the dequantized f32 payload is a plain
//!   Linear for bound purposes).
//! - `Embedding` mirrors `trace_to_graph_layerspec_embed.rs`: a Linear with
//!   zeros weight + per-dimension midpoint bias (sound but conservative table
//!   lookup relaxation), plus a Reshape when the op adds an embedding dim.
//! - `Narrow` mirrors `translate_narrow` (Slice with axis/start/end).
//! - `Expand` mirrors `translate_expand` (chained Tile per broadcast dim;
//!   identity Reshape when no dim expands).
//! - `Conv3d` remains the fail-closed `UnsupportedOp` refusal: NN's LayerSpec
//!   path has no Conv3d emission (its conv dispatch rejects it), so there is
//!   no ground-truth lowering to port.

use std::collections::HashMap;

use ndarray::{Array2, IxDyn};
use ny_build::{AttributeValue, DataType, LayerSpec, WeightRef};
use ny_core::{LayerType, NyError, Result};

use crate::schema::{TraceNode, TraceOp, WeightPayload};

use super::{
    dim_as_i64, first_input, insert_payload, op_name, ops_core, shape_to_i64, simple_spec,
    weight_f32, Ctx, NodeOutput,
};

/// Translate a conv/linear-family op (`Conv3d`, `ConvTranspose1d`,
/// `ConvTranspose2d`, `QLinear`, `Embedding`, `Narrow`, `Expand`) node.
///
/// `Conv3d` (no NN LayerSpec emission exists) refuses with the exact
/// [`NyError::UnsupportedOp`] error the pre-split catch-all arm produced.
pub(super) fn translate_conv_linear(
    node: &TraceNode,
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    _node_names: &HashMap<u64, String>,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    match &node.op {
        // QLinear routes to the shared Linear translator (NN:
        // `TraceOp::QLinear { weight, bias } => translate_linear(...)`).
        TraceOp::QLinear { weight, bias } => {
            ops_core::translate_linear(name, weight, bias, input_tensors, output_tensor, ctx)
        }

        TraceOp::ConvTranspose1d {
            weight,
            bias,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        } => translate_conv_transpose1d(
            name,
            weight,
            bias,
            *padding,
            *output_padding,
            *stride,
            *dilation,
            *groups,
            input_tensors,
            output_tensor,
            ctx,
        ),

        TraceOp::ConvTranspose2d {
            weight,
            bias,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        } => translate_conv_transpose2d(
            name,
            weight,
            bias,
            *padding,
            *output_padding,
            *stride,
            *dilation,
            *groups,
            input_tensors,
            output_tensor,
            ctx,
        ),

        TraceOp::Embedding { weight } => translate_embedding(
            name,
            weight,
            input_tensors,
            output_tensor,
            &node.output_shape,
            ctx,
        ),

        // Narrow preserves rank: output rank == input rank, so the
        // trailing-relative axis is encoded against the output shape.
        // Mirrors NN's post-rework dispatch (nn d7144ea7).
        TraceOp::Narrow { dim, start, length } => translate_narrow(
            name,
            super::trailing_axis(*dim, node.output_shape.len(), "Narrow axis")?,
            *start,
            *length,
            input_tensors.to_vec(),
            output_tensor,
        ),

        TraceOp::Expand { target_shape } => translate_expand(
            name,
            target_shape,
            input_tensors.to_vec(),
            output_tensor,
            ctx,
        ),

        // Conv3d and anything mis-routed here: fail-closed refusal (same
        // type, same message shape as the pre-split catch-all arm).
        _ => Err(NyError::UnsupportedOp(format!(
            "{} not supported in NY trace translation",
            op_name(&node.op)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers (minimal copies of NN helpers missing from mod.rs)
// ---------------------------------------------------------------------------

/// Convert `i64` to `usize`, rejecting negatives/overflow.
///
/// Minimal copy of NN's `checked_i64_to_usize` — not available in `mod.rs`
/// helpers (dedupe later).
fn checked_i64_to_usize(val: i64, context: &str) -> Result<usize> {
    usize::try_from(val).map_err(|_| {
        NyError::InternalError(format!("{context}: dimension {val} out of usize range"))
    })
}

// ---------------------------------------------------------------------------
// ConvTranspose1d (mirrors NN trace_to_graph_layerspec_conv_transpose.rs)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn translate_conv_transpose1d(
    name: &str,
    weight: &WeightPayload,
    bias: &Option<WeightPayload>,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if dilation != 1 {
        return Err(NyError::UnsupportedOp(
            "ConvTranspose1d dilation != 1 not supported".to_string(),
        ));
    }
    let w_name = format!("{name}_weight");
    insert_payload(ctx, weight, &w_name, "ConvTranspose1d weight")?;
    if let Some(b) = bias {
        insert_payload(ctx, b, &format!("{name}_bias"), "ConvTranspose1d bias")?;
    }

    let stride_i64 = dim_as_i64(stride, "ConvTranspose1d stride")?;
    let pad_i64 = dim_as_i64(padding, "ConvTranspose1d padding")?;
    let groups_i64 = dim_as_i64(groups, "ConvTranspose1d groups")?;

    let mut attrs = HashMap::new();
    attrs.insert(
        "strides".to_string(),
        AttributeValue::Ints(vec![stride_i64]),
    );
    attrs.insert(
        "pads".to_string(),
        AttributeValue::Ints(vec![pad_i64, pad_i64]),
    );
    attrs.insert("group".to_string(), AttributeValue::Int(groups_i64));

    // When output_padding != 0, decompose into ConvTranspose1d(output_padding=0)
    // followed by a right-side zero-pad via a Linear layer. NY's
    // ConvTranspose1dLayer has no output_padding field (NN #2558).
    if output_padding != 0 {
        return translate_conv_transpose1d_with_output_padding(
            name,
            weight,
            bias,
            output_padding,
            &w_name,
            attrs,
            input_tensors,
            output_tensor,
            ctx,
        );
    }

    let mut spec_inputs = input_tensors.to_vec();
    spec_inputs.push(w_name.clone());
    if bias.is_some() {
        spec_inputs.push(format!("{name}_bias"));
    }

    Ok(NodeOutput::one(LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::ConvTranspose1d,
        inputs: spec_inputs,
        outputs: vec![output_tensor.to_string()],
        weights: Some(WeightRef {
            name: w_name,
            shape: weight.shape.clone(),
            original_dtype: DataType::Float32,
        }),
        attributes: attrs,
    }))
}

/// Decompose `ConvTranspose1d(output_padding=P)` into two LayerSpecs:
///
/// 1. `ConvTranspose1d(output_padding=0)` → intermediate `[B, C, T_mid]`
/// 2. `Linear` zero-pad → `[B, C, T_out]` where `T_out = T_mid + output_padding`
///
/// The Linear uses an identity matrix at rows `[0..T_mid]` and zero rows at
/// `[T_mid..T_out]`, exactly mirroring NN's decomposition (NN #2558).
#[allow(clippy::too_many_arguments)]
fn translate_conv_transpose1d_with_output_padding(
    name: &str,
    weight: &WeightPayload,
    bias: &Option<WeightPayload>,
    output_padding: usize,
    w_name: &str,
    attrs: HashMap<String, AttributeValue>,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    // Look up input spatial length from the first input tensor's recorded shape.
    let input_name = input_tensors.first().ok_or_else(|| {
        NyError::InternalError("ConvTranspose1d output_padding: no input tensors".to_string())
    })?;
    let input_shape = ctx.tensor_shapes.get(input_name).cloned().ok_or_else(|| {
        NyError::InternalError(format!(
            "ConvTranspose1d output_padding: input shape for '{input_name}' not recorded"
        ))
    })?;
    // Input layout: [B, C, T] — spatial dim is the last element.
    if input_shape.len() < 3 {
        return Err(NyError::UnsupportedOp(format!(
            "ConvTranspose1d output_padding: expected >= 3D input, got {}D",
            input_shape.len()
        )));
    }
    let in_len_i64 = *input_shape.last().ok_or_else(|| {
        NyError::InternalError(
            "ConvTranspose1d output_padding: empty input shape (internal error)".to_string(),
        )
    })?;
    let in_len = checked_i64_to_usize(in_len_i64, "ConvTranspose1d input length")?;

    // Extract kernel_size from weight shape [in_ch, out_ch/groups, kernel_size].
    let kernel_size = weight.shape.get(2).copied().ok_or_else(|| {
        NyError::ModelLoad("ConvTranspose1d output_padding: weight rank < 3".to_string())
    })?;

    // Recover stride and padding from attrs (already computed by caller).
    let stride = match attrs.get("strides") {
        Some(AttributeValue::Ints(v)) if !v.is_empty() => {
            checked_i64_to_usize(v[0], "ConvTranspose1d stride attr")?
        }
        _ => 1,
    };
    let padding = match attrs.get("pads") {
        Some(AttributeValue::Ints(v)) if !v.is_empty() => {
            checked_i64_to_usize(v[0], "ConvTranspose1d padding attr")?
        }
        _ => 0,
    };

    // ConvTranspose1d output without output_padding (dilation=1):
    // T_mid = (in_len - 1) * stride - 2 * padding + kernel_size
    let t_mid = in_len
        .checked_sub(1)
        .and_then(|v| v.checked_mul(stride))
        .and_then(|v| v.checked_add(kernel_size))
        .and_then(|v| v.checked_sub(2usize.checked_mul(padding)?))
        .ok_or_else(|| {
            NyError::InternalError(format!(
                "ConvTranspose1d output_padding: T_mid overflow (in_len={in_len}, \
                 stride={stride}, padding={padding}, kernel_size={kernel_size})"
            ))
        })?;
    let t_out = t_mid.checked_add(output_padding).ok_or_else(|| {
        NyError::InternalError(format!(
            "ConvTranspose1d output_padding: T_out overflow ({t_mid} + {output_padding})"
        ))
    })?;

    let mut specs = Vec::with_capacity(2);

    // Step 1: ConvTranspose1d(output_padding=0) → "{name}_conv_out".
    let conv_out = format!("{name}_conv_out");
    let mut spec_inputs = input_tensors.to_vec();
    spec_inputs.push(w_name.to_string());
    if bias.is_some() {
        spec_inputs.push(format!("{name}_bias"));
    }

    specs.push(LayerSpec {
        name: conv_out.clone(),
        layer_type: LayerType::ConvTranspose1d,
        inputs: spec_inputs,
        outputs: vec![conv_out.clone()],
        weights: Some(WeightRef {
            name: w_name.to_string(),
            shape: weight.shape.clone(),
            original_dtype: DataType::Float32,
        }),
        attributes: attrs,
    });

    // Record intermediate shape so downstream nodes can look it up.
    let mut mid_shape = input_shape;
    let last_idx = mid_shape.len() - 1;
    mid_shape[last_idx] = dim_as_i64(t_mid, "ConvTranspose1d T_mid")?;
    ctx.tensor_shapes
        .entry(conv_out.clone())
        .or_insert(mid_shape);

    // Step 2: Linear zero-pad [T_out, T_mid] — right-pads T_mid to T_out.
    // Identity matrix at rows [0..T_mid], zero rows at [T_mid..T_out].
    let pad_w_name = format!("{name}_pad_weight");
    let mut pad_weight = Array2::<f32>::zeros((t_out, t_mid));
    for t in 0..t_mid {
        pad_weight[[t, t]] = 1.0;
    }
    let pad_shape = vec![t_out, t_mid];
    ctx.insert_weight(
        &pad_w_name,
        pad_weight
            .into_shape_with_order(IxDyn(&pad_shape))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "ConvTranspose1d output_padding: pad weight reshape: {e}"
                ))
            })?,
    )?;

    let mut linear_attrs = HashMap::new();
    linear_attrs.insert("transB".to_string(), AttributeValue::Int(1));
    specs.push(LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::Linear,
        inputs: vec![conv_out, pad_w_name.clone()],
        outputs: vec![output_tensor.to_string()],
        weights: Some(WeightRef {
            name: pad_w_name,
            shape: pad_shape,
            original_dtype: DataType::Float32,
        }),
        attributes: linear_attrs,
    });

    Ok(NodeOutput { specs })
}

// ---------------------------------------------------------------------------
// ConvTranspose2d (mirrors NN trace_to_graph_layerspec_conv_transpose.rs)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn translate_conv_transpose2d(
    name: &str,
    weight: &WeightPayload,
    bias: &Option<WeightPayload>,
    padding: [usize; 2],
    output_padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if dilation[0] != 1 || dilation[1] != 1 {
        return Err(NyError::UnsupportedOp(
            "ConvTranspose2d dilation != 1 not supported".to_string(),
        ));
    }
    if output_padding[0] != 0 || output_padding[1] != 0 {
        return Err(NyError::UnsupportedOp(
            "ConvTranspose2d output_padding != 0 not supported".to_string(),
        ));
    }

    let w_name = format!("{name}_weight");
    insert_payload(ctx, weight, &w_name, "ConvTranspose2d weight")?;
    if let Some(b) = bias {
        insert_payload(ctx, b, &format!("{name}_bias"), "ConvTranspose2d bias")?;
    }

    let stride_h = dim_as_i64(stride[0], "ConvTranspose2d stride_h")?;
    let stride_w = dim_as_i64(stride[1], "ConvTranspose2d stride_w")?;
    let pad_h = dim_as_i64(padding[0], "ConvTranspose2d padding_h")?;
    let pad_w = dim_as_i64(padding[1], "ConvTranspose2d padding_w")?;
    let groups_i64 = dim_as_i64(groups, "ConvTranspose2d groups")?;

    let mut attrs = HashMap::new();
    attrs.insert(
        "strides".to_string(),
        AttributeValue::Ints(vec![stride_h, stride_w]),
    );
    attrs.insert(
        "pads".to_string(),
        AttributeValue::Ints(vec![pad_h, pad_w, pad_h, pad_w]),
    );
    attrs.insert("group".to_string(), AttributeValue::Int(groups_i64));

    let mut spec_inputs = input_tensors.to_vec();
    spec_inputs.push(w_name.clone());
    if bias.is_some() {
        spec_inputs.push(format!("{name}_bias"));
    }

    Ok(NodeOutput::one(LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::ConvTranspose2d,
        inputs: spec_inputs,
        outputs: vec![output_tensor.to_string()],
        weights: Some(WeightRef {
            name: w_name,
            shape: weight.shape.clone(),
            original_dtype: DataType::Float32,
        }),
        attributes: attrs,
    }))
}

// ---------------------------------------------------------------------------
// Embedding (mirrors NN trace_to_graph_layerspec_embed.rs)
// ---------------------------------------------------------------------------
//
// NY's graph build has no Embedding converter. The embedding is a table lookup
// where any row could be selected, so per-dimension bounds are:
//   lower[d] = min(weight[0..V, d]), upper[d] = max(weight[0..V, d])
//
// Modeled as a Linear layer with zeros weight and midpoint bias. The Embedding
// maps [B, T] → [B, T, D], adding the embedding dimension. NY's Linear
// preserves rank (maps last dim only), so:
//   1. Linear: [B, T] → [B, T*D]  (W = zeros([T*D, T]), b = midpoint tiled T×)
//   2. Reshape: [B, T*D] → [B, T, D]
//
// Since W=zeros, the Linear always outputs the tiled midpoint regardless of
// input. IBP then spreads the ±half-width through downstream layers. This is
// sound (no false negatives) but conservative.
fn translate_embedding(
    name: &str,
    weight: &WeightPayload,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if weight.shape.len() != 2 {
        return Err(NyError::ModelLoad(format!(
            "Embedding: weight must be 2D [V, D], got {}D",
            weight.shape.len()
        )));
    }
    // weight_f32 rejects shape-only placeholders, empty data, and non-finite
    // elements — the same validations NN's translate_embedding performs.
    let data = weight_f32(weight, "Embedding weight")?;
    let num_embeddings = weight.shape[0];
    let embedding_dim = weight.shape[1];
    let expected_len = num_embeddings.checked_mul(embedding_dim).ok_or_else(|| {
        NyError::InternalError(format!(
            "Embedding: V * D overflow ({num_embeddings} * {embedding_dim})"
        ))
    })?;
    if data.len() != expected_len {
        return Err(NyError::ModelLoad(format!(
            "Embedding: weight data length {} does not match shape [{num_embeddings}, {embedding_dim}]",
            data.len()
        )));
    }

    // Compute per-dimension midpoint across all vocabulary rows.
    let mut midpoints = Vec::with_capacity(embedding_dim);
    for d in 0..embedding_dim {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for row in 0..num_embeddings {
            let val = data[row * embedding_dim + d];
            if val < lo {
                lo = val;
            }
            if val > hi {
                hi = val;
            }
        }
        midpoints.push(f32::midpoint(lo, hi));
    }

    let data_input = first_input(input_tensors, "Embedding")?;

    // Determine in_features from the last dimension of the input tensor.
    // NY Linear maps the last dim, so in_features = input_shape[-1].
    let in_features = ctx
        .tensor_shapes
        .get(&data_input)
        .and_then(|s| s.last())
        .map(|&d| d.unsigned_abs() as usize)
        .unwrap_or(1)
        .max(1);

    // Check if the Embedding adds dimensions (output rank > input rank).
    // This is the normal case: Embedding([B, T]) → [B, T, D]. NN reads the
    // pre-recorded output tensor shape from ctx.tensor_shapes; the bridge
    // records that shape after dispatch, so use the traced node's output
    // shape directly (identical value).
    let input_rank = ctx
        .tensor_shapes
        .get(&data_input)
        .map(Vec::len)
        .unwrap_or(1);
    let output_rank = output_shape.len();

    if output_rank > input_rank {
        // Embedding adds dimensions: emit Linear → flat → Reshape to output
        // shape. Linear maps in_features (T) → out_features (T * D).
        let out_features = in_features.checked_mul(embedding_dim).ok_or_else(|| {
            NyError::InternalError(format!(
                "Embedding: in_features * embedding_dim overflow \
                 ({in_features} * {embedding_dim})"
            ))
        })?;

        // Tile midpoint for each input position:
        // [mid[0]..mid[D-1], mid[0]..mid[D-1], ...]
        let tiled_midpoints: Vec<f32> = midpoints
            .iter()
            .copied()
            .cycle()
            .take(out_features)
            .collect();

        let w_name = format!("{name}_weight");
        let zeros_weight = ndarray::ArrayD::from_elem(IxDyn(&[out_features, in_features]), 0.0_f32);
        ctx.insert_weight(&w_name, zeros_weight)?;

        let b_name = format!("{name}_bias");
        let midpoint_arr = ndarray::ArrayD::from_shape_vec(IxDyn(&[out_features]), tiled_midpoints)
            .map_err(|e| NyError::InternalError(format!("Embedding midpoint array: {e}")))?;
        ctx.insert_weight(&b_name, midpoint_arr)?;

        // Linear: input → flat intermediate tensor.
        let linear_name = format!("{name}_emb_linear");
        let linear_out = format!("{linear_name}_out");
        let mut linear_attrs = HashMap::new();
        linear_attrs.insert("transB".to_string(), AttributeValue::Int(1));
        let linear_spec = LayerSpec {
            name: linear_name,
            layer_type: LayerType::Linear,
            inputs: vec![data_input, w_name.clone(), b_name],
            outputs: vec![linear_out.clone()],
            weights: Some(WeightRef {
                name: w_name,
                shape: vec![out_features, in_features],
                original_dtype: DataType::Float32,
            }),
            attributes: linear_attrs,
        };

        // Record intermediate shape: input shape with last dim = out_features.
        if let Some(input_shape) = ctx.tensor_shapes.get(&input_tensors[0]).cloned() {
            let mut lin_shape = input_shape;
            if let Some(last) = lin_shape.last_mut() {
                *last = dim_as_i64(out_features, "Embedding out_features")?;
            }
            ctx.tensor_shapes.insert(linear_out.clone(), lin_shape);
        }

        // Reshape: flat → traced output shape [B, T, D].
        let target_shape = shape_to_i64(output_shape, "Embedding output shape")?;
        let mut reshape_attrs = HashMap::new();
        reshape_attrs.insert("shape".to_string(), AttributeValue::Ints(target_shape));
        let reshape_spec = LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::Reshape,
            inputs: vec![linear_out],
            outputs: vec![output_tensor.to_string()],
            weights: None,
            attributes: reshape_attrs,
        };

        Ok(NodeOutput {
            specs: vec![linear_spec, reshape_spec],
        })
    } else {
        // Output rank == input rank: single Linear suffices (degenerate case).
        let w_name = format!("{name}_weight");
        let zeros_weight =
            ndarray::ArrayD::from_elem(IxDyn(&[embedding_dim, in_features]), 0.0_f32);
        ctx.insert_weight(&w_name, zeros_weight)?;

        let b_name = format!("{name}_bias");
        let midpoint_arr = ndarray::ArrayD::from_shape_vec(IxDyn(&[embedding_dim]), midpoints)
            .map_err(|e| NyError::InternalError(format!("Embedding midpoint array: {e}")))?;
        ctx.insert_weight(&b_name, midpoint_arr)?;

        let mut attrs = HashMap::new();
        attrs.insert("transB".to_string(), AttributeValue::Int(1));

        Ok(NodeOutput::one(LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::Linear,
            inputs: vec![data_input, w_name.clone(), b_name],
            outputs: vec![output_tensor.to_string()],
            weights: Some(WeightRef {
                name: w_name,
                shape: vec![embedding_dim, in_features],
                original_dtype: DataType::Float32,
            }),
            attributes: attrs,
        }))
    }
}

// ---------------------------------------------------------------------------
// Narrow (mirrors NN trace_to_graph_layerspec_shape.rs translate_narrow)
// ---------------------------------------------------------------------------

fn translate_narrow(
    name: &str,
    axis: i64,
    start: usize,
    length: usize,
    input_tensors: Vec<String>,
    output_tensor: &str,
) -> Result<NodeOutput> {
    // `axis` is trailing-relative negative (pre-encoded by the caller via
    // `super::trailing_axis`): ny-build passes negative axes through and the
    // runtime Slice resolves them against the actual rank. Mirrors NN's
    // post-rework `translate_narrow` (nn d7144ea7).
    let axis_i64 = axis;
    let start_i64 = dim_as_i64(start, "Narrow start")?;
    let end = start.checked_add(length).ok_or_else(|| {
        NyError::InternalError(format!(
            "Narrow: start + length overflow ({start} + {length})"
        ))
    })?;
    let end_i64 = dim_as_i64(end, "Narrow end")?;

    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis_i64));
    attrs.insert("start".to_string(), AttributeValue::Int(start_i64));
    attrs.insert("end".to_string(), AttributeValue::Int(end_i64));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Slice,
        input_tensors,
        output_tensor,
        attrs,
    )))
}

// ---------------------------------------------------------------------------
// Expand (mirrors NN trace_to_graph_layerspec_decompose.rs translate_expand)
// ---------------------------------------------------------------------------

/// Expand → chained Tile ops per expanded dimension.
///
/// Expand broadcasts dims of size 1 to size N, duplicating elements. Reshape
/// is element-preserving and cannot model this; Tile correctly duplicates
/// elements along a single axis.
fn translate_expand(
    name: &str,
    target_shape: &[usize],
    input_tensors: Vec<String>,
    output_tensor: &str,
    ctx: &Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(&input_tensors, "Expand")?;

    // Look up input shape to find dims needing expansion (size 1 → size N).
    let input_shape_i64 = ctx.tensor_shapes.get(&data_input).ok_or_else(|| {
        NyError::InternalError(format!("Expand: input shape not found for {data_input}"))
    })?;

    // Collect dims that need tiling.
    let mut tiles: Vec<(usize, usize)> = Vec::new();
    for (i, (&in_dim, &tgt_dim)) in input_shape_i64.iter().zip(target_shape.iter()).enumerate() {
        let in_dim = checked_i64_to_usize(in_dim, &format!("Expand dim {i}"))?;
        if in_dim == 1 && tgt_dim > 1 {
            tiles.push((i, tgt_dim));
        } else if in_dim != tgt_dim {
            return Err(NyError::UnsupportedOp(format!(
                "Expand: dim {i} size {in_dim} cannot expand to {tgt_dim}"
            )));
        }
    }

    if tiles.is_empty() {
        // No expansion needed — emit identity Reshape.
        let mut attrs = HashMap::new();
        attrs.insert(
            "shape".to_string(),
            AttributeValue::Ints(shape_to_i64(target_shape, name)?),
        );
        return Ok(NodeOutput::one(simple_spec(
            name,
            LayerType::Reshape,
            vec![data_input],
            output_tensor,
            attrs,
        )));
    }

    let mut specs = Vec::new();
    let mut prev_out = data_input;
    let total = tiles.len();

    for (idx, (dim, reps)) in tiles.iter().enumerate() {
        let is_last = idx == total - 1;
        let tile_name = if is_last {
            name.to_string()
        } else {
            format!("{name}_tile_d{dim}")
        };
        let tile_out = if is_last {
            output_tensor.to_string()
        } else {
            format!("{tile_name}_out")
        };
        let axis = dim_as_i64(*dim, &format!("Expand tile dim {dim}"))?;
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int(axis));
        attrs.insert(
            "reps".to_string(),
            AttributeValue::Int(dim_as_i64(*reps, "Expand tile reps")?),
        );
        specs.push(simple_spec(
            &tile_name,
            LayerType::Tile,
            vec![prev_out],
            &tile_out,
            attrs,
        ));
        prev_out = tile_out;
    }

    Ok(NodeOutput { specs })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use ny_build::{AttributeValue, GraphModel, GraphNetworkOptions};
    use ny_core::{LayerType, NyError};

    use crate::schema::{ComputationGraph, DType, NodeId, TraceNode, TraceOp, WeightPayload};
    use crate::translate::translate;

    // Local copies of mod.rs's private test helpers (test-only; dedupe later).
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

    fn count(model: &GraphModel, lt: &LayerType) -> usize {
        model
            .network
            .layers
            .iter()
            .filter(|l| &l.layer_type == lt)
            .count()
    }

    fn find<'a>(model: &'a GraphModel, lt: &LayerType) -> &'a ny_build::LayerSpec {
        model
            .network
            .layers
            .iter()
            .find(|l| &l.layer_type == lt)
            .unwrap_or_else(|| panic!("no {lt:?} layer in translated model"))
    }

    fn builds(model: &GraphModel, what: &str) {
        model
            .build_graph_network(GraphNetworkOptions::default())
            .unwrap_or_else(|e| panic!("{what}: GraphModel should build a graph network: {e}"));
    }

    /// ConvTranspose1d (output_padding=0) emits one ConvTranspose1d layer with
    /// NN's strides/pads/group attributes and the kernel WeightRef.
    #[test]
    fn conv_transpose1d_basic() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 4]),
            node(
                1,
                "ct",
                TraceOp::ConvTranspose1d {
                    weight: WeightPayload::f32(vec![0.5, 1.0, -0.5], vec![1, 1, 3]),
                    bias: Some(WeightPayload::f32(vec![0.1], vec![1])),
                    padding: 0,
                    output_padding: 0,
                    stride: 2,
                    dilation: 1,
                    groups: 1,
                },
                &[0],
                &[1, 1, 9],
            ),
        ]);
        let model = translate(&graph).expect("ConvTranspose1d translates");
        assert_eq!(count(&model, &LayerType::ConvTranspose1d), 1);
        let ct = find(&model, &LayerType::ConvTranspose1d);
        assert_eq!(
            ct.attributes.get("strides"),
            Some(&AttributeValue::Ints(vec![2]))
        );
        assert_eq!(
            ct.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![0, 0]))
        );
        assert_eq!(ct.attributes.get("group"), Some(&AttributeValue::Int(1)));
        // Data input + weight + bias.
        assert_eq!(ct.inputs.len(), 3);
        let wref = ct.weights.as_ref().expect("kernel WeightRef");
        assert_eq!(wref.shape, vec![1, 1, 3]);
        assert!(model.weights.contains_key("layer0_trace_1_weight"));
        assert!(model.weights.contains_key("layer0_trace_1_bias"));
        builds(&model, "ConvTranspose1d");
    }

    /// ConvTranspose1d with dilation != 1 refuses (fail-closed, NN parity).
    #[test]
    fn conv_transpose1d_dilation_rejected() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 4]),
            node(
                1,
                "ct",
                TraceOp::ConvTranspose1d {
                    weight: WeightPayload::f32(vec![0.5, 1.0, -0.5], vec![1, 1, 3]),
                    bias: None,
                    padding: 0,
                    output_padding: 0,
                    stride: 1,
                    dilation: 2,
                    groups: 1,
                },
                &[0],
                &[1, 1, 8],
            ),
        ]);
        let err = translate(&graph).expect_err("dilation != 1 must refuse");
        assert!(
            matches!(&err, NyError::UnsupportedOp(m) if m.contains("dilation")),
            "expected dilation refusal, got {err:?}"
        );
    }

    /// ConvTranspose1d with output_padding decomposes into
    /// ConvTranspose1d(output_padding=0) + identity-matrix Linear zero-pad.
    /// Mirrors NN #2558: input [1,1,4], k=3, stride=2 → T_mid=9, T_out=10.
    #[test]
    fn conv_transpose1d_output_padding_decomposes() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 4]),
            node(
                1,
                "ct",
                TraceOp::ConvTranspose1d {
                    weight: WeightPayload::f32(vec![0.5, 1.0, -0.5], vec![1, 1, 3]),
                    bias: Some(WeightPayload::f32(vec![0.1], vec![1])),
                    padding: 0,
                    output_padding: 1,
                    stride: 2,
                    dilation: 1,
                    groups: 1,
                },
                &[0],
                &[1, 1, 10],
            ),
        ]);
        let model = translate(&graph).expect("output_padding decomposition translates");
        assert_eq!(count(&model, &LayerType::ConvTranspose1d), 1);
        assert_eq!(count(&model, &LayerType::Linear), 1, "zero-pad Linear");

        let lin = find(&model, &LayerType::Linear);
        assert_eq!(lin.attributes.get("transB"), Some(&AttributeValue::Int(1)));
        let wref = lin.weights.as_ref().expect("pad WeightRef");
        assert_eq!(wref.shape, vec![10, 9], "pad weight is [T_out, T_mid]");

        // Pad weight: identity rows [0..9], zero row [9].
        let pad_w = model
            .weights
            .get("layer0_trace_1_pad_weight")
            .expect("pad weight stored");
        assert_eq!(pad_w.shape(), &[10, 9]);
        for t in 0..9 {
            assert_eq!(pad_w[[t, t]], 1.0, "identity diagonal at {t}");
        }
        assert!(
            pad_w
                .index_axis(ndarray::Axis(0), 9)
                .iter()
                .all(|&v| v == 0.0),
            "last row is the zero pad row"
        );

        // The intermediate conv output feeds the Linear.
        assert_eq!(lin.inputs[0], "layer0_trace_1_conv_out");
        builds(&model, "ConvTranspose1d output_padding");
    }

    /// ConvTranspose2d emits NN's strides/pads/group attributes.
    #[test]
    fn conv_transpose2d_basic() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 2, 2]),
            node(
                1,
                "ct",
                TraceOp::ConvTranspose2d {
                    weight: WeightPayload::f32(vec![1.0, 0.5, -0.5, 0.3], vec![1, 1, 2, 2]),
                    bias: None,
                    padding: [0, 0],
                    output_padding: [0, 0],
                    stride: [1, 1],
                    dilation: [1, 1],
                    groups: 1,
                },
                &[0],
                &[1, 1, 3, 3],
            ),
        ]);
        let model = translate(&graph).expect("ConvTranspose2d translates");
        assert_eq!(count(&model, &LayerType::ConvTranspose2d), 1);
        let ct = find(&model, &LayerType::ConvTranspose2d);
        assert_eq!(
            ct.attributes.get("strides"),
            Some(&AttributeValue::Ints(vec![1, 1]))
        );
        assert_eq!(
            ct.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![0, 0, 0, 0]))
        );
        assert_eq!(ct.attributes.get("group"), Some(&AttributeValue::Int(1)));
        // Data input + weight (no bias).
        assert_eq!(ct.inputs.len(), 2);
        builds(&model, "ConvTranspose2d");
    }

    /// ConvTranspose2d refusals: dilation != 1 and output_padding != 0.
    #[test]
    fn conv_transpose2d_refusals() {
        let base_weight = || WeightPayload::f32(vec![1.0, 0.5, -0.5, 0.3], vec![1, 1, 2, 2]);
        let cases = [
            (
                TraceOp::ConvTranspose2d {
                    weight: base_weight(),
                    bias: None,
                    padding: [0, 0],
                    output_padding: [0, 0],
                    stride: [1, 1],
                    dilation: [2, 2],
                    groups: 1,
                },
                "dilation",
            ),
            (
                TraceOp::ConvTranspose2d {
                    weight: base_weight(),
                    bias: None,
                    padding: [0, 0],
                    output_padding: [1, 0],
                    stride: [2, 2],
                    dilation: [1, 1],
                    groups: 1,
                },
                "output_padding",
            ),
        ];
        for (op, needle) in cases {
            let graph = ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[1, 1, 2, 2]),
                node(1, "ct", op, &[0], &[1, 1, 4, 4]),
            ]);
            let err = translate(&graph).expect_err("must refuse");
            assert!(
                matches!(&err, NyError::UnsupportedOp(m) if m.contains(needle)),
                "expected {needle} refusal, got {err:?}"
            );
        }
    }

    /// QLinear routes to the Linear translator (NN parity).
    #[test]
    fn qlinear_maps_to_linear() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "ql",
                TraceOp::QLinear {
                    weight: WeightPayload::f32(vec![0.1; 8], vec![2, 4]),
                    bias: Some(WeightPayload::f32(vec![0.0, 0.0], vec![2])),
                },
                &[0],
                &[2],
            ),
        ]);
        let model = translate(&graph).expect("QLinear translates");
        assert_eq!(count(&model, &LayerType::Linear), 1);
        let lin = find(&model, &LayerType::Linear);
        assert_eq!(lin.attributes.get("transB"), Some(&AttributeValue::Int(1)));
        // Data input + weight + bias.
        assert_eq!(lin.inputs.len(), 3);
        assert_eq!(
            lin.weights.as_ref().map(|w| w.shape.clone()),
            Some(vec![2, 4])
        );
        builds(&model, "QLinear");
    }

    /// Embedding [1,1] → [1,1,4] decomposes into Linear (zeros weight,
    /// midpoint bias) + Reshape, with NN's per-dimension midpoints.
    #[test]
    fn embedding_decomposes_to_midpoint_linear_reshape() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1]),
            node(
                1,
                "emb",
                TraceOp::Embedding {
                    weight: WeightPayload::f32(
                        vec![
                            1.0, 2.0, 3.0, 4.0, // row 0
                            5.0, 6.0, 7.0, 8.0, // row 1
                            3.0, 4.0, 5.0, 6.0, // row 2
                        ],
                        vec![3, 4],
                    ),
                },
                &[0],
                &[1, 1, 4],
            ),
        ]);
        let model = translate(&graph).expect("Embedding translates");
        assert_eq!(count(&model, &LayerType::Linear), 1);
        assert_eq!(count(&model, &LayerType::Reshape), 1);

        // Zeros weight [out_features=1*4, in_features=1].
        let w = model
            .weights
            .get("layer0_trace_1_weight")
            .expect("zeros weight stored");
        assert_eq!(w.shape(), &[4, 1]);
        assert!(w.iter().all(|&v| v == 0.0), "Linear weight is all zeros");

        // Midpoint bias: per-dim (lo,hi) = (1,5),(2,6),(3,7),(4,8) → 3,4,5,6.
        let b = model
            .weights
            .get("layer0_trace_1_bias")
            .expect("midpoint bias stored");
        assert_eq!(
            b.iter().copied().collect::<Vec<f32>>(),
            vec![3.0, 4.0, 5.0, 6.0]
        );

        // Reshape targets the traced output shape.
        let rs = find(&model, &LayerType::Reshape);
        assert_eq!(
            rs.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![1, 1, 4]))
        );
        builds(&model, "Embedding");
    }

    /// Embedding with a non-finite weight refuses (NN parity: the error names
    /// the non-finite value).
    #[test]
    fn embedding_non_finite_weight_rejected() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1]),
            node(
                1,
                "emb",
                TraceOp::Embedding {
                    weight: WeightPayload::f32(
                        vec![1.0, 2.0, 3.0, f32::NAN, 5.0, 6.0, 7.0, 8.0],
                        vec![2, 4],
                    ),
                },
                &[0],
                &[1, 1, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("non-finite weight must refuse");
        assert!(
            err.to_string().contains("non-finite"),
            "error names the non-finite value: {err}"
        );
    }

    /// Narrow emits a Slice with a trailing-relative axis and start/end
    /// attributes (byte-faithful to NN's post-rework `translate_narrow`,
    /// nn d7144ea7).
    #[test]
    fn narrow_maps_to_slice() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 3, 4]),
            node(
                1,
                "nr",
                TraceOp::Narrow {
                    dim: 1,
                    start: 1,
                    length: 2,
                },
                &[0],
                &[1, 2, 4],
            ),
        ]);
        let model = translate(&graph).expect("Narrow translates");
        assert_eq!(count(&model, &LayerType::Slice), 1);
        let sl = find(&model, &LayerType::Slice);
        // Trailing-relative: trace dim 1 of rank 3 → axis 1 - 3 = -2.
        assert_eq!(sl.attributes.get("axis"), Some(&AttributeValue::Int(-2)));
        assert_eq!(sl.attributes.get("start"), Some(&AttributeValue::Int(1)));
        assert_eq!(sl.attributes.get("end"), Some(&AttributeValue::Int(3)));
        builds(&model, "Narrow");
    }

    /// Expand [1,4] → [3,4] emits one Tile (axis 0, reps 3).
    #[test]
    fn expand_emits_tile_chain() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 4]),
            node(
                1,
                "ex",
                TraceOp::Expand {
                    target_shape: vec![3, 4],
                },
                &[0],
                &[3, 4],
            ),
        ]);
        let model = translate(&graph).expect("Expand translates");
        assert_eq!(count(&model, &LayerType::Tile), 1);
        let tile = find(&model, &LayerType::Tile);
        assert_eq!(tile.attributes.get("axis"), Some(&AttributeValue::Int(0)));
        assert_eq!(tile.attributes.get("reps"), Some(&AttributeValue::Int(3)));
        builds(&model, "Expand");
    }

    /// Expand with no dim to broadcast emits an identity Reshape; an
    /// incompatible target refuses.
    #[test]
    fn expand_identity_and_incompatible() {
        // Identity: [2, 4] → [2, 4].
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 4]),
            node(
                1,
                "ex",
                TraceOp::Expand {
                    target_shape: vec![2, 4],
                },
                &[0],
                &[2, 4],
            ),
        ]);
        let model = translate(&graph).expect("identity Expand translates");
        assert_eq!(count(&model, &LayerType::Tile), 0);
        assert_eq!(count(&model, &LayerType::Reshape), 1);
        let rs = find(&model, &LayerType::Reshape);
        assert_eq!(
            rs.attributes.get("shape"),
            Some(&AttributeValue::Ints(vec![2, 4]))
        );
        builds(&model, "Expand identity");

        // Incompatible: dim of size 2 cannot expand to 3.
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 4]),
            node(
                1,
                "ex",
                TraceOp::Expand {
                    target_shape: vec![3, 4],
                },
                &[0],
                &[3, 4],
            ),
        ]);
        let err = translate(&graph).expect_err("incompatible Expand must refuse");
        assert!(
            matches!(&err, NyError::UnsupportedOp(m) if m.contains("cannot expand")),
            "expected expand refusal, got {err:?}"
        );
    }

    /// Conv3d stays fail-closed: NN has no LayerSpec emission for it.
    #[test]
    fn conv3d_stays_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 2, 2, 2]),
            node(
                1,
                "c3",
                TraceOp::Conv3d {
                    weight: WeightPayload::f32(vec![0.1; 8], vec![1, 1, 2, 2, 2]),
                    bias: None,
                    padding: [0, 0, 0],
                    stride: [1, 1, 1],
                    dilation: [1, 1, 1],
                    groups: 1,
                },
                &[0],
                &[1, 1, 1, 1, 1],
            ),
        ]);
        let err = translate(&graph).expect_err("Conv3d is unsupported");
        assert!(
            matches!(&err, NyError::UnsupportedOp(m) if m.contains("Conv3d")),
            "error names the op: {err:?}"
        );
    }
}
