// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Extended conv / linear / embedding family: `Conv3d`, `ConvTranspose1d`,
//! `ConvTranspose2d`, `QLinear`, `Embedding`, `Narrow`, `Expand`.
//!
//! Ported from NN's LayerSpec translators:
//!
//! - `ConvTranspose1d` / `ConvTranspose2d` mirror
//!   `trace_to_graph_layerspec_conv_transpose.rs`, extended past NN's
//!   refusals where ny's layers genuinely support the configuration:
//!   both emit the `dilations` attribute (ny-propagate's
//!   `ConvTranspose1dLayer::new_full` / `ConvTranspose2dLayer::new_full`
//!   model dilation natively, threaded through ny-build's
//!   `convert/conv.rs`), and the 2-D case emits `output_padding`
//!   (`ConvTranspose2dLayer` carries it as a field). The 1-D
//!   `output_padding != 0` case keeps the NN #2558 decomposition
//!   (`ConvTranspose1d(output_padding=0)` + identity-matrix Linear
//!   zero-pad, dilation-aware) because ny-build's 1-D converter refuses
//!   the attribute. Fail-closed refusals that REMAIN: 2-D
//!   `output_padding >= stride` (the transposed-conv output size is only
//!   recoverable within one stride step; `ConvTranspose2dLayer::new_full`
//!   enforces the same bound) and 2-D `groups != 1` (the layer models only
//!   the dense groups=1 kernel layout — silently dropping groups would
//!   reinterpret the grouped kernel as dense, which is unsound).
//! - `QLinear` routes to the shared Linear translator, exactly as NN's
//!   `translate_conv_dispatch` does (the dequantized f32 payload is a plain
//!   Linear for bound purposes).
//! - `Embedding` extends `trace_to_graph_layerspec_embed.rs`: a Linear with
//!   zeros weight + per-dimension midpoint bias, then an `EpsilonExpand`
//!   carrying the outward-rounded per-dimension half-spreads (the widening
//!   that makes the index-agnostic surrogate a true enclosure of every row),
//!   plus a Reshape when the op adds an embedding dim.
//! - `Narrow` mirrors `translate_narrow` (Slice with axis/start/end).
//! - `Expand` mirrors `translate_expand` (chained Tile per broadcast dim;
//!   identity Reshape when no dim expands).
//! - `Conv3d` → `LayerType::Conv3d`, mirroring the `Conv2d` attribute set with
//!   one extra spatial axis (`strides`/`dilations` gain a depth entry, `pads`
//!   becomes ONNX's rank-3 begin/end 6-tuple). See [`translate_conv3d`].
//!
//!   This arm used to be a fail-closed refusal, and the refusal was honest:
//!   **ny-propagate had no 3-D convolution layer at all**, so there was no
//!   emission target to wire a lowering TO and no amount of translator work
//!   would have changed it. That premise is gone — `ny_core::LayerType` has a
//!   `Conv3d` variant, `ny_propagate::layers::Conv3dLayer` exists, and
//!   ny-build's `convert_conv3d` builds it (its ONNX path now dispatches a
//!   rank-5 `Conv` kernel there instead of refusing). The layer is IBP-ONLY:
//!   its interval forward is a certified directed-binary64 enclosure and its
//!   CROWN backward fails closed, so a Conv3d node degrades the network to IBP
//!   at that node rather than making it unverifiable — strictly better than the
//!   permissive mode's vacuous `[-inf, +inf]` `OpaqueSkip` carrier, which was
//!   the only escape while the refusal stood.
//!
//!   Unlike `Conv2d`'s trace arm, dilation is EMITTED rather than refused:
//!   `Conv3dLayer::new_dilated` consumes it and the layer's IBP forward reads
//!   `self.dilation`, so dropping the attribute would silently build a
//!   `dilation = 1` layer whose bounds do not enclose the traced op.

use std::collections::HashMap;

use ndarray::{Array2, IxDyn};
use ny_build::{AttributeValue, DataType, LayerSpec, WeightRef};
use ny_core::{LayerType, NyError, Result};
use ny_tensor::sub_up_f32;

use crate::schema::{TraceNode, TraceOp, WeightPayload};

use super::{
    dim_as_i64, first_input, insert_payload, op_name, ops_core, shape_to_i64, simple_spec,
    weight_f32, Ctx, NodeOutput,
};

/// Translate a conv/linear-family op (`Conv3d`, `ConvTranspose1d`,
/// `ConvTranspose2d`, `QLinear`, `Embedding`, `Narrow`, `Expand`) node.
///
/// `Conv3d` lowers to `LayerType::Conv3d` (see [`translate_conv3d`]); the
/// blanket refusal that stood here rested on ny-propagate having no 3-D
/// convolution layer, which is no longer true. Any op mis-routed here keeps
/// the pre-split catch-all's message shape.
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

        // Conv3d: LOWERED. The refusal that stood here was a MISSING-LAYER
        // refusal ("ny-propagate has no 3-D convolution layer to emit"), and
        // that premise no longer holds: `ny_propagate::layers::Conv3dLayer`
        // exists, `ny_core::LayerType::Conv3d` names it, and ny-build's
        // `convert_conv3d` arm builds it (it also now dispatches a rank-5 ONNX
        // Conv kernel there instead of refusing).
        TraceOp::Conv3d {
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
        } => translate_conv3d(
            name,
            weight,
            bias,
            padding,
            stride,
            dilation,
            *groups,
            input_tensors,
            output_tensor,
            ctx,
        ),

        // Anything mis-routed here: fail-closed refusal (same type, same
        // message shape as the pre-split catch-all arm).
        _ => Err(NyError::UnsupportedOp(format!(
            "{} not supported in NY trace translation",
            op_name(&node.op)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Conv3d
// ---------------------------------------------------------------------------

/// Translate `TraceOp::Conv3d` into a `LayerType::Conv3d` spec.
///
/// The attribute set mirrors [`ops_core::translate_conv2d`] with one extra
/// spatial axis throughout: `strides` / `dilations` gain a leading depth entry
/// and `pads` becomes ONNX's rank-3 begin/end form
/// `[d, h, w, d, h, w]` — which ny-build's `parse_symmetric_padding_3d` reads
/// as symmetric and threads into `Conv3dLayer`'s one-value-per-axis padding.
/// The trace carries a single `padding[axis]` per axis (nn's `Conv3d` has no
/// asymmetric form), so the emitted 6-tuple is symmetric by construction and
/// can never hit that parser's asymmetric refusal.
///
/// # Dilation is emitted, not refused
///
/// [`ops_core::translate_conv2d`] refuses `dilation != 1` because ny's
/// `Conv2dLayer` path there cannot consume it. `Conv3dLayer` can: ny-build
/// calls `Conv3dLayer::new_dilated(.., dilation, groups)` and the layer's
/// certified IBP forward reads `self.dilation` directly. Dropping the attribute
/// instead would silently build a `dilation = 1` layer whose bounds do NOT
/// enclose the traced op, so it is threaded through.
///
/// # Soundness / tier
///
/// A 3-D convolution is an affine map, so nothing here approximates: the
/// emitted layer computes the traced function with the traced weights. The
/// layer is IBP-ONLY — its interval forward is a certified directed-binary64
/// enclosure and `Conv3dLayer`'s CROWN backward fails closed — so a Conv3d node
/// degrades the network to IBP at that node rather than making it unverifiable,
/// which is what the old refusal did (the permissive mode's vacuous
/// `[-inf, +inf]` OpaqueSkip carrier was the only escape).
#[allow(clippy::too_many_arguments)]
fn translate_conv3d(
    name: &str,
    weight: &WeightPayload,
    bias: &Option<WeightPayload>,
    padding: &[usize; 3],
    stride: &[usize; 3],
    dilation: &[usize; 3],
    groups: usize,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    // ny-build refuses a non-rank-5 kernel with a bare ShapeMismatch; catching
    // it here names the op and the actual shape instead.
    if weight.shape.len() != 5 {
        return Err(NyError::UnsupportedOp(format!(
            "Conv3d: kernel must be rank 5 [out_c, in_c/group, kD, kH, kW], got {:?}",
            weight.shape
        )));
    }

    let w_name = format!("{name}_weight");
    insert_payload(ctx, weight, &w_name, "Conv3d weight")?;
    if let Some(b) = bias {
        insert_payload(ctx, b, &format!("{name}_bias"), "Conv3d bias")?;
    }

    let sd = dim_as_i64(stride[0], "Conv3d stride_d")?;
    let sh = dim_as_i64(stride[1], "Conv3d stride_h")?;
    let sw = dim_as_i64(stride[2], "Conv3d stride_w")?;
    let pd = dim_as_i64(padding[0], "Conv3d pad_d")?;
    let ph = dim_as_i64(padding[1], "Conv3d pad_h")?;
    let pw = dim_as_i64(padding[2], "Conv3d pad_w")?;
    let dd = dim_as_i64(dilation[0], "Conv3d dilation_d")?;
    let dh = dim_as_i64(dilation[1], "Conv3d dilation_h")?;
    let dw = dim_as_i64(dilation[2], "Conv3d dilation_w")?;

    let mut attrs = HashMap::new();
    attrs.insert(
        "strides".to_string(),
        AttributeValue::Ints(vec![sd, sh, sw]),
    );
    // ONNX pads order: [x1_begin, .., xN_begin, x1_end, .., xN_end].
    attrs.insert(
        "pads".to_string(),
        AttributeValue::Ints(vec![pd, ph, pw, pd, ph, pw]),
    );
    attrs.insert(
        "dilations".to_string(),
        AttributeValue::Ints(vec![dd, dh, dw]),
    );
    attrs.insert(
        "group".to_string(),
        AttributeValue::Int(dim_as_i64(groups, "Conv3d groups")?),
    );

    let mut spec_inputs = input_tensors.to_vec();
    spec_inputs.push(w_name.clone());
    if bias.is_some() {
        spec_inputs.push(format!("{name}_bias"));
    }

    Ok(NodeOutput::one(LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::Conv3d,
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
    let w_name = format!("{name}_weight");
    insert_payload(ctx, weight, &w_name, "ConvTranspose1d weight")?;
    if let Some(b) = bias {
        insert_payload(ctx, b, &format!("{name}_bias"), "ConvTranspose1d bias")?;
    }

    let stride_i64 = dim_as_i64(stride, "ConvTranspose1d stride")?;
    let pad_i64 = dim_as_i64(padding, "ConvTranspose1d padding")?;
    let dilation_i64 = dim_as_i64(dilation, "ConvTranspose1d dilation")?;
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
    // ny-build's `parse_dilation_attr` reads `dilations` and threads it into
    // `ConvTranspose1dLayer::new_full`, whose forward/IBP/CROWN paths all
    // consume `self.dilation` (ny-propagate conv1d/types/transpose.rs) — no
    // refusal needed; dropping the attribute instead would silently build a
    // dilation=1 layer whose bounds do NOT enclose the traced op.
    attrs.insert(
        "dilations".to_string(),
        AttributeValue::Ints(vec![dilation_i64]),
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

    // Recover stride, padding, and dilation from attrs (already computed by
    // the caller).
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
    let dilation = match attrs.get("dilations") {
        Some(AttributeValue::Ints(v)) if !v.is_empty() => {
            checked_i64_to_usize(v[0], "ConvTranspose1d dilation attr")?
        }
        _ => 1,
    };

    // ConvTranspose1d output without output_padding:
    // T_mid = (in_len - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    // (the dilated effective kernel extent; matches PyTorch ConvTranspose1d and
    // ny-propagate `ConvTranspose1dLayer::output_length`). Using the raw
    // kernel_size here with dilation != 1 would size the zero-pad Linear
    // against the wrong T_mid and mis-place every output element — a silent
    // enclosure break, so the arithmetic is derived from the same effective
    // extent the layer itself uses.
    let effective_kernel = kernel_size
        .checked_sub(1)
        .and_then(|v| v.checked_mul(dilation))
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            NyError::InternalError(format!(
                "ConvTranspose1d output_padding: effective kernel overflow \
                 (kernel_size={kernel_size}, dilation={dilation})"
            ))
        })?;
    let t_mid = in_len
        .checked_sub(1)
        .and_then(|v| v.checked_mul(stride))
        .and_then(|v| v.checked_add(effective_kernel))
        .and_then(|v| v.checked_sub(2usize.checked_mul(padding)?))
        .ok_or_else(|| {
            NyError::InternalError(format!(
                "ConvTranspose1d output_padding: T_mid overflow (in_len={in_len}, \
                 stride={stride}, padding={padding}, kernel_size={kernel_size}, \
                 dilation={dilation})"
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
    // Fail-closed refusal: ny-propagate's `ConvTranspose2dLayer` models only
    // the dense groups=1 kernel layout (`new_full` takes no groups argument,
    // and ny-build's `convert_conv_transpose2d` refuses `group != 1`).
    // Emitting anyway would let a grouped kernel `[C_in, C_out/groups, kH, kW]`
    // be reinterpreted as dense — silently-wrong bounds, the worst failure —
    // so refuse at the bridge with the reason spelled out.
    if groups != 1 {
        return Err(NyError::UnsupportedOp(format!(
            "ConvTranspose2d groups={groups} not supported: ny's \
             ConvTranspose2dLayer models only the dense groups=1 kernel layout; \
             lowering a grouped kernel [C_in, C_out/groups, kH, kW] onto it \
             would reinterpret the kernel as dense and produce wrong (not \
             enclosing) bounds"
        )));
    }
    // Fail-closed refusal: output_padding only disambiguates the recovered
    // output size within one stride step, so it must be < stride per
    // dimension (PyTorch ConvTranspose2d contract). ny-propagate's
    // `ConvTranspose2dLayer::new_full` enforces the same bound downstream;
    // refusing here keeps the error at the translation boundary with the
    // trace's own parameters named.
    if output_padding[0] >= stride[0] || output_padding[1] >= stride[1] {
        return Err(NyError::UnsupportedOp(format!(
            "ConvTranspose2d output_padding ({}, {}) must be < stride ({}, {}) \
             per dimension: output_padding only disambiguates the \
             transposed-conv output size within one stride step (PyTorch \
             ConvTranspose2d contract, enforced by ny's ConvTranspose2dLayer)",
            output_padding[0], output_padding[1], stride[0], stride[1]
        )));
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
    let dil_h = dim_as_i64(dilation[0], "ConvTranspose2d dilation_h")?;
    let dil_w = dim_as_i64(dilation[1], "ConvTranspose2d dilation_w")?;
    let outpad_h = dim_as_i64(output_padding[0], "ConvTranspose2d output_padding_h")?;
    let outpad_w = dim_as_i64(output_padding[1], "ConvTranspose2d output_padding_w")?;
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
    // ny-build's `parse_dilation_attr` / `parse_output_padding_attr` read
    // both attributes and thread them into `ConvTranspose2dLayer::new_full`
    // (ny-build convert/conv.rs), whose output-size formula
    // `stride*(in-1) + dilation*(kernel-1) + 1 - 2*pad + output_padding`
    // consumes them natively. Dropping either would silently build a layer
    // whose bounds do not enclose the traced op.
    attrs.insert(
        "dilations".to_string(),
        AttributeValue::Ints(vec![dil_h, dil_w]),
    );
    attrs.insert(
        "output_padding".to_string(),
        AttributeValue::Ints(vec![outpad_h, outpad_w]),
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
// Embedding (midpoint Linear + EpsilonExpand hull widening)
// ---------------------------------------------------------------------------
//
// NY's graph build has no Embedding converter, and which row a lookup selects
// is not a quantity interval propagation can track. The surrogate is instead
// INDEX-VALUE-INDEPENDENT: whatever row is selected, the output lies in the
// per-dimension vocabulary hull
//   lower[d] = min(weight[0..V, d]), upper[d] = max(weight[0..V, d])
//
// Modeled as a Linear with zeros weight and midpoint bias — whose output is
// pinned to the tiled hull midpoint regardless of the input — followed by an
// EpsilonExpand carrying the outward-rounded per-dimension half-spreads, so
// the propagated interval [midpoint - eps, midpoint + eps] encloses
// [lower, upper] and with it every row the lookup could select. The midpoint
// Linear alone (NN's historic lowering) claims the DEGENERATE interval
// [midpoint, midpoint], which excludes almost every actual row; the
// EpsilonExpand widening is what makes the surrogate an enclosure.
//
// The Embedding maps [B, T] → [B, T, D], adding the embedding dimension. NY's
// Linear preserves rank (maps last dim only), so:
//   1. Linear: [B, T] → [B, T*D]  (W = zeros([T*D, T]), b = midpoint tiled T×)
//   2. EpsilonExpand: [B, T*D] → [B, T*D]  (radii = half-spreads tiled T×)
//   3. Reshape: [B, T*D] → [B, T, D]
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

    // Per-dimension vocabulary hull [lo, hi], its midpoint, and an
    // outward-rounded half-spread radius. eps[d] must dominate BOTH gaps in
    // EXACT arithmetic — midpoint - eps <= lo AND midpoint + eps >= hi — or
    // the widened interval would exclude an actual vocabulary row.
    // `f32::midpoint` rounds to nearest, so a naive `(hi - lo) / 2` can land
    // strictly inside a gap (e.g. lo = -2^26, hi = 1: the rounded half-spread
    // 2^25 leaves midpoint + eps = 0 < hi); the max of the two upward-rounded
    // gaps (`sub_up_f32` returns the smallest f32 >= the exact difference)
    // gives eps >= midpoint - lo and eps >= hi - midpoint by construction,
    // whichever side `f32::midpoint` rounded toward. Zero-spread dims
    // collapse to a true 0 radius. EpsilonExpand's IBP then applies the radii
    // with directed rounding (`sub_down_f32`/`add_up_f32`), which only ever
    // moves further outward.
    let mut midpoints = Vec::with_capacity(embedding_dim);
    let mut epsilons = Vec::with_capacity(embedding_dim);
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
        let mid = f32::midpoint(lo, hi);
        let eps = sub_up_f32(mid, lo).max(sub_up_f32(hi, mid)).max(0.0);
        if !eps.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "Embedding: half-spread radius overflows f32 at dim {d} \
                 (lo={lo}, hi={hi}, midpoint={mid})"
            )));
        }
        midpoints.push(mid);
        epsilons.push(eps);
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
        // Embedding adds dimensions: emit Linear → flat → EpsilonExpand →
        // Reshape to output shape. Linear maps in_features (T) →
        // out_features (T * D).
        let out_features = in_features.checked_mul(embedding_dim).ok_or_else(|| {
            NyError::InternalError(format!(
                "Embedding: in_features * embedding_dim overflow \
                 ({in_features} * {embedding_dim})"
            ))
        })?;

        // Tile midpoints and epsilon radii identically for each input
        // position: [v[0]..v[D-1], v[0]..v[D-1], ...] — the same row-major
        // layout the Reshape reads back as [.., T, D].
        let tiled_midpoints: Vec<f32> = midpoints
            .iter()
            .copied()
            .cycle()
            .take(out_features)
            .collect();
        let tiled_epsilons: Vec<f32> = epsilons
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

        // EpsilonExpand: widen the pinned midpoint by the tiled half-spread
        // radii, carried as a constant weight tensor named by inputs[1]
        // (ny-build's convert_epsilon_expand contract).
        let eps_w_name = format!("{name}_epsilons");
        let epsilon_arr =
            ndarray::ArrayD::from_shape_vec(IxDyn(&[out_features]), tiled_epsilons)
                .map_err(|e| NyError::InternalError(format!("Embedding epsilon array: {e}")))?;
        ctx.insert_weight(&eps_w_name, epsilon_arr)?;
        let expand_name = format!("{name}_emb_eps");
        let expand_out = format!("{expand_name}_out");
        let expand_spec = simple_spec(
            &expand_name,
            LayerType::EpsilonExpand,
            vec![linear_out.clone(), eps_w_name],
            &expand_out,
            HashMap::new(),
        );

        // Record intermediate shapes: input shape with last dim =
        // out_features (EpsilonExpand preserves shape).
        if let Some(input_shape) = ctx.tensor_shapes.get(&input_tensors[0]).cloned() {
            let mut lin_shape = input_shape;
            if let Some(last) = lin_shape.last_mut() {
                *last = dim_as_i64(out_features, "Embedding out_features")?;
            }
            ctx.tensor_shapes.insert(linear_out, lin_shape.clone());
            ctx.tensor_shapes.insert(expand_out.clone(), lin_shape);
        }

        // Reshape: flat widened tensor → traced output shape [B, T, D].
        let target_shape = shape_to_i64(output_shape, "Embedding output shape")?;
        let mut reshape_attrs = HashMap::new();
        reshape_attrs.insert("shape".to_string(), AttributeValue::Ints(target_shape));
        let reshape_spec = LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::Reshape,
            inputs: vec![expand_out],
            outputs: vec![output_tensor.to_string()],
            weights: None,
            attributes: reshape_attrs,
        };

        Ok(NodeOutput {
            specs: vec![linear_spec, expand_spec, reshape_spec],
        })
    } else {
        // Output rank == input rank (degenerate case): no Reshape needed —
        // midpoint Linear then the EpsilonExpand hull widening, un-tiled
        // (one radius per embedding dimension).
        let w_name = format!("{name}_weight");
        let zeros_weight =
            ndarray::ArrayD::from_elem(IxDyn(&[embedding_dim, in_features]), 0.0_f32);
        ctx.insert_weight(&w_name, zeros_weight)?;

        let b_name = format!("{name}_bias");
        let midpoint_arr = ndarray::ArrayD::from_shape_vec(IxDyn(&[embedding_dim]), midpoints)
            .map_err(|e| NyError::InternalError(format!("Embedding midpoint array: {e}")))?;
        ctx.insert_weight(&b_name, midpoint_arr)?;

        let eps_w_name = format!("{name}_epsilons");
        let epsilon_arr = ndarray::ArrayD::from_shape_vec(IxDyn(&[embedding_dim]), epsilons)
            .map_err(|e| NyError::InternalError(format!("Embedding epsilon array: {e}")))?;
        ctx.insert_weight(&eps_w_name, epsilon_arr)?;

        let mut attrs = HashMap::new();
        attrs.insert("transB".to_string(), AttributeValue::Int(1));

        let linear_name = format!("{name}_emb_linear");
        let linear_out = format!("{linear_name}_out");
        let linear_spec = LayerSpec {
            name: linear_name,
            layer_type: LayerType::Linear,
            inputs: vec![data_input, w_name.clone(), b_name],
            outputs: vec![linear_out.clone()],
            weights: Some(WeightRef {
                name: w_name,
                shape: vec![embedding_dim, in_features],
                original_dtype: DataType::Float32,
            }),
            attributes: attrs,
        };
        // EpsilonExpand preserves shape, so the intermediate has the traced
        // output shape.
        ctx.tensor_shapes.insert(
            linear_out.clone(),
            shape_to_i64(output_shape, "Embedding linear shape")?,
        );

        let expand_spec = simple_spec(
            name,
            LayerType::EpsilonExpand,
            vec![linear_out, eps_w_name],
            output_tensor,
            HashMap::new(),
        );

        Ok(NodeOutput {
            specs: vec![linear_spec, expand_spec],
        })
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
        assert_eq!(
            ct.attributes.get("dilations"),
            Some(&AttributeValue::Ints(vec![1]))
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

    /// Dilated ConvTranspose1d translates (the old fail-closed refusal is
    /// gone), emits the `dilations` attribute, and the built network's IBP on
    /// a point box matches a directly constructed
    /// `ConvTranspose1dLayer::new_full` — pinning that the attribute really
    /// reaches the layer (a dropped attribute would produce dilation=1
    /// bounds that do NOT enclose the traced op).
    #[test]
    fn conv_transpose1d_dilated_matches_direct_layer() {
        let kernel_data = vec![0.5_f32, 1.0, -0.5];
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 4]),
            node(
                1,
                "ct",
                TraceOp::ConvTranspose1d {
                    weight: WeightPayload::f32(kernel_data.clone(), vec![1, 1, 3]),
                    bias: Some(WeightPayload::f32(vec![0.1], vec![1])),
                    padding: 0,
                    output_padding: 0,
                    stride: 1,
                    dilation: 2,
                    groups: 1,
                },
                &[0],
                // out_len = (4-1)*1 + 2*(3-1) + 1 = 8
                &[1, 1, 8],
            ),
        ]);
        let model = translate(&graph).expect("dilated ConvTranspose1d translates");
        let ct = find(&model, &LayerType::ConvTranspose1d);
        assert_eq!(
            ct.attributes.get("dilations"),
            Some(&AttributeValue::Ints(vec![2]))
        );

        let gn = model
            .build_graph_network(GraphNetworkOptions::default())
            .expect("dilated ConvTranspose1d builds");
        let x = vec![1.0_f32, 2.0, -1.0, 3.0];
        let point = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 4]), x).expect("shape");
        let input = ny_tensor::BoundedTensor::new(point.clone(), point).expect("valid box");
        let out = gn.propagate_ibp(&input).expect("dilated CT1d IBP");

        // Direct layer construction — same kernel/bias/hyperparameters.
        use ny_propagate::BoundPropagation;
        let kernel = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 3]), kernel_data)
            .expect("kernel shape");
        let layer = ny_propagate::layers::ConvTranspose1dLayer::new_full(
            kernel,
            Some(ndarray::Array1::from_vec(vec![0.1])),
            1, // stride
            0, // padding
            2, // dilation
            1, // groups
        )
        .expect("direct layer constructs");
        let reference = layer.propagate_ibp(&input).expect("direct layer IBP");

        assert_eq!(out.shape(), &[1, 1, 8]);
        assert_eq!(out.shape(), reference.shape());
        for (i, (((gl, gu), rl), ru)) in out
            .lower()
            .iter()
            .zip(out.upper().iter())
            .zip(reference.lower().iter())
            .zip(reference.upper().iter())
            .enumerate()
        {
            assert!(
                (gl - rl).abs() < 1e-6 && (gu - ru).abs() < 1e-6,
                "element {i}: graph [{gl}, {gu}] vs direct layer [{rl}, {ru}]"
            );
        }
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

    /// Dilated ConvTranspose1d with output_padding: the decomposition's T_mid
    /// must use the dilated effective kernel extent
    /// (`dilation * (kernel_size - 1) + 1`), and the built network's IBP on a
    /// point box must match the direct `ConvTranspose1dLayer` on the first
    /// T_mid positions with an exact-zero pad tail. A T_mid computed from the
    /// raw kernel_size would mis-size the pad Linear (caught here by both the
    /// shape and the value assertions).
    #[test]
    fn conv_transpose1d_dilated_output_padding_matches_direct_layer() {
        let kernel_data = vec![0.5_f32, 1.0, -0.5];
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 4]),
            node(
                1,
                "ct",
                TraceOp::ConvTranspose1d {
                    weight: WeightPayload::f32(kernel_data.clone(), vec![1, 1, 3]),
                    bias: None,
                    padding: 0,
                    output_padding: 1,
                    stride: 2,
                    dilation: 2,
                    groups: 1,
                },
                &[0],
                // T_mid = (4-1)*2 + 2*(3-1) + 1 = 11; T_out = 12.
                &[1, 1, 12],
            ),
        ]);
        let model = translate(&graph).expect("dilated output_padding CT1d translates");
        let lin = find(&model, &LayerType::Linear);
        assert_eq!(
            lin.weights.as_ref().expect("pad WeightRef").shape,
            vec![12, 11],
            "pad weight sized from the DILATED T_mid"
        );

        let gn = model
            .build_graph_network(GraphNetworkOptions::default())
            .expect("dilated output_padding CT1d builds");
        let x = vec![1.0_f32, 2.0, -1.0, 3.0];
        let point = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 4]), x).expect("shape");
        let input = ny_tensor::BoundedTensor::new(point.clone(), point).expect("valid box");
        let out = gn
            .propagate_ibp(&input)
            .expect("dilated output_padding CT1d IBP");
        assert_eq!(out.shape(), &[1, 1, 12]);

        // Direct layer (no output_padding field on ny's CT1d layer): compare
        // the first T_mid positions, then the zero tail.
        use ny_propagate::BoundPropagation;
        let kernel = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 3]), kernel_data)
            .expect("kernel shape");
        let layer = ny_propagate::layers::ConvTranspose1dLayer::new_full(kernel, None, 2, 0, 2, 1)
            .expect("direct layer constructs");
        let reference = layer.propagate_ibp(&input).expect("direct layer IBP");
        assert_eq!(reference.shape(), &[1, 1, 11]);

        for t in 0..11 {
            let (gl, gu) = (out.lower()[[0, 0, t]], out.upper()[[0, 0, t]]);
            let (rl, ru) = (reference.lower()[[0, 0, t]], reference.upper()[[0, 0, t]]);
            // The pad Linear's IBP is outward-rounded, so the graph interval
            // may carry a few ULPs of directed-rounding slack around the
            // direct layer's value: require ENCLOSURE (never inward) and
            // tightness within float dust.
            assert!(
                gl <= rl + 1e-6 && gu >= ru - 1e-6,
                "position {t}: graph [{gl}, {gu}] must enclose direct layer [{rl}, {ru}]"
            );
            assert!(
                (gl - rl).abs() < 1e-4 && (gu - ru).abs() < 1e-4,
                "position {t}: graph [{gl}, {gu}] vs direct layer [{rl}, {ru}]"
            );
        }
        // The pad tail is the zero row of the pad Linear; its outward-rounded
        // IBP encloses 0 within a subnormal-width sliver.
        let (tl, tu) = (out.lower()[[0, 0, 11]], out.upper()[[0, 0, 11]]);
        assert!(
            tl <= 0.0 && 0.0 <= tu && tl > -1e-6 && tu < 1e-6,
            "pad tail must be a tight enclosure of zero, got [{tl}, {tu}]"
        );
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
        assert_eq!(
            ct.attributes.get("dilations"),
            Some(&AttributeValue::Ints(vec![1, 1]))
        );
        assert_eq!(
            ct.attributes.get("output_padding"),
            Some(&AttributeValue::Ints(vec![0, 0]))
        );
        assert_eq!(ct.attributes.get("group"), Some(&AttributeValue::Int(1)));
        // Data input + weight (no bias).
        assert_eq!(ct.inputs.len(), 2);
        builds(&model, "ConvTranspose2d");
    }

    /// Dilated ConvTranspose2d with output_padding < stride translates (both
    /// old blanket refusals are gone), emits the `dilations` and
    /// `output_padding` attributes, and the built network's IBP on a point
    /// box matches a directly constructed `ConvTranspose2dLayer::new_full` —
    /// pinning that both attributes reach the layer.
    #[test]
    fn conv_transpose2d_dilated_output_padding_matches_direct_layer() {
        let kernel_data = vec![1.0_f32, 0.5, -0.5, 0.3];
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 2, 2]),
            node(
                1,
                "ct",
                TraceOp::ConvTranspose2d {
                    weight: WeightPayload::f32(kernel_data.clone(), vec![1, 1, 2, 2]),
                    bias: Some(WeightPayload::f32(vec![0.25], vec![1])),
                    padding: [0, 0],
                    output_padding: [1, 1],
                    stride: [2, 2],
                    dilation: [2, 2],
                    groups: 1,
                },
                &[0],
                // out = (2-1)*2 + 2*(2-1) + 1 - 0 + 1 = 6 per spatial dim.
                &[1, 1, 6, 6],
            ),
        ]);
        let model = translate(&graph).expect("dilated output_padding CT2d translates");
        let ct = find(&model, &LayerType::ConvTranspose2d);
        assert_eq!(
            ct.attributes.get("dilations"),
            Some(&AttributeValue::Ints(vec![2, 2]))
        );
        assert_eq!(
            ct.attributes.get("output_padding"),
            Some(&AttributeValue::Ints(vec![1, 1]))
        );

        let gn = model
            .build_graph_network(GraphNetworkOptions::default())
            .expect("dilated output_padding CT2d builds");
        let x = vec![1.0_f32, -2.0, 3.0, 0.5];
        let point =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 2, 2]), x).expect("shape");
        let input = ny_tensor::BoundedTensor::new(point.clone(), point).expect("valid box");
        let out = gn.propagate_ibp(&input).expect("dilated CT2d IBP");

        use ny_propagate::BoundPropagation;
        let kernel = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 2, 2]), kernel_data)
            .expect("kernel shape");
        let layer = ny_propagate::layers::ConvTranspose2dLayer::new_full(
            kernel,
            Some(ndarray::Array1::from_vec(vec![0.25])),
            (2, 2), // stride
            (0, 0), // padding
            (2, 2), // dilation
            (1, 1), // output_padding
        )
        .expect("direct layer constructs");
        let reference = layer.propagate_ibp(&input).expect("direct layer IBP");

        assert_eq!(out.shape(), &[1, 1, 6, 6]);
        assert_eq!(out.shape(), reference.shape());
        for (i, (((gl, gu), rl), ru)) in out
            .lower()
            .iter()
            .zip(out.upper().iter())
            .zip(reference.lower().iter())
            .zip(reference.upper().iter())
            .enumerate()
        {
            assert!(
                (gl - rl).abs() < 1e-6 && (gu - ru).abs() < 1e-6,
                "element {i}: graph [{gl}, {gu}] vs direct layer [{rl}, {ru}]"
            );
        }
    }

    /// ConvTranspose2d refusals that REMAIN fail-closed: output_padding >=
    /// stride (only recoverable within one stride step; the layer enforces
    /// the same bound) and grouped kernels (the layer models groups=1 only —
    /// a grouped kernel reinterpreted as dense would be silently wrong).
    #[test]
    fn conv_transpose2d_refusals() {
        let base_weight = || WeightPayload::f32(vec![1.0, 0.5, -0.5, 0.3], vec![1, 1, 2, 2]);
        let cases = [
            (
                TraceOp::ConvTranspose2d {
                    weight: base_weight(),
                    bias: None,
                    padding: [0, 0],
                    output_padding: [2, 0],
                    stride: [2, 2],
                    dilation: [1, 1],
                    groups: 1,
                },
                "output_padding (2, 0) must be < stride (2, 2)",
            ),
            (
                TraceOp::ConvTranspose2d {
                    weight: base_weight(),
                    bias: None,
                    padding: [0, 0],
                    output_padding: [0, 0],
                    stride: [1, 1],
                    dilation: [1, 1],
                    groups: 2,
                },
                "groups=2 not supported",
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
                "expected refusal containing {needle:?}, got {err:?}"
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
    /// midpoint bias) + EpsilonExpand (half-spread radii) + Reshape, with
    /// NN's per-dimension midpoints.
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
        assert_eq!(count(&model, &LayerType::EpsilonExpand), 1);
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

        // Half-spread radii: every dim spans 4.0, so eps = 2.0 (exact — no
        // rounding step needed for these dyadic endpoints).
        let eps = model
            .weights
            .get("layer0_trace_1_epsilons")
            .expect("epsilon radii stored");
        assert_eq!(
            eps.iter().copied().collect::<Vec<f32>>(),
            vec![2.0, 2.0, 2.0, 2.0]
        );

        // The widening sits between the Linear and the Reshape.
        let expand = find(&model, &LayerType::EpsilonExpand);
        assert_eq!(
            expand.inputs,
            vec![
                "layer0_trace_1_emb_linear_out".to_string(),
                "layer0_trace_1_epsilons".to_string(),
            ]
        );
        let rs = find(&model, &LayerType::Reshape);
        assert_eq!(rs.inputs, vec!["layer0_trace_1_emb_eps_out".to_string()]);

        // Reshape targets the traced output shape.
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

    fn conv3d_graph() -> ComputationGraph {
        ComputationGraph::from_nodes(vec![
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
        ])
    }

    /// Conv3d LOWERS now — the refusal that stood here rested on a premise
    /// that no longer holds.
    ///
    /// The old test asserted the message named a MISSING LAYER and pinned that
    /// premise directly: `"Conv3d".parse::<LayerType>() == Unknown`, with the
    /// note that it "starts failing the day a real Conv3d variant lands —
    /// which is exactly when this refusal must be revisited." That day came.
    /// The assertion is kept, inverted: `LayerType::Conv3d` exists, so the
    /// translator must emit it rather than refuse.
    #[test]
    fn conv3d_lowers_now_that_the_layer_exists() {
        assert_eq!(
            "Conv3d".parse::<LayerType>(),
            Ok(LayerType::Conv3d),
            "the layer the old refusal said was missing now exists"
        );

        let model = translate(&conv3d_graph()).expect("Conv3d lowers");
        let spec = model
            .network
            .layers
            .iter()
            .find(|l| l.layer_type == LayerType::Conv3d)
            .expect("a Conv3d spec is emitted");
        assert_eq!(spec.name, "layer0_trace_1");
        assert_eq!(
            spec.inputs,
            vec!["layer0_trace_0_out", "layer0_trace_1_weight"]
        );
        assert_eq!(
            spec.attributes.get("strides"),
            Some(&AttributeValue::Ints(vec![1, 1, 1]))
        );
        // ONNX rank-3 pads: [d_b, h_b, w_b, d_e, h_e, w_e], symmetric by
        // construction from the trace's one-per-axis padding.
        assert_eq!(
            spec.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![0, 0, 0, 0, 0, 0]))
        );
        assert_eq!(
            spec.attributes.get("dilations"),
            Some(&AttributeValue::Ints(vec![1, 1, 1]))
        );
        assert_eq!(spec.attributes.get("group"), Some(&AttributeValue::Int(1)));

        model
            .build_graph_network(GraphNetworkOptions::default())
            .expect("the Conv3d model builds a graph network");
    }

    /// The emitted Conv3d propagates IBP and its interval ENCLOSES the exact
    /// convolution at a point input.
    ///
    /// Geometry: a `[1, 1, 2, 2, 2]` volume against an all-`0.1` `[1,1,2,2,2]`
    /// kernel, so the single output cell is `0.1 * sum(input)` — a number the
    /// test computes independently of the layer.
    #[test]
    fn conv3d_ibp_encloses_the_exact_convolution() {
        let model = translate(&conv3d_graph()).expect("Conv3d lowers");
        let network = model
            .build_graph_network(GraphNetworkOptions::default())
            .expect("Conv3d builds");

        let values = [1.0_f32, -2.0, 3.0, 0.5, -1.5, 4.0, 0.25, -0.75];
        let point =
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1, 2, 2, 2]), values.to_vec())
                .unwrap();
        let bounds = ny_tensor::BoundedTensor::new(point.clone(), point).expect("point bounds");
        let out = network.propagate_ibp(&bounds).expect("Conv3d IBP");
        assert_eq!(out.shape(), &[1, 1, 1, 1, 1]);

        let expected: f32 = values.iter().map(|v| v * 0.1).sum();
        let (lo, hi) = (
            out.lower().as_slice().unwrap()[0],
            out.upper().as_slice().unwrap()[0],
        );
        assert!(
            lo <= expected + 1e-5 && expected - 1e-5 <= hi,
            "exact convolution {expected} must be enclosed by [{lo}, {hi}]"
        );

        // Non-vacuous: a rounding-width interval, not [-inf, +inf].
        assert!(
            (hi - lo) < 1e-3,
            "point box must give a tight interval, got [{lo}, {hi}]"
        );
    }

    /// Dilation is THREADED THROUGH rather than refused (as Conv2d must) or
    /// dropped (which would build a dilation-1 layer whose bounds do not
    /// enclose the traced op). `Conv3dLayer::new_dilated` consumes it.
    #[test]
    fn conv3d_dilation_reaches_the_layer() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 3, 3, 3]),
            node(
                1,
                "c3",
                TraceOp::Conv3d {
                    weight: WeightPayload::f32(vec![0.1; 8], vec![1, 1, 2, 2, 2]),
                    bias: Some(WeightPayload::f32(vec![0.5], vec![1])),
                    padding: [1, 0, 0],
                    stride: [1, 2, 1],
                    dilation: [2, 1, 1],
                    groups: 1,
                },
                &[0],
                &[1, 1, 2, 1, 2],
            ),
        ]);
        let model = translate(&graph).expect("dilated Conv3d lowers");
        let spec = model
            .network
            .layers
            .iter()
            .find(|l| l.layer_type == LayerType::Conv3d)
            .expect("a Conv3d spec is emitted");
        assert_eq!(
            spec.attributes.get("dilations"),
            Some(&AttributeValue::Ints(vec![2, 1, 1])),
            "dilation must survive to the layer"
        );
        assert_eq!(
            spec.attributes.get("strides"),
            Some(&AttributeValue::Ints(vec![1, 2, 1]))
        );
        assert_eq!(
            spec.attributes.get("pads"),
            Some(&AttributeValue::Ints(vec![1, 0, 0, 1, 0, 0])),
            "per-axis padding mirrored into ONNX begin/end form"
        );
        assert_eq!(
            spec.inputs,
            vec![
                "layer0_trace_0_out",
                "layer0_trace_1_weight",
                "layer0_trace_1_bias"
            ],
            "bias is wired as the third operand"
        );
    }

    /// A kernel that is not rank 5 fails closed HERE, naming the op and the
    /// shape, instead of surfacing as ny-build's bare `ShapeMismatch`.
    #[test]
    fn conv3d_refuses_a_non_rank5_kernel() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 1, 2, 2, 2]),
            node(
                1,
                "c3",
                TraceOp::Conv3d {
                    weight: WeightPayload::f32(vec![0.1; 4], vec![1, 1, 2, 2]),
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
        let err = translate(&graph).expect_err("a rank-4 kernel is not a Conv3d kernel");
        assert!(
            matches!(&err, NyError::UnsupportedOp(m)
                if m.contains("Conv3d") && m.contains("rank 5")),
            "got {err:?}"
        );
    }
}
