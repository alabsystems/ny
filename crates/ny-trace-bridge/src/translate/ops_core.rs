// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Already-implemented op arms of `translate_node`'s dispatch, moved verbatim
//! from the pre-split `translate.rs`.
//!
//! Covers the supported core set: Input/Constant/ConstantWeight, unary and
//! parameterized activations (incl. GELU variants, LeakyRelu/PRelu
//! decompositions, the named-activation fallback), binary elementwise + the
//! Sub decomposition, MatMul, Linear, Conv1d/Conv2d (with dilated-Conv1d
//! kernel expansion), reductions, shape ops, Softmax/LogSoftmax,
//! normalizations (LayerNorm/RmsNorm/InstanceNorm/BatchNorm and the GroupNorm
//! decomposition), Clamp, Dropout, the ToDtype cast modeling, and the
//! constant-folding fast paths. Not-yet-ported families live in the sibling
//! `ops_*` stub modules; shared helpers come from `super`.

use std::collections::HashMap;

use ndarray::{ArrayD, Axis, IxDyn};
use ny_build::{AttributeValue, DataType, LayerSpec, WeightRef};
use ny_core::{checked_shape_product, LayerType, NyError, Result};

use crate::schema::{DType, TraceActivation, TraceOp, WeightPayload};

use super::{
    checked_f64_to_f32, dim_as_i64, first_input, insert_payload, insert_scalar_constant, op_name,
    shape_to_i64, simple_spec, validate_eps, weight_f32, Ctx, NodeOutput,
};

/// Cap translator-owned dense materializations so compact malformed traces
/// cannot force unbounded allocations. Matches ny-onnx constant folding.
const MAX_MATERIALIZED_ELEMENTS: usize = 10_000_000;

/// Fallibly materialize a dense f32 array after checked shape/cap validation.
fn materialize_filled_array(shape: &[usize], value: f32, context: &str) -> Result<ArrayD<f32>> {
    let elements = checked_shape_product(shape).ok_or_else(|| {
        NyError::ModelLoad(format!(
            "{context}: shape {shape:?} has a dimension product that overflows usize"
        ))
    })?;
    if elements > MAX_MATERIALIZED_ELEMENTS {
        return Err(NyError::ModelLoad(format!(
            "{context}: shape {shape:?} requires {elements} elements, exceeding the {MAX_MATERIALIZED_ELEMENTS}-element materialization limit"
        )));
    }
    let mut data = Vec::new();
    data.try_reserve_exact(elements).map_err(|error| {
        NyError::ModelLoad(format!(
            "{context}: allocation failed for {elements} elements: {error}"
        ))
    })?;
    data.resize(elements, value);
    ArrayD::from_shape_vec(IxDyn(shape), data).map_err(|error| {
        NyError::ModelLoad(format!(
            "{context}: shape {shape:?} could not be materialized: {error}"
        ))
    })
}

// ---------------------------------------------------------------------------
// Constant / input translators
// ---------------------------------------------------------------------------

/// Translate `TraceOp::Input` — wrap in `Add(input, 0)` identity.
pub(super) fn translate_input(
    name: &str,
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let const_name = format!("{name}_zero");
    insert_scalar_constant(ctx, &const_name, 0.0)?;
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Add,
        vec![format!("{name}_in"), const_name],
        output_tensor,
        HashMap::new(),
    )))
}

/// Translate `TraceOp::Constant` — register the value as a constant weight.
pub(super) fn translate_constant(
    value: f64,
    output_shape: &[usize],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let val_f32 = checked_f64_to_f32(value, "Constant")?;
    let arr = materialize_filled_array(output_shape, val_f32, "Constant")?;
    ctx.insert_weight(output_tensor, arr)?;
    Ok(NodeOutput::none())
}

/// Translate `TraceOp::ConstantWeight` — register the captured data as constant.
pub(super) fn translate_constant_weight(
    weight: &WeightPayload,
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    insert_payload(ctx, weight, output_tensor, "ConstantWeight")?;
    Ok(NodeOutput::none())
}

// ---------------------------------------------------------------------------
// Activation translators
// ---------------------------------------------------------------------------

/// Translate a simple unary activation without changing its input domain.
///
/// In particular, Exp/Softplus/Mish must not be preceded by a Clamp: doing so
/// changes valid values above the f32 exponential threshold. Their propagation
/// layers either evaluate stably or fail closed on an unsafe domain.
pub(super) fn translate_unary_activation(
    op: &TraceOp,
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, op_name(op))?;

    let layer_type = match op {
        TraceOp::Relu => LayerType::ReLU,
        TraceOp::Sigmoid => LayerType::Sigmoid,
        TraceOp::Tanh => LayerType::Tanh,
        TraceOp::Silu => LayerType::SiLU,
        TraceOp::Sqrt => LayerType::Sqrt,
        TraceOp::Abs => LayerType::Abs,
        TraceOp::Recip => LayerType::Reciprocal,
        TraceOp::Log => LayerType::Log,
        TraceOp::Sin => LayerType::Sin,
        TraceOp::Cos => LayerType::Cos,
        TraceOp::Floor => LayerType::Floor,
        TraceOp::Round => LayerType::Round,
        TraceOp::Neg => LayerType::Neg,
        TraceOp::Tan => LayerType::Tan,
        TraceOp::Ceil => LayerType::Ceil,
        TraceOp::Sign => LayerType::Sign,
        TraceOp::HardSigmoid => LayerType::HardSigmoid,
        TraceOp::HardSwish => LayerType::HardSwish,
        TraceOp::Selu => LayerType::Selu,
        TraceOp::Softsign => LayerType::Softsign,
        TraceOp::Exp => LayerType::Exp,
        TraceOp::Softplus => LayerType::Softplus,
        TraceOp::Mish => LayerType::Mish,
        _ => {
            return Err(NyError::UnsupportedOp(format!(
                "not a simple unary activation: {}",
                op_name(op)
            )));
        }
    };

    Ok(NodeOutput::one(simple_spec(
        name,
        layer_type,
        vec![data_input],
        output_tensor,
        HashMap::new(),
    )))
}

/// Translate GELU with the given `approximate` mode ("tanh" or "none").
pub(super) fn translate_gelu(
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    approximate: &str,
) -> NodeOutput {
    let mut attrs = HashMap::new();
    attrs.insert(
        "approximate".to_string(),
        AttributeValue::String(approximate.to_string()),
    );
    NodeOutput::one(simple_spec(
        name,
        LayerType::GELU,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    ))
}

/// Translate `TraceOp::Sqr` as `Pow(2)`.
pub(super) fn translate_sqr(
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "Sqr")?;
    let const_name = format!("{name}_pow2");
    insert_scalar_constant(ctx, &const_name, 2.0)?;
    let mut attrs = HashMap::new();
    attrs.insert("power".to_string(), AttributeValue::Float(2.0));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Pow,
        vec![data_input, const_name],
        output_tensor,
        attrs,
    )))
}

/// Translate Softmax / LogSoftmax (raw dim; backend adjusts axis internally).
pub(super) fn translate_softmax(
    name: &str,
    layer_type: LayerType,
    dim: usize,
    input_tensors: Vec<String>,
    output_tensor: &str,
) -> Result<NodeOutput> {
    let mut attrs = HashMap::new();
    attrs.insert(
        "axis".to_string(),
        AttributeValue::Int(dim_as_i64(dim, "Softmax axis")?),
    );
    Ok(NodeOutput::one(simple_spec(
        name,
        layer_type,
        input_tensors,
        output_tensor,
        attrs,
    )))
}

/// Translate `TraceOp::Clamp` / Clip.
pub(super) fn translate_clamp(
    name: &str,
    min: Option<f64>,
    max: Option<f64>,
    input_tensors: Vec<String>,
    output_tensor: &str,
) -> Result<NodeOutput> {
    let mut attrs = HashMap::new();
    if let Some(lo) = min {
        attrs.insert(
            "min".to_string(),
            AttributeValue::Float(checked_f64_to_f32(lo, "Clamp min")?),
        );
    }
    if let Some(hi) = max {
        attrs.insert(
            "max".to_string(),
            AttributeValue::Float(checked_f64_to_f32(hi, "Clamp max")?),
        );
    }
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Clip,
        input_tensors,
        output_tensor,
        attrs,
    )))
}

/// Translate an identity op via `Add + 0.0`.
///
/// Used for ops that are semantically guaranteed not to change numerical
/// values, such as Dropout in eval mode and `Powf` with exponent 1.
pub(super) fn translate_identity_add_zero(
    name: &str,
    op_desc: &str,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, op_desc)?;
    let const_name = format!("{name}_zero");
    insert_scalar_constant(ctx, &const_name, 0.0)?;
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Add,
        vec![data_input, const_name],
        output_tensor,
        HashMap::new(),
    )))
}

/// Translate `TraceOp::Dropout` as `Add(x, 0)` identity.
pub(super) fn translate_dropout(
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    translate_identity_add_zero(name, "Dropout", input_tensors, output_tensor, ctx)
}

/// Refuse `TraceOp::ToDtype` casts.
///
/// The wire format records only the target dtype. Without the source dtype,
/// even an F32/F64 target may be a precision-losing cast. The bridge therefore
/// cannot prove that any `ToDtype` is an identity and has no sound lowering for
/// it.
pub(super) fn translate_to_dtype(
    target_dtype: DType,
    input_tensors: &[String],
) -> Result<NodeOutput> {
    let _ = first_input(input_tensors, "ToDtype")?;
    Err(NyError::UnsupportedOp(format!(
        "ToDtype target {target_dtype:?} cannot be soundly lowered because the trace does not record the source dtype"
    )))
}

/// Translate `TraceOp::Elu { alpha }`.
pub(super) fn translate_elu(
    name: &str,
    alpha: f64,
    input_tensors: Vec<String>,
    output_tensor: &str,
) -> Result<NodeOutput> {
    let mut attrs = HashMap::new();
    attrs.insert(
        "alpha".to_string(),
        AttributeValue::Float(checked_f64_to_f32(alpha, "Elu alpha")?),
    );
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Elu,
        input_tensors,
        output_tensor,
        attrs,
    )))
}

/// Translate `TraceOp::Celu { alpha }`.
pub(super) fn translate_celu(
    name: &str,
    alpha: f64,
    input_tensors: Vec<String>,
    output_tensor: &str,
) -> Result<NodeOutput> {
    let mut attrs = HashMap::new();
    attrs.insert(
        "alpha".to_string(),
        AttributeValue::Float(checked_f64_to_f32(alpha, "Celu alpha")?),
    );
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Celu,
        input_tensors,
        output_tensor,
        attrs,
    )))
}

/// Translate `TraceOp::LeakyRelu { slope }` via decomposition.
///
/// `LeakyReLU(x, α) = α*x + (1-α)*ReLU(x)`, so NY uses `ReLU`'s correct CROWN
/// linearization rather than a wide `LeakyReLU` relaxation. Emits 4 specs.
pub(super) fn translate_leaky_relu(
    name: &str,
    slope: f64,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let slope_f32 = checked_f64_to_f32(slope, "LeakyRelu slope")?;
    let input_tensor = first_input(input_tensors, "LeakyRelu")?;

    let alpha_const = format!("{name}_lr_alpha");
    insert_scalar_constant(ctx, &alpha_const, slope_f32)?;
    let one_minus_alpha_const = format!("{name}_lr_1ma");
    insert_scalar_constant(ctx, &one_minus_alpha_const, 1.0 - slope_f32)?;

    let alpha_x = format!("{name}_lr_ax");
    let alpha_x_spec = simple_spec(
        &alpha_x,
        LayerType::Mul,
        vec![input_tensor.clone(), alpha_const],
        &alpha_x,
        HashMap::new(),
    );
    let relu_x = format!("{name}_lr_relu");
    let relu_spec = simple_spec(
        &relu_x,
        LayerType::ReLU,
        vec![input_tensor],
        &relu_x,
        HashMap::new(),
    );
    let relu_scaled = format!("{name}_lr_rs");
    let relu_scaled_spec = simple_spec(
        &relu_scaled,
        LayerType::Mul,
        vec![relu_x, one_minus_alpha_const],
        &relu_scaled,
        HashMap::new(),
    );
    let add_spec = simple_spec(
        name,
        LayerType::Add,
        vec![alpha_x, relu_scaled],
        output_tensor,
        HashMap::new(),
    );
    Ok(NodeOutput {
        specs: vec![alpha_x_spec, relu_spec, relu_scaled_spec, add_spec],
    })
}

/// Translate `TraceOp::PRelu { slope }` with a per-channel slope weight.
pub(super) fn translate_prelu(
    name: &str,
    slope: &WeightPayload,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "PRelu")?;
    let slope_name = format!("{name}_slope");
    insert_payload(ctx, slope, &slope_name, "PRelu slope")?;
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::PRelu,
        vec![data_input, slope_name],
        output_tensor,
        HashMap::new(),
    )))
}

/// Translate the generic named-activation fallback.
///
/// Elu/LeakyRelu are deliberately rejected here: routing them through the
/// generic path would silently use default parameters, producing wrong bounds.
/// They must come through the dedicated `Elu { alpha }` / `LeakyRelu { slope }`
/// variants that preserve the actual parameter.
pub(super) fn translate_named_activation(
    name: &str,
    kind: TraceActivation,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    let layer_type = match kind {
        TraceActivation::Relu => LayerType::ReLU,
        TraceActivation::Gelu => {
            return Ok(translate_gelu(name, input_tensors, output_tensor, "tanh"));
        }
        TraceActivation::GeluErf => {
            return Ok(translate_gelu(name, input_tensors, output_tensor, "none"));
        }
        TraceActivation::Silu => LayerType::SiLU,
        TraceActivation::Sigmoid => LayerType::Sigmoid,
        TraceActivation::Tanh => LayerType::Tanh,
        TraceActivation::Log => LayerType::Log,
        TraceActivation::Exp => {
            // Share the direct Exp lowering and its fail-closed propagation.
            return translate_unary_activation(&TraceOp::Exp, name, input_tensors, output_tensor);
        }
        TraceActivation::Mish => {
            return translate_unary_activation(&TraceOp::Mish, name, input_tensors, output_tensor);
        }
        TraceActivation::Elu => {
            return Err(NyError::UnsupportedOp(
                "Activation 'Elu' via generic path rejected — use TraceOp::Elu { alpha } \
                 to preserve the actual alpha parameter"
                    .to_string(),
            ));
        }
        TraceActivation::LeakyRelu => {
            return Err(NyError::UnsupportedOp(
                "Activation 'LeakyRelu' via generic path rejected — use \
                 TraceOp::LeakyRelu { slope } to preserve the actual slope parameter"
                    .to_string(),
            ));
        }
    };
    Ok(NodeOutput::one(simple_spec(
        name,
        layer_type,
        input_tensors.to_vec(),
        output_tensor,
        HashMap::new(),
    )))
}

// ---------------------------------------------------------------------------
// Binary / reduction translators
// ---------------------------------------------------------------------------

/// Translate a simple binary op.
pub(super) fn translate_binary(
    op: &TraceOp,
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::UnsupportedOp(format!(
            "{} requires two inputs, got {}",
            op_name(op),
            input_tensors.len()
        )));
    }
    let layer_type = match op {
        TraceOp::Add => LayerType::Add,
        TraceOp::Mul => LayerType::Mul,
        TraceOp::Div => LayerType::Div,
        TraceOp::Maximum => LayerType::Max,
        TraceOp::Minimum => LayerType::Min,
        _ => {
            return Err(NyError::UnsupportedOp(format!(
                "not a simple binary op: {}",
                op_name(op)
            )));
        }
    };
    Ok(NodeOutput::one(simple_spec(
        name,
        layer_type,
        input_tensors.to_vec(),
        output_tensor,
        HashMap::new(),
    )))
}

/// Translate `TraceOp::Sub` as `Add(lhs, Neg(rhs))`.
pub(super) fn translate_sub(
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
) -> Result<NodeOutput> {
    let lhs = first_input(input_tensors, "Sub")?;
    let rhs = input_tensors
        .get(1)
        .ok_or_else(|| NyError::UnsupportedOp("Sub requires two inputs".to_string()))?
        .clone();
    let neg_name = format!("{name}_neg");
    let neg_out = format!("{neg_name}_out");
    let neg_spec = simple_spec(
        &neg_name,
        LayerType::Neg,
        vec![rhs],
        &neg_out,
        HashMap::new(),
    );
    let add_spec = simple_spec(
        name,
        LayerType::Add,
        vec![lhs, neg_out],
        output_tensor,
        HashMap::new(),
    );
    Ok(NodeOutput {
        specs: vec![neg_spec, add_spec],
    })
}

/// Translate a reduction op.
pub(super) fn translate_reduce(
    op: &TraceOp,
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &Ctx,
) -> Result<NodeOutput> {
    let (layer_type, dim, keepdim) = match op {
        TraceOp::ReduceSum { dim, keepdim } => (LayerType::ReduceSum, *dim, *keepdim),
        TraceOp::ReduceMean { dim, keepdim } => (LayerType::ReduceMean, *dim, *keepdim),
        TraceOp::ReduceMax { dim, keepdim } => (LayerType::ReduceMax, *dim, *keepdim),
        TraceOp::ReduceMin { dim, keepdim } => (LayerType::ReduceMin, *dim, *keepdim),
        _ => {
            return Err(NyError::UnsupportedOp(format!(
                "not a reduce op: {}",
                op_name(op)
            )));
        }
    };
    // ONNX Reduce* axes index the INPUT rank. The input tensor is a
    // predecessor node's output (or a recorded constant), so its shape is
    // always in ctx.tensor_shapes by the time this node is translated; fail
    // closed if not. Trailing-relative negative encoding (see
    // `super::trailing_axis`) is correct in every ny-build conversion regime;
    // mirrors NN's post-rework `translate_reduce` (nn d7144ea7).
    let data_input = first_input(input_tensors, "Reduce")?;
    let input_rank = ctx
        .tensor_shapes
        .get(&data_input)
        .map(Vec::len)
        .ok_or_else(|| {
            NyError::InternalError(format!("Reduce: input shape not found for {data_input}"))
        })?;
    let axis_i64 = super::trailing_axis(dim, input_rank, "reduce axis")?;
    let mut attrs = HashMap::new();
    attrs.insert("axes".to_string(), AttributeValue::Ints(vec![axis_i64]));
    attrs.insert(
        "keepdims".to_string(),
        AttributeValue::Int(i64::from(keepdim)),
    );
    Ok(NodeOutput::one(simple_spec(
        name,
        layer_type,
        input_tensors.to_vec(),
        output_tensor,
        attrs,
    )))
}

// ---------------------------------------------------------------------------
// Linear / convolution translators
// ---------------------------------------------------------------------------

/// Translate `TraceOp::Linear`.
pub(super) fn translate_linear(
    name: &str,
    weight: &WeightPayload,
    bias: &Option<WeightPayload>,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let w_name = format!("{name}_weight");
    insert_payload(ctx, weight, &w_name, "Linear weight")?;
    if let Some(b) = bias {
        insert_payload(ctx, b, &format!("{name}_bias"), "Linear bias")?;
    }

    let mut attrs = HashMap::new();
    // Linear uses transB=1: weight is [out, in], matmul is input @ Wᵀ.
    attrs.insert("transB".to_string(), AttributeValue::Int(1));

    let mut spec_inputs = input_tensors.to_vec();
    spec_inputs.push(w_name.clone());
    if bias.is_some() {
        spec_inputs.push(format!("{name}_bias"));
    }

    Ok(NodeOutput::one(LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::Linear,
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

/// Translate `TraceOp::Conv1d`.
///
/// Dilation > 1 is expanded into a zero-interleaved kernel (matching NN), so the
/// emitted Conv1d uses effective dilation 1.
#[allow(clippy::too_many_arguments)]
pub(super) fn translate_conv1d(
    name: &str,
    weight: &WeightPayload,
    bias: &Option<WeightPayload>,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let w_name = format!("{name}_weight");

    let (eff_w_shape, eff_dilation) = if dilation > 1 {
        let data = weight_f32(weight, "Conv1d weight")?;
        let kernel = ArrayD::from_shape_vec(IxDyn(&weight.shape), data)
            .map_err(|e| NyError::ModelLoad(format!("Conv1d weight shape: {e}")))?;
        let expanded = expand_dilated_conv1d_kernel(&kernel, dilation)?;
        let expanded_shape = expanded.shape().to_vec();
        ctx.insert_weight(&w_name, expanded)?;
        (expanded_shape, 1)
    } else {
        insert_payload(ctx, weight, &w_name, "Conv1d weight")?;
        (weight.shape.clone(), dilation)
    };

    if let Some(b) = bias {
        insert_payload(ctx, b, &format!("{name}_bias"), "Conv1d bias")?;
    }

    let mut attrs = HashMap::new();
    attrs.insert(
        "strides".to_string(),
        AttributeValue::Ints(vec![dim_as_i64(stride, "Conv1d stride")?]),
    );
    let pad = dim_as_i64(padding, "Conv1d padding")?;
    attrs.insert("pads".to_string(), AttributeValue::Ints(vec![pad, pad]));
    attrs.insert(
        "dilations".to_string(),
        AttributeValue::Ints(vec![dim_as_i64(eff_dilation, "Conv1d dilation")?]),
    );
    attrs.insert(
        "group".to_string(),
        AttributeValue::Int(dim_as_i64(groups, "Conv1d groups")?),
    );

    // Conv expects inputs = [data, kernel_weight, optional_bias]; the kernel
    // comes from the WeightRef, not extra traced input nodes.
    let data_input = first_input(input_tensors, "Conv1d")?;
    let mut spec_inputs = vec![data_input, w_name.clone()];
    if bias.is_some() {
        spec_inputs.push(format!("{name}_bias"));
    }

    Ok(NodeOutput::one(LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::Conv1d,
        inputs: spec_inputs,
        outputs: vec![output_tensor.to_string()],
        weights: Some(WeightRef {
            name: w_name,
            shape: eff_w_shape,
            original_dtype: DataType::Float32,
        }),
        attributes: attrs,
    }))
}

/// Translate `TraceOp::Conv2d`.
#[allow(clippy::too_many_arguments)]
pub(super) fn translate_conv2d(
    name: &str,
    weight: &WeightPayload,
    bias: &Option<WeightPayload>,
    padding: &[usize; 2],
    stride: &[usize; 2],
    dilation: &[usize; 2],
    groups: usize,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if dilation[0] != 1 || dilation[1] != 1 {
        return Err(NyError::UnsupportedOp(
            "Conv2d dilation != 1 not yet supported in trace translation".to_string(),
        ));
    }

    let w_name = format!("{name}_weight");
    insert_payload(ctx, weight, &w_name, "Conv2d weight")?;
    if let Some(b) = bias {
        insert_payload(ctx, b, &format!("{name}_bias"), "Conv2d bias")?;
    }

    let sh = dim_as_i64(stride[0], "Conv2d stride_h")?;
    let sw = dim_as_i64(stride[1], "Conv2d stride_w")?;
    let ph = dim_as_i64(padding[0], "Conv2d pad_h")?;
    let pw = dim_as_i64(padding[1], "Conv2d pad_w")?;
    let dh = dim_as_i64(dilation[0], "Conv2d dilation_h")?;
    let dw = dim_as_i64(dilation[1], "Conv2d dilation_w")?;

    let mut attrs = HashMap::new();
    attrs.insert("strides".to_string(), AttributeValue::Ints(vec![sh, sw]));
    attrs.insert(
        "pads".to_string(),
        AttributeValue::Ints(vec![ph, pw, ph, pw]),
    );
    attrs.insert("dilations".to_string(), AttributeValue::Ints(vec![dh, dw]));
    attrs.insert(
        "group".to_string(),
        AttributeValue::Int(dim_as_i64(groups, "Conv2d groups")?),
    );

    let mut spec_inputs = input_tensors.to_vec();
    spec_inputs.push(w_name.clone());
    if bias.is_some() {
        spec_inputs.push(format!("{name}_bias"));
    }

    Ok(NodeOutput::one(LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::Conv2d,
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

/// Expand a dilated Conv1d kernel `[out, in, k]` into an equivalent dense kernel
/// with zero-interleaved taps, so a dilation-1 convolution is exact.
fn expand_dilated_conv1d_kernel(kernel: &ArrayD<f32>, dilation: usize) -> Result<ArrayD<f32>> {
    if kernel.ndim() != 3 {
        return Err(NyError::UnsupportedOp(format!(
            "Conv1d dilation expansion expects rank-3 kernel, got rank {}",
            kernel.ndim()
        )));
    }
    if dilation == 0 {
        return Err(NyError::UnsupportedOp(
            "Conv1d dilation must be >= 1".to_string(),
        ));
    }
    let shape = kernel.shape();
    let (out_c, in_c, k) = (shape[0], shape[1], shape[2]);
    let taps_minus_one = k.checked_sub(1).ok_or_else(|| {
        NyError::ModelLoad("Conv1d dilation expansion requires a non-empty kernel".to_string())
    })?;
    let new_k = taps_minus_one
        .checked_mul(dilation)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| NyError::InternalError("Conv1d dilation expansion overflow".to_string()))?;
    let expanded_shape = [out_c, in_c, new_k];
    let mut expanded = materialize_filled_array(&expanded_shape, 0.0, "Conv1d dilation expansion")?;
    for o in 0..out_c {
        for i in 0..in_c {
            for tap in 0..k {
                expanded[[o, i, tap * dilation]] = kernel[[o, i, tap]];
            }
        }
    }
    Ok(expanded)
}

// ---------------------------------------------------------------------------
// Shape translators
// ---------------------------------------------------------------------------

pub(super) fn translate_reshape(
    name: &str,
    target_shape: &[usize],
    input_tensors: Vec<String>,
    output_tensor: &str,
) -> Result<NodeOutput> {
    let shape_i64 = shape_to_i64(target_shape, "Reshape target_shape")?;
    let mut attrs = HashMap::new();
    attrs.insert("shape".to_string(), AttributeValue::Ints(shape_i64));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Reshape,
        input_tensors,
        output_tensor,
        attrs,
    )))
}

pub(super) fn translate_transpose(
    name: &str,
    dim0: usize,
    dim1: usize,
    output_shape: &[usize],
    input_tensors: Vec<String>,
    output_tensor: &str,
) -> Result<NodeOutput> {
    let ndim = output_shape.len();
    if dim0 >= ndim || dim1 >= ndim {
        return Err(NyError::UnsupportedOp(format!(
            "Transpose: dim0={dim0} or dim1={dim1} exceeds output rank {ndim}"
        )));
    }
    let ndim_i64 = dim_as_i64(ndim, "Transpose ndim")?;
    let mut perm: Vec<i64> = (0..ndim_i64).collect();
    perm.swap(dim0, dim1);
    let mut attrs = HashMap::new();
    attrs.insert("perm".to_string(), AttributeValue::Ints(perm));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Transpose,
        input_tensors,
        output_tensor,
        attrs,
    )))
}

/// `axis` is trailing-relative negative w.r.t. the OUTPUT rank (ONNX
/// Unsqueeze semantics), pre-encoded by the caller via
/// [`super::trailing_axis`]. Mirrors NN's post-rework `translate_unsqueeze`.
pub(super) fn translate_unsqueeze(
    name: &str,
    axis: i64,
    input_tensors: Vec<String>,
    output_tensor: &str,
) -> Result<NodeOutput> {
    let axis_i64 = axis;
    let mut attrs = HashMap::new();
    attrs.insert("axes".to_string(), AttributeValue::Ints(vec![axis_i64]));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Unsqueeze,
        input_tensors,
        output_tensor,
        attrs,
    )))
}

/// `axis` is trailing-relative negative w.r.t. the INPUT rank (ONNX Squeeze
/// semantics), pre-encoded by the caller via [`super::trailing_axis`].
/// Mirrors NN's post-rework `translate_squeeze`.
pub(super) fn translate_squeeze(
    name: &str,
    axis: i64,
    input_tensors: Vec<String>,
    output_tensor: &str,
) -> Result<NodeOutput> {
    let axis_i64 = axis;
    let mut attrs = HashMap::new();
    attrs.insert("axes".to_string(), AttributeValue::Ints(vec![axis_i64]));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Squeeze,
        input_tensors,
        output_tensor,
        attrs,
    )))
}

pub(super) fn translate_permute(
    name: &str,
    axes: &[usize],
    input_tensors: Vec<String>,
    output_tensor: &str,
) -> Result<NodeOutput> {
    // Validate axes form a valid permutation (no out-of-bounds, no duplicates).
    let ndim = axes.len();
    if ndim == 0 {
        return Err(NyError::UnsupportedOp("Permute: empty axes".to_string()));
    }
    let mut seen = vec![false; ndim];
    for (i, &axis) in axes.iter().enumerate() {
        if axis >= ndim {
            return Err(NyError::UnsupportedOp(format!(
                "Permute: axis {axis} at position {i} exceeds rank {ndim}"
            )));
        }
        if seen[axis] {
            return Err(NyError::UnsupportedOp(format!(
                "Permute: duplicate axis {axis}"
            )));
        }
        seen[axis] = true;
    }
    let perm: Vec<i64> = axes
        .iter()
        .map(|&a| dim_as_i64(a, "Permute axis"))
        .collect::<Result<_>>()?;
    let mut attrs = HashMap::new();
    attrs.insert("perm".to_string(), AttributeValue::Ints(perm));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Transpose,
        input_tensors,
        output_tensor,
        attrs,
    )))
}

/// `axis` is trailing-relative negative (Cat preserves rank across all inputs
/// and the output), pre-encoded by the caller via [`super::trailing_axis`];
/// ny-build's Concat passes negative axes through and the runtime layer
/// resolves them against the actual rank. Mirrors NN's post-rework
/// `translate_cat`.
pub(super) fn translate_cat(
    name: &str,
    axis: i64,
    input_tensors: Vec<String>,
    output_tensor: &str,
) -> Result<NodeOutput> {
    let axis_i64 = axis;
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), AttributeValue::Int(axis_i64));
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::Concat,
        input_tensors,
        output_tensor,
        attrs,
    )))
}

// ---------------------------------------------------------------------------
// Normalization translators
// ---------------------------------------------------------------------------

pub(super) fn translate_layer_norm(
    name: &str,
    eps: f64,
    weight: &WeightPayload,
    bias: &WeightPayload,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let eps_f32 = validate_eps(eps, "LayerNorm")?;
    let activation = first_input(input_tensors, "LayerNorm")?;

    let w_name = format!("{name}_weight");
    let b_name = format!("{name}_bias");
    insert_payload(ctx, weight, &w_name, "LayerNorm weight")?;
    insert_payload(ctx, bias, &b_name, "LayerNorm bias")?;

    let mut attrs = HashMap::new();
    attrs.insert("epsilon".to_string(), AttributeValue::Float(eps_f32));
    let norm_shape = shape_to_i64(&weight.shape, "LayerNorm normalized_shape")?;
    attrs.insert(
        "normalized_shape".to_string(),
        AttributeValue::Ints(norm_shape),
    );

    Ok(NodeOutput::one(LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::LayerNorm,
        inputs: vec![activation, w_name.clone(), b_name],
        outputs: vec![output_tensor.to_string()],
        weights: Some(WeightRef {
            name: w_name,
            shape: weight.shape.clone(),
            original_dtype: DataType::Float32,
        }),
        attributes: attrs,
    }))
}

pub(super) fn translate_rms_norm(
    name: &str,
    eps: f64,
    weight: &WeightPayload,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let eps_f32 = validate_eps(eps, "RmsNorm")?;
    let activation = first_input(input_tensors, "RmsNorm")?;

    let w_name = format!("{name}_weight");
    insert_payload(ctx, weight, &w_name, "RmsNorm weight")?;

    let mut attrs = HashMap::new();
    attrs.insert("epsilon".to_string(), AttributeValue::Float(eps_f32));

    Ok(NodeOutput::one(LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::RMSNorm,
        inputs: vec![activation, w_name.clone()],
        outputs: vec![output_tensor.to_string()],
        weights: Some(WeightRef {
            name: w_name,
            shape: weight.shape.clone(),
            original_dtype: DataType::Float32,
        }),
        attributes: attrs,
    }))
}

pub(super) fn translate_instance_norm(
    name: &str,
    eps: f64,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let eps_f32 = validate_eps(eps, "InstanceNorm")?;
    let num_channels = match output_shape.len() {
        0 | 1 => {
            return Err(NyError::UnsupportedOp(
                "InstanceNorm: cannot infer num_channels from scalar/1D output shape".to_string(),
            ));
        }
        2 => output_shape[0],
        _ => output_shape[1],
    };

    // InstanceNorm expects inputs = [data, gamma, beta]; create synthetic
    // identity affine parameters (gamma=ones, beta=zeros).
    let gamma_name = format!("{name}_gamma");
    let beta_name = format!("{name}_beta");
    ctx.insert_weight(
        &gamma_name,
        materialize_filled_array(&[num_channels], 1.0, "InstanceNorm gamma")?,
    )?;
    ctx.insert_weight(
        &beta_name,
        materialize_filled_array(&[num_channels], 0.0, "InstanceNorm beta")?,
    )?;

    let data_input = first_input(input_tensors, "InstanceNorm")?;
    let mut attrs = HashMap::new();
    attrs.insert("epsilon".to_string(), AttributeValue::Float(eps_f32));
    if output_shape.len() == 2 {
        attrs.insert(
            ny_build::INTERNAL_CT_INSTANCE_NORM_ATTR.to_string(),
            AttributeValue::Int(1),
        );
    }
    Ok(NodeOutput::one(simple_spec(
        name,
        LayerType::InstanceNorm,
        vec![data_input, gamma_name, beta_name],
        output_tensor,
        attrs,
    )))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn translate_batch_norm(
    name: &str,
    eps: f64,
    weight: &WeightPayload,
    bias: &WeightPayload,
    running_mean: &WeightPayload,
    running_var: &WeightPayload,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let eps_f32 = validate_eps(eps, "BatchNorm")?;
    let data_input = first_input(input_tensors, "BatchNorm")?;

    let w_name = format!("{name}_weight");
    let b_name = format!("{name}_bias");
    let mean_name = format!("{name}_mean");
    let var_name = format!("{name}_var");
    insert_payload(ctx, weight, &w_name, "BatchNorm weight")?;
    insert_payload(ctx, bias, &b_name, "BatchNorm bias")?;
    insert_payload(ctx, running_mean, &mean_name, "BatchNorm running_mean")?;
    insert_payload(ctx, running_var, &var_name, "BatchNorm running_var")?;

    let mut attrs = HashMap::new();
    attrs.insert("epsilon".to_string(), AttributeValue::Float(eps_f32));

    Ok(NodeOutput::one(LayerSpec {
        name: name.to_string(),
        layer_type: LayerType::BatchNorm,
        inputs: vec![data_input, w_name.clone(), b_name, mean_name, var_name],
        outputs: vec![output_tensor.to_string()],
        weights: Some(WeightRef {
            name: w_name,
            shape: weight.shape.clone(),
            original_dtype: DataType::Float32,
        }),
        attributes: attrs,
    }))
}

/// Translate `TraceOp::GroupNorm` by decomposition (no GroupNorm LayerType in
/// NY): Reshape → InstanceNorm → Reshape → Mul(gamma) → Add(beta).
#[allow(clippy::too_many_arguments)]
pub(super) fn translate_group_norm(
    name: &str,
    num_groups: usize,
    eps: f64,
    weight: &WeightPayload,
    bias: &WeightPayload,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let eps_f32 = validate_eps(eps, "GroupNorm")?;
    if output_shape.len() < 2 {
        return Err(NyError::UnsupportedOp(
            "GroupNorm: output must be at least 2D".to_string(),
        ));
    }
    let num_channels = output_shape[1];
    if num_groups == 0 || !num_channels.is_multiple_of(num_groups) {
        return Err(NyError::UnsupportedOp(format!(
            "GroupNorm: num_channels ({num_channels}) not divisible by num_groups ({num_groups})"
        )));
    }
    let channels_per_group = num_channels / num_groups;
    let batch = output_shape[0];
    let spatial: usize = output_shape[2..]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| NyError::InternalError("GroupNorm spatial dims overflow".to_string()))?;

    let batch_x_groups = batch
        .checked_mul(num_groups)
        .ok_or_else(|| NyError::InternalError("GroupNorm batch*num_groups overflow".to_string()))?;
    let bg_i64 = dim_as_i64(batch_x_groups, "GroupNorm batch*groups")?;
    let cpg_i64 = dim_as_i64(channels_per_group, "GroupNorm channels_per_group")?;
    let sp_i64 = dim_as_i64(spatial, "GroupNorm spatial")?;

    let data_input = first_input(input_tensors, "GroupNorm")?;
    let mut specs = Vec::new();

    // Step 1: Reshape [N, C, *spatial] → [N*G, C/G, spatial_flat].
    let r1_name = format!("{name}_gn_reshape1");
    let r1_out = format!("{r1_name}_out");
    let r1_shape = vec![bg_i64, cpg_i64, sp_i64];
    let mut r1_attrs = HashMap::new();
    r1_attrs.insert("shape".to_string(), AttributeValue::Ints(r1_shape.clone()));
    ctx.tensor_shapes.insert(r1_out.clone(), r1_shape);
    specs.push(simple_spec(
        &r1_name,
        LayerType::Reshape,
        vec![data_input],
        &r1_out,
        r1_attrs,
    ));

    // Step 2: InstanceNorm with synthetic identity affine.
    let in_name = format!("{name}_gn_instnorm");
    let in_out = format!("{in_name}_out");
    let gamma_name = format!("{in_name}_gamma");
    let beta_name = format!("{in_name}_beta");
    ctx.insert_weight(
        &gamma_name,
        materialize_filled_array(
            &[channels_per_group],
            1.0,
            "GroupNorm synthetic InstanceNorm gamma",
        )?,
    )?;
    ctx.insert_weight(
        &beta_name,
        materialize_filled_array(
            &[channels_per_group],
            0.0,
            "GroupNorm synthetic InstanceNorm beta",
        )?,
    )?;
    let mut in_attrs = HashMap::new();
    in_attrs.insert("epsilon".to_string(), AttributeValue::Float(eps_f32));
    in_attrs.insert("num_channels".to_string(), AttributeValue::Int(cpg_i64));
    ctx.tensor_shapes
        .insert(in_out.clone(), vec![bg_i64, cpg_i64, sp_i64]);
    specs.push(simple_spec(
        &in_name,
        LayerType::InstanceNorm,
        vec![r1_out, gamma_name, beta_name],
        &in_out,
        in_attrs,
    ));

    // Step 3: Reshape back → [N, C, *spatial].
    let r2_name = format!("{name}_gn_reshape2");
    let r2_out = format!("{r2_name}_out");
    let r2_shape = shape_to_i64(output_shape, "GroupNorm reshape2")?;
    let mut r2_attrs = HashMap::new();
    r2_attrs.insert("shape".to_string(), AttributeValue::Ints(r2_shape.clone()));
    ctx.tensor_shapes.insert(r2_out.clone(), r2_shape);
    specs.push(simple_spec(
        &r2_name,
        LayerType::Reshape,
        vec![in_out],
        &r2_out,
        r2_attrs,
    ));

    // Step 4: Mul(gamma) broadcast [1, C, 1, ...].
    let mut affine_shape = vec![1usize; output_shape.len()];
    affine_shape[1] = num_channels;
    let gamma_data = weight_f32(weight, "GroupNorm weight")?;
    let gamma_nd = ArrayD::from_shape_vec(IxDyn(&affine_shape), gamma_data)
        .map_err(|e| NyError::ModelLoad(format!("GroupNorm gamma reshape: {e}")))?;
    let gamma_name = format!("{name}_gamma");
    ctx.insert_weight(&gamma_name, gamma_nd)?;
    let mul_name = format!("{name}_gn_mul");
    let mul_out = format!("{mul_name}_out");
    ctx.tensor_shapes.insert(
        mul_out.clone(),
        shape_to_i64(output_shape, "GroupNorm mul")?,
    );
    specs.push(simple_spec(
        &mul_name,
        LayerType::Mul,
        vec![r2_out, gamma_name],
        &mul_out,
        HashMap::new(),
    ));

    // Step 5: Add(beta) — final node uses the parent name.
    let beta_data = weight_f32(bias, "GroupNorm bias")?;
    let beta_nd = ArrayD::from_shape_vec(IxDyn(&affine_shape), beta_data)
        .map_err(|e| NyError::ModelLoad(format!("GroupNorm beta reshape: {e}")))?;
    let beta_name = format!("{name}_beta");
    ctx.insert_weight(&beta_name, beta_nd)?;
    specs.push(simple_spec(
        name,
        LayerType::Add,
        vec![mul_out, beta_name],
        output_tensor,
        HashMap::new(),
    ));

    Ok(NodeOutput { specs })
}

// ---------------------------------------------------------------------------
// Constant folding (mirrors NN's try_constant_fold_*)
// ---------------------------------------------------------------------------

/// Constant-fold a unary op whose sole input is a constant tensor.
pub(super) fn try_constant_fold_unary(
    op: &TraceOp,
    output_tensor: &str,
    input_tensors: &[String],
    ctx: &mut Ctx,
) -> Option<Result<NodeOutput>> {
    let input_name = input_tensors.first()?;
    if !ctx.constant_tensors.contains(input_name.as_str()) {
        return None;
    }
    let input_arr = ctx.weights.get(input_name)?.clone();

    let fold_fn: fn(f32) -> f32 = match op {
        TraceOp::Relu => |v| v.max(0.0),
        TraceOp::Sigmoid => |v| 1.0 / (1.0 + (-v).exp()),
        TraceOp::Tanh => f32::tanh,
        TraceOp::Exp => f32::exp,
        TraceOp::Silu => |v| v / (1.0 + (-v).exp()),
        TraceOp::Sqrt => f32::sqrt,
        TraceOp::Abs => f32::abs,
        TraceOp::Recip => |v| 1.0 / v,
        TraceOp::Log => f32::ln,
        TraceOp::Sin => f32::sin,
        TraceOp::Cos => f32::cos,
        TraceOp::Floor => f32::floor,
        // Round-half-to-even (banker's rounding), matching NN's
        // `f32::round_ties_even`. Spelled out for MSRV 1.75 (the std method is
        // stable only since 1.77).
        TraceOp::Round => round_ties_even,
        TraceOp::Neg => |v| -v,
        TraceOp::Softplus => |v| {
            if v > 0.0 {
                v + (-v).exp().ln_1p()
            } else {
                v.exp().ln_1p()
            }
        },
        _ => return None,
    };

    let result_arr = input_arr.mapv(fold_fn);
    if result_arr.iter().any(|v| !v.is_finite()) {
        return Some(Err(NyError::NumericalInstability(format!(
            "constant fold of {} produced a non-finite value",
            op_name(op)
        ))));
    }
    Some(
        ctx.insert_weight(output_tensor, result_arr)
            .map(|()| NodeOutput::none()),
    )
}

/// Constant-fold a binary op whose both inputs are constant tensors.
pub(super) fn try_constant_fold_binary(
    op: &TraceOp,
    output_tensor: &str,
    input_tensors: &[String],
    ctx: &mut Ctx,
) -> Option<Result<NodeOutput>> {
    if input_tensors.len() != 2 {
        return None;
    }
    let lhs_name = &input_tensors[0];
    let rhs_name = &input_tensors[1];
    if !ctx.constant_tensors.contains(lhs_name.as_str())
        || !ctx.constant_tensors.contains(rhs_name.as_str())
    {
        return None;
    }
    let lhs = ctx.weights.get(lhs_name)?.clone();
    let rhs = ctx.weights.get(rhs_name)?.clone();

    let fold_fn: fn(f32, f32) -> f32 = match op {
        TraceOp::Add => |a, b| a + b,
        TraceOp::Sub => |a, b| a - b,
        TraceOp::Mul => |a, b| a * b,
        TraceOp::Div => |a, b| a / b,
        TraceOp::Maximum => f32::max,
        TraceOp::Minimum => f32::min,
        _ => return None,
    };

    let result_arr = match (lhs.broadcast(rhs.shape()), rhs.broadcast(lhs.shape())) {
        _ if lhs.shape() == rhs.shape() => ndarray::Zip::from(&lhs)
            .and(&rhs)
            .map_collect(|&a, &b| fold_fn(a, b)),
        (Some(lhs_bc), _) => ndarray::Zip::from(&lhs_bc)
            .and(&rhs)
            .map_collect(|&a, &b| fold_fn(a, b)),
        (_, Some(rhs_bc)) => ndarray::Zip::from(&lhs)
            .and(&rhs_bc)
            .map_collect(|&a, &b| fold_fn(a, b)),
        _ => {
            return Some(Err(NyError::UnsupportedOp(format!(
                "constant fold of {}: shapes {:?} and {:?} not broadcast-compatible",
                op_name(op),
                lhs.shape(),
                rhs.shape()
            ))));
        }
    };

    if result_arr.iter().any(|v| !v.is_finite()) {
        return Some(Err(NyError::NumericalInstability(format!(
            "constant fold of {} produced a non-finite value",
            op_name(op)
        ))));
    }
    Some(
        ctx.insert_weight(output_tensor, result_arr)
            .map(|()| NodeOutput::none()),
    )
}

/// Constant-fold a MatMul whose both inputs are constant tensors.
///
/// ONNX/PyTorch matmul semantics: 1-D operands are promoted to 2-D, leading
/// batch dims broadcast, the last dim of lhs contracts with the second-to-last
/// of rhs.
pub(super) fn try_constant_fold_matmul(
    output_tensor: &str,
    input_tensors: &[String],
    ctx: &mut Ctx,
) -> Option<Result<NodeOutput>> {
    if input_tensors.len() != 2 {
        return None;
    }
    let lhs_name = &input_tensors[0];
    let rhs_name = &input_tensors[1];
    if !ctx.constant_tensors.contains(lhs_name.as_str())
        || !ctx.constant_tensors.contains(rhs_name.as_str())
    {
        return None;
    }
    let mut lhs = ctx.weights.get(lhs_name)?.clone();
    let mut rhs = ctx.weights.get(rhs_name)?.clone();

    if lhs.ndim() == 0 || rhs.ndim() == 0 {
        return Some(Err(NyError::UnsupportedOp(format!(
            "constant fold of MatMul requires rank >= 1 operands, got {}-D and {}-D",
            lhs.ndim(),
            rhs.ndim()
        ))));
    }

    let lhs_was_1d = lhs.ndim() == 1;
    let rhs_was_1d = rhs.ndim() == 1;
    if lhs_was_1d {
        let k = lhs.shape()[0];
        lhs = match lhs.into_shape_with_order(IxDyn(&[1, k])) {
            Ok(arr) => arr,
            Err(e) => {
                return Some(Err(NyError::InternalError(format!(
                    "MatMul fold reshape left 1-D operand: {e}"
                ))));
            }
        };
    }
    if rhs_was_1d {
        let k = rhs.shape()[0];
        rhs = match rhs.into_shape_with_order(IxDyn(&[k, 1])) {
            Ok(arr) => arr,
            Err(e) => {
                return Some(Err(NyError::InternalError(format!(
                    "MatMul fold reshape right 1-D operand: {e}"
                ))));
            }
        };
    }

    let lhs_shape = lhs.shape().to_vec();
    let rhs_shape = rhs.shape().to_vec();
    let lhs_batch = &lhs_shape[..lhs_shape.len() - 2];
    let rhs_batch = &rhs_shape[..rhs_shape.len() - 2];
    let max_batch_ndim = lhs_batch.len().max(rhs_batch.len());
    let mut out_batch = vec![1usize; max_batch_ndim];
    for i in 0..max_batch_ndim {
        let lhs_dim = if i < lhs_batch.len() {
            lhs_batch[lhs_batch.len() - 1 - i]
        } else {
            1
        };
        let rhs_dim = if i < rhs_batch.len() {
            rhs_batch[rhs_batch.len() - 1 - i]
        } else {
            1
        };
        if lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1 {
            out_batch[max_batch_ndim - 1 - i] = lhs_dim.max(rhs_dim);
        } else {
            return Some(Err(NyError::UnsupportedOp(format!(
                "constant fold of MatMul: batch dims {lhs_batch:?} and {rhs_batch:?} not broadcast-compatible"
            ))));
        }
    }

    let m = lhs_shape[lhs_shape.len() - 2];
    let k = lhs_shape[lhs_shape.len() - 1];
    let rhs_k = rhs_shape[rhs_shape.len() - 2];
    let n = rhs_shape[rhs_shape.len() - 1];
    if k != rhs_k {
        return Some(Err(NyError::UnsupportedOp(format!(
            "constant fold of MatMul: inner dim mismatch {k} vs {rhs_k}"
        ))));
    }

    let batch_size = match out_batch
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    {
        Some(size) => size,
        None => {
            return Some(Err(NyError::InternalError(format!(
                "MatMul fold batch dims overflow: {out_batch:?}"
            ))));
        }
    };

    let mut result_shape = out_batch.clone();
    result_shape.push(m);
    result_shape.push(n);
    let mut result =
        match materialize_filled_array(&result_shape, 0.0, "constant fold of MatMul output") {
            Ok(result) => result,
            Err(error) => return Some(Err(error)),
        };

    let lhs_offset = out_batch.len().saturating_sub(lhs_batch.len());
    let rhs_offset = out_batch.len().saturating_sub(rhs_batch.len());
    let mut batch_index = vec![0usize; out_batch.len()];
    for linear_idx in 0..batch_size {
        let mut remainder = linear_idx;
        for axis in (0..out_batch.len()).rev() {
            let dim = out_batch[axis];
            batch_index[axis] = remainder % dim;
            remainder /= dim;
        }

        let mut lhs_view = lhs.view().into_dyn();
        for (axis, &dim) in lhs_batch.iter().enumerate() {
            let out_axis = lhs_offset + axis;
            let idx = if dim == 1 { 0 } else { batch_index[out_axis] };
            lhs_view = lhs_view.index_axis_move(Axis(0), idx);
        }
        let lhs_2d = match lhs_view.into_dimensionality::<ndarray::Ix2>() {
            Ok(arr) => arr,
            Err(e) => {
                return Some(Err(NyError::InternalError(format!(
                    "MatMul fold left batch view to 2-D: {e}"
                ))));
            }
        };

        let mut rhs_view = rhs.view().into_dyn();
        for (axis, &dim) in rhs_batch.iter().enumerate() {
            let out_axis = rhs_offset + axis;
            let idx = if dim == 1 { 0 } else { batch_index[out_axis] };
            rhs_view = rhs_view.index_axis_move(Axis(0), idx);
        }
        let rhs_2d = match rhs_view.into_dimensionality::<ndarray::Ix2>() {
            Ok(arr) => arr,
            Err(e) => {
                return Some(Err(NyError::InternalError(format!(
                    "MatMul fold right batch view to 2-D: {e}"
                ))));
            }
        };

        let product = lhs_2d.dot(&rhs_2d);

        let mut out_view = result.view_mut().into_dyn();
        for &idx in &batch_index {
            out_view = out_view.index_axis_move(Axis(0), idx);
        }
        let mut out_2d = match out_view.into_dimensionality::<ndarray::Ix2>() {
            Ok(arr) => arr,
            Err(e) => {
                return Some(Err(NyError::InternalError(format!(
                    "MatMul fold output batch view to 2-D: {e}"
                ))));
            }
        };
        out_2d.assign(&product);
    }

    if lhs_was_1d {
        result = result.index_axis_move(Axis(out_batch.len()), 0);
    }
    if rhs_was_1d {
        let last_axis = result.ndim() - 1;
        result = result.index_axis_move(Axis(last_axis), 0);
    }

    if result.iter().any(|v| !v.is_finite()) {
        return Some(Err(NyError::NumericalInstability(
            "constant fold of MatMul produced a non-finite value".to_string(),
        )));
    }
    Some(
        ctx.insert_weight(output_tensor, result)
            .map(|()| NodeOutput::none()),
    )
}

/// Round-half-to-even (banker's rounding), matching `f32::round_ties_even`.
///
/// Spelled out so the crate compiles on the declared MSRV (1.75); the std
/// method stabilized in 1.77.
fn round_ties_even(v: f32) -> f32 {
    let rounded = v.round(); // ties away from zero
    if (v - v.floor() - 0.5).abs() < f32::EPSILON {
        // Exact half: pick the even neighbor.
        let down = v.floor();
        let up = v.ceil();
        if (down as i64) % 2 == 0 {
            down
        } else {
            up
        }
    } else {
        rounded
    }
}
