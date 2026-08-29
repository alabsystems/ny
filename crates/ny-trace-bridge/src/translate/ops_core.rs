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
    shape_to_i64, simple_spec, validate_constant_variance_eps, validate_eps, weight_f32, Ctx,
    NodeOutput,
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

/// ONNX `Cast` target-dtype code for `INT32`, as carried on the guarded
/// `LayerType::Trunc` lowering's `to` attribute.
///
/// `ny_build::convert::elementwise` reads exactly this attribute to select
/// `TruncLayer::for_int32_cast()`, and refuses every other `to` value — so
/// these two constants are the complete set of integer cast targets the
/// bridge may emit.
const ONNX_CAST_TARGET_INT32: i64 = 6;

/// ONNX `Cast` target-dtype code for `INT64` (see [`ONNX_CAST_TARGET_INT32`]).
const ONNX_CAST_TARGET_INT64: i64 = 7;

/// The ONNX `Cast` `to` code for a target dtype that has a MODELED guarded
/// `Trunc` lowering, or `None` when the target must stay refused.
///
/// Only `INT32` and `INT64` qualify. `ny-build`'s converter rejects any other
/// `to` value outright, and there is no corresponding domain-guarded
/// `TruncLayer` constructor for the narrow/unsigned integer types — where a
/// wrong enclosure would be a wrong `unsat` — so `U32`/`U8`/`Bool` and every
/// float target stay refused.
///
/// Shared with the graph-wide dtype gate
/// (`super::is_guarded_integer_cast`) so the set of admitted cast targets has
/// exactly one definition.
pub(super) fn guarded_integer_cast_target(target_dtype: DType) -> Option<i64> {
    match target_dtype {
        DType::I32 => Some(ONNX_CAST_TARGET_INT32),
        DType::I64 => Some(ONNX_CAST_TARGET_INT64),
        _ => None,
    }
}

/// Translate `TraceOp::ToDtype`.
///
/// The cast node records only its target dtype, but the SOURCE dtype is
/// recoverable from the wire format: the cast's first input tensor is produced
/// by a node whose recorded `output_dtype` is the source dtype
/// (`ctx.tensor_dtypes`, populated alongside `tensor_shapes`). Three resolved
/// families lower; every other combination is refused, because the rounding
/// semantics of the non-F32 endpoint are not modeled and an identity lowering
/// would not enclose the true cast. This check is deliberately independent of
/// the graph-wide F32 gate (which already refuses any reachable non-F32 node)
/// so a later relaxation of that gate cannot silently turn a lossy cast into an
/// identity. An unresolvable producer (dangling input) keeps the historic
/// fail-closed refusal rather than guessing a source dtype.
///
/// # The three lowerable families
///
/// 1. `F32 -> F32` — a cast between identical dtypes changes no value, emitted
///    with the same `Add(x, 0)` idiom as Dropout.
/// 2. `U8`/`Bool -> F32` — a LOSSLESS integer embedding (see the inline note).
/// 3. `F32 -> I32`/`I64` — the guarded `Trunc` lowering ny-onnx already uses
///    for the IDENTICAL ONNX `Cast`, argued below.
///
/// # Why `F32 -> I32`/`I64` is a guarded `Trunc`, and why that is sound
///
/// `ny_build::convert::elementwise` lowers `LayerType::Trunc` carrying
/// `to = 6` (INT32) / `to = 7` (INT64) to `TruncLayer::for_int32_cast()` /
/// `for_int64_cast()`: round-toward-zero, plus a PROOF OBLIGATION discharged at
/// every verdict-bearing IBP / CROWN / Patches / exact-f64-cell boundary that
/// the whole pre-activation interval is finite and inside `[-2^31, 2^31)` /
/// `[-2^63, 2^63)`. Out of range the layer raises `SoundnessRefusal` rather
/// than choosing a value — ONNX leaves float→fixed-point Cast undefined there
/// and ORT's answer (wraparound) is NOT `trunc`.
///
/// Pointwise agreement with the producer. nn's `nn-core`
/// `dyn_tensor/selection/dtype_convert.rs::f32_to_i64` is
/// `arr.mapv(|x| x as i64)` — Rust's `as` on floats IS round-toward-zero —
/// after rejecting non-finite values and anything outside
/// `[-9_223_372_000_000_000_000.0, 9_223_371_500_000_000_000.0]`. Those two
/// f32 literals are exactly `-2^63` and `next_down(2^63)`, i.e. nn's accepted
/// domain and the guarded layer's admitted domain are the SAME closed set of
/// f32 values. So wherever nn does not itself error, `trunc` is its answer.
/// (nn's `to_dtype` has no `(F32, I32)` arm at all — it errors — so the INT32
/// arm cannot arise from an nn trace; it is emitted for other producers of the
/// wire format under the ONNX semantics `ny-build` models, where the same
/// in-range/finite guard makes `trunc` the only admitted behaviour.)
///
/// Exactness of the F32 mirror. The emitted graph carries the result as f32.
/// `trunc(x)` of an f32 IS an f32 (for `|x| >= 2^23` the value is already
/// integral), and inside the guarded domain the mathematical integer
/// `x as i64` equals that same f32 exactly. So the mirror introduces NO
/// rounding: downstream layers see precisely the integer nn computed. `Trunc`'s
/// own IBP is `[trunc(l), trunc(u)]` — the exact interval image of a monotone
/// step — so the enclosure is exact, not merely sound, and strictly TIGHTER
/// than the refusal it replaces without weakening anything: the domain guard
/// keeps every previously-unrepresentable case a hard refusal.
///
/// Wider fan-out is the graph gate's job, not this arm's: see
/// `super::is_guarded_integer_cast` for why an integer-declared cast output may
/// pass the F32 contract at all.
pub(super) fn translate_to_dtype(
    name: &str,
    target_dtype: DType,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "ToDtype")?;
    let Some(&source_dtype) = ctx.tensor_dtypes.get(&data_input) else {
        return Err(NyError::UnsupportedOp(format!(
            "ToDtype target {target_dtype:?} cannot be soundly lowered because the trace does not record the source dtype"
        )));
    };
    if source_dtype == DType::F32 && target_dtype == DType::F32 {
        return translate_identity_add_zero(name, "ToDtype", input_tensors, output_tensor, ctx);
    }
    // F32 -> INT32/INT64: the guarded round-toward-zero `Trunc` (see the doc
    // comment for the pointwise-agreement and exact-mirror arguments). Every
    // other integer target stays refused — `ny-build` rejects any other `to`
    // value and no domain-guarded layer models U32/U8/Bool.
    if source_dtype == DType::F32 {
        if let Some(to) = guarded_integer_cast_target(target_dtype) {
            let mut attrs = HashMap::new();
            attrs.insert("to".to_string(), AttributeValue::Int(to));
            return Ok(NodeOutput::one(simple_spec(
                name,
                LayerType::Trunc,
                vec![data_input],
                output_tensor,
                attrs,
            )));
        }
    }
    // U8/Bool -> F32 is a LOSSLESS integer embedding: every value of the
    // source domain (0..=255 / {0, 1}) is exactly representable in F32, so
    // there is no rounding to model and the cast is an identity on the value
    // set. The graph-wide dtype gate only admits U8/Bool producers whose
    // lowering already models the values as F32 exactly (Compare masks,
    // value-independent index leaves), so the identity composes soundly.
    // Wider integer sources (U32/I32/I64) stay refused: beyond 2^24 the
    // embedding rounds.
    if matches!(source_dtype, DType::U8 | DType::Bool) && target_dtype == DType::F32 {
        return translate_identity_add_zero(name, "ToDtype", input_tensors, output_tensor, ctx);
    }
    let reason = if source_dtype == DType::F32 {
        format!(
            "rounding semantics of the {target_dtype:?} target are not modeled (only I32/I64 \
             have the guarded round-toward-zero Trunc lowering)"
        )
    } else {
        format!("rounding semantics of the {source_dtype:?} source are not modeled")
    };
    Err(NyError::UnsupportedOp(format!(
        "ToDtype {source_dtype:?} -> {target_dtype:?} is refused: {reason}"
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
    // BatchNorm is the ONE normalization whose variance is a translate-time
    // constant, so it takes the constant-variance gate (finite eps >= 0 plus
    // per-channel `var + eps > 0`) rather than the data-variance floor. See
    // `validate_constant_variance_eps` for why that is the weakest sound rule
    // here — and why the data-variance family cannot have it.
    let running_var_data = weight_f32(running_var, "BatchNorm running_var")?;
    let eps_f32 = validate_constant_variance_eps(eps, &running_var_data, "BatchNorm")?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A translation context whose sole producer tensor `p_out` carries the
    /// given recorded dtype (`None` leaves the dtype lookup empty, modeling an
    /// unresolvable producer).
    fn ctx_with_source(dtype: Option<DType>) -> Ctx {
        let mut ctx = Ctx::new(false);
        if let Some(source_dtype) = dtype {
            ctx.tensor_dtypes.insert("p_out".to_string(), source_dtype);
        }
        ctx
    }

    /// Unwrap a refusal (`expect_err` needs `NodeOutput: Debug`).
    fn unwrap_refusal(result: Result<NodeOutput>, what: &str) -> NyError {
        match result {
            Ok(_) => panic!("{what}: expected a refusal, got a lowering"),
            Err(err) => err,
        }
    }

    /// F32 -> F32 with an F32-recorded producer is the one lowerable cast:
    /// the same `Add(x, 0)` identity idiom as Dropout.
    #[test]
    fn to_dtype_f32_identity_lowers_as_add_zero() {
        let mut ctx = ctx_with_source(Some(DType::F32));
        let output = translate_to_dtype(
            "layer0_trace_1",
            DType::F32,
            &["p_out".to_string()],
            "layer0_trace_1_out",
            &mut ctx,
        )
        .expect("F32 -> F32 lowers as identity");
        assert_eq!(output.specs.len(), 1, "one identity Add spec");
        let spec = &output.specs[0];
        assert_eq!(spec.layer_type, LayerType::Add);
        assert_eq!(spec.inputs[0], "p_out");
        assert_eq!(spec.outputs, vec!["layer0_trace_1_out".to_string()]);
        assert!(
            ctx.constant_tensors.contains("layer0_trace_1_zero"),
            "identity Add registers its +0 constant"
        );
    }

    /// A resolved non-identity cast is refused naming BOTH dtypes and the
    /// offending endpoint — never blamed on a missing source dtype. The
    /// non-F32-source arm is unreachable through `translate` today (the
    /// graph-wide F32 gate refuses the producer first) but must stay
    /// fail-closed on its own.
    #[test]
    fn to_dtype_resolved_non_identity_names_both_dtypes() {
        let mut ctx = ctx_with_source(Some(DType::F16));
        let err = unwrap_refusal(
            translate_to_dtype(
                "cast",
                DType::F32,
                &["p_out".to_string()],
                "cast_out",
                &mut ctx,
            ),
            "F16 -> F32",
        );
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("ToDtype F16 -> F32 is refused")
                    && m.contains("F16 source")),
            "got {err:?}"
        );

        let mut ctx = ctx_with_source(Some(DType::F32));
        let err = unwrap_refusal(
            translate_to_dtype(
                "cast",
                DType::U32,
                &["p_out".to_string()],
                "cast_out",
                &mut ctx,
            ),
            "F32 -> U32",
        );
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("ToDtype F32 -> U32 is refused")
                    && m.contains("U32 target")),
            "got {err:?}"
        );
    }

    /// `F32 -> I32`/`I64` lowers to the guarded round-toward-zero `Trunc`,
    /// carrying the ONNX Cast target code `ny-build`'s converter reads to
    /// select `TruncLayer::for_int32_cast()` / `for_int64_cast()` — the same
    /// layer ny-onnx emits for the identical ONNX `Cast`.
    #[test]
    fn to_dtype_f32_to_int_lowers_to_guarded_trunc() {
        for (target, expected_to) in [(DType::I32, 6_i64), (DType::I64, 7_i64)] {
            let mut ctx = ctx_with_source(Some(DType::F32));
            let output = translate_to_dtype(
                "layer0_trace_1",
                target,
                &["p_out".to_string()],
                "layer0_trace_1_out",
                &mut ctx,
            )
            .unwrap_or_else(|e| panic!("F32 -> {target:?} must lower to a guarded Trunc: {e:?}"));
            assert_eq!(output.specs.len(), 1, "{target:?}: one Trunc spec");
            let spec = &output.specs[0];
            assert_eq!(spec.layer_type, LayerType::Trunc, "{target:?}");
            assert_eq!(spec.inputs, vec!["p_out".to_string()], "{target:?}");
            assert_eq!(
                spec.outputs,
                vec!["layer0_trace_1_out".to_string()],
                "{target:?}"
            );
            assert_eq!(
                spec.attributes.get("to"),
                Some(&AttributeValue::Int(expected_to)),
                "{target:?}: the Cast target attribute selects the GUARDED TruncLayer; a plain \
                 Trunc (no `to`) would drop the finite/in-range proof obligation"
            );
        }
    }

    /// The integer-cast relaxation is exactly two targets wide. Every other
    /// F32 target keeps the standing refusal: `ny-build` rejects any other
    /// `to` value and no domain-guarded layer models U32/U8/Bool, where a
    /// wrong enclosure would be a wrong `unsat`.
    #[test]
    fn to_dtype_f32_to_other_targets_stay_refused() {
        for target in [
            DType::U32,
            DType::U8,
            DType::Bool,
            DType::F16,
            DType::Bf16,
            DType::F64,
        ] {
            let mut ctx = ctx_with_source(Some(DType::F32));
            let err = unwrap_refusal(
                translate_to_dtype("cast", target, &["p_out".to_string()], "cast_out", &mut ctx),
                &format!("F32 -> {target:?}"),
            );
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m)
                    if m.contains(&format!("ToDtype F32 -> {target:?} is refused"))
                        && m.contains("only I32/I64")),
                "{target:?}: got {err:?}"
            );
        }
    }

    /// The `Trunc` arm is keyed on the RESOLVED source, not just the target:
    /// a non-F32 source keeps the standing refusal even for an I32/I64 target
    /// (a `F16 -> I64` cast rounds twice, and neither rounding is modeled).
    #[test]
    fn to_dtype_non_f32_source_to_int_stays_refused() {
        for source in [DType::F16, DType::Bf16, DType::F64, DType::U8, DType::I64] {
            for target in [DType::I32, DType::I64] {
                let mut ctx = ctx_with_source(Some(source));
                let err = unwrap_refusal(
                    translate_to_dtype(
                        "cast",
                        target,
                        &["p_out".to_string()],
                        "cast_out",
                        &mut ctx,
                    ),
                    &format!("{source:?} -> {target:?}"),
                );
                assert!(
                    matches!(err, NyError::UnsupportedOp(ref m)
                        if m.contains(&format!("ToDtype {source:?} -> {target:?} is refused"))
                            && m.contains(&format!("{source:?} source"))),
                    "{source:?} -> {target:?}: got {err:?}"
                );
            }
        }
    }

    /// An unresolvable producer fails closed for an integer target too — the
    /// guarded `Trunc` is only correct when the source really is F32, and the
    /// arm must never guess that.
    #[test]
    fn to_dtype_unresolvable_source_to_int_fails_closed() {
        for target in [DType::I32, DType::I64] {
            let mut ctx = ctx_with_source(None);
            let err = unwrap_refusal(
                translate_to_dtype("cast", target, &["p_out".to_string()], "cast_out", &mut ctx),
                &format!("unresolvable producer -> {target:?}"),
            );
            assert!(
                matches!(err, NyError::UnsupportedOp(ref m)
                    if m.contains("does not record the source dtype")),
                "{target:?}: got {err:?}"
            );
        }
    }

    /// The admitted-target table has exactly one definition, shared with the
    /// graph-wide dtype gate, so the emitter and the gate cannot drift apart.
    #[test]
    fn guarded_integer_cast_target_table_is_exactly_int32_and_int64() {
        assert_eq!(guarded_integer_cast_target(DType::I32), Some(6));
        assert_eq!(guarded_integer_cast_target(DType::I64), Some(7));
        for other in [
            DType::F32,
            DType::F16,
            DType::Bf16,
            DType::F64,
            DType::U32,
            DType::U8,
            DType::Bool,
        ] {
            assert_eq!(
                guarded_integer_cast_target(other),
                None,
                "{other:?} has no modeled guarded lowering"
            );
        }
    }

    /// With no recorded producer dtype the cast keeps the historic
    /// fail-closed refusal — even for an F32 target.
    #[test]
    fn to_dtype_unresolvable_source_keeps_fail_closed_refusal() {
        let mut ctx = ctx_with_source(None);
        let err = unwrap_refusal(
            translate_to_dtype(
                "cast",
                DType::F32,
                &["p_out".to_string()],
                "cast_out",
                &mut ctx,
            ),
            "unresolvable producer",
        );
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m)
                if m.contains("does not record the source dtype")),
            "got {err:?}"
        );
    }
}
