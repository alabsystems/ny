// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Kokoro fused-op family: `KokoroFused` in all 5 kinds — `SnakeTensor`,
//! `AdainSnake`, `AdainLeakyRelu`, `AdaLayerNorm`, `FusedAdainResBlock`.
//!
//! Ported from NN's `trace_to_graph_layerspec_snake.rs`,
//! `trace_to_graph_layerspec_decompose_adalayernorm.rs`, and
//! `trace_to_graph_layerspec_decompose_resblock{,_helpers}.rs`, preserving
//! NN's emission exactly:
//!
//! - `SnakeTensor` → one native `LayerType::Snake` with per-channel alpha as a
//!   weight tensor in `inputs[1]` (gc#4117).
//! - `AdainSnake` / `AdainLeakyRelu`: constant gamma/beta → native
//!   `LayerType::AdaIN` with precomputed style weights; variable gamma/beta →
//!   decomposed InstanceNorm + Reshape + Mul + Add (#2987). Kokoro's
//!   `style_gamma = gamma + 1.0` residual convention is preserved on both
//!   paths. The trailing activation is a native Snake or a decomposed
//!   LeakyReLU.
//! - `AdaLayerNorm` → LayerNorm + Add + Mul + Add
//!   (`(1 + gamma) * LayerNorm(x, w, b) + beta`, #2547).
//! - `FusedAdainResBlock` → ~20-30 primitive LayerSpecs: two
//!   `(Linear(style) → split → InstanceNorm → affine → activation → Conv1d)`
//!   phases plus the (optionally scaled) residual (#2547).
//!
//! LeakyReLU is always decomposed to `α·x + (1−α)·ReLU(x)` for tight CROWN
//! bounds (#2977) — native `LeakyReLULayer` returns IBP-wide CROWN bounds.

use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use ny_build::{AttributeValue, DataType, LayerSpec, WeightRef};
use ny_core::{LayerType, NyError, Result};

use crate::schema::{KokoroFusedOp, ResBlockActivation, TraceNode, TraceOp, WeightPayload};

use super::{
    checked_f64_to_f32, dim_as_i64, first_input, insert_payload, insert_scalar_constant, op_name,
    simple_spec, weight_f32, Ctx, NodeOutput,
};

/// Translate a `KokoroFused` op (kinds `SnakeTensor`, `AdainSnake`,
/// `AdainLeakyRelu`, `AdaLayerNorm`, `FusedAdainResBlock`) node.
pub(super) fn translate_kokoro(
    node: &TraceNode,
    name: &str,
    input_tensors: &[String],
    output_tensor: &str,
    _node_names: &HashMap<u64, String>,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let TraceOp::KokoroFused(kind) = &node.op else {
        return Err(NyError::InternalError(format!(
            "translate_kokoro dispatched a non-KokoroFused op ({})",
            op_name(&node.op)
        )));
    };
    let output_shape = node.output_shape.as_slice();
    match kind {
        KokoroFusedOp::SnakeTensor { alpha } => {
            translate_snake_tensor(name, alpha, input_tensors, output_tensor, ctx)
        }
        KokoroFusedOp::AdainSnake { alpha, eps } => translate_adain_snake(
            name,
            alpha,
            *eps,
            input_tensors,
            output_tensor,
            output_shape,
            ctx,
        ),
        KokoroFusedOp::AdainLeakyRelu { eps, slope } => translate_adain_leaky_relu(
            name,
            *eps,
            *slope,
            input_tensors,
            output_tensor,
            output_shape,
            ctx,
        ),
        KokoroFusedOp::AdaLayerNorm {
            norm_weight,
            norm_bias,
            eps,
        } => translate_ada_layer_norm(
            name,
            norm_weight,
            norm_bias,
            *eps,
            input_tensors,
            output_tensor,
            ctx,
        ),
        KokoroFusedOp::FusedAdainResBlock {
            activation,
            adain1_weight,
            adain1_bias,
            adain2_weight,
            adain2_bias,
            conv1_weight,
            conv1_bias,
            conv1_dilation,
            conv1_padding,
            conv2_weight,
            conv2_bias,
            conv2_padding,
            eps,
            residual_scale,
        } => translate_fused_adain_resblock(
            name,
            activation,
            adain1_weight,
            adain1_bias,
            adain2_weight,
            adain2_bias,
            conv1_weight,
            conv1_bias,
            *conv1_dilation,
            *conv1_padding,
            conv2_weight,
            conv2_bias,
            *conv2_padding,
            *eps,
            *residual_scale,
            input_tensors,
            output_tensor,
            output_shape,
            ctx,
        ),
    }
}

// ---------------------------------------------------------------------------
// Per-kind translators
// ---------------------------------------------------------------------------

/// Translate `KokoroFusedOp::SnakeTensor { alpha }`.
///
/// Always emits a single native `LayerType::Snake` with alpha as a weight
/// tensor in `inputs[1]`. NY handles both scalar and per-channel alpha
/// natively via `SnakeLayer::per_channel()` (gc#4117). Mirrors NN's
/// `translate_snake_tensor`.
fn translate_snake_tensor(
    name: &str,
    alpha: &WeightPayload,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    let data_input = first_input(input_tensors, "SnakeTensor")?;
    validate_snake_alpha(alpha, "SnakeTensor")?;
    let spec = emit_native_snake(name, alpha, &data_input, output_tensor, "SnakeTensor", ctx)?;
    Ok(NodeOutput::one(spec))
}

/// Translate `KokoroFusedOp::AdainSnake { alpha, eps }`.
///
/// Constant gamma/beta → native AdaIN + native Snake (2 layers).
/// Variable gamma/beta → decomposed InstanceNorm + Mul + Add + native Snake.
/// Mirrors NN's `translate_adain_snake`.
fn translate_adain_snake(
    name: &str,
    alpha: &WeightPayload,
    eps: f64,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() < 3 {
        return Err(NyError::UnsupportedOp(format!(
            "AdainSnake requires 3 inputs (x, gamma, beta), got {}",
            input_tensors.len()
        )));
    }
    let x_input = input_tensors[0].clone();
    let gamma_input = input_tensors[1].clone();
    let beta_input = input_tensors[2].clone();

    let eps_f32 = validate_adain_eps(eps, "AdainSnake")?;
    validate_snake_alpha(alpha, "AdainSnake")?;

    // Constant gamma/beta → native AdaIN + native Snake.
    if let Some((adain_spec, adain_out)) =
        try_native_adain(name, &x_input, &gamma_input, &beta_input, eps_f32, ctx)?
    {
        let snake_spec =
            emit_native_snake(name, alpha, &adain_out, output_tensor, "AdainSnake", ctx)?;
        return Ok(NodeOutput {
            specs: vec![adain_spec, snake_spec],
        });
    }

    // Variable gamma/beta → decomposed InstanceNorm + Mul + Add + native Snake.
    let num_channels = infer_num_channels(output_shape, "AdainSnake")?;
    let (mut specs, adain_out) = emit_variable_adain(
        name,
        &x_input,
        &gamma_input,
        &beta_input,
        eps_f32,
        num_channels,
        ctx,
    )?;
    let snake_spec = emit_native_snake(name, alpha, &adain_out, output_tensor, "AdainSnake", ctx)?;
    specs.push(snake_spec);
    Ok(NodeOutput { specs })
}

/// Translate `KokoroFusedOp::AdainLeakyRelu { eps, slope }`.
///
/// Constant gamma/beta → native AdaIN + decomposed LeakyRelu.
/// Variable gamma/beta → decomposed InstanceNorm + Mul + Add + decomposed
/// LeakyRelu. Mirrors NN's `translate_adain_leaky_relu`.
fn translate_adain_leaky_relu(
    name: &str,
    eps: f64,
    slope: f64,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() < 3 {
        return Err(NyError::UnsupportedOp(format!(
            "AdainLeakyRelu requires 3 inputs (x, gamma, beta), got {}",
            input_tensors.len()
        )));
    }
    let x_input = input_tensors[0].clone();
    let gamma_input = input_tensors[1].clone();
    let beta_input = input_tensors[2].clone();

    let eps_f32 = validate_adain_eps(eps, "AdainLeakyRelu")?;
    let slope_f32 = slope as f32;
    if !slope_f32.is_finite() {
        return Err(NyError::NumericalInstability(format!(
            "AdainLeakyRelu: slope must be finite, got {slope}"
        )));
    }

    // Constant gamma/beta → native AdaIN + decomposed LeakyRelu.
    if let Some((adain_spec, adain_out)) =
        try_native_adain(name, &x_input, &gamma_input, &beta_input, eps_f32, ctx)?
    {
        let mut specs = vec![adain_spec];
        emit_leaky_relu_specs(name, slope_f32, &adain_out, output_tensor, ctx, &mut specs)?;
        return Ok(NodeOutput { specs });
    }

    // Variable gamma/beta → decomposed InstanceNorm + Mul + Add + LeakyRelu.
    let num_channels = infer_num_channels(output_shape, "AdainLeakyRelu")?;
    let (mut specs, adain_out) = emit_variable_adain(
        name,
        &x_input,
        &gamma_input,
        &beta_input,
        eps_f32,
        num_channels,
        ctx,
    )?;
    emit_leaky_relu_specs(name, slope_f32, &adain_out, output_tensor, ctx, &mut specs)?;
    Ok(NodeOutput { specs })
}

/// Translate `KokoroFusedOp::AdaLayerNorm { norm_weight, norm_bias, eps }`.
///
/// Math: `(1 + gamma) * LayerNorm(x, weight, bias, eps) + beta`.
/// Tensor inputs `[x, gamma, beta]`, decomposed into 4 LayerSpecs. Mirrors
/// NN's `translate_ada_layer_norm`.
fn translate_ada_layer_norm(
    name: &str,
    norm_weight: &WeightPayload,
    norm_bias: &WeightPayload,
    eps: f64,
    input_tensors: &[String],
    output_tensor: &str,
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() < 3 {
        return Err(NyError::UnsupportedOp(format!(
            "AdaLayerNorm requires 3 inputs (x, gamma, beta), got {}",
            input_tensors.len()
        )));
    }
    let x_input = input_tensors[0].clone();
    let gamma_input = input_tensors[1].clone();
    let beta_input = input_tensors[2].clone();

    let eps_f32 = checked_f64_to_f32(eps, "AdaLayerNorm eps")?;
    if eps_f32 < 0.0 {
        return Err(NyError::UnsupportedOp(format!(
            "AdaLayerNorm eps must be non-negative, got {eps}"
        )));
    }

    let weight_name = format!("{name}_norm_weight");
    let bias_name = format!("{name}_norm_bias");
    insert_payload(ctx, norm_weight, &weight_name, "AdaLayerNorm norm_weight")?;
    insert_payload(ctx, norm_bias, &bias_name, "AdaLayerNorm norm_bias")?;

    let ones_const = format!("{name}_ones");
    insert_scalar_constant(ctx, &ones_const, 1.0)?;

    let normed = format!("{name}_normed");
    let scale = format!("{name}_scale");
    let scaled = format!("{name}_scaled");

    let mut eps_attrs = HashMap::new();
    eps_attrs.insert("epsilon".to_string(), AttributeValue::Float(eps_f32));

    let specs = vec![
        simple_spec(
            &normed,
            LayerType::LayerNorm,
            vec![x_input, weight_name, bias_name],
            &normed,
            eps_attrs,
        ),
        simple_spec(
            &scale,
            LayerType::Add,
            vec![gamma_input, ones_const],
            &scale,
            HashMap::new(),
        ),
        simple_spec(
            &scaled,
            LayerType::Mul,
            vec![normed.clone(), scale],
            &scaled,
            HashMap::new(),
        ),
        simple_spec(
            name,
            LayerType::Add,
            vec![scaled, beta_input],
            output_tensor,
            HashMap::new(),
        ),
    ];

    Ok(NodeOutput { specs })
}

/// Translate `KokoroFusedOp::FusedAdainResBlock` to a decomposed LayerSpec
/// chain.
///
/// Tensor inputs `[x, style]`: x is `[B, C_in, T]`, style is `[B, S]`.
/// Two `(AdaIN → activation → Conv1d)` phases plus the residual. Mirrors NN's
/// `translate_fused_adain_resblock`.
#[allow(clippy::too_many_arguments)]
fn translate_fused_adain_resblock(
    name: &str,
    activation: &ResBlockActivation,
    adain1_weight: &WeightPayload,
    adain1_bias: &WeightPayload,
    adain2_weight: &WeightPayload,
    adain2_bias: &WeightPayload,
    conv1_weight: &WeightPayload,
    conv1_bias: &WeightPayload,
    conv1_dilation: usize,
    conv1_padding: usize,
    conv2_weight: &WeightPayload,
    conv2_bias: &WeightPayload,
    conv2_padding: usize,
    eps: f64,
    residual_scale: f64,
    input_tensors: &[String],
    output_tensor: &str,
    output_shape: &[usize],
    ctx: &mut Ctx,
) -> Result<NodeOutput> {
    if input_tensors.len() < 2 {
        return Err(NyError::UnsupportedOp(format!(
            "FusedAdainResBlock requires 2 inputs (x, style), got {}",
            input_tensors.len()
        )));
    }
    let x_input = input_tensors[0].clone();
    let style_input = input_tensors[1].clone();

    let eps_f32 = checked_f64_to_f32(eps, "FusedAdainResBlock eps")?;
    if eps_f32 < 0.0 {
        return Err(NyError::UnsupportedOp(format!(
            "FusedAdainResBlock eps must be non-negative, got {eps}"
        )));
    }

    // Traced Kokoro resblock tensors are [B, C, T].
    if output_shape.len() < 3 {
        return Err(NyError::UnsupportedOp(format!(
            "FusedAdainResBlock requires rank-3 output shape, got rank {}",
            output_shape.len()
        )));
    }
    let c_in = conv1_weight.shape.get(1).copied().ok_or_else(|| {
        NyError::ModelLoad("FusedAdainResBlock: conv1_weight rank < 2".to_string())
    })?;
    let c_out = conv1_weight.shape.first().copied().ok_or_else(|| {
        NyError::ModelLoad("FusedAdainResBlock: conv1_weight is empty".to_string())
    })?;

    let mut specs = Vec::with_capacity(32);

    // Phase 1: AdaIN1 + activation1 + Conv1.
    let phase1_adain_out = emit_adain_phase(
        name,
        "p1",
        &style_input,
        &x_input,
        adain1_weight,
        adain1_bias,
        c_in,
        eps_f32,
        ctx,
        &mut specs,
    )?;

    let activated1_name = format!("{name}_p1_act");
    emit_activation(
        name,
        "p1",
        activation,
        true,
        &phase1_adain_out,
        &activated1_name,
        ctx,
        &mut specs,
    )?;

    let conv1_out_name = format!("{name}_conv1");
    emit_conv1d(
        name,
        "conv1",
        conv1_weight,
        conv1_bias,
        conv1_padding,
        conv1_dilation,
        &activated1_name,
        &conv1_out_name,
        ctx,
        &mut specs,
    )?;

    // Phase 2: AdaIN2 + activation2 + Conv2.
    let phase2_adain_out = emit_adain_phase(
        name,
        "p2",
        &style_input,
        &conv1_out_name,
        adain2_weight,
        adain2_bias,
        c_out,
        eps_f32,
        ctx,
        &mut specs,
    )?;

    let activated2_name = format!("{name}_p2_act");
    emit_activation(
        name,
        "p2",
        activation,
        false,
        &phase2_adain_out,
        &activated2_name,
        ctx,
        &mut specs,
    )?;

    let conv2_out_name = format!("{name}_conv2");
    emit_conv1d(
        name,
        "conv2",
        conv2_weight,
        conv2_bias,
        conv2_padding,
        1, // no dilation
        &activated2_name,
        &conv2_out_name,
        ctx,
        &mut specs,
    )?;

    // Residual: output = (x + conv2_out) * residual_scale.
    let needs_scale = (residual_scale - 1.0).abs() > f64::EPSILON;

    if needs_scale {
        let residual_name = format!("{name}_residual");
        specs.push(simple_spec(
            &residual_name,
            LayerType::Add,
            vec![x_input, conv2_out_name],
            &residual_name,
            HashMap::new(),
        ));

        let scale_f32 = checked_f64_to_f32(residual_scale, "FusedAdainResBlock residual_scale")?;
        let scale_const = format!("{name}_res_scale");
        insert_scalar_constant(ctx, &scale_const, scale_f32)?;

        specs.push(simple_spec(
            name,
            LayerType::Mul,
            vec![residual_name, scale_const],
            output_tensor,
            HashMap::new(),
        ));
    } else {
        specs.push(simple_spec(
            name,
            LayerType::Add,
            vec![x_input, conv2_out_name],
            output_tensor,
            HashMap::new(),
        ));
    }

    Ok(NodeOutput { specs })
}

// ---------------------------------------------------------------------------
// Shared emission helpers (ported from NN's snake / resblock helpers)
// ---------------------------------------------------------------------------

/// Infer num_channels from output shape `[B, C, T]` for norm layers.
///
/// Mirrors NN's `infer_num_channels`.
fn infer_num_channels(output_shape: &[usize], context: &str) -> Result<usize> {
    match output_shape.len() {
        0 | 1 => Err(NyError::UnsupportedOp(format!(
            "{context}: cannot infer num_channels from shape {output_shape:?}"
        ))),
        2 => Ok(output_shape[0]),
        _ => Ok(output_shape[1]),
    }
}

/// Validate that eps casts to a finite, non-negative f32 (AdaIN convention:
/// NN's Adain ops accept eps == 0).
fn validate_adain_eps(eps: f64, op: &str) -> Result<f32> {
    let eps_f32 = eps as f32;
    if !eps_f32.is_finite() || eps_f32 < 0.0 {
        return Err(NyError::NumericalInstability(format!(
            "{op}: eps must be finite and non-negative, got {eps}"
        )));
    }
    Ok(eps_f32)
}

/// Validate Snake alpha: finite (via [`weight_f32`]) and non-zero (the Snake
/// form divides by alpha).
fn validate_snake_alpha(alpha: &WeightPayload, op: &str) -> Result<()> {
    let data = weight_f32(alpha, &format!("{op} alpha"))?;
    for &v in &data {
        if v == 0.0 {
            return Err(NyError::NumericalInstability(format!(
                "{op}: alpha must be finite and non-zero, got {v}"
            )));
        }
    }
    Ok(())
}

/// Emit a native `LayerType::Snake` spec with alpha as a weight tensor.
///
/// The alpha tensor is registered as a weight and passed via `inputs[1]`;
/// NY's Snake converter reads per-channel alpha from `spec.inputs[1]` and
/// constructs `SnakeLayer::per_channel()` (gc#4117). Mirrors NN's
/// `emit_native_snake`. Callers validate alpha first
/// ([`validate_snake_alpha`]).
fn emit_native_snake(
    name: &str,
    alpha: &WeightPayload,
    data_input: &str,
    output_tensor: &str,
    context: &str,
    ctx: &mut Ctx,
) -> Result<LayerSpec> {
    let alpha_name = format!("{name}_alpha");
    let data = weight_f32(alpha, &format!("{context} alpha"))?;
    let alpha_arr = ArrayD::from_shape_vec(IxDyn(&alpha.shape), data)
        .map_err(|e| NyError::ModelLoad(format!("{context}: alpha shape: {e}")))?;
    ctx.insert_weight(&alpha_name, alpha_arr)?;

    Ok(simple_spec(
        name,
        LayerType::Snake,
        vec![data_input.to_string(), alpha_name],
        output_tensor,
        HashMap::new(),
    ))
}

/// Try to emit a native `LayerType::AdaIN` spec when gamma/beta are constant.
///
/// Returns `Some((adain_spec, adain_out_name))` if both gamma and beta are
/// constant tensors in the context, precomputing Kokoro's
/// `style_gamma = gamma + 1.0`. Returns `None` (caller decomposes) if either
/// is a variable graph input. Mirrors NN's `try_native_adain`.
fn try_native_adain(
    name: &str,
    x_input: &str,
    gamma_input: &str,
    beta_input: &str,
    eps_f32: f32,
    ctx: &mut Ctx,
) -> Result<Option<(LayerSpec, String)>> {
    if !ctx.constant_tensors.contains(gamma_input) || !ctx.constant_tensors.contains(beta_input) {
        return Ok(None);
    }
    let (Some(gamma_data), Some(beta_data)) = (
        ctx.weights.get(gamma_input).cloned(),
        ctx.weights.get(beta_input).cloned(),
    ) else {
        return Ok(None);
    };
    // Pre-compute style_gamma = gamma + 1.0 (Kokoro residual convention).
    let style_gamma = gamma_data.mapv(|v| v + 1.0);
    if !style_gamma.iter().all(|v| v.is_finite()) || !beta_data.iter().all(|v| v.is_finite()) {
        return Ok(None);
    }

    let style_gamma_name = format!("{name}_style_gamma");
    let style_beta_name = format!("{name}_style_beta");
    ctx.insert_weight(&style_gamma_name, style_gamma)?;
    ctx.insert_weight(&style_beta_name, beta_data)?;

    let adain_out = format!("{name}_adain");
    let mut eps_attrs = HashMap::new();
    eps_attrs.insert("epsilon".to_string(), AttributeValue::Float(eps_f32));

    let spec = simple_spec(
        &adain_out,
        LayerType::AdaIN,
        vec![x_input.to_string(), style_gamma_name, style_beta_name],
        &adain_out,
        eps_attrs,
    );
    Ok(Some((spec, adain_out)))
}

/// Emit decomposed variable-style AdaIN: InstanceNorm + Reshape + Mul + Add.
///
/// When gamma/beta are variable graph inputs (not constant weights), the
/// compact 3-input `LayerType::AdaIN` form causes gamma-build to default
/// `num_channels=1` (it can't find style params in the weight store).
/// Decomposing into individual layers avoids this (#2987):
///
/// 1. InstanceNorm(x, ones(ch), zeros(ch)) → normalized `[C, T]`
/// 2. Add(gamma, 1.0) → style_gamma `[C]` (Kokoro residual convention)
/// 3. Reshape(style_gamma, [C, 1]) → style_gamma_3d (broadcast-compatible)
/// 4. Mul(normalized, style_gamma_3d) → scaled `[C, T]`
/// 5. Reshape(beta, [C, 1]) → beta_3d
/// 6. Add(scaled, beta_3d) → output `[C, T]`
///
/// Mirrors NN's `emit_variable_adain`.
fn emit_variable_adain(
    name: &str,
    x_input: &str,
    gamma_input: &str,
    beta_input: &str,
    eps_f32: f32,
    num_channels: usize,
    ctx: &mut Ctx,
) -> Result<(Vec<LayerSpec>, String)> {
    let mut specs = Vec::with_capacity(6);

    // 1. InstanceNorm with identity affine (gamma=ones, beta=zeros).
    let norm_name = format!("{name}_instnorm");
    let norm_gamma = format!("{norm_name}_gamma");
    let norm_beta = format!("{norm_name}_beta");
    ctx.insert_weight(
        &norm_gamma,
        ArrayD::from_elem(IxDyn(&[num_channels]), 1.0_f32),
    )?;
    ctx.insert_weight(
        &norm_beta,
        ArrayD::from_elem(IxDyn(&[num_channels]), 0.0_f32),
    )?;
    let mut eps_attrs = HashMap::new();
    eps_attrs.insert("epsilon".to_string(), AttributeValue::Float(eps_f32));
    let norm_out = format!("{norm_name}_out");
    specs.push(simple_spec(
        &norm_name,
        LayerType::InstanceNorm,
        vec![x_input.to_string(), norm_gamma, norm_beta],
        &norm_out,
        eps_attrs,
    ));

    // 2. style_gamma = gamma + 1.0 (Kokoro residual convention).
    let ones_const = format!("{name}_ones");
    insert_scalar_constant(ctx, &ones_const, 1.0)?;
    let style_gamma_name = format!("{name}_style_gamma");
    specs.push(simple_spec(
        &style_gamma_name,
        LayerType::Add,
        vec![gamma_input.to_string(), ones_const],
        &style_gamma_name,
        HashMap::new(),
    ));

    // 3. Reshape style_gamma [C] → [C, 1] for broadcast with [C, T].
    // NY operates in unbatched convention; the normalized output is [C, T] so
    // the style must be [C, 1] for element-wise broadcast.
    let ch_i64 = dim_as_i64(num_channels, "adain num_channels")?;
    let sg_3d_name = format!("{name}_style_gamma_3d");
    let mut sg_attrs = HashMap::new();
    sg_attrs.insert("shape".to_string(), AttributeValue::Ints(vec![ch_i64, 1]));
    specs.push(simple_spec(
        &sg_3d_name,
        LayerType::Reshape,
        vec![style_gamma_name],
        &sg_3d_name,
        sg_attrs,
    ));
    ctx.tensor_shapes
        .insert(sg_3d_name.clone(), vec![ch_i64, 1]);

    // 4. Mul(normalized, style_gamma_3d) → scaled.
    let scaled_name = format!("{name}_scaled");
    specs.push(simple_spec(
        &scaled_name,
        LayerType::Mul,
        vec![norm_out, sg_3d_name],
        &scaled_name,
        HashMap::new(),
    ));

    // 5. Reshape beta [C] → [C, 1] for broadcast with [C, T].
    let beta_3d_name = format!("{name}_beta_3d");
    let mut beta_attrs = HashMap::new();
    beta_attrs.insert("shape".to_string(), AttributeValue::Ints(vec![ch_i64, 1]));
    specs.push(simple_spec(
        &beta_3d_name,
        LayerType::Reshape,
        vec![beta_input.to_string()],
        &beta_3d_name,
        beta_attrs,
    ));
    ctx.tensor_shapes
        .insert(beta_3d_name.clone(), vec![ch_i64, 1]);

    // 6. Add(scaled, beta_3d) → output.
    let adain_out = format!("{name}_adain");
    specs.push(simple_spec(
        &adain_out,
        LayerType::Add,
        vec![scaled_name, beta_3d_name],
        &adain_out,
        HashMap::new(),
    ));

    Ok((specs, adain_out))
}

/// Emit decomposed LeakyReLU specs: `α*x + (1-α)*ReLU(x)` (#2977).
///
/// Mirrors NN's `emit_leaky_relu_specs` (prefix `{name}_lr`).
fn emit_leaky_relu_specs(
    name: &str,
    slope_f32: f32,
    input_tensor: &str,
    output_tensor: &str,
    ctx: &mut Ctx,
    specs: &mut Vec<LayerSpec>,
) -> Result<()> {
    let prefix = format!("{name}_lr");

    let alpha_const = format!("{prefix}_alpha");
    insert_scalar_constant(ctx, &alpha_const, slope_f32)?;
    let one_minus_alpha_const = format!("{prefix}_1ma");
    insert_scalar_constant(ctx, &one_minus_alpha_const, 1.0 - slope_f32)?;

    let alpha_x = format!("{prefix}_ax");
    specs.push(simple_spec(
        &alpha_x,
        LayerType::Mul,
        vec![input_tensor.to_string(), alpha_const],
        &alpha_x,
        HashMap::new(),
    ));
    let relu_x = format!("{prefix}_relu");
    specs.push(simple_spec(
        &relu_x,
        LayerType::ReLU,
        vec![input_tensor.to_string()],
        &relu_x,
        HashMap::new(),
    ));
    let relu_scaled = format!("{prefix}_rs");
    specs.push(simple_spec(
        &relu_scaled,
        LayerType::Mul,
        vec![relu_x, one_minus_alpha_const],
        &relu_scaled,
        HashMap::new(),
    ));
    specs.push(simple_spec(
        output_tensor,
        LayerType::Add,
        vec![alpha_x, relu_scaled],
        output_tensor,
        HashMap::new(),
    ));
    Ok(())
}

/// Emit a single resblock AdaIN phase:
/// Linear(style) → split → InstanceNorm → affine.
///
/// Returns the output tensor name of the affine step. Mirrors NN's
/// `emit_adain_phase`.
#[allow(clippy::too_many_arguments)]
fn emit_adain_phase(
    name: &str,
    phase: &str,
    style_input: &str,
    x_input: &str,
    adain_weight: &WeightPayload,
    adain_bias: &WeightPayload,
    num_channels: usize,
    eps_f32: f32,
    ctx: &mut Ctx,
    specs: &mut Vec<LayerSpec>,
) -> Result<String> {
    if num_channels == 0 {
        return Err(NyError::UnsupportedOp(
            "FusedAdainResBlock: num_channels must be > 0".to_string(),
        ));
    }

    // Style projection: Linear(style, w, b) → proj [B, 2*C].
    let proj_name = format!("{name}_{phase}_proj");
    let w_name = format!("{name}_{phase}_adain_w");
    let b_name = format!("{name}_{phase}_adain_b");
    insert_payload(ctx, adain_weight, &w_name, &format!("AdaIN {phase} weight"))?;
    insert_payload(ctx, adain_bias, &b_name, &format!("AdaIN {phase} bias"))?;

    let mut linear_attrs = HashMap::new();
    linear_attrs.insert("transB".to_string(), AttributeValue::Int(1));
    specs.push(LayerSpec {
        name: proj_name.clone(),
        layer_type: LayerType::Linear,
        inputs: vec![style_input.to_string(), w_name.clone(), b_name],
        outputs: vec![proj_name.clone()],
        weights: Some(WeightRef {
            name: w_name,
            shape: adain_weight.shape.clone(),
            original_dtype: DataType::Float32,
        }),
        attributes: linear_attrs,
    });

    // Narrow(proj, dim=1, 0..C) → gamma, Narrow(proj, dim=1, C..2C) → beta.
    let gamma_name = format!("{name}_{phase}_gamma");
    let beta_name = format!("{name}_{phase}_beta");

    // Channel-split axis, TRAILING-RELATIVE: dim 1 of the conceptual [B, 2C]
    // projection → `-1` (see `super::trailing_axis`; d7144ea7 convention).
    //
    // Deletion-time audit of the historic positive `axis=1` (the legacy nn
    // emission this arm mirrored byte-for-byte): `proj` is a bridge-
    // synthesized intermediate with NO recorded shape, so ny-build's Slice
    // conversion was REGIME-DEPENDENT —
    //   * batched-classified models (every kokoro corpus trace: the resblock
    //     x input is rank >= 2): unknown recorded rank → legacy `axis-1`
    //     adjustment → internal axis 0 on the rank-1 runtime `proj` [2C].
    //     Correct.
    //   * unbatched-classified models (all graph inputs rank <= 1): ONNX
    //     axes convert VERBATIM → axis 1 is out of range for the rank-1
    //     runtime `proj` → fail-closed error (never unsound), exactly the
    //     ResizeBilinear trim-Slice case fixed in d7144ea7.
    // Trailing `-1` selects the same dimension in BOTH regimes (negative
    // axes pass through conversion and resolve against the runtime rank at
    // propagation), so the batched-regime lowering is unchanged (kokoro
    // suites re-gated green) and the unbatched regime stops failing.
    let axis_i64 = super::trailing_axis(1, 2, "AdaIN gamma/beta split axis")?;
    let c_i64 = dim_as_i64(num_channels, "Narrow length")?;
    let c2 = num_channels.checked_mul(2).ok_or_else(|| {
        NyError::InternalError("FusedAdainResBlock: 2 * num_channels overflow".to_string())
    })?;
    let c2_i64 = dim_as_i64(c2, "Narrow end")?;

    let mut gamma_attrs = HashMap::new();
    gamma_attrs.insert("axis".to_string(), AttributeValue::Int(axis_i64));
    gamma_attrs.insert("start".to_string(), AttributeValue::Int(0));
    gamma_attrs.insert("end".to_string(), AttributeValue::Int(c_i64));
    specs.push(simple_spec(
        &gamma_name,
        LayerType::Slice,
        vec![proj_name.clone()],
        &gamma_name,
        gamma_attrs,
    ));

    let mut beta_attrs = HashMap::new();
    beta_attrs.insert("axis".to_string(), AttributeValue::Int(axis_i64));
    beta_attrs.insert("start".to_string(), AttributeValue::Int(c_i64));
    beta_attrs.insert("end".to_string(), AttributeValue::Int(c2_i64));
    specs.push(simple_spec(
        &beta_name,
        LayerType::Slice,
        vec![proj_name],
        &beta_name,
        beta_attrs,
    ));

    // Reshape gamma, beta: [C] → [C, 1] (unbatched convention for NY).
    // gamma-build attribute-based Reshape does NOT strip batch (#2987), so the
    // target must be unbatched [C, 1], not [-1, C, 1].
    let gamma_3d = format!("{name}_{phase}_gamma3d");
    let beta_3d = format!("{name}_{phase}_beta3d");
    let reshape_3d = vec![c_i64, 1];
    let mut reshape_attrs = HashMap::new();
    reshape_attrs.insert("shape".to_string(), AttributeValue::Ints(reshape_3d));
    specs.push(simple_spec(
        &gamma_3d,
        LayerType::Reshape,
        vec![gamma_name],
        &gamma_3d,
        reshape_attrs.clone(),
    ));
    specs.push(simple_spec(
        &beta_3d,
        LayerType::Reshape,
        vec![beta_name],
        &beta_3d,
        reshape_attrs,
    ));

    // InstanceNorm(x, eps) with identity affine (gamma=ones, beta=zeros).
    let normed = format!("{name}_{phase}_normed");
    let norm_gamma = format!("{name}_{phase}_norm_gamma");
    let norm_beta = format!("{name}_{phase}_norm_beta");
    ctx.insert_weight(
        &norm_gamma,
        ArrayD::from_elem(IxDyn(&[num_channels]), 1.0_f32),
    )?;
    ctx.insert_weight(
        &norm_beta,
        ArrayD::from_elem(IxDyn(&[num_channels]), 0.0_f32),
    )?;

    let mut eps_attrs = HashMap::new();
    eps_attrs.insert("epsilon".to_string(), AttributeValue::Float(eps_f32));
    specs.push(simple_spec(
        &normed,
        LayerType::InstanceNorm,
        vec![x_input.to_string(), norm_gamma, norm_beta],
        &normed,
        eps_attrs,
    ));

    // Affine: (1 + gamma) * normed + beta.
    let ones_const = format!("{name}_{phase}_ones");
    insert_scalar_constant(ctx, &ones_const, 1.0)?;

    let scale = format!("{name}_{phase}_scale");
    let scaled = format!("{name}_{phase}_scaled");
    let adain_out = format!("{name}_{phase}_adain");

    specs.push(simple_spec(
        &scale,
        LayerType::Add,
        vec![gamma_3d, ones_const],
        &scale,
        HashMap::new(),
    ));
    specs.push(simple_spec(
        &scaled,
        LayerType::Mul,
        vec![normed, scale],
        &scaled,
        HashMap::new(),
    ));
    specs.push(simple_spec(
        &adain_out,
        LayerType::Add,
        vec![scaled, beta_3d],
        &adain_out,
        HashMap::new(),
    ));

    Ok(adain_out)
}

/// Emit a resblock activation (Snake or LeakyRelu) into the specs list.
///
/// Mirrors NN's resblock-helper `emit_activation`.
#[allow(clippy::too_many_arguments)]
fn emit_activation(
    name: &str,
    phase: &str,
    activation: &ResBlockActivation,
    is_first: bool,
    input_name: &str,
    output_name: &str,
    ctx: &mut Ctx,
    specs: &mut Vec<LayerSpec>,
) -> Result<()> {
    match activation {
        ResBlockActivation::Snake { alpha1, alpha2 } => {
            let alpha = if is_first { alpha1 } else { alpha2 };
            emit_snake_activation(name, phase, alpha, input_name, output_name, ctx, specs)
        }
        ResBlockActivation::LeakyRelu { slope } => {
            // Decompose LeakyReLU(x, α) = α*x + (1-α)*ReLU(x) for tight CROWN
            // bounds (#2977); NN uses the `{output_name}_lr` prefix here.
            let slope_f32 = checked_f64_to_f32(*slope, "LeakyRelu slope")?;
            emit_leaky_relu_specs(output_name, slope_f32, input_name, output_name, ctx, specs)
        }
    }
}

/// Emit the decomposed Snake activation `x + (1/alpha) * sin²(alpha * x)` as
/// 5 LayerSpecs (Mul, Sin, Pow, Mul, Add), constant-folding `1/alpha`
/// (#2413). Mirrors NN's `emit_snake_activation`.
fn emit_snake_activation(
    name: &str,
    phase: &str,
    alpha: &WeightPayload,
    input_name: &str,
    output_name: &str,
    ctx: &mut Ctx,
    specs: &mut Vec<LayerSpec>,
) -> Result<()> {
    let prefix = format!("{name}_{phase}_snake");

    // Insert alpha weight and constant-fold 1/alpha (#2413).
    let alpha_name = format!("{prefix}_alpha");
    let data = weight_f32(alpha, "FusedAdainResBlock Snake alpha")?;
    let alpha_arr = ArrayD::from_shape_vec(IxDyn(&alpha.shape), data)
        .map_err(|e| NyError::ModelLoad(format!("Snake alpha shape: {e}")))?;
    for &v in alpha_arr.iter() {
        if !v.is_finite() || v == 0.0 {
            return Err(NyError::NumericalInstability(format!(
                "Snake alpha must be finite and non-zero, got {v}"
            )));
        }
    }
    ctx.insert_weight(&alpha_name, alpha_arr.clone())?;

    let inv_alpha_name = format!("{prefix}_inv_alpha");
    let inv_alpha_arr = alpha_arr.mapv(|v| 1.0 / v);
    ctx.insert_weight(&inv_alpha_name, inv_alpha_arr)?;

    let pow2_const = format!("{prefix}_pow2");
    insert_scalar_constant(ctx, &pow2_const, 2.0)?;

    // Intermediate names.
    let snake_scaled = format!("{prefix}_scaled");
    let sin_val = format!("{prefix}_sin");
    let sin_sq = format!("{prefix}_sin_sq");
    let weighted = format!("{prefix}_weighted");

    let mut pow_attrs = HashMap::new();
    pow_attrs.insert("power".to_string(), AttributeValue::Float(2.0));

    // 1. scaled = x * alpha
    specs.push(simple_spec(
        &snake_scaled,
        LayerType::Mul,
        vec![input_name.to_string(), alpha_name],
        &snake_scaled,
        HashMap::new(),
    ));
    // 2. sin_val = sin(scaled)
    specs.push(simple_spec(
        &sin_val,
        LayerType::Sin,
        vec![snake_scaled],
        &sin_val,
        HashMap::new(),
    ));
    // 3. sin_sq = sin_val ^ 2
    specs.push(simple_spec(
        &sin_sq,
        LayerType::Pow,
        vec![sin_val, pow2_const],
        &sin_sq,
        pow_attrs,
    ));
    // 4. weighted = sin_sq * (1/alpha)
    specs.push(simple_spec(
        &weighted,
        LayerType::Mul,
        vec![sin_sq, inv_alpha_name],
        &weighted,
        HashMap::new(),
    ));
    // 5. output = x + weighted
    specs.push(simple_spec(
        output_name,
        LayerType::Add,
        vec![input_name.to_string(), weighted],
        output_name,
        HashMap::new(),
    ));

    Ok(())
}

/// Emit a resblock Conv1d LayerSpec with weight/bias insertion.
///
/// For dilated convolutions, expands the kernel (inserts zeros) and sets
/// effective dilation to 1 — same approach as the core `translate_conv1d`.
/// Mirrors NN's resblock-helper `emit_conv1d` (spec name == output name).
#[allow(clippy::too_many_arguments)]
fn emit_conv1d(
    name: &str,
    suffix: &str,
    weight: &WeightPayload,
    bias: &WeightPayload,
    padding: usize,
    dilation: usize,
    input_name: &str,
    output_name: &str,
    ctx: &mut Ctx,
    specs: &mut Vec<LayerSpec>,
) -> Result<()> {
    if dilation == 0 {
        return Err(NyError::UnsupportedOp(format!(
            "FusedAdainResBlock {suffix}: dilation must be >= 1, got 0"
        )));
    }
    let w_name = format!("{name}_{suffix}_weight");
    let b_name = format!("{name}_{suffix}_bias");

    // For dilated convolutions, expand the kernel and set effective dilation to 1.
    let (weight_shape, effective_dilation) = if dilation > 1 {
        let data = weight_f32(weight, &format!("{suffix} weight"))?;
        let kernel = ArrayD::from_shape_vec(IxDyn(&weight.shape), data)
            .map_err(|e| NyError::ModelLoad(format!("{suffix} weight shape: {e}")))?;
        let expanded = expand_dilated_conv1d_kernel(&kernel, dilation)?;
        let shape = expanded.shape().to_vec();
        ctx.insert_weight(&w_name, expanded)?;
        insert_payload(ctx, bias, &b_name, &format!("{suffix} bias"))?;
        (shape, 1_usize)
    } else {
        insert_payload(ctx, weight, &w_name, &format!("{suffix} weight"))?;
        insert_payload(ctx, bias, &b_name, &format!("{suffix} bias"))?;
        (weight.shape.clone(), dilation)
    };

    let pad_i64 = dim_as_i64(padding, &format!("{suffix} padding"))?;
    let dil_i64 = dim_as_i64(effective_dilation, &format!("{suffix} dilation"))?;

    let mut attrs = HashMap::new();
    attrs.insert("strides".to_string(), AttributeValue::Ints(vec![1]));
    attrs.insert(
        "pads".to_string(),
        AttributeValue::Ints(vec![pad_i64, pad_i64]),
    );
    attrs.insert("dilations".to_string(), AttributeValue::Ints(vec![dil_i64]));
    attrs.insert("group".to_string(), AttributeValue::Int(1));

    specs.push(LayerSpec {
        name: output_name.to_string(),
        layer_type: LayerType::Conv1d,
        inputs: vec![input_name.to_string(), w_name.clone(), b_name],
        outputs: vec![output_name.to_string()],
        weights: Some(WeightRef {
            name: w_name,
            shape: weight_shape,
            original_dtype: DataType::Float32,
        }),
        attributes: attrs,
    });

    Ok(())
}

/// Expand a dilated Conv1d kernel `[out, in, k]` into an equivalent dense
/// kernel with zero-interleaved taps, so a dilation-1 convolution is exact.
///
/// Copied from `ops_core::expand_dilated_conv1d_kernel` (private there; dedupe
/// into `mod.rs` later).
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
    let new_k = (k - 1)
        .checked_mul(dilation)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| NyError::InternalError("Conv1d dilation expansion overflow".to_string()))?;
    let mut expanded = ArrayD::<f32>::zeros(IxDyn(&[out_c, in_c, new_k]));
    for o in 0..out_c {
        for i in 0..in_c {
            for tap in 0..k {
                expanded[[o, i, tap * dilation]] = kernel[[o, i, tap]];
            }
        }
    }
    Ok(expanded)
}

#[cfg(test)]
mod tests {
    use super::super::{translate, translate_multi_input};
    use crate::schema::{
        ComputationGraph, DType, KokoroFusedOp, NodeId, ResBlockActivation, TraceNode, TraceOp,
        WeightPayload,
    };
    use ny_build::{AttributeValue, GraphModel, GraphNetworkOptions};
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

    fn count(model: &GraphModel, lt: &LayerType) -> usize {
        model
            .network
            .layers
            .iter()
            .filter(|l| &l.layer_type == lt)
            .count()
    }

    fn find<'m>(model: &'m GraphModel, lt: &LayerType) -> &'m ny_build::LayerSpec {
        model
            .network
            .layers
            .iter()
            .find(|l| &l.layer_type == lt)
            .unwrap_or_else(|| panic!("layer of type {lt:?} present"))
    }

    fn assert_builds(model: &GraphModel, what: &str) {
        model
            .build_graph_network(GraphNetworkOptions::default())
            .unwrap_or_else(|e| panic!("{what} builds a graph network: {e}"));
    }

    /// SnakeTensor emits one native Snake layer with alpha in `inputs[1]`.
    #[test]
    fn snake_tensor_emits_native_snake() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 3]),
            node(
                1,
                "snake",
                TraceOp::KokoroFused(KokoroFusedOp::SnakeTensor {
                    alpha: WeightPayload::f32(vec![0.5, 2.0], vec![1, 2, 1]),
                }),
                &[0],
                &[2, 3],
            ),
        ]);
        let model = translate(&graph).expect("snake translates");
        assert_eq!(count(&model, &LayerType::Snake), 1, "one native Snake");
        let snake = find(&model, &LayerType::Snake);
        assert_eq!(
            snake.inputs,
            vec![
                "layer0_trace_0_out".to_string(),
                "layer0_trace_1_alpha".to_string()
            ],
            "alpha passed as weight tensor in inputs[1]"
        );
        assert!(model.weights.contains_key("layer0_trace_1_alpha"));
        assert_builds(&model, "SnakeTensor model");
    }

    /// SnakeTensor with a zero alpha element is refused (alpha divides).
    #[test]
    fn snake_tensor_zero_alpha_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 3]),
            node(
                1,
                "snake",
                TraceOp::KokoroFused(KokoroFusedOp::SnakeTensor {
                    alpha: WeightPayload::f32(vec![0.5, 0.0], vec![1, 2, 1]),
                }),
                &[0],
                &[2, 3],
            ),
        ]);
        let err = translate(&graph).expect_err("zero alpha refused");
        assert!(
            matches!(err, NyError::NumericalInstability(ref m) if m.contains("non-zero")),
            "zero alpha yields a fail-closed error, got {err:?}"
        );
    }

    /// AdainSnake with constant gamma/beta → native AdaIN + native Snake,
    /// with style_gamma precomputed as gamma + 1.0.
    #[test]
    fn adain_snake_constant_style_native_adain() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 4]),
            node(
                1,
                "g",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![0.5, 0.25], vec![2]),
                },
                &[],
                &[2],
            ),
            node(
                2,
                "b",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![0.1, 0.2], vec![2]),
                },
                &[],
                &[2],
            ),
            node(
                3,
                "as",
                TraceOp::KokoroFused(KokoroFusedOp::AdainSnake {
                    alpha: WeightPayload::f32(vec![1.0, 1.0], vec![1, 2, 1]),
                    eps: 1e-5,
                }),
                &[0, 1, 2],
                &[2, 4],
            ),
        ]);
        let model = translate(&graph).expect("adain-snake translates");
        assert_eq!(count(&model, &LayerType::AdaIN), 1, "one native AdaIN");
        assert_eq!(count(&model, &LayerType::Snake), 1, "one native Snake");
        assert_eq!(
            count(&model, &LayerType::InstanceNorm),
            0,
            "no decomposition on the constant-style path"
        );

        let adain = find(&model, &LayerType::AdaIN);
        assert_eq!(
            adain.attributes.get("epsilon"),
            Some(&AttributeValue::Float(1e-5))
        );
        assert_eq!(
            adain.inputs,
            vec![
                "layer0_trace_0_out".to_string(),
                "layer0_trace_3_style_gamma".to_string(),
                "layer0_trace_3_style_beta".to_string(),
            ]
        );
        // style_gamma = gamma + 1.0 (Kokoro residual convention).
        let sg = model
            .weights
            .get("layer0_trace_3_style_gamma")
            .expect("style_gamma registered");
        assert_eq!(
            sg.iter().copied().collect::<Vec<f32>>(),
            vec![1.5, 1.25],
            "style_gamma precomputed as gamma + 1.0"
        );

        // Snake consumes the AdaIN output.
        let snake = find(&model, &LayerType::Snake);
        assert_eq!(snake.inputs[0], "layer0_trace_3_adain");
        assert_builds(&model, "AdainSnake constant-style model");
    }

    /// AdainSnake with variable gamma/beta decomposes into
    /// InstanceNorm + Add + Reshape + Mul + Add + Snake (no native AdaIN).
    #[test]
    fn adain_snake_variable_style_decomposes() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 4]),
            node(1, "g", TraceOp::Input, &[], &[2]),
            node(2, "b", TraceOp::Input, &[], &[2]),
            node(
                3,
                "as",
                TraceOp::KokoroFused(KokoroFusedOp::AdainSnake {
                    alpha: WeightPayload::f32(vec![1.0, 2.0], vec![1, 2, 1]),
                    eps: 1e-5,
                }),
                &[0, 1, 2],
                &[2, 4],
            ),
        ]);
        let translation = translate_multi_input(&graph).expect("variable-style translates");
        let model = &translation.model;
        assert_eq!(count(model, &LayerType::AdaIN), 0, "no native AdaIN");
        assert_eq!(count(model, &LayerType::InstanceNorm), 1);
        assert_eq!(count(model, &LayerType::Snake), 1);
        // Decomposition emits 2 style Reshapes ([C] → [C, 1]) on top of the
        // 3 multi-input recovery Reshapes.
        assert_eq!(count(model, &LayerType::Reshape), 5);

        let inst = find(model, &LayerType::InstanceNorm);
        assert_eq!(inst.name, "layer0_trace_3_instnorm");
        assert_eq!(
            inst.attributes.get("epsilon"),
            Some(&AttributeValue::Float(1e-5))
        );
        // Identity affine weights for the InstanceNorm.
        assert!(model.weights.contains_key("layer0_trace_3_instnorm_gamma"));
        assert!(model.weights.contains_key("layer0_trace_3_instnorm_beta"));
        // style_gamma = gamma + 1.0 via an Add against the ones constant.
        let sg_add = model
            .network
            .layers
            .iter()
            .find(|l| l.name == "layer0_trace_3_style_gamma")
            .expect("style_gamma Add present");
        assert_eq!(sg_add.layer_type, LayerType::Add);
        assert_eq!(sg_add.inputs[1], "layer0_trace_3_ones");
        assert_builds(model, "AdainSnake variable-style model");
    }

    /// AdainLeakyRelu with constant gamma/beta → native AdaIN + the
    /// `α·x + (1−α)·ReLU(x)` LeakyReLU decomposition (#2977).
    #[test]
    fn adain_leaky_relu_constant_style() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 4]),
            node(
                1,
                "g",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![0.5, 0.5], vec![2]),
                },
                &[],
                &[2],
            ),
            node(
                2,
                "b",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![0.0, 0.0], vec![2]),
                },
                &[],
                &[2],
            ),
            node(
                3,
                "alr",
                TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu {
                    eps: 1e-5,
                    slope: 0.2,
                }),
                &[0, 1, 2],
                &[2, 4],
            ),
        ]);
        let model = translate(&graph).expect("adain-leaky-relu translates");
        assert_eq!(count(&model, &LayerType::AdaIN), 1, "one native AdaIN");
        assert_eq!(count(&model, &LayerType::ReLU), 1, "decomposed LeakyReLU");
        assert_eq!(count(&model, &LayerType::Mul), 2, "α·x and (1−α)·ReLU(x)");
        assert_eq!(count(&model, &LayerType::Snake), 0);

        // Decomposition constants: slope and 1 − slope.
        let alpha = model
            .weights
            .get("layer0_trace_3_lr_alpha")
            .expect("slope constant registered");
        assert_eq!(alpha.iter().copied().collect::<Vec<f32>>(), vec![0.2]);
        let one_minus = model
            .weights
            .get("layer0_trace_3_lr_1ma")
            .expect("1 − slope constant registered");
        assert_eq!(one_minus.iter().copied().collect::<Vec<f32>>(), vec![0.8]);
        assert_builds(&model, "AdainLeakyRelu constant-style model");
    }

    /// AdaLayerNorm decomposes into LayerNorm + Add + Mul + Add
    /// (`(1 + gamma) * LayerNorm(x, w, b) + beta`).
    #[test]
    fn ada_layer_norm_decomposes() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[4]),
            node(
                1,
                "g",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![0.5; 4], vec![4]),
                },
                &[],
                &[4],
            ),
            node(
                2,
                "b",
                TraceOp::ConstantWeight {
                    weight: WeightPayload::f32(vec![0.1; 4], vec![4]),
                },
                &[],
                &[4],
            ),
            node(
                3,
                "aln",
                TraceOp::KokoroFused(KokoroFusedOp::AdaLayerNorm {
                    norm_weight: WeightPayload::f32(vec![1.0; 4], vec![4]),
                    norm_bias: WeightPayload::f32(vec![0.0; 4], vec![4]),
                    eps: 1e-5,
                }),
                &[0, 1, 2],
                &[4],
            ),
        ]);
        let model = translate(&graph).expect("ada-layer-norm translates");
        assert_eq!(count(&model, &LayerType::LayerNorm), 1);
        assert_eq!(count(&model, &LayerType::Mul), 1);
        // Input identity Add + scale Add(gamma, 1) + final Add(scaled, beta).
        assert_eq!(count(&model, &LayerType::Add), 3);

        let ln = find(&model, &LayerType::LayerNorm);
        assert_eq!(ln.name, "layer0_trace_3_normed");
        assert_eq!(
            ln.attributes.get("epsilon"),
            Some(&AttributeValue::Float(1e-5))
        );
        assert_eq!(
            ln.inputs,
            vec![
                "layer0_trace_0_out".to_string(),
                "layer0_trace_3_norm_weight".to_string(),
                "layer0_trace_3_norm_bias".to_string(),
            ]
        );
        assert!(model.weights.contains_key("layer0_trace_3_norm_weight"));
        assert!(model.weights.contains_key("layer0_trace_3_norm_bias"));
        assert!(model.weights.contains_key("layer0_trace_3_ones"));
        // The final Add is the node's own layer producing the node output.
        let final_add = model
            .network
            .layers
            .iter()
            .find(|l| l.name == "layer0_trace_3")
            .expect("final Add present");
        assert_eq!(final_add.layer_type, LayerType::Add);
        assert_eq!(final_add.outputs, vec!["layer0_trace_3_out".to_string()]);
        assert_builds(&model, "AdaLayerNorm model");
    }

    /// Shared resblock fixture: x [1, 2, 4], style [1, 3], K=3 convs.
    fn resblock_graph_impl(
        activation: ResBlockActivation,
        conv1_dilation: usize,
        conv1_padding: usize,
        residual_scale: f64,
    ) -> ComputationGraph {
        // conv1 out T: fixture keeps T=4 through conv1.
        assert_eq!(4 + 2 * conv1_padding - ((3 - 1) * conv1_dilation), 4);
        ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[1, 2, 4]),
            node(1, "s", TraceOp::Input, &[], &[1, 3]),
            node(
                2,
                "rb",
                TraceOp::KokoroFused(KokoroFusedOp::FusedAdainResBlock {
                    activation,
                    adain1_weight: WeightPayload::f32(vec![0.1; 4 * 3], vec![4, 3]),
                    adain1_bias: WeightPayload::f32(vec![0.0; 4], vec![4]),
                    adain2_weight: WeightPayload::f32(vec![0.1; 4 * 3], vec![4, 3]),
                    adain2_bias: WeightPayload::f32(vec![0.0; 4], vec![4]),
                    conv1_weight: WeightPayload::f32(vec![0.1; 2 * 2 * 3], vec![2, 2, 3]),
                    conv1_bias: WeightPayload::f32(vec![0.0; 2], vec![2]),
                    conv1_dilation,
                    conv1_padding,
                    conv2_weight: WeightPayload::f32(vec![0.1; 2 * 2 * 3], vec![2, 2, 3]),
                    conv2_bias: WeightPayload::f32(vec![0.0; 2], vec![2]),
                    conv2_padding: 1,
                    eps: 1e-5,
                    residual_scale,
                }),
                &[0, 1],
                &[1, 2, 4],
            ),
        ])
    }

    /// FusedAdainResBlock (Snake, residual_scale = 1.0): full decomposed
    /// chain — 2 Linear style projections, 4 Slices, 2 InstanceNorms,
    /// 2 Snake decompositions (Sin/Pow), 2 Conv1ds, final residual Add.
    #[test]
    fn fused_adain_resblock_snake_emits_full_chain() {
        let graph = resblock_graph_impl(
            ResBlockActivation::Snake {
                alpha1: WeightPayload::f32(vec![1.0, 2.0], vec![1, 2, 1]),
                alpha2: WeightPayload::f32(vec![0.5, 1.5], vec![1, 2, 1]),
            },
            1,
            1,
            1.0,
        );
        let translation = translate_multi_input(&graph).expect("resblock translates");
        let model = &translation.model;

        assert_eq!(count(model, &LayerType::Linear), 2, "2 style projections");
        assert_eq!(count(model, &LayerType::InstanceNorm), 2, "2 AdaIN norms");
        assert_eq!(count(model, &LayerType::Conv1d), 2, "2 convolutions");
        assert_eq!(count(model, &LayerType::Sin), 2, "2 Snake sin()");
        assert_eq!(count(model, &LayerType::Pow), 2, "2 Snake sin²()");
        // 4 gamma/beta Slices + 2 multi-input recovery Slices.
        assert_eq!(count(model, &LayerType::Slice), 6);

        // Style projection carries transB=1 and the AdaIN weight.
        let proj = model
            .network
            .layers
            .iter()
            .find(|l| l.name == "layer0_trace_2_p1_proj")
            .expect("phase-1 projection present");
        assert_eq!(proj.layer_type, LayerType::Linear);
        assert_eq!(proj.attributes.get("transB"), Some(&AttributeValue::Int(1)));

        // gamma Slice: trailing-relative channel axis (dim 1 of the
        // conceptual [B, 2C] projection → -1), [0, C).
        let gamma_slice = model
            .network
            .layers
            .iter()
            .find(|l| l.name == "layer0_trace_2_p1_gamma")
            .expect("phase-1 gamma slice present");
        assert_eq!(gamma_slice.layer_type, LayerType::Slice);
        assert_eq!(
            gamma_slice.attributes.get("axis"),
            Some(&AttributeValue::Int(-1))
        );
        assert_eq!(
            gamma_slice.attributes.get("start"),
            Some(&AttributeValue::Int(0))
        );
        assert_eq!(
            gamma_slice.attributes.get("end"),
            Some(&AttributeValue::Int(2))
        );

        // Snake alpha1/alpha2 and folded reciprocals are registered.
        assert!(model.weights.contains_key("layer0_trace_2_p1_snake_alpha"));
        assert!(model
            .weights
            .contains_key("layer0_trace_2_p1_snake_inv_alpha"));
        assert!(model.weights.contains_key("layer0_trace_2_p2_snake_alpha"));

        // residual_scale == 1.0: final layer is the plain residual Add.
        let final_layer = model
            .network
            .layers
            .iter()
            .find(|l| l.name == "layer0_trace_2")
            .expect("final residual layer present");
        assert_eq!(final_layer.layer_type, LayerType::Add);
        assert_eq!(final_layer.outputs, vec!["layer0_trace_2_out".to_string()]);
        assert_eq!(
            count(model, &LayerType::Mul),
            6,
            "2 phases × (affine + 2 Snake Muls)"
        );
        assert_builds(model, "FusedAdainResBlock Snake model");
    }

    /// FusedAdainResBlock (LeakyRelu, residual_scale = 1/√2): LeakyReLU
    /// decomposition per phase + scaled residual (Add then Mul).
    #[test]
    fn fused_adain_resblock_leaky_relu_scaled_residual() {
        let graph = resblock_graph_impl(
            ResBlockActivation::LeakyRelu { slope: 0.2 },
            1,
            1,
            std::f64::consts::FRAC_1_SQRT_2,
        );
        let translation = translate_multi_input(&graph).expect("resblock translates");
        let model = &translation.model;

        assert_eq!(count(model, &LayerType::ReLU), 2, "one ReLU per phase");
        assert_eq!(count(model, &LayerType::Sin), 0, "no Snake on the F0 path");

        // Scaled residual: Add(x, conv2) then Mul(residual, res_scale).
        let residual_add = model
            .network
            .layers
            .iter()
            .find(|l| l.name == "layer0_trace_2_residual")
            .expect("residual Add present");
        assert_eq!(residual_add.layer_type, LayerType::Add);
        let final_mul = model
            .network
            .layers
            .iter()
            .find(|l| l.name == "layer0_trace_2")
            .expect("final Mul present");
        assert_eq!(final_mul.layer_type, LayerType::Mul);
        assert_eq!(
            final_mul.inputs,
            vec![
                "layer0_trace_2_residual".to_string(),
                "layer0_trace_2_res_scale".to_string(),
            ]
        );
        let scale = model
            .weights
            .get("layer0_trace_2_res_scale")
            .expect("residual scale constant registered");
        assert_eq!(
            scale.iter().copied().collect::<Vec<f32>>(),
            vec![std::f64::consts::FRAC_1_SQRT_2 as f32]
        );
        assert_builds(model, "FusedAdainResBlock LeakyRelu model");
    }

    /// Dilated conv1 (dilation 3, K=3) expands to a zero-interleaved K=7
    /// kernel with effective dilation 1 — mirroring NN's expansion.
    #[test]
    fn fused_adain_resblock_dilated_conv1_expands_kernel() {
        let graph = resblock_graph_impl(
            ResBlockActivation::LeakyRelu { slope: 0.2 },
            3, // conv1_dilation
            3, // conv1_padding keeps T constant: 2*3 - (3-1)*3 = 0
            1.0,
        );
        let translation = translate_multi_input(&graph).expect("dilated resblock translates");
        let model = &translation.model;

        let conv1 = model
            .network
            .layers
            .iter()
            .find(|l| l.name == "layer0_trace_2_conv1")
            .expect("conv1 present");
        assert_eq!(conv1.layer_type, LayerType::Conv1d);
        assert_eq!(
            conv1.attributes.get("dilations"),
            Some(&AttributeValue::Ints(vec![1])),
            "effective dilation folded to 1"
        );
        let wref = conv1.weights.as_ref().expect("conv1 weight ref");
        assert_eq!(wref.shape, vec![2, 2, 7], "kernel expanded (3-1)*3 + 1 = 7");
        let w = model
            .weights
            .get("layer0_trace_2_conv1_weight")
            .expect("expanded kernel registered");
        assert_eq!(w.shape(), &[2, 2, 7]);
        // Zero-interleaved taps: positions 0, 3, 6 carry the original values.
        let row: Vec<f32> = w
            .index_axis(ndarray::Axis(0), 0)
            .index_axis(ndarray::Axis(0), 0)
            .iter()
            .copied()
            .collect();
        assert_eq!(row, vec![0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1]);
        assert_builds(model, "FusedAdainResBlock dilated model");
    }

    /// FusedAdainResBlock refuses non-rank-3 output shapes (fail closed).
    #[test]
    fn fused_adain_resblock_rank2_refused() {
        let graph = ComputationGraph::from_nodes(vec![
            node(0, "x", TraceOp::Input, &[], &[2, 4]),
            node(1, "s", TraceOp::Input, &[], &[3]),
            node(
                2,
                "rb",
                TraceOp::KokoroFused(KokoroFusedOp::FusedAdainResBlock {
                    activation: ResBlockActivation::LeakyRelu { slope: 0.2 },
                    adain1_weight: WeightPayload::f32(vec![0.1; 4 * 3], vec![4, 3]),
                    adain1_bias: WeightPayload::f32(vec![0.0; 4], vec![4]),
                    adain2_weight: WeightPayload::f32(vec![0.1; 4 * 3], vec![4, 3]),
                    adain2_bias: WeightPayload::f32(vec![0.0; 4], vec![4]),
                    conv1_weight: WeightPayload::f32(vec![0.1; 2 * 2 * 3], vec![2, 2, 3]),
                    conv1_bias: WeightPayload::f32(vec![0.0; 2], vec![2]),
                    conv1_dilation: 1,
                    conv1_padding: 1,
                    conv2_weight: WeightPayload::f32(vec![0.1; 2 * 2 * 3], vec![2, 2, 3]),
                    conv2_bias: WeightPayload::f32(vec![0.0; 2], vec![2]),
                    conv2_padding: 1,
                    eps: 1e-5,
                    residual_scale: 1.0,
                }),
                &[0, 1],
                &[2, 4],
            ),
        ]);
        let err = translate_multi_input(&graph).expect_err("rank-2 output refused");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref m) if m.contains("rank-3")),
            "refusal names the rank requirement, got {err:?}"
        );
    }
}
