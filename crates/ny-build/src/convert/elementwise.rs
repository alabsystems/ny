// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{LayerType, NyError, Result};
use ny_propagate::layers::{
    AbsLayer, ArctanLayer, CausalSoftmaxLayer, CeilLayer, CeluLayer, ClipLayer, CompareLayer,
    CompareOp, CompareTensorLayer, CosLayer, EluLayer, ErfLayer, ExpLayer, FloorLayer, GELULayer,
    GeluApproximation, HardSigmoidLayer, HardSwishLayer, LeakyReLULayer, LogLayer, LogSoftmaxLayer,
    MishLayer, MulConstantLayer, PReluLayer, ReLULayer, ReciprocalLayer, RoundLayer, SeluLayer,
    ShrinkLayer, SiLULayer, SigmoidLayer, SignLayer, SinLayer, SnakeLayer, SoftmaxLayer,
    SoftplusLayer, SoftsignLayer, SqrtLayer, TanLayer, TanhLayer, ThresholdedReluLayer, TruncLayer,
};
use ny_propagate::Layer;
use std::str::FromStr;
use tracing::debug;

use super::{AttributeValue, ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    /// Resolve a Softmax-family ONNX axis for unbatched propagation.
    ///
    /// Delegates to [`ConvertContext::remap_axis_trailing`]: the axis is
    /// re-expressed TRAILING-RELATIVE (negative), which selects the same
    /// semantic dimension whether the runtime tensor kept its ONNX rank
    /// (leading size-1 retained, e.g. Flatten / rank-2 Gemm outputs) or had
    /// its leading batch dim stripped. The legacy `axis - 1` guess normalized
    /// over the WRONG dimension in the first layout (the same defect class as
    /// the pensieve ReduceSum no-op). Ambiguous cases (unknown recorded rank,
    /// batch axis 0 of a rank>1 tensor) refuse conversion — fail-closed.
    /// Load-time axis interpretation only — no bound math.
    fn resolve_softmax_axis(&self, onnx_axis: i32, op_name: &str, spec: &LayerSpec) -> Result<i32> {
        let data_name = spec.inputs.first().map(String::as_str).ok_or_else(|| {
            NyError::ModelLoad(format!("{op_name} '{}' has no data input", spec.name))
        })?;
        let remapped = self.remap_axis_trailing(
            op_name,
            &spec.name,
            data_name,
            i64::from(onnx_axis),
            super::LegacyBatchAxisPolicy::RejectZero,
        )?;
        i32::try_from(remapped).map_err(|_| {
            NyError::ModelLoad(format!(
                "{op_name} '{}': remapped axis {remapped} does not fit i32",
                spec.name
            ))
        })
    }
}

fn validate_positive_finite_param(
    op_name: &str,
    node_name: &str,
    param_name: &str,
    value: f32,
) -> Result<f32> {
    if !value.is_finite() || value <= 0.0 {
        return Err(NyError::ModelLoad(format!(
            "{op_name} {node_name} invalid {param_name} {value}: {param_name} must be finite and > 0"
        )));
    }
    Ok(value)
}

fn validate_finite_param(
    op_name: &str,
    node_name: &str,
    param_name: &str,
    value: f32,
) -> Result<f32> {
    if !value.is_finite() {
        return Err(NyError::ModelLoad(format!(
            "{op_name} {node_name} invalid {param_name} {value}: {param_name} must be finite"
        )));
    }
    Ok(value)
}

fn validate_clip_bounds(op_name: &str, node_name: &str, min_val: f32, max_val: f32) -> Result<()> {
    if min_val.is_nan() || max_val.is_nan() {
        return Err(NyError::ModelLoad(format!(
            "{op_name} {node_name} invalid bounds min={min_val} max={max_val}: bounds must not be NaN"
        )));
    }
    if min_val > max_val {
        return Err(NyError::ModelLoad(format!(
            "{op_name} {node_name} invalid bounds min={min_val} max={max_val}: min must be <= max"
        )));
    }
    Ok(())
}

fn clip_bound_input(
    context: &ConvertContext<'_>,
    spec: &LayerSpec,
    index: usize,
    bound_name: &str,
) -> Result<Option<f32>> {
    let Some(input_name) = spec.inputs.get(index).filter(|name| !name.is_empty()) else {
        return Ok(None);
    };
    let tensor = context.constant_value(input_name).ok_or_else(|| {
        NyError::UnsupportedOp(format!(
            "Clip {} {bound_name} input '{}' must be a constant scalar",
            spec.name, input_name
        ))
    })?;
    if tensor.ndim() != 0 || tensor.len() != 1 {
        return Err(NyError::UnsupportedOp(format!(
            "Clip {} {bound_name} input '{}' must be a rank-0 scalar, got shape {:?}",
            spec.name,
            input_name,
            tensor.shape()
        )));
    }
    Ok(tensor.iter().next().copied())
}

fn prelu_slope_values(context: &ConvertContext<'_>, spec: &LayerSpec) -> Result<Array1<f32>> {
    if spec.inputs.len() != 2 || spec.inputs.iter().any(String::is_empty) {
        return Err(NyError::ModelLoad(format!(
            "PRelu {} requires exactly two non-empty inputs, got {:?}",
            spec.name, spec.inputs
        )));
    }
    for (name, value) in &spec.attributes {
        if name != "consumed_inputs" || !matches!(value, AttributeValue::Ints(_)) {
            return Err(NyError::ModelLoad(format!(
                "PRelu {} has unsupported attribute '{name}'",
                spec.name
            )));
        }
    }

    let slope_name = &spec.inputs[1];
    let slope = context.constant_value(slope_name).ok_or_else(|| {
        NyError::UnsupportedOp(format!(
            "PRelu {} required slope input '{}' must be constant; dynamic slopes are not represented",
            spec.name, slope_name
        ))
    })?;
    if slope.is_empty() {
        return Err(NyError::ModelLoad(format!(
            "PRelu {} has empty slope tensor from '{}'",
            spec.name, slope_name
        )));
    }
    if slope.iter().any(|value| !value.is_finite()) {
        return Err(NyError::ModelLoad(format!(
            "PRelu {} slope tensor '{}' must contain only finite FLOAT32 values",
            spec.name, slope_name
        )));
    }

    let input_name = &spec.inputs[0];
    let input_shape = context.tensor_shapes.get(input_name);
    let slope_shape = slope.shape();
    if slope.len() == 1 {
        // A rank-0 scalar never changes output shape. Tensor-shaped singleton
        // constants are safe only when their unidirectional broadcast is
        // proven not to add a leading dimension to X.
        if !slope_shape.is_empty() {
            let input_shape = input_shape.ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "PRelu {} cannot authenticate singleton slope shape {:?} against unknown input shape '{}'",
                    spec.name, slope_shape, input_name
                ))
            })?;
            if slope_shape.len() > input_shape.len()
                || slope_shape.iter().rev().zip(input_shape.iter().rev()).any(
                    |(&slope_dim, &input_dim)| {
                        slope_dim != 1 && (input_dim <= 0 || slope_dim as i64 != input_dim)
                    },
                )
            {
                return Err(NyError::UnsupportedOp(format!(
                    "PRelu {} singleton slope shape {:?} is not shape-preserving for input shape {:?}",
                    spec.name, slope_shape, input_shape
                )));
            }
        }
        let scalar = slope.iter().next().copied().ok_or_else(|| {
            NyError::ModelLoad(format!("PRelu {} has empty slope tensor", spec.name))
        })?;
        return Ok(Array1::from_elem(1, scalar));
    }

    let input_shape = input_shape.ok_or_else(|| {
        NyError::UnsupportedOp(format!(
            "PRelu {} requires a known input shape to authenticate per-channel slope shape {:?}",
            spec.name, slope_shape
        ))
    })?;
    let channel_axis = if context.model_unbatched { 0 } else { 1 };
    if input_shape.len() <= channel_axis {
        return Err(NyError::UnsupportedOp(format!(
            "PRelu {} input shape {:?} has no represented channel axis",
            spec.name, input_shape
        )));
    }
    let channels = usize::try_from(input_shape[channel_axis])
        .ok()
        .filter(|channels| *channels > 0)
        .ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "PRelu {} cannot authenticate unresolved channel extent in input shape {:?}",
                spec.name, input_shape
            ))
        })?;
    if slope.len() != channels {
        return Err(NyError::UnsupportedOp(format!(
            "PRelu {} slope shape {:?} has {} values, expected {channels} channels for input shape {:?}",
            spec.name,
            slope_shape,
            slope.len(),
            input_shape
        )));
    }

    let canonical_shape: Vec<usize> = if context.model_unbatched {
        std::iter::once(channels)
            .chain(std::iter::repeat_n(1, input_shape.len().saturating_sub(1)))
            .collect()
    } else {
        std::iter::once(channels)
            .chain(std::iter::repeat_n(1, input_shape.len().saturating_sub(2)))
            .collect()
    };
    let batch_explicit_shape: Vec<usize> = if context.model_unbatched {
        Vec::new()
    } else {
        std::iter::once(1)
            .chain(canonical_shape.iter().copied())
            .collect()
    };
    if slope_shape != canonical_shape.as_slice() && slope_shape != batch_explicit_shape.as_slice() {
        return Err(NyError::UnsupportedOp(format!(
            "PRelu {} slope shape {:?} is not the represented per-channel layout {:?}{} for input shape {:?}; last-axis [C] broadcasting is not channel-wise for rank > 2",
            spec.name,
            slope_shape,
            canonical_shape,
            if batch_explicit_shape.is_empty() {
                String::new()
            } else {
                format!(" or {:?}", batch_explicit_shape)
            },
            input_shape
        )));
    }

    let slope_len = slope.len();
    slope.into_shape_with_order(slope_len).map_err(|error| {
        NyError::ModelLoad(format!(
            "PRelu {} could not flatten authenticated slope '{}': {error}",
            spec.name, slope_name
        ))
    })
}

fn tensor_shape_usize(context: &ConvertContext<'_>, name: &str) -> Option<Vec<usize>> {
    let shape = context.tensor_shapes.get(name)?;
    let mut out = Vec::with_capacity(shape.len());
    for &dim in shape {
        if dim <= 0 {
            return None;
        }
        out.push(dim as usize);
    }
    Some(out)
}

fn triangular_diagonal(spec: &LayerSpec, op_name: &str) -> Result<i64> {
    match spec.attributes.get("diagonal") {
        Some(AttributeValue::Int(diagonal)) => Ok(*diagonal),
        Some(other) => Err(NyError::ModelLoad(format!(
            "{op_name} {} invalid diagonal attribute {other:?}: expected Int",
            spec.name
        ))),
        None => Ok(0),
    }
}

fn generate_triangular_mask(shape: &[usize], diagonal: i64, lower: bool) -> Result<ArrayD<f32>> {
    if shape.len() < 2 {
        return Err(NyError::InvalidSpec(format!(
            "triangular mask requires rank >= 2, got shape {shape:?}"
        )));
    }

    let rank = shape.len();
    Ok(ArrayD::from_shape_fn(IxDyn(shape), |idx| {
        let row = idx[rank - 2] as i64;
        let col = idx[rank - 1] as i64;
        let keep = if lower {
            col <= row + diagonal
        } else {
            col >= row + diagonal
        };
        if keep {
            1.0
        } else {
            0.0
        }
    }))
}

fn convert_triangular(
    context: &ConvertContext<'_>,
    spec: &LayerSpec,
    lower: bool,
) -> Result<Option<Layer>> {
    let op_name = if lower { "Tril" } else { "Triu" };
    let input_name = spec.inputs.first().ok_or_else(|| {
        NyError::ModelLoad(format!(
            "{op_name} {} requires one activation input",
            spec.name
        ))
    })?;
    let diagonal = triangular_diagonal(spec, op_name)?;
    let input_shape = tensor_shape_usize(context, input_name).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "{op_name} {} requires known concrete input shape for mask generation",
            spec.name
        ))
    })?;
    let mask = generate_triangular_mask(&input_shape, diagonal, lower)?;
    let layer = MulConstantLayer::try_with_input_shape(mask, input_shape).map_err(|err| {
        NyError::InvalidSpec(format!("{op_name} {} constant invalid: {err}", spec.name))
    })?;
    Ok(Some(Layer::MulConstant(layer)))
}

fn compare_op_from_spec(spec: &LayerSpec) -> Result<CompareOp> {
    match spec.attributes.get("compare_op") {
        Some(AttributeValue::String(op)) => CompareOp::from_str(op).map_err(|err| {
            NyError::ModelLoad(format!(
                "Compare {} has invalid compare_op '{}': {err}",
                spec.name, op
            ))
        }),
        Some(other) => Err(NyError::ModelLoad(format!(
            "Compare {} invalid compare_op attribute {other:?}: expected String",
            spec.name
        ))),
        None => Ok(CompareOp::Gt),
    }
}

fn reverse_compare_op(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Ge => CompareOp::Le,
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Le => CompareOp::Ge,
        CompareOp::Eq => CompareOp::Eq,
        CompareOp::Ne => CompareOp::Ne,
    }
}

fn scalar_compare_constant(
    spec: &LayerSpec,
    input_name: &str,
    constant: &ArrayD<f32>,
) -> Result<f32> {
    if constant.len() != 1 {
        return Err(NyError::UnsupportedOp(format!(
            "Compare {} only supports scalar constants or two activation inputs; '{}' has shape {:?}",
            spec.name,
            input_name,
            constant.shape()
        )));
    }
    let value = constant.iter().next().copied().unwrap_or(0.0);
    if !value.is_finite() {
        return Err(NyError::ModelLoad(format!(
            "Compare {} constant '{}' must be finite, got {}",
            spec.name, input_name, value
        )));
    }
    Ok(value)
}

impl ConvertContext<'_> {
    pub(crate) fn convert_elementwise(&self, spec: &LayerSpec) -> Result<Option<Layer>> {
        match spec.layer_type {
            LayerType::ReLU => Ok(Some(Layer::ReLU(ReLULayer))),
            LayerType::LeakyRelu => {
                let alpha = match spec.attributes.get("alpha") {
                    Some(AttributeValue::Float(v)) => *v,
                    _ => 0.01,
                };
                let layer = LeakyReLULayer::try_new(alpha)
                    .map_err(|err| NyError::InvalidSpec(format!("LeakyRelu alpha: {err}")))?;
                Ok(Some(Layer::LeakyReLU(layer)))
            }
            LayerType::GELU => {
                let approximation = match spec.attributes.get("approximate") {
                    Some(AttributeValue::String(s)) if s == "tanh" => GeluApproximation::Tanh,
                    _ => GeluApproximation::Erf,
                };
                Ok(Some(Layer::GELU(GELULayer::new(approximation))))
            }
            LayerType::SiLU => {
                // Standard ONNX Swish-24 calls this coefficient `alpha`.
                // Retain `beta` only as a strict custom-domain compatibility
                // spelling. `SiLULayer` implements exactly x * sigmoid(x), so
                // any other value changes the authored function.
                if spec.attributes.contains_key("alpha") && spec.attributes.contains_key("beta") {
                    return Err(NyError::ModelLoad(format!(
                        "Swish/SiLU {} cannot define both alpha and beta",
                        spec.name
                    )));
                }
                for (name, attribute) in &spec.attributes {
                    match (name.as_str(), attribute) {
                        ("alpha" | "beta", AttributeValue::Float(value)) if *value == 1.0 => {}
                        ("alpha" | "beta", AttributeValue::Float(value)) => {
                            return Err(NyError::InvalidSpec(format!(
                                "Swish/SiLU {name} {value} unsupported (expected 1.0)"
                            )));
                        }
                        ("alpha" | "beta", _) => {
                            return Err(NyError::ModelLoad(format!(
                                "Swish/SiLU {} attribute '{name}' must be FLOAT",
                                spec.name
                            )));
                        }
                        _ => {
                            return Err(NyError::ModelLoad(format!(
                                "Swish/SiLU {} has unsupported attribute '{name}'",
                                spec.name
                            )));
                        }
                    }
                }
                Ok(Some(Layer::SiLU(SiLULayer::new())))
            }
            LayerType::Sigmoid => Ok(Some(Layer::Sigmoid(SigmoidLayer))),
            LayerType::Tanh => Ok(Some(Layer::Tanh(TanhLayer))),
            LayerType::Erf => {
                if spec.inputs.len() != 1 || spec.inputs[0].is_empty() {
                    return Err(NyError::ModelLoad(format!(
                        "Erf {} requires exactly one non-empty input, got {:?}",
                        spec.name, spec.inputs
                    )));
                }
                if !spec.attributes.is_empty() {
                    return Err(NyError::ModelLoad(format!(
                        "Erf {} has unsupported attributes {:?}; standard ONNX Erf has none",
                        spec.name,
                        spec.attributes.keys().collect::<Vec<_>>()
                    )));
                }
                Ok(Some(Layer::Erf(ErfLayer)))
            }
            LayerType::Softplus => Ok(Some(Layer::Softplus(SoftplusLayer))),
            LayerType::Clip => {
                if !(1..=3).contains(&spec.inputs.len())
                    || spec.inputs[0].is_empty()
                    || spec.outputs.len() != 1
                    || spec.outputs[0].is_empty()
                {
                    return Err(NyError::ModelLoad(format!(
                        "Clip {} has invalid inputs {:?} or outputs {:?}",
                        spec.name, spec.inputs, spec.outputs
                    )));
                }
                for (name, value) in &spec.attributes {
                    let valid = matches!(
                        (name.as_str(), value),
                        ("min" | "max", AttributeValue::Float(_))
                            | ("consumed_inputs", AttributeValue::Ints(_))
                    );
                    if !valid {
                        return Err(NyError::ModelLoad(format!(
                            "Clip {} has unsupported or wrong-typed attribute '{name}'",
                            spec.name
                        )));
                    }
                }
                let min_input = clip_bound_input(self, spec, 1, "min")?;
                let max_input = clip_bound_input(self, spec, 2, "max")?;
                if spec.attributes.contains_key("min") && min_input.is_some()
                    || spec.attributes.contains_key("max") && max_input.is_some()
                {
                    return Err(NyError::ModelLoad(format!(
                        "Clip {} cannot specify the same bound by both attribute and input",
                        spec.name
                    )));
                }
                let min_val = match spec.attributes.get("min") {
                    Some(AttributeValue::Float(v)) => *v,
                    _ => min_input.unwrap_or(f32::NEG_INFINITY),
                };
                let max_val = match spec.attributes.get("max") {
                    Some(AttributeValue::Float(v)) => *v,
                    _ => max_input.unwrap_or(f32::INFINITY),
                };
                validate_clip_bounds("Clip", &spec.name, min_val, max_val)?;
                Ok(Some(Layer::Clip(ClipLayer::new(min_val, max_val))))
            }
            LayerType::Elu => {
                let alpha = match spec.attributes.get("alpha") {
                    Some(AttributeValue::Float(v)) => *v,
                    _ => 1.0,
                };
                let alpha = validate_positive_finite_param("Elu", &spec.name, "alpha", alpha)?;
                let layer = EluLayer::new(alpha);
                Ok(Some(Layer::Elu(layer)))
            }
            LayerType::Selu => {
                let alpha = match spec.attributes.get("alpha") {
                    Some(AttributeValue::Float(value)) => *value,
                    Some(_) => {
                        return Err(NyError::ModelLoad(format!(
                            "Selu {} attribute 'alpha' must be FLOAT",
                            spec.name
                        )));
                    }
                    None => SeluLayer::ALPHA,
                };
                let gamma = match spec.attributes.get("gamma") {
                    Some(AttributeValue::Float(value)) => *value,
                    Some(_) => {
                        return Err(NyError::ModelLoad(format!(
                            "Selu {} attribute 'gamma' must be FLOAT",
                            spec.name
                        )));
                    }
                    None => SeluLayer::LAMBDA,
                };
                if alpha.to_bits() != SeluLayer::ALPHA.to_bits()
                    || gamma.to_bits() != SeluLayer::LAMBDA.to_bits()
                {
                    return Err(NyError::UnsupportedOp(format!(
                        "Selu {} coefficients alpha={alpha}, gamma={gamma} are unsupported; expected the implemented ONNX defaults alpha={}, gamma={}",
                        spec.name,
                        SeluLayer::ALPHA,
                        SeluLayer::LAMBDA
                    )));
                }
                for name in spec.attributes.keys() {
                    if !matches!(name.as_str(), "alpha" | "gamma") {
                        return Err(NyError::ModelLoad(format!(
                            "Selu {} has unsupported attribute '{name}'",
                            spec.name
                        )));
                    }
                }
                Ok(Some(Layer::Selu(SeluLayer::new())))
            }
            LayerType::PRelu => {
                let slope = prelu_slope_values(self, spec)?;
                debug!("PRelu: authenticated {} slope value(s)", slope.len());
                Ok(Some(Layer::PRelu(PReluLayer::new(slope)?)))
            }
            LayerType::HardSigmoid => {
                let alpha = match spec.attributes.get("alpha") {
                    Some(AttributeValue::Float(v)) => *v,
                    _ => 0.2,
                };
                let beta = match spec.attributes.get("beta") {
                    Some(AttributeValue::Float(v)) => *v,
                    _ => 0.5,
                };
                let alpha =
                    validate_positive_finite_param("HardSigmoid", &spec.name, "alpha", alpha)?;
                let beta = validate_finite_param("HardSigmoid", &spec.name, "beta", beta)?;
                let layer = HardSigmoidLayer::new(alpha, beta);
                Ok(Some(Layer::HardSigmoid(layer)))
            }
            LayerType::HardSwish => Ok(Some(Layer::HardSwish(HardSwishLayer::new()))),
            LayerType::Exp => Ok(Some(Layer::Exp(ExpLayer::new()))),
            LayerType::Log => Ok(Some(Layer::Log(LogLayer::new()))),
            LayerType::Celu => {
                let alpha = match spec.attributes.get("alpha") {
                    Some(AttributeValue::Float(v)) => *v,
                    _ => 1.0,
                };
                let alpha = validate_positive_finite_param("Celu", &spec.name, "alpha", alpha)?;
                let layer = CeluLayer::new(alpha);
                Ok(Some(Layer::Celu(layer)))
            }
            LayerType::Mish => Ok(Some(Layer::Mish(MishLayer::new()))),
            LayerType::LogSoftmax => {
                let onnx_axis = match spec.attributes.get("axis") {
                    Some(AttributeValue::Int(v)) => i32::try_from(*v).map_err(|_| {
                        NyError::InvalidSpec(format!("LogSoftmax axis {} out of range", v))
                    })?,
                    _ => -1,
                };
                let axis = self.resolve_softmax_axis(onnx_axis, "LogSoftmax", spec)?;
                Ok(Some(Layer::LogSoftmax(LogSoftmaxLayer::new(axis))))
            }
            LayerType::ThresholdedRelu => {
                let alpha = match spec.attributes.get("alpha") {
                    Some(AttributeValue::Float(v)) => *v,
                    _ => 1.0,
                };
                let layer = ThresholdedReluLayer::try_new(alpha)
                    .map_err(|err| NyError::InvalidSpec(format!("ThresholdedRelu alpha: {err}")))?;
                Ok(Some(Layer::ThresholdedRelu(layer)))
            }
            LayerType::Shrink => {
                let bias = match spec.attributes.get("bias") {
                    Some(AttributeValue::Float(v)) => *v,
                    _ => 0.0,
                };
                let lambd = match spec.attributes.get("lambd") {
                    Some(AttributeValue::Float(v)) => *v,
                    _ => 0.5,
                };
                let layer = ShrinkLayer::try_new(bias, lambd).map_err(|err| {
                    NyError::ModelLoad(format!(
                        "Shrink {} invalid bias={} lambd={}: {}",
                        spec.name, bias, lambd, err
                    ))
                })?;
                Ok(Some(Layer::Shrink(layer)))
            }
            LayerType::Softsign => Ok(Some(Layer::Softsign(SoftsignLayer::new()))),
            LayerType::Snake => {
                let scalar_alpha = match spec.attributes.get("a") {
                    Some(AttributeValue::Float(v)) => *v,
                    _ => 1.0, // default frequency parameter
                };
                if spec.inputs.len() >= 2 {
                    let alpha_name = &spec.inputs[1];
                    if let Some(alpha_arr) = self.weights.get(alpha_name) {
                        let alpha_1d = alpha_arr
                            .clone()
                            .into_shape_with_order(alpha_arr.len())
                            .map_err(|err| {
                                NyError::ModelLoad(format!(
                                    "Snake {} invalid alpha shape from '{}': {err}",
                                    spec.name, alpha_name
                                ))
                            })?;
                        if alpha_1d.is_empty() {
                            return Err(NyError::ModelLoad(format!(
                                "Snake {} has empty alpha tensor from '{}'",
                                spec.name, alpha_name
                            )));
                        }
                        debug!(
                            "Snake: loaded {} alpha values from {}",
                            alpha_1d.len(),
                            alpha_name
                        );
                        Ok(Some(Layer::Snake(SnakeLayer::per_channel(alpha_1d)?)))
                    } else {
                        debug!(
                            "Snake: alpha {} not found in weights, using scalar attribute {}",
                            alpha_name, scalar_alpha
                        );
                        Ok(Some(Layer::Snake(SnakeLayer::new(scalar_alpha)?)))
                    }
                } else {
                    Ok(Some(Layer::Snake(SnakeLayer::new(scalar_alpha)?)))
                }
            }
            LayerType::Floor => Ok(Some(Layer::Floor(FloorLayer::new()))),
            LayerType::Ceil => Ok(Some(Layer::Ceil(CeilLayer::new()))),
            LayerType::Round => Ok(Some(Layer::Round(RoundLayer::new()))),
            // Native ONNX Trunc and the guarded lowering of ONNX Cast to an
            // integer dtype share the same point function, but not the same
            // domain. Floating-point-to-fixed-point Cast is undefined outside
            // the destination range, so retain its `to` attribute and require
            // every verdict-bearing propagation to prove the input finite and
            // in range. A plain Trunc has no `to` attribute and stays
            // unrestricted over finite f32 values.
            LayerType::Trunc => {
                let layer = match spec.attributes.get("to") {
                    None => TruncLayer::new(),
                    Some(AttributeValue::Int(6)) => TruncLayer::for_int32_cast(),
                    Some(AttributeValue::Int(7)) => TruncLayer::for_int64_cast(),
                    Some(other) => {
                        return Err(NyError::ModelLoad(format!(
                            "Trunc layer '{}' carries invalid Cast target attribute {other:?}; \
                             only INT32(6) and INT64(7) may lower through guarded Trunc",
                            spec.name
                        )));
                    }
                };
                Ok(Some(Layer::Trunc(layer)))
            }
            LayerType::Sign => Ok(Some(Layer::Sign(SignLayer::new()))),
            LayerType::Reciprocal => Ok(Some(Layer::Reciprocal(ReciprocalLayer::new()))),
            LayerType::Sin => Ok(Some(Layer::Sin(SinLayer::new()))),
            LayerType::Cos => Ok(Some(Layer::Cos(CosLayer::new()))),
            LayerType::Tan => Ok(Some(Layer::Tan(TanLayer::new()))),
            LayerType::Arctan => Ok(Some(Layer::Arctan(ArctanLayer::new()))),
            LayerType::Softmax => {
                let onnx_axis = match spec.attributes.get("axis") {
                    Some(AttributeValue::Int(v)) => i32::try_from(*v).map_err(|_| {
                        NyError::InvalidSpec(format!("Softmax axis {} out of range", v))
                    })?,
                    _ => -1,
                };
                let axis = self.resolve_softmax_axis(onnx_axis, "Softmax", spec)?;
                Ok(Some(Layer::Softmax(SoftmaxLayer::new(axis))))
            }
            LayerType::CausalSoftmax => {
                let onnx_axis = match spec.attributes.get("axis") {
                    Some(AttributeValue::Int(v)) => i32::try_from(*v).map_err(|_| {
                        NyError::InvalidSpec(format!("CausalSoftmax axis {} out of range", v))
                    })?,
                    _ => -1,
                };
                let axis = self.resolve_softmax_axis(onnx_axis, "CausalSoftmax", spec)?;
                Ok(Some(Layer::CausalSoftmax(CausalSoftmaxLayer::new(axis))))
            }
            LayerType::Neg => Ok(Some(Layer::MulConstant(MulConstantLayer::scalar(-1.0)))),
            LayerType::Triu => convert_triangular(self, spec, false),
            LayerType::Tril => convert_triangular(self, spec, true),
            LayerType::Abs => Ok(Some(Layer::Abs(AbsLayer))),
            LayerType::Sqrt => Ok(Some(Layer::Sqrt(SqrtLayer))),
            LayerType::Compare => {
                if spec.inputs.len() != 2 {
                    return Err(NyError::ModelLoad(format!(
                        "Compare {} requires exactly 2 inputs, got {}",
                        spec.name,
                        spec.inputs.len()
                    )));
                }
                let lhs_name = &spec.inputs[0];
                let rhs_name = &spec.inputs[1];
                let lhs_constant = self.constant_value(lhs_name);
                let rhs_constant = self.constant_value(rhs_name);
                if lhs_constant.is_some() && rhs_constant.is_some() {
                    return Err(NyError::UnsupportedOp(format!(
                        "Compare {} has both constant inputs — should be constant-folded before conversion",
                        spec.name
                    )));
                }
                let op = compare_op_from_spec(spec)?;
                if let Some(constant) = rhs_constant.as_ref() {
                    let threshold = scalar_compare_constant(spec, rhs_name, constant)?;
                    return Ok(Some(Layer::Compare(CompareLayer::new(threshold, op))));
                }
                if let Some(constant) = lhs_constant.as_ref() {
                    let threshold = scalar_compare_constant(spec, lhs_name, constant)?;
                    return Ok(Some(Layer::Compare(CompareLayer::new(
                        threshold,
                        reverse_compare_op(op),
                    ))));
                }
                Ok(Some(Layer::CompareTensor(CompareTensorLayer::new(op))))
            }
            LayerType::CompareTensor => {
                if spec.inputs.len() != 2 {
                    return Err(NyError::ModelLoad(format!(
                        "CompareTensor {} requires exactly 2 inputs, got {}",
                        spec.name,
                        spec.inputs.len()
                    )));
                }
                Ok(Some(Layer::CompareTensor(CompareTensorLayer::new(
                    compare_op_from_spec(spec)?,
                ))))
            }
            _ => Ok(None),
        }
    }
}
