// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tensor shape lookup helpers for static cost analysis.

mod common;
mod deterministic;
mod slice;

use self::common::{
    checked_shape_mul, normalize_indices, normalized_axes, reduction_keepdims,
    shape_inference_error,
};
use self::deterministic::{
    infer_concat_shape, infer_reshape_shape, infer_unsqueeze_shape, infer_weighted_matmul_shape,
};
use self::slice::infer_slice_shape;
use std::collections::{BTreeSet, HashMap};

use ny_core::LayerType;

use super::CostError;
use crate::{AttributeValue, LayerSpec, OnnxModel, TensorSpec};

#[derive(Clone, Copy)]
enum MissingOutputShapeStrategy {
    Concat,
    Convolution,
    FirstRuntimeInput,
    Linear,
    Unsqueeze,
    WeightedMatMul,
    MatchingRuntimeInputs,
    Pad,
    Reduction,
    Reshape,
    Slice,
    Transpose,
}

pub(super) struct ShapeLookup<'a> {
    model: &'a OnnxModel,
    /// Shapes computed during layer-by-layer analysis for decomposed-op
    /// intermediate tensors that aren't in ORT's shape inference output.
    computed: HashMap<String, Vec<usize>>,
}

impl<'a> ShapeLookup<'a> {
    pub(super) fn new(model: &'a OnnxModel) -> Self {
        Self {
            model,
            computed: HashMap::new(),
        }
    }

    /// Register a computed output shape so subsequent layers can find it.
    pub(super) fn register_shape(&mut self, name: String, shape: Vec<usize>) {
        self.computed.insert(name, shape);
    }

    pub(super) fn tensor_shape(&self, tensor_name: &str) -> Result<Vec<usize>, CostError> {
        if let Some(shape) = self.computed.get(tensor_name) {
            return Ok(shape.clone());
        }
        if let Some(shape) = self.model.tensor_shapes().get(tensor_name) {
            return normalize_shape(tensor_name, shape);
        }
        if let Some(spec) = self
            .model
            .network
            .inputs
            .iter()
            .find(|spec| spec.name == tensor_name)
        {
            return self.shape_from_spec(spec);
        }
        if let Some(spec) = self
            .model
            .network
            .outputs
            .iter()
            .find(|spec| spec.name == tensor_name)
        {
            return self.shape_from_spec(spec);
        }
        if let Some(weight) = self.model.weights.get(tensor_name) {
            return Ok(weight.shape().to_vec());
        }

        Err(CostError::invalid_input_shape(
            "static cost estimate",
            format!("missing tensor shape for '{tensor_name}'"),
        ))
    }

    pub(super) fn shape_from_spec(&self, spec: &TensorSpec) -> Result<Vec<usize>, CostError> {
        normalize_shape(&spec.name, &spec.shape)
    }

    /// Best-effort shape inference for decomposed-op intermediates.
    ///
    /// Decomposed ops (ReduceL2 → Pow+ReduceSum+Sqrt, LSTM layout transposes)
    /// create intermediates that ORT never saw. We only recover missing shapes
    /// for audited cases where the output geometry is exactly derivable from
    /// runtime inputs plus the layer's own attributes.
    pub(super) fn infer_output_shape(&self, layer: &LayerSpec) -> Result<Vec<usize>, CostError> {
        let Some(strategy) = missing_output_shape_strategy(layer) else {
            return Err(shape_inference_error(
                layer,
                "missing shape metadata is only auto-recovered for audited decomposed layers"
                    .to_string(),
            ));
        };

        let runtime_shapes = self.runtime_input_shapes(layer)?;
        let Some((_, first_shape)) = runtime_shapes.first() else {
            return Err(shape_inference_error(
                layer,
                "no runtime input has a known shape".to_string(),
            ));
        };

        match strategy {
            MissingOutputShapeStrategy::Concat => infer_concat_shape(layer, &runtime_shapes),
            MissingOutputShapeStrategy::Convolution => {
                infer_convolution_shape(self, layer, first_shape)
            }
            MissingOutputShapeStrategy::FirstRuntimeInput => Ok(first_shape.clone()),
            MissingOutputShapeStrategy::Linear => infer_linear_shape(layer, first_shape, self),
            MissingOutputShapeStrategy::Unsqueeze => {
                infer_unsqueeze_shape(self, layer, first_shape)
            }
            MissingOutputShapeStrategy::WeightedMatMul => {
                infer_weighted_matmul_shape(self, layer, first_shape)
            }
            MissingOutputShapeStrategy::MatchingRuntimeInputs => {
                infer_matching_runtime_shape(layer, &runtime_shapes)
            }
            MissingOutputShapeStrategy::Pad => infer_pad_shape(self, layer, first_shape),
            MissingOutputShapeStrategy::Reduction => {
                infer_reduction_shape(self, layer, first_shape)
            }
            MissingOutputShapeStrategy::Reshape => infer_reshape_shape(self, layer, first_shape),
            MissingOutputShapeStrategy::Slice => infer_slice_shape(self, layer, first_shape),
            MissingOutputShapeStrategy::Transpose => infer_transpose_shape(layer, first_shape),
        }
    }

    fn runtime_input_shapes(
        &self,
        layer: &LayerSpec,
    ) -> Result<Vec<(String, Vec<usize>)>, CostError> {
        super::layer_metadata::activation_input_names(self.model, layer)
            .into_iter()
            .map(|name| self.tensor_shape(name).map(|shape| (name.clone(), shape)))
            .collect()
    }
}

#[cfg(all(test, feature = "external-avoice"))]
pub(super) fn layer_supports_missing_output_shape_fallback(layer: &LayerSpec) -> bool {
    missing_output_shape_strategy(layer).is_some()
}

fn missing_output_shape_strategy(layer: &LayerSpec) -> Option<MissingOutputShapeStrategy> {
    match layer.layer_type {
        LayerType::Concat => Some(MissingOutputShapeStrategy::Concat),
        LayerType::Conv1d
        | LayerType::Conv2d
        | LayerType::ConvTranspose1d
        | LayerType::ConvTranspose2d => Some(MissingOutputShapeStrategy::Convolution),
        LayerType::Linear => Some(MissingOutputShapeStrategy::Linear),
        LayerType::ReLU
        | LayerType::LeakyRelu
        | LayerType::GELU
        | LayerType::SiLU
        | LayerType::Sigmoid
        | LayerType::Tanh
        | LayerType::Softplus
        | LayerType::Softmax
        | LayerType::CausalSoftmax
        | LayerType::LogSoftmax
        | LayerType::Clip
        | LayerType::Elu
        | LayerType::Selu
        | LayerType::PRelu
        | LayerType::HardSigmoid
        | LayerType::HardSwish
        | LayerType::Exp
        | LayerType::Log
        | LayerType::Celu
        | LayerType::Mish
        | LayerType::ThresholdedRelu
        | LayerType::Shrink
        | LayerType::Softsign
        | LayerType::Snake
        | LayerType::Floor
        | LayerType::Ceil
        | LayerType::Round
        | LayerType::Sign
        | LayerType::Reciprocal
        | LayerType::Sin
        | LayerType::Cos
        | LayerType::Tan
        | LayerType::Arctan
        | LayerType::Erf
        | LayerType::RoPE
        | LayerType::LayerNorm
        | LayerType::RMSNorm
        | LayerType::InstanceNorm
        | LayerType::GroupNorm
        | LayerType::AdaIN
        | LayerType::BatchNorm
        | LayerType::Neg
        | LayerType::Triu
        | LayerType::Tril
        | LayerType::Abs
        | LayerType::Sqrt => Some(MissingOutputShapeStrategy::FirstRuntimeInput),
        LayerType::Unsqueeze => Some(MissingOutputShapeStrategy::Unsqueeze),
        LayerType::MatMul => Some(MissingOutputShapeStrategy::WeightedMatMul),
        LayerType::Add
        | LayerType::Sub
        | LayerType::Mul
        | LayerType::Div
        | LayerType::Pow
        | LayerType::Min
        | LayerType::Max => Some(MissingOutputShapeStrategy::MatchingRuntimeInputs),
        LayerType::Pad => Some(MissingOutputShapeStrategy::Pad),
        LayerType::ReduceMean | LayerType::ReduceSum => Some(MissingOutputShapeStrategy::Reduction),
        LayerType::Reshape => Some(MissingOutputShapeStrategy::Reshape),
        LayerType::Slice => Some(MissingOutputShapeStrategy::Slice),
        LayerType::Transpose => Some(MissingOutputShapeStrategy::Transpose),
        _ => None,
    }
}

fn infer_matching_runtime_shape(
    layer: &LayerSpec,
    runtime_shapes: &[(String, Vec<usize>)],
) -> Result<Vec<usize>, CostError> {
    let mut result = runtime_shapes[0].1.clone();
    for (_, shape) in runtime_shapes.iter().skip(1) {
        result = broadcast_shapes(layer, &result, shape)?;
    }
    Ok(result)
}

fn broadcast_shapes(
    layer: &LayerSpec,
    lhs: &[usize],
    rhs: &[usize],
) -> Result<Vec<usize>, CostError> {
    let out_rank = lhs.len().max(rhs.len());
    let mut output = vec![1; out_rank];
    for idx in 0..out_rank {
        let lhs_dim = lhs
            .len()
            .checked_sub(out_rank - idx)
            .and_then(|lhs_idx| lhs.get(lhs_idx))
            .copied()
            .unwrap_or(1);
        let rhs_dim = rhs
            .len()
            .checked_sub(out_rank - idx)
            .and_then(|rhs_idx| rhs.get(rhs_idx))
            .copied()
            .unwrap_or(1);
        output[idx] = match (lhs_dim, rhs_dim) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            (a, b) => {
                return Err(shape_inference_error(
                    layer,
                    format!("runtime input shapes {lhs:?} and {rhs:?} cannot broadcast at trailing dim {idx}: {a} vs {b}"),
                ))
            }
        };
    }
    Ok(output)
}

fn infer_reduction_shape(
    lookup: &ShapeLookup<'_>,
    layer: &LayerSpec,
    input_shape: &[usize],
) -> Result<Vec<usize>, CostError> {
    let axes = normalized_axes(lookup, layer, input_shape.len())?;
    let keepdims = reduction_keepdims(layer)?;
    let reduced_axes = axes.into_iter().collect::<BTreeSet<_>>();

    if keepdims {
        return Ok(input_shape
            .iter()
            .enumerate()
            .map(|(idx, &dim)| if reduced_axes.contains(&idx) { 1 } else { dim })
            .collect());
    }

    Ok(input_shape
        .iter()
        .enumerate()
        .filter_map(|(idx, &dim)| (!reduced_axes.contains(&idx)).then_some(dim))
        .collect())
}

fn infer_pad_shape(
    lookup: &ShapeLookup<'_>,
    layer: &LayerSpec,
    input_shape: &[usize],
) -> Result<Vec<usize>, CostError> {
    let pads = pad_values(lookup, layer)?;
    let rank = input_shape.len();
    if pads.len() != rank * 2 {
        return Err(shape_inference_error(
            layer,
            format!(
                "Pad pads length must be twice input rank {rank}, got {}",
                pads.len()
            ),
        ));
    }

    input_shape
        .iter()
        .enumerate()
        .map(|(axis, dim)| {
            let before = parse_nonnegative_pad(layer, pads[axis], "before")?;
            let after = parse_nonnegative_pad(layer, pads[axis + rank], "after")?;
            dim.checked_add(before)
                .and_then(|value| value.checked_add(after))
                .ok_or_else(|| {
                    shape_inference_error(layer, format!("Pad output dim overflow on axis {axis}"))
                })
        })
        .collect()
}

fn pad_values(lookup: &ShapeLookup<'_>, layer: &LayerSpec) -> Result<Vec<i64>, CostError> {
    if layer.inputs.get(1).is_some_and(|name| !name.is_empty()) {
        return lookup.read_i64_tensor(layer, 1, false, "Pad pads");
    }
    match layer.attributes.get("pads") {
        Some(AttributeValue::Ints(values)) => Ok(values.clone()),
        Some(AttributeValue::Int(value)) => Ok(vec![*value]),
        Some(other) => Err(shape_inference_error(
            layer,
            format!("expected integer Pad pads attribute, got {other:?}"),
        )),
        None => Err(shape_inference_error(
            layer,
            "Pad fallback requires constant pads input or pads attribute".to_string(),
        )),
    }
}

fn parse_nonnegative_pad(layer: &LayerSpec, value: i64, label: &str) -> Result<usize, CostError> {
    if value < 0 {
        return Err(shape_inference_error(
            layer,
            format!("Pad {label} value must be non-negative, got {value}"),
        ));
    }
    usize::try_from(value).map_err(|_| {
        shape_inference_error(
            layer,
            format!("Pad {label} value {value} cannot be represented as usize"),
        )
    })
}

fn infer_convolution_shape(
    lookup: &ShapeLookup<'_>,
    layer: &LayerSpec,
    input_shape: &[usize],
) -> Result<Vec<usize>, CostError> {
    let spatial_rank = convolution_spatial_rank(layer)?;
    if input_shape.len() < spatial_rank + 1 {
        return Err(shape_inference_error(
            layer,
            format!(
                "convolution input rank {} is too small for spatial rank {spatial_rank}",
                input_shape.len()
            ),
        ));
    }

    let weight = layer
        .inputs
        .get(1)
        .and_then(|name| lookup.model.weights.get(name))
        .ok_or_else(|| shape_inference_error(layer, "weight tensor is missing".to_string()))?;
    let weight_shape = weight.shape();
    if weight_shape.len() != spatial_rank + 2 {
        return Err(shape_inference_error(
            layer,
            format!(
                "weight tensor rank {} does not match convolution spatial rank {spatial_rank}",
                weight_shape.len()
            ),
        ));
    }

    let group = parse_group(layer)?;
    let channel_axis = input_shape.len() - spatial_rank - 1;
    let (expected_input_channels, output_channels) = match layer.layer_type {
        LayerType::Conv1d | LayerType::Conv2d => (
            checked_shape_mul(weight_shape[1], group, layer)?,
            weight_shape[0],
        ),
        LayerType::ConvTranspose1d | LayerType::ConvTranspose2d => (
            weight_shape[0],
            checked_shape_mul(weight_shape[1], group, layer)?,
        ),
        _ => unreachable!("convolution strategy only covers convolution layers"),
    };
    if input_shape[channel_axis] != expected_input_channels {
        return Err(shape_inference_error(
            layer,
            format!(
                "input channel dim {} does not match weight/group expectation {}",
                input_shape[channel_axis], expected_input_channels
            ),
        ));
    }

    let mut output_shape = Vec::with_capacity(input_shape.len());
    output_shape.extend_from_slice(&input_shape[..channel_axis]);
    output_shape.push(output_channels);

    let output_spatial = if is_transposed_convolution(layer) {
        infer_conv_transpose_spatial(layer, input_shape, weight_shape, spatial_rank)?
    } else {
        infer_conv_spatial(layer, input_shape, weight_shape, spatial_rank)?
    };
    output_shape.extend(output_spatial);
    Ok(output_shape)
}

fn infer_linear_shape(
    layer: &LayerSpec,
    input_shape: &[usize],
    lookup: &ShapeLookup<'_>,
) -> Result<Vec<usize>, CostError> {
    if gemm_bool_attr(layer, "transA", false)? {
        return Err(shape_inference_error(
            layer,
            "Linear fallback does not support transA=1".to_string(),
        ));
    }
    let weight_name = layer
        .inputs
        .get(1)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            shape_inference_error(layer, "Linear is missing its weight input".to_string())
        })?;
    let weight = lookup.model.weights.get(weight_name).ok_or_else(|| {
        shape_inference_error(
            layer,
            format!("Linear weight '{weight_name}' is not a constant tensor"),
        )
    })?;
    let weight_shape = weight.shape();
    if weight_shape.len() != 2 {
        return Err(shape_inference_error(
            layer,
            format!(
                "Linear weight '{weight_name}' must be rank-2 for missing-shape fallback, got {weight_shape:?}"
            ),
        ));
    }

    let trans_b = linear_trans_b(layer, weight_shape)?;
    let (out_features, in_features) = if trans_b {
        (weight_shape[0], weight_shape[1])
    } else {
        (weight_shape[1], weight_shape[0])
    };
    let input_last = *input_shape.last().ok_or_else(|| {
        shape_inference_error(layer, "Linear input must have rank >= 1".to_string())
    })?;
    if input_last != in_features {
        return Err(shape_inference_error(
            layer,
            format!(
                "Linear input last dim {input_last} does not match weight inner dim {in_features}"
            ),
        ));
    }

    let mut output_shape = input_shape.to_vec();
    *output_shape
        .last_mut()
        .expect("validated Linear input rank >= 1") = out_features;
    Ok(output_shape)
}

fn linear_trans_b(layer: &LayerSpec, weight_shape: &[usize]) -> Result<bool, CostError> {
    match layer.attributes.get("transB") {
        Some(AttributeValue::Int(value)) => return Ok(*value != 0),
        Some(other) => {
            return Err(shape_inference_error(
                layer,
                format!("expected integer transB attribute, got {other:?}"),
            ))
        }
        None => {}
    }

    let Some(weight_ref) = &layer.weights else {
        // ONNX Gemm defaults transB to 0: B is [in_features, out_features].
        return Ok(false);
    };

    if weight_ref.shape.len() == 2
        && weight_shape[0] == weight_ref.shape[1]
        && weight_shape[1] == weight_ref.shape[0]
    {
        // Matches ConvertContext::convert_linear's WeightRef heuristic:
        // actual weight is transposed relative to the canonical [out, in] hint.
        return Ok(false);
    }
    Ok(true)
}

fn gemm_bool_attr(layer: &LayerSpec, attr_name: &str, default: bool) -> Result<bool, CostError> {
    match layer.attributes.get(attr_name) {
        None => Ok(default),
        Some(AttributeValue::Int(value)) => Ok(*value != 0),
        Some(other) => Err(shape_inference_error(
            layer,
            format!("expected integer {attr_name} attribute, got {other:?}"),
        )),
    }
}

fn convolution_spatial_rank(layer: &LayerSpec) -> Result<usize, CostError> {
    match layer.layer_type {
        LayerType::Conv1d | LayerType::ConvTranspose1d => Ok(1),
        LayerType::Conv2d | LayerType::ConvTranspose2d => Ok(2),
        _ => Err(shape_inference_error(
            layer,
            format!("{} is not a convolution layer", layer.layer_type),
        )),
    }
}

fn is_transposed_convolution(layer: &LayerSpec) -> bool {
    matches!(
        layer.layer_type,
        LayerType::ConvTranspose1d | LayerType::ConvTranspose2d
    )
}

fn infer_conv_spatial(
    layer: &LayerSpec,
    input_shape: &[usize],
    weight_shape: &[usize],
    spatial_rank: usize,
) -> Result<Vec<usize>, CostError> {
    let pads = parse_pads(layer, spatial_rank)?;
    let strides = parse_positive_spatial_attr(layer, "strides", spatial_rank, 1)?;
    let dilations = parse_positive_spatial_attr(layer, "dilations", spatial_rank, 1)?;
    let input_spatial = &input_shape[input_shape.len() - spatial_rank..];
    let kernel_spatial = &weight_shape[weight_shape.len() - spatial_rank..];

    (0..spatial_rank)
        .map(|dim| {
            let effective_kernel =
                checked_shape_mul(dilations[dim], kernel_spatial[dim].saturating_sub(1), layer)?
                    .checked_add(1)
                    .ok_or_else(|| {
                        shape_inference_error(layer, "effective kernel size overflow".to_string())
                    })?;
            let padded_input = input_spatial[dim]
                .checked_add(pads[dim].0)
                .and_then(|value| value.checked_add(pads[dim].1))
                .ok_or_else(|| {
                    shape_inference_error(layer, "padded input size overflow".to_string())
                })?;
            if padded_input < effective_kernel {
                return Err(shape_inference_error(
                    layer,
                    format!(
                        "padded input dim {padded_input} is smaller than effective kernel {effective_kernel}"
                    ),
                ));
            }
            Ok((padded_input - effective_kernel) / strides[dim] + 1)
        })
        .collect()
}

fn infer_conv_transpose_spatial(
    layer: &LayerSpec,
    input_shape: &[usize],
    weight_shape: &[usize],
    spatial_rank: usize,
) -> Result<Vec<usize>, CostError> {
    if let Some(output_shape) = parse_output_shape_attr(layer, spatial_rank)? {
        return Ok(output_shape);
    }

    let pads = parse_pads(layer, spatial_rank)?;
    let strides = parse_positive_spatial_attr(layer, "strides", spatial_rank, 1)?;
    let dilations = parse_positive_spatial_attr(layer, "dilations", spatial_rank, 1)?;
    let output_padding = parse_nonnegative_spatial_attr(layer, "output_padding", spatial_rank, 0)?;
    let input_spatial = &input_shape[input_shape.len() - spatial_rank..];
    let kernel_spatial = &weight_shape[weight_shape.len() - spatial_rank..];

    (0..spatial_rank)
        .map(|dim| {
            let effective_kernel =
                checked_shape_mul(dilations[dim], kernel_spatial[dim].saturating_sub(1), layer)?
                    .checked_add(1)
                    .ok_or_else(|| {
                        shape_inference_error(layer, "effective kernel size overflow".to_string())
                    })?;
            let base = strides[dim]
                .checked_mul(input_spatial[dim].saturating_sub(1))
                .and_then(|value| value.checked_add(output_padding[dim]))
                .and_then(|value| value.checked_add(effective_kernel))
                .ok_or_else(|| {
                    shape_inference_error(
                        layer,
                        "transposed convolution output size overflow".to_string(),
                    )
                })?;
            let pad_total = pads[dim].0.checked_add(pads[dim].1).ok_or_else(|| {
                shape_inference_error(layer, "padding size overflow".to_string())
            })?;
            let output_dim = base.checked_sub(pad_total).ok_or_else(|| {
                shape_inference_error(
                    layer,
                    format!(
                        "transposed convolution base size {base} is smaller than padding {pad_total}"
                    ),
                )
            })?;
            if output_dim == 0 {
                return Err(shape_inference_error(
                    layer,
                    "transposed convolution produced zero output dimension".to_string(),
                ));
            }
            Ok(output_dim)
        })
        .collect()
}

fn parse_group(layer: &LayerSpec) -> Result<usize, CostError> {
    match layer.attributes.get("group") {
        None => Ok(1),
        Some(AttributeValue::Int(value)) if *value >= 1 => Ok(*value as usize),
        Some(AttributeValue::Int(value)) => Err(shape_inference_error(
            layer,
            format!("group must be >= 1, got {value}"),
        )),
        Some(other) => Err(shape_inference_error(
            layer,
            format!("expected integer group attribute, got {other:?}"),
        )),
    }
}

fn parse_pads(layer: &LayerSpec, spatial_rank: usize) -> Result<Vec<(usize, usize)>, CostError> {
    let values = parse_spatial_attr_values(layer, "pads", spatial_rank, 0, false)?;
    match values.len() {
        len if len == spatial_rank => Ok(values.into_iter().map(|value| (value, value)).collect()),
        len if len == spatial_rank * 2 => Ok((0..spatial_rank)
            .map(|dim| (values[dim], values[dim + spatial_rank]))
            .collect()),
        len => Err(shape_inference_error(
            layer,
            format!(
                "pads length must be 1, {spatial_rank}, or {}, got {len}",
                spatial_rank * 2
            ),
        )),
    }
}

fn parse_positive_spatial_attr(
    layer: &LayerSpec,
    attr_name: &str,
    spatial_rank: usize,
    default: usize,
) -> Result<Vec<usize>, CostError> {
    let values = parse_spatial_attr_values(layer, attr_name, spatial_rank, default, true)?;
    if values.len() == spatial_rank {
        return Ok(values);
    }
    Err(shape_inference_error(
        layer,
        format!(
            "{attr_name} length must be 1 or {spatial_rank}, got {}",
            values.len()
        ),
    ))
}

fn parse_nonnegative_spatial_attr(
    layer: &LayerSpec,
    attr_name: &str,
    spatial_rank: usize,
    default: usize,
) -> Result<Vec<usize>, CostError> {
    let values = parse_spatial_attr_values(layer, attr_name, spatial_rank, default, false)?;
    if values.len() == spatial_rank {
        return Ok(values);
    }
    Err(shape_inference_error(
        layer,
        format!(
            "{attr_name} length must be 1 or {spatial_rank}, got {}",
            values.len()
        ),
    ))
}

fn parse_spatial_attr_values(
    layer: &LayerSpec,
    attr_name: &str,
    spatial_rank: usize,
    default: usize,
    strictly_positive: bool,
) -> Result<Vec<usize>, CostError> {
    let raw_values = match layer.attributes.get(attr_name) {
        None => return Ok(vec![default; spatial_rank]),
        Some(AttributeValue::Int(value)) => vec![*value],
        Some(AttributeValue::Ints(values)) if values.is_empty() => {
            return Ok(vec![default; spatial_rank]);
        }
        Some(AttributeValue::Ints(values)) => values.clone(),
        Some(other) => {
            return Err(shape_inference_error(
                layer,
                format!("expected integer {attr_name} attribute, got {other:?}"),
            ))
        }
    };

    let parsed = raw_values
        .into_iter()
        .map(|value| parse_usize_attr_value(layer, attr_name, value, strictly_positive))
        .collect::<Result<Vec<_>, _>>()?;
    if parsed.len() == 1 {
        Ok(vec![parsed[0]; spatial_rank])
    } else {
        Ok(parsed)
    }
}

fn parse_output_shape_attr(
    layer: &LayerSpec,
    spatial_rank: usize,
) -> Result<Option<Vec<usize>>, CostError> {
    let raw_values = match layer.attributes.get("output_shape") {
        None => return Ok(None),
        Some(AttributeValue::Int(value)) => vec![*value],
        Some(AttributeValue::Ints(values)) if values.is_empty() => return Ok(None),
        Some(AttributeValue::Ints(values)) => values.clone(),
        Some(other) => {
            return Err(shape_inference_error(
                layer,
                format!("expected integer output_shape attribute, got {other:?}"),
            ))
        }
    };
    if raw_values.len() < spatial_rank {
        return Err(shape_inference_error(
            layer,
            format!(
                "output_shape length must be at least spatial rank {spatial_rank}, got {}",
                raw_values.len()
            ),
        ));
    }
    raw_values[raw_values.len() - spatial_rank..]
        .iter()
        .map(|value| parse_usize_attr_value(layer, "output_shape", *value, true))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn parse_usize_attr_value(
    layer: &LayerSpec,
    attr_name: &str,
    value: i64,
    strictly_positive: bool,
) -> Result<usize, CostError> {
    if strictly_positive && value <= 0 {
        return Err(shape_inference_error(
            layer,
            format!("{attr_name} must be positive, got {value}"),
        ));
    }
    if !strictly_positive && value < 0 {
        return Err(shape_inference_error(
            layer,
            format!("{attr_name} must be non-negative, got {value}"),
        ));
    }
    usize::try_from(value).map_err(|_| {
        shape_inference_error(
            layer,
            format!("{attr_name} value {value} cannot be represented as usize"),
        )
    })
}

fn infer_transpose_shape(
    layer: &LayerSpec,
    input_shape: &[usize],
) -> Result<Vec<usize>, CostError> {
    let perm = normalized_permutation(layer, input_shape.len())?;
    Ok(perm.into_iter().map(|idx| input_shape[idx]).collect())
}

fn normalized_permutation(layer: &LayerSpec, rank: usize) -> Result<Vec<usize>, CostError> {
    let raw_perm = match layer.attributes.get("perm") {
        Some(AttributeValue::Ints(values)) => values.clone(),
        Some(AttributeValue::Int(value)) => vec![*value],
        Some(other) => {
            return Err(shape_inference_error(
                layer,
                format!("expected integer transpose permutation, got {other:?}"),
            ))
        }
        None => (0..rank).rev().map(|idx| idx as i64).collect(),
    };
    let perm = normalize_indices(layer, rank, &raw_perm, "perm")?;
    if perm.len() != rank {
        return Err(shape_inference_error(
            layer,
            format!("transpose perm must have rank {rank}, got {}", perm.len()),
        ));
    }
    if perm.iter().collect::<BTreeSet<_>>().len() != rank {
        return Err(shape_inference_error(
            layer,
            format!("transpose perm must be a permutation of 0..{rank}, got {perm:?}"),
        ));
    }
    Ok(perm)
}

fn normalize_shape(name: &str, shape: &[i64]) -> Result<Vec<usize>, CostError> {
    shape
        .iter()
        .map(|dim| {
            if *dim <= 0 {
                return Err(CostError::invalid_input_shape(
                    "static cost estimate",
                    format!(
                        "tensor '{name}' has dynamic or non-positive dimension {dim}; \
                         export a fixed-shape model before running --cost"
                    ),
                ));
            }
            Ok(*dim as usize)
        })
        .collect()
}
