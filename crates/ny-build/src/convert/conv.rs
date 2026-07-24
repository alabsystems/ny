// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_propagate::layers::{Conv1dLayer, Conv2dLayer, ConvTranspose1dLayer, ConvTranspose2dLayer};
use ny_propagate::Layer;
use tracing::debug;

use super::{AttributeValue, ConvertContext, LayerSpec};

fn parse_non_negative_usize(
    value: i64,
    op_name: &str,
    layer_name: &str,
    attr_name: &str,
) -> Result<usize> {
    if value < 0 {
        return Err(NyError::ModelLoad(format!(
            "{} layer {} has negative {} value {}",
            op_name, layer_name, attr_name, value
        )));
    }
    Ok(value as usize)
}

/// Parse a strictly positive integer (>= 1) for stride attributes.
///
/// Strides of 0 cause divide-by-zero in output-size formulas.
/// Defense-in-depth: constructors also reject stride=0, but catching it
/// at the conversion boundary prevents invalid values from propagating.
/// Part of #2828.
fn parse_positive_usize(
    value: i64,
    op_name: &str,
    layer_name: &str,
    attr_name: &str,
) -> Result<usize> {
    if value <= 0 {
        return Err(NyError::ModelLoad(format!(
            "{} layer {} has non-positive {} value {} (must be >= 1)",
            op_name, layer_name, attr_name, value
        )));
    }
    Ok(value as usize)
}

/// Parse stride attributes, enforcing strictly positive values (>= 1).
///
/// Strides must be >= 1 to prevent divide-by-zero in output-size formulas.
/// Part of #2828.
fn parse_stride_attr(spec: &LayerSpec, op_name: &str, spatial_rank: usize) -> Result<Vec<usize>> {
    parse_positive_spatial_attr(spec, op_name, spatial_rank, "strides")
}

fn parse_dilation_attr(spec: &LayerSpec, op_name: &str, spatial_rank: usize) -> Result<Vec<usize>> {
    parse_positive_spatial_attr(spec, op_name, spatial_rank, "dilations")
}

fn parse_positive_spatial_attr(
    spec: &LayerSpec,
    op_name: &str,
    spatial_rank: usize,
    attr_name: &str,
) -> Result<Vec<usize>> {
    let raw_values: Vec<i64> = match spec.attributes.get(attr_name) {
        None => return Ok(vec![1; spatial_rank]),
        Some(AttributeValue::Int(v)) => vec![*v],
        Some(AttributeValue::Ints(v)) if v.is_empty() => return Ok(vec![1; spatial_rank]),
        Some(AttributeValue::Ints(v)) => v.clone(),
        Some(_) => {
            return Err(NyError::ModelLoad(format!(
                "{} layer {} has invalid {} attribute type",
                op_name, spec.name, attr_name
            )));
        }
    };

    let expanded = if raw_values.len() == 1 {
        vec![raw_values[0]; spatial_rank]
    } else if raw_values.len() == spatial_rank {
        raw_values
    } else {
        return Err(NyError::ModelLoad(format!(
            "{} layer {} has invalid {} length {}, expected 1 or {}",
            op_name,
            spec.name,
            attr_name,
            raw_values.len(),
            spatial_rank
        )));
    };

    expanded
        .into_iter()
        .map(|v| parse_positive_usize(v, op_name, &spec.name, attr_name))
        .collect()
}

fn parse_symmetric_padding_1d(spec: &LayerSpec, op_name: &str) -> Result<usize> {
    let values = match spec.attributes.get("pads") {
        None => return Ok(0),
        Some(AttributeValue::Int(v)) => vec![*v],
        Some(AttributeValue::Ints(v)) if v.is_empty() => return Ok(0),
        Some(AttributeValue::Ints(v)) => v.clone(),
        Some(_) => {
            return Err(NyError::ModelLoad(format!(
                "{} layer {} has invalid pads attribute type",
                op_name, spec.name
            )));
        }
    };

    match values.as_slice() {
        [p] => parse_non_negative_usize(*p, op_name, &spec.name, "pads"),
        [begin, end] if begin == end => {
            parse_non_negative_usize(*begin, op_name, &spec.name, "pads")
        }
        [_, _] => Err(NyError::UnsupportedConfiguration(format!(
            "{} layer {} uses asymmetric pads {:?}, which is not supported",
            op_name, spec.name, values
        ))),
        _ => Err(NyError::ModelLoad(format!(
            "{} layer {} has invalid pads length {}, expected 1 or 2",
            op_name,
            spec.name,
            values.len()
        ))),
    }
}

fn parse_symmetric_padding_2d(spec: &LayerSpec, op_name: &str) -> Result<(usize, usize)> {
    let values = match spec.attributes.get("pads") {
        None => return Ok((0, 0)),
        Some(AttributeValue::Int(v)) => vec![*v],
        Some(AttributeValue::Ints(v)) if v.is_empty() => return Ok((0, 0)),
        Some(AttributeValue::Ints(v)) => v.clone(),
        Some(_) => {
            return Err(NyError::ModelLoad(format!(
                "{} layer {} has invalid pads attribute type",
                op_name, spec.name
            )));
        }
    };

    match values.as_slice() {
        [p] => {
            let pad = parse_non_negative_usize(*p, op_name, &spec.name, "pads")?;
            Ok((pad, pad))
        }
        [h, w] => Ok((
            parse_non_negative_usize(*h, op_name, &spec.name, "pads")?,
            parse_non_negative_usize(*w, op_name, &spec.name, "pads")?,
        )),
        [h_begin, w_begin, h_end, w_end] if h_begin == h_end && w_begin == w_end => Ok((
            parse_non_negative_usize(*h_begin, op_name, &spec.name, "pads")?,
            parse_non_negative_usize(*w_begin, op_name, &spec.name, "pads")?,
        )),
        [_, _, _, _] => Err(NyError::UnsupportedConfiguration(format!(
            "{} layer {} uses asymmetric pads {:?}, which is not supported",
            op_name, spec.name, values
        ))),
        _ => Err(NyError::ModelLoad(format!(
            "{} layer {} has invalid pads length {}, expected 1, 2, or 4",
            op_name,
            spec.name,
            values.len()
        ))),
    }
}

/// Parse the ONNX `output_padding` attribute for a ConvTranspose layer.
///
/// Defaults to zeros. Length 1 broadcasts across all spatial dims; length
/// `spatial_rank` is taken as-is. Reference: ONNX ConvTranspose `output_padding`.
fn parse_output_padding_attr(
    spec: &LayerSpec,
    op_name: &str,
    spatial_rank: usize,
) -> Result<Vec<usize>> {
    let raw_values: Vec<i64> = match spec.attributes.get("output_padding") {
        None => return Ok(vec![0; spatial_rank]),
        Some(AttributeValue::Int(v)) => vec![*v],
        Some(AttributeValue::Ints(v)) if v.is_empty() => return Ok(vec![0; spatial_rank]),
        Some(AttributeValue::Ints(v)) => v.clone(),
        Some(_) => {
            return Err(NyError::ModelLoad(format!(
                "{} layer {} has invalid output_padding attribute type",
                op_name, spec.name
            )));
        }
    };

    let expanded = if raw_values.len() == 1 {
        vec![raw_values[0]; spatial_rank]
    } else if raw_values.len() == spatial_rank {
        raw_values
    } else {
        return Err(NyError::ModelLoad(format!(
            "{} layer {} has invalid output_padding length {}, expected 1 or {}",
            op_name,
            spec.name,
            raw_values.len(),
            spatial_rank
        )));
    };

    expanded
        .into_iter()
        .map(|v| parse_non_negative_usize(v, op_name, &spec.name, "output_padding"))
        .collect()
}

/// Read the ONNX `output_shape` spatial dimensions for a ConvTranspose layer,
/// if present and non-empty.
///
/// `output_shape` may be given as the full spatial output size. When set, it is
/// an alternative to `output_padding`; ONNX derives output_padding from it.
fn parse_output_shape_attr(
    spec: &LayerSpec,
    op_name: &str,
    spatial_rank: usize,
) -> Result<Option<Vec<usize>>> {
    let raw_values: Vec<i64> = match spec.attributes.get("output_shape") {
        None => return Ok(None),
        Some(AttributeValue::Ints(v)) if v.is_empty() => return Ok(None),
        Some(AttributeValue::Ints(v)) => v.clone(),
        Some(AttributeValue::Int(v)) => vec![*v],
        Some(_) => {
            return Err(NyError::ModelLoad(format!(
                "{} layer {} has invalid output_shape attribute type",
                op_name, spec.name
            )));
        }
    };

    // output_shape may include leading batch/channel dims (rank = 2 + spatial)
    // or just the spatial dims. Take the trailing `spatial_rank` entries.
    if raw_values.len() < spatial_rank {
        return Err(NyError::ModelLoad(format!(
            "{} layer {} has output_shape length {}, expected at least {}",
            op_name,
            spec.name,
            raw_values.len(),
            spatial_rank
        )));
    }
    let spatial = &raw_values[raw_values.len() - spatial_rank..];
    let parsed = spatial
        .iter()
        .map(|&v| parse_non_negative_usize(v, op_name, &spec.name, "output_shape"))
        .collect::<Result<Vec<usize>>>()?;
    Ok(Some(parsed))
}

impl ConvertContext<'_> {
    fn ensure_supported_conv_attributes(
        &self,
        spec: &LayerSpec,
        op_name: &str,
        _spatial_rank: usize,
    ) -> Result<()> {
        if let Some(attr) = spec.attributes.get("auto_pad") {
            let auto_pad = match attr {
                AttributeValue::String(s) => s.trim().to_ascii_uppercase(),
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "{} layer {} has invalid auto_pad attribute type",
                        op_name, spec.name
                    )));
                }
            };
            if !auto_pad.is_empty() && auto_pad != "NOTSET" {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "{} layer {} uses auto_pad={}, which is not supported",
                    op_name, spec.name, auto_pad
                )));
            }
        }

        Ok(())
    }

    /// Parse the ONNX `group` attribute, defaulting to 1.
    /// Reference: ONNX Conv operator spec.
    fn parse_group_attr(spec: &LayerSpec, op_name: &str) -> Result<usize> {
        if let Some(attr) = spec.attributes.get("group") {
            let group = match attr {
                AttributeValue::Int(v) => *v,
                _ => {
                    return Err(NyError::ModelLoad(format!(
                        "{} layer {} has invalid group attribute type",
                        op_name, spec.name
                    )));
                }
            };
            if group < 1 {
                return Err(NyError::InvalidSpec(format!(
                    "{} layer {} has group={}, must be >= 1",
                    op_name, spec.name, group
                )));
            }
            Ok(group as usize)
        } else {
            Ok(1)
        }
    }

    fn infer_conv1d_input_length(&self, input_name: &str) -> Option<usize> {
        self.tensor_shapes
            .get(input_name)
            .and_then(|shape| shape.last().copied())
            .filter(|&length| length > 0)
            .and_then(|length| usize::try_from(length).ok())
    }

    /// Infer the trailing two spatial dimensions (height, width) of a 2D conv
    /// input from known static shapes. Returns `[Some(h), Some(w)]` when both
    /// are statically known and positive, otherwise `None` for the unknown dim.
    fn infer_conv2d_input_spatial(&self, input_name: &str) -> [Option<usize>; 2] {
        let Some(shape) = self.tensor_shapes.get(input_name) else {
            return [None, None];
        };
        if shape.len() < 2 {
            return [None, None];
        }
        let to_dim = |v: i64| -> Option<usize> {
            if v > 0 {
                usize::try_from(v).ok()
            } else {
                None
            }
        };
        let h = to_dim(shape[shape.len() - 2]);
        let w = to_dim(shape[shape.len() - 1]);
        [h, w]
    }

    pub(crate) fn convert_conv1d(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "Conv1d layer {} has fewer than 2 inputs",
                spec.name
            )));
        }

        let kernel_name = &spec.inputs[1];
        let kernel = self
            .constant_value(kernel_name)
            .ok_or_else(|| NyError::ModelLoad(format!("Kernel {} not found", kernel_name)))?;

        if kernel.ndim() != 3 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: kernel.shape().to_vec(),
            });
        }

        let bias = if spec.inputs.len() >= 3 {
            let bias_name = &spec.inputs[2];
            let bias_arr = self
                .constant_value(bias_name)
                .ok_or_else(|| NyError::ModelLoad(format!("Bias {} not found", bias_name)))?;
            let bias_1d = bias_arr
                .clone()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    NyError::shape_mismatch(vec![kernel.shape()[0]], bias_arr.shape().to_vec())
                })?;
            Some(bias_1d)
        } else {
            None
        };

        self.ensure_supported_conv_attributes(spec, "Conv1d", 1)?;
        let groups = Self::parse_group_attr(spec, "Conv1d")?;
        let stride = parse_stride_attr(spec, "Conv1d", 1)?[0];
        let dilation = parse_dilation_attr(spec, "Conv1d", 1)?[0];
        let padding = parse_symmetric_padding_1d(spec, "Conv1d")?;

        debug!(
            "Conv1d: kernel {:?}, stride {}, padding {}, dilation {}, groups {}, input_length {:?}",
            kernel.shape(),
            stride,
            padding,
            dilation,
            groups,
            self.infer_conv1d_input_length(&spec.inputs[0]),
        );
        let mut conv = Conv1dLayer::new_full(kernel, bias, stride, padding, dilation, groups)?;
        if let Some(input_length) = self.infer_conv1d_input_length(&spec.inputs[0]) {
            conv.set_input_length(input_length);
        }
        Ok(Layer::Conv1d(conv))
    }
    pub(crate) fn convert_conv2d(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "Conv2d layer {} has fewer than 2 inputs",
                spec.name
            )));
        }

        let kernel_name = &spec.inputs[1];
        let kernel = self
            .constant_value(kernel_name)
            .ok_or_else(|| NyError::ModelLoad(format!("Kernel {} not found", kernel_name)))?;

        if kernel.ndim() != 4 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: kernel.shape().to_vec(),
            });
        }

        let bias = if spec.inputs.len() >= 3 {
            let bias_name = &spec.inputs[2];
            let bias_arr = self
                .constant_value(bias_name)
                .ok_or_else(|| NyError::ModelLoad(format!("Bias {} not found", bias_name)))?;
            let bias_1d = bias_arr
                .clone()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    NyError::shape_mismatch(vec![kernel.shape()[0]], bias_arr.shape().to_vec())
                })?;
            Some(bias_1d)
        } else {
            None
        };

        self.ensure_supported_conv_attributes(spec, "Conv2d", 2)?;
        let groups = Self::parse_group_attr(spec, "Conv2d")?;
        let dilation_values = parse_dilation_attr(spec, "Conv2d", 2)?;
        let dilation = (dilation_values[0], dilation_values[1]);
        let stride_values = parse_stride_attr(spec, "Conv2d", 2)?;
        let stride = (stride_values[0], stride_values[1]);
        let padding = parse_symmetric_padding_2d(spec, "Conv2d")?;

        debug!(
            "Conv2d: kernel {:?}, stride {:?}, padding {:?}, dilation {:?}, groups {}",
            kernel.shape(),
            stride,
            padding,
            dilation,
            groups,
        );
        let mut conv = Conv2dLayer::new_dilated(kernel, bias, stride, padding, dilation, groups)?;
        // Thread the ONNX-inferred input spatial shape (H, W) into the layer so
        // downstream consumers that require it — notably the MIP Conv2d→Linear
        // unfolding for the complete verifier — can recover the spatial dims even
        // when the model input is flat (e.g. Gemm→Reshape→Conv). The shape comes
        // from `tensor_shapes`, populated by ONNX shape inference at load time.
        if let [Some(in_h), Some(in_w)] = self.infer_conv2d_input_spatial(&spec.inputs[0]) {
            conv.set_input_shape(in_h, in_w);
        }
        Ok(Layer::Conv2d(conv))
    }
    fn ensure_conv_transpose_attributes(
        &self,
        spec: &LayerSpec,
        spatial_rank: usize,
    ) -> Result<()> {
        // output_padding and output_shape are now supported (threaded through the
        // layer struct / propagation); only auto_pad remains unsupported.
        self.ensure_supported_conv_attributes(spec, "ConvTranspose", spatial_rank)
    }

    /// Resolve the per-dimension `output_padding` for a ConvTranspose layer,
    /// preferring an explicit `output_padding` attribute, otherwise deriving it
    /// from `output_shape` if given.
    ///
    /// Given `output_shape`, ONNX derives:
    ///   output_padding = output_shape
    ///       - (stride*(in-1) + dilation*(kernel-1) + 1 - 2*pad)
    /// Reference: ONNX ConvTranspose operator spec.
    #[allow(clippy::too_many_arguments)]
    fn resolve_output_padding(
        spec: &LayerSpec,
        op_name: &str,
        spatial_rank: usize,
        input_spatial: &[Option<usize>],
        kernel_spatial: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
    ) -> Result<Vec<usize>> {
        let explicit = parse_output_padding_attr(spec, op_name, spatial_rank)?;
        let Some(output_shape) = parse_output_shape_attr(spec, op_name, spatial_rank)? else {
            return Ok(explicit);
        };

        // Derive output_padding from output_shape. Requires known input dims.
        let mut derived = vec![0usize; spatial_rank];
        for d in 0..spatial_rank {
            let in_dim = input_spatial[d].ok_or_else(|| {
                NyError::UnsupportedConfiguration(format!(
                    "{} layer {} specifies output_shape but input spatial dim {} is unknown; \
                     cannot derive output_padding",
                    op_name, spec.name, d
                ))
            })?;
            let eff_k = dilation[d] * (kernel_spatial[d] - 1) + 1;
            let base = stride[d] * (in_dim.saturating_sub(1)) + eff_k;
            let base = base.checked_sub(2 * padding[d]).ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "{} layer {} output_shape derivation underflow in dim {}",
                    op_name, spec.name, d
                ))
            })?;
            if output_shape[d] < base {
                return Err(NyError::ModelLoad(format!(
                    "{} layer {} output_shape {} is smaller than minimum output {} in dim {}",
                    op_name, spec.name, output_shape[d], base, d
                )));
            }
            derived[d] = output_shape[d] - base;
        }
        Ok(derived)
    }
    pub(crate) fn convert_conv_transpose1d(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "ConvTranspose1d layer {} has fewer than 2 inputs",
                spec.name
            )));
        }

        self.ensure_conv_transpose_attributes(spec, 1)?;
        // ConvTranspose1dLayer does not carry an output_padding/output_shape
        // field, so reject nonzero values rather than silently ignoring them
        // (which would corrupt bounds). 2D ConvTranspose supports these.
        if parse_output_padding_attr(spec, "ConvTranspose1d", 1)?
            .iter()
            .any(|&v| v != 0)
        {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose1d layer {} uses nonzero output_padding, which is not supported",
                spec.name
            )));
        }
        if parse_output_shape_attr(spec, "ConvTranspose1d", 1)?.is_some() {
            return Err(NyError::UnsupportedConfiguration(format!(
                "ConvTranspose1d layer {} specifies output_shape, which is not supported",
                spec.name
            )));
        }
        let groups = Self::parse_group_attr(spec, "ConvTranspose1d")?;

        let kernel_name = &spec.inputs[1];
        let kernel = self
            .constant_value(kernel_name)
            .ok_or_else(|| NyError::ModelLoad(format!("Kernel {} not found", kernel_name)))?;

        if kernel.ndim() != 3 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0, 0],
                got: kernel.shape().to_vec(),
            });
        }

        let bias = if spec.inputs.len() >= 3 {
            let bias_name = &spec.inputs[2];
            let bias_arr = self
                .constant_value(bias_name)
                .ok_or_else(|| NyError::ModelLoad(format!("Bias {} not found", bias_name)))?;
            let bias_1d = bias_arr
                .clone()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    NyError::shape_mismatch(vec![kernel.shape()[1]], bias_arr.shape().to_vec())
                })?;
            Some(bias_1d)
        } else {
            None
        };

        let stride = parse_stride_attr(spec, "ConvTranspose1d", 1)?[0];
        let padding = parse_symmetric_padding_1d(spec, "ConvTranspose1d")?;
        let dilation = parse_dilation_attr(spec, "ConvTranspose1d", 1)?[0];

        debug!(
            "ConvTranspose1d: kernel {:?}, stride {}, padding {}, dilation {}, groups {}, input_length {:?}",
            kernel.shape(),
            stride,
            padding,
            dilation,
            groups,
            self.infer_conv1d_input_length(&spec.inputs[0]),
        );
        let mut conv =
            ConvTranspose1dLayer::new_full(kernel, bias, stride, padding, dilation, groups)?;
        if let Some(input_length) = self.infer_conv1d_input_length(&spec.inputs[0]) {
            conv.set_input_length(input_length);
        }
        Ok(Layer::ConvTranspose1d(conv))
    }
    pub(crate) fn convert_conv_transpose2d(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "ConvTranspose2d layer {} has fewer than 2 inputs",
                spec.name
            )));
        }

        self.ensure_conv_transpose_attributes(spec, 2)?;
        let dilation_values = parse_dilation_attr(spec, "ConvTranspose2d", 2)?;
        let dilation = (dilation_values[0], dilation_values[1]);

        let kernel_name = &spec.inputs[1];
        let kernel = self
            .constant_value(kernel_name)
            .ok_or_else(|| NyError::ModelLoad(format!("Kernel {} not found", kernel_name)))?;

        if kernel.ndim() != 4 {
            return Err(NyError::ShapeMismatch {
                expected: vec![0, 0, 0, 0],
                got: kernel.shape().to_vec(),
            });
        }

        let bias = if spec.inputs.len() >= 3 {
            let bias_name = &spec.inputs[2];
            let bias_arr = self
                .constant_value(bias_name)
                .ok_or_else(|| NyError::ModelLoad(format!("Bias {} not found", bias_name)))?;
            let bias_1d = bias_arr
                .clone()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    NyError::shape_mismatch(vec![kernel.shape()[1]], bias_arr.shape().to_vec())
                })?;
            Some(bias_1d)
        } else {
            None
        };

        let stride_values = parse_stride_attr(spec, "ConvTranspose2d", 2)?;
        let stride = (stride_values[0], stride_values[1]);
        let padding = parse_symmetric_padding_2d(spec, "ConvTranspose2d")?;

        // Kernel layout for ONNX ConvTranspose: (in_c, out_c, kh, kw).
        let kernel_spatial = [kernel.shape()[2], kernel.shape()[3]];
        let input_spatial = self.infer_conv2d_input_spatial(&spec.inputs[0]);
        let output_padding_values = Self::resolve_output_padding(
            spec,
            "ConvTranspose2d",
            2,
            &input_spatial,
            &kernel_spatial,
            &stride_values,
            &[padding.0, padding.1],
            &dilation_values,
        )?;
        let output_padding = (output_padding_values[0], output_padding_values[1]);

        debug!(
            "ConvTranspose2d: kernel {:?}, stride {:?}, padding {:?}, dilation {:?}, output_padding {:?}",
            kernel.shape(),
            stride,
            padding,
            dilation,
            output_padding,
        );
        let conv = ConvTranspose2dLayer::new_full(
            kernel,
            bias,
            stride,
            padding,
            dilation,
            output_padding,
        )?;
        Ok(Layer::ConvTranspose2d(conv))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};
    use ny_core::{LayerType, NyError};
    use ny_propagate::Layer;
    use std::collections::HashMap;

    use super::ConvertContext;
    use crate::{AttributeValue, LayerSpec, WeightStore};
    use std::collections::HashSet;

    fn make_context(weights: &WeightStore) -> ConvertContext<'_> {
        // Static empty maps for test contexts
        static SHAPES: std::sync::LazyLock<HashMap<String, Vec<i64>>> =
            std::sync::LazyLock::new(HashMap::new);
        static CONSTANTS: std::sync::LazyLock<HashSet<String>> =
            std::sync::LazyLock::new(HashSet::new);
        ConvertContext::new(weights, &SHAPES, &CONSTANTS)
    }

    fn make_context_with_evaluated<'a>(
        weights: &'a WeightStore,
        evaluated: &'a HashMap<String, ArrayD<f32>>,
    ) -> ConvertContext<'a> {
        static SHAPES: std::sync::LazyLock<HashMap<String, Vec<i64>>> =
            std::sync::LazyLock::new(HashMap::new);
        static CONSTANTS: std::sync::LazyLock<HashSet<String>> =
            std::sync::LazyLock::new(HashSet::new);
        ConvertContext::with_evaluated_constants(weights, &SHAPES, &CONSTANTS, evaluated)
    }

    fn make_context_with_shapes<'a>(
        weights: &'a WeightStore,
        tensor_shapes: &'a HashMap<String, Vec<i64>>,
    ) -> ConvertContext<'a> {
        static CONSTANTS: std::sync::LazyLock<HashSet<String>> =
            std::sync::LazyLock::new(HashSet::new);
        ConvertContext::new(weights, tensor_shapes, &CONSTANTS)
    }

    fn conv1d_spec(attrs: HashMap<String, AttributeValue>) -> (WeightStore, LayerSpec) {
        let mut ws = WeightStore::new();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 3, 3]), vec![0.0; 2 * 3 * 3])
            .expect("invariant: valid shape");
        let bias =
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0; 2]).expect("invariant: valid shape");
        ws.insert("kernel".to_string(), kernel);
        ws.insert("bias".to_string(), bias);

        let spec = LayerSpec {
            name: "conv1d".to_string(),
            layer_type: LayerType::Conv1d,
            inputs: vec![
                "input".to_string(),
                "kernel".to_string(),
                "bias".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: attrs,
        };
        (ws, spec)
    }

    fn conv_transpose1d_spec(attrs: HashMap<String, AttributeValue>) -> (WeightStore, LayerSpec) {
        let mut ws = WeightStore::new();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[3, 2, 3]), vec![0.0; 3 * 2 * 3])
            .expect("invariant: valid shape");
        let bias =
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0; 2]).expect("invariant: valid shape");
        ws.insert("kernel".to_string(), kernel);
        ws.insert("bias".to_string(), bias);

        let spec = LayerSpec {
            name: "conv_transpose1d".to_string(),
            layer_type: LayerType::ConvTranspose1d,
            inputs: vec![
                "input".to_string(),
                "kernel".to_string(),
                "bias".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: attrs,
        };
        (ws, spec)
    }

    fn conv2d_spec(attrs: HashMap<String, AttributeValue>) -> (WeightStore, LayerSpec) {
        let mut ws = WeightStore::new();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 3, 3, 3]), vec![0.0; 2 * 3 * 3 * 3])
            .expect("invariant: valid shape");
        let bias =
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0; 2]).expect("invariant: valid shape");
        ws.insert("kernel".to_string(), kernel);
        ws.insert("bias".to_string(), bias);

        let spec = LayerSpec {
            name: "conv2d".to_string(),
            layer_type: LayerType::Conv2d,
            inputs: vec![
                "input".to_string(),
                "kernel".to_string(),
                "bias".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: attrs,
        };
        (ws, spec)
    }

    fn conv_transpose2d_spec(attrs: HashMap<String, AttributeValue>) -> (WeightStore, LayerSpec) {
        let mut ws = WeightStore::new();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[3, 2, 3, 3]), vec![0.0; 3 * 2 * 3 * 3])
            .expect("invariant: valid shape");
        let bias =
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0; 2]).expect("invariant: valid shape");
        ws.insert("kernel".to_string(), kernel);
        ws.insert("bias".to_string(), bias);

        let spec = LayerSpec {
            name: "conv_transpose2d".to_string(),
            layer_type: LayerType::ConvTranspose2d,
            inputs: vec![
                "input".to_string(),
                "kernel".to_string(),
                "bias".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: attrs,
        };
        (ws, spec)
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv2d_default_attributes_succeed() {
        let (ws, spec) = conv2d_spec(HashMap::new());
        let context = make_context(&ws);
        let layer = context
            .convert_conv2d(&spec)
            .expect("default conv2d attrs must succeed");
        assert!(matches!(layer, Layer::Conv2d(_)));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv2d_non_unit_dilation_is_accepted() {
        let mut attrs = HashMap::new();
        attrs.insert("dilations".to_string(), AttributeValue::Ints(vec![2, 3]));
        let (ws, spec) = conv2d_spec(attrs);
        let context = make_context(&ws);
        let layer = context
            .convert_conv2d(&spec)
            .expect("Conv2d dilation must now be supported");
        match layer {
            Layer::Conv2d(conv) => assert_eq!(conv.dilation, (2, 3)),
            other => panic!("expected Conv2d layer, got {other:?}"),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv_transpose2d_non_unit_dilation_is_accepted() {
        let mut attrs = HashMap::new();
        attrs.insert("dilations".to_string(), AttributeValue::Ints(vec![2, 2]));
        let (ws, spec) = conv_transpose2d_spec(attrs);
        let context = make_context(&ws);
        let layer = context
            .convert_conv_transpose2d(&spec)
            .expect("ConvTranspose2d dilation must now be supported");
        match layer {
            Layer::ConvTranspose2d(conv) => assert_eq!(conv.dilation, (2, 2)),
            other => panic!("expected ConvTranspose2d layer, got {other:?}"),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv_transpose2d_output_padding_is_accepted() {
        let mut attrs = HashMap::new();
        // stride must exceed output_padding for soundness.
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![2, 2]));
        attrs.insert(
            "output_padding".to_string(),
            AttributeValue::Ints(vec![1, 1]),
        );
        let (ws, spec) = conv_transpose2d_spec(attrs);
        let context = make_context(&ws);
        let layer = context
            .convert_conv_transpose2d(&spec)
            .expect("ConvTranspose2d output_padding must now be supported");
        match layer {
            Layer::ConvTranspose2d(conv) => {
                assert_eq!(conv.output_padding, (1, 1));
                assert_eq!(conv.stride, (2, 2));
            }
            other => panic!("expected ConvTranspose2d layer, got {other:?}"),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv_transpose2d_output_padding_ge_stride_is_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![2, 2]));
        attrs.insert(
            "output_padding".to_string(),
            AttributeValue::Ints(vec![2, 0]),
        );
        let (ws, spec) = conv_transpose2d_spec(attrs);
        let context = make_context(&ws);
        let err = context
            .convert_conv_transpose2d(&spec)
            .expect_err("output_padding >= stride must be rejected for soundness");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("output_padding")),
            "expected UnsupportedConfiguration about output_padding, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv_transpose2d_output_shape_derives_output_padding() {
        use std::collections::HashMap as Map;
        // Kernel [3,2,3,3]: in_c=3, out_c=2, kh=kw=3. Input spatial 4x4, stride 2,
        // pad 0, dilation 1 → base output = 2*(4-1)+3 = 9. output_shape 10 → op=1.
        let mut attrs = HashMap::new();
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![2, 2]));
        attrs.insert(
            "output_shape".to_string(),
            AttributeValue::Ints(vec![10, 10]),
        );
        let (ws, spec) = conv_transpose2d_spec(attrs);
        let tensor_shapes: Map<String, Vec<i64>> =
            Map::from([("input".to_string(), vec![1, 3, 4, 4])]);
        let context = make_context_with_shapes(&ws, &tensor_shapes);
        let layer = context
            .convert_conv_transpose2d(&spec)
            .expect("output_shape should derive output_padding");
        match layer {
            Layer::ConvTranspose2d(conv) => assert_eq!(conv.output_padding, (1, 1)),
            other => panic!("expected ConvTranspose2d layer, got {other:?}"),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv2d_grouped_convolution_is_accepted() {
        // Kernel shape [2, 3, 3, 3]: out_c=2, in_c_per_group=3, kh=3, kw=3.
        // With group=2: total_in_c = 3 * 2 = 6, out_c_per_group = 2 / 2 = 1.
        let mut attrs = HashMap::new();
        attrs.insert("group".to_string(), AttributeValue::Int(2));
        let (ws, spec) = conv2d_spec(attrs);
        let context = make_context(&ws);
        let layer = context
            .convert_conv2d(&spec)
            .expect("group=2 must be accepted for Conv2d (#3770)");
        match layer {
            Layer::Conv2d(conv) => {
                assert_eq!(conv.groups, 2);
                assert_eq!(conv.out_channels(), 2);
                // in_channels = kernel.shape()[1] * groups = 3 * 2 = 6
                assert_eq!(conv.in_channels(), 6);
            }
            _ => panic!("expected Conv2d layer"),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv2d_non_default_auto_pad_is_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "auto_pad".to_string(),
            AttributeValue::String("SAME_UPPER".to_string()),
        );
        let (ws, spec) = conv2d_spec(attrs);
        let context = make_context(&ws);
        let err = context
            .convert_conv2d(&spec)
            .expect_err("non-default auto_pad must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("auto_pad")),
            "expected UnsupportedConfiguration with auto_pad, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv2d_asymmetric_padding_is_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("pads".to_string(), AttributeValue::Ints(vec![1, 2, 0, 2]));
        let (ws, spec) = conv2d_spec(attrs);
        let context = make_context(&ws);
        let err = context
            .convert_conv2d(&spec)
            .expect_err("asymmetric pads must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("asymmetric pads")),
            "expected UnsupportedConfiguration with asymmetric pads, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv_transpose2d_asymmetric_padding_is_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("pads".to_string(), AttributeValue::Ints(vec![1, 0, 0, 0]));
        let (ws, spec) = conv_transpose2d_spec(attrs);
        let context = make_context(&ws);
        let err = context
            .convert_conv_transpose2d(&spec)
            .expect_err("asymmetric pads must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("asymmetric pads")),
            "expected UnsupportedConfiguration with asymmetric pads, got: {err:?}"
        );
    }

    /// Regression test for #2828: stride=0 must be rejected at conversion time.
    /// Now caught by parse_stride_attr (ModelLoad) before reaching the constructor.
    #[ntest::timeout(10000)]
    #[test]
    fn conv2d_zero_stride_is_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![0, 1]));
        let (ws, spec) = conv2d_spec(attrs);
        let context = make_context(&ws);
        let err = context
            .convert_conv2d(&spec)
            .expect_err("stride=0 must be rejected");
        assert!(
            matches!(err, NyError::ModelLoad(ref msg) if msg.contains("stride")),
            "expected ModelLoad with stride, got: {err:?}"
        );
    }

    /// Regression test for #2828: stride=0 in both dims must be rejected.
    #[ntest::timeout(10000)]
    #[test]
    fn conv2d_zero_stride_both_dims_is_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![0, 0]));
        let (ws, spec) = conv2d_spec(attrs);
        let context = make_context(&ws);
        let err = context
            .convert_conv2d(&spec)
            .expect_err("stride=(0,0) must be rejected");
        assert!(
            matches!(err, NyError::ModelLoad(ref msg) if msg.contains("stride")),
            "expected ModelLoad with stride, got: {err:?}"
        );
    }

    /// Regression test for #2828: stride=0 in ConvTranspose2d must be rejected.
    #[ntest::timeout(10000)]
    #[test]
    fn conv_transpose2d_zero_stride_is_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![1, 0]));
        let (ws, spec) = conv_transpose2d_spec(attrs);
        let context = make_context(&ws);
        let err = context
            .convert_conv_transpose2d(&spec)
            .expect_err("stride=0 must be rejected");
        assert!(
            matches!(err, NyError::ModelLoad(ref msg) if msg.contains("stride")),
            "expected ModelLoad with stride, got: {err:?}"
        );
    }

    /// Regression test for #2828: stride=0 in Conv1d must be rejected at conversion.
    #[ntest::timeout(10000)]
    #[test]
    fn conv1d_zero_stride_is_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![0]));
        let (ws, spec) = conv1d_spec(attrs);
        let context = make_context(&ws);
        let err = context
            .convert_conv1d(&spec)
            .expect_err("stride=0 must be rejected");
        assert!(
            matches!(err, NyError::ModelLoad(ref msg) if msg.contains("stride")),
            "expected ModelLoad with stride, got: {err:?}"
        );
    }

    /// Regression test for #2828: negative stride in Conv1d must be rejected.
    #[ntest::timeout(10000)]
    #[test]
    fn conv1d_negative_stride_is_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![-1]));
        let (ws, spec) = conv1d_spec(attrs);
        let context = make_context(&ws);
        let err = context
            .convert_conv1d(&spec)
            .expect_err("negative stride must be rejected");
        assert!(
            matches!(err, NyError::ModelLoad(ref msg) if msg.contains("stride")),
            "expected ModelLoad with stride, got: {err:?}"
        );
    }

    /// Regression test for #2828: stride=0 in ConvTranspose1d must be rejected.
    #[ntest::timeout(10000)]
    #[test]
    fn conv_transpose1d_zero_stride_is_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("strides".to_string(), AttributeValue::Ints(vec![0]));
        let (ws, spec) = conv_transpose1d_spec(attrs);
        let context = make_context(&ws);
        let err = context
            .convert_conv_transpose1d(&spec)
            .expect_err("stride=0 must be rejected");
        assert!(
            matches!(err, NyError::ModelLoad(ref msg) if msg.contains("stride")),
            "expected ModelLoad with stride, got: {err:?}"
        );
    }

    /// Positive control: Conv1d with default attributes (stride=1) must succeed.
    #[ntest::timeout(10000)]
    #[test]
    fn conv1d_default_attributes_succeed() {
        let (ws, spec) = conv1d_spec(HashMap::new());
        let context = make_context(&ws);
        let layer = context
            .convert_conv1d(&spec)
            .expect("default conv1d attrs must succeed");
        assert!(matches!(layer, Layer::Conv1d(_)));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv1d_non_unit_dilation_succeeds() {
        let mut attrs = HashMap::new();
        attrs.insert("dilations".to_string(), AttributeValue::Ints(vec![2]));
        let (ws, spec) = conv1d_spec(attrs);
        let context = make_context(&ws);
        let layer = context
            .convert_conv1d(&spec)
            .expect("Conv1d dilation=2 must be supported");
        match layer {
            Layer::Conv1d(conv) => assert_eq!(conv.dilation, 2),
            other => panic!("expected Conv1d layer, got {other:?}"),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv_transpose1d_non_unit_dilation_succeeds() {
        let mut attrs = HashMap::new();
        attrs.insert("dilations".to_string(), AttributeValue::Ints(vec![2]));
        let (ws, spec) = conv_transpose1d_spec(attrs);
        let context = make_context(&ws);
        let layer = context
            .convert_conv_transpose1d(&spec)
            .expect("ConvTranspose1d dilation=2 must be supported");
        match layer {
            Layer::ConvTranspose1d(conv) => assert_eq!(conv.dilation, 2),
            other => panic!("expected ConvTranspose1d layer, got {other:?}"),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn conv_transpose1d_grouped_convolution_succeeds() {
        let mut ws = WeightStore::new();
        ws.insert(
            "kernel".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4, 2, 3]), vec![0.0; 4 * 2 * 3])
                .expect("invariant: valid shape"),
        );
        ws.insert(
            "bias".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0; 4]).expect("invariant: valid shape"),
        );

        let mut attrs = HashMap::new();
        attrs.insert("group".to_string(), AttributeValue::Int(2));
        let spec = LayerSpec {
            name: "conv_transpose1d_grouped".to_string(),
            layer_type: LayerType::ConvTranspose1d,
            inputs: vec![
                "input".to_string(),
                "kernel".to_string(),
                "bias".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: attrs,
        };

        let context = make_context(&ws);
        let layer = context
            .convert_conv_transpose1d(&spec)
            .expect("ConvTranspose1d group=2 must be supported");
        match layer {
            Layer::ConvTranspose1d(conv) => {
                assert_eq!(conv.groups, 2);
                assert_eq!(conv.in_channels(), 4);
                assert_eq!(conv.out_channels(), 4);
            }
            other => panic!("expected ConvTranspose1d layer, got {other:?}"),
        }
    }

    #[test]
    fn conv1d_accepts_evaluated_constant_kernel_3500() {
        let mut ws = WeightStore::new();
        ws.insert(
            "bias".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0; 2]).unwrap(),
        );
        let spec = LayerSpec {
            name: "conv1d_eval_kernel".to_string(),
            layer_type: LayerType::Conv1d,
            inputs: vec![
                "input".to_string(),
                "kernel".to_string(),
                "bias".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };
        let mut evaluated = HashMap::new();
        evaluated.insert(
            "kernel".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2, 3, 3]), vec![0.0; 2 * 3 * 3]).unwrap(),
        );

        let context = make_context_with_evaluated(&ws, &evaluated);
        let layer = context
            .convert_conv1d(&spec)
            .expect("evaluated constant kernels should be accepted");
        assert!(matches!(layer, Layer::Conv1d(_)));
    }

    #[test]
    fn conv1d_sets_input_length_from_last_input_axis_4354() {
        let (ws, spec) = conv1d_spec(HashMap::new());
        let tensor_shapes = HashMap::from([("input".to_string(), vec![1, 3, 17])]);
        let context = make_context_with_shapes(&ws, &tensor_shapes);

        let layer = context
            .convert_conv1d(&spec)
            .expect("conv1d with static input shape should convert");

        match layer {
            Layer::Conv1d(conv) => assert_eq!(conv.input_length, Some(17)),
            other => panic!("expected Conv1d layer, got {other:?}"),
        }
    }

    #[test]
    fn conv_transpose1d_sets_input_length_from_last_input_axis_4354() {
        let (ws, spec) = conv_transpose1d_spec(HashMap::new());
        let tensor_shapes = HashMap::from([("input".to_string(), vec![1, 2, 11])]);
        let context = make_context_with_shapes(&ws, &tensor_shapes);

        let layer = context
            .convert_conv_transpose1d(&spec)
            .expect("conv transpose with static input shape should convert");

        match layer {
            Layer::ConvTranspose1d(conv) => assert_eq!(conv.input_length, Some(11)),
            other => panic!("expected ConvTranspose1d layer, got {other:?}"),
        }
    }
}
