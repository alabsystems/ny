// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{concatenate, ArrayD, Axis, IxDyn, Slice};
use ny_propagate::layers::normalize_transpose_perm_for_rank;
use std::collections::HashMap;
use tracing::debug;

use super::super::{AttributeValue, ConvertContext, LayerSpec};
use super::{
    adjust_constant_slice_axis, lookup_constant_value, normalize_slice_bound,
    parse_integral_constant_value, propagate_constant_through_layer, resolve_constant_axis,
};

impl ConvertContext<'_> {
    pub(super) fn evaluate_concat_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        if spec.inputs.len() < 2 {
            return None;
        }

        let tensors: Vec<ArrayD<f32>> = spec
            .inputs
            .iter()
            .map(|name| lookup_constant_value(self.weights, evaluated_constants, name))
            .collect::<Option<Vec<_>>>()?;
        let axis = spec
            .attributes
            .get("axis")
            .and_then(|value| match value {
                AttributeValue::Int(axis) => Some(*axis),
                _ => None,
            })
            .unwrap_or(0);
        let axis = resolve_constant_axis(spec, "Concat", tensors[0].ndim(), axis)?;
        let views: Vec<_> = tensors.iter().map(|tensor| tensor.view()).collect();
        debug!("Evaluating {} as Concat along axis {}", spec.name, axis);
        concatenate(Axis(axis), &views).ok()
    }

    pub(super) fn evaluate_squeeze_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        if spec.inputs.is_empty() {
            return None;
        }
        let data = lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[0])?;
        let axes: Vec<i64> = if let Some(AttributeValue::Ints(axes)) = spec.attributes.get("axes") {
            axes.clone()
        } else if spec.inputs.len() >= 2 {
            let axes_tensor =
                lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[1])?;
            axes_tensor.iter().map(|&value| value as i64).collect()
        } else {
            return None;
        };
        let ndim = data.ndim();
        let mut resolved: Vec<usize> = axes
            .iter()
            .filter_map(|&axis| {
                let axis = if axis < 0 {
                    (ndim as i64 + axis) as usize
                } else {
                    axis as usize
                };
                (axis < ndim && data.shape()[axis] == 1).then_some(axis)
            })
            .collect();
        resolved.sort_unstable();
        resolved.dedup();
        let mut shape: Vec<usize> = data.shape().to_vec();
        for &axis in resolved.iter().rev() {
            shape.remove(axis);
        }
        if shape.is_empty() {
            shape.push(1);
        }
        debug!(
            "Evaluating {} as Squeeze: {:?} -> {:?}",
            spec.name,
            data.shape(),
            shape
        );
        data.into_shape_with_order(IxDyn(&shape)).ok()
    }

    pub(super) fn evaluate_slice_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        let data = lookup_constant_value(self.weights, evaluated_constants, spec.inputs.first()?)?;
        let (axis, start, end) = if let (
            Some(AttributeValue::Int(axis)),
            Some(AttributeValue::Int(start)),
            Some(AttributeValue::Int(end)),
        ) = (
            spec.attributes.get("axis"),
            spec.attributes.get("start"),
            spec.attributes.get("end"),
        ) {
            (*axis, *start, *end)
        } else if spec.inputs.len() >= 3 {
            let starts =
                lookup_constant_value(self.weights, evaluated_constants, spec.inputs.get(1)?)?;
            let ends =
                lookup_constant_value(self.weights, evaluated_constants, spec.inputs.get(2)?)?;
            if starts.len() != 1 || ends.len() != 1 {
                debug!(
                    "Slice {} constant evaluation only supports single-axis slicing",
                    spec.name
                );
                return None;
            }

            let axis = if let Some(axis_name) = spec.inputs.get(3).filter(|name| !name.is_empty()) {
                let axes = lookup_constant_value(self.weights, evaluated_constants, axis_name)?;
                if axes.len() != 1 {
                    debug!(
                        "Slice {} constant evaluation only supports one axis entry",
                        spec.name
                    );
                    return None;
                }
                parse_integral_constant_value(spec, "axis", axes.iter().next().copied()?, false)?
            } else {
                0
            };

            if let Some(step_name) = spec.inputs.get(4).filter(|name| !name.is_empty()) {
                let steps = lookup_constant_value(self.weights, evaluated_constants, step_name)?;
                if steps.len() != 1 {
                    debug!(
                        "Slice {} constant evaluation only supports one step entry",
                        spec.name
                    );
                    return None;
                }
                let step = parse_integral_constant_value(
                    spec,
                    "step",
                    steps.iter().next().copied()?,
                    false,
                )?;
                if step != 1 {
                    debug!(
                        "Slice {} constant evaluation only supports step=1 (got {})",
                        spec.name, step
                    );
                    return None;
                }
            }

            (
                axis,
                parse_integral_constant_value(
                    spec,
                    "start",
                    starts.iter().next().copied()?,
                    false,
                )?,
                parse_integral_constant_value(spec, "end", ends.iter().next().copied()?, true)?,
            )
        } else {
            return None;
        };

        let axis = adjust_constant_slice_axis(spec, axis)?;
        let axis = resolve_constant_axis(spec, "Slice", data.ndim(), axis)?;
        let axis_len = data.shape()[axis] as i64;
        let start = normalize_slice_bound(start, axis_len);
        let end = if end == i64::MAX {
            axis_len
        } else {
            normalize_slice_bound(end, axis_len)
        };
        debug!(
            "Evaluating {} as Slice axis={}, start={}, end={}",
            spec.name, axis, start, end
        );
        Some(
            data.slice_axis(Axis(axis), Slice::from(start as isize..end as isize))
                .to_owned(),
        )
    }

    pub(super) fn evaluate_transpose_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        if spec.inputs.is_empty() {
            return None;
        }
        let data = lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[0])?;
        let raw_perm: Vec<usize> =
            if let Some(AttributeValue::Ints(perm)) = spec.attributes.get("perm") {
                // A negative entry cannot be a valid axis; bail (None) rather than
                // silently dropping it, which would corrupt the permutation length.
                perm.iter()
                    .map(|&value| usize::try_from(value).ok())
                    .collect::<Option<Vec<usize>>>()?
            } else {
                (0..data.ndim()).rev().collect()
            };
        // The ONNX `perm` may have been authored for a higher (batched) rank than
        // the materialized constant carries, or be a meaningless over-ranked perm
        // on a rank-≤1 constant (e.g. a vit positional-embedding `{48}` fed to a
        // `perm={0,2,1}` Transpose). `ndarray::permuted_axes` PANICS on any
        // length/range mismatch, so normalize to the constant's actual rank first.
        // Returns `None` (skip const-eval, leaving a runtime layer) when no
        // rank-consistent rewrite is provably equivalent — never an unsound guess.
        let perm = normalize_transpose_perm_for_rank(&raw_perm, data.ndim())?;
        debug!(
            "Evaluating {} as Transpose: {:?} raw_perm={:?} normalized_perm={:?}",
            spec.name,
            data.shape(),
            raw_perm,
            perm
        );
        Some(
            data.view()
                .permuted_axes(perm)
                .as_standard_layout()
                .into_owned(),
        )
    }

    pub(super) fn evaluate_gather_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        if spec.inputs.len() < 2 {
            return None;
        }
        let data = lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[0])?;
        let indices_raw =
            lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[1])?;

        let onnx_axis = spec
            .attributes
            .get("axis")
            .and_then(|value| match value {
                AttributeValue::Int(axis) => Some(*axis),
                _ => None,
            })
            .unwrap_or(0);

        let axis = resolve_constant_axis(spec, "Gather", data.ndim(), onnx_axis)?;
        let axis_len = data.shape()[axis] as i64;

        let mut indices_i64 = Vec::with_capacity(indices_raw.len());
        for &v in indices_raw.iter() {
            if !v.is_finite() {
                debug!(
                    "Gather {} indices contain NaN/Inf at value {}",
                    spec.name, v
                );
                return None;
            }
            let rounded = v.round();
            let idx = rounded as i64;
            let normalized = if idx < 0 { axis_len + idx } else { idx };
            if normalized < 0 || normalized >= axis_len {
                debug!(
                    "Gather {} index {} out of bounds for axis length {}",
                    spec.name, idx, axis_len
                );
                return None;
            }
            indices_i64.push(normalized as usize);
        }

        if indices_raw.shape().is_empty() {
            if indices_i64.len() != 1 {
                debug!(
                    "Gather {} scalar indices expected 1 element, got {}",
                    spec.name,
                    indices_i64.len()
                );
                return None;
            }
            let index = indices_i64[0];
            return Some(data.index_axis(Axis(axis), index).to_owned());
        }

        let selected = data.select(Axis(axis), &indices_i64);
        let mut output_shape =
            Vec::with_capacity(data.shape().len() - 1 + indices_raw.shape().len());
        output_shape.extend_from_slice(&data.shape()[..axis]);
        output_shape.extend_from_slice(indices_raw.shape());
        output_shape.extend_from_slice(&data.shape()[axis + 1..]);

        debug!(
            "Evaluating {} as Gather axis={}, input_shape={:?}, indices_shape={:?}, output_shape={:?}",
            spec.name,
            axis,
            data.shape(),
            indices_raw.shape(),
            output_shape
        );

        selected
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order(IxDyn(&output_shape))
            .ok()
    }

    pub(super) fn evaluate_shape_constant(
        &self,
        spec: &LayerSpec,
        _evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        // ONNX Shape op: returns a 1-D tensor containing the static shape of the input.
        // The input must have a known static shape in tensor_shapes.
        let input_name = spec.inputs.first()?;
        let input_shape = self.tensor_shapes.get(input_name)?;

        // Convert the shape to a 1-D f32 array (matching ONNX Shape output dtype).
        // ONNX Shape opset>=15 supports optional start/end attributes for range slicing.
        let start = spec
            .attributes
            .get("start")
            .and_then(|value| match value {
                AttributeValue::Int(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(0) as usize;
        let end = spec
            .attributes
            .get("end")
            .and_then(|value| match value {
                AttributeValue::Int(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(input_shape.len() as i64) as usize;

        let end = end.min(input_shape.len());
        if start > end || start >= input_shape.len() {
            debug!(
                "Shape {} constant evaluation: invalid range start={} end={} for shape len={}",
                spec.name,
                start,
                end,
                input_shape.len()
            );
            return None;
        }

        let shape_slice: Vec<f32> = input_shape[start..end]
            .iter()
            .map(|&dim| dim as f32)
            .collect();
        debug!(
            "Evaluating Shape {} -> {:?} (dims {} to {})",
            spec.name, shape_slice, start, end
        );
        ArrayD::from_shape_vec(IxDyn(&[shape_slice.len()]), shape_slice).ok()
    }

    pub(super) fn evaluate_unsqueeze_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        // Unsqueeze: insert dimension of size 1 at specified axis.
        // The data and axes must be available as constants.
        if spec.inputs.is_empty() {
            return None;
        }
        let data = lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[0])?;
        let axis: i64 = if let Some(AttributeValue::Ints(axes)) = spec.attributes.get("axes") {
            if axes.len() != 1 {
                return None;
            }
            axes[0]
        } else if spec.inputs.len() >= 2 {
            let axes_tensor =
                lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[1])?;
            if axes_tensor.len() != 1 {
                return None;
            }
            axes_tensor.iter().next().copied()? as i64
        } else {
            return None;
        };

        let ndim = data.ndim() as i64;
        let resolved_axis = if axis < 0 { ndim + 1 + axis } else { axis };
        if resolved_axis < 0 || resolved_axis > ndim {
            debug!(
                "Unsqueeze {} axis {} invalid for rank {}",
                spec.name,
                axis,
                data.ndim()
            );
            return None;
        }

        let mut shape: Vec<usize> = data.shape().to_vec();
        shape.insert(resolved_axis as usize, 1);
        debug!(
            "Evaluating {} as Unsqueeze: {:?} -> {:?} (axis {})",
            spec.name,
            data.shape(),
            shape,
            axis
        );
        data.into_shape_with_order(IxDyn(&shape)).ok()
    }

    pub(super) fn evaluate_cast_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        // Cast: identity for f32 bounds (all computation is in f32).
        // Just return the input as-is.
        if spec.inputs.is_empty() {
            return None;
        }
        lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[0])
    }

    pub(super) fn evaluate_fallback_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        if spec.inputs.len() == 1 {
            let input =
                lookup_constant_value(self.weights, evaluated_constants, spec.inputs.first()?)?;
            let layer = self.convert_layer(spec).ok()?;
            if let Some(output) = propagate_constant_through_layer(&layer, input, &spec.name) {
                return Some(output);
            }
        }
        if spec.inputs.is_empty() {
            if let Some(AttributeValue::Float(value)) = spec.attributes.get("value") {
                if let Some(shape) = self.tensor_shapes.get(&spec.outputs[0]) {
                    let shape_usize: Vec<usize> = shape
                        .iter()
                        .filter_map(|&dim| if dim > 0 { Some(dim as usize) } else { None })
                        .collect();
                    if shape_usize.len() == shape.len() {
                        debug!(
                            "Evaluating {} as constant fill(shape={:?}, value={})",
                            spec.name, shape_usize, value
                        );
                        return Some(ArrayD::from_elem(IxDyn(&shape_usize), *value));
                    }
                }
            }
        }
        debug!(
            "Cannot evaluate constant layer {} of type {:?}",
            spec.name, spec.layer_type
        );
        None
    }
}
