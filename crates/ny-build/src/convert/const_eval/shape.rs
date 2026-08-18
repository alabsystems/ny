// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{concatenate, ArrayD, Axis, IxDyn, Slice};
use ny_propagate::layers::normalize_transpose_perm_for_rank;
use std::collections::HashMap;
use tracing::debug;

use super::super::{AttributeValue, ConvertContext, LayerSpec};
use super::{
    adjust_constant_slice_axis, lookup_constant_value, lookup_integral_constant_values,
    normalize_slice_bound, propagate_constant_through_layer, resolve_constant_axis,
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
            lookup_integral_constant_values(
                self.weights,
                evaluated_constants,
                spec,
                &spec.inputs[1],
                "axis",
                false,
            )?
            .0
        } else {
            return None;
        };
        let ndim = data.ndim();
        let mut resolved: Vec<usize> = axes
            .iter()
            .map(|&axis| {
                let axis = resolve_constant_axis(spec, "Squeeze", ndim, axis)?;
                (data.shape()[axis] == 1).then_some(axis)
            })
            .collect::<Option<Vec<_>>>()?;
        resolved.sort_unstable();
        if resolved.windows(2).any(|window| window[0] == window[1]) {
            return None;
        }
        let mut shape: Vec<usize> = data.shape().to_vec();
        for &axis in resolved.iter().rev() {
            shape.remove(axis);
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
            let starts_name = spec.inputs.get(1)?;
            let ends_name = spec.inputs.get(2)?;
            let (starts, _) = lookup_integral_constant_values(
                self.weights,
                evaluated_constants,
                spec,
                starts_name,
                "start",
                false,
            )?;
            let (ends, _) = lookup_integral_constant_values(
                self.weights,
                evaluated_constants,
                spec,
                ends_name,
                "end",
                true,
            )?;
            if starts.len() != 1 || ends.len() != 1 {
                debug!(
                    "Slice {} constant evaluation only supports single-axis slicing",
                    spec.name
                );
                return None;
            }

            let axis = if let Some(axis_name) = spec.inputs.get(3).filter(|name| !name.is_empty()) {
                let (axes, _) = lookup_integral_constant_values(
                    self.weights,
                    evaluated_constants,
                    spec,
                    axis_name,
                    "axis",
                    false,
                )?;
                if axes.len() != 1 {
                    debug!(
                        "Slice {} constant evaluation only supports one axis entry",
                        spec.name
                    );
                    return None;
                }
                axes[0]
            } else {
                0
            };

            if let Some(step_name) = spec.inputs.get(4).filter(|name| !name.is_empty()) {
                let (steps, _) = lookup_integral_constant_values(
                    self.weights,
                    evaluated_constants,
                    spec,
                    step_name,
                    "step",
                    false,
                )?;
                if steps.len() != 1 {
                    debug!(
                        "Slice {} constant evaluation only supports one step entry",
                        spec.name
                    );
                    return None;
                }
                let step = steps[0];
                if step != 1 {
                    debug!(
                        "Slice {} constant evaluation only supports step=1 (got {})",
                        spec.name, step
                    );
                    return None;
                }
            }

            (axis, starts[0], ends[0])
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
        let (indices_raw, indices_shape) = lookup_integral_constant_values(
            self.weights,
            evaluated_constants,
            spec,
            &spec.inputs[1],
            "index",
            false,
        )?;

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
        for idx in indices_raw {
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

        if indices_shape.is_empty() {
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
        let mut output_shape = Vec::with_capacity(data.shape().len() - 1 + indices_shape.len());
        output_shape.extend_from_slice(&data.shape()[..axis]);
        output_shape.extend_from_slice(&indices_shape);
        output_shape.extend_from_slice(&data.shape()[axis + 1..]);

        debug!(
            "Evaluating {} as Gather axis={}, input_shape={:?}, indices_shape={:?}, output_shape={:?}",
            spec.name,
            axis,
            data.shape(),
            indices_shape,
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
        // Non-positive dimensions are ny's unresolved-shape markers. Folding
        // one into a literal Shape tensor would replace its runtime extent
        // with -1/0 and corrupt downstream shape expressions.
        if input_shape.iter().any(|&dim| dim <= 0) {
            debug!(
                "Shape {} constant evaluation declined for dynamic shape {:?}",
                spec.name, input_shape
            );
            return None;
        }

        // Convert the shape to a 1-D f32 array (matching ONNX Shape output dtype).
        // ONNX Shape opset>=15 supports optional start/end attributes for range slicing.
        let raw_start = spec
            .attributes
            .get("start")
            .and_then(|value| match value {
                AttributeValue::Int(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(0);
        let raw_end = spec
            .attributes
            .get("end")
            .and_then(|value| match value {
                AttributeValue::Int(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(input_shape.len() as i64);

        // Shape-15 normalizes negative bounds relative to the input rank, then
        // clamps both endpoints into [0, rank]. Saturating addition also keeps
        // hostile INT64_MIN attributes from overflowing in debug builds.
        let rank = input_shape.len() as i64;
        let normalize_bound = |bound: i64| {
            if bound < 0 {
                rank.saturating_add(bound).clamp(0, rank)
            } else {
                bound.min(rank)
            }
        };
        let start = normalize_bound(raw_start) as usize;
        let end = normalize_bound(raw_end) as usize;
        if start > end {
            return ArrayD::from_shape_vec(IxDyn(&[0]), Vec::new()).ok();
        }

        let shape_slice: Vec<f32> = input_shape[start..end]
            .iter()
            .map(|&dim| {
                super::super::i64_to_f32_checked(dim, &format!("Shape {} dimension", spec.name))
                    .ok()
            })
            .collect::<Option<Vec<_>>>()?;
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
            let (axes, _) = lookup_integral_constant_values(
                self.weights,
                evaluated_constants,
                spec,
                &spec.inputs[1],
                "axis",
                false,
            )?;
            if axes.len() != 1 {
                return None;
            }
            axes[0]
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
        // Only an ONNX Cast *to* FLOAT32 is an identity in NY's f32 carrier.
        // Integer, boolean, and lower-precision targets change values and must
        // not be materialized by copying the source tensor.
        if spec.inputs.len() != 1
            || spec.outputs.len() != 1
            || spec.attributes.len() != 1
            || !matches!(spec.attributes.get("to"), Some(AttributeValue::Int(1)))
        {
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
        debug!(
            "Cannot evaluate constant layer {} of type {:?}",
            spec.name, spec.layer_type
        );
        None
    }
}
