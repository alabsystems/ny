// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_propagate::layers::ResizeLayer;
use ny_propagate::Layer;

use super::{AttributeValue, ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    pub(crate) fn convert_resize(&self, spec: &LayerSpec) -> Result<Layer> {
        let mode = match spec.attributes.get("mode") {
            Some(AttributeValue::String(mode)) => mode.as_str(),
            None => "nearest",
            other => {
                return Err(NyError::ModelLoad(format!(
                    "Resize {} has invalid mode attribute {:?}",
                    spec.name, other
                )))
            }
        };
        if mode != "nearest" {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Resize {} mode='{}' is not supported (expected 'nearest')",
                spec.name, mode
            )));
        }

        if spec.attributes.contains_key("axes") {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Resize {} axes subsets are not supported; scales/sizes must describe every input axis",
                spec.name
            )));
        }

        // The implemented layer is exact integer replication. That agrees with
        // either the legacy/default pair when BOTH attributes are absent, the
        // explicit asymmetric/floor pair, or the explicit modern default pair
        // half_pixel/round_prefer_floor. A mixed explicit/default pair is
        // opset-dependent, and can select different source pixels.
        let coord_mode = match spec.attributes.get("coordinate_transformation_mode") {
            None => None,
            Some(AttributeValue::String(value)) => Some(value.as_str()),
            Some(other) => {
                return Err(NyError::ModelLoad(format!(
                    "Resize {} has invalid coordinate_transformation_mode attribute {:?}",
                    spec.name, other
                )))
            }
        };
        let nearest_mode = match spec.attributes.get("nearest_mode") {
            None => None,
            Some(AttributeValue::String(value)) => Some(value.as_str()),
            Some(other) => {
                return Err(NyError::ModelLoad(format!(
                    "Resize {} has invalid nearest_mode attribute {:?}",
                    spec.name, other
                )))
            }
        };
        if !matches!(
            (coord_mode, nearest_mode),
            (None, None)
                | (Some("asymmetric"), Some("floor"))
                | (Some("half_pixel"), Some("round_prefer_floor"))
        ) {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Resize {} coordinate/nearest pair {:?}/{:?} is not proven equivalent to integer nearest-neighbor replication",
                spec.name, coord_mode, nearest_mode
            )));
        }

        match spec.attributes.get("antialias") {
            None | Some(AttributeValue::Int(0)) => {}
            Some(AttributeValue::Int(_)) => {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "Resize {} antialiasing is not supported",
                    spec.name
                )))
            }
            Some(other) => {
                return Err(NyError::ModelLoad(format!(
                    "Resize {} has invalid antialias attribute {:?}",
                    spec.name, other
                )))
            }
        }
        match spec.attributes.get("keep_aspect_ratio_policy") {
            None => {}
            Some(AttributeValue::String(policy)) if policy == "stretch" => {}
            Some(AttributeValue::String(policy)) => {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "Resize {} keep_aspect_ratio_policy='{}' is not supported",
                    spec.name, policy
                )));
            }
            Some(other) => {
                return Err(NyError::ModelLoad(format!(
                    "Resize {} has invalid keep_aspect_ratio_policy attribute {:?}",
                    spec.name, other
                )))
            }
        }

        let (scale_h, scale_w) = self.resolve_resize_scales(spec)?;
        Ok(Layer::Resize(ResizeLayer::new(scale_h, scale_w)))
    }

    fn resolve_resize_scales(&self, spec: &LayerSpec) -> Result<(usize, usize)> {
        let sizes_name = spec.inputs.get(3).filter(|name| !name.is_empty());
        let scales_input = if spec.inputs.len() >= 3 {
            spec.inputs.get(2)
        } else {
            spec.inputs.get(1)
        };
        let scales_name = scales_input.filter(|name| !name.is_empty());

        if sizes_name.is_some() && scales_name.is_some() {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Resize {} supplies both scales and sizes; exactly one semantic operand is required",
                spec.name
            )));
        }

        if let Some(sizes_name) = sizes_name {
            let sizes = self
                .discrete_constant_i64(sizes_name, &format!("Resize {} sizes", spec.name))?
                .ok_or_else(|| {
                    NyError::UnsupportedConfiguration(format!(
                        "Resize {} requires sizes input '{}' to be constant",
                        spec.name, sizes_name
                    ))
                })?;
            let (scale_h, scale_w) = self.parse_resize_sizes(spec, &sizes)?;
            self.validate_resize_io_shapes(spec, scale_h, scale_w)?;
            return Ok((scale_h, scale_w));
        }

        if let Some(scales_name) = scales_name {
            let scales = self.constant_value(scales_name).ok_or_else(|| {
                NyError::UnsupportedConfiguration(format!(
                    "Resize {} requires scales input '{}' to be constant",
                    spec.name, scales_name
                ))
            })?;
            let (scale_h, scale_w) = self.parse_resize_scales_tensor(spec, &scales)?;
            self.validate_resize_io_shapes(spec, scale_h, scale_w)?;
            return Ok((scale_h, scale_w));
        }

        Err(NyError::UnsupportedConfiguration(format!(
            "Resize {} requires exactly one constant scales or sizes input",
            spec.name
        )))
    }

    fn parse_resize_scales_tensor(
        &self,
        spec: &LayerSpec,
        scales: &ArrayD<f32>,
    ) -> Result<(usize, usize)> {
        if scales.ndim() != 1 {
            return Err(NyError::ModelLoad(format!(
                "Resize {} scales must be a 1-D tensor, got shape {:?}",
                spec.name,
                scales.shape()
            )));
        }
        let values = scales.iter().copied().collect::<Vec<_>>();
        if values.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "Resize {} scales must have rank >= 2, got {:?}",
                spec.name, values
            )));
        }
        if let Some(input_shape) = spec
            .inputs
            .first()
            .and_then(|name| self.tensor_shapes.get(name))
        {
            if values.len() != input_shape.len() {
                return Err(NyError::ModelLoad(format!(
                    "Resize {} scales length {} does not match input rank {}",
                    spec.name,
                    values.len(),
                    input_shape.len()
                )));
            }
        }

        for &scale in &values[..values.len() - 2] {
            if scale != 1.0 {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "Resize {} only supports spatial scaling; leading scale {} must be 1",
                    spec.name, scale
                )));
            }
        }

        let scale_h = parse_positive_integer(values[values.len() - 2], &spec.name, "scale_h")?;
        let scale_w = parse_positive_integer(values[values.len() - 1], &spec.name, "scale_w")?;
        Ok((scale_h, scale_w))
    }

    fn parse_resize_sizes(&self, spec: &LayerSpec, sizes: &ArrayD<i64>) -> Result<(usize, usize)> {
        if sizes.ndim() != 1 {
            return Err(NyError::ModelLoad(format!(
                "Resize {} sizes must be a 1-D tensor, got shape {:?}",
                spec.name,
                sizes.shape()
            )));
        }
        let input_shape = self
            .tensor_shapes
            .get(&spec.inputs[0])
            .ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "Resize {} requires input tensor shape to interpret sizes",
                    spec.name
                ))
            })?
            .clone();
        let output_sizes = sizes.iter().copied().collect::<Vec<_>>();

        if output_sizes.len() != input_shape.len() {
            return Err(NyError::ModelLoad(format!(
                "Resize {} sizes rank {} does not match input rank {}",
                spec.name,
                output_sizes.len(),
                input_shape.len()
            )));
        }
        if output_sizes.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "Resize {} sizes rank must be >= 2",
                spec.name
            )));
        }

        for axis in 0..output_sizes.len() - 2 {
            let in_dim = positive_known_dim(
                input_shape[axis],
                &spec.name,
                &format!("input leading axis {axis}"),
            )?;
            let out_dim = positive_known_dim(
                output_sizes[axis],
                &spec.name,
                &format!("output leading axis {axis}"),
            )?;
            if out_dim != in_dim {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "Resize {} only supports spatial resizing; axis {} changes {} -> {}",
                    spec.name, axis, in_dim, out_dim
                )));
            }
        }

        let h_axis = output_sizes.len() - 2;
        let w_axis = output_sizes.len() - 1;
        let in_h = positive_known_dim(input_shape[h_axis], &spec.name, "input height")?;
        let in_w = positive_known_dim(input_shape[w_axis], &spec.name, "input width")?;
        let out_h = positive_known_dim(output_sizes[h_axis], &spec.name, "output height")?;
        let out_w = positive_known_dim(output_sizes[w_axis], &spec.name, "output width")?;

        if out_h % in_h != 0 || out_w % in_w != 0 {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Resize {} only supports integer spatial scale factors, got {}x{} -> {}x{}",
                spec.name, in_h, in_w, out_h, out_w
            )));
        }

        Ok((
            usize::try_from(out_h / in_h).map_err(|_| {
                NyError::UnsupportedConfiguration(format!(
                    "Resize {} height scale does not fit usize",
                    spec.name
                ))
            })?,
            usize::try_from(out_w / in_w).map_err(|_| {
                NyError::UnsupportedConfiguration(format!(
                    "Resize {} width scale does not fit usize",
                    spec.name
                ))
            })?,
        ))
    }

    fn validate_resize_io_shapes(
        &self,
        spec: &LayerSpec,
        scale_h: usize,
        scale_w: usize,
    ) -> Result<()> {
        let Some(input_shape) = self.tensor_shapes.get(&spec.inputs[0]) else {
            return Ok(());
        };
        let Some(output_name) = spec.outputs.first() else {
            return Ok(());
        };
        let Some(output_shape) = self.tensor_shapes.get(output_name) else {
            return Ok(());
        };

        validate_resize_shapes(&spec.name, input_shape, output_shape, scale_h, scale_w)
    }
}

fn parse_positive_integer(value: f32, layer_name: &str, label: &str) -> Result<usize> {
    if !value.is_finite() {
        return Err(NyError::InvalidSpec(format!(
            "Resize {} {} must be finite, got {}",
            layer_name, label, value
        )));
    }
    let rounded = value.round();
    if value != rounded || rounded <= 0.0 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "Resize {} {} must be a positive integer, got {}",
            layer_name, label, value
        )));
    }
    if rounded >= i64::MAX as f32 || rounded >= usize::MAX as f32 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "Resize {} {} is outside the non-saturating usize range: {}",
            layer_name, label, value
        )));
    }
    Ok(rounded as usize)
}

fn positive_known_dim(value: i64, layer_name: &str, label: &str) -> Result<i64> {
    if value <= 0 {
        return Err(NyError::ModelLoad(format!(
            "Resize {} requires positive known {} (got {})",
            layer_name, label, value
        )));
    }
    Ok(value)
}

fn validate_resize_shapes(
    layer_name: &str,
    input_shape: &[i64],
    output_shape: &[i64],
    scale_h: usize,
    scale_w: usize,
) -> Result<()> {
    if input_shape.len() != output_shape.len() {
        return Err(NyError::ModelLoad(format!(
            "Resize {} input rank {} does not match output rank {}",
            layer_name,
            input_shape.len(),
            output_shape.len()
        )));
    }
    if input_shape.len() < 2 {
        return Err(NyError::ModelLoad(format!(
            "Resize {} requires rank >= 2 shapes, got {:?} -> {:?}",
            layer_name, input_shape, output_shape
        )));
    }

    for axis in 0..input_shape.len() - 2 {
        let in_dim = input_shape[axis];
        let out_dim = output_shape[axis];
        if in_dim <= 0 || out_dim <= 0 {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Resize {} requires positive known unchanged leading axis {}, got {} -> {}",
                layer_name, axis, in_dim, out_dim
            )));
        }
        if in_dim != out_dim {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Resize {} only supports spatial resizing; axis {} changes {} -> {}",
                layer_name, axis, in_dim, out_dim
            )));
        }
    }

    let h_axis = input_shape.len() - 2;
    let w_axis = input_shape.len() - 1;
    if input_shape[h_axis] > 0
        && output_shape[h_axis] > 0
        && Some(output_shape[h_axis])
            != i64::try_from(scale_h)
                .ok()
                .and_then(|scale| input_shape[h_axis].checked_mul(scale))
    {
        return Err(NyError::UnsupportedConfiguration(format!(
            "Resize {} height scale mismatch: {:?} -> {:?} with scale_h={}",
            layer_name, input_shape, output_shape, scale_h
        )));
    }
    if input_shape[w_axis] > 0
        && output_shape[w_axis] > 0
        && Some(output_shape[w_axis])
            != i64::try_from(scale_w)
                .ok()
                .and_then(|scale| input_shape[w_axis].checked_mul(scale))
    {
        return Err(NyError::UnsupportedConfiguration(format!(
            "Resize {} width scale mismatch: {:?} -> {:?} with scale_w={}",
            layer_name, input_shape, output_shape, scale_w
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightStore;
    use ndarray::{arr1, IxDyn};
    use ny_core::LayerType;
    use std::collections::{HashMap, HashSet};

    fn resize_spec() -> LayerSpec {
        LayerSpec {
            name: "resize".to_string(),
            layer_type: LayerType::Resize,
            inputs: vec!["x".to_string(), "roi".to_string(), "scales".to_string()],
            outputs: vec!["y".to_string()],
            weights: None,
            attributes: HashMap::from([
                (
                    "mode".to_string(),
                    AttributeValue::String("nearest".to_string()),
                ),
                (
                    "coordinate_transformation_mode".to_string(),
                    AttributeValue::String("asymmetric".to_string()),
                ),
                (
                    "nearest_mode".to_string(),
                    AttributeValue::String("floor".to_string()),
                ),
            ]),
        }
    }

    #[test]
    fn convert_resize_from_constant_scales() {
        let mut weights = WeightStore::new();
        weights.insert("scales".to_string(), arr1(&[1.0, 1.0, 2.0, 2.0]).into_dyn());
        let tensor_shapes = HashMap::from([
            ("x".to_string(), vec![1, 8, 10, 12]),
            ("y".to_string(), vec![1, 8, 20, 24]),
        ]);
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);

        let layer = ctx.convert_resize(&resize_spec()).unwrap();
        let Layer::Resize(resize) = layer else {
            panic!("expected Resize layer");
        };
        assert_eq!(resize.scale_h, 2);
        assert_eq!(resize.scale_w, 2);
    }

    #[test]
    fn convert_resize_from_sizes_uses_io_shapes() {
        let mut weights = WeightStore::new();
        weights.insert(
            "sizes".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 8.0, 20.0, 24.0]).unwrap(),
        );
        let tensor_shapes = HashMap::from([
            ("x".to_string(), vec![1, 8, 10, 12]),
            ("y".to_string(), vec![1, 8, 20, 24]),
        ]);
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);

        let mut spec = resize_spec();
        spec.inputs[2].clear();
        spec.inputs.push("sizes".to_string());

        let layer = ctx.convert_resize(&spec).unwrap();
        let Layer::Resize(resize) = layer else {
            panic!("expected Resize layer");
        };
        assert_eq!(resize.scale_h, 2);
        assert_eq!(resize.scale_w, 2);
    }

    #[test]
    fn convert_resize_sizes_prefers_exact_integer_payload() {
        let mut weights = WeightStore::new();
        weights.insert(
            "sizes".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 8.0, 10.0, 12.0]).unwrap(),
        );
        weights.insert_integers(
            "sizes".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1_i64, 8, 20, 24]).unwrap(),
        );
        let tensor_shapes = HashMap::from([
            ("x".to_string(), vec![1, 8, 10, 12]),
            ("y".to_string(), vec![1, 8, 20, 24]),
        ]);
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);
        let mut spec = resize_spec();
        spec.inputs[2].clear();
        spec.inputs.push("sizes".to_string());

        let Layer::Resize(resize) = ctx.convert_resize(&spec).unwrap() else {
            panic!("expected Resize layer");
        };
        assert_eq!((resize.scale_h, resize.scale_w), (2, 2));
    }

    #[test]
    fn convert_resize_rejects_non_spatial_scale() {
        let mut weights = WeightStore::new();
        weights.insert("scales".to_string(), arr1(&[1.0, 2.0, 2.0, 2.0]).into_dyn());
        let tensor_shapes = HashMap::new();
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);

        let err = ctx.convert_resize(&resize_spec()).unwrap_err();
        assert!(
            format!("{err}").contains("only supports spatial scaling"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resize_rejects_adjacent_non_integral_scales_and_sizes() {
        let below_one = f32::from_bits(1.0_f32.to_bits() - 1);
        let above_two = f32::from_bits(2.0_f32.to_bits() + 1);

        assert!(parse_positive_integer(below_one, "resize", "scale_h").is_err());
        assert!(parse_positive_integer(above_two, "resize", "scale_w").is_err());
        assert!(parse_positive_integer(f32::MAX, "resize", "scale_h").is_err());
        for value in [below_one, above_two] {
            let mut weights = WeightStore::new();
            weights.insert(
                "sizes".to_string(),
                ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 8.0, value, 24.0]).unwrap(),
            );
            let tensor_shapes = HashMap::from([("x".to_string(), vec![1, 8, 10, 12])]);
            let constant_tensors = HashSet::new();
            let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);
            let mut spec = resize_spec();
            spec.inputs[2].clear();
            spec.inputs.push("sizes".to_string());
            assert!(ctx.convert_resize(&spec).is_err());
        }
    }

    #[test]
    fn convert_resize_rejects_mixed_coordinate_rounding_defaults() {
        let mut weights = WeightStore::new();
        weights.insert("scales".to_string(), arr1(&[1.0, 1.0, 3.0, 3.0]).into_dyn());
        let tensor_shapes = HashMap::from([
            ("x".to_string(), vec![1, 1, 2, 2]),
            ("y".to_string(), vec![1, 1, 6, 6]),
        ]);
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);

        let mut missing_nearest = resize_spec();
        missing_nearest.attributes.remove("nearest_mode");
        assert!(ctx.convert_resize(&missing_nearest).is_err());

        let mut missing_coord = resize_spec();
        missing_coord
            .attributes
            .remove("coordinate_transformation_mode");
        assert!(ctx.convert_resize(&missing_coord).is_err());
    }

    #[test]
    fn convert_resize_accepts_joint_defaults_for_integer_replication() {
        let mut weights = WeightStore::new();
        weights.insert("scales".to_string(), arr1(&[1.0, 1.0, 3.0, 3.0]).into_dyn());
        let tensor_shapes = HashMap::from([
            ("x".to_string(), vec![1, 1, 2, 2]),
            ("y".to_string(), vec![1, 1, 6, 6]),
        ]);
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);
        let mut spec = resize_spec();
        spec.attributes.remove("coordinate_transformation_mode");
        spec.attributes.remove("nearest_mode");

        let Layer::Resize(layer) = ctx.convert_resize(&spec).unwrap() else {
            panic!("expected Resize layer");
        };
        assert_eq!((layer.scale_h, layer.scale_w), (3, 3));
    }

    #[test]
    fn convert_resize_rejects_axes_subset() {
        let mut weights = WeightStore::new();
        weights.insert("scales".to_string(), arr1(&[2.0, 2.0]).into_dyn());
        let tensor_shapes = HashMap::from([
            ("x".to_string(), vec![1, 3, 4, 5]),
            ("y".to_string(), vec![1, 6, 8, 5]),
        ]);
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);
        let mut spec = resize_spec();
        spec.attributes
            .insert("axes".to_string(), AttributeValue::Ints(vec![1, 2]));

        assert!(ctx
            .convert_resize(&spec)
            .unwrap_err()
            .to_string()
            .contains("axes subsets"));
    }

    #[test]
    fn convert_resize_rejects_dynamic_semantic_operands() {
        let weights = WeightStore::new();
        let tensor_shapes = HashMap::from([
            ("x".to_string(), vec![1, 1, 2, 2]),
            ("y".to_string(), vec![1, 1, 4, 4]),
        ]);
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);

        assert!(ctx.convert_resize(&resize_spec()).is_err());
        let mut sizes_spec = resize_spec();
        sizes_spec.inputs[2].clear();
        sizes_spec.inputs.push("runtime_sizes".to_string());
        assert!(ctx.convert_resize(&sizes_spec).is_err());
    }

    #[test]
    fn convert_resize_sizes_rejects_dynamic_leading_input_dimension() {
        let mut weights = WeightStore::new();
        weights.insert(
            "sizes".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 5.0, 4.0, 4.0]).unwrap(),
        );
        weights.insert_integers(
            "sizes".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1_i64, 5, 4, 4]).unwrap(),
        );
        let tensor_shapes = HashMap::from([
            ("x".to_string(), vec![1, -1, 2, 2]),
            ("y".to_string(), vec![1, 5, 4, 4]),
        ]);
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);
        let mut spec = resize_spec();
        spec.inputs[2].clear();
        spec.inputs.push("sizes".to_string());

        let err = ctx.convert_resize(&spec).unwrap_err();
        assert!(
            err.to_string()
                .contains("positive known input leading axis 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resize_rejects_scale_outside_i64_shape_arithmetic() {
        assert!(parse_positive_integer(i64::MAX as f32, "resize", "scale_h").is_err());
    }
}
