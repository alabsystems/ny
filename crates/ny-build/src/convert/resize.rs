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

        if let Some(AttributeValue::String(coord_mode)) =
            spec.attributes.get("coordinate_transformation_mode")
        {
            if coord_mode != "asymmetric" {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "Resize {} coordinate_transformation_mode='{}' is not supported (expected 'asymmetric')",
                    spec.name, coord_mode
                )));
            }
        }

        if let Some(AttributeValue::String(nearest_mode)) = spec.attributes.get("nearest_mode") {
            if nearest_mode != "floor" {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "Resize {} nearest_mode='{}' is not supported (expected 'floor')",
                    spec.name, nearest_mode
                )));
            }
        }

        let (scale_h, scale_w) = self.resolve_resize_scales(spec)?;
        Ok(Layer::Resize(ResizeLayer::new(scale_h, scale_w)))
    }

    fn resolve_resize_scales(&self, spec: &LayerSpec) -> Result<(usize, usize)> {
        if spec.inputs.len() >= 4 && !spec.inputs[3].is_empty() {
            if let Some(sizes) = self.constant_value(&spec.inputs[3]) {
                return self.parse_resize_sizes(spec, &sizes);
            }
        }

        let scales_input = if spec.inputs.len() >= 3 {
            spec.inputs.get(2)
        } else {
            spec.inputs.get(1)
        };
        if let Some(scales_name) = scales_input.filter(|name| !name.is_empty()) {
            if let Some(scales) = self.constant_value(scales_name) {
                let (scale_h, scale_w) = self.parse_resize_scales_tensor(spec, &scales)?;
                self.validate_resize_io_shapes(spec, scale_h, scale_w)?;
                return Ok((scale_h, scale_w));
            }
        }

        self.scales_from_inferred_shapes(spec)
    }

    fn parse_resize_scales_tensor(
        &self,
        spec: &LayerSpec,
        scales: &ArrayD<f32>,
    ) -> Result<(usize, usize)> {
        let values = scales.iter().copied().collect::<Vec<_>>();
        if values.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "Resize {} scales must have rank >= 2, got {:?}",
                spec.name, values
            )));
        }

        for &scale in &values[..values.len() - 2] {
            if !approx_eq(scale, 1.0) {
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

    fn parse_resize_sizes(&self, spec: &LayerSpec, sizes: &ArrayD<f32>) -> Result<(usize, usize)> {
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
        let output_sizes = sizes
            .iter()
            .copied()
            .map(|value| parse_i64(value, &spec.name, "sizes"))
            .collect::<Result<Vec<_>>>()?;

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
            let in_dim = input_shape[axis];
            let out_dim = output_sizes[axis];
            if in_dim > 0 && out_dim != in_dim {
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

        Ok(((out_h / in_h) as usize, (out_w / in_w) as usize))
    }

    fn scales_from_inferred_shapes(&self, spec: &LayerSpec) -> Result<(usize, usize)> {
        let input_shape = self
            .tensor_shapes
            .get(&spec.inputs[0])
            .ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "Resize {} requires constant scales/sizes or inferred input shape",
                    spec.name
                ))
            })?
            .clone();
        let output_name = spec.outputs.first().ok_or_else(|| {
            NyError::ModelLoad(format!("Resize {} is missing an output name", spec.name))
        })?;
        let output_shape = self
            .tensor_shapes
            .get(output_name)
            .ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "Resize {} requires inferred output shape when scales are not constant",
                    spec.name
                ))
            })?
            .clone();

        scales_from_io_shapes(&spec.name, &input_shape, &output_shape)
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

fn approx_eq(lhs: f32, rhs: f32) -> bool {
    (lhs - rhs).abs() <= 1e-6
}

fn parse_positive_integer(value: f32, layer_name: &str, label: &str) -> Result<usize> {
    if !value.is_finite() {
        return Err(NyError::InvalidSpec(format!(
            "Resize {} {} must be finite, got {}",
            layer_name, label, value
        )));
    }
    let rounded = value.round();
    if !approx_eq(value, rounded) || rounded <= 0.0 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "Resize {} {} must be a positive integer, got {}",
            layer_name, label, value
        )));
    }
    Ok(rounded as usize)
}

fn parse_i64(value: f32, layer_name: &str, label: &str) -> Result<i64> {
    if !value.is_finite() {
        return Err(NyError::InvalidSpec(format!(
            "Resize {} {} must be finite, got {}",
            layer_name, label, value
        )));
    }
    let rounded = value.round();
    if !approx_eq(value, rounded) {
        return Err(NyError::InvalidSpec(format!(
            "Resize {} {} must be integral, got {}",
            layer_name, label, value
        )));
    }
    Ok(rounded as i64)
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
        if in_dim > 0 && out_dim > 0 && in_dim != out_dim {
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
        && output_shape[h_axis] != input_shape[h_axis] * scale_h as i64
    {
        return Err(NyError::UnsupportedConfiguration(format!(
            "Resize {} height scale mismatch: {:?} -> {:?} with scale_h={}",
            layer_name, input_shape, output_shape, scale_h
        )));
    }
    if input_shape[w_axis] > 0
        && output_shape[w_axis] > 0
        && output_shape[w_axis] != input_shape[w_axis] * scale_w as i64
    {
        return Err(NyError::UnsupportedConfiguration(format!(
            "Resize {} width scale mismatch: {:?} -> {:?} with scale_w={}",
            layer_name, input_shape, output_shape, scale_w
        )));
    }

    Ok(())
}

fn scales_from_io_shapes(
    layer_name: &str,
    input_shape: &[i64],
    output_shape: &[i64],
) -> Result<(usize, usize)> {
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
        if in_dim > 0 && out_dim > 0 && in_dim != out_dim {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Resize {} only supports spatial resizing; axis {} changes {} -> {}",
                layer_name, axis, in_dim, out_dim
            )));
        }
    }

    let h_axis = input_shape.len() - 2;
    let w_axis = input_shape.len() - 1;
    let in_h = positive_known_dim(input_shape[h_axis], layer_name, "input height")?;
    let in_w = positive_known_dim(input_shape[w_axis], layer_name, "input width")?;
    let out_h = positive_known_dim(output_shape[h_axis], layer_name, "output height")?;
    let out_w = positive_known_dim(output_shape[w_axis], layer_name, "output width")?;

    if out_h % in_h != 0 || out_w % in_w != 0 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "Resize {} only supports integer spatial scale factors, got {}x{} -> {}x{}",
            layer_name, in_h, in_w, out_h, out_w
        )));
    }

    Ok(((out_h / in_h) as usize, (out_w / in_w) as usize))
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
        spec.inputs.push("sizes".to_string());

        let layer = ctx.convert_resize(&spec).unwrap();
        let Layer::Resize(resize) = layer else {
            panic!("expected Resize layer");
        };
        assert_eq!(resize.scale_h, 2);
        assert_eq!(resize.scale_w, 2);
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
}
