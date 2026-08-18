// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_propagate::layers::{PadLayer, PadMode};
use ny_propagate::Layer;

use super::{AttributeValue, ConvertContext, LayerSpec};
use crate::PAD_PRESERVE_ALL_AXES_ATTR;

impl ConvertContext<'_> {
    pub(crate) fn convert_pad(&self, spec: &LayerSpec) -> Result<Layer> {
        let has_pads_input = spec.inputs.len() >= 2 && !spec.inputs[1].is_empty();
        let preserve_all_axes = match spec.attributes.get(PAD_PRESERVE_ALL_AXES_ATTR) {
            None => false,
            Some(AttributeValue::Int(1)) => true,
            Some(value) => {
                return Err(NyError::ModelLoad(format!(
                    "Pad {} has invalid internal axis-layout certificate {value:?}",
                    spec.name
                )));
            }
        };

        if spec.inputs.get(3).is_some_and(|name| !name.is_empty())
            || spec.attributes.contains_key("axes")
        {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Pad {} axes subsets are not supported",
                spec.name
            )));
        }

        let input_shape = spec
            .inputs
            .first()
            .and_then(|name| self.tensor_shapes.get(name));

        // Opset >= 11: pads come as a second input tensor.
        // Opset < 11: pads come as an attribute "pads" (list of ints).
        if has_pads_input && spec.attributes.contains_key("pads") {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Pad {} supplies pads as both an attribute and an input",
                spec.name
            )));
        }

        let pads = if has_pads_input {
            let pads_tensor = self
                .discrete_constant_i64(&spec.inputs[1], &format!("Pad {} pads", spec.name))?
                .ok_or_else(|| {
                    NyError::UnsupportedConfiguration(format!(
                        "Pad {} requires constant pads input '{}'",
                        spec.name, spec.inputs[1]
                    ))
                })?;
            parse_integer_pad_pairs(
                spec,
                &pads_tensor,
                input_shape,
                self.model_unbatched,
                preserve_all_axes,
            )?
        } else if let Some(AttributeValue::Ints(ints)) = spec.attributes.get("pads") {
            finish_pad_pairs(
                spec,
                ints.clone(),
                input_shape,
                self.model_unbatched,
                preserve_all_axes,
            )?
        } else {
            return Err(NyError::ModelLoad(format!(
                "Pad {} requires pads as input tensor (opset>=11) or attribute (opset<11)",
                spec.name
            )));
        };

        let mode = match spec.attributes.get("mode") {
            None => PadMode::Constant(self.parse_pad_constant_value(spec)?),
            Some(AttributeValue::String(mode)) if mode == "reflect" => PadMode::Reflect,
            Some(AttributeValue::String(mode)) if mode == "constant" => {
                PadMode::Constant(self.parse_pad_constant_value(spec)?)
            }
            Some(AttributeValue::String(mode)) => {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "Pad {} mode '{}' is not supported",
                    spec.name, mode
                )));
            }
            Some(other) => {
                return Err(NyError::ModelLoad(format!(
                    "Pad {} has invalid mode attribute {:?}",
                    spec.name, other
                )));
            }
        };

        Ok(Layer::Pad(PadLayer::new(pads, mode)))
    }

    fn parse_pad_constant_value(&self, spec: &LayerSpec) -> Result<f32> {
        // Opset >= 11: constant_value comes as 3rd input tensor.
        if let Some(value_name) = spec.inputs.get(2).filter(|name| !name.is_empty()) {
            let value = self.constant_value(value_name).ok_or_else(|| {
                NyError::UnsupportedConfiguration(format!(
                    "Pad {} constant mode requires constant value input '{}'",
                    spec.name, value_name
                ))
            })?;
            return parse_pad_scalar(spec, &value);
        }
        // Opset < 11: constant_value comes as "value" attribute.
        if let Some(AttributeValue::Float(v)) = spec.attributes.get("value") {
            return Ok(*v);
        }
        Ok(0.0)
    }
}

fn parse_integer_pad_pairs(
    spec: &LayerSpec,
    pads_tensor: &ArrayD<i64>,
    input_shape: Option<&Vec<i64>>,
    model_unbatched: bool,
    preserve_all_axes: bool,
) -> Result<Vec<(usize, usize)>> {
    if pads_tensor.ndim() != 1 {
        return Err(NyError::ModelLoad(format!(
            "Pad {} pads input must be a 1-D tensor, got shape {:?}",
            spec.name,
            pads_tensor.shape()
        )));
    }
    finish_pad_pairs(
        spec,
        pads_tensor.iter().copied().collect(),
        input_shape,
        model_unbatched,
        preserve_all_axes,
    )
}

fn finish_pad_pairs(
    spec: &LayerSpec,
    pads: Vec<i64>,
    input_shape: Option<&Vec<i64>>,
    model_unbatched: bool,
    preserve_all_axes: bool,
) -> Result<Vec<(usize, usize)>> {
    if !pads.len().is_multiple_of(2) {
        return Err(NyError::ModelLoad(format!(
            "Pad {} pads tensor must contain an even number of values, got {}",
            spec.name,
            pads.len()
        )));
    }

    let onnx_rank = pads.len() / 2;
    if onnx_rank == 0 {
        return Err(NyError::ModelLoad(format!(
            "Pad {} pads tensor is empty",
            spec.name
        )));
    }

    if let Some(shape) = input_shape {
        if shape.len() != onnx_rank {
            return Err(NyError::ModelLoad(format!(
                "Pad {} pads rank {} does not match recorded input rank {}",
                spec.name,
                onnx_rank,
                shape.len()
            )));
        }
    } else if !model_unbatched && onnx_rank > 1 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "Pad {} requires a recorded input rank to distinguish the stripped batch axis",
            spec.name
        )));
    }

    let drop_batch_axis = !model_unbatched && !preserve_all_axes && onnx_rank > 1;
    if drop_batch_axis && (pads[0] != 0 || pads[onnx_rank] != 0) {
        return Err(NyError::UnsupportedConfiguration(format!(
            "Pad {} cannot discard nonzero batch-axis padding ({}, {})",
            spec.name, pads[0], pads[onnx_rank]
        )));
    }
    let start_axis = usize::from(drop_batch_axis);

    (start_axis..onnx_rank)
        .map(|axis| {
            let before = pads[axis];
            let after = pads[axis + onnx_rank];
            let before = usize::try_from(before).map_err(|_| {
                NyError::ModelLoad(format!(
                    "Pad {} axis {} has negative pad_before {}",
                    spec.name, axis, before
                ))
            })?;
            let after = usize::try_from(after).map_err(|_| {
                NyError::ModelLoad(format!(
                    "Pad {} axis {} has negative pad_after {}",
                    spec.name, axis, after
                ))
            })?;
            Ok((before, after))
        })
        .collect()
}

fn parse_pad_scalar(spec: &LayerSpec, value: &ArrayD<f32>) -> Result<f32> {
    if value.len() != 1 {
        return Err(NyError::ModelLoad(format!(
            "Pad {} constant value must be scalar, got shape {:?}",
            spec.name,
            value.shape()
        )));
    }
    value.first().copied().ok_or_else(|| {
        NyError::ModelLoad(format!(
            "Pad {} constant value tensor is unexpectedly empty",
            spec.name
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConvertContext;
    use crate::{AttributeValue, LayerSpec, WeightStore};
    use ndarray::{ArrayD, IxDyn};
    use ny_core::LayerType;
    use std::collections::{HashMap, HashSet};

    fn make_context(shape: Vec<i64>, pads: Vec<f32>) -> ConvertContext<'static> {
        let weights = Box::leak(Box::new({
            let mut weights = WeightStore::new();
            weights.insert(
                "pads".to_string(),
                ArrayD::from_shape_vec(IxDyn(&[pads.len()]), pads).unwrap(),
            );
            weights
        }));
        let tensor_shapes = Box::leak(Box::new(HashMap::from([("x".to_string(), shape)])));
        let constant_tensors = Box::leak(Box::new(HashSet::new()));
        ConvertContext::new(weights, tensor_shapes, constant_tensors)
    }

    fn pad_spec(mode: &str) -> LayerSpec {
        LayerSpec {
            name: "pad".to_string(),
            layer_type: LayerType::Pad,
            inputs: vec!["x".to_string(), "pads".to_string()],
            outputs: vec!["y".to_string()],
            weights: None,
            attributes: HashMap::from([(
                "mode".to_string(),
                AttributeValue::String(mode.to_string()),
            )]),
        }
    }

    #[test]
    fn convert_pad_drops_batch_axis_pairs() {
        let ctx = make_context(vec![1, 128, -1], vec![0.0, 0.0, 2.0, 0.0, 0.0, 2.0]);
        let layer = ctx.convert_pad(&pad_spec("reflect")).unwrap();
        match layer {
            Layer::Pad(pad) => {
                assert_eq!(pad.pads, vec![(0, 0), (2, 2)]);
                assert_eq!(pad.mode, PadMode::Reflect);
            }
            other => panic!("expected Pad layer, got {:?}", other),
        }
    }

    #[test]
    fn convert_pad_rejects_adjacent_non_integer_extent() {
        let extent = f32::from_bits(1.0_f32.to_bits() - 1);
        let ctx = make_context(vec![1, 3, 8], vec![0.0, 0.0, extent, 0.0, 0.0, 1.0]);
        let err = ctx
            .convert_pad(&pad_spec("constant"))
            .expect_err("fractional pad extent must not be rounded");
        assert!(err.to_string().contains("non-integer"));
    }

    #[test]
    fn convert_pad_prefers_exact_integer_extents() {
        let weights = Box::leak(Box::new({
            let mut weights = WeightStore::new();
            weights.insert(
                "pads".to_string(),
                ArrayD::from_shape_vec(IxDyn(&[6]), vec![0.0; 6]).unwrap(),
            );
            weights.insert_integers(
                "pads".to_string(),
                ArrayD::from_shape_vec(IxDyn(&[6]), vec![0, 0, 1, 0, 0, 2]).unwrap(),
            );
            weights
        }));
        let tensor_shapes = Box::leak(Box::new(HashMap::from([("x".to_string(), vec![1, 3, 8])])));
        let constant_tensors = Box::leak(Box::new(HashSet::new()));
        let ctx = ConvertContext::new(weights, tensor_shapes, constant_tensors);
        let layer = ctx.convert_pad(&pad_spec("constant")).unwrap();
        let Layer::Pad(pad) = layer else {
            panic!("expected Pad layer");
        };
        assert_eq!(pad.pads, vec![(0, 0), (1, 2)]);
    }

    /// Old ONNX opset (<11): pads stored as "pads" attribute, not as input tensor.
    /// This is needed for TinyYOLO and other legacy models.
    #[test]
    fn convert_pad_from_attribute_opset_lt_11() {
        // No pads in weight store — only data input present
        let weights = Box::leak(Box::new(WeightStore::new()));
        let tensor_shapes = Box::leak(Box::new(HashMap::from([(
            "x".to_string(),
            vec![1, 3, 8, 8],
        )])));
        let constant_tensors = Box::leak(Box::new(HashSet::new()));
        let ctx = ConvertContext::new(weights, tensor_shapes, constant_tensors);

        // Spec with only 1 input (data) and pads as attribute — opset < 11 style
        let spec = LayerSpec {
            name: "Pad_10".to_string(),
            layer_type: LayerType::Pad,
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            weights: None,
            attributes: HashMap::from([
                (
                    "mode".to_string(),
                    AttributeValue::String("constant".to_string()),
                ),
                (
                    "pads".to_string(),
                    // [batch_before, c_before, h_before, w_before,
                    //  batch_after,  c_after,  h_after,  w_after]
                    AttributeValue::Ints(vec![0, 0, 1, 1, 0, 0, 1, 1]),
                ),
                ("value".to_string(), AttributeValue::Float(0.0)),
            ]),
        };

        let layer = ctx.convert_pad(&spec).unwrap();
        match layer {
            Layer::Pad(pad) => {
                // Batch + channel dims dropped, only spatial dims remain
                assert_eq!(pad.pads, vec![(0, 0), (1, 1), (1, 1)]);
                assert_eq!(pad.mode, PadMode::Constant(0.0));
            }
            other => panic!("expected Pad layer, got {:?}", other),
        }
    }

    /// Old opset with "value" attribute for constant padding.
    #[test]
    fn convert_pad_attribute_with_value() {
        let weights = Box::leak(Box::new(WeightStore::new()));
        let tensor_shapes = Box::leak(Box::new(HashMap::from([("x".to_string(), vec![1, 3, 8])])));
        let constant_tensors = Box::leak(Box::new(HashSet::new()));
        let ctx = ConvertContext::new(weights, tensor_shapes, constant_tensors);

        let spec = LayerSpec {
            name: "pad".to_string(),
            layer_type: LayerType::Pad,
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            weights: None,
            attributes: HashMap::from([
                (
                    "mode".to_string(),
                    AttributeValue::String("constant".to_string()),
                ),
                (
                    "pads".to_string(),
                    AttributeValue::Ints(vec![0, 0, 2, 0, 0, 2]),
                ),
                ("value".to_string(), AttributeValue::Float(-1.0)),
            ]),
        };

        let layer = ctx.convert_pad(&spec).unwrap();
        match layer {
            Layer::Pad(pad) => {
                assert_eq!(pad.pads, vec![(0, 0), (2, 2)]);
                assert_eq!(pad.mode, PadMode::Constant(-1.0));
            }
            other => panic!("expected Pad layer, got {:?}", other),
        }
    }

    #[test]
    fn convert_pad_preserves_exact_attribute_ints() {
        let weights = Box::leak(Box::new(WeightStore::new()));
        let tensor_shapes = Box::leak(Box::new(HashMap::from([(
            "x".to_string(),
            vec![1, 3, 8, 8],
        )])));
        let constant_tensors = Box::leak(Box::new(HashSet::new()));
        let ctx = ConvertContext::new(weights, tensor_shapes, constant_tensors);

        let spec = LayerSpec {
            name: "Pad_lossy".to_string(),
            layer_type: LayerType::Pad,
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            weights: None,
            attributes: HashMap::from([
                (
                    "mode".to_string(),
                    AttributeValue::String("constant".to_string()),
                ),
                (
                    "pads".to_string(),
                    AttributeValue::Ints(vec![0, 0, 16_777_217, 0, 0, 0, 0, 0]),
                ),
                ("value".to_string(), AttributeValue::Float(0.0)),
            ]),
        };

        let Layer::Pad(layer) = ctx.convert_pad(&spec).unwrap() else {
            panic!("expected Pad layer");
        };
        assert_eq!(layer.pads, vec![(0, 0), (16_777_217, 0), (0, 0)]);
    }

    #[test]
    fn convert_pad_rejects_axes_subset() {
        let mut spec = pad_spec("constant");
        spec.inputs.push(String::new());
        spec.inputs.push("axes".to_string());
        let ctx = make_context(vec![1, 3, 8, 8], vec![1.0, 1.0, 1.0, 1.0]);
        let err = ctx.convert_pad(&spec).unwrap_err();
        assert!(err.to_string().contains("axes subsets"));
    }

    #[test]
    fn convert_pad_rejects_nonzero_discarded_batch_padding() {
        let ctx = make_context(
            vec![1, 3, 8, 8],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let err = ctx.convert_pad(&pad_spec("constant")).unwrap_err();
        assert!(err.to_string().contains("batch-axis padding"));
    }

    #[test]
    fn convert_pad_keeps_all_axes_for_globally_unbatched_model() {
        let weights = Box::leak(Box::new({
            let mut weights = WeightStore::new();
            weights.insert(
                "pads".to_string(),
                ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            );
            weights
        }));
        let tensor_shapes = Box::leak(Box::new(HashMap::from([("x".to_string(), vec![3, 8])])));
        let constant_tensors = Box::leak(Box::new(HashSet::new()));
        let ctx = ConvertContext::new(weights, tensor_shapes, constant_tensors)
            .with_model_unbatched(true);

        let Layer::Pad(layer) = ctx.convert_pad(&pad_spec("constant")).unwrap() else {
            panic!("expected Pad layer");
        };
        assert_eq!(layer.pads, vec![(1, 3), (2, 4)]);
    }

    #[test]
    fn convert_pad_keeps_all_axes_with_internal_layout_certificate() {
        let ctx = make_context(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0]);
        let mut spec = pad_spec("constant");
        spec.attributes.insert(
            PAD_PRESERVE_ALL_AXES_ATTR.to_string(),
            AttributeValue::Int(1),
        );

        let Layer::Pad(layer) = ctx.convert_pad(&spec).unwrap() else {
            panic!("expected Pad layer");
        };
        assert_eq!(layer.pads, vec![(1, 3), (2, 4)]);

        spec.attributes.insert(
            PAD_PRESERVE_ALL_AXES_ATTR.to_string(),
            AttributeValue::Int(0),
        );
        assert!(ctx.convert_pad(&spec).is_err());
    }

    #[test]
    fn convert_pad_default_mode_uses_constant_value_input() {
        let mut weights = WeightStore::new();
        weights.insert(
            "pads".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 1.0, 0.0, 1.0]).unwrap(),
        );
        weights.insert(
            "value".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![7.0]).unwrap(),
        );
        let tensor_shapes = HashMap::from([("x".to_string(), vec![1, 4])]);
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);
        let spec = LayerSpec {
            name: "pad_default_mode".to_string(),
            layer_type: LayerType::Pad,
            inputs: vec!["x".to_string(), "pads".to_string(), "value".to_string()],
            outputs: vec!["y".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };

        let Layer::Pad(layer) = ctx.convert_pad(&spec).unwrap() else {
            panic!("expected Pad layer");
        };
        assert_eq!(layer.mode, PadMode::Constant(7.0));
    }
}
