// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_propagate::layers::{PadLayer, PadMode};
use ny_propagate::Layer;

use super::{i64_to_f32_checked, AttributeValue, ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    pub(crate) fn convert_pad(&self, spec: &LayerSpec) -> Result<Layer> {
        let has_pads_input = spec.inputs.len() >= 2 && !spec.inputs[1].is_empty();

        let input_shape = spec
            .inputs
            .first()
            .and_then(|name| self.tensor_shapes.get(name));

        // Opset >= 11: pads come as a second input tensor.
        // Opset < 11: pads come as an attribute "pads" (list of ints).
        let pads = if has_pads_input {
            let pads_tensor = self.constant_value(&spec.inputs[1]).ok_or_else(|| {
                NyError::UnsupportedConfiguration(format!(
                    "Pad {} requires constant pads input '{}'",
                    spec.name, spec.inputs[1]
                ))
            })?;
            parse_pad_pairs(spec, &pads_tensor, input_shape)?
        } else if let Some(AttributeValue::Ints(ints)) = spec.attributes.get("pads") {
            let pads_tensor = ArrayD::from_shape_vec(
                IxDyn(&[ints.len()]),
                ints.iter()
                    .enumerate()
                    .map(|(index, &value)| {
                        i64_to_f32_checked(
                            value,
                            &format!("Pad {} pads attribute[{index}]", spec.name),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
            .map_err(|e| {
                NyError::ModelLoad(format!(
                    "Pad {} failed to create pads tensor from attribute: {}",
                    spec.name, e
                ))
            })?;
            parse_pad_pairs(spec, &pads_tensor, input_shape)?
        } else {
            return Err(NyError::ModelLoad(format!(
                "Pad {} requires pads as input tensor (opset>=11) or attribute (opset<11)",
                spec.name
            )));
        };

        let mode = match spec.attributes.get("mode") {
            None => PadMode::Constant(0.0),
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

fn parse_pad_pairs(
    spec: &LayerSpec,
    pads_tensor: &ArrayD<f32>,
    input_shape: Option<&Vec<i64>>,
) -> Result<Vec<(usize, usize)>> {
    let pads = pads_tensor
        .iter()
        .copied()
        .map(|value| parse_pad_i64(spec, value))
        .collect::<Result<Vec<_>>>()?;
    if pads.len() % 2 != 0 {
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

    let drop_batch_axis = match input_shape {
        Some(shape) if shape.len() == onnx_rank => onnx_rank > 1,
        Some(shape) if shape.len() + 1 == onnx_rank => false,
        _ => onnx_rank > 1,
    };
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

fn parse_pad_i64(spec: &LayerSpec, value: f32) -> Result<i64> {
    if !value.is_finite() {
        return Err(NyError::ModelLoad(format!(
            "Pad {} pads tensor contains non-finite value {}",
            spec.name, value
        )));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-4 {
        return Err(NyError::ModelLoad(format!(
            "Pad {} pads tensor contains non-integer value {}",
            spec.name, value
        )));
    }
    Ok(rounded as i64)
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
    fn convert_pad_rejects_precision_losing_attribute_ints_4149() {
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

        let error = ctx
            .convert_pad(&spec)
            .expect_err("precision-losing pad attribute should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(msg)
                    if msg.contains("precision loss")
                        && msg.contains("Pad Pad_lossy pads attribute[2]")
            ),
            "unexpected error: {error:?}"
        );
    }
}
