// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_propagate::layers::{ArgMaxLayer, ArgMinLayer, ArgSortLayer, TopkLayer, TopkOutputKind};
use ny_propagate::Layer;

use super::{AttributeValue, ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    pub(crate) fn convert_topk(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() != 1 {
            return Err(NyError::ModelLoad(format!(
                "Topk {} requires 1 input, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let axis = resolve_unbatched_axis(spec, "Topk", self)?;
        let k = read_positive_usize_attr(spec, "k", "Topk")?;
        let output = read_topk_output_kind(spec)?;
        Ok(Layer::Topk(TopkLayer::new(k, axis, output)))
    }

    pub(crate) fn convert_argmax(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() != 1 {
            return Err(NyError::ModelLoad(format!(
                "Argmax {} requires 1 input, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let keepdims = read_keepdims_attr(spec, "Argmax")?;
        Ok(Layer::ArgMax(ArgMaxLayer::new(
            resolve_unbatched_axis(spec, "Argmax", self)?,
            keepdims,
        )))
    }

    pub(crate) fn convert_argmin(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() != 1 {
            return Err(NyError::ModelLoad(format!(
                "Argmin {} requires 1 input, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let keepdims = read_keepdims_attr(spec, "Argmin")?;
        Ok(Layer::ArgMin(ArgMinLayer::new(
            resolve_unbatched_axis(spec, "Argmin", self)?,
            keepdims,
        )))
    }

    pub(crate) fn convert_argsort(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() != 1 {
            return Err(NyError::ModelLoad(format!(
                "ArgSort {} requires 1 input, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let descending = match spec.attributes.get("descending") {
            Some(AttributeValue::Int(v)) => *v != 0,
            Some(other) => {
                return Err(NyError::ModelLoad(format!(
                    "ArgSort {} descending attribute must be Int, got {:?}",
                    spec.name, other
                )));
            }
            None => false,
        };

        Ok(Layer::ArgSort(ArgSortLayer::new(
            resolve_unbatched_axis(spec, "ArgSort", self)?,
            descending,
        )))
    }
}

fn resolve_unbatched_axis(spec: &LayerSpec, op: &str, ctx: &ConvertContext<'_>) -> Result<i64> {
    let raw_axis = match spec
        .attributes
        .get("axis")
        .or_else(|| spec.attributes.get("dim"))
    {
        Some(AttributeValue::Int(axis)) => *axis,
        Some(other) => {
            return Err(NyError::ModelLoad(format!(
                "{op} {} axis attribute must be Int, got {:?}",
                spec.name, other
            )));
        }
        None => {
            return Err(NyError::ModelLoad(format!(
                "{op} {} requires axis/dim attribute",
                spec.name
            )));
        }
    };

    // Trailing-relative remap: correct under both internal runtime layouts
    // (leading batch dim stripped OR retained); refuses ambiguous cases.
    // See `ConvertContext::remap_axis_trailing` (#pensieve ReduceSum no-op).
    let data_name = spec
        .inputs
        .first()
        .map(String::as_str)
        .ok_or_else(|| NyError::ModelLoad(format!("{op} '{}' has no data input", spec.name)))?;
    ctx.remap_axis_trailing(
        op,
        &spec.name,
        data_name,
        raw_axis,
        super::LegacyBatchAxisPolicy::RejectZero,
    )
}

fn read_positive_usize_attr(spec: &LayerSpec, key: &str, op: &str) -> Result<usize> {
    let value = match spec.attributes.get(key) {
        Some(AttributeValue::Int(value)) => *value,
        Some(other) => {
            return Err(NyError::ModelLoad(format!(
                "{op} {} attribute '{key}' must be Int, got {:?}",
                spec.name, other
            )));
        }
        None => {
            return Err(NyError::ModelLoad(format!(
                "{op} {} requires '{key}' attribute",
                spec.name
            )));
        }
    };
    if value <= 0 {
        return Err(NyError::ModelLoad(format!(
            "{op} {} attribute '{key}' must be > 0, got {}",
            spec.name, value
        )));
    }
    usize::try_from(value).map_err(|_| {
        NyError::ModelLoad(format!(
            "{op} {} attribute '{key}' does not fit usize: {}",
            spec.name, value
        ))
    })
}

fn read_keepdims_attr(spec: &LayerSpec, op: &str) -> Result<bool> {
    match spec.attributes.get("keepdims") {
        Some(AttributeValue::Int(value)) => Ok(*value != 0),
        Some(other) => Err(NyError::ModelLoad(format!(
            "{op} {} attribute 'keepdims' must be Int, got {:?}",
            spec.name, other
        ))),
        None => Ok(false),
    }
}

fn read_topk_output_kind(spec: &LayerSpec) -> Result<TopkOutputKind> {
    match spec
        .attributes
        .get("output")
        .or_else(|| spec.attributes.get("output_kind"))
    {
        Some(AttributeValue::String(kind)) => match kind.as_str() {
            "values" | "value" => Ok(TopkOutputKind::Values),
            "indices" | "index" => Ok(TopkOutputKind::Indices),
            other => Err(NyError::ModelLoad(format!(
                "Topk {} output attribute must be 'values' or 'indices', got '{}'",
                spec.name, other
            ))),
        },
        Some(AttributeValue::Int(value)) => match *value {
            0 => Ok(TopkOutputKind::Values),
            1 => Ok(TopkOutputKind::Indices),
            other => Err(NyError::ModelLoad(format!(
                "Topk {} output attribute Int must be 0 (values) or 1 (indices), got {}",
                spec.name, other
            ))),
        },
        Some(other) => Err(NyError::ModelLoad(format!(
            "Topk {} output attribute must be String or Int, got {:?}",
            spec.name, other
        ))),
        None => Ok(TopkOutputKind::Values),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightStore;
    use ndarray::{arr1, ArrayD, IxDyn};
    use ny_core::LayerType;
    use ny_propagate::layers::BoundPropagation;
    use ny_tensor::BoundedTensor;
    use std::collections::{HashMap, HashSet};

    fn make_ctx() -> ConvertContext<'static> {
        let weights = Box::leak(Box::new(WeightStore::new()));
        let tensor_shapes = Box::leak(Box::new(HashMap::new()));
        let constant_tensors = Box::leak(Box::new(HashSet::new()));
        ConvertContext::new(weights, tensor_shapes, constant_tensors)
    }

    fn spec(layer_type: LayerType, attrs: HashMap<String, AttributeValue>) -> LayerSpec {
        LayerSpec {
            name: "sel".to_string(),
            layer_type,
            inputs: vec!["input".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: attrs,
        }
    }

    #[test]
    fn convert_topk_keeps_axis_and_k() {
        let ctx = make_ctx();
        let layer = ctx
            .convert_topk(&spec(
                LayerType::Topk,
                HashMap::from([
                    ("axis".to_string(), AttributeValue::Int(-1)),
                    ("k".to_string(), AttributeValue::Int(2)),
                ]),
            ))
            .expect("Topk conversion should succeed");
        let Layer::Topk(layer) = layer else {
            panic!("expected Topk layer");
        };
        assert_eq!(layer.axis, -1);
        assert_eq!(layer.k, 2);
        assert_eq!(layer.output, TopkOutputKind::Values);
        let input = BoundedTensor::new(
            arr1(&[1.0_f32, -2.0, 4.0]).into_dyn(),
            arr1(&[3.0, 0.0, 5.0]).into_dyn(),
        )
        .expect("valid bounded tensor");
        let output = layer
            .propagate_ibp(&input)
            .expect("Topk IBP should succeed");
        assert_eq!(output.shape(), &[2]);
    }

    #[test]
    fn convert_topk_reads_indices_output_flag() {
        let ctx = make_ctx();
        let layer = ctx
            .convert_topk(&spec(
                LayerType::Topk,
                HashMap::from([
                    ("axis".to_string(), AttributeValue::Int(-1)),
                    ("k".to_string(), AttributeValue::Int(2)),
                    (
                        "output".to_string(),
                        AttributeValue::String("indices".to_string()),
                    ),
                ]),
            ))
            .expect("Topk conversion should succeed");
        let Layer::Topk(layer) = layer else {
            panic!("expected Topk layer");
        };
        assert_eq!(layer.output, TopkOutputKind::Indices);
    }

    #[test]
    fn convert_argmax_reads_keepdims_flag() {
        let ctx = make_ctx();
        let layer = ctx
            .convert_argmax(&spec(
                LayerType::Argmax,
                HashMap::from([
                    ("axis".to_string(), AttributeValue::Int(-1)),
                    ("keepdims".to_string(), AttributeValue::Int(1)),
                ]),
            ))
            .expect("Argmax conversion should succeed");
        let Layer::ArgMax(layer) = layer else {
            panic!("expected ArgMax layer");
        };
        assert!(layer.keepdims);
    }

    #[test]
    fn convert_argsort_reads_descending_flag() {
        let ctx = make_ctx();
        let layer = ctx
            .convert_argsort(&spec(
                LayerType::ArgSort,
                HashMap::from([
                    ("axis".to_string(), AttributeValue::Int(-1)),
                    ("descending".to_string(), AttributeValue::Int(1)),
                ]),
            ))
            .expect("ArgSort conversion should succeed");
        let Layer::ArgSort(layer) = layer else {
            panic!("expected ArgSort layer");
        };
        let input = BoundedTensor::concrete(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0_f32, 4.0, 2.0])
                .expect("valid concrete tensor"),
        )
        .expect("bounded tensor should construct");
        let output = layer
            .propagate_ibp(&input)
            .expect("ArgSort IBP should succeed");
        assert_eq!(output.lower().as_slice().unwrap(), &[1.0, 2.0, 0.0]);
        assert_eq!(output.upper().as_slice().unwrap(), &[1.0, 2.0, 0.0]);
    }
}
