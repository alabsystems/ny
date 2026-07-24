// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_propagate::layers::ScatterNdLayer;
use ny_propagate::Layer;

use super::{AttributeValue, ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    pub(crate) fn convert_scatter_nd(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() != 3 {
            return Err(NyError::ModelLoad(format!(
                "ScatterND {} requires 3 inputs (data, indices, updates), got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        if let Some(AttributeValue::String(reduction)) = spec.attributes.get("reduction") {
            if reduction != "none" {
                return Err(NyError::UnsupportedConfiguration(format!(
                    "ScatterND {} reduction='{}' is not supported (expected 'none')",
                    spec.name, reduction
                )));
            }
        }

        let data_constant = self.constant_value(&spec.inputs[0]);
        let indices = self
            .constant_value(&spec.inputs[1])
            .map(parse_indices_i64)
            .transpose()
            .map_err(|err| {
                NyError::ModelLoad(format!("ScatterND {} indices error: {}", spec.name, err))
            })?;
        let updates_constant = self.constant_value(&spec.inputs[2]);

        Ok(Layer::ScatterNd(ScatterNdLayer::new(
            data_constant,
            indices,
            updates_constant,
        )))
    }
}

fn parse_indices_i64(arr: ArrayD<f32>) -> Result<ArrayD<i64>> {
    let shape = arr.shape().to_vec();
    let mut values = Vec::with_capacity(arr.len());
    for &v in &arr {
        if !v.is_finite() {
            return Err(NyError::InvalidSpec(
                "ScatterND indices contain NaN/Inf".to_string(),
            ));
        }
        let rounded = v.round();
        if (v - rounded).abs() > 1e-6 {
            return Err(NyError::InvalidSpec(format!(
                "ScatterND indices must be integers; got {}",
                v
            )));
        }
        values.push(rounded as i64);
    }

    ArrayD::from_shape_vec(IxDyn(&shape), values)
        .map_err(|e| NyError::InvalidSpec(format!("ScatterND indices reshape failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightStore;
    use ny_core::LayerType;
    use std::collections::{HashMap, HashSet};

    fn scatter_spec() -> LayerSpec {
        LayerSpec {
            name: "scatter".to_string(),
            layer_type: LayerType::ScatterND,
            inputs: vec![
                "data".to_string(),
                "indices".to_string(),
                "updates".to_string(),
            ],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        }
    }

    #[test]
    fn convert_scatter_nd_embeds_constant_data_and_indices() {
        let mut weights = WeightStore::new();
        weights.insert(
            "data".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
        );
        weights.insert(
            "indices".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0, 3.0]).unwrap(),
        );
        let tensor_shapes = HashMap::new();
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);

        let layer = ctx.convert_scatter_nd(&scatter_spec()).unwrap();
        let Layer::ScatterNd(scatter) = layer else {
            panic!("expected ScatterNd layer");
        };

        assert_eq!(scatter.activation_input_count(), 1);
        assert!(scatter.has_static_indices());
    }
}
