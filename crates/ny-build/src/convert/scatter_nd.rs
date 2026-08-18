// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

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
            .discrete_constant_i64(&spec.inputs[1], &format!("ScatterND {} indices", spec.name))?;
        let updates_constant = self.constant_value(&spec.inputs[2]);

        Ok(Layer::ScatterNd(ScatterNdLayer::new(
            data_constant,
            indices,
            updates_constant,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightStore;
    use ndarray::{ArrayD, IxDyn};
    use ny_core::LayerType;
    use ny_tensor::BoundedTensor;
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

    #[test]
    fn scatter_nd_rejects_adjacent_non_integer_indices() {
        for value in [
            f32::from_bits(1.0_f32.to_bits() - 1),
            f32::from_bits(1.0_f32.to_bits() + 1),
        ] {
            let mut weights = WeightStore::new();
            weights.insert(
                "indices".to_string(),
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![value]).unwrap(),
            );
            let tensor_shapes = HashMap::new();
            let constant_tensors = HashSet::new();
            let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);
            assert!(ctx.convert_scatter_nd(&scatter_spec()).is_err());
        }
    }

    #[test]
    fn scatter_nd_prefers_exact_integer_indices() {
        let mut weights = WeightStore::new();
        weights.insert(
            "indices".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0]).unwrap(),
        );
        weights.insert_integers(
            "indices".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1_i64]).unwrap(),
        );
        let tensor_shapes = HashMap::new();
        let constant_tensors = HashSet::new();
        let ctx = ConvertContext::new(&weights, &tensor_shapes, &constant_tensors);
        let layer = ctx.convert_scatter_nd(&scatter_spec()).unwrap();
        let Layer::ScatterNd(scatter) = layer else {
            panic!("expected ScatterNd layer");
        };
        let data_point = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
        let data = BoundedTensor::new(data_point.clone(), data_point).unwrap();
        let updates_point = ArrayD::from_shape_vec(IxDyn(&[1]), vec![9.0]).unwrap();
        let updates = BoundedTensor::new(updates_point.clone(), updates_point).unwrap();
        let output = scatter.propagate_ibp_binary(&data, &updates).unwrap();
        assert_eq!(output.lower().as_slice().unwrap(), &[1.0, 9.0]);
    }
}
