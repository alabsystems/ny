// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_propagate::layers::{BilinearCrownLayer, LinearLayer, MatMulLayer};
use ny_propagate::Layer;
use tracing::{debug, warn};

use super::{i64_to_f32_checked, AttributeValue, ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    pub(crate) fn convert_matmul(&self, spec: &LayerSpec) -> Result<Layer> {
        // MatMul in ONNX: C = A @ B
        // If one input is a weight (constant), treat it as a Linear layer
        // Otherwise, treat it as a bounded MatMul

        if spec.inputs.len() != 2 {
            return Err(NyError::ModelLoad(format!(
                "MatMul {} requires exactly 2 inputs, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let input_a = &spec.inputs[0];
        let input_b = &spec.inputs[1];

        // Check if either input is a constant weight
        let a_is_weight = self.weights.get(input_a).is_some();
        let b_is_weight = self.weights.get(input_b).is_some();

        let transpose_b = match spec.attributes.get("transpose_b") {
            Some(AttributeValue::Int(v)) => *v != 0,
            Some(AttributeValue::Float(v)) => *v != 0.0,
            Some(_) => {
                return Err(NyError::ModelLoad(format!(
                    "MatMul {} has invalid transpose_b attribute type",
                    spec.name
                )));
            }
            None => false,
        };

        let scale = match spec.attributes.get("scale") {
            Some(AttributeValue::Float(v)) => Some(*v),
            Some(AttributeValue::Int(v)) => Some(i64_to_f32_checked(
                *v,
                &format!("MatMul {} scale attribute", spec.name),
            )?),
            Some(_) => {
                return Err(NyError::ModelLoad(format!(
                    "MatMul {} has invalid scale attribute type",
                    spec.name
                )));
            }
            None => None,
        };
        if let Some(scale) = scale {
            if !scale.is_finite() {
                return Err(NyError::InvalidSpec(format!(
                    "MatMul {} scale must be finite, got {}",
                    spec.name, scale
                )));
            }
        }

        if b_is_weight && !a_is_weight {
            // Standard case: A @ W where W is a constant weight
            // This is equivalent to a Linear layer (without bias)
            let weight = self.weights.get(input_b).ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "MatMul {} missing constant weight input '{}'",
                    spec.name, input_b
                ))
            })?;

            // ONNX MatMul spec: if B is 1D (K,), promote to (K, 1), compute matmul,
            // then remove the appended dimension. The builder inserts a Squeeze(-1) node
            // after this Linear layer to handle the dimension removal.
            let weight = if weight.ndim() == 1 {
                let k = weight.len();
                debug!(
                    "MatMul {} has 1D weight B ({}), promoting to ({}, 1) per ONNX spec",
                    spec.name, k, k
                );
                weight
                    .clone()
                    .into_shape_with_order(ndarray::IxDyn(&[k, 1]))
                    .map_err(|e| {
                        NyError::ModelLoad(format!("Cannot reshape 1D MatMul weight to 2D: {}", e))
                    })?
            } else {
                weight.clone()
            };

            // MatMul semantics: input shape (*, K), weight shape (K, N), output (*, N)
            // For Linear layer, we need weight shape (N, K) so we transpose
            if weight.ndim() != 2 {
                return Err(NyError::ModelLoad(format!(
                    "MatMul weight must be 2D, got {}D",
                    weight.ndim()
                )));
            }

            let weight_2d = weight
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| NyError::ModelLoad(format!("Cannot reshape MatMul weight: {}", e)))?;

            // MatMul supports an optional `transpose_b` attribute in our fused graphs.
            //
            // - If `transpose_b=false`: MatMul is A @ W, with W expected to be (K, N).
            //   Linear expects (N, K), so we transpose here.
            // - If `transpose_b=true`: MatMul is A @ W^T, with W expected to be (N, K).
            //   Linear expects (N, K), so we use W as-is.
            let mut linear_weight = if transpose_b {
                weight_2d
            } else {
                weight_2d.t().to_owned()
            };
            if let Some(s) = scale {
                linear_weight.mapv_inplace(|v| v * s);
            }
            debug!(
                "MatMul {} with constant second input converted to Linear layer",
                spec.name
            );
            Ok(Layer::Linear(LinearLayer::new(linear_weight, None)?))
        } else if a_is_weight && !b_is_weight {
            // Less common: W @ B where W is constant
            // This is also equivalent to a Linear layer, but weight is already in correct format
            // MatMul semantics: W shape (M, K), B shape (K,), output (M,)
            // For Linear layer, weight should be (out_features, in_features) = (M, K) - no transpose needed
            let weight = self.weights.get(input_a).ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "MatMul {} missing constant weight input '{}'",
                    spec.name, input_a
                ))
            })?;

            // ONNX MatMul spec: if A is 1D (K,), promote to (1, K), compute matmul,
            // then remove the prepended dimension. The builder inserts a Squeeze node
            // after this Linear layer to handle the dimension removal.
            let weight = if weight.ndim() == 1 {
                let k = weight.len();
                debug!(
                    "MatMul {} has 1D weight A ({}), promoting to (1, {}) per ONNX spec",
                    spec.name, k, k
                );
                weight
                    .clone()
                    .into_shape_with_order(ndarray::IxDyn(&[1, k]))
                    .map_err(|e| {
                        NyError::ModelLoad(format!("Cannot reshape 1D MatMul weight to 2D: {}", e))
                    })?
            } else {
                weight.clone()
            };

            if weight.ndim() != 2 {
                return Err(NyError::ModelLoad(format!(
                    "MatMul weight must be 2D, got {}D",
                    weight.ndim()
                )));
            }

            let weight_2d = weight
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| NyError::ModelLoad(format!("Cannot reshape MatMul weight: {}", e)))?;

            // No transpose needed: W @ x expects W in (out, in) format which matches Linear
            let mut linear_weight = weight_2d;
            if let Some(s) = scale {
                linear_weight.mapv_inplace(|v| v * s);
            }
            debug!(
                "MatMul {} with constant first input converted to Linear layer",
                spec.name
            );
            Ok(Layer::Linear(LinearLayer::new(linear_weight, None)?))
        } else if !a_is_weight && !b_is_weight {
            // Neither input is a weight - true bounded MatMul (e.g., Q @ K^T or probs @ V).
            // Use BilinearCrownLayer for all bilinear MatMuls regardless of transpose_b.
            // BilinearCrown uses CompactMcCormick for seq > 64 (#286), avoiding the
            // O(seq^4) dense identity that blocks the MatMul retry path in crown_batched.rs.
            // CompactMcCormick supports both transpose_b configurations (verified by proptests).
            debug!(
                "MatMul {} is a bounded binary operation (both inputs are activations), using BilinearCrownLayer",
                spec.name
            );
            Ok(Layer::BilinearCrown(
                BilinearCrownLayer::try_new(transpose_b, scale).map_err(|err| {
                    NyError::InvalidSpec(format!("MatMul {} scale invalid: {err}", spec.name))
                })?,
            ))
        } else {
            // Both are weights - should be constant folded
            warn!(
                "MatMul {} has both constant inputs - should be constant folded",
                spec.name
            );
            Ok(Layer::MatMul(
                MatMulLayer::try_new(transpose_b, scale).map_err(|err| {
                    NyError::InvalidSpec(format!("MatMul {} scale invalid: {err}", spec.name))
                })?,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightStore;
    use ny_core::LayerType;
    use std::collections::{HashMap, HashSet};

    /// Regression test for #2666: MatMul with extra inputs must be rejected.
    #[test]
    fn convert_matmul_rejects_extra_inputs_2666() {
        let weights = WeightStore::new();
        let shapes = HashMap::new();
        let constants = HashSet::new();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = LayerSpec {
            name: "matmul_op".to_string(),
            layer_type: LayerType::MatMul,
            inputs: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };

        let error = context
            .convert_matmul(&spec)
            .expect_err("MatMul with 3 inputs should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(msg) if msg.contains("exactly 2 inputs")
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn convert_matmul_rejects_precision_losing_scale_attr_4149() {
        let weights = WeightStore::new();
        let shapes = HashMap::new();
        let constants = HashSet::new();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = LayerSpec {
            name: "matmul_lossy_scale".to_string(),
            layer_type: LayerType::MatMul,
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::from([("scale".to_string(), AttributeValue::Int(16_777_217))]),
        };

        let error = context
            .convert_matmul(&spec)
            .expect_err("precision-losing MatMul scale should be rejected");
        assert!(
            matches!(
                &error,
                NyError::ModelLoad(msg)
                    if msg.contains("precision loss")
                        && msg.contains("MatMul matmul_lossy_scale scale attribute")
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn convert_matmul_rejects_non_finite_scale_attr_4307() {
        let weights = WeightStore::new();
        let shapes = HashMap::new();
        let constants = HashSet::new();
        let context = ConvertContext::new(&weights, &shapes, &constants);
        let spec = LayerSpec {
            name: "matmul_nan_scale".to_string(),
            layer_type: LayerType::MatMul,
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::from([("scale".to_string(), AttributeValue::Float(f32::NAN))]),
        };

        let error = context
            .convert_matmul(&spec)
            .expect_err("non-finite MatMul scale should be rejected");
        assert!(
            matches!(
                &error,
                NyError::InvalidSpec(msg)
                    if msg.contains("MatMul matmul_nan_scale scale must be finite")
            ),
            "unexpected error: {error:?}"
        );
    }
}
