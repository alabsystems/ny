// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_propagate::layers::LinearLayer;
use ny_propagate::Layer;
use tracing::debug;

use super::{AttributeValue, ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    pub(crate) fn convert_linear(&self, spec: &LayerSpec) -> Result<Layer> {
        // ONNX Gemm: Y = alpha * A @ B + beta * C
        // Typically: input=A, weight=B, bias=C
        // For transB=1: Y = alpha * A @ B.T + beta * C
        //
        // ONNX Gemm attributes (with defaults):
        //   alpha: float = 1.0
        //   beta: float = 1.0
        //   transA: int = 0
        //   transB: int = 0
        // Reference: https://onnx.ai/onnx/operators/onnx__Gemm.html
        if spec.inputs.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "Linear layer {} has fewer than 2 inputs",
                spec.name
            )));
        }

        // Read ONNX Gemm attributes (#2244)
        let alpha = match spec.attributes.get("alpha") {
            Some(AttributeValue::Float(v)) => *v,
            _ => 1.0,
        };
        let beta = match spec.attributes.get("beta") {
            Some(AttributeValue::Float(v)) => *v,
            _ => 1.0,
        };
        let trans_a = match spec.attributes.get("transA") {
            Some(AttributeValue::Int(v)) => *v != 0,
            _ => false,
        };
        let trans_b_explicit = spec.attributes.get("transB").map(|v| match v {
            AttributeValue::Int(v) => *v != 0,
            _ => false,
        });

        // transA requires input transposition, which ny-propagate doesn't model
        if trans_a {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Gemm node '{}' has transA=1, which requires input transposition \
                 not supported by LinearLayer",
                spec.name
            )));
        }

        // alpha != 1.0 and beta != 1.0 require scaling weights/bias
        if (alpha - 1.0).abs() > f32::EPSILON {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Gemm node '{}' has alpha={}, only alpha=1.0 is supported",
                spec.name, alpha
            )));
        }
        if (beta - 1.0).abs() > f32::EPSILON {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Gemm node '{}' has beta={}, only beta=1.0 is supported",
                spec.name, beta
            )));
        }

        let weight_name = &spec.inputs[1];
        let weight = self
            .weights
            .get(weight_name)
            .ok_or_else(|| NyError::ModelLoad(format!("Weight {} not found", weight_name)))?;

        // Weight shape for LinearLayer: (out_features, in_features)
        let mut weight_2d = weight
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![0, 0],
                got: weight.shape().to_vec(),
            })?;

        // ONNX Gemm transB semantics (ref: alpha-beta-CROWN auto_LiRPA/operators/linear.py:66-79):
        //   transB=1 (common, PyTorch default): B stored as (out_features, in_features)
        //     → matches LinearLayer convention, NO transpose needed
        //   transB=0: B stored as (in_features, out_features) per Y = A @ B
        //     → MUST transpose to (out_features, in_features) for LinearLayer
        match trans_b_explicit {
            Some(true) => {
                // transB=1: weight already in (out, in) form — no transpose
                debug!(
                    "Weight {} shape {:?} with transB=1 — no transpose needed for LinearLayer",
                    weight_name,
                    weight_2d.shape()
                );
            }
            Some(false) => {
                // transB=0: weight in (in, out) form — must transpose
                debug!(
                    "Transposing weight {} (transB=0) from {:?} for LinearLayer",
                    weight_name,
                    weight_2d.shape()
                );
                weight_2d = weight_2d.t().to_owned();
            }
            None => {
                // No transB attribute: ONNX spec defaults to transB=0 (no transpose),
                // meaning weight is stored as (in_features, out_features) and Y = A @ B.
                // Must transpose to (out, in) for LinearLayer.
                //
                // If a WeightRef is available (non-ONNX sources: GGUF, SafeTensors, native),
                // use its shape hint instead, since those formats may store weights either way.
                if let Some(ref weight_ref) = spec.weights {
                    let actual_shape = weight_2d.shape();
                    let expected_shape = &weight_ref.shape;
                    if expected_shape.len() == 2
                        && actual_shape[0] == expected_shape[1]
                        && actual_shape[1] == expected_shape[0]
                    {
                        debug!(
                            "Transposing weight {} from {:?} to {:?} (WeightRef heuristic)",
                            weight_name, actual_shape, expected_shape
                        );
                        weight_2d = weight_2d.t().to_owned();
                    }
                } else {
                    // Pure ONNX model with no WeightRef — apply ONNX default transB=0.
                    // This is the common case for models exported without explicit transB
                    // (e.g., cersyve VNN-COMP 2025 benchmarks).
                    debug!(
                        "Transposing weight {} (ONNX default transB=0) from {:?} for LinearLayer",
                        weight_name,
                        weight_2d.shape()
                    );
                    weight_2d = weight_2d.t().to_owned();
                }
            }
        }

        let bias = if spec.inputs.len() >= 3 {
            let bias_name = &spec.inputs[2];
            let bias_arr = self
                .weights
                .get(bias_name)
                .ok_or_else(|| NyError::ModelLoad(format!("Bias {} not found", bias_name)))?;
            let bias_1d = bias_arr
                .clone()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| {
                    NyError::shape_mismatch(vec![weight_2d.nrows()], bias_arr.shape().to_vec())
                })?;
            Some(bias_1d)
        } else {
            None
        };

        let linear = LinearLayer::new(weight_2d, bias)?;
        Ok(Layer::Linear(linear))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};
    use ny_core::{LayerType, NyError};
    use ny_propagate::Layer;
    use std::collections::HashMap;

    use super::ConvertContext;
    use crate::{AttributeValue, LayerSpec, WeightStore};
    use std::collections::HashSet;

    fn make_context(weights: &WeightStore) -> ConvertContext<'_> {
        static SHAPES: std::sync::LazyLock<HashMap<String, Vec<i64>>> =
            std::sync::LazyLock::new(HashMap::new);
        static CONSTANTS: std::sync::LazyLock<HashSet<String>> =
            std::sync::LazyLock::new(HashSet::new);
        ConvertContext::new(weights, &SHAPES, &CONSTANTS)
    }

    /// Build a Gemm LayerSpec with a (3,2) weight (transB=1 convention: out=3, in=2),
    /// (3,) bias, and given attributes.
    fn gemm_spec_transb1(
        name: &str,
        attrs: HashMap<String, AttributeValue>,
    ) -> (WeightStore, LayerSpec) {
        let mut ws = WeightStore::new();
        // Weight shape (3, 2) with transB=1: out_features=3, in_features=2
        let w = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("invariant: valid shape");
        let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, 0.2, 0.3])
            .expect("invariant: valid shape");
        ws.insert("weight".to_string(), w);
        ws.insert("bias".to_string(), b);

        let spec = LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::Linear,
            inputs: vec![
                "input".to_string(),
                "weight".to_string(),
                "bias".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: attrs,
        };
        (ws, spec)
    }

    /// Build a Gemm LayerSpec with a (2,3) weight (transB=0/ONNX default convention:
    /// in=2, out=3), (3,) bias, and given attributes.
    fn gemm_spec_transb0(
        name: &str,
        attrs: HashMap<String, AttributeValue>,
    ) -> (WeightStore, LayerSpec) {
        let mut ws = WeightStore::new();
        // Weight shape (2, 3) with transB=0: in_features=2, out_features=3
        let w = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
            .expect("invariant: valid shape");
        let b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, 0.2, 0.3])
            .expect("invariant: valid shape");
        ws.insert("weight".to_string(), w);
        ws.insert("bias".to_string(), b);

        let spec = LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::Linear,
            inputs: vec![
                "input".to_string(),
                "weight".to_string(),
                "bias".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: attrs,
        };
        (ws, spec)
    }

    #[ntest::timeout(10000)]
    #[test]
    fn gemm_default_attributes_succeeds() {
        // No transB attribute + no WeightRef → ONNX default transB=0.
        // Weight (2, 3) = (in=2, out=3) per Y = A @ B, transposed to (3, 2) for LinearLayer.
        let (ws, spec) = gemm_spec_transb0("gemm0", HashMap::new());
        let context = make_context(&ws);
        let layer = context
            .convert_linear(&spec)
            .expect("default Gemm attributes must succeed");
        let Layer::Linear(lin) = layer else {
            unreachable!("expected Layer::Linear");
        };
        // Weight (2, 3) transposed to (3, 2) by ONNX default transB=0
        assert_eq!(lin.weight.shape(), &[3, 2]);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn gemm_trans_b_1_no_transpose() {
        // transB=1: weight stored as (out_features=3, in_features=2) — PyTorch convention.
        // No transpose needed for LinearLayer which also stores (out, in).
        // Ref: alpha-beta-CROWN auto_LiRPA/operators/linear.py:73
        let (ws, spec) = gemm_spec_transb1("gemm_transb1", {
            let mut attrs = HashMap::new();
            attrs.insert("transB".to_string(), AttributeValue::Int(1));
            attrs
        });
        let context = make_context(&ws);
        let layer = context
            .convert_linear(&spec)
            .expect("transB=1 must succeed without transpose");
        let Layer::Linear(lin) = layer else {
            unreachable!("expected Layer::Linear");
        };
        // gemm_spec_transb1 creates weight (3, 2) — preserved as-is with transB=1
        assert_eq!(lin.weight.shape(), &[3, 2]);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn gemm_trans_b_0_transposes_weight() {
        // transB=0 explicit: weight stored as (in_features=2, out_features=3) per Y = A @ B.
        // Must transpose to (3, 2) for LinearLayer.
        let mut attrs = HashMap::new();
        attrs.insert("transB".to_string(), AttributeValue::Int(0));
        let (ws, spec) = gemm_spec_transb0("gemm_transb0", attrs);

        let context = make_context(&ws);
        let layer = context
            .convert_linear(&spec)
            .expect("transB=0 must succeed after transpose");
        // Weight (2, 3) transposed to (3, 2)
        let Layer::Linear(lin) = layer else {
            unreachable!("expected Layer::Linear");
        };
        assert_eq!(lin.weight.shape(), &[3, 2]);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn gemm_trans_a_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("transA".to_string(), AttributeValue::Int(1));
        // transA check happens before transB, so weight orientation doesn't matter
        let (ws, spec) = gemm_spec_transb1("gemm_transa", attrs);
        let context = make_context(&ws);
        let err = context
            .convert_linear(&spec)
            .expect_err("transA=1 must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("transA")),
            "expected UnsupportedConfiguration mentioning transA, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn gemm_non_unit_alpha_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("alpha".to_string(), AttributeValue::Float(2.0));
        // alpha check happens before transB, so weight orientation doesn't matter
        let (ws, spec) = gemm_spec_transb1("gemm_alpha", attrs);
        let context = make_context(&ws);
        let err = context
            .convert_linear(&spec)
            .expect_err("alpha!=1.0 must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("alpha")),
            "expected UnsupportedConfiguration mentioning alpha, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn gemm_non_unit_beta_rejected() {
        let mut attrs = HashMap::new();
        attrs.insert("beta".to_string(), AttributeValue::Float(0.5));
        // beta check happens before transB, so weight orientation doesn't matter
        let (ws, spec) = gemm_spec_transb1("gemm_beta", attrs);
        let context = make_context(&ws);
        let err = context
            .convert_linear(&spec)
            .expect_err("beta!=1.0 must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedConfiguration(ref msg) if msg.contains("beta")),
            "expected UnsupportedConfiguration mentioning beta, got: {err:?}"
        );
    }
}
