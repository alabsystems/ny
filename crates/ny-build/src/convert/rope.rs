// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX RoPE (Rotary Position Embedding) conversion.
//!
//! Custom op "RoPE" / "RotaryPositionEmbedding" with:
//!   input[0]: activation tensor (shape [..., head_dim])
//!   input[1]: cos_freqs weight (shape [num_pairs] where num_pairs = head_dim/2)
//!   input[2]: sin_freqs weight (shape [num_pairs])

use ny_core::{NyError, Result};
use ny_propagate::layers::RopeLayer;
use ny_propagate::Layer;

use super::{ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    pub(crate) fn convert_rope(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() < 3 {
            return Err(NyError::ModelLoad(format!(
                "RoPE layer {} requires 3 inputs (activation, cos_freqs, sin_freqs), got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let cos_name = &spec.inputs[1];
        let sin_name = &spec.inputs[2];

        let cos_arr = self.weights.get(cos_name).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "RoPE layer {}: cos_freqs weight '{}' not found",
                spec.name, cos_name
            ))
        })?;

        let sin_arr = self.weights.get(sin_name).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "RoPE layer {}: sin_freqs weight '{}' not found",
                spec.name, sin_name
            ))
        })?;

        // Flatten to 1D vectors — cos/sin may be stored as [1, num_pairs] or [num_pairs].
        let cos_freqs: Vec<f32> = cos_arr.iter().copied().collect();
        let sin_freqs: Vec<f32> = sin_arr.iter().copied().collect();

        let layer = RopeLayer::new(cos_freqs, sin_freqs).map_err(|err| {
            NyError::ModelLoad(format!(
                "RoPE layer {} malformed cos/sin tables: {err}",
                spec.name
            ))
        })?;
        Ok(Layer::RoPE(layer))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};
    use ny_core::LayerType;
    use ny_propagate::Layer;
    use std::collections::{HashMap, HashSet};

    use super::ConvertContext;
    use crate::{LayerSpec, WeightStore};

    fn make_context(weights: &WeightStore) -> ConvertContext<'_> {
        static SHAPES: std::sync::LazyLock<HashMap<String, Vec<i64>>> =
            std::sync::LazyLock::new(HashMap::new);
        static CONSTANTS: std::sync::LazyLock<HashSet<String>> =
            std::sync::LazyLock::new(HashSet::new);
        ConvertContext::new(weights, &SHAPES, &CONSTANTS)
    }

    /// End-to-end test: verifies `convert_layer()` dispatches RoPE correctly
    /// and the resulting layer has the expected number of pairs.
    #[ntest::timeout(10000)]
    #[test]
    fn convert_rope_from_cos_sin_weights() {
        let mut ws = WeightStore::new();
        // 2 pairs (head_dim=4): cos and sin for angles [0.0, π/4]
        let angle0 = 0.0f32;
        let angle1 = std::f32::consts::FRAC_PI_4;
        ws.insert(
            "cos_freqs".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![angle0.cos(), angle1.cos()]).unwrap(),
        );
        ws.insert(
            "sin_freqs".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![angle0.sin(), angle1.sin()]).unwrap(),
        );

        let spec = LayerSpec {
            name: "rope0".to_string(),
            layer_type: LayerType::RoPE,
            inputs: vec![
                "input".to_string(),
                "cos_freqs".to_string(),
                "sin_freqs".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };

        // Test via convert_layer() — not convert_rope() directly — to verify
        // the dispatch arm in mod.rs is wired correctly.
        let context = make_context(&ws);
        let layer = context
            .convert_layer(&spec)
            .expect("RoPE conversion via convert_layer must succeed");
        match &layer {
            Layer::RoPE(rope) => {
                assert_eq!(rope.num_pairs(), 2, "expected 2 pairs for head_dim=4");
                assert_eq!(rope.head_dim(), 4, "expected head_dim=4 for 2 pairs");
            }
            other => panic!("expected Layer::RoPE, got {}", other.layer_type()),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn convert_rope_bf16_quantized_cos_sin_weights() {
        let mut ws = WeightStore::new();
        // cos(1.0) and sin(1.0) rounded independently to bf16, then decoded to f32.
        ws.insert(
            "cos_freqs".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.539_062_5]).unwrap(),
        );
        ws.insert(
            "sin_freqs".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.839_843_75]).unwrap(),
        );

        let spec = LayerSpec {
            name: "rope_bf16".to_string(),
            layer_type: LayerType::RoPE,
            inputs: vec![
                "input".to_string(),
                "cos_freqs".to_string(),
                "sin_freqs".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };

        let context = make_context(&ws);
        let layer = context
            .convert_layer(&spec)
            .expect("bf16-rounded RoPE tables should remain valid");
        match &layer {
            Layer::RoPE(rope) => {
                assert_eq!(rope.num_pairs(), 1, "expected 1 pair for head_dim=2");
                assert_eq!(rope.head_dim(), 2, "expected head_dim=2 for 1 pair");
            }
            other => panic!("expected Layer::RoPE, got {}", other.layer_type()),
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn convert_rope_missing_cos_fails() {
        let mut ws = WeightStore::new();
        ws.insert(
            "sin_freqs".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap(),
        );

        let spec = LayerSpec {
            name: "rope_no_cos".to_string(),
            layer_type: LayerType::RoPE,
            inputs: vec![
                "input".to_string(),
                "cos_freqs".to_string(),
                "sin_freqs".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };

        let context = make_context(&ws);
        let err = context
            .convert_rope(&spec)
            .expect_err("missing cos_freqs must fail");
        assert!(
            err.to_string().contains("cos_freqs"),
            "error should mention cos_freqs: {err}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn convert_rope_missing_sin_fails() {
        let mut ws = WeightStore::new();
        ws.insert(
            "cos_freqs".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 0.707]).unwrap(),
        );

        let spec = LayerSpec {
            name: "rope_no_sin".to_string(),
            layer_type: LayerType::RoPE,
            inputs: vec![
                "input".to_string(),
                "cos_freqs".to_string(),
                "sin_freqs".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };

        let context = make_context(&ws);
        let err = context
            .convert_rope(&spec)
            .expect_err("missing sin_freqs must fail");
        assert!(
            err.to_string().contains("sin_freqs"),
            "error should mention sin_freqs: {err}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn convert_rope_too_few_inputs_fails() {
        let ws = WeightStore::new();
        let spec = LayerSpec {
            name: "rope_no_inputs".to_string(),
            layer_type: LayerType::RoPE,
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };

        let context = make_context(&ws);
        let err = context
            .convert_rope(&spec)
            .expect_err("too few inputs must fail");
        assert!(
            err.to_string().contains("3 inputs"),
            "error should mention 3 inputs: {err}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn convert_rope_non_unit_cos_sin_fails() {
        let mut ws = WeightStore::new();
        ws.insert(
            "cos_freqs".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        );
        ws.insert(
            "sin_freqs".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        );

        let spec = LayerSpec {
            name: "rope_bad_norm".to_string(),
            layer_type: LayerType::RoPE,
            inputs: vec![
                "input".to_string(),
                "cos_freqs".to_string(),
                "sin_freqs".to_string(),
            ],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };

        let context = make_context(&ws);
        let err = context
            .convert_rope(&spec)
            .expect_err("non-unit cos/sin tables must fail");
        let error_text = err.to_string();
        assert!(
            error_text.contains("rope_bad_norm"),
            "error should mention the layer name: {err}"
        );
        assert!(
            error_text.contains("unit-rotation invariant"),
            "error should mention the RoPE invariant: {err}"
        );
    }
}
