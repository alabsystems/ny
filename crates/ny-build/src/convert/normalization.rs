// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_propagate::layers::{
    AdaIN1dLayer, BatchNormLayer, GroupNormLayer, InstanceNorm1dLayer, LayerNormLayer, RmsNormLayer,
};
use ny_propagate::Layer;
use tracing::{debug, warn};

use super::{AttributeValue, ConvertContext, LayerSpec};
use crate::layernorm_mode_from_attrs;

/// Load an optional 1-D weight for LayerNorm. Returns error if the weight
/// exists but has wrong dimensionality. Returns `default_fn()` if missing.
fn load_layernorm_1d_weight(
    weights: &crate::WeightStore,
    layer_name: &str,
    weight_name: &str,
    param_label: &str,
    default_fn: impl FnOnce() -> ndarray::Array1<f32>,
) -> Result<ndarray::Array1<f32>> {
    match weights.get(weight_name) {
        Some(w) => w
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|e| {
                NyError::ModelLoad(format!(
                    "LayerNorm {} {} '{}' has shape {:?}, expected 1-D: {}",
                    layer_name,
                    param_label,
                    weight_name,
                    w.shape(),
                    e
                ))
            }),
        None => Ok(default_fn()),
    }
}

impl ConvertContext<'_> {
    pub(crate) fn convert_layer_norm(&self, spec: &LayerSpec) -> Result<Layer> {
        let attr_norm_size = match spec.attributes.get("normalized_shape") {
            Some(AttributeValue::Ints(dims)) if !dims.is_empty() => {
                // #2983: Guard negative normalized_shape dimension.
                Some(dims.last().copied().unwrap_or(1).max(1) as usize)
            }
            _ => None,
        };

        let norm_size = self.resolve_layernorm_size(spec, attr_norm_size);

        let ny = if spec.inputs.len() >= 2 {
            load_layernorm_1d_weight(self.weights, &spec.name, &spec.inputs[1], "ny", || {
                ndarray::Array1::ones(norm_size)
            })?
        } else {
            ndarray::Array1::ones(norm_size)
        };

        let beta = if spec.inputs.len() >= 3 {
            load_layernorm_1d_weight(self.weights, &spec.name, &spec.inputs[2], "beta", || {
                ndarray::Array1::zeros(norm_size)
            })?
        } else {
            ndarray::Array1::zeros(norm_size)
        };

        let eps = match spec.attributes.get("epsilon") {
            Some(AttributeValue::Float(e)) => *e,
            _ => 1e-5,
        };

        let mode = layernorm_mode_from_attrs(spec);
        Ok(Layer::LayerNorm(
            LayerNormLayer::new(ny, beta, eps)?.with_mode(mode),
        ))
    }

    fn resolve_layernorm_size(&self, spec: &LayerSpec, attr_norm_size: Option<usize>) -> usize {
        if spec.inputs.len() >= 2 {
            let ny_name = &spec.inputs[1];
            if let Some(ny) = self.weights.get(ny_name) {
                ny.len()
            } else if let Some(size) = attr_norm_size {
                debug!(
                    "LayerNorm ny {} not found, using normalized_shape {}",
                    ny_name, size
                );
                size
            } else {
                warn!("LayerNorm ny not found and no normalized_shape, using default size 1");
                1
            }
        } else if let Some(size) = attr_norm_size {
            debug!("LayerNorm has no ny input, using normalized_shape {}", size);
            size
        } else {
            warn!("LayerNorm inputs incomplete and no normalized_shape, using default size 1");
            1
        }
    }
    pub(crate) fn convert_rms_norm(&self, spec: &LayerSpec) -> Result<Layer> {
        // ONNX SimplifiedLayerNormalization / RMSNormalization inputs: X, ny
        // RMSNorm has no beta parameter.
        let norm_size = if spec.inputs.len() >= 2 {
            let ny_name = &spec.inputs[1];
            if let Some(ny) = self.weights.get(ny_name) {
                ny.len()
            } else {
                match spec.attributes.get("normalized_shape") {
                    Some(AttributeValue::Ints(dims)) if !dims.is_empty() => {
                        dims.last().copied().unwrap_or(1).max(1) as usize
                    }
                    _ => {
                        warn!("RMSNorm ny not found and no normalized_shape, using default size 1");
                        1
                    }
                }
            }
        } else {
            warn!("RMSNorm has no ny input, using default size 1");
            1
        };

        let ny = if spec.inputs.len() >= 2 {
            load_layernorm_1d_weight(self.weights, &spec.name, &spec.inputs[1], "ny", || {
                ndarray::Array1::ones(norm_size)
            })?
        } else {
            ndarray::Array1::ones(norm_size)
        };

        let eps = match spec.attributes.get("epsilon") {
            Some(AttributeValue::Float(e)) => *e,
            _ => 1e-5,
        };

        Ok(Layer::RmsNorm(RmsNormLayer::new(ny, eps)?))
    }

    pub(crate) fn convert_instance_norm(&self, spec: &LayerSpec) -> Result<Layer> {
        // ONNX InstanceNormalization inputs: X, scale (ny), B (beta)
        // Attributes: epsilon (float, default 1e-5)
        let num_channels = if spec.inputs.len() >= 2 {
            let ny_name = &spec.inputs[1];
            if let Some(ny) = self.weights.get(ny_name) {
                ny.len()
            } else if let Some(channels) = spec
                .inputs
                .first()
                .and_then(|input_name| self.tensor_shapes.get(input_name))
                .and_then(|shape| shape.get(1))
                .and_then(|&channels| usize::try_from(channels).ok())
                .filter(|&channels| channels > 0)
            {
                warn!(
                    "InstanceNorm ny {} not found, using activation channel count {}",
                    ny_name, channels
                );
                channels
            } else if let Some(channels) = spec
                .inputs
                .first()
                .and_then(|input_name| self.evaluated_constants.get(input_name))
                .and_then(|value| match value.shape() {
                    [1, channels, ..] => Some(*channels),
                    [channels, ..] => Some(*channels),
                    _ => None,
                })
                .filter(|&channels| channels > 0)
            {
                warn!(
                    "InstanceNorm ny {} not found, using evaluated activation channel count {}",
                    ny_name, channels
                );
                channels
            } else {
                warn!(
                    "InstanceNorm ny {} not found, using default size 1",
                    ny_name
                );
                1
            }
        } else {
            warn!("InstanceNorm has no scale input, using default size 1");
            1
        };

        let ny = if spec.inputs.len() >= 2 {
            load_layernorm_1d_weight(self.weights, &spec.name, &spec.inputs[1], "scale", || {
                ndarray::Array1::ones(num_channels)
            })?
        } else {
            ndarray::Array1::ones(num_channels)
        };

        let beta = if spec.inputs.len() >= 3 {
            load_layernorm_1d_weight(self.weights, &spec.name, &spec.inputs[2], "B", || {
                ndarray::Array1::zeros(num_channels)
            })?
        } else {
            ndarray::Array1::zeros(num_channels)
        };

        let eps = match spec.attributes.get("epsilon") {
            Some(AttributeValue::Float(e)) => *e,
            _ => 1e-5,
        };

        debug!(
            "InstanceNorm {}: channels={}, eps={}",
            spec.name, num_channels, eps
        );

        Ok(Layer::InstanceNorm1d(InstanceNorm1dLayer::new(
            ny, beta, eps,
        )?))
    }

    pub(crate) fn convert_group_norm(&self, spec: &LayerSpec) -> Result<Layer> {
        // ONNX GroupNormalization inputs: X, scale (ny), bias (beta)
        // Attributes: epsilon (float, default 1e-5), num_groups (int, required)
        // Part of #3205.
        let num_channels = if spec.inputs.len() >= 2 {
            let ny_name = &spec.inputs[1];
            if let Some(ny) = self.weights.get(ny_name) {
                ny.len()
            } else {
                warn!("GroupNorm ny {} not found, using default size 1", ny_name);
                1
            }
        } else {
            warn!("GroupNorm has no scale input, using default size 1");
            1
        };

        let ny = if spec.inputs.len() >= 2 {
            load_layernorm_1d_weight(self.weights, &spec.name, &spec.inputs[1], "scale", || {
                ndarray::Array1::ones(num_channels)
            })?
        } else {
            ndarray::Array1::ones(num_channels)
        };

        let beta = if spec.inputs.len() >= 3 {
            load_layernorm_1d_weight(self.weights, &spec.name, &spec.inputs[2], "bias", || {
                ndarray::Array1::zeros(num_channels)
            })?
        } else {
            ndarray::Array1::zeros(num_channels)
        };

        let eps = match spec.attributes.get("epsilon") {
            Some(AttributeValue::Float(e)) => *e,
            _ => 1e-5,
        };

        let num_groups = match spec.attributes.get("num_groups") {
            Some(AttributeValue::Int(n)) => *n as usize,
            _ => {
                return Err(NyError::InvalidSpec(format!(
                    "GroupNorm {}: num_groups attribute required",
                    spec.name
                )));
            }
        };

        debug!(
            "GroupNorm {}: channels={}, groups={}, eps={}",
            spec.name, num_channels, num_groups, eps
        );

        Ok(Layer::GroupNorm(GroupNormLayer::new(
            ny, beta, num_groups, eps,
        )?))
    }

    pub(crate) fn convert_adain(&self, spec: &LayerSpec) -> Result<Layer> {
        // AdaIN custom op inputs: X, instance_norm_scale, instance_norm_bias,
        //                          style_gamma, style_beta
        // Attributes: epsilon (float, default 1e-5)
        //
        // AdaIN(x) = style_gamma * InstanceNorm(x; scale, bias, eps) + style_beta
        //
        // Minimum inputs: X, style_gamma, style_beta (3 inputs, default IN ny=1, beta=0)
        // Full inputs: X, in_scale, in_bias, style_gamma, style_beta (5 inputs)
        let eps = match spec.attributes.get("epsilon") {
            Some(AttributeValue::Float(e)) => *e,
            _ => 1e-5,
        };

        let (in_gamma, in_beta, style_gamma, style_beta) = if spec.inputs.len() >= 5 {
            // Full form: X, in_scale, in_bias, style_gamma, style_beta
            let in_gamma = load_layernorm_1d_weight(
                self.weights,
                &spec.name,
                &spec.inputs[1],
                "in_scale",
                || ndarray::Array1::ones(1),
            )?;
            let in_beta = load_layernorm_1d_weight(
                self.weights,
                &spec.name,
                &spec.inputs[2],
                "in_bias",
                || ndarray::Array1::zeros(in_gamma.len()),
            )?;
            let style_gamma = load_layernorm_1d_weight(
                self.weights,
                &spec.name,
                &spec.inputs[3],
                "style_gamma",
                || ndarray::Array1::ones(in_gamma.len()),
            )?;
            let style_beta = load_layernorm_1d_weight(
                self.weights,
                &spec.name,
                &spec.inputs[4],
                "style_beta",
                || ndarray::Array1::zeros(in_gamma.len()),
            )?;
            (in_gamma, in_beta, style_gamma, style_beta)
        } else if spec.inputs.len() >= 3 {
            // Compact form: X, style_gamma, style_beta (default IN ny=1, beta=0)
            let style_gamma = load_layernorm_1d_weight(
                self.weights,
                &spec.name,
                &spec.inputs[1],
                "style_gamma",
                || ndarray::Array1::ones(1),
            )?;
            let style_beta = load_layernorm_1d_weight(
                self.weights,
                &spec.name,
                &spec.inputs[2],
                "style_beta",
                || ndarray::Array1::zeros(style_gamma.len()),
            )?;
            let num_channels = style_gamma.len();
            (
                ndarray::Array1::ones(num_channels),
                ndarray::Array1::zeros(num_channels),
                style_gamma,
                style_beta,
            )
        } else {
            return Err(NyError::ModelLoad(format!(
                "AdaIN {} requires at least 3 inputs (X, style_gamma, style_beta), got {}",
                spec.name,
                spec.inputs.len()
            )));
        };

        let instance_norm = InstanceNorm1dLayer::new(in_gamma, in_beta, eps)?;

        debug!(
            "AdaIN {}: channels={}, eps={}",
            spec.name,
            instance_norm.num_channels(),
            eps
        );

        Ok(Layer::AdaIN1d(AdaIN1dLayer::new(
            instance_norm,
            style_gamma,
            style_beta,
        )?))
    }

    pub(crate) fn convert_batch_norm(&self, spec: &LayerSpec) -> Result<Layer> {
        // ONNX BatchNormalization inputs: X, scale(ny), B(beta), input_mean, input_var
        // For inference mode (the only mode we support), mean and var are fixed (running statistics)
        if spec.inputs.len() < 5 {
            return Err(NyError::ModelLoad(format!(
                "BatchNormalization {} requires 5 inputs (X, scale, B, mean, var), got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let ny_name = &spec.inputs[1];
        let beta_name = &spec.inputs[2];
        let mean_name = &spec.inputs[3];
        let var_name = &spec.inputs[4];

        let ny = self
            .weights
            .get(ny_name)
            .ok_or_else(|| NyError::ModelLoad(format!("BatchNorm scale {} not found", ny_name)))?
            .clone();

        let beta = self
            .weights
            .get(beta_name)
            .ok_or_else(|| NyError::ModelLoad(format!("BatchNorm bias {} not found", beta_name)))?
            .clone();

        let mean = self
            .weights
            .get(mean_name)
            .ok_or_else(|| NyError::ModelLoad(format!("BatchNorm mean {} not found", mean_name)))?
            .clone();

        let var = self
            .weights
            .get(var_name)
            .ok_or_else(|| NyError::ModelLoad(format!("BatchNorm var {} not found", var_name)))?
            .clone();

        let epsilon = match spec.attributes.get("epsilon") {
            Some(AttributeValue::Float(e)) => *e,
            _ => 1e-5,
        };

        Ok(Layer::BatchNorm(BatchNormLayer::new(
            &ny, &beta, &mean, &var, epsilon,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};
    use ny_core::LayerType;
    use ny_propagate::layers::LayerNormMode;
    use std::collections::{HashMap, HashSet};

    use super::ConvertContext;
    use crate::{AttributeValue, LayerSpec, WeightStore};

    fn make_context(weights: &WeightStore) -> ConvertContext<'_> {
        static SHAPES: std::sync::LazyLock<HashMap<String, Vec<i64>>> =
            std::sync::LazyLock::new(HashMap::new);
        static CONSTANTS: std::sync::LazyLock<HashSet<String>> =
            std::sync::LazyLock::new(HashSet::new);
        ConvertContext::new(weights, &SHAPES, &CONSTANTS)
    }

    fn layernorm_spec(inputs: Vec<&str>) -> LayerSpec {
        LayerSpec {
            name: "ln".to_string(),
            layer_type: LayerType::LayerNorm,
            inputs: inputs.into_iter().map(|s| s.to_string()).collect(),
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        }
    }

    /// Wrong-shaped ny (2-D instead of 1-D) must return an error,
    /// not silently fall back to identity ny.
    #[test]
    fn convert_layer_norm_wrong_ny_shape_returns_error() {
        let mut ws = WeightStore::new();
        // ny stored as [1, 4] instead of [4] — a plausible ONNX shape variant
        ws.insert(
            "ny".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![2.0, 3.0, 4.0, 5.0]).unwrap(),
        );
        let ctx = make_context(&ws);
        let spec = layernorm_spec(vec!["input", "ny"]);
        let result = ctx.convert_layer_norm(&spec);
        assert!(result.is_err(), "wrong-shaped ny must return error");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("expected 1-D"),
            "error should mention expected dimensionality: {}",
            err
        );
    }

    /// Wrong-shaped beta (2-D instead of 1-D) must return an error.
    #[test]
    fn convert_layer_norm_wrong_beta_shape_returns_error() {
        let mut ws = WeightStore::new();
        // correct 1-D ny
        ws.insert(
            "ny".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0; 4]).unwrap(),
        );
        // beta stored as [1, 4] instead of [4]
        ws.insert(
            "beta".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
        );
        let ctx = make_context(&ws);
        let spec = layernorm_spec(vec!["input", "ny", "beta"]);
        let result = ctx.convert_layer_norm(&spec);
        assert!(result.is_err(), "wrong-shaped beta must return error");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("expected 1-D"),
            "error should mention expected dimensionality: {}",
            err
        );
    }

    /// Missing ny (weight not in store) should use default ones — not an error.
    #[test]
    fn convert_layer_norm_missing_ny_uses_default() {
        let ws = WeightStore::new();
        let ctx = make_context(&ws);
        let mut spec = layernorm_spec(vec!["input", "ny"]);
        spec.attributes.insert(
            "normalized_shape".to_string(),
            AttributeValue::Ints(vec![3]),
        );
        let result = ctx.convert_layer_norm(&spec);
        assert!(
            result.is_ok(),
            "missing ny should use default ones: {:?}",
            result.err()
        );
    }

    /// Correct 1-D ny and beta should succeed.
    #[test]
    fn convert_layer_norm_correct_shapes_succeeds() {
        let mut ws = WeightStore::new();
        ws.insert(
            "ny".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        );
        ws.insert(
            "beta".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0; 4]).unwrap(),
        );
        let ctx = make_context(&ws);
        let spec = layernorm_spec(vec!["input", "ny", "beta"]);
        let result = ctx.convert_layer_norm(&spec);
        assert!(
            result.is_ok(),
            "correct shapes should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn layernorm_mode_from_attrs_parses_aliases_and_defaults_4176() {
        let mut mean_only = layernorm_spec(vec!["input"]);
        mean_only.attributes.insert(
            "layernorm_mode".to_string(),
            AttributeValue::String("mean_only".to_string()),
        );
        assert_eq!(
            crate::layernorm_mode_from_attrs(&mean_only),
            LayerNormMode::MeanOnly
        );

        let mut deept = layernorm_spec(vec!["input"]);
        deept.attributes.insert(
            "mode".to_string(),
            AttributeValue::String("deept".to_string()),
        );
        assert_eq!(
            crate::layernorm_mode_from_attrs(&deept),
            LayerNormMode::MeanOnly
        );

        let mut standard = layernorm_spec(vec!["input"]);
        standard.attributes.insert(
            "layernorm_mode".to_string(),
            AttributeValue::String("standard".to_string()),
        );
        assert_eq!(
            crate::layernorm_mode_from_attrs(&standard),
            LayerNormMode::Standard
        );

        let mut numeric = layernorm_spec(vec!["input"]);
        numeric
            .attributes
            .insert("layernorm_mode".to_string(), AttributeValue::Int(1));
        assert_eq!(
            crate::layernorm_mode_from_attrs(&numeric),
            LayerNormMode::MeanOnly
        );

        let mut unknown = layernorm_spec(vec!["input"]);
        unknown.attributes.insert(
            "layernorm_mode".to_string(),
            AttributeValue::String("mystery".to_string()),
        );
        assert_eq!(
            crate::layernorm_mode_from_attrs(&unknown),
            LayerNormMode::Standard
        );

        let missing = layernorm_spec(vec!["input"]);
        assert_eq!(
            crate::layernorm_mode_from_attrs(&missing),
            LayerNormMode::Standard
        );
    }

    /// RMSNorm conversion should succeed with valid ny weight.
    #[test]
    fn convert_rmsnorm_succeeds_with_valid_gamma() {
        let mut ws = WeightStore::new();
        ws.insert(
            "ny".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0; 4]).unwrap(),
        );
        let ctx = make_context(&ws);
        let spec = LayerSpec {
            name: "rms_norm_0".to_string(),
            layer_type: LayerType::RMSNorm,
            inputs: vec!["input".to_string(), "ny".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };
        let result = ctx.convert_layer(&spec);
        assert!(
            result.is_ok(),
            "RMSNorm with valid ny should succeed: {:?}",
            result.err()
        );
        // Verify it creates an RmsNorm layer, not a LayerNorm
        let layer = result.unwrap();
        assert_eq!(layer.layer_type(), "RmsNorm");
    }

    /// RMSNorm conversion with missing ny uses defaults.
    #[test]
    fn convert_rmsnorm_missing_ny_uses_default() {
        let ws = WeightStore::new();
        let ctx = make_context(&ws);
        let mut spec = LayerSpec {
            name: "rms_norm_0".to_string(),
            layer_type: LayerType::RMSNorm,
            inputs: vec!["input".to_string(), "ny".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        };
        spec.attributes.insert(
            "normalized_shape".to_string(),
            AttributeValue::Ints(vec![3]),
        );
        let result = ctx.convert_rms_norm(&spec);
        assert!(
            result.is_ok(),
            "RMSNorm with missing ny should use defaults: {:?}",
            result.err()
        );
    }
}
