// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_propagate::layers::{
    CumsumLayer, LogSumExpLayer, ReduceMaxLayer, ReduceMeanLayer, ReduceMinLayer, ReduceSumLayer,
};
use ny_propagate::Layer;
use tracing::debug;

use super::{AttributeValue, ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    /// Read reduction axes from attributes (opset < 13/18) or second input tensor (opset 13+/18+).
    ///
    /// ONNX ReduceSum moved `axes` from attribute to input[1] in opset 13.
    /// ReduceMean, ReduceMax, ReduceMin moved `axes` to input[1] in opset 18.
    /// This helper checks attributes first (backward compat), then falls back
    /// to reading the constant second input.
    fn read_reduction_axes(&self, spec: &LayerSpec) -> Result<Vec<i64>> {
        // 1. Try attributes (opset < 13 for ReduceSum, opset < 18 for others)
        if let Some(AttributeValue::Ints(arr)) = spec.attributes.get("axes") {
            if !arr.is_empty() {
                return Ok(arr.clone());
            }
        }

        // 2. Try second input tensor (opset 13+/18+): axes as constant tensor
        if let Some(axes_name) = spec.inputs.get(1).filter(|n| !n.is_empty()) {
            if let Some(axes_tensor) = self.constant_value(axes_name) {
                // #2360: Validate axis values before f32→i64 cast. NaN as i64 = 0 (wrong
                // axis), Inf as i64 = i64::MAX (out-of-range axis), non-integer values
                // silently round. Reject all three cases.
                let axes: Vec<i64> = axes_tensor
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(idx, v)| {
                        if !v.is_finite() {
                            return Err(NyError::ModelLoad(format!(
                                "{} '{}': reduction axis at index {} is non-finite ({})",
                                spec.layer_type, spec.name, idx, v
                            )));
                        }
                        if v.trunc() != v {
                            return Err(NyError::ModelLoad(format!(
                                "{} '{}': reduction axis at index {} is non-integer ({})",
                                spec.layer_type, spec.name, idx, v
                            )));
                        }
                        Ok(v as i64)
                    })
                    .collect::<Result<_>>()?;
                if !axes.is_empty() {
                    debug!(
                        "{} {} reading axes from input tensor '{}': {:?}",
                        spec.layer_type, spec.name, axes_name, axes
                    );
                    return Ok(axes);
                }
            }
        }

        // 3. Empty = reduce over all axes (ONNX default)
        Ok(Vec::new())
    }

    fn read_cumsum_axis(&self, spec: &LayerSpec) -> Result<i64> {
        let axis_name = spec
            .inputs
            .get(1)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                NyError::ModelLoad(format!(
                    "CumSum {} requires a constant axis input at input[1]",
                    spec.name
                ))
            })?;
        let axis_tensor = self.constant_value(axis_name).ok_or_else(|| {
            NyError::UnsupportedConfiguration(format!(
                "CumSum {} requires input[1] axis '{}' to be a constant tensor",
                spec.name, axis_name
            ))
        })?;

        if axis_tensor.is_empty() {
            return Err(NyError::UnsupportedConfiguration(format!(
                "CumSum {} axis tensor '{}' is empty",
                spec.name, axis_name
            )));
        }
        if axis_tensor.len() != 1 {
            return Err(NyError::UnsupportedConfiguration(format!(
                "CumSum {} axis tensor '{}' must be scalar, got {} elements",
                spec.name,
                axis_name,
                axis_tensor.len()
            )));
        }

        let axis_value = axis_tensor.iter().next().copied().ok_or_else(|| {
            NyError::UnsupportedConfiguration(format!(
                "CumSum {} axis tensor '{}' is empty",
                spec.name, axis_name
            ))
        })?;
        if !axis_value.is_finite() {
            return Err(NyError::UnsupportedConfiguration(format!(
                "CumSum {} axis tensor '{}' must be finite, got {}",
                spec.name, axis_name, axis_value
            )));
        }

        Ok(axis_value.round() as i64)
    }

    /// Remap all reduction axes of `spec` to the trailing-relative internal
    /// encoding (see [`ConvertContext::remap_axis_trailing`]).
    ///
    /// The legacy blanket `axis >= 1 → axis - 1` guess miscompiled reductions
    /// whose runtime input retained its leading size-1 axis (e.g. downstream
    /// of Flatten / rank-2 Gemm outputs): on pensieve, `ReduceSum(axes=[1])`
    /// on a runtime `[1, n]` tensor became a size-1-axis NO-OP and the graph
    /// bounded the wrong function (`w = p/p = 1`). Trailing-relative
    /// (negative) axes select the same semantic dim under both runtime
    /// layouts; the reduction layers resolve them against the actual runtime
    /// rank at propagation time. Ambiguous cases refuse conversion.
    fn remap_reduction_axes(
        &self,
        spec: &LayerSpec,
        op: &str,
        onnx_axes: &[i64],
    ) -> Result<Vec<i64>> {
        let data_name =
            spec.inputs.first().map(String::as_str).ok_or_else(|| {
                NyError::ModelLoad(format!("{op} '{}' has no data input", spec.name))
            })?;
        onnx_axes
            .iter()
            .map(|&axis| {
                self.remap_axis_trailing(
                    op,
                    &spec.name,
                    data_name,
                    axis,
                    super::LegacyBatchAxisPolicy::KeepZeroWarn,
                )
            })
            .collect()
    }

    pub(crate) fn convert_reduce_mean(&self, spec: &LayerSpec) -> Result<Layer> {
        // ReduceMean in ONNX: compute mean over specified axes
        // Attributes: axes (list of ints), keepdims (int, default 1)
        // Opset 18+ moved axes to second input tensor.

        let onnx_axes = self.read_reduction_axes(spec)?;

        // Get keepdims from attributes (default is true/1)
        let keepdims = match spec.attributes.get("keepdims") {
            Some(AttributeValue::Int(v)) => *v != 0,
            _ => true, // Default is to keep dims
        };

        let adjusted_axes = self.remap_reduction_axes(spec, "ReduceMean", &onnx_axes)?;

        debug!(
            "ReduceMean {} with ONNX axes {:?} -> adjusted axes {:?}, keepdims={}",
            spec.name, onnx_axes, adjusted_axes, keepdims
        );

        Ok(Layer::ReduceMean(ReduceMeanLayer::new(
            adjusted_axes,
            keepdims,
        )))
    }
    pub(crate) fn convert_reduce_sum(&self, spec: &LayerSpec) -> Result<Layer> {
        // ReduceSum in ONNX: compute sum over specified axes
        // Attributes: axes (list of ints), keepdims (int, default 1)
        // Opset 13+ moved axes to second input tensor.

        let onnx_axes = self.read_reduction_axes(spec)?;

        // Get keepdims from attributes (default is true/1)
        let keepdims = match spec.attributes.get("keepdims") {
            Some(AttributeValue::Int(v)) => *v != 0,
            _ => true, // Default is to keep dims
        };

        let adjusted_axes = self.remap_reduction_axes(spec, "ReduceSum", &onnx_axes)?;

        debug!(
            "ReduceSum {} with ONNX axes {:?} -> adjusted axes {:?}, keepdims={}",
            spec.name, onnx_axes, adjusted_axes, keepdims
        );

        Ok(Layer::ReduceSum(ReduceSumLayer::new(
            adjusted_axes,
            keepdims,
        )))
    }
    pub(crate) fn convert_reduce_max(&self, spec: &LayerSpec) -> Result<Layer> {
        // Opset 18+ moved axes to second input tensor.
        let onnx_axes = self.read_reduction_axes(spec)?;

        let keepdims = match spec.attributes.get("keepdims") {
            Some(AttributeValue::Int(v)) => *v != 0,
            _ => true,
        };

        let adjusted_axes = self.remap_reduction_axes(spec, "ReduceMax", &onnx_axes)?;

        debug!(
            "ReduceMax {} with ONNX axes {:?} -> adjusted axes {:?}, keepdims={}",
            spec.name, onnx_axes, adjusted_axes, keepdims
        );

        Ok(Layer::ReduceMax(ReduceMaxLayer::new(
            adjusted_axes,
            keepdims,
        )))
    }

    pub(crate) fn convert_reduce_min(&self, spec: &LayerSpec) -> Result<Layer> {
        // Opset 18+ moved axes to second input tensor.
        let onnx_axes = self.read_reduction_axes(spec)?;

        let keepdims = match spec.attributes.get("keepdims") {
            Some(AttributeValue::Int(v)) => *v != 0,
            _ => true,
        };

        let adjusted_axes = self.remap_reduction_axes(spec, "ReduceMin", &onnx_axes)?;

        debug!(
            "ReduceMin {} with ONNX axes {:?} -> adjusted axes {:?}, keepdims={}",
            spec.name, onnx_axes, adjusted_axes, keepdims
        );

        Ok(Layer::ReduceMin(ReduceMinLayer::new(
            adjusted_axes,
            keepdims,
        )))
    }

    pub(crate) fn convert_cumsum(&self, spec: &LayerSpec) -> Result<Layer> {
        // CumSum in ONNX: cumulative sum along an axis.
        // Input[0]: data tensor, Input[1]: axis (scalar constant tensor)
        // Attributes: exclusive (int, default 0), reverse (int, default 0)

        let onnx_axis = self.read_cumsum_axis(spec)?;
        let exclusive =
            matches!(spec.attributes.get("exclusive"), Some(AttributeValue::Int(v)) if *v != 0);
        let reverse =
            matches!(spec.attributes.get("reverse"), Some(AttributeValue::Int(v)) if *v != 0);

        let adjusted_axis = self.remap_reduction_axes(spec, "CumSum", &[onnx_axis])?[0];

        debug!(
            "CumSum {} with ONNX axis {} -> adjusted axis {}, exclusive={}, reverse={}",
            spec.name, onnx_axis, adjusted_axis, exclusive, reverse
        );

        Ok(Layer::CumSum(CumsumLayer::new(
            adjusted_axis,
            exclusive,
            reverse,
        )))
    }

    pub(crate) fn convert_logsumexp(&self, spec: &LayerSpec) -> Result<Layer> {
        // LogSumExp in ny: compute log(sum(exp(x))) over specified axes
        // Attributes: axes (list of ints), keepdims (int, default 1)

        let onnx_axes = self.read_reduction_axes(spec)?;

        let keepdims = match spec.attributes.get("keepdims") {
            Some(AttributeValue::Int(v)) => *v != 0,
            _ => true,
        };

        let adjusted_axes = self.remap_reduction_axes(spec, "LogSumExp", &onnx_axes)?;

        debug!(
            "LogSumExp {} with ONNX axes {:?} -> adjusted axes {:?}, keepdims={}",
            spec.name, onnx_axes, adjusted_axes, keepdims
        );

        Ok(Layer::LogSumExp(LogSumExpLayer::new(
            adjusted_axes,
            keepdims,
        )))
    }
}

#[cfg(test)]
#[path = "reductions_tests.rs"]
mod reductions_tests;
