// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_propagate::layers::{
    CumsumLayer, LogSumExpLayer, ReduceMaxLayer, ReduceMeanLayer, ReduceMinLayer, ReduceSumLayer,
    SkipMergeLayer,
};
use ny_propagate::Layer;
use tracing::debug;

use super::{AttributeValue, ConvertContext, LayerSpec};

enum ReductionAxes {
    Reduce(Vec<i64>),
    Identity,
}

impl ConvertContext<'_> {
    /// Read reduction axes from attributes (opset < 13/18) or second input tensor (opset 13+/18+).
    ///
    /// ONNX ReduceSum moved `axes` from attribute to input[1] in opset 13.
    /// ReduceMean, ReduceMax, ReduceMin moved `axes` to input[1] in opset 18.
    /// This helper checks attributes first (backward compat), then falls back
    /// to reading the constant second input.
    fn read_reduction_axes(&self, spec: &LayerSpec) -> Result<ReductionAxes> {
        let noop_with_empty_axes = match spec.attributes.get("noop_with_empty_axes") {
            None | Some(AttributeValue::Int(0)) => false,
            Some(AttributeValue::Int(1)) => true,
            Some(other) => {
                return Err(NyError::ModelLoad(format!(
                    "{} '{}' has invalid noop_with_empty_axes attribute {:?}",
                    spec.layer_type, spec.name, other
                )))
            }
        };

        let attribute_axes = match spec.attributes.get("axes") {
            None => None,
            Some(AttributeValue::Ints(axes)) => Some(axes),
            Some(other) => {
                return Err(NyError::ModelLoad(format!(
                    "{} '{}' has invalid axes attribute {:?}",
                    spec.layer_type, spec.name, other
                )))
            }
        };
        let input_axes_name = spec.inputs.get(1).filter(|name| !name.is_empty());

        // LayerSpec no longer carries the originating opset. Accepting both
        // schema encodings would require guessing which one ONNX Runtime used.
        if attribute_axes.is_some() && input_axes_name.is_some() {
            return Err(NyError::UnsupportedConfiguration(format!(
                "{} '{}' supplies axes as both an attribute and an input",
                spec.layer_type, spec.name
            )));
        }

        // 1. Attribute form (older opsets).
        if let Some(axes) = attribute_axes {
            return if axes.is_empty() && noop_with_empty_axes {
                Ok(ReductionAxes::Identity)
            } else {
                Ok(ReductionAxes::Reduce(axes.clone()))
            };
        }

        // 2. Try second input tensor (opset 13+/18+): axes as constant tensor
        if let Some(axes_name) = input_axes_name {
            let axes_tensor = self
                .discrete_constant_i64(
                    axes_name,
                    &format!("{} '{}' reduction axes", spec.layer_type, spec.name),
                )?
                .ok_or_else(|| {
                    NyError::UnsupportedConfiguration(format!(
                        "{} '{}' requires axes input '{}' to be constant",
                        spec.layer_type, spec.name, axes_name
                    ))
                })?;
            let axes: Vec<i64> = axes_tensor.iter().copied().collect();
            if !axes.is_empty() {
                debug!(
                    "{} {} reading axes from input tensor '{}': {:?}",
                    spec.layer_type, spec.name, axes_name, axes
                );
            }
            return if axes.is_empty() && noop_with_empty_axes {
                Ok(ReductionAxes::Identity)
            } else {
                Ok(ReductionAxes::Reduce(axes))
            };
        }

        // 3. Missing axes reduces all unless noop_with_empty_axes requests the
        // schema-defined identity behavior.
        if noop_with_empty_axes {
            Ok(ReductionAxes::Identity)
        } else {
            Ok(ReductionAxes::Reduce(Vec::new()))
        }
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
        let axis_tensor = self
            .discrete_constant_i64(axis_name, &format!("CumSum {} axis", spec.name))?
            .ok_or_else(|| {
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

        axis_tensor.iter().next().copied().ok_or_else(|| {
            NyError::UnsupportedConfiguration(format!(
                "CumSum {} axis tensor '{}' is empty",
                spec.name, axis_name
            ))
        })
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

        let ReductionAxes::Reduce(onnx_axes) = self.read_reduction_axes(spec)? else {
            return Ok(Layer::SkipMerge(SkipMergeLayer::new()));
        };

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

        let ReductionAxes::Reduce(onnx_axes) = self.read_reduction_axes(spec)? else {
            return Ok(Layer::SkipMerge(SkipMergeLayer::new()));
        };

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
        let ReductionAxes::Reduce(onnx_axes) = self.read_reduction_axes(spec)? else {
            return Ok(Layer::SkipMerge(SkipMergeLayer::new()));
        };

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
        let ReductionAxes::Reduce(onnx_axes) = self.read_reduction_axes(spec)? else {
            return Ok(Layer::SkipMerge(SkipMergeLayer::new()));
        };

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

        let ReductionAxes::Reduce(onnx_axes) = self.read_reduction_axes(spec)? else {
            return Ok(Layer::SkipMerge(SkipMergeLayer::new()));
        };

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
