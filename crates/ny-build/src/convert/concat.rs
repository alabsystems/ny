// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_propagate::layers::ConcatLayer;
use ny_propagate::Layer;
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use tracing::{debug, warn};

use super::{AttributeValue, ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    pub(crate) fn convert_concat(&self, spec: &LayerSpec) -> Result<Layer> {
        // Concat in ONNX: concatenate tensors along axis
        // For shape-computing Concats (used to build Reshape target shapes), we skip
        // For data Concats (e.g., CLS token + patches in ViT), we create ConcatLayer

        if spec.inputs.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "Concat {} has fewer than 2 inputs",
                spec.name
            )));
        }

        // Check if all inputs are constants (shape-computing concat)
        let all_constants = spec
            .inputs
            .iter()
            .all(|inp| self.weights.get(inp).is_some());

        if all_constants {
            // This is a shape-computing concat that wasn't constant-folded
            debug!(
                "Concat {} is shape-computing (all constant inputs) - skipping",
                spec.name
            );
            return Err(NyError::UnsupportedOp(format!(
                "Concat {} is shape-computing (all constant inputs) - skipped",
                spec.name
            )));
        }

        // Get axis attribute (default 0 for ONNX)
        let onnx_axis = spec
            .attributes
            .get("axis")
            .and_then(|v| match v {
                AttributeValue::Int(i) => Some(*i),
                _ => None,
            })
            .unwrap_or(0);

        // Trailing-relative remap (see `ConvertContext::remap_axis_trailing`):
        // with a KNOWN recorded rank the axis becomes negative (correct
        // whether the runtime inputs kept their ONNX rank or were
        // batch-stripped — the mscn deep-set Concat over retained-rank
        // `[1, d]` inputs miscompiled to a leading-axis stack under the
        // legacy `axis - 1` guess). Unknown rank keeps the legacy adjustment
        // (ny-synthesized subgraphs, e.g. the LSTM unroller, were authored
        // against it). ONNX Concat requires equal-rank inputs, so the first
        // input's recorded rank speaks for all of them.
        let axis = self.remap_axis_trailing(
            "Concat",
            &spec.name,
            &spec.inputs[0],
            onnx_axis,
            super::LegacyBatchAxisPolicy::KeepZeroWarn,
        )?;

        debug!(
            "Concat {} with ONNX axis {} -> adjusted axis {}",
            spec.name, onnx_axis, axis
        );

        // Collect input shapes from weights for constant tensor inputs.
        // For non-constant inputs, we'll get shapes from IBP bounds at runtime.
        // Only pass input_shapes if ALL inputs have known shapes.
        let input_shapes: Vec<Vec<usize>> = spec
            .inputs
            .iter()
            .map(|inp| {
                self.weights
                    .get(inp)
                    .map(|tensor| tensor.shape().to_vec())
                    .unwrap_or_default() // Empty vec if not a constant
            })
            .collect();

        // Only use input_shapes if all shapes are known (non-empty)
        let all_shapes_known = input_shapes.iter().all(|s| !s.is_empty());
        let input_shapes = if all_shapes_known {
            input_shapes
        } else {
            Vec::new()
        };

        // Create BoundedTensors for constant inputs (lower == upper since they're constant)
        let constant_inputs: Vec<Option<BoundedTensor>> = spec
            .inputs
            .iter()
            .map(|inp| {
                self.weights.get(inp).and_then(|tensor| {
                    // Create a BoundedTensor with lower == upper (zero width for constants)
                    BoundedTensor::new(tensor.clone(), tensor.clone())
                        .map_err(|e| {
                            warn!(
                                "Concat {}: failed to create BoundedTensor for input '{}': {}",
                                spec.name, inp, e
                            );
                            e
                        })
                        .ok()
                })
            })
            .collect();

        // Check if any input has constant data
        let has_constants = constant_inputs.iter().any(|c| c.is_some());

        debug!(
            "Concat {} is data concat along axis {} with {} inputs, has_constants={}",
            spec.name,
            axis,
            spec.inputs.len(),
            has_constants
        );

        // Only LEGACY-shifted positive axes (stored as `onnx_axis - 1`) move
        // when the runtime reintroduces a broadcast batch dim. Verbatim axes
        // (unbatched model) and trailing-relative NEGATIVE axes resolve
        // against the actual rank and must NOT shift.
        let restored_batch_axis_shift = !self.model_unbatched && onnx_axis > 0 && axis >= 0;

        if has_constants {
            Ok(Layer::Concat(
                ConcatLayer::with_constants(axis, input_shapes, constant_inputs)
                    .with_restored_batch_axis_shift(restored_batch_axis_shift),
            ))
        } else if !input_shapes.is_empty() {
            // Pass input_shapes for proper CROWN backward propagation when all shapes are known
            Ok(Layer::Concat(
                ConcatLayer::with_input_shapes(axis, input_shapes)
                    .with_restored_batch_axis_shift(restored_batch_axis_shift),
            ))
        } else {
            // Shapes will be determined from IBP bounds at runtime
            Ok(Layer::Concat(
                ConcatLayer::new(axis).with_restored_batch_axis_shift(restored_batch_axis_shift),
            ))
        }
    }
}

impl ConvertContext<'_> {
    /// Convert Concat with access to pre-evaluated constant chains.
    ///
    /// This is used in graph network construction when constant chains
    /// (like ConstantOfShape + Add) have been evaluated and their results
    /// are available.
    pub fn convert_concat_with_evaluated(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Result<Layer> {
        if spec.inputs.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "Concat {} has fewer than 2 inputs",
                spec.name
            )));
        }

        // Check if all inputs are constants (including evaluated ones)
        let all_constants = spec
            .inputs
            .iter()
            .all(|inp| self.weights.get(inp).is_some() || evaluated_constants.contains_key(inp));

        if all_constants {
            debug!(
                "Concat {} is shape-computing (all constant inputs) - skipping",
                spec.name
            );
            return Err(NyError::UnsupportedOp(format!(
                "Concat {} is shape-computing (all constant inputs) - skipped",
                spec.name
            )));
        }

        let onnx_axis = spec
            .attributes
            .get("axis")
            .and_then(|v| match v {
                AttributeValue::Int(i) => Some(*i),
                _ => None,
            })
            .unwrap_or(0);

        // Trailing-relative remap with legacy fallback for unknown recorded
        // ranks — see `convert_concat` above and
        // `ConvertContext::remap_axis_trailing`.
        let axis = self.remap_axis_trailing(
            "Concat",
            &spec.name,
            &spec.inputs[0],
            onnx_axis,
            super::LegacyBatchAxisPolicy::KeepZeroWarn,
        )?;

        debug!(
            "Concat {} (with evaluated) ONNX axis {} -> adjusted axis {}",
            spec.name, onnx_axis, axis
        );

        // Collect input shapes from both weights and evaluated constants
        let input_shapes: Vec<Vec<usize>> = spec
            .inputs
            .iter()
            .map(|inp| {
                self.weights
                    .get(inp)
                    .map(|tensor| tensor.shape().to_vec())
                    .or_else(|| evaluated_constants.get(inp).map(|t| t.shape().to_vec()))
                    .unwrap_or_default()
            })
            .collect();

        // Only use input_shapes if all shapes are known (non-empty)
        let all_shapes_known = input_shapes.iter().all(|s| !s.is_empty());
        let input_shapes = if all_shapes_known {
            input_shapes
        } else {
            Vec::new()
        };

        // Create BoundedTensors from both weights and evaluated constants
        let constant_inputs: Vec<Option<BoundedTensor>> = spec
            .inputs
            .iter()
            .map(|inp| {
                // First try weights
                self.weights
                    .get(inp)
                    .and_then(|tensor| {
                        BoundedTensor::new(tensor.clone(), tensor.clone())
                            .map_err(|e| {
                                warn!(
                                    "Concat {}: failed to create BoundedTensor for input '{}': {}",
                                    spec.name, inp, e
                                );
                                e
                            })
                            .ok()
                    })
                    // Then try evaluated constants
                    .or_else(|| {
                        evaluated_constants.get(inp).and_then(|tensor| {
                            BoundedTensor::new(tensor.clone(), tensor.clone())
                                .map_err(|e| {
                                    warn!(
                                        "Concat {}: failed to create BoundedTensor from evaluated constant '{}': {}",
                                        spec.name, inp, e
                                    );
                                    e
                                })
                                .ok()
                        })
                    })
            })
            .collect();

        let has_constants = constant_inputs.iter().any(|c| c.is_some());

        // Only LEGACY-shifted positive axes (stored as `onnx_axis - 1`) move
        // when the runtime reintroduces a broadcast batch dim. Verbatim axes
        // (unbatched model) and trailing-relative NEGATIVE axes resolve
        // against the actual rank and must NOT shift.
        let restored_batch_axis_shift = !self.model_unbatched && onnx_axis > 0 && axis >= 0;

        if has_constants {
            Ok(Layer::Concat(
                ConcatLayer::with_constants(axis, input_shapes, constant_inputs)
                    .with_restored_batch_axis_shift(restored_batch_axis_shift),
            ))
        } else if !input_shapes.is_empty() {
            // Pass input_shapes for proper CROWN backward propagation when all shapes are known
            Ok(Layer::Concat(
                ConcatLayer::with_input_shapes(axis, input_shapes)
                    .with_restored_batch_axis_shift(restored_batch_axis_shift),
            ))
        } else {
            // Shapes will be determined from IBP bounds at runtime
            Ok(Layer::Concat(
                ConcatLayer::new(axis).with_restored_batch_axis_shift(restored_batch_axis_shift),
            ))
        }
    }
}
