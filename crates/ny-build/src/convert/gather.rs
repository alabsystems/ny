// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_propagate::layers::GatherLayer;
use ny_propagate::Layer;
use tracing::debug;

use super::{AttributeValue, ConvertContext, LayerSpec};

impl ConvertContext<'_> {
    pub(crate) fn convert_gather(&self, spec: &LayerSpec) -> Result<Layer> {
        if spec.inputs.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "Gather {} requires 2 inputs (data, indices), got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        if let Some(layer) = self.try_convert_runtime_last_axis_len_gather(spec)? {
            return Ok(layer);
        }

        let onnx_axis = match spec.attributes.get("axis") {
            Some(AttributeValue::Int(a)) => *a,
            _ => 0, // ONNX default axis is 0
        };
        let data_is_constant = self.is_constant(&spec.inputs[0]);
        let axis = if self.model_unbatched {
            // Unbatched model (#cctsdb B5): ONNX axes describe the internal
            // tensors verbatim — no batch axis was ever stripped anywhere.
            debug!(
                "Gather '{}': unbatched model — ONNX axis {} used verbatim",
                spec.name, onnx_axis
            );
            onnx_axis
        } else if onnx_axis == 0 && data_is_constant {
            // Embedding lookup: constant data (weight table) never had a batch
            // dimension; axis=0 indexes the vocabulary axis. Reference:
            // alpha-beta-CROWN BoundGather (indexing.py:38-39). Part of #3312.
            debug!(
                "Gather '{}': axis=0 with constant data — embedding lookup, no batch adjustment",
                spec.name
            );
            0
        } else {
            // Trailing-relative remap: correct whether the runtime data tensor
            // kept its ONNX rank (leading size-1 retained) or had the batch
            // dim stripped; GatherLayer resolves negative axes against the
            // actual runtime rank. Refuses ambiguous cases (unknown recorded
            // rank; batch axis 0 of a rank>1 activation tensor, per #2848).
            // See `ConvertContext::remap_axis_trailing`.
            self.remap_axis_trailing(
                "Gather",
                &spec.name,
                &spec.inputs[0],
                onnx_axis,
                crate::convert::LegacyBatchAxisPolicy::RejectZero,
            )?
        };

        let indices_name = &spec.inputs[1];
        let mut indices = None;
        let mut indices_shape: Vec<usize> = Vec::new();
        let mut indices_shape_unknown = false;

        if let Some(indices_arr) =
            self.discrete_constant_i64(indices_name, &format!("Gather {} indices", spec.name))?
        {
            indices_shape = indices_arr.shape().to_vec();
            indices = Some(indices_arr);
            debug!(
                "Converting Gather '{}' with constant indices shape {:?}",
                spec.name, indices_shape
            );
        } else if let Some(shape) = self.tensor_shapes.get(indices_name) {
            let (shape, had_unknown) = shape_i64_to_usize(shape);
            indices_shape = shape;
            indices_shape_unknown = had_unknown;
            debug!(
                "Converting Gather '{}' with dynamic indices shape {:?}",
                spec.name, indices_shape
            );
        }

        if indices.is_none() && (indices_shape.is_empty() || indices_shape_unknown) {
            let data_name = &spec.inputs[0];
            let output_name = spec.outputs.first().cloned().unwrap_or_default();
            if let (Some(data_shape), Some(output_shape)) = (
                self.tensor_shapes.get(data_name),
                self.tensor_shapes.get(&output_name),
            ) {
                if let Some(inferred) = infer_indices_shape_from_io(data_shape, output_shape, axis)
                {
                    indices_shape = inferred;
                    debug!(
                        "Converting Gather '{}' inferred indices shape {:?} from IO shapes",
                        spec.name, indices_shape
                    );
                }
            }
        }

        if indices.is_none() && indices_shape.is_empty() {
            debug!(
                "Gather '{}' indices tensor '{}' missing shape; defaulting to scalar indices",
                spec.name, indices_name
            );
        }

        Ok(Layer::Gather(GatherLayer::new(
            axis,
            indices,
            indices_shape,
        )))
    }

    fn try_convert_runtime_last_axis_len_gather(&self, spec: &LayerSpec) -> Result<Option<Layer>> {
        let data_name = &spec.inputs[0];
        let indices_name = &spec.inputs[1];
        let onnx_axis = match spec.attributes.get("axis") {
            Some(AttributeValue::Int(a)) => *a,
            _ => 0,
        };
        if onnx_axis != 0 && onnx_axis != -1 {
            return Ok(None);
        }

        let data_value_available = self.weights.get(data_name).is_some()
            || self.evaluated_constants.contains_key(data_name);
        if data_value_available || !self.constant_tensors.contains(data_name) {
            return Ok(None);
        }

        let indices_arr = match self.discrete_constant_i64(
            indices_name,
            &format!("Gather {} runtime shape-query indices", spec.name),
        )? {
            Some(arr) if arr.len() == 1 => arr,
            _ => return Ok(None),
        };
        let gathered_index = match indices_arr.iter().next().copied() {
            Some(idx) => idx,
            None => return Ok(None),
        };

        let shape_rank = match self.tensor_shapes.get(data_name) {
            Some(shape) if shape.len() == 1 && shape[0] > 0 => shape[0],
            _ => return Ok(None),
        };
        let selects_last_axis = gathered_index == -1 || gathered_index == shape_rank - 1;
        if !selects_last_axis {
            return Ok(None);
        }

        if let Some(output_name) = spec.outputs.first() {
            if let Some(output_shape) = self.tensor_shapes.get(output_name) {
                let scalar_like =
                    output_shape.is_empty() || output_shape.iter().copied().product::<i64>() == 1;
                if !scalar_like {
                    return Ok(None);
                }
            }
        }

        let indices_shape = indices_arr.shape().to_vec();
        debug!(
            "Converting Gather '{}' on shape-derived tensor '{}' to runtime last-axis length query",
            spec.name, data_name
        );
        Ok(Some(Layer::Gather(GatherLayer::runtime_last_axis_len(
            indices_shape,
        ))))
    }
}

fn shape_i64_to_usize(shape: &[i64]) -> (Vec<usize>, bool) {
    let mut had_unknown = false;
    let mut out = Vec::with_capacity(shape.len());
    for &dim in shape {
        if dim > 0 {
            out.push(dim as usize);
        } else {
            out.push(1);
            had_unknown = true;
        }
    }
    (out, had_unknown)
}

fn infer_indices_shape_from_io(
    data_shape: &[i64],
    output_shape: &[i64],
    axis: i64,
) -> Option<Vec<usize>> {
    let data_rank = data_shape.len();
    if data_rank == 0 {
        return None;
    }
    let axis = if axis < 0 {
        let axis = data_rank as i64 + axis;
        if axis < 0 {
            return None;
        }
        axis as usize
    } else {
        axis as usize
    };
    if axis >= data_rank {
        return None;
    }

    let prefix_len = axis;
    let suffix_len = data_rank - axis - 1;
    if output_shape.len() < prefix_len + suffix_len {
        return None;
    }

    // Validate known prefix/suffix dimensions when available.
    for (idx, &dim) in data_shape.iter().enumerate() {
        if idx == axis {
            continue;
        }
        let out_idx = if idx < axis {
            idx
        } else {
            output_shape.len().saturating_sub(suffix_len) + (idx - axis - 1)
        };
        if let (Some(&out_dim), true) = (output_shape.get(out_idx), dim > 0) {
            if out_dim > 0 && out_dim != dim {
                return None;
            }
        }
    }

    let indices_slice_end = output_shape.len().saturating_sub(suffix_len);
    let indices_shape_i64 = &output_shape[prefix_len..indices_slice_end];
    Some(shape_i64_to_usize(indices_shape_i64).0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::ConvertContext;
    use crate::{AttributeValue, LayerSpec, WeightStore};
    use ndarray::arr1;
    use ny_core::LayerType;
    use ny_propagate::layers::BoundPropagation;
    use ny_tensor::BoundedTensor;
    use std::collections::{HashMap, HashSet};

    fn make_context(weights: &WeightStore) -> ConvertContext<'_> {
        static SHAPES: std::sync::LazyLock<HashMap<String, Vec<i64>>> =
            std::sync::LazyLock::new(HashMap::new);
        static CONSTANTS: std::sync::LazyLock<HashSet<String>> =
            std::sync::LazyLock::new(HashSet::new);
        ConvertContext::new(weights, &SHAPES, &CONSTANTS)
    }

    fn make_context_with_shape_constants<'a>(
        weights: &'a WeightStore,
        shapes: &'a HashMap<String, Vec<i64>>,
        constants: &'a HashSet<String>,
    ) -> ConvertContext<'a> {
        ConvertContext::new(weights, shapes, constants)
    }

    fn gather_spec(name: &str, axis: i64) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::Gather,
            inputs: vec!["data".to_string(), "indices".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(axis))]),
        }
    }

    fn gather_spec_no_axis(name: &str) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::Gather,
            inputs: vec!["data".to_string(), "indices".to_string()],
            outputs: vec!["output".to_string()],
            weights: None,
            attributes: HashMap::new(),
        }
    }

    /// Trailing-relative remap: positive ONNX axes become negative
    /// (from-the-end) internal axes, correct under both runtime layouts.
    #[test]
    fn gather_positive_axis_remaps_trailing_relative() {
        let mut ws = WeightStore::new();
        ws.insert("indices".to_string(), arr1(&[0.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 4, 6])]);
        let constants = HashSet::new();
        let m = make_context_with_shape_constants(&ws, &shapes, &constants);
        for (onnx_axis, expected) in [(1_i64, -2_i64), (2, -1)] {
            let spec = gather_spec("g_pos", onnx_axis);
            let layer = m.convert_gather(&spec).unwrap();
            let Layer::Gather(gather) = layer else {
                panic!("expected Gather layer");
            };
            assert_eq!(gather.axis_raw(), expected, "onnx axis {onnx_axis}");
        }
    }

    /// A positive axis with no recorded ONNX shape keeps the legacy
    /// `axis - 1` adjustment (ny-synthesized-subgraph compatibility).
    #[test]
    fn gather_positive_axis_unknown_rank_keeps_legacy() {
        let mut ws = WeightStore::new();
        ws.insert("indices".to_string(), arr1(&[0.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = gather_spec("g_axis1_norank", 1);
        let layer = m.convert_gather(&spec).unwrap();
        let Layer::Gather(gather) = layer else {
            panic!("expected Gather layer");
        };
        assert_eq!(gather.axis_raw(), 0);
    }

    /// Regression test for #2848: Gather with default axis=0 on a rank-2
    /// activation tensor must be rejected (batch axis).
    #[test]
    fn gather_default_axis_0_rejected() {
        let mut ws = WeightStore::new();
        ws.insert("indices".to_string(), arr1(&[0.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 8])]);
        let constants = HashSet::new();
        let m = make_context_with_shape_constants(&ws, &shapes, &constants);
        let spec = gather_spec_no_axis("g_default");
        let err = m.convert_gather(&spec).unwrap_err();
        assert!(
            err.to_string().contains("batch dimension"),
            "Expected batch dimension rejection for default axis=0, got: {err}"
        );
    }

    /// Regression test for #2848: Gather with explicit axis=0 must be rejected.
    #[test]
    fn gather_explicit_axis_0_rejected() {
        let mut ws = WeightStore::new();
        ws.insert("indices".to_string(), arr1(&[0.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 8])]);
        let constants = HashSet::new();
        let m = make_context_with_shape_constants(&ws, &shapes, &constants);
        let spec = gather_spec("g_explicit", 0);
        let err = m.convert_gather(&spec).unwrap_err();
        assert!(
            err.to_string().contains("batch dimension"),
            "Expected batch dimension rejection for axis=0, got: {err}"
        );
    }

    #[test]
    fn gather_axis_1_succeeds() {
        let mut ws = WeightStore::new();
        ws.insert("indices".to_string(), arr1(&[0.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 8])]);
        let constants = HashSet::new();
        let m = make_context_with_shape_constants(&ws, &shapes, &constants);
        let spec = gather_spec("g_axis1", 1);
        let layer = m.convert_gather(&spec).unwrap();
        assert!(matches!(layer, Layer::Gather(_)));
    }

    #[test]
    fn gather_axis_2_succeeds() {
        let mut ws = WeightStore::new();
        ws.insert("indices".to_string(), arr1(&[0.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 4, 8])]);
        let constants = HashSet::new();
        let m = make_context_with_shape_constants(&ws, &shapes, &constants);
        let spec = gather_spec("g_axis2", 2);
        let layer = m.convert_gather(&spec).unwrap();
        assert!(matches!(layer, Layer::Gather(_)));
    }

    #[test]
    fn gather_nan_indices_rejected() {
        let mut ws = WeightStore::new();
        ws.insert("indices".to_string(), arr1(&[f32::NAN]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 8])]);
        let constants = HashSet::new();
        let m = make_context_with_shape_constants(&ws, &shapes, &constants);
        let spec = gather_spec("g_nan", 1);
        let err = m.convert_gather(&spec).unwrap_err();
        assert!(
            err.to_string().contains("NaN"),
            "Expected NaN rejection, got: {err}"
        );
    }

    #[test]
    fn gather_non_integer_indices_rejected() {
        let mut ws = WeightStore::new();
        ws.insert("indices".to_string(), arr1(&[1.5_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 8])]);
        let constants = HashSet::new();
        let m = make_context_with_shape_constants(&ws, &shapes, &constants);
        let spec = gather_spec("g_float", 1);
        let err = m.convert_gather(&spec).unwrap_err();
        assert!(
            err.to_string().contains("integers"),
            "Expected integer requirement, got: {err}"
        );
    }

    #[test]
    fn gather_adjacent_non_integer_indices_rejected() {
        for value in [
            f32::from_bits(1.0_f32.to_bits() - 1),
            f32::from_bits(1.0_f32.to_bits() + 1),
        ] {
            let mut ws = WeightStore::new();
            ws.insert("indices".to_string(), arr1(&[value]).into_dyn());
            let shapes = HashMap::from([("data".to_string(), vec![1_i64, 8])]);
            let constants = HashSet::new();
            let m = make_context_with_shape_constants(&ws, &shapes, &constants);
            assert!(m.convert_gather(&gather_spec("g_adjacent", 1)).is_err());
        }
    }

    #[test]
    fn gather_prefers_exact_integer_indices_over_lossy_float_mirror() {
        let mut ws = WeightStore::new();
        let exact = 16_777_217_i64;
        ws.insert("indices".to_string(), arr1(&[exact as f32]).into_dyn());
        ws.insert_integers("indices".to_string(), arr1(&[exact]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 8])]);
        let constants = HashSet::new();
        let m = make_context_with_shape_constants(&ws, &shapes, &constants);
        let layer = m
            .convert_gather(&gather_spec("g_exact", 1))
            .expect("exact integer payload should be accepted");
        let Layer::Gather(gather) = layer else {
            panic!("expected Gather layer");
        };
        assert_eq!(
            gather.constant_indices().and_then(|values| values.first()),
            Some(&exact),
            "the rounded f32 mirror must not replace the exact ONNX index"
        );
    }

    /// Embedding lookup: axis=0 with constant data should succeed.
    /// When data is a weight tensor (e.g., [vocab_size, embed_dim]), axis=0
    /// indexes vocabulary, not batch dimension. Part of #3312.
    #[test]
    fn gather_axis_0_constant_data_succeeds() {
        use ndarray::array;
        let mut ws = WeightStore::new();
        // Constant data (embedding table) — makes is_constant("data") true
        ws.insert(
            "data".to_string(),
            array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn(),
        );
        ws.insert("indices".to_string(), arr1(&[0.0_f32, 2.0]).into_dyn());
        let m = make_context(&ws);
        let spec = gather_spec("g_embed", 0);
        let layer = m.convert_gather(&spec).unwrap();
        assert!(matches!(layer, Layer::Gather(_)));
    }

    /// axis=0 without constant data still rejected (activation tensor).
    /// This preserves the #2848 guard for non-embedding Gather ops.
    #[test]
    fn gather_axis_0_dynamic_data_rejected() {
        let mut ws = WeightStore::new();
        // Only indices are constant; data is dynamic (not in weights)
        ws.insert("indices".to_string(), arr1(&[0.0_f32]).into_dyn());
        let m = make_context(&ws);
        let spec = gather_spec("g_dynamic", 0);
        let err = m.convert_gather(&spec).unwrap_err();
        assert!(
            err.to_string().contains("batch dimension"),
            "Expected batch dimension rejection for dynamic data axis=0, got: {err}"
        );
    }

    /// cctsdb_yolo_2023 (axis0-load): a genuine rank-1 data tensor (input `[12296]`)
    /// is gathered along axis=0. Per `unbatched_target_shape_from_onnx_shape`, a
    /// length-≤1 ONNX shape is kept verbatim and never batch-stripped, so axis=0 is a
    /// real data axis — it must NOT be rejected as the synthesized batch axis, and the
    /// Gather must produce the correct shape. This mirrors Slice Case 2b in reshape.rs.
    #[test]
    fn gather_axis_0_rank1_nonconstant_data_is_real_axis() {
        let mut ws = WeightStore::new();
        // Constant indices selecting 3 of 6 elements along the (only) data axis.
        ws.insert("indices".to_string(), arr1(&[0.0_f32, 2.0, 4.0]).into_dyn());
        // `data` is a non-constant rank-1 activation tensor: it never had a batch axis.
        let shapes = HashMap::from([("data".to_string(), vec![6_i64])]);
        let constants = HashSet::new();
        let m = make_context_with_shape_constants(&ws, &shapes, &constants);
        let spec = gather_spec("g_rank1_axis0", 0);

        // Must NOT error with the unbatched-mode batch-axis rejection.
        let layer = m
            .convert_gather(&spec)
            .expect("axis=0 on rank-1 non-constant data must convert (genuine data axis)");

        let Layer::Gather(gather) = layer else {
            panic!("expected Gather layer");
        };

        // Verify the produced layer gathers the correct (internal axis 0) elements and
        // yields the correct output shape [3] from a rank-1 input of length 6.
        let input = BoundedTensor::new(
            arr1(&[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]).into_dyn(),
            arr1(&[0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0]).into_dyn(),
        )
        .expect("valid bounded tensor");
        let output = gather
            .propagate_ibp(&input)
            .expect("Gather IBP on rank-1 axis-0 input should succeed");
        assert_eq!(output.shape(), &[3], "expected gathered shape [3]");
        assert_eq!(
            output.lower().iter().copied().collect::<Vec<_>>(),
            vec![0.0_f32, 2.0, 4.0],
            "expected elements at indices [0, 2, 4] along genuine data axis 0"
        );
    }

    /// Default axis (ONNX default 0) on a rank-1 non-constant tensor is likewise a
    /// genuine data axis and must convert.
    #[test]
    fn gather_default_axis_rank1_nonconstant_data_is_real_axis() {
        let mut ws = WeightStore::new();
        ws.insert("indices".to_string(), arr1(&[1.0_f32]).into_dyn());
        let shapes = HashMap::from([("data".to_string(), vec![4_i64])]);
        let constants = HashSet::new();
        let m = make_context_with_shape_constants(&ws, &shapes, &constants);
        let spec = gather_spec_no_axis("g_rank1_default_axis");
        m.convert_gather(&spec)
            .expect("default axis=0 on rank-1 non-constant data must convert");
    }

    /// Guard preserved: a known rank-2 non-constant tensor's axis=0 IS the stripped
    /// batch axis and must still be rejected in unbatched mode (#2848).
    #[test]
    fn gather_axis_0_rank2_nonconstant_data_still_rejected() {
        let mut ws = WeightStore::new();
        ws.insert("indices".to_string(), arr1(&[0.0_f32]).into_dyn());
        // rank-2 shape => leading dim was a stripped batch axis.
        let shapes = HashMap::from([("data".to_string(), vec![1_i64, 8])]);
        let constants = HashSet::new();
        let m = make_context_with_shape_constants(&ws, &shapes, &constants);
        let spec = gather_spec("g_rank2_axis0", 0);
        let err = m.convert_gather(&spec).unwrap_err();
        assert!(
            err.to_string().contains("batch dimension"),
            "Expected batch dimension rejection for rank-2 data axis=0, got: {err}"
        );
    }

    #[test]
    fn gather_shape_query_last_axis_uses_runtime_last_axis_len_layer() {
        let mut ws = WeightStore::new();
        ws.insert("indices".to_string(), ndarray::arr0(2.0_f32).into_dyn());

        let shapes = HashMap::from([
            ("data".to_string(), vec![3]),
            ("output".to_string(), vec![]),
        ]);
        let constants = HashSet::from(["data".to_string()]);
        let m = make_context_with_shape_constants(&ws, &shapes, &constants);
        let spec = gather_spec("g_shape_last_axis", 0);
        let layer = m
            .convert_gather(&spec)
            .expect("shape-query gather should convert");

        match layer {
            Layer::Gather(gather) => {
                assert!(
                    gather.is_runtime_last_axis_len(),
                    "expected runtime last-axis length lowering"
                );
            }
            other => panic!("expected Gather layer, got {other:?}"),
        }
    }
}
