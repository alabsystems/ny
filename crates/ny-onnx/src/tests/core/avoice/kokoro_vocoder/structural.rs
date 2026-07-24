// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common::unbatched_shape_from_input_spec;
use super::graph_support::instance_norm_node_count;
use super::model::{
    assert_kokoro_vocoder_io_shapes, kokoro_har_time_for_features_t,
    load_kokoro_vocoder_with_fixed_aux, KOKORO_VOCODER_FILE, KOKORO_VOCODER_MIN_FIXED_AUX_T,
    KOKORO_VOCODER_STRUCTURAL_T,
};
use super::*;
use ndarray::{ArrayD, IxDyn};
use ort::{session::Session, value::TensorRef};

fn synthetic_kokoro_vocoder_shape_model(output_shape: Vec<i64>) -> OnnxModel {
    OnnxModel::empty_with_network(
        Network {
            name: "synthetic_kokoro_vocoder_shapes".to_string(),
            inputs: vec![
                TensorSpec {
                    name: "features".to_string(),
                    shape: vec![-1, 512, -1],
                    dtype: DataType::Float32,
                },
                TensorSpec {
                    name: "style".to_string(),
                    shape: vec![-1, 128],
                    dtype: DataType::Float32,
                },
                TensorSpec {
                    name: "har".to_string(),
                    shape: vec![-1, 22, -1],
                    dtype: DataType::Float32,
                },
            ],
            outputs: vec![TensorSpec {
                name: "audio".to_string(),
                shape: output_shape,
                dtype: DataType::Float32,
            }],
            layers: vec![],
            param_count: 0,
        },
        WeightStore::new(),
    )
}

/// Run the real Kokoro vocoder via ORT and return (shape, data).
///
/// Shared backend for the shape-only helper and the full waveform helper.
fn kokoro_vocoder_ort_forward(
    feature_t: usize,
    har_t: usize,
    center_value: f32,
) -> Result<(Vec<usize>, Vec<f32>), ort::Error> {
    let path = require_test_model_with_hint(KOKORO_VOCODER_FILE, AVOICE_TEST_MODEL_HINT);
    let mut session = Session::builder()
        .expect("ORT session builder should initialize")
        .commit_from_file(&path)
        .expect("kokoro_vocoder.onnx should load in ORT");

    let features = ArrayD::<f32>::from_elem(IxDyn(&[1, 512, feature_t]), center_value);
    let style = ArrayD::<f32>::zeros(IxDyn(&[1, 128]));
    let har = ArrayD::<f32>::zeros(IxDyn(&[1, 22, har_t]));

    let features_tensor =
        TensorRef::from_array_view(features.view()).expect("features tensor view should build");
    let style_tensor =
        TensorRef::from_array_view(style.view()).expect("style tensor view should build");
    let har_tensor = TensorRef::from_array_view(har.view()).expect("har tensor view should build");

    let outputs = session.run(ort::inputs! {
        "features" => features_tensor,
        "style" => style_tensor,
        "har" => har_tensor,
    })?;

    let mut iter = outputs.iter();
    let (_, output) = iter
        .next()
        .expect("kokoro vocoder ORT forward should expose one output");
    assert!(
        iter.next().is_none(),
        "kokoro vocoder ORT forward should expose exactly one output"
    );

    let (shape, data) = output.try_extract_tensor::<f32>()?;
    let shape_vec = shape
        .iter()
        .map(|&dim| usize::try_from(dim).expect("ORT output dims should be non-negative"))
        .collect();
    Ok((shape_vec, data.to_vec()))
}

fn kokoro_vocoder_output_shape_from_ort(
    feature_t: usize,
    har_t: usize,
) -> Result<Vec<usize>, ort::Error> {
    let (shape, _data) = kokoro_vocoder_ort_forward(feature_t, har_t, 0.0)?;
    Ok(shape)
}

/// Produce a concrete Kokoro vocoder waveform via ONNX Runtime.
///
/// Returns a zero-width `BoundedTensor` with shape `[1, T_audio]` matching
/// the mel128 builder's input contract.  Uses ORT directly, bypassing the
/// slow `to_graph_network()` const-folding path that makes `features_t >= 5`
/// infeasible on CPU.
///
/// Reference: designs/2026-03-15-issue-3719-ort-concrete-speaker-bridge.md
pub(crate) fn kokoro_vocoder_concrete_waveform_from_ort(
    feature_t: usize,
    center_value: f32,
) -> BoundedTensor {
    let har_t = kokoro_har_time_for_features_t(feature_t);
    let (shape, data) = kokoro_vocoder_ort_forward(feature_t, har_t, center_value)
        .expect("kokoro ORT forward should succeed with exported har contract");

    assert_eq!(
        shape.len(),
        3,
        "ORT output should be rank-3 [B, 1, T_audio], got {:?}",
        shape
    );
    assert_eq!(
        shape[0], 1,
        "ORT output batch dim should be 1, got {}",
        shape[0]
    );
    assert_eq!(
        shape[1], 1,
        "ORT output channel dim should be 1, got {}",
        shape[1]
    );

    // Strip batch axis: [1, 1, T_audio] -> [1, T_audio] to match mel128 input.
    let audio_len = shape[2];
    let waveform = ArrayD::from_shape_vec(IxDyn(&[1, audio_len]), data)
        .expect("ORT output data should reshape to [1, T_audio]");
    BoundedTensor::concrete(waveform).expect("ORT waveform should be finite")
}

#[test]
fn test_load_kokoro_vocoder_with_fixed_aux_rejects_windows_below_runtime_floor_3500() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_MIN_FIXED_AUX_T - 1)
    }));
    assert!(
        result.is_err(),
        "fixed-aux helper should reject temporal windows below the verified runtime floor"
    );
}

#[test]
fn test_unbatched_shape_from_input_spec_replaces_dynamic_axes_3500() {
    let input_spec = TensorSpec {
        name: "har".to_string(),
        shape: vec![-1, 22, -1],
        dtype: DataType::Float32,
    };

    assert_eq!(
        unbatched_shape_from_input_spec(&input_spec, KOKORO_VOCODER_STRUCTURAL_T, "har"),
        vec![22, KOKORO_VOCODER_STRUCTURAL_T]
    );
}

#[test]
fn test_load_kokoro_vocoder_with_fixed_aux_uses_exported_har_contract_3500() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    let model = load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_MIN_FIXED_AUX_T);
    let style = model
        .weights
        .get("style")
        .expect("style should be frozen into the weight store");
    let har = model
        .weights
        .get("har")
        .expect("har should be frozen into the weight store");

    assert_eq!(
        style.shape(),
        &[128],
        "frozen style should use the unbatched export contract"
    );
    assert_eq!(
        har.shape(),
        &[
            22,
            kokoro_har_time_for_features_t(KOKORO_VOCODER_MIN_FIXED_AUX_T)
        ],
        "frozen har should use the exported time-axis contract, not features_t directly"
    );
    assert_eq!(
        model
            .network
            .inputs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>(),
        vec!["features"],
        "fixed-aux helper should leave features as the only activation input"
    );
}

#[test]
fn test_assert_kokoro_vocoder_io_shapes_accepts_export_contract_3500() {
    let model = synthetic_kokoro_vocoder_shape_model(vec![-1, 1, -1]);
    assert_kokoro_vocoder_io_shapes(&model);
}

#[test]
fn test_assert_kokoro_vocoder_io_shapes_rejects_non_channel_axis_one_3500() {
    let model = synthetic_kokoro_vocoder_shape_model(vec![-1, 2, -1]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_kokoro_vocoder_io_shapes(&model)
    }));
    assert!(
        result.is_err(),
        "kokoro vocoder shape gate should reject outputs whose channel axis is not 1"
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(60000))]
#[test]
fn test_load_avoice_kokoro_vocoder_3500() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    let path = require_test_model_with_hint(KOKORO_VOCODER_FILE, AVOICE_TEST_MODEL_HINT);
    let model = load_onnx(&path).expect("Failed to load kokoro_vocoder.onnx");

    let input_names: Vec<&str> = model
        .network
        .inputs
        .iter()
        .map(|s| s.name.as_str())
        .collect();
    for expected in &["features", "style", "har"] {
        assert!(
            input_names.contains(expected),
            "expected {expected} in kokoro vocoder input inventory, got {:?}",
            input_names
        );
    }
    assert_eq!(
        input_names.len(),
        3,
        "kokoro vocoder should have exactly 3 inputs (features, style, har), got {:?}",
        input_names
    );
    assert_kokoro_vocoder_io_shapes(&model);

    let layer_types: Vec<LayerType> = model
        .network
        .layers
        .iter()
        .map(|layer| layer.layer_type.clone())
        .collect();
    assert!(
        layer_types.iter().any(|layer_type| {
            matches!(
                layer_type,
                LayerType::ConvTranspose1d | LayerType::ConvTranspose2d
            )
        }),
        "expected at least one transposed convolution stage in kokoro_vocoder.onnx, got {:?}",
        layer_types
    );
    assert!(
        layer_types
            .iter()
            .any(|layer_type| matches!(layer_type, LayerType::Conv1d | LayerType::Conv2d)),
        "expected at least one convolution stage in kokoro_vocoder.onnx, got {:?}",
        layer_types
    );
    assert!(
        layer_types
            .iter()
            .any(|lt| matches!(lt, LayerType::SiLU | LayerType::ReLU | LayerType::LeakyRelu)),
        "expected at least one activation (SiLU/ReLU/LeakyReLU) in kokoro_vocoder.onnx, got {:?}",
        layer_types
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(60000))]
#[test]
fn test_kokoro_vocoder_ort_forward_accepts_exported_har_contract_3500() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    for feature_t in [1, 6] {
        let har_t = kokoro_har_time_for_features_t(feature_t);
        let output_shape = kokoro_vocoder_output_shape_from_ort(feature_t, har_t)
            .expect("kokoro ORT forward should succeed when har_t follows the export contract");

        assert_eq!(
            output_shape.len(),
            3,
            "export-contract ORT forward should return [B, 1, T], got {:?}",
            output_shape
        );
        assert_eq!(
            output_shape[1], 1,
            "export-contract ORT forward should keep waveform channel axis at 1, got {:?}",
            output_shape
        );
        assert!(
            output_shape[2] > 0,
            "export-contract ORT forward should produce at least one waveform sample, got {:?}",
            output_shape
        );

        eprintln!(
            "kokoro ORT forward: features_t={}, har_t={} -> {:?}",
            feature_t, har_t, output_shape
        );
    }
}

#[cfg_attr(not(debug_assertions), ntest::timeout(60000))]
#[test]
fn test_kokoro_vocoder_ort_rejects_equal_axis_har_shape_3500() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    // Use features_t=6 explicitly (not MIN_FIXED_AUX_T) because har_t=1
    // broadcasts universally — the rejection only triggers when har_t > 1
    // but differs from the expected contract.
    let feature_t = 6;
    let expected_har_t = kokoro_har_time_for_features_t(feature_t);
    let err = kokoro_vocoder_output_shape_from_ort(feature_t, feature_t)
        .expect_err("kokoro ORT forward should reject har_t == features_t for the exported model");

    assert_eq!(
        err.code(),
        ort::ErrorCode::RuntimeException,
        "equal-axis har input should fail inside ORT runtime, got {err:?}"
    );
    assert!(
        err.message().contains("Attempting to broadcast")
            || err
                .message()
                .contains(&format!("{feature_t} by {expected_har_t}")),
        "equal-axis har input should expose the Add broadcast failure, got {err:?}"
    );
    assert!(
        expected_har_t > feature_t,
        "the exported har contract should widen time relative to features_t: \
         features_t={feature_t}, expected_har_t={expected_har_t}"
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_graph_avoice_kokoro_vocoder_fixed_aux_fuses_instance_norm_3591() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    let model = load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_STRUCTURAL_T);
    let graph = model
        .to_graph_network()
        .expect("kokoro vocoder graph conversion should succeed");

    assert!(
        instance_norm_node_count(&graph) > 0,
        "kokoro vocoder graph should fuse at least one decomposed InstanceNorm node"
    );
}

// ---------------------------------------------------------------------------
// Packet 0: graph enumeration diagnostic (#3500)
//
// Prints the full node inventory in topological order, counting
// ConvTranspose nodes as natural cut points for prefix subgraph extraction.
// Reference: designs/2026-03-11-issue-3500-shallow-vocoder-subpath.md §Packet 0
//
// Budget: under the corrected har contract (har_t=60*T+1), to_graph_network()
// const-folds the frozen har branch through the full HiFi-GAN upsampler chain
// via IBP's 4x W+/W- scalar loops.  At T=1, har=[22,61] → upsampled to ~18k
// temporal samples through 3 ConvTranspose stages.  This costs ~90-120s in
// debug mode, but with concurrent cargo lock contention the wall time can
// exceed 180s.  Budget raised to 300s to avoid flaky timeouts under load.
// See #3500 commit notes for the const-fold bottleneck analysis.
// ---------------------------------------------------------------------------

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_kokoro_vocoder_graph_node_inventory_3500() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    let model = load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_STRUCTURAL_T);
    let graph = model
        .to_graph_network()
        .expect("graph conversion should succeed");

    let topo = graph.topological_sort().expect("topo sort should succeed");
    eprintln!("kokoro vocoder: {} nodes in topological order", topo.len());
    for (i, name) in topo.iter().enumerate() {
        if let Some(node) = graph.node(name) {
            eprintln!("  [{:3}] {}: {}", i, node.layer().layer_type(), name);
        }
    }

    assert!(topo.len() > 10, "vocoder should have significant depth");

    let upsample_indices: Vec<usize> = topo
        .iter()
        .enumerate()
        .filter(|(_, name)| {
            graph
                .node(name)
                .map(|n| {
                    n.layer().layer_type() == "ConvTranspose1d"
                        || n.layer().layer_type() == "ConvTranspose2d"
                })
                .unwrap_or(false)
        })
        .map(|(i, _)| i)
        .collect();
    eprintln!("upsampling stage indices: {:?}", upsample_indices);
    assert!(
        !upsample_indices.is_empty(),
        "vocoder should have at least one ConvTranspose node"
    );

    let instance_norm_count = instance_norm_node_count(&graph);
    eprintln!("InstanceNorm1d node count: {}", instance_norm_count);
}
