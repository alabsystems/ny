// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::PathBuf;

use ndarray::{ArrayD, IxDyn};
use prost::Message;

use super::{
    estimate_model_cost, estimate_model_timing, CostResult, FamilyTimingCalibration, LayerCost,
    TimingProfile,
};
use crate::onnx_proto;
use crate::test_fixtures::{
    require_test_model, require_test_model_with_hint,
    specialize_kokoro_duration_predictor_for_lstm_unroll, AVOICE_TEST_MODEL_HINT,
};
use crate::{
    load_onnx, load_onnx_bytes, DataType, LayerSpec, Network, OnnxModel, TensorSpec, WeightStore,
};
use ny_core::LayerType;

#[ntest::timeout(10000)]
#[test]
fn test_estimate_model_cost_single_linear_counts_flops_and_memory() {
    let path = require_test_model("single_linear.onnx");
    let model = load_onnx(&path).expect("Failed to load single_linear.onnx");

    let result = estimate_model_cost(&model).expect("cost analysis should succeed");

    assert_eq!(result.layers.len(), 1);
    assert_eq!(result.total_flops, 15, "3 outputs * (2 mul-adds + bias)");
    assert_eq!(result.parameter_bytes, 36, "6 weights + 3 bias values");
    assert_eq!(
        result.peak_activation_bytes, 20,
        "input (2 f32) + output (3 f32) should be live at peak"
    );
    assert_eq!(result.peak_total_bytes, 56);
    assert_eq!(result.layers[0].output_elements, 3);
    assert_eq!(result.layers[0].output_bytes, 12);
    assert_eq!(result.layers[0].activation_input_bytes, 8);
    assert_eq!(result.layers[0].parameter_input_bytes, 36);
    assert_eq!(result.layers[0].total_tensor_traffic_bytes, 56);
    assert_eq!(result.layers[0].timing_family, "dense_mac");
}

#[test]
fn test_estimate_model_cost_skips_importer_folded_constant_layer() {
    let mut weights = WeightStore::new();
    for name in ["weight_a", "weight_b", "folded_mm"] {
        weights.insert(name.to_string(), ArrayD::zeros(IxDyn(&[1, 1])));
    }
    let model = OnnxModel::empty_with_network(
        Network {
            name: "folded_constant_cost".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, 2],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![1, 2],
                dtype: DataType::Float32,
            }],
            layers: vec![
                LayerSpec {
                    name: "folded_matmul".to_string(),
                    layer_type: LayerType::MatMul,
                    inputs: vec!["weight_a".to_string(), "weight_b".to_string()],
                    outputs: vec!["folded_mm".to_string()],
                    weights: None,
                    attributes: std::collections::HashMap::new(),
                },
                LayerSpec {
                    name: "runtime_relu".to_string(),
                    layer_type: LayerType::ReLU,
                    inputs: vec!["input".to_string()],
                    outputs: vec!["output".to_string()],
                    weights: None,
                    attributes: std::collections::HashMap::new(),
                },
            ],
            param_count: 3,
        },
        weights,
    )
    .with_tensor_shapes(std::collections::HashMap::from([
        ("input".to_string(), vec![1, 2]),
        ("output".to_string(), vec![1, 2]),
        ("folded_mm".to_string(), vec![1, 1]),
    ]));

    let result =
        estimate_model_cost(&model).expect("constant-only layer should have zero runtime cost");

    assert_eq!(result.layers.len(), 1);
    assert_eq!(result.layers[0].name, "runtime_relu");
    assert_eq!(result.total_flops, 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_estimate_model_cost_conv_relu_tracks_activation_liveness() {
    let path = require_test_model("conv_relu.onnx");
    let model = load_onnx(&path).expect("Failed to load conv_relu.onnx");

    let result = estimate_model_cost(&model).expect("cost analysis should succeed");

    assert_eq!(result.layers.len(), 2);
    assert_eq!(
        result.layers[0].output_bytes, 72,
        "conv output should be 2x3x3 f32"
    );
    assert_eq!(
        result.layers[1].output_bytes, 72,
        "relu output shape should match conv output"
    );
    assert_eq!(
        result.peak_activation_bytes, 144,
        "peak should occur while both conv output and relu output are live"
    );
    assert!(
        result.total_flops > 0,
        "conv + relu should have non-zero FLOPs"
    );
    assert!(
        result.layers[0].activation_input_bytes > 0,
        "conv should read non-zero activation traffic"
    );
    assert!(
        result.layers[0].parameter_input_bytes > 0,
        "conv should read non-zero parameter traffic"
    );
    assert!(
        result.layers[0].total_tensor_traffic_bytes > result.layers[0].output_bytes,
        "conv tensor traffic should include reads in addition to writes"
    );
    assert_eq!(result.layers[0].timing_family, "convolution");
    assert!(
        result.layers[1].activation_input_bytes > 0,
        "relu should read the conv activation output"
    );
    assert_eq!(result.layers[1].parameter_input_bytes, 0);
    assert!(
        result.layers[1].total_tensor_traffic_bytes > result.layers[1].output_bytes,
        "relu tensor traffic should include the input read"
    );
    assert_eq!(result.layers[1].timing_family, "elementwise");
}

#[test]
fn test_estimate_model_cost_handles_triu_layer_type_4270() {
    let model = OnnxModel::empty_with_network(
        Network {
            name: "triu_cost".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![3, 3],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![3, 3],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "triu".to_string(),
                layer_type: LayerType::Triu,
                inputs: vec!["input".to_string()],
                outputs: vec!["output".to_string()],
                weights: None,
                attributes: std::collections::HashMap::new(),
            }],
            param_count: 0,
        },
        WeightStore::new(),
    )
    .with_tensor_shapes(std::collections::HashMap::from([
        ("input".to_string(), vec![3, 3]),
        ("output".to_string(), vec![3, 3]),
    ]));

    let result = estimate_model_cost(&model).expect("Triu cost analysis should succeed");

    assert_eq!(result.layers.len(), 1);
    assert_eq!(result.layers[0].layer_type, "Triu");
    assert_eq!(result.layers[0].timing_family, "elementwise");
    assert_eq!(result.layers[0].output_shapes, vec![vec![3, 3]]);
    assert_eq!(result.layers[0].output_elements, 9);
    assert_eq!(result.layers[0].flops, 9);
}

#[test]
fn test_estimate_model_timing_uses_profile_and_workspace_slack() {
    let cost = synthetic_timing_cost();
    let profile = synthetic_timing_profile();

    let estimate = estimate_model_timing(&cost, &profile).expect("timing estimate should succeed");

    assert_eq!(estimate.profile_name, "synthetic");
    assert_eq!(estimate.backend, "wgpu");
    assert_eq!(estimate.layers.len(), 2);
    assert_eq!(estimate.layers[0].name, "dense");
    assert_eq!(estimate.layers[1].name, "reshape");
    // Conservative composition: launch_overhead + compute + memory (sum, not max).
    // ceil_time_ns applies next_up bias before ceil, so exact divisions round up by 1.
    //
    // dense: flops=1000/100.0 → exact 10.0 → next_up → 11,
    //        bytes=448/50.0 → 8.96 → next_up → ceil → 9
    //        total = 10 + 11 + 9 = 30
    // reshape: flops=0 → 0, bytes=128/64.0 → exact 2.0 → next_up → 3
    //          total = 5 + 0 + 3 = 8
    assert_eq!(estimate.layers[0].compute_time_ns, 11);
    assert_eq!(estimate.layers[0].memory_time_ns, 9);
    assert_eq!(estimate.layers[0].total_time_ns, 30);
    assert_eq!(estimate.layers[1].compute_time_ns, 0);
    assert_eq!(estimate.layers[1].memory_time_ns, 3);
    assert_eq!(estimate.layers[1].total_time_ns, 8);
    assert_eq!(estimate.total_time_ns, 38);
    assert_eq!(estimate.peak_memory_bytes, 1_024);
    assert!(
        estimate
            .assumptions
            .iter()
            .all(|assumption| !assumption.trim().is_empty()),
        "timing assumptions should be non-empty"
    );
}

fn synthetic_timing_cost() -> CostResult {
    CostResult {
        layers: vec![
            LayerCost {
                name: "dense".to_string(),
                layer_type: "Linear".to_string(),
                output_shapes: vec![vec![1, 8]],
                output_elements: 8,
                flops: 1_000,
                activation_input_bytes: 128,
                parameter_input_bytes: 256,
                output_bytes: 64,
                total_tensor_traffic_bytes: 448,
                timing_family: "dense_mac".to_string(),
                peak_live_activation_bytes: 512,
                cumulative_flops: 1_000,
            },
            LayerCost {
                name: "reshape".to_string(),
                layer_type: "Reshape".to_string(),
                output_shapes: vec![vec![1, 8]],
                output_elements: 8,
                flops: 0,
                activation_input_bytes: 64,
                parameter_input_bytes: 0,
                output_bytes: 64,
                total_tensor_traffic_bytes: 128,
                timing_family: "shape_only".to_string(),
                peak_live_activation_bytes: 512,
                cumulative_flops: 1_000,
            },
        ],
        total_flops: 1_000,
        parameter_bytes: 256,
        peak_activation_bytes: 512,
        peak_total_bytes: 768,
        assumptions: Vec::new(),
    }
}

fn synthetic_timing_profile() -> TimingProfile {
    TimingProfile {
        schema_version: 1,
        profile_name: "synthetic".to_string(),
        backend: "wgpu".to_string(),
        device_info: "unit-test".to_string(),
        workspace_slack_bytes: 256,
        families: BTreeMap::from([
            (
                "dense_mac".to_string(),
                FamilyTimingCalibration {
                    min_effective_flops_per_ns: 100.0,
                    min_effective_bytes_per_ns: 50.0,
                    launch_overhead_ns: 10,
                },
            ),
            (
                "shape_only".to_string(),
                FamilyTimingCalibration {
                    min_effective_flops_per_ns: 1.0,
                    min_effective_bytes_per_ns: 64.0,
                    launch_overhead_ns: 5,
                },
            ),
        ]),
    }
}

fn calibration_profile_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../calibration/m4-max-metal-conservative.json")
}

fn load_m4_max_profile() -> TimingProfile {
    let path = calibration_profile_path();
    let file = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("Failed to open calibration profile at {:?}: {}", path, e));
    serde_json::from_reader(std::io::BufReader::new(file))
        .unwrap_or_else(|e| panic!("Failed to parse calibration profile at {:?}: {}", path, e))
}

#[test]
fn test_m4_max_profile_deserializes_and_round_trips() {
    let profile = load_m4_max_profile();

    assert_eq!(profile.schema_version, 1);
    assert_eq!(profile.backend, "metal");
    assert!(
        profile.profile_name.contains("m4-max"),
        "profile name should identify M4 Max"
    );
    assert!(
        profile.device_info.contains("M4 Max"),
        "device info should mention M4 Max"
    );
    assert_eq!(
        profile.workspace_slack_bytes, 67_108_864,
        "workspace slack should be 64 MiB"
    );

    // All seven timing families must be present.
    let expected_families = [
        "convolution",
        "dense_mac",
        "elementwise",
        "normalization",
        "reduction",
        "shape_only",
        "softmax",
    ];
    for family in &expected_families {
        assert!(
            profile.families.contains_key(*family),
            "profile should contain family '{}'",
            family
        );
        let cal = &profile.families[*family];
        assert!(
            cal.min_effective_flops_per_ns > 0.0 && cal.min_effective_flops_per_ns.is_finite(),
            "family '{}' must have positive finite FLOPs/ns rate",
            family
        );
        assert!(
            cal.min_effective_bytes_per_ns > 0.0 && cal.min_effective_bytes_per_ns.is_finite(),
            "family '{}' must have positive finite bytes/ns rate",
            family
        );
    }

    // Round-trip: serialize and deserialize should produce identical data.
    let json = serde_json::to_string_pretty(&profile).expect("serialize should succeed");
    let reparsed: TimingProfile =
        serde_json::from_str(&json).expect("round-trip deserialize should succeed");
    assert_eq!(reparsed.schema_version, profile.schema_version);
    assert_eq!(reparsed.profile_name, profile.profile_name);
    assert_eq!(reparsed.families.len(), profile.families.len());
}

#[ntest::timeout(10000)]
#[test]
fn test_end_to_end_timing_estimate_with_cnn_model() {
    let path = require_test_model("conv_relu_maxpool.onnx");
    let model = load_onnx(&path).expect("Failed to load conv_relu_maxpool.onnx");
    let profile = load_m4_max_profile();

    let cost = estimate_model_cost(&model).expect("cost analysis should succeed");
    assert!(
        cost.layers.len() >= 3,
        "CNN should have 3+ layers, got {}",
        cost.layers.len()
    );
    assert!(cost.total_flops > 0, "CNN should have non-zero FLOPs");

    let timing = estimate_model_timing(&cost, &profile).expect("timing estimate should succeed");

    assert_eq!(timing.backend, "metal");
    assert!(timing.total_time_ns > 0, "total latency should be positive");
    assert!(
        timing.peak_memory_bytes >= profile.workspace_slack_bytes,
        "peak memory should include workspace slack"
    );
    assert_eq!(timing.layers.len(), cost.layers.len());

    // Every layer should have a positive total time.
    for layer in &timing.layers {
        assert!(
            layer.total_time_ns > 0,
            "layer '{}' should have positive total time",
            layer.name
        );
    }

    // The sum of individual layer times should equal the total.
    let sum: u64 = timing.layers.iter().map(|l| l.total_time_ns).sum();
    assert_eq!(
        sum, timing.total_time_ns,
        "sum of per-layer times should equal total"
    );

    // Sanity: the timing summary should be non-empty and contain key fields.
    let summary = timing.summary();
    assert!(
        summary.contains("Timing Estimate"),
        "summary should have header"
    );
    assert!(summary.contains("metal"), "summary should mention backend");
    assert!(
        summary.contains("Total latency bound"),
        "summary should contain latency bound"
    );
    assert!(
        summary.contains("Peak memory bound"),
        "summary should contain peak memory bound"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_end_to_end_timing_estimate_with_conv_relu_model() {
    let path = require_test_model("conv_relu.onnx");
    let model = load_onnx(&path).expect("Failed to load conv_relu.onnx");
    let profile = load_m4_max_profile();

    let cost = estimate_model_cost(&model).expect("cost analysis should succeed");
    let timing = estimate_model_timing(&cost, &profile).expect("timing estimate should succeed");

    assert_eq!(timing.layers.len(), 2);
    assert_eq!(timing.layers[0].timing_family, "convolution");
    assert_eq!(timing.layers[1].timing_family, "elementwise");

    // Conv layer should have compute time > 0 (non-zero FLOPs) and memory time > 0.
    assert!(
        timing.layers[0].compute_time_ns > 0,
        "conv layer should have non-zero compute time"
    );
    assert!(
        timing.layers[0].memory_time_ns > 0,
        "conv layer should have non-zero memory time"
    );

    // Elementwise (ReLU) layer should have non-zero memory time.
    assert!(
        timing.layers[1].memory_time_ns > 0,
        "relu layer should have non-zero memory time"
    );
}

#[test]
fn test_timing_estimate_rejects_missing_family() {
    let cost = synthetic_timing_cost();
    // Profile with only dense_mac — missing shape_only needed by the reshape layer.
    let incomplete_profile = TimingProfile {
        schema_version: 1,
        profile_name: "incomplete".to_string(),
        backend: "test".to_string(),
        device_info: "test".to_string(),
        workspace_slack_bytes: 0,
        families: BTreeMap::from([(
            "dense_mac".to_string(),
            FamilyTimingCalibration {
                min_effective_flops_per_ns: 100.0,
                min_effective_bytes_per_ns: 50.0,
                launch_overhead_ns: 10,
            },
        )]),
    };

    let result = estimate_model_timing(&cost, &incomplete_profile);
    assert!(
        result.is_err(),
        "should reject profile missing required family"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("shape_only"),
        "error should mention missing family 'shape_only', got: {}",
        err_msg
    );
}

/// M4 Max peak references for profile sanity checks.
const M4_MAX_PEAK_FLOPS_PER_NS: f64 = 53_000.0; // ~53 TFLOPS FP32
const M4_MAX_PEAK_BYTES_PER_NS: f64 = 546.0; // 546 GB/s

#[test]
fn test_m4_max_profile_compute_rates_consistent_with_flops_per_ns_units_3498() {
    let profile = load_m4_max_profile();

    let compute_families = ["dense_mac", "convolution"];
    for family in &compute_families {
        let cal = &profile.families[*family];
        assert!(
            cal.min_effective_flops_per_ns >= 0.001 * M4_MAX_PEAK_FLOPS_PER_NS,
            "family '{}' rate {:.1} FLOPs/ns is below 0.1% of peak ({:.0} FLOPs/ns) — \
             likely a nanoseconds-vs-seconds unit error",
            family,
            cal.min_effective_flops_per_ns,
            M4_MAX_PEAK_FLOPS_PER_NS,
        );
        assert!(
            cal.min_effective_flops_per_ns < M4_MAX_PEAK_FLOPS_PER_NS,
            "family '{}' rate {:.1} FLOPs/ns exceeds theoretical peak {:.0} FLOPs/ns",
            family,
            cal.min_effective_flops_per_ns,
            M4_MAX_PEAK_FLOPS_PER_NS,
        );
    }

    for (family, cal) in &profile.families {
        assert!(
            cal.min_effective_bytes_per_ns > 0.0 && cal.min_effective_bytes_per_ns.is_finite(),
            "family '{}' must have positive finite bytes/ns rate",
            family,
        );
        assert!(
            cal.min_effective_bytes_per_ns < M4_MAX_PEAK_BYTES_PER_NS,
            "family '{}' bytes/ns rate {:.1} exceeds peak bandwidth {:.0} bytes/ns",
            family,
            cal.min_effective_bytes_per_ns,
            M4_MAX_PEAK_BYTES_PER_NS,
        );
    }
}

#[test]
fn test_ceil_time_ns_conservative_for_large_flops_counter_3498() {
    // 2^53 + 1: the smallest u64 that cannot be represented exactly in f64.
    let large_work_units: u64 = 9_007_199_254_740_993;

    let cost = CostResult {
        layers: vec![LayerCost {
            name: "big_dense".to_string(),
            layer_type: "Linear".to_string(),
            output_shapes: vec![vec![1, 1]],
            output_elements: 1,
            flops: large_work_units,
            activation_input_bytes: 0,
            parameter_input_bytes: 0,
            output_bytes: 0,
            total_tensor_traffic_bytes: 0,
            timing_family: "dense_mac".to_string(),
            peak_live_activation_bytes: 0,
            cumulative_flops: large_work_units,
        }],
        total_flops: large_work_units,
        parameter_bytes: 0,
        peak_activation_bytes: 0,
        peak_total_bytes: 0,
        assumptions: Vec::new(),
    };

    let profile = TimingProfile {
        schema_version: 1,
        profile_name: "unit-rate".to_string(),
        backend: "test".to_string(),
        device_info: "test".to_string(),
        workspace_slack_bytes: 0,
        families: BTreeMap::from([(
            "dense_mac".to_string(),
            FamilyTimingCalibration {
                min_effective_flops_per_ns: 1.0,
                min_effective_bytes_per_ns: 1.0,
                launch_overhead_ns: 0,
            },
        )]),
    };

    let estimate = estimate_model_timing(&cost, &profile).expect("timing estimate should succeed");
    // With rate=1.0, ceil(work_units / 1.0) must be >= work_units.
    // The conservative bias in ceil_time_ns ensures the f64 rounding
    // doesn't produce a value below the exact mathematical ceiling.
    assert!(
        estimate.layers[0].compute_time_ns >= large_work_units,
        "compute_time_ns {} must be >= exact work_units {} for certificate soundness",
        estimate.layers[0].compute_time_ns,
        large_work_units,
    );
}

#[test]
fn test_ceil_time_ns_conservative_for_large_bytes_counter_3498() {
    let large_work_units: u64 = 9_007_199_254_740_993;

    let cost = CostResult {
        layers: vec![LayerCost {
            name: "big_memcpy".to_string(),
            layer_type: "Reshape".to_string(),
            output_shapes: vec![vec![1, 1]],
            output_elements: 1,
            flops: 0,
            activation_input_bytes: large_work_units / 2,
            parameter_input_bytes: 0,
            output_bytes: large_work_units / 2 + 1,
            total_tensor_traffic_bytes: large_work_units,
            timing_family: "shape_only".to_string(),
            peak_live_activation_bytes: 0,
            cumulative_flops: 0,
        }],
        total_flops: 0,
        parameter_bytes: 0,
        peak_activation_bytes: 0,
        peak_total_bytes: 0,
        assumptions: Vec::new(),
    };

    let profile = TimingProfile {
        schema_version: 1,
        profile_name: "unit-rate".to_string(),
        backend: "test".to_string(),
        device_info: "test".to_string(),
        workspace_slack_bytes: 0,
        families: BTreeMap::from([(
            "shape_only".to_string(),
            FamilyTimingCalibration {
                min_effective_flops_per_ns: 1.0,
                min_effective_bytes_per_ns: 1.0,
                launch_overhead_ns: 0,
            },
        )]),
    };

    let estimate = estimate_model_timing(&cost, &profile).expect("timing estimate should succeed");
    assert!(
        estimate.layers[0].memory_time_ns >= large_work_units,
        "memory_time_ns {} must be >= exact work_units {} for certificate soundness",
        estimate.layers[0].memory_time_ns,
        large_work_units,
    );
}

// ========================================================================
// Real avoice timing smoke tests (#3498)
//
// Exercise the timing estimator on shape-specialized variants of the real
// avoice ONNX exports from #3554. These are deterministic static-cost
// smoke tests — no runtime benchmarking, no Metal calibration claims.
//
// Reference: designs/2026-03-12-issue-3498-real-export-timing-smoke.md
// ========================================================================

/// Replace all dynamic dimensions in the protobuf with `default_value`.
///
/// Handles both unnamed dynamic dims (DimValue <= 0) and named symbolic dims
/// (DimParam). This makes the model admissible to `estimate_model_cost` which
/// rejects dynamic dimensions.
///
/// Source: adapted from `duration_predictor/real_export.rs:concretize_symbolic_sequence_len`.
fn concretize_all_dynamic_dims(proto: &mut onnx_proto::ModelProto, default_value: i64) {
    use onnx_proto::tensor_shape_proto::dimension::Value;

    let set_dims = |infos: &mut [onnx_proto::ValueInfoProto]| {
        for info in infos {
            let Some(tensor_type) = info.r#type.as_mut().and_then(|ty| ty.tensor_type.as_mut())
            else {
                continue;
            };
            let Some(shape) = tensor_type.shape.as_mut() else {
                continue;
            };
            for dim in &mut shape.dim {
                let is_dynamic = match &dim.value {
                    Some(Value::DimValue(v)) => *v <= 0,
                    Some(Value::DimParam(_)) => true,
                    None => true,
                };
                if is_dynamic {
                    dim.value = Some(Value::DimValue(default_value));
                }
            }
        }
    };

    let Some(graph) = proto.graph.as_mut() else {
        return;
    };
    set_dims(&mut graph.input);
    set_dims(&mut graph.output);
    #[cfg(feature = "onnx-value-info")]
    set_dims(&mut graph.value_info);
}

/// Set the complete shape of a named input in the protobuf.
fn set_proto_input_shape(proto: &mut onnx_proto::ModelProto, input_name: &str, shape: &[i64]) {
    use onnx_proto::tensor_shape_proto::{dimension::Value, Dimension};

    let Some(graph) = proto.graph.as_mut() else {
        return;
    };
    for info in &mut graph.input {
        if info.name != input_name {
            continue;
        }
        let Some(tensor_type) = info.r#type.as_mut().and_then(|ty| ty.tensor_type.as_mut()) else {
            continue;
        };
        tensor_type.shape = Some(onnx_proto::TensorShapeProto {
            dim: shape
                .iter()
                .map(|&v| Dimension {
                    value: Some(Value::DimValue(v)),
                })
                .collect(),
        });
    }
}

// --- Shape-specialized model loaders for timing smoke ---

/// Speaker encoder with dynamic mel-sequence axis specialized to 5.
///
/// Source: `crates/ny-onnx/src/tests/core/avoice/speaker_encoder/mod.rs:15`
pub(super) fn load_speaker_encoder_timing_model() -> OnnxModel {
    let path = require_test_model_with_hint("speaker_encoder.onnx", AVOICE_TEST_MODEL_HINT);
    let bytes = std::fs::read(&path).expect("read speaker encoder bytes");
    let mut proto =
        onnx_proto::ModelProto::decode(bytes.as_slice()).expect("decode speaker encoder proto");
    // Input [B, T, 128]: B and T are dynamic. Replace all dynamic dims with 5,
    // the minimum valid T for the speaker encoder's fixed TDNN pads
    // [2, 2, 3, 4, 0] because reflect padding requires pad < T.
    // Batch becomes 5 too — acceptable for a non-exact smoke test.
    concretize_all_dynamic_dims(&mut proto, 5);
    load_onnx_bytes("speaker_encoder_timing", &proto.encode_to_vec())
        .expect("speaker encoder should load with concretized dims")
}

/// Talker attention with shared sequence axis specialized to 16.
///
/// Source: `crates/ny-onnx/src/tests/core/avoice/talker_attention/mod.rs:45`
pub(super) fn load_talker_attention_timing_model() -> OnnxModel {
    let path = require_test_model_with_hint("talker_attention_layer0.onnx", AVOICE_TEST_MODEL_HINT);
    let bytes = std::fs::read(&path).expect("read talker attention bytes");
    let mut proto =
        onnx_proto::ModelProto::decode(bytes.as_slice()).expect("decode talker attention proto");
    // Four inputs: hidden_states [B, T, H], cos [1, 1, T, 64],
    // sin [1, 1, T, 64], mask [1, 1, T, T]. All dynamic T → 16.
    concretize_all_dynamic_dims(&mut proto, 16);
    load_onnx_bytes("talker_attention_timing", &proto.encode_to_vec())
        .expect("talker attention should load with concretized dims")
}

/// Kokoro vocoder with features_t=1, har_t=61, audio_t=300.
///
/// Source: `crates/ny-onnx/src/tests/core/avoice/kokoro_vocoder/mod.rs:42-136`
pub(super) fn load_kokoro_vocoder_timing_model() -> OnnxModel {
    let path = require_test_model_with_hint("kokoro_vocoder.onnx", AVOICE_TEST_MODEL_HINT);
    let bytes = std::fs::read(&path).expect("read vocoder bytes");
    let mut proto = onnx_proto::ModelProto::decode(bytes.as_slice()).expect("decode vocoder proto");
    // Three inputs: features [B, 512, T_f], style [B, 128], har [B, 22, T_h].
    // har_t = 60 * features_t + 1 = 61 when features_t = 1.
    set_proto_input_shape(&mut proto, "features", &[1, 512, 1]);
    set_proto_input_shape(&mut proto, "style", &[1, 128]);
    set_proto_input_shape(&mut proto, "har", &[1, 22, 61]);
    // Remaining dynamic dims in intermediates/outputs → let ORT infer from
    // concrete inputs. Fallback: any leftover dynamic dims get value 1.
    concretize_all_dynamic_dims(&mut proto, 1);
    load_onnx_bytes("kokoro_vocoder_timing", &proto.encode_to_vec())
        .expect("vocoder should load with concretized dims")
}

/// Duration predictor with symbolic T=4 and unrolled-LSTM rewrite.
///
/// Source: `crates/ny-onnx/src/tests/core/avoice/duration_predictor/real_export.rs:27-46`
const DURATION_PREDICTOR_SEQ_LEN: i64 = 4;

pub(super) fn load_duration_predictor_timing_model() -> OnnxModel {
    let path =
        require_test_model_with_hint("kokoro_duration_predictor.onnx", AVOICE_TEST_MODEL_HINT);
    let bytes = std::fs::read(&path).expect("read duration predictor bytes");
    let mut proto =
        onnx_proto::ModelProto::decode(bytes.as_slice()).expect("decode duration predictor proto");
    specialize_kokoro_duration_predictor_for_lstm_unroll(&mut proto, DURATION_PREDICTOR_SEQ_LEN);
    // Replace any remaining dynamic dims (batch, etc.) with a safe default.
    concretize_all_dynamic_dims(&mut proto, DURATION_PREDICTOR_SEQ_LEN);
    load_onnx_bytes("kokoro_duration_predictor_timing", &proto.encode_to_vec())
        .expect("duration predictor should load after specialization + LSTM rewrite")
}

// --- Common timing smoke helper ---

fn run_real_export_timing_smoke(
    model: &OnnxModel,
    label: &str,
) -> (CostResult, super::TimingEstimate) {
    let profile = load_m4_max_profile();

    let cost = estimate_model_cost(model)
        .unwrap_or_else(|e| panic!("{label}: cost analysis should succeed: {e}"));
    assert!(
        !cost.layers.is_empty(),
        "{label}: cost.layers should be non-empty"
    );
    assert!(
        cost.total_flops > 0,
        "{label}: total FLOPs should be positive"
    );

    let timing = estimate_model_timing(&cost, &profile)
        .unwrap_or_else(|e| panic!("{label}: timing estimate should succeed: {e}"));
    assert_eq!(
        timing.layers.len(),
        cost.layers.len(),
        "{label}: timing layers count should match cost layers"
    );
    assert!(
        timing.total_time_ns > 0,
        "{label}: total latency should be positive"
    );
    assert!(
        timing.peak_memory_bytes >= cost.peak_total_bytes,
        "{label}: timing peak memory ({}) should be >= cost peak total ({})",
        timing.peak_memory_bytes,
        cost.peak_total_bytes
    );

    (cost, timing)
}

#[ntest::timeout(60000)]
#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_speaker_encoder_timing_profile_smoke_3498() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = load_speaker_encoder_timing_model();
    let (cost, timing) = run_real_export_timing_smoke(&model, "speaker encoder");

    // The ECAPA-TDNN speaker encoder is a convolutional architecture:
    // Conv1d blocks + SE-Res2Net + channel attention + statistics pooling.
    let has_expected_family = cost
        .layers
        .iter()
        .any(|layer| layer.timing_family == "convolution" || layer.timing_family == "reduction");
    assert!(
        has_expected_family,
        "speaker encoder should contain convolution or reduction family, got families: {:?}",
        cost.layers
            .iter()
            .map(|l| l.timing_family.as_str())
            .collect::<Vec<_>>()
    );

    eprintln!(
        "speaker encoder timing smoke: {} layers, {} total FLOPs, {} ns total latency",
        timing.layers.len(),
        cost.total_flops,
        timing.total_time_ns
    );
}

#[ntest::timeout(60000)]
#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_talker_attention_timing_profile_smoke_3498() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let model = load_talker_attention_timing_model();
    let (cost, timing) = run_real_export_timing_smoke(&model, "talker attention");

    // Talker attention should contain both dense_mac (MatMul) and softmax families.
    let has_dense = cost.layers.iter().any(|l| l.timing_family == "dense_mac");
    let has_softmax = cost.layers.iter().any(|l| l.timing_family == "softmax");
    assert!(
        has_dense,
        "talker attention should contain dense_mac family"
    );
    assert!(
        has_softmax,
        "talker attention should contain softmax family"
    );

    eprintln!(
        "talker attention timing smoke: {} layers, {} total FLOPs, {} ns total latency",
        timing.layers.len(),
        cost.total_flops,
        timing.total_time_ns
    );
}

#[ntest::timeout(60000)]
#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_kokoro_vocoder_timing_profile_smoke_3498() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    let model = load_kokoro_vocoder_timing_model();
    let (cost, timing) = run_real_export_timing_smoke(&model, "kokoro vocoder");

    // The Kokoro HiFi-GAN vocoder should contain convolution family layers.
    let has_convolution = cost.layers.iter().any(|l| l.timing_family == "convolution");
    assert!(
        has_convolution,
        "kokoro vocoder should contain convolution family, got families: {:?}",
        cost.layers
            .iter()
            .map(|l| l.timing_family.as_str())
            .collect::<Vec<_>>()
    );

    eprintln!(
        "kokoro vocoder timing smoke: {} layers, {} total FLOPs, {} ns total latency",
        timing.layers.len(),
        cost.total_flops,
        timing.total_time_ns
    );
}

#[ntest::timeout(120000)]
#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_duration_predictor_timing_profile_smoke_3498() {
    crate::test_fixtures::assert_test_model_available!("kokoro_duration_predictor.onnx");
    let model = load_duration_predictor_timing_model();
    let (cost, timing) = run_real_export_timing_smoke(&model, "duration predictor");

    // The duration predictor should contain dense_mac (Linear/MatMul) or
    // elementwise families after LSTM unrolling.
    let has_expected_family = cost
        .layers
        .iter()
        .any(|layer| layer.timing_family == "dense_mac" || layer.timing_family == "elementwise");
    assert!(
        has_expected_family,
        "duration predictor should contain dense_mac or elementwise family, got families: {:?}",
        cost.layers
            .iter()
            .map(|l| l.timing_family.as_str())
            .collect::<Vec<_>>()
    );

    eprintln!(
        "duration predictor timing smoke: {} layers, {} total FLOPs, {} ns total latency",
        timing.layers.len(),
        cost.total_flops,
        timing.total_time_ns
    );
}
