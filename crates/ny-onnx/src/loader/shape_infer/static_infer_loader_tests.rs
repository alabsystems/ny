// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Loader-level and ORT-differential tests for the static shape fallback.
//!
//! * A load whose shape-inference child cannot run must now carry the
//!   fallback's exact shapes (the measured x86/qemu failure), and a load whose
//!   fallback cannot resolve a tensor must still carry nothing for it.
//! * With the `ort` feature, every shape the fallback fills on the synthetic
//!   graphs must EQUAL what in-process ORT reports for the same bytes, and a
//!   fallback-backed load must see exactly the metadata an ORT-backed load
//!   sees.

use super::static_infer::infer_tensor_shapes_static;
use super::static_infer_tests::regression_models;
use crate::loader::{load_onnx_bytes_with_config, OnnxLoadConfig, ShapeInferBackend};
use crate::onnx_proto;
use prost::Message;

fn unspawnable_backend(dir: &tempfile::TempDir) -> OnnxLoadConfig {
    OnnxLoadConfig::default().with_shape_infer_backend(ShapeInferBackend::Subprocess {
        exe: dir.path().join("no-such-binary"),
    })
}

fn tensor_value_info(name: &str, shape: &[i64]) -> onnx_proto::ValueInfoProto {
    onnx_proto::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::TensorTypeProto {
                elem_type: 1,
                shape: Some(onnx_proto::TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|&dim| onnx_proto::tensor_shape_proto::Dimension {
                            value: Some(
                                onnx_proto::tensor_shape_proto::dimension::Value::DimValue(dim),
                            ),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

/// The same Add -> Relu model the subprocess tests use: `sum` is undeclared,
/// so only shape inference can supply it.
fn add_relu_model_bytes() -> Vec<u8> {
    let node = |name: &str, op: &str, inputs: &[&str], outputs: &[&str]| onnx_proto::NodeProto {
        input: inputs.iter().map(|s| (*s).to_string()).collect(),
        output: outputs.iter().map(|s| (*s).to_string()).collect(),
        name: name.to_string(),
        op_type: op.to_string(),
        ..Default::default()
    };
    let model = onnx_proto::ModelProto {
        ir_version: 9,
        graph: Some(onnx_proto::GraphProto {
            name: "static_fallback_add_relu".to_string(),
            input: vec![
                tensor_value_info("a", &[1, 3]),
                tensor_value_info("b", &[1, 3]),
            ],
            output: vec![tensor_value_info("out", &[1, 3])],
            node: vec![
                node("add", "Add", &["a", "b"], &["sum"]),
                node("relu", "Relu", &["sum"], &["out"]),
            ],
            ..Default::default()
        }),
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            version: super::DEFAULT_OPSET_VERSION,
            domain: String::new(),
        }],
        ..Default::default()
    };
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");
    bytes
}

#[test]
fn unspawnable_shape_child_loads_with_the_static_fallback_shapes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model = load_onnx_bytes_with_config(
        "add_relu",
        &add_relu_model_bytes(),
        &unspawnable_backend(&dir),
    )
    .expect("a failed shape-infer child must degrade, never fail the load");
    assert_eq!(
        model.tensor_shapes().get("sum").map(Vec::as_slice),
        Some([1, 3].as_slice()),
        "the fallback's exact broadcast shape must reach the loaded model"
    );
}

#[test]
fn linearizenn_concat_converts_when_the_shape_child_cannot_run() {
    // The measured x86/qemu regression: without shapes this load refused the
    // Concat ("has no recorded ONNX shape"); with the fallback it converts.
    let dir = tempfile::tempdir().expect("tempdir");
    let (name, bytes) = regression_models()
        .into_iter()
        .find(|(name, _)| *name == "linearizenn_allinone_tail")
        .expect("regression model present");
    let model = load_onnx_bytes_with_config(name, &bytes, &unspawnable_backend(&dir))
        .unwrap_or_else(|err| panic!("{name}: must load through the static fallback: {err}"));
    assert_eq!(
        model
            .tensor_shapes()
            .get("/controller/MatMul_output_0")
            .map(Vec::as_slice),
        Some([1, 1].as_slice())
    );
    assert!(!model.network.layers.is_empty());
}

#[test]
fn unresolvable_tensor_is_never_fabricated_through_the_loader() {
    // `Add` below opset 7 is legacy-broadcast arithmetic the fallback refuses;
    // its output must stay absent from the loaded model's shape table.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut model = onnx_proto::ModelProto::decode(add_relu_model_bytes().as_slice())
        .expect("decode test model");
    for opset in &mut model.opset_import {
        opset.version = 6;
    }
    let table = infer_tensor_shapes_static(&{
        let mut bytes = Vec::new();
        model.encode(&mut bytes).expect("encode");
        bytes
    });
    assert!(!table.shapes.contains_key("sum"));
    assert!(table.unresolved.iter().any(|name| name == "sum"));
    // The loader-level consequence is the historical no-shape behaviour for
    // that tensor, whether or not the rest of the load accepts opset 6.
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode");
    if let Ok(loaded) =
        load_onnx_bytes_with_config("legacy_add", &bytes, &unspawnable_backend(&dir))
    {
        assert!(!loaded.tensor_shapes().contains_key("sum"));
    }
}

/// On every synthetic graph the fallback resolves, its table must EQUAL the
/// in-process ORT table: same keys, same shapes.
#[cfg(feature = "ort")]
#[test]
fn static_fallback_equals_in_process_ort_on_synthetic_graphs() {
    for (name, bytes) in super::static_infer_tests::differential_models() {
        let ort = super::infer_tensor_shapes_from_ort(&bytes)
            .unwrap_or_else(|err| panic!("{name}: ORT must accept the differential model: {err}"));
        let fallback = infer_tensor_shapes_static(&bytes);
        assert!(fallback.refusal.is_none(), "{name}: {:?}", fallback.refusal);
        assert!(
            !fallback.shapes.is_empty(),
            "{name}: fallback filled nothing"
        );
        for (tensor, dims) in &fallback.shapes {
            assert_eq!(
                ort.get(tensor),
                Some(dims),
                "{name}: '{tensor}' fallback {dims:?} vs ORT {:?}",
                ort.get(tensor)
            );
        }
        let mut missing: Vec<&String> = ort
            .keys()
            .filter(|tensor| !fallback.shapes.contains_key(*tensor))
            .collect();
        missing.sort();
        assert!(
            missing.is_empty(),
            "{name}: fallback left ORT-known tensors unknown: {missing:?}"
        );
    }
}

/// A fallback-backed load of the linearizenn tail sees exactly the metadata
/// an in-process-ORT-backed load sees.
#[cfg(feature = "ort")]
#[test]
fn fallback_backed_load_matches_ort_backed_load() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (name, bytes) = regression_models()
        .into_iter()
        .find(|(name, _)| *name == "linearizenn_allinone_tail")
        .expect("regression model present");
    let with_ort = load_onnx_bytes_with_config(
        name,
        &bytes,
        &OnnxLoadConfig::default().with_shape_infer_backend(ShapeInferBackend::InProcess),
    )
    .unwrap_or_else(|err| panic!("{name}: ORT-backed load: {err}"));
    let with_fallback = load_onnx_bytes_with_config(name, &bytes, &unspawnable_backend(&dir))
        .unwrap_or_else(|err| panic!("{name}: fallback-backed load: {err}"));
    assert_eq!(with_fallback.tensor_shapes(), with_ort.tensor_shapes());
    assert_eq!(
        with_fallback.network.layers.len(),
        with_ort.network.layers.len()
    );
}
