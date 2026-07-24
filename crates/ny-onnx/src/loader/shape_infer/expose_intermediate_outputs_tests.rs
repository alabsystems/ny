// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{
    infer_tensor_shapes_from_ort, infer_tensor_shapes_from_ort_path, insert_shape_if_informative,
};
use crate::onnx_proto;
use prost::Message;
use std::collections::HashMap;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

const WINDOWS_MISSING_MODEL_PATH: &str = r"C:\\missing\\model.onnx";

fn tensor_value_info(name: &str, shape: &[i64]) -> onnx_proto::ValueInfoProto {
    let dims = shape
        .iter()
        .map(|dim| onnx_proto::tensor_shape_proto::Dimension {
            value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                *dim,
            )),
        })
        .collect();
    onnx_proto::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::TensorTypeProto {
                elem_type: 1,
                shape: Some(onnx_proto::TensorShapeProto { dim: dims }),
            }),
        }),
    }
}

fn node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> onnx_proto::NodeProto {
    onnx_proto::NodeProto {
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        ..Default::default()
    }
}

fn model_from_graph(graph: onnx_proto::GraphProto) -> onnx_proto::ModelProto {
    onnx_proto::ModelProto {
        ir_version: super::DEFAULT_IR_VERSION,
        graph: Some(graph),
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            version: super::DEFAULT_OPSET_VERSION,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-shape-infer-test".to_string(),
        ..Default::default()
    }
}

fn build_shape_infer_model_bytes() -> Vec<u8> {
    let graph = onnx_proto::GraphProto {
        name: "shape_infer_graph".to_string(),
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
    };
    let mut model = model_from_graph(graph);
    super::normalize_model_for_ort(&mut model);
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");
    bytes
}

fn build_no_runtime_inputs_model_bytes() -> Vec<u8> {
    let graph = onnx_proto::GraphProto {
        name: "shape_infer_constant_graph".to_string(),
        input: vec![tensor_value_info("weight", &[1, 3])],
        output: vec![tensor_value_info("out", &[1, 3])],
        initializer: vec![onnx_proto::TensorProto {
            name: "weight".to_string(),
            dims: vec![1, 3],
            data_type: 1,
            raw_data: Vec::new(),
            ..Default::default()
        }],
        node: vec![node("id", "Identity", &["weight"], &["out"])],
        ..Default::default()
    };
    let mut model = model_from_graph(graph);
    super::normalize_model_for_ort(&mut model);
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");
    bytes
}

#[test]
fn test_infer_shapes_from_ort_memory_adds_intermediate_outputs() {
    let bytes = build_shape_infer_model_bytes();
    let shapes = infer_tensor_shapes_from_ort(&bytes).expect("shape inference from memory");

    assert_eq!(shapes.get("a").map(Vec::as_slice), Some([1, 3].as_slice()));
    assert_eq!(
        shapes.get("sum").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
    assert_eq!(
        shapes.get("out").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
}

#[test]
fn test_infer_shapes_from_ort_skips_when_no_runtime_inputs() {
    let bytes = build_no_runtime_inputs_model_bytes();
    let shapes = infer_tensor_shapes_from_ort(&bytes).expect("shape inference skipped");

    assert!(shapes.is_empty());
}

#[test]
fn test_infer_shapes_from_ort_path_skips_when_no_runtime_inputs() {
    let bytes = build_no_runtime_inputs_model_bytes();
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let temp_path = temp_dir.path().join("model.onnx");

    let shapes =
        infer_tensor_shapes_from_ort_path(&temp_path, &bytes).expect("shape inference skipped");

    assert!(shapes.is_empty());
    assert!(!temp_path.exists());
}

#[test]
fn test_infer_shapes_from_ort_path_uses_tempfile_in_parent() {
    let bytes = build_shape_infer_model_bytes();
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let temp_path = temp_dir.path().join("model.onnx");

    let shapes =
        infer_tensor_shapes_from_ort_path(&temp_path, &bytes).expect("shape inference from path");

    assert_eq!(
        shapes.get("sum").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
    assert_eq!(
        shapes.get("out").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
}

#[test]
fn test_infer_shapes_from_ort_path_missing_parent_uses_system_temp() {
    let bytes = build_shape_infer_model_bytes();
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let temp_path = temp_dir.path().join("missing-dir").join("model.onnx");

    assert!(!temp_path.parent().expect("missing parent").exists());

    let shapes = infer_tensor_shapes_from_ort_path(&temp_path, &bytes)
        .expect("shape inference from missing parent");

    assert_eq!(
        shapes.get("sum").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
    assert_eq!(
        shapes.get("out").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
    assert!(!temp_path.exists());
}

#[test]
fn test_write_temp_onnx_file_windows_parent_falls_back() {
    let bytes = build_shape_infer_model_bytes();
    let windows_path = std::path::Path::new(WINDOWS_MISSING_MODEL_PATH);
    let temp_path =
        super::write_temp_onnx_file(windows_path, &bytes).expect("temp file for windows path");
    let temp_dir = std::env::temp_dir();

    assert!(
        temp_path.starts_with(&temp_dir),
        "Temp file should be created under system temp dir"
    );
}

#[test]
fn test_write_temp_onnx_file_drive_relative_parent_falls_back() {
    let bytes = build_shape_infer_model_bytes();
    let drive_relative = std::path::Path::new("C:relative.onnx");
    let temp_path = super::write_temp_onnx_file(drive_relative, &bytes)
        .expect("temp file for drive-relative path");
    let temp_dir = std::env::temp_dir();

    assert!(
        temp_path.starts_with(&temp_dir),
        "Temp file should be created under system temp dir"
    );
}

#[test]
fn test_write_temp_onnx_file_rejects_empty_bytes() {
    let windows_path = std::path::Path::new(WINDOWS_MISSING_MODEL_PATH);
    let err = super::write_temp_onnx_file(windows_path, &[]).expect_err("empty bytes should error");

    assert!(err
        .to_string()
        .contains("Refusing to write empty ONNX model bytes"));
}

#[cfg(unix)]
#[test]
fn test_infer_shapes_from_ort_path_readonly_parent_uses_system_temp() {
    let bytes = build_shape_infer_model_bytes();
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let readonly_dir = temp_dir.path().join("readonly");
    fs::create_dir(&readonly_dir).expect("readonly dir");
    let mut perms = fs::metadata(&readonly_dir)
        .expect("readonly metadata")
        .permissions();
    perms.set_mode(0o500);
    fs::set_permissions(&readonly_dir, perms).expect("readonly perms");

    let temp_path = readonly_dir.join("model.onnx");
    let shapes = infer_tensor_shapes_from_ort_path(&temp_path, &bytes)
        .expect("shape inference from readonly parent");

    assert_eq!(
        shapes.get("sum").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
    assert_eq!(
        shapes.get("out").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
    assert!(!temp_path.exists());

    let mut perms = fs::metadata(&readonly_dir)
        .expect("readonly metadata reset")
        .permissions();
    perms.set_mode(0o700);
    fs::set_permissions(&readonly_dir, perms).expect("reset perms");
}

#[test]
fn test_infer_shapes_from_ort_path_gz_uses_memory_bytes() {
    let bytes = build_shape_infer_model_bytes();
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let temp_path = temp_dir.path().join("model.onnx.gz");

    assert!(!temp_path.exists());
    let shapes = infer_tensor_shapes_from_ort_path(&temp_path, &bytes)
        .expect("shape inference from gz path");
    assert!(!temp_path.exists());

    assert_eq!(
        shapes.get("sum").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
    assert_eq!(
        shapes.get("out").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
}

#[test]
fn test_infer_shapes_from_ort_path_gz_case_insensitive() {
    let bytes = build_shape_infer_model_bytes();
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let temp_path = temp_dir.path().join("model.ONNX.GZ");

    assert!(!temp_path.exists());
    let shapes = infer_tensor_shapes_from_ort_path(&temp_path, &bytes)
        .expect("shape inference from gz path");
    assert!(!temp_path.exists());

    assert_eq!(
        shapes.get("sum").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
    assert_eq!(
        shapes.get("out").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
}

#[test]
fn test_insert_shape_if_informative_filters_empty_names() {
    let mut shapes = HashMap::new();
    insert_shape_if_informative(&mut shapes, "", &[1, 3]);
    assert!(shapes.is_empty());
}

#[test]
fn test_insert_shape_if_informative_accepts_scalar_shapes() {
    let mut shapes = HashMap::new();
    insert_shape_if_informative(&mut shapes, "scalar", &[]);
    assert_eq!(shapes.get("scalar").map(Vec::as_slice), Some([].as_slice()));
}

#[test]
fn test_insert_shape_if_informative_skips_non_positive_dims() {
    let mut shapes = HashMap::new();
    insert_shape_if_informative(&mut shapes, "unknown", &[-1, 0]);
    assert!(!shapes.contains_key("unknown"));
}

#[test]
fn test_insert_shape_if_informative_keeps_positive_dims() {
    let mut shapes = HashMap::new();
    insert_shape_if_informative(&mut shapes, "valid", &[-1, 3]);
    assert_eq!(
        shapes.get("valid").map(Vec::as_slice),
        Some([-1, 3].as_slice())
    );
}

fn conv_node(name: &str, x: &str, w: &str, y: &str) -> onnx_proto::NodeProto {
    let mut n = node(name, "Conv", &[x, w], &[y]);
    n.attribute = vec![
        onnx_proto::AttributeProto {
            name: "kernel_shape".to_string(),
            ints: vec![3, 3],
            r#type: 7,
            ..Default::default()
        },
        onnx_proto::AttributeProto {
            name: "pads".to_string(),
            ints: vec![1, 1, 1, 1],
            r#type: 7,
            ..Default::default()
        },
    ];
    n
}

fn conv_weight(name: &str, dims: &[i64]) -> onnx_proto::TensorProto {
    let count: i64 = dims.iter().product();
    onnx_proto::TensorProto {
        name: name.to_string(),
        dims: dims.to_vec(),
        data_type: 1,
        float_data: vec![0.0f32; count as usize],
        ..Default::default()
    }
}

/// Build a tinyimagenet-style residual stub: an intermediate Conv output
/// ('120') feeds a downstream Conv ('Conv_8'). The exporter exposes '120' as a
/// graph output AND mis-annotates it as rank-1 `{64}` (confusing it with a
/// length-64 bias/scale) even though it is really 4-D. `placement` controls
/// where the contradictory `{64}` annotation lives.
enum BadShapePlacement {
    ValueInfo,
    Output,
}

fn build_contradictory_conv_model(placement: BadShapePlacement) -> Vec<u8> {
    let mut output = vec![tensor_value_info("out", &[1, 64, 29, 29])];
    let mut value_info = Vec::new();
    match placement {
        BadShapePlacement::ValueInfo => value_info.push(tensor_value_info("120", &[64])),
        BadShapePlacement::Output => output.push(tensor_value_info("120", &[64])),
    }
    // `..Default::default()` is load-bearing when `onnx-value-info` is off: the
    // cfg-gated `value_info` field disappears and the update fills nothing else.
    #[allow(clippy::needless_update)]
    let graph = onnx_proto::GraphProto {
        name: "residual_conv_stub".to_string(),
        input: vec![tensor_value_info("X", &[1, 3, 29, 29])],
        output,
        initializer: vec![
            conv_weight("w1", &[64, 3, 3, 3]),
            conv_weight("w2", &[64, 64, 3, 3]),
        ],
        node: vec![
            conv_node("Conv_1", "X", "w1", "120"),
            conv_node("Conv_8", "120", "w2", "out"),
        ],
        #[cfg(feature = "onnx-value-info")]
        value_info,
        ..Default::default()
    };
    model_from_graph(graph).encode_to_vec()
}

/// Regression: a contradictory rank-1 annotation on the intermediate Conv
/// output '120' carried in `value_info` must not survive into ORT. ORT must
/// re-derive the true 4-D shape and the downstream Conv_8 ('out') must resolve.
/// Covered by the original `clear_intermediate_value_info_shapes` fix.
#[test]
fn test_contradictory_intermediate_shape_in_value_info_is_repaired() {
    let bytes = build_contradictory_conv_model(BadShapePlacement::ValueInfo);
    let shapes = infer_tensor_shapes_from_ort(&bytes).expect("ORT shape inference must succeed");

    assert_eq!(
        shapes.get("120").map(Vec::as_slice),
        Some([1, 64, 29, 29].as_slice()),
        "intermediate Conv output '120' must resolve to its true 4-D shape, not rank-1 {{64}}"
    );
    assert_eq!(
        shapes.get("out").map(Vec::as_slice),
        Some([1, 64, 29, 29].as_slice()),
        "downstream Conv_8 output must resolve (ORT pass not skipped)"
    );
}

/// Regression for the tinyimagenet_2024 ResNet warning that persisted after the
/// value_info clear: the contradictory rank-1 `{64}` shape on intermediate
/// output '120' lives in `graph.output` (an exposed intermediate), which the
/// value_info clear intentionally skips. `clear_reconsumed_output_shapes` must
/// strip it so ORT's full inference succeeds (no "skipped" fallback) and the
/// downstream Conv_8 gets its proper 4-D input.
#[test]
fn test_contradictory_intermediate_shape_in_output_is_repaired() {
    let bytes = build_contradictory_conv_model(BadShapePlacement::Output);
    let shapes = infer_tensor_shapes_from_ort(&bytes).expect("ORT shape inference must succeed");

    assert_eq!(
        shapes.get("120").map(Vec::as_slice),
        Some([1, 64, 29, 29].as_slice()),
        "re-consumed intermediate '120' exposed in graph.output must resolve to 4-D, \
         not the stale rank-1 {{64}} annotation"
    );
    assert_eq!(
        shapes.get("out").map(Vec::as_slice),
        Some([1, 64, 29, 29].as_slice()),
        "downstream Conv_8 must resolve, proving ORT shape inference was not skipped"
    );
}

/// A genuine terminal model output (produced but never consumed downstream)
/// must keep its declared shape — `clear_reconsumed_output_shapes` must not
/// touch the I/O contract.
#[test]
fn test_terminal_output_shape_is_preserved() {
    let graph = onnx_proto::GraphProto {
        name: "terminal_output".to_string(),
        input: vec![tensor_value_info("X", &[1, 3, 29, 29])],
        output: vec![tensor_value_info("out", &[1, 64, 29, 29])],
        initializer: vec![conv_weight("w1", &[64, 3, 3, 3])],
        node: vec![conv_node("Conv_1", "X", "w1", "out")],
        ..Default::default()
    };
    let bytes = model_from_graph(graph).encode_to_vec();
    let exposed = super::expose_intermediate_outputs(&bytes).expect("expose outputs");
    let decoded =
        onnx_proto::ModelProto::decode(exposed.bytes.as_slice()).expect("decode exposed model");
    let out = decoded
        .graph
        .expect("graph")
        .output
        .into_iter()
        .find(|o| o.name == "out")
        .expect("terminal output present");
    let shape = out
        .r#type
        .and_then(|t| t.tensor_type)
        .and_then(|t| t.shape)
        .expect("terminal output keeps its declared shape");
    assert_eq!(
        shape.dim.len(),
        4,
        "terminal model output must retain its 4-D I/O contract shape"
    );
}

#[test]
fn test_expose_intermediate_outputs_dedupes_output_names() {
    let graph = onnx_proto::GraphProto {
        name: "dedupe_outputs".to_string(),
        input: vec![tensor_value_info("input", &[1, 3])],
        output: Vec::new(),
        node: vec![
            node("first", "Identity", &["input"], &["dup"]),
            node("second", "Identity", &["dup"], &["dup"]),
        ],
        ..Default::default()
    };
    let model = model_from_graph(graph);
    let bytes = model.encode_to_vec();
    let exposed = super::expose_intermediate_outputs(&bytes).expect("expose outputs");
    let decoded = onnx_proto::ModelProto::decode(exposed.bytes.as_slice()).expect("decode exposed");
    let output_count = decoded
        .graph
        .expect("graph")
        .output
        .iter()
        .filter(|output| output.name == "dup")
        .count();

    assert_eq!(output_count, 1);
}

// ─── vit_2023 Transpose-on-rank-mismatch normalization (ORT) ───────────────

fn transpose_node_perm(name: &str, x: &str, y: &str, perm: &[i64]) -> onnx_proto::NodeProto {
    let mut n = node(name, "Transpose", &[x], &[y]);
    n.attribute = vec![onnx_proto::AttributeProto {
        name: "perm".to_string(),
        ints: perm.to_vec(),
        r#type: 7,
        ..Default::default()
    }];
    n
}

fn rank1_initializer(name: &str, len: usize) -> onnx_proto::TensorProto {
    onnx_proto::TensorProto {
        name: name.to_string(),
        dims: vec![len as i64],
        data_type: 1,
        float_data: (0..len).map(|i| i as f32 * 0.01).collect(),
        ..Default::default()
    }
}

/// Regression for vit_2023 native load. A `Transpose perm={0,2,1}` applied to a
/// genuinely rank-1 `{48}` initializer (e.g. a positional embedding) makes ONNX
/// Runtime shape inference reject the node
/// ("[TypeInferenceError] Invalid attribute perm {0, 2, 1}, input shape = {48}")
/// and ABORT the entire pass — so even the healthy MatMul tensor loses its shape
/// and the model can only fall back to unknown bounds.
///
/// After `normalize_transpose_perms_for_ort` rewrites the rank-1 perm to the
/// identity, ORT must (1) succeed and (2) infer the correct shapes for the
/// healthy path. Transposing a rank-1 tensor is the identity, so the rewrite
/// does not change the graph's function.
#[test]
fn test_vit_transpose_rank1_perm_lets_ort_succeed() {
    let wmat = onnx_proto::TensorProto {
        name: "w".to_string(),
        dims: vec![48, 48],
        data_type: 1,
        float_data: vec![0.0f32; 48 * 48],
        ..Default::default()
    };
    // `..Default::default()` is load-bearing when `onnx-value-info` is off: the
    // cfg-gated `value_info` field disappears and the update fills nothing else.
    #[allow(clippy::needless_update)]
    let graph = onnx_proto::GraphProto {
        name: "vit_attn_stub".to_string(),
        input: vec![tensor_value_info("X", &[1, 4, 48])],
        output: vec![
            tensor_value_info("t", &[48]),
            tensor_value_info("good", &[1, 4, 48]),
        ],
        initializer: vec![rank1_initializer("emb", 48), wmat],
        node: vec![
            // Offending: perm len 3 on rank-1 {48} initializer.
            transpose_node_perm("Transpose_emb", "emb", "t", &[0, 2, 1]),
            // Healthy path that must still get a shape once ORT no longer aborts.
            node("MatMul_good", "MatMul", &["X", "w"], &["good"]),
        ],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
        ..Default::default()
    };
    let bytes = model_from_graph(graph).encode_to_vec();

    let shapes = infer_tensor_shapes_from_ort(&bytes)
        .expect("ORT shape inference must succeed after perm normalization");

    // The healthy MatMul output resolves => ORT did NOT abort the whole pass.
    assert_eq!(
        shapes.get("good").map(Vec::as_slice),
        Some([1, 4, 48].as_slice()),
        "healthy MatMul output must resolve, proving the perm/rank mismatch no longer aborts ORT"
    );
    // The transposed rank-1 tensor keeps its {48} shape (identity transpose).
    assert_eq!(
        shapes.get("t").map(Vec::as_slice),
        Some([48].as_slice()),
        "transpose of a rank-1 tensor is the identity: shape stays {{48}}"
    );
}

/// The pre-ORT pass must leave a healthy higher-rank Transpose perm untouched.
#[test]
fn test_vit_transpose_rank3_perm_preserved() {
    // emb3 is a rank-3 {1,4,48} initializer; perm={0,2,1} is valid for it.
    let emb3 = onnx_proto::TensorProto {
        name: "emb3".to_string(),
        dims: vec![1, 4, 48],
        data_type: 1,
        float_data: vec![0.0f32; 4 * 48],
        ..Default::default()
    };
    // `..Default::default()` is load-bearing when `onnx-value-info` is off: the
    // cfg-gated `value_info` field disappears and the update fills nothing else.
    #[allow(clippy::needless_update)]
    let graph = onnx_proto::GraphProto {
        name: "vit_rank3".to_string(),
        input: vec![tensor_value_info("X", &[1, 48, 4])],
        output: vec![tensor_value_info("t", &[1, 48, 4])],
        initializer: vec![emb3],
        node: vec![transpose_node_perm(
            "Transpose_emb3",
            "emb3",
            "t",
            &[0, 2, 1],
        )],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
        ..Default::default()
    };
    let bytes = model_from_graph(graph).encode_to_vec();
    let shapes =
        infer_tensor_shapes_from_ort(&bytes).expect("rank-consistent perm must already pass ORT");
    assert_eq!(
        shapes.get("t").map(Vec::as_slice),
        Some([1, 48, 4].as_slice()),
        "valid rank-3 perm={{0,2,1}} must transpose {{1,4,48}} -> {{1,48,4}} unchanged"
    );
}
