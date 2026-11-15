// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The zero-dim rank probe (`resolve_zero_dim_ranks`): a tensor ONNX Runtime
//! could not type must never be recorded as the scalar `[]`, and a tensor ORT
//! proves rank 0 must still be.
//!
//! ORT's C API reports ZERO dims for both. The synthetic graph below holds
//! both kinds side by side, so dropping every zero-dim report fails the scalar
//! assertions and the old rule (keep every one as `[]`) fails the
//! unknown-rank ones. The expected verdicts were measured on this exact graph
//! with ORT 1.26 (Python) and ORT 1.28 through the C API (the build ny
//! links): probe dims `[0]` for `d`, `k`, `s` and `[-1]` for `y`, `z`.

use super::subprocess::{parse_shape_infer_response, serve_shape_infer_request};
use super::{
    build_rank_probe_model, infer_tensor_shapes_from_ort, infer_tensor_shapes_from_ort_path,
    probe_proves_rank_zero, resolve_zero_dim_ranks, SessionShapes, SessionSource,
    ORT_SHAPE_INFERENCE_TIMEOUT, RANK_PROBE_PREFIX,
};
use crate::onnx_proto;
use prost::Message;
use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

const DT_FLOAT: i32 = 1;
const DT_INT64: i32 = 7;

fn value_info(name: &str, elem_type: i32, dims: Option<&[i64]>) -> onnx_proto::ValueInfoProto {
    onnx_proto::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::TensorTypeProto {
                elem_type,
                shape: dims.map(|dims| onnx_proto::TensorShapeProto {
                    dim: dims
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

fn ints_attr(name: &str, values: &[i64]) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        r#type: onnx_proto::attribute_type::INTS,
        ints: values.to_vec(),
        ..Default::default()
    }
}

fn int_attr(name: &str, value: i64) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        r#type: onnx_proto::attribute_type::INT,
        i: Some(value),
        ..Default::default()
    }
}

fn node(
    name: &str,
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attribute: Vec<onnx_proto::AttributeProto>,
) -> onnx_proto::NodeProto {
    onnx_proto::NodeProto {
        input: inputs.iter().map(|name| (*name).to_string()).collect(),
        output: outputs.iter().map(|name| (*name).to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute,
    }
}

fn model_bytes(graph: onnx_proto::GraphProto, opset: i64) -> Vec<u8> {
    onnx_proto::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: opset,
        }],
        producer_name: "ny-onnx-rank-probe-test".to_string(),
        graph: Some(graph),
        ..Default::default()
    }
    .encode_to_vec()
}

/// The mixed graph (opset 9, IR 8):
/// * `y = Slice(a)` in the opset-9 attribute form with UNSORTED axes `[1, 0]`
///   (the reviewer's idiom): ORT's shape inference leaves `y` untyped although
///   its runtime shape is `[2, 6]`, and `z = Relu(y)` inherits that;
/// * `d = Gather(Shape(x), idx)` with a scalar index and
///   `s = ReduceSum(x, keepdims=0)`: genuine rank-0 intermediates;
/// * `k`: a runtime graph input DECLARED rank 0 (`Mul(x, k)` uses it);
/// * `a`, `sh = Shape(x)` and `xk`: ordinary typed tensors, the controls that
///   prove ORT ran.
fn mixed_rank_model_bytes() -> Vec<u8> {
    let idx = onnx_proto::TensorProto {
        name: "idx".to_string(),
        dims: Vec::new(),
        data_type: DT_INT64,
        int64_data: vec![1],
        ..Default::default()
    };
    let graph = onnx_proto::GraphProto {
        name: "rank_probe_mixed".to_string(),
        input: vec![
            value_info("a", DT_FLOAT, Some(&[5, 7])),
            value_info("x", DT_FLOAT, Some(&[2, 3])),
            value_info("k", DT_FLOAT, Some(&[])),
        ],
        output: vec![
            value_info("z", DT_FLOAT, None),
            value_info("d", DT_INT64, None),
            value_info("s", DT_FLOAT, None),
            value_info("xk", DT_FLOAT, None),
        ],
        initializer: vec![idx],
        node: vec![
            node(
                "slice_unsorted_axes",
                "Slice",
                &["a"],
                &["y"],
                vec![
                    ints_attr("starts", &[1, -3]),
                    ints_attr("ends", &[i64::MAX, -1]),
                    ints_attr("axes", &[1, 0]),
                ],
            ),
            node("relu", "Relu", &["y"], &["z"], Vec::new()),
            node("shape", "Shape", &["x"], &["sh"], Vec::new()),
            node(
                "gather",
                "Gather",
                &["sh", "idx"],
                &["d"],
                vec![int_attr("axis", 0)],
            ),
            node(
                "reduce",
                "ReduceSum",
                &["x"],
                &["s"],
                vec![int_attr("keepdims", 0)],
            ),
            node("mul", "Mul", &["x", "k"], &["xk"], Vec::new()),
        ],
        ..Default::default()
    };
    model_bytes(graph, 9)
}

#[track_caller]
fn assert_mixed_rank_verdicts(shapes: &HashMap<String, Vec<i64>>) {
    // Controls: ORT ran and typed the healthy tensors.
    assert_eq!(shapes.get("a").map(Vec::as_slice), Some([5, 7].as_slice()));
    assert_eq!(shapes.get("sh").map(Vec::as_slice), Some([2].as_slice()));
    assert_eq!(shapes.get("xk").map(Vec::as_slice), Some([2, 3].as_slice()));
    // Proven rank 0: still recorded as the scalar shape.
    for scalar in ["d", "s", "k"] {
        assert_eq!(
            shapes.get(scalar).map(Vec::as_slice),
            Some([].as_slice()),
            "'{scalar}' is proven rank 0 and must stay recorded as []; table = {shapes:?}"
        );
    }
    // Rank unknown to ORT (runtime [2, 6]): recorded as NOTHING, never [].
    for untyped in ["y", "z"] {
        assert_eq!(
            shapes.get(untyped),
            None,
            "'{untyped}' has no ORT-inferred rank and must be absent, not a fabricated scalar"
        );
    }
    assert!(
        shapes
            .keys()
            .all(|name| !name.starts_with(RANK_PROBE_PREFIX)),
        "a rank-probe output leaked into the shape table: {shapes:?}"
    );
}

#[test]
fn unknown_rank_tensor_is_not_recorded_as_scalar_but_proven_scalars_are() {
    let shapes = infer_tensor_shapes_from_ort(&mixed_rank_model_bytes())
        .expect("ORT shape inference must succeed on the mixed graph");
    assert_mixed_rank_verdicts(&shapes);
}

/// The file-backed path loads the probe the way it loads the first session
/// (a temporary file beside the model), and cleans both files up.
#[test]
fn file_backed_inference_applies_the_same_rank_probe() {
    let bytes = mixed_rank_model_bytes();
    let dir = tempfile::tempdir().expect("temp dir");
    let shapes = infer_tensor_shapes_from_ort_path(&dir.path().join("model.onnx"), &bytes)
        .expect("file-backed ORT shape inference must succeed on the mixed graph");
    assert_mixed_rank_verdicts(&shapes);
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .expect("read temp dir")
        .map(|entry| entry.expect("dir entry").path())
        .collect();
    assert!(
        leftovers.is_empty(),
        "session temp files must be removed: {leftovers:?}"
    );
}

/// The subprocess child runs the rank probe and the parent only parses: the
/// served table carries the same verdicts as the in-process one.
#[test]
fn served_shape_table_carries_the_rank_probe_verdicts() {
    let bytes = mixed_rank_model_bytes();
    let mut payload = Vec::new();
    serve_shape_infer_request(&mut bytes.as_slice(), &mut payload).expect("serve");
    let served = parse_shape_infer_response(&payload).expect("parse served payload");
    assert_mixed_rank_verdicts(&served);
    let direct = infer_tensor_shapes_from_ort(&bytes).expect("direct in-process inference");
    assert_eq!(served, direct);
}

/// A runtime graph input declared with NO shape reaches ORT untyped, so it too
/// was reported with zero dims and used to be recorded as a scalar.
#[test]
fn shapeless_graph_input_is_not_recorded_as_scalar() {
    let graph = onnx_proto::GraphProto {
        name: "rank_probe_shapeless_input".to_string(),
        input: vec![
            value_info("x", DT_FLOAT, None),
            value_info("w", DT_FLOAT, Some(&[1, 3])),
        ],
        output: vec![
            value_info("y", DT_FLOAT, None),
            value_info("v", DT_FLOAT, None),
        ],
        node: vec![
            node("relu_x", "Relu", &["x"], &["y"], Vec::new()),
            node("relu_w", "Relu", &["w"], &["v"], Vec::new()),
        ],
        ..Default::default()
    };
    let shapes = infer_tensor_shapes_from_ort(&model_bytes(graph, 13))
        .expect("ORT shape inference must succeed");
    // Control: the typed branch resolved.
    assert_eq!(shapes.get("v").map(Vec::as_slice), Some([1, 3].as_slice()));
    assert_eq!(shapes.get("x"), None, "untyped input recorded: {shapes:?}");
    assert_eq!(shapes.get("y"), None, "untyped output recorded: {shapes:?}");
}

/// Only TENSOR-typed zero-dim reports may join the probe:
/// `Shape` of a sequence is an invalid graph (measured on ORT 1.26 and on the
/// ORT 1.28 C API: "Type 'seq(tensor(float))' of input parameter (sq) of
/// operator (Shape) ... is invalid"), and one invalid probe node sinks every
/// candidate of the model. So the declared scalar input `p` must stay `[]`
/// next to the sequence `sq`.
///
/// The same graph pins the other half of the soundness rule: `e =
/// SequenceAt(SplitToSequence(a, keepdims=0), p)` and `z = Neg(e)` ARE scalars
/// at run time, but ORT's type for them has no shape (probe dims `[-1]` on
/// ORT 1.26 and 1.28), so they are left unknown rather than guessed — the old
/// rule recorded them `[]`, right only by accident.
#[test]
fn sequence_outputs_do_not_sink_the_probe_and_untyped_runtime_scalars_stay_unknown() {
    let graph = onnx_proto::GraphProto {
        name: "rank_probe_sequence".to_string(),
        input: vec![
            value_info("a", DT_FLOAT, Some(&[3])),
            value_info("p", DT_INT64, Some(&[])),
        ],
        output: vec![value_info("z", DT_FLOAT, None)],
        node: vec![
            node(
                "split",
                "SplitToSequence",
                &["a"],
                &["sq"],
                vec![int_attr("keepdims", 0)],
            ),
            node("at", "SequenceAt", &["sq", "p"], &["e"], Vec::new()),
            node("neg", "Neg", &["e"], &["z"], Vec::new()),
        ],
        ..Default::default()
    };
    let shapes = infer_tensor_shapes_from_ort(&model_bytes(graph, 13))
        .expect("ORT shape inference must succeed on the sequence graph");
    // Control: ORT ran and typed the tensor input.
    assert_eq!(shapes.get("a").map(Vec::as_slice), Some([3].as_slice()));
    assert_eq!(
        shapes.get("p").map(Vec::as_slice),
        Some([].as_slice()),
        "the declared scalar input must survive a sequence output in the same model: {shapes:?}"
    );
    for absent in ["sq", "e", "z"] {
        assert_eq!(
            shapes.get(absent),
            None,
            "'{absent}' has no rank-0 ORT type and must be absent: {shapes:?}"
        );
    }
}

/// The mixed graph's exposed form, with its declared scalar `k` and typed `x`
/// as the first session would report them, for driving
/// [`resolve_zero_dim_ranks`] directly.
fn reported_k_and_x(extra_candidate: Option<&str>) -> SessionShapes {
    let mut reported = SessionShapes::default();
    reported.shapes.insert("x".to_string(), vec![2, 3]);
    reported.zero_dims.insert("k".to_string());
    if let Some(extra) = extra_candidate {
        reported.zero_dims.insert(extra.to_string());
    }
    reported
}

#[track_caller]
fn assert_probe_failure_recorded_no_scalar(shapes: &HashMap<String, Vec<i64>>) {
    assert_eq!(
        shapes.get("x").map(Vec::as_slice),
        Some([2, 3].as_slice()),
        "a failed probe must leave the first session's non-empty shapes intact: {shapes:?}"
    );
    assert_eq!(
        shapes.get("k"),
        None,
        "a failed probe proves nothing: even the declared scalar must stay unknown: {shapes:?}"
    );
    assert_eq!(shapes.len(), 1, "nothing else may be recorded: {shapes:?}");
}

/// A probe session ORT REFUSES (here a candidate no node
/// produces; ORT 1.26 and 1.28 both answer "Node input 'no_such_tensor' is not
/// a graph input, initializer, or output of a previous node") records no `[]`
/// at all — not for the unprovable candidate and not for the genuine scalar
/// beside it.
#[test]
fn refused_rank_probe_records_no_scalar() {
    let shapes = resolve_zero_dim_ranks(
        reported_k_and_x(Some("no_such_tensor")),
        &mixed_rank_model_bytes(),
        &SessionSource::Memory,
        Instant::now(),
    );
    assert_probe_failure_recorded_no_scalar(&shapes);
}

/// With the shared budget already spent, the probe is not
/// run and every candidate stays unknown.
#[test]
fn rank_probe_past_the_budget_records_no_scalar() {
    let started = Instant::now()
        .checked_sub(ORT_SHAPE_INFERENCE_TIMEOUT + Duration::from_secs(1))
        .expect("the monotonic clock has run longer than the shape-inference budget");
    let shapes = resolve_zero_dim_ranks(
        reported_k_and_x(None),
        &mixed_rank_model_bytes(),
        &SessionSource::Memory,
        started,
    );
    assert_probe_failure_recorded_no_scalar(&shapes);
}

/// A probe model that cannot be built (bytes with no graph)
/// records no `[]` either.
#[test]
fn unbuildable_rank_probe_records_no_scalar() {
    let shapes = resolve_zero_dim_ranks(
        reported_k_and_x(None),
        &[],
        &SessionSource::Memory,
        Instant::now(),
    );
    assert_probe_failure_recorded_no_scalar(&shapes);
}

/// Control for the three failure tests above: the SAME inputs with a working
/// probe do record the declared scalar, so their "absent" is the failure
/// path's doing and not an artifact of how the tests drive the function.
#[test]
fn working_rank_probe_records_the_declared_scalar() {
    let shapes = resolve_zero_dim_ranks(
        reported_k_and_x(None),
        &mixed_rank_model_bytes(),
        &SessionSource::Memory,
        Instant::now(),
    );
    assert_eq!(shapes.get("k").map(Vec::as_slice), Some([].as_slice()));
    assert_eq!(shapes.get("x").map(Vec::as_slice), Some([2, 3].as_slice()));
    assert_eq!(shapes.len(), 2, "{shapes:?}");
}

/// A model-authored `[]` may reach the ORT table only
/// through a graph-input or TERMINAL-output declaration. A stale `[]` on an
/// intermediate — here `s`, the opset-9 unsorted-axes Slice that ORT cannot
/// type (runtime `[2, 6]`) — is stripped by `expose_intermediate_outputs`
/// before ORT sees it, so the probe reports `[-1]` and `s` (and `z = Neg(s)`)
/// stay unknown. Measured on ORT 1.26 and the ORT 1.28 C API. `k`, declared
/// rank 0 on a runtime graph input (ORT enforces input ranks), is the control
/// that the probe ran.
fn stale_scalar_annotation_graph(s_declared_as_output: bool) -> onnx_proto::GraphProto {
    let mut output = vec![
        value_info("z", DT_FLOAT, None),
        value_info("ak", DT_FLOAT, None),
    ];
    if s_declared_as_output {
        output.insert(0, value_info("s", DT_FLOAT, Some(&[])));
    }
    onnx_proto::GraphProto {
        name: "rank_probe_stale_scalar_annotation".to_string(),
        input: vec![
            value_info("a", DT_FLOAT, Some(&[5, 7])),
            value_info("k", DT_FLOAT, Some(&[])),
        ],
        output,
        node: vec![
            node(
                "slice_unsorted_axes",
                "Slice",
                &["a"],
                &["s"],
                vec![
                    ints_attr("starts", &[1, -3]),
                    ints_attr("ends", &[i64::MAX, -1]),
                    ints_attr("axes", &[1, 0]),
                ],
            ),
            node("neg", "Neg", &["s"], &["z"], Vec::new()),
            node("mul", "Mul", &["a", "k"], &["ak"], Vec::new()),
        ],
        ..Default::default()
    }
}

#[track_caller]
fn assert_stale_scalar_annotation_ignored(shapes: &HashMap<String, Vec<i64>>) {
    // Controls: ORT ran, and the probe proved the declared scalar input.
    assert_eq!(shapes.get("ak").map(Vec::as_slice), Some([5, 7].as_slice()));
    assert_eq!(shapes.get("k").map(Vec::as_slice), Some([].as_slice()));
    for untyped in ["s", "z"] {
        assert_eq!(
            shapes.get(untyped),
            None,
            "'{untyped}' is untyped by ORT; a stale model annotation must not make it []: \
             {shapes:?}"
        );
    }
}

/// A graph output that a node re-consumes, declared `[]` over an untyped
/// producer: `clear_reconsumed_output_shapes` strips the declaration.
#[test]
fn reconsumed_output_declared_scalar_is_not_recorded() {
    let bytes = model_bytes(stale_scalar_annotation_graph(true), 9);
    let shapes = infer_tensor_shapes_from_ort(&bytes).expect("ORT shape inference must succeed");
    assert_stale_scalar_annotation_ignored(&shapes);
}

/// An intermediate `value_info` declaring `[]` over an untyped producer:
/// `clear_intermediate_value_info_shapes` strips the declaration. Gated on the
/// feature that models `value_info` at all, so it can never pass vacuously.
#[cfg(feature = "onnx-value-info")]
#[test]
fn stale_value_info_scalar_is_not_recorded() {
    let mut graph = stale_scalar_annotation_graph(false);
    graph.value_info.push(value_info("s", DT_FLOAT, Some(&[])));
    let shapes = infer_tensor_shapes_from_ort(&model_bytes(graph, 9))
        .expect("ORT shape inference must succeed");
    assert_stale_scalar_annotation_ignored(&shapes);
}

#[test]
fn only_an_exactly_zero_length_shape_output_proves_rank_zero() {
    let zero: &[i64] = &[0];
    let empty: &[i64] = &[];
    let unknown: &[i64] = &[-1];
    let one: &[i64] = &[1];
    let two: &[i64] = &[2];
    let pair: &[i64] = &[0, 0];
    assert!(probe_proves_rank_zero(Some(zero)));
    for unproven in [
        None,
        Some(empty),
        Some(unknown),
        Some(one),
        Some(two),
        Some(pair),
    ] {
        assert!(
            !probe_proves_rank_zero(unproven),
            "{unproven:?} must not prove rank 0"
        );
    }
}

/// The probe model is the input model plus one default-domain `Shape` per
/// candidate, each exposed as a graph output under a name the model does not
/// already use (here the model itself occupies the first probe name).
#[test]
fn rank_probe_model_adds_one_fresh_shape_output_per_candidate() {
    let occupied = format!("{RANK_PROBE_PREFIX}0");
    let graph = onnx_proto::GraphProto {
        name: "rank_probe_names".to_string(),
        input: vec![value_info("x", DT_FLOAT, Some(&[2, 3]))],
        output: vec![value_info(&occupied, DT_FLOAT, None)],
        node: vec![node(
            "relu",
            "Relu",
            &["x"],
            &[occupied.as_str()],
            Vec::new(),
        )],
        ..Default::default()
    };
    let candidates: BTreeSet<String> = ["x".to_string(), occupied.clone()].into_iter().collect();
    let probe =
        build_rank_probe_model(&model_bytes(graph, 13), &candidates).expect("probe model builds");
    let decoded =
        onnx_proto::ModelProto::decode(probe.bytes.as_slice()).expect("probe model decodes");
    let graph = decoded.graph.expect("probe model has a graph");

    assert_eq!(graph.node.len(), 1 + candidates.len());
    assert_eq!(
        graph.node[0].op_type, "Relu",
        "the original node comes first"
    );
    assert_eq!(graph.node[0].output, vec![occupied.clone()]);
    assert_eq!(probe.candidate_by_output.len(), candidates.len());

    let mut measured = BTreeSet::new();
    for probe_node in &graph.node[1..] {
        assert_eq!(probe_node.op_type, "Shape");
        assert!(probe_node.domain.is_empty(), "standard domain only");
        assert!(
            probe_node.attribute.is_empty(),
            "no start/end: the full shape"
        );
        assert_eq!(probe_node.input.len(), 1);
        assert_eq!(probe_node.output.len(), 1);
        let output = &probe_node.output[0];
        assert!(output.starts_with(RANK_PROBE_PREFIX));
        assert_ne!(
            output, &occupied,
            "a probe output must not reuse a model name"
        );
        assert_eq!(
            probe.candidate_by_output.get(output),
            Some(&probe_node.input[0])
        );
        assert!(
            graph.output.iter().any(|info| &info.name == output),
            "probe output '{output}' must be a graph output"
        );
        measured.insert(probe_node.input[0].clone());
    }
    assert_eq!(measured, candidates, "exactly one probe per candidate");

    let mut produced: Vec<&String> = graph.node.iter().flat_map(|n| n.output.iter()).collect();
    let total = produced.len();
    produced.sort();
    produced.dedup();
    assert_eq!(produced.len(), total, "every produced name is unique");
}

/// The measured corpus case (audit 2026-09-28): cctsdb `patch-1` tensor `362`
/// and `patch-3` tensor `onnx::ScatterND_338` are the ScatterND updates of a
/// runtime-computed Reshape, rank 4 at runtime (`[1,1,x,64]`), and ORT 1.28
/// reports them with zero dims. They must be absent from the table, while a
/// proven scalar of the same model stays `[]` and the ScatterND output that
/// consumes them keeps its full shape.
#[cfg(feature = "external-vnncomp")]
#[test]
fn cctsdb_unknown_rank_tensors_are_not_recorded_as_scalars() {
    use super::static_infer_corpus_tests::{
        benchmarks_root, candidate_paths, read_model, sha256_hex,
    };

    struct Case {
        relative: [&'static str; 2],
        sha256: &'static str,
        untyped: &'static str,
        scalar: &'static str,
        typed: &'static str,
        typed_dims: &'static [i64],
    }
    let cases = [
        Case {
            relative: [
                "vnncomp2026/benchmarks/cctsdb_yolo_2023/1.0/onnx/patch-1.onnx",
                "vnncomp2025/benchmarks/cctsdb_yolo_2023/onnx/patch-1.onnx",
            ],
            sha256: "9c39db6eadd86b753d6ff7c6e8107ea1c63ff78c869c617a395e1407668454be",
            untyped: "362",
            scalar: "235",
            typed: "363",
            typed_dims: &[1, 3, 64, 64],
        },
        Case {
            relative: [
                "vnncomp2026/benchmarks/cctsdb_yolo_2023/1.0/onnx/patch-3.onnx",
                "vnncomp2025/benchmarks/cctsdb_yolo_2023/onnx/patch-3.onnx",
            ],
            sha256: "a5696e216de28f1434a6fa563924029c1e1cabf10f992e28a9fd660754f405fa",
            untyped: "onnx::ScatterND_338",
            scalar: "onnx::Cast_235",
            typed: "onnx::Mul_339",
            typed_dims: &[1, 3, 64, 64],
        },
    ];

    let root = benchmarks_root();
    assert!(
        root.is_dir(),
        "VNN-COMP benchmark fixture root is missing at {} (benchmarks/ is gitignored and lives \
         only in the main checkout; run benchmarks/download_benchmarks.sh or link it)",
        root.display()
    );
    for case in &cases {
        let path = case
            .relative
            .iter()
            .flat_map(|relative| candidate_paths(&root, relative))
            .find(|candidate| candidate.is_file())
            .unwrap_or_else(|| panic!("cctsdb model absent: {:?}", case.relative));
        let bytes = read_model(&path);
        assert_eq!(
            sha256_hex(&bytes),
            case.sha256,
            "{} changed under this test",
            path.display()
        );
        let shapes = infer_tensor_shapes_from_ort(&bytes)
            .unwrap_or_else(|err| panic!("{}: ORT shape inference: {err}", path.display()));
        assert_eq!(
            shapes.get(case.typed).map(Vec::as_slice),
            Some(case.typed_dims),
            "{}: control '{}' must keep its ORT shape",
            path.display(),
            case.typed
        );
        assert_eq!(
            shapes.get(case.scalar).map(Vec::as_slice),
            Some([].as_slice()),
            "{}: proven scalar '{}' must stay recorded as []",
            path.display(),
            case.scalar
        );
        assert_eq!(
            shapes.get(case.untyped),
            None,
            "{}: '{}' is rank 4 at runtime and ORT cannot type it; it must not be recorded \
             (least of all as the scalar [])",
            path.display(),
            case.untyped
        );
    }
}
