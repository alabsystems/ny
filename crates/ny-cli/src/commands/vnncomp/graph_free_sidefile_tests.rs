// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #graph-free-candidates: the side file is dark by default, decodes to the
//! exact tensor that ran, is bounded, and NEVER changes the lane's verdict.

use super::*;
use std::fs;
use std::time::Duration;

use super::super::{
    graph_free_falsification_applies, parse_witness_inputs_f64, property_violated_f64,
    try_graph_free_ort_falsify_observed,
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Independent RFC 4648 decoder (NOT the encoder under test run backwards).
fn b64_decode(text: &str) -> Vec<u8> {
    fn sextet(c: u8) -> u32 {
        match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            b'/' => 63,
            other => panic!("not a base64 character: {other}"),
        }
    }
    let bytes = text.as_bytes();
    assert_eq!(
        bytes.len() % 4,
        0,
        "padded base64 has a length divisible by 4"
    );
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for quad in bytes.chunks(4) {
        let pad: usize = quad.iter().map(|&c| usize::from(c == b'=')).sum();
        let mut acc = 0u32;
        for &c in &quad[..4 - pad] {
            acc = (acc << 6) | sextet(c);
        }
        acc <<= 6 * pad as u32;
        let triple = [(acc >> 16) as u8, (acc >> 8) as u8, acc as u8];
        out.extend_from_slice(&triple[..3 - pad]);
    }
    out
}

fn read_records(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .expect("side file exists")
        .lines()
        .map(|line| serde_json::from_str(line).expect("every side-file line is JSON"))
        .collect()
}

fn kinds(records: &[Value]) -> Vec<String> {
    records
        .iter()
        .map(|r| r["kind"].as_str().expect("kind").to_string())
        .collect()
}

/// Decode a record's `x` payload, checking length and sha256 on the way.
fn decode_x(record: &Value) -> Vec<f32> {
    let x = &record["x"];
    assert_eq!(x["encoding"], X_ENCODING);
    let bytes = b64_decode(x["data"].as_str().expect("x.data"));
    assert_eq!(
        x["sha256"].as_str().expect("x.sha256"),
        format!("{:x}", Sha256::digest(&bytes)),
        "the recorded sha256 must be the hash of exactly the decoded payload"
    );
    let len = usize::try_from(x["len"].as_u64().expect("x.len")).expect("len");
    assert_eq!(bytes.len(), len * 4);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// A two-input spec whose unsafe region is `Y_0 >= 0.6`.
fn two_input_spec(dir: &Path) -> (PathBuf, VnnLibSpec) {
    let vnnlib = dir.join("two_input.vnnlib");
    fs::write(
        &vnnlib,
        "(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n\
         (assert (>= X_0 0.0))\n(assert (<= X_0 1.0))\n\
         (assert (>= X_1 0.0))\n(assert (<= X_1 1.0))\n\
         (assert (>= Y_0 0.6))\n",
    )
    .expect("write vnnlib");
    let spec = ny_onnx::vnnlib::load_vnnlib(&vnnlib).expect("spec parses");
    (vnnlib, spec)
}

/// A load-refused model (the runtime-INT64 `Equal` ny's loader refuses, as in
/// the existing graph-free fixtures) computing `y = Sigmoid(logit)` where
/// `logit` is produced by `logit_nodes` from input `x` (`[1, n]`).
fn write_refused_model(
    dir: &Path,
    name: &str,
    n: i64,
    logit_nodes: Vec<ny_onnx::onnx_proto::NodeProto>,
    initializers: Vec<ny_onnx::onnx_proto::TensorProto>,
) -> PathBuf {
    use ny_onnx::onnx_proto::{
        tensor_shape_proto, AttributeProto, GraphProto, ModelProto, OperatorSetIdProto,
        TensorProto, TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
    };
    use prost::Message;

    let value_info = |name: &str, dims: &[i64], elem: i32| ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: elem,
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|&d| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                        })
                        .collect(),
                }),
            }),
        }),
    };
    let cast_to = |to: i64| AttributeProto {
        name: "to".to_string(),
        r#type: 2, // INT
        i: Some(to),
        ..AttributeProto::default()
    };
    let mut nodes = logit_nodes;
    nodes.push(node("Cast", &["logit"], "logit_i64", vec![cast_to(7)]));
    nodes.push(node("Equal", &["logit_i64", "c999"], "eq_b", vec![]));
    nodes.push(node("Cast", &["eq_b"], "eq_f", vec![cast_to(1)]));
    nodes.push(node("Add", &["logit", "eq_f"], "gated", vec![]));
    nodes.push(node("Sigmoid", &["gated"], "y", vec![]));
    let mut initializer = initializers;
    initializer.push(TensorProto {
        dims: vec![1, 1],
        data_type: 7, // INT64
        name: "c999".to_string(),
        int64_data: vec![999],
        ..TensorProto::default()
    });
    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            node: nodes,
            name: name.to_string(),
            initializer,
            input: vec![value_info("x", &[1, n], 1)],
            output: vec![value_info("y", &[1, 1], 1)],
            ..GraphProto::default()
        }),
        ..ModelProto::default()
    };
    let onnx = dir.join(format!("{name}.onnx"));
    fs::write(&onnx, model.encode_to_vec()).expect("write onnx");
    onnx
}

fn node(
    op: &str,
    inputs: &[&str],
    output: &str,
    attrs: Vec<ny_onnx::onnx_proto::AttributeProto>,
) -> ny_onnx::onnx_proto::NodeProto {
    ny_onnx::onnx_proto::NodeProto {
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: vec![output.to_string()],
        name: format!("{op}_{output}"),
        op_type: op.to_string(),
        domain: String::new(),
        attribute: attrs,
    }
}

fn f32_tensor(name: &str, dims: &[i64], values: Vec<f32>) -> ny_onnx::onnx_proto::TensorProto {
    ny_onnx::onnx_proto::TensorProto {
        dims: dims.to_vec(),
        data_type: 1, // FLOAT
        name: name.to_string(),
        float_data: values,
        ..ny_onnx::onnx_proto::TensorProto::default()
    }
}

fn unit_box_spec(dir: &Path, name: &str, n: usize) -> PathBuf {
    let vnnlib = dir.join(format!("{name}.vnnlib"));
    let mut text = String::new();
    for i in 0..n {
        text.push_str(&format!("(declare-const X_{i} Real)\n"));
    }
    text.push_str("(declare-const Y_0 Real)\n");
    for i in 0..n {
        text.push_str(&format!(
            "(assert (>= X_{i} 0.0))\n(assert (<= X_{i} 1.0))\n"
        ));
    }
    text.push_str("(assert (>= Y_0 0.6))\n");
    fs::write(&vnnlib, text).expect("write vnnlib");
    vnnlib
}

// ---------------------------------------------------------------------------
// encoding
// ---------------------------------------------------------------------------

#[test]
fn base64_encoder_matches_the_rfc4648_test_vectors() {
    for (plain, encoded) in [
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foob", "Zm9vYg=="),
        ("fooba", "Zm9vYmE="),
        ("foobar", "Zm9vYmFy"),
    ] {
        assert_eq!(
            base64_encode(plain.as_bytes()),
            encoded,
            "RFC 4648 vector {plain:?}"
        );
    }
    // Every byte value, every tail length, through the independent decoder.
    let all: Vec<u8> = (0..=255u8).chain((0..=255u8).rev()).collect();
    for len in 0..all.len() {
        assert_eq!(b64_decode(&base64_encode(&all[..len])), &all[..len]);
    }
}

#[test]
fn sidefile_path_appends_rather_than_replacing_the_extension() {
    assert_eq!(
        sidefile_path(Path::new("/tmp/result.txt")),
        Path::new("/tmp/result.txt.graph_free_candidates.jsonl"),
        "with_extension would map result.txt and result.csv to the same side file"
    );
}

// ---------------------------------------------------------------------------
// dark by default
// ---------------------------------------------------------------------------

#[test]
fn graph_free_sidefile_is_dark_by_default() {
    let decl = &ny_levers::decls::dark_probes::GRAPH_FREE_CANDIDATE_SIDEFILE;
    assert_eq!(decl.default, ny_levers::DefaultSpec::Bool(false));
    ny_test_utils::env::with_env_edits(|env| {
        env.remove(decl.name);
        assert!(
            CandidateSidefile::from_gates().is_none(),
            "with no preset key and no env override the lane must construct NO recorder"
        );
    });

    let default = crate::preset::PresetConfig::default();
    assert_eq!(default.attack.graph_free_candidate_sidefile, None);
    assert_eq!(
        preset_config_value(&default, decl),
        Some(None),
        "an unset key projects the declaration default into the flight receipt"
    );
    assert_eq!(
        preset_config_value(
            &default,
            &ny_levers::decls::dark_probes::FLAT_SURROGATE_TRANSFER
        ),
        None,
        "the projection claims only its own declaration"
    );
    let mut armed = crate::preset::PresetConfig::default();
    armed.attack.graph_free_candidate_sidefile = Some(true);
    assert_eq!(
        preset_config_value(&armed, decl),
        Some(Some(ny_levers::LeverValue::Bool(true))),
        "an armed yaml must reach the receipt, or the run's evidence contradicts the run"
    );
}

// ---------------------------------------------------------------------------
// the recorder
// ---------------------------------------------------------------------------

#[test]
fn recorder_writes_records_that_decode_to_the_exact_tensor_and_outcome() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (vnnlib, spec) = two_input_spec(dir.path());
    let path = dir.path().join("r.txt.graph_free_candidates.jsonl");
    // A subnormal and a negative zero: the payload must be the BITS, not a
    // decimal rendering of them.
    let x = [0.25_f32, f32::from_bits(1)];
    let x_neg_zero = [-0.0_f32, 1.0];
    {
        let mut rec = CandidateSidefile::for_path(path.clone());
        rec.begin(Path::new("m.onnx"), &vnnlib, 2, &[Some(0.19607843), None]);
        rec.set_stage("center");
        rec.record_candidate(&spec, &x, &[0.7], None, Outcome::RejectedEmit);
        rec.record_candidate(
            &spec,
            &x,
            &[0.7],
            Some(&[0.5_f32][..]),
            Outcome::RejectedDualLevel,
        );
        rec.set_stage("block");
        rec.record_candidate(
            &spec,
            &x_neg_zero,
            &[0.7],
            Some(&[0.7_f32][..]),
            Outcome::RejectedCorroboration {
                radius: 0.02,
                neighbour_margin: -0.1,
                checked: 3,
            },
        );
        rec.record_candidate(
            &spec,
            &x,
            &[0.7],
            Some(&[0.7_f32][..]),
            Outcome::AcceptedByLane { corroborated: 12 },
        );
        let counts = rec.counts();
        assert_eq!(counts.seen, 4);
        assert_eq!(counts.accepted, 1);
        assert_eq!(counts.rejected_emit, 1);
        assert_eq!(counts.rejected_dual_level, 1);
        assert_eq!(counts.rejected_corroboration, 1);
        assert!(rec
            .receipt()
            .starts_with("#graph-free-candidates: v1 armed=1 source=explicit"));
    }
    let records = read_records(&path);
    assert_eq!(
        kinds(&records),
        [
            "header",
            "candidate",
            "candidate",
            "candidate",
            "candidate",
            "summary"
        ]
    );
    let header = &records[0];
    assert_eq!(header["schema"], SCHEMA);
    assert_eq!(header["n_inputs"], 2);
    assert_eq!(header["pinned_count"], 1);
    assert_eq!(header["pinned"][0][0], 0);
    assert_eq!(header["pinned"][0][1].as_f64(), Some(0.19607843));

    let outcomes: Vec<&str> = records[1..5]
        .iter()
        .map(|r| r["outcome"].as_str().expect("outcome"))
        .collect();
    assert_eq!(
        outcomes,
        [
            "rejected_emit",
            "rejected_dual_level",
            "rejected_corroboration",
            "accepted_by_lane"
        ]
    );
    for (record, expected) in records[1..5].iter().zip([x, x, x_neg_zero, x]) {
        let decoded = decode_x(record);
        assert_eq!(
            decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the payload must be the exact f32 bits that ran"
        );
        assert_eq!(
            record["default_outputs_f32_bits"][0],
            format!("{:08x}", 0.7_f32.to_bits())
        );
        let margin = record["default_margin"].as_f64().expect("margin");
        assert!((margin - (f64::from(0.7_f32) - 0.6)).abs() < 1e-12);
    }
    assert_eq!(records[1]["stage"], "center");
    assert_eq!(records[3]["stage"], "block");
    assert!(records[1].get("disable_outputs").is_none());
    let dual = &records[2];
    assert_eq!(
        dual["disable_outputs_f32_bits"][0],
        format!("{:08x}", 0.5_f32.to_bits())
    );
    assert!(dual["disable_margin"].as_f64().expect("disable margin") < 0.0);
    assert_eq!(records[3]["corroboration_radius"].as_f64(), Some(0.02));
    assert_eq!(records[3]["corroboration_checked"], 3);
    assert_eq!(records[4]["corroborated"], 12);
    let summary = &records[5];
    assert_eq!(summary["candidates"], 4);
    assert_eq!(summary["accepted"], 1);
    assert_eq!(summary["accepted_id"], 3);
    assert_eq!(summary["candidate_records"], 4);
}

#[test]
fn candidate_records_are_capped_but_every_outcome_is_counted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (vnnlib, spec) = two_input_spec(dir.path());
    let path = dir.path().join("cap.jsonl");
    let extra = 44_u64;
    {
        let mut rec = CandidateSidefile::for_path(path.clone());
        rec.begin(Path::new("m.onnx"), &vnnlib, 2, &[None, None]);
        for _ in 0..(MAX_CANDIDATE_RECORDS + extra) {
            rec.record_candidate(&spec, &[0.5, 0.5], &[0.7], None, Outcome::RejectedEmit);
        }
        assert_eq!(rec.counts().seen, MAX_CANDIDATE_RECORDS + extra);
        assert_eq!(rec.candidate_records, MAX_CANDIDATE_RECORDS);
        assert_eq!(rec.dropped_cap, extra);
    }
    let records = read_records(&path);
    let summary = records.last().expect("summary");
    assert_eq!(summary["kind"], "summary");
    assert_eq!(summary["candidates"], MAX_CANDIDATE_RECORDS + extra);
    assert_eq!(summary["candidate_records"], MAX_CANDIDATE_RECORDS);
    assert_eq!(summary["dropped_cap"], extra);
}

#[test]
fn near_miss_sampling_is_bounded_and_flushes_the_final_incumbent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (vnnlib, spec) = two_input_spec(dir.path());

    // 70_000 non-violating incumbents: power-of-two indices 1..=32768 are
    // written (16 = the cap), 65536 hits the cap, and the final incumbent is
    // flushed at search end as the one extra record the cap allows.
    let path = dir.path().join("nm.jsonl");
    {
        let mut rec = CandidateSidefile::for_path(path.clone());
        rec.begin(Path::new("m.onnx"), &vnnlib, 2, &[None, None]);
        for k in 1..=70_000_u32 {
            let x = [k as f32, 0.0];
            rec.observe_incumbent(&spec, &x, &[0.5], -0.1, f64::from(k), false);
        }
        rec.observe_incumbent(&spec, &[], &[], f64::NAN, 70_000.0, true);
    }
    let records = read_records(&path);
    let near: Vec<&Value> = records
        .iter()
        .filter(|r| r["kind"] == "near_miss")
        .collect();
    assert_eq!(near.len() as u64, MAX_NEAR_MISS_RECORDS + 1);
    let improvements: Vec<u64> = near
        .iter()
        .map(|r| r["improvement"].as_u64().expect("improvement"))
        .collect();
    let mut expected: Vec<u64> = (0..16).map(|p| 1_u64 << p).collect();
    expected.push(70_000);
    assert_eq!(improvements, expected);
    let last = near.last().expect("final");
    assert_eq!(last["final_incumbent"], true);
    assert_eq!(decode_x(last)[0], 70_000.0);
    assert!(near[..16].iter().all(|r| r["final_incumbent"] == false));
    assert!(near.iter().all(|r| r["outcome"] == "not_offered"));

    // A VIOLATING incumbent is not a near miss: the acceptance stack records
    // it as a candidate, and it retires the stashed older incumbent.
    let path2 = dir.path().join("nm2.jsonl");
    {
        let mut rec = CandidateSidefile::for_path(path2.clone());
        rec.begin(Path::new("m.onnx"), &vnnlib, 2, &[None, None]);
        for k in 1..=3_u32 {
            rec.observe_incumbent(&spec, &[k as f32, 0.0], &[0.5], -0.1, f64::from(k), false);
        }
        rec.observe_incumbent(&spec, &[4.0, 0.0], &[0.7], 0.1, 4.0, false);
        rec.observe_incumbent(&spec, &[], &[], f64::NAN, 4.0, true);
        assert_eq!(rec.improvements, 4);
    }
    let records2 = read_records(&path2);
    let near2: Vec<u64> = records2
        .iter()
        .filter(|r| r["kind"] == "near_miss")
        .map(|r| r["improvement"].as_u64().expect("improvement"))
        .collect();
    assert_eq!(
        near2,
        [1, 2],
        "index 3 was stashed, then retired by the violating incumbent"
    );
}

#[test]
fn a_recorder_without_a_target_records_nothing_and_still_reports() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (vnnlib, spec) = two_input_spec(dir.path());
    let mut rec = CandidateSidefile::new(None, "explicit");
    rec.begin(Path::new("m.onnx"), &vnnlib, 2, &[None, None]);
    rec.record_candidate(&spec, &[0.5, 0.5], &[0.7], None, Outcome::RejectedEmit);
    assert_eq!(rec.counts().seen, 1);
    let receipt = rec.receipt();
    assert!(receipt.contains(" status=no_target "), "{receipt}");
    assert!(receipt.contains(" path=none "), "{receipt}");
    assert!(receipt.contains(" records=0 "), "{receipt}");
    assert!(receipt.contains(" candidates=1 "), "{receipt}");
}

#[test]
fn trusted_oracle_outcome_is_appended_only_for_a_parked_candidate() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("oracle.jsonl");
    fs::write(&path, "").expect("create");

    // An explicit (test) recorder that ACCEPTED a candidate never parks it,
    // so it cannot leak into another run's funnel. (Asserted on the recorder
    // itself, not on the process-global slot, which concurrently running
    // tests share.)
    {
        let (vnnlib, spec) = two_input_spec(dir.path());
        let mut rec = CandidateSidefile::for_path(dir.path().join("explicit.jsonl"));
        rec.begin(Path::new("m.onnx"), &vnnlib, 2, &[None, None]);
        rec.record_candidate(
            &spec,
            &[0.5, 0.5],
            &[0.7],
            Some(&[0.7_f32][..]),
            Outcome::AcceptedByLane { corroborated: 12 },
        );
        assert!(!rec.park_oracle, "an explicit recorder must never park");
        assert_eq!(rec.accepted_id, Some(0));
        assert!(rec.accepted_recorded);
    }

    // Nothing parked for THIS file: the caller hook appends nothing to it.
    record_trusted_oracle_outcome(true, true);
    assert_eq!(
        fs::read_to_string(&path).expect("read"),
        "",
        "nothing parked: nothing appended"
    );

    // The append itself, exercised WITHOUT the process-global slot so a
    // concurrently running funnel test can never consume this test's record.
    let pending = PendingOracle {
        path: path.clone(),
        candidate_id: 7,
        candidate_recorded: true,
    };
    assert!(append_trusted_oracle(&pending, true, false));
    let records = read_records(&path);
    assert_eq!(kinds(&records), ["trusted_oracle"]);
    assert_eq!(records[0]["candidate"], 7);
    assert_eq!(records[0]["candidate_recorded"], true);
    assert_eq!(records[0]["consulted"], true);
    assert_eq!(records[0]["upheld"], false);
}

// ---------------------------------------------------------------------------
// the lane: the side file never changes a verdict
// ---------------------------------------------------------------------------

/// End to end on a load-refused model whose WHOLE box violates (`logit =
/// sum(x) + 1 >= 1`, so `Y >= 0.73 >= 0.6` everywhere): the dark and the
/// armed lane must return the byte-identical witness, and the side file must
/// hold exactly that point, as the accepted candidate, decoding to the
/// witness the lane emitted and replaying as a violation on a fresh ONNX
/// Runtime session.
#[test]
fn armed_sidefile_never_changes_the_lane_verdict_on_a_violating_row() {
    let dir = tempfile::tempdir().expect("tempdir");
    let onnx = write_refused_model(
        dir.path(),
        "whole_box_violating",
        4,
        vec![node("Gemm", &["x", "w", "b"], "logit", vec![])],
        vec![
            f32_tensor("w", &[4, 1], vec![1.0; 4]),
            f32_tensor("b", &[1], vec![1.0]),
        ],
    );
    assert!(
        graph_free_falsification_applies(&onnx),
        "the fixture must be load-refused, or this test is not exercising the lane"
    );
    let vnnlib = unit_box_spec(dir.path(), "whole_box_violating", 4);
    let deadline = || Some(Instant::now() + Duration::from_secs(60));

    let dark = try_graph_free_ort_falsify_observed(&onnx, &vnnlib, deadline(), None)
        .expect("the dark lane converts a whole-box violation");
    let side = dir.path().join("armed.txt.graph_free_candidates.jsonl");
    let armed = try_graph_free_ort_falsify_observed(
        &onnx,
        &vnnlib,
        deadline(),
        Some(CandidateSidefile::for_path(side.clone())),
    )
    .expect("the armed lane converts the same row");
    assert_eq!(
        dark, armed,
        "arming the side file must not change the witness by one byte"
    );

    let records = read_records(&side);
    assert_eq!(kinds(&records), ["header", "candidate", "summary"]);
    let candidate = &records[1];
    assert_eq!(candidate["outcome"], "accepted_by_lane");
    assert_eq!(candidate["stage"], "center");
    assert_eq!(records[0]["pinned_count"], 0);

    let x = decode_x(candidate);
    let witness_inputs = parse_witness_inputs_f64(&armed).expect("witness parses");
    assert_eq!(witness_inputs.len(), x.len());
    for (w, v) in witness_inputs.iter().zip(&x) {
        assert_eq!(
            w.to_bits(),
            f32_to_f64_exact(*v).to_bits(),
            "the recorded tensor must be exactly the emitted witness's f32 view"
        );
    }
    let spec = ny_onnx::vnnlib::load_vnnlib(&vnnlib).expect("spec parses");
    let mut replay = ny_onnx::diff::OrtForward::from_path(&onnx, x.len()).expect("ORT replay");
    let out = replay.run(&x).expect("ORT replay forward");
    let out64: Vec<f64> = out.iter().map(|&v| f32_to_f64_exact(v)).collect();
    assert!(property_violated_f64(&spec, &witness_inputs, &out64));
    assert_eq!(
        candidate["default_outputs_f32_bits"][0],
        format!("{:08x}", out[0].to_bits()),
        "the recorded outputs are this host's default-level ONNX Runtime outputs"
    );
    assert_eq!(records[2]["accepted"], 1);
}

/// End to end on a load-refused KNIFE-EDGE model (`logit = 1 - 1e6 (x0 -
/// 0.5)^2`, 9 free dims so the historical whole-span corroboration radii
/// apply): the centre violates on ny's forward but its neighbourhood does
/// not, so corroboration rejects every candidate. Dark and armed lanes both
/// publish NOTHING, and the armed side file records the rejections with the
/// gate that made them — the exact class the x86 harvest exists to replay.
#[test]
fn armed_sidefile_records_corroboration_rejections_and_the_lane_still_publishes_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut e0 = vec![0.0_f32; 9];
    e0[0] = 1.0;
    let onnx = write_refused_model(
        dir.path(),
        "knife_edge",
        9,
        vec![
            node("Gemm", &["x", "e0", "bneg"], "d", vec![]),
            node("Mul", &["d", "d"], "sq", vec![]),
            node("Mul", &["sq", "k"], "ksq", vec![]),
            node("Sub", &["c1", "ksq"], "logit", vec![]),
        ],
        vec![
            f32_tensor("e0", &[9, 1], e0),
            f32_tensor("bneg", &[1], vec![-0.5]),
            f32_tensor("k", &[1, 1], vec![1.0e6]),
            f32_tensor("c1", &[1, 1], vec![1.0]),
        ],
    );
    assert!(
        graph_free_falsification_applies(&onnx),
        "the fixture must be load-refused, or this test is not exercising the lane"
    );
    let vnnlib = unit_box_spec(dir.path(), "knife_edge", 9);
    let deadline = || Some(Instant::now() + Duration::from_secs(2));

    assert!(
        try_graph_free_ort_falsify_observed(&onnx, &vnnlib, deadline(), None).is_none(),
        "the dark lane must not publish an un-corroborated knife edge"
    );
    let side = dir.path().join("knife.txt.graph_free_candidates.jsonl");
    assert!(
        try_graph_free_ort_falsify_observed(
            &onnx,
            &vnnlib,
            deadline(),
            Some(CandidateSidefile::for_path(side.clone())),
        )
        .is_none(),
        "arming the side file must not make the lane publish anything"
    );

    let records = read_records(&side);
    let candidates: Vec<&Value> = records
        .iter()
        .filter(|r| r["kind"] == "candidate")
        .collect();
    assert!(
        !candidates.is_empty(),
        "the centre violates, so it must be recorded"
    );
    assert_eq!(candidates[0]["stage"], "center");
    assert_eq!(candidates[0]["outcome"], "rejected_corroboration");
    assert!(candidates.len() as u64 <= MAX_CANDIDATE_RECORDS);
    for candidate in &candidates {
        assert_ne!(candidate["outcome"], "accepted_by_lane");
        assert!(
            candidate["default_margin"].as_f64().expect("margin") >= 0.0,
            "only points ny's own forward calls violating reach the acceptance stack"
        );
        let x = decode_x(candidate);
        assert_eq!(x.len(), 9);
        assert!(
            x.iter().all(|v| (0.0..=1.0).contains(v)),
            "recorded point left its box"
        );
    }
    let summary = records.last().expect("summary");
    assert_eq!(summary["kind"], "summary");
    assert_eq!(summary["accepted"], 0);
    assert!(summary["rejected_corroboration"].as_u64().expect("count") >= 1);
}
