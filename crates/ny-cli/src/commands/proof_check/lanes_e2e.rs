// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! B2.2 END TO END: a synthetic ONNX file through the REAL loader, the REAL
//! lane (graph ReLU split, graph input split, multi-objective, lowdim) under
//! an armed lane proof capture, the REAL translation ([`relu_site_map`],
//! [`canonical_clauses`], `stream_lane_proofs`) and the REAL v2 writer, then
//! the independent checker (`ny proof-check`'s engine, through the parent
//! module's test faces — the one ny-cli source that may name the checker) on
//! the bundle: verified. Mutations of a real bundle (a dropped leaf, a flipped split
//! literal) are rejected; garbage β hints never make it wrong.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ndarray::{ArrayD, IxDyn};
use ny_onnx::onnx_proto::tensor_shape_proto::{dimension, Dimension};
use ny_onnx::onnx_proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto,
    TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_propagate::beta_crown::{BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier};
use ny_propagate::certify::{
    reserve_node_id, reset_lane_proof_capture, stream_lane_proofs, take_lane_proof_capture, AbsLit,
    AbsSite, LaneProofInstance, LegacyUnsatAuthority, NodeClose, NodeRecord, ProofSink,
    SiteTensorRecord, SitesRecord, UnsatSource,
};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use prost::Message;
use sha2::{Digest, Sha256};

use super::test_faces::{check, checker_disjuncts, Checked};
use crate::commands::beta_crown::constraint_plan::build_constraint_objective;
use crate::commands::graph_cover_publish::graph_instance_property;
use crate::commands::proof_bundle::{
    bundle_path, canonical_clauses, close_bundle, enable_hint_capture_for_test, relu_site_map,
    Artifact, BundleWriter, Completion, Header, VerdictInfo, FINALISE_WAIT,
};

/// Serializes the tests that arm the PROCESS lane capture.
static LANE_E2E_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn node(
    name: &str,
    op: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Vec<AttributeProto>,
) -> NodeProto {
    NodeProto {
        name: name.to_owned(),
        op_type: op.to_owned(),
        input: inputs.iter().map(|s| (*s).to_owned()).collect(),
        output: outputs.iter().map(|s| (*s).to_owned()).collect(),
        attribute: attrs,
        ..Default::default()
    }
}

fn int(name: &str, v: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_owned(),
        i: Some(v),
        r#type: 2,
        ..Default::default()
    }
}

fn f32_tensor(name: &str, dims: &[i64], v: &[f32]) -> TensorProto {
    TensorProto {
        name: name.to_owned(),
        dims: dims.to_vec(),
        data_type: 1,
        float_data: v.to_vec(),
        ..Default::default()
    }
}

fn vi(name: &str, dims: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_owned(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|&d| Dimension {
                            value: Some(dimension::Value::DimValue(d)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

/// `x [1,2] -> Gemm (z = [[1,1],[1,-1]] x) -> Relu -> Gemm -> y`. One output:
/// `y0 = 1 − relu(x0+x1) − relu(x0−x1)` (true minimum −1 on `[−1, 1]²`, root
/// CROWN −2). Two outputs add `y1 = 1 + relu(x0+x1)` (minimum 1). `named`:
/// ONNX node names set (else the loader names a node by its first output).
fn crossing_onnx(two_outputs: bool, named: bool) -> Vec<u8> {
    let n = |s: &str| if named { s.to_owned() } else { String::new() };
    let (w1, b1, outs): (Vec<f32>, Vec<f32>, i64) = if two_outputs {
        (vec![-1.0, -1.0, 1.0, 0.0], vec![1.0, 1.0], 2)
    } else {
        (vec![-1.0, -1.0], vec![1.0], 1)
    };
    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![ny_onnx::onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            node: vec![
                node(
                    &n("fc0"),
                    "Gemm",
                    &["x", "W0", "B0"],
                    &["z"],
                    vec![int("transB", 1)],
                ),
                node(&n("relu0"), "Relu", &["z"], &["h"], Vec::new()),
                node(
                    &n("fc1"),
                    "Gemm",
                    &["h", "W1", "B1"],
                    &["y"],
                    vec![int("transB", 1)],
                ),
            ],
            initializer: vec![
                f32_tensor("W0", &[2, 2], &[1.0, 1.0, 1.0, -1.0]),
                f32_tensor("B0", &[2], &[0.0, 0.0]),
                f32_tensor("W1", &[outs, 2], &w1),
                f32_tensor("B1", &[outs], &b1),
            ],
            input: vec![vi("x", &[1, 2])],
            output: vec![vi("y", &[1, outs])],
            ..Default::default()
        }),
        ..Default::default()
    };
    model.encode_to_vec()
}

const ONE_OUTPUT: &str = "(declare-const X_0 Real)\n(declare-const X_1 Real)\n\
    (declare-const Y_0 Real)\n\
    (assert (>= X_0 -1))\n(assert (<= X_0 1))\n(assert (>= X_1 -1))\n(assert (<= X_1 1))\n\
    (assert (<= Y_0 -1.25))\n";

const TWO_OUTPUTS: &str = "(declare-const X_0 Real)\n(declare-const X_1 Real)\n\
    (declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
    (assert (>= X_0 -1))\n(assert (<= X_0 1))\n(assert (>= X_1 -1))\n(assert (<= X_1 1))\n\
    (assert (or (and (<= Y_0 -1.25)) (and (<= Y_1 0.5))))\n";

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A value in `[-1, 1)` on a 1/1024 grid (exact in binary32).
    fn weight(&mut self) -> f32 {
        let k = (self.next() % 2048) as i32 - 1024;
        k as f32 / 1024.0
    }
}

/// An MLP as `(w [out][in], b [out])` layers.
type Layers = Vec<(Vec<Vec<f32>>, Vec<f32>)>;

/// A random MLP `2 -> widths... -> 1` on the 1/1024 grid (the graph ledger
/// family's generator).
fn random_layers(rng: &mut Rng, widths: &[usize]) -> Layers {
    let mut fan_in = 2usize;
    let mut out = Vec::new();
    for &n in widths.iter().chain(std::iter::once(&1usize)) {
        let w = (0..n)
            .map(|_| (0..fan_in).map(|_| rng.weight()).collect())
            .collect();
        let b = (0..n).map(|_| rng.weight() * 0.5).collect();
        out.push((w, b));
        fan_in = n;
    }
    out
}

/// The ONNX file of `layers`: `Gemm (transB) -> Relu -> ... -> Gemm`, nodes
/// named `fc<i>` / `relu<i>`.
fn layers_onnx(layers: &Layers) -> Vec<u8> {
    let mut nodes = Vec::new();
    let mut inits = Vec::new();
    let mut prev = "x".to_string();
    let last = layers.len() - 1;
    for (i, (w, b)) in layers.iter().enumerate() {
        let (o, k) = (w.len() as i64, w[0].len() as i64);
        let flat: Vec<f32> = w.iter().flatten().copied().collect();
        inits.push(f32_tensor(&format!("W{i}"), &[o, k], &flat));
        inits.push(f32_tensor(&format!("B{i}"), &[o], b));
        let out = if i == last {
            "y".to_string()
        } else {
            format!("z{i}")
        };
        nodes.push(node(
            &format!("fc{i}"),
            "Gemm",
            &[prev.as_str(), &format!("W{i}"), &format!("B{i}")],
            &[out.as_str()],
            vec![int("transB", 1)],
        ));
        if i < last {
            nodes.push(node(
                &format!("relu{i}"),
                "Relu",
                &[out.as_str()],
                &[&format!("h{i}")],
                Vec::new(),
            ));
            prev = format!("h{i}");
        }
    }
    ModelProto {
        ir_version: 8,
        opset_import: vec![ny_onnx::onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            node: nodes,
            initializer: inits,
            input: vec![vi("x", &[1, 2])],
            output: vec![vi("y", &[1, 1])],
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec()
}

fn eval_layers(layers: &Layers, x: [f64; 2]) -> f64 {
    let mut h: Vec<f64> = x.to_vec();
    let last = layers.len() - 1;
    for (i, (w, b)) in layers.iter().enumerate() {
        let mut next: Vec<f64> = w
            .iter()
            .zip(b)
            .map(|(row, bias)| {
                row.iter()
                    .zip(&h)
                    .map(|(a, v)| f64::from(*a) * v)
                    .sum::<f64>()
                    + f64::from(*bias)
            })
            .collect();
        if i < last {
            for v in &mut next {
                *v = v.max(0.0);
            }
        }
        h = next;
    }
    h[0]
}

/// A threshold STRICTLY below the network's minimum over `[-1, 1]^2` (grid
/// minimum minus a Lipschitz bound on the grid error, minus `slack` times the
/// output range): the property `Y_0 <= t` is then UNSAT.
fn true_threshold(layers: &Layers, slack: f64) -> f32 {
    let n = 100;
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..=n {
        for j in 0..=n {
            let x = [
                -1.0 + 2.0 * f64::from(i) / f64::from(n),
                -1.0 + 2.0 * f64::from(j) / f64::from(n),
            ];
            let v = eval_layers(layers, x);
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let lipschitz: f64 = layers
        .iter()
        .map(|(w, _)| {
            w.iter()
                .map(|row| row.iter().map(|v| f64::from(v.abs())).sum::<f64>())
                .fold(0.0, f64::max)
        })
        .product();
    // A binary32 threshold at or below the bound (rounded down).
    let t = lo - lipschitz / f64::from(n) - slack * (hi - lo) - 1e-3;
    let f = t as f32;
    if f64::from(f) > t {
        f.next_down()
    } else {
        f
    }
}

/// The ReLU-split fixture: the first (net, slack) of the graph ledger
/// family's generator the ReLU-split lane verifies WITH A SPLIT TREE (its
/// tape streams at least two split-literal nodes). VOID — a panic — when
/// none does.
fn relu_split_fixture(dir: &Path, tag: &str, named: bool) -> Loaded {
    let mut rng = Rng(0x2F69_3A1C_7B45_9E0D);
    for net in 0..10 {
        let widths: &[usize] = if net % 2 == 0 { &[4] } else { &[3, 3] };
        let layers = random_layers(&mut rng, widths);
        for slack in [0.02, 0.1, 0.3] {
            let t = true_threshold(&layers, slack);
            let text = ONE_OUTPUT.replace("-1.25", &format!("{:?}", f64::from(t)));
            let mut bytes = layers_onnx(&layers);
            if !named {
                // Strip every node name: the loader then names a node by its
                // first output, and the site map must follow.
                let mut m = ModelProto::decode(bytes.as_slice()).expect("decode");
                if let Some(g) = m.graph.as_mut() {
                    for n in &mut g.node {
                        n.name.clear();
                    }
                }
                bytes = m.encode_to_vec();
            }
            let l = load(
                dir,
                &format!("{tag}_n{net}_s{}", (slack * 100.0) as u32),
                &bytes,
                &text,
            );
            // A split tree: the verified run's tape streams split literals.
            enable_hint_capture_for_test();
            reset_lane_proof_capture();
            let verified = relu_split(&l);
            let mut capture = take_lane_proof_capture();
            // This fixture's tapes only: the capture is process-global and other
            // tests of this binary may run lanes while it is armed (review(b2-lanes)).
            capture.retain_objectives(&l.own());
            if !verified {
                continue;
            }
            let spec = ny_onnx::vnnlib::parse_vnnlib(&l.text).expect("vnnlib");
            let property = graph_instance_property(&spec).expect("instance property");
            let inst = LaneProofInstance::new(&property, relu_site_map(&l.onnx));
            let mut sink = ny_propagate::certify::RecordingSink::default();
            let _ = stream_lane_proofs(&capture, &inst, &mut sink);
            let split = sink.nodes.iter().filter(|n| !n.lits.is_empty()).count();
            if split >= 2 {
                eprintln!(
                    "[proof-lanes-e2e] {tag}: net={net} slack={slack} t={t} split_nodes={split}"
                );
                return l;
            }
        }
    }
    panic!("{tag}: the ReLU-split lane verified no split tree in the family: the e2e pin is VOID");
}

/// One loaded instance: the lane's graph, its f32 input box (outward, batch
/// stripped — as the CLI builds it), the objective of every clause atom (the
/// same encoding the instance door uses) and the file paths.
struct Loaded {
    graph: GraphNetwork,
    input: BoundedTensor,
    objectives: Vec<Vec<f32>>,
    thresholds: Vec<f32>,
    onnx: PathBuf,
    vnnlib: PathBuf,
    text: String,
}

impl Loaded {
    /// This instance's objectives, to pick its own tapes out of the
    /// process-global capture (review(b2-lanes)).
    fn own(&self) -> Vec<(&[f32], f32)> {
        self.objectives
            .iter()
            .map(Vec::as_slice)
            .zip(self.thresholds.iter().copied())
            .collect()
    }
}

fn load(dir: &Path, tag: &str, onnx_bytes: &[u8], vnnlib: &str) -> Loaded {
    let onnx = dir.join(format!("{tag}.onnx"));
    let vpath = dir.join(format!("{tag}.vnnlib"));
    std::fs::write(&onnx, onnx_bytes).expect("onnx");
    std::fs::write(&vpath, vnnlib).expect("vnnlib");
    let model = ny_onnx::load_onnx(&onnx).expect("the loader reads the fixture");
    let graph = model.to_graph_network().expect("graph network");
    let spec = ny_onnx::vnnlib::parse_vnnlib(vnnlib).expect("vnnlib parses");
    let (lo, hi) = spec.split_input_bounds_f32();
    let n = lo.len();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), lo).expect("shape"),
        ArrayD::from_shape_vec(IxDyn(&[n]), hi).expect("shape"),
    )
    .expect("box");
    let groups: Vec<_> = if spec.is_disjunction {
        spec.output_constraint_clauses.clone()
    } else {
        vec![spec.output_constraints.clone()]
    };
    let mut objectives = Vec::new();
    let mut thresholds = Vec::new();
    for g in &groups {
        for c in g {
            let o = build_constraint_objective(c, spec.num_outputs).expect("objective");
            objectives.push(o.spec_coeffs().to_vec());
            thresholds.push(o.threshold());
        }
    }
    Loaded {
        graph,
        input,
        objectives,
        thresholds,
        onnx,
        vnnlib: vpath,
        text: vnnlib.to_owned(),
    }
}

/// The Proof header of a loaded instance.
fn header_of(l: &Loaded) -> Header {
    Header {
        profile: ny_core::proof_profile::ResolvedProofProfile::resolve(
            ny_core::proof_profile::EntryPoint::VnncompV1,
            Some(ny_core::proof_profile::ProofProfile::Proof),
        ),
        onnx: l.onnx.clone(),
        vnnlib: l.vnnlib.clone(),
    }
}

/// Run `lane` under an armed capture, then close the bundle through the REAL
/// publication close (`close_bundle`: the lane tapes' translation and the
/// writer's completion, exactly as `after_publication` runs them) and check
/// it. Returns the check outcome and the bundle.
fn lane_bundle_check(
    dir: &Path,
    tag: &str,
    l: &Loaded,
    source: UnsatSource,
    lane: impl FnOnce(&Loaded) -> bool,
) -> (Checked, PathBuf) {
    enable_hint_capture_for_test();
    reset_lane_proof_capture();
    let verified = lane(l);
    let mut capture = take_lane_proof_capture();
    // This lane's tapes only (the capture is process-global; review(b2-lanes)).
    capture.retain_objectives(&l.own());
    assert!(verified, "{tag}: the lane must verify the fixture");
    assert!(!capture.is_empty(), "{tag}: the lane deposited no tape");
    let results = dir.join(format!("{tag}.txt"));
    let (writer, _sink) = BundleWriter::create(&results, header_of(l)).expect("bundle");
    let artifact = close_bundle(
        writer,
        "unsat",
        Some((Vec::new(), f64::NAN, source.into())),
        Some(capture),
        false,
        Some(&(l.onnx.clone(), l.vnnlib.clone())),
    );
    assert!(
        artifact.render().starts_with("written"),
        "{tag}: {}",
        artifact.render()
    );
    let path = bundle_path(&results);
    (check(&path), path)
}

fn config() -> BetaCrownConfig {
    BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 4000,
        max_depth: 40,
        batch_size: 1,
        timeout: Duration::from_secs(30),
        ..Default::default()
    }
}

fn relu_split(l: &Loaded) -> bool {
    matches!(
        BetaCrownVerifier::new(config())
            .verify_graph_relu_split(&l.graph, &l.input, &l.objectives[0], l.thresholds[0])
            .expect("runs")
            .result,
        BabVerificationStatus::Verified
    )
}

fn assert_verified(o: &Checked, tag: &str) {
    assert_eq!(
        o.verdict, "verified",
        "{tag}: the checker disagrees with the lane: {}",
        o.report_json
    );
    eprintln!("[proof-lanes-e2e] {tag} {}", o.done_line);
}

/// The graph ReLU-split driver's tree: absolute Relu-input literals with the
/// lane's β, per-leaf certificates the checker re-derives, a split cover.
#[test]
fn lane_proofs_end_to_end_relu_split_bundle_is_verified() {
    let _g = LANE_E2E_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    for named in [true, false] {
        let tag = if named { "relu_named" } else { "relu_unnamed" };
        let l = relu_split_fixture(dir.path(), tag, named);
        let (o, path) = lane_bundle_check(dir.path(), tag, &l, UnsatSource::Authorized, relu_split);
        assert_verified(&o, tag);
        let text = std::fs::read_to_string(&path).expect("bundle");
        assert!(
            text.contains("\"name\":\"z"),
            "{tag}: the Relu's ONNX input names the site: {text}"
        );
        let split_nodes = text
            .lines()
            .filter(|l| l.contains("\"kind\":\"node\"") && l.contains("\"lits\":[["))
            .count();
        assert!(split_nodes >= 2, "{tag}: a split tree: {text}");
    }
}

/// The graph input-split driver's boxes (and clip slabs) as input-threshold
/// literals, covered geometrically by the checker.
#[test]
fn lane_proofs_end_to_end_input_split_bundle_is_verified() {
    let _g = LANE_E2E_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    let l = load(
        dir.path(),
        "input_split",
        &crossing_onnx(false, true),
        ONE_OUTPUT,
    );
    let (o, path) = lane_bundle_check(
        dir.path(),
        "input_split",
        &l,
        UnsatSource::Authorized,
        |l| {
            matches!(
                BetaCrownVerifier::new(config())
                    .verify_graph_input_split(&l.graph, &l.input, &l.objectives[0], l.thresholds[0])
                    .expect("runs")
                    .result,
                BabVerificationStatus::Verified
            )
        },
    );
    assert_verified(&o, "input_split");
    let text = std::fs::read_to_string(&path).expect("bundle");
    assert!(text.contains("[\"x\","), "box leaves: {text}");
    let r = ny_cert::json::from_str(&o.report_json).expect("report");
    assert_eq!(r["cover"][0]["kind"].as_str(), Some("box"));
    // review(b2-lanes): a box leaf streams no β, so the checker must count
    // no invalid hint (a `[]` against its `x` literals was one per node).
    assert_eq!(
        r["nodes"]["hints_invalid"].as_u64(),
        Some(0),
        "{}",
        o.report_json
    );
}

/// The multi-objective (disjunctive) driver: its domains record their own
/// closes; a legacy-door token with the lane's stream is still checked.
#[test]
fn lane_proofs_end_to_end_multi_objective_bundle_is_verified() {
    let _g = LANE_E2E_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    let l = load(dir.path(), "mo", &crossing_onnx(true, true), TWO_OUTPUTS);
    assert_eq!(l.objectives.len(), 2);
    let (o, _) = lane_bundle_check(
        dir.path(),
        "mo",
        &l,
        UnsatSource::LegacyUncovered(LegacyUnsatAuthority::BetaCrownVerdict),
        |l| {
            matches!(
                BetaCrownVerifier::new(config())
                    .verify_graph_relu_split_multi_objective(
                        &l.graph,
                        &l.input,
                        &l.objectives,
                        &l.thresholds,
                    )
                    .expect("runs")
                    .result,
                BabVerificationStatus::Verified
            )
        },
    );
    assert_verified(&o, "multi_objective");
}

/// The lowdim tree's leaves and clip slabs over the full input.
#[test]
fn lane_proofs_end_to_end_lowdim_bundle_is_verified() {
    let _g = LANE_E2E_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    let l = load(
        dir.path(),
        "lowdim",
        &crossing_onnx(false, true),
        ONE_OUTPUT,
    );
    let (o, _) = lane_bundle_check(
        dir.path(),
        "lowdim",
        &l,
        UnsatSource::LegacyUncovered(LegacyUnsatAuthority::BetaCrownVerdict),
        |l| {
            matches!(
                ny_propagate::lowdim::prove_conjunction_infeasible(
                    &l.graph,
                    &l.input,
                    &l.objectives,
                    &l.thresholds,
                    Instant::now() + Duration::from_secs(30),
                    &ny_propagate::lowdim::LowDimConfig::default(),
                ),
                ny_propagate::lowdim::LowDimOutcome::Verified(_)
            )
        },
    );
    assert_verified(&o, "lowdim");
    let r = ny_cert::json::from_str(&o.report_json).expect("report");
    assert_eq!(
        r["nodes"]["hints_invalid"].as_u64(),
        Some(0),
        "box leaves stream no β hint to be invalid: {}",
        o.report_json
    );
}

/// Re-seal a bundle after a line edit (a valid `end` record), so the checker
/// judges the CONTENT of the mutation, not its integrity.
fn reseal(path: &Path, edit: impl FnOnce(&mut Vec<String>)) -> Vec<u8> {
    let text = std::fs::read_to_string(path).expect("bundle");
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    lines.pop();
    edit(&mut lines);
    let mut body = String::new();
    for l in &lines {
        body.push_str(l);
        body.push('\n');
    }
    let digest = format!("{:x}", Sha256::digest(body.as_bytes()));
    body.push_str(
        &serde_json::json!({"kind": "end", "lines": lines.len() + 1, "sha256": digest}).to_string(),
    );
    body.push('\n');
    body.into_bytes()
}

fn check_bytes(dir: &Path, name: &str, bytes: &[u8]) -> Checked {
    let p = dir.join(format!("{name}.proof.jsonl"));
    std::fs::write(&p, bytes).expect("write");
    check(&p)
}

/// Mutations of a REAL lane bundle: dropping a split leaf and flipping a
/// split literal each open a cover gap (`invalid`); garbage β hints can only
/// cost confirmations (never `invalid`, `refuted` or a checker fault).
#[test]
fn lane_proofs_end_to_end_mutations_are_rejected() {
    let _g = LANE_E2E_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    let l = relu_split_fixture(dir.path(), "mut", true);
    let (o, path) = lane_bundle_check(dir.path(), "mut", &l, UnsatSource::Authorized, relu_split);
    assert_verified(&o, "mut_base");
    let is_split_leaf = |line: &str| {
        line.contains("\"kind\":\"node\"")
            && line.contains("\"close\":\"bound\"")
            && line.contains("\"lits\":[[")
    };
    // Drop a split leaf.
    let dropped = reseal(&path, |lines| {
        let at = lines
            .iter()
            .position(|x| is_split_leaf(x))
            .expect("a split leaf");
        lines.remove(at);
    });
    let o = check_bytes(dir.path(), "dropped", &dropped);
    assert_eq!(o.verdict, "invalid", "{}", o.report_json);
    // Flip the side of a split leaf's DEEPEST literal: the leaf now repeats
    // its sibling's cell, and its own cell is uncovered.
    let flipped = reseal(&path, |lines| {
        let at = lines
            .iter()
            .position(|x| is_split_leaf(x))
            .expect("a split leaf");
        let mut v: serde_json::Value = serde_json::from_str(&lines[at]).expect("json");
        let lits = v["lits"].as_array_mut().expect("lits");
        let last = lits.last_mut().expect("a literal");
        let side = last[2].as_u64().expect("side");
        last[2] = serde_json::json!(1 - side);
        lines[at] = v.to_string();
    });
    let o = check_bytes(dir.path(), "flipped", &flipped);
    assert!(
        ["invalid", "unconfirmed"].contains(&o.verdict),
        "a flipped literal must not verify: {}",
        o.report_json
    );
    // Garbage β hints.
    let garbage = reseal(&path, |lines| {
        for line in lines.iter_mut().filter(|x| is_split_leaf(x)) {
            let mut v: serde_json::Value = serde_json::from_str(line).expect("json");
            let n = v["lits"].as_array().map_or(0, Vec::len);
            v["hint"]["beta"] = serde_json::json!(vec![1e30; n]);
            *line = v.to_string();
        }
    });
    let o = check_bytes(dir.path(), "garbage", &garbage);
    assert!(
        ["verified", "unconfirmed"].contains(&o.verdict),
        "hints can only cost confirmations: {}",
        o.report_json
    );
}

/// The producer's clause strings are exactly the checker's canonical form for
/// every constraint form ny's VNN-LIB reader produces, including decimals
/// whose f64 value is not the decimal (the FILE's decimal is rendered).
#[test]
fn canonical_clauses_match_the_checkers_own_parse() {
    let head = "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
                (declare-const Y_2 Real)\n(assert (>= X_0 -1))\n(assert (<= X_0 1))\n";
    for body in [
        "(assert (<= Y_0 -1.25))\n",
        "(assert (>= Y_0 3.991125645861615))\n",
        "(assert (< Y_1 0.1))\n(assert (> Y_2 -1e-3))\n",
        "(assert (<= Y_0 Y_1))\n(assert (>= Y_2 Y_1))\n",
        "(assert (or (and (>= Y_1 Y_0)) (and (>= Y_2 Y_0))))\n",
        "(assert (or (and (<= Y_0 0.5) (>= Y_1 2)) (and (< Y_2 Y_0))))\n",
    ] {
        let text = format!("{head}{body}");
        let spec = ny_onnx::vnnlib::parse_vnnlib(&text).expect("ny parses it");
        let ours = canonical_clauses(&spec, &text).expect("rendered");
        let theirs = checker_disjuncts(&text);
        assert_eq!(theirs.len(), 1, "{body}");
        let mut want: Vec<String> = theirs[0].clone();
        let mut got: Vec<String> = ours
            .iter()
            .map(|atoms| {
                let mut a = atoms.clone();
                a.sort();
                a.dedup();
                a.join("&")
            })
            .collect();
        want.sort();
        got.sort();
        assert_eq!(got, want, "{body}");
    }
}

/// Relu sites resolve by the ONNX node name, or — for an unnamed node — by
/// its first output (the loader's naming), to the node's `input[0]`.
#[test]
fn relu_site_map_names_relu_inputs() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (named, key) in [(true, "relu0"), (false, "h")] {
        let p = dir.path().join(format!("m{named}.onnx"));
        std::fs::write(&p, crossing_onnx(false, named)).expect("onnx");
        let m = relu_site_map(&p);
        assert_eq!(m.len(), 1, "{m:?}");
        assert_eq!(m.get(key).map(String::as_str), Some("z"), "{m:?}");
        // The loader names the graph's ReLU node the same way.
        let graph = ny_onnx::load_onnx(&p)
            .expect("load")
            .to_graph_network()
            .expect("graph");
        assert!(
            graph.contains_node(key),
            "graph nodes: {:?}",
            graph.node_names()
        );
    }
}

// ---------------------------------------------------------------------------
// Legacy well-formedness (certified-moat review, 2026-10-08)
// ---------------------------------------------------------------------------

/// The production legacy doors (the test fixture authority excluded).
fn production_authorities() -> Vec<LegacyUnsatAuthority> {
    LegacyUnsatAuthority::ALL
        .into_iter()
        .filter(|a| *a != LegacyUnsatAuthority::TestFixture)
        .collect()
}

/// The report's JSON.
fn report(o: &Checked) -> ny_cert::json::Json {
    ny_cert::json::from_str(&o.report_json).expect("report")
}

fn reason_codes(o: &Checked) -> Vec<String> {
    match &report(o)["reasons"] {
        ny_cert::json::Json::Arr(v) => v
            .iter()
            .filter_map(|x| x["code"].as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}

fn assert_unconfirmed(o: &Checked, tag: &str) {
    assert_eq!(
        o.verdict, "unconfirmed",
        "{tag}: a legacy bundle must be well-formed and unconfirmed: {}",
        o.report_json
    );
    assert!(
        o.done_line.starts_with("PROOF_CHECK_DONE rc=10 "),
        "{tag}: {}",
        o.done_line
    );
    eprintln!("[legacy-e2e] {tag} {}", o.done_line);
}

/// A bundle's lines with every `node` whose `src` is `src` removed, resealed.
fn without_src(path: &Path, src: &str, keep: usize) -> Vec<u8> {
    let needle = format!("\"src\":\"{src}\"");
    reseal(path, |lines| {
        let mut seen = 0usize;
        lines.retain(|x| {
            if x.contains("\"kind\":\"node\"") && x.contains(&needle) {
                seen += 1;
                seen <= keep
            } else {
                true
            }
        });
    })
}

/// THE REPRODUCED FINDING, closed: every production legacy door whose lane
/// streams NO proof (a lane with no tape — here the REAL input-split
/// multi-clause disjunctive driver, which verifies the two-clause instance
/// and deposits nothing; the sequential engine, MIP, GPU DomainList, a
/// root-only fast path and the non-graph doors are the same shape) closes
/// through the REAL publication close into a WELL-FORMED bundle — empty
/// `sites`, canonical `clauses`, one whole-region claim per clause naming
/// the authority — that the independent checker reports `unconfirmed`
/// (exit 10), where the pinned ENG14 binary's bundles were `malformed`
/// (`missing_sites`, exit 30). MUTATION: the claims are load-bearing —
/// dropping one is a cover gap, dropping all is `uncovered_source`, both
/// `invalid`.
#[test]
fn legacy_lanes_without_a_tape_close_well_formed_and_check_unconfirmed() {
    let _g = LANE_E2E_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    let two = load(dir.path(), "mc", &crossing_onnx(true, true), TWO_OUTPUTS);
    let one = load(dir.path(), "one", &crossing_onnx(false, true), ONE_OUTPUT);
    // The REAL tapeless lane.
    enable_hint_capture_for_test();
    reset_lane_proof_capture();
    let verified = matches!(
        BetaCrownVerifier::new(config())
            .verify_graph_input_split_multi_clause_disjunctive(
                &two.graph,
                &two.input,
                &two.objectives,
                &two.thresholds,
                &[1, 1],
                None,
                Some(Instant::now() + Duration::from_secs(30)),
            )
            .expect("runs")
            .result,
        BabVerificationStatus::Verified
    );
    let mut capture = take_lane_proof_capture();
    capture.retain_objectives(&two.own());
    assert!(
        verified,
        "the multi-clause disjunctive lane must verify the fixture (else VOID)"
    );
    assert!(
        capture.is_empty(),
        "the multi-clause disjunctive lane keeps no tape: this is the tapeless shape"
    );
    let mut capture = Some(capture);
    let mut last_two = None;
    for (l, n_clauses) in [(&two, 2usize), (&one, 1usize)] {
        for authority in production_authorities() {
            let tag = format!("{}_{n_clauses}", authority.token());
            let results = dir.path().join(format!("{tag}.txt"));
            let (writer, _sink) = BundleWriter::create(&results, header_of(l)).expect("bundle");
            let artifact = close_bundle(
                writer,
                "unsat",
                Some((
                    Vec::new(),
                    f64::NAN,
                    UnsatSource::LegacyUncovered(authority).into(),
                )),
                capture.take(),
                false,
                Some(&(l.onnx.clone(), l.vnnlib.clone())),
            );
            assert!(
                artifact
                    .render()
                    .ends_with(&format!("schema=ny-proof-bundle/v2 claims={n_clauses}")),
                "{tag}: {}",
                artifact.render()
            );
            let path = bundle_path(&results);
            let o = check(&path);
            assert_unconfirmed(&o, &tag);
            let r = report(&o);
            assert_eq!(
                r["verdict_source"].as_str(),
                Some(authority.token()),
                "{tag}"
            );
            assert_eq!(
                r["nodes"]["opaque"].as_u64(),
                Some(n_clauses as u64),
                "{tag}"
            );
            let ny_cert::json::Json::Arr(cover) = &r["cover"] else {
                panic!("{tag}: no cover rows: {}", o.report_json);
            };
            assert_eq!(cover.len(), n_clauses, "{tag}");
            assert!(
                cover
                    .iter()
                    .all(|c| c["claimed_ok"] == ny_cert::json::Json::Bool(true)),
                "{tag}: every clause is claimed: {}",
                o.report_json
            );
            let text = std::fs::read_to_string(&path).expect("bundle");
            assert!(
                text.contains(&format!("\"src\":\"legacy:{}\"", authority.token())),
                "{tag}: the claims name the authority: {text}"
            );
            if n_clauses == 2 {
                last_two = Some((path, authority));
            }
        }
    }
    let (path, authority) = last_two.expect("a two-clause bundle");
    let src = format!("legacy:{}", authority.token());
    let o = check_bytes(dir.path(), "drop_one", &without_src(&path, &src, 1));
    assert_eq!(o.verdict, "invalid", "{}", o.report_json);
    assert!(
        reason_codes(&o).contains(&"cover_gap".to_string()),
        "{}",
        o.report_json
    );
    let o = check_bytes(dir.path(), "drop_all", &without_src(&path, &src, 0));
    assert_eq!(o.verdict, "invalid", "{}", o.report_json);
    assert!(
        reason_codes(&o).contains(&"uncovered_source".to_string()),
        "{}",
        o.report_json
    );
}

/// THE PARTIAL-RECORD SHAPE (`challenging_certified_training_2026` idx5683
/// on the ENG14 pin: `unsupported` / `uncovered_source`, exit 31): the
/// margin-row lane named the sites and clauses and streamed part of a tree
/// (one side of a split, one clause), then the internal verifier's LEGACY
/// door published. The REAL close completes it with a whole-region claim per
/// clause (a partial stream of another lane is not the verdict's proof):
/// `unconfirmed`. MUTATION: closing the same stream as if it were the
/// token's own proof (`Completion::OwnStream`) leaves the partial clause a
/// cover gap — `invalid`, the shape the completion exists to prevent.
#[test]
fn partial_stream_of_another_lane_under_a_legacy_verdict_checks_unconfirmed() {
    let _g = LANE_E2E_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    let l = load(
        dir.path(),
        "partial",
        &crossing_onnx(true, true),
        TWO_OUTPUTS,
    );
    let spec = ny_onnx::vnnlib::parse_vnnlib(&l.text).expect("vnnlib");
    let list: Vec<(usize, Vec<String>)> = canonical_clauses(&spec, &l.text)
        .expect("canonical clauses")
        .into_iter()
        .enumerate()
        .collect();
    let mut outcomes = Vec::new();
    for own in [false, true] {
        let results = dir.path().join(format!("partial_{own}.txt"));
        let (writer, mut sink) = BundleWriter::create(&results, header_of(&l)).expect("bundle");
        sink.sites(&SitesRecord {
            tensors: vec![SiteTensorRecord {
                id: 0,
                name: "z".to_owned(),
                size: 2,
                op: 0,
            }],
        });
        writer.clauses_list(&list);
        sink.node(&NodeRecord {
            id: reserve_node_id(),
            region: 0,
            lits: vec![AbsLit {
                site: AbsSite { op: 0, index: 0 },
                positive: true,
            }],
            xlits: Vec::new(),
            close: NodeClose::Bound,
            src: "tree",
            dis: Some(vec![0]),
            beta: Vec::new(),
            alpha: None,
            lift: false,
            margin: f64::NAN,
            empty_site: None,
            fd: None,
        });
        let source = UnsatSource::LegacyUncovered(LegacyUnsatAuthority::BetaCrownVerdict).into();
        let artifact = if own {
            writer.finalise(
                "unsat",
                Some(VerdictInfo {
                    source,
                    min_margin: f64::NAN,
                    completion: Completion::OwnStream,
                }),
                FINALISE_WAIT,
            )
        } else {
            close_bundle(
                writer,
                "unsat",
                Some((Vec::new(), f64::NAN, source)),
                None,
                true,
                Some(&(l.onnx.clone(), l.vnnlib.clone())),
            )
        };
        assert!(matches!(artifact, Artifact::Written { .. }), "{artifact:?}");
        outcomes.push(check(&bundle_path(&results)));
    }
    assert_unconfirmed(&outcomes[0], "partial_legacy");
    assert_eq!(
        outcomes[1].verdict, "invalid",
        "the mutation (no whole-region completion) must be a cover gap: {}",
        outcomes[1].report_json
    );
    assert!(reason_codes(&outcomes[1]).contains(&"cover_gap".to_string()));
}

/// THE INPUT-HEAD SHAPE (safenlp, review(eng14)): the route names its clauses
/// and streams ONE whole-region `opaque` `input_head` claim, but no lane
/// names a site — on the ENG14 pin `malformed` (`missing_sites`). The REAL
/// close adds the empty `sites` record and no claim (the route's own claim
/// covers): `unconfirmed`. MUTATION: removing that `sites` record again is
/// the reproduced `missing_sites` (malformed).
#[test]
fn input_head_claim_without_sites_closes_well_formed_and_checks_unconfirmed() {
    let _g = LANE_E2E_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    let l = load(dir.path(), "head", &crossing_onnx(true, true), TWO_OUTPUTS);
    let spec = ny_onnx::vnnlib::parse_vnnlib(&l.text).expect("vnnlib");
    let list: Vec<(usize, Vec<String>)> = canonical_clauses(&spec, &l.text)
        .expect("canonical clauses")
        .into_iter()
        .enumerate()
        .collect();
    let results = dir.path().join("head.txt");
    let (writer, mut sink) = BundleWriter::create(&results, header_of(&l)).expect("bundle");
    writer.clauses_list(&list);
    sink.node(&NodeRecord {
        id: reserve_node_id(),
        region: 0,
        lits: Vec::new(),
        xlits: Vec::new(),
        close: NodeClose::Opaque,
        src: "input_head",
        dis: Some(vec![0, 1]),
        beta: Vec::new(),
        alpha: None,
        lift: false,
        margin: f64::NAN,
        empty_site: None,
        fd: None,
    });
    let artifact = close_bundle(
        writer,
        "unsat",
        Some((vec![0, 1], 0.5, UnsatSource::Authorized.into())),
        None,
        true,
        Some(&(l.onnx.clone(), l.vnnlib)),
    );
    assert!(
        matches!(
            artifact,
            Artifact::Written {
                nodes: 1,
                claims: 0,
                ..
            }
        ),
        "{artifact:?}"
    );
    let path = bundle_path(&results);
    let o = check(&path);
    assert_unconfirmed(&o, "input_head");
    let text = std::fs::read_to_string(&path).expect("bundle");
    assert!(
        text.contains("{\"kind\":\"sites\",\"tensors\":[]}"),
        "{text}"
    );
    let cut = reseal(&path, |lines| {
        lines.retain(|x| !x.contains("\"kind\":\"sites\""));
    });
    let o = check_bytes(dir.path(), "head_no_sites", &cut);
    assert_eq!(o.verdict, "malformed", "{}", o.report_json);
    assert!(reason_codes(&o).contains(&"missing_sites".to_string()));
}

/// THE v2.1 SHAPES: a property this module cannot render (per-clause input
/// boxes, which the graph instance door refuses — the nonlinear and
/// input-disjunction routes) and a VNN-LIB 2.0 dual-network row (relational /
/// isomorphic: the ONNX field is a tuple, not a file — on the ENG14 pin
/// `malformed`, `unknown_field` `error` in the header) close as
/// `ny-proof-bundle/v2.1` with the whole property as one `unavailable`
/// clause and one legacy claim: `unconfirmed`, never parsed past the claims.
/// MUTATION: dropping the claim is `invalid`.
#[test]
fn unrenderable_properties_close_as_v2_1_and_check_unconfirmed() {
    let _g = LANE_E2E_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    let onnx = dir.path().join("v21.onnx");
    std::fs::write(&onnx, crossing_onnx(false, true)).expect("onnx");
    let boxes = dir.path().join("boxes.vnnlib");
    std::fs::write(
        &boxes,
        "(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n\
         (assert (or (and (>= X_0 -1) (<= X_0 0) (>= X_1 -1) (<= X_1 1) (<= Y_0 -1.25)) \
         (and (>= X_0 0) (<= X_0 1) (>= X_1 -1) (<= X_1 1) (<= Y_0 -1.25))))\n",
    )
    .expect("vnnlib");
    let dual = dir.path().join("dual.vnnlib");
    std::fs::write(
        &dual,
        "(vnnlib-version <2.0>)\n(declare-network f\n  (declare-input X_f real [1, 2])\n  \
         (declare-output Y_f real [1, 1])\n)\n(declare-network g\n  (isomorphic-to f)\n  \
         (declare-input X_g real [1, 2])\n  (declare-output Y_g real [1, 1])\n)\n\
         (assert (and (<= X_f[0,0] 1.0) (>= X_f[0,0] -1.0)))\n\
         (assert (and (<= X_f[0,1] 1.0) (>= X_f[0,1] -1.0)))\n\
         (assert (== X_f[0,0] X_g[0,0]))\n(assert (== X_f[0,1] X_g[0,1]))\n\
         (assert (> Y_g[0,0] (+ Y_f[0,0] 0.05)))\n",
    )
    .expect("vnnlib");
    let tuple = PathBuf::from(format!(
        "[('f', '{}'), ('g', '{}')]",
        onnx.display(),
        onnx.display()
    ));
    for (tag, onnx_path, vnnlib, authority, codes) in [
        (
            "boxes",
            onnx,
            boxes,
            LegacyUnsatAuthority::NonlinearIntervalBab,
            &["clauses_unavailable"][..],
        ),
        (
            "dual",
            tuple,
            dual,
            LegacyUnsatAuthority::RelationalImplication,
            &["clauses_unavailable", "inputs_unavailable"][..],
        ),
    ] {
        let results = dir.path().join(format!("{tag}.txt"));
        let header = Header {
            profile: ny_core::proof_profile::ResolvedProofProfile::resolve(
                ny_core::proof_profile::EntryPoint::VnncompV1,
                Some(ny_core::proof_profile::ProofProfile::Proof),
            ),
            onnx: onnx_path.clone(),
            vnnlib: vnnlib.clone(),
        };
        let (writer, _sink) = BundleWriter::create(&results, header).expect("bundle");
        let artifact = close_bundle(
            writer,
            "unsat",
            Some((
                Vec::new(),
                f64::NAN,
                UnsatSource::LegacyUncovered(authority).into(),
            )),
            None,
            false,
            Some(&(onnx_path, vnnlib)),
        );
        assert!(
            artifact
                .render()
                .ends_with("schema=ny-proof-bundle/v2.1 claims=1"),
            "{tag}: {}",
            artifact.render()
        );
        let path = bundle_path(&results);
        let o = check(&path);
        assert_unconfirmed(&o, tag);
        let found = reason_codes(&o);
        for code in codes {
            assert!(
                found.contains(&(*code).to_owned()),
                "{tag}: {code}: {}",
                o.report_json
            );
        }
        let o = check_bytes(
            dir.path(),
            &format!("{tag}_dropped"),
            &without_src(&path, &format!("legacy:{}", authority.token()), 0),
        );
        assert_eq!(o.verdict, "invalid", "{tag}: {}", o.report_json);
    }
}

/// review(eng15): the ORIGINAL-HULL close (b7eb511c6, ROUTER_PROGRAM 10.12)
/// through the merged writer AND the independent checker, the pairing no
/// branch review saw. The route asks for the instance's canonical clauses
/// (the writer's own rendering, ids by position) and streams ONE
/// whole-region `opaque` `original_hull` node over the property's single
/// clause; the close adds only the empty `sites` record (no completion: the
/// stream is the token's own) and the verdict names `original_hull_replay`.
/// The checker reads it well-formed and `unconfirmed` — an opaque claim
/// certifies nothing, so it is never `verified` — and with the hull's node
/// dropped the verdict's clause is uncovered (`invalid`): the claim is what
/// keeps the bundle out of the cover-gap class.
#[cfg(feature = "mip")]
#[test]
fn original_hull_claim_closes_well_formed_and_checks_unconfirmed() {
    let _g = LANE_E2E_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().expect("tempdir");
    let l = load(dir.path(), "hull", &crossing_onnx(false, true), ONE_OUTPUT);
    let results = dir.path().join("hull.txt");
    let (writer, mut sink) = BundleWriter::create(&results, header_of(&l)).expect("bundle");
    writer.request_instance_clauses();
    sink.node(&NodeRecord {
        id: reserve_node_id(),
        region: 0,
        lits: Vec::new(),
        xlits: Vec::new(),
        close: NodeClose::Opaque,
        src: "original_hull",
        dis: Some(vec![0]),
        beta: Vec::new(),
        alpha: None,
        lift: false,
        margin: f64::NAN,
        empty_site: None,
        fd: None,
    });
    let artifact = writer.finalise(
        "unsat",
        Some(VerdictInfo {
            source: crate::commands::unsat_receipt::PublishedUnsatSource::OriginalHull,
            min_margin: 0.25,
            completion: Completion::OwnStream,
        }),
        FINALISE_WAIT,
    );
    assert!(
        matches!(
            artifact,
            Artifact::Written {
                nodes: 1,
                claims: 0,
                ..
            }
        ),
        "{artifact:?}"
    );
    let path = bundle_path(&results);
    let o = check(&path);
    assert_unconfirmed(&o, "original_hull");
    let o = check_bytes(
        dir.path(),
        "hull_dropped",
        &without_src(&path, "original_hull", 0),
    );
    assert_eq!(o.verdict, "invalid", "{}", o.report_json);
}
