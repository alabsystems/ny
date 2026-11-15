// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! B2.3 END TO END: the margin-row lane's remaining routes — the frozen-dual
//! tier (its prunes), the wave β-BaB (its per-class leaves), the classwise
//! class trees (`class_part` frames) and nested epoch subtrees (`subdomain`
//! frames under their parent's prefix) — each through the REAL lane driver
//! (`MarginRowBab::run` / `run_classwise` on the REAL loader's twin spec and
//! a certified-outward root), the REAL `authorize_frame` stream into the REAL
//! v2 writer, then the independent checker (`ny proof-check`'s engine,
//! through the parent's test faces): VERIFIED. Mutations of a real
//! frozen-dual bundle — a dropped prune, a flipped side, a multiplier moved
//! beyond rounding, a prune relabelled to another class — are rejected.
//!
//! Every route finds its instance in a width sweep and fails VOID (never
//! passes vacuously) when no width exercises it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_onnx::onnx_proto::tensor_shape_proto::{dimension, Dimension};
use ny_onnx::onnx_proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TensorShapeProto,
    TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_propagate::margin_row::{
    BabConfig, EpochRuntimePolicy, MarginRowBab, MarginRowOutcome, RootGates, RoundMode, TreeMode,
    TwinNet, WavePolicy,
};
use prost::Message;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::test_faces::{check, Checked};
use crate::commands::margin_row_bab::{extract_twin_spec_named, parse_vnnlib_robustness};
use crate::commands::proof_bundle::{
    bundle_path, enable_hint_capture_for_test, sites_record, BundleWriter, Completion, Header,
    VerdictInfo, FINALISE_WAIT,
};

fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: Vec<AttributeProto>) -> NodeProto {
    NodeProto {
        op_type: op.to_owned(),
        input: inputs.iter().map(|s| (*s).to_owned()).collect(),
        output: outputs.iter().map(|s| (*s).to_owned()).collect(),
        attribute: attrs,
        ..Default::default()
    }
}

fn ints(name: &str, v: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_owned(),
        ints: v.to_vec(),
        r#type: 7,
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

/// The margin-row family net of the B2.1 e2e (2x6x6 input, a strided conv,
/// a residual block, two trunk Relus, a 5-wide head Relu, 4 classes) with
/// deterministic f32 weights.
fn twin_model(seed: u64) -> ModelProto {
    let mut s = seed | 1;
    let mut w = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((((s >> 33) % 1001) as f64 / 1000.0 - 0.5) * 0.7) as f32
            })
            .collect()
    };
    let dim = |d: i64| Dimension {
        value: Some(dimension::Value::DimValue(d)),
    };
    let vi = |name: &str, dims: &[i64]| ValueInfoProto {
        name: name.to_owned(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: dims.iter().map(|&d| dim(d)).collect(),
                }),
            }),
        }),
    };
    let initializer = vec![
        f32_tensor("c1w", &[4, 2, 3, 3], &w(72)),
        f32_tensor("c1b", &[4], &w(4)),
        f32_tensor("c2w", &[4, 4, 3, 3], &w(144)),
        f32_tensor("c2b", &[4], &w(4)),
        f32_tensor("g1w", &[5, 36], &w(180)),
        f32_tensor("g1b", &[5], &w(5)),
        f32_tensor("g2w", &[4, 5], &w(20)),
        f32_tensor("g2b", &[4], &w(4)),
    ];
    let nodes = vec![
        node(
            "Conv",
            &["x", "c1w", "c1b"],
            &["c1"],
            vec![ints("strides", &[2, 2]), ints("pads", &[1, 1, 1, 1])],
        ),
        node("Relu", &["c1"], &["r1"], Vec::new()),
        node(
            "Conv",
            &["r1", "c2w", "c2b"],
            &["c2"],
            vec![ints("pads", &[1, 1, 1, 1])],
        ),
        node("Add", &["c2", "r1"], &["a1"], Vec::new()),
        node("Relu", &["a1"], &["r2"], Vec::new()),
        node("Flatten", &["r2"], &["f"], vec![int("axis", 1)]),
        node(
            "Gemm",
            &["f", "g1w", "g1b"],
            &["g1"],
            vec![int("transB", 1)],
        ),
        node("Relu", &["g1"], &["h"], Vec::new()),
        node("Gemm", &["h", "g2w", "g2b"], &["y"], vec![int("transB", 1)]),
    ];
    ModelProto {
        ir_version: 8,
        graph: Some(GraphProto {
            node: nodes,
            initializer,
            input: vec![vi("x", &[1, 2, 6, 6])],
            output: vec![vi("y", &[1, 4])],
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn robustness_vnnlib(lo: &[f64], hi: &[f64], n_y: usize, t: usize) -> String {
    let mut s = String::new();
    for i in 0..lo.len() {
        s.push_str(&format!("(declare-const X_{i} Real)\n"));
    }
    for j in 0..n_y {
        s.push_str(&format!("(declare-const Y_{j} Real)\n"));
    }
    for i in 0..lo.len() {
        s.push_str(&format!(
            "(assert (>= X_{i} {:?}))\n(assert (<= X_{i} {:?}))\n",
            lo[i], hi[i]
        ));
    }
    s.push_str("(assert (or\n");
    for j in (0..n_y).filter(|&j| j != t) {
        s.push_str(&format!("    (and (>= Y_{j} Y_{t}))\n"));
    }
    s.push_str("))\n");
    s
}

/// The route a run drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    /// The frozen-dual tier on the outer frame (the harness field arms it),
    /// the legacy tree for whatever it hands back.
    Fd,
    /// The wave β-BaB on the REAL CPU f32 search.
    Wave,
    /// The classwise conjunction of per-class trees.
    Classwise,
    /// The legacy tree with Tier-2 epochs at depth 1 (nested subtrees).
    Epoch,
}

/// Register the REAL CPU f32 wave search exactly as `main.rs` does (first
/// install wins; every test registers the same backend).
fn register_wave_backend() {
    ny_propagate::margin_row::set_wave_search_backend_factory(|| {
        Some(Arc::new(ny_bbe::CpuF32Backend::new()) as Arc<dyn ny_core::search::SearchBackend>)
    });
}

fn cfg_for(route: Route, deadline: Instant) -> BabConfig {
    let base = BabConfig {
        frontier: 1,
        frontier_contextual: 1,
        lru_cap: 64,
        max_expansions: 20_000,
        deadline: Some(deadline),
        frozen_dual_tier: Some(false),
        ..BabConfig::default()
    };
    match route {
        Route::Fd => BabConfig {
            frozen_dual_tier: Some(true),
            ..base
        },
        Route::Wave => BabConfig {
            wave: WavePolicy {
                mode: TreeMode::Wave,
                batch: 16,
                cert_threads: 2,
            },
            ..base
        },
        Route::Classwise => base,
        Route::Epoch => BabConfig {
            epoch_policy: EpochRuntimePolicy {
                depth: 1,
                max_attempts: 64,
                ..EpochRuntimePolicy::default()
            },
            ..base
        },
    }
}

/// One proved route: the checker's outcome, the bundle's records, the lane's
/// own counters.
struct RouteRun {
    checked: Checked,
    bundle: PathBuf,
    records: Vec<Value>,
    expansions: usize,
    epochs_closed: usize,
}

fn records(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .expect("bundle")
        .lines()
        .map(|l| serde_json::from_str(l).expect("every bundle line is JSON"))
        .collect()
}

/// The twin net's box centre (72 inputs).
fn twin_centre() -> Vec<f64> {
    (0..72)
        .map(|i| f64::from((i * 37 % 64) as u32) / 64.0)
        .collect()
}

/// Run `route` on the twin net over the box `centre ± eps` with the REAL
/// writer, then the checker. `None` when the lane did not prove it.
fn run_route(dir: &Path, tag: &str, onnx: &Path, eps: f64, route: Route) -> Option<RouteRun> {
    run_route_centred(dir, tag, onnx, &twin_centre(), eps, route)
}

/// [`run_route`] over the box `centre ± eps` of any twin-family net.
fn run_route_centred(
    dir: &Path,
    tag: &str,
    onnx: &Path,
    centre: &[f64],
    eps: f64,
    route: Route,
) -> Option<RouteRun> {
    let (spec, names) = extract_twin_spec_named(onnx).expect("the lane admits the net");
    let net = TwinNet::compile(&spec).expect("compile");
    let mid = ndarray::Array2::from_shape_fn((centre.len(), 1), |(i, _)| centre[i]);
    let (y, _) = net
        .forward_points(&mid, &std::collections::BTreeMap::new())
        .expect("forward");
    let (w2, b2, (n_out, n_y)) = net.gemm2();
    let scores: Vec<f64> = (0..n_out)
        .map(|o| {
            let mut sm = b2[o];
            for k in 0..n_y {
                sm += w2[o * n_y + k] * y[[k, 0]].max(0.0);
            }
            sm
        })
        .collect();
    let t = (0..n_out)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("a class");
    let lo: Vec<f64> = centre.iter().map(|c| c - eps).collect();
    let hi: Vec<f64> = centre.iter().map(|c| c + eps).collect();
    let vnnlib = dir.join(format!("{tag}.vnnlib"));
    std::fs::write(&vnnlib, robustness_vnnlib(&lo, &hi, 4, t)).expect("vnnlib");
    let (plo, phi, pt, adv) = parse_vnnlib_robustness(&vnnlib).expect("the lane parses it");
    assert_eq!(pt, t);
    let root = RootGates::build(&net, &plo, &phi, RoundMode::Outward, None).ok()?;
    let results = dir.join(format!("{tag}.txt"));
    let header = Header {
        profile: ny_core::proof_profile::ResolvedProofProfile::resolve(
            ny_core::proof_profile::EntryPoint::VnncompV1,
            Some(ny_core::proof_profile::ProofProfile::Proof),
        ),
        onnx: onnx.to_path_buf(),
        vnnlib,
    };
    let (writer, sink) = BundleWriter::create(&results, header).expect("bundle");
    writer.clauses(pt, &adv);
    let cfg = cfg_for(route, Instant::now() + Duration::from_secs(30));
    let (out, _sink) = ny_propagate::certify::with_thread_sink(Box::new(sink), || {
        ny_propagate::certify::stream_sites(&sites_record(&names));
        if route == Route::Classwise {
            MarginRowBab::run_classwise(&net, &root, pt, &adv, cfg)
        } else {
            MarginRowBab::run(&net, &root, pt, &adv, cfg)
        }
    });
    let (stats, token) = match out {
        MarginRowOutcome::Unsat(stats, token) => (stats, token),
        MarginRowOutcome::Unknown { reason, stats } => {
            eprintln!(
                "[routes-e2e] route={route:?} eps={eps} unknown reason={reason} expansions={}",
                stats.map_or(0, |s| s.expansions)
            );
            let _ = writer.finalise("unknown", None, FINALISE_WAIT);
            return None;
        }
    };
    let _ = writer.finalise(
        "unsat",
        Some(VerdictInfo {
            source: token.source().into(),
            min_margin: token.min_margin(),
            completion: Completion::OwnStream,
        }),
        FINALISE_WAIT,
    );
    let bundle = bundle_path(&results);
    let checked = check(&bundle);
    eprintln!(
        "[routes-e2e] route={route:?} eps={eps} expansions={} epochs_closed={} {}",
        stats.expansions, stats.epochs_closed, checked.done_line
    );
    Some(RouteRun {
        checked,
        records: records(&bundle),
        bundle,
        expansions: stats.expansions,
        epochs_closed: stats.epochs_closed,
    })
}

const WIDTHS: [f64; 14] = [
    1.0 / 128.0,
    1.0 / 64.0,
    17.0 / 1024.0,
    9.0 / 512.0,
    3.0 / 160.0,
    5.0 / 256.0,
    1.0 / 48.0,
    7.0 / 320.0,
    23.0 / 1024.0,
    3.0 / 128.0,
    1.0 / 40.0,
    7.0 / 256.0,
    1.0 / 32.0,
    9.0 / 256.0,
];

/// Sweep the widths (and two nets) until `wanted` holds of a proved run.
fn find_route(
    tag: &str,
    route: Route,
    wanted: impl Fn(&RouteRun) -> bool,
) -> (tempfile::TempDir, RouteRun) {
    enable_hint_capture_for_test();
    if route == Route::Wave {
        register_wave_backend();
    }
    let dir = tempfile::tempdir().expect("tempdir");
    for seed in [0x5eed_u64, 0xB23, 0xFD03, 0x77] {
        let onnx = dir.path().join(format!("twin{seed:x}.onnx"));
        std::fs::write(&onnx, twin_model(seed).encode_to_vec()).expect("onnx");
        for (k, eps) in WIDTHS.into_iter().enumerate() {
            let Some(run) = run_route(dir.path(), &format!("{tag}{seed:x}_{k}"), &onnx, eps, route)
            else {
                continue;
            };
            if wanted(&run) {
                return (dir, run);
            }
        }
    }
    panic!("{tag}: no width exercised the {route:?} route: the test is VOID");
}

fn nodes_of(run: &RouteRun) -> impl Iterator<Item = &Value> + '_ {
    run.records.iter().filter(|r| r["kind"] == "node")
}

fn assert_verified(run: &RouteRun) {
    assert_eq!(
        run.checked.verdict, "verified",
        "the checker disagrees with the emitter: {}",
        run.checked.report_json
    );
}

/// THE FROZEN-DUAL ROUTE: a class the tier closes by SPLITTING streams its
/// state record and one `fd_prune` node per prune (no `close: fd` claim
/// left), and the checker's own fd rung confirms them: verified.
#[test]
fn route_proofs_end_to_end_frozen_dual_bundle_is_verified() {
    let (_dir, run) = find_route("fd", Route::Fd, |r| {
        nodes_of(r)
            .any(|n| n["src"] == "fd_prune" && n["lits"].as_array().is_some_and(|l| !l.is_empty()))
    });
    assert_verified(&run);
    assert!(run.records.iter().any(|r| r["kind"] == "hint.fd_state"));
    assert!(run.records.iter().any(|r| r["kind"] == "hint.fd_lambda"));
    assert!(
        !nodes_of(&run).any(|n| n["close"] == "fd"),
        "a class with a complete prune trace still streamed its claim"
    );
    let report: Value = serde_json::from_str(&run.checked.report_json).expect("report");
    assert!(report["fd"]["discharges_confirmed"].as_u64().unwrap_or(0) > 0);
    assert!(report["ladder"]["fd"].as_u64().unwrap_or(0) > 0);
}

/// The FD-ONLY route: every class the root leaves open is closed by the
/// tier, so the frame has no tree at all (zero expansions) and its proof is
/// root claims plus prunes: verified.
#[test]
fn route_proofs_end_to_end_frozen_dual_only_bundle_is_verified() {
    let (_dir, run) = find_route("fdonly", Route::Fd, |r| {
        r.expansions == 0 && nodes_of(r).any(|n| n["src"] == "fd_prune")
    });
    assert_verified(&run);
    assert!(nodes_of(&run).all(|n| n["src"] == "fd_prune" || n["close"] == "root"));
}

/// THE WAVE ROUTE: every certified leaf of the wave's per-class ledgers
/// streams with the frame (`src: tree`, ONE class each), and no class keeps
/// the opaque `wave` claim: verified.
#[test]
fn route_proofs_end_to_end_wave_bundle_is_verified() {
    let (_dir, run) = find_route("wave", Route::Wave, |r| {
        r.expansions > 0 && nodes_of(r).any(|n| n["src"] == "tree")
    });
    assert_verified(&run);
    assert!(
        !nodes_of(&run).any(|n| n["close"] == "opaque"),
        "a wave class kept an opaque claim"
    );
    for n in nodes_of(&run).filter(|n| n["src"] == "tree" && n["close"] == "bound") {
        assert_eq!(
            n["dis"].as_array().map(Vec::len),
            Some(1),
            "a wave leaf discharges its row's class only: {n}"
        );
    }
}

/// THE CLASSWISE ROUTE: each class tree authorizes a `class_part` frame that
/// streams its own leaves (one class each) from the instance root; the
/// checker composes them per clause: verified.
#[test]
fn route_proofs_end_to_end_classwise_bundle_is_verified() {
    let (_dir, run) = find_route("classwise", Route::Classwise, |r| {
        r.expansions > 0 && nodes_of(r).any(|n| n["src"] == "tree")
    });
    assert_verified(&run);
    for n in nodes_of(&run).filter(|n| n["src"] == "tree" && n["close"] == "bound") {
        assert_eq!(n["dis"].as_array().map(Vec::len), Some(1), "{n}");
    }
}

/// THE EPOCH ROUTE: a nested Tier-2 subtree's frame streams its own nodes
/// under its parent leaf's absolute prefix; the parent leaf is a `subtree`
/// node claiming nothing, and the checker's cover composes the nested nodes
/// in its place: verified.
#[test]
fn route_proofs_end_to_end_epoch_subtree_bundle_is_verified() {
    let (_dir, run) = find_route("epoch", Route::Epoch, |r| {
        r.epochs_closed > 0 && nodes_of(r).any(|n| n["close"] == "subtree")
    });
    assert_verified(&run);
    let subtrees: Vec<&Value> = nodes_of(&run).filter(|n| n["close"] == "subtree").collect();
    for s in &subtrees {
        let prefix = s["lits"].as_array().expect("lits");
        assert!(
            nodes_of(&run).any(|n| n["close"] != "subtree"
                && n["lits"]
                    .as_array()
                    .is_some_and(|l| l.len() >= prefix.len() && l[..prefix.len()] == prefix[..])),
            "a subtree leaf whose nested frame streamed nothing under its prefix: {s}"
        );
    }
}

/// Re-mint a bundle from its records with a valid `end` (the checker's
/// integrity gate is not what the mutations test).
fn remint(records: &[Value]) -> Vec<u8> {
    let body: Vec<String> = records
        .iter()
        .filter(|r| r["kind"] != "end")
        .map(Value::to_string)
        .collect();
    let mut text = String::new();
    for l in &body {
        text.push_str(l);
        text.push('\n');
    }
    let digest = format!("{:x}", Sha256::digest(text.as_bytes()));
    let end = serde_json::json!({"kind": "end", "lines": body.len() + 1, "sha256": digest});
    text.push_str(&end.to_string());
    text.push('\n');
    text.into_bytes()
}

fn check_bytes(dir: &Path, name: &str, bytes: &[u8]) -> Checked {
    let path = dir.join(format!("{name}.proof.jsonl"));
    std::fs::write(&path, bytes).expect("write");
    check(&path)
}

/// MUTATIONS of a REAL frozen-dual bundle: a dropped prune (cover gap), a
/// flipped side (cover gap), a prune relabelled to another class (the
/// original class's cover has a gap) are INVALID; a multiplier moved far
/// beyond rounding is no longer confirmed by the fd rung (it may only be
/// rescued by another independent rung, never by the hint).
#[test]
fn route_proofs_end_to_end_frozen_dual_mutations_are_rejected() {
    let (dir, run) = find_route("fdmut", Route::Fd, |r| {
        r.checked.verdict == "verified"
            && nodes_of(r).any(|n| {
                n["src"] == "fd_prune" && n["lits"].as_array().is_some_and(|l| !l.is_empty())
            })
    });
    let base = remint(&run.records);
    assert_eq!(
        check_bytes(dir.path(), "remint", &base).verdict,
        "verified",
        "re-minting alone must not change the verdict"
    );
    let at = run
        .records
        .iter()
        .position(|r| {
            r["kind"] == "node"
                && r["src"] == "fd_prune"
                && r["lits"].as_array().is_some_and(|l| !l.is_empty())
        })
        .expect("a split prune");
    // Dropped prune.
    let mut dropped = run.records.clone();
    dropped.remove(at);
    let c = check_bytes(dir.path(), "dropped", &remint(&dropped));
    assert_eq!(c.verdict, "invalid", "{}", c.report_json);
    // Wrong side (the prune's deepest literal flipped).
    let mut flipped = run.records.clone();
    if let Some(lits) = flipped[at]["lits"].as_array_mut() {
        let last = lits.len() - 1;
        let side = lits[last][2].as_u64().expect("side");
        lits[last][2] = serde_json::json!(1 - side);
    }
    let c = check_bytes(dir.path(), "flipped", &remint(&flipped));
    assert_ne!(c.verdict, "verified", "{}", c.report_json);
    // Wrong class.
    let class = run.records[at]["dis"][0].as_u64().expect("class");
    let clauses: Vec<u64> = run
        .records
        .iter()
        .find(|r| r["kind"] == "clauses")
        .and_then(|r| r["list"].as_array())
        .map(|l| l.iter().filter_map(|c| c["id"].as_u64()).collect())
        .expect("clauses");
    let other = *clauses
        .iter()
        .find(|&&c| c != class)
        .expect("another class");
    let mut relabelled = run.records.clone();
    relabelled[at]["dis"] = serde_json::json!([other]);
    let c = check_bytes(dir.path(), "relabelled", &remint(&relabelled));
    assert_eq!(c.verdict, "invalid", "{}", c.report_json);
    // A multiplier moved far beyond rounding: the fd rung no longer confirms
    // that prune.
    let report: Value = serde_json::from_str(&run.checked.report_json).expect("report");
    let fd_before = report["fd"]["discharges_confirmed"]
        .as_u64()
        .expect("fd count");
    let mut moved = run.records;
    let fd = &mut moved[at]["hint"]["fd"];
    for key in ["l0", "l1"] {
        if let Some(v) = fd[key].as_array_mut() {
            for x in v.iter_mut() {
                *x = serde_json::json!(x.as_f64().unwrap_or(0.0) + 1.0e6);
            }
        }
    }
    let c = check_bytes(dir.path(), "moved", &remint(&moved));
    let r: Value = serde_json::from_str(&c.report_json).expect("report");
    assert_eq!(
        r["fd"]["discharges_confirmed"].as_u64(),
        Some(fd_before - 1),
        "a multiplier 1e6 beyond the certified point still confirmed: {}",
        c.report_json
    );
    assert_ne!(c.verdict, "invalid");
    let _ = run.bundle;
}

// ---------------------------------------------------------------------------
// #fd-compact-checkable: the frozen-dual tier on a RANKED COMPACT state
// ---------------------------------------------------------------------------

/// Nuisance conv channels per trunk layer of [`wide_twin_model`].
const NUISANCE: usize = 32;

/// The twin net of [`twin_model`], its main path weight for weight (the same
/// draws in the same order), WIDENED by a third input channel (centred at 0
/// by [`wide_centre`]) that feeds [`NUISANCE`] extra conv channels in both
/// trunk layers with zero biases. Every nuisance neuron is then unstable on
/// every box, so the frame has more than 512 unstable neurons (`2 * 32 * 9 =
/// 576` nuisance alone) and the frozen-dual tier takes the #breakpoint-dual
/// RANKED compact build — the production path of every real cifar100 /
/// tinyimagenet row — at the REAL 512-row threshold. The nuisance never feeds
/// the main channels; it reaches the head only through `coupling`-scaled
/// weights (0: never), so its omitted rows still carry margin coefficients.
fn wide_twin_model(seed: u64, coupling: f32) -> ModelProto {
    let lcg = |seed: u64| {
        let mut s = seed | 1;
        move |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((((s >> 33) % 1001) as f64 / 1000.0 - 0.5) * 0.7) as f32
                })
                .collect()
        }
    };
    let mut w = lcg(seed);
    let (c1w, c1b, c2w, c2b) = (w(72), w(4), w(144), w(4));
    let (g1w, g1b, g2w, g2b) = (w(180), w(5), w(20), w(4));
    let mut nz = lcg(seed ^ 0x9E37_79B9_7F4A_7C15);
    let c = 4 + NUISANCE;
    let mut wc1 = vec![0.0f32; c * 27];
    let mut wc2 = vec![0.0f32; c * c * 9];
    let mut bc1 = vec![0.0f32; c];
    let mut bc2 = vec![0.0f32; c];
    for co in 0..c {
        for k in 0..9 {
            if co < 4 {
                for ci in 0..2 {
                    wc1[co * 27 + ci * 9 + k] = c1w[co * 18 + ci * 9 + k];
                }
            }
        }
        if co >= 4 {
            let r = nz(9);
            wc1[co * 27 + 18..co * 27 + 27].copy_from_slice(&r);
        }
    }
    bc1[..4].copy_from_slice(&c1b);
    bc2[..4].copy_from_slice(&c2b);
    for co in 0..c {
        for ci in 0..c {
            let at = co * c * 9 + ci * 9;
            if co < 4 && ci < 4 {
                wc2[at..at + 9].copy_from_slice(&c2w[co * 36 + ci * 9..co * 36 + ci * 9 + 9]);
            } else if co >= 4 && ci >= 4 {
                let r = nz(9);
                wc2[at..at + 9].copy_from_slice(&r);
            }
        }
    }
    let mut wg1 = vec![0.0f32; 5 * c * 9];
    for o in 0..5 {
        wg1[o * c * 9..o * c * 9 + 36].copy_from_slice(&g1w[o * 36..o * 36 + 36]);
        let r = nz(NUISANCE * 9);
        for (k, v) in r.iter().enumerate() {
            wg1[o * c * 9 + 36 + k] = coupling * v;
        }
    }
    let dim = |d: i64| Dimension {
        value: Some(dimension::Value::DimValue(d)),
    };
    let vi = |name: &str, dims: &[i64]| ValueInfoProto {
        name: name.to_owned(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: dims.iter().map(|&d| dim(d)).collect(),
                }),
            }),
        }),
    };
    let ci = c as i64;
    let initializer = vec![
        f32_tensor("c1w", &[ci, 3, 3, 3], &wc1),
        f32_tensor("c1b", &[ci], &bc1),
        f32_tensor("c2w", &[ci, ci, 3, 3], &wc2),
        f32_tensor("c2b", &[ci], &bc2),
        f32_tensor("g1w", &[5, ci * 9], &wg1),
        f32_tensor("g1b", &[5], &g1b),
        f32_tensor("g2w", &[4, 5], &g2w),
        f32_tensor("g2b", &[4], &g2b),
    ];
    let nodes = vec![
        node(
            "Conv",
            &["x", "c1w", "c1b"],
            &["c1"],
            vec![ints("strides", &[2, 2]), ints("pads", &[1, 1, 1, 1])],
        ),
        node("Relu", &["c1"], &["r1"], Vec::new()),
        node(
            "Conv",
            &["r1", "c2w", "c2b"],
            &["c2"],
            vec![ints("pads", &[1, 1, 1, 1])],
        ),
        node("Add", &["c2", "r1"], &["a1"], Vec::new()),
        node("Relu", &["a1"], &["r2"], Vec::new()),
        node("Flatten", &["r2"], &["f"], vec![int("axis", 1)]),
        node(
            "Gemm",
            &["f", "g1w", "g1b"],
            &["g1"],
            vec![int("transB", 1)],
        ),
        node("Relu", &["g1"], &["h"], Vec::new()),
        node("Gemm", &["h", "g2w", "g2b"], &["y"], vec![int("transB", 1)]),
    ];
    ModelProto {
        ir_version: 8,
        graph: Some(GraphProto {
            node: nodes,
            initializer,
            input: vec![vi("x", &[1, 3, 6, 6])],
            output: vec![vi("y", &[1, 4])],
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// [`wide_twin_model`]'s box centre: the twin net's 72, then the nuisance
/// channel's 36 at 0.
fn wide_centre() -> Vec<f64> {
    let mut c = twin_centre();
    c.extend(std::iter::repeat_n(0.0, 36));
    c
}

/// The `hint.fd_state` records of COMPACT states (`relaxed > unstable`).
fn compact_states(run: &RouteRun) -> Vec<&Value> {
    run.records
        .iter()
        .filter(|r| {
            r["kind"] == "hint.fd_state"
                && r["relaxed"].as_u64().unwrap_or(0) > r["unstable"].as_u64().unwrap_or(u64::MAX)
        })
        .collect()
}

/// Every `hint.fd_lambda` entry `(site id, index)` of state `id`.
fn table_of(records: &[Value], id: u64) -> Vec<(u64, u64)> {
    records
        .iter()
        .filter(|r| r["kind"] == "hint.fd_lambda" && r["id"].as_u64() == Some(id))
        .flat_map(|r| {
            let t = r["t"].as_u64().expect("site id");
            r["idx"]
                .as_array()
                .expect("idx")
                .iter()
                .map(move |i| (t, i.as_u64().expect("index")))
        })
        .collect()
}

/// Sweep the wide net (two seeds, nuisance coupling `1/1024` then `0`) and
/// the widths until the tier closed a class on a COMPACT state by splitting
/// and the lane proved the instance. VOID (panics) when nothing does.
fn find_compact_fd_route(tag: &str) -> (tempfile::TempDir, RouteRun) {
    enable_hint_capture_for_test();
    let dir = tempfile::tempdir().expect("tempdir");
    let centre = wide_centre();
    for seed in [0x5eed_u64, 0xFD03] {
        for coupling in [1.0 / 1024.0, 0.0] {
            let onnx = dir
                .path()
                .join(format!("wide{seed:x}_{}.onnx", coupling > 0.0));
            std::fs::write(&onnx, wide_twin_model(seed, coupling).encode_to_vec()).expect("onnx");
            for (k, eps) in WIDTHS.into_iter().enumerate() {
                let name = format!("{tag}{seed:x}_{}_{k}", coupling > 0.0);
                let Some(run) =
                    run_route_centred(dir.path(), &name, &onnx, &centre, eps, Route::Fd)
                else {
                    continue;
                };
                let compact: Vec<u64> = compact_states(&run)
                    .iter()
                    .filter_map(|r| r["id"].as_u64())
                    .collect();
                eprintln!(
                    "[routes-e2e] compact seed={seed:x} coupling={coupling} eps={eps} \
                     compact_states={} verdict={}",
                    compact.len(),
                    run.checked.verdict
                );
                let split = nodes_of(&run).any(|n| {
                    n["src"] == "fd_prune"
                        && n["lits"].as_array().is_some_and(|l| !l.is_empty())
                        && n["hint"]["fd"]["state"]
                            .as_u64()
                            .is_some_and(|s| compact.contains(&s))
                });
                if split {
                    return (dir, run);
                }
            }
        }
    }
    panic!("{tag}: no width closed a class on a compact frozen-dual state: the test is VOID");
}

/// `#fd-compact-checkable` END TO END at the REAL 512-row threshold: a class
/// the tier closes by splitting on a #breakpoint-dual RANKED COMPACT state
/// (more than 512 unstable neurons) streams its prunes with its CLASS
/// RELAXATION — `hint.fd_lambda` carries one slope per relaxation symbol
/// (`relaxed` of them, more than 512, more than the state's rows) — and no
/// `close: fd` claim; the checker rebuilds its own state from that whole
/// table (`table_entries == lane_relaxed`, every producer slope consumed
/// where its own box finds the neuron unstable) and confirms the prunes:
/// VERIFIED. Then three mutations of that real bundle, each REJECTED by the
/// fd rung:
///
/// * one OMITTED row's slope dropped from the table: the state's declared
///   `relaxed` no longer matches its table, so the checker refuses the
///   state (`fd_table_count`) and the fd rung confirms none of its prunes
///   (another independent rung may still confirm a prune on its own);
/// * every slope of the table perturbed to the opposite end of `[0, 1]`
///   (`lambda < 1/2 -> 1`, else `0`): the checker evaluates THAT, looser,
///   relaxation and confirms fewer prunes;
/// * a prune relabelled to the WRONG class: the right class's cover has a
///   gap, INVALID.
///
/// (A perturbed slope that MATTERS is pinned deterministically by the
/// checker's own compact fixture, `fd_compact_state_with_a_perturbed_omitted_slope_is_not_confirmed`,
/// and refused at the producer by `fd_compact_class_relaxation_mutations_are_refused`.)
#[test]
fn route_proofs_end_to_end_compact_frozen_dual_bundle_is_verified() {
    let (dir, run) = find_compact_fd_route("fdcompact");
    assert_verified(&run);
    let report: Value = serde_json::from_str(&run.checked.report_json).expect("report");
    let compact = compact_states(&run);
    assert_ne!((compact).len(), 0);
    let order_sites = |st: &Value| -> Vec<(u64, u64)> {
        st["order"]
            .as_array()
            .expect("order")
            .iter()
            .map(|p| (p[0].as_u64().expect("site"), p[1].as_u64().expect("index")))
            .collect()
    };
    let mut omitted_entry: Option<(u64, (u64, u64))> = None;
    for st in &compact {
        let id = st["id"].as_u64().expect("id");
        let relaxed = st["relaxed"].as_u64().expect("relaxed");
        let unstable = st["unstable"].as_u64().expect("unstable");
        assert!(
            relaxed > 512,
            "a compact state covers its whole frame: {st}"
        );
        assert!(unstable <= 512 && unstable < relaxed, "{st}");
        let table = table_of(&run.records, id);
        assert_eq!(table.len() as u64, relaxed, "the table is every symbol");
        let distinct: std::collections::BTreeSet<_> = table.iter().copied().collect();
        assert_eq!(distinct.len(), table.len(), "a table entry twice");
        let order = order_sites(st);
        for site in &order {
            assert!(distinct.contains(site), "a split site without its slope");
        }
        if omitted_entry.is_none() {
            omitted_entry = table.iter().find(|e| !order.contains(e)).map(|&e| (id, e));
        }
        let class = st["class"].clone();
        assert!(
            !nodes_of(&run).any(|n| n["close"] == "fd" && n["dis"][0] == class),
            "a compact class with a complete trace still streamed its claim"
        );
        let row = report["fd"]["states"]
            .as_array()
            .expect("fd states")
            .iter()
            .find(|r| r["id"].as_u64() == Some(id))
            .expect("the checker built the compact state");
        assert_eq!(row["built"], Value::Bool(true), "{row}");
        assert_eq!(row["table_entries"].as_u64(), Some(relaxed), "{row}");
        assert_eq!(row["lane_relaxed"].as_u64(), Some(relaxed), "{row}");
        eprintln!(
            "[routes-e2e] compact state id={id} relaxed={relaxed} rows={unstable} \
             checker_unstable={} hinted_lambda={} chord_lambda={} hint_unused={}",
            row["unstable"], row["hinted_lambda"], row["chord_lambda"], row["hint_unused"]
        );
    }
    let fd_before = report["fd"]["discharges_confirmed"]
        .as_u64()
        .expect("fd count");
    assert!(fd_before > 0);
    let base = remint(&run.records);
    assert_eq!(
        check_bytes(dir.path(), "compact_remint", &base).verdict,
        "verified",
        "re-minting alone must not change the verdict"
    );
    // (1) One omitted row's slope dropped.
    let (id, (t, i)) = omitted_entry.expect("a compact state omits a row");
    let mut dropped = run.records.clone();
    let line = dropped
        .iter_mut()
        .find(|r| {
            r["kind"] == "hint.fd_lambda"
                && r["id"].as_u64() == Some(id)
                && r["t"].as_u64() == Some(t)
        })
        .expect("the omitted entry's table line");
    let at = line["idx"]
        .as_array()
        .expect("idx")
        .iter()
        .position(|v| v.as_u64() == Some(i))
        .expect("the omitted entry");
    for key in ["idx", "val", "lo", "hi"] {
        if let Some(v) = line[key].as_array_mut() {
            v.remove(at);
        }
    }
    let c = check_bytes(dir.path(), "compact_dropped", &remint(&dropped));
    let r: Value = serde_json::from_str(&c.report_json).expect("report");
    eprintln!(
        "[routes-e2e] compact dropped verdict={} fd_confirmed={} (honest {fd_before})",
        c.verdict, r["fd"]["discharges_confirmed"]
    );
    let row = r["fd"]["states"]
        .as_array()
        .expect("fd states")
        .iter()
        .find(|x| x["id"].as_u64() == Some(id))
        .expect("the state row");
    assert_eq!(row["refused"].as_str(), Some("fd_table_count"), "{row}");
    assert!(
        r["fd"]["discharges_confirmed"].as_u64().unwrap_or(0) < fd_before,
        "{}",
        c.report_json
    );
    // (2) Every slope perturbed to the opposite end of [0, 1].
    let mut perturbed = run.records.clone();
    for rec in perturbed
        .iter_mut()
        .filter(|r| r["kind"] == "hint.fd_lambda")
    {
        if let Some(v) = rec["val"].as_array_mut() {
            for x in v.iter_mut() {
                let lam = x.as_f64().unwrap_or(0.5);
                *x = serde_json::json!(if lam < 0.5 { 1.0 } else { 0.0 });
            }
        }
    }
    let c = check_bytes(dir.path(), "compact_perturbed", &remint(&perturbed));
    let r: Value = serde_json::from_str(&c.report_json).expect("report");
    eprintln!(
        "[routes-e2e] compact perturbed verdict={} fd_confirmed={} (honest {fd_before})",
        c.verdict, r["fd"]["discharges_confirmed"]
    );
    assert!(
        r["fd"]["discharges_confirmed"].as_u64().unwrap_or(0) < fd_before,
        "{}",
        c.report_json
    );
    // (3) A compact prune relabelled to the wrong class.
    let at = run
        .records
        .iter()
        .position(|r| {
            r["kind"] == "node"
                && r["src"] == "fd_prune"
                && r["lits"].as_array().is_some_and(|l| !l.is_empty())
                && r["hint"]["fd"]["state"].as_u64() == Some(id)
        })
        .expect("a compact split prune");
    let class = run.records[at]["dis"][0].as_u64().expect("class");
    let other = run
        .records
        .iter()
        .find(|r| r["kind"] == "clauses")
        .and_then(|r| r["list"].as_array())
        .and_then(|l| {
            l.iter()
                .filter_map(|c| c["id"].as_u64())
                .find(|&c| c != class)
        })
        .expect("another class");
    let mut relabelled = run.records.clone();
    relabelled[at]["dis"] = serde_json::json!([other]);
    let c = check_bytes(dir.path(), "compact_relabelled", &remint(&relabelled));
    assert_eq!(c.verdict, "invalid", "{}", c.report_json);
    let _ = run.bundle;
}
