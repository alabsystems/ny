// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The B2.1 checker's named tests (brief §9): acceptance, mutations,
//! soundness properties. Every fixture is synthetic (no `benchmarks/`).

mod support;

use ny_cert::json::Json;
use ny_cert::Rat;
use ny_proofcheck::cover::{check_cover, CoverResult, DEFAULT_BUDGET};
use ny_proofcheck::testkit::{alpha_map, lower_bound, net_from_onnx};
use ny_proofcheck::{check_bundle, CheckArgs, FaultInjection, Verdict};
use support::*;

const B0: f64 = 1.0 / 1_048_576.0; // 2^-20

fn base(name: &str, b0: f64, b1: f64) -> Fixture {
    Fixture::new(name, mlp_onnx(b0, b1), &mlp_vnnlib())
}

fn base_bundle(fx: &Fixture) -> Vec<u8> {
    finish(&mlp_bundle_lines(fx, &mlp_leaves()))
}

fn rat(x: f64) -> Rat {
    Rat::from_f64_exact(x).expect("finite")
}

fn rsum(a: Rat, b: Rat) -> Rat {
    a.add(b).expect("rat add")
}

fn rmul(a: Rat, b: Rat) -> Rat {
    a.mul(b).expect("rat mul")
}

/// Exact MLP margin at `x`: relu(x0+x1) + relu(x0-x1) + b0 - b1.
fn exact_mlp_margin(x: [f64; 2], b0: f64, b1: f64) -> Rat {
    let z0 = rsum(rat(x[0]), rat(x[1]));
    let z1 = rsum(rat(x[0]), rat(x[1]).neg());
    let r = |z: Rat| if z.is_negative() { Rat::ZERO } else { z };
    rsum(rsum(r(z0), r(z1)), rsum(rat(b0), rat(b1).neg()))
}

// ---------------------------------------------------------------------------
// Acceptance
// ---------------------------------------------------------------------------

#[test]
fn accepts_a_correct_synthetic_bundle() {
    let fx = base("accept_base", B0, 0.0);
    // The root check alone must FAIL (the split is needed).
    let root_only = finish(&[
        fx.header(),
        sites(&[(0, "z", 3, 1)]),
        clauses(0, &[(1, vec![clause01()])]),
        node_line(0, &[], "root", Some(&[1]), None, None, false, -1.0),
        verdict(&[1], 1),
        published("unsat"),
    ]);
    // Without refinement (no L3, no clip) the root bound alone fails.
    let o = fx.check_with("root_only", &root_only, |a| a.refine = false);
    assert_eq!(o.verdict, Verdict::Unconfirmed, "{}", o.report_json);
    assert_eq!(o.exit_code, 10);
    // B2.2: with refinement the checker's own input clip proves the ROOT
    // node: the root form `2·x0 + b0` confines every counterexample to
    // `x0 <= -b0/2`, where both Relus' chords give a bound of `b0 > 0`. A
    // sound confirmation of a TRUE discharge (the property's exact minimum
    // is b0 > 0), reported at ladder step `clip`.
    let o = fx.check("root_only_clip", &root_only);
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    assert_eq!(
        report(&o)["ladder"]["clip"].as_u64(),
        Some(1),
        "{}",
        o.report_json
    );
    // Every leaf passes.
    let o = fx.check("full", &base_bundle(&fx));
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    assert_eq!(o.exit_code, 0);
    let r = report(&o);
    assert_eq!(r["nodes"]["confirmed"].as_u64(), Some(4));
    // The checker's margin never exceeds the exact minimum b0 - b1 = 2^-20
    // (attained at x = (-1, 0), inside every leaf's closure), and is > 0.
    let min_chk = match &r["margins"]["min_chk"] {
        Json::F64(v) => *v,
        other => panic!("min_chk {other:?}"),
    };
    assert!(min_chk > 0.0);
    let exact = exact_mlp_margin([-1.0, 0.0], B0, 0.0);
    assert!(
        rat(min_chk) <= exact,
        "certified {min_chk} above exact {exact:?}"
    );
}

#[test]
fn accepts_a_synthetic_conv_bn_residual_bundle() {
    let onnx = conv_onnx();
    let lo = vec!["0"; 16];
    let hi = vec!["1"; 16];
    let fx = Fixture::new(
        "accept_conv",
        onnx.clone(),
        &robustness_vnnlib(&lo, &hi, 2, 0, &[1]),
    );
    // The lowering is the whole chain and evaluates like a plain forward.
    let net = net_from_onnx(&onnx).expect("lowers");
    assert_eq!(net.nodes.len(), 10);
    let x: Vec<f64> = (0..16).map(|i| f64::from(i) / 16.0).collect();
    let b = ny_proofcheck::ibp::eval_point(&net, &x);
    let y = (&b.lo[net.output], &b.hi[net.output]);
    assert!(y.0.iter().zip(y.1).all(|(l, h)| l <= h && h - l < 1e-9));
    let sites_line = sites(&[(0, "b1", 32, 2), (1, "a1", 16, 5), (2, "g1", 4, 8)]);
    let root = finish(&[
        fx.header(),
        sites_line.clone(),
        clauses(0, &[(1, vec![clause01()])]),
        node_line(0, &[], "root", Some(&[1]), None, None, false, 1.0),
        verdict(&[1], 1),
        published("unsat"),
    ]);
    let o = fx.check("root", &root);
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    let tree = finish(&[
        fx.header(),
        sites_line,
        clauses(0, &[(1, vec![clause01()])]),
        node_line(
            0,
            &[L::R(0, 5, true)],
            "bound",
            Some(&[1]),
            Some(&[0.0]),
            None,
            false,
            1.0,
        ),
        node_line(
            1,
            &[L::R(0, 5, false)],
            "bound",
            Some(&[1]),
            Some(&[0.0]),
            None,
            false,
            1.0,
        ),
        verdict(&[1], 2),
        published("unsat"),
    ]);
    let o = fx.check("tree", &tree);
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
}

fn three_class_onnx(b0: f64) -> Vec<u8> {
    model(
        "x",
        &[1, 2],
        &[
            node("Gemm", &["x", "W1", "B1"], &["z"], vec![A::I("transB", 1)]),
            node("Relu", &["z"], &["h"], vec![]),
            node("Gemm", &["h", "W2", "B2"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            tf("W1", &[3, 2], &[1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
            tf("B1", &[3], &[0.0, 0.0, 5.0]),
            tf(
                "W2",
                &[3, 3],
                &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            tf("B2", &[3], &[b0, -10.0, 0.0]),
        ],
        "y",
        &[1, 3],
    )
}

fn three_class_fixture(name: &str) -> Fixture {
    Fixture::new(
        name,
        three_class_onnx(B0),
        &robustness_vnnlib(&["-1", "-1"], &["1", "1"], 3, 0, &[1, 2]),
    )
}

fn three_class_lines(fx: &Fixture, leaf_dis: u64) -> Vec<String> {
    vec![
        fx.header(),
        sites(&[(0, "z", 3, 1)]),
        clauses(
            0,
            &[
                (1, vec!["Y0:1,Y1:-1<=0".into()]),
                (2, vec!["Y0:1,Y2:-1<=0".into()]),
            ],
        ),
        node_line(0, &[], "root", Some(&[1]), None, None, false, 10.0),
        node_line(
            1,
            &[L::R(0, 0, false)],
            "retire",
            Some(&[2]),
            Some(&[0.0]),
            None,
            false,
            B0,
        ),
        node_line(
            2,
            &[L::R(0, 0, true)],
            "bound",
            Some(&[leaf_dis]),
            Some(&[1.0]),
            None,
            false,
            B0,
        ),
        verdict(&[1, 2], 3),
        published("unsat"),
    ]
}

#[test]
fn accepts_per_disjunct_cover_with_retire_and_root_discharges() {
    let fx = three_class_fixture("accept_retire_root");
    let o = fx.check("b", &finish(&three_class_lines(&fx, 2)));
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    let r = report(&o);
    match &r["cover"] {
        Json::Arr(rows) => {
            assert_eq!(rows.len(), 2);
            assert!(rows
                .iter()
                .all(|c| matches!(c["confirmed_ok"], Json::Bool(true))));
        }
        other => panic!("cover {other:?}"),
    }
}

#[test]
fn accepts_an_emptiness_node() {
    let fx = base("accept_empty", B0, 0.0);
    // z2 = x0 + x1 + 5 >= 3: the half z2 <= 0 is empty.
    let mut lines = vec![
        fx.header(),
        sites(&[(0, "z", 3, 1)]),
        clauses(0, &[(1, vec![clause01()])]),
        node_line(
            0,
            &[L::R(0, 2, false)],
            "empty",
            None,
            None,
            None,
            false,
            3.0,
        ),
    ];
    for (k, (lits, beta)) in mlp_leaves().into_iter().enumerate() {
        let mut l = vec![L::R(0, 2, true)];
        l.extend(lits);
        let mut b = vec![0.0];
        b.extend(beta);
        lines.push(node_line(
            k as u64 + 1,
            &l,
            "bound",
            Some(&[1]),
            Some(&b),
            None,
            false,
            B0,
        ));
    }
    lines.push(verdict(&[1], 5));
    lines.push(published("unsat"));
    let o = fx.check("b", &finish(&lines));
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    assert!(report(&o)["nodes"]["empty"].as_u64() >= Some(1));
}

fn input_split_onnx(b: f64) -> Vec<u8> {
    // z0 = x0, z1 = x0 + 10 (always active); y0 = h0 - h1/2 + 5 + b
    //    = relu(x0) - x0/2 + b; y1 = 0.
    model(
        "x",
        &[1, 1],
        &[
            node("Gemm", &["x", "W1", "B1"], &["z"], vec![A::I("transB", 1)]),
            node("Relu", &["z"], &["h"], vec![]),
            node("Gemm", &["h", "W2", "B2"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            tf("W1", &[2, 1], &[1.0, 1.0]),
            tf("B1", &[2], &[0.0, 10.0]),
            tf("W2", &[2, 2], &[1.0, -0.5, 0.0, 0.0]),
            tf("B2", &[2], &[5.0 + b, 0.0]),
        ],
        "y",
        &[1, 2],
    )
}

#[test]
fn accepts_input_split_literals_over_a_box() {
    let fx = Fixture::new(
        "accept_input_split",
        input_split_onnx(B0),
        &robustness_vnnlib(&["-1"], &["1"], 2, 0, &[1]),
    );
    let lines = vec![
        fx.header(),
        sites(&[(0, "z", 2, 1)]),
        clauses(0, &[(1, vec![clause01()])]),
        node_line(
            0,
            &[L::X(0, 0.0, true)],
            "bound",
            Some(&[1]),
            None,
            None,
            false,
            B0,
        ),
        node_line(
            1,
            &[L::X(0, 0.0, false)],
            "bound",
            Some(&[1]),
            None,
            None,
            false,
            B0,
        ),
        verdict(&[1], 2),
        published("unsat"),
    ];
    let o = fx.check("b", &finish(&lines));
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    // The x <= 0 half needs the node-box refinement (L3).
    assert!(
        report(&o)["ladder"]["L3"].as_u64() >= Some(1),
        "{}",
        o.report_json
    );
    // Without refinement the same bundle is only unconfirmed.
    let o = fx.check_with("b_norefine", &finish(&lines), |a| a.refine = false);
    assert_eq!(o.verdict, Verdict::Unconfirmed, "{}", o.report_json);
}

#[test]
fn ignores_unknown_hint_records_and_counts_them() {
    let fx = base("hint_ignored", B0, 0.0);
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
    lines.insert(3, r#"{"kind":"hint.future","payload":[1,2,3]}"#.to_string());
    lines.insert(3, r#"{"kind":"hint.lemma","t":0}"#.to_string());
    let o = fx.check("b", &finish(&lines));
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    assert_eq!(report(&o)["nodes"]["hints_ignored"].as_u64(), Some(2));
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

#[test]
fn rejects_flipped_literal_sign() {
    let fx = base("mut_flip", B0, 0.0);
    let mut leaves = mlp_leaves();
    leaves[0].0[1] = L::R(0, 1, false);
    let o = fx.check("b", &finish(&mlp_bundle_lines(&fx, &leaves)));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(reason_codes(&o).contains(&"cover_gap".to_string()));
}

#[test]
fn rejects_dropped_leaf_as_a_cover_gap() {
    let fx = base("mut_drop", B0, 0.0);
    let mut leaves = mlp_leaves();
    leaves.remove(3);
    let o = fx.check("b", &finish(&mlp_bundle_lines(&fx, &leaves)));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(reason_codes(&o).contains(&"cover_gap".to_string()));
}

#[test]
fn rejects_network_coefficient_perturbed_beyond_rounding() {
    // b1 moved so the true minimum is -2^-9 < -1e-3; header re-hashed.
    let b1 = B0 + 1.0 / 512.0;
    let fx = base("mut_perturb_big", B0, b1);
    assert!(exact_mlp_margin([0.0, 0.0], B0, b1) < rat(-1e-3));
    let o = fx.check("b", &base_bundle(&fx));
    assert_eq!(o.exit_code, 21, "{}", o.report_json);
    let r = report(&o);
    assert!(
        !r["refutation"]["counterexample"].is_null(),
        "the witness must be in the report: {}",
        o.report_json
    );
    assert!(r["refutation"]["counterexample"]["vnncomp"]
        .as_str()
        .is_some_and(|s| s.contains("(X_0 ") && s.contains("(Y_1 ")));
}

#[test]
fn accepts_network_coefficient_perturbed_within_rounding_slack() {
    // 2^-40 against the 2^-20 slack.
    let fx = base("mut_perturb_small", B0, 1.0 / 1_099_511_627_776.0);
    let o = fx.check("b", &base_bundle(&fx));
    assert_eq!(o.exit_code, 0, "{}", o.report_json);
}

#[test]
fn rejects_wrong_spec_clause_on_a_discharge() {
    let fx = three_class_fixture("mut_wrong_clause");
    // The z0 >= 0 leaf claims clause 1 instead of 2: clause 2 has a gap.
    let o = fx.check("b", &finish(&three_class_lines(&fx, 1)));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    let r = report(&o);
    let gap_on_2 = match &r["reasons"] {
        Json::Arr(v) => v.iter().any(|x| {
            x["code"].as_str() == Some("cover_gap") && x["detail"]["disjunct"].as_u64() == Some(1)
        }),
        _ => false,
    };
    assert!(
        gap_on_2,
        "the gap must be on the true clause: {}",
        o.report_json
    );
}

#[test]
fn rejects_producer_clause_that_the_vnnlib_does_not_contain() {
    let fx = base("mut_extra_clause", B0, 0.0);
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
    lines[2] = clauses(
        0,
        &[(1, vec![clause01()]), (7, vec!["Y0:1,Y1:-1<=1/2".into()])],
    );
    let o = fx.check("b", &finish(&lines));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(reason_codes(&o).contains(&"clause_mismatch".to_string()));
}

#[test]
fn rejects_truncated_bundle() {
    let fx = base("mut_truncated", B0, 0.0);
    let full = base_bundle(&fx);
    let cut = &full[..full.len() / 2];
    let o = fx.check("cut", cut);
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    // Dropped `end`: every line intact, the trailer missing.
    let lines = mlp_bundle_lines(&fx, &mlp_leaves());
    let mut no_end = lines.join("\n");
    no_end.push('\n');
    let o = fx.check("no_end", no_end.as_bytes());
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    // A tampered body under an intact `end` fails the digest.
    let text = String::from_utf8(full).expect("utf8");
    let tampered = text.replacen("\"margin\":1e-6", "\"margin\":2e-6", 1);
    assert_ne!(tampered, text);
    let o = fx.check("tampered", tampered.as_bytes());
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(reason_codes(&o).contains(&"integrity".to_string()));
}

#[test]
fn rejects_unknown_schema_version() {
    let fx = base("mut_schema", B0, 0.0);
    for (schema, code) in [
        ("ny-proof-bundle/v9", "unknown_schema"),
        ("ny-proof-bundle/v1", "v1_positional_sites"),
    ] {
        let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
        lines[0] = lines[0].replace("ny-proof-bundle/v2", schema);
        let o = fx.check(schema.rsplit('/').next().unwrap_or("x"), &finish(&lines));
        assert_eq!(o.exit_code, 30, "{}", o.report_json);
        assert!(
            reason_codes(&o).contains(&code.to_string()),
            "{}",
            o.report_json
        );
    }
}

#[test]
fn rejects_onnx_or_vnnlib_hash_mismatch() {
    let fx = base("mut_hash", B0, 0.0);
    for (name, onnx_extra, vnn_extra) in [("onnx", true, false), ("vnnlib", false, true)] {
        let mut ob = fx.onnx_bytes.clone();
        let mut vb = fx.vnn_bytes.clone();
        if onnx_extra {
            ob.push(0);
        }
        if vnn_extra {
            vb.push(b' ');
        }
        let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
        lines[0] = header(&fx.onnx, &ob, &fx.vnnlib, &vb);
        let o = fx.check(name, &finish(&lines));
        assert_eq!(o.exit_code, 30, "{}", o.report_json);
        assert!(reason_codes(&o).contains(&"hash_mismatch".to_string()));
    }
}

#[test]
fn rejects_unpublished_or_partial_bundle() {
    let fx = base("mut_unpublished", B0, 0.0);
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
    let n = lines.len();
    lines[n - 1] = published("unknown");
    let o = fx.check("unknown", &finish(&lines));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(reason_codes(&o).contains(&"unpublished".to_string()));
    let partial = fx.dir.join("run.proof.jsonl.partial");
    std::fs::write(&partial, base_bundle(&fx)).expect("write");
    let mut args = CheckArgs::new(partial);
    args.threads = Some(1);
    let o = check_bundle(&args).expect("operational");
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(reason_codes(&o).contains(&"partial_bundle".to_string()));
}

#[test]
fn rejects_unknown_structural_record_or_literal_kind() {
    let fx = base("mut_unknown_kind", B0, 0.0);
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
    lines.insert(3, r#"{"kind":"lemma","node":0}"#.to_string());
    let o = fx.check("record", &finish(&lines));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(reason_codes(&o).contains(&"unknown_record_kind".to_string()));
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
    lines[3] = lines[3].replacen("\"lits\":[[0,0,1]", "\"lits\":[[\"y\",0,1]", 1);
    let o = fx.check("literal", &finish(&lines));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(
        reason_codes(&o).contains(&"unknown_literal_kind".to_string()),
        "{}",
        o.report_json
    );
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
    lines[3] = lines[3].replacen("\"close\":\"bound\"", "\"close\":\"guess\"", 1);
    let o = fx.check("close", &finish(&lines));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
}

#[test]
fn rejects_literal_on_a_tensor_that_is_not_a_relu_input() {
    let fx = base("mut_not_relu", B0, 0.0);
    for (name, tensor, size) in [("output", "y", 2u64), ("relu_out", "h", 3)] {
        let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
        lines[1] = sites(&[(0, tensor, size, 1)]);
        let o = fx.check(name, &finish(&lines));
        assert_eq!(o.exit_code, 30, "{}", o.report_json);
        assert!(reason_codes(&o).contains(&"not_a_relu_input".to_string()));
    }
}

#[test]
fn refuses_unsupported_ops_and_spec_forms() {
    let sig = model(
        "x",
        &[1, 2],
        &[
            node("Gemm", &["x", "W1", "B1"], &["z"], vec![A::I("transB", 1)]),
            node("Sigmoid", &["z"], &["h"], vec![]),
            node("Gemm", &["h", "W2", "B2"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            tf("W1", &[3, 2], &[1.0; 6]),
            tf("B1", &[3], &[0.0; 3]),
            tf("W2", &[2, 3], &[1.0; 6]),
            tf("B2", &[2], &[0.0; 2]),
        ],
        "y",
        &[1, 2],
    );
    let fx = Fixture::new("unsup_op", sig, &mlp_vnnlib());
    let o = fx.check("b", &base_bundle(&fx));
    assert_eq!(o.exit_code, 31, "{}", o.report_json);
    for (name, spec) in [
        (
            "arith",
            "(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n\
             (declare-const Y_1 Real)\n(assert (<= X_0 1))\n(assert (>= X_0 -1))\n\
             (assert (<= X_1 1))\n(assert (>= X_1 -1))\n(assert (<= (+ Y_0 Y_1) 0))\n",
        ),
        (
            "mixed",
            "(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n\
             (declare-const Y_1 Real)\n(assert (<= X_0 1))\n(assert (>= X_0 -1))\n\
             (assert (<= X_1 1))\n(assert (>= X_1 -1))\n(assert (<= X_0 Y_0))\n",
        ),
        (
            "unbounded",
            "(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n\
             (declare-const Y_1 Real)\n(assert (<= X_0 1))\n(assert (>= X_1 -1))\n\
             (assert (<= Y_0 Y_1))\n",
        ),
    ] {
        let fx = Fixture::new(&format!("unsup_spec_{name}"), mlp_onnx(B0, 0.0), spec);
        let o = fx.check("b", &base_bundle(&fx));
        assert_eq!(o.exit_code, 31, "{name}: {}", o.report_json);
    }
}

#[test]
fn thin_true_leaf_is_unconfirmed_not_refuted() {
    let fx = base("thin_leaf", B0, 0.0);
    let mut leaves = mlp_leaves();
    // Without its β the (+,+) leaf's relaxation cannot see the literals.
    leaves[0].1 = vec![0.0, 0.0];
    let o = fx.check("b", &finish(&mlp_bundle_lines(&fx, &leaves)));
    assert_eq!(o.exit_code, 10, "{}", o.report_json);
    assert!(report(&o)["refutation"]["counterexample"].is_null());
}

#[test]
fn opaque_route_claim_is_never_confirmed() {
    let fx = base("opaque", B0, 0.0);
    let lines = vec![
        fx.header(),
        sites(&[(0, "z", 3, 1)]),
        clauses(0, &[(1, vec![clause01()])]),
        node_line(0, &[], "opaque", Some(&[1]), None, None, false, 1.0),
        verdict(&[1], 1),
        published("unsat"),
    ];
    let o = fx.check("b", &finish(&lines));
    assert_eq!(o.exit_code, 10, "{}", o.report_json);
}

// ---------------------------------------------------------------------------
// Soundness and properties
// ---------------------------------------------------------------------------

#[test]
fn garbage_hints_never_confirm_a_false_discharge() {
    // Every leaf's true minimum is 2^-20 - 0.25 < 0.
    let fx = base("garbage_hints", B0, 0.25);
    let mut s: u64 = 0xDEAD_BEEF;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for round in 0..24 {
        let mut lines = vec![
            fx.header(),
            sites(&[(0, "z", 3, 1)]),
            clauses(0, &[(1, vec![clause01()])]),
        ];
        let vals = [-1.0, 0.0, 0.5, 1.0, 3.0, 1e300, -1e300, 0.999_999];
        let pick = |r: u64| vals[(r % vals.len() as u64) as usize];
        lines.push(alpha_line(
            9,
            0,
            &[0, 1, 2],
            &[pick(next()), pick(next()), pick(next())],
        ));
        for (k, (lits, _)) in mlp_leaves().into_iter().enumerate() {
            let beta = vec![pick(next()), pick(next())];
            lines.push(node_line(
                k as u64,
                &lits,
                "bound",
                Some(&[1]),
                Some(&beta),
                Some(9),
                true,
                1.0,
            ));
        }
        lines.push(verdict(&[1], 4));
        lines.push(published("unsat"));
        let o = fx.check(&format!("r{round}"), &finish(&lines));
        let r = report(&o);
        assert_eq!(
            r["nodes"]["confirmed"].as_u64(),
            Some(0),
            "a garbage hint confirmed a false discharge: {}",
            o.report_json
        );
        assert_ne!(o.exit_code, 0);
    }
}

fn random_mlp(seed: u64) -> (Vec<u8>, usize, usize) {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s % 17) as f64 - 8.0) / 8.0
    };
    let (n_in, n_h) = (2 + (seed % 2) as usize, 3 + (seed % 3) as usize);
    let w1: Vec<f64> = (0..n_h * n_in).map(|_| next()).collect();
    let b1: Vec<f64> = (0..n_h).map(|_| next()).collect();
    let w2: Vec<f64> = (0..2 * n_h).map(|_| next()).collect();
    let b2: Vec<f64> = (0..2).map(|_| next()).collect();
    let onnx = model(
        "x",
        &[1, n_in as i64],
        &[
            node("Gemm", &["x", "W1", "B1"], &["z"], vec![A::I("transB", 1)]),
            node("Relu", &["z"], &["h"], vec![]),
            node("Gemm", &["h", "W2", "B2"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            tf("W1", &[n_h as i64, n_in as i64], &w1),
            tf("B1", &[n_h as i64], &b1),
            tf("W2", &[2, n_h as i64], &w2),
            tf("B2", &[2], &b2),
        ],
        "y",
        &[1, 2],
    );
    (onnx, n_in, n_h)
}

/// Exact forward of a lowered 2-layer Gemm/Relu/Gemm net at `x`.
fn exact_forward(net: &ny_proofcheck::lower::Net, x: &[f64]) -> (Vec<Rat>, Vec<Rat>) {
    use ny_proofcheck::lower::Op;
    let mut vals: Vec<Vec<Rat>> = vec![Vec::new(); net.tensors.len()];
    vals[net.input] = x.iter().map(|&v| rat(v)).collect();
    for nd in &net.nodes {
        let out = match &nd.op {
            Op::Gemm(g) => (0..g.n_out)
                .map(|j| {
                    let mut s = rat(g.b[j]);
                    for i in 0..g.n_in {
                        s = rsum(s, rmul(rat(g.w[j * g.n_in + i]), vals[g.x][i]));
                    }
                    s
                })
                .collect(),
            Op::Relu { x } => vals[*x]
                .iter()
                .map(|v| if v.is_negative() { Rat::ZERO } else { *v })
                .collect(),
            other => panic!("unexpected op {other:?}"),
        };
        vals[nd.out] = out;
    }
    let z = vals[net.by_name["z"]].clone();
    (z, vals[net.output].clone())
}

#[test]
fn certified_bound_never_exceeds_exact_rational_value() {
    let mut checked = 0usize;
    for seed in 1..40u64 {
        let (onnx, n_in, n_h) = random_mlp(seed * 7919);
        let net = net_from_onnx(&onnx).expect("lowers");
        let lo = vec![-1.0; n_in];
        let hi = vec![1.0; n_in];
        // Literals: the first two hidden neurons, sides from the seed.
        let lits = [("z", 0usize, seed % 2 == 0), ("z", 1usize, seed % 3 == 0)];
        let beta = [0.25 * (seed % 4) as f64, 0.5];
        let alpha = alpha_map(&net, &[("z", 2, 0.5)]);
        let Some((l, lin, c)) = lower_bound(
            &net,
            &lo,
            &hi,
            &lits,
            &beta,
            Some(&alpha),
            &[(0, 1.0), (1, -1.0)],
            0.0,
        ) else {
            continue;
        };
        let mut s: u64 = seed;
        for _ in 0..64 {
            let x: Vec<f64> = (0..n_in)
                .map(|_| {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    ((s >> 11) % 257) as f64 / 128.0 - 1.0
                })
                .collect();
            let (z, y) = exact_forward(&net, &x);
            let inside = lits.iter().all(|&(_, i, pos)| {
                if pos {
                    !z[i].is_negative()
                } else {
                    !z[i].is_positive()
                }
            });
            if !inside {
                continue;
            }
            let g = rsum(y[0], y[1].neg());
            assert!(
                rat(l) <= g,
                "seed {seed}: certified {l} above exact {g:?} (n_h {n_h})"
            );
            let mut lx = rat(c);
            for (i, &xi) in x.iter().enumerate() {
                lx = rsum(lx, rmul(rat(lin[i]), rat(xi)));
            }
            assert!(
                lx <= g,
                "seed {seed}: linear form above the exact objective"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 100,
        "only {checked} points landed in a region: the test is VOID"
    );
}

#[test]
fn rounding_charges_cover_adversarial_cancellation() {
    // z0 = 2^40 x0 - 2^40 x1 + x2 with x0, x1 the SAME non-dyadic decimal
    // (their boxes are one-ulp intervals): the huge terms cancel in exact
    // arithmetic and the rounding charges must still hold the bound down.
    let big = 1_099_511_627_776.0; // 2^40
    let onnx = model(
        "x",
        &[1, 3],
        &[
            node("Gemm", &["x", "W1", "B1"], &["z"], vec![A::I("transB", 1)]),
            node("Relu", &["z"], &["h"], vec![]),
            node("Gemm", &["h", "W2", "B2"], &["y"], vec![A::I("transB", 1)]),
        ],
        &[
            tf("W1", &[2, 3], &[big, -big, 1.0, big, -big, 0.0]),
            tf("B1", &[2], &[0.0, 0.0]),
            tf("W2", &[2, 2], &[big, -big, 0.0, 0.0]),
            tf("B2", &[2], &[0.0, 0.0]),
        ],
        "y",
        &[1, 2],
    );
    let net = net_from_onnx(&onnx).expect("lowers");
    let v: f64 = 0.1;
    let lo = vec![v.next_down(), v.next_down(), 0.0];
    let hi = vec![v.next_up(), v.next_up(), 1.0];
    let (l, lin, c) =
        lower_bound(&net, &lo, &hi, &[], &[], None, &[(0, 1.0), (1, -1.0)], 0.0).expect("bound");
    for x in [
        [v, v, 0.5],
        [v.next_down(), v.next_up(), 0.0],
        [v.next_up(), v.next_down(), 1.0],
        [v, v.next_up(), 0.25],
    ] {
        let (_, y) = exact_forward(&net, &x);
        let g = rsum(y[0], y[1].neg());
        assert!(rat(l) <= g, "certified {l} above exact {g:?} at {x:?}");
        let mut lx = rat(c);
        for (i, &xi) in x.iter().enumerate() {
            lx = rsum(lx, rmul(rat(lin[i]), rat(xi)));
        }
        assert!(lx <= g, "linear form above the exact objective at {x:?}");
    }
}

#[test]
fn relu_upper_line_is_above_relu_at_both_endpoints_after_rounding() {
    let mut s: u64 = 0xABCD_EF01;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut n = 0;
    for _ in 0..2000 {
        let el = (next() % 600) as i32 - 300;
        let eu = (next() % 600) as i32 - 300;
        let l = -((next() % 1000 + 1) as f64) * 10f64.powi(el);
        let u = ((next() % 1000 + 1) as f64) * 10f64.powi(eu);
        if !(l.is_finite() && u.is_finite() && l < 0.0 && u > 0.0) {
            continue;
        }
        let Some((sl, t)) = ny_proofcheck::arith::relu_chord(l, u) else {
            continue;
        };
        let at_l = rsum(rmul(rat(sl), rat(l)), rat(t));
        let at_u = rsum(rmul(rat(sl), rat(u)), rat(t));
        assert!(!at_l.is_negative(), "line below relu at l={l} u={u}");
        assert!(at_u >= rat(u), "line below relu at u: l={l} u={u}");
        assert!((0.0..=1.0).contains(&sl));
        n += 1;
    }
    assert!(n > 1000, "only {n} chords checked");
}

#[test]
fn beta_sign_convention_matches_the_documented_lagrangian() {
    let net = net_from_onnx(&mlp_onnx(B0, 0.0)).expect("lowers");
    let lo = [-1.0, -1.0];
    let hi = [1.0, 1.0];
    let obj = [(0, 1.0), (1, -1.0)];
    // z >= 0 literals: coefficient of z gains -β:
    // (1-0.25)(x0+x1) + (1-0.5)(x0-x1) = 1.25 x0 + 0.25 x1.
    let (_, lin, c) = lower_bound(
        &net,
        &lo,
        &hi,
        &[("z", 0, true), ("z", 1, true)],
        &[0.25, 0.5],
        None,
        &obj,
        0.0,
    )
    .expect("bound");
    assert_eq!(lin, vec![1.25, 0.25]);
    assert!(c <= B0 && c > B0 - 1e-12);
    // z <= 0 literals: coefficient gains +β: 0.25(x0+x1) + 0.5(x0-x1).
    let (_, lin, _) = lower_bound(
        &net,
        &lo,
        &hi,
        &[("z", 0, false), ("z", 1, false)],
        &[0.25, 0.5],
        None,
        &obj,
        0.0,
    )
    .expect("bound");
    assert_eq!(lin, vec![0.75, -0.25]);
    // The documented dual point closes the (+,+) leaf; the flipped sign
    // convention would double the input coefficient instead.
    let (l, _, _) = lower_bound(
        &net,
        &lo,
        &hi,
        &[("z", 0, true), ("z", 1, true)],
        &[1.0, 1.0],
        None,
        &obj,
        0.0,
    )
    .expect("bound");
    assert!(l > 0.0 && l <= B0);
}

#[test]
fn cover_accepts_any_split_order_and_overlaps_and_refuses_gaps() {
    let s = |i: u64| (0u8, 0u64, i);
    // Left subtree split on a then b; right subtree on c then a.
    let tree = vec![
        vec![(s(1), true), (s(2), true)],
        vec![(s(1), true), (s(2), false)],
        vec![(s(1), false), (s(3), true)],
        vec![(s(1), false), (s(3), false), (s(2), true)],
        vec![(s(1), false), (s(3), false), (s(2), false)],
    ];
    let sorted = |v: &[Vec<((u8, u64, u64), bool)>]| -> Vec<Vec<((u8, u64, u64), bool)>> {
        v.iter()
            .map(|m| {
                let mut m = m.clone();
                m.sort_unstable();
                m
            })
            .collect()
    };
    assert_eq!(
        check_cover(&sorted(&tree), DEFAULT_BUDGET),
        CoverResult::Covered
    );
    let mut rev = tree.clone();
    rev.reverse();
    assert_eq!(
        check_cover(&sorted(&rev), DEFAULT_BUDGET),
        CoverResult::Covered
    );
    // An overlapping extra member (an internal retire node) is harmless.
    let mut over = tree.clone();
    over.push(vec![(s(1), false)]);
    assert_eq!(
        check_cover(&sorted(&over), DEFAULT_BUDGET),
        CoverResult::Covered
    );
    // A missing leaf is a gap with its witness.
    let mut hole = tree;
    hole.remove(3);
    match check_cover(&sorted(&hole), DEFAULT_BUDGET) {
        CoverResult::Gap(w) => {
            assert!(w.contains(&(s(1), false)) && w.contains(&(s(3), false)));
        }
        other => panic!("expected a gap, got {other:?}"),
    }
    assert!(matches!(
        check_cover(&[], DEFAULT_BUDGET),
        CoverResult::Gap(_)
    ));
    // The root member covers everything.
    assert_eq!(
        check_cover(&[Vec::new()], DEFAULT_BUDGET),
        CoverResult::Covered
    );
    // Siblings that split DIFFERENT sites leave a gap.
    let skew = vec![vec![(s(5), false)], vec![(s(7), true)]];
    assert!(matches!(
        check_cover(&sorted(&skew), DEFAULT_BUDGET),
        CoverResult::Gap(_)
    ));
}

#[test]
fn report_is_deterministic_across_thread_counts() {
    let fx = base("determinism", B0, 0.0);
    let bundle = base_bundle(&fx);
    let one = fx.check_with("t1", &bundle, |a| a.threads = Some(1));
    let four = fx.check_with("t4", &bundle, |a| a.threads = Some(4));
    let d1 = report(&one)["result_digest"].as_str().map(str::to_string);
    let d4 = report(&four)["result_digest"].as_str().map(str::to_string);
    // The bundle path differs per run name; the digest covers it, so compare
    // with the paths equalized.
    let norm = |o: &ny_proofcheck::CheckOutcome| {
        let mut r = report(o);
        if let Json::Obj(m) = &mut r {
            m.remove("timing");
            m.remove("result_digest");
            m.remove("bundle");
        }
        ny_cert::json::to_string(&r)
    };
    assert_eq!(norm(&one), norm(&four));
    assert!(d1.is_some() && d4.is_some());
    // Same path, different thread counts: identical digests.
    let path = fx.dir.join("same.proof.jsonl");
    std::fs::write(&path, &bundle).expect("write");
    let run = |t: usize| {
        let mut a = CheckArgs::new(path.clone());
        a.threads = Some(t);
        a.checker_build = "test".into();
        a.report = Some(fx.dir.join(format!("same.{t}.check.json")));
        let o = check_bundle(&a).expect("operational");
        report(&o)["result_digest"].as_str().map(str::to_string)
    };
    assert_eq!(run(1), run(3));
}

#[test]
fn self_containment_trips_on_an_injected_checker_fault() {
    let fx = base("fault", B0, 0.0);
    let o = fx.check_with("b", &base_bundle(&fx), |a| {
        a.fault = Some(FaultInjection::InflateRootLower);
    });
    assert_eq!(o.exit_code, 40, "{}", o.report_json);
    assert!(reason_codes(&o).contains(&"self_containment".to_string()));
}

#[test]
fn done_line_propagates_the_real_exit_code() {
    let fx = base("done_line", B0, 0.0);
    let ok = fx.check("ok", &base_bundle(&fx));
    assert!(
        ok.done_line
            .starts_with("PROOF_CHECK_DONE rc=0 verdict=verified "),
        "{}",
        ok.done_line
    );
    let file = std::fs::read(&ok.report_path).expect("report written");
    assert!(ok.done_line.contains(&format!(
        "report_sha256={}",
        ny_proofcheck::bundle::sha256_hex(&file)
    )));
    let full = base_bundle(&fx);
    let bad = fx.check("bad", &full[..full.len() - 3]);
    assert!(
        bad.done_line
            .starts_with("PROOF_CHECK_DONE rc=30 verdict=malformed "),
        "{}",
        bad.done_line
    );
    let mut leaves = mlp_leaves();
    leaves.remove(0);
    let gap = fx.check("gap", &finish(&mlp_bundle_lines(&fx, &leaves)));
    assert!(
        gap.done_line
            .starts_with("PROOF_CHECK_DONE rc=20 verdict=invalid "),
        "{}",
        gap.done_line
    );
    assert!(gap.done_line.contains(" cover=gap "));
    for o in [&ok, &bad, &gap] {
        assert_eq!(o.exit_code, o.verdict.code());
    }
}
