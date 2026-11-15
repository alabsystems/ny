// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! LEGACY WELL-FORMEDNESS (certified-moat review, 2026-10-08).
//!
//! A legacy lane's published `unsat` streams nothing the checker can
//! re-derive. Its bundle is completed by the producer's writer with an empty
//! `sites`, the instance's `clauses` and one whole-region `opaque` claim per
//! clause (`src: legacy:<authority>`), or — when the producer cannot render
//! the property or hash an input — the `ny-proof-bundle/v2.1` `unavailable`
//! forms. These tests pin the checker's side on synthetic bundles:
//!
//! * such a bundle is WELL-FORMED and `unconfirmed` (exit 10), never
//!   `verified`, whatever its nodes claim (the claims are not certificates);
//! * dropping the legacy claim, or the requirement, is `invalid` (exit 20);
//! * a v2.1 form under a v2 header, or an unavailable clause next to another
//!   clause, is `malformed` (exit 30);
//! * a v2.1 bundle with ordinary clauses is checked exactly like v2.

mod support;

use ny_cert::json::{self, Json, Map};
use ny_proofcheck::bundle::sha256_hex;
use ny_proofcheck::CheckOutcome;
use support::*;

/// `y0 - y1 >= 2^-20 > 0` everywhere: the property is UNSAT.
fn fixture(name: &str, vnn: &str) -> Fixture {
    Fixture::new(name, mlp_onnx(1.0 / 1_048_576.0, 0.0), vnn)
}

fn has(o: &CheckOutcome, code: &str) -> bool {
    reason_codes(o).iter().any(|c| c == code)
}

fn obj(pairs: Vec<(&str, Json)>) -> String {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), v);
    }
    json::to_string(&Json::Obj(m))
}

/// A header naming `schema`; `onnx_unavailable` writes the v2.1 input form.
fn header_v(fx: &Fixture, schema: &str, onnx_unavailable: Option<&str>) -> String {
    let fref = |p: &std::path::Path, b: &[u8]| {
        let mut m = Map::new();
        m.insert("path".into(), Json::Str(p.display().to_string()));
        m.insert("sha256".into(), Json::Str(sha256_hex(b)));
        m.insert("bytes".into(), Json::U64(b.len() as u64));
        Json::Obj(m)
    };
    let onnx = match onnx_unavailable {
        None => fref(&fx.onnx, &fx.onnx_bytes),
        Some(why) => {
            let mut m = Map::new();
            m.insert(
                "path".into(),
                Json::Str("[('f', 'a.onnx'), ('g', 'b.onnx')]".into()),
            );
            m.insert("unavailable".into(), Json::Str(why.into()));
            Json::Obj(m)
        }
    };
    obj(vec![
        ("kind", Json::Str("header".into())),
        ("schema", Json::Str(schema.into())),
        ("build", Json::Str("test-fixture".into())),
        ("profile", Json::Str("proof".into())),
        ("profile_source", Json::Str("cli".into())),
        ("lane", Json::Str("vnncomp_v1".into())),
        ("site_space", Json::Str("onnx-relu-input/v1".into())),
        ("input_space", Json::Str("onnx-input0-flat/v1".into())),
        ("onnx", onnx),
        ("vnnlib", fref(&fx.vnnlib, &fx.vnn_bytes)),
    ])
}

/// The writer's whole-region claim of `clause` (`opaque`, no literal).
fn claim(id: u64, clause: u64, src: &str) -> String {
    obj(vec![
        ("kind", Json::Str("node".into())),
        ("id", Json::U64(id)),
        ("region", Json::U64(0)),
        ("lits", Json::Arr(Vec::new())),
        ("close", Json::Str("opaque".into())),
        ("src", Json::Str(src.into())),
        ("dis", Json::Arr(vec![Json::U64(clause)])),
        ("hint", obj_json(vec![("beta", Json::Arr(Vec::new()))])),
    ])
}

fn obj_json(pairs: Vec<(&str, Json)>) -> Json {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), v);
    }
    Json::Obj(m)
}

fn legacy_verdict(required: &[u64], nodes: u64) -> String {
    obj(vec![
        ("kind", Json::Str("verdict".into())),
        ("source", Json::Str("beta_crown_verdict".into())),
        (
            "required",
            Json::Arr(required.iter().map(|x| Json::U64(*x)).collect()),
        ),
        ("min_margin", Json::Null),
        ("nodes", Json::U64(nodes)),
    ])
}

fn unavailable_clauses(list: &[(u64, Option<&str>)]) -> String {
    obj(vec![
        ("kind", Json::Str("clauses".into())),
        ("region", Json::U64(0)),
        (
            "list",
            Json::Arr(
                list.iter()
                    .map(|(id, why)| match why {
                        Some(why) => obj_json(vec![
                            ("id", Json::U64(*id)),
                            ("unavailable", Json::Str((*why).into())),
                        ]),
                        None => obj_json(vec![
                            ("id", Json::U64(*id)),
                            ("atoms", Json::Arr(vec![Json::Str(clause01())])),
                        ]),
                    })
                    .collect(),
            ),
        ),
    ])
}

/// The v2 legacy bundle the writer completes: empty sites, the canonical
/// clause, one whole-region legacy claim, a verdict requiring the clause.
fn legacy_v2_lines(fx: &Fixture) -> Vec<String> {
    vec![
        fx.header(),
        sites(&[]),
        clauses(0, &[(1, vec![clause01()])]),
        claim(7, 1, "legacy:beta_crown_verdict"),
        legacy_verdict(&[1], 1),
        published("unsat"),
    ]
}

/// The reproduced shape (`missing_sites`, exit 30) next to the completed one
/// (well-formed, `unconfirmed`, exit 10): the completion moves a legacy
/// bundle from the emergency code to the classic sub-moat, never to
/// `verified`. A partial stream of another lane plus the legacy claim is
/// still `unconfirmed` (never a cover gap).
#[test]
fn legacy_bundle_with_its_whole_region_claim_is_well_formed_and_unconfirmed() {
    let fx = fixture("legacy_v2", &mlp_vnnlib());
    // The pre-fix shape: header, verdict(required=[]), published, end.
    let old = vec![fx.header(), legacy_verdict(&[], 0), published("unsat")];
    let o = fx.check("old", &finish(&old));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(has(&o, "missing_sites"), "{}", o.report_json);
    // The completed shape.
    let o = fx.check("completed", &finish(&legacy_v2_lines(&fx)));
    assert_eq!(o.exit_code, 10, "{}", o.report_json);
    assert!(has(&o, "unconfirmed"), "{}", o.report_json);
    let r = report(&o);
    assert_eq!(r["verdict_source"].as_str(), Some("beta_crown_verdict"));
    assert_eq!(r["nodes"]["opaque"].as_u64(), Some(1), "{}", o.report_json);
    assert_eq!(r["cover"][0]["claimed_ok"], Json::Bool(true));
    assert_eq!(r["cover"][0]["confirmed_ok"], Json::Bool(false));
    // + a partial stream (one side of the split) of another lane.
    let mut lines = legacy_v2_lines(&fx);
    lines[1] = sites(&[(0, "z", 3, 1)]);
    lines.insert(
        3,
        node_line(
            0,
            &[L::R(0, 0, true), L::R(0, 1, true)],
            "bound",
            Some(&[1]),
            Some(&[1.0, 1.0]),
            None,
            false,
            1e-6,
        ),
    );
    let o = fx.check("partial", &finish(&lines));
    assert_eq!(o.exit_code, 10, "{}", o.report_json);
}

/// MUTATIONS of the completed legacy bundle: the claim is load-bearing.
/// Dropping it (no node at all) is `invalid` (`uncovered_source`, which used
/// to be `unsupported`), and so is the pre-fix verdict that requires nothing.
#[test]
fn legacy_bundle_mutations_dropping_the_claim_or_the_requirement_are_invalid() {
    let fx = fixture("legacy_mut", &mlp_vnnlib());
    let mut lines = legacy_v2_lines(&fx);
    lines.remove(3);
    lines[3] = legacy_verdict(&[1], 0);
    let o = fx.check("dropped", &finish(&lines));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(has(&o, "uncovered_source"), "{}", o.report_json);
    let mut lines = legacy_v2_lines(&fx);
    lines[4] = legacy_verdict(&[], 1);
    let o = fx.check("unrequired", &finish(&lines));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(has(&o, "required_mismatch"), "{}", o.report_json);
}

/// The v2.1 forms: an UNAVAILABLE clause (the producer could not render the
/// property) and an UNAVAILABLE header input (not one regular file). Neither
/// is read or parsed — the VNN-LIB here is VNN-LIB 2.0, which the checker's
/// own reader refuses, and the ONNX field is a dual-network tuple — and the
/// bundle is `unconfirmed`, never `verified`, even when a node claims the
/// whole region as `empty` or `bound`.
#[test]
fn v2_1_unavailable_bundles_are_well_formed_and_never_verified() {
    let v2_text =
        "(vnnlib-version <2.0>)\n(declare-network f\n  (declare-input X_f real [1, 2])\n  \
                   (declare-output Y_f real [1, 2])\n)\n(assert (<= Y_f[0,0] -5.0))\n";
    let fx = fixture("legacy_v21", v2_text);
    for (tag, onnx_unavailable, code) in [
        ("clause", None, "clauses_unavailable"),
        ("input", Some("not a regular file"), "inputs_unavailable"),
    ] {
        let lines = vec![
            header_v(&fx, "ny-proof-bundle/v2.1", onnx_unavailable),
            sites(&[]),
            unavailable_clauses(&[(0, Some("dual_network_property"))]),
            claim(3, 0, "legacy:relational_implication"),
            legacy_verdict(&[0], 1),
            published("unsat"),
        ];
        let o = fx.check(tag, &finish(&lines));
        assert_eq!(o.exit_code, 10, "{tag}: {}", o.report_json);
        assert!(has(&o, code), "{tag}: {}", o.report_json);
        assert!(has(&o, "clauses_unavailable"), "{tag}: {}", o.report_json);
        let r = report(&o);
        assert_eq!(r["cover"][0]["claimed_ok"], Json::Bool(true), "{tag}");
        assert_eq!(r["cover"][0]["confirmed_ok"], Json::Bool(false), "{tag}");
        assert!(
            o.done_line.contains("verdict=unconfirmed"),
            "{}",
            o.done_line
        );
        // Whole-region claims of every kind: still never verified.
        for close in ["empty", "bound", "root"] {
            let mut l = lines.clone();
            let dis = if close == "empty" {
                None
            } else {
                Some(&[0u64][..])
            };
            l.insert(4, node_line(9, &[], close, dis, None, None, false, 1.0));
            l[5] = legacy_verdict(&[0], 2);
            let o = fx.check(&format!("{tag}_{close}"), &finish(&l));
            assert_eq!(o.exit_code, 10, "{tag} {close}: {}", o.report_json);
        }
    }
}

/// MUTATIONS of a v2.1 bundle: a dropped claim or requirement is `invalid`;
/// a v2.1 form under a v2 header, an unavailable clause next to another
/// clause (or with atoms), and an unknown minor are `malformed`.
#[test]
fn v2_1_mutations_are_invalid_or_malformed() {
    let fx = fixture("legacy_v21_mut", &mlp_vnnlib());
    let base = |schema: &str, clauses_line: String| {
        vec![
            header_v(&fx, schema, None),
            sites(&[]),
            clauses_line,
            claim(3, 0, "legacy:nonlinear_interval_bab"),
            legacy_verdict(&[0], 1),
            published("unsat"),
        ]
    };
    let ok = base(
        "ny-proof-bundle/v2.1",
        unavailable_clauses(&[(0, Some("vnnlib_reparse"))]),
    );
    let o = fx.check("base", &finish(&ok));
    assert_eq!(o.exit_code, 10, "positive control: {}", o.report_json);
    // The claim dropped.
    let mut l = ok.clone();
    l.remove(3);
    l[3] = legacy_verdict(&[0], 0);
    let o = fx.check("dropped", &finish(&l));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    // The claim kept, the clause no longer required.
    let mut l = ok.clone();
    l[4] = legacy_verdict(&[], 1);
    let o = fx.check("unrequired", &finish(&l));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(has(&o, "required_mismatch"), "{}", o.report_json);
    // A claim with a literal is not a whole-region claim.
    let mut l = ok.clone();
    l[1] = sites(&[(0, "z", 3, 1)]);
    l[3] = node_line(
        3,
        &[L::R(0, 0, true)],
        "opaque",
        Some(&[0]),
        None,
        None,
        false,
        0.0,
    );
    let o = fx.check("literal_claim", &finish(&l));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(has(&o, "cover_gap"), "{}", o.report_json);
    // The forms under a v2 header.
    let l = base(
        "ny-proof-bundle/v2",
        unavailable_clauses(&[(0, Some("vnnlib_reparse"))]),
    );
    let o = fx.check("v2_clause", &finish(&l));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(has(&o, "unknown_field"), "{}", o.report_json);
    let mut l = ok.clone();
    l[0] = header_v(&fx, "ny-proof-bundle/v2", Some("not a regular file"));
    let o = fx.check("v2_input", &finish(&l));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(has(&o, "unknown_field"), "{}", o.report_json);
    // An unavailable clause is the ONLY clause.
    let l = base(
        "ny-proof-bundle/v2.1",
        unavailable_clauses(&[(0, Some("vnnlib_reparse")), (1, None)]),
    );
    let o = fx.check("mixed", &finish(&l));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(has(&o, "unavailable_clause_shape"), "{}", o.report_json);
    let mut l = ok.clone();
    l.insert(3, clauses(1, &[(4, vec![clause01()])]));
    let o = fx.check("two_records", &finish(&l));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(has(&o, "unavailable_clause_shape"), "{}", o.report_json);
    // ... with no atoms beside it.
    let mut l = ok.clone();
    l[2] = l[2].replace(
        "\"unavailable\"",
        "\"atoms\":[\"Y0:1,Y1:-1<=0\"],\"unavailable\"",
    );
    assert!(l[2].contains("atoms"), "{}", l[2]);
    let o = fx.check("atoms_too", &finish(&l));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(has(&o, "unknown_field"), "{}", o.report_json);
    // An unknown minor.
    let mut l = ok;
    l[0] = header_v(&fx, "ny-proof-bundle/v2.2", None);
    let o = fx.check("v2_2", &finish(&l));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(has(&o, "unknown_schema"), "{}", o.report_json);
}

/// v2.1 is a superset: a v2.1 bundle with ordinary clauses and checkable
/// nodes is checked exactly like v2 (here: verified), and its dropped leaf
/// is the same cover gap.
#[test]
fn v2_1_with_ordinary_clauses_is_checked_like_v2() {
    let fx = fixture("legacy_v21_superset", &mlp_vnnlib());
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
    let v2 = fx.check("v2", &finish(&lines));
    assert_eq!(v2.exit_code, 0, "positive control: {}", v2.report_json);
    lines[0] = header_v(&fx, "ny-proof-bundle/v2.1", None);
    let o = fx.check("v21", &finish(&lines));
    assert_eq!(o.exit_code, 0, "{}", o.report_json);
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves()[..3]);
    lines[0] = header_v(&fx, "ny-proof-bundle/v2.1", None);
    let o = fx.check("v21_gap", &finish(&lines));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(has(&o, "cover_gap"), "{}", o.report_json);
}

/// review(legacy-bundle) ADVERSARIAL: the v2.1 `unavailable` forms can only
/// DOWNGRADE, never upgrade. A lane stream that is a complete, confirmable
/// cover (`verified` under v2, the positive control) proves nothing once one
/// header input is marked unavailable: its literal-carrying nodes are no
/// whole-region claim, so alone it is a cover gap (`invalid`) — which is why
/// the writer adds the whole-region claims whenever an input is unavailable
/// — and with a whole-region claim beside it only `unconfirmed`, also when
/// `--onnx` names the real network (an override cannot bind an input the
/// producer did not hash). The available input's hash still binds (a wrong
/// vnnlib sha is `hash_mismatch`, malformed). The same under an unavailable
/// CLAUSE: a gap alone, `unconfirmed` with a whole-region claim. Nothing in an
/// unavailable bundle ever reaches `verified`.
#[test]
fn v2_1_unavailable_forms_never_upgrade_a_confirmable_stream() {
    let fx = fixture("legacy_v21_never_upgrade", &mlp_vnnlib());
    let lines = mlp_bundle_lines(&fx, &mlp_leaves());
    let o = fx.check("v2_control", &finish(&lines));
    assert_eq!(o.exit_code, 0, "positive control: {}", o.report_json);
    // `lines` with a whole-region `bound` claim of clause 1 before the verdict.
    let with_claim = |mut l: Vec<String>| {
        let n = l.len();
        l.insert(
            n - 2,
            node_line(9, &[], "bound", Some(&[1]), None, None, false, 1.0),
        );
        let n = l.len();
        l[n - 2] = verdict(&[1], 5);
        l
    };
    for (tag, onnx_unavailable, clause_unavailable, code) in [
        (
            "onnx",
            Some("not a regular file"),
            false,
            "inputs_unavailable",
        ),
        ("clause", None, true, "clauses_unavailable"),
    ] {
        let mut l = lines.clone();
        l[0] = header_v(&fx, "ny-proof-bundle/v2.1", onnx_unavailable);
        if clause_unavailable {
            l[2] = unavailable_clauses(&[(1, Some("hostile_producer"))]);
        }
        // The confirmable stream alone: a gap, never verified.
        let o = fx.check(&format!("{tag}_stream"), &finish(&l));
        assert_eq!(o.exit_code, 20, "{tag}: {}", o.report_json);
        assert!(has(&o, "cover_gap"), "{tag}: {}", o.report_json);
        assert!(has(&o, code), "{tag}: {}", o.report_json);
        // With a whole-region claim: unconfirmed, never verified.
        let l = with_claim(l);
        let o = fx.check(&format!("{tag}_claimed"), &finish(&l));
        assert_eq!(o.exit_code, 10, "{tag}: {}", o.report_json);
        assert!(
            o.done_line.contains("verdict=unconfirmed"),
            "{tag}: {}",
            o.done_line
        );
        let real_onnx = fx.onnx.clone();
        let o = fx.check_with(&format!("{tag}_claimed_override"), &finish(&l), |a| {
            a.onnx = Some(real_onnx);
        });
        assert_eq!(
            o.exit_code, 10,
            "{tag}: an override binds nothing: {}",
            o.report_json
        );
        // The available vnnlib's hash still binds.
        let mut bad = l.clone();
        bad[0] = bad[0].replace(&sha256_hex(&fx.vnn_bytes), &"0".repeat(64));
        assert_ne!(bad[0], l[0], "{tag}: the vnnlib sha was replaced");
        let o = fx.check(&format!("{tag}_claimed_vnnlib_sha"), &finish(&bad));
        assert_eq!(o.exit_code, 30, "{tag}: {}", o.report_json);
        assert!(has(&o, "hash_mismatch"), "{tag}: {}", o.report_json);
    }
}

/// review(eng16) CROSS-BRANCH: the legacy forms (#b2-legacy-wellformed) over a
/// network ONLY the checker's own lowering (#b2-checker-lowering) admits —
/// `support::routed_mlp_onnx`, the base MLP's function through Reshape, a
/// constant-left MatMul, Gather, Slice and Concat (exit 31 `onnx_lowering`
/// before that lowering). Positive control: the lane stream verifies through
/// the routed network. The completed v2 legacy bundle now passes the lowering
/// (its Select ops are in the report) and is `unconfirmed` (10), never
/// `verified` — the shape the `proof_check_real` integrator rows nn4sys /
/// sbench / linz pin since the eng16 merge. Negative control: the same network
/// declared at opset 6 stops at the lowering's legacy-broadcast refusal (31
/// `onnx_lowering`, after the bundle parsed: well-formed, never 30). The v2.1
/// `unavailable` forms never upgrade the confirmable stream: a cover gap
/// alone, at most `unconfirmed` with a whole-region claim.
#[test]
fn legacy_forms_over_a_network_only_the_new_lowering_admits() {
    const B0: f64 = 1.0 / 1_048_576.0;
    let fx = Fixture::new("legacy_routed", routed_mlp_onnx(B0, 0.0), &mlp_vnnlib());
    let stream = mlp_bundle_lines(&fx, &mlp_leaves());
    let o = fx.check("v2_control", &finish(&stream));
    assert_eq!(o.exit_code, 0, "positive control: {}", o.report_json);
    // The completed legacy v2 bundle: past the lowering, unconfirmed.
    let o = fx.check("completed", &finish(&legacy_v2_lines(&fx)));
    assert_eq!(o.exit_code, 10, "{}", o.report_json);
    assert!(has(&o, "unconfirmed"), "{}", o.report_json);
    assert!(!has(&o, "onnx_lowering"), "{}", o.report_json);
    assert!(
        o.done_line.contains("verdict=unconfirmed"),
        "{}",
        o.done_line
    );
    let r = report(&o);
    assert_eq!(
        r["network"]["ops"]["Select"].as_u64(),
        Some(4),
        "the routing forms lowered: {}",
        o.report_json
    );
    assert_eq!(r["nodes"]["opaque"].as_u64(), Some(1), "{}", o.report_json);
    assert_eq!(r["cover"][0]["confirmed_ok"], Json::Bool(false));
    // Negative control: opset 6 is refused at the lowering, well-formed.
    let old = Fixture::new(
        "legacy_routed_opset6",
        routed_mlp_onnx_opset(B0, 0.0, 6),
        &mlp_vnnlib(),
    );
    let o = old.check("completed_opset6", &finish(&legacy_v2_lines(&old)));
    assert_eq!(o.exit_code, 31, "{}", o.report_json);
    assert_eq!(reason_codes(&o), ["onnx_lowering"], "{}", o.report_json);
    assert!(o.report_json.contains("opset 7"), "{}", o.report_json);
    // The v2.1 forms over the routed network: never an upgrade.
    for (tag, onnx_unavailable, clause_unavailable, code) in [
        (
            "onnx",
            Some("not a regular file"),
            false,
            "inputs_unavailable",
        ),
        ("clause", None, true, "clauses_unavailable"),
    ] {
        let mut l = stream.clone();
        l[0] = header_v(&fx, "ny-proof-bundle/v2.1", onnx_unavailable);
        if clause_unavailable {
            l[2] = unavailable_clauses(&[(1, Some("hostile_producer"))]);
        }
        let o = fx.check(&format!("routed_{tag}_stream"), &finish(&l));
        assert_eq!(o.exit_code, 20, "{tag}: {}", o.report_json);
        assert!(has(&o, "cover_gap"), "{tag}: {}", o.report_json);
        assert!(has(&o, code), "{tag}: {}", o.report_json);
        let n = l.len();
        l.insert(n - 2, claim(9, 1, "legacy:beta_crown_verdict"));
        let n = l.len();
        l[n - 2] = verdict(&[1], 5);
        let o = fx.check(&format!("routed_{tag}_claimed"), &finish(&l));
        assert_eq!(o.exit_code, 10, "{tag}: {}", o.report_json);
        assert!(
            o.done_line.contains("verdict=unconfirmed"),
            "{tag}: {}",
            o.done_line
        );
    }
}
