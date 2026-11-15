// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ADVERSARIAL bundles (review(b2) of the B2.1 checker): every test hands
//! the checker a proof that is wrong, incomplete or ill-formed in one
//! specific way and pins that it is NEVER `verified` (exit 0), with the
//! specific verdict and reason the downgrade must carry. Positive controls
//! sit next to the attacks they bracket, so a test that "rejects" because
//! the fixture is broken cannot pass. Every fixture is synthetic (no
//! `benchmarks/`).

mod support;

use ny_cert::json::Json;
use ny_cert::Rat;
use ny_proofcheck::ibp::eval_point;
use ny_proofcheck::testkit::net_from_onnx;
use ny_proofcheck::Verdict;
use support::*;

const B0: f64 = 1.0 / 1_048_576.0; // 2^-20

fn base(name: &str, b0: f64, b1: f64) -> Fixture {
    Fixture::new(name, mlp_onnx(b0, b1), &mlp_vnnlib())
}

fn codes(o: &ny_proofcheck::CheckOutcome) -> Vec<String> {
    reason_codes(o)
}

fn has(o: &ny_proofcheck::CheckOutcome, code: &str) -> bool {
    codes(o).iter().any(|c| c == code)
}

fn count(o: &ny_proofcheck::CheckOutcome, key: &str) -> u64 {
    report(o)["nodes"][key].as_u64().unwrap_or(u64::MAX)
}

/// The witnesses of every `cover_gap` reason, as `(site id, index, side)`.
fn gap_witnesses(o: &ny_proofcheck::CheckOutcome) -> Vec<Vec<(u64, u64, u64)>> {
    let r = report(o);
    let Json::Arr(reasons) = &r["reasons"] else {
        return Vec::new();
    };
    reasons
        .iter()
        .filter(|x| x["code"].as_str() == Some("cover_gap"))
        .map(|x| match &x["detail"]["witness"] {
            Json::Arr(w) => w
                .iter()
                .filter_map(|l| match l {
                    Json::Arr(v) if v.len() == 3 => {
                        Some((v[0].as_u64()?, v[1].as_u64()?, v[2].as_u64()?))
                    }
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Cover attacks
// ---------------------------------------------------------------------------

/// Four leaves, three of them overlapping on `z0 >= 0`, and no leaf for the
/// cell `z0 <= 0, z1 <= 0`: overlapping members must not be read as a cover.
/// The witness names exactly the uncovered cell.
#[test]
fn adversarial_overlapping_but_incomplete_leaves_are_a_gap() {
    let fx = base("adv_overlap", B0, 0.0);
    let leaves = vec![
        (vec![L::R(0, 0, true)], vec![1.0]),
        (vec![L::R(0, 0, true), L::R(0, 1, false)], vec![1.0, 0.0]),
        (vec![L::R(0, 0, true), L::R(0, 1, true)], vec![1.0, 1.0]),
        (vec![L::R(0, 0, false), L::R(0, 1, true)], vec![0.0, 1.0]),
    ];
    let o = fx.check("b", &finish(&mlp_bundle_lines(&fx, &leaves)));
    assert_eq!(o.verdict, Verdict::Invalid, "{}", o.report_json);
    assert_eq!(o.exit_code, 20);
    let w = gap_witnesses(&o);
    assert_eq!(w.len(), 1, "{}", o.report_json);
    let mut cell = w[0].clone();
    cell.sort_unstable();
    assert_eq!(cell, vec![(0, 0, 0), (0, 1, 0)], "{}", o.report_json);
    // Positive control: adding the missing cell's leaf closes the cover.
    let mut full = leaves;
    full.push((vec![L::R(0, 0, false), L::R(0, 1, false)], vec![0.0, 0.0]));
    let o = fx.check("full", &finish(&mlp_bundle_lines(&fx, &full)));
    assert_eq!(o.exit_code, 0, "{}", o.report_json);
}

/// Input-split literals are keyed by their EXACT threshold: `x0 >= 0.5` and
/// `x0 <= 0.25` leave `(0.25, 0.5)` uncovered even though each node's own
/// discharge is true and confirmed. The same threshold spelled two ways
/// (`0.5` / `5e-1`) is the same f64 and does cover.
#[test]
fn adversarial_mismatched_input_thresholds_are_a_gap() {
    let fx = base("adv_thresholds", B0, 0.0);
    let lines = |t_lo: f64| {
        let mut l = vec![
            fx.header(),
            sites(&[(0, "z", 3, 1)]),
            clauses(0, &[(1, vec![clause01()])]),
        ];
        l.push(node_line(
            0,
            &[L::X(0, 0.5, true)],
            "bound",
            Some(&[1]),
            None,
            None,
            false,
            1.0,
        ));
        l.push(node_line(
            1,
            &[L::X(0, t_lo, false)],
            "bound",
            Some(&[1]),
            None,
            None,
            false,
            1.0,
        ));
        l.push(verdict(&[1], 2));
        l.push(published("unsat"));
        l
    };
    let o = fx.check("gap", &finish(&lines(0.25)));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(has(&o, "cover_gap"), "{}", o.report_json);
    assert_eq!(
        count(&o, "confirmed"),
        2,
        "both discharges are true; only the cover fails: {}",
        o.report_json
    );
    // Positive control: one threshold, two spellings.
    let mut same = lines(0.5);
    same[4] = same[4].replacen("\"0.5\"", "\"5e-1\"", 1);
    assert!(same[4].contains("\"5e-1\""), "{}", same[4]);
    let o = fx.check("same", &finish(&same));
    assert_eq!(o.exit_code, 0, "{}", o.report_json);
}

/// A nested frame's `subtree` leaf stands for cells its nested nodes must
/// cover; without them the cell is a gap, whatever the leaf claims.
#[test]
fn adversarial_subtree_leaf_without_nested_nodes_is_a_gap() {
    let fx = base("adv_subtree", B0, 0.0);
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
    let (lits, _) = mlp_leaves()[3].clone();
    lines[6] = node_line(3, &lits, "subtree", Some(&[]), None, None, false, 1.0);
    let o = fx.check("b", &finish(&lines));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(has(&o, "cover_gap"), "{}", o.report_json);
}

/// A token whose `required` list omits a clause the VNN-LIB contains cannot
/// speak for the property (the omitted disjunct is never checked).
#[test]
fn adversarial_required_list_missing_a_spec_clause_is_invalid() {
    let fx = base("adv_required", B0, 0.0);
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
    let n = lines.len();
    lines[n - 2] = verdict(&[], 4);
    let o = fx.check("b", &finish(&lines));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(has(&o, "required_mismatch"), "{}", o.report_json);
}

/// Nodes and clauses for a region the spec does not have: never counted.
#[test]
fn adversarial_region_the_spec_does_not_have_is_refused() {
    let fx = base("adv_region", B0, 0.0);
    // A node in region 1 with no clauses record: malformed.
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
    lines[3] = lines[3].replacen("\"region\":0", "\"region\":1", 1);
    assert!(lines[3].contains("\"region\":1"), "{}", lines[3]);
    let o = fx.check("no_clauses", &finish(&lines));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(has(&o, "unknown_region"), "{}", o.report_json);
    // With a clauses record for region 1: the clause matches nothing.
    lines.insert(3, clauses(1, &[(1, vec![clause01()])]));
    let o = fx.check("with_clauses", &finish(&lines));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(has(&o, "clause_mismatch"), "{}", o.report_json);
}

// ---------------------------------------------------------------------------
// Discharge attacks
// ---------------------------------------------------------------------------

/// A bound that holds only on a SMALLER region, claimed for a larger one:
/// `x0 >= 0.5` truly discharges the clause, `x0 <= 0.5` does not (the
/// property is false there), and copying the good node's margin hint onto
/// the bad node changes nothing. Exactly one node confirms and the witness
/// is a rigorous counterexample.
#[test]
fn adversarial_bound_valid_only_on_a_smaller_region_is_refuted() {
    let fx = base("adv_smaller", B0, 0.25);
    let lines = vec![
        fx.header(),
        sites(&[(0, "z", 3, 1)]),
        clauses(0, &[(1, vec![clause01()])]),
        node_line(
            0,
            &[L::X(0, 0.5, true)],
            "bound",
            Some(&[1]),
            None,
            None,
            false,
            0.75,
        ),
        node_line(
            1,
            &[L::X(0, 0.5, false)],
            "bound",
            Some(&[1]),
            Some(&[1.0]),
            None,
            true,
            0.75,
        ),
        verdict(&[1], 2),
        published("unsat"),
    ];
    let o = fx.check("b", &finish(&lines));
    assert_eq!(o.verdict, Verdict::Refuted, "{}", o.report_json);
    assert_eq!(o.exit_code, 21);
    assert_eq!(count(&o, "confirmed"), 1, "{}", o.report_json);
    let r = report(&o);
    let Json::Arr(unc) = &r["unconfirmed"] else {
        panic!("unconfirmed list: {}", o.report_json);
    };
    assert_eq!(unc.len(), 1, "{}", o.report_json);
    assert_eq!(unc[0]["node"].as_u64(), Some(1), "{}", o.report_json);
    // The counterexample lies in the bad node's half and violates the spec
    // in exact arithmetic: relu(x0+x1) + relu(x0-x1) + b0 - b1 <= 0.
    let Json::Arr(x) = &r["refutation"]["counterexample"]["x"] else {
        panic!("no counterexample: {}", o.report_json);
    };
    let xs: Vec<f64> = x
        .iter()
        .map(|v| match v {
            Json::F64(f) => *f,
            Json::U64(u) => *u as f64,
            Json::I64(i) => *i as f64,
            other => panic!("x entry {other:?}"),
        })
        .collect();
    let rat = |v: f64| Rat::from_f64_exact(v).expect("finite");
    let relu = |z: Rat| if z.is_negative() { Rat::ZERO } else { z };
    let z0 = rat(xs[0]).add(rat(xs[1])).expect("add");
    let z1 = rat(xs[0]).add(rat(xs[1]).neg()).expect("add");
    let m = relu(z0)
        .add(relu(z1))
        .and_then(|s| s.add(rat(B0)))
        .and_then(|s| s.add(rat(0.25).neg()))
        .expect("margin");
    assert!(xs[0] <= 0.5, "{xs:?}");
    assert!(
        !m.is_positive(),
        "not a counterexample: {xs:?} margin {m:?}"
    );
}

/// An `empty` claim (`dis: "*"`) on a region that is NOT empty is never
/// counted as empty; with a false property it confirms nothing.
#[test]
fn adversarial_emptiness_claim_on_a_nonempty_region_confirms_nothing() {
    let fx = base("adv_empty", B0, 0.25);
    let leaves = mlp_leaves();
    let mut lines = mlp_bundle_lines(&fx, &leaves);
    lines[6] = node_line(3, &leaves[3].0, "empty", None, None, None, false, 1.0);
    let o = fx.check("b", &finish(&lines));
    assert_ne!(o.exit_code, 0, "{}", o.report_json);
    assert_eq!(count(&o, "empty"), 0, "{}", o.report_json);
    assert_eq!(count(&o, "confirmed"), 0, "{}", o.report_json);
    // Positive control: a TRULY empty node (z2 = x0 + x1 + 5 >= 3 on the box,
    // so `z2 <= 0` is empty) is certified empty.
    let fx = base("adv_empty_ok", B0, 0.0);
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
    let n = lines.len();
    lines.insert(
        n - 2,
        node_line(
            9,
            &[L::R(0, 2, false)],
            "empty",
            None,
            None,
            None,
            false,
            1.0,
        ),
    );
    let o = fx.check("ok", &finish(&lines));
    assert_eq!(o.exit_code, 0, "{}", o.report_json);
    assert_eq!(count(&o, "empty"), 1, "{}", o.report_json);
}

/// A true minimum margin of exactly ZERO is not a proof: the unsafe atom
/// `Y0 - Y1 <= 0` holds at the minimiser, so the property is SAT. The
/// checker's `L > 0` is strict, so no leaf touching the minimiser confirms.
/// (Its rigorous probe cannot certify an EXACT zero either — directed
/// rounding of `0` products straddles it — so `unconfirmed` is as right as
/// `refuted`; only `verified` would be wrong.)
#[test]
fn adversarial_zero_margin_is_never_verified() {
    let fx = base("adv_zero_margin", 0.0, 0.0);
    let o = fx.check("b", &finish(&mlp_bundle_lines(&fx, &mlp_leaves())));
    assert!(
        matches!(o.verdict, Verdict::Unconfirmed | Verdict::Refuted),
        "{}",
        o.report_json
    );
    assert_ne!(o.exit_code, 0);
    assert!(count(&o, "confirmed") < 4, "{}", o.report_json);
    // A true minimum of -2^-20 is a refutation the probe does find.
    let fx = base("adv_neg_margin", 0.0, B0);
    let o = fx.check("b", &finish(&mlp_bundle_lines(&fx, &mlp_leaves())));
    assert_eq!(o.verdict, Verdict::Refuted, "{}", o.report_json);
}

/// A true margin of one f32 subnormal (2^-149) is positive: whatever the
/// checker can or cannot confirm, it must never claim a counterexample.
#[test]
fn adversarial_subnormal_margin_is_never_refuted() {
    let tiny = f64::from(f32::from_bits(1)); // 2^-149
    let fx = base("adv_subnormal", tiny, 0.0);
    let o = fx.check("b", &finish(&mlp_bundle_lines(&fx, &mlp_leaves())));
    assert!(
        matches!(o.verdict, Verdict::Verified | Verdict::Unconfirmed),
        "{}",
        o.report_json
    );
    assert!(report(&o)["refutation"]["counterexample"].is_null());
}

// ---------------------------------------------------------------------------
// Ill-formed bundles: fail closed
// ---------------------------------------------------------------------------

/// Duplicate node ids, both sides of one site in one node, out-of-range
/// indices, unknown site / clause ids, a site naming a tensor that is not a
/// Relu input or not in the net, a wrong site size, and trailing bytes after
/// `end` are all `malformed` (exit 30) with their own reason.
#[test]
fn adversarial_ill_formed_structure_fails_closed() {
    let fx = base("adv_structure", B0, 0.0);
    let leaves = mlp_leaves();
    type Edit = Box<dyn Fn(&mut Vec<String>)>;
    let cases: Vec<(&str, &str, Edit)> = vec![
        (
            "dup_id",
            "duplicate_node_id",
            Box::new(|l: &mut Vec<String>| {
                l[4] = l[4].replacen("\"id\":1", "\"id\":0", 1);
            }),
        ),
        (
            "both_sides",
            "both_sides",
            Box::new(|l: &mut Vec<String>| {
                l[3] = node_line(
                    0,
                    &[L::R(0, 0, true), L::R(0, 0, false)],
                    "bound",
                    Some(&[1]),
                    None,
                    None,
                    false,
                    1.0,
                );
            }),
        ),
        (
            "index",
            "literal_index",
            Box::new(|l: &mut Vec<String>| {
                l[3] = node_line(
                    0,
                    &[L::R(0, 3, true)],
                    "bound",
                    Some(&[1]),
                    None,
                    None,
                    false,
                    1.0,
                );
            }),
        ),
        (
            "x_index",
            "literal_index",
            Box::new(|l: &mut Vec<String>| {
                l[3] = node_line(
                    0,
                    &[L::X(2, 0.0, true)],
                    "bound",
                    Some(&[1]),
                    None,
                    None,
                    false,
                    1.0,
                );
            }),
        ),
        (
            "site_id",
            "unknown_site_id",
            Box::new(|l: &mut Vec<String>| {
                l[3] = node_line(
                    0,
                    &[L::R(7, 0, true)],
                    "bound",
                    Some(&[1]),
                    None,
                    None,
                    false,
                    1.0,
                );
            }),
        ),
        (
            "clause_id",
            "unknown_clause_id",
            Box::new(|l: &mut Vec<String>| {
                l[3] = l[3].replacen("\"dis\":[1]", "\"dis\":[9]", 1);
            }),
        ),
        (
            "input_site",
            "not_a_relu_input",
            Box::new(|l: &mut Vec<String>| {
                l[1] = sites(&[(0, "x", 2, 1)]);
            }),
        ),
        (
            "no_tensor",
            "unknown_site_tensor",
            Box::new(|l: &mut Vec<String>| {
                l[1] = sites(&[(0, "nope", 3, 1)]);
            }),
        ),
        (
            "site_size",
            "site_size",
            Box::new(|l: &mut Vec<String>| {
                l[1] = sites(&[(0, "z", 4, 1)]);
            }),
        ),
        (
            "empty_not_star",
            "bad_field",
            Box::new(|l: &mut Vec<String>| {
                l[3] = l[3].replacen("\"close\":\"bound\"", "\"close\":\"empty\"", 1);
            }),
        ),
    ];
    for (name, code, edit) in &cases {
        let mut lines = mlp_bundle_lines(&fx, &leaves);
        let before = lines.clone();
        edit(&mut lines);
        assert_ne!(lines, before, "{name}: the edit did not apply");
        let o = fx.check(name, &finish(&lines));
        assert_eq!(o.exit_code, 30, "{name}: {}", o.report_json);
        assert!(has(&o, code), "{name}: want `{code}`: {}", o.report_json);
    }
    // Trailing bytes after a valid `end`.
    let mut bytes = finish(&mlp_bundle_lines(&fx, &leaves));
    bytes.extend_from_slice(br#"{"kind":"node","id":99}"#);
    bytes.push(b'\n');
    let o = fx.check("trailing", &bytes);
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(has(&o, "truncated"), "{}", o.report_json);
    // Positive control: the unedited bundle verifies.
    let o = fx.check("control", &finish(&mlp_bundle_lines(&fx, &leaves)));
    assert_eq!(o.exit_code, 0, "{}", o.report_json);
}

/// Non-finite numbers cannot enter a bundle: a NaN / inf input threshold is
/// `bad_literal`, a bare `NaN` token is a JSON parse failure; finite
/// extremes in hints (±1e308 β and α, a huge claimed margin) on a FALSE
/// property confirm nothing.
#[test]
fn adversarial_nonfinite_and_extreme_numbers_never_confirm() {
    let fx = base("adv_nonfinite", B0, 0.0);
    for (name, t) in [
        ("nan", f64::NAN),
        ("inf", f64::INFINITY),
        ("ninf", f64::NEG_INFINITY),
    ] {
        let lines = vec![
            fx.header(),
            sites(&[(0, "z", 3, 1)]),
            clauses(0, &[(1, vec![clause01()])]),
            node_line(
                0,
                &[L::X(0, t, true)],
                "bound",
                Some(&[1]),
                None,
                None,
                false,
                1.0,
            ),
            node_line(
                1,
                &[L::X(0, t, false)],
                "bound",
                Some(&[1]),
                None,
                None,
                false,
                1.0,
            ),
            verdict(&[1], 2),
            published("unsat"),
        ];
        let o = fx.check(name, &finish(&lines));
        assert_eq!(o.exit_code, 30, "{name}: {}", o.report_json);
        assert!(has(&o, "bad_literal"), "{name}: {}", o.report_json);
    }
    let mut lines = mlp_bundle_lines(&fx, &mlp_leaves());
    lines[3] = lines[3].replacen("\"margin\":1e-6", "\"margin\":NaN", 1);
    assert!(lines[3].contains("NaN"), "{}", lines[3]);
    let o = fx.check("nan_token", &finish(&lines));
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(has(&o, "parse"), "{}", o.report_json);
    // Extremes on a false property (true minimum 2^-20 - 0.25 < 0).
    let fx = base("adv_extremes", B0, 0.25);
    let mut lines = vec![
        fx.header(),
        sites(&[(0, "z", 3, 1)]),
        clauses(0, &[(1, vec![clause01()])]),
        alpha_line(4, 0, &[0, 1, 2], &[1e308, -1e308, 0.5]),
    ];
    for (k, (lits, _)) in mlp_leaves().into_iter().enumerate() {
        let beta = if k % 2 == 0 {
            vec![1e308, -1e308]
        } else {
            vec![f64::MIN_POSITIVE, 1e-308]
        };
        lines.push(node_line(
            k as u64,
            &lits,
            "bound",
            Some(&[1]),
            Some(&beta),
            Some(4),
            true,
            1e308,
        ));
    }
    lines.push(verdict(&[1], 4));
    lines.push(published("unsat"));
    let o = fx.check("extremes", &finish(&lines));
    assert_eq!(count(&o, "confirmed"), 0, "{}", o.report_json);
    assert_ne!(o.exit_code, 0, "{}", o.report_json);
}

/// A legacy (uncovered) token that streams NO nodes claims nothing: since the
/// producer's writer completes every published `unsat` with a whole-region
/// claim per clause (legacy well-formedness), a node-less legacy bundle is a
/// DROPPED claim — `invalid` (`uncovered_source`; it was `unsupported`
/// before the completion existed), never checked as if it were authorized.
/// B2.2: a legacy
/// token whose lane DID stream a proof is checked like any other — the
/// producer's authority never enters the checker's verdict (every discharge
/// and the cover are re-derived), and the report records the source. An
/// `--onnx` override that is not the header's file is a hash mismatch.
#[test]
fn adversarial_legacy_source_and_swapped_inputs_are_refused() {
    let fx = base("adv_legacy", B0, 0.0);
    let legacy = |mut lines: Vec<String>| {
        let n = lines.len();
        lines[n - 2] = lines[n - 2].replacen(
            "\"source\":\"authorized\"",
            "\"source\":\"legacy_uncovered\"",
            1,
        );
        assert!(
            lines[n - 2].contains("legacy_uncovered"),
            "{}",
            lines[n - 2]
        );
        lines
    };
    let lines = legacy(mlp_bundle_lines(&fx, &[]));
    let o = fx.check("legacy", &finish(&lines));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(has(&o, "uncovered_source"), "{}", o.report_json);
    // The same legacy token with the lane's node stream: checked, verified.
    let lines = legacy(mlp_bundle_lines(&fx, &mlp_leaves()));
    let o = fx.check("legacy_streamed", &finish(&lines));
    assert_eq!(o.exit_code, 0, "{}", o.report_json);
    assert_eq!(
        report(&o)["verdict_source"].as_str(),
        Some("legacy_uncovered"),
        "{}",
        o.report_json
    );
    // ... and a legacy stream missing a leaf is a gap like any other.
    let lines = legacy(mlp_bundle_lines(&fx, &mlp_leaves()[..3]));
    let o = fx.check("legacy_gap", &finish(&lines));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(has(&o, "cover_gap"), "{}", o.report_json);
    // A different network passed with --onnx.
    let other = fx.dir.join("other.onnx");
    std::fs::write(&other, mlp_onnx(B0, 0.5)).expect("write other onnx");
    let o = fx.check_with(
        "swapped",
        &finish(&mlp_bundle_lines(&fx, &mlp_leaves())),
        |a| a.onnx = Some(other.clone()),
    );
    assert_eq!(o.exit_code, 30, "{}", o.report_json);
    assert!(has(&o, "hash_mismatch"), "{}", o.report_json);
}

// ---------------------------------------------------------------------------
// Arithmetic edges
// ---------------------------------------------------------------------------

/// The VNN-LIB box is widened OUTWARD for bounds and INWARD for probes, in
/// exact rational terms, for a non-dyadic decimal and for a strict bound.
#[test]
fn adversarial_vnnlib_box_encloses_the_exact_decimals() {
    let text = "(declare-const X_0 Real)\n(declare-const X_1 Real)\n\
                (declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
                (assert (>= X_0 0.1))\n(assert (<= X_0 0.7))\n\
                (assert (> X_1 0.5))\n(assert (<= X_1 1))\n\
                (assert (<= Y_0 Y_1))\n";
    let s = ny_proofcheck::vnnlib::parse(text).expect("parses");
    let r = &s.regions[0];
    let rat = |v: f64| Rat::from_f64_exact(v).expect("finite");
    let q = |n: i128, d: i128| Rat::new(n, d).expect("rat");
    assert!(rat(r.lo_out[0]) < q(1, 10) && q(1, 10) < rat(r.lo_in[0]));
    assert!(rat(r.hi_in[0]) < q(7, 10) && q(7, 10) < rat(r.hi_out[0]));
    // Strict dyadic bound: outward keeps the closure, inward excludes it.
    assert_eq!(r.lo_out[1], 0.5);
    assert!(r.lo_in[1] > 0.5);
    assert_eq!((r.hi_out[1], r.hi_in[1]), (1.0, 1.0));
}

/// An 8-layer chain of 3e38 weights overflows f64 in the point evaluation,
/// and a ZERO weight then meets `inf` (`0 · inf = NaN`). The real network is
/// finite and SAFE (z = 0·a + 5, y0 - y1 = relu(5) - 1 = 4 > 0), so the
/// enclosure of `relu(z)` must contain 5 or be non-finite — never the
/// laundered `[5, 0]` that `f64::max` makes of `[5, NaN]` — and the checker
/// must never report `refuted` (it cannot prove the row either: the root
/// bounds are not finite, so the honest verdict is `unconfirmed`).
#[test]
fn adversarial_overflow_nan_never_manufactures_a_refutation() {
    let w = 3.0e38;
    let mut nodes = Vec::new();
    let mut inits = Vec::new();
    let names: Vec<String> = (0..=8).map(|k| format!("a{k}")).collect();
    for k in 1..=8 {
        let (wn, bn) = (format!("W{k}"), format!("B{k}"));
        nodes.push(N {
            op: "Gemm",
            inputs: vec![names[k - 1].clone(), wn.clone(), bn.clone()],
            outputs: vec![names[k].clone()],
            attrs: vec![A::I("transB", 1)],
        });
        inits.push(tf(&wn, &[1, 1], &[w]));
        inits.push(tf(&bn, &[1], &[0.0]));
    }
    nodes.push(node(
        "Gemm",
        &["a8", "Wz", "Bz"],
        &["z"],
        vec![A::I("transB", 1)],
    ));
    nodes.push(node("Relu", &["z"], &["h"], vec![]));
    nodes.push(node(
        "Gemm",
        &["h", "Wy", "By"],
        &["y"],
        vec![A::I("transB", 1)],
    ));
    inits.push(tf("Wz", &[1, 1], &[0.0]));
    inits.push(tf("Bz", &[1], &[5.0]));
    inits.push(tf("Wy", &[2, 1], &[1.0, 0.0]));
    inits.push(tf("By", &[2], &[-1.0, 0.0]));
    let onnx = model("a0", &[1, 1], &nodes, &inits, "y", &[1, 2]);
    // The rigorous point evaluation keeps the real value or gives up.
    let net = net_from_onnx(&onnx).expect("lowers");
    let b = eval_point(&net, &[1.0e38]);
    let h = net.by_name["h"];
    let (lo, hi) = (b.lo[h][0], b.hi[h][0]);
    assert!(
        !(lo.is_finite() && hi.is_finite()) || (lo <= 5.0 && 5.0 <= hi),
        "relu(z) enclosure [{lo:e}, {hi:e}] excludes the real value 5"
    );
    // End to end: never refuted.
    let vnn = robustness_vnnlib(&["1e38"], &["1.5e38"], 2, 0, &[1]);
    let fx = Fixture::new("adv_overflow", onnx, &vnn);
    let lines = vec![
        fx.header(),
        sites(&[(0, "z", 1, 9)]),
        clauses(0, &[(1, vec![clause01()])]),
        node_line(0, &[], "root", Some(&[1]), None, None, false, 4.0),
        verdict(&[1], 1),
        published("unsat"),
    ];
    let o = fx.check("b", &finish(&lines));
    assert_ne!(o.verdict, Verdict::Refuted, "{}", o.report_json);
    assert!(report(&o)["refutation"]["counterexample"].is_null());
    assert_eq!(o.verdict, Verdict::Unconfirmed, "{}", o.report_json);
}

/// The a-priori rounding CHARGE is load-bearing: a Gemm/Relu/Gemm net at a
/// point box (x = 1, every Relu stable-on, so the relaxation is EXACT) whose
/// 64 input-coefficient terms are full-mantissa f32 products spread over
/// 2^±16 (so the f64 partial sums MUST round) and cancel to a small residual.
/// The uncharged float coefficient lands above the exact value on about half
/// the seeds (by ~1e-11, while the directed steps give back only a few ulps
/// of the residual); the certified bound must stay at or below the exact
/// rational value on every seed. A fixture where the naive sum never
/// overshoots would prove nothing, so that is asserted too (VOID guard; an
/// earlier draft with every term in [0.5, 1) summed EXACTLY and tripped it).
#[test]
fn adversarial_rounding_charge_holds_a_cancelling_bound_below_the_exact_value() {
    let n_h = 64usize;
    let rat = |v: f64| Rat::from_f64_exact(v).expect("finite");
    let mut overshoots = 0usize;
    for seed in 1..=32u64 {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut draws: Vec<u64> = Vec::with_capacity(2 * n_h);
        for _ in 0..2 * n_h {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            draws.push(s);
        }
        // A full 24-bit mantissa in [0.5, 1), exact in f32.
        let mant = |r: u64| {
            f64::from(f32::from_bits(
                0x3F00_0000 | ((r >> 20) as u32 & 0x007F_FFFF),
            ))
        };
        let w1: Vec<f64> = draws[..n_h].iter().map(|&r| mant(r)).collect();
        let mut lam: Vec<f64> = draws[n_h..2 * n_h - 1]
            .iter()
            .map(|&r| {
                let scale = f64::from(2.0f32.powi((r % 33) as i32 - 16));
                let v = mant(r) * scale;
                if (r >> 8) & 1 == 1 {
                    -v
                } else {
                    v
                }
            })
            .collect();
        // The checker accumulates the input coefficient in row order.
        let partial = lam.iter().zip(&w1).fold(0.0f64, |a, (l, w)| a + l * w);
        lam.push(f64::from((-partial / w1[n_h - 1]) as f32));
        let naive = lam.iter().zip(&w1).fold(0.0f64, |a, (l, w)| a + l * w);
        let mut exact = Rat::ZERO;
        for (l, w) in lam.iter().zip(&w1) {
            exact = exact
                .add(rat(*l).mul(rat(*w)).expect("rat mul"))
                .expect("rat add");
        }
        if rat(naive) > exact {
            overshoots += 1;
        }
        let onnx = model(
            "x",
            &[1, 1],
            &[
                node("Gemm", &["x", "W1", "B1"], &["z"], vec![A::I("transB", 1)]),
                node("Relu", &["z"], &["h"], vec![]),
                node("Gemm", &["h", "W2", "B2"], &["y"], vec![A::I("transB", 1)]),
            ],
            &[
                tf("W1", &[n_h as i64, 1], &w1),
                tf("B1", &[n_h as i64], &vec![0.0; n_h]),
                tf(
                    "W2",
                    &[2, n_h as i64],
                    &lam.iter()
                        .copied()
                        .chain(vec![0.0; n_h])
                        .collect::<Vec<f64>>(),
                ),
                tf("B2", &[2], &[0.0, 0.0]),
            ],
            "y",
            &[1, 2],
        );
        let net = net_from_onnx(&onnx).expect("lowers");
        let (l, _, _) = ny_proofcheck::testkit::lower_bound(
            &net,
            &[1.0],
            &[1.0],
            &[],
            &[],
            None,
            &[(0, 1.0), (1, -1.0)],
            0.0,
        )
        .expect("bound");
        assert!(
            rat(l) <= exact,
            "seed {seed}: certified {l:e} above the exact value {exact:?} (naive {naive:e})"
        );
    }
    assert!(
        overshoots > 0,
        "the uncharged sum never overshot the exact value: the fixture is VOID"
    );
}
