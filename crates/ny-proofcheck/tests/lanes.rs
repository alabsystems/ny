// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! B2.2: the checker's lane-agnostic extensions, on synthetic fixtures only.
//!
//! * the admission set grows by the ops the input-split lanes' networks use
//!   (MatMul, Sub, Mul, Div, Neg, Identity) — each lowered to arithmetic the
//!   checker already certifies (Gemm, AddConst, BN's interval-affine form);
//! * a region whose nodes carry input-threshold literals is covered
//!   GEOMETRICALLY (any tiling, not only a split tree);
//! * the checker's own input clip confirms a conjunction no single atom's
//!   bound excludes (what a lane's clip close needs).

mod support;

use ny_proofcheck::ibp::{eval_point, forward};
use ny_proofcheck::testkit::net_from_onnx;
use ny_proofcheck::Verdict;
use support::*;

fn has(o: &ny_proofcheck::CheckOutcome, code: &str) -> bool {
    reason_codes(o).iter().any(|c| c == code)
}

/// acasxu's shape at 2 inputs: `input [1,1,1,2]` minus a constant, Flatten,
/// two MatMul + Add layers with a Relu between, then a Mul and a Div by
/// constants on the logits (exercising every new op on the verdict path):
///
/// * `z = W1ᵀ (x − c)`, `W1 = [[1, 1, 1], [1, −1, 0]]`, bias `(0, 0, 3)`;
/// * `h = relu(z)`, `y' = W2ᵀ h + b2`, `y = (y' · 2) / 4` = `y' / 2`.
///
/// `W2` makes `y0 − y1 = (h0 + h1 + h2 − 3 + margin) / 2`. With `c = (0, 0)`
/// and `x ∈ [−1, 1]²`: `h2 = x0 + 3 ≥ 2` always active, and
/// `h0 + h1 + h2 − 3 = relu(x0+x1) + relu(x0−x1) + x0`, whose minimum is
/// `−1` at `x0 = −1` (both Relus off). So `margin > 1` makes the property
/// true with exact minimum `(margin − 1) / 2` (which plain interval bounds
/// already reach: this fixture tests the cover and the ops, not bound power;
/// the MLP fixtures of `checker.rs` test bound power).
fn acas_like_onnx(margin: f64) -> Vec<u8> {
    model(
        "input",
        &[1, 1, 1, 2],
        &[
            node("Sub", &["input", "avg"], &["xs"], vec![]),
            node("Flatten", &["xs"], &["xf"], vec![A::I("axis", 1)]),
            node("MatMul", &["xf", "W1"], &["m1"], vec![]),
            node("Add", &["m1", "B1"], &["z"], vec![]),
            node("Relu", &["z"], &["h"], vec![]),
            node("MatMul", &["h", "W2"], &["m2"], vec![]),
            node("Add", &["m2", "B2"], &["yp"], vec![]),
            node("Identity", &["yp"], &["yi"], vec![]),
            node("Mul", &["yi", "two"], &["y2"], vec![]),
            node("Div", &["y2", "four"], &["y"], vec![]),
        ],
        &[
            tf("avg", &[1, 1, 1, 2], &[0.0, 0.0]),
            tf("W1", &[2, 3], &[1.0, 1.0, 1.0, 1.0, -1.0, 0.0]),
            tf("B1", &[3], &[0.0, 0.0, 3.0]),
            tf("W2", &[3, 2], &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            tf("B2", &[2], &[margin - 3.0, 0.0]),
            tf("two", &[1], &[2.0]),
            tf("four", &[], &[4.0]),
        ],
        "y",
        &[1, 2],
    )
}

fn acas_like_vnnlib() -> String {
    robustness_vnnlib(&["-1", "-1"], &["1", "1"], 2, 0, &[1])
}

/// The new ops lower to the checker's IR and evaluate to the network's value:
/// every rigorous point enclosure contains the exact output, and an IBP box
/// forward contains the point enclosures.
#[test]
fn lowering_admits_matmul_sub_mul_div_neg_and_encloses_points() {
    let net = net_from_onnx(&acas_like_onnx(1.5)).expect("the acas-like net lowers");
    let kinds: Vec<&str> = net.nodes.iter().map(|n| n.op.kind()).collect();
    assert_eq!(
        kinds,
        [
            "AddConst", "Identity", "Gemm", "AddConst", "Relu", "Gemm", "AddConst", "Identity",
            "Scale", "Scale"
        ],
        "Sub -> AddConst, MatMul -> Gemm, Mul/Div -> Scale"
    );
    for x in [[-1.0, 0.0], [0.25, -0.5], [1.0, 1.0], [-0.75, 0.75]] {
        let b = eval_point(&net, &x);
        let (yl, yh) = (&b.lo[net.output], &b.hi[net.output]);
        let z0 = (x[0] + x[1]).max(0.0);
        let z1 = (x[0] - x[1]).max(0.0);
        let z2 = (x[0] + 3.0).max(0.0);
        let y0 = (z0 + z1 + z2 + 1.5 - 3.0) / 2.0;
        assert!(
            yl[0] <= y0 && y0 <= yh[0],
            "y0 {y0} in [{}, {}]",
            yl[0],
            yh[0]
        );
        assert!(yl[1] <= 0.0 && 0.0 <= yh[1]);
        // Division by 4 is exact here, so the enclosure is tight to an ulp.
        assert!(yh[0] - yl[0] < 1e-12, "{} {}", yl[0], yh[0]);
    }
    // IBP over the box: y0 in [(0 + 0 + 2 - 1.5) / 2, (2 + 2 + 4 - 1.5) / 2],
    // outward rounded.
    let fwd = forward(&net, &[-1.0, -1.0], &[1.0, 1.0]);
    assert!(fwd.lo[net.output][0] <= 0.25 && fwd.hi[net.output][0] >= 3.25);
    // A Neg lowers too (and `c − x` Sub is a Scale).
    let neg = model(
        "x",
        &[1, 2],
        &[
            node("Neg", &["x"], &["n"], vec![]),
            node("Sub", &["k", "n"], &["y"], vec![]),
        ],
        &[tf("k", &[2], &[1.0, 2.0])],
        "y",
        &[1, 2],
    );
    let net = net_from_onnx(&neg).expect("Neg and const − act lower");
    let b = eval_point(&net, &[0.5, -0.25]);
    // y = k − (−x) = k + x, enclosed (the interval-affine form rounds
    // outward even where the value is exact).
    for (j, want) in [(0usize, 1.5f64), (1, 1.75)] {
        let (l, h) = (b.lo[net.output][j], b.hi[net.output][j]);
        assert!(
            l <= want && want <= h && h - l < 1e-12,
            "y{j}: [{l}, {h}] vs {want}"
        );
    }
    // Activation × activation stays outside the admission set.
    let bad = model(
        "x",
        &[1, 2],
        &[node("Mul", &["x", "x"], &["y"], vec![])],
        &[],
        "y",
        &[1, 2],
    );
    assert!(net_from_onnx(&bad).is_err());
    // Division by zero is refused, never lowered to an infinite scale.
    let dz = model(
        "x",
        &[1, 2],
        &[node("Div", &["x", "z"], &["y"], vec![])],
        &[tf("z", &[1], &[0.0])],
        "y",
        &[1, 2],
    );
    assert!(net_from_onnx(&dz).is_err());
}

/// Division by a constant whose reciprocal is NOT a binary fraction is
/// enclosed outward: the point enclosure of `x / 3` contains the exact
/// rational at a point where `fl(x / 3)` is inexact.
#[test]
fn division_scale_is_an_outward_enclosure() {
    let net = net_from_onnx(&model(
        "x",
        &[1, 1],
        &[node("Div", &["x", "three"], &["y"], vec![])],
        &[tf("three", &[1], &[3.0])],
        "y",
        &[1, 1],
    ))
    .expect("lowers");
    // The lowered scale itself encloses the EXACT reciprocal (not only the
    // point evaluation, whose own outward rounding could mask a point scale).
    let r = |v: f64| ny_cert::Rat::from_f64_exact(v).expect("finite");
    let three = r(3.0);
    let scale = net
        .nodes
        .iter()
        .find_map(|n| match &n.op {
            ny_proofcheck::lower::Op::Scale(b) => Some(b.clone()),
            _ => None,
        })
        .expect("Div lowers to a Scale");
    assert!(
        scale.s_lo[0] < scale.s_hi[0],
        "1/3 is not a binary fraction"
    );
    assert!(!r(1.0)
        .sub(r(scale.s_lo[0]).mul(three).unwrap())
        .unwrap()
        .is_negative());
    assert!(!r(scale.s_hi[0])
        .mul(three)
        .unwrap()
        .sub(r(1.0))
        .unwrap()
        .is_negative());
    let b = eval_point(&net, &[1.0]);
    let (lo, hi) = (b.lo[net.output][0], b.hi[net.output][0]);
    // 1/3 lies strictly between two f64 values; the enclosure straddles it.
    let third = 1.0f64 / 3.0;
    assert!(lo < hi, "an inexact quotient is an interval: [{lo}, {hi}]");
    assert!(lo <= third && third <= hi);
    // Exact check: lo * 3 <= 1 <= hi * 3 in exact rationals.
    assert!(!r(1.0).sub(r(lo).mul(three).unwrap()).unwrap().is_negative());
    assert!(!r(hi).mul(three).unwrap().sub(r(1.0)).unwrap().is_negative());
}

/// A box tiling of the input region: leaves are input-split BOXES (their
/// interior faces as `x` literals), not a split tree — a pinwheel of four
/// boxes around a centre box.
fn pinwheel() -> Vec<Vec<L>> {
    let b = |lo: [f64; 2], hi: [f64; 2]| {
        let mut v = Vec::new();
        for d in 0..2u64 {
            if lo[d as usize] > -1.0 {
                v.push(L::X(d, lo[d as usize], true));
            }
            if hi[d as usize] < 1.0 {
                v.push(L::X(d, hi[d as usize], false));
            }
        }
        v
    };
    vec![
        b([-1.0, -1.0], [0.5, -0.5]),
        b([0.5, -1.0], [1.0, 0.5]),
        b([-0.5, 0.5], [1.0, 1.0]),
        b([-1.0, -0.5], [-0.5, 1.0]),
        b([-0.6, -0.6], [0.6, 0.6]),
    ]
}

fn box_bundle(fx: &Fixture, leaves: &[Vec<L>], verdict_source_legacy: bool) -> Vec<u8> {
    let mut lines = vec![
        fx.header(),
        sites(&[(0, "z", 3, 4)]),
        clauses(0, &[(1, vec![clause01()])]),
    ];
    for (k, lits) in leaves.iter().enumerate() {
        lines.push(node_line(
            k as u64,
            lits,
            "bound",
            Some(&[1]),
            None,
            None,
            false,
            0.25,
        ));
    }
    let mut v = verdict(&[1], leaves.len() as u64);
    if verdict_source_legacy {
        v = v.replacen("\"authorized\"", "\"beta_crown_verdict\"", 1);
    }
    lines.push(v);
    lines.push(published("unsat"));
    finish(&lines)
}

/// B2.2 acceptance: an input-split lane's box leaves over a MatMul/Sub
/// network are verified — a non-tree tiling covers geometrically, every
/// leaf's discharge is re-derived by the checker — and the legacy door the
/// lane published through does not matter once it streamed its nodes.
#[test]
fn accepts_an_input_split_box_tiling_over_a_matmul_network() {
    let fx = Fixture::new("b22_box_tiling", acas_like_onnx(1.5), &acas_like_vnnlib());
    for legacy in [false, true] {
        let o = fx.check(
            if legacy { "legacy" } else { "authorized" },
            &box_bundle(&fx, &pinwheel(), legacy),
        );
        assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
        let r = report(&o);
        assert_eq!(
            r["cover"][0]["kind"].as_str(),
            Some("box"),
            "{}",
            o.report_json
        );
        assert_eq!(r["nodes"]["confirmed"].as_u64(), Some(5));
    }
}

/// Mutations of a correct box bundle: a leaf shrunk by one ulp leaves a
/// sliver (gap, `invalid`), a flipped face side opens a hole (gap), a dropped
/// leaf is a gap, and a network whose true minimum is negative is never
/// verified however the leaves tile (every leaf re-derived by the checker).
#[test]
fn rejects_box_bundle_mutations() {
    let fx = Fixture::new("b22_box_mut", acas_like_onnx(1.5), &acas_like_vnnlib());
    // Shrink the first leaf's x0 <= 0.5 face by one ulp: (0.5-ulp, 0.5] in
    // x0 over x1 in [-1, -0.6) is covered by no other box.
    let mut sliver = pinwheel();
    sliver[0][0] = L::X(0, f64::from_bits(0.5f64.to_bits() - 1), false);
    let o = fx.check("sliver", &box_bundle(&fx, &sliver, false));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    assert!(has(&o, "cover_gap"), "{}", o.report_json);
    // Flip the side of the centre box's first face.
    let mut flipped = pinwheel();
    flipped[4][0] = L::X(0, -0.6, false);
    let o = fx.check("flipped", &box_bundle(&fx, &flipped, false));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    // Drop a leaf.
    let dropped: Vec<Vec<L>> = pinwheel().into_iter().skip(1).collect();
    let o = fx.check("dropped", &box_bundle(&fx, &dropped, false));
    assert_eq!(o.exit_code, 20, "{}", o.report_json);
    // A false property (margin 0.5 < 1: minimum −0.25 at x0 = −1): the tiling
    // covers, but no discharge containing x0 = −1 confirms, and the probe
    // finds the counterexample.
    let fx = Fixture::new("b22_box_false", acas_like_onnx(0.5), &acas_like_vnnlib());
    let o = fx.check("false", &box_bundle(&fx, &pinwheel(), false));
    assert_ne!(o.verdict, Verdict::Verified, "{}", o.report_json);
    assert_eq!(o.verdict, Verdict::Refuted, "{}", o.report_json);
}

/// `y = (x0, −x0)` with the unsafe disjunct `y0 <= −1/2 ∧ y1 <= −1/2`, i.e.
/// `x0 <= −1/2 ∧ x0 >= 1/2`: infeasible, yet neither atom's bound alone is
/// positive anywhere near the root (`min y0 + 1/2 = min y1 + 1/2 = −1/2`).
fn conj_onnx() -> Vec<u8> {
    model(
        "x",
        &[1, 1],
        &[node("MatMul", &["x", "W"], &["y"], vec![])],
        &[tf("W", &[1, 2], &[1.0, -1.0])],
        "y",
        &[1, 2],
    )
}

fn conj_vnnlib() -> String {
    "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
     (assert (>= X_0 -1))\n(assert (<= X_0 1))\n\
     (assert (<= Y_0 -0.5))\n(assert (<= Y_1 -0.5))\n"
        .to_string()
}

/// B2.2: the checker's own input clip confirms a CONJUNCTION no single atom
/// excludes — the lane's clip close — and only with refinement on (the clip
/// re-derives on the shrunk box); a clause that IS satisfiable is never
/// clip-confirmed.
#[test]
fn clip_confirms_a_conjunction_no_single_atom_excludes() {
    let fx = Fixture::new("b22_clip", conj_onnx(), &conj_vnnlib());
    let atoms = vec!["Y0:1<=-1/2".to_string(), "Y1:1<=-1/2".to_string()];
    let bundle = |fx: &Fixture| {
        finish(&[
            fx.header(),
            sites(&[]),
            clauses(0, &[(0, atoms.clone())]),
            node_line(0, &[], "bound", Some(&[0]), None, None, false, 0.0),
            verdict(&[0], 1),
            published("unsat"),
        ])
    };
    let o = fx.check("clip", &bundle(&fx));
    assert_eq!(o.verdict, Verdict::Verified, "{}", o.report_json);
    assert_eq!(
        report(&o)["ladder"]["clip"].as_u64(),
        Some(1),
        "{}",
        o.report_json
    );
    let o = fx.check_with("clip_norefine", &bundle(&fx), |a| a.refine = false);
    assert_eq!(o.verdict, Verdict::Unconfirmed, "{}", o.report_json);
    // Satisfiable: y0 <= 1/2 and y1 <= 1/2 (x0 = 0): never confirmed.
    let sat = "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
               (assert (>= X_0 -1))\n(assert (<= X_0 1))\n\
               (assert (<= Y_0 0.5))\n(assert (<= Y_1 0.5))\n";
    let fx = Fixture::new("b22_clip_sat", conj_onnx(), sat);
    let lines = finish(&[
        fx.header(),
        sites(&[]),
        clauses(0, &[(0, vec!["Y0:1<=1/2".into(), "Y1:1<=1/2".into()])]),
        node_line(0, &[], "bound", Some(&[0]), None, None, false, 0.0),
        verdict(&[0], 1),
        published("unsat"),
    ]);
    let o = fx.check("sat", &lines);
    assert_eq!(o.verdict, Verdict::Refuted, "{}", o.report_json);
    assert_eq!(
        report(&o)["ladder"]["clip"].as_u64(),
        Some(0),
        "{}",
        o.report_json
    );
}

/// Site bounds handed to a back-substitution as given (the test's view).
struct GivenView {
    lo: Vec<Vec<f64>>,
    hi: Vec<Vec<f64>>,
}

impl ny_proofcheck::backsub::SiteView for GivenView {
    fn site(&self, t: usize) -> (&[f64], &[f64]) {
        (&self.lo[t], &self.hi[t])
    }
}

/// review(b2-lanes): the clip treats every row's linear form as a CERTIFIED
/// half-space, so a row whose relaxation FAILED (an unstable Relu with an
/// unbounded end meets a negative λ; its term is dropped and the row
/// certifies nothing) must hand back no form. `y = −relu(x)`, `x ∈ [−1, 1]`,
/// with the Relu input's lower end widened to −∞: the row `y` fails, and its
/// dropped-term form (`y >= 0·x + c`, `c ≈ 0`) is FALSE at `x = 1` (`y = −1`).
/// Control: the same row over the finite view certifies `y >= −1` with a form.
#[test]
fn a_failed_row_hands_the_clip_no_linear_form() {
    use ny_proofcheck::backsub::{backsub, Ctx, Seeds};
    use ny_proofcheck::ibp::magnitudes;
    let net = net_from_onnx(&model(
        "x",
        &[1, 1],
        &[
            node("MatMul", &["x", "W1"], &["z"], vec![]),
            node("Relu", &["z"], &["h"], vec![]),
            node("MatMul", &["h", "W2"], &["y"], vec![]),
        ],
        &[tf("W1", &[1, 1], &[1.0]), tf("W2", &[1, 1], &[-1.0])],
        "y",
        &[1, 1],
    ))
    .expect("lowers");
    let (lo, hi) = ([-1.0f64], [1.0f64]);
    let b = forward(&net, &lo, &hi);
    let mags = magnitudes(&net, &b);
    let z = *net.by_name.get("z").expect("the Relu input");
    let seeds = Seeds {
        tensor: net.output,
        rows: vec![vec![(0, 1.0)]],
        kappa0: vec![0.0],
        nterms: 1,
    };
    let run = |view: &GivenView| {
        let ctx = Ctx {
            net: &net,
            mags: &mags,
            view,
            alpha: None,
            betas: &[],
            box_lo: &lo,
            box_hi: &hi,
            chunk_rows: 16,
            want_lin: true,
        };
        backsub(&ctx, &seeds).remove(0)
    };
    let mut wide = GivenView {
        lo: b.lo.clone(),
        hi: b.hi.clone(),
    };
    wide.lo[z][0] = f64::NEG_INFINITY;
    let failed = run(&wide);
    assert!(failed.lb.is_nan(), "the row certifies nothing: {failed:?}");
    assert!(
        failed.lin.is_none(),
        "a failed row must hand back no (false) linear form: {failed:?}"
    );
    let ok = run(&GivenView {
        lo: b.lo.clone(),
        hi: b.hi,
    });
    assert!(ok.lb.is_finite() && ok.lb <= -1.0 + 1e-9, "{ok:?}");
    assert!(ok.lin.is_some(), "a certified row keeps its form: {ok:?}");
}

/// review(b2-lanes): the clip is never tighter than the half-space it cuts
/// by, in several dimensions. `y = (x0 + x1, x0 − x1)` on `[−1, 1]²` with the
/// SATISFIABLE conjunction `y0 <= 0 ∧ y1 <= −1` (e.g. `x = (−1, 0.5)`): the
/// clip may shrink the box to `[−1, 0] × [0, 1]` but must never confirm. A
/// clip that bounds the OTHER coordinates' contribution by its maximum
/// instead of its minimum would clip `x0 + x1 <= 0` to `x0 <= −1, x1 <= −1`,
/// where `x0 − x1 = 0 > −1`: a false `clip` confirmation (the refutation
/// probe may still find the counterexample, so the ladder count is the
/// pin, not only the verdict).
#[test]
fn clip_never_confirms_a_satisfiable_conjunction_in_two_dimensions() {
    let onnx = model(
        "x",
        &[1, 2],
        &[node("MatMul", &["x", "W"], &["y"], vec![])],
        &[tf("W", &[2, 2], &[1.0, 1.0, 1.0, -1.0])],
        "y",
        &[1, 2],
    );
    let text = "(declare-const X_0 Real)\n(declare-const X_1 Real)\n\
                (declare-const Y_0 Real)\n(declare-const Y_1 Real)\n\
                (assert (>= X_0 -1))\n(assert (<= X_0 1))\n\
                (assert (>= X_1 -1))\n(assert (<= X_1 1))\n\
                (assert (<= Y_0 0))\n(assert (<= Y_1 -1))\n";
    let fx = Fixture::new("b22_clip_sat2", onnx, text);
    let lines = finish(&[
        fx.header(),
        sites(&[]),
        clauses(0, &[(0, vec!["Y0:1<=0".into(), "Y1:1<=-1".into()])]),
        node_line(0, &[], "bound", Some(&[0]), None, None, false, 0.0),
        verdict(&[0], 1),
        published("unsat"),
    ]);
    let o = fx.check("sat2", &lines);
    assert_ne!(o.verdict, Verdict::Verified, "{}", o.report_json);
    assert_eq!(
        report(&o)["ladder"]["clip"].as_u64(),
        Some(0),
        "the clip confirmed a satisfiable conjunction: {}",
        o.report_json
    );
    assert!(
        !has(&o, "clause_mismatch"),
        "the fixture's clause must match: {}",
        o.report_json
    );
}
