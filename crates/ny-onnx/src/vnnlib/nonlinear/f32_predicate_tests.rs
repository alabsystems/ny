// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use std::time::Duration;
fn formula(body: &str) -> NonLinearFormula {
    NonLinearFormula::parse(&format!("(vnnlib-version <2.0>)\n(declare-network f (declare-input X float32 [2]) (declare-output Y float32 [1]))\n(assert {body})")).unwrap()
}
fn point(f: &NonLinearFormula, x: [f64; 2], y: f64) -> Option<CaseSplitVerdict> {
    let end = Instant::now() + Duration::from_secs(5);
    f.prepare_f32_predicate(end, &mut || true)?
        .refute_box(&x, &x, &[y], &[y], end, &mut || true)
}

#[test]
fn f32_predicate_underflow_is_a_real_semantics_counter_control() {
    let f = formula(
        "(and (> X[0] 0.0) (== X[1] 0.0) (> Y[0] 0.0) (<= (* (* X[0] 2.0) Y[0]) (* X[1] X[1])))",
    );
    let tiny = f64::from(f32::from_bits(1));
    let x = [tiny, 0.0];
    assert_eq!(f.holds_at(&x, &[tiny]), Some(false));
    assert_eq!(
        f.holds_over_box_split(&x, &x, &[tiny], &[tiny])
            .unwrap()
            .truth,
        Tri::False
    );
    // The effective binary32 program rounds the nonzero product to zero.
    // It has a counterexample, so the separate profile MUST NOT refute it.
    assert_eq!(point(&f, x, tiny).unwrap().truth, Tri::Unknown);
}

#[test]
fn f32_predicate_weak_scalar_promotion_and_minus_order_are_preserved() {
    let f = formula("(== X[0] 16777217.0)");
    let x = [16777216.0, 0.0];
    assert_eq!(f.holds_at(&x, &[0.0]), Some(false));
    assert_eq!(point(&f, x, 0.0).unwrap().truth, Tri::Unknown);
    let f = formula("(!= (- X[0] X[1] 1.0) 0.0)");
    // Reference Minus is a - sum(rest). (2^24 + 1) rounds to 2^24,
    // hence 2^24 - (2^24 + 1) is zero, unlike two rounded subtractions.
    assert_eq!(
        point(&f, [16777216.0, 16777216.0], 0.0).unwrap().truth,
        Tri::False
    );
}

#[test]
fn f32_predicate_grid_excludes_no_representable_point_at_strict_edges() {
    assert!(environment());
    let min = f64::from(f32::from_bits(1));
    assert!(grid(OpenIv {
        lo: 0.0,
        hi: min,
        lo_open: true,
        hi_open: true
    })
    .unwrap()
    .is_none());
    assert_eq!(
        grid(OpenIv {
            lo: 0.0,
            hi: min,
            lo_open: true,
            hi_open: false
        })
        .unwrap(),
        Some(OpenIv::closed(min, min))
    );
    let f = formula("(and (> X[0] 0.0) (< X[0] 1.401298464324817e-45))");
    let end = Instant::now() + Duration::from_secs(5);
    let p = f.prepare_f32_predicate(end, &mut || true).unwrap();
    assert_eq!(
        p.refute_box(&[0.0, 0.0], &[min, 0.0], &[0.0], &[0.0], end, &mut || true)
            .unwrap()
            .truth,
        Tri::False
    );
    let f = formula("(and (> X[0] 0.0) (<= X[0] 1.401298464324817e-45))");
    assert_eq!(point(&f, [min, 0.0], 0.0).unwrap().truth, Tri::Unknown);
    for bits in [
        0,
        1,
        2,
        3,
        0x0080_0000,
        0x3f80_0000,
        0x7f7f_ffff,
        0x8000_0000,
        0x8000_0001,
        0xbf80_0000,
    ] {
        let v = f64::from(f32::from_bits(bits));
        let iv = OpenIv::closed(v, v);
        assert_eq!(grid(iv).unwrap(), Some(iv));
    }
}

#[test]
fn f32_predicate_source_profile_rejects_unsupported_before_short_circuit() {
    let end = Instant::now() + Duration::from_secs(5);
    for src in ["(declare-network)", "(declare-network f)"] {
        assert!(NonLinearFormula::parse(src).is_err());
    }
    for body in [
        "(and (< X[0] X[0]) (> BAD[0] 0))",
        "(and (< X[0] X[0]) (> (/ X[0] 2) 0))",
        "(and (< X[0] X[0]) (+ X[0] 1))",
        "(< X[0])",
    ] {
        assert!(
            formula(body)
                .prepare_f32_predicate(end, &mut || true)
                .is_none(),
            "{body}"
        );
    }
    let src="(vnnlib-version <2.0>) (declare-network f (declare-input X float64 [2]) (declare-output Y float32 [1])) (assert (< X[0] 0))";
    assert!(NonLinearFormula::parse(src)
        .unwrap()
        .prepare_f32_predicate(end, &mut || true)
        .is_none());
    for insertion in ["(declare-hidden H float32 [1])", "(isomorphic-to g)"] {
        let src=format!("(vnnlib-version <2.0>) (declare-network f (declare-input X float32 [2]) (declare-output Y float32 [1]) {insertion}) (assert (< X[0] 0))");
        assert!(NonLinearFormula::parse(&src)
            .unwrap()
            .prepare_f32_predicate(end, &mut || true)
            .is_none());
    }
}

#[test]
fn f32_predicate_pinned_independent_numpy_checker_points_are_never_refuted() {
    let v: serde_json::Value =
        serde_json::from_str(include_str!("f32_reference_points.json")).unwrap();
    assert_eq!(v["point_count"].as_u64(), Some(155));
    let mut count = 0;
    let mut refuted = 0;
    let mut accepted = 0;
    for c in v["cases"].as_array().unwrap() {
        let f = NonLinearFormula::parse(c["source"].as_str().unwrap()).unwrap();
        let end = Instant::now() + Duration::from_secs(5);
        let p = f.prepare_f32_predicate(end, &mut || true).unwrap();
        for sample in c["samples"].as_array().unwrap() {
            let values = |name: &str| -> Vec<f64> {
                sample[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| f64::from(f32::from_bits(v.as_u64().unwrap() as u32)))
                    .collect()
            };
            let x = values("x_bits");
            let y = values("y_bits");
            let expected = sample["expected"].as_bool().unwrap();
            let result = p.refute_box(&x, &x, &y, &y, end, &mut || true).unwrap();
            if result.truth == Tri::False {
                assert!(
                    !expected,
                    "independent positive was refuted: {sample:?} source={}",
                    c["source"]
                );
                refuted += 1;
            }
            if expected {
                accepted += 1;
            }
            count += 1;
        }
    }
    assert_eq!(count, 155);
    assert!(
        accepted > 20 && refuted > 20,
        "nonvacuous census {accepted} {refuted}"
    );
}

#[test]
fn f32_predicate_every_poll_refuses_and_deadline_or_malformed_box_cannot_publish() {
    let f = formula("(and (>= X[0] 0.0) (<= X[0] 0.0) (> (* X[0] Y[0]) 0.0))");
    let run = |fail: usize| {
        let mut n = 0;
        let mut cb = || {
            n += 1;
            n != fail
        };
        let end = Instant::now() + Duration::from_secs(5);
        let result = f
            .prepare_f32_predicate(end, &mut cb)
            .and_then(|p| p.refute_box(&[-1.0, 0.0], &[1.0, 0.0], &[-1.0], &[1.0], end, &mut cb));
        (result, n)
    };
    let (good, polls) = run(usize::MAX);
    assert_eq!(good.unwrap().truth, Tri::False);
    assert!(polls > 10);
    for fail in 1..=polls {
        assert!(run(fail).0.is_none(), "poll {fail}");
    }
    let end = Instant::now() + Duration::from_secs(5);
    let p = f.prepare_f32_predicate(end, &mut || true).unwrap();
    for (l, u) in [(f64::NAN, 1.0), (0.0, f64::INFINITY), (1.0, -1.0)] {
        assert!(p
            .refute_box(&[l, 0.0], &[u, 0.0], &[0.0], &[0.0], end, &mut || true)
            .is_none());
    }
    assert!(p
        .refute_box(&[], &[], &[0.0], &[0.0], end, &mut || true)
        .is_none());
    assert!(p
        .refute_box(
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[0.0],
            &[0.0],
            Instant::now(),
            &mut || true
        )
        .is_none());
    assert!(f
        .prepare_f32_predicate(Instant::now(), &mut || true)
        .is_none());
}

#[test]
fn f32_predicate_full_cartesian_piece_limit_refuses_without_partial_cover() {
    let mut atoms = Vec::new();
    for name in ["X", "Y"] {
        for i in 0..4 {
            atoms.push(format!("(or (> {name}[{i}] 0.0) (< {name}[{i}] 0.0))"));
        }
    }
    let src=format!("(vnnlib-version <2.0>) (declare-network f (declare-input X float32 [4]) (declare-output Y float32 [4])) (assert (and {}))",atoms.join(" "));
    let f = NonLinearFormula::parse(&src).unwrap();
    let end = Instant::now() + Duration::from_secs(5);
    let p = f.prepare_f32_predicate(end, &mut || true).unwrap();
    let result = p
        .refute_box(
            &[-1.0; 4],
            &[1.0; 4],
            &[-1.0; 4],
            &[1.0; 4],
            end,
            &mut || true,
        )
        .unwrap();
    assert_eq!(
        result,
        CaseSplitVerdict {
            truth: Tri::Unknown,
            pieces: 0
        }
    );
}
