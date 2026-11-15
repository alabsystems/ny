// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Containment certification for the QDQ envelope.
//!
//! The oracle below IS the ONNX/onnxruntime FLOAT32 pipeline, under both the
//! divide and the reciprocal-multiply kernel models. Every containment check
//! asserts with ZERO tolerance: the envelope must contain the true value, not
//! approximate it. `planted_violation_is_detected` shows the harness can fail
//! by re-running it against the historical UNCERTIFIED half-scale radius,
//! which the checked-in counterexample (scale 0.1, x = 0x4154_0001) breaks.

use super::*;
use ndarray::ArrayD;

/// True fused QDQ value, divide-kernel model: `fl32(x / s)`.
fn qdq_true_div(x: f32, s: f32, zp: i64, qmin: i64, qmax: i64) -> f32 {
    let d = x / s;
    quantize_tail(d, s, zp, qmin, qmax)
}

/// True fused QDQ value, reciprocal-multiply kernel model: `fl32(x * fl32(1/s))`.
fn qdq_true_mul(x: f32, s: f32, zp: i64, qmin: i64, qmax: i64) -> f32 {
    let d = x * (1.0f32 / s);
    quantize_tail(d, s, zp, qmin, qmax)
}

/// True fused QDQ value, f32 zero-point-add kernel model: the zero point is
/// added in FLOAT32 before rounding (`round(fl32(x/s + zp)) - zp`), which can
/// flip round-half-even parity relative to integer-domain addition.
fn qdq_true_zp_add(x: f32, s: f32, zp: i64, qmin: i64, qmax: i64) -> f32 {
    let d = x / s;
    assert!(!d.is_nan());
    let v = d + zp as f32;
    let r = v.round_ties_even();
    let q = f64::from(r).clamp(qmin as f64, qmax as f64);
    let k = q as i64 - zp;
    (k as f32) * s
}

fn quantize_tail(d: f32, s: f32, zp: i64, qmin: i64, qmax: i64) -> f32 {
    assert!(!d.is_nan(), "test oracle fed a NaN pre-quantizer value");
    // round_ties_even on the f32, exactly as the ONNX spec requires.
    let r = d.round_ties_even();
    // Saturating add + clamp in f64 (exact for these magnitudes), then the
    // exact integer k = q - zp, then ONE rounded f32 multiply.
    let q = (f64::from(r) + zp as f64).clamp(qmin as f64, qmax as f64);
    let k = q as i64 - zp;
    (k as f32) * s
}

fn point(x: f32) -> BoundedTensor {
    let t = ArrayD::from_shape_vec(IxDyn(&[1]), vec![x]).unwrap();
    BoundedTensor::new_allow_infinite(t.clone(), t).unwrap()
}

/// The parameter grid: (scale, zero_point, qmin, qmax).
fn param_grid() -> Vec<(f32, i64, i64, i64)> {
    let mut grid = Vec::new();
    // uint8 cases, including the smart_turn head quantizer (s, zp=0) and its
    // classifier.5 (s=0.0148118, zp=11).
    for &(s, zp) in &[
        (0.009_454_178_f32, 0_i64),
        (0.014_811_8, 11),
        (0.1, 0),
        (0.1, 128),
        (1.0, 255),
        (3.7e-3, 42),
        (128.0, 1),
        (6.1e-5, 200),
    ] {
        grid.push((s, zp, 0, 255));
    }
    // int8 cases.
    for &(s, zp) in &[
        (0.1_f32, 0_i64),
        (0.023, -128),
        (0.023, 127),
        (2.5, -1),
        (7.3e-4, 5),
    ] {
        grid.push((s, zp, -128, 127));
    }
    grid
}

/// Every f32 probe point for one parameter set: dense samples around every
/// bucket edge (the round-half-even tie points), bucket centers, the
/// saturation edges, and far-outside values.
fn probe_points(s: f32, zp: i64, qmin: i64, qmax: i64) -> Vec<f32> {
    let mut xs = Vec::new();
    let s64 = f64::from(s);
    for k in (qmin - zp)..=(qmax - zp) {
        // Bucket k's upper edge in real terms: (k + 1/2) * s — the
        // ties-to-even boundary — and its center.
        let edge = (k as f64 + 0.5) * s64;
        let center = k as f64 * s64;
        for base in [edge, center] {
            let f = base as f32;
            let mut lo3 = f;
            let mut hi3 = f;
            for _ in 0..3 {
                lo3 = next_down_f32(lo3);
                hi3 = next_up_f32(hi3);
                xs.push(lo3);
                xs.push(hi3);
            }
            xs.push(f);
        }
    }
    // Far outside the saturation range on both sides.
    let lo_sat = ((qmin - zp) as f64 * s64) as f32;
    let hi_sat = ((qmax - zp) as f64 * s64) as f32;
    xs.extend_from_slice(&[
        lo_sat - 1000.0 * s,
        hi_sat + 1000.0 * s,
        lo_sat - 3.5e4,
        hi_sat + 3.5e4,
        0.0,
        -0.0,
    ]);
    xs.retain(|x| x.is_finite());
    xs
}

/// Core harness: checks IBP point containment for every probe point under
/// both kernel models. Returns the number of violations instead of panicking
/// so the planted-violation test can PROVE the harness can fail.
fn count_ibp_violations(
    layer: &QdqPerturbationLayer,
    s: f32,
    zp: i64,
    qmin: i64,
    qmax: i64,
) -> usize {
    let mut violations = 0;
    for x in probe_points(s, zp, qmin, qmax) {
        let out = layer.propagate_ibp(&point(x)).expect("IBP must not error");
        let lo = out.lower()[[0]];
        let up = out.upper()[[0]];
        for y in [
            qdq_true_div(x, s, zp, qmin, qmax),
            qdq_true_mul(x, s, zp, qmin, qmax),
            qdq_true_zp_add(x, s, zp, qmin, qmax),
        ] {
            if !(lo <= y && y <= up) {
                violations += 1;
            }
        }
    }
    violations
}

#[test]
fn ibp_point_containment_exhaustive_over_buckets_and_ties() {
    for (s, zp, qmin, qmax) in param_grid() {
        let layer = QdqPerturbationLayer::try_certified(&[s], &[zp], qmin, qmax)
            .expect("certified constructor");
        let violations = count_ibp_violations(&layer, s, zp, qmin, qmax);
        assert_eq!(
            violations, 0,
            "IBP envelope violated {violations} times for scale {s}, zp {zp}, range [{qmin}, {qmax}]"
        );
    }
}

#[test]
fn ibp_interval_containment_covers_interior_points() {
    for (s, zp, qmin, qmax) in param_grid() {
        let layer = QdqPerturbationLayer::try_certified(&[s], &[zp], qmin, qmax)
            .expect("certified constructor");
        let xs = probe_points(s, zp, qmin, qmax);
        // Take a few wide and narrow intervals from the probe set.
        for (i, j) in [(0usize, 40usize), (7, 9), (3, 200), (50, 51)] {
            let (Some(&a), Some(&b)) = (xs.get(i), xs.get(j)) else {
                continue;
            };
            let (l, u) = if a <= b { (a, b) } else { (b, a) };
            let t = ArrayD::from_shape_vec(IxDyn(&[1]), vec![l]).unwrap();
            let t2 = ArrayD::from_shape_vec(IxDyn(&[1]), vec![u]).unwrap();
            let out = layer
                .propagate_ibp(&BoundedTensor::new_allow_infinite(t, t2).unwrap())
                .expect("IBP must not error");
            let (olo, oup) = (out.lower()[[0]], out.upper()[[0]]);
            for step in 0..=32 {
                let x = l + (u - l) * (step as f32) / 32.0;
                for y in [
                    qdq_true_div(x, s, zp, qmin, qmax),
                    qdq_true_mul(x, s, zp, qmin, qmax),
                    qdq_true_zp_add(x, s, zp, qmin, qmax),
                ] {
                    assert!(
                        olo <= y && y <= oup,
                        "interval IBP [{olo}, {oup}] misses y={y} at x={x} (l={l}, u={u}, s={s}, zp={zp})"
                    );
                }
            }
        }
    }
}

#[test]
fn crown_relaxation_contains_true_values() {
    for (s, zp, qmin, qmax) in param_grid() {
        let layer = QdqPerturbationLayer::try_certified(&[s], &[zp], qmin, qmax)
            .expect("certified constructor");
        let s64 = f64::from(s);
        let lo_sat = ((qmin - zp) as f64 * s64) as f32;
        let hi_sat = ((qmax - zp) as f64 * s64) as f32;
        let spans = [
            (lo_sat - 2.0 * s, hi_sat + 2.0 * s),  // crosses both edges
            (lo_sat - 5.0 * s, lo_sat + 0.25 * s), // crosses lower edge
            (hi_sat - 0.25 * s, hi_sat + 5.0 * s), // crosses upper edge
            (lo_sat + 0.1 * s, hi_sat - 0.1 * s),  // interior
            (0.4 * s, 0.6 * s),                    // narrow, near one tie
            (0.5 * s, 0.5 * s),                    // degenerate at a tie
        ];
        for (l, u) in spans {
            if !(l.is_finite() && u.is_finite() && l <= u) {
                continue;
            }
            let relax = layer.relaxation(l, u, 0);
            for step in 0..=64 {
                let x = l + (u - l) * (step as f32) / 64.0;
                let lower_line =
                    f64::from(relax.lower_slope) * f64::from(x) + f64::from(relax.lower_intercept);
                let upper_line =
                    f64::from(relax.upper_slope) * f64::from(x) + f64::from(relax.upper_intercept);
                for y in [
                    qdq_true_div(x, s, zp, qmin, qmax),
                    qdq_true_mul(x, s, zp, qmin, qmax),
                    qdq_true_zp_add(x, s, zp, qmin, qmax),
                ] {
                    assert!(
                        lower_line <= f64::from(y) && f64::from(y) <= upper_line,
                        "CROWN relaxation [{lower_line}, {upper_line}] misses y={y} at x={x} \
                         (l={l}, u={u}, s={s}, zp={zp}, range [{qmin}, {qmax}])"
                    );
                }
            }
        }
    }
}

/// The harness must be ABLE to fail: the historical uncertified half-scale
/// radius (no division-rounding term, no reconstruction term, no directed
/// steps) is broken by the checked-in counterexample scale=0.1,
/// x = f32::from_bits(0x4154_0001), and this test proves the containment
/// harness sees exactly that.
#[test]
fn planted_violation_is_detected() {
    let (s, zp, qmin, qmax) = (0.1_f32, 0_i64, 0_i64, 255_i64);
    let mut layer = QdqPerturbationLayer::try_certified(&[s], &[zp], qmin, qmax)
        .expect("certified constructor");
    // Plant the violation: shrink the envelope to the naive exact-real
    // half-scale radius with no reconstruction slack.
    layer.eps[0] = s * 0.5;
    layer.recon[0] = 0.0;
    let violations = count_ibp_violations(&layer, s, zp, qmin, qmax);
    assert!(
        violations > 0,
        "planted half-scale envelope must be caught by the containment harness"
    );
    // And the specific historical counterexample is among the failures.
    let x = f32::from_bits(0x4154_0001);
    let y = qdq_true_div(x, s, zp, qmin, qmax);
    let out = layer.propagate_ibp(&point(x)).expect("IBP must not error");
    assert!(
        y < out.lower()[[0]],
        "checked-in counterexample must escape the planted envelope: y={y}, lo={}",
        out.lower()[[0]]
    );
}

#[test]
fn per_axis_slots_broadcast_cyclically_over_trailing_axis() {
    // Two rows of four features; slot i applies to feature i in each row.
    let scales = [0.1_f32, 0.2, 0.05, 1.0];
    let zps = [0_i64, 10, 255, 128];
    let layer =
        QdqPerturbationLayer::try_certified(&scales, &zps, 0, 255).expect("certified constructor");
    let xs: Vec<f32> = vec![0.04, 1.3, -0.7, 12.0, 25.6, -3.0, 0.0, 130.0];
    let t = ArrayD::from_shape_vec(IxDyn(&[2, 4]), xs.clone()).unwrap();
    let out = layer
        .propagate_ibp(&BoundedTensor::new_allow_infinite(t.clone(), t).unwrap())
        .expect("IBP must not error");
    for (i, &x) in xs.iter().enumerate() {
        let (row, col) = (i / 4, i % 4);
        let (s, zp) = (scales[col], zps[col]);
        for y in [
            qdq_true_div(x, s, zp, 0, 255),
            qdq_true_mul(x, s, zp, 0, 255),
            qdq_true_zp_add(x, s, zp, 0, 255),
        ] {
            assert!(
                out.lower()[[row, col]] <= y && y <= out.upper()[[row, col]],
                "per-axis slot {col} envelope misses y={y} at x={x}"
            );
        }
    }
}

#[test]
fn constructor_rejects_invalid_parameters() {
    assert!(QdqPerturbationLayer::try_certified(&[], &[], 0, 255).is_err());
    assert!(QdqPerturbationLayer::try_certified(&[0.1], &[], 0, 255).is_err());
    assert!(QdqPerturbationLayer::try_certified(&[0.0], &[0], 0, 255).is_err());
    assert!(QdqPerturbationLayer::try_certified(&[-0.1], &[0], 0, 255).is_err());
    assert!(QdqPerturbationLayer::try_certified(&[f32::NAN], &[0], 0, 255).is_err());
    assert!(QdqPerturbationLayer::try_certified(&[f32::INFINITY], &[0], 0, 255).is_err());
    assert!(QdqPerturbationLayer::try_certified(&[0.1], &[256], 0, 255).is_err());
    assert!(QdqPerturbationLayer::try_certified(&[0.1], &[-1], 0, 255).is_err());
    assert!(QdqPerturbationLayer::try_certified(&[0.1], &[0], 255, 0).is_err());
    assert!(QdqPerturbationLayer::try_certified(&[0.1], &[0], 0, 1 << 25).is_err());
    assert!(QdqPerturbationLayer::try_certified(&[0.1], &[0], 0, 255).is_ok());
    assert!(QdqPerturbationLayer::try_certified(&[0.1], &[-128], -128, 127).is_ok());
}

#[test]
fn nan_input_is_rejected_and_infinite_input_saturates() {
    let layer = QdqPerturbationLayer::try_certified(&[0.1], &[0], 0, 255).expect("certified");
    let nan = ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NAN]).unwrap();
    let fin = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0f32]).unwrap();
    assert!(layer
        .propagate_ibp(&BoundedTensor::new_unchecked(nan, fin).expect("unchecked constructor"))
        .is_err());

    let lo = ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NEG_INFINITY]).unwrap();
    let hi = ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::INFINITY]).unwrap();
    let out = layer
        .propagate_ibp(&BoundedTensor::new_allow_infinite(lo, hi).unwrap())
        .expect("infinite bounds saturate through the clamp");
    // The image of the whole real line is inside the saturation range ± G.
    assert!(out.lower()[[0]].is_finite() && out.upper()[[0]].is_finite());
    assert!(out.lower()[[0]] <= 0.0 && out.upper()[[0]] >= 25.5);
}
