// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Whole-field "no-escape" continuous-domain tolerance tests
//! (`docs/GEOMETRIC_GROUND_TRUTH_PLAN.md`; wave-3 whole-field item).
//!
//! The capability: sampled inspection compares measured points to a nominal and
//! reports max deviation — a spike *between* samples is missed. Here a surrogate
//! field `f` (a `GraphNetwork`) is verified against a nominal ground-truth graph
//! `g` over the WHOLE input box: `∀ x: |f − g| ≤ tol`.
//!
//! - `conforms_over_the_whole_box`: a tilted-plane field within tol → Conforms
//!   + certificate (certified deviation enclosure).
//! - `violation_reports_a_witness_region`: an over-tolerance field → Violates
//!   with a witness point and a locator region.
//! - `agrees_with_point_sampling_on_the_samples`: on the sampled points the
//!   whole-field verdict is consistent with naive point sampling.
//! - `between_sample_spike_is_not_falsely_conforming`: the soundness upgrade —
//!   a narrow bump between the default grid samples is invisible to point
//!   sampling (which would falsely pass) but the whole-field verify never
//!   returns Conforms; refining the witness grid catches it as a Violation.

use ndarray::{arr1, arr2};
use ny_core::Bound;
use ny_groundtruth::{
    signed_plane_distance, verify_whole_field_tolerance, verify_whole_field_tolerance_with,
    VerifyOptions, WholeFieldOutcome,
};
use ny_propagate::layers::{AddLayer, LinearLayer, ReLULayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

/// Nominal: the plane `z = 0` (a flat mating face), `g(x) = x2`.
fn nominal_plane() -> GraphNetwork {
    signed_plane_distance([0.0, 0.0, 1.0], 0.0).expect("plane builds")
}

/// Evaluate a single-output graph at a concrete point via zero-width IBP
/// (a sound enclosure of the exact value).
fn eval1(g: &GraphNetwork, x: [f32; 3]) -> (f32, f32) {
    let arr = arr1(&x).into_dyn();
    let t = BoundedTensor::new(arr.clone(), arr).expect("point tensor is valid");
    let out = g.propagate_ibp(&t).expect("IBP point evaluation succeeds");
    (out.lower()[0], out.upper()[0])
}

/// A tilted-plane measured-field surrogate `f(x) = x0/128 + x2`: it tracks the
/// nominal `g(x) = x2` with a small, whole-field-varying deviation `x0/128`.
/// The normal need not be unit (plane residual denotes the same field), and
/// `1/128` is exactly f32-representable.
fn tilted_field() -> GraphNetwork {
    signed_plane_distance([1.0 / 128.0, 0.0, 1.0], 0.0).expect("tilted plane builds")
}

#[test]
fn conforms_over_the_whole_box() {
    // Deviation f − g = x0/128 ∈ [−1/128, 1/128] ≈ ±0.0078 on x0 ∈ [−1, 1];
    // tol = 0.1 → Conforms, and the difference net is affine so CROWN is exact.
    let f = tilted_field();
    let g = nominal_plane();
    let region = vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
        Bound::new(-0.5, 0.5),
    ];
    match verify_whole_field_tolerance(&f, &g, &region, 0.1).expect("verify runs") {
        WholeFieldOutcome::Conforms { cert } => {
            assert!(
                cert.max_abs_deviation() <= 0.1,
                "certified deviation {} must be within tol",
                cert.max_abs_deviation()
            );
            // The tilt is real: the certified deviation is not identically zero.
            assert!(cert.max_abs_deviation() > 0.0);
            // And it is close to the true 1/128 (CROWN is exact on an affine h).
            assert!(
                (cert.max_abs_deviation() - 1.0 / 128.0).abs() <= 1e-4,
                "expected the certified band ≈ 1/128, got {}",
                cert.max_abs_deviation()
            );
            for b in &cert.deviation_bounds {
                assert!(b.lower() >= -0.1 && b.upper() <= 0.1, "band inside ±tol");
            }
        }
        other => panic!("expected Conforms, got {other:?}"),
    }
}

#[test]
fn violation_reports_a_witness_region() {
    // A constant 0.5 offset field: f − g ≡ 0.5 everywhere > tol = 0.1, so the
    // grid witness search certifies a violation with a concrete point.
    let f = signed_plane_distance([0.0, 0.0, 1.0], 0.5).expect("offset plane builds");
    let g = nominal_plane();
    let region = vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
        Bound::new(-0.5, 0.5),
    ];
    match verify_whole_field_tolerance(&f, &g, &region, 0.1).expect("verify runs") {
        WholeFieldOutcome::Violates {
            witness,
            witness_region,
            difference,
        } => {
            assert_eq!(witness.len(), 3);
            // The witness lies inside the region.
            for (x, b) in witness.iter().zip(region.iter()) {
                assert!(*x >= b.lower() && *x <= b.upper(), "witness inside region");
            }
            // The witness_region is a sub-box of the region containing x*.
            assert_eq!(witness_region.len(), 3);
            for ((x, wb), rb) in witness.iter().zip(&witness_region).zip(&region) {
                assert!(
                    wb.lower() >= rb.lower() && wb.upper() <= rb.upper(),
                    "sub-box"
                );
                assert!(*x >= wb.lower() && *x <= wb.upper(), "x* in witness_region");
            }
            // The certified deviation at x* is out of tolerance.
            assert!(
                difference.lower() > 0.1 || difference.upper() < -0.1,
                "witness deviation {difference:?} must certainly exceed ±0.1"
            );
            // Cross-check against a direct evaluation of each side at x*.
            let x = [witness[0], witness[1], witness[2]];
            let (f_lo, _) = eval1(&f, x);
            let (_, g_hi) = eval1(&g, x);
            assert!(f_lo - g_hi > 0.1, "direct f − g at x* exceeds tol");
        }
        other => panic!("expected Violates, got {other:?}"),
    }
}

#[test]
fn agrees_with_point_sampling_on_the_samples() {
    // Where the whole-field verdict and naive sampling overlap, they must agree.
    let region = vec![
        Bound::new(-1.0, 1.0),
        Bound::new(-1.0, 1.0),
        Bound::new(-0.5, 0.5),
    ];
    let grid = sample_grid(&region, 5);

    // Conforming field: every sample is within tol (sampling would also pass).
    let f = tilted_field();
    let g = nominal_plane();
    assert!(matches!(
        verify_whole_field_tolerance(&f, &g, &region, 0.1).unwrap(),
        WholeFieldOutcome::Conforms { .. }
    ));
    for x in &grid {
        let (fl, fh) = eval1(&f, *x);
        let (gl, gh) = eval1(&g, *x);
        let dev = ((fl - gh).abs()).max((fh - gl).abs());
        assert!(dev <= 0.1, "sample {x:?} deviation {dev} within tol");
    }

    // Violating field: sampling sees the violation at the reported witness.
    let f = signed_plane_distance([0.0, 0.0, 1.0], 0.5).unwrap();
    match verify_whole_field_tolerance(&f, &g, &region, 0.1).unwrap() {
        WholeFieldOutcome::Violates { witness, .. } => {
            let x = [witness[0], witness[1], witness[2]];
            let (fl, _) = eval1(&f, x);
            let (_, gh) = eval1(&g, x);
            assert!(
                fl - gh > 0.1,
                "point sampling confirms the witness violation"
            );
        }
        other => panic!("expected Violates, got {other:?}"),
    }
}

/// The soundness upgrade over sampled inspection. A narrow triangular bump on
/// `x0`, centered at `t0 = 3/8` with half-width `w = 1/16` and peak `a·w`, is
/// added to the nominal field. Its support `[5/16, 7/16]` misses every default
/// (5-per-dim) grid sample `{0, 1/4, 1/2, 3/4, 1}` on `x0 ∈ [0, 1]`, so naive
/// point sampling sees deviation ≡ 0 and would FALSELY pass. The whole-field
/// verify must never return Conforms — and a grid fine enough to sample the
/// bump reports the Violation.
///
/// `f(x) = x2 + hat(x0)`,
/// `hat(t) = a·relu(t − (t0−w)) − 2a·relu(t − t0) + a·relu(t − (t0+w))`.
fn bump_field(a: f32, t0: f32, w: f32) -> GraphNetwork {
    let mut f = GraphNetwork::new();
    // Three ReLU pre-activations t − knot for the hat's three knots.
    f.add_node(GraphNode::from_input(
        "hat_pre",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                Some(arr1(&[-(t0 - w), -t0, -(t0 + w)])),
            )
            .expect("hat_pre linear valid"),
        ),
    ));
    f.add_node(GraphNode::new(
        "hat_relu",
        Layer::ReLU(ReLULayer),
        vec!["hat_pre".to_string()],
    ));
    // hat(x0) = a·r0 − 2a·r1 + a·r2  (one output).
    f.add_node(GraphNode::new(
        "hat",
        Layer::Linear(
            LinearLayer::new(arr2(&[[a, -2.0 * a, a]]), None).expect("hat readout valid"),
        ),
        vec!["hat_relu".to_string()],
    ));
    // base(x) = x2 (the nominal plane field), then f = hat + base.
    f.add_node(GraphNode::from_input(
        "base",
        Layer::Linear(
            LinearLayer::new(arr2(&[[0.0_f32, 0.0, 1.0]]), None).expect("base linear valid"),
        ),
    ));
    f.add_node(GraphNode::binary(
        "out",
        Layer::Add(AddLayer),
        "hat",
        "base",
    ));
    f.set_output("out");
    f
}

#[test]
fn between_sample_spike_is_not_falsely_conforming() {
    // Bump: peak = a·w = 10 · 1/16 = 0.625 ≫ tol = 0.1, support [5/16, 7/16].
    let (a, t0, w) = (10.0_f32, 3.0 / 8.0, 1.0 / 16.0);
    let f = bump_field(a, t0, w);
    let g = nominal_plane();
    let region = vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(-0.5, 0.5),
    ];
    let tol = 0.1;

    // (a) Naive point sampling on the DEFAULT (5-per-dim) grid sees no
    // violation — the bump is zero at every sample. This is the FALSE pass a
    // sampled inspector would report.
    let coarse = sample_grid(&region, 5);
    let mut max_sampled_dev = 0.0_f32;
    for x in &coarse {
        let (fl, fh) = eval1(&f, *x);
        let (gl, gh) = eval1(&g, *x);
        max_sampled_dev = max_sampled_dev.max((fl - gh).abs().max((fh - gl).abs()));
    }
    assert!(
        max_sampled_dev <= tol as f32,
        "the coarse grid must miss the bump (naive sampling would pass), max dev {max_sampled_dev}"
    );

    // (b) The whole-field verify must NOT falsely certify conformance.
    let outcome = verify_whole_field_tolerance(&f, &g, &region, tol).expect("verify runs");
    assert!(
        !matches!(outcome, WholeFieldOutcome::Conforms { .. }),
        "no-escape verify must refuse the false pass, got {outcome:?}"
    );

    // (c) A grid fine enough to sample the peak (t0 = 12/32 lands on a 33-point
    // grid) escalates the sound non-answer into a certified Violation.
    let fine = VerifyOptions {
        witness_grid: 33,
        ..VerifyOptions::default()
    };
    match verify_whole_field_tolerance_with(&f, &g, &region, tol, &fine).expect("verify runs") {
        WholeFieldOutcome::Violates {
            witness,
            difference,
            ..
        } => {
            // The witness sits inside the bump support and is out of tolerance.
            assert!(
                witness[0] >= t0 - w && witness[0] <= t0 + w,
                "witness x0 = {} must be inside the bump support",
                witness[0]
            );
            assert!(
                difference.lower() > tol as f32 || difference.upper() < -(tol as f32),
                "witness deviation {difference:?} must exceed ±tol"
            );
        }
        other => panic!("a peak-sampling grid must find the violation, got {other:?}"),
    }
}

/// Evenly spaced grid samples (endpoints included) over the box, `n` per
/// dimension — the naive point-sampling baseline.
fn sample_grid(region: &[Bound], n: usize) -> Vec<[f32; 3]> {
    assert_eq!(region.len(), 3);
    let axis = |b: &Bound| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                b.lower() + t * (b.upper() - b.lower())
            })
            .collect()
    };
    let (xs, ys, zs) = (axis(&region[0]), axis(&region[1]), axis(&region[2]));
    let mut out = Vec::with_capacity(n * n * n);
    for &x in &xs {
        for &y in &ys {
            for &z in &zs {
                out.push([x, y, z]);
            }
        }
    }
    out
}
