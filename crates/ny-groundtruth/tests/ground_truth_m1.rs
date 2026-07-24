// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M1 golden and verification tests for `ny-groundtruth`
//! (`docs/GEOMETRIC_GROUND_TRUTH_PLAN.md` §3, M1; follows the M0 spike in
//! `crates/ny-propagate/tests/ground_truth_m0.rs`).
//!
//! - **Golden tests**: every builder's graph is evaluated at concrete dyadic
//!   sample points via zero-width IBP — a *sound enclosure* of the exact
//!   value (M0 finding (a)) — and must enclose the exact rational reference
//!   ([`ny_groundtruth::reference`]) with negligible width.
//! - **Dominance verifies**: for every builder, a surrogate `f = g + margin`
//!   (exactly representable constant shift, like M0) is Verified dominant
//!   over `g` on a box via CROWN on the difference network.
//! - **Falsified direction**: `f = g − margin` is rejected with a concrete
//!   witness from the grid search (M0 finding (b)).
//! - **Constant contract** (plan §2.3): inexact, non-finite, derived-inexact,
//!   non-unit-axis, and degenerate parameters are rejected with typed errors
//!   — never silently rounded.

use ndarray::arr1;
use ny_core::Bound;
use ny_groundtruth::{
    cone_residual, cylinder_residual, max_of, min_of, reference, signed_plane_distance,
    sphere_residual, torus_residual, verify_against_ground_truth, with_pose, GroundTruthError,
    GroundTruthOutcome, Pose, Relation,
};
use ny_propagate::layers::AddConstantLayer;
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

// --- helpers ---------------------------------------------------------------

/// Evaluate a single-output graph at a concrete point via zero-width IBP.
/// The result is a sound enclosure of the exact value (M0 finding (a)).
fn eval1(g: &GraphNetwork, x: [f32; 3]) -> (f64, f64) {
    let arr = arr1(&x).into_dyn();
    let t = BoundedTensor::new(arr.clone(), arr).expect("point tensor is valid");
    let out = g.propagate_ibp(&t).expect("IBP point evaluation succeeds");
    assert_eq!(out.lower().len(), 1, "residual graphs have a single output");
    (f64::from(out.lower()[0]), f64::from(out.upper()[0]))
}

/// Assert the graph's sound enclosure at `x` contains the exact reference
/// value and is tight (a few ULPs of directed rounding, not a real gap).
fn assert_golden(g: &GraphNetwork, x: [f32; 3], reference_value: f64) {
    let (lo, hi) = eval1(g, x);
    assert!(
        lo <= reference_value && reference_value <= hi,
        "sound enclosure [{lo}, {hi}] must contain the exact reference {reference_value} at {x:?}"
    );
    assert!(
        hi - lo <= 1e-3 * (1.0 + reference_value.abs()),
        "point enclosure should be tight, got [{lo}, {hi}] (reference {reference_value})"
    );
}

/// Surrogate builder: `f = g + k` via an exactly representable constant
/// shift appended to the graph (the M1 analogue of M0's biased surrogate).
fn plus_constant(mut g: GraphNetwork, k: f32) -> GraphNetwork {
    let out = g.output_name().to_string();
    g.add_node(GraphNode::new(
        "surrogate_shift",
        Layer::AddConstant(AddConstantLayer::new(arr1(&[k]).into_dyn())),
        vec![out],
    ));
    g.set_output("surrogate_shift");
    g
}

fn box_around(center: [f32; 3], half: f32) -> Vec<Bound> {
    center
        .iter()
        .map(|&c| Bound::new(c - half, c + half))
        .collect()
}

fn assert_dominance_verified(build: impl Fn() -> GraphNetwork, region: &[Bound]) {
    let g = build();
    let f = plus_constant(build(), 10.0);
    let outcome = verify_against_ground_truth(&f, &g, Relation::Dominates, region)
        .expect("verification runs");
    match outcome {
        GroundTruthOutcome::Verified { difference_bounds } => {
            assert!(
                difference_bounds.iter().all(|b| b.lower() >= 0.0),
                "Verified dominance must certify nonnegative lower bounds, got {difference_bounds:?}"
            );
        }
        other => panic!("expected Verified dominance for f = g + 10, got {other:?}"),
    }
}

// --- golden tests: builders vs exact rational reference --------------------

#[test]
fn plane_golden_matches_reference() {
    let n = [0.0, 0.0, 1.0];
    let d = -5.0;
    let g = signed_plane_distance(n, d).expect("plane builds");
    for x in [
        [0.0_f32, 0.0, 5.0], // on the plane
        [1.5, -2.0, 7.25],   // above
        [0.0, 0.0, -10.0],   // below
        [-3.25, 8.5, 0.5],
    ] {
        let x64 = [f64::from(x[0]), f64::from(x[1]), f64::from(x[2])];
        assert_golden(&g, x, reference::signed_plane_distance(n, d, x64));
    }
    assert_golden(&g, [0.0, 0.0, 5.0], 0.0);
}

#[test]
fn sphere_golden_matches_reference() {
    let c = [1.0, -2.0, 0.5];
    let r = 1.5;
    let g = sphere_residual(c, r).expect("sphere builds");
    for x in [
        [2.5_f32, -2.0, 0.5], // on the surface
        [1.0, -2.0, 0.5],     // center (inside)
        [3.0, -2.0, 0.5],     // outside
        [1.5, -1.5, 1.0],     // inside, off-axis
    ] {
        let x64 = [f64::from(x[0]), f64::from(x[1]), f64::from(x[2])];
        assert_golden(&g, x, reference::sphere_residual(c, r, x64));
    }
    assert_golden(&g, [2.5, -2.0, 0.5], 0.0);
    assert_golden(&g, [1.0, -2.0, 0.5], -2.25);
}

/// The a3d acceptance shape (plan §5.4): an axis-aligned scanned-rod cylinder.
#[test]
fn cylinder_golden_matches_reference() {
    let a = [0.0, 0.0, 1.0];
    let p = [1.0, -2.0, 0.5];
    let r = 1.5;
    let g = cylinder_residual(a, p, r).expect("cylinder builds");
    for x in [
        [2.5_f32, -2.0, 7.0], // on the surface, far along the axis
        [1.0, -2.0, 0.0],     // on the axis (inside)
        [3.5, -1.0, 4.0],     // outside
        [1.5, -2.5, -3.0],    // inside, off-axis
    ] {
        let x64 = [f64::from(x[0]), f64::from(x[1]), f64::from(x[2])];
        assert_golden(&g, x, reference::cylinder_residual(a, p, r, x64));
    }
    assert_golden(&g, [2.5, -2.0, 7.0], 0.0);
    assert_golden(&g, [1.0, -2.0, 0.0], -2.25);
    assert_golden(&g, [3.5, -1.0, 4.0], 5.0);
}

#[test]
fn cone_golden_matches_reference() {
    let a = [0.0, 1.0, 0.0];
    let q = [0.5, 2.0, -1.0];
    let k = 0.5; // half-angle 45 degrees
    let g = cone_residual(a, q, k).expect("cone builds");
    for x in [
        [1.5_f32, 3.0, -1.0], // on the surface (45 degrees off axis)
        [0.5, 4.0, -1.0],     // on the axis (inside)
        [2.5, 2.0, -1.0],     // orthogonal to the axis at the apex (outside)
        [0.5, 2.0, -1.0],     // the apex itself (residual 0)
    ] {
        let x64 = [f64::from(x[0]), f64::from(x[1]), f64::from(x[2])];
        assert_golden(&g, x, reference::cone_residual(a, q, k, x64));
    }
    assert_golden(&g, [1.5, 3.0, -1.0], 0.0);
    assert_golden(&g, [0.5, 4.0, -1.0], -2.0);
    assert_golden(&g, [2.5, 2.0, -1.0], 2.0);
}

#[test]
fn torus_golden_matches_reference() {
    let a = [0.0, 0.0, 1.0];
    let p = [0.5, 0.0, -1.0];
    let big_r = 2.0;
    let r = 0.5;
    let g = torus_residual(a, p, big_r, r).expect("torus builds");
    for x in [
        [3.0_f32, 0.0, -1.0], // on the surface (outer equator)
        [2.5, 0.0, -1.0],     // inside the tube
        [0.5, 0.0, 1.0],      // on the axis (far outside the tube)
        [0.5, -2.0, -0.5],    // generic off-plane sample
    ] {
        let x64 = [f64::from(x[0]), f64::from(x[1]), f64::from(x[2])];
        assert_golden(&g, x, reference::torus_residual(a, p, big_r, r, x64));
    }
    assert_golden(&g, [3.0, 0.0, -1.0], 0.0);
}

// --- one dominance verify per builder (surrogate = g + margin, like M0) ----

#[test]
fn plane_dominance_verifies() {
    let region = box_around([0.0, 0.0, 0.0], 10.0);
    assert_dominance_verified(
        || signed_plane_distance([0.0, 0.0, 1.0], -5.0).expect("plane builds"),
        &region,
    );
}

#[test]
fn sphere_dominance_verifies() {
    let region = box_around([2.5, -2.0, 0.5], 0.5);
    assert_dominance_verified(
        || sphere_residual([1.0, -2.0, 0.5], 1.5).expect("sphere builds"),
        &region,
    );
}

/// The a3d acceptance shape: dominance over the cylinder residual.
#[test]
fn cylinder_dominance_verifies() {
    let region = box_around([2.5, -2.0, 0.0], 0.5);
    assert_dominance_verified(
        || cylinder_residual([0.0, 0.0, 1.0], [1.0, -2.0, 0.5], 1.5).expect("cylinder builds"),
        &region,
    );
}

#[test]
fn cone_dominance_verifies() {
    let region = box_around([1.5, 3.0, -1.0], 0.25);
    assert_dominance_verified(
        || cone_residual([0.0, 1.0, 0.0], [0.5, 2.0, -1.0], 0.5).expect("cone builds"),
        &region,
    );
}

#[test]
fn torus_dominance_verifies() {
    let region = box_around([3.0, 0.0, -1.0], 0.25);
    assert_dominance_verified(
        || torus_residual([0.0, 0.0, 1.0], [0.5, 0.0, -1.0], 2.0, 0.5).expect("torus builds"),
        &region,
    );
}

// --- falsified direction: witness search (M0 finding (b)) ------------------

#[test]
fn cylinder_violation_is_falsified_with_witness() {
    let build =
        || cylinder_residual([0.0, 0.0, 1.0], [1.0, -2.0, 0.5], 1.5).expect("cylinder builds");
    let g = build();
    let f2 = plus_constant(build(), -10.0); // f2 = g - 10 everywhere
    let region = box_around([2.5, -2.0, 0.0], 0.5);

    let outcome = verify_against_ground_truth(&f2, &g, Relation::Dominates, &region)
        .expect("verification runs");
    match outcome {
        GroundTruthOutcome::Falsified {
            witness,
            difference,
        } => {
            assert_eq!(witness.len(), 3);
            for (x, b) in witness.iter().zip(region.iter()) {
                assert!(
                    *x >= b.lower() && *x <= b.upper(),
                    "witness {witness:?} must lie inside the region"
                );
            }
            assert!(
                difference.upper() < 0.0,
                "witness enclosure must certify f2(x*) - g(x*) < 0, got {difference:?}"
            );
            // Cross-check the witness against the original networks, like M0.
            let x = [witness[0], witness[1], witness[2]];
            let (_, f2_hi) = eval1(&f2, x);
            let (g_lo, _) = eval1(&g, x);
            assert!(
                f2_hi < g_lo,
                "witness must satisfy f2(x*) < g(x*): f2 <= {f2_hi}, g >= {g_lo}"
            );
        }
        other => panic!("expected Falsified with witness for f2 = g - 10, got {other:?}"),
    }
}

// --- AbsBound relation ------------------------------------------------------

#[test]
fn abs_bound_verifies_identical_ground_truths() {
    let build = || sphere_residual([1.0, -2.0, 0.5], 1.5).expect("sphere builds");
    let region = box_around([2.5, -2.0, 0.5], 0.5);
    let outcome = verify_against_ground_truth(&build(), &build(), Relation::AbsBound(2.0), &region)
        .expect("verification runs");
    match outcome {
        GroundTruthOutcome::Verified { difference_bounds } => {
            assert!(difference_bounds
                .iter()
                .all(|b| b.lower() >= -2.0 && b.upper() <= 2.0));
        }
        other => panic!("expected Verified |f - g| <= 2, got {other:?}"),
    }
}

#[test]
fn abs_bound_violation_is_falsified_with_witness() {
    let build = || sphere_residual([1.0, -2.0, 0.5], 1.5).expect("sphere builds");
    let g = build();
    let f = plus_constant(build(), 5.0); // |f - g| = 5 > 2 everywhere
    let region = box_around([2.5, -2.0, 0.5], 0.5);
    let outcome = verify_against_ground_truth(&f, &g, Relation::AbsBound(2.0), &region)
        .expect("verification runs");
    match outcome {
        GroundTruthOutcome::Falsified { difference, .. } => {
            assert!(
                difference.lower() > 2.0,
                "witness enclosure must certify |f - g| > 2, got {difference:?}"
            );
        }
        other => panic!("expected Falsified for |f - g| = 5 > 2, got {other:?}"),
    }
}

// --- compose: pose + min/max ------------------------------------------------

#[test]
fn pose_translation_golden_matches_shifted_reference() {
    let c = [1.0, -2.0, 0.5];
    let r = 1.5;
    let t = [0.5, 1.0, -1.0];
    let g = sphere_residual(c, r).expect("sphere builds");
    let posed = with_pose(&g, &Pose::translation(t).expect("pose builds")).expect("pose composes");
    for x in [[2.0_f32, -3.0, 1.5], [0.5, -2.5, 0.0], [1.0, 0.25, -0.75]] {
        let shifted = [
            f64::from(x[0]) + t[0],
            f64::from(x[1]) + t[1],
            f64::from(x[2]) + t[2],
        ];
        assert_golden(&posed, x, reference::sphere_residual(c, r, shifted));
    }
}

#[test]
fn pose_linear_golden_matches_transformed_reference() {
    let c = [1.0, -2.0, 0.5];
    let r = 1.5;
    // x -> (2*x0 + 0.5, x1, x2): exactly representable non-rigid affine map.
    let pose = Pose::new(
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.5, 0.0, 0.0],
    )
    .expect("pose builds");
    let g = sphere_residual(c, r).expect("sphere builds");
    let posed = with_pose(&g, &pose).expect("pose composes");
    for x in [[0.25_f32, -2.0, 0.5], [-1.0, -1.5, 1.0]] {
        let mapped = [
            2.0 * f64::from(x[0]) + 0.5,
            f64::from(x[1]),
            f64::from(x[2]),
        ];
        assert_golden(&posed, x, reference::sphere_residual(c, r, mapped));
    }
}

#[test]
fn min_max_golden_match_pointwise_min_max_of_references() {
    let c1 = [-1.0, 0.0, 0.0];
    let c2 = [1.0, 0.0, 0.0];
    let r = 1.0;
    let build = |c: [f64; 3]| sphere_residual(c, r).expect("sphere builds");
    let union = min_of(&[build(c1), build(c2)]).expect("min composes");
    let intersection = max_of(&[build(c1), build(c2)]).expect("max composes");
    for x in [
        [0.0_f32, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.5, 0.75, -0.25],
    ] {
        let x64 = [f64::from(x[0]), f64::from(x[1]), f64::from(x[2])];
        let r1 = reference::sphere_residual(c1, r, x64);
        let r2 = reference::sphere_residual(c2, r, x64);
        assert_golden(&union, x, r1.min(r2));
        assert_golden(&intersection, x, r1.max(r2));
    }
}

#[test]
fn min_composition_dominance_verifies() {
    let c1 = [-1.0, 0.0, 0.0];
    let c2 = [1.0, 0.0, 0.0];
    let build = || {
        min_of(&[
            sphere_residual(c1, 1.0).expect("sphere builds"),
            sphere_residual(c2, 1.0).expect("sphere builds"),
        ])
        .expect("min composes")
    };
    let region = box_around([-1.0, 0.0, 0.0], 0.25);
    assert_dominance_verified(build, &region);
}

#[test]
fn compose_rejects_inexact_and_empty() {
    assert!(matches!(
        Pose::translation([0.1, 0.0, 0.0]),
        Err(GroundTruthError::InexactParameter { .. })
    ));
    assert!(matches!(
        min_of(&[]),
        Err(GroundTruthError::InvalidComposition(_))
    ));
}

// --- constant-handling contract (plan §2.3) ---------------------------------

#[test]
fn builders_reject_inexact_parameters() {
    // 0.1 does not round-trip f64 -> f32.
    assert!(matches!(
        sphere_residual([1.0, -2.0, 0.5], 0.1),
        Err(GroundTruthError::InexactParameter { .. })
    ));
    assert!(matches!(
        sphere_residual([0.1, 0.0, 0.0], 1.0),
        Err(GroundTruthError::InexactParameter { .. })
    ));
    assert!(matches!(
        signed_plane_distance([0.0, 0.0, 1.0], 0.2),
        Err(GroundTruthError::InexactParameter { .. })
    ));
}

#[test]
fn builders_reject_non_finite_parameters() {
    assert!(matches!(
        sphere_residual([1.0, -2.0, 0.5], f64::NAN),
        Err(GroundTruthError::NonFiniteParameter { .. })
    ));
    assert!(matches!(
        signed_plane_distance([f64::INFINITY, 0.0, 0.0], 0.0),
        Err(GroundTruthError::NonFiniteParameter { .. })
    ));
}

#[test]
fn builders_reject_inexact_derived_constants() {
    // 8191.5 is f32-exact but its square ((2^14 - 1)^2 / 4, odd 28-bit
    // numerator) is not: exact rational arithmetic catches what a silent
    // f32 multiply would have rounded.
    assert!(matches!(
        sphere_residual([0.0, 0.0, 0.0], 8191.5),
        Err(GroundTruthError::InexactDerivedConstant { .. })
    ));
    // Torus: -4R^2 = -(2^14 - 1)^2 is a 28-bit odd integer, not f32-exact
    // (while R^2 - r^2 = 8192 * 8191 happens to be exact).
    assert!(matches!(
        torus_residual([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 8191.5, 0.5),
        Err(GroundTruthError::InexactDerivedConstant { .. })
    ));
}

#[test]
fn builders_reject_non_unit_axes() {
    assert!(matches!(
        cylinder_residual([1.0, 1.0, 0.0], [0.0, 0.0, 0.0], 1.0),
        Err(GroundTruthError::AxisNotUnit { .. })
    ));
    // Unit over the reals, but the components are not f32-exact: rejected
    // as inexact parameters before the axis check (never silently rounded).
    assert!(matches!(
        cone_residual([0.6, 0.8, 0.0], [0.0, 0.0, 0.0], 0.5),
        Err(GroundTruthError::InexactParameter { .. })
    ));
    assert!(matches!(
        torus_residual([0.0, 2.0, 0.0], [0.0, 0.0, 0.0], 2.0, 0.5),
        Err(GroundTruthError::AxisNotUnit { .. })
    ));
}

#[test]
fn builders_reject_degenerate_parameters() {
    assert!(matches!(
        signed_plane_distance([0.0, 0.0, 0.0], 1.0),
        Err(GroundTruthError::DegenerateParameter { .. })
    ));
    assert!(matches!(
        sphere_residual([0.0, 0.0, 0.0], -1.0),
        Err(GroundTruthError::DegenerateParameter { .. })
    ));
    assert!(matches!(
        cylinder_residual([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 0.0),
        Err(GroundTruthError::DegenerateParameter { .. })
    ));
    assert!(matches!(
        cone_residual([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 1.0),
        Err(GroundTruthError::DegenerateParameter { .. })
    ));
    assert!(matches!(
        torus_residual([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 0.0, 0.5),
        Err(GroundTruthError::DegenerateParameter { .. })
    ));
}
