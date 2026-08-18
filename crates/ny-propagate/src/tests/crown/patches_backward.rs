// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::patches::{
    CrownBounds, PatchGeometry, PatchesData, PatchesLinearBounds, UnstableIdx,
};
use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{
    crown_elementwise_backward_patches, crown_elementwise_backward_patches_with_deadline,
    crown_elementwise_backward_patches_with_poll_for_test,
};
use ndarray::{array, Array1, ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

fn patches_relaxation(lower: f32, upper: f32) -> LinearRelaxation {
    LinearRelaxation::new(
        lower.abs() + 1.0,
        lower.abs() + 0.25,
        upper + 0.5,
        upper + 1.25,
    )
}

/// Build the mixed-sign fixture used by the enclosure check and the value pin.
fn mixed_sign_fixture() -> (PatchesLinearBounds, BoundedTensor) {
    let bounds = PatchesLinearBounds {
        row_count: 1,
        lower_a: PatchesData {
            coeff_err: None,
            patches: Some(
                ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 1, 2]), vec![1.5_f32, -2.0]).unwrap(),
            ),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, 2),
            unstable_idx: None,
        },
        lower_b: array![0.25_f32],
        upper_a: PatchesData {
            coeff_err: None,
            patches: Some(
                ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 1, 2]), vec![-0.75_f32, 3.0]).unwrap(),
            ),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, 2),
            unstable_idx: None,
        },
        upper_b: array![-0.5_f32],
    };
    let pre_activation = BoundedTensor::new(
        array![-1.0_f32, -2.0].into_dyn(),
        array![2.0_f32, 4.0].into_dyn(),
    )
    .unwrap();
    (bounds, pre_activation)
}

#[test]
fn elementwise_patches_rejects_mismatched_side_geometry_in_release_builds() {
    let (mut bounds, pre_activation) = mixed_sign_fixture();
    bounds.upper_a.geometry = PatchGeometry::affine((2, 1), (0, 0, 0, 0));
    let error = crown_elementwise_backward_patches(&bounds, &pre_activation, patches_relaxation)
        .expect_err("one lower/upper geometry must be a hard precondition");
    assert!(matches!(error, NyError::InvalidSpec(_)), "{error:?}");

    let (mut bounds, pre_activation) = mixed_sign_fixture();
    bounds.upper_a.patches = Some(
        bounds
            .upper_a
            .patches
            .take()
            .unwrap()
            .into_shape_with_order(IxDyn(&[1, 1, 1, 1, 2, 1]))
            .unwrap(),
    );
    let error = crown_elementwise_backward_patches(&bounds, &pre_activation, patches_relaxation)
        .expect_err("shared metadata cannot authenticate mismatched coefficient tensors");
    assert!(matches!(error, NyError::ShapeMismatch { .. }), "{error:?}");
}

#[test]
fn elementwise_patches_preserves_anchored_geometry_without_densifying() {
    use crate::bounds::patches::{patches_to_dense_call_sites, reset_patches_to_dense_call_count};

    let (expected, actual) = crate::bounds::patches::test_override::with_eager_err(false, || {
        let (affine, pre_activation) = mixed_sign_fixture();
        let expected =
            match crown_elementwise_backward_patches(&affine, &pre_activation, patches_relaxation)
                .expect("affine oracle")
            {
                CrownBounds::Patches(bounds) => bounds,
                CrownBounds::Dense(_) => panic!("elementwise Patches must stay structured"),
            };
        let mut bounds = affine;
        let anchored = PatchGeometry::anchored(vec![0], vec![0]).unwrap();
        bounds.lower_a.geometry = anchored.clone();
        bounds.upper_a.geometry = anchored;

        reset_patches_to_dense_call_count();
        let actual = match crown_elementwise_backward_patches_with_deadline(
            &bounds,
            &pre_activation,
            std::time::Instant::now() + std::time::Duration::from_secs(30),
            patches_relaxation,
        )
        .expect("Anchored elementwise composition")
        {
            CrownBounds::Patches(bounds) => bounds,
            CrownBounds::Dense(_) => panic!("Anchored elementwise path must not densify"),
        };
        (expected, actual)
    });
    assert!(matches!(
        &actual.lower_a.geometry,
        PatchGeometry::Anchored(_)
    ));
    assert_eq!(actual.lower_a.geometry, actual.upper_a.geometry);
    assert_eq!(actual.lower_a.patches, expected.lower_a.patches);
    assert_eq!(actual.upper_a.patches, expected.upper_a.patches);
    assert_eq!(actual.lower_b, expected.lower_b);
    assert_eq!(actual.upper_b, expected.upper_b);
    assert_eq!(actual.lower_a.coeff_err, expected.lower_a.coeff_err);
    assert_eq!(actual.upper_a.coeff_err, expected.upper_a.coeff_err);
    assert!(
        patches_to_dense_call_sites().is_empty(),
        "native Anchored elementwise propagation called to_dense"
    );
}

#[test]
fn anchored_elementwise_charges_both_subnormal_center_flushes() {
    crate::bounds::patches::test_override::with_eager_err(false, || {
        let (mut bounds, pre_activation) = mixed_sign_fixture();
        let min_subnormal = f32::from_bits(1);
        let coefficients = vec![min_subnormal, -min_subnormal];
        bounds.lower_a.patches =
            Some(ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 1, 2]), coefficients.clone()).unwrap());
        bounds.upper_a.patches =
            Some(ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 1, 2]), coefficients).unwrap());
        let anchored = PatchGeometry::anchored(vec![0], vec![0]).unwrap();
        bounds.lower_a.geometry = anchored.clone();
        bounds.upper_a.geometry = anchored;

        let result =
            match crown_elementwise_backward_patches(&bounds, &pre_activation, |_lower, _upper| {
                LinearRelaxation::identity()
            })
            .expect("Anchored identity relaxation")
            {
                CrownBounds::Patches(bounds) => bounds,
                CrownBounds::Dense(_) => panic!("Anchored elementwise path must stay Patches"),
            };
        for side in [&result.lower_a, &result.upper_a] {
            let stored = side.patches.as_ref().unwrap().as_slice().unwrap();
            for &center in stored {
                let magnitude = center.to_bits() & 0x7fff_ffff;
                assert!(
                    magnitude == 0 || magnitude >= 0x0080_0000,
                    "Anchored coefficient centers must be zero or normal, got {center:e}"
                );
            }
            let error = side.coeff_err.as_ref().unwrap()[0];
            let magnitude = error.to_bits() & 0x7fff_ffff;
            assert!(
                magnitude == 0 || magnitude >= 0x0080_0000,
                "published coefficient error must be zero or normal, got {error:e}"
            );
            for (&center, exact) in stored.iter().zip([
                ny_core::f32_to_f64_exact(min_subnormal),
                ny_core::f32_to_f64_exact(-min_subnormal),
            ]) {
                assert!(
                    (exact - ny_core::f32_to_f64_exact(center)).abs()
                        <= ny_core::f32_to_f64_exact(error),
                    "coeff_err must cover exact product versus the no-subnormal center"
                );
            }
        }
    });
}

fn anchored_tiny_intercept_fixture(explicit_rows: bool) -> (PatchesLinearBounds, BoundedTensor) {
    let geometry = PatchGeometry::anchored(vec![0], vec![0]).unwrap();
    let shape: &[usize] = if explicit_rows {
        &[1, 1, 1, 1, 1, 1, 2]
    } else {
        &[1, 1, 1, 1, 1, 2]
    };
    let make_side = || PatchesData {
        coeff_err: None,
        patches: Some(ArrayD::from_shape_vec(IxDyn(shape), vec![1.0, -1.0]).unwrap()),
        geometry: geometry.clone(),
        identity: false,
        output_shape: (1, 1, 1),
        input_shape: (1, 1, 2),
        unstable_idx: None,
    };
    (
        PatchesLinearBounds {
            row_count: 1,
            lower_a: make_side(),
            lower_b: Array1::from_vec(vec![0.0]),
            upper_a: make_side(),
            upper_b: Array1::from_vec(vec![0.0]),
        },
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 2]), -1.0),
            ArrayD::from_elem(IxDyn(&[1, 1, 2]), 1.0),
        )
        .unwrap(),
    )
}

#[test]
fn anchored_tiny_intercepts_and_cancellation_are_daz_ftz_safe_in_6d_and_7d() {
    crate::bounds::patches::test_override::with_eager_err(false, || {
        let min_subnormal = f32::from_bits(1);
        for explicit_rows in [false, true] {
            for intercept in [min_subnormal, -min_subnormal] {
                let (bounds, pre_activation) = anchored_tiny_intercept_fixture(explicit_rows);
                let lower_before = bounds.lower_a.patches.as_ref().unwrap().clone();
                let upper_before = bounds.upper_a.patches.as_ref().unwrap().clone();
                let lower_bias_before = bounds.lower_b.clone();
                let upper_bias_before = bounds.upper_b.clone();

                let expired = std::time::Instant::now()
                    .checked_sub(std::time::Duration::from_millis(1))
                    .unwrap();
                let refused = crown_elementwise_backward_patches_with_deadline(
                    &bounds,
                    &pre_activation,
                    expired,
                    |_lower, _upper| LinearRelaxation::new(1.0, intercept, 1.0, intercept),
                );
                assert!(refused.as_ref().is_err_and(NyError::is_deadline_exceeded));
                assert_eq!(bounds.lower_a.patches.as_ref().unwrap(), &lower_before);
                assert_eq!(bounds.upper_a.patches.as_ref().unwrap(), &upper_before);
                assert_eq!(bounds.lower_b, lower_bias_before);
                assert_eq!(bounds.upper_b, upper_bias_before);

                let result = match crown_elementwise_backward_patches_with_deadline(
                    &bounds,
                    &pre_activation,
                    std::time::Instant::now() + std::time::Duration::from_secs(30),
                    |_lower, _upper| LinearRelaxation::new(1.0, intercept, 1.0, intercept),
                )
                .expect("Anchored tiny-intercept relaxation")
                {
                    CrownBounds::Patches(bounds) => bounds,
                    CrownBounds::Dense(_) => panic!("Anchored activation must stay Patches"),
                };

                // The exact two-tap intercept fold is +tiny - tiny = 0. The
                // emitted interval must still contain that value if a DAZ/FTZ
                // host erased either relaxation intercept or an intermediate
                // cancellation. Bias and coefficient centers cross the seam
                // only as zero, normal, or non-finite values.
                assert!(result.lower_b[0] <= 0.0, "lower cancellation escaped");
                assert!(result.upper_b[0] >= 0.0, "upper cancellation escaped");
                for &value in result
                    .lower_a
                    .patches
                    .as_ref()
                    .unwrap()
                    .iter()
                    .chain(result.upper_a.patches.as_ref().unwrap().iter())
                    .chain(result.lower_b.iter())
                    .chain(result.upper_b.iter())
                {
                    let magnitude = value.to_bits() & 0x7fff_ffff;
                    assert!(
                        magnitude == 0 || magnitude >= 0x0080_0000,
                        "published Anchored center is subnormal: {value:e}"
                    );
                }
            }
        }
    });
}

#[test]
fn anchored_elementwise_rejects_6d_row_count_mismatch_and_overflow_atomically() {
    let (mut row_mismatch, pre_activation) = mixed_sign_fixture();
    let anchored = PatchGeometry::anchored(vec![0], vec![0]).unwrap();
    row_mismatch.lower_a.geometry = anchored.clone();
    row_mismatch.upper_a.geometry = anchored.clone();
    row_mismatch.row_count = 2;
    let lower_before = row_mismatch.lower_a.patches.as_ref().unwrap().clone();
    let error =
        crown_elementwise_backward_patches(&row_mismatch, &pre_activation, patches_relaxation)
            .expect_err("6D row_count must equal output positions");
    assert!(matches!(error, NyError::ShapeMismatch { .. }), "{error:?}");
    assert_eq!(
        row_mismatch.lower_a.patches.as_ref().unwrap(),
        &lower_before
    );

    let (mut overflow, pre_activation) = mixed_sign_fixture();
    overflow.lower_a.geometry = anchored.clone();
    overflow.upper_a.geometry = anchored;
    overflow.lower_a.input_shape = (usize::MAX, 2, 1);
    overflow.upper_a.input_shape = (usize::MAX, 2, 1);
    let lower_before = overflow.lower_a.patches.as_ref().unwrap().clone();
    let upper_before = overflow.upper_a.patches.as_ref().unwrap().clone();
    let lower_bias_before = overflow.lower_b.clone();
    let upper_bias_before = overflow.upper_b.clone();
    let error = crown_elementwise_backward_patches(&overflow, &pre_activation, patches_relaxation)
        .expect_err("overflowing Anchored metadata must refuse before allocation");
    assert!(matches!(error, NyError::InvalidSpec(_)), "{error:?}");
    assert_eq!(overflow.lower_a.patches.as_ref().unwrap(), &lower_before);
    assert_eq!(overflow.upper_a.patches.as_ref().unwrap(), &upper_before);
    assert_eq!(overflow.lower_b, lower_bias_before);
    assert_eq!(overflow.upper_b, upper_bias_before);
}

#[test]
fn elementwise_patches_rejects_short_6d_coeff_error_and_poison_sanitizes_nan() {
    let (mut short, pre_activation) = mixed_sign_fixture();
    short.lower_a.coeff_err = Some(Array1::from_vec(vec![]));
    let error = crown_elementwise_backward_patches(&short, &pre_activation, patches_relaxation)
        .expect_err("a short 6D error receipt must not imply exact trailing rows");
    assert!(matches!(error, NyError::ShapeMismatch { .. }), "{error:?}");

    crate::bounds::patches::test_override::with_eager_err(false, || {
        let (mut poisoned, pre_activation) = mixed_sign_fixture();
        poisoned.lower_a.coeff_err = Some(Array1::from_vec(vec![f32::NAN]));
        let CrownBounds::Patches(result) =
            crown_elementwise_backward_patches(&poisoned, &pre_activation, patches_relaxation)
                .expect("a malformed value degrades outward instead of being treated as zero")
        else {
            panic!("expected Patches result");
        };
        let error = result.lower_a.coeff_err.as_ref().unwrap()[0];
        assert_eq!(error, f32::INFINITY);
        assert_eq!(result.lower_b[0], f32::NEG_INFINITY);
    });
}

#[test]
fn sparse_patches_refuses_coeff_error_until_sparse_transport_exists() {
    let idx = UnstableIdx {
        channels: vec![0],
        heights: vec![0],
        widths: vec![0],
    };
    let mut bounds = PatchesLinearBounds::sparse_identity((1, 1, 1), (1, 1, 1), idx);
    bounds.lower_a = bounds.lower_a.try_materialize_identity().unwrap();
    bounds.upper_a = bounds.upper_a.try_materialize_identity().unwrap();
    bounds.lower_a.coeff_err = Some(Array1::from_vec(vec![0.25]));
    let pre_activation =
        BoundedTensor::new(array![-1.0_f32].into_dyn(), array![1.0].into_dyn()).unwrap();
    let error = crown_elementwise_backward_patches(&bounds, &pre_activation, patches_relaxation)
        .expect_err("sparse coefficient error must remain a typed refusal");
    assert!(
        matches!(error, NyError::UnsupportedConfiguration(_)),
        "{error:?}"
    );
}

/// #patches-eager-err: with the fold DEFAULT-ON, the property that matters is
/// no longer "the value equals X" but "folding never buys tightness it has not
/// certified". Folding retires an already-certified per-row error OUTWARD into
/// the bias, so at the fold site the relation it produces must ENCLOSE the one
/// it replaces: the lower bias may only go down, the upper bias only up.
///
/// A violation in the other direction would be a FALSE PROOF — a bound narrower
/// than the arithmetic justifies. A widening is merely weaker. This is the check
/// the two optimized-vs-reference moats can no longer make for the folded
/// configuration, since they pin the unfolded kernel by construction.
#[test]
fn eager_fold_never_tightens_against_the_unfolded_path() {
    use crate::bounds::patches::test_override::with_eager_err;

    let run = |enabled: bool| {
        with_eager_err(enabled, || {
            let (bounds, pre_activation) = mixed_sign_fixture();
            let result =
                crown_elementwise_backward_patches(&bounds, &pre_activation, patches_relaxation)
                    .unwrap();
            let CrownBounds::Patches(result) = result else {
                panic!("expected patches output");
            };
            (result.lower_b[0], result.upper_b[0])
        })
    };

    let (unfolded_lo, unfolded_hi) = run(false);
    let (folded_lo, folded_hi) = run(true);

    assert!(
        folded_lo <= unfolded_lo,
        "folding tightened the LOWER bias ({folded_lo} > {unfolded_lo}): a bound narrower \
         than the unfolded path is a false proof, not an improvement"
    );
    assert!(
        folded_hi >= unfolded_hi,
        "folding tightened the UPPER bias ({folded_hi} < {unfolded_hi}): a bound narrower \
         than the unfolded path is a false proof, not an improvement"
    );
    // And the fold must actually be doing something here, or this test would
    // pass vacuously against a no-op gate.
    assert!(
        folded_lo < unfolded_lo || folded_hi > unfolded_hi,
        "fixture no longer exercises the fold: both biases identical ({folded_lo}, {folded_hi})"
    );
}

/// #patches-eager-err: the DEFAULT configuration, asserted positively.
///
/// This is the structural consequence the `*_coeff_err_*_bit_identical` pins
/// used to encode from the other side: with the fold on, the certified per-row
/// coefficient error is retired into the bias at the pre-activation cut, so the
/// outgoing relation carries NO `coeff_err` — and the bias it was folded into
/// has moved outward to pay for it.
///
/// Asserted against the same fixture and against the unfolded run, so it cannot
/// pass by the fold silently becoming a no-op.
#[test]
fn eager_fold_discharges_coeff_err_into_the_bias_by_default() {
    use crate::bounds::patches::test_override::with_eager_err;

    let run = |enabled: bool| {
        with_eager_err(enabled, || {
            let (bounds, pre_activation) = mixed_sign_fixture();
            let result =
                crown_elementwise_backward_patches(&bounds, &pre_activation, patches_relaxation)
                    .unwrap();
            let CrownBounds::Patches(result) = result else {
                panic!("expected patches output");
            };
            (
                result.lower_a.coeff_err.is_some(),
                result.upper_a.coeff_err.is_some(),
                result.lower_b[0],
            )
        })
    };

    let (unfolded_lo_err, unfolded_hi_err, unfolded_lo_b) = run(false);
    let (folded_lo_err, folded_hi_err, folded_lo_b) = run(true);

    assert!(
        unfolded_lo_err || unfolded_hi_err,
        "precondition: the unfolded path must actually emit a carried coeff_err here, \
         otherwise this fixture proves nothing about discharging it"
    );
    assert!(
        !folded_lo_err && !folded_hi_err,
        "with the fold on the carry must be fully discharged, but coeff_err survived \
         (lower={folded_lo_err}, upper={folded_hi_err})"
    );
    assert!(
        folded_lo_b < unfolded_lo_b,
        "the discharged error has to be PAID somewhere: the lower bias must widen \
         ({folded_lo_b} vs {unfolded_lo_b})"
    );
}

/// #patches-eager-err: pins the UNFOLDED kernel. Every expectation below is
/// derived from the patches/relaxation arithmetic itself; the eager error fold
/// is a POLICY applied at the ReLU backward call sites that widens the bias
/// outward on top of that arithmetic. Asserting the derivation with the fold off
/// keeps this test measuring what it was written to measure instead of
/// re-pinning it to constants the derivation no longer explains. The default
/// (folded) configuration is covered by
/// `eager_fold_never_tightens_against_the_unfolded_path` and
/// `eager_fold_discharges_coeff_err_into_the_bias_by_default`.
#[test]
fn test_crown_elementwise_backward_patches_dense_mixed_sign_coefficients() {
    crate::bounds::patches::test_override::with_eager_err(false, || {
        let bounds = PatchesLinearBounds {
            row_count: 1,
            lower_a: PatchesData {
                coeff_err: None,
                patches: Some(
                    ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 1, 2]), vec![1.5_f32, -2.0])
                        .unwrap(),
                ),
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: false,
                output_shape: (1, 1, 1),
                input_shape: (1, 1, 2),
                unstable_idx: None,
            },
            lower_b: array![0.25_f32],
            upper_a: PatchesData {
                coeff_err: None,
                patches: Some(
                    ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 1, 2]), vec![-0.75_f32, 3.0])
                        .unwrap(),
                ),
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: false,
                output_shape: (1, 1, 1),
                input_shape: (1, 1, 2),
                unstable_idx: None,
            },
            upper_b: array![-0.5_f32],
        };
        let pre_activation = BoundedTensor::new(
            array![-1.0_f32, -2.0].into_dyn(),
            array![2.0_f32, 4.0].into_dyn(),
        )
        .unwrap();

        let result =
            crown_elementwise_backward_patches(&bounds, &pre_activation, patches_relaxation)
                .unwrap();
        let CrownBounds::Patches(result) = result else {
            panic!("expected patches output");
        };

        let lower_patches = result.lower_a.patches.as_ref().expect("lower patches");
        let upper_patches = result.upper_a.patches.as_ref().expect("upper patches");

        assert_eq!(lower_patches[[0, 0, 0, 0, 0, 0]], next_down_f32(3.0));
        assert_eq!(lower_patches[[0, 0, 0, 0, 0, 1]], next_down_f32(-9.0));
        assert_eq!(upper_patches[[0, 0, 0, 0, 0, 0]], next_up_f32(-1.5));
        assert_eq!(upper_patches[[0, 0, 0, 0, 0, 1]], next_up_f32(13.5));

        assert_eq!(result.lower_b[0], next_down_f32(-8.375));
        assert_eq!(result.upper_b[0], next_up_f32(14.3125));
    });
}

#[test]
fn test_crown_elementwise_backward_patches_sparse_identity_refuses_without_error_transport() {
    let bounds = PatchesLinearBounds::sparse_identity(
        (1, 1, 3),
        (1, 1, 3),
        UnstableIdx {
            channels: vec![0, 0],
            heights: vec![0, 0],
            widths: vec![0, 2],
        },
    );
    let bounds = PatchesLinearBounds {
        row_count: 2,
        lower_b: array![0.1_f32, -0.2],
        upper_b: array![0.3_f32, 0.4],
        ..bounds
    };
    let pre_activation = BoundedTensor::new(
        array![-1.0_f32, -2.0, -3.0].into_dyn(),
        array![2.0_f32, 4.0, 6.0].into_dyn(),
    )
    .unwrap();

    let error = crown_elementwise_backward_patches(&bounds, &pre_activation, patches_relaxation)
        .expect_err("4D sparse activation cannot drop its intrinsic rounding error");
    assert!(
        matches!(error, NyError::UnsupportedConfiguration(_)),
        "{error:?}"
    );
}

#[test]
fn test_crown_elementwise_backward_patches_sparse_explicit_rows_also_refuses() {
    let idx = UnstableIdx {
        channels: vec![0, 0],
        heights: vec![0, 0],
        widths: vec![0, 1],
    };
    let side = |values: Vec<f32>| PatchesData {
        coeff_err: None,
        patches: Some(ArrayD::from_shape_vec(IxDyn(&[1, 2, 1, 1, 1]), values).unwrap()),
        geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
        identity: false,
        output_shape: (1, 1, 2),
        input_shape: (1, 1, 2),
        unstable_idx: Some(idx.clone()),
    };
    let bounds = PatchesLinearBounds {
        row_count: 1,
        lower_a: side(vec![0.3, -0.7]),
        lower_b: array![0.0],
        upper_a: side(vec![-0.2, 0.6]),
        upper_b: array![0.0],
    };
    let pre_activation = BoundedTensor::new(
        array![-1.0_f32, -2.0].into_dyn(),
        array![2.0, 3.0].into_dyn(),
    )
    .unwrap();
    let error = crown_elementwise_backward_patches(&bounds, &pre_activation, patches_relaxation)
        .expect_err("5D sparse activation cannot drop its intrinsic rounding error");
    assert!(
        matches!(error, NyError::UnsupportedConfiguration(_)),
        "{error:?}"
    );
}

// =====================================================================
// Byte-identity pin (#patches-coeff-err-soundness; 7D explicit-rows
// closure spec §6.4 T4, docs/PATCHES_7D_COEFF_ERR_CLOSURE.md).
//
// Committed against the UNMODIFIED tree: pins the exact bit patterns the
// CURRENT 6D activation backward emits (both coeff_err arrays, both
// biases including the incoming-err intercept discharge, and the full
// composed coefficient tensors). The 7D closure adds a 7D arm beside the
// 6D arm and must keep the 6D path byte-for-byte unchanged — this test
// must pass unmodified after it lands.
// =====================================================================

/// Deterministic non-dyadic mixed-sign fill with exact zeros for the pin.
fn pin_fill(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let k = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            if k.is_multiple_of(9) {
                0.0
            } else {
                (((k >> 7) % 4000) as f32 - 2000.0) * 0.000_731
            }
        })
        .collect()
}

/// Fixed 6D fixture for the byte-identity pin: [2,2,2,2,2,2] patches
/// (64 coefficients/side), stride 1, padding (1,0,1,0) (padding taps
/// exercised), nonzero incoming coeff_err on both sides (with exact-zero
/// rows), non-dyadic biases, mixed-regime pre-activation bounds, and the
/// nonzero-intercept `patches_relaxation` so the intercept discharge is live.
fn run_6d_err_pin_fixture() -> Box<PatchesLinearBounds> {
    let shape = [2usize, 2, 2, 2, 2, 2];
    let n: usize = shape.iter().product();
    let mk = |seed: u32, err: Vec<f32>| PatchesData {
        coeff_err: Some(Array1::from_vec(err)),
        patches: Some(ArrayD::from_shape_vec(IxDyn(&shape), pin_fill(n, seed)).unwrap()),
        geometry: PatchGeometry::affine((1, 1), (1, 0, 1, 0)),
        identity: false,
        output_shape: (2, 2, 2),
        input_shape: (2, 2, 2),
        unstable_idx: None,
    };
    let bounds = PatchesLinearBounds {
        row_count: 8,
        lower_a: mk(
            1,
            vec![1.0e-3, 0.0, 5.0e-4, 2.0e-3, 0.0, 1.0e-6, 7.0e-4, 3.0e-3],
        ),
        lower_b: array![0.13_f32, -0.7, 0.29, -0.011, 0.53, 0.0, -1.21, 0.077],
        upper_a: mk(
            2,
            vec![2.0e-3, 1.0e-4, 0.0, 5.0e-5, 4.0e-4, 0.0, 6.0e-4, 8.0e-4],
        ),
        upper_b: array![0.41_f32, 0.09, -0.33, 0.72, -0.005, 1.13, 0.0, -0.86],
    };
    let pre_activation = BoundedTensor::new(
        array![-1.3_f32, 0.2, -0.45, -2.7, 0.9, -0.05, -1.15, 0.33].into_dyn(),
        array![0.7_f32, 1.9, 0.6, -0.9, 2.1, 1.4, 0.02, 0.87].into_dyn(),
    )
    .unwrap();
    let result =
        crown_elementwise_backward_patches(&bounds, &pre_activation, patches_relaxation).unwrap();
    let CrownBounds::Patches(res) = result else {
        panic!("expected patches output");
    };
    res
}

fn assert_pinned_bits(label: &str, actual: &[f32], expected_bits: &[u32]) {
    assert_eq!(
        actual.len(),
        expected_bits.len(),
        "{label}: length mismatch"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected_bits.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e,
            "{label}[{i}]: got {a:?} (bits {:#010x}), pinned {:#010x}",
            a.to_bits(),
            e
        );
    }
}

/// T4 (spec §6.4): byte-identity pin for the 6D activation backward. Bit
/// literals captured from the UNMODIFIED (pre-closure) tree via the (now
/// deleted) capture harness; the 7D closure must keep this green unmodified.
/// #patches-eager-err: pins the UNFOLDED kernel. Every expectation below is
/// derived from the patches/relaxation arithmetic itself; the eager error fold
/// is a POLICY applied at the ReLU backward call sites that widens the bias
/// outward on top of that arithmetic. Asserting the derivation with the fold off
/// keeps this test measuring what it was written to measure instead of
/// re-pinning it to constants the derivation no longer explains. The default
/// (folded) configuration is covered by
/// `eager_fold_never_tightens_against_the_unfolded_path` and
/// `eager_fold_discharges_coeff_err_into_the_bias_by_default`.
#[test]
fn test_patches_backward_6d_err_byte_identical_pin() {
    crate::bounds::patches::test_override::with_eager_err(false, || {
        const PIN_LOWER_ERR_BITS: [u32; 8] = [
            0x3b9375fc, 0x34366b3d, 0x3b1378de, 0x3c1375fc, 0x34066193, 0x369b1e99, 0x3b4e7535,
            0x3c5d3037,
        ];
        const PIN_UPPER_ERR_BITS: [u32; 8] = [
            0x3c13751c, 0x39ebfb8f, 0x348effab, 0x396c4c82, 0x3aebf15e, 0x33b5d201, 0x3b30f73f,
            0x3b6bf0e9,
        ];
        const PIN_LOWER_B_BITS: [u32; 8] = [
            0x3f3bafe1, 0xc03fa67b, 0xc057abb6, 0xc138b07a, 0x3fb25935, 0xc01ba0c0, 0xc0a12d1a,
            0xc0f2fc28,
        ];
        const PIN_UPPER_B_BITS: [u32; 8] = [
            0x405fdb7e, 0x3f3f51ff, 0x3ec504a9, 0xc01e7001, 0x40876eb0, 0x3ed50231, 0x3e2b57a7,
            0xbf111e35,
        ];
        const PIN_LOWER_PATCH_BITS: [u32; 64] = [
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x3f813ed6, 0x00000000, 0x00000000, 0x40013b3f, 0x00000000, 0x00000000, 0x00000000,
            0xbf8495fb, 0xbfd4fac0, 0x00000000, 0xbd7f1a93, 0x00000000, 0x00000000, 0x00000000,
            0xc05d0458, 0x00000000, 0x3fb06d59, 0xbf96cb51, 0xc0601d35, 0x3fb74b86, 0x00000000,
            0xc0392237, 0x3e4ac7d1, 0xbe19c03e, 0xbf865759, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3fb69447, 0x00000000, 0x00000000,
            0xbd6d2386, 0x00000000, 0x00000000, 0x00000000, 0xbeee6949, 0xbf9f77cc, 0x00000000,
            0x3ec52108, 0x00000000, 0x00000000, 0x00000000, 0xc0388685, 0x00000000, 0xbe17ce03,
            0xbf6a382d, 0xc03e6dfb, 0x3db65866, 0x00000000, 0xc014a463, 0x3edb4933, 0xbd13e98d,
            0x3fd1b0c8,
        ];
        const PIN_UPPER_PATCH_BITS: [u32; 64] = [
            0x00000000, 0x00000000, 0x00000000, 0x3f4b744c, 0x00000000, 0x00000000, 0x00000000,
            0x3fb0dcbb, 0x00000000, 0x00000000, 0x3f86d9a8, 0xbf64d473, 0x00000000, 0x00000000,
            0xbf41c781, 0xbf6b65fa, 0x00000000, 0x00000000, 0x00000000, 0x3f596090, 0x00000000,
            0xc021832b, 0x00000000, 0x3eaaaeda, 0xc01082d6, 0x00000000, 0x3f8b0d2a, 0xc0166cb1,
            0xc0074a3a, 0x3eb777d8, 0xbf1eecdc, 0xbf826b37, 0x00000000, 0x00000000, 0x00000000,
            0x3f876960, 0x00000000, 0x00000000, 0x00000000, 0x3ff9d863, 0x00000000, 0x00000000,
            0xbde34208, 0xbf2175ff, 0x00000000, 0x00000000, 0xbeae393e, 0xbf304107, 0x00000000,
            0x00000000, 0x00000000, 0x3f8b90e8, 0x00000000, 0xc006d873, 0x00000000, 0xbf1ce9dd,
            0xbfe075d3, 0x00000000, 0x3d8a54b9, 0xbfc4fd1a, 0xbfd93f05, 0x3f4666cf, 0xbe18e3de,
            0x3fd7ff40,
        ];

        let res = run_6d_err_pin_fixture();
        assert_pinned_bits(
            "PIN_LOWER_ERR_BITS",
            res.lower_a.coeff_err.as_ref().unwrap().as_slice().unwrap(),
            &PIN_LOWER_ERR_BITS,
        );
        assert_pinned_bits(
            "PIN_UPPER_ERR_BITS",
            res.upper_a.coeff_err.as_ref().unwrap().as_slice().unwrap(),
            &PIN_UPPER_ERR_BITS,
        );
        assert_pinned_bits(
            "PIN_LOWER_B_BITS",
            res.lower_b.as_slice().unwrap(),
            &PIN_LOWER_B_BITS,
        );
        assert_pinned_bits(
            "PIN_UPPER_B_BITS",
            res.upper_b.as_slice().unwrap(),
            &PIN_UPPER_B_BITS,
        );
        assert_pinned_bits(
            "PIN_LOWER_PATCH_BITS",
            res.lower_a.patches.as_ref().unwrap().as_slice().unwrap(),
            &PIN_LOWER_PATCH_BITS,
        );
        assert_pinned_bits(
            "PIN_UPPER_PATCH_BITS",
            res.upper_a.patches.as_ref().unwrap().as_slice().unwrap(),
            &PIN_UPPER_PATCH_BITS,
        );
    });
}

// =====================================================================
// 7D explicit-rows err lift (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §6.4):
// T1 f64 oracle coverage, T2 length-mismatch hard error, T3 gap-only
// emission on exact inputs, T5 serial-vs-parallel bitwise equality.
// The err index on this layout is the SPEC row (axis 0).
// =====================================================================

use ny_core::NyError;

/// 7D fixture geometry: [row=2, oc=2, oh=1, ow=2, ic=2, ki=1, kj=2],
/// stride (1,1), padding (left=1, 0, 0, 0) — the (ow=0, kj=0) taps map to
/// iw = −1 (out-of-bounds padding taps, exercised per §6.4 T1).
const F7_SHAPE: [usize; 7] = [2, 2, 1, 2, 2, 1, 2];
const F7_INPUT_SHAPE: (usize, usize, usize) = (2, 1, 2);
const F7_PADDING: (usize, usize, usize, usize) = (1, 0, 0, 0);

/// Input-neuron flat index for a tap of the T1 fixture, `None` for padding
/// taps. Mirrors the padding predicate + `input_flat` mapping of the
/// production loop.
fn f7_input_flat(oh: usize, ow: usize, ki: usize, kj: usize) -> Option<usize> {
    let (_, in_h, in_w) = F7_INPUT_SHAPE;
    let ih_raw = (oh + ki) as isize; // stride 1, pad_top 0
    let iw_raw = (ow + kj) as isize - 1; // stride 1, pad_left 1
    if ih_raw < 0 || (ih_raw as usize) >= in_h || iw_raw < 0 || (iw_raw as usize) >= in_w {
        None
    } else {
        Some(ih_raw as usize * in_w + iw_raw as usize)
    }
}

fn mk_7d_fixture(
    lower_err: Option<Vec<f32>>,
    upper_err: Option<Vec<f32>>,
) -> (PatchesLinearBounds, BoundedTensor) {
    let n: usize = F7_SHAPE.iter().product();
    let mk = |seed: u32, err: Option<Vec<f32>>| PatchesData {
        coeff_err: err.map(Array1::from_vec),
        patches: Some(ArrayD::from_shape_vec(IxDyn(&F7_SHAPE), pin_fill(n, seed)).unwrap()),
        geometry: PatchGeometry::affine((1, 1), F7_PADDING),
        identity: false,
        output_shape: (2, 1, 2),
        input_shape: F7_INPUT_SHAPE,
        unstable_idx: None,
    };
    let bounds = PatchesLinearBounds {
        row_count: 2,
        lower_a: mk(3, lower_err),
        lower_b: array![0.25_f32, -0.5],
        upper_a: mk(4, upper_err),
        upper_b: array![-0.125_f32, 0.375],
    };
    // 4 input neurons (2, 1, 2); mixed-regime bounds so patches_relaxation
    // yields 4 distinct relaxations with nonzero intercepts.
    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), vec![-1.0_f32, 0.5, -0.75, 0.25]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), vec![2.0_f32, 1.5, 0.5, 0.8]).unwrap(),
    )
    .unwrap();
    (bounds, pre_activation)
}

/// f64 mirrors of the fixture's per-neuron relaxations.
struct RelaxF64 {
    ls: f64,
    li: f64,
    us: f64,
    ui: f64,
}

fn f7_relaxations() -> Vec<RelaxF64> {
    let pre_l = [-1.0_f32, 0.5, -0.75, 0.25];
    let pre_u = [2.0_f32, 1.5, 0.5, 0.8];
    pre_l
        .iter()
        .zip(pre_u.iter())
        .map(|(&l, &u)| {
            let r = patches_relaxation(l, u);
            RelaxF64 {
                ls: f64::from(r.lower_slope),
                li: f64::from(r.lower_intercept),
                us: f64::from(r.upper_slope),
                ui: f64::from(r.upper_intercept),
            }
        })
        .collect()
}

/// Composed coefficient and bias contribution for a candidate true incoming
/// coefficient `c`, LOWER side (sign selection exactly as `compose_lower`).
fn comp_lower_f64(c: f64, r: &RelaxF64) -> (f64, f64) {
    if c > 0.0 {
        (c * r.ls, c * r.li)
    } else if c < 0.0 {
        (c * r.us, c * r.ui)
    } else {
        (0.0, 0.0)
    }
}

/// UPPER-side mirror (sign selection exactly as `compose_upper`).
fn comp_upper_f64(c: f64, r: &RelaxF64) -> (f64, f64) {
    if c > 0.0 {
        (c * r.us, c * r.ui)
    } else if c < 0.0 {
        (c * r.ls, c * r.li)
    } else {
        (0.0, 0.0)
    }
}

/// Candidate true incoming coefficients for stored `a` under row err `e`:
/// the endpoints of `[a−e, a+e]` plus 0 when the interval straddles it.
/// Sufficient for extremal composed coefficient/intercept: both maps are
/// piecewise linear in the true coefficient with the only breakpoint at 0.
fn err_candidates(a: f64, e: f64) -> Vec<f64> {
    let mut c = vec![a - e, a + e];
    if a - e < 0.0 && 0.0 < a + e {
        c.push(0.0);
    }
    c
}

/// Shared f64 oracle for the 7D arm (§6.4 T1 semantics, no tolerance
/// epsilon anywhere): per valid tap every candidate composed coefficient is
/// within the emitted row err of the stored one; padding taps are stored 0
/// exactly; the output biases are outside `b + Σ min/max candidate
/// intercept folds`. `strict_bias` asserts strict inequality (T3: proves
/// the gbar·ABS discharge + directed cast move outward even at e = 0).
fn check_7d_oracle(input: &PatchesLinearBounds, output: &PatchesLinearBounds, strict_bias: bool) {
    let relax = f7_relaxations();
    let [rows, out_c, out_h, out_w, in_c, kh, kw] = F7_SHAPE;
    let (_, in_h, in_w) = F7_INPUT_SHAPE;
    let old_l = input.lower_a.patches.as_ref().unwrap();
    let old_u = input.upper_a.patches.as_ref().unwrap();
    let new_l = output.lower_a.patches.as_ref().unwrap();
    let new_u = output.upper_a.patches.as_ref().unwrap();
    let err_l = output.lower_a.coeff_err.as_ref();
    let err_u = output.upper_a.coeff_err.as_ref();
    if let Some(err_l) = err_l {
        assert_eq!(
            err_l.len(),
            rows,
            "err index is the spec row (len == row_count)"
        );
    }
    if let Some(err_u) = err_u {
        assert_eq!(err_u.len(), rows);
    }

    for row in 0..rows {
        let e_l = input
            .lower_a
            .coeff_err
            .as_ref()
            .map_or(0.0, |e| f64::from(e[row]));
        let e_u = input
            .upper_a
            .coeff_err
            .as_ref()
            .map_or(0.0, |e| f64::from(e[row]));
        let ne_l = err_l.map(|err| f64::from(err[row]));
        let ne_u = err_u.map(|err| f64::from(err[row]));
        let mut bias_min_l = f64::from(input.lower_b[row]);
        let mut bias_max_u = f64::from(input.upper_b[row]);
        let sample_fractions = [0.0f64, 1.0, 0.5, 0.25, 0.75];
        let pre_l = [-1.0f64, 0.5, -0.75, 0.25];
        let pre_u = [2.0f64, 1.5, 0.5, 0.8];
        let mut stored_eval_l = [f64::from(output.lower_b[row]); 5];
        let mut stored_eval_u = [f64::from(output.upper_b[row]); 5];
        let mut oracle_eval_l = [f64::from(input.lower_b[row]); 5];
        let mut oracle_eval_u = [f64::from(input.upper_b[row]); 5];
        let mut saw_padding = false;
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let idx = [row, oc, oh, ow, ic, ki, kj];
                                let Some(pos) = f7_input_flat(oh, ow, ki, kj) else {
                                    // Padding taps: never composed, stored 0
                                    // exactly on both sides.
                                    assert_eq!(new_l[idx], 0.0, "padding tap not 0 (lower)");
                                    assert_eq!(new_u[idx], 0.0, "padding tap not 0 (upper)");
                                    saw_padding = true;
                                    continue;
                                };
                                let input_flat = ic * in_h * in_w + pos;
                                let r = &relax[input_flat];

                                let a_l = f64::from(old_l[idx]);
                                let stored_l = f64::from(new_l[idx]);
                                let mut min_h = f64::INFINITY;
                                let lower_candidates = err_candidates(a_l, e_l);
                                for &cand in &lower_candidates {
                                    let (c_ideal, h) = comp_lower_f64(cand, r);
                                    if let Some(ne_l) = ne_l {
                                        assert!(
                                            (stored_l - c_ideal).abs() <= ne_l,
                                            "lower coeff row {row} tap {idx:?}: \
                                             |{stored_l} - {c_ideal}| > err {ne_l}",
                                        );
                                    }
                                    if h < min_h {
                                        min_h = h;
                                    }
                                }
                                bias_min_l += min_h;

                                let a_u = f64::from(old_u[idx]);
                                let stored_u = f64::from(new_u[idx]);
                                let mut max_h = f64::NEG_INFINITY;
                                let upper_candidates = err_candidates(a_u, e_u);
                                for &cand in &upper_candidates {
                                    let (c_ideal, h) = comp_upper_f64(cand, r);
                                    if let Some(ne_u) = ne_u {
                                        assert!(
                                            (stored_u - c_ideal).abs() <= ne_u,
                                            "upper coeff row {row} tap {idx:?}: \
                                             |{stored_u} - {c_ideal}| > err {ne_u}",
                                        );
                                    }
                                    if h > max_h {
                                        max_h = h;
                                    }
                                }
                                bias_max_u += max_h;

                                if ne_l.is_none() || ne_u.is_none() {
                                    for (sample, &fraction) in sample_fractions.iter().enumerate() {
                                        let y = pre_l[input_flat]
                                            + fraction * (pre_u[input_flat] - pre_l[input_flat]);
                                        stored_eval_l[sample] += stored_l * y;
                                        stored_eval_u[sample] += stored_u * y;
                                        if ne_l.is_none() {
                                            oracle_eval_l[sample] += lower_candidates
                                                .iter()
                                                .map(|&cand| {
                                                    let (c, h) = comp_lower_f64(cand, r);
                                                    c * y + h
                                                })
                                                .fold(f64::INFINITY, f64::min);
                                        }
                                        if ne_u.is_none() {
                                            oracle_eval_u[sample] += upper_candidates
                                                .iter()
                                                .map(|&cand| {
                                                    let (c, h) = comp_upper_f64(cand, r);
                                                    c * y + h
                                                })
                                                .fold(f64::NEG_INFINITY, f64::max);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(saw_padding, "fixture must exercise out-of-bounds taps");

        let out_bl = f64::from(output.lower_b[row]);
        let out_bu = f64::from(output.upper_b[row]);
        if strict_bias {
            assert!(
                out_bl < bias_min_l,
                "lower bias row {row}: {out_bl} not strictly below oracle {bias_min_l} \
                 (gbar·ABS discharge / directed cast missing?)",
            );
            assert!(
                out_bu > bias_max_u,
                "upper bias row {row}: {out_bu} not strictly above oracle {bias_max_u}",
            );
        } else {
            assert!(
                out_bl <= bias_min_l,
                "lower bias row {row}: {out_bl} > oracle min {bias_min_l}",
            );
            assert!(
                out_bu >= bias_max_u,
                "upper bias row {row}: {out_bu} < oracle max {bias_max_u}",
            );
        }
        if ne_l.is_none() {
            for sample in 0..sample_fractions.len() {
                assert!(
                    stored_eval_l[sample] <= oracle_eval_l[sample],
                    "folded lower expression row {row} sample {sample}: {} > oracle {}",
                    stored_eval_l[sample],
                    oracle_eval_l[sample]
                );
            }
        }
        if ne_u.is_none() {
            for sample in 0..sample_fractions.len() {
                assert!(
                    stored_eval_u[sample] >= oracle_eval_u[sample],
                    "folded upper expression row {row} sample {sample}: {} < oracle {}",
                    stored_eval_u[sample],
                    oracle_eval_u[sample]
                );
            }
        }
    }
}

fn run_7d(bounds: &PatchesLinearBounds, pre: &BoundedTensor) -> Box<PatchesLinearBounds> {
    let result = crown_elementwise_backward_patches(bounds, pre, patches_relaxation).unwrap();
    let CrownBounds::Patches(res) = result else {
        panic!("expected patches output");
    };
    res
}

/// T1 (spec §6.4): 7D explicit-rows arm emits per-SPEC-row errs that cover
/// the f64 oracle over every admissible true incoming coefficient (endpoint +
/// sign-flip candidates), with padding taps stored exactly 0 and biases
/// outside the candidate-extremal intercept folds. Row 0 carries nonzero
/// incoming err on both sides (0.75 / 0.5 — sign flips live); row 1 is
/// exact. No tolerance epsilon anywhere.
#[test]
fn test_patches_backward_7d_err_covers_f64_oracle() {
    let (bounds, pre) = mk_7d_fixture(Some(vec![0.75, 0.0]), Some(vec![0.5, 0.0]));
    // Fixture sanity: mixed-sign values with exact zeros among the taps.
    let lp = bounds.lower_a.patches.as_ref().unwrap();
    assert!(lp.iter().any(|&v| v > 0.0) && lp.iter().any(|&v| v < 0.0));
    assert!(lp.iter().any(|&v| v == 0.0));

    let res = run_7d(&bounds, &pre);
    check_7d_oracle(&bounds, &res, false);

    let mut eager = res;
    eager.fold_coeff_err_over_box_eager_with_policy(&pre, true);
    assert!(
        eager.lower_a.coeff_err.is_none() && eager.upper_a.coeff_err.is_none(),
        "valid 7D plain-ReLU carrier must discharge under the explicit gate"
    );
    check_7d_oracle(&bounds, &eager, false);
}

/// T2 (spec §6.4 / I6): a carried 7D err whose length != row_count is a
/// construction bug — hard Err(ShapeMismatch), never a silent under-count.
#[test]
fn test_patches_backward_7d_err_wrong_length_rejected() {
    for (le, ue) in [
        (Some(vec![0.1_f32, 0.2, 0.3]), None),
        (None, Some(vec![0.1_f32])),
    ] {
        let (bounds, pre) = mk_7d_fixture(le, ue);
        let err = crown_elementwise_backward_patches(&bounds, &pre, patches_relaxation)
            .expect_err("length-mismatched 7D err must be rejected");
        assert!(
            matches!(err, NyError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got {err:?}",
        );
    }
}

/// T3 (spec §6.4): exact inputs (None err both sides) still emit Some errs —
/// the directed-rounding gap terms are intrinsic — with each entry tightly
/// bounded by the computed per-row max gap, and biases strictly outside the
/// e=0 oracle (proves the gbar·ABS compose-fold discharge is present and
/// outward).
#[test]
fn test_patches_backward_7d_none_err_emits_gap_only() {
    let (bounds, pre) = mk_7d_fixture(None, None);
    let res = run_7d(&bounds, &pre);

    // Entries: gap-covering from below, within one outward f32 step of the
    // f64 row max gap from above.
    let relax = f7_relaxations();
    let [rows, out_c, out_h, out_w, in_c, kh, kw] = F7_SHAPE;
    let (_, in_h, in_w) = F7_INPUT_SHAPE;
    for row in 0..rows {
        let mut gap_l = 0.0f64;
        let mut gap_u = 0.0f64;
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for ic in 0..in_c {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let Some(pos) = f7_input_flat(oh, ow, ki, kj) else {
                                    continue;
                                };
                                let r = &relax[ic * in_h * in_w + pos];
                                let idx = [row, oc, oh, ow, ic, ki, kj];
                                let a_l = f64::from(bounds.lower_a.patches.as_ref().unwrap()[idx]);
                                let (ideal_l, _) = comp_lower_f64(a_l, r);
                                let stored_l =
                                    f64::from(res.lower_a.patches.as_ref().unwrap()[idx]);
                                gap_l = gap_l.max((stored_l - ideal_l).abs());
                                let a_u = f64::from(bounds.upper_a.patches.as_ref().unwrap()[idx]);
                                let (ideal_u, _) = comp_upper_f64(a_u, r);
                                let stored_u =
                                    f64::from(res.upper_a.patches.as_ref().unwrap()[idx]);
                                gap_u = gap_u.max((stored_u - ideal_u).abs());
                            }
                        }
                    }
                }
            }
        }
        let ne_l = f64::from(res.lower_a.coeff_err.as_ref().expect("Some out")[row]);
        let ne_u = f64::from(res.upper_a.coeff_err.as_ref().expect("Some out")[row]);
        assert!(ne_l >= gap_l, "row {row} lower err {ne_l} < gap {gap_l}");
        assert!(ne_u >= gap_u, "row {row} upper err {ne_u} < gap {gap_u}");
        assert!(
            ne_l <= f64::from(next_up_f32(gap_l as f32)),
            "row {row} lower err {ne_l} looser than the 1-ulp gap bound",
        );
        assert!(
            ne_u <= f64::from(next_up_f32(gap_u as f32)),
            "row {row} upper err {ne_u} looser than the 1-ulp gap bound",
        );
    }

    // Biases: e = 0 oracle, strict (discharge present and outward).
    check_7d_oracle(&bounds, &res, true);
}

/// Rebuild a logically-identical 7D tensor with reversed memory layout so
/// `as_slice()` returns `None` (forces the serial indexed fallback).
fn to_noncontiguous_7d(arr: &ArrayD<f32>) -> ArrayD<f32> {
    let shape = arr.shape().to_vec();
    let rev_shape: Vec<usize> = shape.iter().rev().copied().collect();
    let perm: Vec<usize> = (0..shape.len()).rev().collect();
    let mut buf = ArrayD::<f32>::zeros(IxDyn(&rev_shape));
    buf.view_mut().permuted_axes(perm.clone()).assign(arr);
    let nc = buf.permuted_axes(perm);
    assert_eq!(nc.shape(), arr.shape());
    assert!(nc.as_slice().is_none(), "fixture must be non-contiguous");
    assert_eq!(&nc, arr);
    nc
}

/// T5 (spec §6.4): the parallel row driver and the serial indexed fallback
/// (reached via a non-contiguous, permuted-axes input) produce bitwise
/// identical patches, biases, and err arrays.
#[test]
fn test_patches_backward_7d_serial_fallback_matches_parallel() {
    let (bounds, pre) = mk_7d_fixture(Some(vec![0.75, 0.0]), Some(vec![0.5, 1.0e-3]));
    let res_par = run_7d(&bounds, &pre);

    let mut bounds_nc = bounds.clone();
    bounds_nc.lower_a.patches = Some(to_noncontiguous_7d(
        bounds.lower_a.patches.as_ref().unwrap(),
    ));
    bounds_nc.upper_a.patches = Some(to_noncontiguous_7d(
        bounds.upper_a.patches.as_ref().unwrap(),
    ));
    let res_ser = run_7d(&bounds_nc, &pre);

    let bits = |xs: &[f32]| xs.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let iter_bits = |a: &ArrayD<f32>| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        iter_bits(res_par.lower_a.patches.as_ref().unwrap()),
        iter_bits(res_ser.lower_a.patches.as_ref().unwrap()),
        "lower patches diverge",
    );
    assert_eq!(
        iter_bits(res_par.upper_a.patches.as_ref().unwrap()),
        iter_bits(res_ser.upper_a.patches.as_ref().unwrap()),
        "upper patches diverge",
    );
    assert_eq!(
        bits(res_par.lower_b.as_slice().unwrap()),
        bits(res_ser.lower_b.as_slice().unwrap()),
        "lower bias diverges",
    );
    assert_eq!(
        bits(res_par.upper_b.as_slice().unwrap()),
        bits(res_ser.upper_b.as_slice().unwrap()),
        "upper bias diverges",
    );
    assert_eq!(
        bits(
            res_par
                .lower_a
                .coeff_err
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap()
        ),
        bits(
            res_ser
                .lower_a
                .coeff_err
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap()
        ),
        "lower err diverges",
    );
    assert_eq!(
        bits(
            res_par
                .upper_a
                .coeff_err
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap()
        ),
        bits(
            res_ser
                .upper_a
                .coeff_err
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap()
        ),
        "upper err diverges",
    );
}

fn assert_explicit_row_results_bitwise_equal(
    label: &str,
    actual: &PatchesLinearBounds,
    expected: &PatchesLinearBounds,
) {
    let bits = |xs: &[f32]| xs.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    for (field, actual, expected) in [
        (
            "lower patches",
            actual.lower_a.patches.as_ref().unwrap().as_slice().unwrap(),
            expected
                .lower_a
                .patches
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap(),
        ),
        (
            "upper patches",
            actual.upper_a.patches.as_ref().unwrap().as_slice().unwrap(),
            expected
                .upper_a
                .patches
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap(),
        ),
        (
            "lower bias",
            actual.lower_b.as_slice().unwrap(),
            expected.lower_b.as_slice().unwrap(),
        ),
        (
            "upper bias",
            actual.upper_b.as_slice().unwrap(),
            expected.upper_b.as_slice().unwrap(),
        ),
        (
            "lower coeff error",
            actual
                .lower_a
                .coeff_err
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap(),
            expected
                .lower_a
                .coeff_err
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap(),
        ),
        (
            "upper coeff error",
            actual
                .upper_a
                .coeff_err
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap(),
            expected
                .upper_a
                .coeff_err
                .as_ref()
                .unwrap()
                .as_slice()
                .unwrap(),
        ),
    ] {
        assert_eq!(
            bits(actual),
            bits(expected),
            "{label}: {field} differs from the historical result"
        );
    }
    assert_eq!(actual.row_count, expected.row_count, "{label}: row count");
    assert_eq!(actual.lower_a.geometry, expected.lower_a.geometry);
    assert_eq!(actual.upper_a.geometry, expected.upper_a.geometry);
    assert_eq!(actual.lower_a.output_shape, expected.lower_a.output_shape);
    assert_eq!(actual.upper_a.output_shape, expected.upper_a.output_shape);
    assert_eq!(actual.lower_a.input_shape, expected.lower_a.input_shape);
    assert_eq!(actual.upper_a.input_shape, expected.upper_a.input_shape);
}

fn expect_patches(result: CrownBounds) -> Box<PatchesLinearBounds> {
    let CrownBounds::Patches(result) = result else {
        panic!("expected Patches result");
    };
    result
}

#[test]
fn finite_deadline_explicit_rows_are_bitwise_legacy_in_serial_and_parallel() {
    let (bounds, pre) = mk_7d_fixture(Some(vec![0.75, 0.0]), Some(vec![0.5, 1.0e-3]));
    let legacy = run_7d(&bounds, &pre);
    let serial = expect_patches(
        crown_elementwise_backward_patches_with_poll_for_test(
            &bounds,
            &pre,
            patches_relaxation,
            false,
            &|| Ok(()),
        )
        .expect("serial cooperative transaction"),
    );
    let parallel = expect_patches(
        crown_elementwise_backward_patches_with_poll_for_test(
            &bounds,
            &pre,
            patches_relaxation,
            true,
            &|| Ok(()),
        )
        .expect("parallel cooperative transaction"),
    );
    let deadline = expect_patches(
        crown_elementwise_backward_patches_with_deadline(
            &bounds,
            &pre,
            std::time::Instant::now() + std::time::Duration::from_secs(30),
            patches_relaxation,
        )
        .expect("production finite-deadline transaction"),
    );

    assert_explicit_row_results_bitwise_equal("serial cooperative", &serial, &legacy);
    assert_explicit_row_results_bitwise_equal("parallel cooperative", &parallel, &legacy);
    assert_explicit_row_results_bitwise_equal("production deadline", &deadline, &legacy);
}

#[test]
fn finite_deadline_expired_before_start_publishes_nothing() {
    let (bounds, pre) = mk_7d_fixture(Some(vec![0.75, 0.0]), Some(vec![0.5, 1.0e-3]));
    let lower_before = bounds.lower_a.patches.as_ref().unwrap().clone();
    let upper_before = bounds.upper_a.patches.as_ref().unwrap().clone();
    let expired = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_millis(1))
        .expect("one millisecond fits before the current instant");

    let result = crown_elementwise_backward_patches_with_deadline(
        &bounds,
        &pre,
        expired,
        patches_relaxation,
    );
    assert!(
        result.as_ref().is_err_and(NyError::is_deadline_exceeded),
        "expired authority must fail before work, got {result:?}"
    );
    let published: Option<CrownBounds> = result.ok();
    assert!(
        published.is_none(),
        "an expired transaction cannot publish CrownBounds"
    );
    assert_eq!(
        bounds.lower_a.patches.as_ref().unwrap(),
        &lower_before,
        "expired work cannot mutate its lower input"
    );
    assert_eq!(
        bounds.upper_a.patches.as_ref().unwrap(),
        &upper_before,
        "expired work cannot mutate its upper input"
    );
}

fn mk_deadline_poll_fixture() -> (PatchesLinearBounds, BoundedTensor) {
    // 8,192 coordinates per row: two bounded-cadence polls occur inside each
    // full compose/error pass, rather than only at row boundaries.
    let shape = [2usize, 2, 2, 2, 8, 8, 8];
    let n: usize = shape.iter().product();
    let mk = |seed| PatchesData {
        coeff_err: Some(Array1::from_vec(vec![1.0e-4, 2.0e-4])),
        patches: Some(ArrayD::from_shape_vec(IxDyn(&shape), pin_fill(n, seed)).unwrap()),
        geometry: PatchGeometry::affine((1, 1), (3, 4, 3, 4)),
        identity: false,
        output_shape: (2, 2, 2),
        input_shape: (8, 2, 2),
        unstable_idx: None,
    };
    let bounds = PatchesLinearBounds {
        row_count: 2,
        lower_a: mk(0x51a7),
        lower_b: array![0.25_f32, -0.5],
        upper_a: mk(0x91d3),
        upper_b: array![-0.125_f32, 0.375],
    };
    let pre = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[8, 2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[8, 2, 2]), 2.0_f32),
    )
    .unwrap();
    (bounds, pre)
}

#[test]
fn finite_deadline_midflight_error_discards_partial_scratch() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let (bounds, pre) = mk_deadline_poll_fixture();
    let lower_before = bounds.lower_a.patches.as_ref().unwrap().clone();
    let upper_before = bounds.upper_a.patches.as_ref().unwrap().clone();
    let poll_count = AtomicUsize::new(0);
    let injected = || {
        // Entry + three allocation-boundary polls + row entry pass; fail on
        // the first 4,096-coordinate poll inside compose row zero.
        if poll_count.fetch_add(1, Ordering::SeqCst) >= 5 {
            Err(NyError::DeadlineExceeded(
                "injected explicit-row ReLU deadline".into(),
            ))
        } else {
            Ok(())
        }
    };

    let result = crown_elementwise_backward_patches_with_poll_for_test(
        &bounds,
        &pre,
        patches_relaxation,
        false,
        &injected,
    );
    assert!(
        result.as_ref().is_err_and(NyError::is_deadline_exceeded),
        "injected mid-row expiry must reject the transaction, got {result:?}"
    );
    assert!(
        poll_count.load(Ordering::SeqCst) >= 6,
        "expiry must occur after an in-row bounded-cadence poll"
    );
    let published: Option<CrownBounds> = result.ok();
    assert!(
        published.is_none(),
        "partial coefficient/bias scratch cannot be published"
    );
    assert_eq!(bounds.lower_a.patches.as_ref().unwrap(), &lower_before);
    assert_eq!(bounds.upper_a.patches.as_ref().unwrap(), &upper_before);

    let retry = crown_elementwise_backward_patches_with_poll_for_test(
        &bounds,
        &pre,
        patches_relaxation,
        true,
        &|| Ok(()),
    );
    assert!(
        retry.is_ok(),
        "a fresh parallel transaction must complete after discard"
    );
}

#[test]
fn finite_deadline_midflight_certificate_error_discards_compose_scratch() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let (bounds, pre) = mk_deadline_poll_fixture();
    let poll_count = AtomicUsize::new(0);
    let injected = || {
        // Serial schedule before this failure:
        // entry + 3 allocation boundaries + both 8,192-coordinate compose
        // rows + the between-pass boundary + certificate-row entry. The next
        // callback is the first bounded-cadence poll 4,096 coordinates into
        // the certified coeff-error/intercept-discharge pass.
        if poll_count.fetch_add(1, Ordering::SeqCst) >= 14 {
            Err(NyError::InternalError(
                "injected certificate-pass failure".into(),
            ))
        } else {
            Ok(())
        }
    };

    let result = crown_elementwise_backward_patches_with_poll_for_test(
        &bounds,
        &pre,
        patches_relaxation,
        false,
        &injected,
    );
    assert!(
        matches!(&result, Err(NyError::InternalError(_))),
        "injected certificate-pass error must reject the whole transaction"
    );
    assert!(
        poll_count.load(Ordering::SeqCst) >= 15,
        "the injected failure must occur inside the second full pass"
    );
    let published: Option<CrownBounds> = result.ok();
    assert!(
        published.is_none(),
        "completed compose scratch is not publishable without its certificate pass"
    );
}

#[test]
fn finite_deadline_rejects_malformed_explicit_row_geometry() {
    let (mut bounds, pre) = mk_7d_fixture(Some(vec![0.75, 0.0]), Some(vec![0.5, 1.0e-3]));
    bounds.upper_a.patches = Some(ArrayD::zeros(IxDyn(&[2, 2, 1, 2, 2, 1, 1])));

    let error = crown_elementwise_backward_patches_with_deadline(
        &bounds,
        &pre,
        std::time::Instant::now() + std::time::Duration::from_secs(30),
        patches_relaxation,
    )
    .expect_err("mismatched lower/upper row geometry must be rejected");
    assert!(
        matches!(error, NyError::ShapeMismatch { .. }),
        "malformed geometry must return ShapeMismatch, got {error:?}"
    );
}
