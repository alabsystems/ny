// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::ArrayD;
use num_bigint::BigInt;
use ny_core::require_f64_interval_proof_environment;

use super::{
    box_reconstruction::reconstruct_input_box, relaxed_clip, relaxed_clip_internal_fast,
    relaxed_clip_internal_scalar, relaxed_clip_single_spec_row_fast, RelaxedClipOptions,
    SingleSpecRowClipScratch,
};

/// Exact integer encoding of a finite binary32 value in units of `2^-149`.
fn exact_f32_units(value: f32) -> BigInt {
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 0xff;
    assert_ne!(exponent, 0xff, "finite binary32 required");
    let fraction = bits & 0x7f_ffff;
    let magnitude = if exponent == 0 {
        BigInt::from(fraction)
    } else {
        BigInt::from((1_u32 << 23) | fraction) << (exponent - 1) as usize
    };
    if bits >> 31 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn assert_exact_endpoint_containment(x_l: f32, x_u: f32) {
    require_f64_interval_proof_environment().expect("normal proof environment");
    let (centre, radius) = reconstruct_input_box(x_l, x_u);
    assert!(centre.is_finite());
    assert!(!radius.is_nan() && radius >= 0.0);
    if radius.is_infinite() {
        return;
    }
    let centre = exact_f32_units(centre);
    let radius = exact_f32_units(radius);
    let lower = exact_f32_units(x_l);
    let upper = exact_f32_units(x_u);
    assert!(
        &centre - &radius <= lower,
        "lower endpoint escaped [{x_l:e}, {x_u:e}]"
    );
    assert!(
        &centre + &radius >= upper,
        "upper endpoint escaped [{x_l:e}, {x_u:e}]"
    );
}

#[test]
fn symmetric_box_preserves_center_and_contains_endpoints() {
    require_f64_interval_proof_environment().expect("normal proof environment");
    let (centre, radius) = reconstruct_input_box(-2.0, 2.0);
    assert_eq!(centre, 0.0);
    assert!(
        radius >= 2.0,
        "certified radius shrank below the exact radius"
    );
    assert_exact_endpoint_containment(-2.0, 2.0);
}

#[test]
fn adjacent_float_tie_box_contains_both_endpoints_exactly() {
    require_f64_interval_proof_environment().expect("normal proof environment");
    let x_l = 1.0_f32;
    let x_u = x_l.next_up();
    let unit = 2.0_f32.powi(-23);
    let (centre, radius) = reconstruct_input_box(x_l, x_u);
    assert_eq!(centre, 1.0, "historical tie-to-even anchor moved");
    assert!(
        radius >= unit,
        "radius must reach the adjacent upper endpoint"
    );
    assert_exact_endpoint_containment(x_l, x_u);
}

#[test]
fn endpoint_certificate_covers_extremes_and_subnormals() {
    let seeds = [
        f32::MIN,
        -1.0,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        1.0,
        1.0_f32.next_up(),
        f32::MAX,
    ];
    for &x_l in &seeds {
        for &x_u in &seeds {
            if x_l <= x_u {
                assert_exact_endpoint_containment(x_l, x_u);
            }
        }
    }
}

#[test]
fn tie_rounded_box_keeps_a_strictly_feasible_witness_on_every_path() {
    let unit = 2.0_f32.powi(-23);
    let lo = [0.0_f32, 1.0, 1.0];
    let hi = [unit, 1.0 + unit, 1.0];
    let coefficients = [1.0_f32, -1.0, 1.0];
    let witness = [0.75 * unit, 1.0 + unit, 1.0];
    assert_eq!(1.0 + (1.0 + unit), 2.0, "fixture lost its midpoint tie");
    let objective = coefficients
        .iter()
        .zip(witness)
        .map(|(&a, x)| f64::from(a) * f64::from(x))
        .sum::<f64>();
    assert!(
        objective < 0.0,
        "witness must satisfy the affine constraint"
    );

    let x_l = ArrayD::from_shape_vec(vec![1, 3], lo.to_vec()).unwrap();
    let x_u = ArrayD::from_shape_vec(vec![1, 3], hi.to_vec()).unwrap();
    let l_a = ArrayD::from_shape_vec(vec![1, 1, 3], coefficients.to_vec()).unwrap();
    let bias = ArrayD::from_shape_vec(vec![1, 1], vec![0.0]).unwrap();
    let thresholds = ArrayD::from_shape_vec(vec![1, 1], vec![0.0]).unwrap();
    let assert_retained = |lane: &str, clipped_l: &[f32], clipped_u: &[f32]| {
        for dim in 0..3 {
            assert!(
                clipped_l[dim] <= witness[dim] && witness[dim] <= clipped_u[dim],
                "{lane} removed feasible witness dimension {dim}"
            );
        }
    };
    let options = || RelaxedClipOptions {
        num_iterations: 1,
        is_lower: true,
        preserve_infeasible: false,
    };

    let (lower, upper, verified) =
        relaxed_clip_internal_scalar(&x_l, &x_u, &l_a, &bias, &thresholds, options()).unwrap();
    assert_eq!(verified, vec![false]);
    assert_retained(
        "scalar",
        lower.as_slice().unwrap(),
        upper.as_slice().unwrap(),
    );

    let (lower, upper, verified) =
        relaxed_clip_internal_fast(&x_l, &x_u, &l_a, &bias, &thresholds, options()).unwrap();
    assert_eq!(verified, vec![false]);
    assert_retained("fast", lower.as_slice().unwrap(), upper.as_slice().unwrap());

    let (lower, upper) = relaxed_clip(&x_l, &x_u, &l_a, &bias, &thresholds, 1, true).unwrap();
    assert_retained(
        "public",
        lower.as_slice().unwrap(),
        upper.as_slice().unwrap(),
    );

    let mut row_l = lo.to_vec();
    let mut row_u = hi.to_vec();
    let mut scratch = SingleSpecRowClipScratch::new();
    relaxed_clip_single_spec_row_fast(
        &mut row_l,
        &mut row_u,
        &coefficients,
        &[0.0],
        &[0.0],
        1,
        3,
        1,
        true,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(scratch.row_verified, vec![false]);
    assert_retained("specialized row", &row_l, &row_u);
}
