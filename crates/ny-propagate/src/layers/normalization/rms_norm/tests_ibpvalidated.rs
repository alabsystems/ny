// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IbpValidated and decomposed-helper regressions for RMSNorm CROWN.
//!
//! Keeps the shared-helper parity and fused-IBP envelope assertions separate
//! from the general CROWN regression file so the RMSNorm test modules stay
//! within the 500-line file-size limit.
//!
//! Part of #3821, #3875.

use ndarray::{arr1, arr2, ArrayD, Ix1, Ix2, IxDyn};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::types::RmsNormLayer;
use crate::layers::common::BoundPropagation;
use crate::layers::normalization::decomposed::decomposed_rms_norm_crown_backward;
use crate::{BatchedLinearBounds, LinearBounds};

fn scalar_bounds_to_batched_for_test(bounds: &LinearBounds) -> BatchedLinearBounds {
    BatchedLinearBounds::new(
        bounds.lower_a().clone().into_dyn(),
        bounds.lower_b().clone().into_dyn(),
        bounds.upper_a().clone().into_dyn(),
        bounds.upper_b().clone().into_dyn(),
        vec![bounds.num_inputs()],
        vec![bounds.num_outputs()],
    )
    .expect("scalar bounds should reshape into BatchedLinearBounds")
}

fn batched_bounds_to_scalar_for_test(bounds: &BatchedLinearBounds) -> LinearBounds {
    LinearBounds::new(
        bounds
            .lower_a()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("expected 2D lower_a"),
        bounds
            .lower_b()
            .clone()
            .into_dimensionality::<Ix1>()
            .expect("expected 1D lower_b"),
        bounds
            .upper_a()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("expected 2D upper_a"),
        bounds
            .upper_b()
            .clone()
            .into_dimensionality::<Ix1>()
            .expect("expected 1D upper_b"),
    )
    .expect("converted scalar bounds should be valid")
}

fn assert_interval_within_fused_ibp(actual: &BoundedTensor, fused_ibp: &BoundedTensor) {
    for (idx, (actual_lower, fused_lower)) in actual
        .lower()
        .iter()
        .zip(fused_ibp.lower().iter())
        .enumerate()
    {
        assert!(
            *actual_lower >= *fused_lower - 1e-5,
            "lower[{idx}] escaped fused RmsNorm IBP envelope: actual={} fused={}",
            actual_lower,
            fused_lower,
        );
    }
    for (idx, (actual_upper, fused_upper)) in actual
        .upper()
        .iter()
        .zip(fused_ibp.upper().iter())
        .enumerate()
    {
        assert!(
            *actual_upper <= *fused_upper + 1e-5,
            "upper[{idx}] escaped fused RmsNorm IBP envelope: actual={} fused={}",
            actual_upper,
            fused_upper,
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_rmsnorm_ibpvalidated_scalar_matches_decomposed_helper_3821() -> Result<()> {
    let layer = RmsNormLayer::new(arr1(&[1.5, -0.75, 0.25]), 1e-5)?;
    let bounds = LinearBounds::new(
        arr2(&[[1.0, -0.5, 0.25], [0.0, 0.75, -1.25], [0.2, 0.1, 0.3]]),
        arr1(&[0.0, 0.25, -0.1]),
        arr2(&[[1.0, -0.5, 0.25], [0.0, 0.75, -1.25], [0.2, 0.1, 0.3]]),
        arr1(&[0.0, 0.25, -0.1]),
    )?;
    let pre_act = BoundedTensor::new(
        arr1(&[-1.0, 0.25, 0.5]).into_dyn(),
        arr1(&[0.5, 1.5, 2.0]).into_dyn(),
    )?;

    let actual = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;
    let expected = decomposed_rms_norm_crown_backward(
        &scalar_bounds_to_batched_for_test(&bounds),
        &layer.ny,
        layer.eps,
        &pre_act,
    )?;
    let expected_scalar = batched_bounds_to_scalar_for_test(&expected.bounds);

    assert_eq!(actual.lower_a(), expected_scalar.lower_a());
    assert_eq!(actual.lower_b(), expected_scalar.lower_b());
    assert_eq!(actual.upper_a(), expected_scalar.upper_a());
    assert_eq!(actual.upper_b(), expected_scalar.upper_b());

    let concretized = actual.concretize_sound(&pre_act);
    let fused_ibp = layer.propagate_ibp(&pre_act)?;
    let fused_envelope = bounds.concretize_sound(&fused_ibp);
    assert_interval_within_fused_ibp(&concretized, &fused_envelope);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_rmsnorm_ibpvalidated_batched_matches_decomposed_helper_3821() -> Result<()> {
    let layer = RmsNormLayer::new(arr1(&[0.75, -1.25, 1.5]), 1e-5)?;
    let bounds = BatchedLinearBounds::identity(&[2, 3])?;
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 0.5, -0.25, 1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.5, 1.0, 1.5, 0.75, 2.0, 3.0]).unwrap(),
    )?;

    let actual = layer.propagate_linear_batched_with_bounds(&bounds, &pre_act)?;
    let expected = decomposed_rms_norm_crown_backward(&bounds, &layer.ny, layer.eps, &pre_act)?;

    assert_eq!(actual.lower_a, expected.bounds.lower_a);
    assert_eq!(actual.lower_b, expected.bounds.lower_b);
    assert_eq!(actual.upper_a, expected.bounds.upper_a);
    assert_eq!(actual.upper_b, expected.bounds.upper_b);

    let concretized = actual.concretize_sound(&pre_act)?;
    let fused_ibp = layer.propagate_ibp(&pre_act)?;
    let fused_envelope = bounds.concretize_sound(&fused_ibp)?;
    assert_interval_within_fused_ibp(&concretized, &fused_envelope);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(128) })]

    #[ntest::timeout(60000)]
    #[test]
    fn proptest_decomposed_rmsnorm_crown_contains_forward_output(
        c0 in -1.5f32..1.5,
        c1 in -1.5f32..1.5,
        c2 in -1.5f32..1.5,
        hw in 0.05f32..0.25,
        g0 in -2.0f32..2.0,
        g1 in -2.0f32..2.0,
        g2 in -2.0f32..2.0,
    ) {
        let centers = [c0, c1, c2];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let pre_act = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), upper_v.clone()).unwrap(),
        ).unwrap();
        let flat_input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), upper_v.clone()).unwrap(),
        ).unwrap();
        let layer = RmsNormLayer::new(ndarray::Array1::from_vec(vec![g0, g1, g2]), 1e-5).unwrap();

        let helper = decomposed_rms_norm_crown_backward(
            &scalar_bounds_to_batched_for_test(&LinearBounds::identity(3)),
            &layer.ny,
            layer.eps,
            &pre_act,
        )
        .map_err(|e| {
            TestCaseError::fail(format!(
                "decomposed RMSNorm helper failed: {e}"
            ))
        })?;
        let concrete = helper
            .bounds
            .concretize_sound(&flat_input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "decomposed RMSNorm concretize failed: {e}"
                ))
            })?;

        for s in 0..16_u32 {
            let sample: Vec<f32> = (0..3)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();
            let y = layer.eval(&arr1(&sample)).expect("eval should succeed");

            for i in 0..3 {
                prop_assert!(
                    concrete.lower()[[i]] <= y[i] + 1e-4,
                    "decomposed RMSNorm lower violation at dim {i}: {} > {}",
                    concrete.lower()[[i]],
                    y[i]
                );
                prop_assert!(
                    concrete.upper()[[i]] >= y[i] - 1e-4,
                    "decomposed RMSNorm upper violation at dim {i}: {} < {}",
                    concrete.upper()[[i]],
                    y[i]
                );
            }
        }
    }
}
