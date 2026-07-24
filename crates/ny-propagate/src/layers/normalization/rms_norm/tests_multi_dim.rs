// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-dimensional scalar RMSNorm CROWN regressions.
//!
//! Part of #4148.

use ndarray::{arr1, arr2, Array1, ArrayD, IxDyn};
use ny_core::Result;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::types::RmsNormLayer;
use crate::layers::normalization::decomposed::{
    batched_bounds_to_scalar_multi_dim, scalar_bounds_to_batched_multi_dim,
};
use crate::layers::normalization::LayerNormCrownMode;
use crate::LinearBounds;

/// Trust-verifier regression: RMSNorm IBP used a fixed `[usize; 8]` per-slice
/// index buffer, so a tensor of rank > 8 indexed it out of bounds and PANICKED.
/// The verifier flagged this (index_out_of_bounds in the RMSNorm IBP path); the
/// fix fails closed with a structured error. Assert no panic on a rank-9 input.
#[test]
fn rmsnorm_ibp_rank_over_8_fails_closed_not_panics() {
    use crate::layers::common::BoundPropagation;
    let layer = RmsNormLayer::new(arr1(&[1.0, 1.0]), 1e-5).unwrap();
    // Rank 9: eight leading unit dims + a size-2 normalized last dim.
    let shape = vec![1usize, 1, 1, 1, 1, 1, 1, 1, 2];
    let n: usize = shape.iter().product();
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&shape), vec![0.0f32; n]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&shape), vec![1.0f32; n]).unwrap(),
    )
    .unwrap();
    let result = layer.propagate_ibp(&pre_act);
    assert!(
        result.is_err(),
        "RMSNorm IBP must fail closed (not panic) on rank-9 input"
    );
    // Rank 8 (the supported max) must still work.
    let shape8 = vec![1usize, 1, 1, 1, 1, 1, 1, 2];
    let n8: usize = shape8.iter().product();
    let ok_input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&shape8), vec![0.0f32; n8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&shape8), vec![1.0f32; n8]).unwrap(),
    )
    .unwrap();
    assert!(
        layer.propagate_ibp(&ok_input).is_ok(),
        "RMSNorm IBP must still accept rank-8 input"
    );
}

fn rmsnorm_fixture_4148() -> Result<(RmsNormLayer, LinearBounds, BoundedTensor, usize, usize)> {
    let layer = RmsNormLayer::new(arr1(&[1.5, -0.75, 0.25]), 1e-5)?
        .with_crown_mode(LayerNormCrownMode::IbpValidated);
    let bounds = LinearBounds::new(
        arr2(&[
            [1.0, -0.5, 0.25, 0.75, -0.25, 0.5],
            [-0.2, 0.3, 1.1, -0.4, 0.8, -0.7],
        ]),
        arr1(&[0.25, -0.1]),
        arr2(&[
            [1.1, -0.25, 0.4, 0.85, -0.15, 0.55],
            [0.05, 0.45, 1.35, -0.2, 1.0, -0.35],
        ]),
        arr1(&[0.4, 0.15]),
    )?;
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, -0.25, 0.5, -0.5, 0.75, 1.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.25, 0.75, 1.2, 0.4, 1.4, 2.0]).unwrap(),
    )?;
    Ok((layer, bounds, pre_act, 2, 3))
}

fn rmsnorm_multi_dim_case() -> impl Strategy<Value = (usize, usize, f32, Vec<f32>, Vec<f32>)> {
    (2usize..5, 2usize..5, 0.05f32..0.2).prop_flat_map(|(batch_size, norm_size, hw)| {
        (
            Just(batch_size),
            Just(norm_size),
            Just(hw),
            proptest::collection::vec(-1.5f32..1.5, batch_size * norm_size),
            proptest::collection::vec(-2.0f32..2.0, norm_size),
        )
    })
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_multi_dim_rmsnorm_matches_batched_4148() -> Result<()> {
    let (layer, bounds, pre_act, batch_size, norm_size) = rmsnorm_fixture_4148()?;

    let actual = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;
    let reshaped = pre_act.reshape(&[batch_size, norm_size])?;
    let batched_bounds = scalar_bounds_to_batched_multi_dim(&bounds, batch_size, norm_size)?;
    let expected = layer.propagate_linear_batched_with_bounds(&batched_bounds, &reshaped)?;
    let expected_scalar =
        batched_bounds_to_scalar_multi_dim(&expected, bounds.lower_b(), bounds.upper_b())?;

    assert_eq!(actual.lower_a(), expected_scalar.lower_a());
    assert_eq!(actual.lower_b(), expected_scalar.lower_b());
    assert_eq!(actual.upper_a(), expected_scalar.upper_a());
    assert_eq!(actual.upper_b(), expected_scalar.upper_b());
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(64) })]

    #[ntest::timeout(60000)]
    #[test]
    fn proptest_crown_scalar_multi_dim_rmsnorm_contains_sampled_outputs_4148(
        (batch_size, norm_size, hw, centers, ny) in rmsnorm_multi_dim_case(),
    ) {
        let total = batch_size * norm_size;
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[batch_size, norm_size]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch_size, norm_size]), upper_v.clone()).unwrap(),
        ).unwrap();
        let layer = RmsNormLayer::new(Array1::from_vec(ny), 1e-5)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);
        let result = layer
            .propagate_linear_with_bounds(&LinearBounds::identity(total), &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "multi-dim scalar RMSNorm CROWN failed: {e}"
                ))
            })?;
        let concrete = result.concretize_sound(&input);

        for s in 0..12_u32 {
            let sample: Vec<f32> = (0..total)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2246822519)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            for b in 0..batch_size {
                let start = b * norm_size;
                let end = start + norm_size;
                let x = Array1::from_vec(sample[start..end].to_vec());
                let y = layer.eval(&x).expect("RMSNorm eval should succeed");
                for i in 0..norm_size {
                    let idx = start + i;
                    prop_assert!(
                        concrete.lower()[[idx]] <= y[i] + 1e-4,
                        "multi-dim RMSNorm lower[{idx}] {} > {}",
                        concrete.lower()[[idx]],
                        y[i]
                    );
                    prop_assert!(
                        concrete.upper()[[idx]] >= y[i] - 1e-4,
                        "multi-dim RMSNorm upper[{idx}] {} < {}",
                        concrete.upper()[[idx]],
                        y[i]
                    );
                }
            }
        }
    }
}
