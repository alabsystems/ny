// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GroupNorm `IbpValidated` parity regressions.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::types::GroupNormLayer;
use crate::layers::normalization::{InstanceNorm1dLayer, LayerNormCrownMode};
use crate::{BatchedLinearBounds, LinearBounds};

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibpvalidated_scalar_matches_instance_norm_when_groups_eq_channels_3914(
) -> ny_core::Result<()> {
    let ny = Array1::from_vec(vec![1.5, -0.75]);
    let beta = Array1::from_vec(vec![0.1, -0.25]);
    let eps = 1e-5;
    let gn = GroupNormLayer::new(ny.clone(), beta.clone(), 2, eps)?
        .with_crown_mode(LayerNormCrownMode::IbpValidated);
    let inn =
        InstanceNorm1dLayer::new(ny, beta, eps)?.with_crown_mode(LayerNormCrownMode::IbpValidated);
    let bounds = LinearBounds::new(
        Array2::from_shape_vec(
            (2, 6),
            vec![
                1.0, -0.5, 0.25, 0.0, 0.75, -1.25, 0.2, 0.1, 0.3, -0.4, 0.5, 0.6,
            ],
        )
        .expect("valid scalar GroupNorm lower_a"),
        Array1::from_vec(vec![0.0, -0.1]),
        Array2::from_shape_vec(
            (2, 6),
            vec![
                1.0, -0.5, 0.25, 0.0, 0.75, -1.25, 0.2, 0.1, 0.3, -0.4, 0.5, 0.6,
            ],
        )
        .expect("valid scalar GroupNorm upper_a"),
        Array1::from_vec(vec![0.0, -0.1]),
    )?;
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.25, 0.5, -0.75, 0.0, 1.0])
            .expect("valid scalar GroupNorm lower"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.5, 1.5, 2.0, 0.25, 1.0, 2.5])
            .expect("valid scalar GroupNorm upper"),
    )?;

    let gn_actual = gn.propagate_linear_with_bounds(&bounds, &pre_act)?;
    let inn_actual = inn.propagate_linear_with_bounds(&bounds, &pre_act)?;

    assert_eq!(gn_actual.lower_a(), inn_actual.lower_a());
    assert_eq!(gn_actual.lower_b(), inn_actual.lower_b());
    assert_eq!(gn_actual.upper_a(), inn_actual.upper_a());
    assert_eq!(gn_actual.upper_b(), inn_actual.upper_b());
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibpvalidated_batched_matches_instance_norm_when_groups_eq_channels_3914(
) -> ny_core::Result<()> {
    let ny = Array1::from_vec(vec![0.75, -1.25]);
    let beta = Array1::from_vec(vec![0.0, 0.2]);
    let eps = 1e-5;
    let gn = GroupNormLayer::new(ny.clone(), beta.clone(), 2, eps)?
        .with_crown_mode(LayerNormCrownMode::IbpValidated);
    let inn =
        InstanceNorm1dLayer::new(ny, beta, eps)?.with_crown_mode(LayerNormCrownMode::IbpValidated);
    let bounds = BatchedLinearBounds::identity(&[2, 6])?;
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 6]),
            vec![
                -1.0, 0.0, 0.5, -0.25, 1.0, 2.0, -0.5, 0.25, 1.5, 0.0, 0.75, 2.25,
            ],
        )
        .expect("valid batched GroupNorm lower"),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 6]),
            vec![
                0.5, 1.0, 1.5, 0.75, 2.0, 3.0, 0.25, 1.25, 2.0, 1.0, 1.5, 3.5,
            ],
        )
        .expect("valid batched GroupNorm upper"),
    )?;

    let gn_actual = gn.propagate_linear_batched_with_bounds(&bounds, &pre_act)?;
    let inn_actual = inn.propagate_linear_batched_with_bounds(&bounds, &pre_act)?;

    assert_eq!(gn_actual.lower_a(), inn_actual.lower_a());
    assert_eq!(gn_actual.lower_b(), inn_actual.lower_b());
    assert_eq!(gn_actual.upper_a(), inn_actual.upper_a());
    assert_eq!(gn_actual.upper_b(), inn_actual.upper_b());
    Ok(())
}
