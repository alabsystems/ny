// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Public-API GELU proof coverage for #2519.
//!
//! Covers:
//! - `propagate_linear()` rejecting nonlinear GELU CROWN without pre-activation bounds
//! - large-input IBP using the parallel element-wise path

use ndarray::Array1;
use ny_core::NyError;
use ny_propagate::layers::{gelu_eval, BoundPropagation, GELULayer, GeluApproximation};
use ny_propagate::LinearBounds;
use ny_tensor::BoundedTensor;

const PARALLEL_IBP_LEN_NY: usize = 70_000;

fn make_parallel_point_input_2519() -> BoundedTensor {
    let values = Array1::from_shape_fn(PARALLEL_IBP_LEN_NY, |i| {
        let centered = (i % 257) as f32 - 128.0;
        centered / 16.0
    });
    BoundedTensor::new(values.clone().into_dyn(), values.into_dyn())
        .expect("point intervals should build a valid bounded tensor")
}

#[ntest::timeout(10000)]
#[test]
fn test_gelu_propagate_linear_requires_pre_activation_bounds_2519() {
    let bounds = LinearBounds::identity(1);

    for approximation in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let gelu = GELULayer::new(approximation);
        let err = gelu
            .propagate_linear(&bounds)
            .expect_err("GELU propagate_linear should reject missing pre-activation bounds");

        assert!(
            matches!(&err, NyError::UnsupportedOp(msg) if msg.contains("GELU is nonlinear")),
            "{approximation:?} should report nonlinear GELU unsupported-op, got {err:?}"
        );
    }
}

#[ntest::timeout(20000)]
#[test]
fn test_gelu_ibp_parallel_large_point_intervals_match_point_eval_2519() {
    let input = make_parallel_point_input_2519();
    assert!(
        input.len() > 65_536,
        "test fixture must exceed the parallel threshold"
    );

    for approximation in [GeluApproximation::Erf, GeluApproximation::Tanh] {
        let gelu = GELULayer::new(approximation);
        let output = gelu
            .propagate_ibp(&input)
            .expect("GELU IBP should succeed on large finite point intervals");

        for (idx, ((&x, &lower), &upper)) in input
            .lower()
            .iter()
            .zip(output.lower().iter())
            .zip(output.upper().iter())
            .enumerate()
        {
            let expected = gelu_eval(x, approximation);
            assert!(
                (lower - expected).abs() < 1e-6,
                "{approximation:?} lower[{idx}]={lower} != point eval {expected} for x={x}"
            );
            assert!(
                (upper - expected).abs() < 1e-6,
                "{approximation:?} upper[{idx}]={upper} != point eval {expected} for x={x}"
            );
        }
    }
}
