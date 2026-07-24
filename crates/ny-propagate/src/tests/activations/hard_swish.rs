// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! HardSwish NaN/Inf guard tests.
//!
//! Updated for #2977: CROWN backward domain_guard rejects non-finite pre-activation
//! with NumericalInstability. Previously these tested maximally-loose fallback bounds;
//! now they verify the early rejection is correct.

use crate::*;
use ndarray::{ArrayD, IxDyn};

#[ntest::timeout(10000)]
#[test]
fn test_hardswish_crown_nan_inf_guard_returns_maximally_loose_bounds() {
    // Updated for #2977: domain_guard now rejects non-finite pre-activation.
    let layer = HardSwishLayer::new();
    let linear_bounds = LinearBounds::identity(1);

    for (l, u, desc) in [
        (f32::NEG_INFINITY, 1.0f32, "neg-inf lower"),
        (-1.0f32, f32::NAN, "NaN upper"),
    ] {
        let pre = BoundedTensor::new_unchecked(
            ArrayD::from_elem(IxDyn(&[1]), l),
            ArrayD::from_elem(IxDyn(&[1]), u),
        )
        .unwrap();

        let result = layer.propagate_linear_with_bounds(&linear_bounds, &pre);
        assert!(
            matches!(result, Err(NyError::NumericalInstability(_))),
            "HardSwish ({desc}): non-finite pre-activation should trigger domain_guard: got {:?}",
            result
        );
    }
}

/// Regression test for #1736/#2977: multi-neuron with Inf pre-activation now rejected.
#[ntest::timeout(10000)]
#[test]
fn test_hardswish_crown_nan_contamination_multi_neuron_1736() {
    let layer = HardSwishLayer::new();

    // 2 neurons: neuron 0 has Inf (triggers domain_guard), neuron 1 is normal
    let pre = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
    )
    .unwrap();

    let identity = LinearBounds::identity(2);
    let result = layer.propagate_linear_with_bounds(&identity, &pre);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "HardSwish multi-neuron with Inf: domain_guard should reject: got {:?}",
        result
    );
}

/// Test with non-identity incoming bounds and Inf pre-activation — now rejected.
#[ntest::timeout(10000)]
#[test]
fn test_hardswish_crown_nan_contamination_nonidentity_1736() {
    use ndarray::{Array1, Array2};
    let layer = HardSwishLayer::new();

    // 2 neurons: neuron 0 has Inf, neuron 1 is normal
    let pre = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
    )
    .unwrap();

    let incoming = LinearBounds::new(
        Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap(),
        Array1::zeros(1),
        Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap(),
        Array1::zeros(1),
    )
    .unwrap();

    let result = layer.propagate_linear_with_bounds(&incoming, &pre);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "HardSwish nonidentity with Inf: domain_guard should reject: got {:?}",
        result
    );
}

// =============================================================================
// Proptest: multi-neuron NaN guard (#1736/#2977)
// =============================================================================

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Proptest for #1736/#2977: verify CROWN rejects non-finite pre-activation.
    ///
    /// Setup: 3 neurons, one with ±inf pre-activation (triggers domain_guard),
    /// the other two with random finite bounds.
    /// Updated for #2977: domain_guard now returns NumericalInstability.
    #[ntest::timeout(10000)]
    #[test]
    fn test_hardswish_nan_guard_proptest_1736(
        l1 in -3.0f32..3.0,
        d1 in 0.0f32..6.0,
        l2 in -3.0f32..3.0,
        d2 in 0.0f32..6.0,
    ) {
        let u1 = l1 + d1;
        let u2 = l2 + d2;

        // Neuron 0: ±inf (triggers domain_guard)
        // Neurons 1, 2: normal finite bounds
        let pre = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, l1, l2]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::INFINITY, u1, u2]).unwrap(),
        ).unwrap();

        let layer = HardSwishLayer::new();
        let identity = LinearBounds::identity(3);
        let result = layer.propagate_linear_with_bounds(&identity, &pre);
        prop_assert!(
            matches!(result, Err(NyError::NumericalInstability(_))),
            "HardSwish proptest with Inf neuron: domain_guard should reject: got {:?}",
            result
        );
    }
}
