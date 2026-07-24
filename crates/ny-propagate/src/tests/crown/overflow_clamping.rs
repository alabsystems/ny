// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for CROWN coefficient overflow clamping (#1932).
//!
//! CROWN backward propagation computes `A_new = A @ W` at each layer.
//! For deep networks, A-matrix coefficients grow exponentially and can
//! overflow f32. Issue #1932 adds proactive magnitude clamping: any
//! coefficient exceeding `CROWN_COEFF_MAX` (1e10) triggers row-level
//! degradation to `[-inf, +inf]`, which is sound but maximally loose.
//! The `has_degraded_bounds` post-check then falls back to IBP.

use super::*;
use ndarray::{arr1, Array2};

/// Build a deep Linear→ReLU network where CROWN coefficients grow exponentially.
///
/// Each layer has `dim x dim` weights with spectral norm ~2.3, so after `depth`
/// linear layers the A-matrix coefficients are O(2.3^depth). At depth=40 this
/// exceeds CROWN_COEFF_MAX (1e10) by ~layer 30, exercising the #1932 clamping.
fn build_deep_overflow_network(dim: usize, depth: usize) -> (Network, BoundedTensor) {
    let mut network = Network::new();
    // Weight matrix: 2.0 on diagonal, 0.1 off-diagonal → spectral norm ~2.3.
    // This ensures CROWN A-matrix coefficients grow exponentially (not shrink).
    let w = Array2::from_shape_fn((dim, dim), |(i, j)| if i == j { 2.0 } else { 0.1 });
    let bias = arr1(&vec![0.01_f32; dim]);

    for _ in 0..depth {
        network.add_layer(Layer::Linear(
            LinearLayer::new(w.clone(), Some(bias.clone())).unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
    }

    let lower = arr1(&vec![-1.0_f32; dim]);
    let upper = arr1(&vec![1.0_f32; dim]);
    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).unwrap();
    (network, input)
}

/// Regression test (#1932): deep network CROWN must not produce NaN.
///
/// A 40-layer network with spectral norm ~2.3 per layer causes A-matrix
/// coefficients to exceed CROWN_COEFF_MAX (1e10) by approximately layer 30.
/// Before #1932, this could produce NaN in concretized bounds.
/// After #1932, affected rows degrade to `[-inf, +inf]` (sound).
#[ntest::timeout(10000)]
#[test]
fn test_crown_proactive_coeff_clamping_deep_network_1932() {
    let dim = 4;
    let depth = 40;
    let (network, input) = build_deep_overflow_network(dim, depth);

    // IBP baseline: must produce finite bounds for this network.
    let ibp = network.propagate_ibp(&input).unwrap();
    for i in 0..dim {
        assert!(ibp.lower()[[i]].is_finite(), "IBP lower[{i}] not finite");
        assert!(ibp.upper()[[i]].is_finite(), "IBP upper[{i}] not finite");
    }

    // CROWN: must not produce NaN. May produce Inf from row degradation.
    let crown = network.propagate_crown(&input).unwrap();
    for i in 0..dim {
        assert!(
            !crown.lower()[[i]].is_nan(),
            "CROWN lower[{i}] is NaN — #1932 clamping failed"
        );
        assert!(
            !crown.upper()[[i]].is_nan(),
            "CROWN upper[{i}] is NaN — #1932 clamping failed"
        );
    }

    // If CROWN produced finite bounds, verify soundness at all 2^dim corners.
    let all_finite =
        (0..dim).all(|i| crown.lower()[[i]].is_finite() && crown.upper()[[i]].is_finite());
    if all_finite {
        for corner in 0..(1u32 << dim) {
            let point: Vec<f32> = (0..dim)
                .map(|d| if corner & (1 << d) != 0 { 1.0 } else { -1.0 })
                .collect();
            let pt = BoundedTensor::concrete(arr1(&point).into_dyn()).unwrap();
            let out = network.propagate_ibp(&pt).unwrap();
            for i in 0..dim {
                let y = out.lower()[[i]];
                assert!(
                    y >= crown.lower()[[i]] - 1e-5,
                    "Soundness: corner {corner} output[{i}]={y} < CROWN lower {}",
                    crown.lower()[[i]]
                );
                assert!(
                    y <= crown.upper()[[i]] + 1e-5,
                    "Soundness: corner {corner} output[{i}]={y} > CROWN upper {}",
                    crown.upper()[[i]]
                );
            }
        }
    }
}
