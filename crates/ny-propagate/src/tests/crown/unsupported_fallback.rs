// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for UnsupportedOp fallback paths in CROWN propagation.
//!
//! Two fallback mechanisms exist:
//!
//! 1. **Per-layer IBP concretization** (catch-all in `crown_backward_step`):
//!    When the unary catch-all encounters `UnsupportedOp`, it concretizes the
//!    accumulated CROWN bounds through the failing layer's IBP and continues
//!    backward. This preserves CROWN tightness from layers above. (#3437)
//!
//! 2. **Whole-network IBP fallback** (layer-specific arms like Gather):
//!    When a layer-specific arm returns `IbpFallback`, the CROWN engine falls
//!    back to full-network IBP (for `propagate_crown` and `propagate_crown_fast`)
//!    or to `ibp_fallback_with_constant_linear` (for `propagate_crown_with_linear`).
//!
//! These tests verify each fallback path produces sound bounds.

use super::*;
use crate::layers::GatherLayer;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

/// Helper: build a Linear(3→3) + Gather([0,2]) network where Gather triggers
/// UnsupportedOp in CROWN backward. Returns (network, input).
fn network_with_unsupported_gather() -> (Network, BoundedTensor) {
    let weight = arr2(&[[1.0f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    let bias = arr1(&[1.0f32, 2.0, 3.0]);
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 2]).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::Gather(GatherLayer::new(0, Some(indices), vec![])));

    let input = BoundedTensor::new(
        arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
        arr1(&[4.0f32, 5.0, 6.0]).into_dyn(),
    )
    .unwrap();

    (network, input)
}

/// Assert that bounds contain the expected corner outputs and match IBP.
fn assert_fallback_sound(result: &BoundedTensor, ibp: &BoundedTensor) {
    assert_eq!(result.shape(), ibp.shape());
    assert_eq!(result.lower(), ibp.lower());
    assert_eq!(result.upper(), ibp.upper());

    // Soundness: output[0] = x[0]+1, output[1] = 3*x[2]+3.
    // At x=[1,2,3]: [2, 12].  At x=[4,5,6]: [5, 21].
    let flat = result.flatten();
    assert!(flat.lower()[[0]] <= 2.0, "lower[0]={}", flat.lower()[[0]]);
    assert!(flat.upper()[[0]] >= 5.0, "upper[0]={}", flat.upper()[[0]]);
    assert!(flat.lower()[[1]] <= 12.0, "lower[1]={}", flat.lower()[[1]]);
    assert!(flat.upper()[[1]] >= 21.0, "upper[1]={}", flat.upper()[[1]]);
}

/// Regression test for `propagate_crown` UnsupportedOp fallback (crown.rs:232).
///
/// When the backward loop encounters a layer returning `UnsupportedOp`,
/// `propagate_crown` falls back to full-network IBP.
#[ntest::timeout(10000)]
#[test]
fn propagate_crown_unsupported_op_falls_back_to_ibp() {
    let (network, input) = network_with_unsupported_gather();
    let crown = network.propagate_crown(&input).unwrap();
    let ibp = network.propagate_ibp(&input).unwrap();
    assert_fallback_sound(&crown, &ibp);
}

/// Regression test for `propagate_crown_fast` UnsupportedOp fallback (fast.rs:196).
///
/// When the backward loop in `propagate_crown_fast` encounters `UnsupportedOp`,
/// it falls back to full-network IBP. This tests the catch-all `_` arm.
#[ntest::timeout(10000)]
#[test]
fn propagate_crown_fast_unsupported_op_falls_back_to_ibp() {
    let (network, input) = network_with_unsupported_gather();
    let crown_fast = network.propagate_crown_fast(&input).unwrap();
    let ibp = network.propagate_ibp(&input).unwrap();
    assert_fallback_sound(&crown_fast, &ibp);
}

/// Helper: build a network where ReduceMax(fixed_max_index=false) triggers
/// UnsupportedOp in the unary catch-all of crown_backward_step.
///
/// Network: Linear(3→3, diag(1,2,3)) → ReduceMax(axis=-1, keepdims=true, fixed=false)
///
/// ReduceMax with fixed_max_index=false goes through the catch-all arm and
/// returns UnsupportedOp from propagate_linear_with_bounds. This tests the
/// per-layer IBP concretization path (#3437).
fn network_with_unsupported_reduce_max() -> (Network, BoundedTensor) {
    use crate::layers::ReduceMaxLayer;

    let weight = arr2(&[[1.0f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    let bias = arr1(&[0.0f32, 0.0, 0.0]);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::ReduceMax(ReduceMaxLayer {
        axes: vec![-1],
        keepdims: true,
        fixed_max_index: false,
    }));

    let input = BoundedTensor::new(
        arr1(&[0.0f32, 0.0, 0.0]).into_dyn(),
        arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
    )
    .unwrap();

    (network, input)
}

/// Per-layer IBP concretization: ReduceMax(fixed=false) in catch-all produces
/// sound bounds via per-layer concretization instead of whole-network IBP fallback.
///
/// Part of #3437: crown_backward_step now concretizes accumulated CROWN bounds
/// through the failing layer's IBP, preserving CROWN tightness from layers
/// between the output and the failing layer.
#[ntest::timeout(10000)]
#[test]
fn propagate_crown_per_layer_concretization_reduce_max() {
    let (network, input) = network_with_unsupported_reduce_max();
    let crown = network.propagate_crown(&input).unwrap();
    let ibp = network.propagate_ibp(&input).unwrap();

    // Soundness: CROWN bounds must contain all true outputs.
    // Network: max(x1, 2*x2, 3*x3) for x ∈ [0,1]×[0,2]×[0,3].
    // At corners: max(0,0,0)=0, max(1,0,0)=1, max(0,4,0)=4, max(0,0,9)=9,
    //   max(1,4,9)=9, max(1,4,0)=4, max(1,0,9)=9, max(0,4,9)=9.
    // True range: [0, 9].
    let crown_flat = crown.flatten();
    assert!(
        crown_flat.lower()[[0]] <= 0.0,
        "CROWN lower should be <= 0: got {}",
        crown_flat.lower()[[0]]
    );
    assert!(
        crown_flat.upper()[[0]] >= 9.0,
        "CROWN upper should be >= 9: got {}",
        crown_flat.upper()[[0]]
    );

    // CROWN should be at least as tight as IBP (no worse).
    let ibp_flat = ibp.flatten();
    assert!(
        crown_flat.lower()[[0]] >= ibp_flat.lower()[[0]] - 1e-6,
        "CROWN lower ({}) should be >= IBP lower ({})",
        crown_flat.lower()[[0]],
        ibp_flat.lower()[[0]]
    );
    assert!(
        crown_flat.upper()[[0]] <= ibp_flat.upper()[[0]] + 1e-6,
        "CROWN upper ({}) should be <= IBP upper ({})",
        crown_flat.upper()[[0]],
        ibp_flat.upper()[[0]]
    );
}

/// Per-layer concretization with layers after the unsupported layer.
///
/// Network: Linear(3→3) → ReduceMax(fixed=false) → Sigmoid → MulConstant(2)
/// The Sigmoid+MulConstant layers have valid CROWN backward. ReduceMax triggers
/// per-layer concretization. The accumulated CROWN from Sigmoid+MulConstant is
/// concretized through ReduceMax's IBP.
///
/// Soundness verified by grid sampling over input corners.
#[ntest::timeout(10000)]
#[test]
fn propagate_crown_per_layer_concretization_with_layers_after() {
    use crate::layers::arithmetic::MulConstantLayer;
    use crate::layers::{ReduceMaxLayer, SigmoidLayer};

    let weight = arr2(&[[1.0f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    let bias = arr1(&[0.0f32, -1.0, -2.0]);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::ReduceMax(ReduceMaxLayer {
        axes: vec![-1],
        keepdims: true,
        fixed_max_index: false,
    }));
    network.add_layer(Layer::Sigmoid(SigmoidLayer));
    network.add_layer(Layer::MulConstant(MulConstantLayer::new(
        ArrayD::from_elem(IxDyn(&[1]), 2.0f32),
    )));

    let lower = [0.0f32, 0.0, 0.0];
    let upper = [1.0f32, 1.0, 1.0];
    let input = BoundedTensor::new(arr1(&lower).into_dyn(), arr1(&upper).into_dyn()).unwrap();

    let crown = network.propagate_crown(&input).unwrap();
    let ibp = network.propagate_ibp(&input).unwrap();

    // Soundness check: sample corners and verify all outputs within CROWN bounds.
    let crown_flat = crown.flatten();
    let tol = 1e-3;
    for mask in 0..8u32 {
        let point: Vec<f32> = (0..3)
            .map(|i| {
                if mask & (1 << i) != 0 {
                    upper[i]
                } else {
                    lower[i]
                }
            })
            .collect();
        let pt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), point.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), point).unwrap(),
        )
        .unwrap();
        let out = network.propagate_ibp(&pt).unwrap().flatten();
        let val = out.lower()[[0]];
        assert!(
            crown_flat.lower()[[0]] <= val + tol,
            "CROWN lower {} > sample output {} at corner {:03b}",
            crown_flat.lower()[[0]],
            val,
            mask
        );
        assert!(
            crown_flat.upper()[[0]] >= val - tol,
            "CROWN upper {} < sample output {} at corner {:03b}",
            crown_flat.upper()[[0]],
            val,
            mask
        );
    }

    // CROWN bounds should be at least as tight as IBP.
    let ibp_flat = ibp.flatten();
    assert!(
        crown_flat.lower()[[0]] >= ibp_flat.lower()[[0]] - 1e-6,
        "CROWN lower ({}) should be >= IBP lower ({})",
        crown_flat.lower()[[0]],
        ibp_flat.lower()[[0]]
    );
    assert!(
        crown_flat.upper()[[0]] <= ibp_flat.upper()[[0]] + 1e-6,
        "CROWN upper ({}) should be <= IBP upper ({})",
        crown_flat.upper()[[0]],
        ibp_flat.upper()[[0]]
    );
}

/// NumericalInstability in the unary catch-all should also use per-layer
/// concretization instead of aborting to whole-network IBP.
///
/// Network: ReLU -> Sigmoid
/// Input: [-inf, 1]
///
/// ReLU's CROWN backward rejects the non-finite pre-activation, but its IBP
/// image is finite ([0, 1]). The fix in #3437 should concretize the accumulated
/// Sigmoid CROWN bounds through that finite IBP layer result and continue.
#[ntest::timeout(10000)]
#[test]
fn propagate_crown_per_layer_concretization_numerical_instability_relu() {
    use crate::layers::SigmoidLayer;

    let mut network = Network::new();
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Sigmoid(SigmoidLayer));

    let input = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[1.0f32]).into_dyn(),
    )
    .unwrap();

    let crown = network.propagate_crown(&input).unwrap();
    let ibp = network.propagate_ibp(&input).unwrap();
    let crown_flat = crown.flatten();
    let ibp_flat = ibp.flatten();

    assert!(
        crown_flat.lower()[[0]].is_finite() && crown_flat.upper()[[0]].is_finite(),
        "per-layer concretization should recover finite bounds after ReLU IBP: {:?}",
        crown_flat
    );
    assert!(
        crown_flat.lower()[[0]] <= crown_flat.upper()[[0]],
        "concretized bounds must remain ordered: [{}, {}]",
        crown_flat.lower()[[0]],
        crown_flat.upper()[[0]]
    );

    for x in [-10.0f32, -1.0, 0.0, 1.0] {
        let relu = x.max(0.0);
        let val = 1.0 / (1.0 + (-relu).exp());
        assert!(
            crown_flat.lower()[[0]] <= val + 1e-5,
            "CROWN lower {} > concrete output {} at x={x}",
            crown_flat.lower()[[0]],
            val
        );
        assert!(
            crown_flat.upper()[[0]] >= val - 1e-5,
            "CROWN upper {} < concrete output {} at x={x}",
            crown_flat.upper()[[0]],
            val
        );
    }

    assert!(
        crown_flat.lower()[[0]] >= ibp_flat.lower()[[0]] - 1e-6,
        "CROWN lower ({}) should be >= IBP lower ({})",
        crown_flat.lower()[[0]],
        ibp_flat.lower()[[0]]
    );
    assert!(
        crown_flat.upper()[[0]] <= ibp_flat.upper()[[0]] + 1e-6,
        "CROWN upper ({}) should be <= IBP upper ({})",
        crown_flat.upper()[[0]],
        ibp_flat.upper()[[0]]
    );
}
