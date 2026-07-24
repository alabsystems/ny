// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::ArrayD;

use super::super::{NonlinearBranching, NonlinearBranchingConfig, NonlinearHeuristicMethod};
use crate::layers::misc::SignLayer;
use crate::layers::{Layer, ReLULayer};

#[ntest::timeout(5000)]
#[test]
fn test_uniform_branching_points() {
    let branching = NonlinearBranching::default();
    let points = branching.branching_points(-1.0, 1.0, &Layer::GELU(Default::default()));
    assert_eq!(points.len(), 1);
    assert!((points[0] - 0.0).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
#[allow(deprecated)]
fn test_with_defaults_alias_matches_default_4227() {
    let via_alias = NonlinearBranching::with_defaults();
    let via_default = NonlinearBranching::default();
    let gelu = Layer::GELU(Default::default());

    assert_eq!(via_alias.config(), via_default.config());
    assert_eq!(
        via_alias.branching_points(-1.0, 1.0, &gelu),
        via_default.branching_points(-1.0, 1.0, &gelu),
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_relu_always_branches_at_zero() {
    let branching = NonlinearBranching::default();

    let points = branching.branching_points(-2.0, 3.0, &Layer::ReLU(ReLULayer));
    assert_eq!(points, vec![0.0]);

    let points = branching.branching_points(1.0, 3.0, &Layer::ReLU(ReLULayer));
    assert!(points.is_empty());

    let points = branching.branching_points(-3.0, -1.0, &Layer::ReLU(ReLULayer));
    assert!(points.is_empty());
}

#[ntest::timeout(5000)]
#[test]
fn test_is_splittable() {
    let branching = NonlinearBranching::default();

    assert!(branching.is_splittable(&Layer::ReLU(ReLULayer)));
    assert!(branching.is_splittable(&Layer::GELU(Default::default())));
    assert!(branching.is_splittable(&Layer::Sigmoid(crate::layers::SigmoidLayer)));
    assert!(branching.is_splittable(&Layer::Tanh(crate::layers::TanhLayer)));
    assert!(branching.is_splittable(&Layer::SiLU(Default::default())));
    assert!(branching.is_splittable(&Layer::LeakyReLU(crate::layers::LeakyReLULayer::new(0.01))));
    assert!(branching.is_splittable(&Layer::Softplus(crate::layers::SoftplusLayer)));
    assert!(branching.is_splittable(&Layer::Sin(Default::default())));
    assert!(branching.is_splittable(&Layer::Tan(Default::default())));
    assert!(branching.is_splittable(&Layer::Arctan(Default::default())));
    assert!(branching.is_splittable(&Layer::Elu(crate::layers::EluLayer { alpha: 1.0 })));
    assert!(branching.is_splittable(&Layer::Selu(crate::layers::SeluLayer)));
    assert!(branching.is_splittable(&Layer::Exp(Default::default())));
    assert!(branching.is_splittable(&Layer::Log(Default::default())));
    assert!(branching.is_splittable(&Layer::Mish(Default::default())));
    assert!(branching.is_splittable(&Layer::HardSwish(Default::default())));
    assert!(branching.is_splittable(&Layer::Softsign(crate::layers::SoftsignLayer)));
    assert!(branching.is_splittable(&Layer::Cos(Default::default())));
    assert!(branching.is_splittable(&Layer::Abs(crate::layers::AbsLayer)));

    let linear = crate::layers::LinearLayer::new(
        ndarray::Array2::zeros((2, 2)),
        Some(ndarray::Array1::zeros(2)),
    )
    .unwrap();
    assert!(!branching.is_splittable(&Layer::Linear(linear)));
}

#[ntest::timeout(5000)]
#[test]
fn test_bbps_score_weights_by_activation() {
    let branching = NonlinearBranching::default();
    let gelu = Layer::GELU(Default::default());
    let score_near_zero = branching.compute_score(-0.5, 0.5, &gelu, &[0.0]);
    let score_far = branching.compute_score(5.0, 6.0, &gelu, &[5.5]);
    assert!(score_near_zero > score_far);
}

/// Regression test for #1933: score_neurons must handle non-contiguous tensors.
#[test]
fn test_score_neurons_non_contiguous_1933() {
    let lower_c = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[2, 3]),
        vec![-1.0, -2.0, -3.0, 0.5, 0.1, 0.2],
    )
    .unwrap();
    let upper_c =
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 1.5, 1.1, 1.2])
            .unwrap();
    let lower_f = lower_c.t().to_owned();
    let upper_f = upper_c.t().to_owned();
    assert!(
        lower_f.as_slice().is_none(),
        "test setup: lower should be non-contiguous"
    );

    let bounds = ny_tensor::BoundedTensor::new(lower_f, upper_f).unwrap();
    let branching = NonlinearBranching::new(NonlinearBranchingConfig::default());
    let decisions = branching
        .score_neurons("test_relu", &Layer::ReLU(ReLULayer), &bounds)
        .unwrap();

    assert!(
        !decisions.is_empty(),
        "non-contiguous bounds must still produce branching decisions for unstable neurons"
    );
}

/// Regression test (#2882): score_neurons must skip neurons with NaN/Inf bounds.
#[ntest::timeout(5000)]
#[test]
fn test_score_neurons_nan_bounds_skipped_2882() {
    let branching = NonlinearBranching::default();
    let gelu = Layer::GELU(Default::default());

    let lower =
        ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![-1.0_f32, f32::NAN, -1.0, -1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[4]),
        vec![1.0_f32, 1.0, f32::NAN, f32::INFINITY],
    )
    .unwrap();
    let bounds = ny_tensor::BoundedTensor::new_unchecked(lower, upper).unwrap();

    let decisions = branching
        .score_neurons("test_gelu", &gelu, &bounds)
        .unwrap();
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].neuron_idx, 0);
    assert!(decisions[0].score.is_finite());
}

/// #3769 Slice 3: Sign branches at zero just like ReLU.
#[ntest::timeout(5000)]
#[test]
fn test_sign_branches_at_zero_3769() {
    let branching = NonlinearBranching::default();
    let sign = Layer::Sign(SignLayer::new());

    // Unstable Sign: crosses zero → branch at 0.0
    let points = branching.branching_points(-2.0, 3.0, &sign);
    assert_eq!(points, vec![0.0]);

    // Stable positive: no branching points
    let points = branching.branching_points(1.0, 3.0, &sign);
    assert!(points.is_empty());

    // Stable negative: no branching points
    let points = branching.branching_points(-3.0, -1.0, &sign);
    assert!(points.is_empty());
}

/// #3769 Slice 3: score_neurons skips stable Sign neurons.
#[ntest::timeout(5000)]
#[test]
fn test_sign_score_neurons_skips_stable_3769() {
    let branching = NonlinearBranching::default();
    let sign = Layer::Sign(SignLayer::new());

    // All neurons stable (positive): no decisions
    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.5, 1.0, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.5, 3.0, 4.0]).unwrap();
    let bounds = ny_tensor::BoundedTensor::new(lower, upper).unwrap();
    let decisions = branching
        .score_neurons("sign_node", &sign, &bounds)
        .unwrap();
    assert!(
        decisions.is_empty(),
        "stable Sign neurons must not produce branching decisions"
    );
}

/// #3769 Slice 3: score_neurons returns decisions for unstable Sign neurons.
#[ntest::timeout(5000)]
#[test]
fn test_sign_score_neurons_finds_unstable_3769() {
    let branching = NonlinearBranching::default();
    let sign = Layer::Sign(SignLayer::new());

    // Mix of stable and unstable: only unstable neuron at idx 1 crosses zero
    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.5, -2.0, -3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.5, 3.0, -0.5]).unwrap();
    let bounds = ny_tensor::BoundedTensor::new(lower, upper).unwrap();
    let decisions = branching
        .score_neurons("sign_node", &sign, &bounds)
        .unwrap();
    assert_eq!(
        decisions.len(),
        1,
        "only the unstable neuron should produce a decision"
    );
    assert_eq!(decisions[0].neuron_idx, 1);
    assert_eq!(decisions[0].points, vec![0.0]);
    assert!(decisions[0].score > 0.0);
}

/// Numerical validation of BBPS formula for ReLU/Sign branches.
/// Formula: `width * (1 - dist_to_zero / width)`, reference: `scoring.rs:125-130`
#[ntest::timeout(5000)]
#[test]
fn test_bbps_relu_sign_numerical_values() {
    let branching = NonlinearBranching::default();
    let tol = 1e-5;

    // ReLU: [-1.0, 3.0] → width=4, dist=min(1,3)=1, score=4*(1-1/4)=3.0
    let relu = Layer::ReLU(ReLULayer);
    let score = branching.compute_score(-1.0, 3.0, &relu, &[0.0]);
    assert!(
        (score - 3.0).abs() < tol,
        "ReLU [-1,3] BBPS expected 3.0, got {score}"
    );

    // ReLU: [-0.5, 0.5] → width=1, dist=0.5, score=1*(1-0.5)=0.5
    let score = branching.compute_score(-0.5, 0.5, &relu, &[0.0]);
    assert!(
        (score - 0.5).abs() < tol,
        "ReLU [-0.5,0.5] BBPS expected 0.5, got {score}"
    );

    // Sign: [-1.9, 0.1] → width=2, dist=min(1.9,0.1)=0.1, score=2*(1-0.1/2)=1.9
    let sign = Layer::Sign(SignLayer::new());
    let score = branching.compute_score(-1.9, 0.1, &sign, &[0.0]);
    assert!(
        (score - 1.9).abs() < tol,
        "Sign [-1.9,0.1] BBPS expected 1.9, got {score}"
    );
}

/// Numerical validation of BBPS formula for smooth activations and fallback.
/// GELU/SiLU: `width * (1 + exp(-|center|))`, reference: `scoring.rs:132-135`
/// Sigmoid/Tanh: `width * (1 + exp(-|center|/2))`, reference: `scoring.rs:136-139`
/// Softplus: `width * (1 + exp(-|center|) * 0.5)`, reference: `scoring.rs:140-143`
/// Fallback: `width`, reference: `scoring.rs:144`
#[ntest::timeout(5000)]
#[test]
fn test_bbps_smooth_activation_numerical_values() {
    let branching = NonlinearBranching::default();
    let tol = 1e-5;

    // GELU: [-0.5, 0.5] → width=1, center=0, score=1*(1+exp(0))=2.0
    let gelu = Layer::GELU(Default::default());
    let score = branching.compute_score(-0.5, 0.5, &gelu, &[0.0]);
    assert!(
        (score - 2.0).abs() < tol,
        "GELU centered expected 2.0, got {score}"
    );

    // GELU: [5.0, 6.0] → width=1, center=5.5, score=1*(1+exp(-5.5))
    let expected = 1.0 + (-5.5_f32).exp();
    let score = branching.compute_score(5.0, 6.0, &gelu, &[5.5]);
    assert!(
        (score - expected).abs() < tol,
        "GELU far expected {expected}, got {score}"
    );

    // SiLU: same formula as GELU → [-0.5, 0.5]: score=2.0
    let silu = Layer::SiLU(Default::default());
    let score = branching.compute_score(-0.5, 0.5, &silu, &[0.0]);
    assert!(
        (score - 2.0).abs() < tol,
        "SiLU centered expected 2.0, got {score}"
    );

    // Sigmoid: [-1.0, 1.0] → width=2, center=0, score=2*(1+exp(0))=4.0
    let sigmoid = Layer::Sigmoid(crate::layers::SigmoidLayer);
    let score = branching.compute_score(-1.0, 1.0, &sigmoid, &[0.0]);
    assert!(
        (score - 4.0).abs() < tol,
        "Sigmoid centered expected 4.0, got {score}"
    );

    // Tanh: [2.0, 4.0] → width=2, center=3, score=2*(1+exp(-1.5))
    let expected = 2.0 * (1.0 + (-1.5_f32).exp());
    let tanh = Layer::Tanh(crate::layers::TanhLayer);
    let score = branching.compute_score(2.0, 4.0, &tanh, &[0.0]);
    assert!(
        (score - expected).abs() < tol,
        "Tanh off-center expected {expected}, got {score}"
    );

    // Softplus: [-1.0, 1.0] → width=2, center=0, score=2*(1+1.0*0.5)=3.0
    let softplus = Layer::Softplus(crate::layers::SoftplusLayer);
    let score = branching.compute_score(-1.0, 1.0, &softplus, &[0.0]);
    assert!(
        (score - 3.0).abs() < tol,
        "Softplus centered expected 3.0, got {score}"
    );

    // Fallback (LeakyReLU): score = width = 4.0
    let leaky = Layer::LeakyReLU(crate::layers::LeakyReLULayer::new(0.01));
    let score = branching.compute_score(-1.0, 3.0, &leaky, &[0.0]);
    assert!(
        (score - 4.0).abs() < tol,
        "LeakyReLU fallback expected 4.0, got {score}"
    );
}

/// #3769 Slice 3: BBPS scoring for Sign matches ReLU scoring semantics.
/// The BBPS formula `width * (1 - dist_to_zero / width)` favors intervals where
/// zero is close to one edge (small dist_to_zero), because splitting there makes
/// one child very narrow (effectively solved). Sign and ReLU should produce
/// identical scores for the same bounds.
#[ntest::timeout(5000)]
#[test]
fn test_sign_bbps_score_matches_relu_3769() {
    let branching = NonlinearBranching::default();
    let sign = Layer::Sign(SignLayer::new());
    let relu = Layer::ReLU(ReLULayer);

    // Sign and ReLU should produce identical BBPS scores for the same bounds
    let sign_score = branching.compute_score(-1.0, 3.0, &sign, &[0.0]);
    let relu_score = branching.compute_score(-1.0, 3.0, &relu, &[0.0]);
    assert!(
        (sign_score - relu_score).abs() < 1e-6,
        "Sign BBPS score should match ReLU: sign={sign_score}, relu={relu_score}"
    );

    // BBPS prefers off-center intervals (small dist_to_zero) at equal width
    let score_off_center = branching.compute_score(-1.9, 0.1, &sign, &[0.0]);
    let score_centered = branching.compute_score(-1.0, 1.0, &sign, &[0.0]);
    assert!(
        score_off_center > score_centered,
        "BBPS should favor off-center intervals: off_center={score_off_center} > centered={score_centered}"
    );
}

/// BBPS formula boundary: zero-width interval must produce finite zero score.
/// The ReLU/Sign branch has `width.max(1e-6)` guard (scoring.rs:130) to prevent
/// division by zero. All other branches multiply by width, so zero width → zero score.
/// This test exercises the guard directly since `score_neurons` skips zero-width
/// neurons via `min_branch_width` before reaching `compute_score`.
#[ntest::timeout(5000)]
#[test]
fn test_bbps_zero_width_produces_finite_zero() {
    let branching = NonlinearBranching::default();
    let tol = 1e-6;

    // ReLU: width=0, dist_to_zero=1, score=0*(1-1/1e-6)=0
    let relu = Layer::ReLU(ReLULayer);
    let score = branching.compute_score(1.0, 1.0, &relu, &[0.0]);
    assert!(score.is_finite(), "ReLU zero-width must not be NaN/Inf");
    assert!(
        score.abs() < tol,
        "ReLU zero-width expected ~0, got {score}"
    );

    // GELU: width=0 * (1+exp(-1)) = 0
    let gelu = Layer::GELU(Default::default());
    let score = branching.compute_score(1.0, 1.0, &gelu, &[0.0]);
    assert!(score.is_finite(), "GELU zero-width must not be NaN/Inf");
    assert!(
        score.abs() < tol,
        "GELU zero-width expected ~0, got {score}"
    );

    // Sigmoid: width=0 * (1+exp(0)) = 0
    let sigmoid = Layer::Sigmoid(crate::layers::SigmoidLayer);
    let score = branching.compute_score(0.0, 0.0, &sigmoid, &[0.0]);
    assert!(score.is_finite(), "Sigmoid zero-width must not be NaN/Inf");
    assert!(
        score.abs() < tol,
        "Sigmoid zero-width expected ~0, got {score}"
    );

    // Softplus: width=0 * (1+exp(0)*0.5) = 0
    let softplus = Layer::Softplus(crate::layers::SoftplusLayer);
    let score = branching.compute_score(0.0, 0.0, &softplus, &[0.0]);
    assert!(score.is_finite(), "Softplus zero-width must not be NaN/Inf");
    assert!(
        score.abs() < tol,
        "Softplus zero-width expected ~0, got {score}"
    );

    // Fallback (LeakyReLU): width=0
    let leaky = Layer::LeakyReLU(crate::layers::LeakyReLULayer::new(0.01));
    let score = branching.compute_score(2.0, 2.0, &leaky, &[0.0]);
    assert!(score.is_finite(), "Fallback zero-width must not be NaN/Inf");
    assert!(
        score.abs() < tol,
        "Fallback zero-width expected ~0, got {score}"
    );
}

/// BoundWidth method returns `upper - lower` regardless of activation type.
#[ntest::timeout(5000)]
#[test]
fn test_bound_width_method_ignores_activation_type() {
    let config = NonlinearBranchingConfig {
        method: NonlinearHeuristicMethod::BoundWidth,
        ..Default::default()
    };
    let branching = NonlinearBranching::new(config);

    let relu = Layer::ReLU(ReLULayer);
    let gelu = Layer::GELU(Default::default());
    let sigmoid = Layer::Sigmoid(crate::layers::SigmoidLayer);

    // BoundWidth always returns width = upper - lower
    let score_relu = branching.compute_score(-1.0, 3.0, &relu, &[0.0]);
    let score_gelu = branching.compute_score(-1.0, 3.0, &gelu, &[0.0]);
    let score_sigmoid = branching.compute_score(-1.0, 3.0, &sigmoid, &[0.0]);

    assert!(
        (score_relu - 4.0).abs() < 1e-6,
        "BoundWidth ReLU [-1,3] expected 4.0, got {score_relu}"
    );
    assert!(
        (score_gelu - 4.0).abs() < 1e-6,
        "BoundWidth GELU [-1,3] expected 4.0, got {score_gelu}"
    );
    assert!(
        (score_sigmoid - 4.0).abs() < 1e-6,
        "BoundWidth Sigmoid [-1,3] expected 4.0, got {score_sigmoid}"
    );

    // All three must be identical
    assert!(
        (score_relu - score_gelu).abs() < 1e-6 && (score_gelu - score_sigmoid).abs() < 1e-6,
        "BoundWidth must produce identical scores across activation types"
    );
}
