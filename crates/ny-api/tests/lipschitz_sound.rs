// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! NY ext 2 acceptance: SOUND global Lipschitz certification against hand
//! networks with known Lipschitz constants, including a case where the old
//! optimistic estimate under-reports and the sound bound does not.

use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_api::graph::SequentialNetwork;
use ny_api::layers::{Conv1dLayer, Conv2dLayer, Layer, LinearLayer, ReLULayer, SigmoidLayer};
use ny_api::lipschitz::{certify_upper_bound, NormBoundKind};
use ny_api::probabilistic::estimate_lipschitz_from_network;
use ny_cert::Rat;

fn rat(n: i128) -> Rat {
    Rat::from_int(n)
}

/// Single Linear layer with a known exact operator norm: W = [3 4] has
/// ‖W‖₂ = 5 exactly (rank one ⇒ Frobenius is exact), so the certified bound
/// brackets the true constant exactly.
#[test]
fn single_linear_exact_operator_norm_bracketing() {
    let mut network = SequentialNetwork::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[3.0_f32, 4.0]]), None).expect("valid linear"),
    ));
    let sound = certify_upper_bound(&network).expect("certifiable");
    assert_eq!(sound.bound, rat(5), "rank-one row: bound must be exactly 5");
    assert_eq!(sound.squared_bound, rat(25));
    assert_eq!(sound.per_layer.len(), 1);
    assert_eq!(sound.per_layer[0].norm_kind, NormBoundKind::Frobenius);

    // Cross-check against the f32 spectral norm the estimator uses.
    let estimate = estimate_lipschitz_from_network(&network).expect("estimable");
    assert!(estimate.is_sound, "pure-Linear estimate is flagged sound");
    assert!(
        sound.bound.to_f64_approx() >= f64::from(estimate.value) - 1e-6,
        "sound bound must not undercut the true operator norm"
    );
}

/// Diagonal matrix: the 1/∞ product beats Frobenius and is exact.
/// W = diag(2, 3): ‖W‖₂ = 3, ‖W‖₁·‖W‖∞ = 9 < 13 = ‖W‖_F².
#[test]
fn diagonal_matrix_one_inf_product_is_exact() {
    let mut network = SequentialNetwork::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[2.0_f32, 0.0], [0.0, 3.0]]), None).expect("valid linear"),
    ));
    let sound = certify_upper_bound(&network).expect("certifiable");
    assert_eq!(sound.bound, rat(3));
    assert_eq!(sound.per_layer[0].norm_kind, NormBoundKind::OneInfProduct);
}

/// Composition multiplies: 2·I → ReLU → 3·I has Lipschitz constant exactly 6
/// (attained on the positive orthant), and the certified bound is exactly 6.
#[test]
fn composition_of_scalings_multiplies_exactly() {
    let mut network = SequentialNetwork::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[2.0_f32, 0.0], [0.0, 2.0]]), None).expect("valid linear"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[3.0_f32, 0.0], [0.0, 3.0]]), None).expect("valid linear"),
    ));
    let sound = certify_upper_bound(&network).expect("certifiable");
    assert_eq!(sound.bound, rat(6));
    assert_eq!(sound.squared_bound, rat(36));
    assert_eq!(sound.per_layer.len(), 3);
    assert_eq!(sound.per_layer[1].squared_bound, Rat::ONE);
    assert_eq!(sound.per_layer[1].norm_kind, NormBoundKind::UnitLipschitz);
}

/// Non-exact case: the bound is a certified UPPER bound (r² ≥ Q) that stays
/// within the documented min(√(‖·‖₁‖·‖∞), ‖·‖_F) formula.
/// W = [[1, 1], [0, 1]]: ‖W‖₂ = golden ratio ≈ 1.618; ‖W‖₁‖W‖∞ = 4,
/// ‖W‖_F² = 3 ⇒ certified bound ≈ √3, and √3 ≥ φ.
#[test]
fn irrational_case_is_soundly_rounded_outward() {
    let mut network = SequentialNetwork::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, 1.0], [0.0, 1.0]]), None).expect("valid linear"),
    ));
    let sound = certify_upper_bound(&network).expect("certifiable");
    assert_eq!(sound.squared_bound, rat(3), "Frobenius² = 3 wins over 4");
    // Certified: bound² ≥ 3 exactly.
    assert!(sound.bound.mul(sound.bound).expect("exact") >= rat(3));
    // And tight: within 1e-9 of √3.
    let approx = sound.bound.to_f64_approx();
    assert!(
        (approx - 3.0_f64.sqrt()).abs() < 1e-9,
        "bound ≈ √3, got {approx}"
    );
    // The true operator norm φ ≈ 1.618 is below the certified bound.
    assert!(approx >= 1.618);
}

/// The trap case from the task: the old estimate treats a Conv2d layer
/// optimistically as 1-Lipschitz (`is_sound == false`), under-reporting the
/// true constant; the sound certifier handles it exactly and does not.
#[test]
fn conv_net_where_old_estimate_is_optimistic() {
    // 1×1 conv kernel [4]: the layer scales every pixel by 4 ⇒ true L = 4.
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![4.0_f32]).expect("shape");
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).expect("valid conv");
    let mut network = SequentialNetwork::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));

    let estimate = estimate_lipschitz_from_network(&network).expect("estimable");
    assert!(
        !estimate.is_sound,
        "old estimate must flag Conv2d as unhandled/optimistic"
    );
    assert!(
        estimate.value < 4.0,
        "old estimate under-reports the true constant 4, got {}",
        estimate.value
    );

    let sound = certify_upper_bound(&network).expect("certifiable");
    assert_eq!(sound.bound, rat(4), "sound bound covers the conv scaling");
    assert_eq!(
        sound.per_layer[0].norm_kind,
        NormBoundKind::OneInfProduct,
        "conv layers use the ‖A‖₁·‖A‖∞ bound"
    );
}

/// Conv1d bound vs the materialized unrolled matrix: kernel [1, 2] over a
/// length-3 input (stride 1, no padding) unrolls to A = [[2, 1, 0], [0, 2, 1]]
/// with ‖A‖₁ = ‖A‖∞ = 3 and true ‖A‖₂ = √(σmax(AᵀA)) ≈ 2.723 ≤ 3.
#[test]
fn conv1d_bound_dominates_materialized_operator_norm() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0_f32, 2.0]).expect("shape");
    let conv = Conv1dLayer::new(kernel, None, 1, 0).expect("valid conv");
    let mut network = SequentialNetwork::new();
    network.add_layer(Layer::Conv1d(conv));
    let sound = certify_upper_bound(&network).expect("certifiable");
    assert_eq!(sound.bound, rat(3));

    // Numeric cross-check via the equivalent LinearLayer's spectral norm.
    let unrolled = LinearLayer::new(arr2(&[[2.0_f32, 1.0, 0.0], [0.0, 2.0, 1.0]]), None)
        .expect("valid linear");
    assert!(
        sound.bound.to_f64_approx() >= f64::from(unrolled.spectral_norm()) - 1e-5,
        "certified conv bound must dominate the unrolled spectral norm"
    );
}

/// Grouped conv: each group is an independent scaling, so the operator norm
/// is the max group scaling (5), and the certified bound matches exactly.
#[test]
fn grouped_conv_takes_worst_group() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![2.0_f32, 5.0]).expect("shape");
    let conv = Conv1dLayer::new_full(kernel, None, 1, 0, 1, 2).expect("valid conv");
    let mut network = SequentialNetwork::new();
    network.add_layer(Layer::Conv1d(conv));
    let sound = certify_upper_bound(&network).expect("certifiable");
    assert_eq!(sound.bound, rat(5));
}

/// Fail-closed contract: a layer outside the certified fragment is an error
/// naming the layer — never a silently optimistic number.
#[test]
fn unsupported_layer_is_an_error_not_an_estimate() {
    let mut network = SequentialNetwork::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("valid linear"),
    ));
    network.add_layer(Layer::Sigmoid(SigmoidLayer));
    let err = certify_upper_bound(&network).expect_err("must fail closed");
    let msg = err.to_string();
    assert!(msg.contains("Sigmoid"), "error names the layer: {msg}");

    // The optimistic path, by contrast, accepts this net (Sigmoid is on its
    // 1-Lipschitz allow-list) — documenting the difference in contracts.
    assert!(estimate_lipschitz_from_network(&network).is_ok());
}

/// Weights are converted exactly (dyadic f32 values), so a non-trivial
/// mantissa still yields an exact rational product: L = 0.5 · 0.25 = 1/8.
#[test]
fn dyadic_weights_stay_exact_through_the_product() {
    let mut network = SequentialNetwork::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[0.5_f32]]), None).expect("valid linear"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[0.25_f32]]), None).expect("valid linear"),
    ));
    let sound = certify_upper_bound(&network).expect("certifiable");
    assert_eq!(sound.bound, Rat::new(1, 8).expect("exact"));

    // Bias never enters a Lipschitz bound.
    let mut with_bias = SequentialNetwork::new();
    with_bias.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[0.5_f32]]), Some(arr1(&[100.0_f32]))).expect("valid linear"),
    ));
    let sound_bias = certify_upper_bound(&with_bias).expect("certifiable");
    assert_eq!(sound_bias.bound, Rat::new(1, 2).expect("exact"));
}
