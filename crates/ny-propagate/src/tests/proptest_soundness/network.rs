// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::AlphaCrownConfig;
use crate::layers::trigonometric::SigmoidLayer;
use crate::layers::ReduceMaxLayer;
use crate::{Layer, LinearLayer, Network, ReLULayer};
use ndarray::{arr1, arr2};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{sample_points, sigmoid_eval, valid_interval, FP_TOLERANCE};

// =============================================================================
// MULTI-LAYER NETWORK SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Two-layer network (Linear -> ReLU) soundness.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_linear_relu_network(
        w11 in -3.0f32..3.0,
        w12 in -3.0f32..3.0,
        w21 in -3.0f32..3.0,
        w22 in -3.0f32..3.0,
        b1 in -3.0f32..3.0,
        b2 in -3.0f32..3.0,
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
    ) {
        let weight = arr2(&[[w11, w12], [w21, w22]]);
        let bias = arr1(&[b1, b2]);
        let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));
        network.add_layer(Layer::ReLU(ReLULayer));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let output = network.propagate_ibp(&input).unwrap();

        // Test multiple concrete points
        for x1 in sample_points(l1, u1, 4) {
            for x2 in sample_points(l2, u2, 4) {
                let x = arr1(&[x1, x2]);
                let linear_out = weight.dot(&x) + &bias;
                let relu_out = linear_out.mapv(|v| v.max(0.0));

                for i in 0..2 {
                    prop_assert!(
                        output.lower()[[i]] - FP_TOLERANCE <= relu_out[i] && relu_out[i] <= output.upper()[[i]] + FP_TOLERANCE,
                        "Linear-ReLU network soundness violation: ReLU(Wx+b)[{}]={} not in [{}, {}]",
                        i, relu_out[i], output.lower()[[i]], output.upper()[[i]]
                    );
                }
            }
        }
    }

    /// Three-layer network (Linear -> ReLU -> Linear) soundness.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_linear_relu_linear_network(
        w1_vec in prop::collection::vec(-2.0f32..2.0, 4),
        b1_vec in prop::collection::vec(-2.0f32..2.0, 2),
        w2_vec in prop::collection::vec(-2.0f32..2.0, 2),
        b2 in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = ndarray::Array1::from_vec(b1_vec);
        let w2 = ndarray::Array2::from_shape_vec((1, 2), w2_vec).unwrap();
        let b2_arr = ndarray::Array1::from_vec(vec![b2]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(LinearLayer::new(w2.clone(), Some(b2_arr.clone())).unwrap()));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let output = network.propagate_ibp(&input).unwrap();

        // Test concrete points
        for x1 in sample_points(l1, u1, 4) {
            for x2 in sample_points(l2, u2, 4) {
                let x = arr1(&[x1, x2]);
                let y1 = w1.dot(&x) + &b1;
                let relu_out = y1.mapv(|v| v.max(0.0));
                let final_out = w2.dot(&relu_out) + &b2_arr;

                prop_assert!(
                    output.lower()[[0]] - FP_TOLERANCE <= final_out[0] && final_out[0] <= output.upper()[[0]] + FP_TOLERANCE,
                    "3-layer network soundness violation: output={} not in [{}, {}]",
                    final_out[0], output.lower()[[0]], output.upper()[[0]]
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(100) })]

    /// #3437: when `ReduceMax(fixed=false)` forces the unary catch-all onto the
    /// per-layer IBP concretization path, the resulting network bounds must stay
    /// sound and no worse than plain IBP.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_crown_reduce_max_per_layer_concretization(
        w11 in -2.0f32..2.0,
        w12 in -2.0f32..2.0,
        w21 in -2.0f32..2.0,
        w22 in -2.0f32..2.0,
        b1 in -2.0f32..2.0,
        b2 in -2.0f32..2.0,
        out_w in -2.0f32..2.0,
        out_b in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let hidden_weight = arr2(&[[w11, w12], [w21, w22]]);
        let hidden_bias = arr1(&[b1, b2]);
        let output_weight = arr2(&[[out_w]]);
        let output_bias = arr1(&[out_b]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(hidden_weight.clone(), Some(hidden_bias.clone())).unwrap()
        ));
        network.add_layer(Layer::ReduceMax(ReduceMaxLayer {
            axes: vec![-1],
            keepdims: true,
            fixed_max_index: false,
        }));
        network.add_layer(Layer::Sigmoid(SigmoidLayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(output_weight.clone(), Some(output_bias.clone())).unwrap()
        ));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let crown_output = network.propagate_crown(&input).unwrap();
        let ibp_output = network.propagate_ibp(&input).unwrap();

        prop_assert!(
            crown_output.lower()[[0]].is_finite() && crown_output.upper()[[0]].is_finite(),
            "per-layer concretization should keep finite bounds: [{}, {}]",
            crown_output.lower()[[0]],
            crown_output.upper()[[0]]
        );
        prop_assert!(
            crown_output.lower()[[0]] <= crown_output.upper()[[0]] + FP_TOLERANCE,
            "per-layer concretization produced inverted bounds: [{}, {}]",
            crown_output.lower()[[0]],
            crown_output.upper()[[0]]
        );

        for x1 in sample_points(l1, u1, 4) {
            for x2 in sample_points(l2, u2, 4) {
                let x = arr1(&[x1, x2]);
                let hidden = hidden_weight.dot(&x) + &hidden_bias;
                let reduced = hidden[0].max(hidden[1]);
                let sig = 1.0 / (1.0 + (-reduced).exp());
                let final_out = output_weight[[0, 0]] * sig + output_bias[0];

                prop_assert!(
                    crown_output.lower()[[0]] - FP_TOLERANCE <= final_out
                        && final_out <= crown_output.upper()[[0]] + FP_TOLERANCE,
                    "ReduceMax concretization soundness violation: output={} not in [{}, {}]",
                    final_out,
                    crown_output.lower()[[0]],
                    crown_output.upper()[[0]]
                );
            }
        }

        prop_assert!(
            crown_output.lower()[[0]] >= ibp_output.lower()[[0]] - FP_TOLERANCE,
            "per-layer concretization lower ({}) should be >= IBP lower ({})",
            crown_output.lower()[[0]],
            ibp_output.lower()[[0]]
        );
        prop_assert!(
            crown_output.upper()[[0]] <= ibp_output.upper()[[0]] + FP_TOLERANCE,
            "per-layer concretization upper ({}) should be <= IBP upper ({})",
            crown_output.upper()[[0]],
            ibp_output.upper()[[0]]
        );
    }
}

// =============================================================================
// MULTI-LAYER CROWN COMPOSITION SOUNDNESS TESTS
//
// Part of #2842: The IBP tests above verify forward-bound propagation. These
// CROWN composition tests verify the backward linear relaxation path, which
// is the primary bound-tightening algorithm. CROWN backward composition is
// where soundness bugs hide — a correct single-layer relaxation can produce
// unsound bounds when composed with another layer's relaxation through the
// backward pass.
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// CROWN soundness for Linear -> ReLU (2-layer).
    ///
    /// Verifies that CROWN backward composition through ReLU produces bounds
    /// that contain all true network outputs. This catches sign errors in
    /// backward coefficient propagation through the ReLU relaxation.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_crown_linear_relu(
        w11 in -3.0f32..3.0,
        w12 in -3.0f32..3.0,
        w21 in -3.0f32..3.0,
        w22 in -3.0f32..3.0,
        b1 in -3.0f32..3.0,
        b2 in -3.0f32..3.0,
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
    ) {
        let weight = arr2(&[[w11, w12], [w21, w22]]);
        let bias = arr1(&[b1, b2]);
        let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));
        network.add_layer(Layer::ReLU(ReLULayer));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let crown_output = network.propagate_crown(&input).unwrap();

        for x1 in sample_points(l1, u1, 4) {
            for x2 in sample_points(l2, u2, 4) {
                let x = arr1(&[x1, x2]);
                let linear_out = weight.dot(&x) + &bias;
                let relu_out = linear_out.mapv(|v| v.max(0.0));

                for i in 0..2 {
                    prop_assert!(
                        crown_output.lower()[[i]] - FP_TOLERANCE <= relu_out[i]
                            && relu_out[i] <= crown_output.upper()[[i]] + FP_TOLERANCE,
                        "CROWN Linear-ReLU soundness violation: output[{}]={} not in [{}, {}]",
                        i, relu_out[i], crown_output.lower()[[i]], crown_output.upper()[[i]]
                    );
                }
            }
        }
    }

    /// CROWN soundness for Linear -> ReLU -> Linear (3-layer).
    ///
    /// This is the canonical test for CROWN backward composition: the backward
    /// pass must correctly compose the identity at the output, propagate through
    /// the second linear layer's weight matrix, then through the ReLU relaxation,
    /// and finally through the first linear layer. Incorrect concretization when
    /// lower/upper A-matrices have mixed signs after composition is caught here.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_crown_linear_relu_linear(
        w1_vec in prop::collection::vec(-2.0f32..2.0, 4),
        b1_vec in prop::collection::vec(-2.0f32..2.0, 2),
        w2_vec in prop::collection::vec(-2.0f32..2.0, 2),
        b2 in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = ndarray::Array1::from_vec(b1_vec);
        let w2 = ndarray::Array2::from_shape_vec((1, 2), w2_vec).unwrap();
        let b2_arr = ndarray::Array1::from_vec(vec![b2]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(LinearLayer::new(w2.clone(), Some(b2_arr.clone())).unwrap()));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let crown_output = network.propagate_crown(&input).unwrap();

        for x1 in sample_points(l1, u1, 4) {
            for x2 in sample_points(l2, u2, 4) {
                let x = arr1(&[x1, x2]);
                let y1 = w1.dot(&x) + &b1;
                let relu_out = y1.mapv(|v| v.max(0.0));
                let final_out = w2.dot(&relu_out) + &b2_arr;

                prop_assert!(
                    crown_output.lower()[[0]] - FP_TOLERANCE <= final_out[0]
                        && final_out[0] <= crown_output.upper()[[0]] + FP_TOLERANCE,
                    "CROWN 3-layer soundness violation: output={} not in [{}, {}]",
                    final_out[0], crown_output.lower()[[0]], crown_output.upper()[[0]]
                );
            }
        }
    }

    /// CROWN soundness for Linear -> Sigmoid -> Linear (3-layer, S-shaped activation).
    ///
    /// Sigmoid is an S-shaped activation with a non-trivial CROWN relaxation.
    /// The backward composition through Sigmoid's linear relaxation is different
    /// from ReLU (which is piecewise linear). This catches errors specific to
    /// smooth activation relaxations.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_crown_linear_sigmoid_linear(
        w1_vec in prop::collection::vec(-2.0f32..2.0, 4),
        b1_vec in prop::collection::vec(-2.0f32..2.0, 2),
        w2_vec in prop::collection::vec(-2.0f32..2.0, 2),
        b2 in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = ndarray::Array1::from_vec(b1_vec);
        let w2 = ndarray::Array2::from_shape_vec((1, 2), w2_vec).unwrap();
        let b2_arr = ndarray::Array1::from_vec(vec![b2]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()));
        network.add_layer(Layer::Sigmoid(SigmoidLayer));
        network.add_layer(Layer::Linear(LinearLayer::new(w2.clone(), Some(b2_arr.clone())).unwrap()));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let crown_output = network.propagate_crown(&input).unwrap();

        for x1 in sample_points(l1, u1, 4) {
            for x2 in sample_points(l2, u2, 4) {
                let x = arr1(&[x1, x2]);
                let y1 = w1.dot(&x) + &b1;
                let sig_out = y1.mapv(sigmoid_eval);
                let final_out = w2.dot(&sig_out) + &b2_arr;

                prop_assert!(
                    crown_output.lower()[[0]] - FP_TOLERANCE <= final_out[0]
                        && final_out[0] <= crown_output.upper()[[0]] + FP_TOLERANCE,
                    "CROWN Linear-Sigmoid-Linear soundness violation: output={} not in [{}, {}]",
                    final_out[0], crown_output.lower()[[0]], crown_output.upper()[[0]]
                );
            }
        }
    }

    /// α-CROWN soundness for Linear -> ReLU -> Linear (3-layer).
    ///
    /// Verifies that α-CROWN optimization on a sequential network produces
    /// sound bounds. α-CROWN extends CROWN by optimizing the ReLU lower-bound
    /// slope (α), which should tighten bounds without violating soundness.
    /// Uses few iterations (5) to keep proptest fast while still exercising
    /// the optimization loop.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_alpha_crown_linear_relu_linear(
        w1_vec in prop::collection::vec(-2.0f32..2.0, 4),
        b1_vec in prop::collection::vec(-2.0f32..2.0, 2),
        w2_vec in prop::collection::vec(-2.0f32..2.0, 2),
        b2 in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = ndarray::Array1::from_vec(b1_vec);
        let w2 = ndarray::Array2::from_shape_vec((1, 2), w2_vec).unwrap();
        let b2_arr = ndarray::Array1::from_vec(vec![b2]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(LinearLayer::new(w2.clone(), Some(b2_arr.clone())).unwrap()));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        // Use few iterations to keep proptest fast; soundness must hold regardless
        let config = AlphaCrownConfig {
            iterations: 5,
            ..AlphaCrownConfig::default()
        };
        let alpha_output = network.propagate_alpha_crown_with_config(&input, &config).unwrap();

        for x1 in sample_points(l1, u1, 4) {
            for x2 in sample_points(l2, u2, 4) {
                let x = arr1(&[x1, x2]);
                let y1 = w1.dot(&x) + &b1;
                let relu_out = y1.mapv(|v| v.max(0.0));
                let final_out = w2.dot(&relu_out) + &b2_arr;

                prop_assert!(
                    alpha_output.lower()[[0]] - FP_TOLERANCE <= final_out[0]
                        && final_out[0] <= alpha_output.upper()[[0]] + FP_TOLERANCE,
                    "α-CROWN 3-layer soundness violation: output={} not in [{}, {}]",
                    final_out[0], alpha_output.lower()[[0]], alpha_output.upper()[[0]]
                );
            }
        }
    }

    /// CROWN bounds are at least as tight as IBP for Linear -> ReLU -> Linear.
    ///
    /// CROWN uses backward linear relaxation which should produce bounds at least
    /// as tight as forward IBP for any network with supported layers. This catches
    /// cases where CROWN backward composition produces bounds that are *wider*
    /// than IBP, which would indicate a regression in the relaxation quality.
    #[ntest::timeout(10000)]
    #[test]
    fn crown_tighter_than_ibp_linear_relu_linear(
        w1_vec in prop::collection::vec(-2.0f32..2.0, 4),
        b1_vec in prop::collection::vec(-2.0f32..2.0, 2),
        w2_vec in prop::collection::vec(-2.0f32..2.0, 2),
        b2 in -2.0f32..2.0,
        (l1, u1) in valid_interval(3.0),
        (l2, u2) in valid_interval(3.0),
    ) {
        let w1 = ndarray::Array2::from_shape_vec((2, 2), w1_vec).unwrap();
        let b1 = ndarray::Array1::from_vec(b1_vec);
        let w2 = ndarray::Array2::from_shape_vec((1, 2), w2_vec).unwrap();
        let b2_arr = ndarray::Array1::from_vec(vec![b2]);

        let mut network = Network::new();
        network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2_arr)).unwrap()));

        let input = BoundedTensor::new(
            arr1(&[l1, l2]).into_dyn(),
            arr1(&[u1, u2]).into_dyn()
        ).unwrap();

        let ibp_output = network.propagate_ibp(&input).unwrap();
        let crown_output = network.propagate_crown(&input).unwrap();

        // CROWN lower bound should be >= IBP lower bound (tighter)
        // CROWN upper bound should be <= IBP upper bound (tighter)
        // Allow FP_TOLERANCE for floating-point rounding
        prop_assert!(
            crown_output.lower()[[0]] >= ibp_output.lower()[[0]] - FP_TOLERANCE,
            "CROWN lower bound ({}) is looser than IBP ({}) — regression in CROWN quality",
            crown_output.lower()[[0]], ibp_output.lower()[[0]]
        );
        prop_assert!(
            crown_output.upper()[[0]] <= ibp_output.upper()[[0]] + FP_TOLERANCE,
            "CROWN upper bound ({}) is looser than IBP ({}) — regression in CROWN quality",
            crown_output.upper()[[0]], ibp_output.upper()[[0]]
        );
    }
}
