// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::{
    AddConstantLayer, DivConstantLayer, LogSumExpLayer, MulConstantLayer, SnakeLayer,
    SubConstantLayer,
};
use crate::streaming::*;

fn assert_batched_streaming_crown_contains_concrete_outputs(
    network: &Network,
    input: &BoundedTensor,
    samples_per_dim: usize,
    config: StreamingConfig,
) {
    assert_eq!(
        input.shape(),
        &[2],
        "batched concrete soundness helper currently supports 2D input domains"
    );
    let sampled_points = (samples_per_dim + 1) * (samples_per_dim + 1);
    assert!(
        sampled_points >= 100,
        "batched concrete soundness tests must sample at least 100 points"
    );

    let verifier = StreamingVerifier::new(config);
    let crown_bounds = verifier
        .propagate_crown_batched_streaming(network, input)
        .expect("batched streaming CROWN sound propagation should succeed");

    let input_lower = input.lower();
    let input_upper = input.upper();
    for i in 0..=samples_per_dim {
        for j in 0..=samples_per_dim {
            let x0 = input_lower[[0]]
                + (input_upper[[0]] - input_lower[[0]]) * (i as f32) / (samples_per_dim as f32);
            let x1 = input_lower[[1]]
                + (input_upper[[1]] - input_lower[[1]]) * (j as f32) / (samples_per_dim as f32);
            let point = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![x0, x1])
                .expect("invariant: sample point shape");
            let concrete_input = BoundedTensor::concrete(point).unwrap();
            let concrete_output = network
                .propagate_ibp(&concrete_input)
                .expect("invariant: concrete propagation should succeed");
            let output = concrete_output.lower();

            let tol = 1e-5;
            for k in 0..output.len() {
                assert!(
                    crown_bounds.lower()[[k]] <= output[[k]] + tol,
                    "Batched streaming lower[{k}] = {} > concrete output[{k}] = {} at ({x0}, {x1})",
                    crown_bounds.lower()[[k]],
                    output[[k]]
                );
                assert!(
                    crown_bounds.upper()[[k]] >= output[[k]] - tol,
                    "Batched streaming upper[{k}] = {} < concrete output[{k}] = {} at ({x0}, {x1})",
                    crown_bounds.upper()[[k]],
                    output[[k]]
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_batched_crown_empty_network() {
    let network = Network::new();
    let input = create_input(10);

    let verifier = StreamingVerifier::new(StreamingConfig::default());
    let result = verifier
        .propagate_crown_batched_streaming(&network, &input)
        .unwrap();

    assert_eq!(result.shape(), input.shape());

    // Empty network: output bounds must equal input bounds (identity propagation).
    assert_eq!(
        result.lower(),
        input.lower(),
        "empty network: lower must equal input"
    );
    assert_eq!(
        result.upper(),
        input.upper(),
        "empty network: upper must equal input"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_batched_crown_linear_only() {
    let network = create_test_network(5, 8, 8);
    let input = create_input(8);

    let verifier = StreamingVerifier::new(StreamingConfig::default());
    let batched = verifier
        .propagate_crown_batched_streaming(&network, &input)
        .unwrap();

    // Compare against non-batched streaming CROWN for value correctness.
    let regular = verifier
        .propagate_crown_streaming(&network, &input)
        .unwrap();

    assert_eq!(batched.shape(), regular.shape());
    for (idx, (b, r)) in batched
        .lower()
        .iter()
        .zip(regular.lower().iter())
        .enumerate()
    {
        assert!(
            (b - r).abs() < 1e-5,
            "lower mismatch at {}: batched={} regular={}",
            idx,
            b,
            r
        );
    }
    for (idx, (b, r)) in batched
        .upper()
        .iter()
        .zip(regular.upper().iter())
        .enumerate()
    {
        assert!(
            (b - r).abs() < 1e-5,
            "upper mismatch at {}: batched={} regular={}",
            idx,
            b,
            r
        );
    }
}

/// #3550 regression: zero CPU dense budget forces batched streaming CROWN to
/// reuse the regular streaming fallback instead of allocating a batched identity.
#[ntest::timeout(10000)]
#[test]
fn test_streaming_batched_crown_zero_budget_falls_back_to_regular_3550() {
    crate::tests::with_crown_dense_budget_mb("0", || {
        let network = create_test_network(5, 8, 8);
        let input = create_input(8);
        let verifier = StreamingVerifier::new(StreamingConfig::default());

        let regular = verifier
            .propagate_crown_streaming(&network, &input)
            .unwrap();
        let batched = verifier
            .propagate_crown_batched_streaming(&network, &input)
            .unwrap();

        assert_eq!(batched.shape(), regular.shape());
        assert_eq!(batched.lower(), regular.lower());
        assert_eq!(batched.upper(), regular.upper());
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_batched_crown_constant_arithmetic_layers() {
    let mut network = Network::new();

    let weight1 = Array2::from_shape_fn((8, 8), |(i, j)| {
        let phase = (i * 17 + j * 11) as f32;
        0.1 * phase.sin()
    });
    let weight2 = Array2::from_shape_fn((8, 8), |(i, j)| {
        let phase = (i * 13 + j * 19) as f32;
        0.1 * phase.cos()
    });

    network.add_layer(Layer::Linear(
        LinearLayer::new(weight1, Some(Array1::<f32>::zeros(8))).unwrap(),
    ));
    network.add_layer(Layer::AddConstant(AddConstantLayer::new(
        ArrayD::from_elem(ndarray::IxDyn(&[]), 0.25),
    )));
    network.add_layer(Layer::SubConstant(SubConstantLayer::scalar(0.10)));
    network.add_layer(Layer::MulConstant(MulConstantLayer::scalar(1.5)));
    network.add_layer(Layer::DivConstant(DivConstantLayer::scalar(2.0)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(weight2, Some(Array1::<f32>::zeros(8))).unwrap(),
    ));

    let input = create_input(8);
    let verifier = StreamingVerifier::new(StreamingConfig::default());

    let batched_result = verifier
        .propagate_crown_batched_streaming(&network, &input)
        .unwrap();
    let regular_result = verifier
        .propagate_crown_streaming(&network, &input)
        .unwrap();

    assert_eq!(batched_result.shape(), regular_result.shape());

    for (idx, (lhs, rhs)) in batched_result
        .lower()
        .iter()
        .zip(regular_result.lower().iter())
        .enumerate()
    {
        assert!(
            (lhs - rhs).abs() <= 1e-5,
            "lower mismatch at {}: batched={} regular={}",
            idx,
            lhs,
            rhs
        );
    }

    for (idx, (lhs, rhs)) in batched_result
        .upper()
        .iter()
        .zip(regular_result.upper().iter())
        .enumerate()
    {
        assert!(
            (lhs - rhs).abs() <= 1e-5,
            "upper mismatch at {}: batched={} regular={}",
            idx,
            lhs,
            rhs
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_batched_crown_concrete_soundness_recomputes_between_checkpoints() {
    use crate::layers::ReLULayer;

    let linear1 = LinearLayer::new(
        Array2::from_shape_vec((2, 2), vec![0.45, -0.22, 0.18, 0.34]).unwrap(),
        Some(Array1::from_vec(vec![0.02, -0.04])),
    )
    .unwrap();
    let linear2 = LinearLayer::new(
        Array2::from_shape_vec((2, 2), vec![0.28, 0.14, -0.19, 0.31]).unwrap(),
        Some(Array1::from_vec(vec![-0.01, 0.03])),
    )
    .unwrap();
    let linear3 = LinearLayer::new(
        Array2::from_shape_vec((2, 2), vec![-0.24, 0.27, 0.21, 0.16]).unwrap(),
        Some(Array1::from_vec(vec![0.05, -0.02])),
    )
    .unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear3));

    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![-1.0, -0.8]).unwrap();
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.9, 1.1]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let config = StreamingConfig {
        checkpoint_interval: 2,
        ..Default::default()
    };
    assert_batched_streaming_crown_contains_concrete_outputs(&network, &input, 10, config);
}

#[ntest::timeout(10000)]
#[test]
fn test_streaming_batched_crown_unsupported_logsumexp_falls_back_to_regular() {
    let mut network = Network::new();
    network.add_layer(Layer::LogSumExp(LogSumExpLayer::new(vec![0], false)));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 3]),
            vec![-1.0, -0.5, 0.0, 0.25, 0.75, 1.5],
        )
        .expect("shape should be valid"),
        ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 3]),
            vec![0.0, 0.5, 1.0, 1.25, 1.75, 2.5],
        )
        .expect("shape should be valid"),
    )
    .unwrap();
    let verifier = StreamingVerifier::new(StreamingConfig::default());

    let regular = verifier
        .propagate_crown_streaming(&network, &input)
        .unwrap();
    let batched = verifier
        .propagate_crown_batched_streaming(&network, &input)
        .unwrap();

    assert_eq!(batched.shape(), regular.shape());
    assert_eq!(batched.lower(), regular.lower());
    assert_eq!(batched.upper(), regular.upper());
}

/// Parity test: batched vs non-batched streaming CROWN for networks with
/// element-wise activations. Each is sandwiched between linear layers.
///
/// Part of #1708: verifies that the generic `crown_elementwise_backward_batched`
/// produces equivalent bounds to the non-batched `crown_elementwise_backward`.
#[ntest::timeout(10000)]
#[test]
fn test_streaming_batched_crown_elementwise_activations() {
    use crate::layers::{
        AbsLayer, ArctanLayer, CeilLayer, CeluLayer, ClipLayer, CosLayer, EluLayer, ExpLayer,
        FloorLayer, HardSigmoidLayer, HardSwishLayer, LeakyReLULayer, LogLayer, MishLayer,
        PReluLayer, PowConstantLayer, ReciprocalLayer, RoundLayer, SeluLayer, ShrinkLayer,
        SiLULayer, SigmoidLayer, SignLayer, SinLayer, SoftplusLayer, SoftsignLayer, SqrtLayer,
        TanLayer, TanhLayer, ThresholdedReluLayer,
    };

    let verifier = StreamingVerifier::new(StreamingConfig::default());

    // Helper: build Linear -> Activation -> Linear network and compare batched vs regular
    let test_activation = |name: &str, activation: Layer| {
        let mut network = Network::new();
        // Use small values so exp/log/sqrt/reciprocal are in valid range
        let weight1 = Array2::from_shape_fn((4, 4), |(i, j)| {
            let phase = (i * 7 + j * 3) as f32;
            0.05 * phase.sin()
        });
        let weight2 = Array2::from_shape_fn((4, 4), |(i, j)| {
            let phase = (i * 5 + j * 11) as f32;
            0.05 * phase.cos()
        });
        network.add_layer(Layer::Linear(
            LinearLayer::new(weight1, Some(Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]))).unwrap(),
        ));
        network.add_layer(activation);
        network.add_layer(Layer::Linear(
            LinearLayer::new(weight2, Some(Array1::<f32>::zeros(4))).unwrap(),
        ));

        // Input in [0.1, 0.9] to keep exp/log/sqrt/reciprocal in valid domain
        let lower = ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.1_f32);
        let upper = ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.9_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let batched = verifier
            .propagate_crown_batched_streaming(&network, &input)
            .unwrap();
        let regular = verifier
            .propagate_crown_streaming(&network, &input)
            .unwrap();

        assert_eq!(batched.shape(), regular.shape(), "{}: shape mismatch", name);

        // Tolerance: absolute 1e-3 or relative 1e-2 (whichever is larger).
        // The batched and non-batched paths may have slightly different
        // floating-point operation ordering due to array reshaping, causing
        // small precision differences especially for activations with steep
        // slopes (Sqrt near 0, Reciprocal near 0).
        let tol = |a: f32, b: f32| 1e-3_f32.max(1e-2 * a.abs().max(b.abs()));

        for (idx, (b, r)) in batched
            .lower()
            .iter()
            .zip(regular.lower().iter())
            .enumerate()
        {
            assert!(
                (b - r).abs() <= tol(*b, *r),
                "{}: lower mismatch at {}: batched={} regular={} (tol={})",
                name,
                idx,
                b,
                r,
                tol(*b, *r)
            );
        }
        for (idx, (b, r)) in batched
            .upper()
            .iter()
            .zip(regular.upper().iter())
            .enumerate()
        {
            assert!(
                (b - r).abs() <= tol(*b, *r),
                "{}: upper mismatch at {}: batched={} regular={} (tol={})",
                name,
                idx,
                b,
                r,
                tol(*b, *r)
            );
        }
    };

    test_activation("SiLU", Layer::SiLU(SiLULayer::new()));
    test_activation("Tanh", Layer::Tanh(TanhLayer::new()));
    test_activation("Sigmoid", Layer::Sigmoid(SigmoidLayer::new()));
    test_activation("Exp", Layer::Exp(ExpLayer::new()));
    test_activation("Log", Layer::Log(LogLayer::new()));
    test_activation("Sqrt", Layer::Sqrt(SqrtLayer::new()));
    test_activation("Reciprocal", Layer::Reciprocal(ReciprocalLayer::new()));
    test_activation("Softplus", Layer::Softplus(SoftplusLayer::new()));
    test_activation("HardSwish", Layer::HardSwish(HardSwishLayer::new()));
    test_activation("Mish", Layer::Mish(MishLayer::new()));
    test_activation("Selu", Layer::Selu(SeluLayer::new()));
    test_activation("Softsign", Layer::Softsign(SoftsignLayer::new()));
    test_activation(
        "Snake",
        Layer::Snake(SnakeLayer::new(2.0).expect("test: valid Snake")),
    );
    test_activation("Arctan", Layer::Arctan(ArctanLayer::new()));
    test_activation("Tan", Layer::Tan(TanLayer::new()));
    test_activation("Elu", Layer::Elu(EluLayer::new(1.0)));
    test_activation("Celu", Layer::Celu(CeluLayer::new(1.0)));
    test_activation("Sin", Layer::Sin(SinLayer::new()));
    test_activation("Cos", Layer::Cos(CosLayer::new()));
    test_activation("LeakyRelu", Layer::LeakyReLU(LeakyReLULayer::new(0.01)));
    test_activation(
        "HardSigmoid",
        Layer::HardSigmoid(HardSigmoidLayer::default_params()),
    );
    test_activation("Clip", Layer::Clip(ClipLayer::new(-1.0, 1.0)));
    test_activation(
        "ThresholdedRelu",
        Layer::ThresholdedRelu(ThresholdedReluLayer::new(0.4)),
    );
    test_activation("Abs", Layer::Abs(AbsLayer));
    test_activation(
        "PowConstant",
        Layer::PowConstant(PowConstantLayer::new(2.0)),
    );
    test_activation("Floor", Layer::Floor(FloorLayer::new()));
    test_activation("Ceil", Layer::Ceil(CeilLayer::new()));
    test_activation("Round", Layer::Round(RoundLayer::new()));
    test_activation("Sign", Layer::Sign(SignLayer::new()));
    test_activation("PRelu", Layer::PRelu(PReluLayer::from_scalar(0.25)));
    test_activation("Shrink", Layer::Shrink(ShrinkLayer::new(0.0, 0.5)));
}
