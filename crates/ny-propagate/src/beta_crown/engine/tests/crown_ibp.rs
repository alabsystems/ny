// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::prelude::*;
use crate::tests::crown::helpers::MockGpuCrownEngine;
use ny_test_utils::assert_bounded_tensor_close;

/// Helper to create a deeper network for CROWN-IBP testing
/// Linear 2 -> 4, ReLU, Linear 4 -> 4, ReLU, Linear 4 -> 1
fn deeper_network() -> Network {
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7], [-0.2, 0.8]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let w2 = arr2(&[
        [0.5, -0.3, 0.7, 0.1],
        [-0.4, 0.6, -0.2, 0.5],
        [0.3, 0.2, -0.5, 0.4],
        [-0.1, 0.4, 0.3, -0.6],
    ]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let w3 = arr2(&[[1.0, -0.5, 0.3, 0.2]]);
    let linear3 = LinearLayer::new(w3, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear3));
    network
}

#[ntest::timeout(5000)]
#[test]
fn test_joint_alpha_beta_conv2d_infers_spatial_dims_from_layout() {
    // Regression: Conv2d backward in joint α-β CROWN must not mis-infer (H,W)
    // for NHWC / HWC shaped tensors (common in TensorFlow-exported ONNX).
    let verifier = BetaCrownVerifier::default();

    // Conv2d: in_c=3, out_c=2, kernel=1x1, stride=1, pad=0.
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[2, 3, 1, 1]),
        vec![
            // out 0
            1.0, 0.0, 0.0, //
            // out 1
            0.0, 1.0, 0.0, //
        ],
    )
    .unwrap();
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();
    let layer = Layer::Conv2d(conv);

    let beta_state = BetaState::empty();
    let alpha_state = DomainAlphaState::empty();

    let in_c = 3usize;
    let in_h = 4usize;
    let in_w = 5usize;
    let out_c = 2usize;
    let conv_out_size = out_c * in_h * in_w;

    let mut lower_a = Array2::<f32>::zeros((1, conv_out_size));
    lower_a[[0, 0]] = 1.0;
    let output_bounds = LinearBounds {
        lower_a: lower_a.clone(),
        lower_b: Array1::zeros(1),
        upper_a: lower_a,
        upper_b: Array1::zeros(1),
        lower_a_err: None,
        upper_a_err: None,
    };

    let shapes: Vec<Vec<usize>> = vec![
        vec![in_c, in_h, in_w],    // CHW
        vec![in_h, in_w, in_c],    // HWC
        vec![1, in_c, in_h, in_w], // NCHW
        vec![1, in_h, in_w, in_c], // NHWC
    ];

    for shape in shapes {
        let zeros = ArrayD::<f32>::zeros(IxDyn(&shape));
        let pre_bounds = BoundedTensor::new(zeros.clone(), zeros).unwrap();

        let new_bounds = verifier
            .propagate_layer_backward_with_alpha_beta(
                &layer,
                &output_bounds,
                &pre_bounds,
                None,
                &beta_state,
                &alpha_state,
                None, // No arelu state in tests
                0,
                None, // No GPU engine in tests
            )
            .unwrap();

        assert_eq!(new_bounds.num_inputs(), in_c * in_h * in_w);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_beta_crown_trivial_verified() {
    // Network output is always positive for the given input range.
    let network = simple_network();

    // Use an input box that guarantees positive output to keep the test obvious.
    let input =
        BoundedTensor::new(arr1(&[2.0, 0.0]).into_dyn(), arr1(&[3.0, 1.0]).into_dyn()).unwrap();

    let verifier = BetaCrownVerifier::default();
    let result = verifier.verify(&network, &input, -10.0).unwrap();

    // Should verify since output bounds are [0, inf] and threshold is -10
    assert_eq!(result.result, BabVerificationStatus::Verified);
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_crown_needs_splitting() {
    let network = simple_network();

    // Input that creates unstable neurons
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 100,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, -5.0).unwrap();

    // Should verify (output >= 0 for this network)
    assert_eq!(result.result, BabVerificationStatus::Verified);
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_ibp_tighter_intermediate_bounds() {
    // Test that CROWN-IBP produces tighter intermediate bounds than IBP
    let network = deeper_network();

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Collect bounds with IBP
    let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();

    // Collect bounds with CROWN-IBP
    let crown_ibp_bounds = network.collect_crown_ibp_bounds(&input).unwrap();

    // CROWN-IBP bounds should be at least as tight (usually tighter) than IBP
    let mut crown_ibp_tighter = false;
    for (ibp, crown_ibp) in ibp_bounds.iter().zip(crown_ibp_bounds.iter()) {
        let ibp_width = ibp.max_width();
        let crown_ibp_width = crown_ibp.max_width();

        // CROWN-IBP should never be looser
        assert!(
            crown_ibp_width <= ibp_width + 1e-5,
            "CROWN-IBP should not be looser than IBP"
        );

        // Track if CROWN-IBP is tighter for any layer
        if crown_ibp_width < ibp_width - 1e-5 {
            crown_ibp_tighter = true;
        }
    }

    // For this network, CROWN-IBP should be tighter
    assert!(
        crown_ibp_tighter,
        "CROWN-IBP should be tighter than IBP for deeper networks"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_crown_with_crown_ibp() {
    // Test that β-CROWN works with CROWN-IBP enabled
    let network = deeper_network();

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Run with standard IBP bounds
    let config_ibp = BetaCrownConfig {
        max_domains: 100,
        timeout: Duration::from_secs(10),
        use_alpha_crown: false,
        use_crown_ibp: false,
        ..Default::default()
    };
    let verifier_ibp = BetaCrownVerifier::new(config_ibp);
    let result_ibp = verifier_ibp.verify(&network, &input, -5.0).unwrap();

    // Run with CROWN-IBP bounds
    let config_crown_ibp = BetaCrownConfig {
        max_domains: 100,
        timeout: Duration::from_secs(10),
        use_alpha_crown: false,
        use_crown_ibp: true,
        ..Default::default()
    };
    let verifier_crown_ibp = BetaCrownVerifier::new(config_crown_ibp);
    let result_crown_ibp = verifier_crown_ibp.verify(&network, &input, -5.0).unwrap();

    // Both should verify (property is easy to verify)
    assert_eq!(result_ibp.result, BabVerificationStatus::Verified);
    assert_eq!(result_crown_ibp.result, BabVerificationStatus::Verified);

    // CROWN-IBP should use fewer or equal domains (tighter bounds = less splitting)
    assert!(
        result_crown_ibp.domains_explored <= result_ibp.domains_explored,
        "CROWN-IBP should explore <= IBP domains (got CROWN-IBP={}, IBP={})",
        result_crown_ibp.domains_explored,
        result_ibp.domains_explored,
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_beta_crown_verify_with_engine_preserves_gpu_path_for_crown_ibp() {
    let network = deeper_network();
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();
    let expected = network
        .propagate_crown_with_engine(&input, Some(&NaiveCpuGemmEngine))
        .unwrap();
    let mock_gpu = MockGpuCrownEngine::from_expected(&expected);

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        max_domains: 8,
        timeout: Duration::from_secs(10),
        use_alpha_crown: false,
        use_crown_ibp: true,
        ..Default::default()
    });

    let initial_bounds = verifier
        .compute_initial_bounds_with_early_exit_engine(
            &network,
            &input,
            None,
            Some(&mock_gpu),
            None,
        )
        .unwrap();

    assert_eq!(initial_bounds.shape(), expected.shape());
    assert!(
        mock_gpu.gpu_calls() > 0,
        "use_crown_ibp root bounds should preserve the GPU CROWN fast-path"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_beta_crown_verify_with_engine_reuses_root_crown_ibp_collection() {
    let network = deeper_network();
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();
    let expected = network
        .propagate_crown_with_engine(&input, Some(&NaiveCpuGemmEngine))
        .unwrap();
    let threshold = f32::midpoint(expected.lower_scalar(), expected.upper_scalar());

    let baseline_gpu = MockGpuCrownEngine::from_expected(&expected);
    let baseline_layer_bounds = network
        .collect_crown_ibp_bounds_with_engine_and_deadline(&input, Some(&baseline_gpu), None)
        .unwrap();
    let _baseline_output = network
        .propagate_crown_with_layer_bounds_and_engine_and_deadline_and_limits(
            &input,
            &baseline_layer_bounds,
            Some(&baseline_gpu),
            None,
            None,
        )
        .unwrap();
    let baseline_calls = baseline_gpu.gpu_calls();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        max_domains: 0,
        timeout: Duration::from_secs(10),
        use_alpha_crown: false,
        use_crown_ibp: true,
        ..Default::default()
    });
    let mock_gpu = MockGpuCrownEngine::from_expected(&expected);

    let result = verifier
        .verify_with_engine(&network, &input, threshold, Some(&mock_gpu), None)
        .unwrap();

    assert!(
        matches!(result.result, BabVerificationStatus::Unknown { .. }),
        "max_domains=0 should stop after root setup once the property remains unresolved"
    );
    assert_eq!(
        mock_gpu.gpu_calls(),
        baseline_calls,
        "root verification should reuse one CROWN-IBP collection instead of rebuilding it"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_ibp_root_budget_gate_matches_plain_ibp_bounds_4244() {
    let network = deeper_network();
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    let budgeted = BetaCrownVerifier::new(BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: true,
        max_crown_ibp_nodes: Some(0),
        ..Default::default()
    })
    .compute_initial_bounds_and_layer_bounds_engine(&network, &input, None, None, None)
    .unwrap();

    let plain_ibp = BetaCrownVerifier::new(BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: false,
        ..Default::default()
    })
    .compute_initial_bounds_and_layer_bounds_engine(&network, &input, None, None, None)
    .unwrap();

    let budgeted_layer_bounds = budgeted
        .root_layer_bounds
        .expect("#4244 budgeted root path should still cache IBP layer bounds");
    let plain_layer_bounds = plain_ibp
        .root_layer_bounds
        .expect("#4244 plain IBP root path should cache layer bounds");

    assert_eq!(
        budgeted_layer_bounds.len(),
        plain_layer_bounds.len(),
        "#4244 root budget gate should preserve layer-bound count"
    );
    for (index, (actual, expected)) in budgeted_layer_bounds
        .iter()
        .zip(plain_layer_bounds.iter())
        .enumerate()
    {
        assert_bounded_tensor_close(actual, expected, 1e-6, &format!("#4244 root layer {index}"));
    }
    assert_bounded_tensor_close(
        &budgeted.output_bounds,
        &plain_ibp.output_bounds,
        1e-6,
        "#4244 root budget output",
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_input_split_child_preserves_gpu_path_for_crown_recomputation() {
    // The mock engine's fast GPU CROWN backward is UNSOUND by contract
    // (provides_sound_gpu_crown = false), and the process gate defaults to
    // sound-required — which masks the mock and makes `gpu_calls() == 0`
    // deterministic in isolation. Take the ONE shared gate lock, which
    // releases the gate for this scope (the documented pattern for tests that
    // exercise the fast path) and serializes against other gate-flipping
    // tests.
    let _gate = crate::sound_gpu_gate::test_lock::lock_gate();
    let network = deeper_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let parent = BabDomain::root_with_input(Vec::new(), 0.0, 0.0, &input).unwrap();
    let child_input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[0.0, 1.0]).into_dyn()).unwrap();
    let expected = network
        .propagate_crown_with_engine(&child_input, Some(&NaiveCpuGemmEngine))
        .unwrap();
    let mock_gpu = MockGpuCrownEngine::from_expected(&expected);

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: true,
        ..Default::default()
    });

    let child = verifier
        .create_input_split_child(
            &network,
            &input,
            &parent,
            0,
            -1.0,
            0.0,
            -100.0,
            None,
            Some(&mock_gpu),
        )
        .unwrap();

    let child = child.expect("expected a valid split child");
    assert!(
        (child.lower_bound - expected.lower_scalar()).abs() < 1e-6,
        "split child lower bound should match direct CROWN recomputation"
    );
    assert!(
        (child.upper_bound - expected.upper_scalar()).abs() < 1e-6,
        "split child upper bound should match direct CROWN recomputation"
    );
    assert!(
        mock_gpu.gpu_calls() > 0,
        "input-split child recomputation should preserve the GPU CROWN fast-path"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_input_split_child_budget_gate_matches_plain_ibp_bounds_4244() {
    let network = deeper_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let parent = BabDomain::root_with_input(Vec::new(), 0.0, 0.0, &input).unwrap();
    let child_input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[0.0, 1.0]).into_dyn()).unwrap();
    let expected_layer_bounds = network
        .collect_ibp_bounds_with_deadline(&child_input, None)
        .unwrap();
    let expected_output = network
        .propagate_crown_with_precomputed_ibp_and_limits(
            &child_input,
            expected_layer_bounds.clone(),
            None,
            None,
            None,
        )
        .unwrap();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: true,
        max_crown_ibp_nodes: Some(0),
        ..Default::default()
    });

    let child = verifier
        .create_input_split_child(&network, &input, &parent, 0, -1.0, 0.0, -100.0, None, None)
        .unwrap()
        .expect("#4244 expected a valid split child");

    assert_eq!(
        child.layer_bounds().len(),
        expected_layer_bounds.len(),
        "#4244 child budget gate should preserve layer-bound count"
    );
    for (index, (actual, expected)) in child
        .layer_bounds()
        .iter()
        .zip(expected_layer_bounds.iter())
        .enumerate()
    {
        assert_bounded_tensor_close(
            actual.as_ref(),
            expected,
            1e-6,
            &format!("#4244 child layer {index}"),
        );
    }
    assert!(
        (child.lower_bound() - expected_output.lower_scalar()).abs() < 1e-6,
        "#4244 child lower bound should reuse plain-IBP CROWN output"
    );
    assert!(
        (child.upper_bound() - expected_output.upper_scalar()).abs() < 1e-6,
        "#4244 child upper bound should reuse plain-IBP CROWN output"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_coefficient_computation() {
    // Test that CROWN coefficients are computed correctly
    let network = simple_network();

    // Create input bounds
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Get layer bounds
    let layer_bounds = network.collect_ibp_bounds(&input).unwrap();

    // Create root domain
    let domain = BabDomain::root(layer_bounds, 0.0, 1.0).unwrap();

    // Compute coefficients
    let verifier = BetaCrownVerifier::default();
    let coeffs = verifier
        .compute_crown_coefficients(&network, &domain)
        .unwrap();

    // ReLU is at layer 1 (2 neurons, both unstable over [-2,2]).
    // Verify we got coefficients for both neurons at the ReLU layer.
    let relu_coeffs: Vec<_> = coeffs
        .iter()
        .filter(|((layer, _), _)| *layer == 1)
        .collect();
    assert_eq!(
        relu_coeffs.len(),
        2,
        "simple_network has 2 unstable ReLU neurons at layer 1, got {} coefficients",
        relu_coeffs.len(),
    );

    // Branching importance scores are the sum of absolute backward coefficients
    // through the output layer (see compute_crown_coefficients). They are always
    // non-negative, and strictly positive for unstable neurons with a non-trivial
    // path to the output.
    for &((layer, neuron), coeff) in &relu_coeffs {
        assert!(
            *coeff > 0.0,
            "CROWN branching coefficient for layer {} neuron {} = {} should be positive \
             for an unstable neuron",
            layer,
            neuron,
            coeff,
        );
    }
}

/// #cgan-bn11-budget: the preset-configurable per-node CROWN-IBP time budget
/// on `BetaCrownConfig` is stamped onto every engine-configured graph clone,
/// so the collector's budget computation (`crown_tighten.rs`) sees the preset
/// value. Default config = all-None = the built-in #3499/#4413 constants.
#[test]
fn test_configured_graph_carries_crown_ibp_per_node_time_budget() {
    use crate::types::CrownIbpPerNodeTimeBudget;

    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(arr2(&[[1.0_f32, -0.5]]), None).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");

    // Default config: no overrides reach the graph (old constants).
    let default_verifier = BetaCrownVerifier::default();
    let configured = default_verifier.configured_graph_for_crown(&graph);
    assert_eq!(
        configured.crown_ibp_per_node_time_budget,
        CrownIbpPerNodeTimeBudget::default(),
    );

    // cgan_2023-shaped config: cap 150 s, floor unset.
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        crown_ibp_per_node_cap_secs: Some(150.0),
        ..Default::default()
    });
    let configured = verifier.configured_graph_for_crown(&graph);
    assert_eq!(
        configured.crown_ibp_per_node_time_budget,
        CrownIbpPerNodeTimeBudget {
            floor_secs: None,
            cap_secs: Some(150.0),
        },
    );
    // Clones (how BaB paths receive the graph) inherit the stamp.
    assert_eq!(
        configured.clone().crown_ibp_per_node_time_budget,
        configured.crown_ibp_per_node_time_budget,
    );
}
