// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! LayerNorm and ConvTranspose2d CROWN propagation tests for GraphNetwork.

use crate::tests::crown::helpers::CountingGemmEngine;
use crate::*;
use ndarray::{arr1, Array1, ArrayD, IxDyn};
use ny_core::NyError;

fn assert_exact_crown_ibp_fallback_to_ibp(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_name: &str,
    layer_type: &str,
    detail_substring: &str,
    context: &str,
    expected_fallback_count: usize,
) {
    let ibp = graph
        .propagate_ibp(input)
        .unwrap_or_else(|err| panic!("{context}: IBP should succeed: {err}"));
    let with_status = graph
        .collect_crown_ibp_bounds_dag_with_status(input)
        .unwrap_or_else(|err| panic!("{context}: CROWN-IBP should fallback, not fail: {err}"));

    assert!(with_status.has_fallbacks(), "{context}: expected fallback");
    assert_eq!(
        with_status.fallback_count(),
        expected_fallback_count,
        "{context}: fallback count"
    );
    assert_eq!(
        with_status.provenance_for_node(node_name),
        Some(BoundsProvenance::ForwardFallback(
            CrownIbpFallbackReason::CrownPropagationError
        )),
        "{context}: provenance must report forward fallback"
    );

    let fallback_bounds = with_status
        .bounds
        .get(node_name)
        .unwrap_or_else(|| panic!("{context}: missing {node_name} bounds"));
    assert_eq!(
        fallback_bounds.shape(),
        ibp.shape(),
        "{context}: shape mismatch"
    );
    assert_eq!(
        fallback_bounds.lower(),
        ibp.lower(),
        "{context}: lower bounds diverged from exact IBP fallback"
    );
    assert_eq!(
        fallback_bounds.upper(),
        ibp.upper(),
        "{context}: upper bounds diverged from exact IBP fallback"
    );

    let event = with_status
        .fallback_events
        .first()
        .unwrap_or_else(|| panic!("{context}: missing fallback event"));
    assert_eq!(event.layer_type, layer_type, "{context}: layer type");
    assert_eq!(
        event.reason,
        CrownIbpFallbackReason::CrownPropagationError,
        "{context}: fallback reason"
    );
    assert!(
        event.details.contains(detail_substring),
        "{context}: expected detail substring `{detail_substring}`, got `{}`",
        event.details
    );
}

/// Assert CROWN bounds are at least as tight as IBP bounds (element-wise).
fn assert_crown_at_least_as_tight_as_ibp(crown: &BoundedTensor, ibp: &BoundedTensor, tol: f32) {
    assert_eq!(crown.shape(), ibp.shape(), "shape mismatch");
    for (i, (cl, il)) in crown.lower().iter().zip(ibp.lower().iter()).enumerate() {
        assert!(
            *cl >= *il - tol,
            "CROWN lower [{i}] ({cl}) < IBP lower ({il})"
        );
    }
    for (i, (cu, iu)) in crown.upper().iter().zip(ibp.upper().iter()).enumerate() {
        assert!(
            *cu <= *iu + tol,
            "CROWN upper [{i}] ({cu}) > IBP upper ({iu})"
        );
    }
}

/// Sample sqrt(x) over [lower_clamped, upper] and verify containment in bounds.
fn assert_sqrt_soundness_by_sampling(
    bounds: &BoundedTensor,
    lower_clamped: &[f32],
    upper_vals: &[f32],
) {
    for k in 0..=10 {
        let t = k as f32 / 10.0;
        for dim in 0..lower_clamped.len() {
            let x = lower_clamped[dim] + t * (upper_vals[dim] - lower_clamped[dim]);
            let y = x.sqrt();
            assert!(
                y >= bounds.lower()[[dim]] - 1e-5,
                "sqrt({x}) = {y} < CROWN lower {} at dim {dim}",
                bounds.lower()[[dim]]
            );
            assert!(
                y <= bounds.upper()[[dim]] + 1e-5,
                "sqrt({x}) = {y} > CROWN upper {} at dim {dim}",
                bounds.upper()[[dim]]
            );
        }
    }
}

fn build_downstream_sqrt_fallback_graph() -> (GraphNetwork, BoundedTensor) {
    use ndarray::arr2;

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("sqrt", Layer::Sqrt(SqrtLayer::new())));
    graph.add_node(GraphNode::new(
        "post",
        Layer::Linear(
            LinearLayer::new(arr2(&[[2.0_f32]]), Some(arr1(&[0.5_f32]))).expect("valid linear"),
        ),
        vec!["sqrt".to_string()],
    ));
    graph.set_output("post");

    let input = BoundedTensor::new(arr1(&[-0.25_f32]).into_dyn(), arr1(&[4.0_f32]).into_dyn())
        .expect("valid bounded input");
    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_with_layernorm() {
    // Test CROWN propagation through GraphNetwork with LayerNorm (using sampling mode)
    use crate::layers::LayerNormCrownMode;
    use ndarray::arr2;

    let mut graph = GraphNetwork::new();

    // Create: Linear -> LayerNorm
    let w = arr2(&[[1.0_f32, -0.5, 0.3], [0.5, 1.0, -0.2], [-0.3, 0.2, 1.0]]);
    let linear = LinearLayer::new(w, Some(arr1(&[0.1, -0.1, 0.0]))).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));

    let ny = arr1(&[1.0_f32, 1.0, 1.0]);
    let beta = arr1(&[0.0_f32, 0.0, 0.0]);
    let ln = LayerNormLayer::new(ny, beta, 1e-5).unwrap();
    graph.add_node(GraphNode::new(
        "layernorm",
        Layer::LayerNorm(ln.clone()),
        vec!["linear".to_string()],
    ));

    graph.set_output("layernorm");

    // Set all LayerNorm nodes to sampling mode for this test
    graph.set_layernorm_crown_mode(LayerNormCrownMode::Sampling);

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();

    // Get CROWN and IBP bounds
    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify soundness by sampling
    let linear_node = graph.nodes.get("linear").unwrap();
    let linear_layer = match &linear_node.layer {
        Layer::Linear(l) => l,
        _ => panic!("Expected Linear"),
    };

    for i in 0..50 {
        let t0 = (i * 17 % 50) as f32 / 50.0;
        let t1 = (i * 31 % 50) as f32 / 50.0;
        let t2 = (i * 47 % 50) as f32 / 50.0;

        let x_sample = arr1(&[-0.5 + t0, -0.5 + t1, -0.5 + t2]);

        // Forward through Linear
        let linear_out: Array1<f32> =
            linear_layer.weight.dot(&x_sample) + linear_layer.bias.as_ref().unwrap();

        // Forward through LayerNorm
        let ln_out = ln.eval(&linear_out).unwrap();

        // Check bounds
        for j in 0..3 {
            assert!(
                ln_out[j] >= crown_bounds.lower()[[j]] - 1e-3,
                "Sample {} output {} = {} < CROWN lower bound {} at dim {}",
                i,
                ln_out[j],
                ln_out[j],
                crown_bounds.lower()[[j]],
                j
            );
            assert!(
                ln_out[j] <= crown_bounds.upper()[[j]] + 1e-3,
                "Sample {} output {} = {} > CROWN upper bound {} at dim {}",
                i,
                ln_out[j],
                ln_out[j],
                crown_bounds.upper()[[j]],
                j
            );
        }
    }

    // CROWN should not be much worse than IBP
    let crown_width: f32 = (0..3)
        .map(|i| crown_bounds.upper()[[i]] - crown_bounds.lower()[[i]])
        .sum();
    let ibp_width: f32 = (0..3)
        .map(|i| ibp_bounds.upper()[[i]] - ibp_bounds.lower()[[i]])
        .sum();

    println!(
        "GraphNetwork LayerNorm: IBP width = {}, CROWN width = {}",
        ibp_width, crown_width
    );

    // Allow some tolerance - CROWN might not always be tighter for LayerNorm
    assert!(
        crown_width <= ibp_width * 2.0,
        "CROWN width {} should not be much worse than IBP width {}",
        crown_width,
        ibp_width
    );
}

fn build_soundness_refusal_graph() -> (GraphNetwork, BoundedTensor) {
    use crate::layers::LayerNormCrownMode;
    use ndarray::arr2;

    let hidden = 4;
    let mut network = Network::new();
    let weight1 = arr2(&[[0.3, -0.2], [-0.1, 0.4], [0.2, 0.1], [0.0, -0.3]]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(weight1, Some(arr1(&[0.1, -0.1, 0.0, 0.05]))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::LayerNorm(
        LayerNormLayer::new_default(hidden, 1e-5).unwrap(),
    ));
    let weight2 = arr2(&[[0.2, 0.3, -0.1, 0.4], [-0.4, 0.1, 0.5, -0.2]]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(weight2, Some(arr1(&[0.0, 0.05]))).unwrap(),
    ));

    let mut graph = GraphNetwork::from_sequential(&network).expect("sequential graph conversion");
    assert_eq!(
        graph.set_layernorm_crown_mode(LayerNormCrownMode::Sound),
        1,
        "expected exactly one LayerNorm node"
    );

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .expect("valid bounded input");
    (graph, input)
}

/// After #4113, Sqrt CROWN backward handles negative pre-activation lower bounds
/// by clamping the relaxation domain to [0, u] instead of returning
/// UnsupportedConfiguration. This produces tighter CROWN-IBP bounds than a
/// wholesale IBP fallback.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_ibp_sqrt_unsupported_configuration_falls_back_to_exact_ibp_3499() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("sqrt", Layer::Sqrt(SqrtLayer::new())));
    graph.set_output("sqrt");

    // Input includes negative lower bound -- previously caused fallback, now
    // handled by domain clamping inside the Sqrt relaxation (#4113).
    let input = BoundedTensor::new(
        arr1(&[-0.25_f32, 0.0, 4.0]).into_dyn(),
        arr1(&[0.25_f32, 1.0, 9.0]).into_dyn(),
    )
    .expect("valid bounded input");

    let ibp = graph
        .propagate_ibp(&input)
        .expect("IBP should succeed for Sqrt with clamped negative domain");
    let with_status = graph
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .expect("CROWN-IBP should succeed without fallback after #4113");

    // No fallback -- Sqrt CROWN backward now proceeds with clamped domain.
    assert!(
        !with_status.has_fallbacks(),
        "#3499/#4113: expected no fallback, got {}: {:?}",
        with_status.fallback_count(),
        with_status.fallback_events
    );
    assert_eq!(
        with_status.provenance_for_node("sqrt"),
        Some(BoundsProvenance::Crown),
        "#3499/#4113: sqrt node should have Crown provenance"
    );

    let crown_bounds = with_status.bounds.get("sqrt").expect("missing sqrt bounds");
    assert_crown_at_least_as_tight_as_ibp(crown_bounds, &ibp, 1e-6);
    assert_sqrt_soundness_by_sampling(crown_bounds, &[0.0, 0.0, 4.0], &[0.25, 1.0, 9.0]);
}

/// After #4113, Sqrt CROWN backward no longer falls back on negative
/// pre-activation lower bounds.  The downstream Linear output node receives
/// proper CROWN bounds.  The intermediate sqrt node itself gets a
/// `DemandDrivenSkip` provenance because no downstream nonlinear consumer
/// demands tightened pre-activation bounds at the sqrt producer (#3775).
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_ibp_downstream_sqrt_unsupported_configuration_falls_back_to_exact_ibp_3840() {
    let (graph, input) = build_downstream_sqrt_fallback_graph();
    let ibp = graph.propagate_ibp(&input).expect("IBP should succeed");
    let with_status = graph
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .expect("#3840/#4113: CROWN-IBP should succeed");

    // The sqrt node is an intermediate producer with only a linear consumer
    // downstream, so the demand-driven policy skips CROWN tightening for it.
    assert_eq!(
        with_status.provenance_for_node("sqrt"),
        Some(BoundsProvenance::ForwardFallback(
            CrownIbpFallbackReason::DemandDrivenSkip
        )),
        "#3840/#4113: sqrt provenance should be DemandDrivenSkip (no nonlinear consumer)"
    );
    // The output `post` node gets Crown provenance -- no propagation error.
    assert_eq!(
        with_status.provenance_for_node("post"),
        Some(BoundsProvenance::Crown),
        "#3840/#4113: post provenance should be Crown, not CrownPropagationError fallback"
    );

    // No CrownPropagationError fallback events. DemandDrivenSkip does not
    // record fallback events, so the events list should be empty.
    assert!(
        !with_status.has_fallbacks(),
        "#3840/#4113: expected 0 fallback events, got {}: {:?}",
        with_status.fallback_count(),
        with_status.fallback_events
    );

    let post_bounds = with_status.bounds.get("post").expect("missing post bounds");
    assert_crown_at_least_as_tight_as_ibp(post_bounds, &ibp, 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_ibp_layernorm_numerical_instability_falls_back_to_exact_ibp_3499() {
    use crate::layers::LayerNormCrownMode;

    let mut graph = GraphNetwork::new();
    let layernorm = LayerNormLayer::new(arr1(&[1e35_f32, 1e35, 1e35]), arr1(&[0.0, 0.0, 0.0]), 0.0)
        .expect("valid LayerNorm")
        .with_crown_mode(LayerNormCrownMode::Sampling);
    graph.add_node(GraphNode::from_input(
        "layernorm",
        Layer::LayerNorm(layernorm),
    ));
    graph.set_output("layernorm");
    assert_eq!(
        graph.set_layernorm_crown_mode(LayerNormCrownMode::Sampling),
        1,
        "expected exactly one LayerNorm node"
    );

    let input = BoundedTensor::new(
        arr1(&[5.0_f32, 5.0, 5.0]).into_dyn(),
        arr1(&[5.0_f32, 5.0, 5.0]).into_dyn(),
    )
    .expect("valid point input");

    assert_exact_crown_ibp_fallback_to_ibp(
        &graph,
        &input,
        "layernorm",
        "LayerNorm",
        "non-finite Jacobian or eval output",
        "#3499 LayerNorm NumericalInstability fallback",
        1,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_public_crown_soundness_refusal_does_not_retry_fixed_slope_3706() {
    let (graph, input) = build_soundness_refusal_graph();

    let alpha_engine = CountingGemmEngine::new();
    let alpha_error = graph
        .propagate_alpha_crown_with_config_and_engine(
            &input,
            &AlphaCrownConfig::default(),
            Some(&alpha_engine),
        )
        .expect_err("LayerNorm Sound mode should refuse graph alpha-CROWN");
    assert!(
        matches!(alpha_error, NyError::SoundnessRefusal(_)),
        "expected SoundnessRefusal from alpha-CROWN path, got {alpha_error}"
    );
    let alpha_calls = alpha_engine.gemm_calls();
    assert!(
        alpha_calls > 0,
        "alpha-CROWN path should exercise GEMM before hitting the LayerNorm refusal"
    );

    let public_engine = CountingGemmEngine::new();
    let public_error = graph
        .propagate_crown_with_engine(&input, Some(&public_engine))
        .expect_err("public graph CROWN path should propagate SoundnessRefusal");
    assert!(
        matches!(public_error, NyError::SoundnessRefusal(_)),
        "expected SoundnessRefusal from public graph CROWN path, got {public_error}"
    );
    assert_eq!(
        public_engine.gemm_calls(),
        alpha_calls,
        "public graph CROWN path retried fixed-slope CROWN after policy refusal"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_conv_transpose2d_crown_matches_ibp() {
    // GraphNetwork ConvTranspose2d should use CROWN path without falling back.
    let mut kernel = ArrayD::zeros(IxDyn(&[1, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = 1.0;
    kernel[[0, 0, 1, 0]] = 1.0;
    kernel[[0, 0, 1, 1]] = 1.0;
    let conv = ConvTranspose2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::ConvTranspose2d(conv)));
    graph.set_output("conv");

    let mut input_data = ArrayD::zeros(IxDyn(&[1, 2, 2]));
    input_data[[0, 0, 0]] = 1.0;
    input_data[[0, 0, 1]] = 2.0;
    input_data[[0, 1, 0]] = 3.0;
    input_data[[0, 1, 1]] = 4.0;
    let input = BoundedTensor::concrete(input_data).unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    assert_eq!(crown_bounds.shape(), ibp_bounds.shape());
    assert_eq!(crown_bounds.shape(), &[1, 3, 3]);

    let expected = [[1.0, 3.0, 2.0], [4.0, 10.0, 6.0], [3.0, 7.0, 4.0]];
    for (h, expected_row) in expected.iter().enumerate() {
        for (w, &value) in expected_row.iter().enumerate() {
            assert!(
                (ibp_bounds.lower()[[0, h, w]] - value).abs() < 1e-6,
                "ibp lower[{},{}] = {}",
                h,
                w,
                ibp_bounds.lower()[[0, h, w]]
            );
            assert!(
                (ibp_bounds.upper()[[0, h, w]] - value).abs() < 1e-6,
                "ibp upper[{},{}] = {}",
                h,
                w,
                ibp_bounds.upper()[[0, h, w]]
            );
            assert!(
                (crown_bounds.lower()[[0, h, w]] - value).abs() < 1e-6,
                "crown lower[{},{}] = {}",
                h,
                w,
                crown_bounds.lower()[[0, h, w]]
            );
            assert!(
                (crown_bounds.upper()[[0, h, w]] - value).abs() < 1e-6,
                "crown upper[{},{}] = {}",
                h,
                w,
                crown_bounds.upper()[[0, h, w]]
            );
        }
    }
}
