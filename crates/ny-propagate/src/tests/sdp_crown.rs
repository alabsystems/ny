// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::network::{GraphNetwork, GraphNode};
use crate::types::{PropagationConfig, PropagationMethod};
use crate::verifier::Verifier;
use ndarray::{arr1, arr2};
use ny_core::{Bound, NyError, VerificationSpec};

/// Regression test for #2584 item 3: near-zero-width crossing ReLU intervals
/// in the SDP-CROWN ReLU backward path must remain finite and sound.
#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_crown_near_zero_width_crossing_soundness() {
    let relu = ReLULayer::new();
    let epsilon = 1e-12_f32;
    let pre_activation =
        BoundedTensor::new(arr1(&[-epsilon]).into_dyn(), arr1(&[epsilon]).into_dyn()).unwrap();
    let incoming = LinearBounds::identity(1);
    let x_hat = arr1(&[0.0_f32]);
    let rho = 1.0_f32;

    let backward = relu
        .propagate_linear_with_bounds_sdp(&incoming, &pre_activation, &x_hat, rho)
        .expect("SDP-CROWN ReLU backward should succeed on near-zero crossing interval");

    assert!(
        backward.lower_a.iter().all(|v| v.is_finite()),
        "lower_a contains non-finite values"
    );
    assert!(
        backward.upper_a.iter().all(|v| v.is_finite()),
        "upper_a contains non-finite values"
    );
    assert!(
        backward.lower_b.iter().all(|v| v.is_finite()),
        "lower_b contains non-finite values"
    );
    assert!(
        backward.upper_b.iter().all(|v| v.is_finite()),
        "upper_b contains non-finite values"
    );

    // Check pointwise soundness at the endpoints and kink.
    let tol = 1e-12_f32;
    for x in [-epsilon, 0.0_f32, epsilon] {
        let point = BoundedTensor::new(arr1(&[x]).into_dyn(), arr1(&[x]).into_dyn()).unwrap();
        let concrete = backward.concretize(&point);
        let y = x.max(0.0);
        let lower = concrete.lower()[[0]];
        let upper = concrete.upper()[[0]];

        assert!(
            lower <= y + tol,
            "SDP-CROWN lower unsound at x={x}: lower={lower} > ReLU(x)={y}"
        );
        assert!(
            upper >= y - tol,
            "SDP-CROWN upper unsound at x={x}: upper={upper} < ReLU(x)={y}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sdp_crown_relu_offset_zero_center_matches_closed_form() {
    // Example from SDP-CROWN paper (Figure 1):
    // f(x) = -ReLU(x1) - ReLU(x2) on B2(0, 1).
    //
    // Standard LiRPA/CROWN on the enclosing box produces g = [-0.5, -0.5] and offset -1.
    // SDP-CROWN improves the offset to -sqrt(0.5) ≈ -0.7071 (sqrt(2) tighter).
    let c = [-1.0f32, -1.0f32];
    let g = [-0.5f32, -0.5f32];
    let x_hat = [0.0f32, 0.0f32];
    let rho = 1.0f32;

    let h = crate::sdp_crown::relu_sdp_offset_opt(&c, &g, &x_hat, rho).unwrap();
    let expected = -(0.5f32).sqrt();
    assert!(
        (h - expected).abs() < 1e-4,
        "h={}, expected={}",
        h,
        expected
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_network_sdp_crown_matches_paper_example_bounds() {
    // Network: y = [-1, -1] · ReLU(x), input set x ∈ B2(0, 1).
    //
    // True range:
    // - min occurs at x = (1/sqrt(2), 1/sqrt(2)) => y = -sqrt(2)
    // - max occurs for x <= 0 => y = 0
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));

    let w = arr2(&[[-1.0, -1.0]]);
    let linear = LinearLayer::new(w, None).unwrap();
    net.add_layer(Layer::Linear(linear));

    // Provide the enclosing box x_hat ± rho for IBP slope selection.
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let x_hat = arr1(&[0.0, 0.0]);
    let rho = 1.0f32;

    let out = net.propagate_sdp_crown(&input, &x_hat, rho).unwrap();
    let lo = out.lower().as_slice().unwrap()[0];
    let up = out.upper().as_slice().unwrap()[0];

    let expected_lo = -(2.0f32).sqrt();
    let expected_up = 0.0f32;
    assert!(
        (lo - expected_lo).abs() < 1e-3,
        "lo={}, expected={}",
        lo,
        expected_lo
    );
    assert!(
        (up - expected_up).abs() < 1e-6,
        "up={}, expected={}",
        up,
        expected_up
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_network_sdp_crown_rejects_non_enclosing_box() {
    let mut net = Network::new();
    let w = arr2(&[[1.0, 0.0]]);
    let linear = LinearLayer::new(w, None).unwrap();
    net.add_layer(Layer::Linear(linear));

    // Box does NOT enclose x_hat ± rho (rho=2.0 but bounds are [0, 1]).
    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let x_hat = arr1(&[0.0, 0.0]);
    let rho = 2.0f32;

    let result = net.propagate_sdp_crown(&input, &x_hat, rho);
    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_network_sdp_crown_rejects_nonfinite_xhat() {
    let mut net = Network::new();
    net.add_layer(Layer::ReLU(ReLULayer));

    let w = arr2(&[[1.0, 0.0]]);
    let linear = LinearLayer::new(w, None).unwrap();
    net.add_layer(Layer::Linear(linear));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let x_hat = arr1(&[f32::NAN, 0.0]);
    let rho = 1.0f32;

    let result = net.propagate_sdp_crown(&input, &x_hat, rho);
    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_try_to_sequential_linear_relu_succeeds() {
    // Build a sequential Linear -> ReLU -> Linear graph
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0, 0.5], [0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear1".to_string(),
        layer: Layer::Linear(linear1),
        inputs: vec!["_input".to_string()],
    });

    graph.add_node(GraphNode {
        name: "relu".to_string(),
        layer: Layer::ReLU(ReLULayer),
        inputs: vec!["linear1".to_string()],
    });

    let w2 = arr2(&[[1.0, -1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear2".to_string(),
        layer: Layer::Linear(linear2),
        inputs: vec!["relu".to_string()],
    });

    graph.set_output("linear2");

    // Should successfully convert to Network
    let network = graph.try_to_sequential_network();
    assert!(
        network.is_some(),
        "Sequential Linear/ReLU graph should convert to Network"
    );

    let net = network.unwrap();
    assert_eq!(net.layers.len(), 3);
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_try_to_sequential_with_gelu_fails() {
    // Build a sequential Linear -> GELU graph (GELU not supported for SDP-CROWN)
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0, 0.5], [0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear1".to_string(),
        layer: Layer::Linear(linear1),
        inputs: vec!["_input".to_string()],
    });

    graph.add_node(GraphNode {
        name: "gelu".to_string(),
        layer: Layer::GELU(GELULayer::default()),
        inputs: vec!["linear1".to_string()],
    });

    graph.set_output("gelu");

    // Should return None since GELU is not supported
    let network = graph.try_to_sequential_network();
    assert!(
        network.is_none(),
        "Graph with GELU should not convert to Network for SDP-CROWN"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_try_to_sequential_with_branch_fails() {
    // Build a graph with a branch (Add layer needs two inputs)
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0], [0.5]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear1".to_string(),
        layer: Layer::Linear(linear1),
        inputs: vec!["_input".to_string()],
    });

    let w2 = arr2(&[[0.5], [1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear2".to_string(),
        layer: Layer::Linear(linear2),
        inputs: vec!["_input".to_string()],
    });

    graph.add_node(GraphNode {
        name: "add".to_string(),
        layer: Layer::Add(AddLayer),
        inputs: vec!["linear1".to_string(), "linear2".to_string()],
    });

    graph.set_output("add");

    // Should return None due to branch (Add is binary)
    let network = graph.try_to_sequential_network();
    assert!(
        network.is_none(),
        "Graph with branch should not convert to Network"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graphnetwork_sdp_crown_via_verifier_rejects_box_spec() {
    // Build a sequential Linear -> ReLU -> Linear graph
    let mut graph = GraphNetwork::new();
    // Input: 2D, Output after Linear1: 2D
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear1".to_string(),
        layer: Layer::Linear(linear1),
        inputs: vec!["_input".to_string()],
    });

    graph.add_node(GraphNode {
        name: "relu".to_string(),
        layer: Layer::ReLU(ReLULayer),
        inputs: vec!["linear1".to_string()],
    });

    // Output: 1D
    let w2 = arr2(&[[-1.0, -1.0]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode {
        name: "linear2".to_string(),
        layer: Layer::Linear(linear2),
        inputs: vec!["relu".to_string()],
    });

    graph.set_output("linear2");

    // A VerificationSpec can only declare per-element ℓ∞ bounds. SDP-CROWN's
    // bounds hold over an ℓ2 ball, and no ball soundly answers a box question:
    // over the box [-1,1]^2 the true min of y = -ReLU(x1) - ReLU(x2) is -2
    // (at the corner (1,1)), while the inscribed ball B2(0, 1) only reaches
    // -sqrt(2). The verifier must therefore refuse box specs outright.
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new(-10.0, 10.0)],
        Some(5000),
        Some(vec![2]),
    )
    .expect("valid test spec");

    let config = PropagationConfig {
        method: PropagationMethod::SdpCrown,
        max_iterations: 100,
        tolerance: 1e-4,
        use_gpu: false,
        ..Default::default()
    };
    let verifier = Verifier::new(config);

    let result = verifier.verify_graph(&graph, &spec);
    assert!(
        matches!(result, Err(NyError::UnsupportedOp(_))),
        "SDP-CROWN must reject ℓ∞ box specs, got {result:?}"
    );
}

// ============== Non-finite c/g rejection tests (issue #2752) ==============
//
// These verify that relu_sdp_offset_for_lambda and relu_sdp_offset_opt reject
// non-finite c/g coefficients instead of silently swallowing NaN via f64::min().

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_nan_c() {
    let c = [f32::NAN, 0.5];
    let g = [0.3, 0.4];
    let x_hat = [0.0, 0.0];
    let result = crate::sdp_crown::relu_sdp_offset_for_lambda(&c, &g, &x_hat, 0.1, 1.0);
    assert!(result.is_err(), "NaN in c must be rejected");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_nan_g() {
    let c = [0.5, 0.3];
    let g = [f32::NAN, 0.4];
    let x_hat = [0.0, 0.0];
    let result = crate::sdp_crown::relu_sdp_offset_for_lambda(&c, &g, &x_hat, 0.1, 1.0);
    assert!(result.is_err(), "NaN in g must be rejected");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_inf_c() {
    let c = [f32::INFINITY, 0.5];
    let g = [0.3, 0.4];
    let x_hat = [0.0, 0.0];
    let result = crate::sdp_crown::relu_sdp_offset_for_lambda(&c, &g, &x_hat, 0.1, 1.0);
    assert!(result.is_err(), "Inf in c must be rejected");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_for_lambda_inf_g() {
    let c = [0.5, 0.3];
    let g = [f32::NEG_INFINITY, 0.4];
    let x_hat = [0.0, 0.0];
    let result = crate::sdp_crown::relu_sdp_offset_for_lambda(&c, &g, &x_hat, 0.1, 1.0);
    assert!(result.is_err(), "Inf in g must be rejected");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_nan_c() {
    let c = [f32::NAN];
    let g = [0.5];
    let x_hat = [0.0];
    let result = crate::sdp_crown::relu_sdp_offset_opt(&c, &g, &x_hat, 0.1);
    assert!(result.is_err(), "NaN in c must be rejected by opt path");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_nan_g() {
    let c = [0.5];
    let g = [f32::NAN];
    let x_hat = [0.0];
    let result = crate::sdp_crown::relu_sdp_offset_opt(&c, &g, &x_hat, 0.1);
    assert!(result.is_err(), "NaN in g must be rejected by opt path");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_inf_c() {
    let c = [f32::INFINITY];
    let g = [0.5];
    let x_hat = [0.0];
    let result = crate::sdp_crown::relu_sdp_offset_opt(&c, &g, &x_hat, 1.0);
    assert!(result.is_err(), "Inf in c must be rejected by opt path");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_inf_g() {
    let c = [0.5];
    let g = [f32::NEG_INFINITY];
    let x_hat = [0.0];
    let result = crate::sdp_crown::relu_sdp_offset_opt(&c, &g, &x_hat, 1.0);
    assert!(result.is_err(), "Inf in g must be rejected by opt path");
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_nan_c_rho_zero_branch() {
    // Ensure the rho=0 early-return path also rejects NaN c
    let c = [f32::NAN];
    let g = [0.5];
    let x_hat = [1.0];
    let result = crate::sdp_crown::relu_sdp_offset_opt(&c, &g, &x_hat, 0.0);
    assert!(
        result.is_err(),
        "NaN in c must be rejected even on rho=0 path"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_sdp_offset_opt_nan_g_xhat_zero_branch() {
    // Ensure the x_hat≈0 closed-form path also rejects NaN g
    let c = [0.5];
    let g = [f32::NAN];
    let x_hat = [0.0];
    let result = crate::sdp_crown::relu_sdp_offset_opt(&c, &g, &x_hat, 1.0);
    assert!(
        result.is_err(),
        "NaN in g must be rejected even on x_hat=0 path"
    );
}
