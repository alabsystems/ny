// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for MulBinary CROWN backward through graph BaB.
//!
//! Verifies that MulBinary (element-wise multiply) nodes are handled correctly
//! in the graph input-split BaB path via canonical dispatch → McCormick
//! envelopes. Part of #3439.

use super::prelude::*;
use crate::layers::binary_ops::MulBinaryLayer;
use crate::layers::SiLULayer;

/// SwiGLU-like graph: x -> SiLU(gate=x) ; x -> up=x ; Mul(SiLU(gate), up) -> sum.
///
/// gate and up are both functions of the same network input, so the true bound
/// is tight, but IBP/CROWN decorrelate the multiplicative node. GenBaB must split
/// the MulBinary node to close the McCormick gap.
fn swiglu_mul_graph() -> GraphNetwork {
    // gate branch: identity Linear from input -> SiLU
    let w_gate = arr2(&[[1.0_f32]]);
    let linear_gate = LinearLayer::new(w_gate, None).expect("linear_gate");
    // up branch: identity Linear from input
    let w_up = arr2(&[[1.0_f32]]);
    let linear_up = LinearLayer::new(w_up, None).expect("linear_up");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "gate_lin",
        Layer::Linear(linear_gate),
    ));
    graph.add_node(GraphNode::new(
        "silu",
        Layer::SiLU(SiLULayer::new()),
        vec!["gate_lin".to_string()],
    ));
    graph.add_node(GraphNode::from_input("up", Layer::Linear(linear_up)));
    graph.add_node(GraphNode::binary(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        "silu",
        "up",
    ));
    graph.set_output("mul");
    graph
}

/// GenBaB BaB must descend past depth 0 on a SwiGLU-like `Mul(SiLU(x), x)` graph
/// and verify sub-domains — the McCormick gap that IBP/CROWN leaves at the root is
/// closed by splitting the MulBinary node's input intervals.
///
/// Regression for the GenBaB MulBinary blocker (#mul-genbab): GenBaB splits the
/// MulBinary node, the split constraint tightens the correct input's
/// pre-activation node, and child propagation succeeds so BaB can descend.
#[ntest::timeout(30000)]
#[test]
fn test_genbab_mul_binary_swiglu_descends_past_depth0() {
    use crate::beta_crown::nonlinear_branching::NonlinearBranchingConfig;

    let graph = swiglu_mul_graph();
    // x in [-2, 2]: SiLU(x)*x. True min over [-2,2] ≈ -0.16 (near x≈-1.1), so
    // SiLU(x)*x >= -0.5 holds. The McCormick relaxation decorrelates the product
    // at the root, so BaB must split the MulBinary node to make progress.
    let input = BoundedTensor::new(arr1(&[-2.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn())
        .expect("finite bounds");

    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::GenBaB(NonlinearBranchingConfig {
            num_candidates: 4,
            ..Default::default()
        }),
        use_alpha_crown: false,
        max_domains: 500,
        timeout: Duration::from_secs(20),
        ..Default::default()
    };

    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split(&graph, &input, &[1.0_f32], -0.5)
        .expect("verify should not hard-error");

    // Must not falsely claim violation (true min ≈ -0.16 > -0.5).
    assert!(
        !matches!(
            result.result,
            BabVerificationStatus::Violated { .. } | BabVerificationStatus::PotentialViolation
        ),
        "SiLU(x)*x >= -0.5 holds: {:?}",
        result.result
    );

    // The blocker symptom was "Unknown after exploring 1 domain at depth 0"
    // (child propagation failed). Post-fix, BaB descends and verifies sub-domains.
    assert!(
        result.max_depth_reached >= 1,
        "GenBaB must descend past depth 0 on the MulBinary graph, got depth {}",
        result.max_depth_reached
    );
    assert!(
        result.domains_verified >= 1,
        "GenBaB must verify at least one MulBinary sub-domain, got {}",
        result.domains_verified
    );
}

/// DAG graph: input → (linear_a, linear_b) → MulBinary → linear_out → output.
///
/// Computes `y = W_out * (W_a * x ⊙ W_b * x) + b_out` where `⊙` is element-wise.
/// This is a minimal Lyapunov-like DAG structure (cf. lsnc_relu benchmark).
fn mul_binary_dag_graph() -> GraphNetwork {
    // Branch A: x → [2, -1] * x (2-dim output)
    let w_a = arr2(&[[2.0_f32, 0.0], [0.0, -1.0]]);
    let linear_a = LinearLayer::new(w_a, None).expect("linear_a should build");

    // Branch B: x → [1, 3] * x (2-dim output)
    let w_b = arr2(&[[1.0_f32, 0.0], [0.0, 3.0]]);
    let linear_b = LinearLayer::new(w_b, None).expect("linear_b should build");

    // MulBinary: element-wise multiply of branch_a and branch_b outputs.
    // Result: [2*x0*x0, -3*x1*x1] = [2*x0^2, -3*x1^2]
    let mul = MulBinaryLayer;

    // Output: sum to scalar: 2*x0^2 - 3*x1^2 + 0.5
    let w_out = arr2(&[[1.0_f32, 1.0]]);
    let linear_out = LinearLayer::new(w_out, Some(arr1(&[0.5]))).expect("linear_out should build");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("branch_a", Layer::Linear(linear_a)));
    graph.add_node(GraphNode::from_input("branch_b", Layer::Linear(linear_b)));
    graph.add_node(GraphNode::binary(
        "mul",
        Layer::MulBinary(mul),
        "branch_a",
        "branch_b",
    ));
    graph.add_node(GraphNode::new(
        "output",
        Layer::Linear(linear_out),
        vec!["mul".to_string()],
    ));
    graph.set_output("output");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_input_split_mul_binary_dag_explores_domains_3439() {
    // Verify that graph input-split BaB can handle MulBinary nodes without
    // UnsupportedOp errors. This tests the full path: graph BaB → canonical
    // dispatch → MulBinaryLayer::propagate_linear_binary (McCormick envelopes).
    let graph = mul_binary_dag_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite bounds");

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 16,
        max_depth: 4,
        timeout: Duration::from_secs(5),
        ..Default::default()
    });

    // Threshold chosen so the property is not trivially verified or falsified:
    // f(x) = 2*x0^2 - 3*x1^2 + 0.5, verifying f(x) < threshold.
    // At x=(0,0): f=0.5, at x=(1,0): f=2.5, at x=(0,1): f=-2.5
    let result = verifier
        .verify_graph_input_split(&graph, &input, &[1.0], -0.5)
        .expect("graph input split with MulBinary should not error");

    // Key assertion: BaB explored at least one domain through MulBinary CROWN.
    // The exact verification status depends on bound tightness (McCormick relaxation),
    // so we don't assert Verified/Unknown — just that domains were explored.
    assert!(
        result.domains_explored > 0,
        "BaB should explore domains through MulBinary CROWN backward, got 0"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_input_split_mul_binary_dag_alpha_crown_3439() {
    // Same DAG but with alpha-CROWN enabled. Verifies that DAG alpha-CROWN
    // handles MulBinary nodes (via propagate_dag_alpha_crown path).
    let graph = mul_binary_dag_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("finite bounds");

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: true,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 16,
        max_depth: 4,
        timeout: Duration::from_secs(5),
        ..Default::default()
    });

    let result = verifier
        .verify_graph_input_split(&graph, &input, &[1.0], -0.5)
        .expect("graph input split with MulBinary + alpha-CROWN should not error");

    assert!(
        result.domains_explored > 0,
        "Alpha-CROWN BaB should explore domains through MulBinary, got 0"
    );
}
