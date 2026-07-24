// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::graph::crown_fallback_chain;
use crate::{
    GraphNetwork, GraphNode, Layer, LinearLayer, MulBinaryRelaxationMode, PropagationConfig,
    PropagationMethod, ReLULayer, Verifier,
};
use ndarray::{arr1, arr2};
use ny_core::{Bound, Result, VerificationResult, VerificationSpec};
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;
use std::time::{Duration, Instant};

fn build_deadline_test_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let l1 = LinearLayer::new(
        arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]),
        Some(arr1(&[0.1, -0.1])),
    )
    .unwrap();
    let l2 = LinearLayer::new(
        arr2(&[[2.0_f32, -1.0], [1.0, 2.0]]),
        Some(arr1(&[0.0, 0.0])),
    )
    .unwrap();
    graph.add_node(GraphNode::from_input("l1", Layer::Linear(l1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["l1".into()],
    ));
    graph.add_node(GraphNode::new("l2", Layer::Linear(l2), vec!["relu".into()]));
    graph.set_output("l2");
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();
    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_fallback_chain_deadline_fallback_reports_ibp_method_3398() {
    let (graph, input) = build_deadline_test_graph();
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();

    let (bounds, method) = crown_fallback_chain(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
        Some(expired),
    )
    .unwrap();

    assert_eq!(method, PropagationMethod::Ibp);
    let ibp = graph.propagate_ibp(&input).unwrap();
    for (&d, &i) in bounds.lower().iter().zip(ibp.lower().iter()) {
        assert!((d - i).abs() < 1e-6, "lower mismatch: deadline={d} ibp={i}");
    }
    for (&d, &i) in bounds.upper().iter().zip(ibp.upper().iter()) {
        assert!((d - i).abs() < 1e-6, "upper mismatch: deadline={d} ibp={i}");
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_graph_with_engine_middle_relaxation_uses_batched_fallback_engine_3622() -> Result<()>
{
    let (graph, _) = build_deadline_test_graph();
    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Crown,
        mul_binary_relaxation: MulBinaryRelaxationMode::Middle,
        ..Default::default()
    });
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-0.5, 0.5), Bound::new(-0.5, 0.5)],
        vec![Bound::new(-5.0, 5.0), Bound::new(-5.0, 5.0)],
        Some(5_000),
        None,
    )?;
    let engine = CountingGemmEngine::new();

    let result = verifier.verify_graph_with_engine(&graph, &spec, Some(&engine))?;

    assert!(
        matches!(result, VerificationResult::Verified { .. }),
        "expected graph verifier to keep the property verified, got {result:?}"
    );
    let calls = engine.gemm_calls();
    assert!(
        calls > 0,
        "#3622 regression: verify_graph_with_engine should thread GemmEngine through the first batched CROWN fallback attempt, got {calls} GEMM calls"
    );
    Ok(())
}
