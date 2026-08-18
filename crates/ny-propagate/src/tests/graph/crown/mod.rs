// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork CROWN propagation tests.
//!
//! Split from monolithic `crown.rs` (1572 LOC) into submodules:
//! - `batched`: Batched CROWN parity and fallback tests
//! - `binary_ops`: Add and MatMul backward propagation tests
//! - `layernorm`: LayerNorm and ConvTranspose2d CROWN tests
//! - `spec_guided`: Spec-guided CROWN (#593) tests
//! - `regression_2099`: Regression tests for #2099 (empty-input panics)

mod batched;
mod batched_engine;
mod binary_ops;
mod block_wise;
mod block_wise_deadline;
mod block_wise_engine;
mod block_wise_explicit_spec;
mod classifier_head_3813;
mod crown_ibp_engine;
mod crown_ibp_fast_path_demand;
mod crown_ibp_gpu_fast_path;
mod crown_ibp_gpu_fast_path_conv1d;
mod div_parity;
mod expand_like;
mod fallback_reason;
mod graph_backward;
mod graph_backward_engine;
mod guard_coverage_4280;
mod layernorm;
mod memory_budget;
mod nan_guards;
mod norm_stats;
mod patches;
mod per_position_engine;
mod regression_2099;
mod regression_2817;
mod regression_4146;
mod regression_4243;
mod spec_guided;
mod truncated_backward_3813;
mod where_parity;

use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
use crate::*;
use ndarray::{arr1, arr2};
use ny_core::NyError;

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_sequential() {
    // Test CROWN propagation on a sequential graph
    let mut graph = GraphNetwork::new();

    // Build: input -> linear -> relu
    let weight = arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[0.1_f32, -0.1]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let _ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify soundness: sample points should be within CROWN bounds
    let test_points = vec![
        arr1(&[-0.5_f32, -0.5]),
        arr1(&[0.5_f32, 0.5]),
        arr1(&[0.0_f32, 0.0]),
        arr1(&[-0.5_f32, 0.5]),
        arr1(&[0.5_f32, -0.5]),
    ];

    for point in test_points {
        let linear_out = arr1(&[
            point[[0]] + 0.5 * point[[1]] + 0.1,
            -0.5 * point[[0]] + point[[1]] - 0.1,
        ]);
        let relu_out = linear_out.mapv(|v| v.max(0.0));

        for i in 0..2 {
            assert!(
                relu_out[[i]] >= crown_bounds.lower()[[i]] - 1e-5,
                "CROWN: Point {:?}: output[{}]={} < lower={}",
                point,
                i,
                relu_out[[i]],
                crown_bounds.lower()[[i]]
            );
            assert!(
                relu_out[[i]] <= crown_bounds.upper()[[i]] + 1e-5,
                "CROWN: Point {:?}: output[{}]={} > upper={}",
                point,
                i,
                relu_out[[i]],
                crown_bounds.upper()[[i]]
            );
        }
    }

    // Note: CROWN's lower bound can be looser than IBP for ReLU outputs in some cases
    // because the linear relaxation y >= α*x can produce negative values when α < 1 and x < 0.
    // The key property is that CROWN bounds are SOUND (all true outputs are contained).
    // For tightness comparisons, see test_graph_network_crown_tighter_than_ibp which uses
    // upper bound comparison where CROWN is typically tighter.
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_with_silu() {
    let mut graph = GraphNetwork::new();

    let weight = arr2(&[[1.0_f32]]);
    let bias = arr1(&[0.0_f32]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "silu",
        Layer::SiLU(SiLULayer::new()),
        vec!["linear".to_string()],
    ));
    graph.set_output("silu");

    let input =
        BoundedTensor::new(arr1(&[-2.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn()).unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let silu = SiLULayer::new();
    for x in [-2.0_f32, -1.0, 0.0, 1.0, 2.0] {
        let y = silu.eval(x);
        assert!(
            y >= crown_bounds.lower()[[0]] - 1e-5,
            "SiLU({})={} below CROWN lower {}",
            x,
            y,
            crown_bounds.lower()[[0]]
        );
        assert!(
            y <= crown_bounds.upper()[[0]] + 1e-5,
            "SiLU({})={} above CROWN upper {}",
            x,
            y,
            crown_bounds.upper()[[0]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_skip_merge_multi_input_errors() {
    let mut graph = GraphNetwork::new();

    let weight = arr2(&[[1.0_f32]]);
    let bias = arr1(&[0.0_f32]);
    let linear_a = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();
    let linear_b = LinearLayer::new(weight, Some(bias)).unwrap();

    graph.add_node(GraphNode::from_input("a", Layer::Linear(linear_a)));
    graph.add_node(GraphNode::from_input("b", Layer::Linear(linear_b)));
    graph.add_node(GraphNode::new(
        "skip_merge",
        Layer::SkipMerge(SkipMergeLayer::new()),
        vec!["a".to_string(), "b".to_string()],
    ));
    graph.set_output("skip_merge");

    let input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let err = graph.propagate_crown(&input).unwrap_err();
    assert!(
        err.to_string().contains("SkipMerge node"),
        "expected SkipMerge multi-input error, got: {}",
        err
    );
}

/// Regression test for #2062: CROWN backward propagation must report fallback
/// provenance when a layer is unsupported (NonZero has data-dependent output shape).
///
/// Without provenance, callers cannot distinguish tight CROWN bounds from
/// vacuous IBP fallback bounds. This test verifies that
/// `propagate_crown_with_provenance` returns `ForwardFallback` provenance
/// when CROWN backward cannot handle a layer.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_backward_reports_fallback_provenance_for_unsupported_layer_2062() {
    // Build: input -> linear -> nonzero
    // NonZero has data-dependent output shape, so CROWN backward returns Unsupported
    // and the function falls back to IBP forward bounds.
    let mut graph = GraphNetwork::new();

    let weight = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let bias = arr1(&[0.0_f32, 0.0]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "nonzero",
        Layer::NonZero(NonZeroLayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("nonzero");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    // The existing API still works (returns bounds without provenance)
    let bounds = graph.propagate_crown(&input).unwrap();
    assert!(!bounds.lower().iter().any(|v| v.is_nan()));

    // The new provenance API reports the fallback
    let result = graph.propagate_crown_with_provenance(&input).unwrap();
    assert!(
        result.is_fallback(),
        "Expected ForwardFallback provenance for unsupported NonZero layer, got {:?}",
        result.provenance
    );
    assert_eq!(
        result.provenance,
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::CrownPropagationError),
        "Fallback reason should be CrownPropagationError for unsupported layer"
    );

    // Bounds should be sound (same as IBP)
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    assert_eq!(result.bounds.shape(), ibp_bounds.shape());
}

/// Test that `propagate_crown_with_provenance` returns `Crown` provenance
/// when CROWN backward propagation succeeds without fallback.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_backward_reports_crown_provenance_on_success_2062() {
    let mut graph = GraphNetwork::new();

    let weight = arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[0.1_f32, -0.1]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let result = graph.propagate_crown_with_provenance(&input).unwrap();
    assert_eq!(
        result.provenance,
        BoundsProvenance::Crown,
        "Successful CROWN backward should report Crown provenance, got {:?}",
        result.provenance
    );
    assert!(!result.is_fallback());
}

/// Regression test for #2502: forward-bound tightening must apply to non-softmax
/// DAGs as well, keeping final CROWN bounds no wider than forward IBP bounds.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_non_softmax_tightening_keeps_bounds_within_ibp_2502() {
    // Build: input -> linear -> gelu -> linear (no softmax layers).
    let mut graph = GraphNetwork::new();

    let linear1 = LinearLayer::new(
        arr2(&[[3.5_f32, -2.1], [1.7, 2.8]]),
        Some(arr1(&[0.2_f32, -0.3])),
    )
    .unwrap();
    let linear2 = LinearLayer::new(
        arr2(&[[4.0_f32, -3.2], [-2.5, 3.8]]),
        Some(arr1(&[0.1_f32, -0.2])),
    )
    .unwrap();

    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::new(GeluApproximation::Erf)),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        arr1(&[-3.0_f32, -2.0]).into_dyn(),
        arr1(&[3.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    assert_eq!(crown_bounds.shape(), ibp_bounds.shape());

    for (((cl, cu), il), iu) in crown_bounds
        .lower()
        .iter()
        .zip(crown_bounds.upper().iter())
        .zip(ibp_bounds.lower().iter())
        .zip(ibp_bounds.upper().iter())
    {
        assert!(
            *cl >= *il - 1e-4,
            "CROWN lower bound should be no looser than IBP lower: crown={} ibp={}",
            cl,
            il
        );
        assert!(
            *cu <= *iu + 1e-4,
            "CROWN upper bound should be no looser than IBP upper: crown={} ibp={}",
            cu,
            iu
        );
    }
}

/// Helper: build a simple linear -> relu -> linear graph for deadline tests.
fn build_deadline_test_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let l1 = LinearLayer::new(
        arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]),
        Some(arr1(&[0.1, -0.1])),
    );
    let l2 = LinearLayer::new(
        arr2(&[[2.0_f32, -1.0], [1.0, 2.0]]),
        Some(arr1(&[0.0, 0.0])),
    );
    graph.add_node(GraphNode::from_input("l1", Layer::Linear(l1.unwrap())));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["l1".into()],
    ));
    graph.add_node(GraphNode::new(
        "l2",
        Layer::Linear(l2.unwrap()),
        vec!["relu".into()],
    ));
    graph.set_output("l2");
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();
    (graph, input)
}

fn assert_bounds_match_ibp(deadline_bounds: &BoundedTensor, ibp: &BoundedTensor) {
    for (&d, &i) in deadline_bounds.lower().iter().zip(ibp.lower().iter()) {
        assert!((d - i).abs() < 1e-6, "lower mismatch: deadline={d} ibp={i}");
    }
    for (&d, &i) in deadline_bounds.upper().iter().zip(ibp.upper().iter()) {
        assert!((d - i).abs() < 1e-6, "upper mismatch: deadline={d} ibp={i}");
    }
}

/// An already-expired authority cannot launch a fresh no-deadline IBP pass.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_deadline_expired_returns_typed_deadline_3398() {
    use std::time::{Duration, Instant};
    let (graph, input) = build_deadline_test_graph();

    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let error = graph
        .propagate_crown_with_engine_and_deadline(&input, None, Some(expired))
        .expect_err("expired authority must refuse before forward-bound collection");
    assert!(matches!(error, NyError::DeadlineExceeded(_)));
}

/// The legacy Dense seed has no cooperative finite implementation. A live
/// authority publishes the checked forward enclosure before constructing its
/// quadratic identity, while the no-deadline lane retains fixed-slope CROWN.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_deadline_future_dense_seed_uses_checked_forward_3398() {
    use std::time::{Duration, Instant};
    let (graph, input) = build_deadline_test_graph();

    let future = Instant::now() + Duration::from_hours(1);
    let result = graph
        .propagate_crown_with_engine_and_deadline(&input, None, Some(future))
        .unwrap();
    assert_eq!(
        result.provenance,
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::CrownPropagationError),
    );
    let ibp = graph.propagate_ibp(&input).unwrap();
    for (&forward, &plain) in result.bounds.lower().iter().zip(ibp.lower().iter()) {
        assert!(
            forward >= plain - 1e-6,
            "collected lower bound {forward} must not be looser than IBP {plain}"
        );
    }
    for (&forward, &plain) in result.bounds.upper().iter().zip(ibp.upper().iter()) {
        assert!(
            forward <= plain + 1e-6,
            "collected upper bound {forward} must not be looser than IBP {plain}"
        );
    }
}

/// The relaxation-aware wrapper must preserve typed deadline authority before
/// any fallback recomputation.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_relaxation_deadline_expired_returns_typed_deadline_3398() {
    use std::time::{Duration, Instant};
    let (graph, input) = build_deadline_test_graph();

    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let error = graph
        .propagate_crown_with_engine_relaxation_and_deadline(
            &input,
            None,
            MulBinaryRelaxationMode::default(),
            Some(expired),
        )
        .expect_err("expired relaxation request must not launch fallback work");
    assert!(matches!(error, NyError::DeadlineExceeded(_)));
}

/// #3398: the batched deadline wrapper must expose IBP fallback provenance, not raw bounds alone.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_batched_deadline_expired_returns_ibp_provenance_3398() {
    use std::time::{Duration, Instant};
    let (graph, input) = build_deadline_test_graph();

    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let result = graph
        .propagate_crown_batched_with_relaxation_and_deadline(
            &input,
            MulBinaryRelaxationMode::default(),
            Some(expired),
        )
        .unwrap();
    assert_eq!(
        result.provenance,
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded),
    );

    let ibp = graph.propagate_ibp(&input).unwrap();
    assert_bounds_match_ibp(&result.bounds, &ibp);
}
