// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::Array1;
use ny_propagate::layers::ReLULayer;
use ny_propagate::types::{BoundsProvenance, CrownIbpFallbackReason};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;
use std::sync::MutexGuard;
use std::time::{Duration, Instant};

const CROWN_DENSE_BUDGET_ENV: &str = "NY_DENSE_BUDGET_MB";

fn build_relu_graph(input_dim: usize) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        Array1::from_elem(input_dim, -1.0_f32).into_dyn(),
        Array1::from_elem(input_dim, 1.0_f32).into_dyn(),
    )
    .expect("bounded relu input should be valid");

    (graph, input)
}

fn assert_bounds_match(actual: &BoundedTensor, expected: &BoundedTensor, label: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: shape mismatch actual={:?} expected={:?}",
        actual.shape(),
        expected.shape()
    );
    assert_eq!(
        actual.lower(),
        expected.lower(),
        "{label}: lower bounds should stay at IBP parity"
    );
    assert_eq!(
        actual.upper(),
        expected.upper(),
        "{label}: upper bounds should stay at IBP parity"
    );
}

/// Dense-budget override, routed through the blessed env choke point
/// (`ny_test_utils::env`, clippy env wall): holds the process-wide env lock
/// for the guard's lifetime and restores the previous value on drop.
/// Field order matters: `_var` restores before `_lock` releases.
struct DenseBudgetEnvGuard {
    _var: ny_test_utils::env::ScopedEnvVar,
    _lock: MutexGuard<'static, ()>,
}

impl DenseBudgetEnvGuard {
    fn set_mb(mb: usize) -> Self {
        let lock = ny_test_utils::env::lock_env();
        let var = ny_test_utils::env::ScopedEnvVar::set(CROWN_DENSE_BUDGET_ENV, &mb.to_string());
        Self {
            _var: var,
            _lock: lock,
        }
    }
}

#[test]
fn test_crown_ibp_deadline_fallback_reports_deadline_exceeded_reason_3499() {
    let (graph, input) = build_relu_graph(8);
    let output_name = graph.output_name().to_string();
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP node-bound collection should succeed");
    let ibp_output = ibp_bounds
        .get(&output_name)
        .expect("IBP output bounds should exist")
        .clone();

    let result = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(
            &input,
            ibp_bounds,
            Some(
                Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("subtracting one second from Instant::now() should succeed"),
            ),
        )
        .expect("expired-deadline CROWN-IBP should fall back to IBP");

    assert_eq!(
        result.provenance.get(&output_name),
        Some(&BoundsProvenance::ForwardFallback(
            CrownIbpFallbackReason::DeadlineExceeded,
        )),
        "expired global deadline should classify the output fallback as DeadlineExceeded"
    );
    assert_eq!(
        result.fallback_events.len(),
        1,
        "single-node graph should emit exactly one fallback event"
    );
    assert_eq!(
        result.fallback_events[0].reason,
        CrownIbpFallbackReason::DeadlineExceeded,
        "fallback event reason should preserve the deadline classification"
    );

    let fallback_output = result
        .bounds
        .get(&output_name)
        .expect("fallback output bounds should exist");
    assert_bounds_match(
        fallback_output,
        &ibp_output,
        "deadline-exceeded fallback output",
    );
}

/// Budget-exceeding CROWN-IBP targets stream through the objective-chunked
/// backward instead of degrading to IBP: a dim-400 identity pair needs
/// 2*400*400*4 = 1.28 MB > the 1 MB budget, so the collection reroutes the
/// node through `propagate_crown_to_node_chunked` with an auto chunk size
/// that fits the budget. The result must still be genuine CROWN output
/// (provenance `Crown`, no fallback events) and at least as tight as IBP.
#[test]
fn test_crown_ibp_memory_budget_overflow_reroutes_chunked_crown_3499() {
    let _guard = DenseBudgetEnvGuard::set_mb(1);
    let (graph, input) = build_relu_graph(400);
    let output_name = graph.output_name().to_string();
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP node-bound collection should succeed");
    let ibp_output = ibp_bounds
        .get(&output_name)
        .expect("IBP output bounds should exist")
        .clone();

    let result = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(&input, ibp_bounds, None)
        .expect("memory-budgeted CROWN-IBP should succeed via the chunked backward");

    assert_eq!(
        result.provenance.get(&output_name),
        Some(&BoundsProvenance::Crown),
        "budget overflow must reroute through the chunked backward and stay on CROWN provenance"
    );
    assert!(
        result.fallback_events.is_empty(),
        "chunked reroute must not emit fallback events, got {:?}",
        result.fallback_events
    );

    let crown_output = result
        .bounds
        .get(&output_name)
        .expect("chunked CROWN output bounds should exist");
    assert_eq!(
        crown_output.shape(),
        ibp_output.shape(),
        "chunked CROWN output shape should match IBP"
    );
    // CROWN-IBP intersects with the precomputed IBP map, so every element
    // must be at least as tight as IBP (and non-inverted).
    for (idx, ((&cl, &cu), (&il, &iu))) in crown_output
        .lower()
        .iter()
        .zip(crown_output.upper().iter())
        .zip(ibp_output.lower().iter().zip(ibp_output.upper().iter()))
        .enumerate()
    {
        assert!(
            cl <= cu,
            "chunked CROWN output inverted at [{idx}]: lower={cl} > upper={cu}"
        );
        assert!(
            cl >= il && cu <= iu,
            "chunked CROWN output looser than IBP at [{idx}]: crown=[{cl}, {cu}] ibp=[{il}, {iu}]"
        );
    }
}
