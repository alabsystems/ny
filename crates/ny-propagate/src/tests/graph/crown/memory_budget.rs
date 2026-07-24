// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
use ndarray::{arr1, arr2};

use crate::tests::graph::memory_budget_fixture::build_avgpool_memory_budget_graph;
use crate::*;

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_memory_budget_falls_back_to_ibp_3515() {
    tests::with_crown_dense_budget_mb("0", || {
        let (graph, input) = build_avgpool_memory_budget_graph();

        let ibp = graph.propagate_ibp(&input).unwrap();
        let crown = graph.propagate_crown_with_provenance(&input).unwrap();

        assert_eq!(
            crown.provenance,
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::MemoryBudgetExceeded)
        );
        assert_eq!(crown.bounds.lower(), ibp.lower());
        assert_eq!(crown.bounds.upper(), ibp.upper());
    });
}

/// Build a simple MLP graph (non-CNN) for batched CROWN budget tests.
///
/// Linear → ReLU → Linear, 2-dim input/output. Non-CNN so batched CROWN
/// uses Dense identity (not Patches), which triggers the budget guard.
fn build_mlp_budget_test_graph() -> (GraphNetwork, BoundedTensor) {
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

/// #3550 regression: Sequential `Network::propagate_crown_batched()` with zero
/// budget falls back gracefully (not error). With zero budget, the batched path
/// falls back to sequential CROWN, which also hits the zero-budget guard and
/// falls back to IBP. The key contract is no crash/error and sound bounds.
#[ntest::timeout(10000)]
#[test]
fn test_network_batched_crown_zero_budget_falls_back_to_unbatched_3550() {
    tests::with_crown_dense_budget_mb("0", || {
        // Build MLP: Linear → ReLU → Linear
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, 0.5], [-0.5, 1.0]]),
                Some(arr1(&[0.1, -0.1])),
            )
            .unwrap(),
        ));
        network.add_layer(Layer::ReLU(ReLULayer));
        network.add_layer(Layer::Linear(
            LinearLayer::new(
                arr2(&[[2.0_f32, -1.0], [1.0, 2.0]]),
                Some(arr1(&[0.0, 0.0])),
            )
            .unwrap(),
        ));

        let input = BoundedTensor::new(
            arr1(&[-0.5_f32, -0.5]).into_dyn(),
            arr1(&[0.5_f32, 0.5]).into_dyn(),
        )
        .unwrap();

        let ibp_bounds = network.propagate_ibp(&input).unwrap();

        // With zero budget, batched CROWN should fall back gracefully (not error).
        let batched_bounds = network.propagate_crown_batched(&input).unwrap();

        // Fallback bounds must be sound: at least as wide as IBP.
        for (&bl, &il) in batched_bounds.lower().iter().zip(ibp_bounds.lower().iter()) {
            assert!(
                bl <= il + 1e-6,
                "lower bound unsound: batched={bl} > ibp={il}"
            );
        }
        for (&bu, &iu) in batched_bounds.upper().iter().zip(ibp_bounds.upper().iter()) {
            assert!(
                bu >= iu - 1e-6,
                "upper bound unsound: batched={bu} < ibp={iu}"
            );
        }
    });
}

/// #3550 regression: Graph `propagate_crown_batched_with_provenance()` with
/// zero budget falls back through the existing DAG-CROWN provenance path.
#[ntest::timeout(10000)]
#[test]
fn test_graph_batched_crown_zero_budget_falls_back_with_provenance_3550() {
    tests::with_crown_dense_budget_mb("0", || {
        let (graph, input) = build_mlp_budget_test_graph();
        let expected = graph.propagate_crown_with_provenance(&input).unwrap();
        let batched = graph
            .propagate_crown_batched_with_provenance(&input)
            .unwrap();

        assert!(
            matches!(
                batched.provenance,
                BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::MemoryBudgetExceeded)
            ),
            "expected MemoryBudgetExceeded provenance, got: {:?}",
            batched.provenance
        );
        assert_eq!(batched.bounds.lower(), expected.bounds.lower());
        assert_eq!(batched.bounds.upper(), expected.bounds.upper());
    });
}

/// #3550 regression: `propagate_crown_batched()` remains a thin wrapper over
/// the provenance-returning graph batched entrypoint.
#[ntest::timeout(10000)]
#[test]
fn test_graph_batched_crown_zero_budget_wrapper_returns_fallback_bounds_3550() {
    tests::with_crown_dense_budget_mb("0", || {
        let (graph, input) = build_mlp_budget_test_graph();
        let expected = graph
            .propagate_crown_batched_with_provenance(&input)
            .unwrap();
        let bounds = graph.propagate_crown_batched(&input).unwrap();

        assert_eq!(bounds.lower(), expected.bounds.lower());
        assert_eq!(bounds.upper(), expected.bounds.upper());
    });
}

/// #3550 regression: `BatchedCrownBounds::into_batched_dense_checked()` rejects
/// Patches-to-Dense materialization under zero budget.
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_bounds_checked_densification_rejects_under_zero_budget_3550() {
    use crate::bounds::patches::PatchesLinearBounds;
    use crate::bounds::patches_batched::BatchedCrownBounds;

    tests::with_crown_dense_budget_mb("0", || {
        let shape = (1, 2, 2); // 1 channel, 2x2 → 4-dim dense
        let plb = PatchesLinearBounds::identity(shape, shape);
        let bcb = BatchedCrownBounds::Patches(Box::new(plb));

        let result = bcb.into_batched_dense_checked("test:patches_to_dense");
        assert!(result.is_err(), "expected CpuMemoryExceeded error");

        let err = result.unwrap_err();
        assert!(
            matches!(err, NyError::CpuMemoryExceeded { .. }),
            "expected CpuMemoryExceeded, got: {err}"
        );
    });
}

/// #3550 regression: `BatchedCrownBounds::ensure_batched_dense_checked()` rejects
/// Patches-to-Dense in-place materialization under zero budget.
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_bounds_ensure_checked_rejects_under_zero_budget_3550() {
    use crate::bounds::patches::PatchesLinearBounds;
    use crate::bounds::patches_batched::BatchedCrownBounds;

    tests::with_crown_dense_budget_mb("0", || {
        let shape = (1, 2, 2);
        let plb = PatchesLinearBounds::identity(shape, shape);
        let mut bcb = BatchedCrownBounds::Patches(Box::new(plb));

        let result = bcb.ensure_batched_dense_checked("test:ensure_patches_to_dense");
        assert!(result.is_err(), "expected CpuMemoryExceeded error");

        let err = result.unwrap_err();
        assert!(
            matches!(err, NyError::CpuMemoryExceeded { .. }),
            "expected CpuMemoryExceeded, got: {err}"
        );
        // Patches should remain unconverted after rejection.
        assert!(
            bcb.is_patches(),
            "Patches should not be converted after budget rejection"
        );
    });
}

/// #3550 regression: Checked densification passes through Dense variant
/// without a budget check (already materialized).
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_bounds_checked_dense_passthrough_3550() {
    use crate::bounds::patches_batched::BatchedCrownBounds;
    use crate::bounds::BatchedLinearBounds;
    use ndarray::{ArrayD, IxDyn};

    tests::with_crown_dense_budget_mb("0", || {
        // Dense variant should pass through even with zero budget.
        let blb = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::zeros(IxDyn(&[4, 4])),
            ArrayD::zeros(IxDyn(&[4])),
            ArrayD::zeros(IxDyn(&[4, 4])),
            ArrayD::zeros(IxDyn(&[4])),
            vec![4],
            vec![4],
        );
        let bcb = BatchedCrownBounds::Dense(blb);
        let result = bcb.into_batched_dense_checked("test:dense_passthrough");
        assert!(
            result.is_ok(),
            "Dense variant should pass through without budget check"
        );
    });
}
