// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Deadline and tightening regressions for block-wise CROWN.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::layers::linear::LinearLayer;
use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
use crate::*;

fn build_single_block_identity_graph(hidden: usize) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let identity = LinearLayer::new(Array2::eye(hidden), None).unwrap();
    graph.add_node(GraphNode::new(
        "layer0_identity",
        Layer::Linear(identity),
        vec![NETWORK_INPUT.to_string()],
    ));
    graph.set_output("layer0_identity");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_per_block_crown_tightens_to_block_ibp_output_4242() {
    tests::with_crown_dense_budget_mb("2048", || {
        let graph = build_single_block_identity_graph(2);
        let block_input = BoundedTensor::new(
            Array1::from_vec(vec![-0.75_f32, 0.125]).into_dyn(),
            Array1::from_vec(vec![1.5_f32, 2.75]).into_dyn(),
        )
        .unwrap();
        let exec_order = graph.exec_order().unwrap();
        let block_nodes = GraphNetwork::collect_block_nodes(exec_order);
        let nodes_in_block = block_nodes
            .get(&0)
            .expect("identity graph should produce a single layer0 block");
        let block_node_bounds = graph
            .collect_block_ibp_bounds(nodes_in_block, &block_input)
            .unwrap();
        let ibp_output = block_node_bounds
            .get("layer0_identity")
            .expect("block IBP output must exist");

        let (crown_output, _, provenance) = graph
            .crown_backward_within_block(nodes_in_block, &block_node_bounds, &block_input)
            .unwrap();
        assert_eq!(
            provenance,
            BoundsProvenance::Crown,
            "block-wise CROWN on identity graph must not fall back to forward bounds"
        );

        assert_eq!(crown_output.shape(), ibp_output.shape());
        for ((&crown_l, &crown_u), (&ibp_l, &ibp_u)) in crown_output
            .lower()
            .iter()
            .zip(crown_output.upper().iter())
            .zip(ibp_output.lower().iter().zip(ibp_output.upper().iter()))
        {
            assert!(
                (crown_l - ibp_l).abs() <= 1e-6,
                "tightened per-block CROWN lower {} must match block IBP lower {}",
                crown_l,
                ibp_l
            );
            assert!(
                (crown_u - ibp_u).abs() <= 1e-6,
                "tightened per-block CROWN upper {} must match block IBP upper {}",
                crown_u,
                ibp_u
            );
        }
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_per_block_crown_expired_deadline_returns_ibp_output_4242() {
    use std::time::{Duration, Instant};

    tests::with_crown_dense_budget_mb("2048", || {
        let graph = build_single_block_identity_graph(2);
        let block_input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[2])), 0.5_f32).unwrap();
        let exec_order = graph.exec_order().unwrap();
        let block_nodes = GraphNetwork::collect_block_nodes(exec_order);
        let nodes_in_block = block_nodes
            .get(&0)
            .expect("identity graph should produce a single layer0 block");
        let block_node_bounds = graph
            .collect_block_ibp_bounds(nodes_in_block, &block_input)
            .unwrap();
        let ibp_output = block_node_bounds
            .get("layer0_identity")
            .expect("block IBP output must exist");
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();

        let (deadline_output, norm_stats, provenance) = graph
            .crown_backward_within_block_with_engine(
                nodes_in_block,
                &block_node_bounds,
                &block_input,
                None,
                None,
                Some(expired),
            )
            .unwrap();

        assert_eq!(
            provenance,
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded),
            "expired deadline should produce ForwardFallback(DeadlineExceeded) provenance, got {:?}",
            provenance
        );
        assert!(
            norm_stats.is_empty(),
            "deadline fallback should return before recording normalization stats"
        );
        assert_eq!(deadline_output.shape(), ibp_output.shape());
        for ((&actual_l, &actual_u), (&expected_l, &expected_u)) in deadline_output
            .lower()
            .iter()
            .zip(deadline_output.upper().iter())
            .zip(ibp_output.lower().iter().zip(ibp_output.upper().iter()))
        {
            assert!(
                (actual_l - expected_l).abs() <= 1e-6,
                "expired deadline lower {} must match IBP lower {}",
                actual_l,
                expected_l
            );
            assert!(
                (actual_u - expected_u).abs() <= 1e-6,
                "expired deadline upper {} must match IBP upper {}",
                actual_u,
                expected_u
            );
        }
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_public_per_block_crown_expired_deadline_reports_fallback_4256() {
    use std::time::{Duration, Instant};

    tests::with_crown_dense_budget_mb("2048", || {
        let graph = build_single_block_identity_graph(2);
        let epsilon = 0.5_f32;
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[2])), epsilon).unwrap();
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();

        let result = graph
            .propagate_crown_block_wise_with_deadline(&input, epsilon, Some(expired))
            .unwrap();

        assert_eq!(
            result.total_blocks, 1,
            "identity graph should yield one block"
        );
        let block = &result.blocks[0];
        assert!(
            !block.crown_successful,
            "expired deadline fallback should mark per-block CROWN as unsuccessful"
        );
        assert_eq!(
            block.crown_max_width, block.ibp_max_width,
            "expired deadline fallback should reuse the IBP width"
        );
        assert_eq!(
            block.crown_ibp_ratio, 1.0,
            "expired deadline fallback should report a 1.0 CROWN/IBP ratio"
        );
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_propagate_crown_within_graph_expired_deadline_returns_ibp_output_4242() {
    use std::time::{Duration, Instant};

    tests::with_crown_dense_budget_mb("2048", || {
        let graph = build_single_block_identity_graph(2);
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[2])), 0.5_f32).unwrap();
        let ibp_output = graph.propagate_ibp(&input).unwrap();
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();

        let deadline_output = graph
            .propagate_crown_within_graph_with_engine_and_deadline(&input, None, Some(expired))
            .unwrap();

        assert_eq!(deadline_output.shape(), ibp_output.shape());
        for ((&actual_l, &actual_u), (&expected_l, &expected_u)) in deadline_output
            .lower()
            .iter()
            .zip(deadline_output.upper().iter())
            .zip(ibp_output.lower().iter().zip(ibp_output.upper().iter()))
        {
            assert!(
                (actual_l - expected_l).abs() <= 1e-6,
                "public expired deadline lower {} must match IBP lower {}",
                actual_l,
                expected_l
            );
            assert!(
                (actual_u - expected_u).abs() <= 1e-6,
                "public expired deadline upper {} must match IBP upper {}",
                actual_u,
                expected_u
            );
        }
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_propagate_crown_within_graph_expired_deadline_surfaces_provenance_4256() {
    use std::time::{Duration, Instant};

    tests::with_crown_dense_budget_mb("2048", || {
        let graph = build_single_block_identity_graph(2);
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[2])), 0.5_f32).unwrap();
        let ibp_output = graph.propagate_ibp(&input).unwrap();
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();

        let result = graph
            .propagate_crown_within_graph_with_provenance_and_deadline(&input, Some(expired))
            .unwrap();

        assert_eq!(
            result.provenance,
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded),
            "expired deadline should surface ForwardFallback(DeadlineExceeded)"
        );
        assert_eq!(result.bounds.shape(), ibp_output.shape());
        for ((&actual_l, &actual_u), (&expected_l, &expected_u)) in result
            .bounds
            .lower()
            .iter()
            .zip(result.bounds.upper().iter())
            .zip(ibp_output.lower().iter().zip(ibp_output.upper().iter()))
        {
            assert!(
                (actual_l - expected_l).abs() <= 1e-6,
                "provenance lower {} must match IBP lower {}",
                actual_l,
                expected_l
            );
            assert!(
                (actual_u - expected_u).abs() <= 1e-6,
                "provenance upper {} must match IBP upper {}",
                actual_u,
                expected_u
            );
        }
    });
}
