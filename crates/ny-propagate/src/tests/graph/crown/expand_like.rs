// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched GraphNetwork CROWN regressions for `ExpandLikeLastAxis`.

use crate::types::BoundsProvenance;
use crate::*;
use ndarray::Array2;

fn assert_matches_ibp_with_directed_rounding(crown: &BoundedTensor, ibp: &BoundedTensor) {
    let crown_lower = crown.lower().as_slice_memory_order().unwrap();
    let crown_upper = crown.upper().as_slice_memory_order().unwrap();
    let ibp_lower = ibp.lower().as_slice_memory_order().unwrap();
    let ibp_upper = ibp.upper().as_slice_memory_order().unwrap();

    // Tightness tolerance: 5e-6 allows ~10 ULPs at f32 value ~4.0.
    // Multi-step CROWN backward accumulates ~3 ULPs per matrix multiply;
    // observed max delta: 1.4e-6 at upper[3] (crown=4.0000014, ibp=4.0).
    let tightness_tol = 5e-6;
    for i in 0..crown_lower.len() {
        assert!(
            crown_lower[i] <= ibp_lower[i],
            "lower[{i}] CROWN must be sound (≤ IBP): crown={} ibp={}",
            crown_lower[i],
            ibp_lower[i]
        );
        assert!(
            ibp_lower[i] - crown_lower[i] <= tightness_tol,
            "lower[{i}] CROWN gap too large: crown={} ibp={} delta={:.2e}",
            crown_lower[i],
            ibp_lower[i],
            ibp_lower[i] - crown_lower[i]
        );
        assert!(
            crown_upper[i] >= ibp_upper[i],
            "upper[{i}] CROWN must be sound (≥ IBP): crown={} ibp={}",
            crown_upper[i],
            ibp_upper[i]
        );
        assert!(
            crown_upper[i] - ibp_upper[i] <= tightness_tol,
            "upper[{i}] CROWN gap too large: crown={} ibp={} delta={:.2e}",
            crown_upper[i],
            ibp_upper[i],
            crown_upper[i] - ibp_upper[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_batched_expand_like_last_axis_with_network_input_source() {
    tests::with_crown_dense_budget_mb("2048", || {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "reference",
            Layer::Tile(TileLayer::new(1, 3)),
        ));
        graph.add_node(GraphNode::binary(
            "expanded",
            Layer::ExpandLikeLastAxis(ExpandLikeLastAxisLayer::new()),
            NETWORK_INPUT,
            "reference",
        ));
        graph.set_output("expanded");

        let input = BoundedTensor::new(
            Array2::from_shape_vec((2, 1), vec![1.0_f32, 3.0])
                .unwrap()
                .into_dyn(),
            Array2::from_shape_vec((2, 1), vec![2.0_f32, 4.0])
                .unwrap()
                .into_dyn(),
        )
        .unwrap();

        let crown = graph
            .propagate_crown_batched_with_provenance(&input)
            .unwrap();
        let ibp = graph.propagate_ibp(&input).unwrap();

        assert_eq!(
            crown.provenance,
            BoundsProvenance::Crown,
            "ExpandLikeLastAxis batched graph path should stay on CROWN"
        );
        assert_eq!(crown.bounds.shape(), &[2, 3]);
        assert_eq!(ibp.shape(), &[2, 3]);
        assert_eq!(
            ibp.lower().as_slice_memory_order(),
            Some(&[1.0, 1.0, 1.0, 3.0, 3.0, 3.0][..])
        );
        assert_eq!(
            ibp.upper().as_slice_memory_order(),
            Some(&[2.0, 2.0, 2.0, 4.0, 4.0, 4.0][..])
        );
        assert_matches_ibp_with_directed_rounding(&crown.bounds, &ibp);
    });
}
