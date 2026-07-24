// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::AlphaCrownConfig;
use crate::*;

use super::memory_budget_fixture::build_avgpool_memory_budget_graph;

fn total_width(bounds: &BoundedTensor) -> f32 {
    bounds
        .upper()
        .iter()
        .zip(bounds.lower().iter())
        .map(|(&upper, &lower)| upper - lower)
        .sum()
}

#[ntest::timeout(10000)]
#[test]
fn test_dag_alpha_crown_memory_budget_falls_back_to_ibp_3515() {
    let (graph, input) = build_avgpool_memory_budget_graph();
    let config = AlphaCrownConfig {
        iterations: 1,
        adaptive_skip: false,
        ..AlphaCrownConfig::default()
    };

    let ibp = graph.propagate_ibp(&input).unwrap();
    let alpha_unbudgeted = tests::with_crown_dense_budget_mb("2048", || {
        graph
            .propagate_alpha_crown_with_config(&input, &config)
            .unwrap()
    });
    assert!(
        total_width(&alpha_unbudgeted) + 1e-4 < total_width(&ibp),
        "unbudgeted alpha-CROWN should be tighter than IBP for this nonlinear graph"
    );

    let (crown_budgeted, alpha_budgeted) = tests::with_crown_dense_budget_mb("0", || {
        (
            graph.propagate_crown_with_provenance(&input).unwrap(),
            graph
                .propagate_alpha_crown_with_config(&input, &config)
                .unwrap(),
        )
    });
    assert_eq!(
        crown_budgeted.provenance,
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::MemoryBudgetExceeded)
    );

    assert_eq!(alpha_budgeted.lower(), ibp.lower());
    assert_eq!(alpha_budgeted.upper(), ibp.upper());
    assert_eq!(alpha_budgeted.lower(), crown_budgeted.bounds.lower());
    assert_eq!(alpha_budgeted.upper(), crown_budgeted.bounds.upper());
}
