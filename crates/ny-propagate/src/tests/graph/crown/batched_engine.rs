// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Public graph batched CROWN engine-threading regressions.

use ny_test_utils::assert_bounded_tensor_close;

use super::crown_ibp_engine::build_two_linear_relu_graph;
use crate::tests::crown::helpers::{assert_bounds_finite, CountingGemmEngine};

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_batched_with_provenance_and_engine_threads_backward_3959() {
    crate::tests::with_crown_dense_budget_mb("2048", || {
        let (graph, input) = build_two_linear_relu_graph();
        let baseline = graph
            .propagate_crown_batched_with_provenance(&input)
            .expect("#3959 baseline graph batched CROWN should succeed");

        // Engine-agnostic cache serving may reuse the baseline's valid map.
        // Clone resets that cache so this call still tests engine plumbing.
        #[allow(clippy::redundant_clone)]
        let collection_graph = graph.clone();
        let collection_engine = CountingGemmEngine::new();
        collection_graph
            .collect_crown_ibp_bounds_dag_with_engine(&input, Some(&collection_engine))
            .expect("#3959 graph CROWN-IBP collection should succeed");
        let collection_calls = collection_engine.gemm_calls();
        assert!(
            collection_calls > 0,
            "#3959 regression: graph CROWN-IBP collection should already exercise GemmEngine"
        );

        // Fresh clone: the input-keyed CROWN-IBP collection cache
        // (#cgan-collection-cache) would otherwise serve the collection just
        // computed above, hiding the collection-phase engine calls this
        // assertion compares against. `Clone` resets the cache.
        #[allow(clippy::redundant_clone)] // the clone's side effect (cache reset) is the point
        let engine_graph = graph.clone();
        let engine = CountingGemmEngine::new();
        let with_engine = engine_graph
            .propagate_crown_batched_with_provenance_and_engine(&input, Some(&engine))
            .expect("#3959 engine-aware graph batched CROWN should succeed");

        assert_bounds_finite(
            &with_engine.bounds,
            "graph batched CROWN with engine output",
        );
        assert_eq!(
            with_engine.provenance, baseline.provenance,
            "#3959 regression: graph batched engine path changed provenance"
        );
        assert_bounded_tensor_close(
            &with_engine.bounds,
            &baseline.bounds,
            1e-5,
            "#3959 graph propagate_crown_batched_with_provenance_and_engine parity",
        );
        let total_calls = engine.gemm_calls();
        assert!(
            total_calls > collection_calls,
            "#3959 regression: graph batched CROWN should add backward GemmEngine calls beyond \
             CROWN-IBP collection (total={total_calls}, collection={collection_calls})"
        );
    });
}
