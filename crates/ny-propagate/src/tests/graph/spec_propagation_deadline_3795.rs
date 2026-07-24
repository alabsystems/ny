// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::types::{BoundsProvenance, CrownBackwardResult, CrownIbpFallbackReason};
use crate::*;
use ndarray::{arr1, Array2, ArrayD, IxDyn};
use ny_core::{GemmEngine, NaiveCpuGemmEngine, Result};
use std::thread;
use std::time::{Duration, Instant};

struct SleepingGemmEngine {
    sleep_per_call: Duration,
}

impl GemmEngine for SleepingGemmEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        thread::sleep(self.sleep_per_call);
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }
}

fn build_large_conv_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[1, 1, 3, 3]),
        vec![1.0_f32, 0.0, -1.0, 0.5, 0.25, -0.5, 1.0, 0.0, -1.0],
    )
    .unwrap();
    let conv =
        Conv2dLayer::with_input_shape(kernel, Some(arr1(&[0.1_f32])), (1, 1), (1, 1), 32, 32)
            .expect("valid conv2d");
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.set_output("conv");
    graph
}

fn build_large_conv_case() -> (
    GraphNetwork,
    BoundedTensor,
    std::collections::HashMap<String, BoundedTensor>,
    Array2<f32>,
) {
    let graph = build_large_conv_graph();
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 32, 32]), -0.25_f32),
        ArrayD::from_elem(IxDyn(&[1, 32, 32]), 0.25_f32),
    )
    .unwrap();
    let node_bounds = graph.collect_node_bounds(&input).expect("conv node bounds");
    let output_flat_dim = node_bounds.get("conv").expect("conv output bounds").len();
    let mut spec_matrix = Array2::zeros((1, output_flat_dim));
    spec_matrix[[0, output_flat_dim / 2]] = 1.0;
    (graph, input, node_bounds, spec_matrix)
}

fn run_spec_guided_conv2d_deadline_probe(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> CrownBackwardResult {
    graph
        .propagate_crown_with_specs_and_provenance_and_engine_with_node_bounds_and_deadline(
            input,
            spec_matrix,
            engine,
            node_bounds,
            deadline,
        )
        .expect("deadline-aware Conv2d spec CROWN should fall back cleanly")
}

/// Regression for #3795/#3881: a single large Conv2d node must honor the verifier
/// deadline from inside the node-local GEMM path instead of relying on an
/// external watchdog. Sub-floor per-node budgets keep the global deadline
/// so CROWN LinearBounds are preserved on short-budget tiny graphs (#3881).
#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_conv2d_per_node_deadline_fallback_3795() {
    // Keep the legacy pair path active: this regression specifically exercises
    // its deadline-aware GemmEngine calls rather than the default dead-f32 skip.
    tests::with_serialized_env_vars(&[("NY_CONV_SKIP_DEAD_F32", "0")], || {
        let (graph, input, node_bounds, spec_matrix) = build_large_conv_case();

        // Case 1: deadline already expired → immediate IBP fallback (DeadlineExceeded)
        let expired = run_spec_guided_conv2d_deadline_probe(
            &graph,
            &input,
            &node_bounds,
            &spec_matrix,
            None,
            Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
        );
        assert_eq!(
            expired.provenance,
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded)
        );

        // Case 2: sub-floor per-node budget with remaining time → CROWN backward
        // proceeds with global deadline instead of falling back to IBP (#3881).
        // This preserves LinearBounds for split scoring on short-budget tiny graphs.
        let slow_engine = SleepingGemmEngine {
            sleep_per_call: Duration::from_millis(5),
        };
        let result = run_spec_guided_conv2d_deadline_probe(
            &graph,
            &input,
            &node_bounds,
            &spec_matrix,
            Some(&slow_engine),
            Some(Instant::now() + Duration::from_millis(100)),
        );
        assert_eq!(
            result.provenance,
            BoundsProvenance::Crown,
            "sub-floor budget should complete CROWN backward with global deadline (#3881)"
        );
    });
}
