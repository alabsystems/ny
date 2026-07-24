// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regressions for graph `forward+crown` root-output routing.
//! Part of #4354.

use super::prelude::*;
use crate::beta_crown::engine::graph::forward_mode_test_support::{
    assert_bounds_close_4354, build_forward_mode_graph_fixture_4354,
    expected_forward_root_output_4354, plain_graph_crown_output_4354,
};

#[ntest::timeout(10000)]
#[test]
fn test_compute_initial_graph_bounds_forward_mode_uses_forward_bootstrap_4354() {
    let (graph, input) = build_forward_mode_graph_fixture_4354();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        use_alpha_crown: false,
        use_forward_bounds: true,
        timeout: Duration::from_secs(1),
        ..Default::default()
    });

    let (node_bounds, output_bounds) = verifier
        .compute_initial_graph_bounds(&graph, &input, None)
        .expect("forward+crown precompute should succeed on the fixture graph");
    let expected_node_bounds = graph
        .collect_forward_linear_bounds_dag_with_engine(&input, None)
        .expect("forward-linear bootstrap should succeed on the fixture graph");
    let expected_output = expected_forward_root_output_4354(&graph, &input)
        .expect("identity-spec forward CROWN should succeed on the fixture graph");
    let plain_output =
        plain_graph_crown_output_4354(&graph, &input, verifier.config.crown_backward_layers)
            .expect("plain graph CROWN should succeed on the fixture graph");

    assert_bounds_close_4354(
        node_bounds
            .get("relu")
            .expect("precomputed node bounds should include relu"),
        expected_node_bounds
            .get("relu")
            .expect("forward-linear bootstrap should include relu"),
        "forward bootstrap node reuse",
    );
    assert!(
        plain_output
            .lower()
            .iter()
            .zip(expected_output.lower().iter())
            .chain(
                plain_output
                    .upper()
                    .iter()
                    .zip(expected_output.upper().iter())
            )
            .any(|(plain, expected)| (plain - expected).abs() > 1e-6),
        "fixture must distinguish forward+crown output from plain DAG-CROWN so the routing regression stays observable"
    );
    assert_bounds_close_4354(
        &output_bounds,
        &expected_output,
        "compute_initial_graph_bounds forward output",
    );
}
