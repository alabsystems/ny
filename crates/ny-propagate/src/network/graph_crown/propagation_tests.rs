// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::spec_propagation::SpecCrownRequest;
use super::GraphNetworkCrownExt;
use crate::layers::{Layer, LinearLayer};
use crate::network::{GraphNetwork, GraphNode};
use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
use crate::MulBinaryRelaxationMode;
use ndarray::{arr1, arr2, array};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};

fn test_input() -> BoundedTensor {
    BoundedTensor::new(
        array![-1.0_f32, -0.5].into_dyn(),
        array![0.25_f32, 1.0].into_dyn(),
    )
    .expect("bounded tensor should construct")
}

fn single_linear_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(arr2(&[[1.5_f32, -0.25]]), Some(arr1(&[0.75_f32])))
        .expect("linear layer should construct");
    graph.add_node(GraphNode::from_input("lin", Layer::Linear(linear)));
    graph.set_output("lin");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_backward_empty_graph_fast_paths_preserve_input_4205() {
    let graph = GraphNetwork::new();
    let input = test_input();

    let plain = GraphNetworkCrownExt::crown_backward_with_relaxation(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
    )
    .expect("empty graph CROWN should succeed");
    assert_eq!(plain.lower(), input.lower());
    assert_eq!(plain.upper(), input.upper());

    let with_provenance = GraphNetworkCrownExt::crown_backward_with_relaxation_and_provenance(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
    )
    .expect("empty graph provenance CROWN should succeed");
    assert_eq!(with_provenance.bounds.lower(), input.lower());
    assert_eq!(with_provenance.bounds.upper(), input.upper());
    assert_eq!(with_provenance.provenance, BoundsProvenance::Crown);

    let truncated =
        GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
            Some(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .expect("system uptime exceeds 1ms"),
            ),
            Some(0),
        )
        .expect("empty graph truncation path should succeed");
    assert_eq!(truncated.bounds.lower(), input.lower());
    assert_eq!(truncated.bounds.upper(), input.upper());
    assert_eq!(truncated.provenance, BoundsProvenance::Crown);
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_backward_deadline_falls_back_to_exact_ibp_4205() {
    let graph = single_linear_graph();
    let input = test_input();
    let ibp = graph
        .propagate_ibp(&input)
        .expect("IBP baseline should succeed");

    let result = GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("system uptime exceeds 1ms"),
        ),
    )
    .expect("expired deadline should fall back to IBP");

    assert_eq!(
        result.provenance,
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DeadlineExceeded)
    );
    assert_eq!(result.bounds.lower(), ibp.lower());
    assert_eq!(result.bounds.upper(), ibp.upper());
}

/// #dedup-root-collections Fix B: the `_with_node_bounds` variant fed the SAME
/// map the internal Step-1 collection would produce must yield bit-identical
/// output bounds and provenance to the legacy entry point — the precollected
/// path only skips the redundant collection, never changes the backward math.
/// An extra NETWORK_INPUT entry (inserted by the DAG alpha init wiring) must
/// be ignored.
#[ntest::timeout(10000)]
#[test]
fn test_crown_backward_with_precollected_node_bounds_bit_identical() {
    use crate::layers::ReLULayer;
    use crate::network::core::NETWORK_INPUT;

    let mut graph = GraphNetwork::new();
    let lin1 = LinearLayer::new(
        arr2(&[[1.0_f32, -0.5], [0.25, 0.75]]),
        Some(arr1(&[0.1_f32, -0.2])),
    )
    .expect("linear layer should construct");
    let lin2 = LinearLayer::new(arr2(&[[0.5_f32, -1.0]]), Some(arr1(&[0.3_f32])))
        .expect("linear layer should construct");
    graph.add_node(GraphNode::from_input("lin1", Layer::Linear(lin1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer::new()),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(lin2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("lin2");

    let input = test_input();

    // This small ReLU graph selects the per-node CROWN-IBP Step-1 collection;
    // precollect the identical map the legacy path would collect internally.
    assert!(graph.should_collect_per_node_crown_ibp_intermediates());
    let mut precollected = graph
        .collect_crown_ibp_bounds_dag_with_status_and_deadline(&input, None, None)
        .expect("CROWN-IBP collection should succeed")
        .bounds;

    let legacy = GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation(
        &graph,
        &input,
        None,
        MulBinaryRelaxationMode::default(),
        None,
        None,
    )
    .expect("legacy path should succeed");

    let with_bounds =
        GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
            None,
            None,
            Some(&precollected),
        )
        .expect("precollected-bounds path should succeed");

    assert_eq!(with_bounds.provenance, legacy.provenance);
    assert_eq!(with_bounds.bounds.lower(), legacy.bounds.lower());
    assert_eq!(with_bounds.bounds.upper(), legacy.bounds.upper());

    // The DAG alpha init map also carries a NETWORK_INPUT entry — must be inert.
    precollected.insert(NETWORK_INPUT.to_string(), input.clone());
    let with_input_entry =
        GraphNetworkCrownExt::crown_backward_with_relaxation_and_deadline_and_truncation_with_node_bounds(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
            None,
            None,
            Some(&precollected),
        )
        .expect("precollected-bounds path with NETWORK_INPUT entry should succeed");
    assert_eq!(with_input_entry.provenance, legacy.provenance);
    assert_eq!(with_input_entry.bounds.lower(), legacy.bounds.lower());
    assert_eq!(with_input_entry.bounds.upper(), legacy.bounds.upper());
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_backward_spec_wrappers_match_spec_request_builder_4205() {
    let graph = single_linear_graph();
    let input = test_input();
    let spec_matrix = arr2(&[[1.0_f32]]);

    let expected = SpecCrownRequest::new(&graph, &input, &spec_matrix, None)
        .mul_binary_relaxation(MulBinaryRelaxationMode::default())
        .run()
        .expect("spec request builder should succeed");
    let actual = GraphNetworkCrownExt::crown_backward_specs_with_relaxation(
        &graph,
        &input,
        &spec_matrix,
        None,
        MulBinaryRelaxationMode::default(),
    )
    .expect("trait spec wrapper should succeed");

    assert_eq!(actual.lower(), expected.lower());
    assert_eq!(actual.upper(), expected.upper());

    let (expected_bounds, expected_linear) =
        SpecCrownRequest::new(&graph, &input, &spec_matrix, None)
            .mul_binary_relaxation(MulBinaryRelaxationMode::default())
            .run_with_linear()
            .expect("spec request builder should return linear bounds");
    let (actual_bounds, actual_linear) =
        GraphNetworkCrownExt::crown_backward_specs_linear_with_relaxation(
            &graph,
            &input,
            &spec_matrix,
            None,
            MulBinaryRelaxationMode::default(),
        )
        .expect("trait spec-linear wrapper should succeed");

    assert_eq!(actual_bounds.lower(), expected_bounds.lower());
    assert_eq!(actual_bounds.upper(), expected_bounds.upper());

    let expected_linear = expected_linear.expect("builder should capture linear bounds");
    let actual_linear = actual_linear.expect("wrapper should capture linear bounds");
    assert_eq!(actual_linear.lower_a(), expected_linear.lower_a());
    assert_eq!(actual_linear.lower_b(), expected_linear.lower_b());
    assert_eq!(actual_linear.upper_a(), expected_linear.upper_a());
    assert_eq!(actual_linear.upper_b(), expected_linear.upper_b());
}

// #margin-subset-alpha: root CROWN backward margin-subset seeding tests.
mod margin_subset_root_backward {
    use super::GraphNetworkCrownExt;
    use crate::layers::{Layer, LinearLayer, ReLULayer};
    use crate::network::{GraphNetwork, GraphNode};
    use crate::output_margin_seed::MarginOutputSeedGuard;
    use crate::MulBinaryRelaxationMode;
    use ndarray::{arr1, arr2, Array2};
    use ny_tensor::BoundedTensor;

    /// input(2) -> Linear(2->3) "pre" -> ReLU "act" -> Linear(3->600) "out".
    /// 600 outputs put the OUTPUT node at/above the margin-subset engagement
    /// width; the unstable ReLUs make CROWN strictly tighter than IBP.
    fn wide_output_net() -> (GraphNetwork, BoundedTensor) {
        let pre = LinearLayer::new(
            arr2(&[[1.0_f32, -0.5], [0.25, 0.75], [-0.6, 0.4]]),
            Some(arr1(&[0.05_f32, -0.1, 0.02])),
        )
        .expect("pre");
        let weights = Array2::from_shape_fn((600, 3), |(i, j)| {
            let v = ((i * 7 + j * 13) % 11) as f32 / 11.0 - 0.5;
            if v == 0.0 {
                0.3
            } else {
                v
            }
        });
        let out = LinearLayer::new(weights, None).expect("out");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("pre", Layer::Linear(pre)));
        graph.add_node(GraphNode::new(
            "act",
            Layer::ReLU(ReLULayer),
            vec!["pre".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(out),
            vec!["act".to_string()],
        ));
        graph.set_output("out");
        let input = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5]).into_dyn(),
            arr1(&[1.0_f32, 0.75]).into_dyn(),
        )
        .expect("input");
        (graph, input)
    }

    /// With published indices the root CROWN backward's referenced rows are
    /// BIT-IDENTICAL to the full-width backward; every unreferenced row keeps
    /// a sound (equal-or-looser) enclosure of the full-width row.
    #[ntest::timeout(30000)]
    #[test]
    fn root_backward_scatters_published_margin_rows() {
        let (graph, input) = wide_output_net();

        // Full-width reference (no publication on this thread).
        let full = GraphNetworkCrownExt::crown_backward_with_relaxation(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
        )
        .expect("full-width root backward");

        let _guard = MarginOutputSeedGuard::publish(vec![200, 5]);
        let subset = GraphNetworkCrownExt::crown_backward_with_relaxation(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
        )
        .expect("margin-subset root backward");

        assert_eq!(subset.shape(), full.shape());
        for i in 0..600 {
            if i == 5 || i == 200 {
                assert_eq!(
                    subset.lower()[[i]],
                    full.lower()[[i]],
                    "referenced lower row {i} must match the full-width backward"
                );
                assert_eq!(
                    subset.upper()[[i]],
                    full.upper()[[i]],
                    "referenced upper row {i} must match the full-width backward"
                );
            } else {
                // Unreferenced rows keep the node's sound forward enclosure —
                // never tighter than the full-width CROWN row.
                assert!(
                    subset.lower()[[i]] <= full.lower()[[i]],
                    "unreferenced lower row {i} must enclose the full-width row"
                );
                assert!(
                    subset.upper()[[i]] >= full.upper()[[i]],
                    "unreferenced upper row {i} must enclose the full-width row"
                );
            }
        }
    }

    /// Fail-closed: without a publication the behavior is full-width even at
    /// engagement width (no accidental engagement from a stale thread-local).
    #[ntest::timeout(30000)]
    #[test]
    fn root_backward_unpublished_is_full_width() {
        let (graph, input) = wide_output_net();
        let a = GraphNetworkCrownExt::crown_backward_with_relaxation(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
        )
        .expect("run a");
        let b = GraphNetworkCrownExt::crown_backward_with_relaxation(
            &graph,
            &input,
            None,
            MulBinaryRelaxationMode::default(),
        )
        .expect("run b");
        assert_eq!(a.lower(), b.lower());
        assert_eq!(a.upper(), b.upper());
    }
}
