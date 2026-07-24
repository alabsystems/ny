// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness/equivalence tests for upstream-bound inheritance in graph BaB.
//!
//! When a BaB split adds a constraint at a single ReLU node, only nodes
//! downstream of that node can have intermediate bounds that differ from the
//! parent domain. `compute_constrained_forward_bounds` therefore reuses the
//! parent's bounds verbatim for every node NOT in the split node's
//! downstream-reachable set and recomputes only the affected (downstream)
//! nodes.
//!
//! These tests assert that the cached forward pass produces bounds that are
//! EQUAL — element-wise, to within `TOL` — to a full recomputation (every node
//! re-propagated from the inherited seed). Equality must hold for BOTH:
//!   - upstream / sibling-branch nodes that were reused, and
//!   - downstream nodes that were recomputed.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{arr1, arr2};
use ny_test_utils::assert_bounded_tensor_close;

use crate::beta_crown::{GraphNeuronConstraint, GraphSplitHistory};
use crate::{
    AddLayer, BetaCrownConfig, BetaCrownVerifier, BoundedTensor, GraphNetwork, GraphNode, Layer,
    LinearLayer, ReLULayer, NETWORK_INPUT,
};

use super::TOL;

/// Residual DAG with two independent ReLU branches that merge through an Add:
///
/// ```text
///   _input → main_lin → main_relu ↘
///                                  residual (Add) → out_lin
///   _input → skip_lin → skip_relu ↗
/// ```
///
/// Splitting `main_relu` makes `{main_relu, residual, out_lin}` downstream and
/// leaves `{main_lin, skip_lin, skip_relu}` provably unaffected — so the test
/// exercises both reuse (sibling branch + upstream) and recompute (downstream).
fn build_residual_two_branch_graph() -> GraphNetwork {
    let main_lin = LinearLayer::new(arr2(&[[1.0, -0.5], [0.3, 0.8]]), Some(arr1(&[0.1, -0.2])))
        .expect("valid main_lin");
    let skip_lin = LinearLayer::new(arr2(&[[0.2, 2.0], [-1.5, 0.4]]), Some(arr1(&[0.0, 0.5])))
        .expect("valid skip_lin");
    let out_lin = LinearLayer::new(arr2(&[[0.7, -0.3], [1.1, 0.9]]), Some(arr1(&[0.0, 0.0])))
        .expect("valid out_lin");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("main_lin", Layer::Linear(main_lin)));
    graph.add_node(GraphNode::new(
        "main_relu",
        Layer::ReLU(ReLULayer),
        vec!["main_lin".to_string()],
    ));
    graph.add_node(GraphNode::from_input("skip_lin", Layer::Linear(skip_lin)));
    graph.add_node(GraphNode::new(
        "skip_relu",
        Layer::ReLU(ReLULayer),
        vec!["skip_lin".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "residual",
        Layer::Add(AddLayer),
        "main_relu",
        "skip_relu",
    ));
    graph.add_node(GraphNode::new(
        "out_lin",
        Layer::Linear(out_lin),
        vec!["residual".to_string()],
    ));
    graph.set_output("out_lin");
    graph
}

fn input_box() -> BoundedTensor {
    BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
        .expect("valid input box")
}

/// `descendants_inclusive` must return exactly the forward-reachable set from
/// the split node — its membership drives which bounds may be reused.
#[test]
fn descendants_inclusive_partitions_residual_graph() {
    let graph = build_residual_two_branch_graph();

    let down = graph
        .descendants_inclusive(&["main_relu".to_string()])
        .expect("descendants");
    // Downstream of main_relu: itself + residual + out_lin.
    assert!(down.contains("main_relu"));
    assert!(down.contains("residual"));
    assert!(down.contains("out_lin"));
    // Upstream / sibling branch: NOT downstream.
    assert!(!down.contains("main_lin"));
    assert!(!down.contains("skip_lin"));
    assert!(!down.contains("skip_relu"));

    // A NETWORK_INPUT seed marks everything downstream (whole net depends on it).
    let all = graph
        .descendants_inclusive(&[NETWORK_INPUT.to_string()])
        .expect("descendants from input");
    for name in [
        "main_lin",
        "main_relu",
        "skip_lin",
        "skip_relu",
        "residual",
        "out_lin",
    ] {
        assert!(all.contains(name), "input must reach {name}");
    }
}

/// Cached child forward bounds (upstream inheritance ON) must equal the full
/// recomputation (inheritance OFF) for EVERY node — element-wise to `TOL`.
#[test]
fn upstream_cache_matches_full_recompute_active_split() {
    assert_cache_equals_full_recompute(true);
}

/// Same equivalence guarantee for the inactive (x < 0) branch of the split.
#[test]
fn upstream_cache_matches_full_recompute_inactive_split() {
    assert_cache_equals_full_recompute(false);
}

fn assert_cache_equals_full_recompute(is_active: bool) {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_residual_two_branch_graph();
    let input = input_box();

    // Parent domain bounds: unconstrained root forward pass.
    let (parent_cache, _) = verifier
        .compute_constrained_forward_bounds(&graph, &input, &GraphSplitHistory::new(), None, None)
        .expect("parent forward bounds");
    // #cone-delta increment 2: the forward cache is already Arc-shared.
    let parent_seed: HashMap<String, Arc<BoundedTensor>> = parent_cache;

    // Child adds exactly one split constraint on main_relu[0].
    let child_history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "main_relu".to_string(),
        neuron_idx: 0,
        is_active,
        score: 0.0,
    });

    // Cached path (upstream inheritance ON).
    let (cached, cached_input) = verifier
        .compute_constrained_forward_bounds(
            &graph,
            &input,
            &child_history,
            Some(&parent_seed),
            None,
        )
        .expect("cached child forward bounds");

    // Reference path: same seed + history, inheritance OFF (recompute every node).
    let (full, full_input) = verifier
        .compute_constrained_forward_bounds_inner(
            &graph,
            &input,
            &child_history,
            Some(&parent_seed),
            None,
            false,
        )
        .expect("full child forward bounds");

    assert_bounded_tensor_close(&cached_input, &full_input, TOL, "constrained_input");
    assert_eq!(
        cached.len(),
        full.len(),
        "cache node-count mismatch: cached={}, full={}",
        cached.len(),
        full.len()
    );

    // The split is applied as a pre-activation tightening on `main_lin` (the
    // input feeding `main_relu`), so the affected set is seeded there — see the
    // soundness note in `compute_constrained_forward_bounds_inner`.
    let downstream = graph
        .descendants_inclusive(&["main_lin".to_string()])
        .expect("descendants");
    // The pre-activation node itself must be inside the affected set.
    assert!(downstream.contains("main_lin"));
    assert!(downstream.contains("main_relu"));

    for (node, full_bounds) in &full {
        let cached_bounds = cached
            .get(node)
            .unwrap_or_else(|| panic!("cached cache missing node '{node}'"));
        // Equivalence must hold for BOTH reused (upstream) and recomputed
        // (downstream) nodes.
        assert_bounded_tensor_close(cached_bounds, full_bounds, TOL, node);

        // Sanity: reused nodes must be byte-for-byte identical to the parent
        // seed (the optimization simply keeps the seed for them).
        if !downstream.contains(node.as_str()) {
            let parent = parent_seed
                .get(node)
                .unwrap_or_else(|| panic!("parent seed missing reused node '{node}'"));
            assert_bounded_tensor_close(
                cached_bounds,
                parent.as_ref(),
                TOL,
                &format!("{node} (reused == parent)"),
            );
        }
    }
}
