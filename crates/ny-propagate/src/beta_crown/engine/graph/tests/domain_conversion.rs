// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for graph domain conversion utilities.
//!
//! Extracted from `domain_conversion.rs` inline `#[cfg(test)]` block.
//! Part of #1876.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{arr1, arr2, Array1};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::batched_domain::{CachedLinearBounds, DomainMetadata, PickedDomains, ProcessedDomains};
use crate::beta_crown::branching::{GenBabConstraint, GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::engine::graph::domain_conversion::{
    branch_relu_from_picked, graph_domain_from_picked, processed_from_graph_domains_direct,
    processed_from_graph_domains_with_la,
};
use crate::beta_crown::state::GraphBetaState;
use crate::beta_crown::GraphBabDomain;
use crate::{GraphNetwork, GraphNode, Layer, ReLULayer};

fn make_bounded_tensor(lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
    BoundedTensor::new(
        Array1::from_vec(lower).into_dyn(),
        Array1::from_vec(upper).into_dyn(),
    )
    .expect("test bounds must be valid")
}

fn make_domain(
    layer_a: (Vec<f32>, Vec<f32>),
    layer_b: (Vec<f32>, Vec<f32>),
    input: (Vec<f32>, Vec<f32>),
    lower_bound: f32,
    upper_bound: f32,
    depth: usize,
    history: GraphSplitHistory,
) -> GraphBabDomain {
    let mut node_bounds = HashMap::new();
    node_bounds.insert(
        "layer_a".to_string(),
        Arc::new(make_bounded_tensor(layer_a.0, layer_a.1)),
    );
    node_bounds.insert(
        "layer_b".to_string(),
        Arc::new(make_bounded_tensor(layer_b.0, layer_b.1)),
    );
    let input_bounds = Arc::new(make_bounded_tensor(input.0, input.1));

    GraphBabDomain {
        beta_state: GraphBetaState::from_history(&history).unwrap(),
        alpha_state: crate::beta_crown::state::GraphDomainAlphaState::empty(),
        history,
        node_bounds,
        lower_bound,
        upper_bound,
        depth,
        priority: upper_bound,
        input_bounds,
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    }
}

fn make_cached_la(seed: f32) -> CachedLinearBounds {
    let mut lower_a = HashMap::new();
    lower_a.insert("layer_a".to_string(), arr2(&[[seed, seed + 1.0]]));

    let mut upper_a = HashMap::new();
    upper_a.insert("layer_a".to_string(), arr2(&[[seed + 2.0, seed + 3.0]]));

    let mut lower_b = HashMap::new();
    lower_b.insert("layer_a".to_string(), arr1(&[seed + 4.0]));

    let mut upper_b = HashMap::new();
    upper_b.insert("layer_a".to_string(), arr1(&[seed + 5.0]));

    CachedLinearBounds {
        lower_a,
        upper_a,
        lower_b,
        upper_b,
    }
}

fn assert_cached_la_equal(
    left: Option<&CachedLinearBounds>,
    right: Option<&CachedLinearBounds>,
    idx: usize,
) {
    match (left, right) {
        (None, None) => {}
        (Some(left_cache), Some(right_cache)) => {
            assert_eq!(
                left_cache.lower_a.len(),
                right_cache.lower_a.len(),
                "metadata[{idx}] lower_a count mismatch"
            );
            assert_eq!(
                left_cache.upper_a.len(),
                right_cache.upper_a.len(),
                "metadata[{idx}] upper_a count mismatch"
            );
            for (name, left_arr) in &left_cache.lower_a {
                assert_eq!(
                    Some(left_arr),
                    right_cache.lower_a.get(name),
                    "metadata[{idx}] lower_a mismatch at layer {name}"
                );
            }
            for (name, left_arr) in &left_cache.upper_a {
                assert_eq!(
                    Some(left_arr),
                    right_cache.upper_a.get(name),
                    "metadata[{idx}] upper_a mismatch at layer {name}"
                );
            }
            for (name, left_arr) in &left_cache.lower_b {
                assert_eq!(
                    Some(left_arr),
                    right_cache.lower_b.get(name),
                    "metadata[{idx}] lower_b mismatch at layer {name}"
                );
            }
            for (name, left_arr) in &left_cache.upper_b {
                assert_eq!(
                    Some(left_arr),
                    right_cache.upper_b.get(name),
                    "metadata[{idx}] upper_b mismatch at layer {name}"
                );
            }
        }
        _ => panic!("metadata[{idx}] cached_la presence mismatch"),
    }
}

fn assert_processed_domains_equal(
    left: &ProcessedDomains,
    right: &ProcessedDomains,
    layer_names: &[String],
) {
    assert_eq!(left.global_lbs, right.global_lbs);
    assert_eq!(left.global_ubs, right.global_ubs);
    assert_eq!(left.keep_mask, right.keep_mask);
    assert_eq!(left.input_lowers, right.input_lowers);
    assert_eq!(left.input_uppers, right.input_uppers);

    for layer_name in layer_names {
        assert_eq!(
            left.layer_lowers.get(layer_name),
            right.layer_lowers.get(layer_name),
            "layer_lowers mismatch for {layer_name}"
        );
        assert_eq!(
            left.layer_uppers.get(layer_name),
            right.layer_uppers.get(layer_name),
            "layer_uppers mismatch for {layer_name}"
        );
    }

    assert_eq!(left.metadata.len(), right.metadata.len());
    for (idx, (left_meta, right_meta)) in left.metadata.iter().zip(&right.metadata).enumerate() {
        assert_eq!(left_meta.lower_bound, right_meta.lower_bound);
        assert_eq!(left_meta.upper_bound, right_meta.upper_bound);
        assert_eq!(left_meta.depth, right_meta.depth);
        assert_eq!(left_meta.constraints, right_meta.constraints);
        assert_cached_la_equal(
            left_meta.cached_la.as_deref(),
            right_meta.cached_la.as_deref(),
            idx,
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_processed_from_graph_domains_direct_matches_legacy_path() -> Result<()> {
    let layer_names = vec!["layer_a".to_string(), "layer_b".to_string()];

    let mut history0 = GraphSplitHistory::new();
    history0.add_constraint(GraphNeuronConstraint {
        node_name: "relu_0".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.8,
    });
    history0.add_genbab_constraints_for_split(vec![
        GenBabConstraint::new("gelu_0".to_string(), 1, -0.25, true, 0.4).unwrap(),
        GenBabConstraint::new("gelu_0".to_string(), 1, 0.5, false, 0.4).unwrap(),
    ]);
    history0.add_constraint(GraphNeuronConstraint {
        node_name: "relu_1".to_string(),
        neuron_idx: 2,
        is_active: false,
        score: 0.2,
    });

    let mut history1 = GraphSplitHistory::new();
    history1.add_genbab_constraint(
        GenBabConstraint::new("sigmoid_0".to_string(), 3, 0.1, true, 0.6).unwrap(),
    );
    history1.add_constraint(GraphNeuronConstraint {
        node_name: "relu_3".to_string(),
        neuron_idx: 1,
        is_active: true,
        score: 0.1,
    });

    let domains = vec![
        make_domain(
            (vec![-1.0, 0.2], vec![0.1, 1.3]),
            (vec![-0.5, -0.1], vec![0.5, 0.7]),
            (vec![-2.0, -1.0], vec![2.0, 1.0]),
            -0.4,
            1.6,
            3,
            history0,
        ),
        make_domain(
            (vec![-0.8, 0.0], vec![0.2, 1.4]),
            (vec![-0.3, 0.2], vec![0.9, 1.2]),
            (vec![-1.5, -0.5], vec![1.5, 0.5]),
            -0.1,
            1.1,
            2,
            history1,
        ),
    ];

    let legacy = processed_from_graph_domains_with_la(&domains, &layer_names, true, None)?;
    let direct = processed_from_graph_domains_direct(&domains, &layer_names, None)?;

    assert_processed_domains_equal(&legacy, &direct, &layer_names);
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_processed_from_graph_domains_with_la_direct_dispatch() -> Result<()> {
    let layer_names = vec!["layer_a".to_string(), "layer_b".to_string()];

    let domains = vec![
        make_domain(
            (vec![-1.0, 0.1], vec![0.4, 1.0]),
            (vec![-0.2, -0.3], vec![0.6, 0.8]),
            (vec![-1.2, -0.4], vec![1.2, 0.4]),
            -0.3,
            1.0,
            1,
            GraphSplitHistory::new(),
        ),
        make_domain(
            (vec![-0.9, 0.2], vec![0.3, 1.2]),
            (vec![-0.1, -0.2], vec![0.7, 0.9]),
            (vec![-1.1, -0.6], vec![1.1, 0.6]),
            -0.2,
            0.9,
            1,
            GraphSplitHistory::new(),
        ),
    ];

    let cached_la = vec![
        Arc::new(make_cached_la(1.0)),
        Arc::new(make_cached_la(10.0)),
    ];
    let direct =
        processed_from_graph_domains_direct(&domains, &layer_names, Some(cached_la.clone()))?;
    let dispatched =
        processed_from_graph_domains_with_la(&domains, &layer_names, false, Some(cached_la))?;

    assert_processed_domains_equal(&direct, &dispatched, &layer_names);
    assert_eq!(
        dispatched.metadata[0]
            .cached_la
            .as_ref()
            .expect("cached la for domain 0")
            .lower_a
            .get("layer_a"),
        Some(&arr2(&[[1.0, 2.0]]))
    );
    assert_eq!(
        dispatched.metadata[1]
            .cached_la
            .as_ref()
            .expect("cached la for domain 1")
            .upper_a
            .get("layer_a"),
        Some(&arr2(&[[12.0, 13.0]]))
    );

    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_processed_from_graph_domains_direct_missing_layer_bounds() {
    let mut node_bounds = HashMap::new();
    node_bounds.insert(
        "layer_a".to_string(),
        Arc::new(make_bounded_tensor(vec![-1.0, 0.0], vec![0.5, 1.0])),
    );
    let history = GraphSplitHistory::new();
    let domain = GraphBabDomain {
        beta_state: GraphBetaState::empty(),
        alpha_state: crate::beta_crown::state::GraphDomainAlphaState::empty(),
        history,
        node_bounds,
        lower_bound: -0.2,
        upper_bound: 0.8,
        depth: 0,
        priority: 0.8,
        input_bounds: Arc::new(make_bounded_tensor(vec![-1.0, -1.0], vec![1.0, 1.0])),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };
    let layer_names = vec!["layer_a".to_string(), "layer_b".to_string()];

    let err = processed_from_graph_domains_direct(&[domain], &layer_names, None)
        .expect_err("missing layer bounds should fail");
    assert!(
        err.to_string()
            .contains("Missing bounds for layer 'layer_b'"),
        "unexpected error: {err}"
    );
}

/// Verify that `branch_relu_from_picked` produces identical children to the old path:
/// `graph_domain_from_picked` → `with_constraint` (active) + `with_constraint` (inactive).
///
/// This is the parity test for Direction 2 of #1668.
#[ntest::timeout(10000)]
#[test]
fn test_branch_relu_from_picked_matches_legacy_path() -> Result<()> {
    use crate::LinearLayer;
    use ndarray::{arr2, ArrayD as AD, IxDyn as Ix};

    // Build a minimal graph: input(2) -> linear(2->2) -> relu
    let mut graph = GraphNetwork::new();
    let weight = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let linear = LinearLayer::new(weight, None)?;
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    let layer_names = vec!["linear".to_string()];

    // Build PickedDomains with 2 domains. Pre-activation layer "linear" has
    // mixed positive/negative values so both active and inactive branches are feasible.
    let mut layer_lowers = HashMap::new();
    let mut layer_uppers = HashMap::new();
    // Domain 0: linear bounds [-1.0, 0.3], Domain 1: linear bounds [-0.5, -0.2]
    layer_lowers.insert(
        "linear".to_string(),
        AD::<f32>::from_shape_vec(Ix(&[2, 2]), vec![-1.0, 0.3, -0.5, -0.2]).unwrap(),
    );
    layer_uppers.insert(
        "linear".to_string(),
        AD::<f32>::from_shape_vec(Ix(&[2, 2]), vec![0.5, 1.0, 0.8, 0.1]).unwrap(),
    );

    let picked = PickedDomains {
        batch_size: 2,
        layer_lowers,
        layer_uppers,
        input_lowers: AD::from_shape_vec(Ix(&[2, 2]), vec![-2.0, -1.0, -1.5, -0.5]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[2, 2]), vec![2.0, 1.0, 1.5, 0.5]).unwrap(),
        global_lbs: vec![-0.4, -0.1],
        global_ubs: vec![1.6, 1.1],
        metadata: vec![
            DomainMetadata {
                lower_bound: -0.4,
                upper_bound: 1.6,
                depth: 1,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                node_bounds_override: None,
                alpha_state: None,
            },
            DomainMetadata {
                lower_bound: -0.1,
                upper_bound: 1.1,
                depth: 2,
                constraints: vec![
                    // Existing ReLU constraint on relu node, neuron 1
                    ("relu".to_string(), 1, true, None),
                ],
                cached_la: None,
                needs_bounding: false,
                node_bounds_override: None,
                alpha_state: None,
            },
        ],
    };

    // Branch domain 0 on relu neuron 0 (pre-activation: linear neuron 0, l=-1.0 u=0.5)
    // Both active (u>=0) and inactive (l<=0) should be feasible.
    let (fast_active, fast_inactive, had_propagation_failure) =
        branch_relu_from_picked(0, &picked, &graph, "relu", 0, 0.7, &layer_names, false)?;
    assert!(
        !had_propagation_failure,
        "valid branch should not report propagation failure"
    );

    // Old path: materialize parent, then branch
    let parent = graph_domain_from_picked(0, &picked, &layer_names, false, None)?;
    let active_constraint = GraphNeuronConstraint {
        node_name: "relu".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.7,
    };
    let legacy_active = parent.with_constraint(&graph, active_constraint, false)?;
    let inactive_constraint = GraphNeuronConstraint {
        node_name: "relu".to_string(),
        neuron_idx: 0,
        is_active: false,
        score: 0.7,
    };
    let legacy_inactive = parent.with_constraint(&graph, inactive_constraint, false)?;

    // Both paths should produce feasible children
    assert!(
        fast_active.is_some(),
        "fast active child should be feasible"
    );
    assert!(
        fast_inactive.is_some(),
        "fast inactive child should be feasible"
    );
    assert!(
        legacy_active.is_some(),
        "legacy active child should be feasible"
    );
    assert!(
        legacy_inactive.is_some(),
        "legacy inactive child should be feasible"
    );

    let fa = fast_active.unwrap();
    let fi = fast_inactive.unwrap();
    let la = legacy_active.unwrap();
    let li = legacy_inactive.unwrap();

    // Compare scalar fields
    assert_eq!(fa.lower_bound, la.lower_bound, "active lower_bound");
    assert_eq!(fa.upper_bound, la.upper_bound, "active upper_bound");
    assert_eq!(fa.depth, la.depth, "active depth");
    assert_eq!(fi.lower_bound, li.lower_bound, "inactive lower_bound");
    assert_eq!(fi.upper_bound, li.upper_bound, "inactive upper_bound");
    assert_eq!(fi.depth, li.depth, "inactive depth");

    // Compare node_bounds
    for name in &layer_names {
        let fa_b = fa.node_bounds.get(name).expect("fast active bounds");
        let la_b = la.node_bounds.get(name).expect("legacy active bounds");
        assert_eq!(fa_b.lower(), la_b.lower(), "active {name} lower");
        assert_eq!(fa_b.upper(), la_b.upper(), "active {name} upper");

        let fi_b = fi.node_bounds.get(name).expect("fast inactive bounds");
        let li_b = li.node_bounds.get(name).expect("legacy inactive bounds");
        assert_eq!(fi_b.lower(), li_b.lower(), "inactive {name} lower");
        assert_eq!(fi_b.upper(), li_b.upper(), "inactive {name} upper");
    }

    // Compare input bounds
    assert_eq!(
        fa.input_bounds.lower(),
        la.input_bounds.lower(),
        "active input lower"
    );
    assert_eq!(
        fa.input_bounds.upper(),
        la.input_bounds.upper(),
        "active input upper"
    );
    assert_eq!(
        fi.input_bounds.lower(),
        li.input_bounds.lower(),
        "inactive input lower"
    );
    assert_eq!(
        fi.input_bounds.upper(),
        li.input_bounds.upper(),
        "inactive input upper"
    );

    // Compare histories: same number of constraints
    assert_eq!(
        fa.history.constraints.len(),
        la.history.constraints.len(),
        "active constraint count"
    );
    assert_eq!(
        fi.history.constraints.len(),
        li.history.constraints.len(),
        "inactive constraint count"
    );

    // Now test domain 1 with existing constraints, branching on neuron 1
    // Domain 1 pre-activation neuron 1: l=-0.2, u=0.1 — both branches feasible
    let (fast_active2, fast_inactive2, had_propagation_failure2) =
        branch_relu_from_picked(1, &picked, &graph, "relu", 1, 0.3, &layer_names, false)?;
    assert!(
        !had_propagation_failure2,
        "valid branch should not report propagation failure"
    );
    let parent2 = graph_domain_from_picked(1, &picked, &layer_names, false, None)?;
    let legacy_active2 = parent2.with_constraint(
        &graph,
        GraphNeuronConstraint {
            node_name: "relu".to_string(),
            neuron_idx: 1,
            is_active: true,
            score: 0.3,
        },
        false,
    )?;
    let legacy_inactive2 = parent2.with_constraint(
        &graph,
        GraphNeuronConstraint {
            node_name: "relu".to_string(),
            neuron_idx: 1,
            is_active: false,
            score: 0.3,
        },
        false,
    )?;

    assert!(fast_active2.is_some() && legacy_active2.is_some());
    assert!(fast_inactive2.is_some() && legacy_inactive2.is_some());

    let fa2 = fast_active2.unwrap();
    let la2 = legacy_active2.unwrap();
    assert_eq!(fa2.depth, la2.depth, "domain 1 active depth");
    assert_eq!(
        fa2.history.constraints.len(),
        la2.history.constraints.len(),
        "domain 1 active constraint count"
    );
    // Domain 1 had 1 existing constraint + 1 new = 2 total
    assert_eq!(fa2.history.constraints.len(), 2);

    Ok(())
}

/// Test that `branch_relu_from_picked` correctly returns None for infeasible branches.
#[ntest::timeout(5000)]
#[test]
fn test_branch_relu_from_picked_infeasible() -> Result<()> {
    use crate::LinearLayer;
    use ndarray::{arr2, ArrayD as AD, IxDyn as Ix};

    let mut graph = GraphNetwork::new();
    let weight = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    let linear = LinearLayer::new(weight, None)?;
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    let layer_names = vec!["linear".to_string()];

    // Domain with all-positive pre-activation: l=0.5, u=1.0
    // active is feasible (u >= 0), inactive is INfeasible (l > 0)
    let mut layer_lowers = HashMap::new();
    let mut layer_uppers = HashMap::new();
    layer_lowers.insert(
        "linear".to_string(),
        AD::<f32>::from_shape_vec(Ix(&[1, 2]), vec![0.5, 0.1]).unwrap(),
    );
    layer_uppers.insert(
        "linear".to_string(),
        AD::<f32>::from_shape_vec(Ix(&[1, 2]), vec![1.0, 0.8]).unwrap(),
    );

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers,
        layer_uppers,
        input_lowers: AD::from_shape_vec(Ix(&[1, 2]), vec![-1.0, -1.0]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 2]), vec![1.0, 1.0]).unwrap(),
        global_lbs: vec![-0.5],
        global_ubs: vec![0.5],
        metadata: vec![DomainMetadata {
            lower_bound: -0.5,
            upper_bound: 0.5,
            depth: 0,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
    };

    // Branch on neuron 0 (l=0.5 > 0): inactive should be infeasible
    let (active, inactive, had_propagation_failure) =
        branch_relu_from_picked(0, &picked, &graph, "relu", 0, 0.5, &layer_names, false)?;
    assert!(
        !had_propagation_failure,
        "infeasible branch pruning should not be marked as propagation failure"
    );
    assert!(
        active.is_some(),
        "active branch should be feasible (u=1.0 >= 0)"
    );
    assert!(
        inactive.is_none(),
        "inactive branch should be infeasible (l=0.5 > 0)"
    );

    // Verify the legacy path agrees
    let parent = graph_domain_from_picked(0, &picked, &layer_names, false, None)?;
    let legacy_inactive = parent.with_constraint(
        &graph,
        GraphNeuronConstraint {
            node_name: "relu".to_string(),
            neuron_idx: 0,
            is_active: false,
            score: 0.5,
        },
        false,
    )?;
    assert!(
        legacy_inactive.is_none(),
        "legacy inactive should also be infeasible"
    );

    Ok(())
}

/// Regression for #2784: parent input bound materialization failures (for example
/// NaN-contaminated picked input bounds) must report propagation failure instead
/// of aborting the whole GPU batch.
#[ntest::timeout(5000)]
#[test]
fn test_branch_relu_from_picked_parent_input_nan_marks_propagation_failure_2784() -> Result<()> {
    use ndarray::{ArrayD as AD, IxDyn as Ix};

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");
    let layer_names: Vec<String> = vec![];

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: AD::from_shape_vec(Ix(&[1, 1]), vec![f32::NAN]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 1]), vec![1.0]).unwrap(),
        global_lbs: vec![-1.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -1.0,
            upper_bound: 1.0,
            depth: 0,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
    };

    let (active, inactive, had_propagation_failure) =
        branch_relu_from_picked(0, &picked, &graph, "relu", 0, 0.5, &layer_names, false)?;
    assert!(
        active.is_none(),
        "active child should be dropped on NaN input"
    );
    assert!(
        inactive.is_none(),
        "inactive child should be dropped on NaN input"
    );
    assert!(
        had_propagation_failure,
        "NaN parent input bounds must mark propagation failure"
    );
    Ok(())
}

/// Regression for #2784: layer bound materialization failures must report
/// propagation failure (Unknown), not bubble as hard errors from fast-path
/// branching.
#[ntest::timeout(5000)]
#[test]
fn test_branch_relu_from_picked_layer_bounds_nan_marks_propagation_failure_2784() -> Result<()> {
    use crate::LinearLayer;
    use ndarray::{arr2, ArrayD as AD, IxDyn as Ix};

    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(arr2(&[[1.0_f32]]), None)?;
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");
    let layer_names = vec!["linear".to_string()];

    let mut layer_lowers = HashMap::new();
    let mut layer_uppers = HashMap::new();
    layer_lowers.insert(
        "linear".to_string(),
        AD::from_shape_vec(Ix(&[1, 1]), vec![f32::NAN]).unwrap(),
    );
    layer_uppers.insert(
        "linear".to_string(),
        AD::from_shape_vec(Ix(&[1, 1]), vec![1.0]).unwrap(),
    );
    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers,
        layer_uppers,
        input_lowers: AD::from_shape_vec(Ix(&[1, 1]), vec![-1.0]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 1]), vec![1.0]).unwrap(),
        global_lbs: vec![-1.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -1.0,
            upper_bound: 1.0,
            depth: 0,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
    };

    let (active, inactive, had_propagation_failure) =
        branch_relu_from_picked(0, &picked, &graph, "relu", 0, 0.5, &layer_names, false)?;
    assert!(
        active.is_none(),
        "active child should be dropped on NaN layer bounds"
    );
    assert!(
        inactive.is_none(),
        "inactive child should be dropped on NaN layer bounds"
    );
    assert!(
        had_propagation_failure,
        "NaN layer bounds must mark propagation failure"
    );
    Ok(())
}

/// Regression for #2934: when BOTH neuron_lower and neuron_upper are NaN,
/// IEEE 754 makes both `neuron_upper >= 0.0` and `neuron_lower <= 0.0` return
/// false. Before the fix, this triggered the `!active && !inactive` path
/// returning `(None, None, false)` — silently dropping the domain without
/// setting `had_propagation_failure`. This could cause false Verified results
/// because the caller wouldn't know a domain was lost.
#[ntest::timeout(5000)]
#[test]
fn test_branch_relu_from_picked_nan_neuron_bounds_marks_propagation_failure_2934() -> Result<()> {
    use crate::LinearLayer;
    use ndarray::{arr2, ArrayD as AD, IxDyn as Ix};

    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]), None)?;
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");
    let layer_names = vec!["linear".to_string()];

    // Neuron 0 has NaN bounds (both lower and upper), neuron 1 is valid.
    // This is the exact scenario from #2934: NaN neuron bounds cause both
    // feasibility checks to fail via IEEE 754 comparison semantics.
    let mut layer_lowers = HashMap::new();
    let mut layer_uppers = HashMap::new();
    layer_lowers.insert(
        "linear".to_string(),
        AD::<f32>::from_shape_vec(Ix(&[1, 2]), vec![f32::NAN, -0.5]).unwrap(),
    );
    layer_uppers.insert(
        "linear".to_string(),
        AD::<f32>::from_shape_vec(Ix(&[1, 2]), vec![f32::NAN, 0.5]).unwrap(),
    );
    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers,
        layer_uppers,
        input_lowers: AD::from_shape_vec(Ix(&[1, 2]), vec![-1.0, -1.0]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 2]), vec![1.0, 1.0]).unwrap(),
        global_lbs: vec![-1.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -1.0,
            upper_bound: 1.0,
            depth: 0,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
    };

    // Branch on neuron 0 which has NaN bounds. Before fix: would return
    // (None, None, false) — silent domain loss. After fix: (None, None, true).
    let (active, inactive, had_propagation_failure) =
        branch_relu_from_picked(0, &picked, &graph, "relu", 0, 0.5, &layer_names, false)?;
    assert!(
        active.is_none(),
        "active child should be None for NaN neuron bounds"
    );
    assert!(
        inactive.is_none(),
        "inactive child should be None for NaN neuron bounds"
    );
    assert!(
        had_propagation_failure,
        "NaN neuron bounds MUST set had_propagation_failure=true to prevent silent domain loss"
    );

    // Also test with Inf bounds — same issue applies.
    let mut layer_lowers_inf = HashMap::new();
    let mut layer_uppers_inf = HashMap::new();
    layer_lowers_inf.insert(
        "linear".to_string(),
        AD::<f32>::from_shape_vec(Ix(&[1, 2]), vec![f32::NEG_INFINITY, -0.5]).unwrap(),
    );
    layer_uppers_inf.insert(
        "linear".to_string(),
        AD::<f32>::from_shape_vec(Ix(&[1, 2]), vec![f32::INFINITY, 0.5]).unwrap(),
    );
    let picked_inf = PickedDomains {
        batch_size: 1,
        layer_lowers: layer_lowers_inf,
        layer_uppers: layer_uppers_inf,
        input_lowers: AD::from_shape_vec(Ix(&[1, 2]), vec![-1.0, -1.0]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 2]), vec![1.0, 1.0]).unwrap(),
        global_lbs: vec![-1.0],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -1.0,
            upper_bound: 1.0,
            depth: 0,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
    };

    let (active_inf, inactive_inf, had_failure_inf) =
        branch_relu_from_picked(0, &picked_inf, &graph, "relu", 0, 0.5, &layer_names, false)?;
    assert!(
        active_inf.is_none(),
        "Inf bounds should not produce children"
    );
    assert!(
        inactive_inf.is_none(),
        "Inf bounds should not produce children"
    );
    assert!(
        had_failure_inf,
        "Inf neuron bounds MUST set had_propagation_failure=true"
    );

    Ok(())
}

/// Verify that cached lA survives the full DomainList round-trip:
/// add (with cached_la) → pick_out → graph_domain_from_picked → verify cached_la present.
///
/// This is Direction 4 of #1669: ensure the lA reuse pipeline is wired end-to-end.
#[test]
fn test_cached_la_round_trip_through_domain_list() {
    use crate::batched_domain::{
        CachedLinearBounds, DomainList, DomainListConfig, ProcessedDomains,
    };
    use ndarray::{Array2, ArrayD as AD, IxDyn as Ix};

    let layer_names = vec!["layer_a".to_string()];

    // Create a CachedLinearBounds with known data (A matrices + bias)
    let mut lower_a = HashMap::new();
    lower_a.insert(
        "layer_a".to_string(),
        Array2::from_shape_vec((1, 2), vec![0.5, -0.3]).unwrap(),
    );
    let mut upper_a = HashMap::new();
    upper_a.insert(
        "layer_a".to_string(),
        Array2::from_shape_vec((1, 2), vec![0.7, 0.2]).unwrap(),
    );
    let mut lower_b = HashMap::new();
    lower_b.insert("layer_a".to_string(), Array1::from_vec(vec![0.1]));
    let mut upper_b = HashMap::new();
    upper_b.insert("layer_a".to_string(), Array1::from_vec(vec![-0.1]));
    let cached = CachedLinearBounds {
        lower_a,
        upper_a,
        lower_b,
        upper_b,
    };

    // Build ProcessedDomains with the cached lA
    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "layer_a".to_string(),
        AD::from_shape_vec(Ix(&[1, 2]), vec![-1.0, -0.5]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "layer_a".to_string(),
        AD::from_shape_vec(Ix(&[1, 2]), vec![1.0, 0.5]).unwrap(),
    );
    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: AD::from_shape_vec(Ix(&[1, 2]), vec![-1.0, -1.0]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 2]), vec![1.0, 1.0]).unwrap(),
        global_lbs: vec![-0.5],
        global_ubs: vec![0.5],
        metadata: vec![DomainMetadata {
            lower_bound: -0.5,
            upper_bound: 0.5,
            depth: 1,
            constraints: vec![],
            cached_la: Some(Arc::new(cached)),
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
        keep_mask: vec![true],
    };

    // Add to DomainList and pick_out
    let mut layer_shapes = HashMap::new();
    layer_shapes.insert("layer_a".to_string(), vec![2usize]);
    let config = DomainListConfig {
        layer_names: layer_names.clone(),
        layer_shapes,
        input_shape: vec![2],
        ..DomainListConfig::default()
    };
    let mut domain_list = DomainList::new(config).expect("DomainList creation should succeed");
    domain_list.add(processed).expect("add should succeed");

    let picked = domain_list
        .pick_out_batched(
            1,
            crate::batched_domain::BatchedDomainOptions {
                enable_interm_transfer: false,
            },
        )
        .unwrap();
    assert_eq!(picked.batch_size, 1, "should pick 1 domain");

    // Reconstruct GraphBabDomain and verify cached_la
    let domain = graph_domain_from_picked(0, &picked, &layer_names, false, None)
        .expect("domain reconstruction should succeed");

    assert!(
        domain.cached_la.is_some(),
        "cached_la should survive the DomainList round-trip"
    );

    let la = domain.cached_la.as_ref().unwrap();
    assert_eq!(la.lower_a.len(), 1, "should have 1 layer cached");
    assert!(
        la.lower_a.contains_key("layer_a"),
        "should have layer_a cached"
    );

    let lower_a_mat = la.lower_a.get("layer_a").unwrap();
    assert_eq!(lower_a_mat.shape(), &[1, 2], "cached lA shape should match");
    assert!(
        (lower_a_mat[[0, 0]] - 0.5).abs() < 1e-6,
        "cached lA value should survive round-trip"
    );
    assert!(
        (lower_a_mat[[0, 1]] - (-0.3)).abs() < 1e-6,
        "cached lA value should survive round-trip"
    );

    // Verify bias terms survive the round-trip
    let lower_b_vec = la.lower_b.get("layer_a").unwrap();
    assert_eq!(lower_b_vec.len(), 1, "cached lower_b shape should match");
    assert!(
        (lower_b_vec[0] - 0.1).abs() < 1e-6,
        "cached lower_b value should survive round-trip"
    );
    let upper_b_vec = la.upper_b.get("layer_a").unwrap();
    assert_eq!(upper_b_vec.len(), 1, "cached upper_b shape should match");
    assert!(
        (upper_b_vec[0] - (-0.1)).abs() < 1e-6,
        "cached upper_b value should survive round-trip"
    );
}

/// Regression for #1845: `graph_domain_from_picked` must initialize non-empty
/// alpha state when graph context is available.
#[test]
fn test_graph_domain_from_picked_initializes_alpha_from_graph_1845() {
    use ndarray::{ArrayD as AD, IxDyn as Ix};

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let layer_names: Vec<String> = vec![];
    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: AD::from_shape_vec(Ix(&[1, 2]), vec![-1.0, -0.8]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 2]), vec![2.0, 1.5]).unwrap(),
        global_lbs: vec![-1.0],
        global_ubs: vec![2.0],
        metadata: vec![DomainMetadata {
            lower_bound: -1.0,
            upper_bound: 2.0,
            depth: 0,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
    };

    let domain = graph_domain_from_picked(0, &picked, &layer_names, false, Some(&graph))
        .expect("domain reconstruction should succeed");

    assert!(
        !domain.alpha_state.is_empty(),
        "alpha state should be initialized from graph bounds"
    );
    assert!(
        domain.alpha_state.neuron("relu", 0).is_some(),
        "first unstable neuron should have alpha entry"
    );
    assert!(
        domain.alpha_state.neuron("relu", 1).is_some(),
        "second unstable neuron should have alpha entry"
    );
}

/// Regression for #1845: `branch_relu_from_picked` must warm-start child alpha
/// from parent metadata so optimized alpha survives across BaB iterations.
#[test]
fn test_branch_relu_from_picked_warm_starts_alpha_1845() -> Result<()> {
    use crate::beta_crown::state::{AlphaNeuronState, GraphDomainAlphaState};
    use ndarray::{ArrayD as AD, IxDyn as Ix};

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");
    let layer_names: Vec<String> = vec![];

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[2.0, 2.0]).into_dyn())?;
    let mut parent_alpha = GraphDomainAlphaState::from_graph_bounds(
        &graph,
        &HashMap::new(),
        &GraphSplitHistory::new(),
        &input_bounds,
    );
    assert!(
        !parent_alpha.is_empty(),
        "parent alpha initialization should capture unstable neurons"
    );
    // Insert or update neuron (relu, 1) with the warm-start alpha value.
    if let Some(n) = parent_alpha.neuron_mut("relu", 1) {
        n.set_alpha(0.1234);
    } else {
        parent_alpha.insert("relu".to_string(), 1, AlphaNeuronState::new(0.1234));
    }

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: AD::from_shape_vec(Ix(&[1, 2]), vec![-1.0, -1.0]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 2]), vec![2.0, 2.0]).unwrap(),
        global_lbs: vec![-1.0],
        global_ubs: vec![2.0],
        metadata: vec![DomainMetadata {
            lower_bound: -1.0,
            upper_bound: 2.0,
            depth: 0,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: Some(parent_alpha.into()),
        }],
    };

    let (active, inactive, had_propagation_failure) =
        branch_relu_from_picked(0, &picked, &graph, "relu", 0, 0.5, &layer_names, false)?;
    assert!(
        !had_propagation_failure,
        "valid branch should not report propagation failure"
    );

    let active = active.expect("active branch should be feasible");
    let inactive = inactive.expect("inactive branch should be feasible");
    for child in [&active, &inactive] {
        assert!(
            !child.alpha_state.is_empty(),
            "child alpha state should stay non-empty after branching"
        );
        assert!(
            (child.alpha_state.alpha("relu", 1) - 0.1234).abs() < 1e-6,
            "child should inherit optimized parent alpha for unconstrained neuron"
        );
        assert!(
            child.alpha_state.neuron("relu", 0).is_none(),
            "branched neuron becomes constrained and should not remain optimizable"
        );
    }

    Ok(())
}

/// Regression for #1845: alpha state must survive the DomainList
/// pick_out/add round-trip.
#[test]
fn test_alpha_state_round_trip_through_domain_list_1845() {
    use crate::batched_domain::{DomainList, DomainListConfig, ProcessedDomains};
    use crate::beta_crown::state::{AlphaNeuronState, GraphDomainAlphaState};
    use ndarray::{ArrayD as AD, IxDyn as Ix};

    let layer_names = vec!["layer_a".to_string()];
    let mut alpha_state = GraphDomainAlphaState::empty();
    alpha_state.insert("relu".to_string(), 1, AlphaNeuronState::new(0.42));

    let mut layer_lowers = HashMap::new();
    layer_lowers.insert(
        "layer_a".to_string(),
        AD::from_shape_vec(Ix(&[1, 2]), vec![-1.0, -0.5]).unwrap(),
    );
    let mut layer_uppers = HashMap::new();
    layer_uppers.insert(
        "layer_a".to_string(),
        AD::from_shape_vec(Ix(&[1, 2]), vec![1.0, 0.5]).unwrap(),
    );

    let processed = ProcessedDomains {
        layer_lowers,
        layer_uppers,
        input_lowers: AD::from_shape_vec(Ix(&[1, 2]), vec![-1.0, -1.0]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 2]), vec![1.0, 1.0]).unwrap(),
        global_lbs: vec![-0.5],
        global_ubs: vec![0.5],
        metadata: vec![DomainMetadata {
            lower_bound: -0.5,
            upper_bound: 0.5,
            depth: 1,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: Some(alpha_state.into()),
        }],
        keep_mask: vec![true],
    };

    let mut layer_shapes = HashMap::new();
    layer_shapes.insert("layer_a".to_string(), vec![2usize]);
    let config = DomainListConfig {
        layer_names: layer_names.clone(),
        layer_shapes,
        input_shape: vec![2],
        ..DomainListConfig::default()
    };
    let mut domain_list = DomainList::new(config).expect("DomainList creation should succeed");
    domain_list.add(processed).expect("add should succeed");

    let picked = domain_list
        .pick_out_batched(
            1,
            crate::batched_domain::BatchedDomainOptions {
                enable_interm_transfer: false,
            },
        )
        .unwrap();
    assert_eq!(picked.batch_size, 1, "should pick one domain");

    let domain = graph_domain_from_picked(0, &picked, &layer_names, false, None)
        .expect("domain reconstruction should succeed");
    assert_eq!(
        domain.alpha_state.len(),
        1,
        "alpha metadata should round-trip"
    );
    assert!(
        (domain.alpha_state.alpha("relu", 1) - 0.42).abs() < 1e-6,
        "optimized alpha value should survive DomainList round-trip"
    );
}

// =============================================================================
// Input-split tests (Part of #1891)
// =============================================================================

/// Verify that `select_input_split_dimension` picks the widest input dimension.
///
/// With 5 input dimensions of varying widths, it should select the one with the
/// largest (upper - lower) span.
#[ntest::timeout(5000)]
#[test]
fn test_select_input_split_dimension_picks_widest() {
    use crate::beta_crown::engine::graph::domain_conversion::select_input_split_dimension;
    use ndarray::{ArrayD as AD, IxDyn as Ix};

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        // 5 input dimensions with widths: 1.0, 4.0, 0.5, 3.0, 2.0
        // Dimension 1 (width 4.0) should be selected.
        input_lowers: AD::from_shape_vec(Ix(&[1, 5]), vec![-1.0, -2.0, 0.0, -1.5, 0.5]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 5]), vec![0.0, 2.0, 0.5, 1.5, 2.5]).unwrap(),
        global_lbs: vec![-0.5],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -0.5,
            upper_bound: 1.0,
            depth: 0,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
    };

    let (best_dim, midpoint) =
        select_input_split_dimension(&picked, 0).expect("should select a dimension");

    assert_eq!(best_dim, 1, "should pick dimension 1 (width 4.0)");
    assert!(
        (midpoint - 0.0).abs() < 1e-6,
        "midpoint of [-2.0, 2.0] should be 0.0, got {}",
        midpoint
    );
}

/// Verify that `select_input_split_dimension` works with a batch (non-zero index).
#[ntest::timeout(5000)]
#[test]
fn test_select_input_split_dimension_batch_index() {
    use crate::beta_crown::engine::graph::domain_conversion::select_input_split_dimension;
    use ndarray::{ArrayD as AD, IxDyn as Ix};

    let picked = PickedDomains {
        batch_size: 2,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        // Domain 0: widths [1.0, 2.0] → dim 1
        // Domain 1: widths [3.0, 0.5] → dim 0
        input_lowers: AD::from_shape_vec(Ix(&[2, 2]), vec![0.0, -1.0, -1.5, 0.0]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[2, 2]), vec![1.0, 1.0, 1.5, 0.5]).unwrap(),
        global_lbs: vec![-0.5, -0.3],
        global_ubs: vec![1.0, 0.8],
        metadata: vec![
            DomainMetadata {
                lower_bound: -0.5,
                upper_bound: 1.0,
                depth: 0,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                node_bounds_override: None,
                alpha_state: None,
            },
            DomainMetadata {
                lower_bound: -0.3,
                upper_bound: 0.8,
                depth: 1,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                node_bounds_override: None,
                alpha_state: None,
            },
        ],
    };

    let (dim0, mid0) =
        select_input_split_dimension(&picked, 0).expect("should select for domain 0");
    assert_eq!(dim0, 1, "domain 0 should pick dim 1 (width 2.0)");
    assert!(
        (mid0 - 0.0).abs() < 1e-6,
        "midpoint of [-1.0, 1.0] should be 0.0"
    );

    let (dim1, mid1) =
        select_input_split_dimension(&picked, 1).expect("should select for domain 1");
    assert_eq!(dim1, 0, "domain 1 should pick dim 0 (width 3.0)");
    assert!(
        (mid1 - 0.0).abs() < 1e-6,
        "midpoint of [-1.5, 1.5] should be 0.0"
    );
}

/// Verify that `select_input_split_dimension` returns an error when all dimensions
/// have zero or non-finite width.
#[ntest::timeout(5000)]
#[test]
fn test_select_input_split_dimension_zero_width_error() {
    use crate::beta_crown::engine::graph::domain_conversion::select_input_split_dimension;
    use ndarray::{ArrayD as AD, IxDyn as Ix};

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        // All dimensions have zero width (lower == upper).
        input_lowers: AD::from_shape_vec(Ix(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap(),
        global_lbs: vec![-0.5],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -0.5,
            upper_bound: 1.0,
            depth: 5,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
    };

    let result = select_input_split_dimension(&picked, 0);
    assert!(
        result.is_err(),
        "should error when all dimensions have zero width"
    );
}

/// Verify that `branch_input_split_from_picked` creates correct left/right children
/// that bisect the selected input dimension.
#[ntest::timeout(5000)]
#[test]
fn test_branch_input_split_from_picked_bisects_input() -> Result<()> {
    use crate::beta_crown::engine::graph::domain_conversion::branch_input_split_from_picked;
    use ndarray::{ArrayD as AD, IxDyn as Ix};

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        // 3 input dimensions: [-2, 2], [-1, 1], [0, 4]
        input_lowers: AD::from_shape_vec(Ix(&[1, 3]), vec![-2.0, -1.0, 0.0]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 3]), vec![2.0, 1.0, 4.0]).unwrap(),
        global_lbs: vec![-0.5],
        global_ubs: vec![1.5],
        metadata: vec![DomainMetadata {
            lower_bound: -0.5,
            upper_bound: 1.5,
            depth: 3,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
    };

    // Split dimension 0 (range [-2, 2], midpoint = 0).
    let (left_opt, right_opt) = branch_input_split_from_picked(0, &picked, 0, false)?;

    let left = left_opt.expect("left child should be Some");
    let right = right_opt.expect("right child should be Some");

    // Left child: input_uppers[0] = midpoint = 0.0
    let left_lower = left.input_lowers.as_slice().unwrap();
    let left_upper = left.input_uppers.as_slice().unwrap();
    assert!(
        (left_lower[0] - (-2.0)).abs() < 1e-6,
        "left lower[0] unchanged"
    );
    assert!(
        (left_upper[0] - 0.0).abs() < 1e-6,
        "left upper[0] = midpoint"
    );
    // Other dimensions unchanged.
    assert!(
        (left_lower[1] - (-1.0)).abs() < 1e-6,
        "left lower[1] unchanged"
    );
    assert!(
        (left_upper[1] - 1.0).abs() < 1e-6,
        "left upper[1] unchanged"
    );
    assert!(
        (left_lower[2] - 0.0).abs() < 1e-6,
        "left lower[2] unchanged"
    );
    assert!(
        (left_upper[2] - 4.0).abs() < 1e-6,
        "left upper[2] unchanged"
    );

    // Right child: input_lowers[0] = midpoint = 0.0
    let right_lower = right.input_lowers.as_slice().unwrap();
    let right_upper = right.input_uppers.as_slice().unwrap();
    assert!(
        (right_lower[0] - 0.0).abs() < 1e-6,
        "right lower[0] = midpoint"
    );
    assert!(
        (right_upper[0] - 2.0).abs() < 1e-6,
        "right upper[0] unchanged"
    );
    // Other dimensions unchanged.
    assert!(
        (right_lower[1] - (-1.0)).abs() < 1e-6,
        "right lower[1] unchanged"
    );
    assert!(
        (right_upper[1] - 1.0).abs() < 1e-6,
        "right upper[1] unchanged"
    );
    assert!(
        (right_lower[2] - 0.0).abs() < 1e-6,
        "right lower[2] unchanged"
    );
    assert!(
        (right_upper[2] - 4.0).abs() < 1e-6,
        "right upper[2] unchanged"
    );

    // Both children should have depth = parent_depth + 1.
    assert_eq!(left.metadata[0].depth, 4, "left depth = parent 3 + 1");
    assert_eq!(right.metadata[0].depth, 4, "right depth = parent 3 + 1");

    // Input-split children have empty constraints (no ReLU history).
    assert!(
        left.metadata[0].constraints.is_empty(),
        "left constraints empty"
    );
    assert!(
        right.metadata[0].constraints.is_empty(),
        "right constraints empty"
    );

    // Input-split children have no cached_la or alpha_state.
    assert!(
        left.metadata[0].cached_la.is_none(),
        "left cached_la = None"
    );
    assert!(
        left.metadata[0].alpha_state.is_none(),
        "left alpha_state = None"
    );
    assert!(
        right.metadata[0].cached_la.is_none(),
        "right cached_la = None"
    );
    assert!(
        right.metadata[0].alpha_state.is_none(),
        "right alpha_state = None"
    );

    // Layer bounds should be empty (children must recompute from scratch).
    assert!(left.layer_lowers.is_empty(), "left layer_lowers empty");
    assert!(left.layer_uppers.is_empty(), "left layer_uppers empty");
    assert!(right.layer_lowers.is_empty(), "right layer_lowers empty");
    assert!(right.layer_uppers.is_empty(), "right layer_uppers empty");

    Ok(())
}

/// Verify that `branch_input_split_from_picked` returns (None, None) when the
/// selected dimension has zero width.
#[ntest::timeout(5000)]
#[test]
fn test_branch_input_split_from_picked_zero_width_returns_none() -> Result<()> {
    use crate::beta_crown::engine::graph::domain_conversion::branch_input_split_from_picked;
    use ndarray::{ArrayD as AD, IxDyn as Ix};

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        // Dimension 1 has zero width (lower == upper).
        input_lowers: AD::from_shape_vec(Ix(&[1, 2]), vec![0.0, 1.5]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 2]), vec![1.0, 1.5]).unwrap(),
        global_lbs: vec![-0.5],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -0.5,
            upper_bound: 1.0,
            depth: 0,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
    };

    // Split on dimension 1 which has zero width.
    let (left, right) = branch_input_split_from_picked(0, &picked, 1, false)?;
    assert!(left.is_none(), "left should be None for zero-width split");
    assert!(right.is_none(), "right should be None for zero-width split");

    Ok(())
}

/// Verify that `branch_input_split_from_picked` returns an error for out-of-bounds
/// split dimension.
#[ntest::timeout(5000)]
#[test]
fn test_branch_input_split_from_picked_invalid_dim() {
    use crate::beta_crown::engine::graph::domain_conversion::branch_input_split_from_picked;
    use ndarray::{ArrayD as AD, IxDyn as Ix};

    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: AD::from_shape_vec(Ix(&[1, 2]), vec![0.0, -1.0]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 2]), vec![1.0, 1.0]).unwrap(),
        global_lbs: vec![-0.5],
        global_ubs: vec![1.0],
        metadata: vec![DomainMetadata {
            lower_bound: -0.5,
            upper_bound: 1.0,
            depth: 0,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
    };

    // Dimension 5 is out of range for 2-dimensional input.
    let result = branch_input_split_from_picked(0, &picked, 5, false);
    assert!(result.is_err(), "should error for out-of-bounds split dim");
}

/// Verify that input-split children partition the parent domain exactly —
/// left child upper union right child lower covers the original range.
#[ntest::timeout(5000)]
#[test]
fn test_branch_input_split_children_partition_parent() -> Result<()> {
    use crate::beta_crown::engine::graph::domain_conversion::{
        branch_input_split_from_picked, select_input_split_dimension,
    };
    use ndarray::{ArrayD as AD, IxDyn as Ix};

    // Use ACAS-Xu-like 5-dimensional input.
    let picked = PickedDomains {
        batch_size: 1,
        layer_lowers: HashMap::new(),
        layer_uppers: HashMap::new(),
        input_lowers: AD::from_shape_vec(Ix(&[1, 5]), vec![0.6, -3.0, -3.0, 100.0, 0.0]).unwrap(),
        input_uppers: AD::from_shape_vec(Ix(&[1, 5]), vec![0.6799, 3.0, 3.0, 1200.0, 1200.0])
            .unwrap(),
        global_lbs: vec![-10.0],
        global_ubs: vec![10.0],
        metadata: vec![DomainMetadata {
            lower_bound: -10.0,
            upper_bound: 10.0,
            depth: 0,
            constraints: vec![],
            cached_la: None,
            needs_bounding: false,
            node_bounds_override: None,
            alpha_state: None,
        }],
    };

    // Select best dimension.
    let (best_dim, midpoint) = select_input_split_dimension(&picked, 0)?;

    // Dimension 3 or 4 (width 1100 or 1200) should be selected.
    // Dim 4 has width 1200, dim 3 has width 1100.
    assert_eq!(best_dim, 4, "should pick dim 4 (width 1200)");

    // Branch.
    let (left_opt, right_opt) = branch_input_split_from_picked(0, &picked, best_dim, false)?;
    let left = left_opt.expect("left child should be Some");
    let right = right_opt.expect("right child should be Some");

    let parent_lower = picked.input_lowers.as_slice().unwrap()[best_dim];
    let parent_upper = picked.input_uppers.as_slice().unwrap()[best_dim];

    // Left child: [parent_lower, midpoint]
    let left_lower = left.input_lowers.as_slice().unwrap()[best_dim];
    let left_upper = left.input_uppers.as_slice().unwrap()[best_dim];
    assert!(
        (left_lower - parent_lower).abs() < 1e-6,
        "left inherits parent lower"
    );
    assert!(
        (left_upper - midpoint).abs() < 1e-6,
        "left upper = midpoint"
    );

    // Right child: [midpoint, parent_upper]
    let right_lower = right.input_lowers.as_slice().unwrap()[best_dim];
    let right_upper = right.input_uppers.as_slice().unwrap()[best_dim];
    assert!(
        (right_lower - midpoint).abs() < 1e-6,
        "right lower = midpoint"
    );
    assert!(
        (right_upper - parent_upper).abs() < 1e-6,
        "right inherits parent upper"
    );

    // Children partition the parent exactly.
    assert!(
        (left_upper - right_lower).abs() < 1e-6,
        "left upper should meet right lower at midpoint"
    );
    assert!(
        (left_lower - parent_lower).abs() < 1e-6 && (right_upper - parent_upper).abs() < 1e-6,
        "children should cover the full parent range"
    );

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests for processed_from_backward_results
// Part of #3463: last uncovered function in the domain_conversion module.
// ──────────────────────────────────────────────────────────────────────────────

/// Helper: build a node cache (backward results) for a single domain.
/// Returns `HashMap<String, BoundedTensor>` keyed by layer names "layer_a", "layer_b".
fn make_node_cache(
    layer_a: (Vec<f32>, Vec<f32>),
    layer_b: (Vec<f32>, Vec<f32>),
) -> HashMap<String, Arc<BoundedTensor>> {
    let mut cache = HashMap::new();
    cache.insert(
        "layer_a".to_string(),
        Arc::new(make_bounded_tensor(layer_a.0, layer_a.1)),
    );
    cache.insert(
        "layer_b".to_string(),
        Arc::new(make_bounded_tensor(layer_b.0, layer_b.1)),
    );
    cache
}

/// Happy path: 2 domains, both kept. Verify stacked layer bounds, input bounds,
/// scalar bounds, metadata, and keep_mask.
#[ntest::timeout(5000)]
#[test]
fn test_backward_results_two_domains_all_kept() -> Result<()> {
    use crate::beta_crown::engine::graph::domain_conversion::processed_from_backward_results;

    let layer_names = vec!["layer_a".to_string(), "layer_b".to_string()];

    // Domain 0: backward results have updated layer bounds
    let cache_0 = make_node_cache((vec![1.0, 2.0], vec![3.0, 4.0]), (vec![5.0], vec![6.0]));
    // Domain 1: different backward results
    let cache_1 = make_node_cache(
        (vec![10.0, 20.0], vec![30.0, 40.0]),
        (vec![50.0], vec![60.0]),
    );

    // Children provide input bounds and metadata (history, depth)
    let child_0 = make_domain(
        (vec![0.0, 0.0], vec![1.0, 1.0]), // stale node bounds (ignored by backward_results)
        (vec![0.0], vec![1.0]),
        (vec![0.0, 0.0], vec![1.0, 1.0]), // input bounds ARE used
        -5.0,
        5.0,
        0,
        GraphSplitHistory::new(),
    );
    let child_1 = make_domain(
        (vec![0.0, 0.0], vec![1.0, 1.0]),
        (vec![0.0], vec![1.0]),
        (vec![0.5, 0.5], vec![1.5, 1.5]), // different input bounds
        -3.0,
        3.0,
        1,
        GraphSplitHistory::new(),
    );

    let children = vec![child_0, child_1];
    let lower_bounds = vec![-2.0, -1.0]; // updated objective bounds
    let upper_bounds = vec![2.0, 1.0];
    let keep_mask = vec![true, true];
    let node_caches = vec![cache_0, cache_1];

    let result = processed_from_backward_results(
        node_caches,
        &children,
        &lower_bounds,
        &upper_bounds,
        &keep_mask,
        &layer_names,
        None,
    )?;

    // Verify batch size
    assert_eq!(result.keep_mask.len(), 2);
    assert!(result.keep_mask.iter().all(|&k| k));

    // Verify layer bounds are from backward results (node_caches), not from children
    let la_lower = result.layer_lowers.get("layer_a").expect("layer_a lower");
    assert_eq!(la_lower.shape(), &[2, 2]); // [batch=2, features=2]
    assert_eq!(la_lower[[0, 0]], 1.0); // from cache_0
    assert_eq!(la_lower[[0, 1]], 2.0);
    assert_eq!(la_lower[[1, 0]], 10.0); // from cache_1
    assert_eq!(la_lower[[1, 1]], 20.0);

    let la_upper = result.layer_uppers.get("layer_a").expect("layer_a upper");
    assert_eq!(la_upper[[0, 0]], 3.0);
    assert_eq!(la_upper[[1, 0]], 30.0);

    let lb_lower = result.layer_lowers.get("layer_b").expect("layer_b lower");
    assert_eq!(lb_lower.shape(), &[2, 1]);
    assert_eq!(lb_lower[[0, 0]], 5.0);
    assert_eq!(lb_lower[[1, 0]], 50.0);

    // Verify input bounds come from children (not backward results)
    assert_eq!(result.input_lowers.shape(), &[2, 2]);
    assert_eq!(result.input_lowers[[0, 0]], 0.0); // from child_0
    assert_eq!(result.input_lowers[[1, 0]], 0.5); // from child_1

    // Verify scalar bounds
    assert_eq!(result.global_lbs, vec![-2.0, -1.0]);
    assert_eq!(result.global_ubs, vec![2.0, 1.0]);

    // Verify metadata
    assert_eq!(result.metadata.len(), 2);
    assert_eq!(result.metadata[0].depth, 0);
    assert_eq!(result.metadata[1].depth, 1);
    assert_eq!(result.metadata[0].lower_bound, -2.0);
    assert_eq!(result.metadata[1].lower_bound, -1.0);

    Ok(())
}

/// Partial keep_mask: 3 domains, middle one filtered out. Verify only kept domains
/// appear in the stacked output.
#[ntest::timeout(5000)]
#[test]
fn test_backward_results_partial_keep_mask() -> Result<()> {
    use crate::beta_crown::engine::graph::domain_conversion::processed_from_backward_results;

    let layer_names = vec!["layer_a".to_string()];

    let cache_0 = {
        let mut c = HashMap::new();
        c.insert(
            "layer_a".to_string(),
            Arc::new(make_bounded_tensor(vec![1.0], vec![2.0])),
        );
        c
    };
    let cache_1 = {
        let mut c = HashMap::new();
        c.insert(
            "layer_a".to_string(),
            Arc::new(make_bounded_tensor(vec![10.0], vec![20.0])),
        );
        c
    };
    let cache_2 = {
        let mut c = HashMap::new();
        c.insert(
            "layer_a".to_string(),
            Arc::new(make_bounded_tensor(vec![100.0], vec![200.0])),
        );
        c
    };

    let children: Vec<GraphBabDomain> = (0..3)
        .map(|i| {
            make_domain(
                (vec![0.0], vec![1.0]),
                (vec![0.0], vec![1.0]),
                (vec![i as f32], vec![i as f32 + 1.0]),
                -(i as f32),
                i as f32,
                i,
                GraphSplitHistory::new(),
            )
        })
        .collect();

    let lower_bounds = vec![-1.0, -2.0, -3.0];
    let upper_bounds = vec![1.0, 2.0, 3.0];
    let keep_mask = vec![true, false, true]; // domain 1 filtered out

    let result = processed_from_backward_results(
        vec![cache_0, cache_1, cache_2],
        &children,
        &lower_bounds,
        &upper_bounds,
        &keep_mask,
        &layer_names,
        None,
    )?;

    // Only 2 kept domains
    assert_eq!(result.keep_mask.len(), 2);
    assert!(result.keep_mask.iter().all(|&k| k));

    // Layer bounds: domain 0 and domain 2 (skip domain 1)
    let la_lower = result.layer_lowers.get("layer_a").expect("layer_a");
    assert_eq!(la_lower.shape(), &[2, 1]);
    assert_eq!(la_lower[[0, 0]], 1.0); // from cache_0
    assert_eq!(la_lower[[1, 0]], 100.0); // from cache_2 (skip cache_1)

    // Input bounds: from child 0 and child 2
    assert_eq!(result.input_lowers[[0, 0]], 0.0); // child 0
    assert_eq!(result.input_lowers[[1, 0]], 2.0); // child 2

    // Scalar bounds: only kept domains
    assert_eq!(result.global_lbs, vec![-1.0, -3.0]);
    assert_eq!(result.global_ubs, vec![1.0, 3.0]);

    // Metadata: depths from child 0 and child 2
    assert_eq!(result.metadata[0].depth, 0);
    assert_eq!(result.metadata[1].depth, 2);

    Ok(())
}

/// All domains filtered out (keep_mask all false) — returns empty ProcessedDomains.
#[ntest::timeout(5000)]
#[test]
fn test_backward_results_all_filtered_returns_empty() -> Result<()> {
    use crate::beta_crown::engine::graph::domain_conversion::processed_from_backward_results;

    let layer_names = vec!["layer_a".to_string()];
    let cache = {
        let mut c = HashMap::new();
        c.insert(
            "layer_a".to_string(),
            Arc::new(make_bounded_tensor(vec![1.0], vec![2.0])),
        );
        c
    };
    let child = make_domain(
        (vec![0.0], vec![1.0]),
        (vec![0.0], vec![1.0]),
        (vec![0.0], vec![1.0]),
        0.0,
        1.0,
        0,
        GraphSplitHistory::new(),
    );

    let result = processed_from_backward_results(
        vec![cache],
        &[child],
        &[-1.0],
        &[1.0],
        &[false], // nothing kept
        &layer_names,
        None,
    )?;

    assert!(result.keep_mask.is_empty());
    assert!(result.global_lbs.is_empty());
    assert!(result.global_ubs.is_empty());
    assert!(result.metadata.is_empty());
    assert!(result.layer_lowers.is_empty());
    assert!(result.layer_uppers.is_empty());

    Ok(())
}

/// Batch size mismatch between node_caches and children triggers InternalError.
#[ntest::timeout(5000)]
#[test]
fn test_backward_results_batch_mismatch_returns_error() {
    use crate::beta_crown::engine::graph::domain_conversion::processed_from_backward_results;

    let layer_names = vec!["layer_a".to_string()];
    let cache = {
        let mut c = HashMap::new();
        c.insert(
            "layer_a".to_string(),
            Arc::new(make_bounded_tensor(vec![1.0], vec![2.0])),
        );
        c
    };

    // 1 node_cache but 0 children → mismatch
    let result = processed_from_backward_results(
        vec![cache],
        &[],
        &[-1.0],
        &[1.0],
        &[true],
        &layer_names,
        None,
    );
    assert!(result.is_err(), "batch mismatch must return error");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("batch mismatch"),
        "error should mention batch mismatch, got: {err_msg}"
    );
}

/// Cached lA flows through to metadata for kept domains.
#[ntest::timeout(5000)]
#[test]
fn test_backward_results_cached_la_threaded_to_metadata() -> Result<()> {
    use crate::beta_crown::engine::graph::domain_conversion::processed_from_backward_results;

    let layer_names = vec!["layer_a".to_string()];

    let cache_0 = {
        let mut c = HashMap::new();
        c.insert(
            "layer_a".to_string(),
            Arc::new(make_bounded_tensor(vec![1.0], vec![2.0])),
        );
        c
    };
    let cache_1 = {
        let mut c = HashMap::new();
        c.insert(
            "layer_a".to_string(),
            Arc::new(make_bounded_tensor(vec![3.0], vec![4.0])),
        );
        c
    };

    let children: Vec<GraphBabDomain> = (0..2)
        .map(|_| {
            make_domain(
                (vec![0.0], vec![1.0]),
                (vec![0.0], vec![1.0]),
                (vec![0.0], vec![1.0]),
                0.0,
                1.0,
                0,
                GraphSplitHistory::new(),
            )
        })
        .collect();

    // Cached lA: provided per kept domain. Both domains are kept, so 2 entries.
    let la_0 = Arc::new(make_cached_la(10.0));
    let la_1 = Arc::new(make_cached_la(20.0));

    let result = processed_from_backward_results(
        vec![cache_0, cache_1],
        &children,
        &[0.0, 0.0],
        &[1.0, 1.0],
        &[true, true],
        &layer_names,
        Some(vec![la_0.clone(), la_1.clone()]),
    )?;

    // Metadata should contain the cached lA
    assert!(
        result.metadata[0].cached_la.is_some(),
        "domain 0 should have cached_la"
    );
    assert!(
        result.metadata[1].cached_la.is_some(),
        "domain 1 should have cached_la"
    );

    // Verify the lA values were preserved
    let meta_la_0 = result.metadata[0].cached_la.as_ref().unwrap();
    let expected_la_0_lower = la_0.lower_a.get("layer_a").unwrap();
    let actual_la_0_lower = meta_la_0.lower_a.get("layer_a").unwrap();
    assert_eq!(actual_la_0_lower, expected_la_0_lower);

    let meta_la_1 = result.metadata[1].cached_la.as_ref().unwrap();
    let expected_la_1_lower = la_1.lower_a.get("layer_a").unwrap();
    let actual_la_1_lower = meta_la_1.lower_a.get("layer_a").unwrap();
    assert_eq!(actual_la_1_lower, expected_la_1_lower);

    Ok(())
}

/// Cached lA with partial keep_mask: lA entries are consumed in kept-domain order.
/// 3 domains, middle filtered out, lA has 2 entries for the 2 kept domains.
#[ntest::timeout(5000)]
#[test]
fn test_backward_results_cached_la_with_partial_keep() -> Result<()> {
    use crate::beta_crown::engine::graph::domain_conversion::processed_from_backward_results;

    let layer_names = vec!["layer_a".to_string()];

    let caches: Vec<HashMap<String, Arc<BoundedTensor>>> = (0..3)
        .map(|i| {
            let mut c = HashMap::new();
            c.insert(
                "layer_a".to_string(),
                Arc::new(make_bounded_tensor(vec![i as f32], vec![i as f32 + 1.0])),
            );
            c
        })
        .collect();

    let children: Vec<GraphBabDomain> = (0..3)
        .map(|_| {
            make_domain(
                (vec![0.0], vec![1.0]),
                (vec![0.0], vec![1.0]),
                (vec![0.0], vec![1.0]),
                0.0,
                1.0,
                0,
                GraphSplitHistory::new(),
            )
        })
        .collect();

    // 2 cached lA entries for the 2 kept domains
    let la_first_kept = Arc::new(make_cached_la(100.0));
    let la_second_kept = Arc::new(make_cached_la(200.0));

    let result = processed_from_backward_results(
        caches,
        &children,
        &[0.0, 0.0, 0.0],
        &[1.0, 1.0, 1.0],
        &[true, false, true], // domains 0 and 2 kept
        &layer_names,
        Some(vec![la_first_kept, la_second_kept]),
    )?;

    assert_eq!(result.metadata.len(), 2);

    // First kept domain (idx 0) gets la_first_kept
    let meta_0_la = result.metadata[0].cached_la.as_ref().unwrap();
    let actual_seed_0 = meta_0_la.lower_a.get("layer_a").unwrap()[[0, 0]];
    assert_eq!(actual_seed_0, 100.0);

    // Second kept domain (idx 2) gets la_second_kept
    let meta_1_la = result.metadata[1].cached_la.as_ref().unwrap();
    let actual_seed_1 = meta_1_la.lower_a.get("layer_a").unwrap()[[0, 0]];
    assert_eq!(actual_seed_1, 200.0);

    Ok(())
}
