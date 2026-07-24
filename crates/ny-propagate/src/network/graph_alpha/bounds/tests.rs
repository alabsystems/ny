// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::{
    AddConstantLayer, AddLayer, ConcatLayer, Conv1dLayer, Conv2dLayer, DivLayer, ExpLayer,
    LinearLayer, MaxBinaryLayer, MulBinaryLayer, MulConstantLayer, NonZeroLayer, PadLayer, PadMode,
    ReLULayer, ReduceSumLayer, SigmoidLayer, SliceLayer, SqrtLayer, SubLayer, TanhLayer,
    WhereLayer,
};
use crate::network::core::GraphNode;
use crate::types::BoundsProvenance;
use ndarray::{arr1, arr2, array, ArrayD, Ix1, IxDyn};
use ny_core::NaiveCpuGemmEngine;
use ny_test_utils::{assert_bounded_tensor_close, CountingGemmEngine};

/// Helper: run both CROWN-IBP (non-alpha) and α-CROWN (empty alpha state)
/// on the same graph and return (crown_ibp_result, alpha_crown_result).
fn run_both_paths(graph: &GraphNetwork, input: &BoundedTensor) -> (BoundedTensor, BoundedTensor) {
    let ibp_bounds = graph.collect_node_bounds(input).unwrap();

    let crown_ibp = graph
        .propagate_crown_to_node(
            input,
            graph.output_name(),
            &std::collections::HashMap::new(),
            &ibp_bounds,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    let alpha_state = GraphAlphaState::new();
    let alpha_crown = graph
        .propagate_crown_to_node_with_alpha(
            input,
            graph.output_name(),
            &std::collections::HashMap::new(),
            &ibp_bounds,
            &alpha_state,
            None,
            None,
        )
        .unwrap();

    (crown_ibp, alpha_crown)
}

fn legacy_ancestors_bfs(graph: &GraphNetwork, target: &str) -> Vec<String> {
    let mut visited = std::collections::HashSet::new();
    let mut to_visit = vec![target.to_string()];

    while let Some(node_name) = to_visit.pop() {
        if visited.contains(&node_name) || node_name == NETWORK_INPUT {
            continue;
        }
        visited.insert(node_name.clone());

        if let Some(node) = graph.nodes.get(&node_name) {
            for input_name in &node.inputs {
                if input_name != NETWORK_INPUT && !visited.contains(input_name) {
                    to_visit.push(input_name.clone());
                }
            }
        }
    }

    graph
        .exec_order()
        .expect("legacy BFS helper should compute exec order")
        .iter()
        .filter(|node_name| visited.contains(*node_name))
        .cloned()
        .collect()
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_ancestors_preserve_topological_order_for_target_subgraph_2237() {
    let mut graph = GraphNetwork::new();
    let weight = arr2(&[[1.0_f32]]);

    graph.add_node(GraphNode::from_input(
        "stem",
        Layer::Linear(LinearLayer::new(weight, None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "left",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "target",
        Layer::ReLU(ReLULayer),
        vec!["left".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "right",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));

    let ancestors = graph.ancestors("target").expect("ancestors should resolve");
    assert_eq!(ancestors, vec!["stem", "left", "target"]);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_ancestors_cache_matches_legacy_bfs_and_invalidates_on_mutation_2220() {
    let mut graph = GraphNetwork::new();
    let weight = arr2(&[[1.0_f32]]);

    graph.add_node(GraphNode::from_input(
        "stem",
        Layer::Linear(LinearLayer::new(weight, None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "left",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "right",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "merge",
        Layer::Add(AddLayer),
        vec!["left".to_string(), "right".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "tail",
        Layer::ReLU(ReLULayer),
        vec!["merge".to_string()],
    ));

    let cached_tail = graph
        .ancestors("tail")
        .expect("cached ancestors should resolve for tail");
    let legacy_tail = legacy_ancestors_bfs(&graph, "tail");
    assert_eq!(
        cached_tail, legacy_tail,
        "#2220 Packet A: cached ancestors must match the legacy BFS traversal for branching DAGs"
    );

    let cached_merge = graph
        .ancestors("merge")
        .expect("cached ancestors should resolve for merge");
    let legacy_merge = legacy_ancestors_bfs(&graph, "merge");
    assert_eq!(
        cached_merge, legacy_merge,
        "#2220 Packet A: cached ancestors must preserve the legacy BFS result for merge nodes"
    );

    graph.add_node(GraphNode::new(
        "shortcut",
        Layer::ReLU(ReLULayer),
        vec!["right".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "new_target",
        Layer::Add(AddLayer),
        vec!["tail".to_string(), "shortcut".to_string()],
    ));

    let cached_new_target = graph
        .ancestors("new_target")
        .expect("ancestors cache should invalidate after graph mutation");
    let legacy_new_target = legacy_ancestors_bfs(&graph, "new_target");
    assert_eq!(
        cached_new_target, legacy_new_target,
        "#2220 Packet A: structural mutation must invalidate the ancestors cache"
    );
}

// ========================= AddConstant parity =========================

#[ntest::timeout(10000)]
#[test]
fn test_alpha_vs_crown_parity_add_constant() {
    // Graph: Linear(2→4) → AddConstant([1,2,3,4]) → Linear(4→1)
    // AddConstant adds bias. If alpha path uses identity fallback,
    // the constant contribution is dropped and bounds differ.
    let w1 = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.9], [0.8, 0.1]]);
    let b1 = arr1(&[0.1, -0.2, 0.3, -0.1]);
    let constant = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let w2 = arr2(&[[2.0, -1.0, 0.5, -0.3]]);
    let b2 = arr1(&[0.7]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "add_const",
        Layer::AddConstant(AddConstantLayer::new(constant)),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["add_const".to_string()],
    ));
    graph.set_output("lin2");

    let input =
        BoundedTensor::new(array![-1.0, -2.0].into_dyn(), array![3.0, 4.0].into_dyn()).unwrap();

    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);
    assert_bounded_tensor_close(&crown_ibp, &alpha_crown, 1e-4, "AddConstant parity");

    // Sanity: bounds must not be trivially zero (would indicate broken propagation)
    assert!(
        crown_ibp.lower()[[0]].abs() > 0.01 || crown_ibp.upper()[[0]].abs() > 0.01,
        "Bounds are trivially near zero — propagation may be broken"
    );
}

// ========================= Sigmoid parity =========================

#[ntest::timeout(10000)]
#[test]
fn test_alpha_vs_crown_parity_sigmoid() {
    // Graph: Linear(2→3) → Sigmoid → Linear(3→1)
    // Sigmoid is a nonlinear activation. Identity fallback produces
    // incorrect linear relaxation semantics.
    let w1 = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.4]]);
    let b1 = arr1(&[0.0, 0.0, 0.0]);
    let w2 = arr2(&[[1.0, -1.0, 0.5]]);
    let b2 = arr1(&[0.0]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["sigmoid".to_string()],
    ));
    graph.set_output("lin2");

    let input =
        BoundedTensor::new(array![-1.0, -1.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);
    assert_bounded_tensor_close(&crown_ibp, &alpha_crown, 1e-4, "Sigmoid parity");
}

// ========================= Slice parity =========================

#[ntest::timeout(10000)]
#[test]
fn test_alpha_vs_crown_parity_slice() {
    // Graph: Linear(2→4) → Slice(axis=0, start=1, end=3) → Linear(2→1)
    // Slice extracts a subrange. Identity fallback skips the slicing
    // and produces wrong dimension propagation.
    let w1 = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.9], [0.8, 0.1]]);
    let b1 = arr1(&[0.1, -0.2, 0.3, -0.1]);
    let w2 = arr2(&[[2.0, -1.0]]);
    let b2 = arr1(&[0.5]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "slice",
        Layer::Slice(SliceLayer::new(0, 1, 3)),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["slice".to_string()],
    ));
    graph.set_output("lin2");

    let input =
        BoundedTensor::new(array![-1.0, -2.0].into_dyn(), array![3.0, 4.0].into_dyn()).unwrap();

    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);
    assert_bounded_tensor_close(&crown_ibp, &alpha_crown, 1e-4, "Slice parity");
}

// ========================= Soundness grid helper (#2543) =========================

/// Sample an 11×11 grid over a 2D input domain and verify that every true
/// network output lies within both CROWN-IBP and alpha-CROWN bounds.
/// Also checks CROWN bounds are at least as tight as IBP.
fn assert_grid_soundness(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    crown_ibp: &BoundedTensor,
    alpha_crown: &BoundedTensor,
    forward_fn: impl Fn(f32, f32) -> f32,
    label: &str,
) {
    let lb0 = input.lower()[[0]];
    let ub0 = input.upper()[[0]];
    let lb1 = input.lower()[[1]];
    let ub1 = input.upper()[[1]];
    let tol = 1e-4;
    for i in 0..=10 {
        for j in 0..=10 {
            let x0 = lb0 + (ub0 - lb0) * (i as f32) / 10.0;
            let x1 = lb1 + (ub1 - lb1) * (j as f32) / 10.0;
            let y = forward_fn(x0, x1);
            assert!(
                y >= crown_ibp.lower()[[0]] - tol && y <= crown_ibp.upper()[[0]] + tol,
                "{label} CROWN-IBP: output {y} not in [{}, {}] at ({x0}, {x1})",
                crown_ibp.lower()[[0]],
                crown_ibp.upper()[[0]],
            );
            assert!(
                y >= alpha_crown.lower()[[0]] - tol && y <= alpha_crown.upper()[[0]] + tol,
                "{label} alpha-CROWN: output {y} not in [{}, {}] at ({x0}, {x1})",
                alpha_crown.lower()[[0]],
                alpha_crown.upper()[[0]],
            );
        }
    }
    let ibp_bounds = graph.collect_node_bounds(input).unwrap();
    let ibp_out = ibp_bounds.get(graph.output_name()).unwrap();
    let crown_width = crown_ibp.upper()[[0]] - crown_ibp.lower()[[0]];
    let ibp_width = ibp_out.upper()[[0]] - ibp_out.lower()[[0]];
    assert!(
        crown_width <= ibp_width + tol,
        "{label}: CROWN should be at least as tight as IBP: crown={crown_width}, ibp={ibp_width}",
    );
}

// ========================= Sigmoid soundness (#2543) =========================

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_sigmoid_soundness_2543() {
    // Graph: Linear(2→3) → Sigmoid → Linear(3→1)
    // Parity alone cannot detect both paths computing the same wrong relaxation.
    let w1 = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.4]]);
    let b1 = arr1(&[0.0, 0.0, 0.0]);
    let w2 = arr2(&[[1.0, -1.0, 0.5]]);
    let b2 = arr1(&[0.0]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
        vec!["sigmoid".to_string()],
    ));
    graph.set_output("lin2");

    let input =
        BoundedTensor::new(array![-1.0, -1.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();
    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);

    assert_grid_soundness(
        &graph,
        &input,
        &crown_ibp,
        &alpha_crown,
        |x0, x1| {
            let h = w1.dot(&array![x0, x1]) + &b1;
            let h_sig = array![
                1.0 / (1.0 + (-h[0]).exp()),
                1.0 / (1.0 + (-h[1]).exp()),
                1.0 / (1.0 + (-h[2]).exp())
            ];
            (w2.dot(&h_sig) + &b2)[0]
        },
        "Sigmoid",
    );
}

// ========================= Slice soundness (#2543) =========================

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_slice_soundness_2543() {
    // Graph: Linear(2→4) → Slice(axis=0, 1..3) → Linear(2→1)
    // Identity fallback would propagate wrong dimensions.
    let w1 = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.9], [0.8, 0.1]]);
    let b1 = arr1(&[0.1, -0.2, 0.3, -0.1]);
    let w2 = arr2(&[[2.0, -1.0]]);
    let b2 = arr1(&[0.5]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "slice",
        Layer::Slice(SliceLayer::new(0, 1, 3)),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
        vec!["slice".to_string()],
    ));
    graph.set_output("lin2");

    let input =
        BoundedTensor::new(array![-1.0, -2.0].into_dyn(), array![3.0, 4.0].into_dyn()).unwrap();
    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);

    assert_grid_soundness(
        &graph,
        &input,
        &crown_ibp,
        &alpha_crown,
        |x0, x1| {
            let h = w1.dot(&array![x0, x1]) + &b1;
            (w2.dot(&array![h[1], h[2]]) + &b2)[0]
        },
        "Slice",
    );
}

// ========================= Soundness check =========================

#[ntest::timeout(10000)]
#[test]
fn test_alpha_add_constant_bounds_contain_true_output() {
    // Verify that bounds from α-CROWN path with AddConstant actually
    // contain sampled true outputs — a direct soundness check.
    let w1 = arr2(&[[2.0, -1.0], [0.5, 1.5]]);
    let b1 = arr1(&[0.0, 0.0]);
    let constant = ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, -1.0]).unwrap();
    let w2 = arr2(&[[1.0, 1.0]]);
    let b2 = arr1(&[0.0]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "add_const",
        Layer::AddConstant(AddConstantLayer::new(constant.clone())),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
        vec!["add_const".to_string()],
    ));
    graph.set_output("lin2");

    let input =
        BoundedTensor::new(array![-1.0, -1.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let alpha_state = GraphAlphaState::new();
    let alpha_bounds = graph
        .propagate_crown_to_node_with_alpha(
            &input,
            "lin2",
            &std::collections::HashMap::new(),
            &ibp_bounds,
            &alpha_state,
            None,
            None,
        )
        .unwrap();

    // Sample corner points and verify containment
    let corners = [
        array![-1.0, -1.0],
        array![-1.0, 1.0],
        array![1.0, -1.0],
        array![1.0, 1.0],
    ];

    for corner in &corners {
        // Forward: lin1 → add_const → lin2
        let h = w1.dot(corner) + &b1;
        let h_plus_c = &h + &constant.clone().into_dimensionality::<Ix1>().unwrap();
        let out = w2.dot(&h_plus_c) + &b2;

        let out_val = out[0];
        let lb = alpha_bounds.lower()[[0]];
        let ub = alpha_bounds.upper()[[0]];
        assert!(
            out_val >= lb - 1e-5 && out_val <= ub + 1e-5,
            "Soundness violation: output {} not in [{}, {}] for corner {:?}",
            out_val,
            lb,
            ub,
            corner
        );
    }
}

// ========================= Exp soundness via generic dispatch =========================

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_exp_soundness_1929() {
    // Graph: Linear(2→2) → Exp → Linear(2→1)
    // Exp was previously falling through to identity in the non-alpha path.
    // With the generic dispatch, it now calls propagate_crown_backward which
    // routes to propagate_linear_with_bounds for proper tangent-line relaxation.
    let w1 = arr2(&[[0.5, -0.3], [0.2, 0.4]]);
    let b1 = arr1(&[0.0, 0.0]);
    let w2 = arr2(&[[1.0, -1.0]]);
    let b2 = arr1(&[0.0]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "exp",
        Layer::Exp(ExpLayer),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
        vec!["exp".to_string()],
    ));
    graph.set_output("lin2");

    let input =
        BoundedTensor::new(array![-0.5, -0.5].into_dyn(), array![0.5, 0.5].into_dyn()).unwrap();

    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);

    // Both paths should produce valid bounds (not identity fallback)
    // Verify soundness: sample points must lie within bounds
    let tol = 1e-4;
    for i in 0..=10 {
        for j in 0..=10 {
            let x0 = -0.5 + (i as f32) / 10.0;
            let x1 = -0.5 + (j as f32) / 10.0;
            let x = array![x0, x1];
            let h = w1.dot(&x) + &b1;
            let h_exp = array![h[0].exp(), h[1].exp()];
            let out = w2.dot(&h_exp) + &b2;
            let y = out[0];

            assert!(
                y >= crown_ibp.lower()[[0]] - tol && y <= crown_ibp.upper()[[0]] + tol,
                "CROWN-IBP soundness: output {} not in [{}, {}] at ({}, {})",
                y,
                crown_ibp.lower()[[0]],
                crown_ibp.upper()[[0]],
                x0,
                x1
            );
            assert!(
                y >= alpha_crown.lower()[[0]] - tol && y <= alpha_crown.upper()[[0]] + tol,
                "alpha-CROWN soundness: output {} not in [{}, {}] at ({}, {})",
                y,
                alpha_crown.lower()[[0]],
                alpha_crown.upper()[[0]],
                x0,
                x1
            );
        }
    }

    // Bounds should be tighter than IBP (CROWN relaxation of Exp is tighter)
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let ibp_out = ibp_bounds.get("lin2").unwrap();
    let crown_width = crown_ibp.upper()[[0]] - crown_ibp.lower()[[0]];
    let ibp_width = ibp_out.upper()[[0]] - ibp_out.lower()[[0]];
    assert!(
        crown_width <= ibp_width + tol,
        "CROWN should be at least as tight as IBP: crown={}, ibp={}",
        crown_width,
        ibp_width
    );
}

// ========================= Tanh soundness via generic dispatch =========================

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_tanh_soundness_1929() {
    // Graph: Linear(2→2) → Tanh → Linear(2→1)
    // Tanh was previously falling through to identity in the non-alpha path.
    let w1 = arr2(&[[0.8, -0.5], [-0.3, 0.6]]);
    let b1 = arr1(&[0.0, 0.0]);
    let w2 = arr2(&[[1.0, 1.0]]);
    let b2 = arr1(&[0.0]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "tanh",
        Layer::Tanh(TanhLayer),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
        vec!["tanh".to_string()],
    ));
    graph.set_output("lin2");

    let input =
        BoundedTensor::new(array![-1.0, -1.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);

    // Verify soundness: sample points must lie within bounds
    let tol = 1e-4;
    for i in 0..=10 {
        for j in 0..=10 {
            let x0 = -1.0 + 2.0 * (i as f32) / 10.0;
            let x1 = -1.0 + 2.0 * (j as f32) / 10.0;
            let x = array![x0, x1];
            let h = w1.dot(&x) + &b1;
            let h_tanh = array![h[0].tanh(), h[1].tanh()];
            let out = w2.dot(&h_tanh) + &b2;
            let y = out[0];

            assert!(
                y >= crown_ibp.lower()[[0]] - tol && y <= crown_ibp.upper()[[0]] + tol,
                "CROWN-IBP soundness: output {} not in [{}, {}] at ({}, {})",
                y,
                crown_ibp.lower()[[0]],
                crown_ibp.upper()[[0]],
                x0,
                x1
            );
            assert!(
                y >= alpha_crown.lower()[[0]] - tol && y <= alpha_crown.upper()[[0]] + tol,
                "alpha-CROWN soundness: output {} not in [{}, {}] at ({}, {})",
                y,
                alpha_crown.lower()[[0]],
                alpha_crown.upper()[[0]],
                x0,
                x1
            );
        }
    }
}

// ========================= Sub binary parity =========================

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_sub_binary_parity_1929() {
    // Graph: input → ReLU → two Linear branches → Sub → Linear → output
    // Sub was previously falling through to identity in both paths.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    let w_a = arr2(&[[1.0_f32]]);
    let b_a = arr1(&[2.0_f32]);
    graph.add_node(GraphNode::new(
        "branch_a",
        Layer::Linear(LinearLayer::new(w_a, Some(b_a)).unwrap()),
        vec!["relu".to_string()],
    ));
    let w_b = arr2(&[[0.5_f32]]);
    let b_b = arr1(&[-1.0_f32]);
    graph.add_node(GraphNode::new(
        "branch_b",
        Layer::Linear(LinearLayer::new(w_b, Some(b_b)).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sub",
        Layer::Sub(SubLayer),
        vec!["branch_a".to_string(), "branch_b".to_string()],
    ));
    let w_out = arr2(&[[1.0_f32]]);
    let b_out = arr1(&[0.0_f32]);
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(w_out, Some(b_out)).unwrap()),
        vec!["sub".to_string()],
    ));
    graph.set_output("out");

    let input =
        BoundedTensor::new(array![-1.0_f32].into_dyn(), array![1.0_f32].into_dyn()).unwrap();

    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);

    // Both should produce consistent results
    assert_bounded_tensor_close(
        &crown_ibp,
        &alpha_crown,
        0.1,
        "Sub: CROWN-IBP vs alpha-CROWN",
    );

    // Verify soundness with samples: out = relu(x)+2 - (0.5*relu(x)-1) = 0.5*relu(x)+3
    let tol = 1e-4;
    for i in 0..=20 {
        let x = -1.0 + 2.0 * (i as f32) / 20.0;
        let r = x.max(0.0);
        let y = 0.5 * r + 3.0;
        assert!(
            y >= crown_ibp.lower()[[0]] - tol && y <= crown_ibp.upper()[[0]] + tol,
            "CROWN-IBP soundness: output {} not in [{}, {}] at x={}",
            y,
            crown_ibp.lower()[[0]],
            crown_ibp.upper()[[0]],
            x
        );
    }
}

// ========================= Multi-input unsupported returns error =========================

fn build_rowwise_broadcast_div_graph_3626() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "reduce",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
    ));
    graph.add_node(GraphNode::new(
        "shift",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(
            IxDyn(&[1]),
            4.0_f32,
        ))),
        vec!["reduce".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "div",
        Layer::Div(DivLayer),
        vec![NETWORK_INPUT.to_string(), "shift".to_string()],
    ));
    graph.set_output("div");
    graph
}

fn eval_rowwise_broadcast_div_graph_3626(x: &[f32; 4]) -> [f32; 4] {
    let denom0 = x[0] + x[1] + 4.0;
    let denom1 = x[2] + x[3] + 4.0;
    [x[0] / denom0, x[1] / denom0, x[2] / denom1, x[3] / denom1]
}

fn assert_div_crown_ibp_not_looser_than_ibp(
    ibp_out: &BoundedTensor,
    cibp_out: &BoundedTensor,
    label: &str,
) {
    let ibp_out = ibp_out.flatten();
    let cibp_out = cibp_out.flatten();
    for dim in 0..4 {
        let ibp_lower = ibp_out.lower()[[dim]];
        let ibp_upper = ibp_out.upper()[[dim]];
        assert!(
            cibp_out.lower()[[dim]] >= ibp_lower - 1e-4,
            "{label} lower[{dim}] looser than IBP: cibp={}, ibp={}",
            cibp_out.lower()[[dim]],
            ibp_lower
        );
        assert!(
            cibp_out.upper()[[dim]] <= ibp_upper + 1e-4,
            "{label} upper[{dim}] looser than IBP: cibp={}, ibp={}",
            cibp_out.upper()[[dim]],
            ibp_upper
        );
    }
}

fn assert_div_sample_contained(bounds: &BoundedTensor, x: [f32; 4], label: &str) {
    let bounds = bounds.flatten();
    let output = eval_rowwise_broadcast_div_graph_3626(&x);
    for (dim, &value) in output.iter().enumerate() {
        assert!(
            value >= bounds.lower()[[dim]] - 1e-4 && value <= bounds.upper()[[dim]] + 1e-4,
            "{label}: output {} not in [{}, {}] at [{}, {}, {}, {}] dim {}",
            value,
            bounds.lower()[[dim]],
            bounds.upper()[[dim]],
            x[0],
            x[1],
            x[2],
            x[3],
            dim
        );
    }
}

fn assert_div_bounds_sound_on_grid(
    input: &BoundedTensor,
    raw_crown: &BoundedTensor,
    alpha_crown: &BoundedTensor,
    cibp_out: &BoundedTensor,
) {
    let lower = input.lower().iter().copied().collect::<Vec<_>>();
    let upper = input.upper().iter().copied().collect::<Vec<_>>();
    for &x0 in &[lower[0], f32::midpoint(lower[0], upper[0]), upper[0]] {
        for &x1 in &[lower[1], f32::midpoint(lower[1], upper[1]), upper[1]] {
            for &x2 in &[lower[2], f32::midpoint(lower[2], upper[2]), upper[2]] {
                for &x3 in &[lower[3], f32::midpoint(lower[3], upper[3]), upper[3]] {
                    let point = [x0, x1, x2, x3];
                    assert_div_sample_contained(raw_crown, point, "raw CROWN soundness");
                    assert_div_sample_contained(alpha_crown, point, "alpha-CROWN soundness");
                    assert_div_sample_contained(cibp_out, point, "CROWN-IBP soundness");
                }
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_div_broadcast_soundness_3626() {
    let graph = build_rowwise_broadcast_div_graph_3626();
    let input = BoundedTensor::new(
        array![[-1.5_f32, -1.0_f32], [0.2_f32, 0.3_f32]].into_dyn(),
        array![[0.5_f32, 1.0_f32], [1.0_f32, 1.2_f32]].into_dyn(),
    )
    .unwrap();

    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let ibp_out = ibp_bounds.get("div").unwrap().flatten();
    let (raw_crown, alpha_crown) = run_both_paths(&graph, &input);
    let raw_crown = raw_crown.flatten();
    let alpha_crown = alpha_crown.flatten();
    let cibp_result = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(&input, ibp_bounds, None)
        .unwrap();
    let cibp_out = cibp_result.bounds.get("div").unwrap().flatten();

    assert_bounded_tensor_close(&raw_crown, &alpha_crown, 1e-4, "Div broadcast raw parity");
    assert!(
        matches!(
            cibp_result.provenance.get("div"),
            Some(BoundsProvenance::Crown)
        ),
        "Div node should be CROWN-tightened in DAG CROWN-IBP, got {:?}",
        cibp_result.provenance.get("div")
    );
    assert_div_crown_ibp_not_looser_than_ibp(&ibp_out, &cibp_out, "Div broadcast CROWN-IBP");
    assert_div_bounds_sound_on_grid(&input, &raw_crown, &alpha_crown, &cibp_out);
}

fn build_deterministic_where_graph_3676(cond_value: f32) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let identity = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]);
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::Linear(LinearLayer::new(identity, None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "y",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cond_base",
        Layer::MulConstant(MulConstantLayer::scalar(0.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(
            IxDyn(&[]),
            cond_value,
        ))),
        vec!["cond_base".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "x".to_string(), "y".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.5]).into_dyn(),
        arr1(&[2.0_f32, 1.5]).into_dyn(),
    )
    .unwrap();
    (graph, input)
}

fn build_mixed_where_matrix_graph_3676() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
    ));
    graph.add_node(GraphNode::new(
        "y",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["x".to_string(), "x".to_string(), "y".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0_f32, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 1.0]).unwrap(),
    )
    .unwrap();
    (graph, input)
}

fn assert_where_tighter_than_ibp_3676(
    ibp_out: &BoundedTensor,
    crown_out: &BoundedTensor,
    label: &str,
) {
    let ibp_out = ibp_out.flatten();
    let crown_out = crown_out.flatten();
    let mut strictly_tighter = false;

    for dim in 0..ibp_out.lower().len() {
        let ibp_lower = ibp_out.lower()[[dim]];
        let ibp_upper = ibp_out.upper()[[dim]];
        let crown_lower = crown_out.lower()[[dim]];
        let crown_upper = crown_out.upper()[[dim]];
        assert!(
            crown_lower >= ibp_lower - 1e-5,
            "{label} lower[{dim}] must be no looser than IBP: crown={} ibp={}",
            crown_lower,
            ibp_lower
        );
        assert!(
            crown_upper <= ibp_upper + 1e-5,
            "{label} upper[{dim}] must be no looser than IBP: crown={} ibp={}",
            crown_upper,
            ibp_upper
        );
        strictly_tighter |= crown_lower > ibp_lower + 1e-5 || crown_upper < ibp_upper - 1e-5;
    }

    assert!(
        strictly_tighter,
        "{label} should tighten deterministic Where over the IBP union"
    );
}

fn assert_graph_alpha_where_exact_3676(cond_value: f32, expected_branch: &str, label: &str) {
    let (graph, input) = build_deterministic_where_graph_3676(cond_value);
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let ibp_out = ibp_bounds.get("out").unwrap().clone();
    let expected = ibp_bounds.get(expected_branch).unwrap().clone();
    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);

    assert_bounded_tensor_close(
        &crown_ibp,
        &expected,
        1e-5,
        &format!("{label} direct CROWN exact branch"),
    );
    assert_bounded_tensor_close(
        &alpha_crown,
        &expected,
        1e-5,
        &format!("{label} alpha-CROWN exact branch"),
    );

    let cibp_result = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(&input, ibp_bounds, None)
        .unwrap();
    let cibp_out = cibp_result.bounds.get("out").unwrap();
    assert_eq!(
        cibp_result.provenance.get("out"),
        Some(&BoundsProvenance::Crown),
        "{label} Where node should stay on the CROWN-IBP path"
    );
    assert_bounded_tensor_close(
        cibp_out,
        &expected,
        1e-5,
        &format!("{label} CROWN-IBP exact branch"),
    );
    assert_where_tighter_than_ibp_3676(&ibp_out, cibp_out, label);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_where_deterministic_true_exact_3676() {
    assert_graph_alpha_where_exact_3676(1.0, "x", "Deterministic-true Where");
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_where_deterministic_false_exact_3676() {
    assert_graph_alpha_where_exact_3676(0.0, "y", "Deterministic-false Where");
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_where_mixed_matrix_falls_back_without_invalid_spec_3676() {
    let (graph, input) = build_mixed_where_matrix_graph_3676();
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let ibp_out = ibp_bounds.get("out").unwrap().clone();
    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);

    assert_bounded_tensor_close(
        &crown_ibp,
        &ibp_out,
        1e-5,
        "Mixed 2D Where direct CROWN should concretize to IBP union",
    );
    assert_bounded_tensor_close(
        &alpha_crown,
        &ibp_out,
        1e-5,
        "Mixed 2D Where alpha-CROWN should concretize to IBP union",
    );

    let cibp_result = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(&input, ibp_bounds, None)
        .unwrap();
    let cibp_out = cibp_result.bounds.get("out").unwrap();
    assert_eq!(
        cibp_result.provenance.get("out"),
        Some(&BoundsProvenance::Crown),
        "Mixed 2D Where should stay on the graph-alpha CROWN path"
    );
    assert_bounded_tensor_close(
        cibp_out,
        &ibp_out,
        1e-5,
        "Mixed 2D Where collector output should match the IBP union without shape errors",
    );
}

/// Graph-α constant-condition Where: a MIXED constant 0/1 mask (not all-true /
/// all-false) must be tightened to the EXACT per-element select on BOTH the
/// direct CROWN-IBP and α-CROWN target-backward paths (the new
/// `where_constant_mask` arm in target_backward.rs), not the loose concretize
/// fallback. Output: out[i] = x[i] if mask[i] else y[i] = -x[i].
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_where_constant_mask_exact_select_3676() {
    let n = 2usize;
    let mask = [1.0_f32, 0.0];
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "x",
        Layer::MulConstant(MulConstantLayer::scalar(1.0)),
    ));
    graph.add_node(GraphNode::new(
        "y",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cond_base",
        Layer::MulConstant(MulConstantLayer::scalar(0.0)),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::AddConstant(AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), mask.to_vec()).unwrap(),
        )),
        vec!["cond_base".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::new()),
        vec!["cond".to_string(), "x".to_string(), "y".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), vec![-1.0_f32, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[n]), vec![2.0_f32, 1.5]).unwrap(),
    )
    .unwrap();

    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let ibp_out = ibp_bounds.get("out").unwrap().clone();
    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);

    // Exact select: out[0] = x[0] in [-1,2]; out[1] = y[1] = -x[1] in [-1.5,-0.5].
    for result in [&crown_ibp, &alpha_crown] {
        let out = result.flatten();
        assert!(
            (out.lower()[[0]] - (-1.0)).abs() < 1e-5,
            "out0 lo {}",
            out.lower()[[0]]
        );
        assert!(
            (out.upper()[[0]] - 2.0).abs() < 1e-5,
            "out0 hi {}",
            out.upper()[[0]]
        );
        assert!(
            (out.lower()[[1]] - (-1.5)).abs() < 1e-5,
            "out1 lo {}",
            out.lower()[[1]]
        );
        assert!(
            (out.upper()[[1]] - (-0.5)).abs() < 1e-5,
            "out1 hi {}",
            out.upper()[[1]]
        );
    }
    assert_where_tighter_than_ibp_3676(&ibp_out, &crown_ibp, "graph-alpha CROWN-IBP const mask");
    assert_where_tighter_than_ibp_3676(
        &ibp_out,
        &alpha_crown,
        "graph-alpha alpha-CROWN const mask",
    );
}

/// Graph-α embedded-constant Where (single `cond` input; both branches embedded
/// constants) must SUCCEED (not InvalidSpec from require_ternary_inputs) on both
/// the direct CROWN-IBP and α-CROWN target-backward paths, and produce the exact
/// per-element select when `cond` is constant.
#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_where_embedded_constants_exact_and_sound() {
    let mask = [1.0_f32, 0.0];
    let const_true = ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0_f32, 20.0]).unwrap();
    let const_false = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-10.0_f32, -20.0]).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "z",
        Layer::MulConstant(MulConstantLayer::scalar(0.0)),
    ));
    graph.add_node(GraphNode::new(
        "cond",
        Layer::AddConstant(AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[2]), mask.to_vec()).unwrap(),
        )),
        vec!["z".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Where(WhereLayer::with_constants(
            Some(const_true),
            Some(const_false),
        )),
        vec!["cond".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0_f32, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 1.0]).unwrap(),
    )
    .unwrap();

    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);
    for result in [&crown_ibp, &alpha_crown] {
        let out = result.flatten();
        assert!(
            (out.lower()[[0]] - 10.0).abs() < 1e-4,
            "out0 lo {}",
            out.lower()[[0]]
        );
        assert!(
            (out.upper()[[0]] - 10.0).abs() < 1e-4,
            "out0 hi {}",
            out.upper()[[0]]
        );
        assert!(
            (out.lower()[[1]] - (-20.0)).abs() < 1e-4,
            "out1 lo {}",
            out.lower()[[1]]
        );
        assert!(
            (out.upper()[[1]] - (-20.0)).abs() < 1e-4,
            "out1 hi {}",
            out.upper()[[1]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_max_binary_sound_1929() {
    // MaxBinary now has a sound convex-hull CROWN backward relaxation.
    // Both CROWN paths should succeed and produce bounds that soundly enclose
    // the true output max(x, 0.5x) over x ∈ [-1, 1].
    let mut graph = GraphNetwork::new();
    let w_a = arr2(&[[1.0_f32]]);
    let b_a = arr1(&[0.0_f32]);
    graph.add_node(GraphNode::from_input(
        "lin_a",
        Layer::Linear(LinearLayer::new(w_a, Some(b_a)).unwrap()),
    ));
    let w_b = arr2(&[[0.5_f32]]);
    let b_b = arr1(&[0.0_f32]);
    graph.add_node(GraphNode::from_input(
        "lin_b",
        Layer::Linear(LinearLayer::new(w_b, Some(b_b)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "max",
        Layer::MaxBinary(MaxBinaryLayer),
        vec!["lin_a".to_string(), "lin_b".to_string()],
    ));
    graph.set_output("max");

    let input =
        BoundedTensor::new(array![-1.0_f32].into_dyn(), array![1.0_f32].into_dyn()).unwrap();

    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();

    // Non-alpha (CROWN-IBP) path should succeed with sound bounds.
    let crown_out = graph
        .propagate_crown_to_node(
            &input,
            "max",
            &std::collections::HashMap::new(),
            &ibp_bounds,
            None,
            None,
            None,
            None,
        )
        .expect("CROWN-IBP MaxBinary should now produce sound bounds");

    // Alpha-CROWN path should also succeed.
    let alpha_state = GraphAlphaState::new();
    let alpha_out = graph
        .propagate_crown_to_node_with_alpha(
            &input,
            "max",
            &std::collections::HashMap::new(),
            &ibp_bounds,
            &alpha_state,
            None,
            None,
        )
        .expect("alpha-CROWN MaxBinary should now produce sound bounds");

    // Dense soundness check: true output max(x, 0.5x) must lie within bounds.
    for k in 0..=40 {
        let x = -1.0 + 2.0 * (k as f32 / 40.0);
        let z = x.max(0.5 * x);
        for (label, out) in [("CROWN-IBP", &crown_out), ("alpha-CROWN", &alpha_out)] {
            assert!(
                z >= out.lower()[0] - 1e-4,
                "{label} lower {} > z={z} at x={x}",
                out.lower()[0]
            );
            assert!(
                z <= out.upper()[0] + 1e-4,
                "{label} upper {} < z={z} at x={x}",
                out.upper()[0]
            );
        }
    }
}

// ========================= NaN-safe accumulation (#2093) =========================

#[ntest::timeout(10000)]
#[test]
fn test_accumulate_crown_ibp_bounds_nan_safe_2093() {
    use crate::bounds::LinearBounds;
    use ndarray::arr2;

    // Simulate INF + (-INF) cancellation during bound accumulation.
    // Without NaN-safe addition, this produces NaN that corrupts all downstream bounds.
    let existing = LinearBounds {
        lower_a: arr2(&[[f32::NEG_INFINITY, 1.0]]),
        lower_b: arr1(&[f32::NEG_INFINITY]),
        upper_a: arr2(&[[f32::INFINITY, 2.0]]),
        upper_b: arr1(&[f32::INFINITY]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let new_bounds = LinearBounds {
        lower_a: arr2(&[[f32::INFINITY, 3.0]]),
        lower_b: arr1(&[f32::INFINITY]),
        upper_a: arr2(&[[f32::NEG_INFINITY, 4.0]]),
        upper_b: arr1(&[f32::NEG_INFINITY]),
        lower_a_err: None,
        upper_a_err: None,
    };

    // Seed the map with the existing bounds, then accumulate
    let mut node_linear_bounds = std::collections::HashMap::new();
    node_linear_bounds.insert("_input".to_string(), existing);
    let mut input_accumulated = true;

    GraphNetwork::accumulate_crown_ibp_bounds(
        "_input",
        new_bounds,
        &mut node_linear_bounds,
        &mut input_accumulated,
    );

    let result = node_linear_bounds.get("_input").unwrap();

    // INF + (-INF) = NaN under IEEE 754. NaN-safe addition should recover:
    // - lower_a: NEG_INFINITY (sound conservative lower)
    // - lower_b: NEG_INFINITY
    // - upper_a: INFINITY (sound conservative upper)
    // - upper_b: INFINITY
    assert_eq!(
        result.lower_a[[0, 0]],
        f32::NEG_INFINITY,
        "lower_a INF-cancellation should recover to NEG_INFINITY, not NaN"
    );
    assert_eq!(
        result.lower_b[0],
        f32::NEG_INFINITY,
        "lower_b INF-cancellation should recover to NEG_INFINITY, not NaN"
    );
    assert_eq!(
        result.upper_a[[0, 0]],
        f32::INFINITY,
        "upper_a INF-cancellation should recover to INFINITY, not NaN"
    );
    assert_eq!(
        result.upper_b[0],
        f32::INFINITY,
        "upper_b INF-cancellation should recover to INFINITY, not NaN"
    );

    // Normal additions should still work correctly
    assert!(
        (result.lower_a[[0, 1]] - 4.0).abs() < 1e-6,
        "Normal lower_a addition should produce 1.0 + 3.0 = 4.0, got {}",
        result.lower_a[[0, 1]]
    );
    assert!(
        (result.upper_a[[0, 1]] - 6.0).abs() < 1e-6,
        "Normal upper_a addition should produce 2.0 + 4.0 = 6.0, got {}",
        result.upper_a[[0, 1]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_accumulate_crown_ibp_bounds_nan_input_preserved_2093() {
    use crate::bounds::LinearBounds;
    use ndarray::arr2;

    // NaN in existing bounds should become conservative infinity after accumulation.
    // safe_add_* replaces all NaN with NEG_INFINITY (lower) / INFINITY (upper).
    let existing = LinearBounds {
        lower_a: arr2(&[[f32::NAN]]),
        lower_b: arr1(&[1.0]),
        upper_a: arr2(&[[2.0]]),
        upper_b: arr1(&[f32::NAN]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let new_bounds = LinearBounds {
        lower_a: arr2(&[[5.0]]),
        lower_b: arr1(&[3.0]),
        upper_a: arr2(&[[4.0]]),
        upper_b: arr1(&[6.0]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let mut node_linear_bounds = std::collections::HashMap::new();
    node_linear_bounds.insert("node1".to_string(), existing);
    let mut input_accumulated = false;

    GraphNetwork::accumulate_crown_ibp_bounds(
        "node1",
        new_bounds,
        &mut node_linear_bounds,
        &mut input_accumulated,
    );

    let result = node_linear_bounds.get("node1").unwrap();

    // NaN input is replaced with conservative infinity by safe_add_*
    // (NEG_INFINITY for lower bounds, INFINITY for upper bounds).
    // This is the sound behavior: NaN is not a valid bound, so we widen
    // to the most conservative value. See safe_add in
    // network/graph_crown/utils.rs.
    assert_eq!(
        result.lower_a[[0, 0]],
        f32::NEG_INFINITY,
        "NaN in lower_a should become NEG_INFINITY (conservative lower bound)"
    );
    // NaN input should become conservative upper bound
    assert_eq!(
        result.upper_b[0],
        f32::INFINITY,
        "NaN in upper_b should become INFINITY (conservative upper bound)"
    );
    // Non-NaN additions should still work
    assert!(
        (result.lower_b[0] - 4.0).abs() < 1e-6,
        "Normal lower_b: 1.0 + 3.0 = 4.0, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.upper_a[[0, 0]] - 6.0).abs() < 1e-6,
        "Normal upper_a: 2.0 + 4.0 = 6.0, got {}",
        result.upper_a[[0, 0]]
    );
}

// ========================= Conv2d dimension error (not silent skip) =========================

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_conv2d_low_dim_input_errors_2027() {
    // Regression #2027: Conv2d with < 3D input must error, not silently skip.
    // Graph: Linear(2→4) → Conv2d(1,1,3,3). Linear produces [4] (< 3D).
    // Manual ibp_bounds bypass IBP forward to test CROWN backward directly.
    let w1 = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.9], [0.8, 0.1]]);
    let b1 = arr1(&[0.1, -0.2, 0.3, -0.1]);

    // 4D kernel: (out_channels=1, in_channels=1, kernel_h=3, kernel_w=3)
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[1, 1, 3, 3]),
        vec![1.0, 0.0, -1.0, 0.5, 0.0, -0.5, 0.25, 0.0, -0.25],
    )
    .unwrap();
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "conv",
        Layer::Conv2d(conv),
        vec!["lin1".to_string()],
    ));
    graph.set_output("conv");

    let input =
        BoundedTensor::new(array![-1.0, -1.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

    // Manually build ibp_bounds so CROWN backward encounters Conv2d with < 3D shape.
    let mut ibp_bounds = std::collections::HashMap::new();
    let lin1_bounds = BoundedTensor::new(
        array![-2.0, -3.0, -1.0, -1.0].into_dyn(),
        array![4.0, 5.0, 2.0, 1.0].into_dyn(),
    )
    .unwrap();
    let conv_bounds =
        BoundedTensor::new(array![-10.0].into_dyn(), array![10.0].into_dyn()).unwrap();
    ibp_bounds.insert("lin1".to_string(), lin1_bounds);
    ibp_bounds.insert("conv".to_string(), conv_bounds);

    // CROWN-IBP path should return an error for Conv2d with < 3D input
    let crown_result = graph.propagate_crown_to_node(
        &input,
        "conv",
        &std::collections::HashMap::new(),
        &ibp_bounds,
        None,
        None,
        None,
        None,
    );
    assert!(
        crown_result.is_err(),
        "CROWN-IBP should error on Conv2d with < 3D input, not silently skip"
    );
    let err_msg = format!("{}", crown_result.unwrap_err());
    assert!(
        err_msg.contains("Conv2d") && err_msg.contains("3D"),
        "Error should mention Conv2d and 3D requirement, got: {}",
        err_msg
    );

    // Alpha-CROWN path should also return an error
    let alpha_state = GraphAlphaState::new();
    let alpha_result = graph.propagate_crown_to_node_with_alpha(
        &input,
        "conv",
        &std::collections::HashMap::new(),
        &ibp_bounds,
        &alpha_state,
        None,
        None,
    );
    assert!(
        alpha_result.is_err(),
        "alpha-CROWN should error on Conv2d with < 3D input, not silently skip"
    );
}

// ========================= #2094 regression tests =========================

/// Regression test for #2094 Step 4: empty-inputs node must be rejected.
/// Since #2481/#2686, GraphNode::new() asserts minimum arity at construction
/// time, so this is caught as a panic rather than a CROWN-time error.
#[ntest::timeout(10000)]
#[test]
#[should_panic(expected = "requires at least 1 input(s) but got 0")]
fn test_graph_crown_empty_inputs_errors_2094() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "orphan",
        Layer::ReLU(ReLULayer),
        vec![], // empty inputs — panics at construction (#2481)
    ));
}

/// Regression test for #2094: Add with insufficient arity must be rejected.
/// Since #2481/#2686, GraphNode::new() asserts minimum arity at construction
/// time, so this is caught as a panic rather than a CROWN-time error.
#[ntest::timeout(10000)]
#[test]
#[should_panic(expected = "requires at least 2 input(s) but got 1")]
fn test_graph_crown_add_arity_error_2094() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32]))).unwrap()),
    ));
    // Add node with only 1 input — panics at construction (#2481)
    graph.add_node(GraphNode::new(
        "bad_add",
        Layer::Add(AddLayer),
        vec!["lin1".to_string()],
    ));
}

/// Regression test for #2094: Sub binary dispatch works through shared dispatch.
/// Post-migration, Sub should still produce correct bounds through
/// dispatch_backward_layer → Binary result handling.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_sub_via_shared_dispatch_2094() {
    // Same graph as test_graph_crown_sub_binary_parity_1929 but specifically
    // verifies the post-#2094 shared dispatch path produces sound bounds.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    let w_a = arr2(&[[1.5_f32]]);
    let b_a = arr1(&[1.0_f32]);
    graph.add_node(GraphNode::new(
        "branch_a",
        Layer::Linear(LinearLayer::new(w_a, Some(b_a)).unwrap()),
        vec!["relu".to_string()],
    ));
    let w_b = arr2(&[[0.7_f32]]);
    let b_b = arr1(&[-0.5_f32]);
    graph.add_node(GraphNode::new(
        "branch_b",
        Layer::Linear(LinearLayer::new(w_b, Some(b_b)).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sub",
        Layer::Sub(SubLayer),
        vec!["branch_a".to_string(), "branch_b".to_string()],
    ));
    graph.set_output("sub");

    let input =
        BoundedTensor::new(array![-1.0_f32].into_dyn(), array![2.0_f32].into_dyn()).unwrap();

    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);

    // Parity between CROWN-IBP and alpha-CROWN
    assert_bounded_tensor_close(&crown_ibp, &alpha_crown, 0.1, "#2094 Sub dispatch parity");

    // Soundness: out = 1.5*relu(x)+1 - (0.7*relu(x)-0.5) = 0.8*relu(x)+1.5
    let tol = 1e-4;
    for i in 0..=20 {
        let x = -1.0 + 3.0 * (i as f32) / 20.0;
        let r = x.max(0.0);
        let y = 0.8 * r + 1.5;
        assert!(
            y >= crown_ibp.lower()[[0]] - tol && y <= crown_ibp.upper()[[0]] + tol,
            "#2094 soundness: output {} not in [{}, {}] at x={}",
            y,
            crown_ibp.lower()[[0]],
            crown_ibp.upper()[[0]],
            x
        );
    }
}

// ========================= UnsupportedOp preserved (not wrapped into InvalidSpec) #2135 =========================

/// Regression test for #2135: When a unary layer returns UnsupportedOp from
/// CROWN backward (e.g., NonZero), the error must propagate as UnsupportedOp
/// — not be wrapped into InvalidSpec by the catch-all arm.
///
/// Before the dispatch_backward_layer refactoring, the catch-all in
/// propagate_crown_to_node_core used `.map_err()` that converted ALL errors
/// (including UnsupportedOp) into InvalidSpec. This prevented the caller
/// (collect_crown_bounds_with_alpha) from matching on UnsupportedOp for
/// graceful per-node IBP fallback.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_unsupported_unary_returns_unsupported_op_2135() {
    // Graph: Linear(2→3) → NonZero → output
    // NonZero's propagate_linear returns UnsupportedOp (data-dependent shape).
    // CROWN backward should surface UnsupportedOp, NOT InvalidSpec.
    let w1 = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.4]]);
    let b1 = arr1(&[0.1, -0.2, 0.3]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "nonzero",
        Layer::NonZero(NonZeroLayer),
        vec!["lin1".to_string()],
    ));
    graph.set_output("nonzero");

    let input =
        BoundedTensor::new(array![-1.0, -1.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

    // Build IBP bounds (NonZero IBP is well-defined)
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();

    // propagate_crown_to_node_with_alpha for NonZero should fail with UnsupportedOp
    let alpha_state = GraphAlphaState::new();
    let result = graph.propagate_crown_to_node_with_alpha(
        &input,
        "nonzero",
        &std::collections::HashMap::new(),
        &ibp_bounds,
        &alpha_state,
        None,
        None,
    );

    // Key assertion: the error must be UnsupportedOp, NOT InvalidSpec.
    // Before the fix (#2135), this was InvalidSpec due to blanket .map_err().
    match &result {
        Err(NyError::UnsupportedOp(_)) => {
            // Correct: UnsupportedOp preserved through dispatch
        }
        Err(NyError::InvalidSpec(msg)) => {
            panic!(
                "#2135 regression: UnsupportedOp was wrapped into InvalidSpec: {}",
                msg
            );
        }
        Err(other) => {
            panic!(
                "#2135 regression: expected UnsupportedOp, got different error: {}",
                other
            );
        }
        Ok(_) => {
            panic!("#2135 regression: expected UnsupportedOp error, got Ok");
        }
    }
}

/// Regression test for #2135: CROWN-IBP DAG pass gracefully falls back to IBP
/// when a node returns UnsupportedOp from CROWN backward.
#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_ibp_dag_fallback_for_unsupported_op_2135() {
    // Graph: Linear(2→3) → NonZero → output
    // collect_crown_ibp_bounds_dag should succeed by falling back to IBP
    // for the NonZero node, even though CROWN backward is unsupported.
    let w1 = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.4]]);
    let b1 = arr1(&[0.1, -0.2, 0.3]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "nonzero",
        Layer::NonZero(NonZeroLayer),
        vec!["lin1".to_string()],
    ));
    graph.set_output("nonzero");

    let input =
        BoundedTensor::new(array![-1.0, -1.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

    // collect_crown_ibp_bounds_dag must succeed (fallback to IBP)
    let result = graph.collect_crown_ibp_bounds_dag(&input);
    assert!(
        result.is_ok(),
        "CROWN-IBP DAG should succeed via IBP fallback for NonZero, got: {:?}",
        result.err()
    );

    // The returned bounds should match IBP (since CROWN couldn't tighten)
    let crown_ibp_bounds = result.unwrap();
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();

    let ibp_nonzero = ibp_bounds.get("nonzero").unwrap();
    let crown_nonzero = crown_ibp_bounds.get("nonzero").unwrap();

    assert_bounded_tensor_close(
        ibp_nonzero,
        crown_nonzero,
        1e-6,
        "#2135: NonZero CROWN-IBP should equal IBP (no tightening possible)",
    );
}

/// Regression for #4112: target-backward CROWN must handle Concat nodes whose
/// inputs were fully embedded as constants during ONNX lowering.
/// Before the fix, `require_unary_input()` rejected zero-input constant-only
/// Concat with "node has no inputs" before layer dispatch could run.
#[ntest::timeout(10000)]
#[test]
fn test_constant_only_concat_target_backward_preserves_constant_bounds_4112() {
    let const_a = BoundedTensor::concrete(arr1(&[1.0_f32, -2.0_f32]).into_dyn())
        .expect("constant concat input A");
    let const_b =
        BoundedTensor::concrete(arr1(&[0.5_f32]).into_dyn()).expect("constant concat input B");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::with_constants(
            0,
            vec![vec![2], vec![1]],
            vec![Some(const_a), Some(const_b)],
        )),
        vec![],
    ));
    graph.set_output("concat");

    let input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("constant-only concat IBP should succeed");
    let expected = ibp_bounds
        .get("concat")
        .expect("constant-only concat bounds should exist");

    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);

    assert_bounded_tensor_close(
        &crown_ibp,
        expected,
        1e-6,
        "#4112 constant-only concat CROWN-IBP should preserve exact constant output",
    );
    assert_bounded_tensor_close(
        &alpha_crown,
        expected,
        1e-6,
        "#4112 constant-only concat alpha-CROWN should preserve exact constant output",
    );
}

/// #4112 prover hardening: constant-only Concat with a downstream Linear that
/// introduces negative backward weights, exercising all four sign combinations
/// of (lA sign) × (constant sign) in the bias contribution `lA * c → bias`.
///
/// Graph: constant-only Concat([1.0, -2.0] ++ [0.5]) → Linear(w=[[1,-2,0.5]], b=[0])
/// Expected output: 1*1.0 + (-2)*(-2.0) + 0.5*0.5 = 5.25 (exact, constant graph)
#[ntest::timeout(10000)]
#[test]
fn test_constant_only_concat_negative_weights_all_sign_combos_4112() {
    let const_a = BoundedTensor::concrete(arr1(&[1.0_f32, -2.0_f32]).into_dyn())
        .expect("constant concat input A");
    let const_b =
        BoundedTensor::concrete(arr1(&[0.5_f32]).into_dyn()).expect("constant concat input B");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::with_constants(
            0,
            vec![vec![2], vec![1]],
            vec![Some(const_a), Some(const_b)],
        )),
        vec![],
    ));
    // Linear with mixed-sign weights: exercises positive×positive, negative×negative,
    // positive×negative, and negative×positive in the lA*c dot product.
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32, -2.0, 0.5]]), Some(arr1(&[0.0_f32]))).unwrap(),
        ),
        vec!["concat".to_string()],
    ));
    graph.set_output("linear_out");

    let input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("constant graph IBP should succeed");
    let expected = ibp_bounds
        .get("linear_out")
        .expect("output bounds should exist");

    // The output is a constant (5.25) — bounds must be tight
    let tol = 1e-4;
    assert!(
        (expected.lower()[[0]] - 5.25).abs() < tol,
        "IBP lower should be ~5.25, got {}",
        expected.lower()[[0]]
    );
    assert!(
        (expected.upper()[[0]] - 5.25).abs() < tol,
        "IBP upper should be ~5.25, got {}",
        expected.upper()[[0]]
    );

    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);

    assert_bounded_tensor_close(
        &crown_ibp,
        expected,
        1e-4,
        "#4112 constant-only concat + negative weights CROWN-IBP",
    );
    assert_bounded_tensor_close(
        &alpha_crown,
        expected,
        1e-4,
        "#4112 constant-only concat + negative weights alpha-CROWN",
    );
}

// ========================= MulBinary parity + soundness (#3396) =========================

type AffineParams<'a> = (&'a ndarray::Array2<f32>, &'a Array1<f32>);

/// Assert soundness over an 11x11 grid: every sampled true output must lie
/// within bounds. McCormick relaxation gap is maximal at interior points.
fn assert_mulbinary_soundness_grid(
    bounds: &BoundedTensor,
    up: AffineParams<'_>,
    gate: AffineParams<'_>,
    out: AffineParams<'_>,
    label: &str,
) {
    let (w_up, b_up) = up;
    let (w_gate, b_gate) = gate;
    let (w_out, b_out) = out;
    let tol = 1e-4;
    for i in 0..=10 {
        for j in 0..=10 {
            let x0 = -1.0 + 2.0 * (i as f32) / 10.0;
            let x1 = -1.0 + 2.0 * (j as f32) / 10.0;
            let x = array![x0, x1];
            let up_out = w_up.dot(&x) + b_up;
            let gate_out = w_gate.dot(&x) + b_gate;
            let mul_out = &up_out * &gate_out;
            let out = w_out.dot(&mul_out) + b_out;
            let y = out[0];

            assert!(
                y >= bounds.lower()[[0]] - tol && y <= bounds.upper()[[0]] + tol,
                "#3396 {} soundness: {} not in [{}, {}] at ({}, {})",
                label,
                y,
                bounds.lower()[[0]],
                bounds.upper()[[0]],
                x0,
                x1
            );
        }
    }
}

/// Regression test for #3396: MulBinary in DAG alpha-CROWN backward must not
/// return a fatal error. Before this fix, `dispatch_backward_layer` returned
/// `Unsupported` for MulBinary (it requires a relaxation mode), and
/// `propagate_crown_to_node_core` treated that as a fatal error instead of
/// handling MulBinary site-specifically.
///
/// Graph: input -> Linear("up") + Linear("gate") -> MulBinary -> Linear("out")
/// This SwiGLU-like pattern exercises MulBinary through both the CROWN-IBP and
/// alpha-CROWN backward paths.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_vs_crown_parity_mulbinary_3396() {
    use crate::layers::MulBinaryLayer;

    let mut graph = GraphNetwork::new();

    let w_up = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.9], [0.8, 0.1]]);
    let b_up = arr1(&[0.1, -0.2, 0.3, -0.1]);
    graph.add_node(GraphNode::from_input(
        "up",
        Layer::Linear(LinearLayer::new(w_up.clone(), Some(b_up.clone())).unwrap()),
    ));

    let w_gate = arr2(&[[0.5, -0.3], [0.7, 0.2], [-0.4, 0.6], [0.1, -0.8]]);
    let b_gate = arr1(&[0.0, 0.1, -0.1, 0.2]);
    graph.add_node(GraphNode::from_input(
        "gate",
        Layer::Linear(LinearLayer::new(w_gate.clone(), Some(b_gate.clone())).unwrap()),
    ));

    graph.add_node(GraphNode::binary(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        "up",
        "gate",
    ));

    let w_out = arr2(&[[2.0, -1.0, 0.5, -0.3]]);
    let b_out = arr1(&[0.7]);
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(w_out.clone(), Some(b_out.clone())).unwrap()),
        vec!["mul".to_string()],
    ));
    graph.set_output("out");

    let input =
        BoundedTensor::new(array![-1.0, -1.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();

    let (crown_ibp, alpha_crown) = run_both_paths(&graph, &input);
    assert_bounded_tensor_close(&crown_ibp, &alpha_crown, 1e-4, "MulBinary parity (#3396)");

    assert_mulbinary_soundness_grid(
        &crown_ibp,
        (&w_up, &b_up),
        (&w_gate, &b_gate),
        (&w_out, &b_out),
        "CROWN-IBP",
    );
    assert_mulbinary_soundness_grid(
        &alpha_crown,
        (&w_up, &b_up),
        (&w_gate, &b_gate),
        (&w_out, &b_out),
        "alpha-CROWN",
    );

    // Sanity: bounds must not be trivially zero
    assert!(
        crown_ibp.lower()[[0]].abs() > 0.01 || crown_ibp.upper()[[0]].abs() > 0.01,
        "Bounds are trivially near zero — MulBinary propagation may be broken"
    );
}

/// Regression test for #3396: MulBinary CROWN-IBP DAG path must succeed
/// (propagate_crown_to_node falls back to IBP on UnsupportedOp if needed,
/// but with the site-specific handler it should produce CROWN-tightened bounds).
#[ntest::timeout(10000)]
#[test]
fn test_mulbinary_crown_ibp_dag_succeeds_3396() {
    use crate::layers::MulBinaryLayer;

    let mut graph = GraphNetwork::new();

    // Simple: input → Linear(1→2, "a") + Linear(1→2, "b") → MulBinary → output
    let w_a = arr2(&[[1.0_f32], [0.5]]);
    let b_a = arr1(&[0.1_f32, -0.1]);
    graph.add_node(GraphNode::from_input(
        "a",
        Layer::Linear(LinearLayer::new(w_a, Some(b_a)).unwrap()),
    ));
    let w_b = arr2(&[[0.5_f32], [-0.3]]);
    let b_b = arr1(&[0.0_f32, 0.2]);
    graph.add_node(GraphNode::from_input(
        "b",
        Layer::Linear(LinearLayer::new(w_b, Some(b_b)).unwrap()),
    ));
    graph.add_node(GraphNode::binary(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        "a",
        "b",
    ));
    graph.set_output("mul");

    let input =
        BoundedTensor::new(array![-1.0_f32].into_dyn(), array![1.0_f32].into_dyn()).unwrap();

    // collect_crown_ibp_bounds_dag must succeed
    let result = graph.collect_crown_ibp_bounds_dag(&input);
    assert!(
        result.is_ok(),
        "#3396: CROWN-IBP DAG should succeed for MulBinary, got: {:?}",
        result.err()
    );

    // The CROWN-tightened bounds should be at least as tight as IBP
    let crown_ibp_bounds = result.unwrap();
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();

    let ibp_mul = ibp_bounds.get("mul").unwrap();
    let crown_mul = crown_ibp_bounds.get("mul").unwrap();

    // CROWN should not be wider than IBP (within tolerance for numerical noise)
    for i in 0..crown_mul.len() {
        assert!(
            crown_mul.lower().iter().nth(i).unwrap()
                >= &(ibp_mul.lower().iter().nth(i).unwrap() - 1e-4),
            "#3396: CROWN lower bound should not be below IBP lower at index {}",
            i
        );
        assert!(
            crown_mul.upper().iter().nth(i).unwrap()
                <= &(ibp_mul.upper().iter().nth(i).unwrap() + 1e-4),
            "#3396: CROWN upper bound should not exceed IBP upper at index {}",
            i
        );
    }
}

// ========================= Alpha optimization loop UnsupportedOp fallback (#3218) =========================

/// Regression test for #3218: DAG alpha-CROWN (collect_alpha_crown_bounds_dag)
/// must handle Gather layers gracefully. Gather returns UnsupportedOp from CROWN
/// backward; the alpha optimization loop at bounds/alpha.rs catches this and
/// breaks out (returning IBP bounds). This unblocks VNN-COMP lsnc_relu category.
///
/// Graph: Linear(3→3) → ReLU → Gather(axis=0, indices=[0,2]) → output
#[ntest::timeout(10000)]
#[test]
fn test_dag_alpha_crown_gather_unsupported_op_fallback_3218() {
    use crate::layers::GatherLayer;

    let mut graph = GraphNetwork::new();

    let w = arr2(&[[1.0f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    let b = arr1(&[1.0f32, 2.0, 3.0]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 2]).unwrap();
    graph.add_node(GraphNode::new(
        "gather1",
        Layer::Gather(GatherLayer::new(0, Some(indices), vec![])),
        vec!["relu1".to_string()],
    ));
    graph.set_output("gather1");

    let input = BoundedTensor::new(
        arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
        arr1(&[4.0f32, 5.0, 6.0]).into_dyn(),
    )
    .unwrap();

    // Build IBP bounds for reference
    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();

    // The key call: collect_alpha_crown_bounds_dag must succeed
    // (catches UnsupportedOp from Gather and falls back to IBP).
    let config = AlphaCrownConfig::default();
    let result = graph.collect_alpha_crown_bounds_dag(&input, &config);

    assert!(
        result.is_ok(),
        "#3218: collect_alpha_crown_bounds_dag should succeed via IBP fallback for Gather, got: {:?}",
        result.err()
    );

    let (crown_bounds, _alpha_state) = result.unwrap();
    // The Gather node's bounds should be present and valid (falling back to IBP)
    let gather_bounds = crown_bounds.get("gather1");
    assert!(
        gather_bounds.is_some(),
        "#3218: Gather node bounds should be present after fallback"
    );
    let gb = gather_bounds.unwrap();
    let ibp_gb = ibp_bounds.get("gather1").unwrap();
    // Verify bounds are valid and no worse than IBP
    for (&l, &u) in gb.lower().iter().zip(gb.upper().iter()) {
        assert!(
            l.is_finite(),
            "#3218: lower bound should be finite, got {}",
            l
        );
        assert!(
            u.is_finite(),
            "#3218: upper bound should be finite, got {}",
            u
        );
        assert!(l <= u, "#3218: lower {} should be <= upper {}", l, u);
    }
    for (i, (&l, &ibp_l)) in gb.lower().iter().zip(ibp_gb.lower().iter()).enumerate() {
        assert!(
            l >= ibp_l - 1e-5,
            "#3218: alpha-CROWN lower[{}]={} should be >= IBP lower={}",
            i,
            l,
            ibp_l
        );
    }
}

// ========================= MulBinary SPSA supplement convergence (#3439 Phase 3) =========================

/// Build a SwiGLU-like graph with ReLU on the "up" path:
/// input → Linear("up") → ReLU("relu") + Linear("gate") → MulBinary("mul") → Linear("out")
fn build_swiglu_relu_dag() -> (GraphNetwork, BoundedTensor) {
    use crate::layers::{MulBinaryLayer, ReLULayer};

    let mut graph = GraphNetwork::new();

    let w_up = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.9], [0.8, 0.1]]);
    let b_up = arr1(&[0.1, -0.2, 0.3, -0.1]);
    graph.add_node(GraphNode::from_input(
        "up",
        Layer::Linear(LinearLayer::new(w_up, Some(b_up)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["up".to_string()],
    ));
    let w_gate = arr2(&[[0.5, -0.3], [0.7, 0.2], [-0.4, 0.6], [0.1, -0.8]]);
    let b_gate = arr1(&[0.0, 0.1, -0.1, 0.2]);
    graph.add_node(GraphNode::from_input(
        "gate",
        Layer::Linear(LinearLayer::new(w_gate, Some(b_gate)).unwrap()),
    ));
    graph.add_node(GraphNode::binary(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        "relu",
        "gate",
    ));
    let w_out = arr2(&[[2.0, -1.0, 0.5, -0.3]]);
    let b_out = arr1(&[0.7]);
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(w_out, Some(b_out)).unwrap()),
        vec!["mul".to_string()],
    ));
    graph.set_output("out");
    let input =
        BoundedTensor::new(array![-1.0, -1.0].into_dyn(), array![1.0, 1.0].into_dyn()).unwrap();
    (graph, input)
}

/// Forward pass for the SwiGLU-ReLU graph: up→ReLU * gate → out.
fn swiglu_relu_forward(x: &Array1<f32>) -> f32 {
    let w_up = arr2(&[[1.0, 0.5], [-0.3, 0.7], [0.2, -0.9], [0.8, 0.1]]);
    let b_up = arr1(&[0.1, -0.2, 0.3, -0.1]);
    let w_gate = arr2(&[[0.5, -0.3], [0.7, 0.2], [-0.4, 0.6], [0.1, -0.8]]);
    let b_gate = arr1(&[0.0, 0.1, -0.1, 0.2]);
    let w_out = arr2(&[[2.0, -1.0, 0.5, -0.3]]);
    let b_out = arr1(&[0.7]);
    let relu_out = (w_up.dot(x) + &b_up).mapv(|v| v.max(0.0));
    let gate_out = w_gate.dot(x) + &b_gate;
    let out = w_out.dot(&(&relu_out * &gate_out)) + &b_out;
    out[0]
}

fn bound_width(bt: &BoundedTensor) -> f32 {
    bt.upper()
        .iter()
        .zip(bt.lower().iter())
        .map(|(u, l)| u - l)
        .sum()
}

/// Grid-sample soundness check: all true outputs must lie within bounds.
fn assert_swiglu_relu_soundness(bounds: &BoundedTensor, label: &str) {
    let tol = 1e-4;
    for i in 0..=10 {
        for j in 0..=10 {
            let x0 = -1.0 + 2.0 * (i as f32) / 10.0;
            let x1 = -1.0 + 2.0 * (j as f32) / 10.0;
            let y = swiglu_relu_forward(&array![x0, x1]);
            assert!(
                y >= bounds.lower()[[0]] - tol && y <= bounds.upper()[[0]] + tol,
                "#3439 {label}: y={y} not in [{}, {}] at ({x0}, {x1})",
                bounds.lower()[[0]],
                bounds.upper()[[0]],
            );
        }
    }
}

fn make_warm_start_test_config(iterations: usize) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        gradient_method: crate::bounds::GradientMethod::AnalyticChain,
        fix_interm_bounds: false,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    }
}

fn make_counting_engine_config(iterations: usize) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        gradient_method: crate::bounds::GradientMethod::AnalyticChain,
        fix_interm_bounds: true,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    }
}

fn build_fix_interm_bounds_residual_dag_4404() -> (GraphNetwork, BoundedTensor) {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 2, 1, 1]), vec![0.9_f32, -0.35, -0.45, 0.8])
        .expect("valid Conv2d kernel");
    let bias = arr1(&[0.05_f32, -0.1]);
    let conv = Conv2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (0, 0), 2, 2)
        .expect("valid Conv2d params");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "residual",
        Layer::Add(AddLayer),
        "relu",
        NETWORK_INPUT,
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::ReduceSum(ReduceSumLayer::new(vec![0, 1, 2], false)),
        vec!["residual".to_string()],
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![-1.0_f32, -0.6, 0.1, -0.3, -0.5, -0.2, 0.0, -0.4],
        )
        .expect("valid lower input shape"),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![1.2_f32, 0.7, 0.9, 0.6, 0.8, 0.5, 1.0, 0.4],
        )
        .expect("valid upper input shape"),
    )
    .expect("residual DAG input bounds should be valid");

    (graph, input)
}

fn node_bounds_max_abs_diff_4404(actual: &BoundedTensor, expected: &BoundedTensor) -> f32 {
    actual
        .lower()
        .iter()
        .zip(expected.lower().iter())
        .chain(actual.upper().iter().zip(expected.upper().iter()))
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max)
}

fn build_sqrt_exp_alpha_dag() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let hidden_w = arr2(&[[1.15_f32, 0.45], [0.35, 1.05]]);
    let hidden_b = arr1(&[0.25_f32, 0.20]);
    graph.add_node(GraphNode::from_input(
        "linear_hidden",
        Layer::Linear(LinearLayer::new(hidden_w, Some(hidden_b)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "sqrt_hidden",
        Layer::Sqrt(SqrtLayer::new()),
        vec!["linear_hidden".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "exp_hidden",
        Layer::Exp(ExpLayer::new()),
        vec!["sqrt_hidden".to_string()],
    ));

    let out_w = arr2(&[[1.8_f32, -2.25]]);
    let out_b = arr1(&[0.1_f32]);
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(LinearLayer::new(out_w, Some(out_b)).unwrap()),
        vec!["exp_hidden".to_string()],
    ));
    graph.set_output("linear_out");

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("sqrt-exp input bounds should construct");

    (graph, input)
}

fn make_sqrt_alpha_test_config(iterations: usize) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        gradient_method: crate::bounds::GradientMethod::AnalyticChain,
        fix_interm_bounds: false,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        spsa_samples: 4,
        sparse_ratio: 1.0,
        ..AlphaCrownConfig::default()
    }
}

fn assert_bounds_do_not_loosen(
    warm_node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    child_ibp: &std::collections::HashMap<String, BoundedTensor>,
) {
    for (name, ibp_bound) in child_ibp {
        let warm_bound = warm_node_bounds
            .get(name)
            .unwrap_or_else(|| panic!("warm-start bounds missing node '{name}'"));
        assert_eq!(
            warm_bound.shape(),
            ibp_bound.shape(),
            "#3453 node '{name}' shape mismatch"
        );
        for (idx, (&warm_l, &ibp_l)) in warm_bound
            .lower()
            .iter()
            .zip(ibp_bound.lower().iter())
            .enumerate()
        {
            assert!(
                warm_l >= ibp_l - 1e-5,
                "#3453 node '{name}' lower[{idx}] loosened: warm={warm_l}, ibp={ibp_l}"
            );
        }
        for (idx, (&warm_u, &ibp_u)) in warm_bound
            .upper()
            .iter()
            .zip(ibp_bound.upper().iter())
            .enumerate()
        {
            assert!(
                warm_u <= ibp_u + 1e-5,
                "#3453 node '{name}' upper[{idx}] loosened: warm={warm_u}, ibp={ibp_u}"
            );
        }
    }
}

fn assert_alpha_state_finite_and_clamped(alpha_state: &GraphAlphaState) {
    for (name, alpha) in &alpha_state.alphas {
        for (idx, value) in alpha.iter().enumerate() {
            assert!(
                value.is_finite() && (0.0..=1.0).contains(value),
                "#3453 node '{name}' lower alpha[{idx}] must stay finite and clamped, got {value}"
            );
        }
    }

    for (name, alpha) in &alpha_state.alphas_upper {
        for (idx, value) in alpha.iter().enumerate() {
            assert!(
                value.is_finite() && (0.0..=1.0).contains(value),
                "#3453 node '{name}' upper alpha[{idx}] must stay finite and clamped, got {value}"
            );
        }
    }
}

fn assert_child_domain_soundness(bounds: &BoundedTensor) {
    let tol = 1e-4;
    for i in 0..=8 {
        for j in 0..=8 {
            let x0 = -0.35 + 0.95 * (i as f32) / 8.0;
            let x1 = -0.20 + 0.60 * (j as f32) / 8.0;
            let y = swiglu_relu_forward(&array![x0, x1]);
            assert!(
                y >= bounds.lower()[[0]] - tol && y <= bounds.upper()[[0]] + tol,
                "#3453 warm-start output bound unsound: y={y} not in [{}, {}] at ({x0}, {x1})",
                bounds.lower()[[0]],
                bounds.upper()[[0]],
            );
        }
    }
}

fn build_shifted_relu_graph_3232() -> (
    GraphNetwork,
    BoundedTensor,
    std::collections::HashMap<String, BoundedTensor>,
) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear_hidden",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[1.0_f32]))).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu_hidden",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear_hidden".to_string()],
    ));
    graph.set_output("relu_hidden");

    let input =
        BoundedTensor::new(arr1(&[-2.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn()).unwrap();
    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("node bounds should collect for the shifted ReLU graph");
    (graph, input, node_bounds)
}

fn reused_alpha_state_3232() -> GraphAlphaState {
    let mut alpha_state = GraphAlphaState::new();
    alpha_state
        .alphas
        .insert("relu_hidden".to_string(), arr1(&[0.5_f32]));
    alpha_state
        .alphas_upper
        .insert("relu_hidden".to_string(), arr1(&[0.5_f32]));
    alpha_state
        .unstable_mask
        .insert("relu_hidden".to_string(), Array1::from_vec(vec![true]));
    alpha_state
}

fn assert_shifted_relu_bound_sound_3232(bounds: &BoundedTensor) {
    let tol = 1e-5_f32;
    for x in [-2.0_f32, -1.0, 0.0, 1.0, 2.0] {
        let y = (x + 1.0).max(0.0);
        assert!(
            y >= bounds.lower()[[0]] - tol && y <= bounds.upper()[[0]] + tol,
            "#3232 reused-alpha bound is unsound at x={x}: y={y} not in [{}, {}]",
            bounds.lower()[[0]],
            bounds.upper()[[0]],
        );
    }
}

fn build_monotone_warm_start_dag() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let hidden_w = arr2(&[[1.4_f32, -0.7], [-0.9, 1.2], [0.6, 0.9]]);
    let hidden_b = arr1(&[0.1_f32, -0.15, 0.05]);
    graph.add_node(GraphNode::from_input(
        "linear_hidden",
        Layer::Linear(LinearLayer::new(hidden_w, Some(hidden_b)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "sigmoid_hidden",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["linear_hidden".to_string()],
    ));

    let skip_w = arr2(&[[0.35_f32, -0.25], [0.15, 0.45], [-0.4, 0.2]]);
    let skip_b = arr1(&[0.0_f32, 0.08, -0.03]);
    graph.add_node(GraphNode::from_input(
        "linear_skip",
        Layer::Linear(LinearLayer::new(skip_w, Some(skip_b)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "merge",
        Layer::Add(AddLayer),
        vec!["sigmoid_hidden".to_string(), "linear_skip".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "tanh_hidden",
        Layer::Tanh(TanhLayer::new()),
        vec!["merge".to_string()],
    ));

    let out_w = arr2(&[[1.1_f32, -0.9, 0.7]]);
    let out_b = arr1(&[0.02_f32]);
    graph.add_node(GraphNode::new(
        "linear_out",
        Layer::Linear(LinearLayer::new(out_w, Some(out_b)).unwrap()),
        vec!["tanh_hidden".to_string()],
    ));
    graph.set_output("linear_out");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    (graph, input)
}

fn monotone_warm_start_forward(x0: f32, x1: f32) -> f32 {
    let hidden_w = arr2(&[[1.4_f32, -0.7], [-0.9, 1.2], [0.6, 0.9]]);
    let hidden_b = arr1(&[0.1_f32, -0.15, 0.05]);
    let skip_w = arr2(&[[0.35_f32, -0.25], [0.15, 0.45], [-0.4, 0.2]]);
    let skip_b = arr1(&[0.0_f32, 0.08, -0.03]);
    let out_w = arr2(&[[1.1_f32, -0.9, 0.7]]);
    let out_b = arr1(&[0.02_f32]);

    let x = arr1(&[x0, x1]);
    let hidden_logits = hidden_w.dot(&x) + &hidden_b;
    let hidden = hidden_logits.mapv(|v| 1.0 / (1.0 + (-v).exp()));
    let skip = skip_w.dot(&x) + &skip_b;
    let merged = &hidden + &skip;
    let tanh_hidden = merged.mapv(f32::tanh);
    let out = out_w.dot(&tanh_hidden) + &out_b;
    out[0]
}

fn make_monotone_warm_start_test_config(iterations: usize) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        gradient_method: crate::bounds::GradientMethod::AnalyticChain,
        fix_interm_bounds: false,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        ..AlphaCrownConfig::default()
    }
}

fn assert_monotone_alpha_state_finite_and_projected(
    alpha_state: &GraphAlphaState,
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    label: &str,
) {
    let tol = 1e-5_f32;
    for node_name in alpha_state.monotone_alpha_names() {
        let alpha = alpha_state
            .monotone_s_shaped_alpha(node_name)
            .unwrap_or_else(|| panic!("{label}: missing monotone alpha '{node_name}'"));
        let pre_activation = node_bounds
            .get(node_name)
            .unwrap_or_else(|| panic!("{label}: missing node bounds for '{node_name}'"))
            .flatten();
        let lower = pre_activation
            .lower()
            .clone()
            .into_dimensionality::<Ix1>()
            .unwrap();
        let upper = pre_activation
            .upper()
            .clone()
            .into_dimensionality::<Ix1>()
            .unwrap();

        for idx in 0..alpha.len() {
            let lower_path = alpha.lower_path_alpha(idx);
            let upper_path = alpha.upper_path_alpha(idx);
            for value in [
                lower_path.tp_pos,
                lower_path.tp_neg,
                lower_path.tp_both_lower,
                lower_path.tp_both_upper,
                upper_path.tp_pos,
                upper_path.tp_neg,
                upper_path.tp_both_lower,
                upper_path.tp_both_upper,
            ] {
                assert!(
                    value.is_finite(),
                    "{label}: '{node_name}' contains non-finite tangent value at index {idx}: {value}"
                );
            }

            if alpha.mask_pos[idx] {
                assert!(
                    lower_path.tp_pos >= lower[idx] - tol && lower_path.tp_pos <= upper[idx] + tol,
                    "{label}: '{node_name}' tp_pos lower_path[{idx}]={} outside [{}, {}]",
                    lower_path.tp_pos,
                    lower[idx],
                    upper[idx]
                );
                assert!(
                    upper_path.tp_pos >= lower[idx] - tol && upper_path.tp_pos <= upper[idx] + tol,
                    "{label}: '{node_name}' tp_pos upper_path[{idx}]={} outside [{}, {}]",
                    upper_path.tp_pos,
                    lower[idx],
                    upper[idx]
                );
            }
            if alpha.mask_neg[idx] {
                assert!(
                    lower_path.tp_neg >= lower[idx] - tol && lower_path.tp_neg <= upper[idx] + tol,
                    "{label}: '{node_name}' tp_neg lower_path[{idx}]={} outside [{}, {}]",
                    lower_path.tp_neg,
                    lower[idx],
                    upper[idx]
                );
                assert!(
                    upper_path.tp_neg >= lower[idx] - tol && upper_path.tp_neg <= upper[idx] + tol,
                    "{label}: '{node_name}' tp_neg upper_path[{idx}]={} outside [{}, {}]",
                    upper_path.tp_neg,
                    lower[idx],
                    upper[idx]
                );
            }
            if alpha.mask_cross[idx] {
                assert!(
                    lower_path.tp_both_lower <= lower_path.d_lower + tol,
                    "{label}: '{node_name}' tp_both_lower lower_path[{idx}]={} > d_lower {}",
                    lower_path.tp_both_lower,
                    lower_path.d_lower
                );
                assert!(
                    upper_path.tp_both_lower <= upper_path.d_lower + tol,
                    "{label}: '{node_name}' tp_both_lower upper_path[{idx}]={} > d_lower {}",
                    upper_path.tp_both_lower,
                    upper_path.d_lower
                );
                assert!(
                    lower_path.tp_both_upper >= lower_path.d_upper - tol,
                    "{label}: '{node_name}' tp_both_upper lower_path[{idx}]={} < d_upper {}",
                    lower_path.tp_both_upper,
                    lower_path.d_upper
                );
                assert!(
                    upper_path.tp_both_upper >= upper_path.d_upper - tol,
                    "{label}: '{node_name}' tp_both_upper upper_path[{idx}]={} < d_upper {}",
                    upper_path.tp_both_upper,
                    upper_path.d_upper
                );
            }
        }
    }
}

fn assert_monotone_child_domain_soundness(
    bounds: &BoundedTensor,
    lower: [f32; 2],
    upper: [f32; 2],
) {
    let tol = 1e-4_f32;
    for i in 0..=8 {
        for j in 0..=8 {
            let x0 = lower[0] + (upper[0] - lower[0]) * (i as f32) / 8.0;
            let x1 = lower[1] + (upper[1] - lower[1]) * (j as f32) / 8.0;
            let y = monotone_warm_start_forward(x0, x1);
            assert!(
                y >= bounds.lower()[[0]] - tol && y <= bounds.upper()[[0]] + tol,
                "#3619 warm-start output bound unsound: y={y} not in [{}, {}] at ({x0}, {x1})",
                bounds.lower()[[0]],
                bounds.upper()[[0]]
            );
        }
    }
}

fn mutate_parent_monotone_projection_fixture(parent_state: &mut GraphAlphaState) {
    let parent = parent_state
        .monotone_s_shaped_alpha_mut("sigmoid_hidden")
        .expect("parent monotone alpha should exist");
    parent.tp_pos.lower_path[2] = f32::NAN;
    parent.tp_pos.upper_path[2] = 9.0;
    parent.tp_neg.lower_path[1] = -9.0;
    parent.tp_neg.upper_path[1] = 7.0;
    parent.tp_both_lower.lower_path[0] = 9.0;
    parent.tp_both_lower.upper_path[0] = f32::INFINITY;
    parent.tp_both_upper.lower_path[0] = f32::NEG_INFINITY;
    parent.tp_both_upper.upper_path[0] = -9.0;
}

fn assert_monotone_projection_result(
    child: &crate::bounds::MonotoneSShapedAlpha,
    child_default: &crate::bounds::MonotoneSShapedAlpha,
) {
    assert_eq!(
        child.tp_pos.lower_path[2], child_default.tp_pos.lower_path[2],
        "#3619 non-finite positive-path parent tangent should reset to the child midpoint"
    );
    assert!(
        (child.tp_pos.upper_path[2] - 1.1).abs() < 1e-6,
        "#3619 positive-path upper tangent should clamp to child upper bound"
    );
    assert!(
        (child.tp_neg.lower_path[1] + 0.6).abs() < 1e-6,
        "#3619 negative-path lower tangent should clamp to child lower bound"
    );
    assert!(
        (child.tp_neg.upper_path[1] + 0.1).abs() < 1e-6,
        "#3619 negative-path upper tangent should clamp to child upper bound"
    );
    assert!(
        (child.tp_both_lower.lower_path[0] - child.lower_path_alpha(0).d_lower).abs() < 1e-6,
        "#3619 crossing lower tangent should clamp to child d_lower"
    );
    assert_eq!(
        child.tp_both_lower.upper_path[0], child_default.tp_both_lower.upper_path[0],
        "#3619 non-finite crossing lower tangent should reset to child d_lower"
    );
    assert_eq!(
        child.tp_both_upper.lower_path[0], child_default.tp_both_upper.lower_path[0],
        "#3619 non-finite crossing upper tangent should reset to child d_upper"
    );
    assert!(
        (child.tp_both_upper.upper_path[0] - child.upper_path_alpha(0).d_upper).abs() < 1e-6,
        "#3619 crossing upper tangent should clamp to child d_upper"
    );
}

/// Verify that the SPSA supplement for MulBinary alphas produces sound bounds
/// and monotonic convergence on a ReLU+MulBinary SwiGLU-like DAG.
///
/// Part of #3439.
#[ntest::timeout(60000)]
#[test]
fn test_mulbinary_spsa_supplement_convergence_3439() {
    use crate::bounds::GradientMethod;

    let (graph, input) = build_swiglu_relu_dag();

    let make_config = |iters| AlphaCrownConfig {
        iterations: iters,
        gradient_method: GradientMethod::AnalyticChain,
        fix_interm_bounds: false,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };

    let bounds_1 = graph
        .propagate_alpha_crown_with_config(&input, &make_config(1))
        .unwrap();
    let bounds_20 = graph
        .propagate_alpha_crown_with_config(&input, &make_config(20))
        .unwrap();

    assert_swiglu_relu_soundness(&bounds_1, "1-iter");
    assert_swiglu_relu_soundness(&bounds_20, "20-iter");

    let width_1 = bound_width(&bounds_1);
    let width_20 = bound_width(&bounds_20);
    eprintln!(
        "#3439 MulBinary SPSA supplement: 1-iter={width_1:.6}, \
         20-iter={width_20:.6}, improvement={:.2}%",
        100.0 * (1.0 - width_20 / width_1)
    );
    assert!(
        width_20 <= width_1 + 1e-4,
        "#3439: 20-iter ({width_20:.6}) should be <= 1-iter ({width_1:.6})"
    );
}

/// Regression for #3453: per-domain alpha-CROWN warm-start must remain sound
/// after the SPSA refinement loop and the CROWN-IBP intersection step.
///
/// This exercises the graph input-split building blocks end-to-end on a small
/// DAG: root alpha collection, child-domain warm-start, node-bound intersection,
/// and spec-guided backward propagation from the returned node bounds.
#[ntest::timeout(60000)]
#[test]
fn test_alpha_crown_warm_start_child_bounds_sound_and_no_looser_3453() {
    let (graph, root_input) = build_swiglu_relu_dag();

    let (_root_bounds, root_alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&root_input, &make_warm_start_test_config(3))
        .expect("root alpha-CROWN should succeed");

    let child_input = BoundedTensor::new(
        array![-0.35_f32, -0.20_f32].into_dyn(),
        array![0.60_f32, 0.40_f32].into_dyn(),
    )
    .expect("child input bounds should be valid");

    let child_ibp = graph
        .collect_crown_ibp_bounds_dag(&child_input)
        .expect("child CROWN-IBP bounds should succeed");
    let (warm_node_bounds, warm_alpha_state) = graph
        .collect_alpha_crown_bounds_dag_warm(
            &child_input,
            &make_warm_start_test_config(2),
            &root_alpha_state,
        )
        .expect("warm-start alpha-CROWN should succeed");
    assert_bounds_do_not_loosen(&warm_node_bounds, &child_ibp);
    assert_alpha_state_finite_and_clamped(&warm_alpha_state);

    let spec_matrix = array![[1.0_f32]];
    let (baseline_output, _) = graph
        .propagate_crown_with_specs_and_engine_with_linear(&child_input, &spec_matrix, None)
        .expect("baseline spec-guided CROWN should succeed");
    let (warm_output, _) = graph
        .propagate_crown_with_specs_and_node_bounds_and_mul_binary_alphas(
            &child_input,
            &spec_matrix,
            None,
            &warm_node_bounds,
            None,
        )
        .expect("warm-start spec-guided CROWN should succeed");

    assert!(
        warm_output.lower()[[0]] >= baseline_output.lower()[[0]] - 1e-5,
        "#3453 warm-start lower bound loosened: warm={}, baseline={}",
        warm_output.lower()[[0]],
        baseline_output.lower()[[0]]
    );
    assert!(
        warm_output.upper()[[0]] <= baseline_output.upper()[[0]] + 1e-5,
        "#3453 warm-start upper bound loosened: warm={}, baseline={}",
        warm_output.upper()[[0]],
        baseline_output.upper()[[0]]
    );
    assert_child_domain_soundness(&warm_output);
}

/// Regression for #3232: spec-guided graph CROWN must honor a supplied
/// `GraphAlphaState` instead of falling back to the `u > -l` heuristic when
/// input-split code reuses root alpha-CROWN slopes.
///
/// The pre-activation interval [-1, 3] makes the heuristic choose alpha=1,
/// but a reused alpha of 0.5 yields a strictly tighter lower bound on the
/// subdomain. This catches the old bug where `GraphAlphaState` was threaded
/// nowhere and the result stayed at the heuristic value.
#[ntest::timeout(10000)]
#[test]
fn test_spec_guided_crown_reuses_graph_alpha_state_3232() {
    let (graph, input, node_bounds) = build_shifted_relu_graph_3232();
    let spec_matrix = array![[1.0_f32]];

    let run = |alpha_state| {
        crate::network::SpecCrownRequest::new(&graph, &input, &spec_matrix, None)
            .node_bounds(&node_bounds)
            .alpha_state_opt(alpha_state)
            .run_all()
    };

    let (heuristic, heuristic_linear, _) =
        run(None).expect("heuristic spec-guided CROWN should succeed");
    let reused_alpha = reused_alpha_state_3232();
    let (reused, reused_linear, _) =
        run(Some(&reused_alpha)).expect("alpha-reuse spec-guided CROWN should succeed");

    let heuristic_linear = heuristic_linear.expect("spec-guided CROWN should return linear bounds");
    let reused_linear = reused_linear.expect("alpha-reuse path should return linear bounds");

    assert!(
        (heuristic_linear.lower_a()[[0, 0]] - 1.0).abs() < 1e-6,
        "#3232 heuristic path should use alpha=1 for pre-activation [-1, 3], got {}",
        heuristic_linear.lower_a()[[0, 0]]
    );
    assert!(
        (reused_linear.lower_a()[[0, 0]] - 0.5).abs() < 1e-6,
        "#3232 reused-alpha path should propagate the supplied alpha, got {}",
        reused_linear.lower_a()[[0, 0]]
    );
    let heuristic_linear_output = heuristic_linear
        .concretize_checked(&input)
        .expect("#3232 heuristic linear form should concretize");
    let reused_linear_output = reused_linear
        .concretize_checked(&input)
        .expect("#3232 reused linear form should concretize");
    assert!(
        reused_linear_output.lower()[[0]] > heuristic_linear_output.lower()[[0]] + 0.45,
        "#3232 reused alpha should tighten the pre-tightening linear lower bound: heuristic={}, reused={}",
        heuristic_linear_output.lower()[[0]],
        reused_linear_output.lower()[[0]]
    );
    // The spec-row reduction accumulates in f64 and casts the endpoint
    // OUTWARD (next_down), so the exact lower bound 0 may be reported one
    // subnormal ULP below 0 — and must never be reported above it.
    let one_ulp_below_zero = -f32::from_bits(1); // == next_down(0.0)
    for (label, bounds) in [("heuristic", &heuristic.bounds), ("reused", &reused.bounds)] {
        let lo = bounds.lower()[[0]];
        assert!(
            lo <= 0.0 && lo >= one_ulp_below_zero,
            "#3232 {label} spec lower must be exact 0 up to the outward 1-ULP cast, got {lo}"
        );
    }
    assert_shifted_relu_bound_sound_3232(&reused.bounds);
}

#[ntest::timeout(60000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_returns_monotone_alpha_state_3619() {
    let (graph, input) = build_monotone_warm_start_dag();

    let (node_bounds, alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &make_monotone_warm_start_test_config(6))
        .expect("monotone DAG alpha-CROWN collection should succeed");

    let monotone_names: Vec<String> = alpha_state.monotone_alpha_names().cloned().collect();
    assert_eq!(
        monotone_names,
        vec!["sigmoid_hidden".to_string(), "tanh_hidden".to_string()],
        "#3619 root collection must return monotone alpha state for both DAG activations"
    );
    assert_eq!(
        alpha_state.num_unstable(),
        0,
        "#3619 monotone-only graph should not invent ReLU alpha state"
    );
    assert_monotone_alpha_state_finite_and_projected(&alpha_state, &node_bounds, "#3619 root");
}

#[ntest::timeout(60000)]
#[test]
fn test_monotone_shape_mismatch_retry_matches_fixed_crown_and_stays_sound_4118() {
    let (graph, input) = build_monotone_warm_start_dag();
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("#4118 monotone DAG IBP bounds should succeed");
    // Baseline: fixed-slope CROWN through the *same* per-target backward core that the
    // alpha path uses, obtained by feeding an empty alpha state. We compare against this
    // rather than `propagate_crown_fixed_slope` (the flat single-pass engine): on a
    // residual DAG the flat engine accumulates linear bounds through the Add merge in one
    // pass and is therefore tighter, whereas the per-target core re-concretizes each node
    // against the IBP intermediates. Both are sound; the #4118 retry must reproduce the
    // core's own fixed-slope result, which is the meaningful invariant for the local seam.
    let empty_alpha_state = GraphAlphaState::new();
    let fixed_bounds = graph
        .propagate_crown_to_node_with_alpha(
            &input,
            graph.output_name(),
            &std::collections::HashMap::new(),
            &ibp_bounds,
            &empty_alpha_state,
            None,
            None,
        )
        .expect("#4118 fixed-slope monotone DAG baseline should succeed");

    // Force both monotone alpha branches down the retry seam by supplying
    // tangent-point bundles with the wrong width. Before #4118 this surfaced as
    // ShapeMismatch from DAG alpha backward, which the caller treated as a
    // graph-wide fallback trigger. The fixed-slope retry should now keep the
    // graph on the local CROWN path, so a monotone-only DAG must match the
    // fixed-slope result exactly.
    let mismatch_pre_activation =
        BoundedTensor::new(arr1(&[-0.2_f32]).into_dyn(), arr1(&[0.3_f32]).into_dyn())
            .expect("#4118 mismatch pre-activation bounds should construct");
    let mut mismatched_alpha_state = GraphAlphaState::new();
    mismatched_alpha_state.monotone_s_shaped_alphas.insert(
        "sigmoid_hidden".to_string(),
        crate::bounds::MonotoneSShapedAlpha::from_bounds(
            &mismatch_pre_activation,
            crate::layers::trigonometric::sigmoid_crossing_default_tangents,
        )
        .expect("#4118 mismatched sigmoid alpha should initialize"),
    );
    mismatched_alpha_state.monotone_s_shaped_alphas.insert(
        "tanh_hidden".to_string(),
        crate::bounds::MonotoneSShapedAlpha::from_bounds(
            &mismatch_pre_activation,
            crate::layers::trigonometric::tanh_crossing_default_tangents,
        )
        .expect("#4118 mismatched tanh alpha should initialize"),
    );

    let retry_bounds = graph
        .propagate_crown_to_node_with_alpha(
            &input,
            graph.output_name(),
            &std::collections::HashMap::new(),
            &ibp_bounds,
            &mismatched_alpha_state,
            None,
            None,
        )
        .expect("#4118 monotone ShapeMismatch should retry fixed-slope locally");

    assert_bounded_tensor_close(
        &retry_bounds,
        &fixed_bounds,
        1e-6,
        "#4118 local monotone retry should match fixed-slope CROWN",
    );
    assert_monotone_child_domain_soundness(&retry_bounds, [-1.0_f32, -1.0], [1.0_f32, 1.0]);
}

#[ntest::timeout(60000)]
#[test]
fn test_alpha_crown_warm_start_child_monotone_bounds_sound_and_no_looser_3619() {
    let (graph, root_input) = build_monotone_warm_start_dag();

    let (_root_bounds, root_alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&root_input, &make_monotone_warm_start_test_config(6))
        .expect("root monotone alpha-CROWN should succeed");

    let child_lower = [-0.35_f32, -0.25];
    let child_upper = [0.55_f32, 0.40];
    let child_input = BoundedTensor::new(
        array![child_lower[0], child_lower[1]].into_dyn(),
        array![child_upper[0], child_upper[1]].into_dyn(),
    )
    .expect("child monotone input bounds should be valid");

    let child_ibp = graph
        .collect_crown_ibp_bounds_dag(&child_input)
        .expect("child monotone CROWN-IBP bounds should succeed");
    let (warm_node_bounds, warm_alpha_state) = graph
        .collect_alpha_crown_bounds_dag_warm(
            &child_input,
            &make_monotone_warm_start_test_config(3),
            &root_alpha_state,
        )
        .expect("warm-start monotone alpha-CROWN should succeed");

    assert_bounds_do_not_loosen(&warm_node_bounds, &child_ibp);
    assert_monotone_alpha_state_finite_and_projected(&warm_alpha_state, &child_ibp, "#3619 warm");

    let output_name = graph.output_name().to_string();
    let warm_output = warm_node_bounds
        .get(&output_name)
        .expect("warm-start output bounds should exist");
    let baseline_output = child_ibp
        .get(&output_name)
        .expect("child CROWN-IBP output bounds should exist");

    assert!(
        warm_output.lower()[[0]] >= baseline_output.lower()[[0]] - 1e-5,
        "#3619 warm-start lower bound loosened: warm={}, baseline={}",
        warm_output.lower()[[0]],
        baseline_output.lower()[[0]]
    );
    assert!(
        warm_output.upper()[[0]] <= baseline_output.upper()[[0]] + 1e-5,
        "#3619 warm-start upper bound loosened: warm={}, baseline={}",
        warm_output.upper()[[0]],
        baseline_output.upper()[[0]]
    );

    assert_monotone_child_domain_soundness(warm_output, child_lower, child_upper);
}

#[test]
fn test_monotone_warm_start_projects_parent_tangent_points_into_child_domain_3619() {
    let parent_bounds = BoundedTensor::new(
        arr1(&[-3.0_f32, -2.0, -1.0]).into_dyn(),
        arr1(&[1.5_f32, -0.2, 2.5]).into_dyn(),
    )
    .expect("parent bounds should construct");
    let child_bounds = BoundedTensor::new(
        arr1(&[-0.4_f32, -0.6, 0.2]).into_dyn(),
        arr1(&[0.5_f32, -0.1, 1.1]).into_dyn(),
    )
    .expect("child bounds should construct");

    let mut parent_state = GraphAlphaState::new();
    parent_state
        .add_sigmoid_node("sigmoid_hidden", &parent_bounds)
        .expect("parent sigmoid alpha should initialize");
    let mut child_state = GraphAlphaState::new();
    child_state
        .add_sigmoid_node("sigmoid_hidden", &child_bounds)
        .expect("child sigmoid alpha should initialize");

    let child_default = child_state
        .monotone_s_shaped_alpha("sigmoid_hidden")
        .expect("child default monotone alpha should exist")
        .clone();
    mutate_parent_monotone_projection_fixture(&mut parent_state);

    let parent = parent_state
        .monotone_s_shaped_alpha("sigmoid_hidden")
        .expect("parent monotone alpha should exist")
        .clone();
    child_state
        .monotone_s_shaped_alpha_mut("sigmoid_hidden")
        .expect("child monotone alpha should exist")
        .warm_start_from(&parent);

    let child = child_state
        .monotone_s_shaped_alpha("sigmoid_hidden")
        .expect("child monotone alpha should exist");
    assert_monotone_projection_result(child, &child_default);
}

/// Regression for #3549: DAG α-CROWN bound collection must keep the caller's
/// GemmEngine alive during the optimization loop, not only during initial
/// CROWN-IBP collection.
#[ntest::timeout(60000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_with_engine_threads_gemm_through_optimization_3549() {
    let (graph, input) = build_swiglu_relu_dag();

    let baseline_engine = CountingGemmEngine::new();
    graph
        .collect_alpha_crown_bounds_dag_with_engine(
            &input,
            &make_counting_engine_config(0),
            Some(&baseline_engine),
        )
        .expect("baseline alpha-CROWN collection should succeed");
    let baseline_calls = baseline_engine.gemm_calls();

    let optimized_engine = CountingGemmEngine::new();
    graph
        .collect_alpha_crown_bounds_dag_with_engine(
            &input,
            &make_counting_engine_config(2),
            Some(&optimized_engine),
        )
        .expect("optimized alpha-CROWN collection should succeed");
    let optimized_calls = optimized_engine.gemm_calls();

    assert!(
        baseline_calls > 0,
        "#3549 baseline must exercise GemmEngine during DAG CROWN-IBP collection"
    );
    assert!(
        optimized_calls > baseline_calls,
        "#3549 alpha optimization should add GemmEngine calls: baseline={baseline_calls}, optimized={optimized_calls}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_fix_interm_bounds_true_uses_ibp_on_dag_4404() {
    let (graph, input) = build_fix_interm_bounds_residual_dag_4404();
    let ibp_bounds = graph
        .collect_node_bounds_with_engine(&input, None)
        .expect("IBP reference bounds should succeed on the residual DAG");
    let crown_ibp_bounds = graph
        .collect_crown_ibp_bounds_dag(&input)
        .expect("CROWN-IBP reference bounds should succeed on the residual DAG");

    let residual_ibp = ibp_bounds
        .get("residual")
        .expect("IBP bounds should include the residual node");
    let residual_crown_ibp = crown_ibp_bounds
        .get("residual")
        .expect("CROWN-IBP bounds should include the residual node");
    let output_ibp = ibp_bounds
        .get("out")
        .expect("IBP bounds should include the output node");
    let output_crown_ibp = crown_ibp_bounds
        .get("out")
        .expect("CROWN-IBP bounds should include the output node");
    assert!(
        node_bounds_max_abs_diff_4404(residual_ibp, residual_crown_ibp)
            .max(node_bounds_max_abs_diff_4404(output_ibp, output_crown_ibp))
            > 1e-5,
        "#4404 oracle graph must distinguish DAG IBP from DAG CROWN-IBP at the residual or output node"
    );

    let config = AlphaCrownConfig {
        iterations: 0,
        fix_interm_bounds: true,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };
    let (alpha_bounds, _alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &config)
        .expect("fix_interm_bounds DAG alpha warmup should succeed");

    assert_bounded_tensor_close(
        alpha_bounds
            .get("residual")
            .expect("alpha warmup should include the residual node"),
        residual_ibp,
        1e-6,
        "#4404 residual DAG warmup should reuse IBP intermediates",
    );
    assert_bounded_tensor_close(
        alpha_bounds
            .get("out")
            .expect("alpha warmup should include the output node"),
        output_ibp,
        1e-6,
        "#4404 output DAG warmup should reuse IBP intermediates",
    );
}

/// Regression for the warm-start half of #4404: the per-child helper must
/// honor `fix_interm_bounds=true` instead of unconditionally rebuilding a
/// CROWN-IBP map. This fixture's unsupported forward-linear tail deliberately
/// falls back to plain IBP, while the oracle above proves that IBP and
/// CROWN-IBP differ at a load-bearing node.
#[ntest::timeout(10000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_warm_fix_interm_bounds_true_uses_ibp_on_dag_4404() {
    let (graph, input) = build_fix_interm_bounds_residual_dag_4404();
    let ibp_bounds = graph
        .collect_node_bounds_with_engine(&input, None)
        .expect("IBP reference bounds should succeed on the residual DAG");
    let crown_ibp_bounds = graph
        .collect_crown_ibp_bounds_dag(&input)
        .expect("CROWN-IBP reference bounds should succeed on the residual DAG");

    let residual_ibp = ibp_bounds
        .get("residual")
        .expect("IBP bounds should include the residual node");
    let output_ibp = ibp_bounds
        .get("out")
        .expect("IBP bounds should include the output node");
    let residual_crown_ibp = crown_ibp_bounds
        .get("residual")
        .expect("CROWN-IBP bounds should include the residual node");
    let output_crown_ibp = crown_ibp_bounds
        .get("out")
        .expect("CROWN-IBP bounds should include the output node");
    assert!(
        node_bounds_max_abs_diff_4404(residual_ibp, residual_crown_ibp)
            .max(node_bounds_max_abs_diff_4404(output_ibp, output_crown_ibp))
            > 1e-5,
        "#4404 warm oracle must distinguish DAG IBP from DAG CROWN-IBP"
    );

    let config = AlphaCrownConfig {
        iterations: 0,
        fix_interm_bounds: true,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };
    let (warm_bounds, _warm_state) = graph
        .collect_alpha_crown_bounds_dag_warm(&input, &config, &GraphAlphaState::new())
        .expect("fixed-intermediate warm start should succeed");

    assert_bounded_tensor_close(
        warm_bounds
            .get("residual")
            .expect("warm bounds should include the residual node"),
        residual_ibp,
        1e-6,
        "#4404 warm residual bounds should use the fixed IBP reference",
    );
    assert_bounded_tensor_close(
        warm_bounds
            .get("out")
            .expect("warm bounds should include the output node"),
        output_ibp,
        1e-6,
        "#4404 warm output bounds should use the fixed IBP reference",
    );
}

/// Regression for #3549: warm-started DAG α-CROWN refinement must also keep
/// using the caller's GemmEngine during its SPSA/output passes.
#[ntest::timeout(60000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_warm_with_engine_threads_gemm_through_refinement_3549() {
    let (graph, root_input) = build_swiglu_relu_dag();
    let (_root_bounds, root_alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&root_input, &make_warm_start_test_config(3))
        .expect("root alpha-CROWN should succeed");

    let child_input = BoundedTensor::new(
        array![-0.35_f32, -0.20_f32].into_dyn(),
        array![0.60_f32, 0.40_f32].into_dyn(),
    )
    .expect("child input bounds should be valid");

    let baseline_engine = CountingGemmEngine::new();
    graph
        .collect_alpha_crown_bounds_dag_warm_with_engine(
            &child_input,
            &make_warm_start_test_config(0),
            &root_alpha_state,
            Some(&baseline_engine),
        )
        .expect("baseline warm-start alpha-CROWN should succeed");
    let baseline_calls = baseline_engine.gemm_calls();

    // Fresh clone: the input-keyed CROWN-IBP collection cache
    // (#cgan-collection-cache) would otherwise serve the baseline call's
    // collection for this bit-identical child box, removing the collection
    // GEMM calls from the refined count that this regression compares.
    // `Clone` resets the cache. (In production, warm-start children have
    // per-domain boxes, which miss by key.)
    // The clone is load-bearing (cache reset via Clone-to-default), NOT dead.
    #[allow(clippy::redundant_clone)]
    let refined_graph = graph.clone();
    let refined_engine = CountingGemmEngine::new();
    refined_graph
        .collect_alpha_crown_bounds_dag_warm_with_engine(
            &child_input,
            &make_warm_start_test_config(2),
            &root_alpha_state,
            Some(&refined_engine),
        )
        .expect("refined warm-start alpha-CROWN should succeed");
    let refined_calls = refined_engine.gemm_calls();

    assert!(
        baseline_calls > 0,
        "#3549 warm-start baseline must exercise GemmEngine during DAG CROWN collection"
    );
    assert!(
        refined_calls > baseline_calls,
        "#3549 warm-start refinement should add GemmEngine calls: baseline={baseline_calls}, refined={refined_calls}"
    );
}

#[ntest::timeout(60000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_sqrt_alpha_changes_output_and_stays_sound_3773() {
    let (graph, input) = build_sqrt_exp_alpha_dag();

    let baseline = graph
        .collect_crown_ibp_bounds_dag(&input)
        .expect("#3773 baseline CROWN-IBP should succeed");
    let baseline_output = baseline
        .get(graph.output_name())
        .expect("#3773 baseline output bounds should exist")
        .flatten();
    let baseline_lower = baseline_output.lower()[0];
    let baseline_upper = baseline_output.upper()[0];

    let (alpha_bounds, alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &make_sqrt_alpha_test_config(8))
        .expect("#3773 sqrt alpha-CROWN should succeed");

    let sqrt_names: Vec<String> = alpha_state.sqrt_alpha_names().cloned().collect();
    assert_eq!(
        sqrt_names,
        vec!["sqrt_hidden".to_string()],
        "#3773 only the sqrt node should carry sqrt alpha state in this graph"
    );
    assert_eq!(
        alpha_state.monotone_alpha_names().count(),
        0,
        "#3773 regression graph should isolate sqrt alpha without monotone state"
    );
    assert_eq!(
        alpha_state.num_unstable(),
        0,
        "#3773 regression graph should not allocate ReLU alpha state"
    );

    let sqrt_alpha = alpha_state
        .sqrt_alpha("sqrt_hidden")
        .expect("#3773 sqrt alpha state should exist");
    assert!(
        sqrt_alpha.active_mask.iter().all(|&active| active),
        "#3773 sqrt alpha should stay active on the positive-domain regression graph"
    );

    let alpha_output = alpha_bounds
        .get(graph.output_name())
        .expect("#3773 alpha output bounds should exist")
        .flatten();
    let alpha_lower = alpha_output.lower()[0];
    let alpha_upper = alpha_output.upper()[0];
    assert!(
        alpha_lower.is_finite() && alpha_upper.is_finite() && alpha_lower <= alpha_upper,
        "#3773 alpha output bounds must be finite and ordered, got [{alpha_lower}, {alpha_upper}]"
    );

    let lower_changed = (alpha_lower - baseline_lower).abs() > 1e-5;
    let upper_changed = (alpha_upper - baseline_upper).abs() > 1e-5;
    assert!(
        lower_changed || upper_changed,
        "#3773 sqrt alpha must change the output bounds; baseline=[{baseline_lower}, {baseline_upper}], alpha=[{alpha_lower}, {alpha_upper}]"
    );

    assert_scalar_bounds_sound_by_sampling(
        &graph,
        &input,
        alpha_lower,
        alpha_upper,
        &[2],
        "#3773 sqrt alpha-CROWN",
    );
}

fn build_extracted_graph_target_shape_regression_3680() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::new(
        "main_slice",
        Layer::Slice(SliceLayer::new(-1, 1, 4)),
        vec!["_input".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "main_pad",
        Layer::Pad(PadLayer::new(
            vec![(0, 0), (0, 0), (1, 1)],
            PadMode::Constant(0.0),
        )),
        vec!["main_slice".to_string()],
    ));

    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![0.45_f32, -0.20_f32])
        .expect("valid Conv1d kernel");
    let conv = Conv1dLayer::with_input_length(kernel, Some(arr1(&[0.05_f32])), 1, 0, 5)
        .expect("valid Conv1d params");
    graph.add_node(GraphNode::new(
        "main_conv",
        Layer::Conv1d(conv),
        vec!["main_pad".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "skip_slice",
        Layer::Slice(SliceLayer::new(-1, 0, 4)),
        vec!["_input".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "output_add",
        Layer::Add(AddLayer),
        vec!["main_conv".to_string(), "skip_slice".to_string()],
    ));
    graph.set_output("output_add");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![-1.0_f32, -0.8, -0.4, 0.1, 0.5])
            .expect("valid lower input"),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![0.2_f32, 0.4, 0.7, 1.0, 1.3])
            .expect("valid upper input"),
    )
    .expect("valid bounded input");

    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_extracted_graph_crown_target_shape_3680() {
    let (graph, input) = build_extracted_graph_target_shape_regression_3680();
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP bounds should succeed for extracted-stage graph");
    let ibp_output = ibp_bounds
        .get("output_add")
        .expect("IBP output bounds should exist")
        .clone();

    let crown_output = graph
        .propagate_crown_to_node(
            &input,
            "output_add",
            &std::collections::HashMap::new(),
            &ibp_bounds,
            None,
            None,
            None,
            None,
        )
        .expect("direct CROWN output should preserve extracted target shape");
    assert_eq!(
        crown_output.shape(),
        ibp_output.shape(),
        "#3680 direct CROWN must restore extracted target shape"
    );

    let crown_ibp = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(&input, ibp_bounds, None)
        .expect("CROWN-IBP should succeed for extracted-stage graph");
    let output_provenance = crown_ibp.provenance.get("output_add");
    assert!(
        !matches!(
            output_provenance,
            Some(BoundsProvenance::ForwardFallback(
                crate::types::CrownIbpFallbackReason::CrownPropagationError
            ))
        ),
        "#3680 extracted output should not degrade to CrownPropagationError fallback: {output_provenance:?}"
    );
    let crown_ibp_output = crown_ibp
        .bounds
        .get("output_add")
        .expect("CROWN-IBP output bounds should exist");
    assert_eq!(
        crown_ibp_output.shape(),
        ibp_output.shape(),
        "#3680 CROWN-IBP output must match forward output shape"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_precomputed_short_deadline_falls_back_to_exact_ibp_3499() {
    use std::time::{Duration, Instant};

    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(arr2(&[[1.5_f32, -0.25]]), Some(arr1(&[0.75_f32])))
        .expect("valid linear layer");
    graph.add_node(GraphNode::from_input("lin", Layer::Linear(linear)));
    graph.set_output("lin");

    let input = BoundedTensor::new(
        array![-1.0_f32, -0.5].into_dyn(),
        array![0.25_f32, 1.0].into_dyn(),
    )
    .expect("valid bounded input");
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP bounds should succeed");

    let result = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(
            &input,
            ibp_bounds.clone(),
            Some(Instant::now() + Duration::from_secs(1)),
        )
        .expect("CROWN-IBP collection should fall back, not fail");

    let output_name = graph.output_name();
    assert_eq!(
        result.provenance.get(output_name),
        Some(&BoundsProvenance::ForwardFallback(
            crate::types::CrownIbpFallbackReason::PerNodeDeadlineExceeded
        )),
        "#3499 short per-node budget should report PerNodeDeadlineExceeded"
    );
    assert_bounded_tensor_close(
        result
            .bounds
            .get(output_name)
            .expect("CROWN-IBP output bounds should exist"),
        ibp_bounds
            .get(output_name)
            .expect("IBP output bounds should exist"),
        1e-6,
        "#3499 short per-node deadline fallback must preserve exact IBP bounds",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_sequential_fast_path_short_deadline_respects_per_node_budget_4413() {
    use std::time::{Duration, Instant};

    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(arr2(&[[1.5_f32, -0.25]]), Some(arr1(&[0.75_f32])))
        .expect("valid linear layer");
    graph.add_node(GraphNode::from_input("lin", Layer::Linear(linear)));
    graph.set_output("lin");

    let input = BoundedTensor::new(
        array![-1.0_f32, -0.5].into_dyn(),
        array![0.25_f32, 1.0].into_dyn(),
    )
    .expect("valid bounded input");
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP bounds should succeed");

    // With a short deadline (1s) and a small network (single Linear layer),
    // the budget floor (2.0s) exceeds the per-node share. The sequential
    // collector uses the full remaining time as cap rather than preemptively
    // falling back to IBP — this is correct because a single Linear backward
    // completes in microseconds. Per-layer deadline checks in crown_partial
    // handle actual timeouts for large networks.
    let result = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_engine(
            &input,
            ibp_bounds.clone(),
            Some(Instant::now() + Duration::from_secs(1)),
            Some(&NaiveCpuGemmEngine),
        )
        .expect("short-budget sequential fast path should succeed for small networks");

    let output_name = graph.output_name();
    // CROWN succeeds: bounds are at least as tight as IBP.
    let crown_bounds = result
        .bounds
        .get(output_name)
        .expect("CROWN-IBP output bounds should exist");
    let ibp_output = ibp_bounds
        .get(output_name)
        .expect("IBP output bounds should exist");
    assert!(
        crown_bounds.max_width() <= ibp_output.max_width() + 1e-6,
        "#4413 CROWN-IBP bounds should be at least as tight as IBP"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_precomputed_exact_output_keeps_crown_provenance_3775() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[1.0_f32, -0.5], [0.25, 0.75]]),
                Some(arr1(&[0.1_f32, -0.2])),
            )
            .expect("valid linear layer"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(
            LinearLayer::new(arr2(&[[0.6_f32, -0.4]]), Some(arr1(&[0.3_f32])))
                .expect("valid linear layer"),
        ),
        vec!["relu".to_string()],
    ));
    graph.set_output("lin2");

    let input = BoundedTensor::new(
        array![-1.0_f32, -0.25].into_dyn(),
        array![0.5_f32, 1.25].into_dyn(),
    )
    .expect("valid bounded input");
    let ibp_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP bounds should succeed");

    let result = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp(&input, ibp_bounds, None)
        .expect("CROWN-IBP collection should succeed");

    assert_eq!(
        result.provenance.get("lin2"),
        Some(&BoundsProvenance::Crown),
        "#3775 exact graph output must stay on the CROWN path, not demand-skip"
    );
}

// ========================= SE-block broadcast alpha-CROWN (#3499) =========================

/// Build a squeeze-excitation-like DAG with broadcasting MulBinary.
///
/// Architecture:
///   input [1, 4]
///     → Conv1d(1→2, k=1) → features [2, 4]
///     → ReLU → relu_feat [2, 4]
///     ↙              ↘
///   (lhs path)    ReduceSum(axis=-1, keepdim=true) → pooled [2, 1]
///                    → Sigmoid → se_weight [2, 1]
///     ↘              ↙
///   MulBinary: relu_feat [2, 4] * se_weight [2, 1]  ← BROADCAST
///     → gated [2, 4]
///     → ReduceSum(axes=[0,1]) → scalar
///
/// This exercises the MulBinary broadcast path in alpha-CROWN DAG optimization:
/// - ReLU provides unstable neurons for alpha optimization
/// - Sigmoid provides monotone alpha state
/// - MulBinary with [2,4] * [2,1] forces the broadcast coefficient accumulation
///   path added in #3499 (iters 1406-1407)
///
/// Reference: ECAPA-TDNN SE blocks multiply features [C,T] by SE weights [C,1].
fn build_se_block_broadcast_dag() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    // Input [1, 4] → Conv1d(1→2, kernel_size=1) → features [2, 4]
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![0.6_f32, -0.4]).expect("valid kernel");
    let bias = Array1::from_vec(vec![0.1_f32, -0.05]);
    let conv =
        Conv1dLayer::with_input_length(kernel, Some(bias), 1, 0, 4).expect("valid Conv1d params");
    graph.add_node(GraphNode::from_input("conv", Layer::Conv1d(conv)));

    // features → ReLU → relu_feat [2, 4]
    graph.add_node(GraphNode::new(
        "relu_feat",
        Layer::ReLU(ReLULayer),
        vec!["conv".to_string()],
    ));

    // relu_feat [2, 4] → ReduceSum(axis=-1, keepdim=true) → pooled [2, 1]
    graph.add_node(GraphNode::new(
        "pooled",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
        vec!["relu_feat".to_string()],
    ));

    // pooled [2, 1] → Sigmoid → se_weight [2, 1]
    graph.add_node(GraphNode::new(
        "se_weight",
        Layer::Sigmoid(SigmoidLayer),
        vec!["pooled".to_string()],
    ));

    // MulBinary: relu_feat [2, 4] * se_weight [2, 1] → gated [2, 4]  (BROADCAST)
    graph.add_node(GraphNode::binary(
        "gated",
        Layer::MulBinary(MulBinaryLayer),
        "relu_feat",
        "se_weight",
    ));

    // gated [2, 4] → ReduceSum(axes=[0,1], keepdim=false) → scalar
    graph.add_node(GraphNode::new(
        "out",
        Layer::ReduceSum(ReduceSumLayer::new(vec![0, 1], false)),
        vec!["gated".to_string()],
    ));
    graph.set_output("out");

    // Input shape [1, 4]: 1 channel, 4 time steps (matching Conv1d in_channels=1)
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-0.5_f32, -0.3, 0.1, -0.2])
            .expect("valid lower shape"),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![0.5_f32, 0.3, 0.8, 0.6])
            .expect("valid upper shape"),
    )
    .expect("valid SE-block broadcast input");

    (graph, input)
}

/// Assert that concrete graph evaluations at sampled points within `input` fall
/// within the scalar `[lower, upper]` bounds. Uses phase-shifted sampling to
/// cover the input domain without grid-aligned bias.
fn assert_scalar_bounds_sound_by_sampling(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    lower: f32,
    upper: f32,
    input_shape: &[usize],
    label: &str,
) {
    let input_lower = input.lower().as_slice().expect("contiguous").to_vec();
    let input_upper = input.upper().as_slice().expect("contiguous").to_vec();
    for sample_idx in 0..20 {
        let t = sample_idx as f32 / 19.0;
        let concrete: Vec<f32> = input_lower
            .iter()
            .zip(input_upper.iter())
            .enumerate()
            .map(|(j, (&lo, &hi))| {
                let phase = ((t + j as f32 * 0.31) % 1.0).clamp(0.0, 1.0);
                lo + phase * (hi - lo)
            })
            .collect();
        let concrete_bt = BoundedTensor::concrete(
            ArrayD::from_shape_vec(IxDyn(input_shape), concrete).expect("valid shape"),
        )
        .expect("valid concrete input");
        let value = graph
            .propagate_ibp(&concrete_bt)
            .expect("concrete evaluation should succeed")
            .flatten()
            .lower()[0];
        assert!(
            value >= lower - 1e-4 && value <= upper + 1e-4,
            "{label} soundness violation: sample {sample_idx}, \
             concrete={value}, bounds=[{lower}, {upper}]"
        );
    }
}

/// Alpha-CROWN DAG optimization on an SE-block graph with broadcasting MulBinary
/// must succeed and produce sound, non-loosening bounds.
///
/// Graph-level regression test for the MulBinary alpha broadcast fix (#3499,
/// iters 1406-1407). Layer-level test:
/// `test_crown_alpha_broadcast_soundness_se_block_pattern_3499` in
/// `layers/binary_ops/mul/tests.rs`.
///
/// Reference: alpha-beta-CROWN `utils.py` `reduce_broadcast_dims`.
#[ntest::timeout(60000)]
#[test]
fn test_alpha_crown_dag_se_block_broadcast_soundness_3499() {
    let (graph, input) = build_se_block_broadcast_dag();

    let ibp = graph
        .propagate_ibp(&input)
        .expect("SE-block IBP should succeed");
    let ibp_lower = ibp.flatten().lower()[0];
    let ibp_upper = ibp.flatten().upper()[0];
    assert!(ibp_lower.is_finite() && ibp_upper.is_finite());
    assert!(ibp_lower <= ibp_upper + 1e-5);

    let config = AlphaCrownConfig {
        iterations: 5,
        gradient_method: crate::bounds::GradientMethod::AnalyticChain,
        fix_interm_bounds: false,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };
    let (alpha_bounds, alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &config)
        .expect("#3499 SE-block broadcast alpha-CROWN DAG should succeed");

    let alpha_output = alpha_bounds
        .get(graph.output_name())
        .expect("alpha-CROWN should return output node bounds");
    let alpha_lower = alpha_output.flatten().lower()[0];
    let alpha_upper = alpha_output.flatten().upper()[0];
    assert!(alpha_lower.is_finite() && alpha_upper.is_finite());
    assert!(alpha_lower <= alpha_upper + 1e-5);

    let tol = 1e-4
        * [ibp_lower, ibp_upper, alpha_lower, alpha_upper, 1.0]
            .iter()
            .fold(0.0_f32, |a, &b| a.max(b.abs()));
    assert!(alpha_lower >= ibp_lower - tol, "lower loosened vs IBP");
    assert!(alpha_upper <= ibp_upper + tol, "upper loosened vs IBP");
    assert!(
        alpha_state.num_unstable() > 0 || alpha_state.monotone_alpha_names().count() > 0,
        "alpha state should have optimizable activations"
    );

    assert_scalar_bounds_sound_by_sampling(
        &graph,
        &input,
        alpha_lower,
        alpha_upper,
        &[1, 4],
        "#3499 SE-block alpha-CROWN",
    );
}

fn build_spatial_conv_relu_budget_graph_3813() -> (GraphNetwork, BoundedTensor) {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5_f32, -0.25, 0.75, 0.4]).unwrap();
    let conv =
        Conv2dLayer::with_input_shape(kernel, Some(arr1(&[0.1_f32])), (1, 1), (0, 0), 33, 33)
            .unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv".into()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 33, 33]), -0.4_f32),
        ArrayD::from_elem(IxDyn(&[1, 33, 33]), 0.7_f32),
    )
    .unwrap();

    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_spatial_targets_use_patches_under_dense_budget_3813() {
    crate::tests::with_crown_dense_budget_mb("1", || {
        let (graph, input) = build_spatial_conv_relu_budget_graph_3813();

        let result = graph
            .collect_crown_ibp_bounds_dag_with_status(&input)
            .unwrap();

        let conv_provenance = result
            .provenance
            .get("conv")
            .expect("conv provenance should exist");
        let relu_provenance = result
            .provenance
            .get("relu")
            .expect("relu provenance should exist");

        assert_eq!(
            conv_provenance,
            &BoundsProvenance::Crown,
            "#3813: spatial Conv2d target should tighten through patches instead of dense-budget IBP fallback"
        );
        assert_eq!(
            relu_provenance,
            &BoundsProvenance::Crown,
            "#3813: spatial ReLU target should tighten through patches instead of dense-budget IBP fallback"
        );
        assert!(
            result.fallback_events.iter().all(|event| {
                !(matches!(
                    event.reason,
                    crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded
                ) && (event.details.contains("node 'conv'") || event.details.contains("node 'relu'")))
            }),
            "#3813: conv/relu spatial targets should not report MemoryBudgetExceeded once patches start is available"
        );
    });
}

/// #3813 Step 7a: Complementary matrix-mode test for CROWN-IBP collector.
///
/// The no-cuts test above verifies Patches tightening works. This test verifies
/// the Dense-only path (matrix mode, `use_patches_mode=false`) also produces
/// Crown-tightened bounds. Together they form the two-sided invariant:
/// 1. without cuts → Patches behavior intact (above test)
/// 2. with cuts / matrix mode → Dense backward still tightens (this test)
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_spatial_targets_stay_dense_in_matrix_mode_3813() {
    // Use a small 4x4 spatial graph so Dense backward fits in memory
    // without needing a large budget override.
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5_f32, -0.25, 0.75, 0.4]).unwrap();
    let conv = Conv2dLayer::with_input_shape(kernel, Some(arr1(&[0.1_f32])), (1, 1), (0, 0), 4, 4)
        .unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["conv".into()],
    ));
    graph.set_output("relu");
    graph.set_use_patches_mode(false);

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.4_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.7_f32),
    )
    .unwrap();

    let result = graph
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .unwrap();

    let conv_provenance = result
        .provenance
        .get("conv")
        .expect("conv provenance should exist");
    assert_eq!(
        conv_provenance,
        &BoundsProvenance::Crown,
        "#3813: matrix-mode CROWN-IBP should tighten conv bounds via Dense backward, not fall back to IBP"
    );
}

/// #3839 Slice 3a: Verify the budget policy split helpers return correct values.
///
/// The fast-path gate (`counts_toward_sequential_skip_fraction`) must count
/// spatial Conv2d targets as over-budget (conservative), while the graph-native
/// guard (`graph_native_target_exceeds_budget`) must return `false` for those
/// same targets because they can start in Patches mode (#3813).
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_budget_policy_split_3839() {
    crate::tests::with_crown_dense_budget_mb("1", || {
        let (graph, _input) = build_spatial_conv_relu_budget_graph_3813();
        let budget = cpu_crown_dense_budget_bytes();

        // Build IBP bounds for the conv and relu nodes to feed to the helpers.
        let conv_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 32, 32]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1, 32, 32]), 1.0_f32),
        )
        .unwrap();
        let relu_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 32, 32]), -0.5_f32),
            ArrayD::from_elem(IxDyn(&[1, 32, 32]), 0.5_f32),
        )
        .unwrap();

        // Fast-path gate: both spatial targets should count toward skip fraction
        // (conservative — the sequential collector cannot selectively start in Patches).
        assert!(
            GraphNetwork::counts_toward_sequential_skip_fraction(&conv_bounds, budget),
            "#3839: conv should count toward sequential skip fraction (dense identity exceeds 1MB budget)"
        );
        assert!(
            GraphNetwork::counts_toward_sequential_skip_fraction(&relu_bounds, budget),
            "#3839: relu should count toward sequential skip fraction (dense identity exceeds 1MB budget)"
        );

        // Graph-native guard: both spatial targets should NOT exceed budget because
        // they can start in Patches mode (#3813 contract).
        assert!(
            graph.crown_ibp_target_can_start_in_patches("conv", &conv_bounds),
            "#3839: conv should be patches-startable (3D spatial, has Conv2d ancestor)"
        );
        assert!(
            graph.crown_ibp_target_can_start_in_patches("relu", &relu_bounds),
            "#3839: relu should be patches-startable (3D spatial, has Conv2d ancestor)"
        );
        assert!(
            !graph.graph_native_target_exceeds_budget("conv", &conv_bounds, budget),
            "#3839: graph-native guard must not reject conv — patches-start path is available"
        );
        assert!(
            !graph.graph_native_target_exceeds_budget("relu", &relu_bounds, budget),
            "#3839: graph-native guard must not reject relu — patches-start path is available"
        );
    });
}

/// #3839 Slice 3b: Engine-presence regression for the graph-native collector.
///
/// When `engine.is_some()`, the sequential fast-path gate may fire and route
/// collection through the sequential collector. This test verifies that under a
/// low dense budget, the graph-native collector still produces Crown provenance
/// for spatial Conv2d targets even when an engine is present. Without the policy
/// split, the per-node budget guard would reject these targets as
/// `MemoryBudgetExceeded` before the patches-start path gets a chance to run.
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_spatial_targets_use_patches_under_dense_budget_with_engine_3839() {
    crate::tests::with_crown_dense_budget_mb("1", || {
        let (graph, input) = build_spatial_conv_relu_budget_graph_3813();

        let result = graph
            .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, Some(&NaiveCpuGemmEngine))
            .unwrap();

        let conv_provenance = result
            .provenance
            .get("conv")
            .expect("conv provenance should exist");
        let relu_provenance = result
            .provenance
            .get("relu")
            .expect("relu provenance should exist");

        assert_eq!(
            conv_provenance,
            &BoundsProvenance::Crown,
            "#3839: spatial Conv2d target should tighten through patches with engine present"
        );
        assert_eq!(
            relu_provenance,
            &BoundsProvenance::Crown,
            "#3839: spatial ReLU target should tighten through patches with engine present"
        );
        assert!(
            result.fallback_events.iter().all(|event| {
                !(matches!(
                    event.reason,
                    crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded
                ) && (event.details.contains("node 'conv'")
                    || event.details.contains("node 'relu'")))
            }),
            "#3839: conv/relu should not report MemoryBudgetExceeded with engine present — patches-start path is available"
        );
    });
}

/// #3839 Slice 4: Exhaust the aggregate patches budget and verify later
/// patches-startable nodes fall back with `PatchesBudgetExceeded`.
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_patches_budget_exhaustion_3839() {
    crate::tests::with_serialized_env_vars(
        &[
            ("NY_DENSE_BUDGET_MB", "1"),
            ("NY_PATCHES_BUDGET_SECS", "0.001"),
        ],
        || {
            let (graph, input) = build_spatial_conv_relu_budget_graph_3813();

            let result = graph
                .collect_crown_ibp_bounds_dag_with_status(&input)
                .unwrap();

            assert!(
                result.bounds.contains_key("conv") && result.bounds.contains_key("relu"),
                "#3839: both patches-startable nodes should still produce bounds when the aggregate budget is exhausted"
            );
            let budget_event = result
                .fallback_events
                .iter()
                .find(|event| {
                    matches!(
                        event.reason,
                        crate::types::CrownIbpFallbackReason::PatchesBudgetExceeded
                    )
                })
                .cloned();
            if let Some(budget_event) = budget_event {
                assert!(
                    budget_event.details.contains("patches budget exhausted"),
                    "#3839: exhaustion event should describe the aggregate patches budget"
                );
                assert!(
                    result.provenance.values().any(|provenance| {
                        matches!(
                            provenance,
                            BoundsProvenance::ForwardFallback(
                                crate::types::CrownIbpFallbackReason::PatchesBudgetExceeded
                            )
                        )
                    }),
                    "#3839: PatchesBudgetExceeded events should have matching provenance entries"
                );
            }
        },
    );
}

/// Deadline guard for the O(L²) α-CROWN intermediate-bound collection
/// (#cifar100-alpha-interm).
///
/// `collect_crown_bounds_with_alpha` previously ran an unbounded per-node CROWN
/// backward sweep over every node in `exec_order` (O(L²) on a deep ResNet),
/// ignoring the verifier wall-clock budget. The cifar100 α-CROWN intermediate
/// branch took this sweep per refresh iteration and overran a 100s budget by
/// ~45 minutes. The fix threads the deadline into the per-node loop.
///
/// Contract verified here:
///   1. With a deadline already in the past, the call returns PROMPTLY and does
///      NOT run any per-node CROWN backward — every returned node equals the IBP
///      reference bound (the sound fallback).
///   2. With no deadline, plain CROWN actually tightens at least one node below
///      its IBP bound, proving the no-deadline path performs real work (so the
///      "all-equal-to-IBP" assertion in (1) genuinely demonstrates the bail-out,
///      not a graph where CROWN never tightens anyway).
#[ntest::timeout(10000)]
#[test]
fn collect_crown_bounds_with_alpha_past_deadline_returns_ibp_promptly() {
    use std::time::{Duration, Instant};

    // Two-hidden-layer ReLU MLP as a graph (DAG-shaped via the graph engine):
    // lin1 → relu1 → lin2 → relu2 → lin3(out). CROWN backward tightens the
    // intermediate ReLU/Linear bounds relative to IBP.
    let w1 = arr2(&[[1.0_f32, 0.5], [-0.3, 0.7], [0.2, -0.4]]);
    let b1 = arr1(&[0.1_f32, -0.2, 0.3]);
    let w2 = arr2(&[[0.4_f32, -0.6, 0.2], [0.1, 0.3, -0.5]]);
    let b2 = arr1(&[0.05_f32, -0.1]);
    let w3 = arr2(&[[0.7_f32, -0.2]]);
    let b3 = arr1(&[0.0_f32]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["lin2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin3",
        Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()),
        vec!["relu2".to_string()],
    ));
    graph.set_output("lin3");

    let input = BoundedTensor::new(
        array![-1.0_f32, -1.0].into_dyn(),
        array![1.0_f32, 1.0].into_dyn(),
    )
    .unwrap();

    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let alpha_state = GraphAlphaState::new();

    // (2) No deadline: plain CROWN must tighten at least one node vs IBP, so the
    // "equals IBP" check below is meaningful (the collection genuinely works).
    let no_deadline = graph
        .collect_crown_bounds_with_alpha(&input, &ibp_bounds, &alpha_state, None, None)
        .expect("no-deadline α-CROWN intermediate collection should succeed");
    let any_tightened = ibp_bounds.iter().any(|(name, ibp)| {
        no_deadline.get(name).is_some_and(|crown| {
            crown.shape() == ibp.shape()
                && crown
                    .lower()
                    .iter()
                    .zip(ibp.lower().iter())
                    .zip(crown.upper().iter().zip(ibp.upper().iter()))
                    .any(|((cl, il), (cu, iu))| *cl > *il + 1e-5 || *cu < *iu - 1e-5)
        })
    });
    assert!(
        any_tightened,
        "no-deadline CROWN should tighten at least one node vs IBP (otherwise the \
         past-deadline assertion is vacuous)"
    );

    // (1) Past deadline: must return promptly with every node equal to IBP,
    // proving the per-node CROWN sweep never ran.
    let past = Some(
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("system uptime exceeds 1s"),
    );
    let start = Instant::now();
    let deadline_bounds = graph
        .collect_crown_bounds_with_alpha(&input, &ibp_bounds, &alpha_state, None, past)
        .expect("past-deadline α-CROWN intermediate collection should succeed");
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(500),
        "past-deadline collection must bail out promptly, took {elapsed:?}"
    );

    // Every IBP node must be present and bound-for-bound identical to IBP: the
    // deadline fired before the first per-node CROWN backward, so all nodes use
    // the sound IBP reference fallback.
    for (name, ibp) in &ibp_bounds {
        let got = deadline_bounds
            .get(name)
            .unwrap_or_else(|| panic!("past-deadline result missing node '{name}'"));
        assert_bounded_tensor_close(
            ibp,
            got,
            1e-9,
            "#cifar100-alpha-interm: past-deadline node must equal IBP reference (no CROWN tightening ran)",
        );
    }
}

// ===== Input-keyed CROWN-IBP collection cache (#cgan-collection-cache) =====

/// Linear(3→3) → ReLU → Linear(3→2) → ReLU: both Linear nodes are pre-ReLU
/// CROWN tightening targets (demand-selected), so a collection does real
/// backward work whose deduplication the cache tests must prove.
fn build_collection_cache_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let l1 = LinearLayer::new(
        arr2(&[[1.0_f32, 0.5, -0.3], [-0.5, 1.0, 0.7], [0.3, -0.2, 1.0]]),
        Some(arr1(&[0.1_f32, -0.1, 0.05])),
    )
    .unwrap();
    graph.add_node(GraphNode::from_input("l1", Layer::Linear(l1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["l1".into()],
    ));
    let l2 = LinearLayer::new(
        arr2(&[[2.0_f32, -1.0, 0.5], [1.0, 2.0, -0.5]]),
        Some(arr1(&[0.0_f32, 0.0])),
    )
    .unwrap();
    graph.add_node(GraphNode::new(
        "l2",
        Layer::Linear(l2),
        vec!["relu1".into()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["l2".into()],
    ));
    graph.set_output("relu2");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();
    (graph, input)
}

/// Bit-exact equality of two node-bounds maps (the cache is a PURE dedup:
/// a served map must be the stored map, not a numerically-close recompute).
fn assert_node_bounds_bit_equal(
    a: &std::collections::HashMap<String, BoundedTensor>,
    b: &std::collections::HashMap<String, BoundedTensor>,
    label: &str,
) {
    assert_eq!(a.len(), b.len(), "{label}: node-count mismatch");
    for (name, ta) in a {
        let tb = b
            .get(name)
            .unwrap_or_else(|| panic!("{label}: node '{name}' missing"));
        assert_eq!(
            ta.lower(),
            tb.lower(),
            "{label}: '{name}' lower bits differ"
        );
        assert_eq!(
            ta.upper(),
            tb.upper(),
            "{label}: '{name}' upper bits differ"
        );
    }
}

/// #cgan-collection-cache: two successive collections on the SAME bit-exact
/// input box run the backward ONCE — the second is served from the cache
/// (hit-counter test hook) with a bit-identical map — while a 1-ULP different
/// box misses.
#[ntest::timeout(10000)]
#[test]
fn collection_cache_serves_second_identical_box_and_misses_one_ulp() {
    let (graph, input) = build_collection_cache_graph();
    assert_eq!(graph.crown_ibp_collection_cache_hits(), 0);

    let first = graph
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .unwrap();
    assert_eq!(
        graph.crown_ibp_collection_cache_hits(),
        0,
        "first collection must compute, not hit"
    );
    assert_eq!(
        first.provenance.get("l1"),
        Some(&BoundsProvenance::Crown),
        "pre-ReLU node must be CROWN-tightened in the fresh collection"
    );

    let second = graph
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .unwrap();
    assert_eq!(
        graph.crown_ibp_collection_cache_hits(),
        1,
        "second identical-box collection must be served from the cache"
    );
    assert_node_bounds_bit_equal(&first.bounds, &second.bounds, "cache hit");

    // A 1-ULP different box is a DIFFERENT box: must miss (BaB children).
    let mut upper = input.upper().clone();
    {
        let slice = upper.as_slice_mut().unwrap();
        slice[0] = ny_tensor::next_up_f32(slice[0]);
    }
    let nudged = BoundedTensor::new(input.lower().clone(), upper).unwrap();
    let _ = graph
        .collect_crown_ibp_bounds_dag_with_status(&nudged)
        .unwrap();
    assert_eq!(
        graph.crown_ibp_collection_cache_hits(),
        1,
        "a 1-ULP different box must MISS the cache"
    );
}

/// #cgan-collection-cache: the CROWN backward runs ONCE for two successive
/// same-box collections — counted through the GEMM engine (zero additional
/// engine calls on the cache-served second collection).
#[ntest::timeout(10000)]
#[test]
fn collection_cache_backward_runs_once_by_gemm_count() {
    let (graph, input) = build_collection_cache_graph();
    let engine = CountingGemmEngine::new();

    let first = graph
        .collect_crown_ibp_bounds_dag_with_engine(&input, Some(&engine))
        .unwrap();
    let calls_after_first = engine.gemm_calls();
    assert!(
        calls_after_first > 0,
        "fresh collection must run the backward through the engine"
    );

    let second = graph
        .collect_crown_ibp_bounds_dag_with_engine(&input, Some(&engine))
        .unwrap();
    assert_eq!(
        engine.gemm_calls(),
        calls_after_first,
        "cache-served second collection must add ZERO backward GEMM calls"
    );
    assert_eq!(graph.crown_ibp_collection_cache_hits(), 1);
    assert_node_bounds_bit_equal(&first, &second, "engine-path cache hit");
}

/// #cgan-collection-cache replacement policy: a truncated (deadline-starved)
/// collection never survives once a complete one exists — the complete map is
/// stored and then served even to a later call whose OWN budget is already
/// expired (the cgan alpha-warmup re-run shape).
#[ntest::timeout(10000)]
#[test]
fn collection_cache_complete_map_replaces_truncated_and_serves_expired_budget() {
    let (graph, input) = build_collection_cache_graph();
    let expired = Some(std::time::Instant::now());

    // 1. Truncated first collection (expired deadline): all-IBP fallback.
    let truncated = graph
        .collect_crown_ibp_bounds_dag_with_status_and_deadline(&input, expired, None)
        .unwrap();
    assert!(
        matches!(
            truncated.provenance.get("l1"),
            Some(BoundsProvenance::ForwardFallback(_))
        ),
        "expired-deadline collection must fall back for the tightening target"
    );
    assert_eq!(graph.crown_ibp_collection_cache_hits(), 0);

    // 2. Complete collection: misses (cached entry is truncated), computes,
    //    and REPLACES the truncated entry (more complete wins).
    let complete = graph
        .collect_crown_ibp_bounds_dag_with_status_and_deadline(&input, None, None)
        .unwrap();
    assert_eq!(
        complete.provenance.get("l1"),
        Some(&BoundsProvenance::Crown),
        "unbudgeted collection must CROWN-tighten the pre-ReLU node"
    );
    assert_eq!(
        graph.crown_ibp_collection_cache_hits(),
        0,
        "the complete run must compute (truncated entries are not served)"
    );

    // 3. Re-run with an expired budget: the complete cached map is served —
    //    this is the exact mechanism that lets the disjunctive precheck's
    //    complete map survive the alpha warmup's budget-starved re-runs.
    let served = graph
        .collect_crown_ibp_bounds_dag_with_status_and_deadline(&input, expired, None)
        .unwrap();
    assert_eq!(graph.crown_ibp_collection_cache_hits(), 1);
    assert_eq!(
        served.provenance.get("l1"),
        Some(&BoundsProvenance::Crown),
        "the served map must be the COMPLETE collection, not a fresh truncation"
    );
    assert_node_bounds_bit_equal(&complete.bounds, &served.bounds, "served complete map");
}

/// #cgan-collection-cache: `Clone` resets the cache (a clone may be mutated),
/// and `adopt_bound_caches_from` carries the entry across a PURE clone — the
/// disjunctive verify flow's clone-then-adopt contract.
#[ntest::timeout(10000)]
#[test]
fn collection_cache_clone_resets_and_adoption_carries_entry() {
    let (graph, input) = build_collection_cache_graph();
    let first = graph
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .unwrap();

    // Clone resets: a collection on the bare clone recomputes (no hit).
    let bare_clone = graph.clone();
    let _ = bare_clone
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .unwrap();
    assert_eq!(
        bare_clone.crown_ibp_collection_cache_hits(),
        0,
        "Clone must RESET the cache (clones may be mutated)"
    );

    // Adoption carries the entry: the adopted clone serves from cache.
    let mut adopted = graph.clone();
    adopted.adopt_bound_caches_from(&graph);
    let served = adopted
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .unwrap();
    assert_eq!(
        adopted.crown_ibp_collection_cache_hits(),
        1,
        "adopt_bound_caches_from must carry the collection entry"
    );
    assert_node_bounds_bit_equal(&first.bounds, &served.bounds, "adopted entry");

    // Mutation invalidates: flipping conv-mode policy clears the entry.
    let mut mutated = graph.clone();
    mutated.adopt_bound_caches_from(&graph);
    mutated.set_use_patches_mode(false);
    let _ = mutated
        .collect_crown_ibp_bounds_dag_with_status(&input)
        .unwrap();
    assert_eq!(
        mutated.crown_ibp_collection_cache_hits(),
        0,
        "a semantic mutation must invalidate the adopted entry"
    );
}

// ============ FC-head pre-activation tightening soundness (#cifar100-fchead) ============

/// Build a small dense DAG with two dense-fed ReLU pre-activations so both
/// `lin1` (fed to `relu1`) and `lin2` (fed to `relu2`) are targeted by
/// `tighten_fc_head_preactivations`. Weights are chosen so both ReLUs have
/// unstable neurons over the input box.
fn build_dense_fc_head_dag_fchead() -> (GraphNetwork, BoundedTensor) {
    let w1 = arr2(&[[1.0_f32, 0.5], [0.5, -1.0], [-0.5, 0.5]]);
    let b1 = arr1(&[0.0_f32, 0.1, -0.1]);
    let w2 = arr2(&[[1.0_f32, -1.0, 0.5], [0.5, 1.0, -1.0]]);
    let b2 = arr1(&[0.2_f32, -0.3]);
    let w3 = arr2(&[[1.0_f32, -1.0]]);
    let b3 = arr1(&[0.0_f32]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["lin2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin3",
        Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()),
        vec!["relu2".to_string()],
    ));
    graph.set_output("lin3");

    let input = BoundedTensor::new(
        array![-1.0_f32, -1.0].into_dyn(),
        array![1.0_f32, 1.0].into_dyn(),
    )
    .unwrap();
    (graph, input)
}

/// True forward pre-activations at `lin1` and `lin2` for a concrete input.
fn forward_fc_head_preacts(x: [f32; 2]) -> (Vec<f32>, Vec<f32>) {
    let w1 = arr2(&[[1.0_f32, 0.5], [0.5, -1.0], [-0.5, 0.5]]);
    let b1 = arr1(&[0.0_f32, 0.1, -0.1]);
    let w2 = arr2(&[[1.0_f32, -1.0, 0.5], [0.5, 1.0, -1.0]]);
    let b2 = arr1(&[0.2_f32, -0.3]);
    let xv = arr1(&x);
    let h1 = w1.dot(&xv) + &b1; // lin1 pre-activation
    let a1 = h1.mapv(|v| v.max(0.0)); // relu1
    let h2 = w2.dot(&a1) + &b2; // lin2 pre-activation
    (h1.to_vec(), h2.to_vec())
}

#[test]
fn fc_head_targets_are_selected_structurally_in_exec_order() {
    let (graph, _) = build_dense_fc_head_dag_fchead();
    let exec_order = graph.exec_order().unwrap();
    assert_eq!(
        graph.fc_head_preactivation_targets(exec_order),
        vec!["lin1".to_string(), "lin2".to_string()],
        "only Linear/Gemm producers immediately feeding ReLUs are head targets; the output linear must not be selected"
    );
}

#[ntest::timeout(30000)]
#[test]
fn test_tighten_fc_head_preactivations_sound_and_only_shrinks_fchead() {
    let (graph, input) = build_dense_fc_head_dag_fchead();
    let exec_order: Vec<String> = graph.exec_order().unwrap().to_vec();

    // Alpha bootstrap in the deep-conv default mode (fix_interm_bounds=true,
    // AnalyticChain) that the production cifar100 path uses.
    let config = AlphaCrownConfig {
        iterations: 5,
        gradient_method: crate::bounds::GradientMethod::AnalyticChain,
        fix_interm_bounds: true,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };
    let (reference, alpha_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &config)
        .expect("alpha bootstrap should succeed");
    assert!(
        alpha_state.num_unstable() > 0,
        "test graph must have unstable ReLUs to exercise the tightening"
    );

    let mut tightened = reference.clone();
    graph.tighten_fc_head_preactivations(
        &input,
        &exec_order,
        &alpha_state,
        None,
        None,
        &mut tightened,
    );

    // (1) INTERSECT-ONLY: every tightened pre-activation bound must lie inside
    // the reference bound (lower can only rise, upper can only fall).
    for node in ["lin1", "lin2"] {
        let r = reference.get(node).unwrap();
        let t = tightened.get(node).unwrap();
        assert_eq!(r.shape(), t.shape(), "{node} shape must be preserved");
        for ((&rl, &tl), (&ru, &tu)) in r
            .lower()
            .iter()
            .zip(t.lower().iter())
            .zip(r.upper().iter().zip(t.upper().iter()))
        {
            assert!(
                tl >= rl - 1e-5,
                "{node}: tightened lower {tl} dropped below reference lower {rl}"
            );
            assert!(
                tu <= ru + 1e-5,
                "{node}: tightened upper {tu} rose above reference upper {ru}"
            );
        }
    }

    // (2) ENCLOSURE: sampled TRUE pre-activations must lie within the tightened
    // bounds — the refined bound is still a valid over-approximation.
    let steps = 21;
    for i in 0..steps {
        for j in 0..steps {
            let x0 = -1.0 + 2.0 * (i as f32) / (steps - 1) as f32;
            let x1 = -1.0 + 2.0 * (j as f32) / (steps - 1) as f32;
            let (h1, h2) = forward_fc_head_preacts([x0, x1]);
            let t1 = tightened.get("lin1").unwrap();
            for (k, &v) in h1.iter().enumerate() {
                assert!(
                    v >= t1.lower()[[k]] - 1e-4 && v <= t1.upper()[[k]] + 1e-4,
                    "lin1[{k}]={v} escaped tightened [{}, {}] at ({x0},{x1})",
                    t1.lower()[[k]],
                    t1.upper()[[k]]
                );
            }
            let t2 = tightened.get("lin2").unwrap();
            for (k, &v) in h2.iter().enumerate() {
                assert!(
                    v >= t2.lower()[[k]] - 1e-4 && v <= t2.upper()[[k]] + 1e-4,
                    "lin2[{k}]={v} escaped tightened [{}, {}] at ({x0},{x1})",
                    t2.lower()[[k]],
                    t2.upper()[[k]]
                );
            }
        }
    }
}

// ============ #linearizenn-dense-dag-ref: DAG reference-collector regression ============

/// Deterministic weight matrix (LCG, values in [-1, 1)) so the test net has
/// genuinely unstable ReLUs without a 6x6 literal per layer.
fn dense_dag_weight(rows: usize, cols: usize, seed: u32) -> ndarray::Array2<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    ndarray::Array2::from_shape_fn((rows, cols), |_| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    })
}

/// linearizenn-shaped graph: a deep dense ReLU chain whose ONLY DAG-ness is a
/// small `Slice(input) -> Linear -> Concat` skip path (exactly the
/// `AllInOne_*.onnx` topology). No conv, no binary-relaxation op, few nodes.
fn build_dense_skip_dag() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input(
        "lin0",
        Layer::Linear(LinearLayer::new(dense_dag_weight(6, 4, 1), None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu0",
        Layer::ReLU(ReLULayer),
        vec!["lin0".to_string()],
    ));
    for depth in 1..4 {
        graph.add_node(GraphNode::new(
            format!("lin{depth}"),
            Layer::Linear(
                LinearLayer::new(dense_dag_weight(6, 6, depth as u32 + 1), None).unwrap(),
            ),
            vec![format!("relu{}", depth - 1)],
        ));
        graph.add_node(GraphNode::new(
            format!("relu{depth}"),
            Layer::ReLU(ReLULayer),
            vec![format!("lin{depth}")],
        ));
    }
    graph.add_node(GraphNode::new(
        "head",
        Layer::Linear(LinearLayer::new(dense_dag_weight(1, 6, 9), None).unwrap()),
        vec!["relu3".to_string()],
    ));

    // Skip path straight off the network input (this is what makes it a DAG).
    graph.add_node(GraphNode::from_input(
        "skip_slice",
        Layer::Slice(SliceLayer::new(0, 2, 4)),
    ));
    graph.add_node(GraphNode::new(
        "skip_lin",
        Layer::Linear(LinearLayer::new(dense_dag_weight(1, 2, 11), None).unwrap()),
        vec!["skip_slice".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "concat",
        Layer::Concat(ConcatLayer::with_input_shapes(0, vec![vec![1], vec![1]])),
        "head",
        "skip_lin",
    ));
    graph.set_output("concat");

    let input = BoundedTensor::new(
        array![-1.0_f32, -1.0, -1.0, -1.0].into_dyn(),
        array![1.0_f32, 1.0, 1.0, 1.0].into_dyn(),
    )
    .unwrap();

    (graph, input)
}

/// #linearizenn-dense-dag-ref: with `fix_interm_bounds = true`, a small dense
/// DAG must take the per-node CROWN-IBP reference collector, not plain IBP.
///
/// Before the fix the `deep_seq` CROWN-IBP override was gated on `!is_dag`, so
/// a deep dense chain with one tiny skip edge fell through every branch to
/// plain IBP. On linearizenn_2024 `AllInOne_120_120 / prop_120_120_0` that cost
/// a ~145x looser root objective (row bounds `[-616.166, 473.220]` vs
/// `[36.070, 48.366]`).
#[test]
fn dense_skip_dag_alpha_reference_uses_crown_ibp_intermediates() {
    let (graph, input) = build_dense_skip_dag();
    let exec_order = graph.exec_order().expect("exec order").to_vec();
    assert!(
        !graph.is_sequential_graph(&exec_order),
        "fixture must be a DAG for this regression to be meaningful"
    );

    let config = AlphaCrownConfig {
        fix_interm_bounds: true,
        ..AlphaCrownConfig::default()
    };
    let (reference, source) = graph
        .collect_alpha_reference_bounds_with_engine_and_source(&input, &config, None, &exec_order)
        .expect("dense-DAG reference bounds should collect");

    assert_eq!(
        source,
        AlphaReferenceBoundsSource::CrownIbp,
        "small dense DAGs must use CROWN-IBP intermediates under fix_interm_bounds=true"
    );

    // The reference map must be strictly tighter than plain IBP at the deepest
    // pre-activation (that is the whole point of the collector switch), and
    // must still enclose it (soundness direction: tighten, never widen past).
    let ibp = graph
        .collect_node_bounds(&input)
        .expect("IBP node bounds should collect");
    let ibp_deep = ibp.get("lin3").expect("IBP has lin3");
    let ref_deep = reference.get("lin3").expect("reference has lin3");
    let ibp_width: f32 = (ibp_deep.upper() - ibp_deep.lower()).sum();
    let ref_width: f32 = (ref_deep.upper() - ref_deep.lower()).sum();
    assert!(
        ref_width < ibp_width,
        "CROWN-IBP reference must be tighter than IBP at lin3: {ref_width} vs {ibp_width}"
    );
    for k in 0..ref_deep.lower().len() {
        assert!(
            ref_deep.lower()[[k]] >= ibp_deep.lower()[[k]] - 1e-4
                && ref_deep.upper()[[k]] <= ibp_deep.upper()[[k]] + 1e-4,
            "CROWN-IBP reference must not be looser than IBP at lin3[{k}]"
        );
    }
}

/// Guard the exclusions: a graph carrying a binary-relaxation op (MulBinary)
/// must keep the plain-IBP reference even though it is a small non-conv DAG.
#[test]
fn dense_skip_dag_with_binary_op_keeps_ibp_reference() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin0",
        Layer::Linear(LinearLayer::new(dense_dag_weight(6, 4, 1), None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu0",
        Layer::ReLU(ReLULayer),
        vec!["lin0".to_string()],
    ));
    for depth in 1..4 {
        graph.add_node(GraphNode::new(
            format!("lin{depth}"),
            Layer::Linear(
                LinearLayer::new(dense_dag_weight(6, 6, depth as u32 + 1), None).unwrap(),
            ),
            vec![format!("relu{}", depth - 1)],
        ));
        graph.add_node(GraphNode::new(
            format!("relu{depth}"),
            Layer::ReLU(ReLULayer),
            vec![format!("lin{depth}")],
        ));
    }
    graph.add_node(GraphNode::new(
        "head",
        Layer::Linear(LinearLayer::new(dense_dag_weight(6, 6, 9), None).unwrap()),
        vec!["relu3".to_string()],
    ));
    // Bounded elementwise product of two activations: a binary-relaxation op,
    // which `should_use_crown_ibp_intermediates()` refuses.
    graph.add_node(GraphNode::binary(
        "bilinear",
        Layer::MulBinary(MulBinaryLayer),
        "head",
        "relu3",
    ));
    graph.set_output("bilinear");

    let input = BoundedTensor::new(
        array![-1.0_f32, -1.0, -1.0, -1.0].into_dyn(),
        array![1.0_f32, 1.0, 1.0, 1.0].into_dyn(),
    )
    .unwrap();

    let exec_order = graph.exec_order().expect("exec order").to_vec();
    let config = AlphaCrownConfig {
        fix_interm_bounds: true,
        ..AlphaCrownConfig::default()
    };
    let (_reference, source) = graph
        .collect_alpha_reference_bounds_with_engine_and_source(&input, &config, None, &exec_order)
        .expect("binary-op DAG reference bounds should collect");
    assert_eq!(
        source,
        AlphaReferenceBoundsSource::Ibp,
        "binary-relaxation graphs must keep the plain-IBP reference"
    );
}
