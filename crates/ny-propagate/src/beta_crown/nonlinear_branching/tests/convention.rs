// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #genbab-selector-convention: the GenBaB selector must score an activation on
//! its INPUT (pre-activation) interval — `node_bounds[node.inputs[0]]` — which
//! is the interval `with_general_split` subdivides. Production `node_bounds` is
//! keyed by node name and holds each node's OUTPUT bounds, so the pre-fix
//! lookup `node_bounds[node_name]` read the POST-activation interval.
//!
//! Each fixture below stores bounds exactly as production does: the input's
//! interval under the input node, and the activation's OUTPUT interval under
//! the activation. The first two tests FAIL on the pre-fix selector.

use std::collections::HashMap;

use ndarray::arr1;
use ny_tensor::BoundedTensor;

use super::super::{points_strictly_partition, NonlinearBranching, NonlinearBranchingConfig};
use crate::beta_crown::branching::LayerRef;
use crate::layers::{Layer, LinearLayer, ReLULayer, SigmoidLayer};
use crate::network::{GraphNetwork, GraphNode};

fn interval(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(arr1(lower).into_dyn(), arr1(upper).into_dyn()).expect("finite interval")
}

/// `lin` (identity, from the network input) -> `act`.
fn linear_then(act: Layer, width: usize) -> GraphNetwork {
    let mut weights = ndarray::Array2::<f32>::zeros((width, width));
    for i in 0..width {
        weights[[i, i]] = 1.0;
    }
    let linear = LinearLayer::new(weights, None).expect("identity linear");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("lin", Layer::Linear(linear)));
    graph.add_node(GraphNode::new("act", act, vec!["lin".to_string()]));
    graph.set_output("act");
    graph
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A ReLU whose PRE-activation interval straddles 0 is a GenBaB candidate,
/// split at 0, and its decision carries the pre-activation interval.
///
/// Pre-fix: the selector read the post-activation interval `[0, 3]`, whose
/// lower bound is >= 0, so `score_neurons` dropped the neuron as stable and
/// GenBaB never branched a ReLU.
#[ntest::timeout(5000)]
#[test]
fn genbab_selector_scores_relu_on_its_pre_activation_interval() {
    let graph = linear_then(Layer::ReLU(ReLULayer), 1);
    let mut node_bounds = HashMap::new();
    node_bounds.insert("lin".to_string(), interval(&[-1.0], &[3.0]));
    // Production stores the ReLU's POST-activation interval under its own name.
    node_bounds.insert("act".to_string(), interval(&[0.0], &[3.0]));

    let decisions = NonlinearBranching::default()
        .decisions(&graph, &node_bounds, &["act".to_string()])
        .expect("selector should not error");

    assert_eq!(
        decisions.len(),
        1,
        "a ReLU whose pre-activation straddles 0 must be a GenBaB candidate \
         (the pre-fix selector read the post-activation [0, 3] and dropped it)"
    );
    let decision = &decisions[0];
    assert_eq!(decision.layer, LayerRef::Name("act".to_string()));
    assert_eq!(decision.points, vec![0.0]);
    assert_eq!(
        decision.original_bounds,
        (-1.0, 3.0),
        "the decision must carry the INPUT interval it splits"
    );
    assert!(points_strictly_partition(-1.0, 3.0, &decision.points));
}

/// The ReLU-half ENGAGEMENT observable: once the selector has scored an unstable
/// ReLU, the process-cumulative `[genbab-select]` summary reports
/// `relu_candidates >= 1`. (`relu=` counts only TOP picks and can stay 0 with
/// the fix working.) The counters are process-global and monotone, so `>= 1`
/// holds regardless of other tests running in parallel.
#[ntest::timeout(5000)]
#[test]
fn genbab_select_summary_reports_relu_candidates() {
    let graph = linear_then(Layer::ReLU(ReLULayer), 1);
    let mut node_bounds = HashMap::new();
    node_bounds.insert("lin".to_string(), interval(&[-1.0], &[3.0]));
    node_bounds.insert("act".to_string(), interval(&[0.0], &[3.0]));

    let decisions = NonlinearBranching::default()
        .decisions(&graph, &node_bounds, &["act".to_string()])
        .expect("selector should not error");
    assert_eq!(decisions.len(), 1, "fixture premise: one unstable ReLU");

    let line = super::super::genbab_select_summary_line()
        .expect("a selection ran in this process, so the summary line must print");
    let relu_candidates: u64 = line
        .split_whitespace()
        .find_map(|field| field.strip_prefix("relu_candidates="))
        .expect("the summary line must carry relu_candidates=")
        .parse()
        .expect("relu_candidates is an integer");
    assert!(
        relu_candidates >= 1,
        "an unstable ReLU was scored, so relu_candidates must be >= 1: {line}"
    );
}

/// A Sigmoid's split point is computed in INPUT space and lies strictly inside
/// the input interval, so both children are non-empty and strictly narrower.
///
/// Pre-fix: the point was the midpoint of the OUTPUT interval
/// `[sigmoid(2), sigmoid(6)] ~ [0.881, 0.998]`, i.e. ~0.939, which lies OUTSIDE
/// the input interval `[2, 6]` that `with_general_split` clamps: one empty
/// child plus one unchanged child — a no-op split.
#[ntest::timeout(5000)]
#[test]
fn genbab_selector_sigmoid_split_point_lies_inside_input_interval() {
    let graph = linear_then(Layer::Sigmoid(SigmoidLayer::new()), 1);
    let mut node_bounds = HashMap::new();
    node_bounds.insert("lin".to_string(), interval(&[2.0], &[6.0]));
    node_bounds.insert(
        "act".to_string(),
        interval(&[sigmoid(2.0)], &[sigmoid(6.0)]),
    );

    let decisions = NonlinearBranching::default()
        .decisions(&graph, &node_bounds, &["act".to_string()])
        .expect("selector should not error");

    assert_eq!(decisions.len(), 1, "the sigmoid neuron must be a candidate");
    let decision = &decisions[0];
    assert_eq!(decision.original_bounds, (2.0, 6.0));
    assert_eq!(decision.points.len(), 1);
    let point = decision.points[0];
    assert!(
        point > 2.0 && point < 6.0,
        "split point {point} must lie strictly inside the INPUT interval [2, 6] \
         (the pre-fix output-space midpoint was ~0.939)"
    );
    assert!(points_strictly_partition(2.0, 6.0, &decision.points));
}

/// The progress predicate itself, including the f32 corner and unsorted points.
#[ntest::timeout(5000)]
#[test]
fn genbab_points_strictly_partition_predicate() {
    assert!(points_strictly_partition(2.0, 6.0, &[4.0]));
    assert!(points_strictly_partition(2.0, 6.0, &[3.0, 5.0]));
    assert!(points_strictly_partition(-1.0, 1.0, &[0.0]));
    // Output-space point outside the input interval: the pre-fix no-op.
    assert!(!points_strictly_partition(2.0, 6.0, &[0.939]));
    // On an endpoint: one child degenerate, the other the whole parent.
    assert!(!points_strictly_partition(2.0, 6.0, &[2.0]));
    assert!(!points_strictly_partition(2.0, 6.0, &[6.0]));
    // Unsorted / repeated / non-finite / empty / degenerate parent.
    assert!(!points_strictly_partition(2.0, 6.0, &[5.0, 3.0]));
    assert!(!points_strictly_partition(2.0, 6.0, &[4.0, 4.0]));
    assert!(!points_strictly_partition(2.0, 6.0, &[f32::NAN]));
    assert!(!points_strictly_partition(2.0, 6.0, &[]));
    assert!(!points_strictly_partition(2.0, 2.0, &[2.0]));
    assert!(!points_strictly_partition(f32::NEG_INFINITY, 6.0, &[4.0]));
}

/// A ULP-wide input interval has NO f32 strictly inside it, so the selector
/// must emit no decision for it rather than a point that rounds onto an
/// endpoint (a split that cannot make progress). The width clears the default
/// `min_branch_width` (1e-6), so only the partition guard can refuse it.
#[ntest::timeout(5000)]
#[test]
fn genbab_selector_refuses_a_split_that_cannot_partition() {
    let lower = 1.0e6_f32;
    let upper = f32::from_bits(lower.to_bits() + 1);
    assert!(upper - lower >= 1e-6, "fixture must clear min_branch_width");

    let graph = linear_then(Layer::Sigmoid(SigmoidLayer::new()), 1);
    let mut node_bounds = HashMap::new();
    node_bounds.insert("lin".to_string(), interval(&[lower], &[upper]));
    node_bounds.insert("act".to_string(), interval(&[0.99], &[1.0]));

    let decisions = NonlinearBranching::default()
        .decisions(&graph, &node_bounds, &["act".to_string()])
        .expect("selector should not error");
    assert!(
        decisions.is_empty(),
        "no f32 lies strictly inside [{lower}, {upper}], so no split can make \
         progress; got {decisions:?}"
    );
}

/// Every production consumer branches on `decisions[0]`
/// (`engine/branching/genbab.rs::select_genbab_branch`,
/// `multi_objective/domain_process.rs::process_graph_domain_genbab`), and the
/// sort is stable, so the TOP decision does not depend on `num_candidates`.
/// This is why delivering `nonlinear_split.num_candidates: 100` is
/// value-neutral until a consumer (the BBPS filter) reads `decisions[1..]`.
#[ntest::timeout(5000)]
#[test]
fn genbab_top_decision_is_independent_of_num_candidates() {
    let graph = linear_then(Layer::Sigmoid(SigmoidLayer::new()), 4);
    let mut node_bounds = HashMap::new();
    node_bounds.insert(
        "lin".to_string(),
        interval(&[-1.0, -3.0, 0.5, -0.25], &[1.0, 2.0, 4.0, 0.25]),
    );
    node_bounds.insert("act".to_string(), interval(&[0.0; 4], &[1.0; 4]));

    let top = |num_candidates: usize| {
        let decisions = NonlinearBranching::new(NonlinearBranchingConfig {
            num_candidates,
            ..Default::default()
        })
        .decisions(&graph, &node_bounds, &["act".to_string()])
        .expect("selector should not error");
        assert!(decisions.len() <= num_candidates);
        let first = decisions
            .first()
            .expect("four wide sigmoid neurons")
            .clone();
        (first.neuron_idx, first.points, first.original_bounds)
    };

    assert_eq!(top(1), top(100));
    assert_eq!(top(1), top(4));
}
