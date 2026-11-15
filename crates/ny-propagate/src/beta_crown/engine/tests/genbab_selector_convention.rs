// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #genbab-selector-convention: children-cover-parent SOUNDNESS tests through
//! the PRODUCTION GenBaB path — `find_splittable_graph_nodes` ->
//! `select_genbab_branch` -> `BranchingDecision::to_splits` ->
//! `GraphBabDomain::with_general_split` — on real root domains whose
//! `node_bounds` come from `collect_node_bounds` (each node's OUTPUT interval,
//! exactly as BaB stores them).
//!
//! The invariant asserted for every split:
//!   1. COVERAGE (soundness): the union of the children's intervals on the
//!      split neuron is EXACTLY the parent interval, and every sampled value of
//!      the parent interval (endpoints and split point included) lies in some
//!      child. A split that dropped part of the parent would let BaB publish
//!      `Verified` over a region it never bounded.
//!   2. PROGRESS: every child is non-empty (`with_general_split` returned
//!      `Some`) and strictly narrower than the parent. Before the fix an
//!      output-space point gave one EMPTY child and one child EQUAL to the
//!      parent, so BaB re-split the same neuron until the depth cap.
//!
//! The activations take the NETWORK INPUT directly, so each child's interval is
//! observable as its narrowed `input_bounds` — a direct reading of the region
//! the child covers, not a re-derivation of it. One test uses an intermediate
//! pre-activation node and reads the recorded GenBaB constraints instead.

use super::prelude::*;
use crate::beta_crown::domain::GraphCrownContext;
use crate::beta_crown::nonlinear_branching::{
    points_strictly_partition, NonlinearBranching, NonlinearBranchingConfig,
};
use crate::layers::{SigmoidLayer, SinLayer};

const SAMPLES: usize = 1000;

fn single_activation_on_input(name: &str, layer: Layer) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(name, layer));
    graph.set_output(name);
    graph
}

fn root_for(graph: &GraphNetwork, lower: f32, upper: f32) -> (BoundedTensor, GraphBabDomain) {
    let input = BoundedTensor::new(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn())
        .expect("finite input box");
    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("IBP node bounds must compute");
    let root =
        GraphBabDomain::root(node_bounds, -10.0, 10.0, &input, false).expect("valid root domain");
    (input, root)
}

fn scalar_interval(bounds: &BoundedTensor) -> (f32, f32) {
    let flat = bounds.flatten();
    assert_eq!(flat.len(), 1, "fixtures are one-neuron");
    (flat.lower()[[0]], flat.upper()[[0]])
}

/// Select through the production path and build every child. Panics (test
/// failure) if the node is not splittable, no decision is produced, or a
/// child is EMPTY — each of which is the pre-fix behaviour on these fixtures.
fn split_through_production_path(
    graph: &GraphNetwork,
    root: &GraphBabDomain,
    node: &str,
) -> (crate::beta_crown::BranchingDecision, Vec<GraphBabDomain>) {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let genbab = NonlinearBranching::new(NonlinearBranchingConfig::default());
    let nodes = vec![node.to_string()];

    let splittable = verifier.find_splittable_graph_nodes(graph, root, &nodes, &genbab);
    assert_eq!(
        splittable, nodes,
        "`find_splittable_graph_nodes` (which reads node.inputs[0]) must report {node}"
    );
    let decision = verifier
        .select_genbab_branch(graph, root, &splittable, &genbab)
        .expect("selection must not error")
        .unwrap_or_else(|| {
            panic!(
                "the selector must agree with find_splittable_graph_nodes and \
                 produce a decision for {node} (pre-fix it read the POST-activation \
                 interval)"
            )
        });

    let mut children = Vec::new();
    for split in decision.to_splits().expect("valid decision") {
        let child = root
            .with_general_split(graph, split, false)
            .expect("with_general_split must not error")
            .unwrap_or_else(|| {
                panic!(
                    "EMPTY child from decision {decision:?}: its sibling is the \
                     unchanged parent, so the split is a no-op"
                )
            });
        children.push(child);
    }
    (decision, children)
}

/// Coverage + progress over explicit child intervals of the split neuron.
fn assert_children_cover_parent(parent: (f32, f32), children: &[(f32, f32)], points: &[f32]) {
    let (pl, pu) = parent;
    assert!(pl < pu, "fixture parent must be a proper interval");
    assert!(
        children.len() >= 2,
        "a split must produce at least two children, got {children:?}"
    );
    for &(cl, cu) in children {
        assert!(cl <= cu, "child [{cl}, {cu}] is empty");
        assert!(
            cl >= pl && cu <= pu,
            "child [{cl}, {cu}] escapes parent [{pl}, {pu}]"
        );
        assert!(
            cu - cl < pu - pl,
            "child [{cl}, {cu}] is not strictly narrower than parent [{pl}, {pu}]: no progress"
        );
    }

    // Exact union: sorted children must start at pl, end at pu, and chain with
    // no gap (touching at a split point is enough — the intervals are closed).
    let mut sorted = children.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(
        sorted[0].0, pl,
        "union must start at the parent lower bound"
    );
    assert_eq!(
        sorted[sorted.len() - 1].1,
        pu,
        "union must end at the parent upper bound"
    );
    for pair in sorted.windows(2) {
        assert!(
            pair[1].0 <= pair[0].1,
            "gap between children [{}, {}] and [{}, {}]: part of the parent is \
             covered by NO child",
            pair[0].0,
            pair[0].1,
            pair[1].0,
            pair[1].1
        );
    }

    // Sampled containment: every sampled pre-activation value of the parent,
    // plus both endpoints and every split point, lies in at least one child.
    let mut samples: Vec<f32> = (0..=SAMPLES)
        .map(|k| (pl + (pu - pl) * (k as f32 / SAMPLES as f32)).clamp(pl, pu))
        .collect();
    samples.push(pl);
    samples.push(pu);
    samples.extend_from_slice(points);
    for x in samples {
        assert!(
            children.iter().any(|&(cl, cu)| cl <= x && x <= cu),
            "parent value {x} is covered by no child of {children:?}"
        );
    }
}

/// Sigmoid on the input box [2, 6]. Pre-fix the point was the midpoint of the
/// OUTPUT interval (~0.939), outside [2, 6]: `with_general_split` returned an
/// EMPTY lower child and an upper child equal to the parent.
#[ntest::timeout(10000)]
#[test]
fn genbab_split_children_cover_parent_sigmoid() {
    let graph = single_activation_on_input("sig", Layer::Sigmoid(SigmoidLayer::new()));
    let (input, root) = root_for(&graph, 2.0, 6.0);
    let parent = scalar_interval(&input);

    let (decision, children) = split_through_production_path(&graph, &root, "sig");
    assert_eq!(
        decision.original_bounds, parent,
        "the decision must carry the INPUT interval, not sigmoid's output interval"
    );
    assert!(points_strictly_partition(
        parent.0,
        parent.1,
        &decision.points
    ));

    let intervals: Vec<(f32, f32)> = children
        .iter()
        .map(|child| scalar_interval(child.input_bounds.as_ref()))
        .collect();
    assert_children_cover_parent(parent, &intervals, &decision.points);
}

/// ReLU on the input box [-1, 3]. Pre-fix the ReLU was never a candidate: its
/// post-activation interval [0, 3] never straddles 0. The GenBaB split at 0 on a
/// ReLU must be recorded as a standard ReLU constraint.
#[ntest::timeout(10000)]
#[test]
fn genbab_split_children_cover_parent_relu() {
    let graph = single_activation_on_input("relu", Layer::ReLU(ReLULayer));
    let (input, root) = root_for(&graph, -1.0, 3.0);
    let parent = scalar_interval(&input);

    let (decision, children) = split_through_production_path(&graph, &root, "relu");
    assert_eq!(decision.points, vec![0.0]);
    assert_eq!(decision.original_bounds, parent);

    for child in &children {
        assert_eq!(
            child.history.constraints.len(),
            1,
            "a 0-split on a ReLU is a ReLU constraint"
        );
        assert_eq!((child.history.genbab_constraints).len(), 0);
    }
    let intervals: Vec<(f32, f32)> = children
        .iter()
        .map(|child| scalar_interval(child.input_bounds.as_ref()))
        .collect();
    assert_children_cover_parent(parent, &intervals, &decision.points);
}

/// Sin on the SYMMETRIC input box [-1, 1]: the uniform midpoint is EXACTLY 0.0.
/// `NeuronSplit::is_relu_split` looks at the bounds alone, so before the fix
/// `with_general_split` recorded a ReLU `GraphNeuronConstraint` on a Sin node,
/// which `build_constraint_lookups` rejects ("references non-ReLU/Sign node"):
/// both children failed propagation and the domain was lost. It must be a GenBaB
/// split-point constraint, and the child must propagate.
#[ntest::timeout(10000)]
#[test]
fn genbab_zero_point_split_on_non_relu_is_a_genbab_constraint() {
    let graph = single_activation_on_input("sin", Layer::Sin(SinLayer::new()));
    let (input, root) = root_for(&graph, -1.0, 1.0);
    let parent = scalar_interval(&input);

    let (decision, children) = split_through_production_path(&graph, &root, "sin");
    assert_eq!(
        decision.points,
        vec![0.0],
        "fixture premise: the midpoint of [-1, 1] is exactly 0.0"
    );

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let objective = [1.0_f32];
    for child in &children {
        assert!(
            child.history.constraints.is_empty(),
            "a 0.0 split on a Sin node must NOT become a ReLU constraint: {:?}",
            child.history.constraints
        );
        assert_eq!(child.history.genbab_constraints.len(), 1);
        assert_eq!(child.history.genbab_constraints[0].split_point(), 0.0);

        let context = GraphCrownContext::new(&child.history, None, Some(&root.node_bounds), None);
        verifier
            .propagate_crown_with_graph_constraints(
                &graph,
                child.input_bounds.as_ref(),
                &context,
                Some(&child.beta_state),
                Some(&objective),
            )
            .expect("the child of a 0.0 Sin split must propagate");
    }
    let intervals: Vec<(f32, f32)> = children
        .iter()
        .map(|child| scalar_interval(child.input_bounds.as_ref()))
        .collect();
    assert_children_cover_parent(parent, &intervals, &decision.points);
}

/// Sigmoid fed by an INTERMEDIATE node (`lin`, identity Linear) — the shape of
/// every ml4acopf activation. The child's interval is read from the GenBaB
/// constraint it records (upper branch: x >= p; lower branch: x <= p), which is
/// exactly the clamp the constrained forward pass applies to `lin`.
#[ntest::timeout(10000)]
#[test]
fn genbab_split_children_cover_parent_intermediate_pre_activation() {
    let linear = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("identity linear");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("lin", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["lin".to_string()],
    ));
    graph.set_output("sig");

    let (_input, root) = root_for(&graph, 2.0, 6.0);
    let parent = scalar_interval(
        root.node_bounds
            .get("lin")
            .expect("IBP bounds for the pre-activation node")
            .as_ref(),
    );

    let (decision, children) = split_through_production_path(&graph, &root, "sig");
    assert_eq!(
        decision.original_bounds, parent,
        "the decision must carry node_bounds[\"lin\"], the sigmoid's INPUT"
    );

    let intervals: Vec<(f32, f32)> = children
        .iter()
        .map(|child| {
            let mut interval = parent;
            for constraint in &child.history.genbab_constraints {
                assert_eq!(constraint.node_name(), "sig");
                if constraint.is_upper_branch() {
                    interval.0 = interval.0.max(constraint.split_point());
                } else {
                    interval.1 = interval.1.min(constraint.split_point());
                }
            }
            assert!(
                !child.history.genbab_constraints.is_empty(),
                "every child must record the split it represents"
            );
            interval
        })
        .collect();
    assert_children_cover_parent(parent, &intervals, &decision.points);
}
