// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the `#crown-repropagate` forward sweep.
//!
//! Every test injects [`Options`] directly instead of setting environment
//! variables, so nothing here depends on (or perturbs) the process-global
//! environment that cargo's parallel test threads share.

use super::crown_repropagate::{sweep, Options};
use crate::layers::{
    AddLayer, DivLayer, ExpLayer, Layer, LinearLayer, MulBinaryLayer, ReLULayer, SubLayer,
};
use crate::network::core::{GraphNetwork, GraphNode};
use ndarray::{Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;

fn tensor(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let shape = IxDyn(&[lower.len()]);
    BoundedTensor::new(
        ArrayD::from_shape_vec(shape.clone(), lower.to_vec()).expect("lower"),
        ArrayD::from_shape_vec(shape, upper.to_vec()).expect("upper"),
    )
    .expect("bounded tensor")
}

fn identity_linear(dim: usize) -> Layer {
    let mut data = vec![0.0_f32; dim * dim];
    for i in 0..dim {
        data[i * dim + i] = 1.0;
    }
    let weight = Array2::from_shape_vec((dim, dim), data).expect("identity weight");
    Layer::Linear(LinearLayer::new(weight, None).expect("identity linear"))
}

fn armed() -> Options {
    Options {
        binary_arm: true,
        debug: false,
    }
}

fn unary_only() -> Options {
    Options {
        binary_arm: false,
        debug: false,
    }
}

/// The defect this whole lane exists to fix, in miniature.
///
/// `x -> relu` where the stored `relu` bound is the STALE, untightened one (width 8)
/// while its input has since been tightened to width 0.5. A monotone 1-Lipschitz
/// ReLU cannot widen its input, so the sweep must pull `relu` down to its input's
/// width — exactly the `Add_8 (0.6461) -> Relu_9 (8.2257)` violation measured on
/// TinyYOLO.
#[ntest::timeout(10000)]
#[test]
fn sweep_pulls_a_stale_consumer_down_to_its_tightened_producer() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("stem", identity_linear(2)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));

    let input = tensor(&[0.0, 0.0], &[1.0, 1.0]);
    let mut bounds = HashMap::new();
    // Producer: already CROWN-tightened.
    bounds.insert("stem".to_string(), tensor(&[0.5, 0.5], &[1.0, 1.0]));
    // Consumer: the stale value a demand-skip stored, far wider than possible.
    bounds.insert("relu".to_string(), tensor(&[0.0, 0.0], &[8.0, 8.0]));

    let exec_order = vec!["stem".to_string(), "relu".to_string()];
    let stats = sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        armed(),
        &mut bounds,
    );

    let relu = &bounds["relu"];
    assert_eq!(relu.lower().as_slice().unwrap(), &[0.5, 0.5]);
    assert_eq!(relu.upper().as_slice().unwrap(), &[1.0, 1.0]);
    assert!(stats.repropagated >= 1, "stats: {stats:?}");
}

/// Compounding is the point: a single topological pass must carry tightening
/// through a CHAIN, not just one hop. `relu_a`'s repaired bound has to feed
/// `relu_b` in the same sweep.
#[ntest::timeout(10000)]
#[test]
fn sweep_compounds_along_a_chain_in_one_topological_pass() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("stem", identity_linear(1)));
    graph.add_node(GraphNode::new(
        "relu_a",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu_b",
        Layer::ReLU(ReLULayer),
        vec!["relu_a".to_string()],
    ));

    let input = tensor(&[0.0], &[1.0]);
    let mut bounds = HashMap::new();
    bounds.insert("stem".to_string(), tensor(&[0.25], &[0.75]));
    bounds.insert("relu_a".to_string(), tensor(&[0.0], &[9.0]));
    bounds.insert("relu_b".to_string(), tensor(&[0.0], &[9.0]));

    let exec_order = vec![
        "stem".to_string(),
        "relu_a".to_string(),
        "relu_b".to_string(),
    ];
    sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        armed(),
        &mut bounds,
    );

    assert_eq!(bounds["relu_a"].upper().as_slice().unwrap(), &[0.75]);
    // The second hop only reaches 0.75 if it consumed relu_a's REPAIRED bound.
    assert_eq!(bounds["relu_b"].upper().as_slice().unwrap(), &[0.75]);
}

/// The binary arm (`#crown-repropagate-binary`) is the increment that lets the
/// tightening cross a residual join. With it disarmed the `Add` must stay stale —
/// that is the unary-only behavior the A/B measures against.
#[ntest::timeout(10000)]
#[test]
fn binary_arm_carries_tightening_across_an_add_and_the_sub_gate_disarms_it() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("stem", identity_linear(1)));
    graph.add_node(GraphNode::new(
        "left",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "join",
        Layer::Add(AddLayer),
        vec!["left".to_string(), "stem".to_string()],
    ));

    let input = tensor(&[0.0], &[1.0]);
    let seed = |bounds: &mut HashMap<String, BoundedTensor>| {
        bounds.insert("stem".to_string(), tensor(&[0.25], &[0.75]));
        bounds.insert("left".to_string(), tensor(&[0.25], &[0.75]));
        bounds.insert("join".to_string(), tensor(&[-40.0], &[40.0]));
    };
    let exec_order = vec!["stem".to_string(), "left".to_string(), "join".to_string()];

    let mut armed_bounds = HashMap::new();
    seed(&mut armed_bounds);
    let armed_stats = sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        armed(),
        &mut armed_bounds,
    );
    // 0.25+0.25 .. 0.75+0.75, widened by at most one ULP in each direction.
    assert!(armed_bounds["join"].lower().as_slice().unwrap()[0] <= 0.5);
    assert!(armed_bounds["join"].lower().as_slice().unwrap()[0] > 0.49);
    assert!(armed_bounds["join"].upper().as_slice().unwrap()[0] >= 1.5);
    assert!(armed_bounds["join"].upper().as_slice().unwrap()[0] < 1.51);
    assert_eq!(armed_stats.binary_repropagated, 1, "{armed_stats:?}");

    let mut unary_bounds = HashMap::new();
    seed(&mut unary_bounds);
    let unary_stats = sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        unary_only(),
        &mut unary_bounds,
    );
    assert_eq!(unary_bounds["join"].lower().as_slice().unwrap(), &[-40.0]);
    assert_eq!(unary_bounds["join"].upper().as_slice().unwrap(), &[40.0]);
    assert_eq!(unary_stats.binary_repropagated, 0, "{unary_stats:?}");
}

/// The binary result must be widened OUTWARD by a ULP before it is intersected in.
/// Each endpoint of an `Add`/`Sub`/`Mul`/`Div` interval is a single f32 operation
/// rounded to NEAREST, which can round INWARD; a map this sweep intersects into
/// cannot absorb that. `0.1 + 0.2` is the classic witness: the f32 sum is not the
/// real sum, and the stored lower bound must not exceed the true one.
#[ntest::timeout(10000)]
#[test]
fn binary_transfer_is_widened_outward_so_nearest_rounding_cannot_shave_the_bound() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("a", identity_linear(1)));
    graph.add_node(GraphNode::new(
        "b",
        Layer::ReLU(ReLULayer),
        vec!["a".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "sum",
        Layer::Add(AddLayer),
        vec!["a".to_string(), "b".to_string()],
    ));

    let input = tensor(&[0.1], &[0.1]);
    let mut bounds = HashMap::new();
    bounds.insert("a".to_string(), tensor(&[0.1], &[0.1]));
    bounds.insert("b".to_string(), tensor(&[0.2], &[0.2]));
    bounds.insert("sum".to_string(), tensor(&[-10.0], &[10.0]));
    let exec_order = vec!["a".to_string(), "b".to_string(), "sum".to_string()];

    sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        armed(),
        &mut bounds,
    );

    let plain = 0.1_f32 + 0.2_f32;
    let lower = bounds["sum"].lower().as_slice().unwrap()[0];
    let upper = bounds["sum"].upper().as_slice().unwrap()[0];
    assert!(
        lower < plain && upper > plain,
        "binary transfer must be strictly outside the round-to-nearest sum: \
         [{lower}, {upper}] vs {plain}"
    );
}

/// Refusal, not corruption, is the contract for an unadmitted op. `Exp` rests on a
/// faithful-libm ASSUMPTION rather than a certificate, so it must be left alone even
/// though its interval transfer exists and would "work".
#[ntest::timeout(10000)]
#[test]
fn uncertified_transcendental_is_refused_and_leaves_its_bound_untouched() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("stem", identity_linear(1)));
    graph.add_node(GraphNode::new(
        "exp",
        Layer::Exp(ExpLayer),
        vec!["stem".to_string()],
    ));

    let input = tensor(&[0.0], &[1.0]);
    let mut bounds = HashMap::new();
    bounds.insert("stem".to_string(), tensor(&[0.0], &[0.1]));
    bounds.insert("exp".to_string(), tensor(&[-5.0], &[5.0]));
    let exec_order = vec!["stem".to_string(), "exp".to_string()];

    let stats = sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        armed(),
        &mut bounds,
    );

    assert_eq!(bounds["exp"].lower().as_slice().unwrap(), &[-5.0]);
    assert_eq!(bounds["exp"].upper().as_slice().unwrap(), &[5.0]);
    assert!(stats.refused >= 1, "{stats:?}");
}

/// A disjoint recompute means the two enclosures contradict each other, which can
/// only happen under a bug upstream. Adopting the union would WIDEN a sound bound
/// and adopting the intersection would CROSS it (`lower > upper`), and a crossed
/// bound certifies anything. The sweep must fail closed and keep what it had.
#[ntest::timeout(10000)]
#[test]
fn disjoint_recompute_fails_closed_and_never_widens_or_crosses() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("stem", identity_linear(1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));

    let input = tensor(&[0.0], &[1.0]);
    let mut bounds = HashMap::new();
    // relu(stem) lands in [5, 6]; the stored bound claims [0, 1]. Disjoint.
    bounds.insert("stem".to_string(), tensor(&[5.0], &[6.0]));
    bounds.insert("relu".to_string(), tensor(&[0.0], &[1.0]));
    let exec_order = vec!["stem".to_string(), "relu".to_string()];

    let stats = sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        armed(),
        &mut bounds,
    );

    let relu = &bounds["relu"];
    assert_eq!(relu.lower().as_slice().unwrap(), &[0.0]);
    assert_eq!(relu.upper().as_slice().unwrap(), &[1.0]);
    assert!(
        relu.lower().as_slice().unwrap()[0] <= relu.upper().as_slice().unwrap()[0],
        "a crossed bound escaped the fail-closed guard"
    );
    assert!(stats.refused >= 1, "{stats:?}");
}

/// The sweep is strictly non-widening. A LOOSER recompute (the stored bound was
/// already CROWN-tightened past what IBP can reproduce) must be discarded, not
/// adopted — otherwise arming the gate could REGRESS a bound.
#[ntest::timeout(10000)]
#[test]
fn a_looser_recompute_never_replaces_an_already_tighter_stored_bound() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("stem", identity_linear(1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));

    let input = tensor(&[0.0], &[1.0]);
    let mut bounds = HashMap::new();
    bounds.insert("stem".to_string(), tensor(&[0.0], &[1.0]));
    // CROWN already proved a much tighter box than relu([0,1]) = [0,1].
    bounds.insert("relu".to_string(), tensor(&[0.4], &[0.6]));
    let exec_order = vec!["stem".to_string(), "relu".to_string()];

    sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        armed(),
        &mut bounds,
    );

    assert_eq!(bounds["relu"].lower().as_slice().unwrap(), &[0.4]);
    assert_eq!(bounds["relu"].upper().as_slice().unwrap(), &[0.6]);
}

/// A node the collection never bounded must not gain an entry: the sweep only ever
/// narrows what the loop produced. Its consumers then find nothing and decline too.
#[ntest::timeout(10000)]
#[test]
fn sweep_never_invents_an_entry_for_an_unbounded_node() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("stem", identity_linear(1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));

    let input = tensor(&[0.0], &[1.0]);
    let mut bounds = HashMap::new();
    bounds.insert("stem".to_string(), tensor(&[0.0], &[0.5]));
    // "relu" deliberately absent.
    let exec_order = vec!["stem".to_string(), "relu".to_string()];

    sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        armed(),
        &mut bounds,
    );

    assert!(!bounds.contains_key("relu"));
}

/// MulBinary is ml4acopf's power-flow frontier. Interval multiplication treats its
/// two operands as independent, which OVER-approximates their dependence and is
/// therefore sound; the sweep must admit it and carry the tightening across.
#[ntest::timeout(10000)]
#[test]
fn mul_binary_is_admitted_and_carries_tightening_across_the_product() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("stem", identity_linear(1)));
    graph.add_node(GraphNode::new(
        "gate",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "prod",
        Layer::MulBinary(MulBinaryLayer),
        vec!["stem".to_string(), "gate".to_string()],
    ));

    let input = tensor(&[0.0], &[1.0]);
    let mut bounds = HashMap::new();
    bounds.insert("stem".to_string(), tensor(&[2.0], &[3.0]));
    bounds.insert("gate".to_string(), tensor(&[2.0], &[3.0]));
    bounds.insert("prod".to_string(), tensor(&[-100.0], &[100.0]));
    let exec_order = vec!["stem".to_string(), "gate".to_string(), "prod".to_string()];

    let stats = sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        armed(),
        &mut bounds,
    );

    let lower = bounds["prod"].lower().as_slice().unwrap()[0];
    let upper = bounds["prod"].upper().as_slice().unwrap()[0];
    assert!((3.9..=4.0).contains(&lower), "lower={lower}");
    assert!((9.0..=9.1).contains(&upper), "upper={upper}");
    assert_eq!(stats.binary_repropagated, 1, "{stats:?}");
}

/// Sub is elementwise and single-rounded like Add, so it takes the same arm.
#[ntest::timeout(10000)]
#[test]
fn sub_is_admitted_by_the_binary_arm() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("stem", identity_linear(1)));
    graph.add_node(GraphNode::new(
        "other",
        Layer::ReLU(ReLULayer),
        vec!["stem".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "diff",
        Layer::Sub(SubLayer),
        vec!["stem".to_string(), "other".to_string()],
    ));

    let input = tensor(&[0.0], &[1.0]);
    let mut bounds = HashMap::new();
    bounds.insert("stem".to_string(), tensor(&[4.0], &[5.0]));
    bounds.insert("other".to_string(), tensor(&[1.0], &[2.0]));
    bounds.insert("diff".to_string(), tensor(&[-50.0], &[50.0]));
    let exec_order = vec!["stem".to_string(), "other".to_string(), "diff".to_string()];

    let stats = sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        armed(),
        &mut bounds,
    );

    let lower = bounds["diff"].lower().as_slice().unwrap()[0];
    let upper = bounds["diff"].upper().as_slice().unwrap()[0];
    assert!((1.9..=2.0).contains(&lower), "lower={lower}");
    assert!((4.0..=4.1).contains(&upper), "upper={upper}");
    assert_eq!(stats.binary_repropagated, 1, "{stats:?}");
}

/// A Div whose denominator straddles zero produces a non-finite interval. That is a
/// valid enclosure, so the sweep must neither crash nor let a NaN through: the
/// stored bound simply survives.
#[ntest::timeout(10000)]
#[test]
fn div_through_zero_leaves_the_stored_bound_sound() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("num", identity_linear(1)));
    graph.add_node(GraphNode::new(
        "den",
        Layer::ReLU(ReLULayer),
        vec!["num".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "quot",
        Layer::Div(DivLayer),
        vec!["num".to_string(), "den".to_string()],
    ));

    let input = tensor(&[-1.0], &[1.0]);
    let mut bounds = HashMap::new();
    // `num` spans zero, so `den = relu(num)` is [0, 1] — a denominator that
    // straddles zero and drives the quotient to ±inf.
    bounds.insert("num".to_string(), tensor(&[-1.0], &[1.0]));
    bounds.insert("den".to_string(), tensor(&[-1.0], &[1.0]));
    bounds.insert("quot".to_string(), tensor(&[-7.0], &[7.0]));
    let exec_order = vec!["num".to_string(), "den".to_string(), "quot".to_string()];

    sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        armed(),
        &mut bounds,
    );

    let quot = &bounds["quot"];
    let lower = quot.lower().as_slice().unwrap()[0];
    let upper = quot.upper().as_slice().unwrap()[0];
    assert!(!lower.is_nan() && !upper.is_nan(), "[{lower}, {upper}]");
    assert!(lower <= upper, "[{lower}, {upper}]");
    // The ±inf recompute carries no information, so the stored enclosure must be
    // exactly preserved — narrowed by nothing, and certainly never widened to ±inf.
    assert_eq!(lower, -7.0, "[{lower}, {upper}]");
    assert_eq!(upper, 7.0, "[{lower}, {upper}]");
}

/// The whole sweep is a no-op on a graph it cannot admit anything from, and it
/// reports that honestly rather than silently doing nothing.
#[ntest::timeout(10000)]
#[test]
fn stats_distinguish_repropagated_from_refused() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("stem", identity_linear(1)));
    graph.add_node(GraphNode::new(
        "exp",
        Layer::Exp(ExpLayer),
        vec!["stem".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["exp".to_string()],
    ));

    let input = tensor(&[0.0], &[1.0]);
    let mut bounds = HashMap::new();
    bounds.insert("stem".to_string(), tensor(&[0.0], &[0.5]));
    bounds.insert("exp".to_string(), tensor(&[0.5], &[0.9]));
    bounds.insert("relu".to_string(), tensor(&[0.0], &[9.0]));
    let exec_order = vec!["stem".to_string(), "exp".to_string(), "relu".to_string()];

    let stats = sweep(
        &graph,
        &input,
        &exec_order,
        None,
        None,
        armed(),
        &mut bounds,
    );

    // exp refused; relu repropagated off exp's (untouched but valid) bound.
    assert_eq!(stats.refused, 1, "{stats:?}");
    assert_eq!(stats.repropagated, 2, "{stats:?}");
    assert_eq!(bounds["relu"].upper().as_slice().unwrap(), &[0.9]);
}
