// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Taint-gated IBP degrade tests (#cctsdb never-hard-error, Phase A).
//!
//! Nodes forward-reachable from an OpaqueSkip ("tainted") already carry no
//! information beyond `[-inf, +inf]`. When bound computation at such a node
//! fails structurally (ShapeMismatch / InvalidSpec / UnsupportedOp), the graph
//! IBP loop substitutes conservative unbounded bounds of the declared shape
//! instead of aborting — so the pass always reaches the network output.
//! Errors at UNtainted nodes still abort (they indicate real bugs).
//!
//! Soundness invariant: every substitution is `[-inf, +inf]`, which
//! over-approximates any op output.

use crate::layers::{AddLayer, OpaqueSkipLayer, ReshapeLayer, SigmoidLayer};
use crate::{BoundedTensor, GraphNetwork, GraphNode, Layer};
use ndarray::arr1;

/// `strict_ibp_env_restores_abort` mutates the process-global NY_STRICT_IBP
/// env var, which every test in this module implicitly reads through
/// `propagate_ibp`. Serialize all tests in this module to avoid the race.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn box_input(len: usize) -> BoundedTensor {
    let lower = arr1(&vec![-1.0_f32; len]).into_dyn();
    let upper = arr1(&vec![1.0_f32; len]).into_dyn();
    BoundedTensor::new(lower, upper).unwrap()
}

fn assert_all_unbounded(bounds: &BoundedTensor) {
    assert!(
        bounds
            .lower()
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_negative()),
        "lower bounds must be -inf, got {:?}",
        bounds.lower()
    );
    assert!(
        bounds
            .upper()
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_positive()),
        "upper bounds must be +inf, got {:?}",
        bounds.upper()
    );
}

/// A shape mismatch at a node DOWNSTREAM of an OpaqueSkip must degrade to
/// unbounded bounds (declared shape or first-input fallback), not abort.
///
/// Graph: input[2] -> opaque (declared shape [3]) -> add(opaque, input)
/// The binary Add sees [3] vs [2] and errors; the degrade path substitutes
/// unbounded bounds so propagation completes.
#[ntest::timeout(10000)]
#[test]
fn tainted_shape_mismatch_degrades_to_unbounded() {
    let _guard = env_guard();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "opaque",
        Layer::OpaqueSkip(OpaqueSkipLayer::with_output_shape(vec![3])),
    ));
    graph.add_node(GraphNode::new(
        "add_mismatched",
        Layer::Add(AddLayer),
        vec!["opaque".to_string(), crate::NETWORK_INPUT.to_string()],
    ));
    graph.set_output("add_mismatched");

    let result = graph
        .propagate_ibp(&box_input(2))
        .expect("tainted shape mismatch must degrade, not abort");
    assert_all_unbounded(&result);
}

/// The same failure at an UNtainted node (no OpaqueSkip anywhere) must still
/// abort — degrading there would mask real conversion/propagation bugs.
#[ntest::timeout(10000)]
#[test]
fn untainted_error_still_aborts() {
    let _guard = env_guard();
    let mut graph = GraphNetwork::new();
    // Reshape [2] -> [5]: element count mismatch, structural error.
    graph.add_node(GraphNode::from_input(
        "bad_reshape",
        Layer::Reshape(ReshapeLayer::new(vec![5])),
    ));
    graph.set_output("bad_reshape");

    let result = graph.propagate_ibp(&box_input(2));
    assert!(
        result.is_err(),
        "structural error at an untainted node must abort, got {:?}",
        result.map(|b| b.shape().to_vec())
    );
}

/// Shape-carrying OpaqueSkip: the declared output shape (not the input
/// shape) shapes the conservative bounds through the graph path.
#[ntest::timeout(10000)]
#[test]
fn opaque_skip_declared_shape_reaches_graph_output() {
    let _guard = env_guard();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "opaque",
        Layer::OpaqueSkip(OpaqueSkipLayer::with_output_shape(vec![4, 2])),
    ));
    graph.set_output("opaque");

    let result = graph.propagate_ibp(&box_input(3)).unwrap();
    assert_eq!(result.shape(), &[4, 2]);
    assert_all_unbounded(&result);
}

/// Fail-closed output guard: a tainted OUTPUT node must never return finite
/// bounds, even when downstream range-clamped ops (Sigmoid) produce them —
/// element alignment under conservative shape substitutions is not trusted.
///
/// Graph: input[2] -> opaque[2] -> sigmoid (finite [0,1] from +-inf input).
#[ntest::timeout(10000)]
#[test]
fn tainted_output_fails_closed_to_unbounded() {
    let _guard = env_guard();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "opaque",
        Layer::OpaqueSkip(OpaqueSkipLayer::new()),
    ));
    graph.add_node(GraphNode::new(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["opaque".to_string()],
    ));
    graph.set_output("sigmoid");

    let result = graph.propagate_ibp(&box_input(2)).unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_all_unbounded(&result);
}

/// An untainted graph that succeeds is untouched by the degrade machinery
/// (no OpaqueSkip => empty taint set => identical behavior).
#[ntest::timeout(10000)]
#[test]
fn untainted_graph_bounds_unchanged() {
    let _guard = env_guard();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer::new()),
    ));
    graph.set_output("sigmoid");

    let result = graph.propagate_ibp(&box_input(2)).unwrap();
    assert_eq!(result.shape(), &[2]);
    for (&l, &u) in result.lower().iter().zip(result.upper().iter()) {
        assert!(l.is_finite() && u.is_finite());
        assert!((0.0..=1.0).contains(&l) && (0.0..=1.0).contains(&u));
        assert!(l <= u);
    }
}

/// NY_STRICT_IBP=1 escape hatch restores abort-on-error at tainted nodes.
///
/// Env-var mutation: run in a dedicated test to avoid cross-test races; the
/// variable is removed before any assertion that could early-exit.
#[ntest::timeout(10000)]
#[test]
fn strict_ibp_env_restores_abort() {
    let _guard = env_guard();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "opaque",
        Layer::OpaqueSkip(OpaqueSkipLayer::with_output_shape(vec![3])),
    ));
    graph.add_node(GraphNode::new(
        "add_mismatched",
        Layer::Add(AddLayer),
        vec!["opaque".to_string(), crate::NETWORK_INPUT.to_string()],
    ));
    graph.set_output("add_mismatched");

    // Route the mutation through the blessed env choke point (clippy env
    // wall): hold the process-wide env lock for cross-module safety on top of
    // this module's reader-serializing ENV_LOCK, and restore via guard drop.
    let strict_result = {
        let _env_lock = ny_test_utils::env::lock_env();
        let _strict = ny_test_utils::env::ScopedEnvVar::set("NY_STRICT_IBP", "1");
        graph.propagate_ibp(&box_input(2))
    };

    assert!(
        strict_result.is_err(),
        "NY_STRICT_IBP=1 must restore the abort-on-error behavior"
    );

    // And without the env var the same graph degrades gracefully.
    let degraded = graph
        .propagate_ibp(&box_input(2))
        .expect("default mode must degrade");
    assert_all_unbounded(&degraded);
}
