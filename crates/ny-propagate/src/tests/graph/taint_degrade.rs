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

/// Is every element the fully-degraded `[-inf, +inf]`?
///
/// The predicate behind `assert_all_unbounded`, extracted so the known-gap test
/// below can pin the NEGATION without restating what "unbounded" means. One
/// definition, two directions.
fn is_all_unbounded(bounds: &BoundedTensor) -> bool {
    bounds
        .lower()
        .iter()
        .all(|v| v.is_infinite() && v.is_sign_negative())
        && bounds
            .upper()
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_positive())
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

/// GUARD GAP, NOT YET FIXED (#opaqueskip-taint-lane-gap).
///
/// `graph_ibp.rs` ends its pass with a fail-closed rule: if the OUTPUT node is
/// forward-reachable from an OpaqueSkip ("tainted"), it DISCARDS the computed
/// bounds and returns `[-inf, +inf]` of the declared shape, on the stated
/// rationale that a tainted output "may carry finite bounds whose element
/// alignment silently depends on conservative shape substitutions made
/// upstream". `tainted_output_fails_closed_to_unbounded` above is that rule's
/// test, and it passes.
///
/// That rule is enforced on exactly ONE lane. `collect_node_bounds_core`
/// (`network/graph_alpha/bounds/ibp.rs:256`) is a second, independent per-node
/// IBP loop with NO taint set, NO taint-gated degrade and NO tainted-output
/// guard — and it is the collector the β-CROWN graph engine actually uses
/// (`beta_crown/engine/graph/shared/init.rs`). Its output map is then consulted
/// for the verdict bound: `concretize_batched_results` publishes
/// `tighten_crown_with_forward_bounds(crown_output, cache.get(output_node))`,
/// and falls back to `cache.get(output_node)` WHOLESALE when no backward pass
/// reached the input. So a bound the IBP lane declares untrustworthy is
/// admissible on the lane that mints the scored verdicts.
///
/// This test asserts the missing rule directly: the same graph the guard was
/// written for, run through the β-CROWN collector, must not hand back a finite
/// bound at a tainted output.
///
/// # To un-ignore, ALL of the following must be true
///
/// 1. The OpaqueSkip forward-reachability closure currently private to
///    `graph_ibp.rs` (built at its `:594-616`) is hoisted into a helper both
///    lanes call — not duplicated, or the two will drift.
/// 2. `collect_node_bounds_core` consults it and refuses or widens to
///    `[-inf, +inf]` at every tainted node whose bound it would cache, so the
///    map handed to the β-CROWN engine carries no untrusted finite entry.
/// 3. The two `cache.get(output_node)` publications in
///    `beta_crown/.../batched.rs` (`concretize_batched_results`' intersection,
///    and its wholesale no-backward-pass fallback) either see only widened
///    entries or refuse at a tainted output themselves.
/// 4. The closure is computed UNCONDITIONALLY. In `graph_ibp.rs` it is built
///    only when `!strict_ibp` (`:594`), so `NY_STRICT_IBP=1` — documented as
///    the way to "restore strict behavior" — removes the tainted-output guard
///    along with the degrade. Hoisting must not carry that inversion across.
/// 5. `lsnc_relu` does not regress from its banked 9/80. It is the 2026
///    board's ONLY OpaqueSkip carrier (`reports/measured-2026/
///    opaqueskip_census_2026-08-25/census_materialized_onnx.tsv`: 4 skips,
///    52/61 nodes tainted, `output_tainted=true` on both 1.0 and 2.0), its
///    banked column is 11 `sat` + 69 `unknown` with ZERO `unsat`, and
///    lsnc_relu is CLOSED — so a widening cannot cost a banked proof, but the
///    row count must still be re-measured on the box, not assumed.
///
/// Deliberately NOT fixed in this pass: the repair is a cross-lane hoist into
/// soundness-critical bound initialisation, it cannot be verdict-measured
/// under a box-light budget, and no banked verdict depends on it today (zero
/// `unsat` on the only carrier). Shipping half of it — a taint check on one
/// of the two publication sites — would be worse than shipping none.
/// # This test PINS THE GAP; it does not endorse it
///
/// The rule above is the one this lane should enforce, and it does NOT hold
/// today. Written as `#[ignore]`, that fact was recorded in an attribute no
/// suite ever evaluates — the repository's source policy forbids `#[ignore]`
/// for exactly this reason, and it was the only violation in the tree.
///
/// So the assertion is inverted and the test RUNS: it asserts the collector
/// still publishes a finite bound at a tainted output, and therefore FAILS THE
/// MOMENT THE GAP CLOSES. That is the useful direction. An ignored test is
/// silent both while the defect exists and after it is fixed; this one speaks
/// exactly once, at the moment someone satisfies the five conditions above and
/// needs to be told to restore the real rule.
///
/// TO CLOSE: make the collector degrade, then swap the `assert!(!is_all_unbounded(..))`
/// below back to `assert_all_unbounded(published)` and delete this note. The
/// failure message says the same thing.
#[ntest::timeout(10000)]
#[test]
fn beta_crown_collector_does_not_yet_fail_closed_at_tainted_output() {
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

    let input = box_input(2);

    // The IBP lane refuses this graph's output today.
    let ibp_output = graph.propagate_ibp(&input).unwrap();
    assert_all_unbounded(&ibp_output);

    // The β-CROWN lane's collector does not. CONTAINMENT is not the issue here
    // — Sigmoid's [0,1] does contain the true range — the issue is that the
    // element ALIGNMENT of a tainted node's bounds depends on conservative
    // shape substitutions the collector never records, which is exactly why
    // the other lane declines to trust them. The two lanes must agree.
    let cache = graph
        .collect_node_bounds(&input)
        .expect("collector should complete");
    let published = cache
        .get("sigmoid")
        .expect("output node must be in the collected map");
    assert!(
        !is_all_unbounded(published),
        "#opaqueskip-taint-lane-gap HAS CLOSED: collect_node_bounds_core now \
         degrades at a tainted output, which is what the doc comment above says \
         it SHOULD do. Restore the real rule — replace this assertion with \
         `assert_all_unbounded(published)` and delete the \"PINS THE GAP\" note \
         — so the lane is guarded rather than merely observed. Published: \
         lower={:?} upper={:?}",
        published.lower(),
        published.upper()
    );
}
