// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #warm-dag: when a child's backward pass may start from its parent's cached
//! lA block instead of the output.
//!
//! The lA warm start (`BetaCrownConfig::enable_la_warm_start`, default on and
//! passed as `!no_la_warm_start` by the ny-cli β-CROWN path) seeds a child
//! domain's backward pass at `history.last_branch_node()` =: `N` with ONE block:
//! the coefficients `(A_N, b_N)` that the parent's pass held for `N` when it
//! popped `N` (captured at `constraints/backward/dispatch.rs`
//! `capture_constrained_linear_bounds`, at the "capture intermediate lA" step
//! of `propagation/batched.rs`, and at the root in
//! `network/graph_crown/spec_propagation/core.rs`). The three sites that
//! consume such a seed ask [`la_warm_start_seed_is_complete`] first:
//! `constraints/backward/setup.rs` and the two `WithLaCapture` arms of
//! `propagation/batched.rs` (scalar-objective core and dense-spec core).
//!
//! # Why a single block is not always a bound
//!
//! A reverse-topological backward pass keeps, at every step, a set of pending
//! blocks `{M: (A_M, b_M)}` and an input accumulator `(A_in, b_in)` with, for
//! every point of the parent region (lower side; the upper side is symmetric),
//!
//! ```text
//!   f(x) >= sum_M (A_M · y_M(x) + b_M) + A_in · x + b_in.
//! ```
//!
//! When `N` is popped the capture records `(A_N, b_N)` only. A child seeded
//! with it therefore bounds `f − sum_{M != N} (A_M·y_M + b_M) − A_in·x − b_in`,
//! not `f`, unless every other term is identically zero at that moment. They
//! are NOT zero whenever the output reaches the input by a path that avoids
//! `N` (a skip connection: the skip operand's block is still pending), or a
//! binary / n-ary node sits between the output and `N` (Add, Sub, Concat,
//! OpaqueSkip, … move the bias accumulated above them into the separate
//! network-input bias channel, `network/backward_dispatch/dispatch.rs`
//! `BackwardDispatchResult::{Binary, Nary}`). Both ends of the interval are
//! affected, in lower and in upper mode alike. Measured: on
//! `out = Σ_i relu(x_i) + x_i`, `x ∈ [-1,1]²` (true range `[-2, 4]`), the
//! seeded multi-objective lane published `Verified` for `out < 3` at
//! `explored=1`, and the DomainList lane published `Verified` for
//! `upper(out) < 3` (regressions in `graph/tests/warm_start_seed.rs`).
//!
//! # The condition this module enforces
//!
//! Walk from the output node towards the input, following the unique input of
//! each node. If the walk reaches `N` while passing ONLY through nodes with
//! exactly one graph input and a unary layer (`!is_binary() && !is_ternary()`),
//! the seed is complete:
//!
//! * The pass seeds only the output. A unary node forwards its block only to
//!   `inputs[0]` (`BackwardDispatchResult::Single` / `PassThrough`), so by
//!   induction over the reverse order the walk nodes are the only nodes that
//!   ever hold a block before `N` is popped.
//! * No walk node above `N` has the network input as its input (the walk
//!   would have hit `NETWORK_INPUT` first), and none is binary/n-ary, so nothing
//!   was written to the input accumulator: `A_in = 0`, `b_in = 0`.
//! * Hence at the moment `N` is popped `f(x) >= A_N · y_N(x) + b_N` on the
//!   parent region, which contains the child's region, and the parent's
//!   relaxations and β terms folded above `N` remain valid there (the child's
//!   split set is a superset of the parent's, β >= 0). The child re-derives
//!   everything below `N` itself, including its own split at `N`.
//!
//! Any node below `N` (closer to the input) is unrestricted: skip connections
//! there are processed by the child's own pass exactly as by a cold pass. On a
//! residual network the seed is therefore kept only for splits in the final
//! unary chain (for example a classifier head after the last residual Add);
//! every other child falls back to the cold pass from the output, which costs
//! time, never a verdict.
//!
//! This check is structural and cheap (at most one hash lookup per node on the
//! walk), so the seed sites call it per domain.

use crate::{GraphNetwork, NETWORK_INPUT};

/// Whether a backward pass seeded ONLY at `branch_node`, with the block the
/// parent's pass held there, encloses the objective at `output_node`.
///
/// Returns `false` (the caller then seeds at the output) for anything it
/// cannot prove complete: an unknown node, a node with zero or several graph
/// inputs, a binary/ternary layer, reaching `NETWORK_INPUT` before
/// `branch_node`, or a walk longer than the graph (a malformed cyclic graph).
pub(in crate::beta_crown::engine::graph) fn la_warm_start_seed_is_complete(
    graph: &GraphNetwork,
    output_node: &str,
    branch_node: &str,
) -> bool {
    if branch_node == NETWORK_INPUT {
        return false;
    }
    let mut current = output_node;
    // Each step moves to a strictly earlier node of a DAG, so a well-formed
    // walk takes at most `nodes.len()` steps; the bound only guards malformed
    // (cyclic) graphs.
    for _ in 0..=graph.nodes.len() {
        if current == branch_node {
            return true;
        }
        if current == NETWORK_INPUT {
            return false;
        }
        let Some(node) = graph.nodes.get(current) else {
            return false;
        };
        if node.inputs.len() != 1 || node.layer.is_binary() || node.layer.is_ternary() {
            return false;
        }
        current = node.inputs[0].as_str();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::la_warm_start_seed_is_complete;
    use crate::{AddLayer, GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};
    use ndarray::arr2;

    fn identity() -> Layer {
        Layer::Linear(LinearLayer::new(arr2(&[[1.0f32, 0.0], [0.0, 1.0]]), None).expect("linear"))
    }

    fn node(name: &str, layer: Layer, inputs: &[&str]) -> GraphNode {
        GraphNode::new(name, layer, inputs.iter().map(|s| s.to_string()).collect())
    }

    /// `x -> l1 -> r1 -> add(r1, l1) -> l2 -> r2 -> l3 (output)`.
    /// `r1` sits inside the residual block, `r2` in the unary head.
    fn residual_then_head() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("l1", identity()));
        graph.add_node(node("r1", Layer::ReLU(ReLULayer), &["l1"]));
        graph.add_node(node("add", Layer::Add(AddLayer), &["r1", "l1"]));
        graph.add_node(node("l2", identity(), &["add"]));
        graph.add_node(node("r2", Layer::ReLU(ReLULayer), &["l2"]));
        graph.add_node(node("l3", identity(), &["r2"]));
        graph.set_output("l3");
        graph
    }

    #[test]
    fn warm_seed_guard_refuses_a_split_inside_a_residual_block() {
        let graph = residual_then_head();
        assert!(!la_warm_start_seed_is_complete(&graph, "l3", "r1"));
        // `l1` is reached only through the Add as well.
        assert!(!la_warm_start_seed_is_complete(&graph, "l3", "l1"));
    }

    #[test]
    fn warm_seed_guard_keeps_a_split_in_the_unary_head() {
        let graph = residual_then_head();
        assert!(la_warm_start_seed_is_complete(&graph, "l3", "r2"));
        assert!(la_warm_start_seed_is_complete(&graph, "l3", "l3"));
    }

    #[test]
    fn warm_seed_guard_keeps_every_node_of_a_chain() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("l1", identity()));
        graph.add_node(node("r1", Layer::ReLU(ReLULayer), &["l1"]));
        graph.add_node(node("l2", identity(), &["r1"]));
        graph.add_node(node("r2", Layer::ReLU(ReLULayer), &["l2"]));
        graph.add_node(node("l3", identity(), &["r2"]));
        graph.set_output("l3");
        for branch in ["r1", "l2", "r2", "l3"] {
            assert!(
                la_warm_start_seed_is_complete(&graph, "l3", branch),
                "chain branch {branch} must keep its warm start"
            );
        }
    }

    #[test]
    fn warm_seed_guard_refuses_unknown_nodes_and_the_input() {
        let graph = residual_then_head();
        assert!(!la_warm_start_seed_is_complete(&graph, "l3", "missing"));
        assert!(!la_warm_start_seed_is_complete(
            &graph,
            "l3",
            crate::NETWORK_INPUT
        ));
        assert!(!la_warm_start_seed_is_complete(&graph, "missing", "r2"));
    }
}
