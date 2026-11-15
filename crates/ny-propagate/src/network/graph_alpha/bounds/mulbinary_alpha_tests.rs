// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mulbinary-alpha-router: graphs whose ONLY optimizable relaxations are
//! multiplicative must reach the DAG alpha optimizer.
//!
//! Before this router fix the gate tested only ReLU/Sigmoid/Tanh/Sqrt, so an
//! elementwise product — `MulBinary`, of which SwiGLU's `up * silu(gate)` is the
//! canonical frontend shape — left through the batched door, and the batched
//! optimizer has no MulBinary alpha entry point. Such graphs were therefore
//! never alpha-optimized at all.
//!
//! Every route assertion below reads the router's own probe
//! ([`route_probe`]); none infers a route from timing or from bound values.

use crate::bounds::AlphaCrownConfig;
use crate::layers::{Layer, LinearLayer, MulBinaryLayer, ReLULayer, SiLULayer};
use crate::network::core::{GraphNetwork, GraphNode};
use crate::network::graph_alpha::propagate_dag::route_probe::{self, DagAlphaRoute};
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

fn tensor(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let shape = IxDyn(&[lower.len()]);
    BoundedTensor::new(
        ArrayD::from_shape_vec(shape.clone(), lower.to_vec()).expect("lower"),
        ArrayD::from_shape_vec(shape, upper.to_vec()).expect("upper"),
    )
    .expect("bounded tensor")
}

fn config(iterations: usize) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..Default::default()
    }
}

/// Route actually taken by `propagate_alpha_crown_with_config`, read from the
/// router probe rather than inferred.
fn route_of(graph: &GraphNetwork, input: &BoundedTensor, cfg: &AlphaCrownConfig) -> DagAlphaRoute {
    route_probe::reset();
    graph
        .propagate_alpha_crown_with_config(input, cfg)
        .expect("alpha-CROWN should succeed");
    route_probe::last().expect("the DAG router must record a route")
}

/// Assert `actual` is elementwise no looser than `reference`.
fn assert_no_looser(actual: &BoundedTensor, reference: &BoundedTensor, what: &str) {
    assert_eq!(
        actual.lower().len(),
        reference.lower().len(),
        "{what}: element counts differ"
    );
    for (idx, ((&a_l, &a_u), (&r_l, &r_u))) in actual
        .lower()
        .iter()
        .zip(actual.upper().iter())
        .zip(reference.lower().iter().zip(reference.upper().iter()))
        .enumerate()
    {
        assert!(a_l >= r_l, "{what}: lower[{idx}] regressed {r_l} -> {a_l}");
        assert!(a_u <= r_u, "{what}: upper[{idx}] regressed {r_u} -> {a_u}");
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const UP_W: [[f32; 2]; 2] = [[0.9, -0.4], [0.3, 0.7]];
const UP_B: [f32; 2] = [0.2, -0.1];
const GATE_W: [[f32; 2]; 2] = [[-0.6, 0.8], [0.5, 0.25]];
const GATE_B: [f32; 2] = [0.05, 0.15];

/// `up * silu(gate)` off the shared network input — the SwiGLU FFN shape.
///
/// `SiLU` carries no alpha state, so `MulBinary` is the graph's ONLY
/// optimizable node.
fn swiglu_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "up",
        Layer::Linear(LinearLayer::new(arr2(&UP_W), Some(arr1(&UP_B))).expect("up")),
    ));
    graph.add_node(GraphNode::from_input(
        "gate",
        Layer::Linear(LinearLayer::new(arr2(&GATE_W), Some(arr1(&GATE_B))).expect("gate")),
    ));
    graph.add_node(GraphNode::new(
        "silu",
        Layer::SiLU(SiLULayer::new()),
        vec!["gate".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "swiglu",
        Layer::MulBinary(MulBinaryLayer),
        vec!["up".to_string(), "silu".to_string()],
    ));
    graph.set_output("swiglu");

    (graph, tensor(&[-0.5, -0.5], &[0.5, 0.5]))
}

/// Independent oracle for [`swiglu_fixture`]: `up * (gate * sigmoid(gate))`,
/// evaluated in f64 so the reference is not itself an f32 relaxation.
fn swiglu_reference(x: [f32; 2]) -> [f32; 2] {
    let mut out = [0.0_f32; 2];
    for (row, slot) in out.iter_mut().enumerate() {
        let up = (UP_W[row][0] as f64) * (x[0] as f64)
            + (UP_W[row][1] as f64) * (x[1] as f64)
            + UP_B[row] as f64;
        let gate = (GATE_W[row][0] as f64) * (x[0] as f64)
            + (GATE_W[row][1] as f64) * (x[1] as f64)
            + GATE_B[row] as f64;
        let silu = gate / (1.0 + (-gate).exp());
        *slot = (up * silu) as f32;
    }
    out
}

/// The minimal "only optimizable node is a product" graph: two linear branches
/// multiplied elementwise, no activation of any family anywhere.
fn bare_product_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "left",
        Layer::Linear(LinearLayer::new(arr2(&UP_W), Some(arr1(&UP_B))).expect("left")),
    ));
    graph.add_node(GraphNode::from_input(
        "right",
        Layer::Linear(LinearLayer::new(arr2(&GATE_W), Some(arr1(&GATE_B))).expect("right")),
    ));
    graph.add_node(GraphNode::new(
        "product",
        Layer::MulBinary(MulBinaryLayer),
        vec!["left".to_string(), "right".to_string()],
    ));
    graph.set_output("product");

    (graph, tensor(&[-0.5, -0.5], &[0.5, 0.5]))
}

/// A pure-linear DAG with no optimizable node at all — the control that must
/// keep taking the batched early return.
fn no_optimizable_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "left",
        Layer::Linear(LinearLayer::new(arr2(&UP_W), Some(arr1(&UP_B))).expect("left")),
    ));
    graph.add_node(GraphNode::from_input(
        "right",
        Layer::Linear(LinearLayer::new(arr2(&GATE_W), Some(arr1(&GATE_B))).expect("right")),
    ));
    graph.add_node(GraphNode::new(
        "sum",
        Layer::Add(crate::layers::AddLayer),
        vec!["left".to_string(), "right".to_string()],
    ));
    graph.set_output("sum");

    (graph, tensor(&[-0.5, -0.5], &[0.5, 0.5]))
}

/// A ReLU that is provably stable (strictly positive pre-activation) feeding a
/// product. This reaches the SECOND gate: alpha state exists, but nothing in
/// the ReLU part of it is optimizable.
fn stable_relu_product_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(
            LinearLayer::new(
                arr2(&[[0.5_f32, 0.25], [0.25, 0.5]]),
                Some(arr1(&[2.0_f32, 2.0])),
            )
            .expect("pre"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".to_string()],
    ));
    graph.add_node(GraphNode::from_input(
        "other",
        Layer::Linear(LinearLayer::new(arr2(&GATE_W), Some(arr1(&GATE_B))).expect("other")),
    ));
    graph.add_node(GraphNode::new(
        "product",
        Layer::MulBinary(MulBinaryLayer),
        vec!["relu".to_string(), "other".to_string()],
    ));
    graph.set_output("product");

    // Input box keeps `pre` in [2.625, 3.375] — the ReLU has zero unstable
    // neurons, so the second gate used to drop this graph into fixed-slope
    // CROWN even though its MulBinary alphas were live.
    (graph, tensor(&[0.5, 0.5], &[1.0, 1.0]))
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

/// The headline fix: a graph whose only optimizable nodes are `MulBinary`
/// enters the alpha optimizer instead of leaving through the batched door.
#[ntest::timeout(60000)]
#[test]
fn mulbinary_only_graph_enters_the_alpha_optimizer() {
    let (graph, input) = bare_product_fixture();
    assert_eq!(
        route_of(&graph, &input, &config(20)),
        DagAlphaRoute::OptimizerLoop,
        "a MulBinary-only graph must reach the DAG alpha optimizer"
    );
}

/// Same claim for the shape that motivated it.
///
/// Holds the shared env lock: alpha-CROWN admits through
/// `cpu_crown_dense_budget_bytes`, which reads `NY_DENSE_BUDGET_MB`, and a
/// sibling's temporary `0` reached this test once under `--test-threads=4`
/// (`CpuMemoryExceeded { required_bytes: 240, budget_bytes: 0, .. }`,
/// 2026-10-04; it passes alone 3/3 on the branch and on main). The wall is
/// 300 s because a guarded reader also waits behind queued writers.
#[ntest::timeout(300000)]
#[test]
fn mulbinary_swiglu_graph_enters_the_alpha_optimizer() {
    let _env = crate::tests::lock_env_shared();
    let (graph, input) = swiglu_fixture();
    assert_eq!(
        route_of(&graph, &input, &config(20)),
        DagAlphaRoute::OptimizerLoop,
        "SwiGLU's up * silu(gate) must reach the DAG alpha optimizer"
    );
}

/// A stable ReLU alongside a product reaches the SECOND gate. `MulBinary`
/// alphas are live there too, so that gate must not early-return either.
#[ntest::timeout(60000)]
#[test]
fn mulbinary_with_only_stable_relus_still_enters_the_alpha_optimizer() {
    let (graph, input) = stable_relu_product_fixture();
    assert_eq!(
        route_of(&graph, &input, &config(20)),
        DagAlphaRoute::OptimizerLoop,
        "live MulBinary alphas must hold a stable-ReLU graph in the optimizer"
    );
}

/// Control: recognizing products must not drag every graph into the optimizer.
/// A DAG with no optimizable node at all keeps its batched early return.
#[ntest::timeout(60000)]
#[test]
fn mulbinary_router_leaves_non_multiplicative_graphs_on_the_batched_route() {
    let (graph, input) = no_optimizable_fixture();
    assert_eq!(
        route_of(&graph, &input, &config(20)),
        DagAlphaRoute::BatchedEarlyReturn,
        "a graph with no optimizable node must keep the batched early return"
    );
}

// ---------------------------------------------------------------------------
// Tightness: the new route may never be looser than the one it replaced
// ---------------------------------------------------------------------------

/// The re-routed answer must be elementwise no looser than the bound the
/// batched path — the pre-fix route — returns for the same graph.
#[ntest::timeout(60000)]
#[test]
fn mulbinary_alpha_route_is_never_looser_than_the_batched_route() {
    for (name, (graph, input)) in [
        ("bare product", bare_product_fixture()),
        ("swiglu", swiglu_fixture()),
    ] {
        let cfg = config(20);
        // Exactly what `init_dag_alpha_state` returned before the fix.
        let batched = graph
            .propagate_alpha_crown_batched(&input, &cfg, None)
            .expect("batched route should succeed");
        route_probe::reset();
        let alpha = graph
            .propagate_alpha_crown_with_config(&input, &cfg)
            .expect("alpha route should succeed");
        assert_eq!(
            route_probe::last(),
            Some(DagAlphaRoute::OptimizerLoop),
            "{name}: fixture must exercise the re-routed arm"
        );
        assert_no_looser(&alpha, &batched, name);
    }
}

/// And on these fixtures it is not merely equal: recognizing the product buys
/// real tightening, which is the point of the fix.
#[ntest::timeout(60000)]
#[test]
fn mulbinary_alpha_route_strictly_tightens_the_swiglu_fixture() {
    let (graph, input) = swiglu_fixture();
    let cfg = config(20);
    let batched = graph
        .propagate_alpha_crown_batched(&input, &cfg, None)
        .expect("batched route should succeed");
    let alpha = graph
        .propagate_alpha_crown_with_config(&input, &cfg)
        .expect("alpha route should succeed");
    assert_no_looser(&alpha, &batched, "swiglu");
    let strictly_tighter = alpha
        .lower()
        .iter()
        .zip(batched.lower().iter())
        .any(|(a, b)| a > b)
        || alpha
            .upper()
            .iter()
            .zip(batched.upper().iter())
            .any(|(a, b)| a < b);
    assert!(
        strictly_tighter,
        "the alpha route should tighten at least one endpoint; \
         alpha=[{:?}, {:?}] batched=[{:?}, {:?}]",
        alpha.lower(),
        alpha.upper(),
        batched.lower(),
        batched.upper(),
    );
}

/// More iterations may never lose ground: the optimizer's best box is seeded
/// from the initial bound and only tightens, so this is a route-level check
/// that the re-routing did not introduce a non-monotone exit.
#[ntest::timeout(60000)]
#[test]
fn mulbinary_alpha_route_is_monotone_in_iterations() {
    let (graph, input) = swiglu_fixture();
    let few = graph
        .propagate_alpha_crown_with_config(&input, &config(2))
        .expect("2-iteration alpha");
    let many = graph
        .propagate_alpha_crown_with_config(&input, &config(40))
        .expect("40-iteration alpha");
    assert_no_looser(&many, &few, "swiglu iterations 2 -> 40");
}

// ---------------------------------------------------------------------------
// Soundness
// ---------------------------------------------------------------------------

/// The alpha-optimized multiplicative bound must still ENCLOSE the network.
/// A tighter bound is only correct while it remains an enclosure, so this
/// samples the concrete SwiGLU on a dense grid over the input box and checks
/// every sample against the returned interval.
#[ntest::timeout(60000)]
#[test]
fn mulbinary_alpha_bounds_enclose_sampled_swiglu_evaluations() {
    let (graph, input) = swiglu_fixture();
    let bounds = graph
        .propagate_alpha_crown_with_config(&input, &config(40))
        .expect("alpha route should succeed");

    let lo = [input.lower()[[0]], input.lower()[[1]]];
    let hi = [input.upper()[[0]], input.upper()[[1]]];
    const STEPS: usize = 40;
    // f32 evaluation slack only — NOT bound slack. The enclosure claim itself
    // is exact; this absorbs the f32 round-off of the oracle's own arithmetic.
    const TOL: f32 = 1e-5;
    let mut widest_lower_slack = f32::INFINITY;
    let mut widest_upper_slack = f32::INFINITY;
    for i in 0..=STEPS {
        for j in 0..=STEPS {
            let x = [
                lo[0] + (hi[0] - lo[0]) * i as f32 / STEPS as f32,
                lo[1] + (hi[1] - lo[1]) * j as f32 / STEPS as f32,
            ];
            let y = swiglu_reference(x);
            for (row, &value) in y.iter().enumerate() {
                let l = bounds.lower()[[row]];
                let u = bounds.upper()[[row]];
                assert!(
                    value >= l - TOL,
                    "sample {value} at x={x:?} fell below the alpha lower bound {l} (row {row})"
                );
                assert!(
                    value <= u + TOL,
                    "sample {value} at x={x:?} rose above the alpha upper bound {u} (row {row})"
                );
                widest_lower_slack = widest_lower_slack.min(value - l);
                widest_upper_slack = widest_upper_slack.min(u - value);
            }
        }
    }
    // Guard against a vacuous pass: the grid must actually approach both faces,
    // otherwise "enclosed" would hold for an arbitrarily loose box.
    assert!(
        widest_lower_slack.is_finite() && widest_upper_slack.is_finite(),
        "the sample grid produced no finite comparison"
    );
}

/// The bare product has an exactly computable range, so its alpha bound can be
/// checked against the true reachable set rather than against a sampling of it.
#[ntest::timeout(60000)]
#[test]
fn mulbinary_alpha_bounds_enclose_sampled_bare_product_evaluations() {
    let (graph, input) = bare_product_fixture();
    let bounds = graph
        .propagate_alpha_crown_with_config(&input, &config(40))
        .expect("alpha route should succeed");

    const STEPS: usize = 60;
    const TOL: f32 = 1e-5;
    for i in 0..=STEPS {
        for j in 0..=STEPS {
            let x = [
                -0.5 + i as f32 / STEPS as f32,
                -0.5 + j as f32 / STEPS as f32,
            ];
            for row in 0..2 {
                let left = UP_W[row][0] * x[0] + UP_W[row][1] * x[1] + UP_B[row];
                let right = GATE_W[row][0] * x[0] + GATE_W[row][1] * x[1] + GATE_B[row];
                let value = left * right;
                let l = bounds.lower()[[row]];
                let u = bounds.upper()[[row]];
                assert!(
                    value >= l - TOL && value <= u + TOL,
                    "product sample {value} at x={x:?} escaped [{l}, {u}] (row {row})"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The non-regression safety net itself
// ---------------------------------------------------------------------------

/// The re-routed arm must CAPTURE the previous route's bound as an elementwise
/// floor, which is what makes "never looser than the batched route" structural
/// rather than fixture-specific. On these fixtures the optimizer dominates the
/// floor anyway, so no bound comparison can witness the floor being taken —
/// assert the router's own record of capturing it instead.
#[ntest::timeout(60000)]
#[test]
fn mulbinary_reroute_captures_the_previous_route_bound_as_a_floor() {
    for (name, (graph, input)) in [
        ("bare product", bare_product_fixture()),
        ("swiglu", swiglu_fixture()),
        ("stable relu product", stable_relu_product_fixture()),
    ] {
        route_probe::reset();
        graph
            .propagate_alpha_crown_with_config(&input, &config(3))
            .expect("alpha route should succeed");
        assert_eq!(
            route_probe::last(),
            Some(DagAlphaRoute::OptimizerLoop),
            "{name}: fixture must exercise the re-routed arm"
        );
        // The stable-ReLU fixture re-routes at the SECOND gate, whose fallback
        // is the very CROWN bound the optimizer already seeds from — no floor
        // is needed or captured there.
        let expected_floor = name != "stable relu product";
        assert_eq!(
            route_probe::floor_captured(),
            expected_floor,
            "{name}: unexpected batched-floor capture"
        );
    }
}

/// Routes that did not change must not start paying for a floor.
#[ntest::timeout(60000)]
#[test]
fn mulbinary_router_captures_no_floor_on_unchanged_routes() {
    let (graph, input) = no_optimizable_fixture();
    route_probe::reset();
    graph
        .propagate_alpha_crown_with_config(&input, &config(3))
        .expect("alpha-CROWN should succeed");
    assert_eq!(route_probe::last(), Some(DagAlphaRoute::BatchedEarlyReturn));
    assert!(
        !route_probe::floor_captured(),
        "an unchanged route must not pay for a floor"
    );
}

/// The SECOND gate's replaced answer is fixed-slope CROWN, so its re-routed
/// arm carries the same contract: never looser than what it used to return.
#[ntest::timeout(60000)]
#[test]
fn mulbinary_stable_relu_route_is_never_looser_than_fixed_slope_crown() {
    let (graph, input) = stable_relu_product_fixture();
    let cfg = config(20);
    // Exactly the bound the second gate returned before the fix.
    let crown = graph
        .propagate_crown_with_engine_and_deadline(&input, None, cfg.deadline)
        .expect("fixed-slope CROWN should succeed")
        .bounds;
    route_probe::reset();
    let alpha = graph
        .propagate_alpha_crown_with_config(&input, &cfg)
        .expect("alpha route should succeed");
    assert_eq!(
        route_probe::last(),
        Some(DagAlphaRoute::OptimizerLoop),
        "fixture must exercise the re-routed second gate"
    );
    assert_no_looser(&alpha, &crown, "stable relu product");
}

/// And that arm's bound must still enclose the concrete network.
#[ntest::timeout(60000)]
#[test]
fn mulbinary_stable_relu_bounds_enclose_sampled_evaluations() {
    let (graph, input) = stable_relu_product_fixture();
    let bounds = graph
        .propagate_alpha_crown_with_config(&input, &config(40))
        .expect("alpha route should succeed");

    const PRE_W: [[f32; 2]; 2] = [[0.5, 0.25], [0.25, 0.5]];
    const PRE_B: [f32; 2] = [2.0, 2.0];
    const STEPS: usize = 50;
    const TOL: f32 = 1e-5;
    for i in 0..=STEPS {
        for j in 0..=STEPS {
            let x = [
                0.5 + 0.5 * i as f32 / STEPS as f32,
                0.5 + 0.5 * j as f32 / STEPS as f32,
            ];
            for row in 0..2 {
                let pre = PRE_W[row][0] * x[0] + PRE_W[row][1] * x[1] + PRE_B[row];
                let relu = pre.max(0.0);
                let other = GATE_W[row][0] * x[0] + GATE_W[row][1] * x[1] + GATE_B[row];
                let value = relu * other;
                let l = bounds.lower()[[row]];
                let u = bounds.upper()[[row]];
                assert!(
                    value >= l - TOL && value <= u + TOL,
                    "sample {value} at x={x:?} escaped [{l}, {u}] (row {row})"
                );
            }
        }
    }
}
