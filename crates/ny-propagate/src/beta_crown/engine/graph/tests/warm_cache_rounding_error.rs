// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #warm-cache-err regressions: a `CachedLinearBounds` block must never be
//! tighter than the certified bound it was captured from.
//!
//! A backward block `(A, b)` carries a certified per-coefficient error `E`
//! (`LinearBounds::{lower,upper}_a_err`, the f32 rounding enclosure of the
//! Linear backward's `A·W`). The true coefficient lies in `[A-E, A+E]`, so
//! the block bounds the objective only once `Σ_j E_ij·max|y_j|` is charged.
//! `CachedLinearBounds` stores `A` and `b` only. Every capture site that
//! publishes into it therefore has to fold `E` into `b` over the node's
//! certified box first, or refuse the entry; dropping `E` publishes a block
//! that is tighter than the certified one, and a warm-started child seeded
//! with it can close on a bound that is not certified.
//!
//! The fixtures make the dropped error an actual soundness violation, not
//! only a missing certificate. The objective row `c = [1, 1]` through
//! `W2 = [[-1, 1, ..], [-2^-26, -2^-27, ..]]` has true ReLU-output
//! coefficients `[-(1 + 2^-26), 1 - 2^-27]`, which round to `[-1, 1]` in f32.
//! With zero biases no outward bias nudge hides the gap. The upper-mode row
//! `c = [-1, -1]` mirrors it.
//!
//! Three kinds of test:
//!
//! * CAPTURE observables (`twin_relu_head`, `y0 = y1 = relu(x)`): every
//!   capture path (root spec-guided CROWN without a deadline, the constrained
//!   backward, the scalar-objective / dense-spec / per-domain-objective
//!   batched cores) must publish blocks whose lower plane lies below, and
//!   upper plane above, the exact objective at every witness point of the
//!   captured region. Checked in exact f64 arithmetic.
//! * SEED-SITE verdicts and END-TO-END BaB (`interval_head`). The same
//!   coefficient-error channel also carries the weight radius of an
//!   `IntervalConst` layer (`out = w·relu(x0)` for EVERY `w ∈ [0.5, 1.5]`,
//!   `layers/interval_const`), so a dropped error there is a whole weight
//!   interval, not an ulp. A child split on `x0` active, warm-started from a
//!   parent's cache, must enclose the true range over all `w` and must not
//!   clear a threshold the property violates; the BaB lanes must never report
//!   `Verified` on it. (With the pure-rounding fixture the seed sites' own
//!   certified f32 concretization charges, `~n·u·Σ|A||x|`, exceeded the
//!   dropped `<= u/2` coefficient rounding in every arrangement tried, so the
//!   rounding case is pinned at the capture boundary above, exactly.)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{arr1, arr2, Array2};
use ny_core::NaiveCpuGemmEngine;

use crate::batched_domain::{BatchedDomains, CachedLinearBounds};
use crate::beta_crown::domain::GraphCrownContext;
use crate::beta_crown::engine::graph::propagation::BatchedBackwardContext;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::state::GraphBetaState;
use crate::beta_crown::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, GraphBabDomain, GraphNeuronConstraint,
    GraphSplitHistory,
};
use crate::layers::IntervalConstLayer;
use crate::network::SpecCrownRequest;
use crate::{BoundedTensor, GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer, NETWORK_INPUT};

/// 2^-26: the rounding the stored `-1` hides in coefficient 0.
const D0: f32 = 1.0 / 67_108_864.0;
/// 2^-27: the rounding the stored `1` hides in coefficient 1.
const D1: f32 = 1.0 / 134_217_728.0;
/// `interval_head` weight enclosure: `w ∈ [W_MID - W_RAD, W_MID + W_RAD]`.
const W_MID: f32 = 1.0;
const W_RAD: f32 = 0.5;
/// A threshold between the worst member (`1.5`) and the midpoint (`1.0`).
const T_MID: f32 = 1.25;

/// Lower-mode row (`c·out > t` is proved by the lower bound).
const LOWER_ROW: [f32; 2] = [1.0, 1.0];
/// Upper-mode row (`c·out < t` is proved by the upper bound).
const UPPER_ROW: [f32; 2] = [-1.0, -1.0];

/// Witness inputs, all inside the unit box.
const WITNESSES: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];

struct Fixture {
    graph: GraphNetwork,
    input: BoundedTensor,
}

fn node(name: &str, layer: Layer, inputs: &[&str]) -> GraphNode {
    GraphNode::new(name, layer, inputs.iter().map(|s| s.to_string()).collect())
}

fn linear(w: Array2<f32>) -> Layer {
    Layer::Linear(LinearLayer::new(w, None).expect("valid linear"))
}

/// `z = [x, x]`, `y = relu(z)`, `out = W2·y` with
/// `W2 = [[-1, 1], [-2^-26, -2^-27]]`, `x ∈ [-1, 1]`. No biases anywhere.
fn twin_relu_head() -> Fixture {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        linear(arr2(&[[1.0f32], [1.0]])),
    ));
    graph.add_node(node("relu1", Layer::ReLU(ReLULayer), &["linear1"]));
    graph.add_node(node(
        "linear2",
        linear(arr2(&[[-1.0f32, 1.0], [-D0, -D1]])),
        &["relu1"],
    ));
    graph.set_output("linear2");
    let input =
        BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).expect("box");
    Fixture { graph, input }
}

/// `y = relu(x)` on the input, `out = w·y0 + 0·y1` through an
/// `IntervalConst` layer with `w ∈ [0.5, 1.5]`, `x ∈ [-1, 1]^2` (`x1` is a
/// zero-weight unstable neuron, so a BaB lane has a second split to make).
fn interval_head() -> Fixture {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu1", Layer::ReLU(ReLULayer)));
    let center = LinearLayer::new(arr2(&[[W_MID, 0.0]]), None).expect("center");
    let layer =
        IntervalConstLayer::try_new_linear(center, arr2(&[[W_RAD, 0.0]])).expect("interval weight");
    graph.add_node(node("iconst", Layer::IntervalConst(layer), &["relu1"]));
    graph.set_output("iconst");
    let input = BoundedTensor::new(
        arr1(&[-1.0f32, -1.0]).into_dyn(),
        arr1(&[1.0f32, 1.0]).into_dyn(),
    )
    .expect("box");
    Fixture { graph, input }
}

/// Exact (f64) value of `row · out` for `twin_relu_head` at input `x`. Every
/// product and sum is exact in f64: the operands are ±1, ±2^-26, ±2^-27 and
/// a dyadic `x`.
fn twin_objective_at(row: &[f32; 2], x: f64) -> f64 {
    let y = x.max(0.0);
    let out0 = -y + y;
    let out1 = -(D0 as f64) * y - (D1 as f64) * y;
    row[0] as f64 * out0 + row[1] as f64 * out1
}

/// Exact value of the `twin_relu_head` node a cached block's columns
/// multiply, at input `x`.
fn twin_node_value(name: &str, x: f64) -> Option<Vec<f64>> {
    let y = x.max(0.0);
    match name {
        NETWORK_INPUT => Some(vec![x]),
        "linear1" => Some(vec![x, x]),
        "relu1" => Some(vec![y, y]),
        "linear2" => Some(vec![-y + y, -(D0 as f64) * y - (D1 as f64) * y]),
        _ => None,
    }
}

fn plane(coeffs: ndarray::ArrayView1<'_, f32>, bias: f32, at: &[f64]) -> f64 {
    coeffs
        .iter()
        .zip(at)
        .fold(bias as f64, |acc, (&a, &v)| acc + a as f64 * v)
}

/// Every published block must enclose the objective at every witness point:
/// `lower(y(x)) <= f(x) <= upper(y(x))`, row by row. Also refuses a vacuous
/// run: the ReLU-output block (the warm-start seed) must be published.
fn assert_cache_encloses(site: &str, cache: &CachedLinearBounds, rows: &[[f32; 2]]) {
    let mut violations = Vec::new();
    for (name, lower_a) in &cache.lower_a {
        let upper_a = &cache.upper_a[name];
        let lower_b = &cache.lower_b[name];
        let upper_b = &cache.upper_b[name];
        assert_eq!(
            lower_a.nrows(),
            rows.len(),
            "{site}: block '{name}' has {} rows, expected {}",
            lower_a.nrows(),
            rows.len()
        );
        for &x in &WITNESSES {
            let at = twin_node_value(name, x)
                .unwrap_or_else(|| panic!("{site}: unexpected cached node '{name}'"));
            for (row_idx, row) in rows.iter().enumerate() {
                let truth = twin_objective_at(row, x);
                let lo = plane(lower_a.row(row_idx), lower_b[row_idx], &at);
                let hi = plane(upper_a.row(row_idx), upper_b[row_idx], &at);
                if lo > truth {
                    violations.push(format!(
                        "node={name} row={row_idx} x={x}: lower plane {lo:e} > f {truth:e}"
                    ));
                }
                if hi < truth {
                    violations.push(format!(
                        "node={name} row={row_idx} x={x}: upper plane {hi:e} < f {truth:e}"
                    ));
                }
            }
        }
    }
    let mut nodes: Vec<&String> = cache.lower_a.keys().collect();
    nodes.sort();
    println!(
        "[warm-err] site={site} nodes={nodes:?} violations={}",
        violations.len()
    );
    assert!(
        cache.lower_a.contains_key("relu1"),
        "{site}: no ReLU-output block was published ({nodes:?}); the capture path is \
         dark or refused, so this test proves nothing"
    );
    assert!(
        violations.is_empty(),
        "{site}: a cached block is TIGHTER than the certified bound it was captured \
         from (dropped coefficient error): {violations:#?}"
    );
}

fn node_bounds(fx: &Fixture) -> HashMap<String, BoundedTensor> {
    fx.graph
        .collect_node_bounds(&fx.input)
        .expect("node bounds")
}

fn arc_node_bounds(fx: &Fixture) -> HashMap<String, Arc<BoundedTensor>> {
    node_bounds(fx)
        .into_iter()
        .map(|(name, bounds)| (name, Arc::new(bounds)))
        .collect()
}

fn warm_config() -> BetaCrownConfig {
    BetaCrownConfig {
        enable_la_warm_start: true,
        enable_cuts: false,
        ..Default::default()
    }
}

fn root_domain(fx: &Fixture) -> GraphBabDomain {
    GraphBabDomain::root(node_bounds(fx), -100.0, 100.0, &fx.input, false).expect("root")
}

fn split(neuron_idx: usize) -> GraphNeuronConstraint {
    GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx,
        is_active: true,
        score: 0.0,
    }
}

/// Neuron 0 split active; last branch node `relu1`.
fn active_child(fx: &Fixture) -> GraphBabDomain {
    let child = root_domain(fx)
        .with_constraint(&fx.graph, split(0), false)
        .expect("with_constraint")
        .expect("feasible");
    assert_eq!(child.history().last_branch_node(), Some("relu1"));
    child
}

fn batched_context_run<T>(
    domain: &GraphBabDomain,
    run: impl FnOnce(&BatchedBackwardContext) -> T,
) -> T {
    let domains = vec![domain];
    let layer_names = vec!["relu1".to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched");
    let ctx = BatchedBackwardContext::from_domains(&domains, &batched).expect("context");
    run(&ctx)
}

// ---------------------------------------------------------------------------
// Capture paths (all on the root domain)
// ---------------------------------------------------------------------------

/// Root spec-guided CROWN capture (`spec_propagation/core.rs`
/// `capture_node_linear_bounds`, `fallback.rs` `capture_final_linear_bounds`).
fn root_spec_cache(fx: &Fixture, row: &[f32], deadline: Option<Instant>) -> CachedLinearBounds {
    let bounds = node_bounds(fx);
    let spec = Array2::from_shape_vec((1, row.len()), row.to_vec()).expect("spec row");
    let (_out, cache) = SpecCrownRequest::new(&fx.graph, &fx.input, &spec, None)
        .node_bounds(&bounds)
        .deadline_opt(deadline)
        .run_with_backward_cache()
        .expect("root spec-guided CROWN");
    cache.expect("root spec-guided CROWN published no cache")
}

/// Constrained backward capture (`constraints/backward/dispatch.rs`
/// `capture_constrained_linear_bounds`, published by `finalize.rs`).
fn constrained_cache(fx: &Fixture, row: &[f32]) -> CachedLinearBounds {
    let bounds = arc_node_bounds(fx);
    let history = GraphSplitHistory::new();
    let beta_state = GraphBetaState::from_history(&history).expect("beta state");
    let context = GraphCrownContext::new(&history, None, Some(&bounds), None);
    let (_out, _cache, captured) = BetaCrownVerifier::new(warm_config())
        .propagate_crown_with_graph_beta_and_cache(
            &fx.graph,
            &fx.input,
            &context,
            &beta_state,
            Some(row),
            None,
            true,
        )
        .expect("constrained pass");
    captured.expect("constrained pass published no cache")
}

/// Scalar-objective batched core capture (`propagation/batched.rs`, the
/// DomainList lane; published by `gpu_bab/batched_gpu.rs`).
fn batched_cache(fx: &Fixture, row: &[f32]) -> CachedLinearBounds {
    let root = root_domain(fx);
    let result = batched_context_run(&root, |ctx| {
        BetaCrownVerifier::new(warm_config())
            .propagate_crown_batched_with_context_capture_la(
                &fx.graph,
                ctx,
                row,
                &NaiveCpuGemmEngine,
            )
            .expect("batched pass")
    });
    let mut maps = result
        .intermediate_la
        .expect("batched pass captured nothing");
    CachedLinearBounds::from_linear_bounds_map(maps.remove(0))
}

/// Dense-spec batched core capture, both rows in one pass.
fn dense_spec_cache(fx: &Fixture, spec: &Array2<f32>) -> CachedLinearBounds {
    let root = root_domain(fx);
    let result = batched_context_run(&root, |ctx| {
        BetaCrownVerifier::new(warm_config())
            .propagate_crown_batched_with_context_specs_capture_la(
                &fx.graph,
                ctx,
                spec,
                &NaiveCpuGemmEngine,
            )
            .expect("dense-spec pass")
    });
    let mut maps = result
        .intermediate_la
        .expect("dense-spec pass captured nothing");
    CachedLinearBounds::from_linear_bounds_map(maps.remove(0))
}

/// Per-domain-objective batched core capture (`multi_objective/per_disjunct_eval.rs`).
fn per_domain_objective_caches(fx: &Fixture) -> Vec<CachedLinearBounds> {
    let bounds = arc_node_bounds(fx);
    let plan = fx.graph.dispatch_plan().expect("plan");
    let result = BetaCrownVerifier::new(warm_config())
        .propagate_crown_batched_backward_core_per_domain_obj(
            &fx.graph,
            2,
            plan,
            &[bounds.clone(), bounds],
            &[fx.input.clone(), fx.input.clone()],
            &[None, None],
            &[None, None],
            &[LOWER_ROW.to_vec(), UPPER_ROW.to_vec()],
            &NaiveCpuGemmEngine,
            true,
            None,
        )
        .expect("per-domain-objective pass");
    result
        .intermediate_la
        .expect("per-domain-objective pass captured nothing")
        .into_iter()
        .map(CachedLinearBounds::from_linear_bounds_map)
        .collect()
}

#[test]
fn warm_err_root_spec_capture_charges_coefficient_error() {
    let fx = twin_relu_head();
    for row in [LOWER_ROW, UPPER_ROW] {
        // Control: the finite-deadline capture already folds (core.rs); it
        // must pass on every tree, which calibrates the witness check.
        let far = Instant::now() + Duration::from_secs(600);
        assert_cache_encloses(
            &format!("root_spec/deadline row={row:?}"),
            &root_spec_cache(&fx, &row, Some(far)),
            &[row],
        );
        assert_cache_encloses(
            &format!("root_spec/no_deadline row={row:?}"),
            &root_spec_cache(&fx, &row, None),
            &[row],
        );
    }
    println!("[warm-err] TEST_COMPLETE name=warm_err_root_spec_capture");
}

#[test]
fn warm_err_constrained_capture_charges_coefficient_error() {
    let fx = twin_relu_head();
    for row in [LOWER_ROW, UPPER_ROW] {
        assert_cache_encloses(
            &format!("constrained row={row:?}"),
            &constrained_cache(&fx, &row),
            &[row],
        );
    }
    println!("[warm-err] TEST_COMPLETE name=warm_err_constrained_capture");
}

#[test]
fn warm_err_batched_capture_charges_coefficient_error() {
    let fx = twin_relu_head();
    for row in [LOWER_ROW, UPPER_ROW] {
        assert_cache_encloses(
            &format!("batched row={row:?}"),
            &batched_cache(&fx, &row),
            &[row],
        );
    }
    println!("[warm-err] TEST_COMPLETE name=warm_err_batched_capture");
}

#[test]
fn warm_err_dense_spec_capture_charges_coefficient_error() {
    let fx = twin_relu_head();
    assert_cache_encloses(
        "dense_spec",
        &dense_spec_cache(&fx, &arr2(&[LOWER_ROW, UPPER_ROW])),
        &[LOWER_ROW, UPPER_ROW],
    );
    println!("[warm-err] TEST_COMPLETE name=warm_err_dense_spec_capture");
}

#[test]
fn warm_err_per_domain_objective_capture_charges_coefficient_error() {
    let fx = twin_relu_head();
    let caches = per_domain_objective_caches(&fx);
    assert_eq!(caches.len(), 2);
    assert_cache_encloses("per_domain_obj row=LOWER", &caches[0], &[LOWER_ROW]);
    assert_cache_encloses("per_domain_obj row=UPPER", &caches[1], &[UPPER_ROW]);
    println!("[warm-err] TEST_COMPLETE name=warm_err_per_domain_objective_capture");
}

// ---------------------------------------------------------------------------
// Scaled boxes: the fold must be charged over the node's OWN box
// ---------------------------------------------------------------------------

/// Input half-width `2^-8`.
const SCALED_S: f32 = 1.0 / 256.0;
/// First-layer gain `2^10`, so `z = K·x ∈ [-4, 4]^2`.
const SCALED_K: f32 = 1024.0;

/// `z = K·x` over TWO independent inputs, `y = relu(z)`, `out = W2·y` with
/// `twin_relu_head`'s `W2`; `x ∈ [-2^-8, 2^-8]^2`. No biases.
///
/// In `twin_relu_head` every box a block could be folded over has magnitude
/// 1 and the input box has length 1, so the fold was only ever checked for
/// "some box of this length": a fold over the WRONG box of the right length
/// charged exactly the same. Here the network-input box has the SAME length
/// as the ReLU-output box (2) but 1/1024 of its magnitude (`2^-8` vs `4`), so
/// a relu1 block folded over the input box instead of `[0, 4]^2` charges
/// `Σ E_j·2^-8` where `Σ E_j·4` is owed, and its lower plane rises above the
/// objective by up to `(2^-26 + 2^-27)·4` at `y = (4, 4)`.
fn scaled_twin_relu_head() -> Fixture {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        linear(arr2(&[[SCALED_K, 0.0], [0.0, SCALED_K]])),
    ));
    graph.add_node(node("relu1", Layer::ReLU(ReLULayer), &["linear1"]));
    graph.add_node(node(
        "linear2",
        linear(arr2(&[[-1.0f32, 1.0], [-D0, -D1]])),
        &["relu1"],
    ));
    graph.set_output("linear2");
    let input = BoundedTensor::new(
        arr1(&[-SCALED_S, -SCALED_S]).into_dyn(),
        arr1(&[SCALED_S, SCALED_S]).into_dyn(),
    )
    .expect("box");
    Fixture { graph, input }
}

/// Witness inputs: the 5x5 dyadic grid over `[-2^-8, 2^-8]^2`.
fn scaled_witnesses() -> Vec<[f64; 2]> {
    let s = f64::from(SCALED_S);
    let axis = [-s, -s / 2.0, 0.0, s / 2.0, s];
    axis.iter()
        .flat_map(|&a| axis.iter().map(move |&b| [a, b]))
        .collect()
}

/// Exact (f64) value of the node a `scaled_twin_relu_head` block's columns
/// multiply, at input `x`. `K·x` is a power-of-two scaling of a dyadic, and
/// every other operand is `±1`, `2^-26` or `2^-27`, so each value is exact.
fn scaled_node_value(name: &str, x: &[f64; 2]) -> Option<Vec<f64>> {
    let k = f64::from(SCALED_K);
    let z = [k * x[0], k * x[1]];
    let y = [z[0].max(0.0), z[1].max(0.0)];
    match name {
        NETWORK_INPUT => Some(x.to_vec()),
        "linear1" => Some(z.to_vec()),
        "relu1" => Some(y.to_vec()),
        "linear2" => Some(vec![
            -y[0] + y[1],
            -f64::from(D0) * y[0] - f64::from(D1) * y[1],
        ]),
        _ => None,
    }
}

/// `assert_cache_encloses` for `scaled_twin_relu_head`, over its 2-D witnesses.
fn assert_scaled_cache_encloses(site: &str, cache: &CachedLinearBounds, rows: &[[f32; 2]]) {
    let mut violations = Vec::new();
    for (name, lower_a) in &cache.lower_a {
        let upper_a = &cache.upper_a[name];
        let lower_b = &cache.lower_b[name];
        let upper_b = &cache.upper_b[name];
        assert_eq!(lower_a.nrows(), rows.len(), "{site}: block '{name}' rows");
        for x in scaled_witnesses() {
            let at = scaled_node_value(name, &x)
                .unwrap_or_else(|| panic!("{site}: unexpected cached node '{name}'"));
            let out = scaled_node_value("linear2", &x).expect("output value");
            for (row_idx, row) in rows.iter().enumerate() {
                let truth = f64::from(row[0]) * out[0] + f64::from(row[1]) * out[1];
                let lo = plane(lower_a.row(row_idx), lower_b[row_idx], &at);
                let hi = plane(upper_a.row(row_idx), upper_b[row_idx], &at);
                if lo > truth {
                    violations.push(format!(
                        "node={name} row={row_idx} x={x:?}: lower plane {lo:e} > f {truth:e}"
                    ));
                }
                if hi < truth {
                    violations.push(format!(
                        "node={name} row={row_idx} x={x:?}: upper plane {hi:e} < f {truth:e}"
                    ));
                }
            }
        }
    }
    let mut nodes: Vec<&String> = cache.lower_a.keys().collect();
    nodes.sort();
    println!(
        "[warm-err] scaled site={site} nodes={nodes:?} violations={}",
        violations.len()
    );
    assert!(
        cache.lower_a.contains_key("relu1"),
        "{site}: no ReLU-output block was published ({nodes:?}); the capture path is \
         dark or refused, so this test proves nothing"
    );
    assert!(
        violations.is_empty(),
        "{site}: a cached block is TIGHTER than the certified bound it was captured \
         from (coefficient error folded over the wrong box): {violations:#?}"
    );
}

/// Every capture path, on the scaled fixture. GREEN on a correct tree; RED
/// against a fold over the network-input box (same length, 1/1024 of the
/// magnitude) at the constrained capture site, which `twin_relu_head` cannot
/// see as a wrong box (its input box has length 1, so there the mutant only
/// surfaces as a refused block).
#[test]
fn warm_err_scaled_head_folds_over_the_node_box() {
    let fx = scaled_twin_relu_head();
    let far = Instant::now() + Duration::from_secs(600);
    for row in [LOWER_ROW, UPPER_ROW] {
        assert_scaled_cache_encloses(
            &format!("root_spec/deadline row={row:?}"),
            &root_spec_cache(&fx, &row, Some(far)),
            &[row],
        );
        assert_scaled_cache_encloses(
            &format!("root_spec/no_deadline row={row:?}"),
            &root_spec_cache(&fx, &row, None),
            &[row],
        );
        assert_scaled_cache_encloses(
            &format!("constrained row={row:?}"),
            &constrained_cache(&fx, &row),
            &[row],
        );
        assert_scaled_cache_encloses(
            &format!("batched row={row:?}"),
            &batched_cache(&fx, &row),
            &[row],
        );
    }
    assert_scaled_cache_encloses(
        "dense_spec",
        &dense_spec_cache(&fx, &arr2(&[LOWER_ROW, UPPER_ROW])),
        &[LOWER_ROW, UPPER_ROW],
    );
    let caches = per_domain_objective_caches(&fx);
    assert_eq!(caches.len(), 2);
    assert_scaled_cache_encloses("per_domain_obj row=LOWER", &caches[0], &[LOWER_ROW]);
    assert_scaled_cache_encloses("per_domain_obj row=UPPER", &caches[1], &[UPPER_ROW]);
    println!("[warm-err] TEST_COMPLETE name=warm_err_scaled_head_folds_over_the_node_box");
}

// ---------------------------------------------------------------------------
// Seed sites: a warm-started child never clears a violated threshold
// ---------------------------------------------------------------------------

/// `interval_head` rows: lower mode proves `-out > t`, upper mode `out < t`.
const ILOWER: [f32; 1] = [-1.0];
const IUPPER: [f32; 1] = [1.0];

/// Exact range of `row·out` over the active child's region
/// (`x0 ∈ [0, 1]`) and over EVERY `w ∈ [0.5, 1.5]`.
fn interval_child_range(row: &[f32]) -> (f64, f64) {
    let worst = (W_MID + W_RAD) as f64;
    if row[0] < 0.0 {
        (-worst, 0.0)
    } else {
        (0.0, worst)
    }
}

/// The child's bound must enclose its true range, and must not certify the
/// violated property: `-out > -1.25` in lower mode (true min `-1.5`),
/// `out < 1.25` in upper mode (true max `1.5`).
fn interval_child_violations(site: &str, row: &[f32], lower: f32, upper: f32) -> Vec<String> {
    let (lo, hi) = interval_child_range(row);
    println!(
        "[warm-err] SEED site={site} row={row:?} child=[{lower:e}, {upper:e}] truth=[{lo:e}, {hi:e}]"
    );
    let mut out = Vec::new();
    if (lower as f64) > lo {
        out.push(format!(
            "{site} row={row:?}: child lower {lower:e} exceeds the true min {lo:e}"
        ));
    }
    if (upper as f64) < hi {
        out.push(format!(
            "{site} row={row:?}: child upper {upper:e} is below the true max {hi:e}"
        ));
    }
    if row[0] < 0.0 && lower > -T_MID {
        out.push(format!(
            "{site}: lower {lower:e} > {:e} would VERIFY `-out > -1.25`, which is FALSE \
             (min {lo:e})",
            -T_MID
        ));
    }
    if row[0] > 0.0 && upper < T_MID {
        out.push(format!(
            "{site}: upper {upper:e} < {T_MID:e} would VERIFY `out < 1.25`, which is FALSE \
             (max {hi:e})"
        ));
    }
    out
}

/// Seed the active child from `cache` through the scalar-objective batched
/// core (`WithLaCapture`, the DomainList lane's seed site).
fn batched_seeded_child(fx: &Fixture, row: &[f32], cache: CachedLinearBounds) -> (f32, f32) {
    let mut child = active_child(fx);
    child.cached_la = Some(Arc::new(cache));
    let result = batched_context_run(&child, |ctx| {
        BetaCrownVerifier::new(warm_config())
            .propagate_crown_batched_with_context_capture_la(
                &fx.graph,
                ctx,
                row,
                &NaiveCpuGemmEngine,
            )
            .expect("seeded batched pass")
    });
    let flat = result.results[0].0.flatten();
    (flat.lower()[[0]], flat.upper()[[0]])
}

/// Seed the active child from `cache` through the constrained backward
/// (`constraints/backward/setup.rs` seed site).
fn constrained_seeded_child(fx: &Fixture, row: &[f32], cache: &CachedLinearBounds) -> (f32, f32) {
    let child = active_child(fx);
    let bounds: HashMap<String, Arc<BoundedTensor>> = child.node_bounds.clone();
    let history = child.history().clone();
    let beta_state = GraphBetaState::from_history(&history).expect("beta state");
    let context = GraphCrownContext::new(&history, None, Some(&bounds), None);
    let (out, _, _) = BetaCrownVerifier::new(warm_config())
        .propagate_crown_with_graph_beta_and_cache(
            &fx.graph,
            &fx.input,
            &context,
            &beta_state,
            Some(row),
            Some(cache),
            false,
        )
        .expect("seeded constrained pass");
    let flat = out.flatten();
    (flat.lower()[[0]], flat.upper()[[0]])
}

/// Seed the active child through the dense-spec batched core.
fn dense_spec_seeded_child(
    fx: &Fixture,
    spec: &Array2<f32>,
    cache: CachedLinearBounds,
) -> Vec<(f32, f32)> {
    let mut child = active_child(fx);
    child.cached_la = Some(Arc::new(cache));
    let result = batched_context_run(&child, |ctx| {
        BetaCrownVerifier::new(warm_config())
            .propagate_crown_batched_with_context_specs_capture_la(
                &fx.graph,
                ctx,
                spec,
                &NaiveCpuGemmEngine,
            )
            .expect("seeded dense-spec pass")
    });
    let flat = result.results[0].output_bounds.flatten();
    (0..spec.nrows())
        .map(|i| (flat.lower()[[i]], flat.upper()[[i]]))
        .collect()
}

#[test]
fn warm_err_interval_seeded_child_never_certifies_a_violated_threshold() {
    let fx = interval_head();
    let mut violations = Vec::new();
    for row in [&ILOWER[..], &IUPPER[..]] {
        let cases: [(&str, (f32, f32)); 6] = [
            (
                "batched_site<-batched_capture",
                batched_seeded_child(&fx, row, batched_cache(&fx, row)),
            ),
            (
                "batched_site<-root_spec_no_deadline",
                batched_seeded_child(&fx, row, root_spec_cache(&fx, row, None)),
            ),
            (
                "batched_site<-constrained_capture",
                batched_seeded_child(&fx, row, constrained_cache(&fx, row)),
            ),
            (
                "constrained_site<-constrained_capture",
                constrained_seeded_child(&fx, row, &constrained_cache(&fx, row)),
            ),
            (
                "constrained_site<-batched_capture",
                constrained_seeded_child(&fx, row, &batched_cache(&fx, row)),
            ),
            (
                "constrained_site<-root_spec_no_deadline",
                constrained_seeded_child(&fx, row, &root_spec_cache(&fx, row, None)),
            ),
        ];
        for (site, (lower, upper)) in cases {
            violations.extend(interval_child_violations(site, row, lower, upper));
        }
    }
    let spec = arr2(&[ILOWER, IUPPER]);
    let rows = dense_spec_seeded_child(&fx, &spec, dense_spec_cache(&fx, &spec));
    for (row, (lower, upper)) in [&ILOWER[..], &IUPPER[..]].into_iter().zip(rows) {
        violations.extend(interval_child_violations(
            "dense_spec_site<-dense_spec_capture",
            row,
            lower,
            upper,
        ));
    }
    assert!(
        violations.is_empty(),
        "a warm-started child reported an uncertified bound: {violations:#?}"
    );
    println!("[warm-err] TEST_COMPLETE name=warm_err_interval_seeded_child");
}

// ---------------------------------------------------------------------------
// End-to-end BaB: the violated property is never Verified
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Outcome {
    verified: bool,
    errored: bool,
    explored: usize,
    label: String,
}

fn outcome(result: ny_core::Result<BetaCrownResult>) -> Outcome {
    match result {
        Ok(result) => Outcome {
            verified: matches!(result.result, BabVerificationStatus::Verified),
            errored: false,
            explored: result.domains_explored,
            label: format!(
                "{:?} explored={} max_depth={}",
                result.result, result.domains_explored, result.max_depth_reached
            ),
        },
        Err(error) => Outcome {
            verified: false,
            errored: true,
            explored: 0,
            label: format!("Err({error})"),
        },
    }
}

fn bab_config(warm: bool, batch_size: usize, verify_upper_bound: bool) -> BetaCrownConfig {
    BetaCrownConfig {
        enable_la_warm_start: warm,
        enable_cuts: false,
        verify_upper_bound,
        timeout: Duration::from_secs(30),
        max_domains: 200,
        max_depth: 10,
        batch_size,
        ..Default::default()
    }
}

/// DomainList lane: lower mode proves `row·out > t`, upper mode `row·out < t`.
fn domain_list(
    fx: &Fixture,
    row: &[f32],
    t: f32,
    upper: bool,
    warm: bool,
    engine: bool,
) -> Outcome {
    let naive = NaiveCpuGemmEngine;
    let engine: Option<&dyn ny_core::GemmEngine> = if engine { Some(&naive) } else { None };
    outcome(
        BetaCrownVerifier::new(bab_config(warm, 2, upper))
            .verify_graph_gpu_domain_list(&fx.graph, &fx.input, row, t, engine, None),
    )
}

/// Multi-objective lane in lower mode (`row·out > t`), sequential or batched.
fn multi_objective(fx: &Fixture, row: &[f32], t: f32, warm: bool, batched: bool) -> Outcome {
    let naive = NaiveCpuGemmEngine;
    let engine: Option<&dyn ny_core::GemmEngine> = if batched { Some(&naive) } else { None };
    outcome(
        BetaCrownVerifier::new(bab_config(warm, if batched { 4 } else { 1 }, false))
            .verify_graph_relu_split_multi_objective_with_engine(
                &fx.graph,
                &fx.input,
                &[row.to_vec()],
                &[t],
                engine,
                None,
            ),
    )
}

#[ntest::timeout(300000)]
#[test]
fn warm_err_interval_bab_never_verifies_the_violated_property() {
    let fx = interval_head();
    let mut unsound = Vec::new();
    let mut errored = Vec::new();
    let mut true_verified = 0usize;
    let mut branched = false;
    let mut record = |lane: String, falsy: Outcome, truthy: Outcome| {
        println!("[warm-err] BAB {lane} FALSE: {}", falsy.label);
        println!("[warm-err] BAB {lane} TRUE: {}", truthy.label);
        if falsy.verified {
            unsound.push(format!("{lane}: {}", falsy.label));
        }
        if falsy.errored || truthy.errored {
            errored.push(format!("{lane}: {} / {}", falsy.label, truthy.label));
        }
        branched |= falsy.explored > 1;
        true_verified += usize::from(truthy.verified);
    };
    for warm in [false, true] {
        for engine in [false, true] {
            // Lower mode: `-out > -1.25` is FALSE (min -1.5); `-out > -2` is TRUE.
            record(
                format!("domain_list/lower warm={warm} engine={engine}"),
                domain_list(&fx, &ILOWER, -T_MID, false, warm, engine),
                domain_list(&fx, &ILOWER, -2.0, false, warm, engine),
            );
            // Upper mode: `out < 1.25` is FALSE (max 1.5); `out < 2` is TRUE.
            record(
                format!("domain_list/upper warm={warm} engine={engine}"),
                domain_list(&fx, &IUPPER, T_MID, true, warm, engine),
                domain_list(&fx, &IUPPER, 2.0, true, warm, engine),
            );
        }
        for batched in [false, true] {
            record(
                format!("multi_objective/lower warm={warm} batched={batched}"),
                multi_objective(&fx, &ILOWER, -T_MID, warm, batched),
                multi_objective(&fx, &ILOWER, -2.0, warm, batched),
            );
        }
    }
    assert!(
        unsound.is_empty(),
        "a violated property was reported Verified (unsound): {unsound:#?}"
    );
    assert!(errored.is_empty(), "a lane errored: {errored:#?}");
    assert!(
        true_verified > 0,
        "the TRUE property verified in no lane; the run is vacuous"
    );
    assert!(
        branched,
        "no lane branched on the violated property, so no child was seeded"
    );
    println!("[warm-err] TEST_COMPLETE name=warm_err_interval_bab");
}
