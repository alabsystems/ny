// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #warm-dag regressions: the lA warm start must never seed a child with a
//! block that is not a bound on the objective.
//!
//! A child's backward pass may start at its last branch node `N` from the
//! block its parent held there only when the output reaches `N` through a
//! unary chain (`propagation/warm_start_guard.rs` has the argument). On a
//! residual graph the parent still holds the skip operand's block and the
//! Add's diverted bias when it pops `N`, so a seed at `N` bounds a different
//! function. Before the guard this published `Verified` on FALSE properties in
//! the multi-objective lane (lower mode, sequential and batched) and in the
//! DomainList lane (upper mode).
//!
//! Two kinds of test:
//!
//! * SEED-SITE observables. Each of the three seed sites (constrained
//!   `setup.rs`, the scalar-objective batched core, the dense-spec batched
//!   core) is handed a deliberately WRONG ("tagged") cached block at the branch
//!   node. If the site consumes it, the bound moves far away from the cold
//!   pass; if it refuses, the bound is the cold pass exactly. A chain graph and
//!   a split in a residual net's unary head must CONSUME it (the warm start is
//!   still live where it is sound, so these tests are not vacuous); a split
//!   inside a residual block must REFUSE it.
//! * End-to-end BaB on small nets whose range is known by enumeration: a FALSE
//!   property must never come back `Verified`, in any lane / mode / warm-start
//!   setting, and a TRUE one must (non-vacuity).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use ndarray::{arr1, arr2, Array1, Array2};
use ny_core::NaiveCpuGemmEngine;

use crate::batched_domain::{BatchedDomains, CachedLinearBounds};
use crate::beta_crown::domain::GraphCrownContext;
use crate::beta_crown::engine::graph::propagation::BatchedBackwardContext;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::state::GraphBetaState;
use crate::beta_crown::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BranchingHeuristic, GraphBabDomain,
    GraphNeuronConstraint, GraphSplitHistory,
};
use crate::{
    AddLayer, BoundedTensor, GraphNetwork, GraphNode, Layer, LinearBounds, LinearLayer, ReLULayer,
};

// ---------------------------------------------------------------------------
// Networks (input x in [-1, 1]^2, one scalar output)
// ---------------------------------------------------------------------------

fn linear(w: Array2<f32>, b: Option<Array1<f32>>) -> Layer {
    Layer::Linear(LinearLayer::new(w, b).expect("valid linear"))
}

fn node(name: &str, layer: Layer, inputs: &[&str]) -> GraphNode {
    GraphNode::new(name, layer, inputs.iter().map(|s| s.to_string()).collect())
}

/// `out = Σ_i relu(x_i) + x_i`: the ReLU is inside a residual block, so the
/// output reaches the input around it. Range `[-2, 4]`.
fn residual_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        linear(arr2(&[[1.0, 0.0], [0.0, 1.0]]), None),
    ));
    graph.add_node(node("relu1", Layer::ReLU(ReLULayer), &["linear1"]));
    graph.add_node(node("add1", Layer::Add(AddLayer), &["relu1", "linear1"]));
    graph.add_node(node(
        "linear2",
        linear(arr2(&[[1.0, 1.0]]), None),
        &["add1"],
    ));
    graph.set_output("linear2");
    graph
}

fn residual_out(x: [f32; 2]) -> f32 {
    x[0].max(0.0) + x[0] + x[1].max(0.0) + x[1]
}

/// `out = relu(x0) + relu(x1)`: a pure chain. Range `[0, 2]`.
fn chain_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        linear(arr2(&[[1.0, 0.0], [0.0, 1.0]]), None),
    ));
    graph.add_node(node("relu1", Layer::ReLU(ReLULayer), &["linear1"]));
    graph.add_node(node(
        "linear2",
        linear(arr2(&[[1.0, 1.0]]), None),
        &["relu1"],
    ));
    graph.set_output("linear2");
    graph
}

fn chain_out(x: [f32; 2]) -> f32 {
    x[0].max(0.0) + x[1].max(0.0)
}

/// `out = relu(x0 - 0.25) + relu(x1 + 0.5) - 0.5`: a chain whose blocks carry
/// bias, so a seed must bring the bias accumulated above the branch node.
/// Range `[-0.5, 1.75]`.
fn biased_chain_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        linear(arr2(&[[1.0, 0.0], [0.0, 1.0]]), Some(arr1(&[-0.25, 0.5]))),
    ));
    graph.add_node(node("relu1", Layer::ReLU(ReLULayer), &["linear1"]));
    graph.add_node(node(
        "linear2",
        linear(arr2(&[[1.0, 1.0]]), Some(arr1(&[-0.5]))),
        &["relu1"],
    ));
    graph.set_output("linear2");
    graph
}

fn biased_chain_out(x: [f32; 2]) -> f32 {
    (x[0] - 0.25).max(0.0) + (x[1] + 0.5).max(0.0) - 0.5
}

/// A residual block followed by a unary head:
/// `a = relu(x) + x`, `h = relu(W a + c)`, `out = h0 + h1`.
/// A split on `relu2` (head) may keep its warm start; a split on `relu1`
/// (inside the block) may not.
fn residual_then_head_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        linear(arr2(&[[1.0, 0.0], [0.0, 1.0]]), None),
    ));
    graph.add_node(node("relu1", Layer::ReLU(ReLULayer), &["linear1"]));
    graph.add_node(node("add1", Layer::Add(AddLayer), &["relu1", "linear1"]));
    graph.add_node(node(
        "linear2",
        linear(arr2(&[[1.0, -1.0], [0.5, 1.0]]), Some(arr1(&[0.2, -0.3]))),
        &["add1"],
    ));
    graph.add_node(node("relu2", Layer::ReLU(ReLULayer), &["linear2"]));
    graph.add_node(node(
        "linear3",
        linear(arr2(&[[1.0, 1.0]]), None),
        &["relu2"],
    ));
    graph.set_output("linear3");
    graph
}

fn residual_then_head_out(x: [f32; 2]) -> f32 {
    let a = [x[0].max(0.0) + x[0], x[1].max(0.0) + x[1]];
    let h0 = (a[0] - a[1] + 0.2).max(0.0);
    let h1 = (0.5 * a[0] + a[1] - 0.3).max(0.0);
    h0 + h1
}

fn unit_box() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-1.0f32, -1.0]).into_dyn(),
        arr1(&[1.0f32, 1.0]).into_dyn(),
    )
    .expect("input box")
}

/// Range of `f` over a 201 x 201 grid of the unit box. Every net here is
/// piecewise linear with Lipschitz constant <= 6, so the true max is within
/// 0.06 of the grid max, and the grid max itself is attained.
fn grid_range(f: fn([f32; 2]) -> f32) -> (f32, f32) {
    let n = 200;
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for i in 0..=n {
        for j in 0..=n {
            let x = [
                -1.0 + 2.0 * i as f32 / n as f32,
                -1.0 + 2.0 * j as f32 / n as f32,
            ];
            let v = f(x);
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    (lo, hi)
}

// ---------------------------------------------------------------------------
// Seed-site observables
// ---------------------------------------------------------------------------

/// A deliberately wrong cached block for `branch_node`: an (almost) constant
/// `TAG_BIAS + TAG_COEFF · y`. Consuming it pins the child's interval near
/// `1.25`, which lies INSIDE every child's forward (IBP) output box here and
/// about 1.0 away from a cold interval end. It must stay inside that box: the
/// batched cores intersect CROWN with the forward box (#2904) and fall back on
/// a disjoint pair, so a first version tagged with 25.0 made a CONSUMED seed on
/// the chain read exactly as the cold interval.
const TAG_COEFF: f32 = 0.01;
const TAG_BIAS: f32 = 1.25;

fn tagged_block(rows: usize, width: usize) -> LinearBounds {
    let a = Array2::from_elem((rows, width), TAG_COEFF);
    let b = Array1::from_elem(rows, TAG_BIAS);
    LinearBounds::new(a.clone(), b.clone(), a, b).expect("finite tagged block")
}

fn tagged_cache(branch_node: &str, rows: usize, width: usize) -> CachedLinearBounds {
    let mut map = HashMap::new();
    map.insert(branch_node.to_string(), tagged_block(rows, width));
    CachedLinearBounds::from_linear_bounds_map(map)
}

fn split(branch_node: &str) -> GraphNeuronConstraint {
    GraphNeuronConstraint {
        node_name: branch_node.to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    }
}

fn warm_config(warm: bool) -> BetaCrownConfig {
    BetaCrownConfig {
        enable_la_warm_start: warm,
        enable_cuts: false,
        ..Default::default()
    }
}

/// `(cold, seeded)` output interval of one child whose last split is on
/// `branch_node`, through the constrained backward (`setup.rs` seed site).
fn constrained_cold_and_seeded(
    graph: &GraphNetwork,
    branch_node: &str,
) -> ((f32, f32), (f32, f32)) {
    let input = unit_box();
    let node_bounds: HashMap<String, Arc<BoundedTensor>> = graph
        .collect_node_bounds(&input)
        .expect("node bounds")
        .into_iter()
        .map(|(k, v)| (k, Arc::new(v)))
        .collect();
    let width = node_bounds[branch_node].len();
    let history = GraphSplitHistory::new().with_constraint(split(branch_node));
    assert_eq!(history.last_branch_node(), Some(branch_node));
    let beta_state = GraphBetaState::from_history(&history).expect("beta state");
    let context = GraphCrownContext::new(&history, None, Some(&node_bounds), None);
    let cache = tagged_cache(branch_node, 1, width);
    let run = |seed: Option<&CachedLinearBounds>| {
        let (out, _, _) = BetaCrownVerifier::new(warm_config(true))
            .propagate_crown_with_graph_beta_and_cache(
                graph,
                &input,
                &context,
                &beta_state,
                Some(&[1.0]),
                seed,
                false,
            )
            .expect("constrained pass");
        let flat = out.flatten();
        (flat.lower()[[0]], flat.upper()[[0]])
    };
    (run(None), run(Some(&cache)))
}

fn child_domain(
    graph: &GraphNetwork,
    branch_node: &str,
    cache: CachedLinearBounds,
) -> GraphBabDomain {
    let input = unit_box();
    let bounds = graph.collect_node_bounds(&input).expect("node bounds");
    let root = GraphBabDomain::root(bounds, -100.0, 100.0, &input, false).expect("root domain");
    let mut child = root
        .with_constraint(graph, split(branch_node), false)
        .expect("with_constraint")
        .expect("feasible child");
    child.cached_la = Some(Arc::new(cache));
    child
}

/// `(cold, seeded)` through the scalar-objective batched core
/// (`WithLaCapture`, the DomainList lane's seed site).
fn batched_cold_and_seeded(graph: &GraphNetwork, branch_node: &str) -> ((f32, f32), (f32, f32)) {
    let width = graph.collect_node_bounds(&unit_box()).expect("node bounds")[branch_node].len();
    let child = child_domain(graph, branch_node, tagged_cache(branch_node, 1, width));
    let domains = vec![&child];
    let layer_names = vec![branch_node.to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched");
    let ctx = BatchedBackwardContext::from_domains(&domains, &batched).expect("context");
    let run = |warm: bool| {
        let result = BetaCrownVerifier::new(warm_config(warm))
            .propagate_crown_batched_with_context_capture_la(
                graph,
                &ctx,
                &[1.0],
                &NaiveCpuGemmEngine,
            )
            .expect("batched pass");
        let flat = result.results[0].0.flatten();
        (flat.lower()[[0]], flat.upper()[[0]])
    };
    (run(false), run(true))
}

/// `(cold, seeded)` through the dense-spec batched core (`WithLaCapture`).
fn dense_spec_cold_and_seeded(graph: &GraphNetwork, branch_node: &str) -> ((f32, f32), (f32, f32)) {
    let width = graph.collect_node_bounds(&unit_box()).expect("node bounds")[branch_node].len();
    let child = child_domain(graph, branch_node, tagged_cache(branch_node, 1, width));
    let domains = vec![&child];
    let layer_names = vec![branch_node.to_string()];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched");
    let ctx = BatchedBackwardContext::from_domains(&domains, &batched).expect("context");
    let spec = arr2(&[[1.0f32]]);
    let run = |warm: bool| {
        let result = BetaCrownVerifier::new(warm_config(warm))
            .propagate_crown_batched_with_context_specs_capture_la(
                graph,
                &ctx,
                &spec,
                &NaiveCpuGemmEngine,
            )
            .expect("dense-spec pass");
        let flat = result.results[0].output_bounds.flatten();
        (flat.lower()[[0]], flat.upper()[[0]])
    };
    (run(false), run(true))
}

fn assert_refused(site: &str, graph: &str, cold: (f32, f32), seeded: (f32, f32)) {
    println!("[warm-dag] site={site} graph={graph} cold={cold:?} seeded={seeded:?}");
    assert!(
        cold.0.is_finite() && cold.1.is_finite(),
        "{site}/{graph}: cold pass must be finite, got {cold:?}"
    );
    // A refused seed runs the cold pass itself; the tolerance only absorbs
    // reduction-order noise; a consumed tag block moves an end by ~1.
    assert!(
        (seeded.0 - cold.0).abs() <= 1e-5 && (seeded.1 - cold.1).abs() <= 1e-5,
        "{site}/{graph}: the incomplete warm-start seed was CONSUMED — the seeded pass \
         {seeded:?} differs from the cold pass {cold:?}"
    );
}

fn assert_consumed(site: &str, graph: &str, cold: (f32, f32), seeded: (f32, f32)) {
    println!("[warm-dag] site={site} graph={graph} cold={cold:?} seeded={seeded:?}");
    assert!(
        cold.0.is_finite() && cold.1.is_finite(),
        "{site}/{graph}: cold pass must be finite, got {cold:?}"
    );
    assert!(
        (seeded.0 - cold.0).abs() > 0.9 || (seeded.1 - cold.1).abs() > 0.9,
        "{site}/{graph}: a complete warm-start seed was NOT consumed (seeded {seeded:?} == \
         cold {cold:?}); the seed site is dark and the refusal tests prove nothing"
    );
}

#[test]
fn warm_dag_constrained_seed_site_refuses_residual_and_keeps_chains() {
    let (c, s) = constrained_cold_and_seeded(&residual_graph(), "relu1");
    assert_refused("constrained", "residual/relu1", c, s);
    let (c, s) = constrained_cold_and_seeded(&residual_then_head_graph(), "relu1");
    assert_refused("constrained", "residual_then_head/relu1", c, s);
    let (c, s) = constrained_cold_and_seeded(&chain_graph(), "relu1");
    assert_consumed("constrained", "chain/relu1", c, s);
    let (c, s) = constrained_cold_and_seeded(&residual_then_head_graph(), "relu2");
    assert_consumed("constrained", "residual_then_head/relu2", c, s);
    println!("[warm-dag] TEST_COMPLETE name=warm_dag_constrained_seed_site");
}

#[test]
fn warm_dag_batched_seed_site_refuses_residual_and_keeps_chains() {
    let (c, s) = batched_cold_and_seeded(&residual_graph(), "relu1");
    assert_refused("batched", "residual/relu1", c, s);
    let (c, s) = batched_cold_and_seeded(&residual_then_head_graph(), "relu1");
    assert_refused("batched", "residual_then_head/relu1", c, s);
    let (c, s) = batched_cold_and_seeded(&chain_graph(), "relu1");
    assert_consumed("batched", "chain/relu1", c, s);
    let (c, s) = batched_cold_and_seeded(&residual_then_head_graph(), "relu2");
    assert_consumed("batched", "residual_then_head/relu2", c, s);
    println!("[warm-dag] TEST_COMPLETE name=warm_dag_batched_seed_site");
}

#[test]
fn warm_dag_dense_spec_seed_site_refuses_residual_and_keeps_chains() {
    let (c, s) = dense_spec_cold_and_seeded(&residual_graph(), "relu1");
    assert_refused("dense_spec", "residual/relu1", c, s);
    let (c, s) = dense_spec_cold_and_seeded(&residual_then_head_graph(), "relu1");
    assert_refused("dense_spec", "residual_then_head/relu1", c, s);
    let (c, s) = dense_spec_cold_and_seeded(&chain_graph(), "relu1");
    assert_consumed("dense_spec", "chain/relu1", c, s);
    let (c, s) = dense_spec_cold_and_seeded(&residual_then_head_graph(), "relu2");
    assert_consumed("dense_spec", "residual_then_head/relu2", c, s);
    println!("[warm-dag] TEST_COMPLETE name=warm_dag_dense_spec_seed_site");
}

// ---------------------------------------------------------------------------
// End-to-end BaB: FALSE never Verified, TRUE Verified
// ---------------------------------------------------------------------------

struct Net {
    name: &'static str,
    graph: GraphNetwork,
    hi: f32,
}

fn nets() -> Vec<Net> {
    let cases: [(&'static str, GraphNetwork, fn([f32; 2]) -> f32); 4] = [
        ("residual", residual_graph(), residual_out),
        (
            "residual_then_head",
            residual_then_head_graph(),
            residual_then_head_out,
        ),
        ("chain", chain_graph(), chain_out),
        ("biased_chain", biased_chain_graph(), biased_chain_out),
    ];
    cases
        .into_iter()
        .map(|(name, graph, f)| {
            let (lo, hi) = grid_range(f);
            println!("[warm-dag] net={name} grid_range=[{lo}, {hi}]");
            Net { name, graph, hi }
        })
        .collect()
}

/// A threshold every net exceeds somewhere (the grid max is attained).
fn false_threshold(net: &Net) -> f32 {
    net.hi - 0.25
}

/// A threshold above the true max (grid max + Lipschitz slack).
fn true_threshold(net: &Net) -> f32 {
    net.hi + 0.5
}

#[derive(Debug)]
struct Outcome {
    verified: bool,
    errored: bool,
    explored: usize,
    label: String,
}

fn outcome(result: BetaCrownResult) -> Outcome {
    Outcome {
        verified: matches!(result.result, BabVerificationStatus::Verified),
        errored: false,
        explored: result.domains_explored,
        label: format!(
            "{:?} explored={} max_depth={}",
            result.result, result.domains_explored, result.max_depth_reached
        ),
    }
}

/// DomainList lane (`verify_graph_gpu_domain_list`) in UPPER mode: proves
/// `out < threshold` by the upper bound of `out`.
fn domain_list_upper(graph: &GraphNetwork, threshold: f32, warm: bool, engine: bool) -> Outcome {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(30),
        max_domains: 200,
        max_depth: 10,
        batch_size: 2,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..warm_config(warm)
    });
    let naive = NaiveCpuGemmEngine;
    let engine: Option<&dyn ny_core::GemmEngine> = if engine { Some(&naive) } else { None };
    match verifier.verify_graph_gpu_domain_list(graph, &unit_box(), &[1.0], threshold, engine, None)
    {
        Ok(result) => outcome(result),
        Err(error) => Outcome {
            verified: false,
            errored: true,
            explored: 0,
            label: format!("Err({error})"),
        },
    }
}

/// Multi-objective ReLU-split lane in LOWER mode on the sign-normalized row
/// (`out < c` is proved as `-out > -c`), sequential or batched.
fn multi_objective_lower(
    graph: &GraphNetwork,
    threshold: f32,
    warm: bool,
    batched: bool,
) -> Outcome {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        timeout: Duration::from_secs(30),
        max_domains: 200,
        max_depth: 10,
        batch_size: if batched { 4 } else { 1 },
        ..warm_config(warm)
    });
    let naive = NaiveCpuGemmEngine;
    let engine: Option<&dyn ny_core::GemmEngine> = if batched { Some(&naive) } else { None };
    match verifier.verify_graph_relu_split_multi_objective_with_engine(
        graph,
        &unit_box(),
        &[vec![-1.0]],
        &[-threshold],
        engine,
        None,
    ) {
        Ok(result) => outcome(result),
        Err(error) => Outcome {
            verified: false,
            errored: true,
            explored: 0,
            label: format!("Err({error})"),
        },
    }
}

/// Run `lane` over every net and arm.
///
/// * FALSE must never verify (the soundness assertion).
/// * No arm may return `Err` (a failing lane would pass the first check
///   vacuously).
/// * TRUE must verify in at least one arm per net, and in EVERY arm on the
///   residual net (the regression target, root-verifiable).
/// * On the residual net at least one arm must branch on the FALSE property,
///   or no child was ever seeded and the net proves nothing.
fn run_lane(
    lane: &str,
    arms: &[(&str, bool, bool)],
    run: impl Fn(&GraphNetwork, f32, bool, bool) -> Outcome,
) {
    let mut unsound = Vec::new();
    let mut errored = Vec::new();
    let mut vacuous = Vec::new();
    for net in nets() {
        let mut branched = false;
        let mut true_verified = 0usize;
        for &(arm, warm, flag) in arms {
            let f = run(&net.graph, false_threshold(&net), warm, flag);
            println!(
                "[warm-dag] ROW lane={lane} net={} truth=FALSE arm={arm} {}",
                net.name, f.label
            );
            if f.verified {
                unsound.push(format!("{}/{arm}: {}", net.name, f.label));
            }
            if f.errored {
                errored.push(format!("{}/{arm}/FALSE: {}", net.name, f.label));
            }
            branched |= f.explored > 1;
            let t = run(&net.graph, true_threshold(&net), warm, flag);
            println!(
                "[warm-dag] ROW lane={lane} net={} truth=TRUE arm={arm} {}",
                net.name, t.label
            );
            if t.errored {
                errored.push(format!("{}/{arm}/TRUE: {}", net.name, t.label));
            }
            if t.verified {
                true_verified += 1;
            } else if net.name == "residual" {
                vacuous.push(format!(
                    "{}/{arm}: TRUE not verified: {}",
                    net.name, t.label
                ));
            }
        }
        if true_verified == 0 {
            vacuous.push(format!("{}: TRUE verified in no arm", net.name));
        }
        if net.name == "residual" && !branched {
            vacuous.push(format!(
                "{}: no arm branched on the FALSE property, so no child was seeded",
                net.name
            ));
        }
    }
    assert!(
        unsound.is_empty(),
        "{lane}: FALSE property reported Verified (unsound) on: {unsound:#?}"
    );
    assert!(errored.is_empty(), "{lane}: lane errored on: {errored:#?}");
    assert!(vacuous.is_empty(), "{lane}: vacuous run: {vacuous:#?}");
}

#[ntest::timeout(300000)]
#[test]
fn warm_dag_multi_objective_lower_mode_never_verifies_a_false_property() {
    run_lane(
        "multi_objective_lower",
        &[
            ("sequential,cold", false, false),
            ("sequential,warm", true, false),
            ("batched,cold", false, true),
            ("batched,warm", true, true),
        ],
        multi_objective_lower,
    );
    println!("[warm-dag] TEST_COMPLETE name=warm_dag_multi_objective_lower_mode");
}

#[ntest::timeout(300000)]
#[test]
fn warm_dag_domain_list_upper_mode_never_verifies_a_false_property() {
    run_lane(
        "domain_list_upper",
        &[
            ("no_engine,cold", false, false),
            ("no_engine,warm", true, false),
            ("cpu_engine,cold", false, true),
            ("cpu_engine,warm", true, true),
        ],
        domain_list_upper,
    );
    println!("[warm-dag] TEST_COMPLETE name=warm_dag_domain_list_upper_mode");
}
