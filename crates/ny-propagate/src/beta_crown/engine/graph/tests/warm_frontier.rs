// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #la-warm-frontier: the frontier-complete lA warm start on DAG graphs.
//!
//! Every test runs on small random DAGs (2-D input, dyadic weights) of four
//! shapes — a residual block, a diamond (one ReLU feeding two branches that
//! are re-joined by an Add), a Concat of a deep and a shallow branch, and a
//! skip from the network input — whose exact range over any child region is
//! enumerated on a grid in f64. Seed sites: the constrained pass (single row
//! and a two-row spec matrix), the scalar-objective batched core and the
//! dense-spec batched core. Three families:
//!
//! * SELF-RESUME = COLD (the completeness theorem). A child's cold pass with
//!   capture records a frontier at its own branch node; resuming that same
//!   child from it must reproduce the cold bound up to the coefficient-error
//!   fold (<= 1e-4 here). A frontier missing any pending block or any diverted
//!   bias moves the bound by O(1).
//! * PARENT -> CHILD VALIDITY. A child resumed from its PARENT's frontier must
//!   enclose the enumerated range of the objective over the child's region, for
//!   every branch node the #warm-dag guard refuses, both split phases, every
//!   neuron — and the resume must actually happen (counter).
//! * DARK = BIT-IDENTICAL. Key off: nothing is captured, a frontier-carrying
//!   cache is ignored bit for bit, and arming does not perturb the main cache
//!   blocks or the guard-accepted single-block seed.
//!
//! Plus end-to-end BaB with the key ARMED: a FALSE property is never
//! `Verified` in any lane / mode, a TRUE one is, and the lanes really resumed
//! children from frontiers on residual graphs.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use ndarray::{arr2, Array1, Array2};
use ny_core::NaiveCpuGemmEngine;

use crate::batched_domain::{BatchedDomains, CachedLinearBounds};
use crate::beta_crown::domain::GraphCrownContext;
use crate::beta_crown::engine::graph::propagation::BatchedBackwardContext;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::state::GraphBetaState;
use crate::beta_crown::{
    BabVerificationStatus, BetaCrownConfig, BranchingHeuristic, GraphBabDomain,
    GraphNeuronConstraint, GraphSplitHistory,
};
use crate::{
    AddLayer, BoundedTensor, ConcatLayer, GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer,
    NETWORK_INPUT,
};

// ---------------------------------------------------------------------------
// A tiny DAG language with an exact f64 evaluator
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Op {
    // Boxed weights keep `Lin` within the workspace's 64-byte
    // clippy::large_enum_variant threshold of the two-name variants.
    Lin {
        src: &'static str,
        w: Box<Array2<f32>>,
        b: Box<Array1<f32>>,
    },
    Relu(&'static str),
    Add(&'static str, &'static str),
    Concat(&'static str, &'static str),
}

#[derive(Clone)]
struct Dag {
    shape: &'static str,
    nodes: Vec<(&'static str, Op)>,
    output: &'static str,
    /// ReLU nodes a split can name, with their width.
    relus: Vec<(&'static str, usize)>,
}

impl Dag {
    fn graph(&self) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        for (name, op) in &self.nodes {
            let node = match op {
                Op::Lin { src, w, b } => {
                    let layer = Layer::Linear(
                        LinearLayer::new((**w).clone(), Some((**b).clone())).expect("linear"),
                    );
                    if *src == NETWORK_INPUT {
                        GraphNode::from_input(*name, layer)
                    } else {
                        GraphNode::new(*name, layer, vec![src.to_string()])
                    }
                }
                Op::Relu(src) => {
                    GraphNode::new(*name, Layer::ReLU(ReLULayer), vec![src.to_string()])
                }
                Op::Add(a, b) => GraphNode::new(
                    *name,
                    Layer::Add(AddLayer),
                    vec![a.to_string(), b.to_string()],
                ),
                Op::Concat(a, b) => GraphNode::new(
                    *name,
                    Layer::Concat(ConcatLayer::new(0)),
                    vec![a.to_string(), b.to_string()],
                ),
            };
            graph.add_node(node);
        }
        graph.set_output(self.output);
        graph
    }

    /// Every node's value at `x`, in f64 (weights are dyadic: exact in f32).
    fn eval(&self, x: [f64; 2]) -> HashMap<&'static str, Vec<f64>> {
        let mut values: HashMap<&'static str, Vec<f64>> = HashMap::new();
        values.insert(NETWORK_INPUT, x.to_vec());
        for (name, op) in &self.nodes {
            let value = match op {
                Op::Lin { src, w, b } => {
                    let input = &values[src];
                    (0..w.nrows())
                        .map(|i| {
                            f64::from(b[i])
                                + (0..w.ncols())
                                    .map(|j| f64::from(w[[i, j]]) * input[j])
                                    .sum::<f64>()
                        })
                        .collect()
                }
                Op::Relu(src) => values[src].iter().map(|v| v.max(0.0)).collect(),
                Op::Add(a, b) => values[a]
                    .iter()
                    .zip(&values[b])
                    .map(|(p, q)| p + q)
                    .collect(),
                Op::Concat(a, b) => values[a].iter().chain(&values[b]).copied().collect(),
            };
            values.insert(*name, value);
        }
        values
    }

    fn relu_input(&self, relu: &str) -> &'static str {
        self.nodes
            .iter()
            .find_map(|(name, op)| match op {
                Op::Relu(src) if *name == relu => Some(*src),
                _ => None,
            })
            .expect("relu node")
    }

    /// `(min, max)` of `row · out` over the grid points of the region cut out
    /// by `splits`, or `None` when no grid point lies in it.
    fn range(&self, splits: &[GraphNeuronConstraint], row: f64) -> Option<(f64, f64)> {
        let n = 80;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..=n {
            for j in 0..=n {
                let x = [
                    -1.0 + 2.0 * i as f64 / n as f64,
                    -1.0 + 2.0 * j as f64 / n as f64,
                ];
                let values = self.eval(x);
                let inside = splits.iter().all(|split| {
                    let z = values[self.relu_input(&split.node_name)][split.neuron_idx];
                    if split.is_active {
                        z >= 0.0
                    } else {
                        z <= 0.0
                    }
                });
                if inside {
                    let f = row * values[self.output][0];
                    lo = lo.min(f);
                    hi = hi.max(f);
                }
            }
        }
        (lo <= hi).then_some((lo, hi))
    }
}

/// Deterministic dyadic weights in `[-1, 1]` (multiples of 1/8, exact in f32).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let k = ((self.0 >> 33) % 17) as i32 - 8;
        k as f32 / 8.0
    }

    fn mat(&mut self, rows: usize, cols: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |_| self.next())
    }

    fn vec(&mut self, len: usize) -> Array1<f32> {
        Array1::from_shape_fn(len, |_| self.next() * 0.5)
    }
}

fn lin(src: &'static str, rng: &mut Lcg, rows: usize, cols: usize) -> Op {
    Op::Lin {
        src,
        w: Box::new(rng.mat(rows, cols)),
        b: Box::new(rng.vec(rows)),
    }
}

/// Residual block then a unary head:
/// `l1 = W x; r1 = relu(l1); l2 = W r1; r2 = relu(l2); a1 = r2 + l1;
///  l3 = W a1; r3 = relu(l3); out = w r3`.
fn residual(seed: u64) -> Dag {
    let mut r = Lcg(seed);
    Dag {
        shape: "residual",
        nodes: vec![
            ("l1", lin(NETWORK_INPUT, &mut r, 3, 2)),
            ("r1", Op::Relu("l1")),
            ("l2", lin("r1", &mut r, 3, 3)),
            ("r2", Op::Relu("l2")),
            ("a1", Op::Add("r2", "l1")),
            ("l3", lin("a1", &mut r, 2, 3)),
            ("r3", Op::Relu("l3")),
            ("out", lin("r3", &mut r, 1, 2)),
        ],
        output: "out",
        relus: vec![("r1", 3), ("r2", 3), ("r3", 2)],
    }
}

/// Diamond (a multi-consumer ReLU re-joined by an Add):
/// `r1 = relu(W x); ra = relu(Wa r1); rb = relu(Wb r1); out = w (ra + rb)`.
fn diamond(seed: u64) -> Dag {
    let mut r = Lcg(seed);
    Dag {
        shape: "diamond",
        nodes: vec![
            ("l1", lin(NETWORK_INPUT, &mut r, 2, 2)),
            ("r1", Op::Relu("l1")),
            ("la", lin("r1", &mut r, 2, 2)),
            ("lb", lin("r1", &mut r, 2, 2)),
            ("ra", Op::Relu("la")),
            ("rb", Op::Relu("lb")),
            ("add", Op::Add("ra", "rb")),
            ("out", lin("add", &mut r, 1, 2)),
        ],
        output: "out",
        relus: vec![("r1", 2), ("ra", 2), ("rb", 2)],
    }
}

/// Concat of a deep and a shallow branch:
/// `r1 = relu(W x); r2 = relu(W r1); out = w [r2; r1]`.
fn concat(seed: u64) -> Dag {
    let mut r = Lcg(seed);
    Dag {
        shape: "concat",
        nodes: vec![
            ("l1", lin(NETWORK_INPUT, &mut r, 2, 2)),
            ("r1", Op::Relu("l1")),
            ("l2", lin("r1", &mut r, 2, 2)),
            ("r2", Op::Relu("l2")),
            ("cat", Op::Concat("r2", "r1")),
            ("out", lin("cat", &mut r, 1, 4)),
        ],
        output: "out",
        relus: vec![("r1", 2), ("r2", 2)],
    }
}

/// A skip from the network input (an affine read of `x` joined late):
/// `r1 = relu(W x); r2 = relu(W r1); sk = W x; out = w (r2 + sk)`.
fn input_skip(seed: u64) -> Dag {
    let mut r = Lcg(seed);
    Dag {
        shape: "input_skip",
        nodes: vec![
            ("l1", lin(NETWORK_INPUT, &mut r, 2, 2)),
            ("r1", Op::Relu("l1")),
            ("l2", lin("r1", &mut r, 2, 2)),
            ("r2", Op::Relu("l2")),
            ("sk", lin(NETWORK_INPUT, &mut r, 2, 2)),
            ("add", Op::Add("r2", "sk")),
            ("out", lin("add", &mut r, 1, 2)),
        ],
        output: "out",
        relus: vec![("r1", 2), ("r2", 2)],
    }
}

fn dags() -> Vec<Dag> {
    let mut all = Vec::new();
    for seed in [3u64, 11, 29, 47] {
        all.push(residual(seed));
        all.push(diamond(seed ^ 0x9e37));
        all.push(concat(seed ^ 0x51ed));
        all.push(input_skip(seed ^ 0x2c1b));
    }
    all
}

fn unit_box() -> BoundedTensor {
    BoundedTensor::new(
        ndarray::arr1(&[-1.0f32, -1.0]).into_dyn(),
        ndarray::arr1(&[1.0f32, 1.0]).into_dyn(),
    )
    .expect("input box")
}

fn config(armed: bool) -> BetaCrownConfig {
    BetaCrownConfig {
        enable_la_warm_start: true,
        la_warm_frontier: armed,
        enable_cuts: false,
        ..Default::default()
    }
}

fn split(node: &str, neuron: usize, active: bool) -> GraphNeuronConstraint {
    GraphNeuronConstraint {
        node_name: node.to_string(),
        neuron_idx: neuron,
        is_active: active,
        score: 0.0,
    }
}

/// The branch nodes whose single-block seed the #warm-dag guard refuses.
fn guarded_relus(dag: &Dag, graph: &GraphNetwork) -> Vec<(&'static str, usize)> {
    dag.relus
        .iter()
        .copied()
        .filter(|(relu, _)| {
            !crate::beta_crown::engine::graph::propagation::warm_start_guard_for_tests(
                graph, dag.output, relu,
            )
        })
        .collect()
}

type Interval = Vec<(f32, f32)>;

fn rows_of(bounds: &BoundedTensor) -> Interval {
    let flat = bounds.flatten();
    flat.lower()
        .iter()
        .zip(flat.upper().iter())
        .map(|(&l, &u)| (l, u))
        .collect()
}

fn bits(interval: &Interval) -> Vec<(u32, u32)> {
    interval
        .iter()
        .map(|(l, u)| (l.to_bits(), u.to_bits()))
        .collect()
}

// ---------------------------------------------------------------------------
// The seed sites, driven one child at a time
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Site {
    /// Constrained pass, single objective row (`setup.rs`).
    Constrained,
    /// Constrained pass, two-row spec matrix (`setup.rs`, multi-objective).
    ConstrainedSpec,
    /// Scalar-objective batched core (DomainList).
    Batched,
    /// Dense-spec batched core, two rows.
    DenseSpec,
}

const SITES: [Site; 4] = [
    Site::Constrained,
    Site::ConstrainedSpec,
    Site::Batched,
    Site::DenseSpec,
];

/// The objective rows each site bounds (`row · out`).
fn site_rows(site: Site) -> Vec<f64> {
    match site {
        Site::Constrained | Site::Batched => vec![1.0],
        Site::ConstrainedSpec | Site::DenseSpec => vec![1.0, -1.0],
    }
}

fn spec_matrix() -> Array2<f32> {
    arr2(&[[1.0f32], [-1.0]])
}

/// One pass of `site` over the domain cut out by `history`, seeded from
/// `seed` (when given), capturing its own cache. Returns the output interval
/// per row and the captured cache (with its frontiers attached).
fn run_site(
    verifier: &BetaCrownVerifier,
    site: Site,
    graph: &GraphNetwork,
    history: &GraphSplitHistory,
    seed: Option<&CachedLinearBounds>,
) -> (Interval, Option<CachedLinearBounds>) {
    let input = unit_box();
    let base: HashMap<String, BoundedTensor> = graph.collect_node_bounds(&input).expect("bounds");
    match site {
        Site::Constrained | Site::ConstrainedSpec => {
            let node_bounds: HashMap<String, Arc<BoundedTensor>> =
                base.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
            let beta_state = GraphBetaState::from_history(history).expect("beta state");
            let context = GraphCrownContext::new(history, None, Some(&node_bounds), None);
            let (out, _, cache) = if site == Site::Constrained {
                verifier
                    .propagate_crown_with_graph_beta_and_cache(
                        graph,
                        &input,
                        &context,
                        &beta_state,
                        Some(&[1.0]),
                        seed,
                        true,
                    )
                    .expect("constrained pass")
            } else {
                verifier
                    .propagate_crown_with_graph_beta_and_spec_matrix(
                        graph,
                        &input,
                        &context,
                        &beta_state,
                        &spec_matrix(),
                        seed,
                        true,
                    )
                    .expect("constrained spec pass")
            };
            (rows_of(&out), cache)
        }
        Site::Batched | Site::DenseSpec => {
            let root = GraphBabDomain::root(base, -100.0, 100.0, &input, false).expect("root");
            let mut domain = root;
            for constraint in history.constraints.iter().cloned() {
                domain = domain
                    .with_constraint(graph, constraint, false)
                    .expect("with_constraint")
                    .expect("the tests only build non-empty children");
            }
            domain.cached_la = seed.map(|cache| Arc::new(cache.clone()));
            let domains = vec![&domain];
            let layer_names: Vec<String> = graph
                .nodes
                .iter()
                .filter(|(_, node)| matches!(node.layer, Layer::ReLU(_)))
                .map(|(name, _)| name.clone())
                .collect();
            let batched =
                BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched");
            let ctx = BatchedBackwardContext::from_domains(&domains, &batched).expect("context");
            if site == Site::Batched {
                let result = verifier
                    .propagate_crown_batched_with_context_capture_la(
                        graph,
                        &ctx,
                        &[1.0],
                        &NaiveCpuGemmEngine,
                    )
                    .expect("batched pass");
                let cache = captured_cache(result.intermediate_la, result.intermediate_frontiers);
                (rows_of(&result.results[0].0), cache)
            } else {
                let result = verifier
                    .propagate_crown_batched_with_context_specs_capture_la(
                        graph,
                        &ctx,
                        &spec_matrix(),
                        &NaiveCpuGemmEngine,
                    )
                    .expect("dense-spec pass");
                let cache = captured_cache(result.intermediate_la, result.intermediate_frontiers);
                (rows_of(&result.results[0].output_bounds), cache)
            }
        }
    }
}

/// Assemble domain 0's cache exactly as the production consumers do.
fn captured_cache(
    la: Option<Vec<HashMap<String, crate::LinearBounds>>>,
    frontiers: Option<Vec<HashMap<String, crate::batched_domain::WarmFrontier>>>,
) -> Option<CachedLinearBounds> {
    let map = la?.into_iter().next()?;
    let mut cache = CachedLinearBounds::from_linear_bounds_map(map);
    if let Some(frontiers) = frontiers.and_then(|all| all.into_iter().next()) {
        cache.attach_frontiers(frontiers);
    }
    Some(cache)
}

fn history_of(splits: &[GraphNeuronConstraint]) -> GraphSplitHistory {
    splits
        .iter()
        .cloned()
        .fold(GraphSplitHistory::new(), |history, constraint| {
            history.with_constraint(constraint)
        })
}

/// Children of the root with a single split: every guarded branch node, every
/// neuron, both phases, when the child is non-empty on the grid.
fn children(dag: &Dag, graph: &GraphNetwork) -> Vec<Vec<GraphNeuronConstraint>> {
    let mut out = Vec::new();
    for (relu, width) in guarded_relus(dag, graph) {
        for neuron in 0..width {
            for active in [true, false] {
                let splits = vec![split(relu, neuron, active)];
                if dag.range(&splits, 1.0).is_some() {
                    out.push(splits);
                }
            }
        }
    }
    out
}

fn child_is_buildable(graph: &GraphNetwork, splits: &[GraphNeuronConstraint]) -> bool {
    let input = unit_box();
    let base = graph.collect_node_bounds(&input).expect("bounds");
    let mut domain = GraphBabDomain::root(base, -100.0, 100.0, &input, false).expect("root");
    for constraint in splits.iter().cloned() {
        match domain.with_constraint(graph, constraint, false) {
            Ok(Some(child)) => domain = child,
            _ => return false,
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Family 1: self-resume = cold (completeness)
// ---------------------------------------------------------------------------

#[ntest::timeout(600000)]
#[test]
fn warm_frontier_self_resume_equals_cold_at_every_seed_site() {
    let mut checked = 0usize;
    let mut input_coefficients_seen = 0usize;
    for dag in dags() {
        let graph = dag.graph();
        for splits in children(&dag, &graph) {
            if !child_is_buildable(&graph, &splits) {
                continue;
            }
            let history = history_of(&splits);
            for site in SITES {
                let verifier = BetaCrownVerifier::new(config(true));
                let (cold, own) = run_site(&verifier, site, &graph, &history, None);
                let own = own.expect("an armed capturing pass returns a cache");
                let frontier = own.frontier(&splits[0].node_name).unwrap_or_else(|| {
                    panic!(
                        "{}/{site:?}/{}: the cold pass recorded no frontier at its branch node",
                        dag.shape, splits[0].node_name
                    )
                });
                input_coefficients_seen += frontier
                    .members()
                    .filter(|(key, block)| *key == NETWORK_INPUT && !block.is_bias_only())
                    .count();
                let before = verifier.la_warm_frontier_counts();
                let (resumed, _) = run_site(&verifier, site, &graph, &history, Some(&own));
                let after = verifier.la_warm_frontier_counts();
                assert_eq!(
                    after.resumed(),
                    before.resumed() + 1,
                    "{}/{site:?}/{:?}: the child was not resumed from its frontier ({after:?})",
                    dag.shape,
                    splits
                );
                for (row, ((cl, cu), (wl, wu))) in cold.iter().zip(&resumed).enumerate() {
                    let tol = 1e-4 * (1.0 + cl.abs().max(cu.abs()));
                    assert!(
                        (cl - wl).abs() <= tol && (cu - wu).abs() <= tol,
                        "{}/{site:?}/{:?} row {row}: self-resume [{wl}, {wu}] != cold [{cl}, {cu}] — \
                         the frontier is not the complete backward state",
                        dag.shape,
                        splits
                    );
                    assert!(
                        *wl <= cl + tol && *wu >= cu - tol,
                        "{}/{site:?}/{:?} row {row}: self-resume TIGHTER than cold",
                        dag.shape,
                        splits
                    );
                }
                checked += 1;
            }
        }
    }
    println!(
        "[la-warm-frontier] self_resume checked={checked} input_coefficient_members={input_coefficients_seen}"
    );
    assert!(checked >= 60, "too few self-resume cases ran: {checked}");
    assert!(
        input_coefficients_seen > 0,
        "no frontier carried input-channel coefficients: the input-skip shape is vacuous"
    );
    println!("[la-warm-frontier] TEST_COMPLETE name=warm_frontier_self_resume_equals_cold_at_every_seed_site");
}

// ---------------------------------------------------------------------------
// Family 2: parent -> child validity against enumeration
// ---------------------------------------------------------------------------

#[ntest::timeout(900000)]
#[test]
fn warm_frontier_parent_resume_encloses_the_child_range_at_every_seed_site() {
    let mut checked = 0usize;
    let mut looser = 0usize;
    let mut tighter = 0usize;
    for dag in dags() {
        let graph = dag.graph();
        for site in SITES {
            let verifier = BetaCrownVerifier::new(config(true));
            let (_, parent) = run_site(&verifier, site, &graph, &GraphSplitHistory::new(), None);
            let parent = parent.expect("root cache");
            for splits in children(&dag, &graph) {
                if !child_is_buildable(&graph, &splits) {
                    continue;
                }
                let history = history_of(&splits);
                let before = verifier.la_warm_frontier_counts();
                let (warm, _) = run_site(&verifier, site, &graph, &history, Some(&parent));
                let after = verifier.la_warm_frontier_counts();
                assert_eq!(
                    after.resumed(),
                    before.resumed() + 1,
                    "{}/{site:?}/{:?}: the child was not resumed from its parent's frontier \
                     ({after:?})",
                    dag.shape,
                    splits
                );
                let (cold, _) = run_site(
                    &BetaCrownVerifier::new(config(false)),
                    site,
                    &graph,
                    &history,
                    None,
                );
                for (row, (&sign, ((wl, wu), (cl, cu)))) in site_rows(site)
                    .iter()
                    .zip(warm.iter().zip(&cold))
                    .enumerate()
                {
                    let (lo, hi) = dag.range(&splits, sign).expect("non-empty child");
                    assert!(
                        f64::from(*wl) <= lo + 1e-5 && f64::from(*wu) >= hi - 1e-5,
                        "UNSOUND {}/{site:?}/{:?} row {row}: resumed [{wl}, {wu}] does not \
                         enclose the enumerated range [{lo}, {hi}] (cold [{cl}, {cu}])",
                        dag.shape,
                        splits
                    );
                    if *wl < cl - 1e-5 || *wu > cu + 1e-5 {
                        looser += 1;
                    }
                    if *wl > cl + 1e-5 || *wu < cu - 1e-5 {
                        tighter += 1;
                    }
                }
                checked += 1;
            }
        }
    }
    println!(
        "[la-warm-frontier] parent_resume checked={checked} rows_looser_than_cold={looser} \
         rows_tighter_than_cold={tighter}"
    );
    assert!(checked >= 60, "too few parent-resume cases ran: {checked}");
    println!("[la-warm-frontier] TEST_COMPLETE name=warm_frontier_parent_resume_encloses_the_child_range_at_every_seed_site");
}

// ---------------------------------------------------------------------------
// Family 3: dark = bit-identical
// ---------------------------------------------------------------------------

#[ntest::timeout(600000)]
#[test]
fn warm_frontier_dark_is_bit_identical_and_captures_nothing() {
    let mut checked = 0usize;
    for dag in dags() {
        let graph = dag.graph();
        for site in SITES {
            // Arming never perturbs the main cache blocks.
            let dark = BetaCrownVerifier::new(config(false));
            let armed = BetaCrownVerifier::new(config(true));
            let root = GraphSplitHistory::new();
            let (dark_root, dark_cache) = run_site(&dark, site, &graph, &root, None);
            let (armed_root, armed_cache) = run_site(&armed, site, &graph, &root, None);
            let dark_cache = dark_cache.expect("dark cache");
            let armed_cache = armed_cache.expect("armed cache");
            assert_eq!(bits(&dark_root), bits(&armed_root));
            assert_eq!(
                dark_cache.frontier_count(),
                0,
                "a dark pass captured a frontier"
            );
            assert!(
                armed_cache.frontier_count() > 0,
                "{}/{site:?}: armed captured nothing",
                dag.shape
            );
            for (map_dark, map_armed) in [
                (&dark_cache.lower_a, &armed_cache.lower_a),
                (&dark_cache.upper_a, &armed_cache.upper_a),
            ] {
                assert_eq!(map_dark.len(), map_armed.len());
                for (node, block) in map_dark {
                    let other = &map_armed[node];
                    assert!(
                        block
                            .iter()
                            .zip(other.iter())
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "arming perturbed the main cache block at {node}"
                    );
                }
            }
            for (map_dark, map_armed) in [
                (&dark_cache.lower_b, &armed_cache.lower_b),
                (&dark_cache.upper_b, &armed_cache.upper_b),
            ] {
                for (node, bias) in map_dark {
                    let other = &map_armed[node];
                    assert!(
                        bias.iter()
                            .zip(other.iter())
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "arming perturbed the main cache bias at {node}"
                    );
                }
            }
            // A dark child ignores a frontier-carrying cache bit for bit: it
            // equals the same child seeded from the frontier-free cache, and
            // (the guard refusing) the cold pass.
            let mut stripped = armed_cache.clone();
            stripped.frontiers.clear();
            for splits in children(&dag, &graph).into_iter().take(3) {
                if !child_is_buildable(&graph, &splits) {
                    continue;
                }
                let history = history_of(&splits);
                let (with_frontier, _) =
                    run_site(&dark, site, &graph, &history, Some(&armed_cache));
                let (without, _) = run_site(&dark, site, &graph, &history, Some(&stripped));
                let (cold, _) = run_site(&dark, site, &graph, &history, None);
                assert_eq!(bits(&with_frontier), bits(&without));
                assert_eq!(bits(&with_frontier), bits(&cold));
                checked += 1;
            }
            assert_eq!(
                dark.la_warm_frontier_counts(),
                Default::default(),
                "a dark verifier recorded frontier activity"
            );
        }
    }
    assert!(checked >= 20, "too few dark cases ran: {checked}");
    println!("[la-warm-frontier] TEST_COMPLETE name=warm_frontier_dark_is_bit_identical_and_captures_nothing");
}

/// A split in the final unary chain keeps the historical single-block seed
/// bit for bit when armed (the frontier path runs only where the guard
/// refuses).
#[ntest::timeout(300000)]
#[test]
fn warm_frontier_armed_keeps_the_guard_accepted_single_seed_bit_identical() {
    let mut checked = 0usize;
    for seed in [3u64, 11, 29] {
        let dag = residual(seed);
        let graph = dag.graph();
        for site in SITES {
            let root = GraphSplitHistory::new();
            let dark = BetaCrownVerifier::new(config(false));
            let armed = BetaCrownVerifier::new(config(true));
            let (_, dark_cache) = run_site(&dark, site, &graph, &root, None);
            let (_, armed_cache) = run_site(&armed, site, &graph, &root, None);
            for neuron in 0..2 {
                for active in [true, false] {
                    let splits = vec![split("r3", neuron, active)];
                    if dag.range(&splits, 1.0).is_none() || !child_is_buildable(&graph, &splits) {
                        continue;
                    }
                    let history = history_of(&splits);
                    let before = armed.la_warm_frontier_counts();
                    let (dark_out, _) =
                        run_site(&dark, site, &graph, &history, dark_cache.as_ref());
                    let (armed_out, _) =
                        run_site(&armed, site, &graph, &history, armed_cache.as_ref());
                    let after = armed.la_warm_frontier_counts();
                    assert_eq!(bits(&dark_out), bits(&armed_out));
                    assert_eq!(after.resumed(), before.resumed(), "a chain split resumed");
                    assert_eq!(
                        after.single_seed,
                        before.single_seed + 1,
                        "single seed not used"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked >= 8, "too few chain cases ran: {checked}");
    println!("[la-warm-frontier] TEST_COMPLETE name=warm_frontier_armed_keeps_the_guard_accepted_single_seed_bit_identical");
}

/// A frontier that does not fit the child is refused and the child runs the
/// cold pass exactly: (a) a one-row frontier read by a two-row spec pass;
/// (b) PAIRING — on the diamond, the frontier recorded at `rb` moved as-is
/// onto branch node `r1` / `ra` (its digest names `rb`'s block, not theirs);
/// (c) STRUCTURE — the same members re-captured as a correctly PAIRED
/// frontier of `r1` (member `ra` does not precede the branch node) and of `ra`
/// (a member names the branch node itself); (d) FOREIGN BLOCK — a node's own
/// frontier next to a different block for that node (only the digest can
/// tell).
#[ntest::timeout(300000)]
#[test]
fn warm_frontier_invalid_frontiers_are_refused_and_run_cold() {
    use crate::batched_domain::{FrontierCapture, WarmFrontier};
    let dag = diamond(0xb ^ 0x9e37);
    let graph = dag.graph();
    let armed = BetaCrownVerifier::new(config(true));
    let (_, parent) = run_site(
        &armed,
        Site::Constrained,
        &graph,
        &GraphSplitHistory::new(),
        None,
    );
    let parent = parent.expect("root cache");
    let rb_frontier = parent
        .frontier("rb")
        .expect("the diamond records a frontier at rb")
        .clone();
    assert!(
        rb_frontier.members().any(|(key, _)| key == "ra"),
        "the rb frontier must carry the pending ra block"
    );
    let rb_members: Vec<(String, crate::LinearBounds)> = rb_frontier
        .members()
        .map(|(key, block)| (key.to_string(), block.to_linear_bounds().expect("rebuild")))
        .collect();
    let mut refusals = 0usize;
    for branch in ["r1", "ra"] {
        let Some(splits) = (0..2)
            .flat_map(|neuron| [true, false].map(|active| vec![split(branch, neuron, active)]))
            .find(|splits| dag.range(splits, 1.0).is_some() && child_is_buildable(&graph, splits))
        else {
            continue;
        };
        let history = history_of(&splits);
        let (cold, _) = run_site(&armed, Site::Constrained, &graph, &history, None);

        // (a) wrong rows.
        let before = armed.la_warm_frontier_counts();
        let (spec_cold, _) = run_site(&armed, Site::ConstrainedSpec, &graph, &history, None);
        let (spec_seeded, _) = run_site(
            &armed,
            Site::ConstrainedSpec,
            &graph,
            &history,
            Some(&parent),
        );
        let after = armed.la_warm_frontier_counts();
        assert_eq!(
            bits(&spec_cold),
            bits(&spec_seeded),
            "{branch}: wrong-row seed consumed"
        );
        assert_eq!(after.resumed(), before.resumed());
        assert!(after.refused_invalid > before.refused_invalid);

        // (b) pairing: a frontier moved next to a block it was not captured with.
        let branch_block = parent.linear_bounds(branch).expect("branch block");
        let forged = match WarmFrontier::capture(&branch_block, rb_members.clone(), |_| None, None)
            .expect("capture")
        {
            FrontierCapture::Recorded(frontier) => frontier,
            other => panic!("re-capture must record: {other:?}"),
        };
        // (d) a node's OWN frontier next to a foreign block for that node (what
        // a per-node cache merge produces when a fresh pass overwrites the
        // block but not the frontier): structurally perfect, digest differs.
        let own = parent.frontier(branch).expect("own frontier").clone();
        for (case, frontier) in [
            ("pairing", rb_frontier.clone()),
            ("structure", forged),
            ("foreign_block", own),
        ] {
            let mut moved = parent.clone();
            moved.frontiers.insert(branch.to_string(), frontier);
            if case == "foreign_block" {
                let lower = moved.lower_b.get_mut(branch).expect("branch bias");
                lower.mapv_inplace(|v| v - 1.0);
            }
            let before = armed.la_warm_frontier_counts();
            let (seeded, _) = run_site(&armed, Site::Constrained, &graph, &history, Some(&moved));
            let after = armed.la_warm_frontier_counts();
            assert_eq!(
                bits(&seeded),
                bits(&cold),
                "{branch}/{case}: an invalid frontier was consumed"
            );
            assert_eq!(
                after.resumed(),
                before.resumed(),
                "{branch}/{case}: resumed"
            );
            assert_eq!(
                after.refused_invalid,
                before.refused_invalid + 1,
                "{branch}/{case}: not refused as invalid"
            );
            refusals += 1;
        }
    }
    assert_eq!(refusals, 6, "every refusal case must run");
    println!("[la-warm-frontier] TEST_COMPLETE name=warm_frontier_invalid_frontiers_are_refused_and_run_cold");
}

// ---------------------------------------------------------------------------
// End-to-end BaB with the key armed
// ---------------------------------------------------------------------------

struct Outcome {
    verified: bool,
    errored: bool,
    explored: usize,
    label: String,
}

fn outcome(result: ny_core::Result<crate::beta_crown::BetaCrownResult>) -> Outcome {
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

fn bab_config(armed: bool) -> BetaCrownConfig {
    BetaCrownConfig {
        timeout: Duration::from_secs(30),
        max_domains: 200,
        max_depth: 10,
        ..config(armed)
    }
}

/// `(outcome, resumed)` of the multi-objective lane in lower mode on the
/// sign-normalized row (`out < c` as `-out > -c`).
fn multi_objective_lower(graph: &GraphNetwork, threshold: f32, batched: bool) -> (Outcome, u64) {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        batch_size: if batched { 4 } else { 1 },
        ..bab_config(true)
    });
    let naive = NaiveCpuGemmEngine;
    let engine: Option<&dyn ny_core::GemmEngine> = if batched { Some(&naive) } else { None };
    let result = verifier.verify_graph_relu_split_multi_objective_with_engine(
        graph,
        &unit_box(),
        &[vec![-1.0]],
        &[-threshold],
        engine,
        None,
    );
    (
        outcome(result),
        verifier.la_warm_frontier_counts().resumed(),
    )
}

/// `(outcome, resumed)` of the DomainList lane in UPPER mode (`out < c`).
fn domain_list_upper(graph: &GraphNetwork, threshold: f32, engine: bool) -> (Outcome, u64) {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        verify_upper_bound: true,
        batch_size: 2,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..bab_config(true)
    });
    let naive = NaiveCpuGemmEngine;
    let engine: Option<&dyn ny_core::GemmEngine> = if engine { Some(&naive) } else { None };
    let result =
        verifier.verify_graph_gpu_domain_list(graph, &unit_box(), &[1.0], threshold, engine, None);
    (
        outcome(result),
        verifier.la_warm_frontier_counts().resumed(),
    )
}

fn e2e_dags() -> Vec<Dag> {
    vec![
        residual(3),
        residual(29),
        diamond(0xb ^ 0x9e37),
        input_skip(0x2f ^ 0x2c1b),
    ]
}

fn run_e2e(lane: &str, arms: &[bool], run: impl Fn(&GraphNetwork, f32, bool) -> (Outcome, u64)) {
    let mut unsound = Vec::new();
    let mut errored = Vec::new();
    let mut vacuous = Vec::new();
    let mut resumed_total = 0u64;
    for dag in e2e_dags() {
        let graph = dag.graph();
        let (_, hi) = dag.range(&[], 1.0).expect("range");
        let hi = hi as f32;
        let false_threshold = hi - 0.05;
        let true_threshold = hi + 1.0;
        let mut true_verified = 0usize;
        for &arm in arms {
            let (f, resumed) = run(&graph, false_threshold, arm);
            resumed_total += resumed;
            println!(
                "[la-warm-frontier] ROW lane={lane} net={} truth=FALSE arm={arm} resumed={resumed} {}",
                dag.shape, f.label
            );
            if f.verified {
                unsound.push(format!("{}/{arm}: {}", dag.shape, f.label));
            }
            if f.errored {
                errored.push(format!("{}/{arm}/FALSE: {}", dag.shape, f.label));
            }
            let _ = f.explored;
            let (t, resumed) = run(&graph, true_threshold, arm);
            resumed_total += resumed;
            println!(
                "[la-warm-frontier] ROW lane={lane} net={} truth=TRUE arm={arm} resumed={resumed} {}",
                dag.shape, t.label
            );
            if t.errored {
                errored.push(format!("{}/{arm}/TRUE: {}", dag.shape, t.label));
            }
            true_verified += usize::from(t.verified);
        }
        if true_verified == 0 {
            vacuous.push(format!("{}: TRUE verified in no arm", dag.shape));
        }
    }
    println!("[la-warm-frontier] lane={lane} resumed_total={resumed_total}");
    assert!(
        unsound.is_empty(),
        "{lane}: FALSE property reported Verified (UNSOUND) on: {unsound:#?}"
    );
    assert!(errored.is_empty(), "{lane}: lane errored on: {errored:#?}");
    assert!(vacuous.is_empty(), "{lane}: vacuous run: {vacuous:#?}");
    assert!(
        resumed_total > 0,
        "{lane}: no child was ever resumed from a frontier on residual graphs \
         (the key is armed but the seeds were silently refused)"
    );
}

#[ntest::timeout(600000)]
#[test]
fn warm_frontier_multi_objective_never_verifies_a_false_property_and_resumes() {
    run_e2e(
        "multi_objective_lower",
        &[false, true],
        multi_objective_lower,
    );
    println!("[la-warm-frontier] TEST_COMPLETE name=warm_frontier_multi_objective_never_verifies_a_false_property_and_resumes");
}

#[ntest::timeout(600000)]
#[test]
fn warm_frontier_domain_list_never_verifies_a_false_property_and_resumes() {
    run_e2e("domain_list_upper", &[false, true], domain_list_upper);
    println!("[la-warm-frontier] TEST_COMPLETE name=warm_frontier_domain_list_never_verifies_a_false_property_and_resumes");
}

// ---------------------------------------------------------------------------
// Review (adversarial): multi-generation descent over nested frontiers
// ---------------------------------------------------------------------------
//
// The families above resume depth-1 children from the ROOT's frontier only.
// The production lanes go deeper: a resumed child's own pass CAPTURES the
// frontiers its children resume from (a nested state: members inherited from
// the ancestor's frontier plus the child's own pending blocks), the
// multi-objective lane UNIONS a fresh capture over the inherited cache per
// node (#la-union-merge), and the DomainList keeps the fresh capture alone.
// These tests walk three generations on richer DAGs (two stacked residual
// blocks whose second skip carries the first block's output; a three-consumer
// ReLU re-joined by two Adds plus an input skip) at every seed site, against
// the enumerated range, and pin the exact identity a resumed capture must
// satisfy.

impl Dag {
    /// `(min, max)` of `c · out` over the grid points of the region cut out by
    /// `splits` (the multi-row generalization of [`Dag::range`]).
    fn range_of(&self, splits: &[GraphNeuronConstraint], c: &[f64]) -> Option<(f64, f64)> {
        let n = 80;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..=n {
            for j in 0..=n {
                let x = [
                    -1.0 + 2.0 * i as f64 / n as f64,
                    -1.0 + 2.0 * j as f64 / n as f64,
                ];
                let values = self.eval(x);
                let inside = splits.iter().all(|split| {
                    let z = values[self.relu_input(&split.node_name)][split.neuron_idx];
                    if split.is_active {
                        z >= 0.0
                    } else {
                        z <= 0.0
                    }
                });
                if inside {
                    let f: f64 = c.iter().zip(&values[self.output]).map(|(a, v)| a * v).sum();
                    lo = lo.min(f);
                    hi = hi.max(f);
                }
            }
        }
        (lo <= hi).then_some((lo, hi))
    }
}

/// Two residual blocks in sequence; the second block's skip carries the first
/// block's OUTPUT, so a split inside block 1 is popped with BOTH skips' blocks
/// and both Adds' diverted bias pending (a nested frontier):
/// `l1 = W x; r1 = relu(l1); l2 = W r1; r2 = relu(l2); a1 = r2 + l1;
///  l3 = W a1; r3 = relu(l3); l4 = W r3; r4 = relu(l4); a2 = r4 + a1;
///  out = W a2` (`heads` output rows).
fn double_residual(seed: u64, heads: usize) -> Dag {
    let mut r = Lcg(seed);
    Dag {
        shape: "double_residual",
        nodes: vec![
            ("l1", lin(NETWORK_INPUT, &mut r, 3, 2)),
            ("r1", Op::Relu("l1")),
            ("l2", lin("r1", &mut r, 3, 3)),
            ("r2", Op::Relu("l2")),
            ("a1", Op::Add("r2", "l1")),
            ("l3", lin("a1", &mut r, 3, 3)),
            ("r3", Op::Relu("l3")),
            ("l4", lin("r3", &mut r, 3, 3)),
            ("r4", Op::Relu("l4")),
            ("a2", Op::Add("r4", "a1")),
            ("out", lin("a2", &mut r, heads, 3)),
        ],
        output: "out",
        relus: vec![("r1", 3), ("r2", 3), ("r3", 3), ("r4", 3)],
    }
}

/// One ReLU with THREE consumers re-joined by two Adds, plus a late skip from
/// the network input (up to three pending blocks and three diverted biases at
/// the inner pops):
/// `r1 = relu(W x); ra = relu(Wa r1); rb = relu(Wb r1); rc = relu(Wc r1);
///  s1 = ra + rb; s2 = s1 + rc; sk = W x; s3 = s2 + sk; out = W s3`.
fn three_consumer(seed: u64, heads: usize) -> Dag {
    let mut r = Lcg(seed);
    Dag {
        shape: "three_consumer",
        nodes: vec![
            ("l1", lin(NETWORK_INPUT, &mut r, 2, 2)),
            ("r1", Op::Relu("l1")),
            ("la", lin("r1", &mut r, 2, 2)),
            ("lb", lin("r1", &mut r, 2, 2)),
            ("lc", lin("r1", &mut r, 2, 2)),
            ("ra", Op::Relu("la")),
            ("rb", Op::Relu("lb")),
            ("rc", Op::Relu("lc")),
            ("s1", Op::Add("ra", "rb")),
            ("s2", Op::Add("s1", "rc")),
            ("sk", lin(NETWORK_INPUT, &mut r, 2, 2)),
            ("s3", Op::Add("s2", "sk")),
            ("out", lin("s3", &mut r, heads, 2)),
        ],
        output: "out",
        relus: vec![("r1", 2), ("ra", 2), ("rb", 2), ("rc", 2)],
    }
}

fn descent_dags() -> Vec<Dag> {
    vec![
        residual(3),
        residual(29),
        diamond(0xb ^ 0x9e37),
        concat(0x1d ^ 0x51ed),
        input_skip(0x2f ^ 0x2c1b),
        double_residual(5, 1),
        double_residual(17, 1),
        three_consumer(7, 1),
        three_consumer(23, 1),
    ]
}

/// Evenly spaced deterministic sample of at most `cap` items.
fn sample<T: Clone>(items: Vec<T>, cap: usize) -> Vec<T> {
    if items.len() <= cap {
        return items;
    }
    (0..cap)
        .map(|k| items[k * items.len() / cap].clone())
        .collect()
}

/// Every extension of `prefix` by one split on ANY ReLU (guarded or chain)
/// that is not already split, non-empty on the grid and buildable, with its
/// enumerated range of `out`.
fn extensions(
    dag: &Dag,
    graph: &GraphNetwork,
    prefix: &[GraphNeuronConstraint],
) -> Vec<(Vec<GraphNeuronConstraint>, (f64, f64))> {
    let mut out = Vec::new();
    for &(relu, width) in &dag.relus {
        for neuron in 0..width {
            if prefix
                .iter()
                .any(|s| s.node_name == relu && s.neuron_idx == neuron)
            {
                continue;
            }
            for active in [true, false] {
                let mut splits = prefix.to_vec();
                splits.push(split(relu, neuron, active));
                let Some(range) = dag.range(&splits, 1.0) else {
                    continue;
                };
                if child_is_buildable(graph, &splits) {
                    out.push((splits, range));
                }
            }
        }
    }
    out
}

/// `Err` when some row of `bounds` fails to enclose the enumerated range of
/// that row's objective (`site_rows`: `+out`, and `-out` for the second row).
fn encloses(site: Site, range: (f64, f64), bounds: &Interval) -> Result<(), String> {
    let rows = site_rows(site);
    if rows.len() != bounds.len() {
        return Err(format!(
            "{} rows for {} objectives",
            bounds.len(),
            rows.len()
        ));
    }
    for (row, (&sign, &(l, u))) in rows.iter().zip(bounds).enumerate() {
        let (lo, hi) = if sign > 0.0 {
            range
        } else {
            (-range.1, -range.0)
        };
        if !(f64::from(l) <= lo + 1e-5 && f64::from(u) >= hi - 1e-5) {
            return Err(format!(
                "row {row}: [{l}, {u}] does not enclose the enumerated [{lo}, {hi}]"
            ));
        }
    }
    Ok(())
}

type Plan = Vec<(
    Vec<GraphNeuronConstraint>,
    (f64, f64),
    Vec<(
        Vec<GraphNeuronConstraint>,
        (f64, f64),
        Option<(Vec<GraphNeuronConstraint>, (f64, f64))>,
    )>,
)>;

/// Three generations per DAG, fixed across sites: up to 10 children, up to 5
/// grandchildren each, and one great-grandchild per grandchild.
fn descent_plan(dag: &Dag, graph: &GraphNetwork) -> Plan {
    sample(extensions(dag, graph, &[]), 10)
        .into_iter()
        .map(|(h1, r1)| {
            let gen2: Vec<_> = sample(extensions(dag, graph, &h1), 5)
                .into_iter()
                .map(|(h2, r2)| {
                    let gen3 = sample(extensions(dag, graph, &h2), 1).into_iter().next();
                    (h2, r2, gen3)
                })
                .collect();
            (h1, r1, gen2)
        })
        .collect()
}

/// ADVERSARIAL DESCENT. At every seed site, on nine DAGs: root (cold,
/// capturing) -> child resumed from the root -> grandchild seeded from (a) the
/// child's FRESH capture alone (the DomainList cache; a resumed child's
/// frontiers are nested: ancestor members + its own pending blocks) and (b)
/// the per-node UNION of that capture over the root cache (the
/// multi-objective #la-union-merge cache) -> great-grandchild seeded from the
/// grandchild's fresh capture. Every bound at every generation must enclose
/// the enumerated range of the objective over that generation's region; the
/// walk must really resume from captures that RESUMED passes made (counter),
/// and from union caches.
#[ntest::timeout(2400000)]
#[test]
fn warm_frontier_review_multi_generation_descent_encloses_every_region() {
    let mut checked = 0usize;
    let mut gen1_resumed = 0u64;
    let mut gen2_resumed_from_resumed_capture = 0u64;
    let mut gen2_resumed_from_union = 0u64;
    let mut gen3_resumed = 0u64;
    let mut unsound = Vec::new();
    for dag in descent_dags() {
        let graph = dag.graph();
        let plan = descent_plan(&dag, &graph);
        for site in SITES {
            let verifier = BetaCrownVerifier::new(config(true));
            let (_, root) = run_site(&verifier, site, &graph, &GraphSplitHistory::new(), None);
            let root = root.expect("root cache");
            let mut run = |h: &[GraphNeuronConstraint],
                           range: (f64, f64),
                           cache: &CachedLinearBounds,
                           what: &str|
             -> (bool, Option<CachedLinearBounds>) {
                let before = verifier.la_warm_frontier_counts().resumed();
                let (bounds, captured) =
                    run_site(&verifier, site, &graph, &history_of(h), Some(cache));
                let resumed = verifier.la_warm_frontier_counts().resumed() > before;
                if let Err(why) = encloses(site, range, &bounds) {
                    unsound.push(format!(
                        "{}/{site:?}/{what} {:?} resumed={resumed}: {why}",
                        dag.shape,
                        h.iter()
                            .map(|s| (s.node_name.as_str(), s.neuron_idx, s.is_active))
                            .collect::<Vec<_>>()
                    ));
                }
                checked += 1;
                (resumed, captured)
            };
            for (h1, r1, gen2) in &plan {
                let (resumed1, c1) = run(h1, *r1, &root, "gen1");
                gen1_resumed += u64::from(resumed1);
                let Some(c1) = c1 else { continue };
                let mut union1 = root.clone();
                union1.union_with_fresh(c1.clone());
                for (h2, r2, gen3) in gen2 {
                    let (resumed2, c2) = run(h2, *r2, &c1, "gen2-fresh");
                    if resumed1 && resumed2 {
                        gen2_resumed_from_resumed_capture += 1;
                    }
                    let (resumed2u, _) = run(h2, *r2, &union1, "gen2-union");
                    gen2_resumed_from_union += u64::from(resumed2u);
                    if let (Some(c2), Some((h3, r3))) = (c2, gen3) {
                        let (resumed3, _) = run(h3, *r3, &c2, "gen3-fresh");
                        gen3_resumed += u64::from(resumed3);
                    }
                }
            }
        }
    }
    println!(
        "[la-warm-frontier] review_descent checked={checked} gen1_resumed={gen1_resumed} \
         gen2_resumed_from_resumed_capture={gen2_resumed_from_resumed_capture} \
         gen2_resumed_from_union={gen2_resumed_from_union} gen3_resumed={gen3_resumed} \
         unsound={}",
        unsound.len()
    );
    assert!(
        unsound.is_empty(),
        "UNSOUND descent bounds ({}): {:#?}",
        unsound.len(),
        &unsound[..unsound.len().min(40)]
    );
    assert!(checked >= 500, "too few descent cases ran: {checked}");
    assert!(gen1_resumed > 0, "no child resumed from the root frontier");
    assert!(
        gen2_resumed_from_resumed_capture > 0,
        "no grandchild resumed from a frontier that a RESUMED pass captured: \
         the nested-frontier case is vacuous"
    );
    assert!(
        gen2_resumed_from_union > 0,
        "no grandchild resumed from a union cache: the #la-union-merge case is vacuous"
    );
    assert!(
        gen3_resumed > 0,
        "no great-grandchild resumed: the third generation is vacuous"
    );
    println!("[la-warm-frontier] TEST_COMPLETE name=warm_frontier_review_multi_generation_descent_encloses_every_region");
}

/// EXACT IDENTITY FOR RESUMED CAPTURES ("resumed passes capture too"). A child
/// resumed from its parent's frontier re-captures, at its own branch node,
/// the very state it was seeded with; resuming the SAME child again from that
/// capture must reproduce the first resumed bound (up to the coefficient-error
/// fold). One generation deeper, a grandchild's cold-vs-self-resume identity
/// must hold on a cache captured by a pass that was itself resumed. A capture
/// that dropped or duplicated any member of the nested state moves the bound
/// by O(1).
#[ntest::timeout(1200000)]
#[test]
fn warm_frontier_review_resumed_capture_reproduces_the_resumed_bound() {
    let mut checked = 0usize;
    let mut mismatches = Vec::new();
    for dag in descent_dags() {
        let graph = dag.graph();
        for site in SITES {
            let verifier = BetaCrownVerifier::new(config(true));
            let (_, root) = run_site(&verifier, site, &graph, &GraphSplitHistory::new(), None);
            let root = root.expect("root cache");
            for (h1, _) in sample(extensions(&dag, &graph, &[]), 8) {
                let branch = h1.last().expect("split").node_name.clone();
                let before = verifier.la_warm_frontier_counts().resumed();
                let (first, c1) = run_site(&verifier, site, &graph, &history_of(&h1), Some(&root));
                if verifier.la_warm_frontier_counts().resumed() == before {
                    continue; // a chain split or a missing frontier: not this family
                }
                let c1 = c1.expect("an armed capturing resumed pass returns a cache");
                assert!(
                    c1.frontier(&branch).is_some(),
                    "{}/{site:?}/{branch}: a resumed pass did not re-capture its seed frontier",
                    dag.shape
                );
                let before = verifier.la_warm_frontier_counts().resumed();
                let (again, _) = run_site(&verifier, site, &graph, &history_of(&h1), Some(&c1));
                assert_eq!(
                    verifier.la_warm_frontier_counts().resumed(),
                    before + 1,
                    "{}/{site:?}/{branch}: the re-captured frontier was not used",
                    dag.shape
                );
                for (row, ((fl, fu), (al, au))) in first.iter().zip(&again).enumerate() {
                    let tol = 1e-4 * (1.0 + fl.abs().max(fu.abs()));
                    if (fl - al).abs() > tol || (fu - au).abs() > tol {
                        mismatches.push(format!(
                            "{}/{site:?}/{branch} row {row}: first [{fl}, {fu}] vs re-seeded [{al}, {au}]",
                            dag.shape
                        ));
                    }
                }
                // One generation deeper: every grandchild of this resumed
                // child, cold with capture vs resumed from its OWN capture.
                for (h2, _) in sample(extensions(&dag, &graph, &h1), 3) {
                    let branch2 = h2.last().expect("split").node_name.clone();
                    let (cold2, own2) =
                        run_site(&verifier, site, &graph, &history_of(&h2), Some(&c1));
                    let Some(own2) = own2 else { continue };
                    if own2.frontier(&branch2).is_none() {
                        continue;
                    }
                    let before = verifier.la_warm_frontier_counts().resumed();
                    let (self2, _) =
                        run_site(&verifier, site, &graph, &history_of(&h2), Some(&own2));
                    if verifier.la_warm_frontier_counts().resumed() == before {
                        continue;
                    }
                    for (row, ((cl, cu), (wl, wu))) in cold2.iter().zip(&self2).enumerate() {
                        let tol = 1e-4 * (1.0 + cl.abs().max(cu.abs()));
                        if (cl - wl).abs() > tol || (cu - wu).abs() > tol {
                            mismatches.push(format!(
                                "{}/{site:?}/{branch}->{branch2} row {row}: seeded [{cl}, {cu}] vs \
                                 self-resumed [{wl}, {wu}]",
                                dag.shape
                            ));
                        }
                    }
                    checked += 1;
                }
                checked += 1;
            }
        }
    }
    println!(
        "[la-warm-frontier] review_resumed_capture checked={checked} mismatches={}",
        mismatches.len()
    );
    assert!(
        mismatches.is_empty(),
        "a resumed pass's capture is not its complete state: {mismatches:#?}"
    );
    assert!(
        checked >= 60,
        "too few resumed-capture cases ran: {checked}"
    );
    println!("[la-warm-frontier] TEST_COMPLETE name=warm_frontier_review_resumed_capture_reproduces_the_resumed_bound");
}

/// `(outcome, resumed)` of the multi-objective lane on TWO objectives (`out_0`
/// and `out_1` of a two-head DAG), so the root's multi-row capture is SLICED
/// per objective (`split_multi_row`), unioned per node and — batched —
/// re-STACKED for the spec-matrix pass: every row of every member must stay
/// paired with its own objective through that round trip.
fn multi_objective_two_heads(
    graph: &GraphNetwork,
    thresholds: [f32; 2],
    batched: bool,
    armed: bool,
) -> (Outcome, u64) {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        batch_size: if batched { 4 } else { 1 },
        ..bab_config(armed)
    });
    let naive = NaiveCpuGemmEngine;
    let engine: Option<&dyn ny_core::GemmEngine> = if batched { Some(&naive) } else { None };
    let result = verifier.verify_graph_relu_split_multi_objective_with_engine(
        graph,
        &unit_box(),
        &[vec![1.0, 0.0], vec![0.0, 1.0]],
        &thresholds,
        engine,
        None,
    );
    (
        outcome(result),
        verifier.la_warm_frontier_counts().resumed(),
    )
}

/// END TO END on two objectives: for each head `k`, a threshold 0.05 ABOVE
/// that head's enumerated minimum makes the property FALSE (the other head
/// keeps a true threshold); the lane must never answer `Verified`, in either
/// mode, armed or dark; the easy both-true thresholds must verify somewhere
/// (vacuity control); and the armed lane must have resumed children from
/// frontiers. TRUE rows that dark verifies and armed does not are printed as
/// `armed_lost` (a completeness cost of resuming above-N relaxations, not a
/// soundness failure).
#[ntest::timeout(900000)]
#[test]
fn warm_frontier_review_two_objective_rows_never_verify_a_false_head() {
    let dags = [
        double_residual(5, 2),
        double_residual(41, 2),
        three_consumer(7, 2),
        three_consumer(59, 2),
    ];
    let mut unsound = Vec::new();
    let mut errored = Vec::new();
    let mut vacuous = Vec::new();
    let mut resumed_total = 0u64;
    let mut armed_lost = Vec::new();
    for dag in &dags {
        let graph = dag.graph();
        let min0 = dag.range_of(&[], &[1.0, 0.0]).expect("range").0 as f32;
        let min1 = dag.range_of(&[], &[0.0, 1.0]).expect("range").0 as f32;
        // TRUE_EASY (1.0 below each head's grid minimum) is the vacuity
        // control: the lane must verify SOMETHING on every DAG, or "never
        // Verified on a FALSE head" proves nothing. TRUE (0.25 below) is
        // recorded only: on some seeds no arm closes it within the domain
        // limit (double_residual seed 41 hits `Domain limit 200` dark too).
        let cases = [
            ("FALSE0", [min0 + 0.05, min1 - 1.0], false),
            ("FALSE1", [min0 - 1.0, min1 + 0.05], false),
            ("TRUE", [min0 - 0.25, min1 - 0.25], true),
            ("TRUE_EASY", [min0 - 1.0, min1 - 1.0], true),
        ];
        let mut true_verified = 0usize;
        for (label, thresholds, truth) in cases {
            for batched in [false, true] {
                let mut dark_verified = false;
                for armed in [false, true] {
                    let (o, resumed) =
                        multi_objective_two_heads(&graph, thresholds, batched, armed);
                    if armed {
                        resumed_total += resumed;
                        // Completeness, not soundness: a resumed child keeps
                        // its ancestor's relaxations above the branch node, so
                        // the armed tree can be deeper than the dark one.
                        if truth && dark_verified && !o.verified {
                            armed_lost.push(format!(
                                "{}/{label}/batched={batched}: dark Verified, armed {}",
                                dag.shape, o.label
                            ));
                        }
                    } else {
                        dark_verified = o.verified;
                    }
                    println!(
                        "[la-warm-frontier] ROW lane=multi_objective_two_heads net={} case={label} \
                         batched={batched} armed={armed} resumed={resumed} {}",
                        dag.shape, o.label
                    );
                    if o.verified && !truth {
                        unsound.push(format!(
                            "{}/{label}/batched={batched}/armed={armed}: {}",
                            dag.shape, o.label
                        ));
                    }
                    if o.errored {
                        errored.push(format!(
                            "{}/{label}/batched={batched}/armed={armed}: {}",
                            dag.shape, o.label
                        ));
                    }
                    if truth {
                        true_verified += usize::from(o.verified);
                    }
                }
            }
        }
        if true_verified == 0 {
            vacuous.push(format!("{}: TRUE verified in no arm", dag.shape));
        }
    }
    println!(
        "[la-warm-frontier] review_two_heads resumed_total={resumed_total} armed_lost={} {armed_lost:?}",
        armed_lost.len()
    );
    assert!(
        unsound.is_empty(),
        "FALSE head reported Verified (UNSOUND): {unsound:#?}"
    );
    assert!(errored.is_empty(), "lane errored: {errored:#?}");
    assert!(vacuous.is_empty(), "vacuous: {vacuous:#?}");
    assert!(
        resumed_total > 0,
        "no child resumed from a frontier on the two-objective lane: the row \
         slicing / stacking round trip is vacuous"
    );
    println!("[la-warm-frontier] TEST_COMPLETE name=warm_frontier_review_two_objective_rows_never_verify_a_false_head");
}

// ---------------------------------------------------------------------------
// Review (adversarial): every capture site records the SAME frontier structure
// ---------------------------------------------------------------------------
//
// The self-resume family proves completeness only at the four SEED sites,
// whose captures it re-reads. The multi-objective ROOT pass
// (`spec_propagation/core.rs`) is a fifth, independent CAPTURE site that no
// seed-site family exercises: its frontiers are consumed only by the root's
// children inside the end-to-end lanes. All five passes walk the same
// reverse dispatch order (`exec_order` reversed), so the set of nodes still
// pending when a pass pops `N` is a property of the GRAPH, not of the pass's
// relaxation. A capture site that drops a pending skip block or the
// network-input channel therefore shows up as a structural disagreement with
// the other four, deterministically, independent of whether the dropped term
// happens to loosen or tighten a bound.

/// `(key, cols)` of every member of `frontier` that is not identically zero
/// (a site may or may not deposit an all-zero block; a dropped NON-zero
/// member is what makes a frontier incomplete).
fn frontier_structure(frontier: &crate::batched_domain::WarmFrontier) -> Vec<(String, usize)> {
    let mut keys: Vec<(String, usize)> = frontier
        .members()
        .filter(|(_, block)| {
            let lb = block.to_linear_bounds().expect("rebuild a stored member");
            !(lb.lower_a().iter().all(|&v| v == 0.0)
                && lb.upper_a().iter().all(|&v| v == 0.0)
                && lb.lower_b().iter().all(|&v| v == 0.0)
                && lb.upper_b().iter().all(|&v| v == 0.0))
        })
        .map(|(key, block)| (key.to_string(), block.cols()))
        .collect();
    keys.sort();
    keys
}

/// The multi-objective ROOT spec pass exactly as `multi_objective/root.rs`
/// issues it (historical request: precomputed node bounds, cache capture,
/// the verifier's counters handed over only while armed), on the two-row
/// spec `[+out; -out]`.
fn root_spec_cache(verifier: &BetaCrownVerifier, graph: &GraphNetwork) -> CachedLinearBounds {
    let input = unit_box();
    let base: HashMap<String, BoundedTensor> = graph.collect_node_bounds(&input).expect("bounds");
    let spec = spec_matrix();
    let (_out, cache) = crate::network::SpecCrownRequest::new(graph, &input, &spec, None)
        .node_bounds(&base)
        .capture_cache()
        .warm_frontier_opt(verifier.la_warm_frontier_armed())
        .run_with_cache()
        .expect("root spec-guided CROWN");
    cache.expect("the root spec pass published a cache")
}

/// CROSS-SITE STRUCTURE. On every DAG (both families), the frontier each of
/// the five capture sites records at every guarded branch node, on the root
/// domain, must name the same non-zero members with the same widths as the
/// constrained pass's. The positive control requires comparisons in which the
/// reference carries a NON-zero network-input channel (the term the
/// #warm-dag defect dropped) and in which the root spec pass recorded a
/// frontier at all.
#[ntest::timeout(600000)]
#[test]
fn warm_frontier_review_every_capture_site_records_the_same_frontier_structure() {
    let mut compared = 0usize;
    let mut with_input_channel = 0usize;
    let mut root_spec_frontiers = 0usize;
    let mut mismatches = Vec::new();
    let mut all = dags();
    all.extend(descent_dags());
    for dag in all {
        let graph = dag.graph();
        let verifier = BetaCrownVerifier::new(config(true));
        let root = GraphSplitHistory::new();
        let mut sources: Vec<(String, CachedLinearBounds)> = SITES
            .iter()
            .map(|&site| {
                let (_, cache) = run_site(&verifier, site, &graph, &root, None);
                (
                    format!("{site:?}"),
                    cache.expect("an armed capturing root pass returns a cache"),
                )
            })
            .collect();
        sources.push(("RootSpec".to_string(), root_spec_cache(&verifier, &graph)));
        root_spec_frontiers += sources[4].1.frontier_count();
        for (relu, _) in guarded_relus(&dag, &graph) {
            let Some(reference) = sources[0].1.frontier(relu) else {
                continue;
            };
            let expected = frontier_structure(reference);
            if expected.iter().any(|(key, _)| key == NETWORK_INPUT) {
                with_input_channel += 1;
            }
            for (name, cache) in &sources[1..] {
                match cache.frontier(relu) {
                    Some(frontier) => {
                        let got = frontier_structure(frontier);
                        if got != expected {
                            mismatches.push(format!(
                                "{}/{relu}: {name} {got:?} vs Constrained {expected:?}",
                                dag.shape
                            ));
                        }
                    }
                    None => mismatches.push(format!(
                        "{}/{relu}: {name} recorded no frontier (Constrained {expected:?})",
                        dag.shape
                    )),
                }
                compared += 1;
            }
        }
    }
    println!(
        "[la-warm-frontier] review_cross_site compared={compared} with_input_channel={with_input_channel} \
         root_spec_frontiers={root_spec_frontiers} mismatches={}",
        mismatches.len()
    );
    assert!(
        mismatches.is_empty(),
        "capture sites disagree on the pending frontier (an incomplete capture): {mismatches:#?}"
    );
    assert!(
        compared >= 100,
        "too few cross-site comparisons: {compared}"
    );
    assert!(
        with_input_channel > 0,
        "no comparison carried a non-zero input channel: the #warm-dag term is unexercised"
    );
    assert!(
        root_spec_frontiers > 0,
        "the root spec pass recorded no frontier: its capture site is unexercised"
    );
    println!("[la-warm-frontier] TEST_COMPLETE name=warm_frontier_review_every_capture_site_records_the_same_frontier_structure");
}

/// ROOT-SPEC PARENT -> CHILD VALIDITY. The multi-objective lane's depth-1
/// children resume from the ROOT spec pass's frontiers (sliced per objective
/// or kept two-row). Every guarded child, both phases, every neuron, resumed
/// at the two-row constrained spec site from the root spec cache, must
/// enclose the enumerated range of `[+out; -out]` over its region, and must
/// really resume.
#[ntest::timeout(600000)]
#[test]
fn warm_frontier_review_root_spec_frontier_resumes_children_soundly() {
    let mut checked = 0usize;
    let mut resumed = 0u64;
    let mut unsound = Vec::new();
    let mut all = dags();
    all.extend(descent_dags());
    for dag in all {
        let graph = dag.graph();
        let verifier = BetaCrownVerifier::new(config(true));
        let root = root_spec_cache(&verifier, &graph);
        for splits in children(&dag, &graph) {
            if !child_is_buildable(&graph, &splits) {
                continue;
            }
            let Some(range) = dag.range(&splits, 1.0) else {
                continue;
            };
            let before = verifier.la_warm_frontier_counts().resumed();
            let (bounds, _) = run_site(
                &verifier,
                Site::ConstrainedSpec,
                &graph,
                &history_of(&splits),
                Some(&root),
            );
            let did = verifier.la_warm_frontier_counts().resumed() > before;
            resumed += u64::from(did);
            if let Err(why) = encloses(Site::ConstrainedSpec, range, &bounds) {
                unsound.push(format!(
                    "{}/{:?} resumed={did}: {why}",
                    dag.shape,
                    splits
                        .iter()
                        .map(|s| (s.node_name.as_str(), s.neuron_idx, s.is_active))
                        .collect::<Vec<_>>()
                ));
            }
            checked += 1;
        }
    }
    println!(
        "[la-warm-frontier] review_root_spec checked={checked} resumed={resumed} unsound={}",
        unsound.len()
    );
    assert!(
        unsound.is_empty(),
        "UNSOUND child bound resumed from the ROOT spec frontier: {unsound:#?}"
    );
    assert!(checked >= 100, "too few root-spec children: {checked}");
    assert!(
        resumed > 0,
        "no child resumed from a root spec frontier: the root capture site is unexercised"
    );
    println!("[la-warm-frontier] TEST_COMPLETE name=warm_frontier_review_root_spec_frontier_resumes_children_soundly");
}
