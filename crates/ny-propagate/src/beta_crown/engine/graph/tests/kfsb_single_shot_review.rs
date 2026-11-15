// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! review(dag-throughput): INDEPENDENT enumeration-exact soundness checks of
//! `bab.kfsb_single_shot` on small residual DAGs the builder's fixtures do not
//! contain — two chained residual Adds in a multi-channel conv block, a dense
//! two-block residual, and a three-way fan-in — each with a TWO-output head so
//! the conjunctive lane carries several atoms, as yolo's 5-atom spec does.
//! Ground truth is an f64 evaluation of every node on an 81 x 81 grid of the
//! 2-D input box (dyadic weights; a FALSE property always has a concrete grid
//! witness with >= 1e-3 slack, so "never Verified" is an exact obligation).
//!
//! * DOMAINLIST LANE (`verify_graph_gpu_domain_list`, the second production
//!   caller, which the builder never ran armed): a FALSE property is never
//!   `Verified`, armed or dark, in both bound modes; armed reaches the
//!   single-shot scorer and dark never does.
//! * CONJUNCTIVE MULTI-ATOM LANE: a FALSE conjunction (a grid witness in the
//!   unsafe intersection) is never `Verified`, with and without an engine.
//! * EVERY DOMAIN ON AN ARMED-PICK TREE: a BaB tree grown with the ARMED
//!   selector's picks and bounded by the unchanged child pipeline
//!   (`evaluate_graph_child_bounds`) has, on every domain, an objective bound
//!   AND every intermediate node bound that encloses the enumerated values on
//!   that domain's region; every armed pick is an unstable, unconstrained
//!   neuron of its domain; every `Ok(None)` (empty) child is really empty.
//! * PARALLEL = SERIAL: the rayon fan-out of candidate simulations returns
//!   bit-identical child bounds and the same pick on 1 thread and on 8.

use std::collections::HashMap;
use std::time::Duration;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{GemmEngine, NaiveCpuGemmEngine};

use crate::beta_crown::engine::branching::kfsb_shared::GraphKfsbCandidate;
use crate::beta_crown::engine::graph::gpu_bab::kfsb_single_shot_calls_for_tests;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::beta_crown::{
    BabVerificationStatus, BetaCrownConfig, BranchingHeuristic, GraphBabDomain,
    GraphNeuronConstraint,
};
use crate::layers::{Conv2dLayer, FlattenLayer};
use crate::{
    AddLayer, BoundedTensor, GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer, NETWORK_INPUT,
};

// ---------------------------------------------------------------------------
// DAG language with an exact f64 evaluator over every node.
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Op {
    Lin {
        src: &'static str,
        w: Array2<f32>,
        b: Array1<f32>,
    },
    /// 1-row Conv2d, kernel `[out_c, in_c, 1, kw]`, stride 1, column pad.
    Conv {
        src: &'static str,
        kernel: Vec<f32>,
        bias: Vec<f32>,
        out_c: usize,
        in_c: usize,
        kw: usize,
        pad: usize,
        width: usize,
    },
    Relu(&'static str),
    Add(&'static str, &'static str),
    Flatten(&'static str),
}

#[derive(Clone)]
struct Dag {
    shape: String,
    input_shape: Vec<usize>,
    nodes: Vec<(&'static str, Op)>,
    output: &'static str,
}

const GRID: usize = 80;

impl Dag {
    fn graph(&self) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        for (name, op) in &self.nodes {
            let (layer, inputs): (Layer, Vec<String>) = match op {
                Op::Lin { src, w, b } => (
                    Layer::Linear(LinearLayer::new(w.clone(), Some(b.clone())).expect("linear")),
                    vec![src.to_string()],
                ),
                Op::Conv {
                    src,
                    kernel,
                    bias,
                    out_c,
                    in_c,
                    kw,
                    pad,
                    width,
                } => {
                    let kernel =
                        ArrayD::from_shape_vec(IxDyn(&[*out_c, *in_c, 1, *kw]), kernel.clone())
                            .expect("kernel shape");
                    let conv = Conv2dLayer::with_input_shape(
                        kernel,
                        Some(Array1::from_vec(bias.clone())),
                        (1, 1),
                        (0, *pad),
                        1,
                        *width,
                    )
                    .expect("conv");
                    (Layer::Conv2d(conv), vec![src.to_string()])
                }
                Op::Relu(src) => (Layer::ReLU(ReLULayer), vec![src.to_string()]),
                Op::Add(a, b) => (Layer::Add(AddLayer), vec![a.to_string(), b.to_string()]),
                Op::Flatten(src) => (Layer::Flatten(FlattenLayer::new(0)), vec![src.to_string()]),
            };
            let node = if inputs.len() == 1 && inputs[0] == NETWORK_INPUT {
                GraphNode::from_input(*name, layer)
            } else {
                GraphNode::new(*name, layer, inputs)
            };
            graph.add_node(node);
        }
        graph.set_output(self.output);
        graph
    }

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
                Op::Conv {
                    src,
                    kernel,
                    bias,
                    out_c,
                    in_c,
                    kw,
                    pad,
                    width,
                } => {
                    let input = &values[src];
                    let out_w = width + 2 * pad + 1 - kw;
                    let mut out = Vec::with_capacity(out_c * out_w);
                    for o in 0..*out_c {
                        for col in 0..out_w {
                            let mut acc = f64::from(bias[o]);
                            for c in 0..*in_c {
                                for k in 0..*kw {
                                    let src_col = col as isize + k as isize - *pad as isize;
                                    if src_col < 0 || src_col as usize >= *width {
                                        continue;
                                    }
                                    let weight = kernel[((o * in_c + c) * kw) + k];
                                    acc += f64::from(weight) * input[c * width + src_col as usize];
                                }
                            }
                            out.push(acc);
                        }
                    }
                    out
                }
                Op::Relu(src) => values[src].iter().map(|v| v.max(0.0)).collect(),
                Op::Add(a, b) => values[a]
                    .iter()
                    .zip(&values[b])
                    .map(|(p, q)| p + q)
                    .collect(),
                Op::Flatten(src) => values[src].clone(),
            };
            values.insert(*name, value);
        }
        values
    }

    /// Every node's exact value at every grid point (computed once per DAG).
    fn grid(&self) -> Vec<HashMap<&'static str, Vec<f64>>> {
        let mut all = Vec::with_capacity((GRID + 1) * (GRID + 1));
        for i in 0..=GRID {
            for j in 0..=GRID {
                let x = [
                    -1.0 + 2.0 * i as f64 / GRID as f64,
                    -1.0 + 2.0 * j as f64 / GRID as f64,
                ];
                all.push(self.eval(x));
            }
        }
        all
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

    fn relu_nodes(&self) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|(_, op)| matches!(op, Op::Relu(_)))
            .map(|(name, _)| name.to_string())
            .collect()
    }

    fn input_box(&self) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&self.input_shape), -1.0f32),
            ArrayD::from_elem(IxDyn(&self.input_shape), 1.0f32),
        )
        .expect("input box")
    }

    /// Whether grid point `values` lies in the CLOSED region cut by `splits`.
    fn inside(
        &self,
        values: &HashMap<&'static str, Vec<f64>>,
        splits: &[GraphNeuronConstraint],
    ) -> bool {
        splits.iter().all(|split| {
            let z = values[self.relu_input(&split.node_name)][split.neuron_idx];
            if split.is_active {
                z >= 0.0
            } else {
                z <= 0.0
            }
        })
    }
}

fn dot(row: &[f32], out: &[f64]) -> f64 {
    row.iter().zip(out).map(|(c, y)| f64::from(*c) * y).sum()
}

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

    fn vec(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next() * 0.5).collect()
    }
}

fn lin(src: &'static str, rng: &mut Lcg, rows: usize, cols: usize) -> Op {
    Op::Lin {
        src,
        w: rng.mat(rows, cols),
        b: Array1::from_vec(rng.vec(rows)),
    }
}

fn conv(src: &'static str, rng: &mut Lcg, out_c: usize, in_c: usize, kw: usize, pad: usize) -> Op {
    Op::Conv {
        src,
        kernel: (0..out_c * in_c * kw).map(|_| rng.next()).collect(),
        bias: rng.vec(out_c),
        out_c,
        in_c,
        kw,
        pad,
        width: 2,
    }
}

/// Two CHAINED residual Adds over a 3-channel conv trunk, a ReLU after the
/// residual sum (ResNet order) and a 2-channel head:
/// `c0 = conv1x1(x); r0; c1 = conv1x3(r0); r1; a1 = r1 + c0; c2 = conv1x3(a1);
///  r2; a2 = r2 + a1; r3 = relu(a2); c3 = conv1x2(r3) [2 ch]; out = flatten`.
fn conv_res2(seed: u64) -> Dag {
    let mut r = Lcg(seed);
    Dag {
        shape: format!("conv_res2/{seed:#x}"),
        input_shape: vec![1, 1, 2],
        nodes: vec![
            ("c0", conv(NETWORK_INPUT, &mut r, 3, 1, 1, 0)),
            ("r0", Op::Relu("c0")),
            ("c1", conv("r0", &mut r, 3, 3, 3, 1)),
            ("r1", Op::Relu("c1")),
            ("a1", Op::Add("r1", "c0")),
            ("c2", conv("a1", &mut r, 3, 3, 3, 1)),
            ("r2", Op::Relu("c2")),
            ("a2", Op::Add("r2", "a1")),
            ("r3", Op::Relu("a2")),
            ("c3", conv("r3", &mut r, 2, 3, 2, 0)),
            ("out", Op::Flatten("c3")),
        ],
        output: "out",
    }
}

/// Dense two-block residual with a skip carried across both blocks.
fn dense_res2(seed: u64) -> Dag {
    let mut r = Lcg(seed);
    Dag {
        shape: format!("dense_res2/{seed:#x}"),
        input_shape: vec![2],
        nodes: vec![
            ("l0", lin(NETWORK_INPUT, &mut r, 4, 2)),
            ("r0", Op::Relu("l0")),
            ("l1", lin("r0", &mut r, 4, 4)),
            ("r1", Op::Relu("l1")),
            ("a1", Op::Add("r1", "l0")),
            ("ra", Op::Relu("a1")),
            ("l2", lin("ra", &mut r, 4, 4)),
            ("r2", Op::Relu("l2")),
            ("a2", Op::Add("r2", "a1")),
            ("out", lin("a2", &mut r, 2, 4)),
        ],
        output: "out",
    }
}

/// Three-way fan-in: two parallel ReLU branches and the pre-activation skip
/// summed by chained Adds before a ReLU.
fn fan_in3(seed: u64) -> Dag {
    let mut r = Lcg(seed);
    Dag {
        shape: format!("fan_in3/{seed:#x}"),
        input_shape: vec![2],
        nodes: vec![
            ("l0", lin(NETWORK_INPUT, &mut r, 3, 2)),
            ("r0", Op::Relu("l0")),
            ("la", lin("r0", &mut r, 3, 3)),
            ("ra", Op::Relu("la")),
            ("lb", lin("r0", &mut r, 3, 3)),
            ("rb", Op::Relu("lb")),
            ("s1", Op::Add("ra", "rb")),
            ("s2", Op::Add("s1", "l0")),
            ("rs", Op::Relu("s2")),
            ("out", lin("rs", &mut r, 2, 3)),
        ],
        output: "out",
    }
}

fn dags() -> Vec<Dag> {
    let mut all = Vec::new();
    for seed in [5u64, 17, 101] {
        all.push(conv_res2(seed ^ 0x51ed));
        all.push(dense_res2(seed ^ 0x0b0e));
        all.push(fan_in3(seed ^ 0x3c3c));
    }
    all
}

/// The two output rows and their difference.
const ROWS: [[f32; 2]; 3] = [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]];

fn kfsb_config(armed: bool) -> BetaCrownConfig {
    BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::Kfsb,
        fsb_candidates: 3,
        use_multi_objective_critical_kfsb: true,
        kfsb_single_shot: armed,
        enable_cuts: false,
        timeout: Duration::from_secs(8),
        max_domains: 200,
        max_depth: 12,
        ..Default::default()
    }
}

fn grid_range(grid: &[HashMap<&'static str, Vec<f64>>], output: &str, row: &[f32]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for values in grid {
        let f = dot(row, &values[output]);
        lo = lo.min(f);
        hi = hi.max(f);
    }
    (lo, hi)
}

// ---------------------------------------------------------------------------
// 1. DomainList lane (`batched_gpu.rs` caller) end to end.
// ---------------------------------------------------------------------------

#[ntest::timeout(1200000)]
#[test]
fn review_kfsb_single_shot_domain_list_lane_never_verifies_a_false_property() {
    let engine = NaiveCpuGemmEngine;
    let mut unsound = Vec::new();
    let mut errored = Vec::new();
    let mut armed_calls = 0u64;
    let mut dark_calls = 0u64;
    let mut false_rows_branched = [0usize; 2];
    let mut false_rows = 0usize;
    let mut true_closes = [0usize; 2];
    let mut true_rows = 0usize;
    for dag in dags() {
        let graph = dag.graph();
        let input = dag.input_box();
        let grid = dag.grid();
        for row in &ROWS[..2] {
            let (lo, hi) = grid_range(&grid, dag.output, row);
            let width = (hi - lo).max(1e-3);
            for verify_upper in [true, false] {
                // FALSE: a grid point violates by >= 1e-3. TRUE: margin beyond
                // the grid extreme (reported only — the grid is not exhaustive).
                let delta = (0.02 * width).max(1e-3);
                let (false_t, true_t) = if verify_upper {
                    (hi - delta, hi + 0.1 * width + 0.05)
                } else {
                    (lo + delta, lo - 0.1 * width - 0.05)
                };
                for (truth, threshold) in [("FALSE", false_t), ("TRUE", true_t)] {
                    if truth == "FALSE" {
                        false_rows += 1;
                    } else {
                        true_rows += 1;
                    }
                    for armed in [false, true] {
                        let verifier = BetaCrownVerifier::new(BetaCrownConfig {
                            verify_upper_bound: verify_upper,
                            batch_size: 4,
                            ..kfsb_config(armed)
                        });
                        let before = kfsb_single_shot_calls_for_tests();
                        let result = verifier.verify_graph_gpu_domain_list(
                            &graph,
                            &input,
                            row,
                            threshold as f32,
                            Some(&engine as &dyn GemmEngine),
                            None,
                        );
                        let calls = kfsb_single_shot_calls_for_tests() - before;
                        if armed {
                            armed_calls += calls;
                        } else {
                            dark_calls += calls;
                        }
                        let label = format!(
                            "{} row={row:?} upper={verify_upper} {truth} t={threshold:.4} \
                             grid=[{lo:.4},{hi:.4}] armed={armed} calls={calls}",
                            dag.shape
                        );
                        match result {
                            Ok(result) => {
                                let verified =
                                    matches!(result.result, BabVerificationStatus::Verified);
                                println!(
                                    "[kfsb-review] DOMAINLIST {label}: {:?} explored={} depth={}",
                                    result.result,
                                    result.domains_explored,
                                    result.max_depth_reached
                                );
                                if truth == "FALSE" {
                                    if verified {
                                        unsound.push(label);
                                    }
                                    if result.max_depth_reached >= 1 {
                                        false_rows_branched[usize::from(armed)] += 1;
                                    }
                                } else if verified {
                                    true_closes[usize::from(armed)] += 1;
                                }
                            }
                            Err(error) => errored.push(format!("{label}: Err({error})")),
                        }
                    }
                }
            }
        }
    }
    println!(
        "[kfsb-review] DOMAINLIST summary false_rows={false_rows} branched dark={} armed={} \
         true_rows={true_rows} true_closes dark={} armed={} armed_calls={armed_calls} \
         dark_calls={dark_calls} errored={}",
        false_rows_branched[0],
        false_rows_branched[1],
        true_closes[0],
        true_closes[1],
        errored.len()
    );
    assert!(
        unsound.is_empty(),
        "DomainList lane reported Verified on a FALSE property (UNSOUND): {unsound:#?}"
    );
    assert!(errored.is_empty(), "DomainList lane errored: {errored:#?}");
    assert_eq!(
        dark_calls, 0,
        "dark must never reach the single-shot scorer"
    );
    assert!(
        armed_calls > 0,
        "VOID: the armed DomainList lane never reached the single-shot scorer"
    );
    assert!(
        false_rows_branched[1] * 2 >= false_rows,
        "VOID-ish: fewer than half the FALSE rows branched armed ({} of {false_rows})",
        false_rows_branched[1]
    );
    println!("[kfsb-review] TEST_COMPLETE name=review_kfsb_single_shot_domain_list_lane_never_verifies_a_false_property");
}

// ---------------------------------------------------------------------------
// 2. Conjunctive multi-atom lane (`branching/graph.rs` critical route).
// ---------------------------------------------------------------------------

#[ntest::timeout(1200000)]
#[test]
fn review_kfsb_single_shot_multi_atom_conjunction_never_verifies_a_false_property() {
    let engine = NaiveCpuGemmEngine;
    let objectives: Vec<Vec<f32>> = ROWS.iter().map(|r| r.to_vec()).collect();
    let mut unsound = Vec::new();
    let mut errored = Vec::new();
    let mut armed_calls = 0u64;
    let mut dark_calls = 0u64;
    let mut false_cases = 0usize;
    let mut false_branched_armed = 0usize;
    let mut true_closes = [0usize; 2];
    let mut true_cases = 0usize;
    for dag in dags() {
        let graph = dag.graph();
        let input = dag.input_box();
        let grid = dag.grid();
        let atoms = |values: &HashMap<&'static str, Vec<f64>>| -> Vec<f64> {
            ROWS.iter().map(|r| dot(r, &values[dag.output])).collect()
        };
        // Witnesses: the grid point minimising the summed atoms (a corner of
        // the unsafe intersection: tight, so the lane has to branch) and a
        // pseudo-random interior grid point.
        let corner = grid
            .iter()
            .min_by(|a, b| {
                let sa: f64 = atoms(a).iter().sum();
                let sb: f64 = atoms(b).iter().sum();
                sa.total_cmp(&sb)
            })
            .expect("grid");
        let interior = &grid[(GRID + 1) * (GRID / 3) + (2 * GRID / 3)];
        let mut cases: Vec<(&str, Vec<f32>)> = Vec::new();
        for (tag, witness) in [("corner", corner), ("interior", interior)] {
            // Unsafe region = { x : atom_i(x) <= t_i for all i }; the witness is
            // in it with >= 1e-3 slack on every atom, so the property is FALSE.
            let t: Vec<f32> = atoms(witness)
                .iter()
                .map(|v| {
                    let t = (v + 1e-3) as f32;
                    assert!(f64::from(t) >= v + 5e-4, "threshold rounding ate the slack");
                    t
                })
                .collect();
            cases.push((tag, t));
        }
        // TRUE (report only): atom 0 holds with margin on the grid.
        let (lo0, hi0) = grid_range(&grid, dag.output, &ROWS[0]);
        cases.push((
            "TRUE",
            vec![(lo0 - 0.1 * (hi0 - lo0) - 0.05) as f32, 1.0e6, 1.0e6],
        ));
        for (tag, thresholds) in cases {
            let is_false = tag != "TRUE";
            if is_false {
                false_cases += 1;
            } else {
                true_cases += 1;
            }
            for (armed, with_engine) in [(false, true), (true, true), (true, false)] {
                let verifier = BetaCrownVerifier::new(kfsb_config(armed));
                let before = kfsb_single_shot_calls_for_tests();
                let result = verifier
                    .verify_graph_relu_split_multi_objective_conjunctive_with_engine(
                        &graph,
                        &input,
                        &objectives,
                        &thresholds,
                        with_engine.then_some(&engine as &dyn GemmEngine),
                        None,
                    );
                let calls = kfsb_single_shot_calls_for_tests() - before;
                if armed {
                    armed_calls += calls;
                } else {
                    dark_calls += calls;
                }
                let label = format!(
                    "{} {tag} t={thresholds:?} armed={armed} engine={with_engine} calls={calls}",
                    dag.shape
                );
                match result {
                    Ok(result) => {
                        let verified = matches!(result.result, BabVerificationStatus::Verified);
                        println!(
                            "[kfsb-review] CONJ {label}: {:?} explored={} depth={}",
                            result.result, result.domains_explored, result.max_depth_reached
                        );
                        if is_false {
                            if verified {
                                unsound.push(label);
                            }
                            if armed && with_engine && result.max_depth_reached >= 1 {
                                false_branched_armed += 1;
                            }
                        } else if verified && with_engine {
                            true_closes[usize::from(armed)] += 1;
                        }
                    }
                    Err(error) => errored.push(format!("{label}: Err({error})")),
                }
            }
        }
    }
    println!(
        "[kfsb-review] CONJ summary false_cases={false_cases} false_branched_armed=\
         {false_branched_armed} true_cases={true_cases} true_closes dark={} armed={} \
         armed_calls={armed_calls} dark_calls={dark_calls} errored={}",
        true_closes[0],
        true_closes[1],
        errored.len()
    );
    assert!(
        unsound.is_empty(),
        "conjunctive lane reported Verified on a FALSE conjunction (UNSOUND): {unsound:#?}"
    );
    assert!(errored.is_empty(), "conjunctive lane errored: {errored:#?}");
    assert_eq!(
        dark_calls, 0,
        "dark must never reach the single-shot scorer"
    );
    assert!(
        armed_calls > 0,
        "VOID: the armed conjunctive lane never reached the single-shot scorer"
    );
    assert!(
        false_branched_armed > 0,
        "VOID: no FALSE conjunction branched on the armed lane"
    );
    println!("[kfsb-review] TEST_COMPLETE name=review_kfsb_single_shot_multi_atom_conjunction_never_verifies_a_false_property");
}

// ---------------------------------------------------------------------------
// 3. Every domain of an armed-pick BaB tree is bounded validly on its region.
// ---------------------------------------------------------------------------

#[ntest::timeout(1200000)]
#[test]
fn review_kfsb_single_shot_every_domain_on_armed_pick_trees_is_enclosed() {
    let engine = NaiveCpuGemmEngine;
    let tol = 1e-4;
    let max_depth = 5usize;
    let mut checked_objective = 0usize;
    let mut checked_nodes = 0usize;
    let mut empty_children = 0usize;
    let mut picks = 0usize;
    let mut violations = Vec::new();
    for dag in dags() {
        let graph = dag.graph();
        let input = dag.input_box();
        let grid = dag.grid();
        let relu_nodes = dag.relu_nodes();
        let node_bounds = graph.collect_node_bounds(&input).expect("ibp bounds");
        for (row_idx, row) in [ROWS[0].to_vec(), vec![-1.0, 0.0], ROWS[2].to_vec()]
            .into_iter()
            .enumerate()
        {
            for with_engine in [false, true] {
                let engine_ref = with_engine.then_some(&engine as &dyn GemmEngine);
                let verifier = BetaCrownVerifier::new(kfsb_config(true));
                let root = GraphBabDomain::root(node_bounds.clone(), -100.0, 100.0, &input, false)
                    .expect("root domain");
                let mut queue = vec![root];
                let mut domains = 0usize;
                while let Some(domain) = queue.pop() {
                    if domain.depth >= max_depth || domains >= 48 {
                        continue;
                    }
                    let unstable =
                        verifier.find_unstable_graph_neurons(&graph, &domain, &relu_nodes);
                    if unstable.is_empty() {
                        continue;
                    }
                    let (node, neuron, _score) = verifier
                        .select_graph_branch_kfsb_in_gpu_batched(
                            &graph, &domain, &unstable, &row, engine_ref,
                        )
                        .expect("armed selector");
                    picks += 1;
                    let tag = format!(
                        "{} row#{row_idx} engine={with_engine} depth={} pick={node}[{neuron}]",
                        dag.shape, domain.depth
                    );
                    if !unstable.iter().any(|(n, i)| *n == node && *i == neuron) {
                        violations.push(format!("{tag}: pick is not an unstable neuron"));
                        continue;
                    }
                    if domain
                        .history
                        .constraints
                        .iter()
                        .any(|c| c.node_name == node && c.neuron_idx == neuron)
                    {
                        violations.push(format!("{tag}: pick is already split"));
                        continue;
                    }
                    for is_active in [true, false] {
                        let constraint = GraphNeuronConstraint {
                            node_name: node.clone(),
                            neuron_idx: neuron,
                            is_active,
                            score: 0.0,
                        };
                        let mut splits = domain.history.constraints.clone();
                        splits.push(constraint.clone());
                        let region: Vec<&HashMap<&'static str, Vec<f64>>> =
                            grid.iter().filter(|v| dag.inside(v, &splits)).collect();
                        let Some(mut child) = domain
                            .with_constraint(&graph, constraint, false)
                            .expect("with_constraint")
                        else {
                            // EMPTY claim: no grid point may lie strictly on the
                            // excluded side of the parent's region.
                            empty_children += 1;
                            let pre = dag.relu_input(&node);
                            let parent: Vec<_> = grid
                                .iter()
                                .filter(|v| dag.inside(v, &domain.history.constraints))
                                .collect();
                            if parent.iter().any(|v| {
                                let z = v[pre][neuron];
                                if is_active {
                                    z > tol
                                } else {
                                    z < -tol
                                }
                            }) {
                                violations.push(format!(
                                    "{tag} active={is_active}: Ok(None) on a non-empty child"
                                ));
                            }
                            continue;
                        };
                        let bounded = verifier
                            .evaluate_graph_child_bounds(
                                &graph,
                                &mut child,
                                &domain.node_bounds,
                                &row,
                                None,
                                engine_ref,
                            )
                            .expect("child bounding");
                        if !bounded || region.is_empty() {
                            continue;
                        }
                        // Objective bound on the child's region.
                        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
                        for v in &region {
                            let f = dot(&row, &v[dag.output]);
                            lo = lo.min(f);
                            hi = hi.max(f);
                        }
                        checked_objective += 1;
                        if f64::from(child.lower_bound) > lo + tol
                            || f64::from(child.upper_bound) < hi - tol
                        {
                            violations.push(format!(
                                "{tag} active={is_active}: objective [{}, {}] misses grid \
                                 [{lo}, {hi}] over {} points",
                                child.lower_bound,
                                child.upper_bound,
                                region.len()
                            ));
                        }
                        // Every intermediate node bound on the child's region:
                        // enumerated per-entry min/max, then one comparison each.
                        for (name, bounds) in &child.node_bounds {
                            let flat = bounds.flatten();
                            let Some(key) = dag
                                .nodes
                                .iter()
                                .map(|(n, _)| *n)
                                .find(|n| *n == name.as_str())
                            else {
                                continue;
                            };
                            let mut mins = vec![f64::INFINITY; flat.len()];
                            let mut maxs = vec![f64::NEG_INFINITY; flat.len()];
                            for v in &region {
                                for (idx, z) in v[key].iter().enumerate().take(flat.len()) {
                                    mins[idx] = mins[idx].min(*z);
                                    maxs[idx] = maxs[idx].max(*z);
                                }
                            }
                            for idx in 0..flat.len() {
                                let (l, u) = (flat.lower()[[idx]], flat.upper()[[idx]]);
                                checked_nodes += 1;
                                let slack_lo = tol + 1e-5 * mins[idx].abs();
                                let slack_hi = tol + 1e-5 * maxs[idx].abs();
                                if f64::from(l) > mins[idx] + slack_lo
                                    || f64::from(u) < maxs[idx] - slack_hi
                                {
                                    violations.push(format!(
                                        "{tag} active={is_active}: node {name}[{idx}] bound \
                                         [{l}, {u}] misses enumerated [{}, {}]",
                                        mins[idx], maxs[idx]
                                    ));
                                }
                            }
                        }
                        domains += 1;
                        queue.push(child);
                    }
                }
            }
        }
    }
    println!(
        "[kfsb-review] TREE picks={picks} checked_objective={checked_objective} \
         checked_node_entries={checked_nodes} empty_children={empty_children} \
         violations={}",
        violations.len()
    );
    assert!(
        checked_objective > 200 && picks > 100,
        "positive control: too few armed-pick domains checked (objective={checked_objective}, \
         picks={picks})"
    );
    assert!(
        violations.is_empty(),
        "a domain on an armed-pick tree is not enclosed: {:#?}",
        &violations[..violations.len().min(20)]
    );
    println!("[kfsb-review] TEST_COMPLETE name=review_kfsb_single_shot_every_domain_on_armed_pick_trees_is_enclosed");
}

// ---------------------------------------------------------------------------
// 4. Parallel candidate simulation is bit-identical to a 1-thread run.
// ---------------------------------------------------------------------------

#[ntest::timeout(600000)]
#[test]
fn review_kfsb_single_shot_parallel_simulations_are_bit_identical_to_serial() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    let eight = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("8-thread pool");
    let mut compared = 0usize;
    let mut mismatches = Vec::new();
    for dag in dags() {
        let graph = dag.graph();
        let input = dag.input_box();
        let node_bounds = graph.collect_node_bounds(&input).expect("ibp bounds");
        let domain =
            GraphBabDomain::root(node_bounds, -100.0, 100.0, &input, false).expect("root domain");
        let verifier = BetaCrownVerifier::new(kfsb_config(true));
        let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &dag.relu_nodes());
        let candidates: Vec<GraphKfsbCandidate> = unstable
            .iter()
            .map(|(node, idx)| GraphKfsbCandidate {
                node_name: node.clone(),
                neuron_idx: *idx,
                main_score: 0.0,
                backup_score: 0.0,
            })
            .collect();
        for row in ROWS {
            let run = |pool: &rayon::ThreadPool| {
                pool.install(|| {
                    let sims = verifier
                        .kfsb_single_shot_child_bounds_for_tests(&graph, &domain, &row, &candidates)
                        .expect("simulations");
                    let pick = verifier
                        .select_graph_branch_kfsb_in_gpu_batched(
                            &graph, &domain, &unstable, &row, None,
                        )
                        .expect("pick");
                    (sims, pick)
                })
            };
            let (s1, p1) = run(&one);
            let (s8, p8) = run(&eight);
            let bits = |s: &[(Option<(f32, f32)>, Option<(f32, f32)>)]| -> Vec<Option<(u32, u32)>> {
                s.iter()
                    .flat_map(|(a, b)| [*a, *b])
                    .map(|side| side.map(|(l, u)| (l.to_bits(), u.to_bits())))
                    .collect()
            };
            compared += s1.len();
            if bits(&s1) != bits(&s8)
                || p1.0 != p8.0
                || p1.1 != p8.1
                || p1.2.to_bits() != p8.2.to_bits()
            {
                mismatches.push(format!(
                    "{} row={row:?}: 1-thread vs 8-thread differ",
                    dag.shape
                ));
            }
        }
    }
    println!(
        "[kfsb-review] PARALLEL compared_candidates={compared} mismatches={}",
        mismatches.len()
    );
    assert!(
        compared > 30,
        "positive control: too few candidates compared ({compared})"
    );
    assert!(
        mismatches.is_empty(),
        "parallel simulation is not deterministic: {mismatches:#?}"
    );
    println!("[kfsb-review] TEST_COMPLETE name=review_kfsb_single_shot_parallel_simulations_are_bit_identical_to_serial");
}

// ---------------------------------------------------------------------------
// 5. The armed PRESCORE ranks by the objective it is seeded with.
// ---------------------------------------------------------------------------

/// `kfsb_single_shot_dark_is_the_historical_pick_and_armed_follows_the_critical_row`
/// asserts that "the objective-seeded prescore must still follow the
/// conjunction-critical row", but its fixture has two unstable neurons and
/// `fsb_candidates: 2`, so BOTH are simulated and the prescore never decides
/// the pick: an identity-seeded prescore passes it too. This pins the
/// prescore itself on the same fixture. Zero bias, so the BaBSR main score is
/// `|min(c, 0)| * u * (-l) / (u - l)` with `c = row · W2`:
/// `x0 in [-3, 1]` (ratio 0.75) and `x1 in [-0.5, 0.5]` (ratio 0.25), so row
/// `[1, 0]` (c = [-5, -0.1]) ranks neuron 0 first (3.75 vs 0.025) and row
/// `[0, 1]` (c = [-0.1, -5]) ranks neuron 1 first (1.25 vs 0.075). The legacy
/// identity seed averages both rows (1.9125 vs 0.6375) and ranks neuron 0
/// first for EITHER row, so it fails the second case.
#[ntest::timeout(60000)]
#[test]
fn review_kfsb_single_shot_prescore_ranks_by_the_seeded_objective() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(
            LinearLayer::new(
                ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0]]),
                Some(ndarray::arr1(&[0.0, 0.0])),
            )
            .expect("linear1"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(
            LinearLayer::new(ndarray::arr2(&[[-5.0, -0.1], [-0.1, -5.0]]), None).expect("linear2"),
        ),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");
    let input = BoundedTensor::new(
        ndarray::arr1(&[-3.0, -0.5]).into_dyn(),
        ndarray::arr1(&[1.0, 0.5]).into_dyn(),
    )
    .expect("input");
    let node_bounds = graph.collect_node_bounds(&input).expect("ibp bounds");
    let domain =
        GraphBabDomain::root(node_bounds, -100.0, 100.0, &input, false).expect("root domain");
    let verifier = BetaCrownVerifier::new(kfsb_config(true));
    let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &["relu".to_string()]);
    assert_eq!(
        unstable.len(),
        2,
        "both neurons must be unstable: {unstable:?}"
    );
    for (row, expect) in [([1.0f32, 0.0], 0usize), ([0.0, 1.0], 1usize)] {
        let ranked = verifier
            .kfsb_single_shot_prescore_for_tests(&graph, &domain, &unstable, &row)
            .expect("armed prescore");
        assert_eq!(ranked.len(), 2, "every unstable neuron is scored");
        assert_eq!(
            (ranked[0].node_name.as_str(), ranked[0].neuron_idx),
            ("relu", expect),
            "row {row:?}: the armed prescore must rank by the seeded objective \
             (scores {:?})",
            ranked
                .iter()
                .map(|c| (c.neuron_idx, c.main_score))
                .collect::<Vec<_>>()
        );
    }
    println!("[kfsb-review] TEST_COMPLETE name=review_kfsb_single_shot_prescore_ranks_by_the_seeded_objective");
}
