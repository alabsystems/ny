// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #kfsb-single-shot: the αβ-CROWN-style graph kFSB scorer
//! (`bab.kfsb_single_shot`).
//!
//! The scorer is ADVISORY (it only picks which ReLU to partition), so the
//! soundness obligations are about the pipeline around it, checked here on
//! small residual DAGs — a Linear residual block, a diamond, a skip from the
//! network input, and a CONV residual block (Conv -> ReLU -> Conv -> ReLU ->
//! Add(skip) -> Conv) — whose exact range over any split region is enumerated
//! in f64 on an 81 x 81 grid of the 2-D input box:
//!
//! * CONTAINMENT: every single-shot child simulation (one constrained CROWN
//!   pass with inherited α/β) encloses the enumerated range of the objective
//!   over that child's region, for every unstable candidate, both phases and
//!   both objective signs — the simulation is a valid bound, not a heuristic
//!   number.
//! * END TO END (the verdict obligation): the conjunctive multi-objective
//!   sequential lane on its aggregation-critical kFSB route never reports
//!   `Verified` for a FALSE property, verifies the TRUE one, and — armed —
//!   really ran the single-shot scorer (counter); dark it never does.
//! * DARK = HISTORICAL: with the key off the selector's pick is the
//!   historical scorer's pick (same code path), and the counter stays 0.
//! * OBJECTIVE DIRECTION: the armed prescore is the objective-seeded BaBSR,
//!   so on the conjunction-critical fixture the armed scorer still picks the
//!   neuron the critical row depends on.

use std::collections::HashMap;
use std::time::Duration;

use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};

use crate::beta_crown::domain::{MultiObjectiveGraphBabDomain, ObjectiveAggregation};
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
// A tiny DAG language (Linear / 1-row Conv / ReLU / Add / Flatten) with an
// exact f64 evaluator. Inputs are the 2-D box [-1, 1]^2; a conv DAG reads it
// as a [1, 1, 2] feature map (one channel, one row, two columns).
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Op {
    Lin {
        src: &'static str,
        w: Array2<f32>,
        b: Array1<f32>,
    },
    /// 1-row Conv2d: kernel `[out_c, in_c, 1, kw]`, stride 1, column padding
    /// `pad`, over an input of `in_c` channels x 1 row x `width` columns.
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
pub(super) struct Dag {
    pub(super) shape: &'static str,
    /// The network input tensor shape (`[2]` or `[1, 1, 2]`).
    input_shape: Vec<usize>,
    nodes: Vec<(&'static str, Op)>,
    output: &'static str,
}

impl Dag {
    pub(super) fn graph(&self) -> GraphNetwork {
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

    /// Every node's value at `x` (flattened, channel-major), in f64. Weights
    /// are dyadic, hence exact in f32.
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

    fn relu_input(&self, relu: &str) -> &'static str {
        self.nodes
            .iter()
            .find_map(|(name, op)| match op {
                Op::Relu(src) if *name == relu => Some(*src),
                _ => None,
            })
            .expect("relu node")
    }

    /// `(min, max)` of `row · out[0]` over the grid points of the region cut
    /// out by `splits`, or `None` when no grid point lies in it.
    pub(super) fn range(&self, splits: &[GraphNeuronConstraint], row: f64) -> Option<(f64, f64)> {
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

    pub(super) fn input_box(&self) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&self.input_shape), -1.0f32),
            ArrayD::from_elem(IxDyn(&self.input_shape), 1.0f32),
        )
        .expect("input box")
    }

    pub(super) fn relu_nodes(&self) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|(_, op)| matches!(op, Op::Relu(_)))
            .map(|(name, _)| name.to_string())
            .collect()
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

/// `l1 = W x; r1 = relu(l1); l2 = W r1; r2 = relu(l2); a1 = r2 + l1;
///  l3 = W a1; r3 = relu(l3); out = w r3`.
fn residual(seed: u64) -> Dag {
    let mut r = Lcg(seed);
    Dag {
        shape: "residual",
        input_shape: vec![2],
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
    }
}

/// `r1 = relu(W x); ra = relu(Wa r1); rb = relu(Wb r1); out = w (ra + rb)`.
fn diamond(seed: u64) -> Dag {
    let mut r = Lcg(seed);
    Dag {
        shape: "diamond",
        input_shape: vec![2],
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
    }
}

/// `r1 = relu(W x); r2 = relu(W r1); sk = W x; out = w (r2 + sk)`.
fn input_skip(seed: u64) -> Dag {
    let mut r = Lcg(seed);
    Dag {
        shape: "input_skip",
        input_shape: vec![2],
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
    }
}

/// The TinyYOLO block shape in miniature (x read as a [1, 1, 2] map):
/// `c1 = conv1x1(x) [2 ch]; r1 = relu(c1); c2 = conv1x3(r1, pad 1);
///  r2 = relu(c2); a1 = r2 + c1; c3 = conv1x2(a1) [1 ch, 1 col];
///  out = flatten(c3)`.
pub(super) fn conv_residual(seed: u64) -> Dag {
    let mut r = Lcg(seed);
    let k1: Vec<f32> = (0..2).map(|_| r.next()).collect();
    let b1 = r.vec(2);
    let k2: Vec<f32> = (0..2 * 2 * 3).map(|_| r.next()).collect();
    let b2 = r.vec(2);
    let k3: Vec<f32> = (0..2 * 2).map(|_| r.next()).collect();
    let b3 = r.vec(1);
    Dag {
        shape: "conv_residual",
        input_shape: vec![1, 1, 2],
        nodes: vec![
            (
                "c1",
                Op::Conv {
                    src: NETWORK_INPUT,
                    kernel: k1,
                    bias: b1,
                    out_c: 2,
                    in_c: 1,
                    kw: 1,
                    pad: 0,
                    width: 2,
                },
            ),
            ("r1", Op::Relu("c1")),
            (
                "c2",
                Op::Conv {
                    src: "r1",
                    kernel: k2,
                    bias: b2,
                    out_c: 2,
                    in_c: 2,
                    kw: 3,
                    pad: 1,
                    width: 2,
                },
            ),
            ("r2", Op::Relu("c2")),
            ("a1", Op::Add("r2", "c1")),
            (
                "c3",
                Op::Conv {
                    src: "a1",
                    kernel: k3,
                    bias: b3,
                    out_c: 1,
                    in_c: 2,
                    kw: 2,
                    pad: 0,
                    width: 2,
                },
            ),
            ("out", Op::Flatten("c3")),
        ],
        output: "out",
    }
}

pub(super) fn dags() -> Vec<Dag> {
    let mut all = Vec::new();
    for seed in [3u64, 11, 29, 47] {
        all.push(residual(seed));
        all.push(diamond(seed ^ 0x9e37));
        all.push(input_skip(seed ^ 0x2c1b));
        all.push(conv_residual(seed ^ 0x7f4a));
    }
    all
}

pub(super) fn split(node: &str, neuron: usize, active: bool) -> GraphNeuronConstraint {
    GraphNeuronConstraint {
        node_name: node.to_string(),
        neuron_idx: neuron,
        is_active: active,
        score: 0.0,
    }
}

pub(super) fn kfsb_config(armed: bool) -> BetaCrownConfig {
    BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::Kfsb,
        fsb_candidates: 3,
        use_multi_objective_critical_kfsb: true,
        kfsb_single_shot: armed,
        enable_cuts: false,
        timeout: Duration::from_secs(30),
        max_domains: 200,
        max_depth: 10,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// CONTAINMENT: every single-shot child simulation encloses the child's range
// ---------------------------------------------------------------------------

#[ntest::timeout(600000)]
#[test]
fn kfsb_single_shot_child_simulations_enclose_the_enumerated_child_range() {
    let tol = 1e-4;
    let mut checked = 0usize;
    let mut violations = Vec::new();
    for dag in dags() {
        let graph = dag.graph();
        let input = dag.input_box();
        let node_bounds = graph.collect_node_bounds(&input).expect("ibp bounds");
        let domain =
            GraphBabDomain::root(node_bounds, -100.0, 100.0, &input, false).expect("root domain");
        let verifier = BetaCrownVerifier::new(kfsb_config(true));
        let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &dag.relu_nodes());
        for row in [1.0f32, -1.0] {
            let objective = [row];
            let candidates: Vec<_> = unstable
                .iter()
                .map(|(node, idx)| {
                    crate::beta_crown::engine::branching::kfsb_shared::GraphKfsbCandidate {
                        node_name: node.clone(),
                        neuron_idx: *idx,
                        main_score: 0.0,
                        backup_score: 0.0,
                    }
                })
                .collect();
            let sims = verifier
                .kfsb_single_shot_child_bounds_for_tests(&graph, &domain, &objective, &candidates)
                .expect("single-shot simulations");
            assert_eq!(sims.len(), candidates.len());
            for (candidate, (active, inactive)) in candidates.iter().zip(sims) {
                for (is_active, side) in [(true, active), (false, inactive)] {
                    let splits = [split(&candidate.node_name, candidate.neuron_idx, is_active)];
                    let Some((lo, hi)) = dag.range(&splits, f64::from(row)) else {
                        continue;
                    };
                    let Some((lower, upper)) = side else {
                        // An infeasible child (`None`) must really be empty
                        // on the grid only if CROWN proved it; a `None` for a
                        // non-empty grid region is a refused/non-finite pass,
                        // which the scorer treats as -inf: still advisory.
                        continue;
                    };
                    checked += 1;
                    if f64::from(lower) > lo + tol || f64::from(upper) < hi - tol {
                        violations.push(format!(
                            "{} {}[{}] active={is_active} row={row}: sim [{lower}, {upper}] \
                             misses grid [{lo}, {hi}]",
                            dag.shape, candidate.node_name, candidate.neuron_idx
                        ));
                    }
                }
            }
        }
    }
    println!("[kfsb-single-shot] containment checked={checked}");
    assert!(
        checked > 40,
        "positive control: too few simulations checked ({checked})"
    );
    assert!(
        violations.is_empty(),
        "single-shot child simulation is not a valid bound: {violations:#?}"
    );
    println!("[kfsb-single-shot] TEST_COMPLETE name=kfsb_single_shot_child_simulations_enclose_the_enumerated_child_range");
}

// ---------------------------------------------------------------------------
// END TO END on the conjunctive multi-objective sequential lane
// ---------------------------------------------------------------------------

struct Outcome {
    verified: bool,
    errored: bool,
    label: String,
}

/// Conjunctive two-atom property on the sequential lane (critical kFSB route):
/// atom 0 is `out < threshold` (row `-out > -threshold`), atom 1 is the
/// never-provable `out > 1e6`, so the conjunction is verified iff atom 0 is.
fn conjunctive(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    threshold: f32,
    armed: bool,
) -> (Outcome, u64) {
    let verifier = BetaCrownVerifier::new(kfsb_config(armed));
    let before = kfsb_single_shot_calls_for_tests();
    let result = verifier.verify_graph_relu_split_multi_objective_conjunctive_with_engine(
        graph,
        input,
        &[vec![-1.0], vec![1.0]],
        &[-threshold, 1.0e6],
        None,
        None,
    );
    let calls = kfsb_single_shot_calls_for_tests() - before;
    let outcome = match result {
        Ok(result) => Outcome {
            verified: matches!(result.result, BabVerificationStatus::Verified),
            errored: false,
            label: format!(
                "{:?} explored={} max_depth={}",
                result.result, result.domains_explored, result.max_depth_reached
            ),
        },
        Err(error) => Outcome {
            verified: false,
            errored: true,
            label: format!("Err({error})"),
        },
    };
    (outcome, calls)
}

#[ntest::timeout(600000)]
#[test]
fn kfsb_single_shot_conjunctive_lane_never_verifies_a_false_property() {
    let mut unsound = Vec::new();
    let mut errored = Vec::new();
    let mut vacuous = Vec::new();
    let mut armed_calls = 0u64;
    let mut dark_calls = 0u64;
    // TRUE-property closes per arm: [dark, armed].
    let mut true_closes = [0usize; 2];
    let all = dags();
    for dag in all.iter() {
        let graph = dag.graph();
        let input = dag.input_box();
        let (_, hi) = dag.range(&[], 1.0).expect("range");
        // FALSE: just below the enumerated maximum, so the lane must branch
        // (the kFSB route runs) and must never verify. TRUE: a clear margin
        // (the warm-frontier e2e's), so a vacuous lane is caught; the tight
        // TRUE (+0.25) is reported, not asserted — on some seeds this lane
        // ends `unresolved_no_branch` there in BOTH arms (a completeness
        // limit of the sequential lane, not a scorer effect).
        let false_threshold = hi as f32 - 0.05;
        let true_threshold = hi as f32 + 1.0;
        let tight_threshold = hi as f32 + 0.25;
        let mut true_verified = 0usize;
        for armed in [false, true] {
            let (f, calls) = conjunctive(&graph, &input, false_threshold, armed);
            let (t, calls_t) = conjunctive(&graph, &input, true_threshold, armed);
            let (tight, calls_tight) = conjunctive(&graph, &input, tight_threshold, armed);
            if armed {
                armed_calls += calls + calls_t + calls_tight;
            } else {
                dark_calls += calls + calls_t + calls_tight;
            }
            println!(
                "[kfsb-single-shot] ROW net={} armed={armed} FALSE: {} | TRUE: {} | TIGHT: {}",
                dag.shape, f.label, t.label, tight.label
            );
            if f.verified {
                unsound.push(format!("{}/armed={armed}: {}", dag.shape, f.label));
            }
            for (truth, o) in [("FALSE", &f), ("TRUE", &t), ("TIGHT", &tight)] {
                if o.errored {
                    errored.push(format!("{}/armed={armed}/{truth}: {}", dag.shape, o.label));
                }
            }
            true_verified += usize::from(t.verified);
            true_closes[usize::from(armed)] += usize::from(t.verified);
        }
        if true_verified == 0 {
            vacuous.push(format!("{}: TRUE verified in no arm", dag.shape));
        }
    }
    println!(
        "[kfsb-single-shot] e2e armed_calls={armed_calls} dark_calls={dark_calls} \
         true_closes_dark={} true_closes_armed={} of {} vacuous={vacuous:?}",
        true_closes[0],
        true_closes[1],
        all.len()
    );
    assert!(
        unsound.is_empty(),
        "FALSE property reported Verified (UNSOUND) on: {unsound:#?}"
    );
    assert!(errored.is_empty(), "lane errored on: {errored:#?}");
    // Vacuity is judged over the whole fixture set: on a few seeds this lane
    // ends `unresolved_no_branch` on the margin-1.0 TRUE property in BOTH arms
    // (children past `beta_max_depth` carry their inherited β only, so a fully
    // split region is still bounded over its box). The scorer must not lose a
    // close the historical scorer makes on this set.
    assert!(
        true_closes[0] + 2 >= all.len() && true_closes[1] + 2 >= all.len(),
        "vacuous run: TRUE closes dark={} armed={} of {}: {vacuous:#?}",
        true_closes[0],
        true_closes[1],
        all.len()
    );
    assert!(
        true_closes[1] >= true_closes[0],
        "the armed scorer lost a TRUE close the historical scorer makes: dark={} armed={}",
        true_closes[0],
        true_closes[1]
    );
    assert_eq!(
        dark_calls, 0,
        "the dark key must never reach the single-shot scorer"
    );
    assert!(
        armed_calls > 0,
        "the armed key never reached the single-shot scorer (VOID: no branching happened \
         on the critical kFSB route)"
    );
    println!("[kfsb-single-shot] TEST_COMPLETE name=kfsb_single_shot_conjunctive_lane_never_verifies_a_false_property");
}

// ---------------------------------------------------------------------------
// DARK = HISTORICAL, and OBJECTIVE DIRECTION
// ---------------------------------------------------------------------------

/// The conjunction-critical fixture of
/// `auto_conjunctive_kfsb_reaches_critical_row_without_env_gate`: intercept
/// prefers neuron 0, the critical row depends on neuron 1.
fn critical_fixture() -> (GraphNetwork, BoundedTensor, MultiObjectiveGraphBabDomain) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(arr2(&[[-5.0, -0.1], [-0.1, -5.0]]), None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");
    let input =
        BoundedTensor::new(arr1(&[-3.0, -0.5]).into_dyn(), arr1(&[1.0, 0.5]).into_dyn()).unwrap();
    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let domain = MultiObjectiveGraphBabDomain::root_with_aggregation(
        node_bounds,
        vec![(-100.0, 0.0), (9.0, 20.0)],
        &input,
        &[-98.0, 10.0],
        false,
        ObjectiveAggregation::Conjunctive,
    )
    .unwrap();
    (graph, input, domain)
}

#[ntest::timeout(60000)]
#[test]
fn kfsb_single_shot_dark_is_the_historical_pick_and_armed_follows_the_critical_row() {
    let (graph, _input, domain) = critical_fixture();
    let objectives = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let thresholds = [-98.0, 10.0];
    let relu_nodes = vec!["relu".to_string()];
    // Serialize against every env-mutating test (the selector reads a few
    // legacy experiment gates); the empty list names no variable, so this file
    // adds no lever-name literal site (source_frontier ratchet).
    let no_vars: &[&str] = &[];
    crate::tests::with_serialized_env_vars_removed(no_vars, || {
        let pick = |config: BetaCrownConfig| {
            let verifier = BetaCrownVerifier::new(config);
            let unstable = verifier.find_unstable_graph_neurons_multi(&graph, &domain, &relu_nodes);
            let before = kfsb_single_shot_calls_for_tests();
            let pick = verifier
                .select_graph_branch_multi(
                    &graph,
                    &domain,
                    &unstable,
                    &objectives,
                    &thresholds,
                    None,
                )
                .expect("selector");
            (pick, kfsb_single_shot_calls_for_tests() - before)
        };
        let base = BetaCrownConfig {
            branching_heuristic: BranchingHeuristic::Kfsb,
            fsb_candidates: 2,
            kfsb_reduce_op: crate::beta_crown::config::KfsbReduceOp::Max,
            beta_iterations: 0,
            use_multi_objective_critical_kfsb: true,
            ..Default::default()
        };
        let (dark, dark_calls) = pick(BetaCrownConfig {
            kfsb_single_shot: false,
            ..base.clone()
        });
        // `beta_iterations: 0` makes the historical child evaluation a
        // single inherited-β pass too, so the dark pick is the historical
        // scorer's own pick on this fixture.
        assert_eq!(dark.1, 1, "historical critical-row kFSB picks neuron 1");
        assert_eq!(dark_calls, 0, "dark never reaches the single-shot scorer");
        let (armed, armed_calls) = pick(BetaCrownConfig {
            kfsb_single_shot: true,
            ..base
        });
        assert_eq!(
            armed_calls, 1,
            "armed reaches the single-shot scorer once per decision"
        );
        assert_eq!(
            (armed.0.as_str(), armed.1),
            ("relu", 1),
            "the objective-seeded prescore must still follow the conjunction-critical row"
        );
    });
    println!("[kfsb-single-shot] TEST_COMPLETE name=kfsb_single_shot_dark_is_the_historical_pick_and_armed_follows_the_critical_row");
}
