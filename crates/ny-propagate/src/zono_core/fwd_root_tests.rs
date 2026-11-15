// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness pins for the V4 forward root screen (`#fwd-root-screen`) on
//! synthetic graphs: exact-rational band checks, pointwise form + box
//! containment on a DAG exercising every op, the bilinear sign-slip and
//! positional-alignment mutations (which must go RED under the form check),
//! the probe maps, refusals, and the dark default.
//!
//! Real-ONNX containment on every ml4acopf model, the true-sat controls, the
//! C1 union and the F1 replica parity live in ny-cli
//! (`verify/fwd_root_screen_tests.rs`), where ONNX loading is available.
//!
//! Every check is POINTWISE and counts what it checked; a test whose
//! precondition (unstable ReLUs, products, fresh symbols, a mutation actually
//! armed) does not hold fails instead of passing vacuously.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use ny_tensor::BoundedTensor;

use super::dag::{DagForwardOptions, DagOp, DagProgram};
use super::elementwise::{
    fault as elementwise_fault, floor_band, pwl_band, smooth_band, Band, SelectSource, SmoothKind,
};
use super::form_check::{check_point, summarize, ContainmentReport};
use super::fwd_root::{forward_root_screen_plan, run_forward_root_screen};
use super::program::ZonoNodeView;
use super::tests::Rng;
use super::ZonoRefusal;
use crate::beta_crown::config::BetaCrownConfig;
use crate::layers::{
    AddConstantLayer, AddLayer, ConcatLayer, CosLayer, DivConstantLayer, ExpLayer, FloorLayer,
    GatherLayer, Layer, LinearLayer, MulBinaryLayer, MulConstantLayer, PowConstantLayer, ReLULayer,
    SigmoidLayer, SinLayer, SliceLayer, SqueezeLayer, SubConstantLayer, SubLayer, TransposeLayer,
    UnsqueezeLayer,
};
use crate::{GraphNetwork, GraphNode};

const BIG_BUDGET: u64 = 1 << 32;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[allow(clippy::cast_possible_truncation)]
fn rand_f32(rng: &mut Rng, n: usize, scale: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.sym() * scale) as f32).collect()
}

fn linear(rng: &mut Rng, rows: usize, cols: usize, scale: f64) -> Layer {
    let w = Array2::from_shape_vec((rows, cols), rand_f32(rng, rows * cols, scale)).expect("w");
    let b = Array1::from_vec(rand_f32(rng, rows, 0.3));
    Layer::Linear(LinearLayer::new(w, Some(b)).expect("linear"))
}

fn node(name: &str, layer: Layer, inputs: &[&str]) -> GraphNode {
    GraphNode::new(
        name,
        layer,
        inputs.iter().map(|input| (*input).to_string()).collect(),
    )
}

fn vector(values: &[f32]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("vector")
}

fn indices(values: &[i64]) -> ArrayD<i64> {
    ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("indices")
}

fn scalar(value: f32) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(&[]), value)
}

fn input_box(lo: &[f32], hi: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lo.len()]), lo.to_vec()).expect("lo"),
        ArrayD::from_shape_vec(IxDyn(&[hi.len()]), hi.to_vec()).expect("hi"),
    )
    .expect("box")
}

/// A miniature ml4acopf graph using EVERY op of the V4 set: Linear trunk with
/// ReLUs, Slice, static Gather with duplicates, live-live Sub/Add, Sin/Cos,
/// bilinear products, `Pow(x, 2)` and `Mul(x, x)`, per-element constants
/// (Add/Sub-reverse/Mul/Div), Sigmoid, the Floor range reduction, a rank-1
/// `Transpose -> Linear -> Transpose` flow block, the ReLU lookup-table motif,
/// and an N-ary Concat output.
fn mini_acopf_graph(seed: u64) -> GraphNetwork {
    let mut rng = Rng::new(seed);
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input("h1", linear(&mut rng, 8, 4, 0.9)));
    g.add_node(node("r1", Layer::ReLU(ReLULayer), &["h1"]));
    g.add_node(node("h2", linear(&mut rng, 8, 8, 0.6), &["r1"]));
    g.add_node(node("r2", Layer::ReLU(ReLULayer), &["h2"]));
    g.add_node(node("trunk", linear(&mut rng, 7, 8, 0.6), &["r2"]));
    g.add_node(node(
        "va",
        Layer::Slice(SliceLayer::new(0, 0, 3)),
        &["trunk"],
    ));
    g.add_node(node(
        "vm_raw",
        Layer::Slice(SliceLayer::new(0, 3, 6)),
        &["trunk"],
    ));
    g.add_node(node(
        "vm",
        Layer::AddConstant(AddConstantLayer::new(vector(&[1.0, 1.1, 0.9]))),
        &["vm_raw"],
    ));
    g.add_node(node(
        "gi",
        Layer::Gather(GatherLayer::new(
            0,
            Some(indices(&[0, 2, 1, 0])),
            Vec::new(),
        )),
        &["va"],
    ));
    g.add_node(node(
        "gj",
        Layer::Gather(GatherLayer::new(
            0,
            Some(indices(&[1, 0, 2, 2])),
            Vec::new(),
        )),
        &["va"],
    ));
    g.add_node(node("theta", Layer::Sub(SubLayer), &["gi", "gj"]));
    g.add_node(node("c", Layer::Cos(CosLayer::new()), &["theta"]));
    g.add_node(node("s", Layer::Sin(SinLayer::new()), &["theta"]));
    g.add_node(node(
        "mi",
        Layer::Gather(GatherLayer::new(
            0,
            Some(indices(&[0, 1, 2, 0])),
            Vec::new(),
        )),
        &["vm"],
    ));
    g.add_node(node(
        "mj",
        Layer::Gather(GatherLayer::new(
            0,
            Some(indices(&[2, 2, 1, 1])),
            Vec::new(),
        )),
        &["vm"],
    ));
    g.add_node(node("p", Layer::MulBinary(MulBinaryLayer), &["mi", "mj"]));
    g.add_node(node("pc", Layer::MulBinary(MulBinaryLayer), &["p", "c"]));
    g.add_node(node("ps", Layer::MulBinary(MulBinaryLayer), &["p", "s"]));
    g.add_node(node(
        "sq",
        Layer::PowConstant(PowConstantLayer::new(2.0)),
        &["mi"],
    ));
    g.add_node(node("mm", Layer::MulBinary(MulBinaryLayer), &["mj", "mj"]));
    g.add_node(node(
        "k1",
        Layer::MulConstant(MulConstantLayer::new(vector(&[0.5, -1.25, 2.0, 0.75]))),
        &["pc"],
    ));
    g.add_node(node("flow", Layer::Add(AddLayer), &["k1", "sq"]));
    g.add_node(node(
        "rev",
        Layer::SubConstant(SubConstantLayer::new_reverse(scalar(0.5))),
        &["ps"],
    ));
    g.add_node(node(
        "d",
        Layer::DivConstant(DivConstantLayer::scalar(3.0)),
        &["mm"],
    ));
    g.add_node(node(
        "sig_in",
        Layer::Slice(SliceLayer::new(0, 5, 7)),
        &["trunk"],
    ));
    g.add_node(node(
        "sig",
        Layer::Sigmoid(SigmoidLayer::new()),
        &["sig_in"],
    ));
    // Floor range reduction r = t - 2pi floor((t + pi) / 2pi).
    g.add_node(node(
        "shifted",
        Layer::AddConstant(AddConstantLayer::new(scalar(std::f32::consts::PI))),
        &["theta"],
    ));
    g.add_node(node(
        "turns",
        Layer::DivConstant(DivConstantLayer::scalar(std::f32::consts::TAU)),
        &["shifted"],
    ));
    g.add_node(node("whole", Layer::Floor(FloorLayer::new()), &["turns"]));
    g.add_node(node(
        "wrap",
        Layer::MulConstant(MulConstantLayer::scalar(std::f32::consts::TAU)),
        &["whole"],
    ));
    g.add_node(node("wrapped", Layer::Sub(SubLayer), &["theta", "wrap"]));
    // Rank-1 flow block: Transpose(perm [1, 0]) -> Linear -> Transpose.
    g.add_node(node(
        "col",
        Layer::Transpose(TransposeLayer::new(vec![1, 0])),
        &["va"],
    ));
    g.add_node(node("lin", linear(&mut rng, 2, 3, 0.8), &["col"]));
    g.add_node(node(
        "row",
        Layer::Transpose(TransposeLayer::new(vec![1, 0])),
        &["lin"],
    ));
    // The ReLU lookup-table motif (C1's LinearRow spelling).
    g.add_node(node(
        "tcol",
        Layer::Unsqueeze(UnsqueezeLayer::new(-1)),
        &["va"],
    ));
    g.add_node(node(
        "tshift",
        Layer::SubConstant(SubConstantLayer::new(vector(&[-1.0, -0.4, 0.1, 0.7, 1.3]))),
        &["tcol"],
    ));
    g.add_node(node("thinge", Layer::ReLU(ReLULayer), &["tshift"]));
    let delta = Array2::from_shape_vec((1, 5), vec![0.3f32, 0.6, -0.9, 0.4, -0.2]).expect("delta");
    g.add_node(node(
        "tslope",
        Layer::Linear(LinearLayer::new(delta, Some(Array1::from_vec(vec![0.1f32]))).expect("t")),
        &["thinge"],
    ));
    g.add_node(node(
        "tflat",
        Layer::Squeeze(SqueezeLayer::new(-1)),
        &["tslope"],
    ));
    g.add_node(node(
        "out",
        Layer::Concat(ConcatLayer::new(0)),
        &[
            "flow", "rev", "d", "sig", "wrapped", "row", "tflat", "c", "trunk",
        ],
    ));
    g.set_output("out");
    g
}

/// Sample points of the box: every other point uniform, the rest vertices.
fn sample_points(lo: &[f32], hi: &[f32], count: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut points = vec![lo
        .iter()
        .zip(hi)
        .map(|(&l, &h)| 0.5 * f64::from(l) + 0.5 * f64::from(h))
        .collect::<Vec<f64>>()];
    for k in 0..count {
        let point = lo
            .iter()
            .zip(hi)
            .map(|(&l, &h)| {
                let (l, h) = (f64::from(l), f64::from(h));
                if k % 2 == 0 {
                    (l + (h - l) * rng.unit()).clamp(l, h)
                } else if rng.unit() < 0.5 {
                    l
                } else {
                    h
                }
            })
            .collect();
        points.push(point);
    }
    points
}

/// Box and pointwise-form containment of every computed node at every point.
#[derive(Debug, Default)]
struct DagContainment {
    form: ContainmentReport,
    box_checked: usize,
    box_violations: usize,
    nodes: usize,
}

fn dag_containment(
    graph: &GraphNetwork,
    lo: &[f32],
    hi: &[f32],
    points: usize,
    seed: u64,
    intersect_ibp: bool,
) -> (DagContainment, super::dag::DagForwardStats) {
    let input = input_box(lo, hi);
    let (program, ibp) = DagProgram::compile_with_ibp(graph, &input, None).expect("compile");
    let mut rng = Rng::new(seed);
    let pts = sample_points(lo, hi, points, &mut rng);
    let flat: Vec<f64> = pts.iter().flatten().copied().collect();
    let mut values: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    program
        .evaluate_points(&flat, pts.len(), |_, name, v| {
            values.entry(name.to_string()).or_default().push(v.to_vec());
        })
        .expect("evaluate");
    let mut report = DagContainment::default();
    let opts = DagForwardOptions {
        intersect_ibp,
        byte_budget: BIG_BUDGET,
    };
    let mut observer = |view: &super::dag::DagNodeView<'_>| {
        let Some(node_values) = values.get(view.node) else {
            return;
        };
        let summary = summarize(&ZonoNodeView {
            step: None,
            node: view.node,
            shape: view.shape,
            state: view.state,
            abs_g: view.abs_g,
            input_symbols: view.input_symbols,
        });
        report.nodes += 1;
        for (point, v) in pts.iter().zip(node_values) {
            check_point(&summary, point, v, &mut report.form);
            for (j, &value) in v.iter().enumerate() {
                report.box_checked += 1;
                let tol = 1e-12 * (1.0 + value.abs());
                if value < view.lo[j] - tol || value > view.hi[j] + tol {
                    report.box_violations += 1;
                }
            }
        }
    };
    let outcome = program
        .forward(&input, &opts, &ibp, None, Some(&mut observer))
        .expect("forward");
    (report, outcome.stats)
}

// ---------------------------------------------------------------------------
// Containment on the full op set
// ---------------------------------------------------------------------------

/// Every op of the V4 set, the root box and random sub-boxes, both with and
/// without the IBP intersection: zero box violations and zero pointwise form
/// violations at every node, with products, squares, bands and unstable ReLUs
/// all actually exercised. The program's own f64 semantics are first
/// cross-checked against ny's pointwise forward node by node (a wiring error
/// would otherwise corrupt the sampled "truth" and the enclosure alike).
#[test]
fn fwd_root_screen_synthetic_dag_contains_samples() {
    let graph = mini_acopf_graph(0xF00D_0001);
    let root_lo = [-1.0f32, -0.8, -1.2, -0.5];
    let root_hi = [1.0f32, 0.9, 0.7, 1.1];
    cross_check_semantics(&graph, &root_lo, &root_hi);
    let mut rng = Rng::new(0xF00D_0002);
    let mut total = DagContainment::default();
    let mut exercised = super::dag::DagForwardStats::default();
    for trial in 0..8 {
        let (lo, hi): (Vec<f32>, Vec<f32>) = if trial == 0 {
            (root_lo.to_vec(), root_hi.to_vec())
        } else {
            root_lo
                .iter()
                .zip(&root_hi)
                .map(|(&l, &h)| {
                    #[allow(clippy::cast_possible_truncation)]
                    let a = (f64::from(l) + (f64::from(h) - f64::from(l)) * rng.unit()) as f32;
                    #[allow(clippy::cast_possible_truncation)]
                    let b = (f64::from(l) + (f64::from(h) - f64::from(l)) * rng.unit()) as f32;
                    (a.min(b), a.max(b))
                })
                .unzip()
        };
        for intersect in [true, false] {
            let (report, stats) = dag_containment(&graph, &lo, &hi, 64, 17 + trial, intersect);
            total.form.absorb(&report.form);
            total.box_checked += report.box_checked;
            total.box_violations += report.box_violations;
            total.nodes += report.nodes;
            exercised.fresh_relu += stats.fresh_relu;
            exercised.fresh_mul += stats.fresh_mul;
            exercised.fresh_square += stats.fresh_square;
            exercised.fresh_band += stats.fresh_band;
        }
    }
    assert!(total.box_checked > 10_000, "checked too little: {total:?}");
    assert!(
        total.form.checked > 10_000,
        "form checked too little: {total:?}"
    );
    assert!(
        exercised.fresh_relu > 0
            && exercised.fresh_mul > 0
            && exercised.fresh_square > 0
            && exercised.fresh_band > 0,
        "precondition: every relaxation kind must be exercised, got {exercised:?}"
    );
    assert_eq!(
        total.box_violations, 0,
        "box containment violated: {total:?}"
    );
    assert_eq!(
        total.form.form_violations, 0,
        "pointwise form containment violated: {total:?}"
    );
}

/// The program's f64 point semantics agree with ny's own pointwise forward at
/// every node it computes (f32 tolerance).
fn cross_check_semantics(graph: &GraphNetwork, lo: &[f32], hi: &[f32]) {
    let input = input_box(lo, hi);
    let (program, _) = DagProgram::compile_with_ibp(graph, &input, None).expect("compile");
    let mut rng = Rng::new(0x5E4A);
    let mut compared = 0usize;
    for point in sample_points(lo, hi, 6, &mut rng) {
        #[allow(clippy::cast_possible_truncation)]
        let x32: Vec<f32> = point.iter().map(|&v| v as f32).collect();
        let degenerate = input_box(&x32, &x32);
        let reference = graph
            .collect_node_activations_pointwise(&degenerate, None)
            .expect("ny pointwise forward");
        let exact: Vec<f64> = x32.iter().map(|&v| f64::from(v)).collect();
        program
            .evaluate_points(&exact, 1, |_, name, values| {
                let Some(theirs) = reference.get(name) else {
                    return;
                };
                let theirs: Vec<f32> = theirs.lower().iter().copied().collect();
                assert_eq!(theirs.len(), values.len(), "`{name}` length");
                let scale = theirs.iter().fold(1.0f64, |m, v| m.max(f64::from(v.abs())));
                for (j, (&mine, &t)) in values.iter().zip(&theirs).enumerate() {
                    assert!(
                        (mine - f64::from(t)).abs() <= 1e-4 * scale,
                        "`{name}`[{j}]: program {mine} vs ny {t}"
                    );
                    compared += 1;
                }
            })
            .expect("evaluate");
    }
    assert!(compared > 0, "semantic cross-check compared nothing: VOID");
}

// ---------------------------------------------------------------------------
// Mutation pins (must go RED under the form check)
// ---------------------------------------------------------------------------

/// A correlated bilinear product: `p = a * b` with `a, b` affine in the same
/// two inputs (centres away from zero, small box, so the first-order term
/// dominates the second-order bound).
fn product_graph() -> GraphNetwork {
    let mut g = GraphNetwork::new();
    let wa = Array2::from_shape_vec((3, 2), vec![1.0f32, 0.5, -0.7, 1.2, 0.3, -0.9]).expect("wa");
    let wb = Array2::from_shape_vec((3, 2), vec![0.8f32, -1.1, 0.6, 0.4, -1.3, 0.2]).expect("wb");
    let ba = Array1::from_vec(vec![2.0f32, -1.5, 1.8]);
    let bb = Array1::from_vec(vec![-2.2f32, 1.7, 2.4]);
    g.add_node(GraphNode::from_input(
        "a",
        Layer::Linear(LinearLayer::new(wa, Some(ba)).expect("a")),
    ));
    g.add_node(GraphNode::from_input(
        "b",
        Layer::Linear(LinearLayer::new(wb, Some(bb)).expect("b")),
    ));
    g.add_node(node("p", Layer::MulBinary(MulBinaryLayer), &["a", "b"]));
    g.set_output("p");
    g
}

/// Named V4 test 3 — the zonotope analogue of F1's McCormick-sign mutation:
/// the product's second first-order term enters with the WRONG sign. Clean,
/// the form check passes; mutated, it must report violations. The form check
/// is the gate because ny's sound IBP box can mask a wrong form.
#[test]
fn fwd_root_screen_mutation_mccormick_sign_fails() {
    let graph = product_graph();
    let lo = [-0.05f32, -0.04];
    let hi = [0.05f32, 0.06];
    let (clean, clean_stats) = dag_containment(&graph, &lo, &hi, 64, 3, true);
    assert!(clean_stats.fresh_mul > 0, "precondition: the product ran");
    assert!(clean.form.checked > 0, "clean run checked nothing");
    assert_eq!(clean.form.form_violations, 0, "clean product: {clean:?}");
    assert_eq!(clean.box_violations, 0, "clean product box: {clean:?}");

    elementwise_fault::set(elementwise_fault::ElementwiseFault::ProductCrossSign);
    let (mutated, _) = dag_containment(&graph, &lo, &hi, 64, 3, true);
    elementwise_fault::set(elementwise_fault::ElementwiseFault::None);
    assert!(
        mutated.form.form_violations > 0,
        "the sign-slipped product must fail the pointwise form check: {mutated:?}"
    );
    // The box check is reported, not asserted: under the IBP intersection a
    // wrong FORM can still hide inside a sound box (F1: box checks missed the
    // McCormick-sign defect the form check caught).
    eprintln!(
        "[fwd-root-test] mutation box_violations={} form_violations={}",
        mutated.box_violations, mutated.form.form_violations
    );
}

/// Two ReLU branches with equal-width unstable ranges (equal DeepZ offsets),
/// concatenated, sliced apart and subtracted: correct ID alignment keeps the
/// two fresh symbols independent; a positional merge makes the second branch's
/// symbol claim the first's, cancelling a real error term.
fn concat_merge_graph() -> GraphNetwork {
    let mut g = GraphNetwork::new();
    let wa = Array2::from_shape_vec((1, 2), vec![1.0f32, 0.5]).expect("wa");
    let wb = Array2::from_shape_vec((1, 2), vec![0.5f32, 1.0]).expect("wb");
    g.add_node(GraphNode::from_input(
        "za",
        Layer::Linear(LinearLayer::new(wa, None::<Array1<f32>>).expect("za")),
    ));
    g.add_node(GraphNode::from_input(
        "zb",
        Layer::Linear(LinearLayer::new(wb, None::<Array1<f32>>).expect("zb")),
    ));
    g.add_node(node("a", Layer::ReLU(ReLULayer), &["za"]));
    g.add_node(node("b", Layer::ReLU(ReLULayer), &["zb"]));
    g.add_node(node("cat", Layer::Concat(ConcatLayer::new(0)), &["a", "b"]));
    g.add_node(node("sa", Layer::Slice(SliceLayer::new(0, 0, 1)), &["cat"]));
    g.add_node(node("sb", Layer::Slice(SliceLayer::new(0, 1, 2)), &["cat"]));
    g.add_node(node("out", Layer::Sub(SubLayer), &["sa", "sb"]));
    g.set_output("out");
    g
}

/// Named V4 test 4: symbol IDs are explicit, never positional. The Concat
/// merge (this module's `select`) and the Sub (the core's `add_aligned`) each
/// have a test-only positional fault; clean runs pass the form check, and
/// EACH positional mutation must go red.
#[test]
fn fwd_root_screen_symbol_ids_not_positional() {
    let graph = concat_merge_graph();
    let lo = [-1.0f32, -1.0];
    let hi = [1.0f32, 1.0];
    let (clean, stats) = dag_containment(&graph, &lo, &hi, 96, 5, true);
    assert!(
        stats.fresh_relu >= 2,
        "precondition: both branches unstable"
    );
    assert_eq!(
        clean.form.form_violations, 0,
        "clean concat merge: {clean:?}"
    );

    elementwise_fault::set(elementwise_fault::ElementwiseFault::PositionalSelect);
    let (select_mutated, _) = dag_containment(&graph, &lo, &hi, 96, 5, true);
    elementwise_fault::set(elementwise_fault::ElementwiseFault::None);
    assert!(
        select_mutated.form.form_violations > 0,
        "a positional Concat merge must fail the form check: {select_mutated:?}"
    );

    // The core's aligned add under its own positional fault: two branches whose
    // fresh symbols sit at the same row position.
    let mut direct = GraphNetwork::new();
    let wa = Array2::from_shape_vec((1, 2), vec![1.0f32, 0.5]).expect("wa");
    let wb = Array2::from_shape_vec((1, 2), vec![0.5f32, 1.0]).expect("wb");
    direct.add_node(GraphNode::from_input(
        "za",
        Layer::Linear(LinearLayer::new(wa, None::<Array1<f32>>).expect("za")),
    ));
    direct.add_node(GraphNode::from_input(
        "zb",
        Layer::Linear(LinearLayer::new(wb, None::<Array1<f32>>).expect("zb")),
    ));
    direct.add_node(node("a", Layer::ReLU(ReLULayer), &["za"]));
    direct.add_node(node("b", Layer::ReLU(ReLULayer), &["zb"]));
    direct.add_node(node("out", Layer::Sub(SubLayer), &["a", "b"]));
    direct.set_output("out");
    let (clean_add, _) = dag_containment(&direct, &lo, &hi, 96, 7, true);
    assert_eq!(
        clean_add.form.form_violations, 0,
        "clean aligned add: {clean_add:?}"
    );
    super::ids::fault::set_positional_alignment(true);
    let (add_mutated, _) = dag_containment(&direct, &lo, &hi, 96, 7, true);
    super::ids::fault::set_positional_alignment(false);
    assert!(
        add_mutated.form.form_violations > 0,
        "a positional aligned add must fail the form check: {add_mutated:?}"
    );
}

// ---------------------------------------------------------------------------
// Exact-rational band checks
// ---------------------------------------------------------------------------

fn rat(v: f64) -> BigRational {
    BigRational::from_float(v).expect("finite binary64 value")
}

fn int(v: u64) -> BigRational {
    BigRational::from_integer(BigInt::from(v))
}

/// Rational enclosure of `sin x` (Taylor, Lagrange remainder `|x|^(2N+1)/(2N+1)!`).
fn sin_bracket(x: &BigRational) -> (BigRational, BigRational) {
    let x2 = x * x;
    let mut term = x.clone();
    let mut sum = BigRational::zero();
    for k in 0..24u64 {
        if k % 2 == 0 {
            sum += &term;
        } else {
            sum -= &term;
        }
        term = term * &x2 / int((2 * k + 2) * (2 * k + 3));
    }
    let r = term.abs();
    (&sum - &r, &sum + &r)
}

/// Rational enclosure of `cos x` (remainder `|x|^(2N)/(2N)!`).
fn cos_bracket(x: &BigRational) -> (BigRational, BigRational) {
    let x2 = x * x;
    let mut term = int(1);
    let mut sum = BigRational::zero();
    for k in 0..24u64 {
        if k % 2 == 0 {
            sum += &term;
        } else {
            sum -= &term;
        }
        term = term * &x2 / int((2 * k + 1) * (2 * k + 2));
    }
    let r = term.abs();
    (&sum - &r, &sum + &r)
}

/// Rational enclosure of `e^y` for `|y| <= 16` (remainder
/// `e^|y| |y|^N / N! <= 3^ceil|y| |y|^N / N!`).
fn exp_bracket(y: &BigRational) -> (BigRational, BigRational) {
    let mut term = int(1);
    let mut sum = BigRational::zero();
    let terms = 90u64;
    for n in 0..terms {
        sum += &term;
        term = term * y / int(n + 1);
    }
    let mut bound = int(1);
    let mut k = BigRational::zero();
    while k < y.abs() {
        bound *= int(3);
        k += int(1);
    }
    let r = term.abs() * bound;
    (&sum - &r, &sum + &r)
}

fn sigmoid_bracket(x: &BigRational) -> (BigRational, BigRational) {
    let (e_lo, e_hi) = exp_bracket(&-x.clone());
    assert!(e_lo > BigRational::zero(), "exp bracket must stay positive");
    let one = int(1);
    (&one / (&one + &e_hi), &one / (&one + &e_lo))
}

/// Check `lambda x + mu_lo <= f(x) <= lambda x + mu_hi` in exact rationals at
/// the endpoints, the midpoint, points one ulp inside, and random interior
/// points of every interval. Returns the number of points checked.
fn check_band_rational<F>(band: &Band, lo: &[f64], hi: &[f64], rng: &mut Rng, bracket: F) -> usize
where
    F: Fn(&BigRational) -> (BigRational, BigRational),
{
    let mut checked = 0usize;
    for j in 0..band.len() {
        let (l, u) = (lo[j], hi[j]);
        let mut xs = vec![l, u, 0.5 * l + 0.5 * u];
        if u > l {
            xs.push(l.next_up().min(u));
            xs.push(u.next_down().max(l));
            for _ in 0..12 {
                xs.push((l + (u - l) * rng.unit()).clamp(l, u));
            }
        }
        let lam = rat(band.lambda[j]);
        let (mlo, mhi) = (rat(band.mu_lo[j]), rat(band.mu_hi[j]));
        for x in xs {
            let xr = rat(x);
            let (f_lo, f_hi) = bracket(&xr);
            let lower = &lam * &xr + &mlo;
            let upper = &lam * &xr + &mhi;
            assert!(
                lower <= f_lo,
                "band lower {} exceeds f at x={x:e} on [{l:e}, {u:e}]",
                band.mu_lo[j]
            );
            assert!(
                f_hi <= upper,
                "band upper {} below f at x={x:e} on [{l:e}, {u:e}]",
                band.mu_hi[j]
            );
            checked += 1;
        }
    }
    checked
}

/// Intervals of assorted widths (from one ulp to several units), including
/// degenerate points, centred in `[-center_span, center_span]`.
fn band_intervals(rng: &mut Rng, center_span: f64, count: usize) -> (Vec<f64>, Vec<f64>) {
    let widths = [0.0, 1e-12, 1e-7, 1e-3, 0.05, 0.4, 1.3, 3.0];
    (0..count)
        .map(|k| {
            let c = center_span * rng.sym();
            let w = widths[k % widths.len()] * (0.5 + rng.unit());
            (c - 0.5 * w, c + 0.5 * w)
        })
        .unzip()
}

fn smooth_rational(kind: SmoothKind, span: f64, seed: u64) {
    let mut rng = Rng::new(seed);
    let (lo, hi) = band_intervals(&mut rng, span, 40);
    let band = smooth_band(kind, &lo, &hi).expect("smooth band");
    let bracket = |x: &BigRational| match kind {
        SmoothKind::Sin => sin_bracket(x),
        SmoothKind::Cos => cos_bracket(x),
        SmoothKind::Sigmoid => sigmoid_bracket(x),
    };
    let checked = check_band_rational(&band, &lo, &hi, &mut rng, bracket);
    assert!(checked > 200, "checked only {checked} points");
}

/// Named V4 test 2 (sin): the chord band contains `sin` at every sampled point
/// in exact rational arithmetic.
#[test]
fn fwd_root_screen_band_exact_rational_sin() {
    smooth_rational(SmoothKind::Sin, 2.5, 0x0B51);
}

/// Named V4 test 2 (cos).
#[test]
fn fwd_root_screen_band_exact_rational_cos() {
    smooth_rational(SmoothKind::Cos, 2.5, 0x0B52);
}

/// Named V4 test 2 (sigmoid), including wide mixed-curvature intervals.
#[test]
fn fwd_root_screen_band_exact_rational_sigmoid() {
    smooth_rational(SmoothKind::Sigmoid, 7.0, 0x0B53);
}

/// Named V4 test 2 (floor): the `[x - 1, x]` band and the exact constant.
#[test]
fn fwd_root_screen_band_exact_rational_floor() {
    let mut rng = Rng::new(0x0B54);
    let (mut lo, mut hi) = band_intervals(&mut rng, 4.0, 40);
    // Intervals that sit exactly on and just around integer boundaries.
    for (a, b) in [
        (2.0, 2.0),
        (2.0, 3.0),
        (-1.0, -0.5),
        (1.999_999_9, 2.000_000_1),
    ] {
        lo.push(a);
        hi.push(b);
    }
    let band = floor_band(&lo, &hi).expect("floor band");
    let checked = check_band_rational(&band, &lo, &hi, &mut rng, |x| {
        let floor = BigRational::from_integer(x.floor().to_integer());
        (floor.clone(), floor)
    });
    assert!(checked > 200, "checked only {checked} points");
    assert!(
        band.lambda.contains(&0.0) && band.lambda.contains(&1.0),
        "precondition: both the constant and the [x-1, x] arm are exercised"
    );
}

/// The lookup-table band is exact in rationals too (C1's point enclosures).
#[test]
fn fwd_root_screen_band_exact_rational_pwl() {
    let thresholds = [-1.5f32, -0.6, -0.05, 0.4, 1.1, 1.9];
    let weights = [0.4f32, -0.9, 1.3, -0.35, 0.8, -1.1];
    let bias = 0.125f32;
    let mut rng = Rng::new(0x0B55);
    let (lo, hi) = band_intervals(&mut rng, 2.0, 40);
    let band = pwl_band(&lo, &hi, &thresholds, &weights, bias).expect("pwl band");
    let checked = check_band_rational(&band, &lo, &hi, &mut rng, |x| {
        let mut f = rat(f64::from(bias));
        for (&b, &w) in thresholds.iter().zip(&weights) {
            let d = x - rat(f64::from(b));
            if d > BigRational::zero() {
                f += rat(f64::from(w)) * d;
            }
        }
        (f.clone(), f)
    });
    assert!(checked > 200, "checked only {checked} points");
}

// ---------------------------------------------------------------------------
// Probes, motif, refusals, dark default
// ---------------------------------------------------------------------------

/// The probed selection maps are ny's own: Slice windows, Gather with
/// duplicates, the rank-1 Transpose identity, Concat offsets; and the lookup
/// table compiles to ONE exact band with its hinge nodes skipped.
#[test]
fn fwd_root_screen_probe_maps_match_ny_semantics() {
    let graph = mini_acopf_graph(0xF00D_0003);
    let input = input_box(&[-1.0, -1.0, -1.0, -1.0], &[1.0, 1.0, 1.0, 1.0]);
    let (program, _) = DagProgram::compile_with_ibp(&graph, &input, None).expect("compile");
    let step = |name: &str| {
        program
            .steps()
            .iter()
            .find(|s| s.node == name)
            .unwrap_or_else(|| panic!("no step `{name}`"))
    };
    let live = |elems: &[usize]| -> Vec<SelectSource> {
        elems
            .iter()
            .map(|&elem| SelectSource::Live { input: 0, elem })
            .collect()
    };
    match &step("vm_raw").op {
        DagOp::Select { sources, identity } => {
            assert_eq!(sources, &live(&[3, 4, 5]));
            assert!(!*identity);
        }
        other => panic!("vm_raw compiled to {other:?}"),
    }
    match &step("gi").op {
        DagOp::Select { sources, .. } => assert_eq!(sources, &live(&[0, 2, 1, 0])),
        other => panic!("gi compiled to {other:?}"),
    }
    match &step("col").op {
        DagOp::Select { identity, .. } => {
            assert!(*identity, "a rank-1 Transpose is an identity relabel")
        }
        other => panic!("col compiled to {other:?}"),
    }
    match &step("out").op {
        DagOp::Select { sources, .. } => {
            assert_eq!(sources[0], SelectSource::Live { input: 0, elem: 0 });
            assert_eq!(sources[4], SelectSource::Live { input: 1, elem: 0 });
            assert_eq!(sources.len(), 4 + 4 + 4 + 2 + 4 + 2 + 3 + 4 + 7);
        }
        other => panic!("out compiled to {other:?}"),
    }
    match &step("tslope").op {
        DagOp::Pwl { thresholds, .. } => assert_eq!(thresholds.len(), 5),
        other => panic!("the lookup-table head compiled to {other:?}"),
    }
    assert_eq!(step("tslope").inputs, vec!["tcol".to_string()]);
    let skipped = program.skipped();
    assert!(
        skipped.contains(&"tshift".to_string()) && skipped.contains(&"thinge".to_string()),
        "the hinge nodes must be skipped, got {skipped:?}"
    );
    assert!(program.census().contains("pwl:1"), "{}", program.census());
}

/// Unsupported layers and exponents refuse at compile time (fail closed).
#[test]
fn fwd_root_screen_refuses_unsupported_ops() {
    let mut exp = GraphNetwork::new();
    let w = Array2::from_shape_vec((2, 2), vec![1.0f32, 0.0, 0.0, 1.0]).expect("w");
    exp.add_node(GraphNode::from_input(
        "z",
        Layer::Linear(LinearLayer::new(w.clone(), None::<Array1<f32>>).expect("z")),
    ));
    exp.add_node(node("e", Layer::Exp(ExpLayer::new()), &["z"]));
    exp.set_output("e");
    let input = input_box(&[-1.0, -1.0], &[1.0, 1.0]);
    match DagProgram::compile_with_ibp(&exp, &input, None) {
        Err(ZonoRefusal::UnsupportedLayer { node, .. }) => assert_eq!(node, "e"),
        other => panic!("Exp must refuse, got {other:?}"),
    }

    let mut cube = GraphNetwork::new();
    cube.add_node(GraphNode::from_input(
        "z",
        Layer::Linear(LinearLayer::new(w, None::<Array1<f32>>).expect("z")),
    ));
    cube.add_node(node(
        "c",
        Layer::PowConstant(PowConstantLayer::new(3.0)),
        &["z"],
    ));
    cube.set_output("c");
    assert!(
        matches!(
            DagProgram::compile_with_ibp(&cube, &input, None),
            Err(ZonoRefusal::UnsupportedGraph(_))
        ),
        "x^3 must refuse"
    );
    match run_forward_root_screen(&cube, &input, BIG_BUDGET, None, true) {
        Err(decline) => assert_eq!(decline.census, "-"),
        Ok(_) => panic!("the screen must decline an unsupported graph"),
    }
}

/// A program's IBP map encloses the node values over the box it was computed
/// on and over nothing else: an intersecting pass on any other box refuses
/// (the stale-map-across-domains class), while the pure pass reads no map.
#[test]
fn fwd_root_screen_refuses_a_stale_ibp_map() {
    let graph = product_graph();
    let root = input_box(&[-0.05, -0.04], &[0.05, 0.06]);
    let child = input_box(&[-0.05, -0.04], &[0.0, 0.06]);
    let (program, ibp) = DagProgram::compile_with_ibp(&graph, &root, None).expect("compile");
    let intersecting = DagForwardOptions {
        intersect_ibp: true,
        byte_budget: BIG_BUDGET,
    };
    program
        .forward(&root, &intersecting, &ibp, None, None)
        .expect("the map's own box runs");
    match program.forward(&child, &intersecting, &ibp, None, None) {
        Err(ZonoRefusal::Internal(why)) => assert!(why.contains("intersect_ibp"), "{why}"),
        other => panic!("a stale IBP map must refuse, got {other:?}"),
    }
    let pure = DagForwardOptions {
        intersect_ibp: false,
        byte_budget: BIG_BUDGET,
    };
    program
        .forward(&child, &pure, &ibp, None, None)
        .expect("the pure pass reads no map");
}

/// A byte budget below the program's first live state refuses (never runs).
#[test]
fn fwd_root_screen_memory_budget_refuses_instead_of_running() {
    let graph = mini_acopf_graph(0xF00D_0004);
    let input = input_box(&[-1.0, -1.0, -1.0, -1.0], &[1.0, 1.0, 1.0, 1.0]);
    match run_forward_root_screen(&graph, &input, 1024, None, true) {
        Err(decline) => {
            assert!(
                matches!(decline.reason, ZonoRefusal::Memory { .. }),
                "expected a memory refusal, got {:?}",
                decline.reason
            );
            assert_ne!(decline.census, "-");
        }
        Ok(_) => panic!("a 1 KiB budget cannot admit the pass"),
    }
}

/// Dark by default: the config default is off, no plan exists, and serde fills
/// both fields when absent; an armed config yields a plan whose source is the
/// typed config, never the legacy environment.
#[test]
fn fwd_root_screen_is_dark_by_default() {
    let default = BetaCrownConfig::default();
    assert!(!default.forward_root_screen);
    assert_eq!(default.forward_root_screen_max_mb, 8192);
    assert!(BetaCrownConfig::forward_root_screen_max_mb_is_valid(
        default.forward_root_screen_max_mb
    ));
    assert!(!BetaCrownConfig::forward_root_screen_max_mb_is_valid(0));
    assert!(forward_root_screen_plan(&default).is_none());
    let absent: BetaCrownConfig = serde_json::from_str("{}").expect("empty config");
    assert!(!absent.forward_root_screen);
    assert_eq!(absent.forward_root_screen_max_mb, 8192);
    let armed = BetaCrownConfig {
        forward_root_screen: true,
        ..BetaCrownConfig::default()
    };
    let plan = forward_root_screen_plan(&armed).expect("armed plan");
    assert_eq!(plan.source, ny_levers::Source::Config);
    assert_ne!(plan.source, ny_levers::Source::LegacyEnv);
    assert_eq!(plan.max_bytes, 8192u64 << 20);
}
