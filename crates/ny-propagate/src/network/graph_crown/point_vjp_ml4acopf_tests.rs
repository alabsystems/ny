// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #ml4acopf-vjp — gradient checks for the variants admitted into the exact
//! point-VJP fragment for the ml4acopf_2024 power-flow nets.
//!
//! Every admitted variant gets a small hand-built [`GraphNetwork`] that uses
//! it, and each test asserts THREE things:
//!
//! 1. the static gate admits the net (`supports_attack_point_gradient()`),
//! 2. [`GraphNetwork::attack_point_gradient`] returns `Some` and matches a
//!    CENTRAL FINITE DIFFERENCE of the graph's own concrete forward
//!    ([`GraphNetwork::propagate_concrete_point`]) to ~1e-3 relative at several
//!    random points, and
//! 3. where the derivative has a closed form, the gradient matches it to ~1e-4
//!    (a finite difference cannot see a slope-0 surrogate at an isolated point,
//!    a closed form can).
//!
//! The last test pins the other side: variants deliberately NOT admitted keep
//! the net outside the fragment, and the diagnostic names the blocking node.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::layers::{
    AddConstantLayer, AddLayer, ConcatLayer, CosLayer, DivConstantLayer, DivLayer, FloorLayer,
    GatherLayer, Layer, LinearLayer, MulBinaryLayer, MulConstantLayer, PowConstantLayer, ReLULayer,
    SigmoidLayer, SinLayer, SliceLayer, SqueezeLayer, SubConstantLayer, SubLayer, TanhLayer,
    TransposeLayer, UnsqueezeLayer, WhereLayer,
};
use crate::network::{GraphNetwork, GraphNode};

/// Central-difference half step. Large enough that f32 forward noise (a few
/// ULP of O(1) outputs, ~1e-6) contributes ~1e-4 to the quotient; small enough
/// that the O(h²) truncation of these O(1)-curvature nets stays ~1e-5.
const FD_STEP: f32 = 4e-3;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Deterministic xorshift64* stream (no `rand` dependency in the fixture).
struct TestRng(u64);

impl TestRng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit() as f32
    }

    fn vector(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }

    fn matrix(&mut self, rows: usize, cols: usize, scale: f32) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |_| self.uniform(-scale, scale))
    }
}

fn linear_layer(weight: Array2<f32>, bias: Option<Vec<f32>>) -> Layer {
    Layer::Linear(LinearLayer::new(weight, bias.map(Array1::from_vec)).expect("valid linear"))
}

fn scalar(value: f32) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(&[]), value)
}

fn node(name: &str, layer: Layer, inputs: &[&str]) -> GraphNode {
    GraphNode::new(
        name,
        layer,
        inputs.iter().map(|input| (*input).to_string()).collect(),
    )
}

fn as_input(x: &[f32]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(&[x.len()]), x.to_vec()).expect("1-D input")
}

/// The graph's own concrete (point) forward at `x`, flattened.
fn concrete_output(graph: &GraphNetwork, x: &[f32]) -> Vec<f32> {
    let input = BoundedTensor::concrete(as_input(x)).expect("concrete input");
    let out = graph
        .propagate_concrete_point(&input, None, None)
        .expect("concrete point forward");
    out.lower().iter().copied().collect()
}

/// `spec · y(x)` in f64.
fn objective(graph: &GraphNetwork, spec: &[f32], x: &[f32]) -> f64 {
    let out = concrete_output(graph, x);
    assert_eq!(out.len(), spec.len(), "spec row must cover every output");
    out.iter()
        .zip(spec)
        .map(|(&y, &s)| f64::from(y) * f64::from(s))
        .sum()
}

/// The point-VJP under test; panics unless the net is admitted.
fn point_gradient(graph: &GraphNetwork, spec: &[f32], x: &[f32]) -> Vec<f32> {
    let row = Array2::from_shape_vec((1, spec.len()), spec.to_vec()).expect("spec row");
    let grad = graph
        .attack_point_gradient(&as_input(x), &row, None, None)
        .expect("attack_point_gradient must not error on an admitted net")
        .expect("an admitted net must yield Some(gradient)");
    assert_eq!(
        grad.shape(),
        &[x.len()],
        "gradient must have the input shape"
    );
    let grad: Vec<f32> = grad.iter().copied().collect();
    assert!(
        grad.iter().all(|g| g.is_finite()),
        "gradient must be finite, got {grad:?}"
    );
    grad
}

/// Compare the point-VJP against a central difference of the concrete
/// forward, coordinate by coordinate. With `kink_gate`, a coordinate whose
/// one-sided differences disagree (the ±h step crossed a ReLU/Floor kink, where
/// the exact gradient is one-sided and a central difference is meaningless) is
/// skipped. Returns how many coordinates were compared.
fn compare_with_central_difference(
    label: &str,
    graph: &GraphNetwork,
    spec: &[f32],
    x: &[f32],
    kink_gate: bool,
) -> usize {
    let grad = point_gradient(graph, spec, x);
    let f0 = objective(graph, spec, x);
    let mut compared = 0usize;
    for i in 0..x.len() {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += FD_STEP;
        xm[i] -= FD_STEP;
        // The step actually taken in f32.
        let hp = f64::from(xp[i]) - f64::from(x[i]);
        let hm = f64::from(x[i]) - f64::from(xm[i]);
        let fp = objective(graph, spec, &xp);
        let fm = objective(graph, spec, &xm);
        let central = (fp - fm) / (hp + hm);
        if kink_gate {
            // Gate at the comparison tolerance itself: when only one side of the
            // step crosses a kink, the exact one-sided gradient is one of the two
            // one-sided quotients, so agreement within `tol` bounds
            // |central − exact| by tol/2. A looser gate would let a small slope
            // change through and fail the comparison below spuriously.
            let forward = (fp - f0) / hp;
            let backward = (f0 - fm) / hm;
            if (forward - backward).abs() > 1e-3 * (1.0 + central.abs()) {
                continue;
            }
        }
        let g = f64::from(grad[i]);
        let tol = 1e-3 * (1.0 + g.abs().max(central.abs()));
        assert!(
            (g - central).abs() <= tol,
            "{label}: coord {i} at x = {x:?}: point-VJP {g} vs central difference {central} \
             (tol {tol})"
        );
        compared += 1;
    }
    compared
}

/// Admission + finite-difference agreement at `points` random points drawn
/// from `[lo, hi]^n`, with a random spec row.
fn assert_admitted_and_matches_fd(
    label: &str,
    graph: &GraphNetwork,
    input_dim: usize,
    (lo, hi): (f32, f32),
    points: usize,
    seed: u64,
) {
    assert!(
        graph.supports_attack_point_gradient(),
        "{label}: the net must be inside the exact point-VJP fragment (first refusal: {:?})",
        graph.first_attack_point_gradient_refusal()
    );
    assert_eq!(graph.first_attack_point_gradient_refusal(), None);
    let mut rng = TestRng::new(seed);
    let out_dim = concrete_output(graph, &vec![f32::midpoint(lo, hi); input_dim]).len();
    for _ in 0..points {
        let x = rng.vector(input_dim, lo, hi);
        let spec = rng.vector(out_dim, -1.0, 1.0);
        let compared = compare_with_central_difference(label, graph, &spec, &x, false);
        assert_eq!(
            compared, input_dim,
            "{label}: every coordinate must be compared"
        );
    }
}

/// `grad_i = Σ_j spec_j · f'(z_j) · W[j, i]` with `z = W x + b`, in f64 — the
/// closed-form gradient of `spec · f(W x + b)` for an element-wise `f`.
fn closed_form_elementwise_gradient(
    weight: &Array2<f32>,
    bias: &[f32],
    x: &[f32],
    spec: &[f32],
    derivative: impl Fn(f64) -> f64,
) -> Vec<f64> {
    let (rows, cols) = weight.dim();
    let mut grad = vec![0.0f64; cols];
    for j in 0..rows {
        let z: f64 = (0..cols)
            .map(|i| f64::from(weight[[j, i]]) * f64::from(x[i]))
            .sum::<f64>()
            + f64::from(bias[j]);
        let scale = f64::from(spec[j]) * derivative(z);
        for (i, g) in grad.iter_mut().enumerate() {
            *g += scale * f64::from(weight[[j, i]]);
        }
    }
    grad
}

/// `Linear(W, b) -> <element-wise layer>` as a graph.
fn affine_then(weight: &Array2<f32>, bias: &[f32], layer: Layer) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "z",
        linear_layer(weight.clone(), Some(bias.to_vec())),
    ));
    graph.add_node(node("act", layer, &["z"]));
    graph.set_output("act");
    graph
}

/// Point-VJP vs the closed form at `x`, to 1e-4 relative.
fn assert_matches_closed_form(label: &str, grad: &[f32], reference: &[f64]) {
    assert_eq!(grad.len(), reference.len());
    for (i, (&g, &r)) in grad.iter().zip(reference).enumerate() {
        let tol = 1e-4 * (1.0 + r.abs());
        assert!(
            (f64::from(g) - r).abs() <= tol,
            "{label}: coord {i}: point-VJP {g} vs closed form {r} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// Linear graph plumbing and the live×live product
// ---------------------------------------------------------------------------

/// Live×live `MulBinary`, including the square `Mul(a, a)` whose two routed
/// cotangent rows must SUM on the shared input (d(a²) = 2a).
#[test]
fn point_vjp_live_live_mul_binary_matches_finite_differences() {
    let mut rng = TestRng::new(0x4D55_4C31);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "a",
        linear_layer(rng.matrix(4, 3, 0.8), Some(rng.vector(4, -0.3, 0.3))),
    ));
    graph.add_node(GraphNode::from_input(
        "b",
        linear_layer(rng.matrix(4, 3, 0.8), Some(rng.vector(4, -0.3, 0.3))),
    ));
    graph.add_node(node("ab", Layer::MulBinary(MulBinaryLayer), &["a", "b"]));
    graph.add_node(node("aa", Layer::MulBinary(MulBinaryLayer), &["a", "a"]));
    graph.add_node(node("out", Layer::Add(AddLayer), &["ab", "aa"]));
    graph.set_output("out");
    assert_admitted_and_matches_fd("mul_binary", &graph, 3, (-1.0, 1.0), 4, 11);
}

/// Live−live `Sub` with a reconverging input: `(a − b) − a == −b`, so the
/// gradient is EXACTLY `−spec · W_b` whatever `W_a` is.
#[test]
fn point_vjp_live_live_sub_matches_finite_differences() {
    let mut rng = TestRng::new(0x5355_4231);
    let w_a = rng.matrix(3, 4, 1.0);
    let w_b = rng.matrix(3, 4, 1.0);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "a",
        linear_layer(w_a, Some(rng.vector(3, -0.5, 0.5))),
    ));
    graph.add_node(GraphNode::from_input(
        "b",
        linear_layer(w_b.clone(), Some(rng.vector(3, -0.5, 0.5))),
    ));
    graph.add_node(node("d", Layer::Sub(SubLayer), &["a", "b"]));
    graph.add_node(node("out", Layer::Sub(SubLayer), &["d", "a"]));
    graph.set_output("out");
    assert_admitted_and_matches_fd("sub", &graph, 4, (-1.0, 1.0), 3, 12);

    let x = rng.vector(4, -1.0, 1.0);
    let spec = rng.vector(3, -1.0, 1.0);
    let grad = point_gradient(&graph, &spec, &x);
    let reference: Vec<f64> = (0..4)
        .map(|i| {
            -(0..3)
                .map(|j| f64::from(spec[j]) * f64::from(w_b[[j, i]]))
                .sum::<f64>()
        })
        .collect();
    assert_matches_closed_form("sub", &grad, &reference);
}

/// N-ary `Concat` with a REPEATED input (`[a, b, a]`): each slot's cotangent
/// must land on its own input and the two `a` slots must sum.
#[test]
fn point_vjp_concat_matches_finite_differences() {
    let mut rng = TestRng::new(0x434F_4E43);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "a",
        linear_layer(rng.matrix(3, 4, 0.8), Some(rng.vector(3, -0.3, 0.3))),
    ));
    graph.add_node(GraphNode::from_input(
        "b",
        linear_layer(rng.matrix(2, 4, 0.8), Some(rng.vector(2, -0.3, 0.3))),
    ));
    graph.add_node(node(
        "cat",
        Layer::Concat(ConcatLayer::new(0)),
        &["a", "b", "a"],
    ));
    // A product AFTER the concat makes the gradient position-dependent, so a
    // mis-routed slot cannot pass by symmetry.
    graph.add_node(node(
        "out",
        Layer::MulBinary(MulBinaryLayer),
        &["cat", "cat"],
    ));
    graph.set_output("out");
    assert_admitted_and_matches_fd("concat", &graph, 4, (-1.0, 1.0), 3, 13);
}

/// `Slice` into two overlapping windows of one tensor, recombined by a product.
#[test]
fn point_vjp_slice_matches_finite_differences() {
    let mut rng = TestRng::new(0x534C_4943);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "h",
        linear_layer(rng.matrix(6, 3, 0.8), Some(rng.vector(6, -0.3, 0.3))),
    ));
    graph.add_node(node("lo", Layer::Slice(SliceLayer::new(0, 1, 4)), &["h"]));
    graph.add_node(node("hi", Layer::Slice(SliceLayer::new(0, 3, 6)), &["h"]));
    graph.add_node(node("out", Layer::MulBinary(MulBinaryLayer), &["lo", "hi"]));
    graph.set_output("out");
    assert_admitted_and_matches_fd("slice", &graph, 3, (-1.0, 1.0), 3, 14);
}

/// Static-index `Gather` with DUPLICATE indices (the ml4acopf branch→bus
/// incidence gathers): duplicated reads must scatter-add on the backward.
#[test]
fn point_vjp_gather_with_static_duplicate_indices_matches_finite_differences() {
    let mut rng = TestRng::new(0x4741_5448);
    let from_bus = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0i64, 2, 2, 4]).expect("indices");
    let to_bus = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1i64, 3, 0, 0]).expect("indices");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "v",
        linear_layer(rng.matrix(5, 3, 0.8), Some(rng.vector(5, -0.3, 0.3))),
    ));
    graph.add_node(node(
        "vi",
        Layer::Gather(GatherLayer::new(0, Some(from_bus), Vec::new())),
        &["v"],
    ));
    graph.add_node(node(
        "vj",
        Layer::Gather(GatherLayer::new(0, Some(to_bus), Vec::new())),
        &["v"],
    ));
    graph.add_node(node("out", Layer::MulBinary(MulBinaryLayer), &["vi", "vj"]));
    graph.set_output("out");
    assert_admitted_and_matches_fd("gather", &graph, 3, (-1.0, 1.0), 3, 15);
}

/// The ml4acopf-linear piecewise-linear activation head, verbatim in shape:
/// `Unsqueeze(-1) -> SubConstant(thresholds, broadcast) -> ReLU ->
/// MatMul(delta_m) [= Linear + builder-inserted Squeeze(-1)] -> AddConstant`.
/// Piecewise linear, so coordinates whose ±h step crosses a threshold are gated.
#[test]
fn point_vjp_unsqueeze_squeeze_piecewise_linear_head_matches_finite_differences() {
    let mut rng = TestRng::new(0x5351_5545);
    let thresholds = ArrayD::from_shape_vec(IxDyn(&[5]), vec![-1.0f32, -0.45, 0.05, 0.6, 1.2])
        .expect("thresholds");
    let delta_m =
        Array2::from_shape_vec((1, 5), vec![0.2f32, 0.5, 0.9, -0.4, -0.7]).expect("delta_m");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "z",
        linear_layer(rng.matrix(4, 3, 1.0), Some(rng.vector(4, -0.3, 0.3))),
    ));
    graph.add_node(node(
        "col",
        Layer::Unsqueeze(UnsqueezeLayer::new(-1)),
        &["z"],
    ));
    graph.add_node(node(
        "shift",
        Layer::SubConstant(SubConstantLayer::new(thresholds)),
        &["col"],
    ));
    graph.add_node(node("hinge", Layer::ReLU(ReLULayer), &["shift"]));
    graph.add_node(node("slopes", linear_layer(delta_m, None), &["hinge"]));
    graph.add_node(node(
        "flat",
        Layer::Squeeze(SqueezeLayer::new(-1)),
        &["slopes"],
    ));
    graph.add_node(node(
        "out",
        Layer::AddConstant(AddConstantLayer::new(scalar(0.25))),
        &["flat"],
    ));
    graph.set_output("out");
    assert!(
        graph.supports_attack_point_gradient(),
        "piecewise-linear head must be admitted (first refusal: {:?})",
        graph.first_attack_point_gradient_refusal()
    );

    let mut compared = 0usize;
    let points = 6usize;
    for _ in 0..points {
        let x = rng.vector(3, -1.0, 1.0);
        let spec = rng.vector(4, -1.0, 1.0);
        compared += compare_with_central_difference("pwl_head", &graph, &spec, &x, true);
    }
    assert!(
        compared >= points * 3 / 2,
        "too few kink-free coordinates were compared ({compared} of {})",
        points * 3
    );
}

/// The ml4acopf flow block `Transpose(perm [1, 0]) -> constant-left MatMul ->
/// Transpose(perm [1, 0])` as ny lowers it for a batched model: the ONNX perm
/// is kept verbatim and the MatMul becomes a last-axis `Linear` on the rank-1
/// internal tensor. In the graph's own (batch-stripped) convention the rank-1
/// Transposes are identities and the block is exact — which is why the attack
/// lanes must hand such a net `attack_point_gradient_input_shape([1, n]) == [n]`
/// rather than the declared `[1, n]` (whose Transpose would yield `[n, 1]` and
/// fail the Linear's in-features check).
#[test]
fn point_vjp_transpose_into_constant_left_linear_matches_finite_differences() {
    let mut rng = TestRng::new(0x5452_4E53);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "h",
        linear_layer(rng.matrix(6, 4, 0.8), Some(rng.vector(6, -0.3, 0.3))),
    ));
    graph.add_node(node(
        "load",
        Layer::Slice(SliceLayer::new(-1, 0, 3)),
        &["h"],
    ));
    graph.add_node(node(
        "col",
        Layer::Transpose(TransposeLayer::new(vec![1, 0])),
        &["load"],
    ));
    graph.add_node(node(
        "flow",
        linear_layer(rng.matrix(5, 3, 1.0), None),
        &["col"],
    ));
    graph.add_node(node(
        "row",
        Layer::Transpose(TransposeLayer::new(vec![1, 0])),
        &["flow"],
    ));
    graph.add_node(node("out", Layer::Sin(SinLayer::new()), &["row"]));
    graph.set_output("out");
    assert_eq!(
        graph.attack_point_gradient_input_shape(&[1, 4]),
        vec![4],
        "a net using the #ml4acopf-vjp additions takes the batch-stripped shape"
    );
    assert_admitted_and_matches_fd("transpose_linear", &graph, 4, (-1.0, 1.0), 3, 41);
}

/// The attack-lane input-shape rule: verbatim for every net that uses none of
/// the #ml4acopf-vjp additions (so every historically admitted net is
/// byte-identical), leading size-1 axis dropped otherwise — and nothing else.
#[test]
fn attack_point_gradient_input_shape_strips_the_batch_axis_only_on_addition_nets() {
    let mut legacy = GraphNetwork::new();
    legacy.add_node(GraphNode::from_input(
        "z",
        linear_layer(Array2::<f32>::eye(3), None),
    ));
    legacy.add_node(node("r", Layer::ReLU(ReLULayer), &["z"]));
    legacy.set_output("r");
    assert!(legacy.supports_attack_point_gradient());
    assert_eq!(
        legacy.attack_point_gradient_input_shape(&[1, 3]),
        vec![1, 3]
    );
    assert_eq!(legacy.attack_point_gradient_input_shape(&[3]), vec![3]);

    let addition = affine_then(
        &Array2::<f32>::eye(3),
        &[0.0; 3],
        Layer::Sigmoid(SigmoidLayer::new()),
    );
    assert_eq!(addition.attack_point_gradient_input_shape(&[1, 3]), vec![3]);
    assert_eq!(
        addition.attack_point_gradient_input_shape(&[1, 1, 3]),
        vec![1, 3]
    );
    assert_eq!(addition.attack_point_gradient_input_shape(&[3]), vec![3]);
    assert_eq!(
        addition.attack_point_gradient_input_shape(&[2, 3]),
        vec![2, 3]
    );
    assert_eq!(addition.attack_point_gradient_input_shape(&[1]), vec![1]);
}

// ---------------------------------------------------------------------------
// Smooth / piecewise-constant element-wise ops (the analytic-derivative arm)
// ---------------------------------------------------------------------------

#[test]
fn point_vjp_sigmoid_uses_the_exact_derivative() {
    let mut rng = TestRng::new(0x5349_474D);
    let weight = rng.matrix(4, 3, 1.5);
    let bias = rng.vector(4, -0.5, 0.5);
    let graph = affine_then(&weight, &bias, Layer::Sigmoid(SigmoidLayer::new()));
    assert_admitted_and_matches_fd("sigmoid", &graph, 3, (-1.5, 1.5), 4, 21);

    let x = rng.vector(3, -1.5, 1.5);
    let spec = rng.vector(4, -1.0, 1.0);
    let reference = closed_form_elementwise_gradient(&weight, &bias, &x, &spec, |z| {
        let s = 1.0 / (1.0 + (-z).exp());
        s * (1.0 - s)
    });
    assert_matches_closed_form("sigmoid", &point_gradient(&graph, &spec, &x), &reference);
}

/// Sin: random points AND the two regimes where the CROWN relaxation degrades
/// to a slope-0 constant — a pre-activation box straddling π (the inflection)
/// or 2π (the normalisation wrap). The exact derivative there is cos(π) = −1
/// and cos(2π) = 1.
#[test]
fn point_vjp_sin_is_exact_including_the_pi_and_two_pi_boundaries() {
    let mut rng = TestRng::new(0x5349_4E30);
    let weight = rng.matrix(3, 3, 1.5);
    let bias = rng.vector(3, -0.5, 0.5);
    let graph = affine_then(&weight, &bias, Layer::Sin(SinLayer::new()));
    assert_admitted_and_matches_fd("sin", &graph, 3, (-2.0, 2.0), 4, 22);

    // Identity affine map, so the Sin pre-activation IS x (up to the certified
    // forward's ULP widening, which is what can straddle the boundary).
    let identity = Array2::<f32>::eye(3);
    let zero_bias = [0.0f32; 3];
    let boundary = affine_then(&identity, &zero_bias, Layer::Sin(SinLayer::new()));
    let x = [PI, TAU, 0.3f32];
    let spec = [1.0f32, 1.0, 1.0];
    let grad = point_gradient(&boundary, &spec, &x);
    let reference = closed_form_elementwise_gradient(&identity, &zero_bias, &x, &spec, f64::cos);
    assert_matches_closed_form("sin@boundary", &grad, &reference);
    assert!(
        (grad[0] + 1.0).abs() < 1e-4 && (grad[1] - 1.0).abs() < 1e-4,
        "d sin at π and 2π must be −1 and +1, got {grad:?}"
    );
}

/// Cos: random points AND its inflection points π/2 and 3π/2, where the exact
/// derivative −sin(z) is −1 and +1.
#[test]
fn point_vjp_cos_is_exact_including_its_inflection_points() {
    let mut rng = TestRng::new(0x434F_5330);
    let weight = rng.matrix(3, 3, 1.5);
    let bias = rng.vector(3, -0.5, 0.5);
    let graph = affine_then(&weight, &bias, Layer::Cos(CosLayer::new()));
    assert_admitted_and_matches_fd("cos", &graph, 3, (-2.0, 2.0), 4, 23);

    let identity = Array2::<f32>::eye(3);
    let zero_bias = [0.0f32; 3];
    let boundary = affine_then(&identity, &zero_bias, Layer::Cos(CosLayer::new()));
    let x = [FRAC_PI_2, 3.0 * FRAC_PI_2, TAU];
    let spec = [1.0f32, 1.0, 1.0];
    let grad = point_gradient(&boundary, &spec, &x);
    let reference =
        closed_form_elementwise_gradient(&identity, &zero_bias, &x, &spec, |z| -z.sin());
    assert_matches_closed_form("cos@inflection", &grad, &reference);
    assert!(
        (grad[0] + 1.0).abs() < 1e-4 && (grad[1] - 1.0).abs() < 1e-4,
        "d cos at π/2 and 3π/2 must be −1 and +1, got {grad:?}"
    );
}

/// `PowConstant` for the ml4acopf square (p = 2) over negative and positive
/// bases, an odd power (p = 3), and a fractional power on a positive base.
#[test]
fn point_vjp_pow_constant_uses_the_exact_derivative() {
    let mut rng = TestRng::new(0x504F_5733);
    for p in [2.0f32, 3.0] {
        let weight = rng.matrix(3, 3, 1.0);
        let bias = rng.vector(3, -0.5, 0.5);
        let graph = affine_then(&weight, &bias, Layer::PowConstant(PowConstantLayer::new(p)));
        let label = format!("pow{p}");
        assert_admitted_and_matches_fd(&label, &graph, 3, (-1.0, 1.0), 3, 24);

        let x = rng.vector(3, -1.0, 1.0);
        let spec = rng.vector(3, -1.0, 1.0);
        let reference = closed_form_elementwise_gradient(&weight, &bias, &x, &spec, |z| {
            f64::from(p) * z.powi(p as i32 - 1)
        });
        assert_matches_closed_form(&label, &point_gradient(&graph, &spec, &x), &reference);
    }

    // √z on a strictly positive base (identity affine map, x in [0.5, 2]).
    let identity = Array2::<f32>::eye(3);
    let zero_bias = [0.0f32; 3];
    let sqrt = affine_then(
        &identity,
        &zero_bias,
        Layer::PowConstant(PowConstantLayer::new(0.5)),
    );
    assert_admitted_and_matches_fd("pow0.5", &sqrt, 3, (0.5, 2.0), 3, 25);
}

/// ONNX `Neg` lowers to `MulConstant(-1)` (ny-build elementwise.rs). Already in
/// the fragment; pinned here because the ml4acopf reactive-flow term is
/// `Neg(Mul(Mul(vi, vj), Sin(θi − θj)))`.
#[test]
fn point_vjp_neg_as_mul_constant_matches_finite_differences() {
    let mut rng = TestRng::new(0x4E45_4730);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "z",
        linear_layer(rng.matrix(3, 3, 1.0), Some(rng.vector(3, -0.5, 0.5))),
    ));
    graph.add_node(node("s", Layer::Sin(SinLayer::new()), &["z"]));
    graph.add_node(node("zs", Layer::MulBinary(MulBinaryLayer), &["z", "s"]));
    graph.add_node(node(
        "neg",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        &["zs"],
    ));
    graph.set_output("neg");
    assert_admitted_and_matches_fd("neg", &graph, 3, (-1.5, 1.5), 3, 26);
}

/// ONNX `Div` by a constant lowers to `DivConstant` (already admitted); a live
/// numerator that is itself a product checks the 1/c scaling composes.
#[test]
fn point_vjp_div_constant_matches_finite_differences() {
    let mut rng = TestRng::new(0x4449_5630);
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "a",
        linear_layer(rng.matrix(3, 3, 1.0), Some(rng.vector(3, -0.5, 0.5))),
    ));
    graph.add_node(GraphNode::from_input(
        "b",
        linear_layer(rng.matrix(3, 3, 1.0), Some(rng.vector(3, -0.5, 0.5))),
    ));
    graph.add_node(node("ab", Layer::MulBinary(MulBinaryLayer), &["a", "b"]));
    graph.add_node(node(
        "out",
        Layer::DivConstant(DivConstantLayer::scalar(3.0)),
        &["ab"],
    ));
    graph.set_output("out");
    assert_admitted_and_matches_fd("div_constant", &graph, 3, (-1.0, 1.0), 3, 27);
}

/// The ml4acopf-linear-residual periodic range reduction
/// `r = t − 2π·floor(t / 2π)` with `t = z + π`. Floor's derivative is exactly
/// 0 a.e., so `dr/dz = 1`: through an identity affine map the gradient must be
/// the spec row itself. (A straight-through Floor would give `1 − 1 = 0`.)
#[test]
fn point_vjp_floor_range_reduction_keeps_the_exact_unit_slope() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "z",
        linear_layer(Array2::<f32>::eye(3), None),
    ));
    graph.add_node(node(
        "t",
        Layer::AddConstant(AddConstantLayer::new(scalar(PI))),
        &["z"],
    ));
    graph.add_node(node(
        "turns",
        Layer::DivConstant(DivConstantLayer::scalar(TAU)),
        &["t"],
    ));
    graph.add_node(node("whole", Layer::Floor(FloorLayer::new()), &["turns"]));
    graph.add_node(node(
        "wrap",
        Layer::MulConstant(MulConstantLayer::scalar(TAU)),
        &["whole"],
    ));
    graph.add_node(node("r", Layer::Sub(SubLayer), &["t", "wrap"]));
    graph.set_output("r");
    assert!(
        graph.supports_attack_point_gradient(),
        "Floor range reduction must be admitted (first refusal: {:?})",
        graph.first_attack_point_gradient_refusal()
    );

    // Points chosen so t / 2π sits well inside one integer cell: every ±h step
    // stays on one Floor plateau.
    let spec = [0.7f32, -1.3, 0.4];
    for x in [[0.3f32, 4.0, -2.5], [7.0, -7.0, 10.0]] {
        let grad = point_gradient(&graph, &spec, &x);
        for (i, (&g, &s)) in grad.iter().zip(&spec).enumerate() {
            assert!(
                (g - s).abs() < 1e-5,
                "coord {i}: d r / d z must be exactly 1 (grad {g}, spec {s})"
            );
        }
        let compared = compare_with_central_difference("floor", &graph, &spec, &x, true);
        assert_eq!(compared, 3, "no ±h step here crosses a Floor boundary");
    }
}

/// The analytic slopes themselves, including the undefined cases that must
/// degrade to 0 rather than poison the gradient with NaN/inf.
#[test]
fn exact_point_derivative_slopes_match_calculus_and_never_emit_nan() {
    use super::{pow_constant_point_derivative, ExactPointDerivative};

    let close = |a: f32, b: f64| (f64::from(a) - b).abs() <= 1e-6 * (1.0 + b.abs());
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    for z in [-3.0f64, -0.5, 0.0, 0.25, 2.0] {
        assert!(close(
            ExactPointDerivative::Sigmoid.slope(z),
            sig(z) * (1.0 - sig(z))
        ));
        assert!(close(ExactPointDerivative::Sin.slope(z), z.cos()));
        assert!(close(ExactPointDerivative::Cos.slope(z), -z.sin()));
        assert_eq!(ExactPointDerivative::Zero.slope(z), 0.0);
    }
    assert!(close(ExactPointDerivative::Pow(2.0).slope(-3.0), -6.0));
    assert!(close(ExactPointDerivative::Pow(3.0).slope(-2.0), 12.0));
    assert!(close(ExactPointDerivative::Pow(0.5).slope(4.0), 0.25));
    assert!(close(ExactPointDerivative::Pow(0.0).slope(5.0), 0.0));
    // Undefined / infinite derivatives degrade to 0.
    assert_eq!(ExactPointDerivative::Pow(0.5).slope(-1.0), 0.0);
    assert_eq!(ExactPointDerivative::Pow(0.5).slope(0.0), 0.0);
    assert_eq!(ExactPointDerivative::Pow(-1.0).slope(0.0), 0.0);
    assert_eq!(ExactPointDerivative::Pow(0.0).slope(0.0), 0.0);
    assert!(pow_constant_point_derivative(0.5, -1.0).is_nan());
    // Extreme sigmoid arguments saturate to a finite 0, not NaN.
    assert_eq!(ExactPointDerivative::Sigmoid.slope(1.0e4), 0.0);
    assert_eq!(ExactPointDerivative::Sigmoid.slope(-1.0e4), 0.0);
}

// ---------------------------------------------------------------------------
// End to end: an ml4acopf-shaped power-flow block
// ---------------------------------------------------------------------------

/// A miniature of the ml4acopf output head: Gemm trunk -> Slice into
/// (magnitudes, angles) -> Sigmoid-scaled magnitudes -> static-index Gathers
/// onto branch endpoints -> angle differences (live−live Sub) -> Sin/Cos ->
/// live×live products `vi·vj·cos`, `−vi·vj·sin` (Neg) and `vi²` (Pow) ->
/// Concat of every residual. Checked against finite differences with the
/// disjunct-style single-output spec rows ml4acopf uses AND random rows.
#[test]
fn point_vjp_ml4acopf_shaped_power_flow_graph_matches_finite_differences() {
    let mut rng = TestRng::new(0x4143_4F50);
    let from_bus = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0i64, 1, 0]).expect("indices");
    let to_bus = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1i64, 2, 2]).expect("indices");
    let gather = |indices: &ArrayD<i64>| {
        Layer::Gather(GatherLayer::new(0, Some(indices.clone()), Vec::new()))
    };

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "trunk",
        linear_layer(rng.matrix(6, 4, 0.8), Some(rng.vector(6, -0.3, 0.3))),
    ));
    graph.add_node(node(
        "vm_raw",
        Layer::Slice(SliceLayer::new(0, 0, 3)),
        &["trunk"],
    ));
    graph.add_node(node(
        "va",
        Layer::Slice(SliceLayer::new(0, 3, 6)),
        &["trunk"],
    ));
    graph.add_node(node(
        "vm_sig",
        Layer::Sigmoid(SigmoidLayer::new()),
        &["vm_raw"],
    ));
    graph.add_node(node(
        "vm_scaled",
        Layer::MulConstant(MulConstantLayer::scalar(0.2)),
        &["vm_sig"],
    ));
    graph.add_node(node(
        "vm",
        Layer::AddConstant(AddConstantLayer::new(scalar(0.9))),
        &["vm_scaled"],
    ));
    graph.add_node(node("vmi", gather(&from_bus), &["vm"]));
    graph.add_node(node("vmj", gather(&to_bus), &["vm"]));
    graph.add_node(node("vai", gather(&from_bus), &["va"]));
    graph.add_node(node("vaj", gather(&to_bus), &["va"]));
    graph.add_node(node("dtheta", Layer::Sub(SubLayer), &["vai", "vaj"]));
    graph.add_node(node("cos", Layer::Cos(CosLayer::new()), &["dtheta"]));
    graph.add_node(node("sin", Layer::Sin(SinLayer::new()), &["dtheta"]));
    graph.add_node(node(
        "vv",
        Layer::MulBinary(MulBinaryLayer),
        &["vmi", "vmj"],
    ));
    graph.add_node(node(
        "p_flow",
        Layer::MulBinary(MulBinaryLayer),
        &["vv", "cos"],
    ));
    graph.add_node(node(
        "q_flow",
        Layer::MulBinary(MulBinaryLayer),
        &["vv", "sin"],
    ));
    graph.add_node(node(
        "q_neg",
        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
        &["q_flow"],
    ));
    graph.add_node(node(
        "vm_sq",
        Layer::PowConstant(PowConstantLayer::new(2.0)),
        &["vmi"],
    ));
    graph.add_node(node("p_res", Layer::Sub(SubLayer), &["vm_sq", "p_flow"]));
    graph.add_node(node(
        "residuals",
        Layer::Concat(ConcatLayer::new(0)),
        &["p_res", "q_neg", "vm"],
    ));
    graph.set_output("residuals");

    // Random spec rows at random points.
    assert_admitted_and_matches_fd("power_flow", &graph, 4, (-1.0, 1.0), 3, 31);

    // Disjunct-style rows: `Y_k >= c` differentiates to e_k, `Y_k <= -c` to −e_k.
    let x = rng.vector(4, -1.0, 1.0);
    let out_dim = concrete_output(&graph, &x).len();
    assert_eq!(out_dim, 9);
    for k in 0..out_dim {
        for sign in [1.0f32, -1.0] {
            let mut spec = vec![0.0f32; out_dim];
            spec[k] = sign;
            let compared =
                compare_with_central_difference("power_flow_e_k", &graph, &spec, &x, false);
            assert_eq!(compared, 4);
        }
    }
}

// ---------------------------------------------------------------------------
// The refusal side: variants deliberately left OUT of the fragment
// ---------------------------------------------------------------------------

/// Deliberately NOT admitted, and why:
/// - live/live `Div`: `dispatch_backward_layer` refuses it (its reciprocal
///   handler lives in the graph CROWN coordinator) and no benchmark needs it;
/// - live `Where` (ternary): its condition input is a comparison whose CROWN
///   backward is IBP-only, so there is no exact route; ml4acopf's Where cones
///   are all constant and fold away at load time;
/// - `Tanh`: smooth and would be easy, but no benchmark in scope needs it and
///   an unmeasured admission changes the post-BaB reserve ledger;
/// - DYNAMIC-index `Gather`: its backward is a conservative union, not a
///   Jacobian.
///
/// Each must keep the net outside the fragment, `attack_point_gradient` must
/// return `Ok(None)`, and the diagnostic must name the blocking node.
#[test]
fn point_vjp_refuses_nets_with_unadmitted_variants() {
    let mut rng = TestRng::new(0x5245_4655);
    let w = |rng: &mut TestRng| linear_layer(rng.matrix(2, 2, 1.0), Some(vec![0.1, -0.1]));

    let mut div = GraphNetwork::new();
    div.add_node(GraphNode::from_input("a", w(&mut rng)));
    div.add_node(GraphNode::from_input("b", w(&mut rng)));
    div.add_node(node("ratio", Layer::Div(DivLayer), &["a", "b"]));
    div.set_output("ratio");

    let mut where_live = GraphNetwork::new();
    where_live.add_node(GraphNode::from_input("cond", w(&mut rng)));
    where_live.add_node(GraphNode::from_input("then", w(&mut rng)));
    where_live.add_node(GraphNode::from_input("else", w(&mut rng)));
    where_live.add_node(node(
        "select",
        Layer::Where(WhereLayer::new()),
        &["cond", "then", "else"],
    ));
    where_live.set_output("select");

    let mut tanh = GraphNetwork::new();
    tanh.add_node(GraphNode::from_input("z", w(&mut rng)));
    tanh.add_node(node("squash", Layer::Tanh(TanhLayer::new()), &["z"]));
    tanh.set_output("squash");

    let mut dynamic_gather = GraphNetwork::new();
    dynamic_gather.add_node(GraphNode::from_input("z", w(&mut rng)));
    dynamic_gather.add_node(node(
        "pick",
        Layer::Gather(GatherLayer::new(0, None, vec![2])),
        &["z"],
    ));
    dynamic_gather.set_output("pick");

    let x = as_input(&[0.3, -0.2]);
    let row = Array2::from_shape_vec((1, 2), vec![1.0f32, -1.0]).expect("spec row");
    for (graph, blocker, layer_type) in [
        (&div, "ratio", "Div"),
        (&where_live, "select", "Where"),
        (&tanh, "squash", "Tanh"),
        (&dynamic_gather, "pick", "Gather"),
    ] {
        assert!(
            !graph.supports_attack_point_gradient(),
            "{blocker}: a net containing {layer_type} must stay OUTSIDE the fragment"
        );
        assert_eq!(
            graph.first_attack_point_gradient_refusal(),
            Some((blocker.to_string(), layer_type)),
            "the diagnostic must name the blocking node"
        );
        assert!(
            graph
                .attack_point_gradient(&x, &row, None, None)
                .expect("an unadmitted net is a typed fallback, not an error")
                .is_none(),
            "{blocker}: attack_point_gradient must decline with Ok(None)"
        );
    }

    // And the static-index Gather that IS admitted differs only in its indices.
    let mut static_gather = GraphNetwork::new();
    static_gather.add_node(GraphNode::from_input("z", w(&mut rng)));
    static_gather.add_node(node(
        "pick",
        Layer::Gather(GatherLayer::new(
            0,
            Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1i64, 0]).expect("indices")),
            Vec::new(),
        )),
        &["z"],
    ));
    static_gather.set_output("pick");
    assert!(static_gather.supports_attack_point_gradient());
    assert_eq!(static_gather.first_attack_point_gradient_refusal(), None);
}
