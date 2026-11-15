// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-attack-twin — the exact-reductions widening of the point-VJP fragment.
//!
//! The fixture is the Lyapunov shape of `lsnc_relu`: `V = sum(d ⊙ (P d))` with
//! `d = W_d x + b_d` (a `Linear -> Linear -> MulBinary -> ReduceSum` cone), next
//! to a ReLU branch `R = sum(ReLU(W_h x + b_h))`, concatenated into a two-output
//! head. Three facts are pinned:
//!
//! 1. The STANDARD fragment is unchanged: it still refuses the net, names the
//!    `ReduceSum` node, and `attack_point_gradient` still returns `Ok(None)` —
//!    every historical caller is byte-identical.
//! 2. The widened fragment admits exactly this net (no other refusal).
//! 3. The widened walk is the EXACT point-Jacobian: it matches the closed form
//!    `dV/dx = W_dᵀ (P + Pᵀ) d` to 1e-4 relative, and a kink-gated central
//!    difference of the graph's own concrete forward on mixed spec rows.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::layers::{ConcatLayer, Layer, LinearLayer, MulBinaryLayer, ReLULayer, ReduceSumLayer};
use crate::network::{GraphNetwork, GraphNode};

/// Central-difference half step (see `point_vjp_ml4acopf_tests.rs`).
const FD_STEP: f32 = 4e-3;

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

struct LyapunovFixture {
    graph: GraphNetwork,
    w_d: Array2<f32>,
    b_d: Vec<f32>,
    p: Array2<f32>,
}

/// `[V, R]` with `V = sum(d ⊙ (P d))`, `d = W_d x + b_d`, and
/// `R = sum(ReLU(W_h x + b_h))`, over a 3-D input.
fn lyapunov_fixture() -> LyapunovFixture {
    let mut rng = TestRng::new(0x4C59_4150);
    let w_d = rng.matrix(3, 3, 1.0);
    let b_d = rng.vector(3, -0.2, 0.2);
    let p = rng.matrix(3, 3, 0.9);
    let w_h = rng.matrix(4, 3, 1.0);
    let b_h = rng.vector(4, -0.3, 0.3);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "d",
        linear_layer(w_d.clone(), Some(b_d.clone())),
    ));
    graph.add_node(node("pd", linear_layer(p.clone(), None), &["d"]));
    graph.add_node(node("dpd", Layer::MulBinary(MulBinaryLayer), &["d", "pd"]));
    graph.add_node(node(
        "v",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
        &["dpd"],
    ));
    graph.add_node(GraphNode::from_input("h", linear_layer(w_h, Some(b_h))));
    graph.add_node(node("hr", Layer::ReLU(ReLULayer), &["h"]));
    graph.add_node(node(
        "r",
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
        &["hr"],
    ));
    graph.add_node(node("out", Layer::Concat(ConcatLayer::new(0)), &["v", "r"]));
    graph.set_output("out");
    LyapunovFixture { graph, w_d, b_d, p }
}

fn widened_gradient(graph: &GraphNetwork, spec: &[f32], x: &[f32]) -> Vec<f32> {
    let row = Array2::from_shape_vec((1, spec.len()), spec.to_vec()).expect("spec row");
    let grad = graph
        .attack_point_gradient_with_exact_reductions(&as_input(x), &row, None, None)
        .expect("the widened point-VJP must not error on an admitted net")
        .expect("an admitted net must yield Some(gradient)");
    assert_eq!(
        grad.shape(),
        &[x.len()],
        "gradient must have the input shape"
    );
    let grad: Vec<f32> = grad.iter().copied().collect();
    assert!(
        grad.iter().all(|g| g.is_finite()),
        "gradient must be finite: {grad:?}"
    );
    grad
}

fn objective(graph: &GraphNetwork, spec: &[f32], x: &[f32]) -> f64 {
    let input = BoundedTensor::concrete(as_input(x)).expect("concrete input");
    let out = graph
        .propagate_concrete_point(&input, None, None)
        .expect("concrete point forward");
    let out: Vec<f32> = out.lower().iter().copied().collect();
    assert_eq!(out.len(), spec.len(), "spec row must cover every output");
    out.iter()
        .zip(spec)
        .map(|(&y, &s)| f64::from(y) * f64::from(s))
        .sum()
}

/// (1) The standard fragment is byte-identical: it refuses the net, names the
/// FIRST `ReduceSum`, and the standard walk declines with `Ok(None)`.
#[test]
fn point_vjp_standard_fragment_still_refuses_reduce_sum() {
    let fixture = lyapunov_fixture();
    let graph = &fixture.graph;
    assert!(
        !graph.supports_attack_point_gradient(),
        "the standard fragment must keep refusing ReduceSum"
    );
    assert_eq!(
        graph.first_attack_point_gradient_refusal(),
        Some(("v".to_string(), "ReduceSum")),
        "the standard diagnostic must name the first ReduceSum"
    );
    let row = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).expect("spec row");
    let standard = graph
        .attack_point_gradient(&as_input(&[0.1, -0.2, 0.3]), &row, None, None)
        .expect("the standard gate declines, it does not error");
    assert!(
        standard.is_none(),
        "the standard walk must still return Ok(None) on a ReduceSum net"
    );
}

/// (2) + (3) The widened fragment admits the net and its walk is the exact
/// point-Jacobian of the Lyapunov quadratic and the ReLU head.
#[test]
fn point_vjp_exact_reductions_fragment_matches_the_lyapunov_closed_form() {
    let fixture = lyapunov_fixture();
    let graph = &fixture.graph;
    assert!(
        graph.supports_attack_point_gradient_with_exact_reductions(),
        "the widened fragment must admit the Lyapunov net (first refusal: {:?})",
        graph.first_attack_point_gradient_refusal_with_exact_reductions()
    );
    assert_eq!(
        graph.first_attack_point_gradient_refusal_with_exact_reductions(),
        None
    );

    let mut rng = TestRng::new(0x5245_4455);
    let mut compared = 0usize;
    let mut attempted = 0usize;
    for _ in 0..6 {
        let x = rng.vector(3, -1.0, 1.0);

        // Closed form of dV/dx = W_dᵀ (P + Pᵀ) d, in f64 (no ReLU on this path).
        let d: Vec<f64> = (0..3)
            .map(|j| {
                (0..3)
                    .map(|i| f64::from(fixture.w_d[[j, i]]) * f64::from(x[i]))
                    .sum::<f64>()
                    + f64::from(fixture.b_d[j])
            })
            .collect();
        let sym_d: Vec<f64> = (0..3)
            .map(|j| {
                (0..3)
                    .map(|k| (f64::from(fixture.p[[j, k]]) + f64::from(fixture.p[[k, j]])) * d[k])
                    .sum::<f64>()
            })
            .collect();
        let reference: Vec<f64> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| f64::from(fixture.w_d[[j, i]]) * sym_d[j])
                    .sum::<f64>()
            })
            .collect();
        let grad = widened_gradient(graph, &[1.0, 0.0], &x);
        for (i, (&g, &r)) in grad.iter().zip(&reference).enumerate() {
            let tol = 1e-4 * (1.0 + r.abs());
            assert!(
                (f64::from(g) - r).abs() <= tol,
                "coord {i} at x = {x:?}: widened point-VJP {g} vs closed form {r} (tol {tol})"
            );
        }

        // Mixed spec row through BOTH reductions, against a central difference
        // of the graph's own concrete forward. A central difference is EXACT on
        // the quadratic and on a linear ReLU piece, so two step sizes agree
        // unless a step crossed a ReLU kink; that disagreement is the gate
        // (a one-sided-difference gate would also reject every curved coord).
        let spec = rng.vector(2, -1.0, 1.0);
        let grad = widened_gradient(graph, &spec, &x);
        let central_at = |h: f32, i: usize| -> f64 {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += h;
            xm[i] -= h;
            let taken = f64::from(xp[i]) - f64::from(xm[i]);
            (objective(graph, &spec, &xp) - objective(graph, &spec, &xm)) / taken
        };
        for i in 0..3 {
            attempted += 1;
            let coarse = central_at(FD_STEP, i);
            let fine = central_at(0.5 * FD_STEP, i);
            if (coarse - fine).abs() > 1e-3 * (1.0 + fine.abs()) {
                continue; // a step crossed a ReLU kink; the gradient is one-sided
            }
            let g = f64::from(grad[i]);
            let tol = 1e-3 * (1.0 + g.abs().max(fine.abs()));
            assert!(
                (g - fine).abs() <= tol,
                "coord {i} at x = {x:?}: widened point-VJP {g} vs central difference \
                 {fine} (tol {tol})"
            );
            compared += 1;
        }
    }
    assert!(
        compared * 2 >= attempted,
        "the kink gate must leave most coordinates comparable ({compared}/{attempted})"
    );
}
