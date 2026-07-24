// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Gradient-check tests for [`GraphNetwork::attack_point_gradient`].
//!
//! Two independent oracles on a small ResNet-style whitelist graph
//! (Conv2d -> ReLU -> Conv2d -> Add(residual) -> Flatten -> Linear):
//!
//! 1. **Exact-CROWN oracle** — the SAME extraction `ny-cli graph_pgd_exact.rs`
//!    uses: `collect_node_bounds` ->
//!    `propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline` ->
//!    `(lower_a + upper_a) * 0.5` row 0. At a concrete point this is the exact
//!    point-Jacobian, so `attack_point_gradient` must match it to 1e-3 relative.
//!
//! 2. **Finite differences** — central difference of `spec_row · output` w.r.t.
//!    each input coordinate, with a forward/backward-agreement gate that skips
//!    coordinates whose perturbation crosses a ReLU kink. Matches to ~1e-2.

use crate::layers::{AddLayer, Conv2dLayer, FlattenLayer, Layer, LinearLayer, ReLULayer};
use crate::network::{GraphNetwork, GraphNode};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

/// Deterministic SplitMix64-style generator (no rand dependency in the fixture),
/// mirroring `forward_linear/tests_image.rs`.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-scale, scale].
    fn next_f32(&mut self, scale: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        (unit * 2.0 - 1.0) * scale
    }
}

fn random_kernel(
    rng: &mut Lcg,
    out_c: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    scale: f32,
) -> ArrayD<f32> {
    let v: Vec<f32> = (0..out_c * in_c * kh * kw)
        .map(|_| rng.next_f32(scale))
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, kh, kw]), v).expect("kernel shape")
}

fn random_bias(rng: &mut Lcg, n: usize, scale: f32) -> Array1<f32> {
    Array1::from_iter((0..n).map(|_| rng.next_f32(scale)))
}

/// Build a small ResNet-style whitelist graph plus a concrete input point and a
/// spec row. Input `(2, 4, 4)`; the residual `Add(conv2, relu1)` makes `relu1`
/// feed BOTH `conv2` and the skip `Add` — exercising cotangent fan-in.
fn build_residual_conv_graph() -> (GraphNetwork, ArrayD<f32>, Array2<f32>) {
    let mut rng = Lcg::new(0xA11CE);
    let mut graph = GraphNetwork::new();

    // conv1: (3, 2, 3, 3), stride 1, pad 1, input 4x4 -> (3, 4, 4)
    let conv1 = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 3, 2, 3, 3, 0.5),
        Some(random_bias(&mut rng, 3, 0.2)),
        (1, 1),
        (1, 1),
        4,
        4,
    )
    .expect("conv1");
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));

    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".to_string()],
    ));

    // conv2: (3, 3, 3, 3), stride 1, pad 1, input 4x4 -> (3, 4, 4)
    let conv2 = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 3, 3, 3, 3, 0.5),
        Some(random_bias(&mut rng, 3, 0.2)),
        (1, 1),
        (1, 1),
        4,
        4,
    )
    .expect("conv2");
    graph.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(conv2),
        vec!["relu1".to_string()],
    ));

    // Residual skip: relu1 is consumed by BOTH conv2 and this Add.
    graph.add_node(GraphNode::new(
        "add1",
        Layer::Add(AddLayer),
        vec!["conv2".to_string(), "relu1".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "flatten",
        Layer::Flatten(FlattenLayer::new(0)),
        vec!["add1".to_string()],
    ));

    // Linear: (4, 48) since flattened (3,4,4) = 48 -> 4 outputs.
    let mut w = Array2::<f32>::zeros((4, 48));
    for v in w.iter_mut() {
        *v = rng.next_f32(0.4);
    }
    let linear = LinearLayer::new(w, Some(random_bias(&mut rng, 4, 0.2))).expect("linear");
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(linear),
        vec!["flatten".to_string()],
    ));
    graph.set_output("out");

    // Concrete input point (a single point, not a box) with values around O(0.6)
    // so ReLU pre-activations are typically well away from 0.
    let x_vec: Vec<f32> = (0..2 * 4 * 4).map(|_| rng.next_f32(0.6)).collect();
    let x = ArrayD::from_shape_vec(IxDyn(&[2, 4, 4]), x_vec).expect("input shape");

    // spec row (1, 4): a linear combination of the 4 outputs (an attack margin).
    let spec_row = Array2::from_shape_vec(
        (1, 4),
        (0..4).map(|_| rng.next_f32(1.0)).collect::<Vec<_>>(),
    )
    .expect("spec row");

    (graph, x, spec_row)
}

/// Evaluate the concrete network output at `x` (midpoint of the degenerate box).
fn eval_output(graph: &GraphNetwork, x: &ArrayD<f32>) -> ArrayD<f32> {
    let ib = BoundedTensor::concrete(x.clone()).expect("concrete input");
    let nb = graph
        .collect_node_bounds_with_engine_and_deadline(&ib, None, None)
        .expect("forward node bounds");
    let out = nb.get(graph.output_name()).expect("output node bounds");
    (out.lower() + out.upper()) * 0.5
}

/// `spec_row · output` (both flattened).
fn spec_dot(spec_row: &Array2<f32>, out: &ArrayD<f32>) -> f32 {
    spec_row
        .row(0)
        .iter()
        .zip(out.iter())
        .map(|(&a, &b)| a * b)
        .sum()
}

#[test]
fn attack_point_gradient_matches_exact_crown_oracle() {
    let (graph, x, spec_row) = build_residual_conv_graph();

    // Under test.
    let grad = graph
        .attack_point_gradient(&x, &spec_row, None, None)
        .expect("attack_point_gradient must not error")
        .expect("graph is in the supported fragment -> Some");
    assert_eq!(grad.shape(), x.shape(), "gradient must have input shape");
    let grad_flat: Vec<f32> = grad.iter().copied().collect();

    // Reference oracle: EXACTLY what ny-cli graph_pgd_exact.rs computes.
    let ib = BoundedTensor::concrete(x).expect("concrete input");
    let node_bounds = graph
        .collect_node_bounds_with_engine_and_deadline(&ib, None, None)
        .expect("node bounds");
    let (_spec_bounds, linear) = graph
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            &ib,
            &spec_row,
            None,
            &node_bounds,
            None,
        )
        .expect("spec CROWN with linear extraction");
    let linear = linear.expect("whitelist graph -> linear extraction is Some");
    let ref_row = (linear.lower_a() + linear.upper_a()) * 0.5;
    let ref_flat: Vec<f32> = ref_row.row(0).to_vec();

    assert_eq!(ref_flat.len(), grad_flat.len());
    for (i, (&g, &r)) in grad_flat.iter().zip(ref_flat.iter()).enumerate() {
        let tol = 1e-3 * (1.0 + r.abs());
        assert!(
            (g - r).abs() <= tol,
            "grad[{i}] = {g} disagrees with exact-CROWN oracle {r} (tol {tol})",
        );
    }
}

#[test]
fn attack_point_gradient_matches_finite_differences() {
    let (graph, x, spec_row) = build_residual_conv_graph();

    let grad = graph
        .attack_point_gradient(&x, &spec_row, None, None)
        .expect("attack_point_gradient must not error")
        .expect("supported fragment -> Some");
    let grad_flat: Vec<f32> = grad.iter().copied().collect();

    let h = 1e-3f32;
    let s0 = spec_dot(&spec_row, &eval_output(&graph, &x));

    let mut checked = 0usize;
    for i in 0..x.len() {
        let mut xp: Vec<f32> = x.iter().copied().collect();
        let mut xm = xp.clone();
        xp[i] += h;
        xm[i] -= h;
        let xp = ArrayD::from_shape_vec(IxDyn(x.shape()), xp).unwrap();
        let xm = ArrayD::from_shape_vec(IxDyn(x.shape()), xm).unwrap();

        let sp = spec_dot(&spec_row, &eval_output(&graph, &xp));
        let sm = spec_dot(&spec_row, &eval_output(&graph, &xm));

        let forward = (sp - s0) / h;
        let backward = (s0 - sm) / h;
        let central = (sp - sm) / (2.0 * h);

        // Kink gate: the network is piecewise-linear, so within one linear piece
        // forward and backward differences agree to float noise. A large mismatch
        // means the +h/-h perturbation crossed a ReLU boundary -> skip (the exact
        // gradient is the one-sided slope, which finite differences can't see
        // across a kink).
        if (forward - backward).abs() > 0.05 * (1.0 + central.abs()) {
            continue;
        }

        let g = grad_flat[i];
        let tol = 1e-2 * (1.0 + g.abs().max(central.abs())) + 1e-3;
        assert!(
            (central - g).abs() <= tol,
            "coord {i}: FD central {central} vs grad {g} (tol {tol})",
        );
        checked += 1;
    }

    assert!(
        checked >= x.len() / 2,
        "finite-difference check exercised too few coordinates ({checked}/{}); \
         most should be on a smooth linear piece",
        x.len()
    );
}

#[test]
fn attack_point_gradient_rejects_unsupported_fragment() {
    // A graph containing a non-whitelist layer (Sigmoid) must return Ok(None).
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "sig",
        Layer::Sigmoid(crate::layers::SigmoidLayer),
    ));
    graph.set_output("sig");

    let x = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1f32, -0.2, 0.3]).unwrap();
    let spec_row = Array2::from_shape_vec((1, 3), vec![1.0f32, 0.0, 0.0]).unwrap();
    let out = graph
        .attack_point_gradient(&x, &spec_row, None, None)
        .expect("must not error");
    assert!(out.is_none(), "non-whitelist graph must yield Ok(None)");
}

/// cora_2024 MLP fragment: DivConstant (mnist /255-style normalization) and
/// AddConstant (unfused Gemm bias) are affine and must produce the EXACT
/// point-Jacobian through the VJP. Analytic oracle, hand-composed:
/// y = W2 @ relu(W1 @ (x / c + b) + b1), spec_row = [1]
/// dy/dx = (spec · W2) · diag(relu_mask) · W1 · (1/c)
#[test]
fn attack_point_gradient_exact_through_div_and_add_constant() {
    use crate::layers::{AddConstantLayer, DivConstantLayer};

    let mut graph = GraphNetwork::new();
    // x / 2.0
    graph.add_node(GraphNode::from_input(
        "div",
        Layer::DivConstant(DivConstantLayer::scalar(2.0)),
    ));
    // + [0.1, 0.2]
    let add_c = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1f32, 0.2]).unwrap();
    graph.add_node(GraphNode::new(
        "addc",
        Layer::AddConstant(AddConstantLayer::new(add_c)),
        vec!["div".to_string()],
    ));
    // W1 = [[1, 2], [3, -1]], b1 = [0.05, -0.05]
    let w1 = Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, -1.0]).unwrap();
    let b1 = Array1::from_vec(vec![0.05f32, -0.05]);
    graph.add_node(GraphNode::new(
        "lin1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
        vec!["addc".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["lin1".to_string()],
    ));
    // W2 = [[1, -2]]
    let w2 = Array2::from_shape_vec((1, 2), vec![1.0f32, -2.0]).unwrap();
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(LinearLayer::new(w2, None).unwrap()),
        vec!["relu".to_string()],
    ));
    graph.set_output("lin2");

    // x = [0.5, -1.0]:
    //   x/2 = [0.25, -0.5]; +[0.1, 0.2] = [0.35, -0.3]
    //   z = W1 @ [0.35, -0.3] + b1 = [0.35 - 0.6 + 0.05, 1.05 + 0.3 - 0.05] = [-0.2, 1.3]
    //   relu mask = [0, 1]
    //   dy/dx = [1, -2] @ diag([0,1]) @ W1 @ diag(0.5) = -2 * [3, -1] * 0.5 = [-3, 1]
    let x = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5f32, -1.0]).unwrap();
    let spec_row = Array2::from_shape_vec((1, 1), vec![1.0f32]).unwrap();

    let grad = graph
        .attack_point_gradient(&x, &spec_row, None, None)
        .expect("attack_point_gradient must not error")
        .expect("DivConstant/AddConstant fragment must be supported");
    let g: Vec<f32> = grad.iter().copied().collect();
    assert_eq!(g.len(), 2);
    assert!(
        (g[0] - (-3.0)).abs() < 1e-4 && (g[1] - 1.0).abs() < 1e-4,
        "expected exact gradient [-3, 1], got {g:?}"
    );
}
