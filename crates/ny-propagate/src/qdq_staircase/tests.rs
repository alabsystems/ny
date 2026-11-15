// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for the dark shared-input staircase tightening (#qdq-crown-followup).
//!
//! Every armed test holds `arm_lock()` — the gate is process-global, so two
//! tests flipping it concurrently would race. The default state (OFF) is
//! itself pinned by `disarmed_by_default_and_keeps_plain_product_bounds`.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ndarray::{arr1, arr2};
use ny_tensor::BoundedTensor;

use super::{qdq_staircase_ibp_armed, set_qdq_staircase_ibp};
use crate::layers::arithmetic::{AddConstantLayer, DivConstantLayer};
use crate::layers::binary_ops::MulBinaryLayer;
use crate::layers::linear::LinearLayer;
use crate::layers::normalization::math_common::square_interval_bounds;
use crate::layers::trigonometric::ErfLayer;
use crate::layers::Layer;
use crate::network::{GraphNetwork, GraphNode};

fn arm_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        // A poisoned lock only means another armed test panicked; the guard's
        // Drop already disarmed the global, so the state is safe to reuse.
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// RAII disarm so a panicking test cannot leave the global armed.
struct Armed(#[allow(dead_code)] MutexGuard<'static, ()>);
impl Armed {
    fn new() -> Self {
        let guard = arm_lock();
        set_qdq_staircase_ibp(true);
        Self(guard)
    }
}
impl Drop for Armed {
    fn drop(&mut self) {
        set_qdq_staircase_ibp(false);
    }
}

/// x -> identity Linear (2 elements), then MulBinary(x, x).
fn square_fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    let identity =
        LinearLayer::new(arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]), None).expect("identity layer");
    graph.add_node(GraphNode::from_input("x", Layer::Linear(identity)));
    graph.add_node(GraphNode::new(
        "sq",
        Layer::MulBinary(MulBinaryLayer),
        vec!["x".to_string(), "x".to_string()],
    ));
    graph.set_output("sq");
    let input = BoundedTensor::new(
        arr1(&[-3.0_f32, 1.0]).into_dyn(),
        arr1(&[2.0_f32, 4.0]).into_dyn(),
    )
    .expect("input box");
    (graph, input)
}

/// x -> Div(c) -> Erf -> Add(1) -> Mul(x, ·): the erf-GELU staircase, with
/// the authored f32 sqrt2 divisor smart_turn carries.
fn gelu_fixture(add_value: f32) -> (GraphNetwork, BoundedTensor) {
    let c = std::f32::consts::SQRT_2;
    let mut graph = GraphNetwork::new();
    let identity =
        LinearLayer::new(arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]), None).expect("identity layer");
    graph.add_node(GraphNode::from_input("x", Layer::Linear(identity)));
    graph.add_node(GraphNode::new(
        "gdiv",
        Layer::DivConstant(DivConstantLayer::new(ndarray::ArrayD::from_elem(
            ndarray::IxDyn(&[]),
            c,
        ))),
        vec!["x".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "gerf",
        Layer::Erf(ErfLayer),
        vec!["gdiv".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "gadd",
        Layer::AddConstant(AddConstantLayer::new(ndarray::ArrayD::from_elem(
            ndarray::IxDyn(&[]),
            add_value,
        ))),
        vec!["gerf".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "gmul",
        Layer::MulBinary(MulBinaryLayer),
        vec!["x".to_string(), "gadd".to_string()],
    ));
    graph.set_output("gmul");
    let input = BoundedTensor::new(
        arr1(&[-4.0_f32, -1.0]).into_dyn(),
        arr1(&[4.0_f32, 0.5]).into_dyn(),
    )
    .expect("input box");
    (graph, input)
}

fn two_node_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    name: &str,
    operand: &str,
) -> ((Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>)) {
    let bounds = graph
        .collect_node_bounds(input)
        .expect("collect_node_bounds");
    let read = |n: &str| {
        let bt = bounds.get(n).expect("node bounds present");
        (
            bt.lower().iter().copied().collect::<Vec<f32>>(),
            bt.upper().iter().copied().collect::<Vec<f32>>(),
        )
    };
    (read(name), read(operand))
}

fn node_bounds(graph: &GraphNetwork, input: &BoundedTensor, name: &str) -> (Vec<f32>, Vec<f32>) {
    two_node_bounds(graph, input, name, name).0
}

/// Default-off pin: disarmed (the shipped state), MulBinary(a, a) keeps the
/// sign-ignorant plain product bound bit-identically — the exact looseness
/// the dark gate exists to repair, and the bit-identical default every other
/// family relies on.
#[ntest::timeout(10000)]
#[test]
fn disarmed_by_default_and_keeps_plain_product_bounds() {
    let _guard = arm_lock();
    assert!(
        !qdq_staircase_ibp_armed(),
        "the staircase tightening must default OFF"
    );
    let (graph, input) = square_fixture();
    let (lower, upper) = node_bounds(&graph, &input, "sq");
    // Plain interval product of [-3, 2] x [-3, 2]: lower = min cross-product
    // (-3 * 2 = -6), upper = max (9). The tightened square image would be
    // [0, 9] — asserting the NEGATIVE lower pins that the dark path stayed
    // cold.
    assert!(
        lower[0] < -5.9,
        "disarmed square must keep the plain product lower, got {}",
        lower[0]
    );
    assert!(upper[0] >= 8.9, "plain product upper, got {}", upper[0]);
}

/// Armed: MulBinary(a, a) publishes the exact square image (lower 0 on a
/// sign-crossing interval), bit-equal to `square_interval_bounds`, and a
/// concrete sample sweep confirms containment of the true x^2.
#[ntest::timeout(10000)]
#[test]
fn armed_same_input_square_uses_the_exact_square_image() {
    let _armed = Armed::new();
    let (graph, input) = square_fixture();
    // The MulBinary node's operand box is the IDENTITY LINEAR node's IBP
    // output — outward-rounded, so a hair wider than the raw input box. The
    // dense reference must square THAT box, exactly as the dispatch does.
    let ((lower, upper), (x_lower, x_upper)) = two_node_bounds(&graph, &input, "sq", "x");

    for (i, (&l, &u)) in x_lower.iter().zip(x_upper.iter()).enumerate() {
        let (want_l, want_u) = square_interval_bounds(l, u);
        assert_eq!(
            lower[i].to_bits(),
            want_l.to_bits(),
            "square lower[{i}]: got {}, want {want_l}",
            lower[i]
        );
        assert_eq!(
            upper[i].to_bits(),
            want_u.to_bits(),
            "square upper[{i}]: got {}, want {want_u}",
            upper[i]
        );
        // Containment of the true image at sampled points (a plant that
        // forgets the sign-crossing case inverts the interval, the
        // intersection falls back to the plain NEGATIVE lower, and the
        // bit-equality above goes red; this sweep additionally proves the
        // published box CONTAINS x^2 everywhere).
        for step in 0..=20 {
            let x = l + (u - l) * (step as f32 / 20.0);
            let sq = x * x;
            assert!(
                lower[i] <= sq && sq <= upper[i],
                "x^2 = {sq} at x = {x} escapes [{}, {}]",
                lower[i],
                upper[i]
            );
        }
    }
    // The tightening actually moved the sign-crossing element: lower is
    // exactly 0, not the plain product's -6.
    assert_eq!(lower[0].to_bits(), 0.0_f32.to_bits());
}

/// Armed: the erf-GELU product publishes the exact 1-D image
/// `c·√2·GELU_erf(x·√2/c)` intersected with the plain product — the lower
/// bound rises from the factor-wise -8 to the function's global minimum
/// (≈ -0.34 for this scale), and a dense concrete sweep in f64 confirms
/// containment of the true `x·(1 + erf(x/c))`.
#[ntest::timeout(10000)]
#[test]
fn armed_erf_gelu_product_uses_the_exact_one_d_image() {
    let _armed = Armed::new();
    let (graph, input) = gelu_fixture(1.0);
    let (lower, upper) = node_bounds(&graph, &input, "gmul");

    let c = f64::from(std::f32::consts::SQRT_2);
    for (i, (&l, &u)) in input.lower().iter().zip(input.upper().iter()).enumerate() {
        // True-image containment sweep (the discriminating half: a planted
        // wrong out-scale caps the upper below the true image at x = u and
        // this sweep goes red).
        for step in 0..=400 {
            let x = l as f64 + (u as f64 - l as f64) * (step as f64 / 400.0);
            let h = x * (1.0 + libm::erf(x / c));
            assert!(
                (lower[i] as f64) <= h + 1e-6 && h - 1e-6 <= (upper[i] as f64),
                "x(1+erf(x/c)) = {h} at x = {x} escapes [{}, {}] (elem {i})",
                lower[i],
                upper[i]
            );
        }
    }
    // The tightening engaged: element 0 spans [-4, 4], so the plain product
    // lower is -8 while the true function minimum is 2·min GELU ≈ -0.34
    // (scaled); the published lower must reflect the 1-D image, not the
    // factor-wise product.
    assert!(
        lower[0] > -0.5,
        "erf-gelu lower must rise to the 1-D image minimum, got {}",
        lower[0]
    );
    assert!(
        lower[0] < -0.3,
        "erf-gelu lower must still contain the global minimum ≈ -0.34, got {}",
        lower[0]
    );
    assert!(
        upper[0] >= 7.9,
        "erf-gelu upper must keep the x = 4 endpoint image ≈ 7.99, got {}",
        upper[0]
    );
}

/// The motif match is exact: an Add constant that is not 1.0 keeps the plain
/// product bound (the identity `x·(a + erf(x/c)) = c·√2·GELU(...)` only
/// holds at a = 1, so tightening there would be unsound).
#[ntest::timeout(10000)]
#[test]
fn armed_gelu_motif_refuses_wrong_add_constant() {
    let _armed = Armed::new();
    let (graph, input) = gelu_fixture(0.9);
    let (lower, _upper) = node_bounds(&graph, &input, "gmul");
    // Plain product of [-4, 4] x [-0.1..., 1.9]: lower well below the GELU
    // image minimum. If the motif fired despite a != 1, this would read
    // ≈ -0.34 instead.
    assert!(
        lower[0] < -7.0,
        "a != 1.0 must keep the plain product bound, got {}",
        lower[0]
    );
}
