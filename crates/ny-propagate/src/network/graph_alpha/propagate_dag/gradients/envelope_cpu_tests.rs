// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #alpha-envelope-grad-cpu — end-to-end delivery and SOUNDNESS of the typed
//! `alpha_envelope_grad` key on the CPU chain-rule gradient.
//!
//! Before this fix the CPU chain gradient resolved the envelope rule env-only,
//! so `bab.alpha_crown.alpha_envelope_grad: true` in a yaml was DARK on every
//! lane that runs it (malbeware, cora and vit all print `[chain-alpha-grad]
//! armed` and then `[alpha-ascent] ... frac=1.0000`). These tests drive the
//! PUBLIC α-CROWN entry point with nothing but `AlphaCrownConfig` — the exact
//! field the preset writes — and hold the env layer ABSENT, so the only route
//! by which the rule can arm is the typed value.
//!
//! Two lanes, because the fix touches two call sites:
//! * the DAG lane (non-sequential graph) — `compute_dag_gradients`'
//!   AnalyticChain replay;
//! * the sequential lane — `sequential_gradients::analytic_chain_gradients`.
//!
//! Evidence of engagement is the chain gradient's own per-call tally (the same
//! code that prints the unconditional `[alpha-envelope-grad]` line), so a run
//! in which the replay never happened fails loudly instead of passing VOID.
//!
//! SOUNDNESS. The envelope rule only changes WHICH α the ascent walks to. Every
//! α in `[0,1]` selects a valid lower relaxation, α is clamped on every write,
//! and every bound the ascent keeps is a CROWN pass evaluated at such an α. The
//! containment checks below are the empirical side of that argument: the
//! published enclosure must contain the true network output at every sampled
//! input, in BOTH arms, on both lanes.

use super::{chain_replay_test_serial, reset_chain_replay_latch};
use crate::bounds::{AlphaCrownConfig, GradientMethod};
use crate::layers::{AddLayer, Layer, LinearLayer, ReLULayer};
use crate::network::core::{GraphNetwork, GraphNode};
use crate::network::graph_alpha::backward::gradients::{
    take_cpu_envelope_tally, CpuEnvelopeTally, EnvelopeEngagement,
};
use ndarray::{arr1, arr2, Array1, Array2};
use ny_levers::decls::root_alpha::ALPHA_ENVELOPE_GRAD;
use ny_tensor::BoundedTensor;

/// Sampling slack for containment, matching the sibling DAG soundness tests
/// (`src/tests/graph/alpha_crown_dag.rs`): f32 bound arithmetic against an f64
/// reference forward.
const CONTAINMENT_TOL: f64 = 1e-5;

// ---------------------------------------------------------------------------
// DAG fixture: a residual block, so `is_sequential_graph` is false and the
// public entry point takes the DAG α-CROWN lane.
//
//   x[3] -> lin0 -> relu0 -> lin1 -> relu1 --\
//                      \                      add -> relu2 -> lin_out[2]
//                       \-> lin_skip --------/
// ---------------------------------------------------------------------------

fn w0() -> Array2<f64> {
    arr2(&[
        [0.6, -0.3, 0.4],
        [-0.2, 0.5, 0.3],
        [0.1, -0.4, 0.7],
        [0.3, 0.2, -0.5],
    ])
}
fn b0() -> Array1<f64> {
    arr1(&[0.1, -0.05, 0.05, -0.1])
}
fn w1() -> Array2<f64> {
    arr2(&[
        [0.4, -0.2, 0.3, -0.1],
        [-0.1, 0.5, -0.2, 0.3],
        [0.2, -0.3, 0.6, -0.2],
        [-0.3, 0.1, -0.1, 0.4],
    ])
}
fn w_skip() -> Array2<f64> {
    arr2(&[
        [0.9, 0.05, -0.05, 0.0],
        [0.0, 0.85, 0.1, -0.05],
        [-0.05, 0.0, 0.9, 0.05],
        [0.05, -0.05, 0.0, 0.85],
    ])
}
fn w_out() -> Array2<f64> {
    arr2(&[[0.4, -0.3, 0.2, 0.1], [-0.2, 0.5, -0.1, 0.3]])
}

#[allow(clippy::cast_possible_truncation)]
fn f32_matrix(m: &Array2<f64>) -> Array2<f32> {
    m.mapv(|v| v as f32)
}
#[allow(clippy::cast_possible_truncation)]
fn f32_vector(v: &Array1<f64>) -> Array1<f32> {
    v.mapv(|x| x as f32)
}

fn linear(w: &Array2<f64>, b: Option<&Array1<f64>>) -> Layer {
    Layer::Linear(LinearLayer::new(f32_matrix(w), b.map(f32_vector)).expect("fixture linear layer"))
}

fn residual_dag() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("lin0", linear(&w0(), Some(&b0()))));
    graph.add_node(GraphNode::new(
        "relu0",
        Layer::ReLU(ReLULayer),
        vec!["lin0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin1",
        linear(&w1(), None),
        vec!["relu0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin_skip",
        linear(&w_skip(), None),
        vec!["relu0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["relu1".to_string(), "lin_skip".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin_out",
        linear(&w_out(), None),
        vec!["relu2".to_string()],
    ));
    graph.set_output("lin_out");
    graph
}

fn relu(v: Array1<f64>) -> Array1<f64> {
    v.mapv(|x| x.max(0.0))
}

/// Independent f64 reference forward of [`residual_dag`].
fn residual_forward(x: &Array1<f64>) -> Array1<f64> {
    let h0 = relu(w0().dot(x) + b0());
    let h1 = relu(w1().dot(&h0));
    let h2 = relu(&h1 + &w_skip().dot(&h0));
    w_out().dot(&h2)
}

// ---------------------------------------------------------------------------
// Sequential fixture: a plain MLP, so the public entry point takes the
// sequential α-CROWN lane (`compute_sequential_gradients`).
// ---------------------------------------------------------------------------

fn s1() -> Array2<f64> {
    arr2(&[
        [0.7, -0.5, 0.4],
        [-0.6, 0.9, 0.3],
        [0.5, 0.8, -0.7],
        [-0.4, -0.6, 0.8],
    ])
}
fn sb1() -> Array1<f64> {
    arr1(&[0.05, -0.10, 0.15, -0.05])
}
fn s2() -> Array2<f64> {
    arr2(&[
        [0.6, -0.7, 0.5, 0.3],
        [-0.5, 0.4, -0.6, 0.7],
        [0.8, 0.3, -0.4, -0.5],
        [0.3, -0.8, 0.6, 0.4],
    ])
}
fn sb2() -> Array1<f64> {
    arr1(&[-0.05, 0.10, -0.15, 0.20])
}
fn s3() -> Array2<f64> {
    arr2(&[[0.9, -0.6, 0.7, -0.8], [-0.3, 0.5, -0.4, 0.6]])
}
fn sb3() -> Array1<f64> {
    arr1(&[0.1, -0.2])
}

fn sequential_mlp() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("lin1", linear(&s1(), Some(&sb1()))));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        linear(&s2(), Some(&sb2())),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["lin2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin3",
        linear(&s3(), Some(&sb3())),
        vec!["relu2".to_string()],
    ));
    graph.set_output("lin3");
    graph
}

/// Independent f64 reference forward of [`sequential_mlp`].
fn sequential_forward(x: &Array1<f64>) -> Array1<f64> {
    let h1 = relu(s1().dot(x) + sb1());
    let h2 = relu(s2().dot(&h1) + sb2());
    s3().dot(&h2) + sb3()
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn input_box(half_width: f32) -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-half_width; 3]).into_dyn(),
        arr1(&[half_width; 3]).into_dyn(),
    )
    .expect("fixture input box")
}

/// Deterministic sample of the box: the full 5x5x5 grid (corners and faces
/// included) plus 200 interior points from a fixed LCG, so the check is
/// reproducible and needs no RNG dependency.
#[allow(clippy::cast_precision_loss)]
fn sample_points(half_width: f64) -> Vec<Array1<f64>> {
    let mut points = Vec::new();
    for i in 0..=4 {
        for j in 0..=4 {
            for k in 0..=4 {
                let at = |t: i32| half_width * (f64::from(t) / 2.0 - 1.0);
                points.push(arr1(&[at(i), at(j), at(k)]));
            }
        }
    }
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut unit = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..200 {
        let x: Vec<f64> = (0..3)
            .map(|_| half_width * 2.0f64.mul_add(unit(), -1.0))
            .collect();
        points.push(Array1::from_vec(x));
    }
    points
}

/// The published enclosure must contain the reference output at every sample.
fn assert_contains_sampled_outputs(
    label: &str,
    bounds: &BoundedTensor,
    half_width: f64,
    forward: impl Fn(&Array1<f64>) -> Array1<f64>,
) {
    let lower = bounds.lower();
    let upper = bounds.upper();
    let lo = lower.as_slice().expect("contiguous lower bounds");
    let hi = upper.as_slice().expect("contiguous upper bounds");
    for x in sample_points(half_width) {
        let y = forward(&x);
        assert_eq!(y.len(), lo.len(), "{label}: output width");
        for (d, &value) in y.iter().enumerate() {
            let (l, u) = (f64::from(lo[d]), f64::from(hi[d]));
            assert!(
                l.is_finite() && u.is_finite(),
                "{label}: output[{d}] enclosure must be finite, got [{l}, {u}]"
            );
            assert!(
                value >= l - CONTAINMENT_TOL,
                "{label}: SOUNDNESS VIOLATION output[{d}]={value} < certified lower {l} at x={x}"
            );
            assert!(
                value <= u + CONTAINMENT_TOL,
                "{label}: SOUNDNESS VIOLATION output[{d}]={value} > certified upper {u} at x={x}"
            );
        }
    }
}

fn ascent_config(armed: bool) -> AlphaCrownConfig {
    let config = AlphaCrownConfig {
        iterations: 20,
        alpha_envelope_grad: armed,
        spsa_samples: 1,
        sparse_ratio: 1.0,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        ..AlphaCrownConfig::default()
    };
    assert!(
        matches!(config.gradient_method, GradientMethod::AnalyticChain),
        "the shipped default gradient method must be AnalyticChain, or these tests \
         no longer exercise the CPU chain gradient"
    );
    assert!(
        config.deadline.is_none(),
        "no deadline: the chain replay takes its unbudgeted historical route"
    );
    config
}

/// The per-arm engagement contract read from the chain gradient's own tally.
fn assert_engagement(label: &str, armed: bool, tally: &CpuEnvelopeTally) {
    assert!(
        tally.calls >= 1,
        "{label}: the CPU chain gradient never ran, so this arm is VOID, not a pass \
         (did the chain replay fall back or get skipped?), got {tally:?}"
    );
    let (choice, _) = tally.last.expect("a call records its choice");
    assert_eq!(
        choice.source,
        Some(ny_levers::Source::Config),
        "{label}: with the env layer absent the typed config value must decide"
    );
    if armed {
        assert_eq!(
            tally.armed, tally.calls,
            "{label}: the typed key must arm EVERY CPU chain-gradient call, got {tally:?}"
        );
        assert!(
            tally.engaged >= 1,
            "{label}: armed but never applied — the envelope binding was unavailable \
             on every call, got {tally:?}"
        );
    } else {
        assert_eq!(
            (tally.armed, tally.engaged),
            (0, 0),
            "{label}: the unarmed arm must run the local rule on every call, got {tally:?}"
        );
        assert_eq!(
            tally.last.map(|(_, engagement)| engagement),
            Some(EnvelopeEngagement::Off),
            "{label}: the unarmed arm must report the local rule"
        );
    }
}

/// DAG lane (residual graph through the public entry point): the typed key
/// reaches `compute_dag_gradients`' AnalyticChain replay, and the certified
/// enclosure contains every sampled true output in both arms.
#[ntest::timeout(60000)]
#[test]
fn dag_alpha_envelope_grad_typed_key_engages_and_bounds_contain_sampled_outputs() {
    ny_test_utils::env::with_serialized_env_vars_removed(&[ALPHA_ENVELOPE_GRAD.name], || {
        // The landing-gate latch is process-global; serialize with the tests
        // that toggle it and start from a clean latch, or a concurrent
        // `abandon_chain_replay` would skip the replay and void this test.
        let _serial = chain_replay_test_serial();
        let graph = residual_dag();
        let half_width = 0.5f32;
        let input = input_box(half_width);
        for armed in [false, true] {
            reset_chain_replay_latch();
            let _ = take_cpu_envelope_tally();
            let bounds = graph
                .propagate_alpha_crown_with_config(&input, &ascent_config(armed))
                .expect("DAG alpha-CROWN on the residual fixture");
            let tally = take_cpu_envelope_tally();
            let label = format!("dag armed={armed}");
            assert_engagement(&label, armed, &tally);
            assert_contains_sampled_outputs(
                &label,
                &bounds,
                f64::from(half_width),
                residual_forward,
            );
        }
    });
}

/// Sequential lane (plain MLP through the public entry point): the typed key
/// reaches `analytic_chain_gradients`, and the certified enclosure contains
/// every sampled true output in both arms. The input box is wide enough that
/// every hidden neuron is unstable, so α is live everywhere.
#[ntest::timeout(60000)]
#[test]
fn sequential_alpha_envelope_grad_typed_key_engages_and_bounds_contain_sampled_outputs() {
    ny_test_utils::env::with_serialized_env_vars_removed(&[ALPHA_ENVELOPE_GRAD.name], || {
        let graph = sequential_mlp();
        let half_width = 1.0f32;
        let input = input_box(half_width);
        for armed in [false, true] {
            let _ = take_cpu_envelope_tally();
            let bounds = graph
                .propagate_alpha_crown_with_config(&input, &ascent_config(armed))
                .expect("sequential alpha-CROWN on the MLP fixture");
            let tally = take_cpu_envelope_tally();
            let label = format!("sequential armed={armed}");
            assert_engagement(&label, armed, &tally);
            assert_contains_sampled_outputs(
                &label,
                &bounds,
                f64::from(half_width),
                sequential_forward,
            );
        }
    });
}
