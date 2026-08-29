// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Normalization-epsilon gate reconciliation: one rule per layer FAMILY,
//! enforced at the translate boundary.
//!
//! Three independent gates used to disagree about which epsilons a trace may
//! carry:
//!
//! | gate | rule |
//! | --- | --- |
//! | nn (`nn-core/src/layers` `validate_eps`) | finite, `eps >= 0` |
//! | ny-trace-bridge (`translate::validate_eps`, before) | finite, `eps > 0` |
//! | ny-propagate (`validate_norm_eps`) and ny-build (`normalization_fusion`) | finite, `eps >= NORMALIZATION_MIN_EPS` |
//!
//! nn's rule is authoritative for what a real trace CAN carry, so `eps == 0`
//! and `0 < eps < 1e-12` both reach the bridge. Under the old bridge rule the
//! second window translated "successfully" and then blew up inside
//! `GraphModel::build_graph_network` — a fail-closed refusal, but at the wrong
//! boundary and with a message naming a converted layer instead of the traced
//! op. These tests pin the reconciliation:
//!
//! 1. **Data-variance family** (LayerNorm, RmsNorm, InstanceNorm, GroupNorm,
//!    AdaLayerNorm, AdaIN, FusedAdainResBlock) — variance comes from the
//!    input, so a box containing a constant vector has `var == 0` and the
//!    denominator `sqrt(var + eps)` is `sqrt(eps)`. `eps == 0` is a genuine
//!    division by zero; below `NORMALIZATION_MIN_EPS` the layers' bound
//!    derivations are not validated and ny-propagate refuses. The bridge now
//!    refuses the same set, at translate time.
//! 2. **BatchNorm** — variance is a captured constant, so `eps == 0` is sound
//!    exactly when every channel has `running_var > 0`. That is the WEAKEST
//!    SOUND gate and it is what `BatchNormLayer::new` enforces; the bridge now
//!    mirrors it verbatim instead of blanket-refusing `eps == 0`.
//! 3. **No gap either way** — everything the bridge accepts must survive
//!    `build_graph_network` (no deferred refusal), and everything it refuses
//!    must be refused by `translate` itself (no deep refusal).

use ny_core::NyError;
use ny_propagate::layers::NORMALIZATION_MIN_EPS;
use ny_trace_bridge::schema::{
    ComputationGraph, DType, KokoroFusedOp, NodeId, TraceNode, TraceOp, WeightPayload,
};
use ny_trace_bridge::translate::translate;

fn node(id: u64, name: &str, op: TraceOp, inputs: &[u64], shape: &[usize]) -> TraceNode {
    TraceNode::new(
        NodeId(id),
        name,
        op,
        inputs.iter().map(|&i| NodeId(i)).collect(),
        shape.to_vec(),
        DType::F32,
    )
}

fn ones(len: usize) -> WeightPayload {
    WeightPayload::f32(vec![1.0; len], vec![len])
}

fn zeros(len: usize) -> WeightPayload {
    WeightPayload::f32(vec![0.0; len], vec![len])
}

/// Every DATA-VARIANCE normalization the bridge lowers, parameterized by eps.
///
/// Each entry is `(label, graph builder)`. `AdaLayerNorm` /
/// `FusedAdainResBlock` live in the Kokoro family and had their own
/// `eps >= 0` rules; they are included precisely because those rules were the
/// loosest of the set.
fn data_variance_graphs(eps: f64) -> Vec<(&'static str, ComputationGraph)> {
    vec![
        (
            "LayerNorm",
            ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[2, 4]),
                node(
                    1,
                    "ln",
                    TraceOp::LayerNorm {
                        eps,
                        weight: ones(4),
                        bias: zeros(4),
                    },
                    &[0],
                    &[2, 4],
                ),
            ]),
        ),
        (
            "RmsNorm",
            ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[2, 4]),
                node(
                    1,
                    "rn",
                    TraceOp::RmsNorm {
                        eps,
                        weight: ones(4),
                    },
                    &[0],
                    &[2, 4],
                ),
            ]),
        ),
        (
            "InstanceNorm",
            ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[1, 3, 4]),
                node(1, "in", TraceOp::InstanceNorm { eps }, &[0], &[1, 3, 4]),
            ]),
        ),
        (
            "GroupNorm",
            ComputationGraph::from_nodes(vec![
                node(0, "x", TraceOp::Input, &[], &[1, 4, 6]),
                node(
                    1,
                    "gn",
                    TraceOp::GroupNorm {
                        num_groups: 2,
                        eps,
                        weight: ones(4),
                        bias: zeros(4),
                    },
                    &[0],
                    &[1, 4, 6],
                ),
            ]),
        ),
        (
            "AdaLayerNorm",
            kokoro_graph(
                KokoroFusedOp::AdaLayerNorm {
                    norm_weight: ones(4),
                    norm_bias: zeros(4),
                    eps,
                },
                &[1, 2, 4],
            ),
        ),
        (
            "AdainLeakyRelu",
            kokoro_graph(
                KokoroFusedOp::AdainLeakyRelu { eps, slope: 0.1 },
                &[1, 4, 3],
            ),
        ),
        (
            "AdainSnake",
            kokoro_graph(
                KokoroFusedOp::AdainSnake {
                    alpha: WeightPayload::f32(vec![1.0; 4], vec![1, 4, 1]),
                    eps,
                },
                &[1, 4, 3],
            ),
        ),
    ]
}

/// A `[x, gamma, beta]` Kokoro fused-op graph with constant style parameters.
fn kokoro_graph(op: KokoroFusedOp, shape: &[usize]) -> ComputationGraph {
    let channels = shape[1];
    // 1-D per-channel style parameters: what the native AdaIN conversion
    // expects, so the constant-gamma/beta fast path builds.
    let affine = WeightPayload::f32(vec![0.0; channels], vec![channels]);
    let affine_shape = vec![channels];
    ComputationGraph::from_nodes(vec![
        node(0, "x", TraceOp::Input, &[], shape),
        node(
            1,
            "gamma",
            TraceOp::ConstantWeight {
                weight: affine.clone(),
            },
            &[],
            &affine_shape,
        ),
        node(
            2,
            "beta",
            TraceOp::ConstantWeight { weight: affine },
            &[],
            &affine_shape,
        ),
        node(3, "fused", TraceOp::KokoroFused(op), &[0, 1, 2], shape),
    ])
}

fn build_batch_norm_graph(eps: f64, running_var: Vec<f32>) -> ComputationGraph {
    let channels = running_var.len();
    ComputationGraph::from_nodes(vec![
        node(0, "x", TraceOp::Input, &[], &[1, channels, 3]),
        node(
            1,
            "bn",
            TraceOp::BatchNorm {
                eps,
                weight: ones(channels),
                bias: zeros(channels),
                running_mean: zeros(channels),
                running_var: WeightPayload::f32(running_var, vec![channels]),
            },
            &[0],
            &[1, channels, 3],
        ),
    ])
}

/// The bridge's restated floor must equal ny-propagate's, bit for bit. If
/// ny-propagate ever moves `NORMALIZATION_MIN_EPS`, this fails rather than
/// letting the two gates drift apart silently.
#[test]
fn bridge_floor_mirrors_ny_propagate_floor() {
    // The bridge's private `translate::NORMALIZATION_MIN_EPS` is this literal.
    assert_eq!(
        (1e-12_f32).to_bits(),
        NORMALIZATION_MIN_EPS.to_bits(),
        "ny-propagate's normalization floor moved; update the bridge's mirror"
    );
}

/// nn accepts `eps == 0`, so real traces carry it. Every data-variance
/// normalization must refuse it AT TRANSLATE TIME, naming the traced op and
/// the floor — never by silently succeeding and failing later inside graph
/// construction.
#[test]
fn data_variance_norms_refuse_zero_eps_at_translate_time() {
    for (label, graph) in data_variance_graphs(0.0) {
        match translate(&graph) {
            Err(NyError::NumericalInstability(message)) => {
                assert!(
                    message.contains(label) && message.contains("eps"),
                    "{label}: message must name the op and eps, got {message}"
                );
                assert!(
                    message.contains("1e-12") || message.contains("0.000000000001"),
                    "{label}: message must name the floor, got {message}"
                );
            }
            other => panic!("{label}: eps == 0 must fail closed at translate, got {other:?}"),
        }
    }
}

/// The `(0, NORMALIZATION_MIN_EPS)` window is the one the old bridge rule let
/// through. It must now be refused at translate time — including the single
/// f32 ULP below the floor, and f64 values that only underflow into the
/// window after the f32 narrowing.
#[test]
fn data_variance_norms_refuse_below_the_floor() {
    let just_below = f64::from(f32::from_bits(NORMALIZATION_MIN_EPS.to_bits() - 1));
    for eps in [
        just_below,
        1e-13,
        1e-30,
        1e-45,  // f32 subnormal: positive, but far below the floor
        1e-320, // f64 subnormal: narrows to 0.0f32
        f64::MIN_POSITIVE,
    ] {
        for (label, graph) in data_variance_graphs(eps) {
            assert!(
                matches!(translate(&graph), Err(NyError::NumericalInstability(_))),
                "{label}: eps {eps} is below the propagate floor and must be refused here"
            );
        }
    }
}

/// Non-finite and negative epsilons stay refused (NaN must not slip through a
/// bare `<` comparison).
#[test]
fn data_variance_norms_refuse_non_finite_and_negative_eps() {
    for eps in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e-5, -0.0] {
        for (label, graph) in data_variance_graphs(eps) {
            assert!(
                matches!(translate(&graph), Err(NyError::NumericalInstability(_))),
                "{label}: eps {eps} must be refused"
            );
        }
    }
}

/// An eps that overflows f32 is refused too (it would become `inf`).
#[test]
fn data_variance_norms_refuse_eps_overflowing_f32() {
    for (label, graph) in data_variance_graphs(1e300) {
        assert!(
            matches!(translate(&graph), Err(NyError::NumericalInstability(_))),
            "{label}: eps 1e300 overflows f32 and must be refused"
        );
    }
}

/// EXACTLY at the floor the bridge must accept — and what it accepts must
/// then BUILD. This is the no-gap direction: if the bridge's gate were
/// stricter than ny-propagate's, verifiable traces would be lost; if it were
/// looser, this build would fail.
#[test]
fn data_variance_norms_accept_the_floor_and_build() {
    for eps in [f64::from(NORMALIZATION_MIN_EPS), 1e-9, 1e-5, 1e-3] {
        for (label, graph) in data_variance_graphs(eps) {
            let model = translate(&graph)
                .unwrap_or_else(|e| panic!("{label}: eps {eps} must translate, got {e:?}"));
            model
                .build_graph_network(ny_build::GraphNetworkOptions::default())
                .unwrap_or_else(|e| {
                    panic!("{label}: eps {eps} translated but failed to build: {e:?}")
                });
        }
    }
}

/// BatchNorm's variance is a translate-time CONSTANT, so `eps == 0` is not a
/// division by zero as long as every channel already has a positive variance.
/// nn accepts it, `BatchNormLayer::new` accepts it, and the bridge must not be
/// the odd gate out: it translates AND builds.
#[test]
fn batch_norm_accepts_zero_eps_when_every_channel_variance_is_positive() {
    let graph = build_batch_norm_graph(0.0, vec![1.0, 0.25, 4.0]);
    let model = translate(&graph).expect("BatchNorm with eps == 0 and positive var translates");
    model
        .build_graph_network(ny_build::GraphNetworkOptions::default())
        .expect("BatchNorm with eps == 0 and positive var builds");
}

/// ... and the genuinely division-by-zero-able configuration is still
/// refused, at translate time, naming the offending channel.
#[test]
fn batch_norm_refuses_zero_eps_with_a_zero_variance_channel() {
    let graph = build_batch_norm_graph(0.0, vec![1.0, 0.0, 4.0]);
    match translate(&graph) {
        Err(NyError::NumericalInstability(message)) => {
            assert!(
                message.contains("BatchNorm") && message.contains("channel 1"),
                "message must name the offending channel, got {message}"
            );
        }
        other => panic!("var == 0 with eps == 0 must fail closed at translate, got {other:?}"),
    }
}

/// A tiny eps rescues a zero-variance channel — exactly as
/// `BatchNormLayer::new` computes it, so translate and build agree.
#[test]
fn batch_norm_zero_variance_channel_is_rescued_by_a_positive_eps() {
    let graph = build_batch_norm_graph(1e-5, vec![1.0, 0.0, 4.0]);
    let model = translate(&graph).expect("positive eps rescues the zero-variance channel");
    model
        .build_graph_network(ny_build::GraphNetworkOptions::default())
        .expect("builds");
}

/// A negative variance that cancels eps to exactly zero, and one that drives
/// the sum negative, are both refused — the check is on `var + eps`, not on
/// eps alone.
#[test]
fn batch_norm_refuses_a_non_positive_variance_sum() {
    for (variance, eps) in [
        (-1e-5_f32, 1e-5_f64),
        (-1.0_f32, 0.5_f64),
        (-4.0_f32, 0.0_f64),
    ] {
        let graph = build_batch_norm_graph(eps, vec![1.0, variance]);
        assert!(
            matches!(translate(&graph), Err(NyError::NumericalInstability(_))),
            "var {variance} + eps {eps} is not strictly positive and must be refused"
        );
    }
}

/// BatchNorm still refuses negative and non-finite epsilons: relaxing the
/// floor is not relaxing finiteness.
#[test]
fn batch_norm_refuses_negative_and_non_finite_eps() {
    for eps in [-1e-5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e300] {
        let graph = build_batch_norm_graph(eps, vec![1.0, 1.0]);
        assert!(
            matches!(translate(&graph), Err(NyError::NumericalInstability(_))),
            "BatchNorm eps {eps} must be refused"
        );
    }
}

/// Sub-floor epsilons that the DATA-variance family refuses are FINE for
/// BatchNorm — the two families genuinely have different weakest sound gates,
/// and the reconciliation must not collapse them into one rule.
#[test]
fn batch_norm_accepts_sub_floor_eps_the_data_variance_family_refuses() {
    let eps = 1e-30;
    // Refused for the data-variance family ...
    for (label, graph) in data_variance_graphs(eps) {
        assert!(
            matches!(translate(&graph), Err(NyError::NumericalInstability(_))),
            "{label}: eps {eps} must be refused"
        );
    }
    // ... but accepted (and buildable) for BatchNorm, whose variance is a
    // constant that already keeps the denominator away from zero.
    let graph = build_batch_norm_graph(eps, vec![1.0, 2.0]);
    let model = translate(&graph).expect("BatchNorm tolerates a sub-floor eps");
    model
        .build_graph_network(ny_build::GraphNetworkOptions::default())
        .expect("and it builds");
}
