// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Deep-stack IBP overflow must SATURATE to a conservative infinite box, not
//! hard-error.
//!
//! With real transformer weights (the nn PLBert / ALBERT-style stack under
//! Kokoro production weights), `GraphNetwork::propagate_ibp` legitimately
//! produces bounds that overflow f32 under outward directed rounding
//! (`add_up_f32` near `f32::MAX` → `+inf`). An infinite bound is sound and
//! maximally conservative — the enclosure `[-inf, +inf]` contains every real
//! value — so propagation must carry it through and record a (possibly
//! vacuous) result. A strict `BoundedTensor::new` site that rejects the
//! infinity with `NumericalInstability("... NaN or Inf")` instead aborts
//! verification entirely, violating the conservative-bounds convention
//! (`new_allow_infinite` / `new_repaired(Conservative)`, as Linear IBP uses).
//!
//! The graphs below reproduce the overflow WITHOUT the Kokoro weights: a chain
//! of Linear layers with weight magnitude 1e12 on input [-1, 1] escalates the
//! interval 4e12 → 1.6e25 → 6.4e37 → overflow (±inf), and the saturated box
//! then flows through the layer types the PLBert encoder uses (Add residual,
//! GELU, LayerNorm, attention MatMul, Softmax, SelfAttention).
//!
//! Sites this repro caught and that must stay saturating:
//! - GELU IBP (`layers/softmax/gelu/mod.rs`): strict `BoundedTensor::new`
//!   rejected the legitimate `GELU(+inf) = +inf` upper endpoint.
//! - LayerNorm IBP (`layers/normalization/layer_norm/ibp/common.rs`): the
//!   entry guard rejected ±inf input bounds outright instead of returning the
//!   conservative `fallback_output_bounds` enclosure.
//!
//! NaN must STILL fail closed at both sites: NaN carries no bound information,
//! and a silently repaired NaN could mask a real numerical bug.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_propagate::layers::{
    AddLayer, GELULayer, LayerNormLayer, LinearLayer, MatMulLayer, SelfAttentionLayer, SoftmaxLayer,
};
use ny_propagate::{BoundPropagation, GraphNetwork, GraphNode, Layer, NETWORK_INPUT};
use ny_tensor::BoundedTensor;

const HIDDEN: usize = 4;
const SEQ: usize = 2;

/// A `HIDDEN x HIDDEN` Linear with every weight equal to `magnitude`.
fn big_linear(magnitude: f32) -> LinearLayer {
    let weight = Array2::from_elem((HIDDEN, HIDDEN), magnitude);
    LinearLayer::new(weight, None).expect("valid linear layer")
}

fn layer_norm() -> LayerNormLayer {
    LayerNormLayer::new(Array1::ones(HIDDEN), Array1::zeros(HIDDEN), 1e-5)
        .expect("valid layer norm")
}

/// Input box [-1, 1] of shape [SEQ, HIDDEN].
fn unit_input() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ, HIDDEN]), -1.0),
        ArrayD::from_elem(IxDyn(&[SEQ, HIDDEN]), 1.0),
    )
    .expect("valid input box")
}

/// A fully saturated box [-inf, +inf] of shape [SEQ, HIDDEN].
fn saturated_input() -> BoundedTensor {
    BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(IxDyn(&[SEQ, HIDDEN]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[SEQ, HIDDEN]), f32::INFINITY),
    )
    .expect("valid saturated box")
}

/// A box with a NaN endpoint (test-utils constructor bypasses validation).
fn nan_input() -> BoundedTensor {
    let mut upper = ArrayD::from_elem(IxDyn(&[SEQ, HIDDEN]), 1.0);
    upper[[0, 0]] = f32::NAN;
    BoundedTensor::new_unchecked(ArrayD::from_elem(IxDyn(&[SEQ, HIDDEN]), -1.0), upper)
        .expect("test-only NaN box")
}

/// The overflowing Linear chain: |bounds| go 4e12 → 1.6e25 → 6.4e37 → ±inf.
fn add_overflow_chain(graph: &mut GraphNetwork) {
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(big_linear(1e12)),
    ));
    graph.add_node(GraphNode::new(
        "lin2",
        Layer::Linear(big_linear(1e12)),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin3",
        Layer::Linear(big_linear(1e12)),
        vec!["lin2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "lin4",
        Layer::Linear(big_linear(1e12)),
        vec!["lin3".to_string()],
    ));
}

/// A PLBert FFN segment on the saturated path: the segment output itself is a
/// sound VACUOUS (infinite) box — the case nn records for a transformer
/// segment whose bounds legitimately overflowed.
#[ntest::timeout(60000)]
#[test]
fn ffn_segment_on_overflowed_box_returns_saturated_bounds() {
    let mut graph = GraphNetwork::new();
    add_overflow_chain(&mut graph);
    graph.add_node(GraphNode::binary(
        "res1",
        Layer::Add(AddLayer),
        "lin4",
        NETWORK_INPUT,
    ));
    // GELU on the saturated box: range is [min GELU ≈ -0.17, +inf).
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["res1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "ffn_out",
        Layer::Linear(big_linear(1e12)),
        vec!["gelu".to_string()],
    ));
    graph.set_output("ffn_out");

    let bounds = graph
        .propagate_ibp(&unit_input())
        .expect("deep-stack IBP overflow must saturate, not error");

    assert_eq!(bounds.shape(), &[SEQ, HIDDEN]);
    assert!(
        bounds
            .lower()
            .iter()
            .chain(bounds.upper().iter())
            .any(|v| v.is_infinite()),
        "the overflowed segment output must contain infinite (saturated) entries"
    );
    assert!(
        bounds
            .lower()
            .iter()
            .chain(bounds.upper().iter())
            .all(|v| !v.is_nan()),
        "saturated bounds must not contain NaN"
    );
    assert!(
        bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .all(|(l, u)| l <= u),
        "saturated bounds must remain a valid (non-inverted) box"
    );
}

/// The full PLBert-style encoder segment: overflow chain, residual Add,
/// LayerNorm, Q/K/V projections, scaled QK^T MatMul, Softmax, probs@V MatMul,
/// GELU FFN, final residual + LayerNorm. Must complete without error; every
/// node output is a sound box (LayerNorm may legally re-tighten via its
/// conservative sqrt(n-1) fallback enclosure).
#[ntest::timeout(60000)]
#[test]
fn full_plbert_chain_on_overflowed_box_propagates_without_error() {
    let mut graph = GraphNetwork::new();
    add_overflow_chain(&mut graph);
    graph.add_node(GraphNode::binary(
        "res1",
        Layer::Add(AddLayer),
        "lin4",
        NETWORK_INPUT,
    ));
    // LayerNorm directly on the saturated box (the site that hard-errored).
    graph.add_node(GraphNode::new(
        "ln1",
        Layer::LayerNorm(layer_norm()),
        vec!["res1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "q",
        Layer::Linear(big_linear(1e12)),
        vec!["ln1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "k",
        Layer::Linear(big_linear(1e12)),
        vec!["ln1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "v",
        Layer::Linear(big_linear(1e12)),
        vec!["ln1".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, Some(0.5))),
        "q",
        "k",
    ));
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(-1)),
        vec!["scores".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "ctx",
        Layer::MatMul(MatMulLayer::new(false, None)),
        "probs",
        "v",
    ));
    graph.add_node(GraphNode::binary(
        "res2",
        Layer::Add(AddLayer),
        "ctx",
        "ln1",
    ));
    graph.add_node(GraphNode::new(
        "ln2",
        Layer::LayerNorm(layer_norm()),
        vec!["res2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "ffn1",
        Layer::Linear(big_linear(1e12)),
        vec!["ln2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["ffn1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "ffn2",
        Layer::Linear(big_linear(1e12)),
        vec!["gelu".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "res3",
        Layer::Add(AddLayer),
        "ffn2",
        "ln2",
    ));
    graph.add_node(GraphNode::new(
        "ln3",
        Layer::LayerNorm(layer_norm()),
        vec!["res3".to_string()],
    ));
    graph.set_output("ln3");

    let bounds = graph
        .propagate_ibp(&unit_input())
        .expect("full PLBert-style chain must propagate through the overflow, not error");

    assert_eq!(bounds.shape(), &[SEQ, HIDDEN]);
    assert!(
        bounds
            .lower()
            .iter()
            .chain(bounds.upper().iter())
            .all(|v| !v.is_nan()),
        "final bounds must not contain NaN"
    );
    assert!(
        bounds
            .lower()
            .iter()
            .zip(bounds.upper().iter())
            .all(|(l, u)| l <= u),
        "final bounds must remain a valid (non-inverted) box"
    );
}

/// Each PLBert layer type, driven directly on a fully saturated box, must
/// return a sound box (never an error). This is the bisect harness that
/// located the GELU and LayerNorm strict-constructor sites.
#[ntest::timeout(60000)]
#[test]
fn each_plbert_layer_saturates_on_infinite_box() {
    let input = saturated_input();

    let drive_unary = |name: &str, layer: Layer| {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("n", layer));
        graph.set_output("n");
        let bounds = graph
            .propagate_ibp(&input)
            .unwrap_or_else(|e| panic!("{name} on saturated box must not error: {e}"));
        assert!(
            bounds
                .lower()
                .iter()
                .chain(bounds.upper().iter())
                .all(|v| !v.is_nan()),
            "{name} on saturated box must not emit NaN"
        );
    };

    drive_unary("Linear", Layer::Linear(big_linear(1e12)));
    drive_unary("LayerNorm", Layer::LayerNorm(layer_norm()));
    drive_unary("GELU", Layer::GELU(GELULayer::default()));
    drive_unary("Softmax", Layer::Softmax(SoftmaxLayer::new(-1)));

    // Binary / ternary layers with both operands saturated.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::binary(
        "n",
        Layer::Add(AddLayer),
        NETWORK_INPUT,
        NETWORK_INPUT,
    ));
    graph.set_output("n");
    graph
        .propagate_ibp(&input)
        .expect("Add on saturated boxes must not error");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::binary(
        "n",
        Layer::MatMul(MatMulLayer::new(true, Some(0.5))),
        NETWORK_INPUT,
        NETWORK_INPUT,
    ));
    graph.set_output("n");
    graph
        .propagate_ibp(&input)
        .expect("attention MatMul (QK^T) on saturated boxes must not error");

    // Fused attention path (Q = K = V = saturated).
    SelfAttentionLayer::standard()
        .propagate_ibp_ternary(&input, &input, &input)
        .expect("SelfAttention on saturated Q/K/V must not error");
}

/// NaN must still fail closed at the two sites the overflow repro fixed:
/// saturation is for legitimate ±inf only, never for NaN.
#[ntest::timeout(60000)]
#[test]
fn nan_input_still_fails_closed_at_fixed_sites() {
    let input = nan_input();

    let gelu = GELULayer::default();
    let err = gelu
        .propagate_ibp(&input)
        .expect_err("GELU IBP must fail closed on NaN input");
    assert!(
        matches!(err, ny_core::NyError::NumericalInstability(_)),
        "GELU NaN rejection must be NumericalInstability, got: {err:?}"
    );

    let ln = layer_norm();
    let err = ln
        .propagate_ibp(&input)
        .expect_err("LayerNorm IBP must fail closed on NaN input");
    assert!(
        matches!(err, ny_core::NyError::NumericalInstability(_)),
        "LayerNorm NaN rejection must be NumericalInstability, got: {err:?}"
    );
}
