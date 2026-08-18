// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DFL / expectation-decode simplex-envelope tightening tests.
//!
//! A Distribution-Focal-Loss style decode head computes
//! `y = softmax(logits) · w` where `w = [0, 1, .., K-1]` are constant bin
//! indices. Because softmax outputs are a probability simplex (`sum p_i = 1`,
//! `p_i >= 0`), the decoded value is a CONVEX COMBINATION of the bin weights and
//! provably lies in `[min_i w_i, max_i w_i] = [0, K-1]`.
//!
//! Term-wise IBP drops the `sum p_i = 1` constraint (it only keeps `p_i ∈ [0,1]`)
//! and therefore over-counts the upper bound as `sum_i 1 * w_i = K(K-1)/2`
//! (e.g. 120 for 16 bins instead of 15). The `ibp::dfl_envelope` tightening
//! intersects the IBP interval with the convex-combination envelope, restoring
//! the tight, still-sound bound.

use crate::*;
use ndarray::{Array1, ArrayD, IxDyn};

/// Bin-index weight row `[0, 1, .., K-1]` as a `(1, K)` Linear weight.
fn bin_index_weight(k: usize) -> ndarray::Array2<f32> {
    let row: Vec<f32> = (0..k).map(|i| i as f32).collect();
    ndarray::Array2::from_shape_vec((1, k), row).expect("valid (1,K) bin-index weight")
}

/// Build `logits --Softmax--> probs --Linear(bin idx)--> y` over a 1-D `[K]`
/// input. Returns `(graph, input_bounds)` with the requested logit interval.
fn dfl_linear_graph(k: usize, lo: f32, hi: f32) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(-1)),
    ));
    let decode = LinearLayer::new(bin_index_weight(k), None).expect("valid DFL Linear");
    graph.add_node(GraphNode::new(
        "decode",
        Layer::Linear(decode),
        vec!["probs".to_string()],
    ));
    graph.set_output("decode");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[k]), lo),
        ArrayD::from_elem(IxDyn(&[k]), hi),
    )
    .expect("valid logit bounds");
    (graph, input)
}

/// The naive term-wise IBP upper bound (sum of all weights) for K bins.
fn naive_termwise_upper(k: usize) -> f32 {
    (0..k).map(|i| i as f32).sum()
}

#[test]
fn dfl_linear_envelope_tightens_upper_bound() {
    // 16 bins: naive term-wise IBP would report upper = 0+1+..+15 = 120; the
    // convex-combination envelope proves the decode never exceeds 15.
    let k = 16;
    let (graph, input) = dfl_linear_graph(k, -2.0, 2.0);
    let out = graph.propagate_ibp(&input).expect("DFL IBP");
    assert_eq!(out.shape(), &[1]);

    let upper = out.upper()[[0]];
    let naive = naive_termwise_upper(k); // 120
    assert!(
        upper <= (k as f32 - 1.0) + 1e-3,
        "DFL upper {upper} must be tightened to <= K-1 = {}, naive term-wise = {naive}",
        k as f32 - 1.0
    );
    // Sanity: the tightening must have actually fired (far below naive).
    assert!(
        upper < naive * 0.5,
        "DFL upper {upper} should be far below the naive term-wise bound {naive}"
    );
}

#[test]
fn dfl_linear_envelope_tightens_lower_bound() {
    // Lower bin index is 0, so a convex combination is >= 0 regardless of logits.
    let k = 8;
    let (graph, input) = dfl_linear_graph(k, -5.0, 5.0);
    let out = graph.propagate_ibp(&input).expect("DFL IBP");
    let lower = out.lower()[[0]];
    assert!(
        lower >= -1e-4,
        "DFL lower {lower} must be >= min bin index (0)"
    );
    let upper = out.upper()[[0]];
    assert!(
        upper <= (k as f32 - 1.0) + 1e-3,
        "DFL upper {upper} must be <= K-1 = {}",
        k as f32 - 1.0
    );
}

#[test]
fn dfl_deadline_none_is_exact_legacy_envelope_path() {
    let (graph, input) = dfl_linear_graph(16, -2.0, 2.0);
    let expected = graph
        .propagate_ibp_with_engine(&input, None)
        .expect("legacy DFL IBP");
    let actual = graph
        .propagate_ibp_with_engine_and_deadline(&input, None, None)
        .expect("deadline=None DFL IBP");

    assert_eq!(actual.lower(), expected.lower());
    assert_eq!(actual.upper(), expected.upper());
    assert!(
        actual.upper()[[0]] <= 15.0 + 1e-3,
        "deadline=None must retain the historical DFL tightening"
    );
}

#[test]
fn dfl_finite_deadline_refuses_optional_unpolled_envelope() {
    let k = 16;
    let (graph, input) = dfl_linear_graph(k, -2.0, 2.0);
    let legacy = graph.propagate_ibp(&input).expect("legacy DFL IBP");
    let finite = graph
        .propagate_ibp_with_engine_and_deadline(
            &input,
            None,
            Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
        )
        .expect("live finite-deadline DFL IBP");

    assert!(
        legacy.upper()[[0]] <= (k - 1) as f32 + 1e-3,
        "legacy path should establish that the DFL envelope is applicable"
    );
    assert!(
        finite.upper()[[0]] > (k - 1) as f32 + 1e-3,
        "finite authority must return the untightened box instead of entering \
         unpolled DFL postprocessing"
    );
    let feasible_equal_logits_decode = (k - 1) as f32 / 2.0;
    assert!(
        finite.lower()[[0]] <= feasible_equal_logits_decode
            && finite.upper()[[0]] >= feasible_equal_logits_decode,
        "refusing an optional tightening must retain a sound box"
    );

    let expired = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_millis(1))
        .expect("Instant supports a 1ms subtraction");
    let error = graph
        .propagate_ibp_with_engine_and_deadline(&input, None, Some(expired))
        .expect_err("expired finite DFL graph propagation must fail");
    assert!(error.is_deadline_exceeded());
}

#[test]
fn dfl_linear_envelope_is_sound_vs_sampled_simplex() {
    // Soundness: every feasible decode value (over the full simplex of softmax
    // outputs, since with wide logits softmax can approach any vertex) must lie
    // within the tightened bounds. We exercise the simplex vertices and a few
    // interior mixtures directly through the constant bin-index weights.
    let k = 10;
    let (graph, input) = dfl_linear_graph(k, -8.0, 8.0);
    let out = graph.propagate_ibp(&input).expect("DFL IBP");
    let (lo, hi) = (out.lower()[[0]], out.upper()[[0]]);

    let weights: Vec<f32> = (0..k).map(|i| i as f32).collect();

    // Vertices: all mass on bin i -> decode = w_i.
    for &w in &weights {
        assert!(
            w >= lo - 1e-3 && w <= hi + 1e-3,
            "vertex decode {w} outside tightened bounds [{lo}, {hi}]"
        );
    }
    // Interior convex combinations with assorted (normalized) weightings.
    let mixes = [
        vec![0.5_f32; k],
        (0..k).map(|i| (i + 1) as f32).collect::<Vec<_>>(),
        (0..k).map(|i| (k - i) as f32).collect::<Vec<_>>(),
    ];
    for raw in &mixes {
        let s: f32 = raw.iter().sum();
        let p: Vec<f32> = raw.iter().map(|v| v / s).collect();
        let decode: f32 = p.iter().zip(&weights).map(|(pi, wi)| pi * wi).sum();
        assert!(
            decode >= lo - 1e-3 && decode <= hi + 1e-3,
            "convex-combination decode {decode} outside tightened bounds [{lo}, {hi}]"
        );
    }
}

#[test]
fn dfl_matmul_envelope_tightens_against_constant_weight() {
    // Same DFL structure but expressed as a MatMul against a *constant* weight
    // operand B (shape (1, K), used transposed so we contract over K). The
    // constant operand is manufactured from the perturbed input via
    // MulConstant(0) -> AddConstant(bin weights), giving a zero-width tensor.
    let k = 16;
    let mut graph = GraphNetwork::new();

    // A = softmax(logits), shape (1, K).
    graph.add_node(GraphNode::from_input(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(-1)),
    ));

    // bzero = input * 0  -> concrete zeros, shape (1, K).
    graph.add_node(GraphNode::from_input(
        "bzero",
        Layer::MulConstant(MulConstantLayer::scalar(0.0)),
    ));
    // bconst = bzero + [0,1,..,K-1] -> concrete bin weights, shape (1, K).
    let weights_row: Vec<f32> = (0..k).map(|i| i as f32).collect();
    let weight_const =
        ArrayD::from_shape_vec(IxDyn(&[1, k]), weights_row).expect("valid (1,K) constant");
    graph.add_node(GraphNode::new(
        "bconst",
        Layer::AddConstant(AddConstantLayer::new(weight_const)),
        vec!["bzero".to_string()],
    ));

    // out = A @ B^T  with B = bconst (1, K): out[0,0] = sum_k p_k * w_k.
    graph.add_node(GraphNode::binary(
        "decode",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "probs",
        "bconst",
    ));
    graph.set_output("decode");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, k]), -2.0_f32),
        ArrayD::from_elem(IxDyn(&[1, k]), 2.0_f32),
    )
    .expect("valid logit bounds");

    let out = graph.propagate_ibp(&input).expect("DFL MatMul IBP");
    let upper = out
        .upper()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let lower = out.lower().iter().copied().fold(f32::INFINITY, f32::min);
    let naive = naive_termwise_upper(k); // 120

    assert!(
        upper <= (k as f32 - 1.0) + 1e-3,
        "DFL MatMul upper {upper} must be tightened to <= K-1 = {}, naive = {naive}",
        k as f32 - 1.0
    );
    assert!(
        lower >= -1e-3,
        "DFL MatMul lower {lower} must be >= min bin index (0)"
    );
    assert!(
        upper < naive * 0.5,
        "DFL MatMul upper {upper} should be far below the naive term-wise bound {naive}"
    );
}

/// Guard: a Linear with a NON-zero bias is not a pure convex combination, so
/// the envelope must NOT fire (bound stays the conservative term-wise value).
/// This protects the soundness precondition rather than the tightening itself.
#[test]
fn dfl_linear_with_bias_does_not_apply_envelope() {
    let k = 4;
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(-1)),
    ));
    let bias = Array1::from_vec(vec![100.0_f32]);
    let decode = LinearLayer::new(bin_index_weight(k), Some(bias)).expect("valid biased Linear");
    graph.add_node(GraphNode::new(
        "decode",
        Layer::Linear(decode),
        vec!["probs".to_string()],
    ));
    graph.set_output("decode");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[k]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[k]), 1.0_f32),
    )
    .expect("valid logit bounds");
    let out = graph.propagate_ibp(&input).expect("biased DFL IBP");
    // With +100 bias, the true range is [100, 103]; the envelope [0, K-1] must
    // NOT be intersected in (it would unsoundly clamp the upper to 3).
    let upper = out.upper()[[0]];
    assert!(
        upper >= 100.0,
        "biased decode upper {upper} must reflect the +100 bias, not the unbiased envelope"
    );
}
