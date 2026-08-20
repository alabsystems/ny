// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Synthetic speaker embedding distance verification for avoice (#3499).
//!
//! This module executes the pre-ONNX slice from
//! `designs/2026-03-10-avoice-crown-capability-triage.md`:
//! - speaker encoder alone
//! - vocoder -> speaker encoder composition
//! - cosine-distance head on the composed graph
//!
//! The real `speaker_encoder.onnx` smoke now lives in
//! `ny-onnx::tests::core::avoice_speaker`; these tests keep the fast
//! synthetic graph for standalone CROWN/soundness coverage and for the later
//! vocoder -> speaker composition slice. The operator surface still matches the
//! ECAPA-style path: Conv1d, Sigmoid, Tanh, Softmax, MulBinary, Sub,
//! ReduceSum, Sqrt, and Concat.

use super::*;
use ndarray::{Array1, ArrayD, IxDyn};
use proptest::prelude::{prop_assert, proptest, ProptestConfig};

const WAVEFORM_LEN: usize = 8;
const MEL_LEN: usize = 12;
const INPUT_EPSILON: f32 = 0.01;
const SOUNDNESS_SAMPLES: usize = 12;
const COSINE_DISTANCE_MAX: f32 = 0.1;

fn bounded_constant_input(shape: &[usize], center: f32, epsilon: f32) -> BoundedTensor {
    BoundedTensor::from_epsilon(ArrayD::from_elem(IxDyn(shape), center), epsilon)
        .expect("valid bounded input")
}

fn concrete_constant_input(shape: &[usize], center: f32) -> BoundedTensor {
    BoundedTensor::concrete(ArrayD::from_elem(IxDyn(shape), center)).expect("valid concrete input")
}

fn conv1d(shape: [usize; 3], values: &[f32], bias: &[f32], input_length: usize) -> Conv1dLayer {
    let kernel = ArrayD::from_shape_vec(IxDyn(&shape), values.to_vec()).expect("valid kernel");
    let bias = Some(Array1::from_vec(bias.to_vec()));
    Conv1dLayer::with_input_length(kernel, bias, 1, 0, input_length).expect("valid conv1d")
}

fn add_unary(graph: &mut GraphNetwork, name: &str, layer: Layer, input: &str) {
    graph.add_node(GraphNode::new(name, layer, vec![input.to_string()]));
}

fn add_binary(graph: &mut GraphNetwork, name: &str, layer: Layer, lhs: &str, rhs: &str) {
    graph.add_node(GraphNode::binary(name, layer, lhs, rhs));
}

fn add_speaker_encoder(
    graph: &mut GraphNetwork,
    input: &str,
    waveform_len: usize,
    prefix: &str,
) -> String {
    let feat = format!("{prefix}_feat");
    add_unary(
        graph,
        &feat,
        Layer::Conv1d(conv1d(
            [2, 1, 1],
            &[0.35, -0.22],
            &[0.05, 0.03],
            waveform_len,
        )),
        input,
    );

    let gate_logits = format!("{prefix}_gate_logits");
    add_unary(
        graph,
        &gate_logits,
        Layer::Conv1d(conv1d(
            [2, 2, 1],
            &[0.45, 0.10, -0.08, 0.38],
            &[-0.10, 0.02],
            waveform_len,
        )),
        &feat,
    );

    let gate = format!("{prefix}_gate");
    add_unary(graph, &gate, Layer::Sigmoid(SigmoidLayer), &gate_logits);

    let gated = format!("{prefix}_gated");
    add_binary(
        graph,
        &gated,
        Layer::MulBinary(MulBinaryLayer),
        &feat,
        &gate,
    );

    let attn_hidden = format!("{prefix}_attn_hidden");
    add_unary(
        graph,
        &attn_hidden,
        Layer::Conv1d(conv1d(
            [2, 2, 1],
            &[0.32, 0.18, -0.14, 0.27],
            &[0.01, -0.02],
            waveform_len,
        )),
        &gated,
    );

    let attn_tanh = format!("{prefix}_attn_tanh");
    add_unary(graph, &attn_tanh, Layer::Tanh(TanhLayer), &attn_hidden);

    let attn_logits = format!("{prefix}_attn_logits");
    add_unary(
        graph,
        &attn_logits,
        Layer::Conv1d(conv1d(
            [2, 2, 1],
            &[0.22, 0.07, 0.06, 0.19],
            &[0.0, -0.03],
            waveform_len,
        )),
        &attn_tanh,
    );

    let attn_weights = format!("{prefix}_attn_weights");
    add_unary(
        graph,
        &attn_weights,
        Layer::Softmax(SoftmaxLayer::new(-1)),
        &attn_logits,
    );

    let weighted = format!("{prefix}_weighted");
    add_binary(
        graph,
        &weighted,
        Layer::MulBinary(MulBinaryLayer),
        &gated,
        &attn_weights,
    );

    let mean = format!("{prefix}_mean");
    add_unary(
        graph,
        &mean,
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
        &weighted,
    );

    let centered = format!("{prefix}_centered");
    add_binary(graph, &centered, Layer::Sub(SubLayer), &gated, &mean);

    let centered_sq = format!("{prefix}_centered_sq");
    add_binary(
        graph,
        &centered_sq,
        Layer::MulBinary(MulBinaryLayer),
        &centered,
        &centered,
    );

    let weighted_sq = format!("{prefix}_weighted_sq");
    add_binary(
        graph,
        &weighted_sq,
        Layer::MulBinary(MulBinaryLayer),
        &centered_sq,
        &attn_weights,
    );

    let variance = format!("{prefix}_variance");
    add_unary(
        graph,
        &variance,
        Layer::ReduceSum(ReduceSumLayer::new(vec![-1], true)),
        &weighted_sq,
    );

    let variance_eps = format!("{prefix}_variance_eps");
    add_unary(
        graph,
        &variance_eps,
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), 1e-4))),
        &variance,
    );

    let std = format!("{prefix}_std");
    add_unary(graph, &std, Layer::Sqrt(SqrtLayer::new()), &variance_eps);

    let embedding = format!("{prefix}_embedding");
    graph.add_node(GraphNode::new(
        &embedding,
        Layer::Concat(ConcatLayer::new(0)),
        vec![mean, std],
    ));

    embedding
}

fn add_vocoder(
    graph: &mut GraphNetwork,
    input: &str,
    mel_len: usize,
    prefix: &str,
) -> (String, usize) {
    let conv1 = format!("{prefix}_conv1");
    add_unary(
        graph,
        &conv1,
        Layer::Conv1d(conv1d(
            [2, 1, 3],
            &[0.18, 0.24, 0.18, -0.10, 0.22, -0.10],
            &[0.02, -0.01],
            mel_len,
        )),
        input,
    );

    let silu = format!("{prefix}_silu");
    add_unary(graph, &silu, Layer::SiLU(SiLULayer), &conv1);

    let output_len = mel_len - 4;
    let waveform = format!("{prefix}_waveform");
    add_unary(
        graph,
        &waveform,
        Layer::Conv1d(conv1d(
            [1, 2, 3],
            &[0.20, 0.12, 0.08, 0.05, 0.08, 0.05],
            &[0.0],
            mel_len - 2,
        )),
        &silu,
    );

    (waveform, output_len)
}

fn build_speaker_encoder_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let output = add_speaker_encoder(&mut graph, "_input", WAVEFORM_LEN, "enc");
    graph.set_output(&output);
    graph
}

fn build_vocoder_speaker_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let (waveform, waveform_len) = add_vocoder(&mut graph, "_input", MEL_LEN, "voc");
    let output = add_speaker_encoder(&mut graph, &waveform, waveform_len, "enc");
    graph.set_output(&output);
    graph
}

fn embedding_l2_norm(embedding: &ArrayD<f32>) -> f32 {
    embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

fn concrete_cosine_distance(embedding: &ArrayD<f32>, reference_embedding: &ArrayD<f32>) -> f32 {
    assert_eq!(
        embedding.shape(),
        reference_embedding.shape(),
        "cosine distance expects matching embedding shapes"
    );

    let embedding_norm = embedding_l2_norm(embedding);
    let reference_norm = embedding_l2_norm(reference_embedding);
    assert!(
        embedding_norm > 0.0,
        "cosine distance requires positive embedding norm, got {embedding_norm}"
    );
    assert!(
        reference_norm > 0.0,
        "cosine distance requires positive reference norm, got {reference_norm}"
    );

    let dot = embedding
        .iter()
        .zip(reference_embedding.iter())
        .map(|(&lhs, &rhs)| lhs * rhs)
        .sum::<f32>();
    1.0 - dot / (embedding_norm * reference_norm)
}

fn normalized_reference_weights(reference_embedding: &ArrayD<f32>) -> ArrayD<f32> {
    let reference_norm = embedding_l2_norm(reference_embedding);
    assert!(
        reference_norm > 0.0,
        "reference embedding norm must stay positive for cosine distance"
    );
    reference_embedding.mapv(|value| value / reference_norm)
}

fn add_cosine_dot(graph: &mut GraphNetwork, normalized_reference: ArrayD<f32>) -> &'static str {
    let dot_terms = "cosine_dot_terms";
    add_unary(
        graph,
        dot_terms,
        Layer::MulConstant(MulConstantLayer::new(normalized_reference)),
        "enc_embedding",
    );

    let dot = "cosine_dot";
    add_unary(
        graph,
        dot,
        Layer::ReduceSum(ReduceSumLayer::new(vec![0, 1], false)),
        dot_terms,
    );
    dot
}

fn add_embedding_norm_reciprocal(graph: &mut GraphNetwork) -> &'static str {
    let embedding_sq = "cosine_embedding_sq";
    add_unary(
        graph,
        embedding_sq,
        Layer::PowConstant(PowConstantLayer::new(2.0)),
        "enc_embedding",
    );

    let embedding_sq_sum = "cosine_embedding_sq_sum";
    add_unary(
        graph,
        embedding_sq_sum,
        Layer::ReduceSum(ReduceSumLayer::new(vec![0, 1], false)),
        embedding_sq,
    );

    let embedding_norm = "cosine_embedding_norm";
    add_unary(
        graph,
        embedding_norm,
        Layer::Sqrt(SqrtLayer::new()),
        embedding_sq_sum,
    );

    let embedding_norm_recip = "cosine_embedding_norm_recip";
    add_unary(
        graph,
        embedding_norm_recip,
        Layer::Reciprocal(ReciprocalLayer::new()),
        embedding_norm,
    );
    embedding_norm_recip
}

fn build_cosine_distance_graph_from_encoder(reference_embedding: ArrayD<f32>) -> GraphNetwork {
    let mut graph = build_vocoder_speaker_graph();
    let dot = add_cosine_dot(
        &mut graph,
        normalized_reference_weights(&reference_embedding),
    );
    let embedding_norm_recip = add_embedding_norm_reciprocal(&mut graph);
    let similarity = "cosine_similarity";
    add_binary(
        &mut graph,
        similarity,
        Layer::MulBinary(MulBinaryLayer),
        dot,
        embedding_norm_recip,
    );

    let distance = "cosine_distance";
    add_unary(
        &mut graph,
        distance,
        Layer::SubConstant(SubConstantLayer::new_reverse(ArrayD::from_elem(
            IxDyn(&[]),
            1.0,
        ))),
        similarity,
    );
    graph.set_output(distance);
    graph
}

fn compute_reference_embedding() -> ArrayD<f32> {
    let graph = build_vocoder_speaker_graph();
    let concrete_input = concrete_constant_input(&[1, MEL_LEN], 0.25);
    let output = graph
        .propagate_ibp(&concrete_input)
        .expect("reference embedding evaluation should succeed");
    output.lower().clone()
}

fn assert_bounds_sound(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    crown_bounds: &BoundedTensor,
    label: &str,
) {
    let flat_lower = crown_bounds
        .flatten()
        .lower()
        .as_slice()
        .expect("contiguous lower")
        .to_vec();
    let flat_upper = crown_bounds
        .flatten()
        .upper()
        .as_slice()
        .expect("contiguous upper")
        .to_vec();
    let input_lower = input
        .lower()
        .as_slice()
        .expect("contiguous input lower")
        .to_vec();
    let input_upper = input
        .upper()
        .as_slice()
        .expect("contiguous input upper")
        .to_vec();
    let input_dim = input_lower.len();

    for sample_idx in 0..SOUNDNESS_SAMPLES {
        let t = sample_idx as f32 / (SOUNDNESS_SAMPLES.saturating_sub(1) as f32);
        let mut concrete = ArrayD::zeros(input.lower().raw_dim());
        let concrete_slice = concrete.as_slice_mut().expect("contiguous concrete input");
        for j in 0..input_dim {
            let phase = ((t + j as f32 * 0.17) % 1.0).clamp(0.0, 1.0);
            concrete_slice[j] = input_lower[j] + phase * (input_upper[j] - input_lower[j]);
        }

        let concrete_bt = BoundedTensor::concrete(concrete).expect("valid concrete input");
        let concrete_output = graph
            .propagate_ibp(&concrete_bt)
            .expect("concrete graph evaluation should succeed");
        let flat_output = concrete_output.flatten();
        let values = flat_output.lower().as_slice().expect("contiguous output");

        for (dim, (&value, (&lower, &upper))) in values
            .iter()
            .zip(flat_lower.iter().zip(flat_upper.iter()))
            .enumerate()
        {
            assert!(
                value >= lower - 1e-4 && value <= upper + 1e-4,
                "{label}: sample {sample_idx}, dim {dim} = {value} not in [{lower}, {upper}]",
            );
        }
    }
}

#[ntest::timeout(60000)]
#[test]
fn test_speaker_encoder_crown_embedding_bounds_3499() {
    let graph = build_speaker_encoder_graph();
    let input = bounded_constant_input(&[1, WAVEFORM_LEN], 0.15, INPUT_EPSILON);

    let crown = graph
        .propagate_crown(&input)
        .expect("speaker encoder CROWN should succeed");

    assert_eq!(crown.shape(), &[4, 1]);
    for (&lower, &upper) in crown.lower().iter().zip(crown.upper().iter()) {
        assert!(
            lower.is_finite() && upper.is_finite(),
            "non-finite encoder bound"
        );
        assert!(
            lower <= upper + 1e-5,
            "inverted encoder bound: {lower} > {upper}"
        );
    }

    assert_bounds_sound(&graph, &input, &crown, "speaker encoder");
}

#[ntest::timeout(60000)]
#[test]
fn test_vocoder_speaker_encoder_crown_embedding_bounds_3499() {
    let graph = build_vocoder_speaker_graph();
    let input = bounded_constant_input(&[1, MEL_LEN], 0.25, INPUT_EPSILON);

    let crown = graph
        .propagate_crown(&input)
        .expect("vocoder->speaker encoder CROWN should succeed");

    assert_eq!(crown.shape(), &[4, 1]);
    for (&lower, &upper) in crown.lower().iter().zip(crown.upper().iter()) {
        assert!(
            lower.is_finite() && upper.is_finite(),
            "non-finite composed bound"
        );
        assert!(
            lower <= upper + 1e-5,
            "inverted composed bound: {lower} > {upper}"
        );
    }

    assert_bounds_sound(&graph, &input, &crown, "vocoder->speaker encoder");
}

#[ntest::timeout(60000)]
#[test]
fn test_vocoder_speaker_cosine_distance_is_nonvacuous_3499() {
    let reference_embedding = compute_reference_embedding();
    let graph = build_cosine_distance_graph_from_encoder(reference_embedding.clone());
    let input = bounded_constant_input(&[1, MEL_LEN], 0.25, INPUT_EPSILON);
    // Use a non-uniform sample inside the same epsilon box so the manual-vs-
    // graph comparison stays away from the near-zero cancellation at the
    // reference center.
    let sample_input = BoundedTensor::concrete(
        ArrayD::from_shape_vec(
            IxDyn(&[1, MEL_LEN]),
            (0..MEL_LEN)
                .map(|idx| if idx % 2 == 0 { 0.24 } else { 0.26 })
                .collect(),
        )
        .expect("valid sample shape"),
    )
    .expect("valid sample input");
    let sample_embedding = build_vocoder_speaker_graph()
        .propagate_ibp(&sample_input)
        .expect("sample embedding evaluation should succeed")
        .lower()
        .clone();
    let expected_distance = concrete_cosine_distance(&sample_embedding, &reference_embedding);
    let concrete_distance = graph
        .propagate_ibp(&sample_input)
        .expect("cosine distance evaluation should succeed")
        .flatten()
        .lower()[0];
    assert!(
        (concrete_distance - expected_distance).abs() <= 5e-5,
        "cosine distance head mismatch: graph={concrete_distance}, expected={expected_distance}"
    );

    let ibp = graph
        .propagate_ibp(&input)
        .expect("cosine distance IBP should succeed");
    let crown = graph
        .propagate_crown(&input)
        .expect("cosine distance CROWN should succeed");

    let ibp_upper = ibp.flatten().upper()[0];
    let crown_upper = crown.flatten().upper()[0];

    assert!(
        crown_upper.is_finite() && crown_upper <= COSINE_DISTANCE_MAX,
        "cosine distance upper bound should stay non-vacuous (< {}), got {crown_upper}",
        COSINE_DISTANCE_MAX
    );
    assert!(
        crown_upper <= ibp_upper + 1e-5,
        "cosine distance CROWN upper {crown_upper} should not exceed IBP upper {ibp_upper}",
    );

    assert_bounds_sound(&graph, &input, &crown, "vocoder->speaker cosine distance");
}

#[ntest::timeout(60000)]
#[test]
fn test_speaker_encoder_crown_tighter_than_ibp_3499() {
    let graph = build_speaker_encoder_graph();
    let input = bounded_constant_input(&[1, WAVEFORM_LEN], 0.15, INPUT_EPSILON);

    let ibp = graph
        .propagate_ibp(&input)
        .expect("speaker encoder IBP should succeed");
    let crown = graph
        .propagate_crown(&input)
        .expect("speaker encoder CROWN should succeed");

    let ibp_flat = ibp.flatten();
    let crown_flat = crown.flatten();
    let ibp_widths: Vec<f32> = ibp_flat
        .lower()
        .iter()
        .zip(ibp_flat.upper().iter())
        .map(|(&l, &u)| u - l)
        .collect();
    let crown_widths: Vec<f32> = crown_flat
        .lower()
        .iter()
        .zip(crown_flat.upper().iter())
        .map(|(&l, &u)| u - l)
        .collect();

    // CROWN bounds should be at least as tight as IBP on every dimension
    for (dim, (cw, iw)) in crown_widths.iter().zip(ibp_widths.iter()).enumerate() {
        assert!(
            *cw <= *iw + 1e-5,
            "dim {dim}: CROWN width {cw} exceeds IBP width {iw}"
        );
    }

    // Verify CROWN bounds are finite and non-inverted (core correctness).
    // Strict tightness (CROWN < IBP) is not guaranteed for small synthetic
    // models where the nonlinear relaxation gap can vanish.
    for (dim, cw) in crown_widths.iter().enumerate() {
        assert!(
            cw.is_finite() && *cw >= 0.0,
            "dim {dim}: invalid CROWN width {cw}"
        );
    }
}

// ---------------------------------------------------------------------------
// Proptests: soundness sampling with randomised inputs
// ---------------------------------------------------------------------------

// CASE COUNTS ARE MATCHED TO MEASURED PER-CASE COST, which is why these three
// tests sit in three blocks rather than one.  They previously shared a single
// `with_cases(200)`, and because `propagate_crown` on these graphs costs ~0.6s
// in a debug build, that made the last test alone a 119.8s serial run -- the
// single longest test in the crate, roughly two thirds of the entire suite's
// wall time, and the tail that every other test ends up waiting behind at high
// `--test-threads`.  Measured 2026-08-19 on this box, 200 cases each:
//
//   proptest_cosine_distance_nonnegative_3499        0.54s   (IBP only)
//   proptest_speaker_encoder_bounds_contain_concrete 25.84s  (+ CROWN)
//   proptest_cosine_distance_crown_bounds_noninvert 119.75s  (+ CROWN)
//
// The setup is NOT the cost and hoisting it would buy nothing: the IBP test
// rebuilds the reference embedding and the whole encoder graph 200 times too,
// and finishes in half a second.  It is `propagate_crown` end to end.
//
// So the cheap property keeps its full 200 cases and the two CROWN properties
// are cut until they cost seconds instead of minutes.  Re-measured after:
//
//   nonnegative      200 cases   0.54s -> 0.51s   (unchanged)
//   contain_concrete  64 cases  25.84s -> 8.42s
//   noninverted       16 cases 119.75s -> 15.05s
//
// -- about 122 seconds of serial test time.  Note the last is 0.94s/case
// rather than the 0.60s the 200-case run implies; a few seconds of that 15s
// is fixed process and lazy-init startup, not per-case work.
//
// Any seed that ever fails is persisted to .proptest-regressions and replays
// on every subsequent run regardless of case count, so lowering the count
// forfeits future random exploration, never a bug already found.
proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(64) })]

    /// CROWN bounds on the speaker encoder contain all concrete outputs
    /// for randomly sampled inputs within the bounded region.
    #[test]
    fn proptest_speaker_encoder_bounds_contain_concrete_3499(
        // 8 mix coefficients for WAVEFORM_LEN=8 inputs, each in [0, 1]
        mix in proptest::collection::vec(0.0f32..1.0, 8),
    ) {
        let graph = build_speaker_encoder_graph();
        let input = bounded_constant_input(&[1, WAVEFORM_LEN], 0.15, INPUT_EPSILON);

        let crown = graph
            .propagate_crown(&input)
            .expect("speaker encoder CROWN should succeed");

        // Construct a concrete input within the bounded region
        let input_lower = input.lower().as_slice().expect("contiguous").to_vec();
        let input_upper = input.upper().as_slice().expect("contiguous").to_vec();
        let concrete_vals: Vec<f32> = input_lower
            .iter()
            .zip(input_upper.iter())
            .zip(mix.iter())
            .map(|((&l, &u), &m)| l + m * (u - l))
            .collect();
        let concrete_input = BoundedTensor::concrete(
            ArrayD::from_shape_vec(IxDyn(&[1, WAVEFORM_LEN]), concrete_vals)
                .expect("valid shape"),
        )
        .expect("valid concrete input");

        let concrete_output = graph
            .propagate_ibp(&concrete_input)
            .expect("concrete evaluation should succeed");

        let crown_flat = crown.flatten();
        let output_flat = concrete_output.flatten();
        let crown_lower = crown_flat.lower().as_slice().expect("contiguous").to_vec();
        let crown_upper = crown_flat.upper().as_slice().expect("contiguous").to_vec();
        let output_vals = output_flat.lower().as_slice().expect("contiguous").to_vec();

        for (dim, (&val, (&lo, &hi))) in output_vals
            .iter()
            .zip(crown_lower.iter().zip(crown_upper.iter()))
            .enumerate()
        {
            prop_assert!(
                val >= lo - 1e-4 && val <= hi + 1e-4,
                "dim {}: concrete output {} not in CROWN bounds [{}, {}]",
                dim, val, lo, hi
            );
        }
    }

}

// IBP only, ~2.7ms per case -- cheap enough to keep the full 200.
proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Concrete cosine distance from the reference embedding is always in [0, 2]
    /// for inputs within the bounded region.
    #[test]
    fn proptest_cosine_distance_nonnegative_3499(
        mix in proptest::collection::vec(0.0f32..1.0, 12),
    ) {
        let reference_embedding = compute_reference_embedding();
        let graph = build_cosine_distance_graph_from_encoder(reference_embedding);
        let input = bounded_constant_input(&[1, MEL_LEN], 0.25, INPUT_EPSILON);

        // Sample a concrete input
        let input_lower = input.lower().as_slice().expect("contiguous").to_vec();
        let input_upper = input.upper().as_slice().expect("contiguous").to_vec();
        let concrete_vals: Vec<f32> = input_lower
            .iter()
            .zip(input_upper.iter())
            .zip(mix.iter())
            .map(|((&l, &u), &m)| l + m * (u - l))
            .collect();
        let concrete_input = BoundedTensor::concrete(
            ArrayD::from_shape_vec(IxDyn(&[1, MEL_LEN]), concrete_vals)
                .expect("valid shape"),
        )
        .expect("valid concrete input");

        let concrete_output = graph
            .propagate_ibp(&concrete_input)
            .expect("cosine distance evaluation should succeed");

        let distance = concrete_output.flatten().lower()[0];
        // f32 accumulated rounding across ~8 ops (MulConstant, ReduceSum,
        // PowConstant, Sqrt, Reciprocal, MulBinary, SubConstant) can push
        // cosine distance slightly below zero.  Reciprocal amplifies upstream
        // error when ||embedding|| is small.  Observed: -5.2e-6 on the
        // regression seed.  Tolerance 1e-4 matches the CROWN containment
        // tolerance on line 657 and gives ~20x margin.
        prop_assert!(
            distance >= -1e-4,
            "cosine distance should be non-negative (f32 tolerance -1e-4), got {}",
            distance
        );
        prop_assert!(
            distance <= 2.0 + 1e-4,
            "cosine distance should stay within [0, 2], got {}",
            distance
        );
    }

}

// ~0.6s per case in a debug build: the most expensive property in the crate.
proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(16) })]

    /// CROWN cosine-distance bounds stay finite and non-inverted across
    /// randomly perturbed input centers.
    #[test]
    fn proptest_cosine_distance_crown_bounds_noninverted_3499(
        center in proptest::collection::vec(0.1f32..0.4, 12),
    ) {
        let reference_embedding = compute_reference_embedding();
        let graph = build_cosine_distance_graph_from_encoder(reference_embedding);

        let center_arr = ArrayD::from_shape_vec(IxDyn(&[1, MEL_LEN]), center)
            .expect("valid shape");
        let input = BoundedTensor::from_epsilon(center_arr, INPUT_EPSILON)
            .expect("valid bounded input");

        let crown = graph
            .propagate_crown(&input)
            .expect("cosine distance CROWN should succeed");

        let flat = crown.flatten();
        let lower = flat.lower().as_slice().expect("contiguous").to_vec();
        let upper = flat.upper().as_slice().expect("contiguous").to_vec();

        for (dim, (&lo, &hi)) in lower.iter().zip(upper.iter()).enumerate() {
            prop_assert!(
                lo.is_finite() && hi.is_finite(),
                "dim {}: non-finite bound lo={}, hi={}",
                dim, lo, hi
            );
            prop_assert!(
                lo <= hi + 1e-4,
                "dim {}: inverted bounds lo={} > hi={}",
                dim, lo, hi
            );
        }
    }
}
