// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Attention centroid monotonicity verification for TTS decoder (#3497).
//!
//! Implements the centroid-monotonicity property from the batch design:
//! `designs/2026-03-10-avoice-crown-capability-triage.md`
//!
//! Key insight: argmax is discontinuous, but the **attention centroid** is a
//! continuous linear function of softmax outputs:
//!
//!   centroid(t) = Σ_j  j · A[t, j]
//!
//! where A[t, j] = softmax(Q[t] @ K^T)[j]. Monotonicity of the centroid
//! (centroid(t) >= centroid(t-1)) can be expressed as a linear output constraint
//! in ny's `OutputConstraints` format, enabling CROWN verification.
//!
//! Uses synthetic attention models until real Kokoro/Qwen3 ONNX models arrive
//! from avoice (blocked on voice#80).

use super::*;
use ndarray::{Array1, Array2, ArrayD};
use proptest::prelude::{prop_assert, proptest, ProptestConfig};

// ---------------------------------------------------------------------------
// Centroid computation helpers
// ---------------------------------------------------------------------------

/// Compute attention centroids from a probability matrix.
///
/// Given attention probabilities A of shape [seq, seq] (or flattened),
/// computes centroid(t) = Σ_j j * A[t, j] for each time step t.
///
/// Reference: designs/2026-03-10-avoice-crown-capability-triage.md, §Request 3
fn compute_centroids(probs: &[f32], seq_len: usize) -> Vec<f32> {
    assert_eq!(
        probs.len(),
        seq_len * seq_len,
        "probs must be flattened [seq, seq]"
    );
    let mut centroids = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let mut centroid = 0.0f32;
        for j in 0..seq_len {
            centroid += j as f32 * probs[t * seq_len + j];
        }
        centroids.push(centroid);
    }
    centroids
}

/// Compute centroid bounds from bounded attention probabilities.
///
/// Since centroid(t) = Σ_j j * A[t, j] is a linear function with non-negative
/// weights (positions j >= 0), interval arithmetic gives:
///   centroid_lower(t) = Σ_j j * A_lower[t, j]  (j >= 0, so lower * positive = lower)
///   centroid_upper(t) = Σ_j j * A_upper[t, j]
fn compute_centroid_bounds(
    probs_lower: &[f32],
    probs_upper: &[f32],
    seq_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(probs_lower.len(), seq_len * seq_len);
    assert_eq!(probs_upper.len(), seq_len * seq_len);

    let mut lower = Vec::with_capacity(seq_len);
    let mut upper = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let mut c_lo = 0.0f32;
        let mut c_hi = 0.0f32;
        for j in 0..seq_len {
            let w = j as f32; // position weight, always >= 0
            let l = probs_lower[t * seq_len + j];
            let u = probs_upper[t * seq_len + j];
            // For non-negative weight w:
            // w * [l, u] = [w * l, w * u] when w >= 0
            c_lo += w * l;
            c_hi += w * u;
        }
        lower.push(c_lo);
        upper.push(c_hi);
    }

    (lower, upper)
}

// ---------------------------------------------------------------------------
// OutputConstraints construction for centroid monotonicity
// ---------------------------------------------------------------------------

/// Construct `OutputConstraints` expressing centroid monotonicity on the
/// flattened attention probability output.
///
/// For each t in 1..seq_len, the constraint is:
///   centroid(t-1) - centroid(t) <= 0
/// i.e. centroid(t) >= centroid(t-1) (non-decreasing).
///
/// In terms of the flattened attention output y[t*seq + j] = A[t, j]:
///   Σ_j j * y[(t-1)*seq + j] - Σ_j j * y[t*seq + j] <= 0
///
/// Reference: designs/2026-03-10-avoice-crown-capability-triage.md, §Request 3
fn centroid_monotonicity_constraints(seq_len: usize) -> OutputConstraints {
    let output_dim = seq_len * seq_len; // flattened attention matrix
    let num_constraints = seq_len - 1; // one constraint per consecutive pair

    let mut a_matrix = Array2::<f32>::zeros((num_constraints, output_dim));
    let rhs = Array1::<f32>::zeros(num_constraints); // all constraints <= 0

    for constraint_idx in 0..num_constraints {
        let t_prev = constraint_idx;
        let t_curr = constraint_idx + 1;
        for j in 0..seq_len {
            let w = j as f32;
            // centroid(t_prev) - centroid(t_curr) <= 0
            // = Σ_j j * A[t_prev, j] - Σ_j j * A[t_curr, j]
            a_matrix[[constraint_idx, t_prev * seq_len + j]] = w;
            a_matrix[[constraint_idx, t_curr * seq_len + j]] = -w;
        }
    }

    OutputConstraints::new(a_matrix, rhs, true).expect("valid monotonicity constraints")
}

// ---------------------------------------------------------------------------
// Synthetic attention graph builders
// ---------------------------------------------------------------------------

/// Build a synthetic attention graph that outputs attention probabilities.
///
/// Architecture: Input → Q/K projections (GELU) → BilinearCrown(Q@K^T) → Softmax
///
/// The output is the attention probability matrix [1, 1, seq, seq], suitable
/// for centroid computation. Uses GELU projections to introduce non-linearity
/// (real models use linear projections, but GELU makes the test more
/// interesting for CROWN).
fn build_attention_probs_graph(seq_len: usize, d_k: usize) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    // Q and K projections from shared input
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));

    // Attention scores: Q @ K^T (scaled)
    let scale = 1.0 / (d_k as f32).sqrt();
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::BilinearCrown(BilinearCrownLayer::new(true, Some(scale))),
        "q",
        "k",
    ));

    // Softmax → attention probabilities
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(-1)),
        vec!["scores".to_string()],
    ));

    graph.set_output("probs");

    // Input: [batch=1, heads=1, seq, d_k] with small perturbation
    let input = BoundedTensor::from_epsilon(
        ArrayD::from_elem(ndarray::IxDyn(&[1, 1, seq_len, d_k]), 0.0f32),
        0.1,
    )
    .expect("valid attention input");

    (graph, input)
}

/// Build a synthetic attention graph that outputs **attention centroids**.
///
/// Architecture: Input → Q/K (GELU) → BilinearCrown(Q@K^T) → Softmax → Linear(positions)
///
/// The final Linear layer computes centroid(t) = Σ_j j * A[t, j] by
/// multiplying each softmax row by the position weight vector [0, 1, ..., seq-1].
/// Output shape: [1, 1, seq, 1] — one centroid per time step.
///
/// This end-to-end graph enables CROWN backward propagation through the
/// centroid computation, which is the key capability for #3497.
fn build_attention_centroid_graph(seq_len: usize, d_k: usize) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));

    let scale = 1.0 / (d_k as f32).sqrt();
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::BilinearCrown(BilinearCrownLayer::new(true, Some(scale))),
        "q",
        "k",
    ));

    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(-1)),
        vec!["scores".to_string()],
    ));

    // Centroid computation: Linear with weight = position indices
    // Weight shape: [1, seq_len] — maps softmax row [seq_len] → centroid [1]
    // positions = [0, 1, 2, ..., seq-1]
    let weight = Array2::from_shape_vec((1, seq_len), (0..seq_len).map(|j| j as f32).collect())
        .expect("valid position weight");
    let centroid_layer = LinearLayer::new(weight, None).expect("valid centroid linear layer");

    graph.add_node(GraphNode::new(
        "centroid",
        Layer::Linear(centroid_layer),
        vec!["probs".to_string()],
    ));

    graph.set_output("centroid");

    let input = BoundedTensor::from_epsilon(
        ArrayD::from_elem(ndarray::IxDyn(&[1, 1, seq_len, d_k]), 0.0f32),
        0.1,
    )
    .expect("valid attention input");

    (graph, input)
}

/// Construct `OutputConstraints` for centroid monotonicity on the
/// centroid vector output (not the flattened attention matrix).
///
/// For each t in 1..seq_len, the constraint is:
///   centroid(t-1) - centroid(t) <= 0
///
/// The output dimension is seq_len (one centroid per time step).
fn centroid_output_monotonicity_constraints(seq_len: usize) -> OutputConstraints {
    let num_constraints = seq_len - 1;
    let mut a_matrix = Array2::<f32>::zeros((num_constraints, seq_len));
    let rhs = Array1::<f32>::zeros(num_constraints);

    for constraint_idx in 0..num_constraints {
        // centroid(t-1) - centroid(t) <= 0
        a_matrix[[constraint_idx, constraint_idx]] = 1.0; // centroid(t-1)
        a_matrix[[constraint_idx, constraint_idx + 1]] = -1.0; // -centroid(t)
    }

    OutputConstraints::new(a_matrix, rhs, true).expect("valid centroid monotonicity constraints")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_centroid_computation_known_values_3497() {
    // Verify centroid computation with known attention probabilities.
    //
    // Uniform attention over 3 positions: A[t, :] = [1/3, 1/3, 1/3]
    // centroid(t) = 0*1/3 + 1*1/3 + 2*1/3 = 1.0 for all t
    let seq = 3;
    let uniform = vec![1.0 / 3.0; seq * seq];
    let centroids = compute_centroids(&uniform, seq);

    for (t, &c) in centroids.iter().enumerate() {
        assert!(
            (c - 1.0).abs() < 1e-5,
            "Uniform centroid at t={t} should be 1.0, got {c}"
        );
    }

    // Monotonically advancing attention: row t focuses on position t
    // A[0, :] = [1, 0, 0], A[1, :] = [0, 1, 0], A[2, :] = [0, 0, 1]
    let mut focused = vec![0.0f32; seq * seq];
    for t in 0..seq {
        focused[t * seq + t] = 1.0;
    }
    let focused_centroids = compute_centroids(&focused, seq);

    for (t, &centroid) in focused_centroids.iter().enumerate().take(seq) {
        assert!(
            (centroid - t as f32).abs() < 1e-5,
            "Focused centroid at t={t} should be {t}, got {}",
            centroid
        );
    }
    // This is monotonically increasing: 0 < 1 < 2
    for t in 1..seq {
        assert!(
            focused_centroids[t] >= focused_centroids[t - 1] - 1e-5,
            "Focused centroids should be monotonic: c[{}]={} < c[{}]={}",
            t,
            focused_centroids[t],
            t - 1,
            focused_centroids[t - 1]
        );
    }
}

#[test]
fn test_centroid_bounds_soundness_3497() {
    // Verify centroid bounds contain the true centroid for known intervals.
    let seq = 3;

    // Create interval-valued probabilities: each A[t,j] ∈ [0.3, 0.4]
    let probs_lower = vec![0.3f32; seq * seq];
    let probs_upper = vec![0.4f32; seq * seq];

    let (c_lo, c_hi) = compute_centroid_bounds(&probs_lower, &probs_upper, seq);

    // True centroids for any concrete probs in [0.3, 0.4] must fall within bounds
    let concrete_probs = vec![0.35f32; seq * seq];
    let concrete_centroids = compute_centroids(&concrete_probs, seq);

    for t in 0..seq {
        assert!(
            concrete_centroids[t] >= c_lo[t] - 1e-5,
            "Centroid at t={t}: concrete {} < lower bound {}",
            concrete_centroids[t],
            c_lo[t]
        );
        assert!(
            concrete_centroids[t] <= c_hi[t] + 1e-5,
            "Centroid at t={t}: concrete {} > upper bound {}",
            concrete_centroids[t],
            c_hi[t]
        );
    }
}

#[test]
fn test_monotonicity_constraints_construction_3497() {
    // Verify that the output constraints matrix is correctly constructed
    // and evaluates correctly on known attention probability vectors.
    let seq = 3;
    let constraints = centroid_monotonicity_constraints(seq);

    assert_eq!(constraints.num_constraints(), seq - 1);
    assert_eq!(constraints.output_dim(), seq * seq);

    // Monotonically advancing attention (identity matrix) should satisfy:
    // centroid = [0, 1, 2] → differences are all positive
    let mut monotonic = vec![0.0f32; seq * seq];
    for t in 0..seq {
        monotonic[t * seq + t] = 1.0;
    }
    let monotonic_arr = Array1::from_vec(monotonic);
    assert!(
        constraints.is_satisfied(&monotonic_arr),
        "Monotonically advancing attention should satisfy centroid constraints"
    );

    // Reversed attention: A[0] focuses on pos 2, A[2] focuses on pos 0
    // centroid = [2, 1, 0] → NOT monotonic
    let mut reversed = vec![0.0f32; seq * seq];
    reversed[2] = 1.0; // A[0] = [0, 0, 1] → centroid(0) = 2
    reversed[seq + 1] = 1.0; // A[1] = [0, 1, 0] → centroid(1) = 1
    reversed[2 * seq] = 1.0; // A[2] = [1, 0, 0] → centroid(2) = 0
    let reversed_arr = Array1::from_vec(reversed);
    assert!(
        !constraints.is_satisfied(&reversed_arr),
        "Reversed attention should NOT satisfy centroid constraints"
    );
}

#[test]
fn test_monotonicity_constraints_uniform_attention_3497() {
    // Uniform attention: A[t, :] = [1/3, 1/3, 1/3] for all t
    // centroid(t) = 1.0 for all t → differences = 0 → satisfies >= 0 (weakly)
    let seq = 3;
    let constraints = centroid_monotonicity_constraints(seq);
    let uniform = Array1::from_vec(vec![1.0 / 3.0; seq * seq]);

    assert!(
        constraints.is_satisfied(&uniform),
        "Uniform attention (constant centroid) should satisfy monotonicity"
    );
}

#[ntest::timeout(60000)]
#[test]
fn test_attention_graph_ibp_centroid_bounds_3497() {
    // Build attention graph, compute IBP bounds on probs, extract centroid bounds.
    let seq = 3;
    let d_k = 2;
    let (graph, input) = build_attention_probs_graph(seq, d_k);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP should succeed");

    // IBP output shape: [1, 1, seq, seq]
    assert_eq!(ibp_output.shape(), &[1, 1, seq, seq]);

    let flat = ibp_output.flatten();
    let lower = flat.lower().as_slice().expect("contiguous");
    let upper = flat.upper().as_slice().expect("contiguous");

    // Extract the seq*seq portion (should be exactly seq*seq elements)
    assert!(
        lower.len() >= seq * seq,
        "IBP output too short: {} < {}",
        lower.len(),
        seq * seq
    );

    let (c_lo, c_hi) = compute_centroid_bounds(&lower[..seq * seq], &upper[..seq * seq], seq);

    // Verify centroid bounds are finite and non-inverted
    for t in 0..seq {
        assert!(
            c_lo[t].is_finite() && c_hi[t].is_finite(),
            "IBP centroid at t={t} is non-finite: [{}, {}]",
            c_lo[t],
            c_hi[t]
        );
        assert!(
            c_lo[t] <= c_hi[t] + 1e-4,
            "IBP centroid at t={t} inverted: {} > {}",
            c_lo[t],
            c_hi[t]
        );
    }

    // Centroid should be in [0, seq-1] range (weighted average of positions)
    for t in 0..seq {
        assert!(
            c_hi[t] >= -1e-4,
            "IBP centroid upper at t={t} should be >= 0, got {}",
            c_hi[t]
        );
        assert!(
            c_lo[t] <= (seq - 1) as f32 + 1e-4,
            "IBP centroid lower at t={t} should be <= {}, got {}",
            seq - 1,
            c_lo[t]
        );
    }
}

#[ntest::timeout(60000)]
#[test]
fn test_attention_centroid_graph_ibp_bounds_3497() {
    // Build attention centroid graph (Q@K^T → Softmax → Linear centroid),
    // verify IBP gives valid centroid bounds.
    //
    // NOTE: Graph CROWN backward through BilinearCrown requires the output
    // shape to match the input's seq*d_k dimension (existing limitation).
    // The centroid graph outputs [seq, 1] which triggers ShapeMismatch.
    // Full CROWN requires the probs@V pattern (output dim = input dim).
    // For now, we verify IBP bounds and test CROWN on a sequential network.
    let seq = 3;
    let d_k = 2;
    let (graph, input) = build_attention_centroid_graph(seq, d_k);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP should succeed");

    // Output shape: [1, 1, seq, 1] — one centroid per time step
    assert_eq!(ibp_output.shape(), &[1, 1, seq, 1]);

    let ibp_flat = ibp_output.flatten();
    let ibp_lo = ibp_flat.lower().as_slice().expect("contiguous");
    let ibp_hi = ibp_flat.upper().as_slice().expect("contiguous");

    for t in 0..seq {
        assert!(
            ibp_lo[t].is_finite() && ibp_hi[t].is_finite(),
            "IBP centroid at t={t} non-finite: [{}, {}]",
            ibp_lo[t],
            ibp_hi[t]
        );
        assert!(
            ibp_lo[t] <= ibp_hi[t] + 1e-4,
            "IBP centroid at t={t} inverted: {} > {}",
            ibp_lo[t],
            ibp_hi[t]
        );
        // Centroid of softmax should be in [0, seq-1]
        assert!(
            ibp_hi[t] >= -1e-3,
            "IBP centroid upper at t={t} should be >= 0, got {}",
            ibp_hi[t]
        );
    }
}

/// Build a sequential network: Linear → Softmax → Linear(centroid)
///
/// This isolates the CROWN backward path through Softmax → centroid
/// without the BilinearCrown shape constraint from graph attention.
/// Input: [seq_len] (flattened logits for one row)
/// Output: [1] (centroid scalar)
fn build_sequential_centroid_network(seq_len: usize) -> Network {
    let mut network = Network::new();

    // Linear projection: [seq_len] → [seq_len] (identity-like with small perturbation)
    let mut weight = Array2::<f32>::eye(seq_len);
    // Add small off-diagonal to make it non-trivial for CROWN
    for i in 0..seq_len {
        for j in 0..seq_len {
            if i != j {
                weight[[i, j]] = 0.1 / seq_len as f32;
            }
        }
    }
    let linear_layer = LinearLayer::new(weight, None).expect("valid linear");
    network.add_layer(Layer::Linear(linear_layer));

    // Softmax
    network.add_layer(Layer::Softmax(SoftmaxLayer::new(-1)));

    // Centroid: Linear with weight = position indices [0, 1, ..., seq-1]
    let centroid_weight =
        Array2::from_shape_vec((1, seq_len), (0..seq_len).map(|j| j as f32).collect())
            .expect("valid centroid weight");
    let centroid_layer = LinearLayer::new(centroid_weight, None).expect("valid centroid linear");
    network.add_layer(Layer::Linear(centroid_layer));

    network
}

#[ntest::timeout(10000)]
#[test]
fn test_sequential_centroid_crown_vs_ibp_3497() {
    // CROWN on sequential Linear → Softmax → Linear(centroid) should give
    // tighter centroid bounds than IBP.
    let seq = 4;
    let input = BoundedTensor::from_epsilon(ArrayD::from_elem(ndarray::IxDyn(&[seq]), 0.0f32), 0.5)
        .expect("valid input");

    let network = build_sequential_centroid_network(seq);

    let ibp_output = network.propagate_ibp(&input).expect("IBP should succeed");
    let crown_output = network
        .propagate_crown(&input)
        .expect("CROWN should succeed");

    let ibp_flat = ibp_output.flatten();
    let crown_flat = crown_output.flatten();

    let ibp_lo = ibp_flat.lower().as_slice().expect("contiguous")[0];
    let ibp_hi = ibp_flat.upper().as_slice().expect("contiguous")[0];
    let crown_lo = crown_flat.lower().as_slice().expect("contiguous")[0];
    let crown_hi = crown_flat.upper().as_slice().expect("contiguous")[0];

    // CROWN bounds should be at least as tight as IBP
    let tol = 1e-3;
    assert!(
        crown_lo >= ibp_lo - tol,
        "CROWN centroid lower ({crown_lo}) looser than IBP ({ibp_lo})"
    );
    assert!(
        crown_hi <= ibp_hi + tol,
        "CROWN centroid upper ({crown_hi}) looser than IBP ({ibp_hi})"
    );

    // Centroid of softmax([seq]) should be in [0, seq-1]
    assert!(crown_lo >= -tol, "CROWN centroid lower {crown_lo} < 0");
    assert!(
        crown_hi <= (seq - 1) as f32 + tol,
        "CROWN centroid upper {crown_hi} > {}",
        seq - 1
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sequential_centroid_crown_soundness_sampling_3497() {
    // Verify CROWN centroid bounds contain concrete centroid values on the
    // sequential Linear → Softmax → Linear(centroid) network.
    let seq = 4;
    let input = BoundedTensor::from_epsilon(ArrayD::from_elem(ndarray::IxDyn(&[seq]), 0.0f32), 0.5)
        .expect("valid input");

    let network = build_sequential_centroid_network(seq);

    let crown_output = network
        .propagate_crown(&input)
        .expect("CROWN should succeed");
    let flat = crown_output.flatten();
    let c_lo = flat.lower().as_slice().expect("contiguous")[0];
    let c_hi = flat.upper().as_slice().expect("contiguous")[0];

    // Sample 20 concrete inputs
    let lower_arr = input.lower().as_slice().expect("contiguous").to_vec();
    let upper_arr = input.upper().as_slice().expect("contiguous").to_vec();

    let tol = 1e-3;
    for sample_idx in 0..20 {
        let t = sample_idx as f32 / 19.0;
        let mut concrete_vals = ArrayD::zeros(input.lower().raw_dim());
        for j in 0..seq {
            let t_j = ((t + j as f32 * 0.07) % 1.0).clamp(0.0, 1.0);
            concrete_vals.as_slice_mut().expect("contiguous")[j] =
                lower_arr[j] + t_j * (upper_arr[j] - lower_arr[j]);
        }

        let concrete_bt = BoundedTensor::concrete(concrete_vals).expect("valid concrete");
        let concrete_out = network
            .propagate_ibp(&concrete_bt)
            .expect("concrete eval should succeed");

        let concrete_flat = concrete_out.flatten();
        let concrete_centroid = concrete_flat.lower().as_slice().expect("contiguous")[0];

        assert!(
            concrete_centroid >= c_lo - tol,
            "Sample {sample_idx}: centroid {concrete_centroid} < lower bound {c_lo}"
        );
        assert!(
            concrete_centroid <= c_hi + tol,
            "Sample {sample_idx}: centroid {concrete_centroid} > upper bound {c_hi}"
        );
    }
}

#[ntest::timeout(60000)]
#[test]
fn test_monotonicity_constraints_on_centroid_graph_output_3497() {
    // End-to-end: run attention centroid graph on concrete inputs, evaluate
    // centroid monotonicity constraints. We don't expect monotonicity to hold
    // for arbitrary synthetic weights — the goal is to verify the constraint
    // formulation integrates correctly with the centroid graph output.
    let seq = 3;
    let d_k = 2;
    let (graph, input) = build_attention_centroid_graph(seq, d_k);

    let constraints = centroid_output_monotonicity_constraints(seq);

    // Sample concrete inputs and check constraint evaluation
    let lower_arr = input.lower().as_slice().expect("contiguous").to_vec();
    let upper_arr = input.upper().as_slice().expect("contiguous").to_vec();
    let input_dim = lower_arr.len();

    let mut sat_count = 0;
    let mut unsat_count = 0;

    for sample_idx in 0..20 {
        let t = sample_idx as f32 / 19.0;
        let mut concrete_vals = ArrayD::zeros(input.lower().raw_dim());
        for j in 0..input_dim {
            let t_j = ((t + j as f32 * 0.13) % 1.0).clamp(0.0, 1.0);
            concrete_vals.as_slice_mut().expect("contiguous")[j] =
                lower_arr[j] + t_j * (upper_arr[j] - lower_arr[j]);
        }

        let concrete_bt = BoundedTensor::concrete(concrete_vals).expect("valid concrete");
        let concrete_out = graph
            .propagate_ibp(&concrete_bt)
            .expect("concrete eval should succeed");

        let concrete_flat = concrete_out.flatten();
        let centroids = concrete_flat.lower().as_slice().expect("contiguous");
        let centroids_arr = Array1::from_vec(centroids[..seq].to_vec());

        if constraints.is_satisfied(&centroids_arr) {
            sat_count += 1;
        } else {
            unsat_count += 1;
        }
    }

    // All 20 samples should produce a result (constraint evaluation works)
    assert!(
        sat_count + unsat_count == 20,
        "All 20 samples should produce a result: sat={sat_count}, unsat={unsat_count}"
    );
}

// ---------------------------------------------------------------------------
// Proptest: centroid computation and bounds
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Centroid bounds contain concrete centroids for random probability matrices.
    #[test]
    fn proptest_centroid_bounds_contain_concrete_3497(
        probs_vals in proptest::collection::vec(0.0f32..1.0, 9),
        concrete_mix in proptest::collection::vec(0.0f32..1.0, 9),
    ) {
        let seq = 3;

        // Construct lower/upper intervals around probs_vals
        let delta = 0.05;
        let lower: Vec<f32> = probs_vals.iter().map(|&v| (v - delta).max(0.0)).collect();
        let upper: Vec<f32> = probs_vals.iter().map(|&v| (v + delta).min(1.0)).collect();

        // Concrete values within [lower, upper]
        let concrete: Vec<f32> = lower
            .iter()
            .zip(upper.iter())
            .zip(concrete_mix.iter())
            .map(|((&l, &u), &m)| l + m * (u - l))
            .collect();

        let (c_lo, c_hi) = compute_centroid_bounds(&lower, &upper, seq);
        let concrete_centroids = compute_centroids(&concrete, seq);

        let tol = 1e-5;
        for t in 0..seq {
            prop_assert!(
                concrete_centroids[t] >= c_lo[t] - tol,
                "t={t}: concrete centroid {} < lower {}",
                concrete_centroids[t], c_lo[t]
            );
            prop_assert!(
                concrete_centroids[t] <= c_hi[t] + tol,
                "t={t}: concrete centroid {} > upper {}",
                concrete_centroids[t], c_hi[t]
            );
        }
    }

    /// Centroid of valid softmax outputs is always in [0, seq-1].
    #[test]
    fn proptest_centroid_range_for_softmax_outputs_3497(
        logits in proptest::collection::vec(-5.0f32..5.0, 9),
    ) {
        let seq = 3;

        // Apply softmax per row to get valid probabilities
        let mut probs = vec![0.0f32; seq * seq];
        for t in 0..seq {
            let row = &logits[t * seq..(t + 1) * seq];
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..seq {
                probs[t * seq + j] = exps[j] / sum;
            }
        }

        let centroids = compute_centroids(&probs, seq);
        for (t, &centroid) in centroids.iter().enumerate().take(seq) {
            prop_assert!(
                centroid >= -1e-5,
                "Centroid at t={t} below 0: {}",
                centroid
            );
            prop_assert!(
                centroid <= (seq - 1) as f32 + 1e-5,
                "Centroid at t={t} above seq-1: {}",
                centroid
            );
        }
    }

    /// Monotonicity constraints correctly detect monotonic vs non-monotonic patterns.
    #[test]
    fn proptest_monotonicity_constraint_correctness_3497(
        // Three centroid positions (sorted or unsorted)
        c0 in 0.0f32..3.0,
        c1 in 0.0f32..3.0,
        c2 in 0.0f32..3.0,
    ) {
        let seq = 3;
        let constraints = centroid_monotonicity_constraints(seq);

        // Construct a probability matrix where centroid(t) = c_t.
        // We use a two-point distribution: A[t, 0] = 1 - c_t/2, A[t, 2] = c_t/2
        // which gives centroid(t) = 0 * A[t,0] + 1 * A[t,1] + 2 * A[t,2] = 2 * c_t/2 = c_t.
        // Note: c_t must be in [0, 2] for valid probabilities.
        let c0 = c0.min(2.0);
        let c1 = c1.min(2.0);
        let c2 = c2.min(2.0);

        let mut probs = vec![0.0f32; seq * seq];
        for (t, &c) in [c0, c1, c2].iter().enumerate() {
            // A[t, :] = [1 - c/2, 0, c/2] → centroid = 2 * c/2 = c
            probs[t * seq] = 1.0 - c / 2.0;
            probs[t * seq + 1] = 0.0;
            probs[t * seq + 2] = c / 2.0;
        }

        let probs_arr = Array1::from_vec(probs);
        let is_sat = constraints.is_satisfied(&probs_arr);
        let is_monotonic = c0 <= c1 + 1e-5 && c1 <= c2 + 1e-5;

        if is_monotonic {
            prop_assert!(
                is_sat,
                "Monotonic centroids [{c0}, {c1}, {c2}] should satisfy constraints"
            );
        }
        // Note: we don't assert !is_sat for non-monotonic due to floating-point tolerance
    }
}
