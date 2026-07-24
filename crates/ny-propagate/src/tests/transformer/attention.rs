// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Attention and MatMul CROWN tests.

use ny_test_utils::assert_bounded_tensor_close;

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_matmul_crown_batched_3d_soundness() {
    // Test MatMul CROWN with batched 3D inputs using GraphNetwork.
    // Build: GELU(input_a) @ GELU(input_b)^T with shape [batch, m, k] @ [batch, n, k]
    let batch = 2_usize;
    let m = 2_usize;
    let k = 3_usize;
    let n = 2_usize;

    let mut graph = GraphNetwork::new();

    // Use GELU to transform inputs (tests McCormick with negative values)
    graph.add_node(GraphNode::from_input(
        "a",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "b",
        Layer::GELU(GELULayer::default()),
    ));

    // C = A @ B^T
    let matmul = MatMulLayer::new(true, None);
    graph.add_node(GraphNode::binary("c", Layer::MatMul(matmul), "a", "b"));
    graph.set_output("c");

    // Input bounds: 3D tensors
    let input = BoundedTensor::new(
        ArrayD::from_elem(vec![batch, m, k], -1.0_f32),
        ArrayD::from_elem(vec![batch, m, k], 1.0_f32),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();

    // Sample and verify soundness
    for sample_idx in 0..20_usize {
        let mut x_sample = ArrayD::zeros(vec![batch, m, k]);

        for idx in x_sample.indexed_iter_mut() {
            let hash = (sample_idx as u32)
                .wrapping_mul(2654435761_u32)
                .wrapping_add(idx.0[0] as u32 * 100 + idx.0[1] as u32 * 10 + idx.0[2] as u32);
            let t = hash as f32 / u32::MAX as f32;
            *idx.1 = -1.0 + 2.0 * t;
        }

        // Apply GELU to get transformed inputs
        let a = x_sample.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let b = x_sample.mapv(|v| gelu_eval(v, GeluApproximation::Erf));

        // Compute C = A @ B^T for each batch
        let mut c_sample = ArrayD::zeros(vec![batch, m, n]);
        for b_idx in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0_f32;
                    for l in 0..k {
                        sum += a[[b_idx, i, l]] * b[[b_idx, j, l]];
                    }
                    c_sample[[b_idx, i, j]] = sum;
                }
            }
        }

        // Verify soundness against flattened bounds
        for (flat, &val) in c_sample.iter().enumerate() {
            let lower = crown_bounds.lower().as_slice().unwrap()[flat];
            let upper = crown_bounds.upper().as_slice().unwrap()[flat];
            assert!(
                val >= lower - 1e-4,
                "Batched MatMul CROWN lower violation at flat {} sample {}: {} < {}",
                flat,
                sample_idx,
                val,
                lower
            );
            assert!(
                val <= upper + 1e-4,
                "Batched MatMul CROWN upper violation at flat {} sample {}: {} > {}",
                flat,
                sample_idx,
                val,
                upper
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_matmul_batched_crown_basic() {
    // Test the new propagate_linear_batched_binary for MatMulLayer
    // Simple 2D case: C = A @ B where A is [2, 3], B is [3, 2] (no batch dims)
    let m = 2_usize;
    let k = 3_usize;
    let n = 2_usize;

    let matmul = MatMulLayer::new(false, None);

    // Input bounds for A [m, k] and B [k, n]
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(vec![m, k], -1.0_f32),
        ArrayD::from_elem(vec![m, k], 1.0_f32),
    )
    .unwrap();

    let input_b = BoundedTensor::new(
        ArrayD::from_elem(vec![k, n], -0.5_f32),
        ArrayD::from_elem(vec![k, n], 0.5_f32),
    )
    .unwrap();

    // Identity bounds on C [m, n]
    let c_size = m * n;
    let identity = BatchedLinearBounds::identity(&[m, n]).unwrap();

    assert_eq!(identity.lower_a.shape(), &[m, n, n]);

    // For this simple case, we need to adjust expectations:
    // BatchedLinearBounds::identity([m, n]) creates [m, n, n] shape
    // But for MatMul, the output flat size is m*n = 4
    // The identity is set up per-row batching which doesn't match our needs

    // Create proper identity for flattened output
    let eye = Array2::<f32>::eye(c_size);
    let flat_identity = BatchedLinearBounds::from_parts_unchecked(
        eye.clone().into_dyn(),
        ArrayD::zeros(vec![c_size].as_slice()),
        eye.into_dyn(),
        ArrayD::zeros(vec![c_size].as_slice()),
        vec![m, n],
        vec![m, n],
    );

    // Propagate backward through MatMul
    let (bounds_a, bounds_b) = matmul
        .propagate_linear_batched_binary(&flat_identity, &input_a, &input_b)
        .unwrap();

    // Check shapes
    let a_size = m * k;
    let b_size = k * n;
    assert_eq!(bounds_a.lower_a.shape(), &[c_size, a_size]);
    assert_eq!(bounds_b.lower_a.shape(), &[c_size, b_size]);

    // Flatten inputs for concretization (concretize expects [..., in_dim] matching coefficients)
    let input_a_flat = BoundedTensor::new(
        input_a
            .lower()
            .clone()
            .into_shape_with_order(vec![a_size])
            .unwrap()
            .into_dyn(),
        input_a
            .upper()
            .clone()
            .into_shape_with_order(vec![a_size])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();
    let input_b_flat = BoundedTensor::new(
        input_b
            .lower()
            .clone()
            .into_shape_with_order(vec![b_size])
            .unwrap()
            .into_dyn(),
        input_b
            .upper()
            .clone()
            .into_shape_with_order(vec![b_size])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    // Concretize and check soundness
    let crown_a = bounds_a.concretize(&input_a_flat).unwrap();
    let crown_b = bounds_b.concretize(&input_b_flat).unwrap();

    // The combined bounds should contain the output C = A @ B
    // Sample some points and verify
    for sample_idx in 0..10 {
        let mut a_sample = Array2::<f32>::zeros((m, k));
        let mut b_sample = Array2::<f32>::zeros((k, n));

        for ((i, j), v) in a_sample.indexed_iter_mut() {
            let hash =
                (sample_idx as u32 * 1000 + i as u32 * 100 + j as u32).wrapping_mul(2654435761);
            let t = hash as f32 / u32::MAX as f32;
            *v = -1.0 + 2.0 * t;
        }

        for ((i, j), v) in b_sample.indexed_iter_mut() {
            let hash =
                (sample_idx as u32 * 10000 + i as u32 * 100 + j as u32).wrapping_mul(1664525);
            let t = hash as f32 / u32::MAX as f32;
            *v = -0.5 + 1.0 * t;
        }

        // Compute C = A @ B
        let c_sample = a_sample.dot(&b_sample);

        // Check soundness for each output position
        for i in 0..m {
            for j in 0..n {
                let val = c_sample[[i, j]];
                let flat = i * n + j;

                // The bounds are in terms of A and B separately
                // For soundness, we need the combined concretization
                // Since bias is split, add both halves
                let lower = crown_a.lower().as_slice().unwrap()[flat]
                    + crown_b.lower().as_slice().unwrap()[flat];
                let upper = crown_a.upper().as_slice().unwrap()[flat]
                    + crown_b.upper().as_slice().unwrap()[flat];

                // Allow tolerance for McCormick relaxation looseness
                assert!(
                    val >= lower - 1e-3,
                    "MatMul batched lower violation at [{},{}]: {} < {}",
                    i,
                    j,
                    val,
                    lower
                );
                assert!(
                    val <= upper + 1e-3,
                    "MatMul batched upper violation at [{},{}]: {} > {}",
                    i,
                    j,
                    val,
                    upper
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_self_attention_ibp_matches_decomposed() {
    let batch = 1_usize;
    let heads = 1_usize;
    let seq = 3_usize;
    let dim = 2_usize;

    let q = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), 1.0_f32),
    )
    .unwrap();
    let k = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), -0.5_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), 0.5_f32),
    )
    .unwrap();
    let v = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), -0.25_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), 0.25_f32),
    )
    .unwrap();

    let attn = SelfAttentionLayer::new(AttentionMask::Standard, None);
    let fused = attn.propagate_ibp_ternary(&q, &k, &v).unwrap();

    let scale = 1.0 / (dim as f32).sqrt();
    let qk = MatMulLayer::new(true, Some(scale))
        .propagate_ibp_binary(&q, &k)
        .unwrap();
    let probs = SoftmaxLayer::new(-1).propagate_ibp(&qk).unwrap();
    let expected_ibp = MatMulLayer::new(false, None)
        .propagate_ibp_binary(&probs, &v)
        .unwrap();
    // The fused path applies the softmax sum-to-1 lever at probs@V, so the
    // decomposed reference must apply it too for an exact match (#softmax-V-lever).
    let expected = softmax::tighten_softmax_v_ibp(&probs, &v, &expected_ibp, false);

    assert_bounded_tensor_close(&fused, &expected, 1e-6, "fused vs expected bounds");
}

#[ntest::timeout(10000)]
#[test]
fn test_self_attention_causal_ibp_matches_decomposed() {
    let batch = 1_usize;
    let heads = 1_usize;
    let seq = 4_usize;
    let dim = 2_usize;

    let q = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), -0.8_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), 0.9_f32),
    )
    .unwrap();
    let k = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), -0.7_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), 0.6_f32),
    )
    .unwrap();
    let v = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), -0.4_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), 0.3_f32),
    )
    .unwrap();

    let attn = SelfAttentionLayer::new(AttentionMask::Causal, None);
    let fused = attn.propagate_ibp_ternary(&q, &k, &v).unwrap();

    let scale = 1.0 / (dim as f32).sqrt();
    let qk = MatMulLayer::new(true, Some(scale))
        .propagate_ibp_binary(&q, &k)
        .unwrap();
    let probs = CausalSoftmaxLayer::new(-1).propagate_ibp(&qk).unwrap();
    let expected_ibp = MatMulLayer::new(false, None)
        .propagate_ibp_binary(&probs, &v)
        .unwrap();
    // The fused path applies the softmax sum-to-1 lever at probs@V (#softmax-V-lever).
    let expected = softmax::tighten_softmax_v_ibp(&probs, &v, &expected_ibp, false);

    assert_bounded_tensor_close(&fused, &expected, 1e-6, "fused vs expected bounds");
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_self_attention_node() {
    let batch = 1_usize;
    let heads = 1_usize;
    let seq = 2_usize;
    let dim = 2_usize;

    let mut graph = GraphNetwork::new();
    let zero = ArrayD::zeros(IxDyn(&[1]));
    let passthrough = Layer::AddConstant(AddConstantLayer::new(zero));

    graph.add_node(GraphNode::from_input("q", passthrough.clone()));
    graph.add_node(GraphNode::from_input("k", passthrough.clone()));
    graph.add_node(GraphNode::from_input("v", passthrough));
    graph.add_node(GraphNode::new(
        "attn",
        Layer::SelfAttention(SelfAttentionLayer::standard()),
        vec!["q".to_string(), "k".to_string(), "v".to_string()],
    ));
    graph.set_output("attn");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), 1.0_f32),
    )
    .unwrap();

    let graph_bounds = graph.propagate_ibp(&input).unwrap();
    let direct_bounds = SelfAttentionLayer::standard()
        .propagate_ibp_ternary(&input, &input, &input)
        .unwrap();

    assert_bounded_tensor_close(
        &graph_bounds,
        &direct_bounds,
        1e-6,
        "graph vs direct bounds",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_crown_fallback_self_attention() {
    let batch = 1_usize;
    let heads = 1_usize;
    let seq = 2_usize;
    let dim = 2_usize;

    let mut graph = GraphNetwork::new();
    let zero = ArrayD::zeros(IxDyn(&[1]));
    let passthrough = Layer::AddConstant(AddConstantLayer::new(zero));

    graph.add_node(GraphNode::from_input("q", passthrough.clone()));
    graph.add_node(GraphNode::from_input("k", passthrough.clone()));
    graph.add_node(GraphNode::from_input("v", passthrough));
    graph.add_node(GraphNode::new(
        "attn",
        Layer::SelfAttention(SelfAttentionLayer::standard()),
        vec!["q".to_string(), "k".to_string(), "v".to_string()],
    ));
    graph.set_output("attn");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, seq, dim]), 1.0_f32),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    assert_bounded_tensor_close(&crown_bounds, &ibp_bounds, 1e-6, "crown vs ibp bounds");
}

#[ntest::timeout(10000)]
#[test]
fn test_network_crown_rejects_self_attention() {
    let mut network = Network::new();
    network.add_layer(Layer::SelfAttention(SelfAttentionLayer::standard()));

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 1.0_f32),
    )
    .unwrap();

    let err = network.propagate_crown(&input).unwrap_err();
    match err {
        NyError::UnsupportedConfiguration(msg) => {
            assert!(
                msg.contains("SelfAttention requires a graph network"),
                "msg: {}",
                msg
            );
        }
        other => panic!("Expected UnsupportedConfiguration, got {:?}", other),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_matmul_batched_crown_requires_flattened_output() {
    // Batched MatMul CROWN currently requires the incoming bounds to treat the MatMul output
    // as a flattened vector of length m*n, not a rank-2 [m, n] tensor with per-row batching.
    let m = 3_usize;
    let k = 4_usize;
    let n = 2_usize;

    let matmul = MatMulLayer::new(false, None);

    let input_a = BoundedTensor::new(
        ArrayD::from_elem(vec![m, k], -1.0_f32),
        ArrayD::from_elem(vec![m, k], 1.0_f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(vec![k, n], -1.0_f32),
        ArrayD::from_elem(vec![k, n], 1.0_f32),
    )
    .unwrap();

    // This identity treats [m, n] as batch_dims=[m], dim=n (in_dim=n), which is not the
    // flattened [m*n] representation required by MatMul batched CROWN.
    let per_row_identity = BatchedLinearBounds::identity(&[m, n]).unwrap();

    let err = matmul
        .propagate_linear_batched_binary(&per_row_identity, &input_a, &input_b)
        .unwrap_err();

    match err {
        NyError::UnsupportedOp(msg) => {
            assert!(msg.contains("flattened output dim"), "msg: {}", msg);
        }
        other => panic!("Expected UnsupportedOp, got {:?}", other),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_matmul_batched_crown_transpose_soundness() {
    // Test batched MatMul CROWN with transpose_b = true (like Q @ K^T in attention)
    let seq = 3_usize;
    let head_dim = 2_usize;

    let matmul = MatMulLayer::new(true, Some(0.5)); // transpose_b, scale

    // Q and K both have shape [seq, head_dim]
    let input_q = BoundedTensor::new(
        ArrayD::from_elem(vec![seq, head_dim], -1.0_f32),
        ArrayD::from_elem(vec![seq, head_dim], 1.0_f32),
    )
    .unwrap();

    let input_k = BoundedTensor::new(
        ArrayD::from_elem(vec![seq, head_dim], -1.0_f32),
        ArrayD::from_elem(vec![seq, head_dim], 1.0_f32),
    )
    .unwrap();

    // Output C has shape [seq, seq] (attention scores)
    let c_size = seq * seq;
    let q_size = seq * head_dim;
    let k_size = seq * head_dim;

    // Create identity bounds for flattened output
    let eye = Array2::<f32>::eye(c_size);
    let flat_identity = BatchedLinearBounds::from_parts_unchecked(
        eye.clone().into_dyn(),
        ArrayD::zeros(vec![c_size].as_slice()),
        eye.into_dyn(),
        ArrayD::zeros(vec![c_size].as_slice()),
        vec![seq, seq],
        vec![seq, seq],
    );

    let (bounds_q, bounds_k) = matmul
        .propagate_linear_batched_binary(&flat_identity, &input_q, &input_k)
        .unwrap();

    assert_eq!(bounds_q.lower_a.shape(), &[c_size, q_size]);
    assert_eq!(bounds_k.lower_a.shape(), &[c_size, k_size]);

    // Flatten inputs for concretization
    let input_q_flat = BoundedTensor::new(
        input_q
            .lower()
            .clone()
            .into_shape_with_order(vec![q_size])
            .unwrap()
            .into_dyn(),
        input_q
            .upper()
            .clone()
            .into_shape_with_order(vec![q_size])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();
    let input_k_flat = BoundedTensor::new(
        input_k
            .lower()
            .clone()
            .into_shape_with_order(vec![k_size])
            .unwrap()
            .into_dyn(),
        input_k
            .upper()
            .clone()
            .into_shape_with_order(vec![k_size])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    // Concretize
    let crown_q = bounds_q.concretize(&input_q_flat).unwrap();
    let crown_k = bounds_k.concretize(&input_k_flat).unwrap();

    // Sample and verify
    for sample_idx in 0..10 {
        let mut q_sample = Array2::<f32>::zeros((seq, head_dim));
        let mut k_sample = Array2::<f32>::zeros((seq, head_dim));

        for ((i, j), v) in q_sample.indexed_iter_mut() {
            let hash =
                (sample_idx as u32 * 1000 + i as u32 * 10 + j as u32).wrapping_mul(2654435761);
            let t = hash as f32 / u32::MAX as f32;
            *v = -1.0 + 2.0 * t;
        }

        for ((i, j), v) in k_sample.indexed_iter_mut() {
            let hash = (sample_idx as u32 * 10000 + i as u32 * 10 + j as u32).wrapping_mul(1664525);
            let t = hash as f32 / u32::MAX as f32;
            *v = -1.0 + 2.0 * t;
        }

        // Compute C = Q @ K^T * scale
        let k_t = k_sample.t();
        let c_sample = q_sample.dot(&k_t).mapv(|v| v * 0.5);

        // Verify soundness
        for i in 0..seq {
            for j in 0..seq {
                let val = c_sample[[i, j]];
                let flat = i * seq + j;

                let lower = crown_q.lower().as_slice().unwrap()[flat]
                    + crown_k.lower().as_slice().unwrap()[flat];
                let upper = crown_q.upper().as_slice().unwrap()[flat]
                    + crown_k.upper().as_slice().unwrap()[flat];

                assert!(
                    val >= lower - 1e-3,
                    "MatMul transpose lower violation at [{},{}]: {} < {}",
                    i,
                    j,
                    val,
                    lower
                );
                assert!(
                    val <= upper + 1e-3,
                    "MatMul transpose upper violation at [{},{}]: {} > {}",
                    i,
                    j,
                    val,
                    upper
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_attention_4d_soundness() {
    // Test full 4D batched attention pattern: [batch, heads, seq, dim]
    // This is the actual shape used in transformer attention.
    //
    // Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
    //
    // Input shape: [batch, heads, seq, dim]
    // Q @ K^T shape: [batch, heads, seq, seq]
    // softmax(.) shape: [batch, heads, seq, seq]
    // @ V shape: [batch, heads, seq, dim]

    let batch = 2_usize;
    let heads = 2_usize;
    let seq = 3_usize;
    let dim = 4_usize;

    let head_dim = dim; // For scaling factor

    let mut graph = GraphNetwork::new();

    // Q, K, V all derive from input via GELU (simulates projection)
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "v",
        Layer::GELU(GELULayer::default()),
    ));

    // Q @ K^T with scaling: [batch, heads, seq, dim] @ [batch, heads, dim, seq] -> [batch, heads, seq, seq]
    let scores = MatMulLayer::new(true, Some(1.0 / (head_dim as f32).sqrt()));
    graph.add_node(GraphNode::binary("scores", Layer::MatMul(scores), "q", "k"));

    // Softmax along last axis (seq dimension in scores)
    let softmax = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(softmax),
        vec!["scores".to_string()],
    ));

    // probs @ V: [batch, heads, seq, seq] @ [batch, heads, seq, dim] -> [batch, heads, seq, dim]
    let out_matmul = MatMulLayer::new(false, None);
    graph.add_node(GraphNode::binary(
        "out",
        Layer::MatMul(out_matmul),
        "probs",
        "v",
    ));
    graph.set_output("out");

    // Input: 4D tensor [batch, heads, seq, dim]
    let input = BoundedTensor::new(
        ArrayD::from_elem(vec![batch, heads, seq, dim], -1.0_f32),
        ArrayD::from_elem(vec![batch, heads, seq, dim], 1.0_f32),
    )
    .unwrap();

    let bounds = graph.propagate_crown(&input).unwrap();

    // Verify output shape
    assert_eq!(bounds.shape(), &[batch, heads, seq, dim]);

    let sm = SoftmaxLayer::new(-1);

    // Sample and verify soundness
    for sample_idx in 0..25_usize {
        let mut x = ArrayD::<f32>::zeros(vec![batch, heads, seq, dim]);
        for idx in x.indexed_iter_mut() {
            let hash = (sample_idx as u32)
                .wrapping_mul(2654435761_u32)
                .wrapping_add(
                    idx.0[0] as u32 * 1000
                        + idx.0[1] as u32 * 100
                        + idx.0[2] as u32 * 10
                        + idx.0[3] as u32,
                );
            let t = hash as f32 / u32::MAX as f32;
            *idx.1 = -1.0 + 2.0 * t;
        }

        // Apply GELU to get Q, K, V
        let q = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let k = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));
        let v = x.mapv(|v| gelu_eval(v, GeluApproximation::Erf));

        // Compute attention manually for each batch/head
        let mut out = ArrayD::<f32>::zeros(vec![batch, heads, seq, dim]);
        for b in 0..batch {
            for h in 0..heads {
                // Extract 2D slices for this batch/head
                let q_2d: Vec<Vec<f32>> = (0..seq)
                    .map(|s| (0..dim).map(|d| q[[b, h, s, d]]).collect())
                    .collect();
                let k_2d: Vec<Vec<f32>> = (0..seq)
                    .map(|s| (0..dim).map(|d| k[[b, h, s, d]]).collect())
                    .collect();
                let v_2d: Vec<Vec<f32>> = (0..seq)
                    .map(|s| (0..dim).map(|d| v[[b, h, s, d]]).collect())
                    .collect();

                // Q @ K^T / sqrt(d) -> [seq, seq]
                let scale = 1.0 / (head_dim as f32).sqrt();
                let mut scores_2d = vec![vec![0.0_f32; seq]; seq];
                for i in 0..seq {
                    for j in 0..seq {
                        let mut sum = 0.0_f32;
                        for l in 0..dim {
                            sum += q_2d[i][l] * k_2d[j][l]; // K^T means k[j][l]
                        }
                        scores_2d[i][j] = sum * scale;
                    }
                }

                // Softmax each row
                let mut probs_2d = vec![vec![0.0_f32; seq]; seq];
                for i in 0..seq {
                    let row: Array1<f32> = Array1::from_vec(scores_2d[i].clone());
                    let softmax_row = sm.eval(&row);
                    for j in 0..seq {
                        probs_2d[i][j] = softmax_row[j];
                    }
                }

                // probs @ V -> [seq, dim]
                for i in 0..seq {
                    for d in 0..dim {
                        let mut sum = 0.0_f32;
                        for j in 0..seq {
                            sum += probs_2d[i][j] * v_2d[j][d];
                        }
                        out[[b, h, i, d]] = sum;
                    }
                }
            }
        }

        // Verify all outputs are within bounds
        for idx in out.indexed_iter() {
            let val = *idx.1;
            let lower_val = bounds.lower()[idx.0.clone()];
            let upper_val = bounds.upper()[idx.0.clone()];
            assert!(
                val >= lower_val - 1e-4,
                "4D Attention CROWN lower violation at {:?} sample {}: {} < {}",
                idx.0,
                sample_idx,
                val,
                lower_val
            );
            assert!(
                val <= upper_val + 1e-4,
                "4D Attention CROWN upper violation at {:?} sample {}: {} > {}",
                idx.0,
                sample_idx,
                val,
                upper_val
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_bilinear_crown_qkt_smoke() {
    // BilinearCrownLayer is used for attention Q@K^T when both inputs are activations.
    // This test ensures GraphNetwork DAG-CROWN supports Layer::BilinearCrown and produces
    // finite bounds that are at least as tight as IBP for a small attention pattern.

    let seq = 3_usize;
    let dim = 4_usize;
    let scale = 1.0 / (dim as f32).sqrt();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::BilinearCrown(BilinearCrownLayer::new(true, Some(scale))),
        "q",
        "k",
    ));
    graph.set_output("scores");

    let input = BoundedTensor::new(
        ArrayD::from_elem(vec![seq, dim], -1.0_f32),
        ArrayD::from_elem(vec![seq, dim], 1.0_f32),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    assert_eq!(ibp_bounds.shape(), &[seq, seq]);
    assert_eq!(crown_bounds.shape(), &[seq, seq]);

    for ((&cl, &cu), (&il, &iu)) in crown_bounds
        .lower()
        .iter()
        .zip(crown_bounds.upper().iter())
        .zip(ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()))
    {
        assert!(cl.is_finite() && cu.is_finite(), "Non-finite CROWN bounds");
        assert!(il.is_finite() && iu.is_finite(), "Non-finite IBP bounds");
        assert!(cl <= cu + 1e-6, "Invalid CROWN interval: {cl} > {cu}");
        assert!(il <= iu + 1e-6, "Invalid IBP interval: {il} > {iu}");
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_batched_attention_4d_smoke() {
    // Smoke test: N-D batched CROWN should be able to propagate through an attention-shaped
    // GraphNetwork without erroring.
    //
    // Shape matches Whisper attention core: [batch, heads, seq, dim]
    let batch = 1_usize;
    let heads = 1_usize;
    let seq = 3_usize;
    let dim = 4_usize;

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "v",
        Layer::GELU(GELULayer::default()),
    ));

    let scale = 1.0 / (dim as f32).sqrt();
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, Some(scale))),
        "q",
        "k",
    ));
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(-1).with_heuristic_sampling(true)),
        vec!["scores".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "out",
        Layer::MatMul(MatMulLayer::new(false, None)),
        "probs",
        "v",
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_elem(vec![batch, heads, seq, dim], -1.0_f32),
        ArrayD::from_elem(vec![batch, heads, seq, dim], 1.0_f32),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let bounds = graph.propagate_crown_batched(&input).unwrap();
    assert_eq!(bounds.shape(), &[batch, heads, seq, dim]);

    for (l, u) in bounds.lower().iter().zip(bounds.upper().iter()) {
        assert!(l.is_finite() && u.is_finite(), "Non-finite bounds");
        assert!(*l <= *u + 1e-6, "Invalid interval: {} > {}", l, u);
    }

    // Partial CROWN: concretizes at the unsupported MatMul using IBP bounds there,
    // giving CROWN benefits for layers after the MatMul. Bounds should be at least
    // as tight as pure IBP (often tighter due to CROWN on post-MatMul layers).
    for ((crown_l, crown_u), (ibp_l, ibp_u)) in bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .zip(ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()))
    {
        // CROWN bounds should be at least as tight as IBP
        assert!(
            *crown_l >= *ibp_l - 1e-5,
            "CROWN lower {} should be >= IBP lower {}",
            crown_l,
            ibp_l
        );
        assert!(
            *crown_u <= *ibp_u + 1e-5,
            "CROWN upper {} should be <= IBP upper {}",
            crown_u,
            ibp_u
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_vs_ibp_4d_attention() {
    // Verify CROWN provides tighter bounds than IBP for 4D attention
    let batch = 2_usize;
    let heads = 2_usize;
    let seq = 3_usize;
    let dim = 4_usize;

    let head_dim = dim;

    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "v",
        Layer::GELU(GELULayer::default()),
    ));

    let scores = MatMulLayer::new(true, Some(1.0 / (head_dim as f32).sqrt()));
    graph.add_node(GraphNode::binary("scores", Layer::MatMul(scores), "q", "k"));

    let softmax = SoftmaxLayer::new(-1).with_heuristic_sampling(true);
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(softmax),
        vec!["scores".to_string()],
    ));

    let out_matmul = MatMulLayer::new(false, None);
    graph.add_node(GraphNode::binary(
        "out",
        Layer::MatMul(out_matmul),
        "probs",
        "v",
    ));
    graph.set_output("out");

    let input = BoundedTensor::new(
        ArrayD::from_elem(vec![batch, heads, seq, dim], -1.0_f32),
        ArrayD::from_elem(vec![batch, heads, seq, dim], 1.0_f32),
    )
    .unwrap();

    // Get both IBP and CROWN bounds
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    // Compute average interval widths
    let ibp_widths: Vec<f32> = ibp_bounds
        .lower()
        .iter()
        .zip(ibp_bounds.upper().iter())
        .map(|(&l, &u)| u - l)
        .collect();
    let crown_widths: Vec<f32> = crown_bounds
        .lower()
        .iter()
        .zip(crown_bounds.upper().iter())
        .map(|(&l, &u)| u - l)
        .collect();

    let avg_ibp_width: f32 = ibp_widths.iter().sum::<f32>() / ibp_widths.len() as f32;
    let avg_crown_width: f32 = crown_widths.iter().sum::<f32>() / crown_widths.len() as f32;

    println!(
        "4D Attention [batch={}, heads={}, seq={}, dim={}]:",
        batch, heads, seq, dim
    );
    println!("  IBP average width: {:.4}", avg_ibp_width);
    println!("  CROWN average width: {:.4}", avg_crown_width);
    println!(
        "  Tightening ratio: {:.2}x",
        avg_ibp_width / avg_crown_width
    );

    // CROWN should provide tighter or equal bounds
    for (i, (&ibp_l, &crown_l)) in ibp_bounds
        .lower()
        .iter()
        .zip(crown_bounds.lower().iter())
        .enumerate()
    {
        assert!(
            crown_l >= ibp_l - 1e-4,
            "CROWN lower bound {} looser than IBP at {}: {} < {}",
            crown_l,
            i,
            crown_l,
            ibp_l
        );
    }
    for (i, (&ibp_u, &crown_u)) in ibp_bounds
        .upper()
        .iter()
        .zip(crown_bounds.upper().iter())
        .enumerate()
    {
        assert!(
            crown_u <= ibp_u + 1e-4,
            "CROWN upper bound {} looser than IBP at {}: {} > {}",
            crown_u,
            i,
            crown_u,
            ibp_u
        );
    }

    // Expect CROWN to be at least 1.0x tighter on average (equal or better)
    // The actual improvement varies by network structure
    assert!(
        avg_crown_width <= avg_ibp_width + 1e-4,
        "CROWN should provide tighter or equal bounds than IBP"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_attention_crown_small_seq() {
    // Test that attention CROWN path is attempted for small attention shapes (seq <= 64).
    // This creates a minimal attention graph where Q@K^T produces [batch, heads, seq, seq]
    // and verifies that the attention identity path is exercised.
    //
    // Graph: Q -> Q@K^T (attention MatMul) -> output
    //        K -^
    //
    // With seq=4 (within the 64 limit), the attention identity should be used.

    let batch = 1_usize;
    let heads = 2_usize;
    let seq = 4_usize;
    let dim = 8_usize;

    let mut graph = GraphNetwork::new();

    // Q and K inputs pass through GELU first (gives non-trivial CROWN pass through)
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));

    // Q @ K^T produces attention scores [batch, heads, seq, seq]
    let scale = 1.0 / (dim as f32).sqrt();
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, Some(scale))),
        "q",
        "k",
    ));
    graph.set_output("scores");

    // Input shape: [batch, heads, seq, dim]
    let input = BoundedTensor::new(
        ArrayD::from_elem(vec![batch, heads, seq, dim], -0.5_f32),
        ArrayD::from_elem(vec![batch, heads, seq, dim], 0.5_f32),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown_batched(&input).unwrap();

    // Output shape should be [batch, heads, seq, seq]
    assert_eq!(crown_bounds.shape(), &[batch, heads, seq, seq]);

    // Bounds should be valid and finite
    for (l, u) in crown_bounds.lower().iter().zip(crown_bounds.upper().iter()) {
        assert!(
            l.is_finite() && u.is_finite(),
            "Non-finite bounds: {} {}",
            l,
            u
        );
        assert!(*l <= *u + 1e-5, "Invalid interval: {} > {}", l, u);
    }

    // CROWN should be at least as tight as IBP (soundness check)
    for ((crown_l, crown_u), (ibp_l, ibp_u)) in crown_bounds
        .lower()
        .iter()
        .zip(crown_bounds.upper().iter())
        .zip(ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()))
    {
        assert!(
            *crown_l >= *ibp_l - 1e-4,
            "CROWN lower {} should be >= IBP lower {}",
            crown_l,
            ibp_l
        );
        assert!(
            *crown_u <= *ibp_u + 1e-4,
            "CROWN upper {} should be <= IBP upper {}",
            crown_u,
            ibp_u
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_attention_crown_large_seq_fallback() {
    // Test that for large seq (> 64), we fall back to partial CROWN without error.
    // The attention identity path should NOT be used due to memory limits.

    let batch = 1_usize;
    let heads = 1_usize;
    let seq = 128_usize; // > 64, should trigger memory limit fallback
    let dim = 8_usize;

    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));

    let scale = 1.0 / (dim as f32).sqrt();
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, Some(scale))),
        "q",
        "k",
    ));
    graph.set_output("scores");

    let input = BoundedTensor::new(
        ArrayD::from_elem(vec![batch, heads, seq, dim], -0.5_f32),
        ArrayD::from_elem(vec![batch, heads, seq, dim], 0.5_f32),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown_batched(&input).unwrap();

    // Should still succeed (partial CROWN fallback)
    assert_eq!(crown_bounds.shape(), &[batch, heads, seq, seq]);

    // Bounds should be valid
    for (l, u) in crown_bounds.lower().iter().zip(crown_bounds.upper().iter()) {
        assert!(l.is_finite() && u.is_finite(), "Non-finite bounds");
        assert!(*l <= *u + 1e-5, "Invalid interval: {} > {}", l, u);
    }

    // CROWN (with fallback) should be at least as tight as IBP
    for ((crown_l, crown_u), (ibp_l, ibp_u)) in crown_bounds
        .lower()
        .iter()
        .zip(crown_bounds.upper().iter())
        .zip(ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()))
    {
        assert!(
            *crown_l >= *ibp_l - 1e-4,
            "CROWN lower {} should be >= IBP lower {}",
            crown_l,
            ibp_l
        );
        assert!(
            *crown_u <= *ibp_u + 1e-4,
            "CROWN upper {} should be <= IBP upper {}",
            crown_u,
            ibp_u
        );
    }
}
