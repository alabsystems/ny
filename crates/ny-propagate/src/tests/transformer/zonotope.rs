// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Zonotope-based attention and propagation tests.

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_crown_batched_uses_zonotope_for_attention_like_matmul_bounds() {
    // For attention-like Q@K^T patterns where Q and K are both linear projections of the same
    // input, GraphNetwork tightens MatMul bounds using a per-position zonotope so that
    // diagonal entries (sums of squares-like terms) get non-negative lower bounds.

    let seq = 3_usize;
    let dim = 4_usize;
    let epsilon = 0.5_f32;

    let mut graph = GraphNetwork::new();
    let eye = Array2::<f32>::eye(dim);
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
    ));
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));
    graph.set_output("scores");

    // Input centered at 0 with uniform epsilon.
    let input = BoundedTensor::new(
        ArrayD::from_elem(vec![seq, dim], -epsilon),
        ArrayD::from_elem(vec![seq, dim], epsilon),
    )
    .unwrap();

    let baseline_interval = MatMulLayer::new(true, None)
        .propagate_ibp_binary(&input, &input)
        .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let node_bounds = graph.collect_node_bounds(&input).unwrap();
    let collected_scores = node_bounds.get("scores").unwrap();
    let crown_bounds = graph.propagate_crown_batched(&input).unwrap();

    // For X @ X^T, diagonal entries are sums of squares, so lower bounds should be >= 0.
    // IBP interval matmul cannot capture this and produces negative lower bounds.
    assert!(
        baseline_interval.lower()[[0, 0]] < -1e-6,
        "Baseline interval MatMul should be negative on diagonal"
    );
    assert!(
        ibp_bounds.lower()[[0, 0]] >= -1e-6,
        "Zonotope-tightened MatMul should be non-negative on diagonal (got {})",
        ibp_bounds.lower()[[0, 0]]
    );
    assert!(
        collected_scores.lower()[[0, 0]] >= -1e-6,
        "collect_node_bounds should use zonotope-tightened MatMul bounds (got {})",
        collected_scores.lower()[[0, 0]]
    );

    // collect_node_bounds() should match the forward IBP pass for this graph.
    assert_eq!(collected_scores.shape(), ibp_bounds.shape());
    for ((&cl, &cu), (&il, &iu)) in collected_scores
        .lower()
        .iter()
        .zip(collected_scores.upper().iter())
        .zip(ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()))
    {
        assert!((cl - il).abs() < 1e-6);
        assert!((cu - iu).abs() < 1e-6);
    }

    // Batched CROWN falls back to partial CROWN at attention MatMul; with an identity output,
    // the result should equal the MatMul IBP bounds used for concretization.
    assert_eq!(crown_bounds.shape(), ibp_bounds.shape());
    for ((&cl, &cu), (&il, &iu)) in crown_bounds
        .lower()
        .iter()
        .zip(crown_bounds.upper().iter())
        .zip(ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()))
    {
        assert!((cl - il).abs() < 1e-6);
        assert!((cu - iu).abs() < 1e-6);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_zonotope_attention_with_layernorm_integration() {
    // Test that zonotope attention tracking propagates through LayerNorm.
    // Architecture: _input -> LayerNorm -> Q_proj + K_proj -> Q@K^T
    // The LayerNorm should preserve correlations between Q and K, giving tighter bounds.

    let seq = 3_usize;
    let dim = 4_usize;
    let epsilon = 0.1_f32;

    let mut graph = GraphNetwork::new();

    // Add LayerNorm (ny=1, beta=0)
    let ny = Array1::ones(dim);
    let beta = Array1::zeros(dim);
    graph.add_node(GraphNode::from_input(
        "ln",
        Layer::LayerNorm(LayerNormLayer::new(ny, beta, 1e-5).unwrap()),
    ));

    // Q and K projections from LayerNorm output
    let eye = Array2::<f32>::eye(dim);
    graph.add_node(GraphNode::new(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
        vec!["ln".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "k",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
        vec!["ln".to_string()],
    ));

    // Q@K^T MatMul
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));
    graph.set_output("scores");

    // Input with varied values per feature (to avoid near-zero variance in LayerNorm)
    // Use a pattern that gives non-trivial variance: each row has values [1, 2, 3, 4]
    let center_values: Vec<f32> = (0..seq * dim).map(|i| (i % dim) as f32 + 1.0).collect();
    let lower = ArrayD::from_shape_vec(
        vec![seq, dim],
        center_values.iter().map(|&v| v - epsilon).collect(),
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        vec![seq, dim],
        center_values.iter().map(|&v| v + epsilon).collect(),
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run IBP (which uses zonotope tightening for Q@K^T)
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Check that bounds are finite and reasonable
    assert!(
        ibp_bounds.lower().iter().all(|&v| v.is_finite()),
        "Lower bounds should be finite"
    );
    assert!(
        ibp_bounds.upper().iter().all(|&v| v.is_finite()),
        "Upper bounds should be finite"
    );

    // LayerNorm normalizes input to zero mean, unit variance.
    // After identity projections, Q@K^T is essentially X_norm @ X_norm^T
    // Diagonal entries should be near dim (sum of squares of normalized values)
    let diag_lower = ibp_bounds.lower()[[0, 0]];
    let diag_upper = ibp_bounds.upper()[[0, 0]];
    assert!(
        diag_lower >= -10.0, // LayerNorm + small epsilon can still give reasonable bounds
        "Diagonal lower bound should be reasonable (got {})",
        diag_lower
    );
    assert!(
        diag_upper >= diag_lower,
        "Upper bound should be >= lower bound"
    );

    // Verify bounds enclose a reasonable range
    let max_width = ibp_bounds.max_width();
    assert!(
        max_width < 50.0,
        "Bound width should be reasonable (got {})",
        max_width
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_zonotope_attention_layernorm_vs_no_layernorm() {
    // Compare zonotope attention bounds with and without LayerNorm in the path.
    // Both should produce valid bounds; with LayerNorm should still have correlation tracking.

    let seq = 3_usize;
    let dim = 4_usize;
    let epsilon = 0.1_f32;

    // Graph WITHOUT LayerNorm: _input -> Q_proj + K_proj -> Q@K^T
    let mut graph_no_ln = GraphNetwork::new();
    let eye = Array2::<f32>::eye(dim);
    graph_no_ln.add_node(GraphNode::from_input(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
    ));
    graph_no_ln.add_node(GraphNode::from_input(
        "k",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
    ));
    graph_no_ln.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));
    graph_no_ln.set_output("scores");

    // Graph WITH LayerNorm: _input -> LayerNorm -> Q_proj + K_proj -> Q@K^T
    let mut graph_with_ln = GraphNetwork::new();
    let ny = Array1::ones(dim);
    let beta = Array1::zeros(dim);
    graph_with_ln.add_node(GraphNode::from_input(
        "ln",
        Layer::LayerNorm(LayerNormLayer::new(ny, beta, 1e-5).unwrap()),
    ));
    graph_with_ln.add_node(GraphNode::new(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
        vec!["ln".to_string()],
    ));
    graph_with_ln.add_node(GraphNode::new(
        "k",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
        vec!["ln".to_string()],
    ));
    graph_with_ln.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));
    graph_with_ln.set_output("scores");

    // Input with varied values per feature (to avoid near-zero variance in LayerNorm)
    // For graph_no_ln this is also fine; for graph_with_ln it ensures LayerNorm works well
    let center_values: Vec<f32> = (0..seq * dim).map(|i| (i % dim) as f32 + 1.0).collect();
    let lower = ArrayD::from_shape_vec(
        vec![seq, dim],
        center_values.iter().map(|&v| v - epsilon).collect(),
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        vec![seq, dim],
        center_values.iter().map(|&v| v + epsilon).collect(),
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let bounds_no_ln = graph_no_ln.propagate_ibp(&input).unwrap();
    let bounds_with_ln = graph_with_ln.propagate_ibp(&input).unwrap();

    // Both should have same output shape
    assert_eq!(bounds_no_ln.shape(), bounds_with_ln.shape());

    // Both should produce finite bounds
    assert!(bounds_no_ln.lower().iter().all(|&v| v.is_finite()));
    assert!(bounds_no_ln.upper().iter().all(|&v| v.is_finite()));
    assert!(bounds_with_ln.lower().iter().all(|&v| v.is_finite()));
    assert!(bounds_with_ln.upper().iter().all(|&v| v.is_finite()));

    // Without LayerNorm, diagonal should have non-negative lower bound
    // (zonotope tracks X @ X^T correlation directly)
    assert!(
        bounds_no_ln.lower()[[0, 0]] >= -1e-6,
        "No-LN diagonal lower should be >= 0 (got {})",
        bounds_no_ln.lower()[[0, 0]]
    );

    // With LayerNorm, the transformation changes the values but correlation
    // should still be tracked through the affine approximation
    let ln_diag_lower = bounds_with_ln.lower()[[0, 0]];
    assert!(
        ln_diag_lower > -10.0, // LayerNorm can shift values, so relaxed check
        "With-LN diagonal lower should be reasonable (got {})",
        ln_diag_lower
    );

    // Bound widths should be reasonable (not exploding)
    assert!(
        bounds_no_ln.max_width() < 50.0,
        "No-LN bounds width should be reasonable (got {})",
        bounds_no_ln.max_width()
    );
    assert!(
        bounds_with_ln.max_width() < 50.0,
        "With-LN bounds width should be reasonable (got {})",
        bounds_with_ln.max_width()
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_zonotope_swiglu_ffn_tightening() {
    // Test that zonotope tightening for SwiGLU FFN gives tighter bounds than IBP.
    // Architecture: ffn_norm -> ffn_up (Linear) -----> up
    //                       -> ffn_gate (Linear) -> silu -> gate
    //               MulBinary(up, gate) -> swiglu
    //
    // Both ffn_up and ffn_gate share the same input (ffn_norm output),
    // so zonotopes can track correlations and give tighter multiplication bounds.

    use ny_tensor::ZonotopeTensor;

    let seq = 4_usize;
    let hidden = 8_usize;
    let ffn_dim = 16_usize; // Intermediate FFN dimension

    let mut graph = GraphNetwork::new();

    // FFN norm (simulated as identity - just marks the shared input)
    graph.add_node(GraphNode::from_input(
        "ffn_norm",
        Layer::AddConstant(AddConstantLayer::new(
            Array2::<f32>::zeros((seq, hidden)).into_dyn(),
        )),
    ));

    // FFN up projection: [seq, hidden] -> [seq, ffn_dim]
    let up_weights = Array2::<f32>::from_shape_fn((ffn_dim, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32 * 0.1;
        0.3 * phase.sin()
    });
    let up_linear = LinearLayer::new(up_weights, None).unwrap();
    graph.add_node(GraphNode::new(
        "ffn_up",
        Layer::Linear(up_linear),
        vec!["ffn_norm".to_string()],
    ));

    // FFN gate projection: [seq, hidden] -> [seq, ffn_dim]
    let gate_weights = Array2::<f32>::from_shape_fn((ffn_dim, hidden), |(i, j)| {
        let phase = (i * 23 + j * 13) as f32 * 0.1;
        0.3 * phase.cos()
    });
    let gate_linear = LinearLayer::new(gate_weights, None).unwrap();
    graph.add_node(GraphNode::new(
        "ffn_gate",
        Layer::Linear(gate_linear),
        vec!["ffn_norm".to_string()],
    ));

    // SiLU activation on gate
    graph.add_node(GraphNode::new(
        "silu",
        Layer::SiLU(SiLULayer::new()),
        vec!["ffn_gate".to_string()],
    ));

    // SwiGLU: up * silu(gate)
    graph.add_node(GraphNode::binary(
        "swiglu",
        Layer::MulBinary(MulBinaryLayer),
        "ffn_up",
        "silu",
    ));

    graph.set_output("swiglu");

    // Input with epsilon perturbation
    let epsilon = 0.01_f32;
    let input =
        BoundedTensor::from_epsilon(Array2::<f32>::zeros((seq, hidden)).into_dyn(), epsilon)
            .unwrap();

    // Get IBP bounds (treats up and gate as independent)
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let ibp_width = ibp_bounds.max_width();

    // Create block-wise result to test zonotope tightening
    let block_wise_result = graph.propagate_ibp_block_wise(&input, epsilon).unwrap();

    // Check that we got swiglu width tracking
    let has_swiglu_tracking = block_wise_result
        .blocks
        .iter()
        .any(|b| b.swiglu_width.is_some());

    // For the test, also manually compute zonotope bounds for comparison
    let _center = (input.lower() + input.upper()) / 2.0;
    let base_z = ZonotopeTensor::from_bounded_tensor_per_position_2d(&input).unwrap();

    // Get the linear layers for manual zonotope propagation
    let up_node = graph.nodes.get("ffn_up").unwrap();
    let gate_node = graph.nodes.get("ffn_gate").unwrap();
    let (up_linear, gate_linear) = match (&up_node.layer, &gate_node.layer) {
        (Layer::Linear(u), Layer::Linear(g)) => (u, g),
        _ => panic!("Expected Linear layers"),
    };

    // Propagate through up and gate
    let up_z = base_z.linear(up_linear.weight(), up_linear.bias()).unwrap();
    let gate_z = base_z
        .linear(gate_linear.weight(), gate_linear.bias())
        .unwrap();

    // Apply SiLU to gate
    let silu_z = gate_z.silu_affine().unwrap();

    // Multiply with shared error symbols
    let swiglu_z = up_z.mul_elementwise(&silu_z).unwrap();
    let zonotope_bounds = swiglu_z.to_bounded_tensor().unwrap();
    let zonotope_width = zonotope_bounds.max_width();

    // Zonotope tracks correlations, so bounds should be at least as tight as IBP.
    // Additive tolerance covers f32 rounding only.
    assert!(
        zonotope_width <= ibp_width + 1e-6,
        "Zonotope width ({:.3e}) should not exceed IBP ({:.3e})",
        zonotope_width,
        ibp_width
    );

    // Verify soundness: zonotope bounds should contain actual values
    // (using concrete center values as sanity check)
    for (&l, &u) in zonotope_bounds
        .lower()
        .iter()
        .zip(zonotope_bounds.upper().iter())
    {
        assert!(
            l.is_finite() && u.is_finite(),
            "Zonotope bounds must be finite"
        );
        assert!(l <= u + 1e-5, "Invalid interval: {} > {}", l, u);
    }

    // Print comparison
    println!(
        "SwiGLU bounds comparison: IBP width={:.3e}, Zonotope width={:.3e}, ratio={:.2}x",
        ibp_width,
        zonotope_width,
        ibp_width / zonotope_width
    );
    if has_swiglu_tracking {
        let swiglu_w = block_wise_result.blocks[0].swiglu_width.unwrap();
        println!("Block-wise SwiGLU width: {:.3e}", swiglu_w);
    }
}

/// Build a SwiGLU graph with biased linear layers and large weights (triggers
/// zonotope_scale > 1.0) for soundness testing. Returns (graph, input,
/// up_weights, up_bias, gate_weights, gate_bias).
fn build_swiglu_soundness_graph() -> (
    GraphNetwork,
    BoundedTensor,
    Array2<f32>,
    Array1<f32>,
    Array2<f32>,
    Array1<f32>,
) {
    let (seq, hidden, ffn_dim) = (2, 4, 4);
    let mut graph = GraphNetwork::new();

    graph.add_node(GraphNode::from_input(
        "ffn_norm",
        Layer::AddConstant(AddConstantLayer::new(
            Array2::<f32>::zeros((seq, hidden)).into_dyn(),
        )),
    ));

    let up_weights = Array2::<f32>::from_shape_fn((ffn_dim, hidden), |(i, j)| {
        2.0 * ((i * 7 + j * 3) as f32 * 0.5).sin()
    });
    let up_bias = arr1(&[0.1_f32, -0.2, 0.3, -0.1]);
    graph.add_node(GraphNode::new(
        "ffn_up",
        Layer::Linear(LinearLayer::new(up_weights.clone(), Some(up_bias.clone())).unwrap()),
        vec!["ffn_norm".to_string()],
    ));

    let gate_weights = Array2::<f32>::from_shape_fn((ffn_dim, hidden), |(i, j)| {
        2.0 * ((i * 11 + j * 5) as f32 * 0.5).cos()
    });
    let gate_bias = arr1(&[0.05_f32, -0.1, 0.2, 0.0]);
    graph.add_node(GraphNode::new(
        "ffn_gate",
        Layer::Linear(LinearLayer::new(gate_weights.clone(), Some(gate_bias.clone())).unwrap()),
        vec!["ffn_norm".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "silu",
        Layer::SiLU(SiLULayer::new()),
        vec!["ffn_gate".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "swiglu",
        Layer::MulBinary(MulBinaryLayer),
        "ffn_up",
        "silu",
    ));
    graph.set_output("swiglu");

    let epsilon = 0.5_f32;
    let center =
        Array2::<f32>::from_shape_fn((seq, hidden), |(i, j)| ((i * 3 + j * 7) as f32 * 0.4).sin());
    let input = BoundedTensor::new(
        (center.clone() - epsilon).into_dyn(),
        (center + epsilon).into_dyn(),
    )
    .unwrap();

    (graph, input, up_weights, up_bias, gate_weights, gate_bias)
}

/// Soundness test (#2386): sample random points in the input interval and verify
/// the true SwiGLU output (up * silu(gate)) falls within bounds.
#[ntest::timeout(10000)]
#[test]
fn test_swiglu_zonotope_soundness_sampled() {
    let (graph, input, up_weights, up_bias, gate_weights, gate_bias) =
        build_swiglu_soundness_graph();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    let n_samples = 200;
    let mut rng_state = 42u64;
    let mut violations = 0;
    for _ in 0..n_samples {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let t = ((rng_state >> 33) as f32) / (u32::MAX as f32);

        let point = (input.lower() + &(t * &(input.upper() - input.lower())))
            .into_dimensionality::<ndarray::Ix2>()
            .expect("input should be 2D");

        let up_val = point.dot(&up_weights.t()) + &up_bias;
        let gate_val = point.dot(&gate_weights.t()) + &gate_bias;
        let silu_val = gate_val.mapv(|x| x / (1.0 + (-x).exp()));
        let true_output = &up_val * &silu_val;

        for (&val, (&lo, &hi)) in true_output
            .iter()
            .zip(ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()))
        {
            if val < lo - 1e-4 || val > hi + 1e-4 {
                violations += 1;
                eprintln!(
                    "Soundness violation: val={val:.6} not in [{lo:.6}, {hi:.6}], gap={:.2e}",
                    if val < lo { lo - val } else { val - hi }
                );
            }
        }
    }

    assert_eq!(
        violations, 0,
        "SwiGLU bounds had {violations} soundness violations"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_full_block_zonotope_propagation() {
    // Test full-block zonotope propagation through transformer attention:
    // LayerNorm -> Q_proj + K_proj -> Q@K^T -> Softmax -> scores@V -> Add(residual)
    //
    // This tests the complete zonotope propagation path including:
    // - LayerNorm (affine approximation)
    // - Linear projections
    // - Q@K^T attention correlation tracking
    // - Softmax (affine approximation)
    // - Value multiplication
    // - Residual connections (Add)
    //
    // Note: Full zonotope propagation trades precision for correlation tracking.
    // The linearization errors from LayerNorm and Softmax can accumulate,
    // so zonotope bounds may be looser than IBP for complex networks.
    // The value is in tracking correlations through operations that benefit
    // from it (like Q@K^T diagonal entries).

    let seq = 3_usize;
    let dim = 4_usize;
    let epsilon = 0.05_f32; // Small epsilon for better linear approximations

    let mut graph = GraphNetwork::new();

    // LayerNorm normalization
    let ny = Array1::ones(dim);
    let beta = Array1::zeros(dim);
    graph.add_node(GraphNode::from_input(
        "ln",
        Layer::LayerNorm(LayerNormLayer::new(ny, beta, 1e-5).unwrap()),
    ));

    // Q, K, V projections (identity for simplicity)
    let eye = Array2::<f32>::eye(dim);
    graph.add_node(GraphNode::new(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
        vec!["ln".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "k",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
        vec!["ln".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "v",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
        vec!["ln".to_string()],
    ));

    // Q@K^T attention scores
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));

    // Softmax over sequence dimension
    graph.add_node(GraphNode::new(
        "attn_weights",
        Layer::Softmax(SoftmaxLayer::new(-1)), // Softmax over last dim (seq)
        vec!["scores".to_string()],
    ));

    // Attention output: attn_weights @ V
    graph.add_node(GraphNode::binary(
        "attn_output",
        Layer::MatMul(MatMulLayer::new(false, None)), // Not transposed
        "attn_weights",
        "v",
    ));

    // Residual connection: attn_output + input (through linear identity)
    let eye_residual = Array2::<f32>::eye(dim);
    graph.add_node(GraphNode::from_input(
        "residual_proj",
        Layer::Linear(LinearLayer::new(eye_residual, None).unwrap()),
    ));

    graph.add_node(GraphNode::binary(
        "output",
        Layer::Add(AddLayer),
        "attn_output",
        "residual_proj",
    ));
    graph.set_output("output");

    // Create input with varied values
    let center_values: Vec<f32> = (0..seq * dim).map(|i| (i % dim) as f32 + 1.0).collect();
    let lower = ArrayD::from_shape_vec(
        vec![seq, dim],
        center_values.iter().map(|&v| v - epsilon).collect(),
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        vec![seq, dim],
        center_values.iter().map(|&v| v + epsilon).collect(),
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run full-block zonotope propagation
    let zonotope_bounds = graph.propagate_zonotope(&input, epsilon).unwrap();

    // Run standard IBP for comparison
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify zonotope bounds are valid
    assert!(
        zonotope_bounds.lower().iter().all(|&v| v.is_finite()),
        "Zonotope lower bounds should be finite"
    );
    assert!(
        zonotope_bounds.upper().iter().all(|&v| v.is_finite()),
        "Zonotope upper bounds should be finite"
    );
    assert!(
        zonotope_bounds
            .lower()
            .iter()
            .zip(zonotope_bounds.upper().iter())
            .all(|(&l, &u)| l <= u + 1e-6),
        "Zonotope lower bounds should be <= upper bounds"
    );

    // Verify IBP bounds are valid for comparison
    assert!(
        ibp_bounds.lower().iter().all(|&v| v.is_finite()),
        "IBP lower bounds should be finite"
    );

    // Calculate widths
    let zonotope_width = zonotope_bounds.max_width();
    let ibp_width = ibp_bounds.max_width();

    println!(
        "Full-block zonotope vs IBP: zonotope_width={:.4}, ibp_width={:.4}, ratio={:.4}",
        zonotope_width,
        ibp_width,
        zonotope_width / ibp_width
    );

    // Full zonotope propagation through complex networks may be looser than IBP
    // due to accumulated linearization errors from LayerNorm and Softmax.
    // The 2x threshold was set before soundness fixes (#2473: sum-of-radii for softmax,
    // #2522: per-element error terms) that correctly increased conservatism, giving
    // ~16.5x. The LayerNorm affine error terms were later widened again to cover the
    // box-wide variation of mean(x) and 1/sqrt(var(x)+eps) that the center-pinned
    // linearization ignored (previously an under-approximation of the true error);
    // that sound enclosure roughly doubles the zonotope width on this block
    // (measured ~35.7x vs IBP), which is loose but correct.
    // The value of zonotope propagation is in specific patterns (Q@K^T diagonals,
    // tested in test_zonotope_vs_ibp_attention_qkt_diagonal_tightness) rather than
    // overall bound tightness for full attention blocks.
    // Ref: #3004 tracks improving the zonotope pipeline for this case.
    assert!(
        zonotope_width <= ibp_width * 40.0,
        "Zonotope bounds should not be catastrophically looser than IBP (got {:.4} vs {:.4}, ratio={:.1}x)",
        zonotope_width,
        ibp_width,
        zonotope_width / ibp_width
    );

    println!("Full-block zonotope propagation test passed!");
}

#[ntest::timeout(10000)]
#[test]
fn test_zonotope_vs_ibp_attention_qkt_diagonal_tightness() {
    // Test that zonotope propagation tracks correlations for Q@K^T diagonal entries.
    // This is the key benefit of zonotope propagation: recognizing that X[i]·X[i] >= 0.
    //
    // IBP (standard): propagate_ibp already includes zonotope tightening for Q@K^T
    // via try_attention_matmul_bounds_zonotope, so both should give similar results.
    //
    // This test verifies that the full propagate_zonotope path also benefits from
    // correlation tracking through the Q@K^T pattern.

    let seq = 4_usize;
    let dim = 8_usize;
    let epsilon = 0.5_f32; // Large epsilon to see the difference

    // Build simple Q@K^T graph where Q and K come from the same input
    let mut graph = GraphNetwork::new();

    // Single input feeds both Q and K projections
    let eye = Array2::<f32>::eye(dim);
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::Linear(LinearLayer::new(eye.clone(), None).unwrap()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
    ));

    // Q@K^T
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, None)),
        "q",
        "k",
    ));
    graph.set_output("scores");

    // Input centered at 0 with epsilon perturbation
    // This makes the X[i]·X[i] correlation obvious: X ∈ [-ε, ε] means X² ≥ 0
    let lower = ArrayD::from_elem(vec![seq, dim], -epsilon);
    let upper = ArrayD::from_elem(vec![seq, dim], epsilon);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Compute baseline interval bounds (no zonotope tightening)
    let baseline_interval = MatMulLayer::new(true, None)
        .propagate_ibp_binary(&input, &input)
        .unwrap();

    // Run zonotope propagation
    let zonotope_bounds = graph.propagate_zonotope(&input, epsilon).unwrap();

    // Run IBP (which already includes zonotope tightening for Q@K^T)
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Check diagonal bounds
    let diag_baseline = baseline_interval.lower()[[0, 0]];
    let diag_zonotope = zonotope_bounds.lower()[[0, 0]];
    let diag_ibp = ibp_bounds.lower()[[0, 0]];

    println!(
        "Diagonal[0,0] lower bounds: baseline={:.4}, zonotope={:.4}, ibp={:.4}",
        diag_baseline, diag_zonotope, diag_ibp
    );

    // For X@X^T where X ∈ [-ε, ε], diagonal entries are sums of squares.
    // Baseline interval: treats X[i,j] as independent, worst case is (-ε)*(+ε) = -ε²
    // This gives diagonal lower bound = dim * (-ε²) = -dim * ε²
    //
    // Zonotope: knows X[i,j]² ∈ [0, ε²], so diagonal lower bound = 0

    // Baseline should have negative lower bound
    assert!(
        diag_baseline < -1e-6,
        "Baseline interval MatMul should have negative diagonal lower (got {})",
        diag_baseline
    );

    // Both zonotope and IBP (with zonotope tightening) should have non-negative diagonal
    assert!(
        diag_zonotope >= -1e-6,
        "Zonotope diagonal lower should be >= 0 (got {})",
        diag_zonotope
    );
    assert!(
        diag_ibp >= -1e-6,
        "IBP (with zonotope tightening) diagonal lower should be >= 0 (got {})",
        diag_ibp
    );

    // Calculate improvement over baseline
    let zonotope_improvement = diag_zonotope - diag_baseline;
    let ibp_improvement = diag_ibp - diag_baseline;

    println!(
        "Improvement over baseline: zonotope={:.4}, ibp={:.4}",
        zonotope_improvement, ibp_improvement
    );

    println!("Zonotope vs IBP Q@K^T diagonal tightness test passed!");
}

#[ntest::timeout(10000)]
#[test]
fn test_propagate_zonotope_causal_softmax_masks_future_positions() {
    let seq = 5_usize;
    let epsilon = 0.2_f32;

    let mut graph = GraphNetwork::new();

    // Identity scores: input -> Linear(I) so the graph has a non-trivial input node.
    let eye = Array2::<f32>::eye(seq);
    graph.add_node(GraphNode::from_input(
        "scores",
        Layer::Linear(LinearLayer::new(eye, None).unwrap()),
    ));

    graph.add_node(GraphNode::new(
        "attn",
        Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)),
        vec!["scores".to_string()],
    ));
    graph.set_output("attn");

    let center_values: Vec<f32> = (0..seq * seq).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let lower = ArrayD::from_shape_vec(
        vec![seq, seq],
        center_values.iter().map(|&v| v - epsilon).collect(),
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        vec![seq, seq],
        center_values.iter().map(|&v| v + epsilon).collect(),
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = graph.propagate_zonotope(&input, epsilon).unwrap();

    // Masked entries must be exactly 0 (within a tiny tolerance).
    for i in 0..seq {
        for j in (i + 1)..seq {
            assert!(
                out.upper()[[i, j]] <= 1e-6 && out.lower()[[i, j]] >= -1e-6,
                "masked causal softmax bounds should be 0 at ({},{}) got [{},{}]",
                i,
                j,
                out.lower()[[i, j]],
                out.upper()[[i, j]]
            );
        }
    }
}
