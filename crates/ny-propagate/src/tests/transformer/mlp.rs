// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! MLP-style transformer tests.

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_mlp_style_crown_vs_ibp_1d() {
    // Test CROWN vs IBP on an MLP-style network with 1D input.
    // This simulates a single position of a transformer MLP.
    //
    // MLP structure: LayerNorm → Linear (expand 4x) → GELU → Linear (contract)
    //
    // Key insight: For transformers, the MLP operates independently per position.
    // If CROWN works on 1D, we can potentially apply it per-position.

    let hidden_dim = 8_usize; // Small for test
    let intermediate_dim = hidden_dim * 4; // 4x expansion like transformers

    // Create weights
    // Linear1: hidden_dim -> intermediate_dim (expand)
    let mut w1 = Array2::<f32>::zeros((intermediate_dim, hidden_dim));
    for i in 0..intermediate_dim {
        for j in 0..hidden_dim {
            // Kaiming-style initialization
            w1[[i, j]] = ((i * 13 + j * 7) % 17) as f32 / 17.0 * 0.2 - 0.1;
        }
    }
    let b1 = Array1::<f32>::zeros(intermediate_dim);

    // Linear2: intermediate_dim -> hidden_dim (contract)
    let mut w2 = Array2::<f32>::zeros((hidden_dim, intermediate_dim));
    for i in 0..hidden_dim {
        for j in 0..intermediate_dim {
            w2[[i, j]] = ((i * 11 + j * 3) % 13) as f32 / 13.0 * 0.2 - 0.1;
        }
    }
    let b2 = Array1::<f32>::zeros(hidden_dim);

    // Build GraphNetwork: LayerNorm → Linear1 → GELU → Linear2
    let mut graph = GraphNetwork::new();

    // LayerNorm (identity scale, zero bias)
    let ny = Array1::ones(hidden_dim);
    let beta = Array1::zeros(hidden_dim);
    let ln = LayerNormLayer::new(ny, beta, 1e-5).unwrap();
    graph.add_node(GraphNode::from_input("layernorm", Layer::LayerNorm(ln)));

    // Linear1 (expand)
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(linear1),
        vec!["layernorm".to_string()],
    ));

    // GELU — use adaptive (non-sound) mode for this tightness test, since the
    // sound mode includes safety margins that can make CROWN wider than IBP.
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::adaptive(GeluApproximation::Erf)),
        vec!["linear1".to_string()],
    ));

    // Linear2 (contract)
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu".to_string()],
    ));

    graph.set_output("linear2");

    // Enable sampling mode for LayerNorm CROWN to allow bound propagation
    use crate::layers::LayerNormCrownMode;
    graph.set_layernorm_crown_mode(LayerNormCrownMode::Sampling);

    // Fair method comparison: disable the normalization → Linear L2/Cauchy-Schwarz
    // IBP lever for this whole test. That lever is, by deliberate design, a
    // *plain-IBP-only* tightening (`crate::l2_lever_gate`): it is gated OFF during
    // any CROWN bound pass because the CROWN backward relaxations never consume the
    // sphere and re-paying its cost per CROWN forward sweep was a 15×+ hang for zero
    // verification benefit. So on a normalization-fronted FFN the lever makes plain
    // IBP genuinely tighter than raw CROWN — which is real and intended (production
    // still benefits via the `min(CROWN, IBP)` intersection), but it is NOT a CROWN
    // regression and it is not what this CROWN-vs-IBP *method* test means to measure.
    // With the lever off, both sides are the box method and the historical invariant
    // "CROWN is at least as tight as the box IBP it refines" holds again.
    let _l2_lever_off = l2_lever_gate::L2LeverGuard::disabled();

    println!(
        "
=== MLP-style Network (1D): CROWN vs IBP ==="
    );
    println!(
        "Structure: LayerNorm[{}] → Linear[{}→{}] → GELU → Linear[{}→{}]",
        hidden_dim, hidden_dim, intermediate_dim, intermediate_dim, hidden_dim
    );

    // Test with small epsilon (tight input bounds)
    let center = Array1::from_elem(hidden_dim, 0.0_f32);
    let epsilon = 0.01_f32;
    let input = BoundedTensor::new(
        (center.clone() - epsilon).into_dyn(),
        (center.clone() + epsilon).into_dyn(),
    )
    .unwrap();

    println!(
        "
Input: center=0, epsilon={}",
        epsilon
    );

    // Get IBP bounds
    let ibp_result = graph.propagate_ibp(&input);
    let ibp_bounds = ibp_result.unwrap();
    let ibp_max_width = ibp_bounds.max_width();

    // Get CROWN bounds
    let crown_result = graph.propagate_crown(&input);
    let crown_bounds = crown_result.unwrap();
    let crown_max_width = crown_bounds.max_width();

    println!("IBP max width: {:.6}", ibp_max_width);
    println!("CROWN max width: {:.6}", crown_max_width);
    println!(
        "CROWN tightening ratio: {:.2}x",
        ibp_max_width / crown_max_width
    );

    // CROWN should be at least as tight as IBP
    assert!(
        crown_max_width <= ibp_max_width + 1e-6,
        "CROWN width {} should be <= IBP width {}",
        crown_max_width,
        ibp_max_width
    );

    // Test with larger epsilon (looser input bounds)
    let epsilon_large = 0.1_f32;
    let input_large = BoundedTensor::new(
        (center.clone() - epsilon_large).into_dyn(),
        (center + epsilon_large).into_dyn(),
    )
    .unwrap();

    println!(
        "
Input: center=0, epsilon={}",
        epsilon_large
    );

    let ibp_large = graph.propagate_ibp(&input_large).unwrap();
    let crown_large = graph.propagate_crown(&input_large).unwrap();

    let ibp_large_width = ibp_large.max_width();
    let crown_large_width = crown_large.max_width();

    println!("IBP max width: {:.6}", ibp_large_width);
    println!("CROWN max width: {:.6}", crown_large_width);
    println!(
        "CROWN tightening ratio: {:.2}x",
        ibp_large_width / crown_large_width
    );

    // Note: CROWN may not always be tighter than IBP for non-convex functions
    // like LayerNorm, due to linearization error. We check they're comparable.
    let ratio = crown_large_width / ibp_large_width;
    println!("CROWN/IBP ratio: {:.2}", ratio);
    // Allow CROWN to be up to 1.5x looser (due to LayerNorm linearization)
    assert!(
        ratio <= 1.5,
        "CROWN width {} should not be much worse than IBP width {} (ratio {:.2}x)",
        crown_large_width,
        ibp_large_width,
        ratio
    );

    // Test with realistic Whisper-like input width after attention
    // Input width ~1e4 (simulating post-attention bounds)
    let large_width_input = BoundedTensor::new(
        Array1::from_elem(hidden_dim, -5000.0_f32).into_dyn(),
        Array1::from_elem(hidden_dim, 5000.0_f32).into_dyn(),
    )
    .unwrap();

    println!(
        "
Input: width=10000 (simulating post-attention bounds)"
    );

    let ibp_post_attn = graph.propagate_ibp(&large_width_input).unwrap();
    let crown_post_attn = graph.propagate_crown(&large_width_input).unwrap();

    let ibp_post_attn_width = ibp_post_attn.max_width();
    let crown_post_attn_width = crown_post_attn.max_width();

    println!("IBP max width: {:.6e}", ibp_post_attn_width);
    println!("CROWN max width: {:.6e}", crown_post_attn_width);
    println!(
        "CROWN tightening ratio: {:.2}x",
        ibp_post_attn_width / crown_post_attn_width
    );

    // Note: Full soundness verification would require complete forward pass
    // through LayerNorm → Linear → GELU → Linear, which is complex.
    // The key results are captured in the width comparisons above.

    // Key result: measure how much CROWN helps vs IBP for MLP
    println!(
        "
=== Summary ==="
    );
    println!(
        "With epsilon=0.01: CROWN is {:.1}x tighter than IBP",
        ibp_max_width / crown_max_width
    );
    println!(
        "With epsilon=0.1:  CROWN is {:.1}x tighter than IBP",
        ibp_large_width / crown_large_width
    );
    println!(
        "With width=10000:  CROWN is {:.1}x tighter than IBP",
        ibp_post_attn_width / crown_post_attn_width
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mlp_crown_without_layernorm() {
    // Test MLP without LayerNorm to isolate the effect of Linear→GELU→Linear

    let hidden_dim = 16_usize;
    let intermediate_dim = hidden_dim * 4;

    // Create random-ish weights
    let mut w1 = Array2::<f32>::zeros((intermediate_dim, hidden_dim));
    for i in 0..intermediate_dim {
        for j in 0..hidden_dim {
            w1[[i, j]] = ((i * 13 + j * 7) % 17) as f32 / 17.0 * 0.3 - 0.15;
        }
    }
    let b1 = Array1::<f32>::zeros(intermediate_dim);

    let mut w2 = Array2::<f32>::zeros((hidden_dim, intermediate_dim));
    for i in 0..hidden_dim {
        for j in 0..intermediate_dim {
            w2[[i, j]] = ((i * 11 + j * 3) % 13) as f32 / 13.0 * 0.3 - 0.15;
        }
    }
    let b2 = Array1::<f32>::zeros(hidden_dim);

    // Build GraphNetwork: Linear1 → GELU → Linear2
    let mut graph = GraphNetwork::new();

    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["linear1".to_string()],
    ));

    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["gelu".to_string()],
    ));

    graph.set_output("linear2");

    println!(
        "
=== MLP without LayerNorm: CROWN vs IBP ==="
    );
    println!(
        "Structure: Linear[{}→{}] → GELU → Linear[{}→{}]",
        hidden_dim, intermediate_dim, intermediate_dim, hidden_dim
    );

    // Test with various epsilons
    for &epsilon in &[0.01_f32, 0.1, 1.0] {
        let center = Array1::from_elem(hidden_dim, 0.0_f32);
        let input = BoundedTensor::new(
            (center.clone() - epsilon).into_dyn(),
            (center + epsilon).into_dyn(),
        )
        .unwrap();

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();
        let crown_bounds = graph.propagate_crown(&input).unwrap();

        let ibp_width = ibp_bounds.max_width();
        let crown_width = crown_bounds.max_width();
        let ratio = ibp_width / crown_width;

        println!(
            "epsilon={:.2}: IBP={:.4}, CROWN={:.4}, ratio={:.2}x",
            epsilon, ibp_width, crown_width, ratio
        );

        // CROWN should be at least as tight
        assert!(
            crown_width <= ibp_width + 1e-5,
            "CROWN width {} should be <= IBP width {}",
            crown_width,
            ibp_width
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_network_propagate_crown_batched_transformer_scale() {
    // Test batched CROWN on transformer-scale input: [batch=1, seq=4, hidden=384]
    // This is a key test for verifying transformer verification works

    let mut network = Network::new();

    // MLP: hidden -> 4*hidden -> hidden (transformer MLP pattern)
    let hidden = 64; // Reduced from 384 for test speed, but tests same structure
    let expansion = 4;

    // Up projection: hidden -> 4*hidden
    // Use deterministic initialization: Xavier-like scaling with position-based variation
    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let weight1 = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
        let phase = (i * 17 + j * 31) as f32;
        scale1 * (phase.sin() * 0.5)
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));

    // GELU
    network.add_layer(Layer::GELU(GELULayer::default()));

    // Down projection: 4*hidden -> hidden
    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();
    let weight2 = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
        let phase = (i * 23 + j * 37) as f32;
        scale2 * (phase.cos() * 0.5)
    });
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    // Input: [batch=1, seq=4, hidden]
    let batch = 1;
    let seq = 4;
    let total_elements = batch * seq * hidden;
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[batch, seq, hidden]), vec![-0.1; total_elements]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[batch, seq, hidden]), vec![0.1; total_elements]).unwrap(),
    )
    .unwrap();

    // Run batched CROWN
    let batched_result = network.propagate_crown_batched(&input).unwrap();

    // Verify output shape matches input shape (MLP preserves shape)
    assert_eq!(batched_result.shape(), &[batch, seq, hidden]);

    // Verify all bounds are finite and valid
    let mut valid_count = 0;
    let mut finite_count = 0;
    for ((l, u), _) in batched_result
        .lower()
        .iter()
        .zip(batched_result.upper().iter())
        .zip(0..total_elements)
    {
        if l.is_finite() && u.is_finite() {
            finite_count += 1;
        }
        if *l <= *u {
            valid_count += 1;
        }
    }

    assert_eq!(finite_count, total_elements, "All bounds should be finite");
    assert_eq!(valid_count, total_elements, "All bounds should be valid");

    // Measure bound widths
    let avg_width: f32 = batched_result
        .lower()
        .iter()
        .zip(batched_result.upper().iter())
        .map(|(l, u)| u - l)
        .sum::<f32>()
        / total_elements as f32;

    println!(
        "Transformer MLP batched CROWN: shape {:?}, avg bound width: {:.4}",
        batched_result.shape(),
        avg_width
    );

    // Bounds should not explode (reasonable width for small perturbation)
    assert!(
        avg_width < 10.0,
        "Bound width should be reasonable (< 10), got {}",
        avg_width
    );
}
