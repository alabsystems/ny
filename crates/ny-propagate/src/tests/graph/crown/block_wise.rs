// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-block CROWN tests: tightness comparison and soundness sampling.
//!
//! Tests for `GraphNetwork::propagate_crown_block_wise()`.
//! Part of #3221.

use ndarray::{Array1, Array2, ArrayD, IxDyn};

use ny_tensor::BoundedTensor;

use crate::layers::binary_ops::AddLayer;
use crate::layers::linear::LinearLayer;
use crate::layers::normalization::layer_norm::LayerNormLayer;
use crate::*;

/// Build a 2-block FFN-only transformer graph for testing.
///
/// Each block: LayerNorm -> Linear_up -> GELU -> Linear_down -> Add(residual)
/// Node names follow the `layer<N>_<suffix>` convention for block detection.
fn build_two_block_ffn_graph(hidden: usize, expansion: usize) -> GraphNetwork {
    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();

    let mut graph = GraphNetwork::new();

    for block_idx in 0..2 {
        let prefix = format!("layer{}", block_idx);
        let block_input_name = if block_idx == 0 {
            NETWORK_INPUT.to_string()
        } else {
            format!("layer{}_add", block_idx - 1)
        };

        // LayerNorm
        let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();
        graph.add_node(GraphNode::new(
            format!("{}_norm", prefix),
            Layer::LayerNorm(ln),
            vec![block_input_name.clone()],
        ));

        // Linear up (hidden -> hidden*expansion)
        let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(i, j)| {
            let phase = (i * 17 + j * 31 + block_idx * 97) as f32;
            scale1 * phase.sin() * 0.15
        });
        let linear_up = LinearLayer::new(weight_up, None).unwrap();
        graph.add_node(GraphNode::new(
            format!("{}_ffn_up", prefix),
            Layer::Linear(linear_up),
            vec![format!("{}_norm", prefix)],
        ));

        // GELU activation
        let gelu = GELULayer::default();
        graph.add_node(GraphNode::new(
            format!("{}_ffn_act", prefix),
            Layer::GELU(gelu),
            vec![format!("{}_ffn_up", prefix)],
        ));

        // Linear down (hidden*expansion -> hidden)
        let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(i, j)| {
            let phase = (i * 23 + j * 37 + block_idx * 71) as f32;
            scale2 * phase.cos() * 0.15
        });
        let linear_down = LinearLayer::new(weight_down, None).unwrap();
        graph.add_node(GraphNode::new(
            format!("{}_ffn_down", prefix),
            Layer::Linear(linear_down),
            vec![format!("{}_ffn_act", prefix)],
        ));

        // Residual Add: block_input + ffn_down
        let add = AddLayer;
        graph.add_node(GraphNode::new(
            format!("{}_add", prefix),
            Layer::Add(add),
            vec![block_input_name, format!("{}_ffn_down", prefix)],
        ));
    }

    graph.set_output("layer1_add");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_per_block_crown_vs_ibp_two_blocks() {
    // Per-block CROWN should provide tighter bounds than IBP within
    // each block because CROWN exploits activation (GELU) correlations.
    //
    // Expected: crown_ibp_ratio < 1.0 for blocks with activations.
    // Part of #3221.
    //
    // Serialized against budget=0 tests to prevent CROWN fallback (#3515).
    tests::with_crown_dense_budget_mb("2048", || {
        let hidden = 4;
        let expansion = 2;
        let epsilon = 0.01_f32;

        let graph = build_two_block_ffn_graph(hidden, expansion);

        // Create input bounds
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();

        // Run per-block CROWN
        let result = graph.propagate_crown_block_wise(&input, epsilon).unwrap();

        println!("=== Per-Block CROWN vs IBP ===");
        println!(
            "{:<10} {:>12} {:>12} {:>12} {:>8}",
            "Block", "IBP width", "CROWN width", "Ratio", "Success"
        );
        println!("{}", "-".repeat(58));

        for block in &result.blocks {
            println!(
                "{:<10} {:>12.6} {:>12.6} {:>12.4} {:>8}",
                block.block_name,
                block.ibp_max_width,
                block.crown_max_width,
                block.crown_ibp_ratio,
                block.crown_successful,
            );
        }

        // Assertions
        assert_eq!(result.total_blocks, 2, "Expected 2 blocks");

        for block in &result.blocks {
            assert!(
                block.crown_successful,
                "CROWN should succeed for FFN-only block {}",
                block.block_name
            );
            assert!(
                block.ibp_max_width > 0.0,
                "IBP width should be positive for block {}",
                block.block_name
            );
            assert!(
                block.crown_max_width > 0.0,
                "CROWN width should be positive for block {}",
                block.block_name
            );
            // Per-block CROWN bounds should be comparable to IBP. With directed
            // rounding on normalization CROWN backward coefficients (#3344), CROWN
            // can be slightly wider than IBP for small models (n=4) where rounding
            // overhead dominates. The soundness test (test_per_block_crown_soundness_sampling)
            // verifies the bounds still contain all concrete evaluations.
            assert!(
                block.crown_ibp_ratio <= 1.15,
                "Per-block CROWN should not be significantly wider than IBP for block {} \
             (ratio={:.4}, IBP={:.6}, CROWN={:.6})",
                block.block_name,
                block.crown_ibp_ratio,
                block.ibp_max_width,
                block.crown_max_width,
            );
        }

        // For at least one block, CROWN should be at least as tight as IBP.
        let any_comparable = result.blocks.iter().any(|b| b.crown_ibp_ratio < 1.0);
        assert!(
            any_comparable,
            "Per-block CROWN should be at least as tight as IBP for at least one block"
        );
    });
}

/// #3550 regression: zero CPU dense budget disables per-block CROWN and reuses
/// the already-computed IBP block widths instead of allocating batched identities.
#[ntest::timeout(10000)]
#[test]
fn test_per_block_crown_zero_budget_uses_ibp_width_3550() {
    tests::with_crown_dense_budget_mb("0", || {
        let epsilon = 0.01_f32;
        let graph = build_two_block_ffn_graph(4, 2);
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[4])), epsilon).unwrap();

        let result = graph.propagate_crown_block_wise(&input, epsilon).unwrap();

        assert_eq!(
            result.total_blocks, 2,
            "expected both blocks to be reported"
        );
        for block in &result.blocks {
            assert!(
                !block.crown_successful,
                "zero budget should disable per-block CROWN for {}",
                block.block_name
            );
            assert_eq!(
                block.crown_max_width, block.ibp_max_width,
                "failed block {} should reuse its IBP width",
                block.block_name
            );
            assert_eq!(
                block.crown_ibp_ratio, 1.0,
                "failed block {} should report a 1.0 IBP ratio",
                block.block_name
            );
        }
    });
}

/// #3550 regression: alpha per-block CROWN skips optimization when the fixed
/// batched backward path is disabled by the shared CPU dense budget.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_per_block_crown_zero_budget_skips_alpha_3550() {
    tests::with_crown_dense_budget_mb("0", || {
        let epsilon = 0.01_f32;
        let graph = build_two_block_ffn_graph(4, 2);
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[4])), epsilon).unwrap();

        let result = graph
            .propagate_alpha_crown_block_wise(&input, epsilon)
            .unwrap();

        assert_eq!(
            result.total_blocks, 2,
            "expected both blocks to be reported"
        );
        for block in &result.blocks {
            assert!(
                !block.crown_successful,
                "zero budget should disable alpha-CROWN's fixed baseline for {}",
                block.block_name
            );
            assert_eq!(
                block.crown_max_width, block.ibp_max_width,
                "failed alpha block {} should reuse its IBP width",
                block.block_name
            );
            assert!(
                block.alpha_crown_max_width.is_none(),
                "alpha optimization should be skipped for {}",
                block.block_name
            );
            assert!(
                block.alpha_crown_ibp_ratio.is_none(),
                "alpha ratio should be absent when optimization is skipped for {}",
                block.block_name
            );
        }
    });
}

#[ntest::timeout(60000)]
#[test]
fn test_phase3_whole_network_vs_per_block_crown() {
    // Phase 3 measurement: per-block CROWN's value proposition.
    // Whole-network CROWN through LayerNorm ≈ IBP (normalization kills correlation),
    // but per-block CROWN captures within-block GELU correlations (ratio ≈ 0.3).
    // Reference: designs/2026-03-03-per-block-crown-transformer-verification.md
    // Part of #3221.
    //
    // Serialized against budget=0 tests to prevent CROWN fallback (#3515).
    tests::with_crown_dense_budget_mb("2048", || {
        let graph = build_two_block_ffn_graph(4, 2);
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[4])), 0.01).unwrap();

        // 1. Whole-network IBP baseline, 2. Whole-network CROWN (through LayerNorm).
        let ibp_width = graph.propagate_ibp(&input).unwrap().max_width();
        let crown_width = graph.propagate_crown_batched(&input).unwrap().max_width();
        let whole_ratio = if ibp_width > f32::EPSILON {
            crown_width / ibp_width
        } else {
            1.0
        };

        // 3. Per-block CROWN: CROWN within blocks, IBP at boundaries.
        let block_result = graph.propagate_crown_block_wise(&input, 0.01).unwrap();
        let (ratio_sum, ratio_count) = block_result
            .blocks
            .iter()
            .filter(|b| b.crown_successful)
            .fold((0.0_f32, 0usize), |(s, c), b| {
                (s + b.crown_ibp_ratio, c + 1)
            });
        let avg_ratio = if ratio_count > 0 {
            ratio_sum / ratio_count as f32
        } else {
            1.0
        };

        println!("=== Phase 3: Whole-Network vs Per-Block CROWN ===");
        println!("End-to-end: IBP={ibp_width:.6}, CROWN={crown_width:.6}, ratio={whole_ratio:.4}");
        for b in &block_result.blocks {
            println!(
                "  {}: IBP={:.6}, CROWN={:.6}, ratio={:.4}",
                b.block_name, b.ibp_max_width, b.crown_max_width, b.crown_ibp_ratio
            );
        }
        println!("Avg per-block={avg_ratio:.4}, whole-net={whole_ratio:.4}");

        // Whole-network CROWN not dramatically tighter than IBP.
        // Lowered from 0.5 to 0.3: LayerNorm IbpValidated decomposition (#3972)
        // improved CROWN tightness through normalization layers.
        assert!(
            whole_ratio >= 0.3,
            "Whole-net ratio={whole_ratio:.4}: firewall violated"
        );
        // Per-block CROWN provides bounds comparable to IBP. With directed rounding
        // on normalization CROWN backward coefficients (#3344), the per-block ratio
        // increased from ~0.47 to ~1.04 for small models (n=4) where rounding
        // overhead dominates. The soundness test still passes — bounds contain all
        // concrete evaluations. For larger models, CROWN's advantage re-emerges
        // because rounding overhead is proportionally smaller.
        assert!(
            avg_ratio < 1.15,
            "Per-block ratio={avg_ratio:.4}, expected < 1.15"
        );
        // Per-block should be comparable to or better than whole-network.
        // Whole-network CROWN through LayerNorm ≈ IBP, so per-block (which avoids
        // cross-normalization composition) should not be significantly worse.
        assert!(
            avg_ratio < whole_ratio + 0.1,
            "Per-block ({avg_ratio:.4}) should not be much worse than whole-net ({whole_ratio:.4})"
        );
    });
}

#[ntest::timeout(60000)]
#[test]
fn test_per_block_crown_soundness_sampling() {
    // Soundness test: per-block CROWN bounds must contain concrete evaluations.
    //
    // For each block, creates a fresh epsilon-ball as block input, gets CROWN
    // bounds, then evaluates concrete points (corners + random) through the
    // block's layers via IBP with degenerate (point) intervals. Verifies that
    // every concrete output falls within the CROWN lower/upper bounds.
    //
    // Part of #3221.
    //
    // Serialized against budget=0 tests to prevent CROWN fallback (#3515).
    tests::with_crown_dense_budget_mb("2048", || {
        let hidden = 4;
        let expansion = 2;
        let epsilon = 0.05_f32;
        let n_random_samples = 20;

        let graph = build_two_block_ffn_graph(hidden, expansion);

        let exec_order = graph.exec_order().unwrap();
        let block_nodes_map = GraphNetwork::collect_block_nodes(exec_order);
        assert!(!block_nodes_map.is_empty(), "Should detect blocks");

        for nodes_in_block in block_nodes_map.values() {
            // Fresh epsilon-ball as block input (same as propagate_crown_block_wise).
            let block_input =
                BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();

            // Get IBP bounds at each node within the block.
            let block_node_bounds = graph
                .collect_block_ibp_bounds(nodes_in_block, &block_input)
                .unwrap();

            // Get CROWN bounds for this block.
            let (crown_bounds, _stats, provenance) = graph
                .crown_backward_within_block(nodes_in_block, &block_node_bounds, &block_input)
                .unwrap();
            // Regression guard: silent fallback would widen bounds without warning (#4256).
            assert_eq!(
                provenance,
                BoundsProvenance::Crown,
                "block-wise CROWN on FFN block must not fall back to forward bounds"
            );

            let crown_lower = crown_bounds.lower();
            let crown_upper = crown_bounds.upper();
            let output_dim = crown_lower.len();

            // Generate concrete sample points within the epsilon-ball.
            // Corners (2^hidden = 16 for hidden=4) plus random interior points.
            let mut sample_points: Vec<ArrayD<f32>> = Vec::new();

            // Corner points: all combinations of ±epsilon.
            let n_corners = 1_usize << hidden;
            for corner_idx in 0..n_corners {
                let point: Vec<f32> = (0..hidden)
                    .map(|d| {
                        if corner_idx & (1 << d) != 0 {
                            epsilon
                        } else {
                            -epsilon
                        }
                    })
                    .collect();
                sample_points.push(ArrayD::from_shape_vec(IxDyn(&[hidden]), point).unwrap());
            }

            // Center point.
            sample_points.push(ArrayD::zeros(IxDyn(&[hidden])));

            // Random interior points using deterministic pseudo-random.
            for s in 0..n_random_samples {
                let point: Vec<f32> = (0..hidden)
                    .map(|d| {
                        // Deterministic hash-based pseudo-random in [-epsilon, epsilon].
                        let hash = ((s * 7919 + d * 104729 + 31) % 10000) as f32 / 10000.0;
                        (hash * 2.0 - 1.0) * epsilon
                    })
                    .collect();
                sample_points.push(ArrayD::from_shape_vec(IxDyn(&[hidden]), point).unwrap());
            }

            // Evaluate each concrete point through the block and check containment.
            for (sample_idx, point) in sample_points.iter().enumerate() {
                // Create degenerate BoundedTensor (lower = upper = point).
                let point_bt = BoundedTensor::new(point.clone(), point.clone()).unwrap();

                // Evaluate through the block's layers using IBP (degenerate = exact).
                let concrete_bounds = graph
                    .collect_block_ibp_bounds(nodes_in_block, &point_bt)
                    .unwrap();
                let last_node = nodes_in_block.last().unwrap();
                let concrete_output = concrete_bounds.get(last_node).unwrap();
                let concrete_vals = concrete_output.lower(); // lower == upper for degenerate

                // Check containment: concrete_vals[d] ∈ [crown_lower[d], crown_upper[d]].
                for d in 0..output_dim {
                    let val = concrete_vals[[d]];
                    let lo = crown_lower[[d]];
                    let hi = crown_upper[[d]];
                    assert!(
                        val >= lo - 1e-6 && val <= hi + 1e-6,
                        "CROWN soundness violation: sample {} dim {}: \
                     val={:.8} not in [{:.8}, {:.8}] (gap: lo={:.2e}, hi={:.2e})",
                        sample_idx,
                        d,
                        val,
                        lo,
                        hi,
                        lo - val,
                        val - hi,
                    );
                }
            }
        }
    });
}

#[ntest::timeout(60000)]
#[test]
fn test_phase4_alpha_crown_per_block() {
    // Phase 4: alpha-CROWN per block with optimizable GELU slopes.
    // Alpha-CROWN should produce bounds at least as tight as fixed CROWN (alpha=0.5).
    // Reference: designs/2026-03-03-per-block-crown-transformer-verification.md Phase 4.
    // Part of #3221.
    //
    // Serialized against budget=0 tests to prevent CROWN fallback (#3515).
    tests::with_crown_dense_budget_mb("2048", || {
        let hidden = 4;
        let expansion = 2;
        let epsilon = 0.01_f32;

        let graph = build_two_block_ffn_graph(hidden, expansion);
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();

        let fixed_result = graph.propagate_crown_block_wise(&input, epsilon).unwrap();
        let alpha_result = graph
            .propagate_alpha_crown_block_wise(&input, epsilon)
            .unwrap();

        println!("=== Phase 4: Alpha-CROWN Per Block ===");
        assert_eq!(
            fixed_result.total_blocks, alpha_result.total_blocks,
            "Block count mismatch"
        );

        for (fb, ab) in fixed_result.blocks.iter().zip(alpha_result.blocks.iter()) {
            let aw = ab.alpha_crown_max_width.unwrap_or(fb.crown_max_width);
            let improvement = if fb.crown_max_width > f32::EPSILON {
                (1.0 - aw / fb.crown_max_width) * 100.0
            } else {
                0.0
            };
            println!(
                "{:<10} IBP={:.6} CROWN={:.6} alpha={:.6} ({:.1}%)",
                ab.block_name, ab.ibp_max_width, fb.crown_max_width, aw, improvement,
            );
            assert!(
                ab.crown_successful,
                "Alpha-CROWN failed for {}",
                ab.block_name
            );
            if let Some(a) = ab.alpha_crown_max_width {
                assert!(
                    a <= fb.crown_max_width + 1e-6,
                    "Alpha worse for {}",
                    ab.block_name
                );
                assert!(a.is_finite(), "Alpha non-finite for {}", ab.block_name);
            }
            if let Some(ar) = ab.alpha_crown_ibp_ratio {
                assert!(
                    ar <= ab.crown_ibp_ratio + 1e-6,
                    "Alpha ratio worse for {}",
                    ab.block_name
                );
            }
        }
    });
}

#[ntest::timeout(60000)]
#[test]
fn test_phase4_alpha_crown_wider_epsilon() {
    // Alpha-CROWN optimization within a single FFN block (no LayerNorm).
    // The block structure is Linear_up → GELU → Linear_down, with weights
    // scaled so GELU inputs land in the nonlinear convex region [-√2, √2].
    //
    // Without LayerNorm, the GELU pre-activation bounds are directly controlled
    // by epsilon and the weight matrix. This isolates alpha's effect.
    //
    // Reference: In alpha-beta-CROWN, per-neuron alpha optimization produces
    // 2-15% tighter bounds on GELU/SiLU networks (vnncomp benchmarks).
    // Part of #3221 Phase 4.
    //
    // Serialized against budget=0 tests to prevent CROWN fallback (#3515).
    tests::with_crown_dense_budget_mb("2048", || {
        let hidden = 8;
        let expansion = 4;
        let epsilon = 0.3_f32;

        // Build a single-block graph: Linear_up → GELU → Linear_down.
        // Use layerN_ prefix for block detection.
        let mut graph = GraphNetwork::new();

        // Scale weights so GELU inputs are in [-1.0, 1.0] ⊂ [-√2, √2].
        // With hidden=8 inputs at ±0.3 and weight scale ~0.15, max GELU input
        // ≈ 8 * 0.3 * 0.15 = 0.36, well within the convex region.
        let weight_up = Array2::from_shape_fn((expansion * hidden, hidden), |(i, j)| {
            let phase = (i * 13 + j * 29) as f32;
            0.15 * phase.sin()
        });
        let bias_up: Array1<f32> = (0..expansion * hidden)
            .map(|i| 0.3 * ((i * 7) as f32).sin())
            .collect();
        let linear_up = LinearLayer::new(weight_up, Some(bias_up)).unwrap();
        graph.add_node(GraphNode::new(
            "layer0_ffn_up".to_string(),
            Layer::Linear(linear_up),
            vec![NETWORK_INPUT.to_string()],
        ));

        let gelu = GELULayer::default();
        graph.add_node(GraphNode::new(
            "layer0_ffn_act".to_string(),
            Layer::GELU(gelu),
            vec!["layer0_ffn_up".to_string()],
        ));

        let weight_down = Array2::from_shape_fn((hidden, expansion * hidden), |(i, j)| {
            let phase = (i * 19 + j * 41) as f32;
            0.15 * phase.cos()
        });
        let linear_down = LinearLayer::new(weight_down, None).unwrap();
        graph.add_node(GraphNode::new(
            "layer0_ffn_down".to_string(),
            Layer::Linear(linear_down),
            vec!["layer0_ffn_act".to_string()],
        ));

        // Add residual for block detection.
        let add = AddLayer;
        graph.add_node(GraphNode::new(
            "layer0_add".to_string(),
            Layer::Add(add),
            vec![NETWORK_INPUT.to_string(), "layer0_ffn_down".to_string()],
        ));
        graph.set_output("layer0_add");

        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();

        let fixed_result = graph.propagate_crown_block_wise(&input, epsilon).unwrap();
        let alpha_result = graph
            .propagate_alpha_crown_block_wise(&input, epsilon)
            .unwrap();

        println!(
            "=== Phase 4: Alpha-CROWN Single FFN Block (eps={}) ===",
            epsilon
        );
        println!(
            "{:<10} {:>10} {:>12} {:>12} {:>8}",
            "Block", "IBP", "CROWN(0.5)", "alpha-best", "Improv%"
        );
        println!("{}", "-".repeat(56));

        for (fb, ab) in fixed_result.blocks.iter().zip(alpha_result.blocks.iter()) {
            let aw = ab.alpha_crown_max_width.unwrap_or(fb.crown_max_width);
            let improvement = if fb.crown_max_width > f32::EPSILON {
                (1.0 - aw / fb.crown_max_width) * 100.0
            } else {
                0.0
            };
            println!(
                "{:<10} {:>10.4} {:>12.6} {:>12.6} {:>7.1}%",
                ab.block_name, ab.ibp_max_width, fb.crown_max_width, aw, improvement,
            );

            assert!(
                ab.crown_successful,
                "Alpha-CROWN failed for {}",
                ab.block_name
            );
            if let Some(a) = ab.alpha_crown_max_width {
                // Alpha-CROWN must be at least as tight as fixed CROWN.
                assert!(
                    a <= fb.crown_max_width + 1e-6,
                    "Alpha worse for {}: alpha={:.6} > fixed={:.6}",
                    ab.block_name,
                    a,
                    fb.crown_max_width,
                );
                assert!(a.is_finite(), "Alpha non-finite for {}", ab.block_name);
            }
        }

        // Per-block CROWN should be tighter than IBP for this FFN block.
        for b in &fixed_result.blocks {
            assert!(
                b.crown_ibp_ratio < 0.95,
                "CROWN should tighten FFN bounds, ratio={:.4}",
                b.crown_ibp_ratio
            );
        }
    });
}

#[ntest::timeout(60000)]
#[test]
fn test_alpha_crown_executes_and_fixed_crown_is_sound_sampling() {
    // Coverage split:
    // - verify the alpha-CROWN block-wise API succeeds on each detected block
    // - verify sampled concrete evaluations stay within the fixed block-CROWN bounds
    //
    // This test does not inspect optimized alpha bounded tensors directly.
    // `propagate_alpha_crown_block_wise()` currently reports per-block width
    // summaries, not the optimized bounded tensors themselves, so the direct
    // soundness witness available at this layer remains `crown_backward_within_block`.
    //
    // Part of #3221 Phase 4.
    //
    // Serialized against budget=0 tests to prevent CROWN fallback (#3515).
    tests::with_crown_dense_budget_mb("2048", || {
        let hidden = 4;
        let expansion = 2;
        let epsilon = 0.05_f32;
        let n_random_samples = 20;

        let graph = build_two_block_ffn_graph(hidden, expansion);

        // Run alpha-CROWN to verify it succeeds.
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();
        let alpha_result = graph
            .propagate_alpha_crown_block_wise(&input, epsilon)
            .unwrap();
        assert!(!alpha_result.blocks.is_empty(), "Should detect blocks");
        for block in &alpha_result.blocks {
            assert!(
                block.crown_successful,
                "Alpha-CROWN should succeed for block {}",
                block.block_name
            );
        }

        // Get the internal block structure for direct evaluation.
        let exec_order = graph.exec_order().unwrap();
        let block_nodes_map = GraphNetwork::collect_block_nodes(exec_order);

        for nodes_in_block in block_nodes_map.values() {
            let block_input =
                BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();

            let block_node_bounds = graph
                .collect_block_ibp_bounds(nodes_in_block, &block_input)
                .unwrap();

            // Get fixed CROWN bounds for this block. Alpha-CROWN is guaranteed
            // to be at least as tight as fixed CROWN (alpha=0.5), so if fixed
            // CROWN is sound, alpha-CROWN is also sound. But we test fixed CROWN
            // containment directly — any violation here means CROWN itself is broken.
            let (crown_bounds, _stats, provenance) = graph
                .crown_backward_within_block(nodes_in_block, &block_node_bounds, &block_input)
                .unwrap();
            assert_eq!(
                provenance,
                BoundsProvenance::Crown,
                "block-wise CROWN on FFN block must not fall back to forward bounds"
            );

            let crown_lower = crown_bounds.lower();
            let crown_upper = crown_bounds.upper();
            let output_dim = crown_lower.len();

            // Generate concrete sample points.
            let mut sample_points: Vec<ArrayD<f32>> = Vec::new();

            let n_corners = 1_usize << hidden;
            for corner_idx in 0..n_corners {
                let point: Vec<f32> = (0..hidden)
                    .map(|d| {
                        if corner_idx & (1 << d) != 0 {
                            epsilon
                        } else {
                            -epsilon
                        }
                    })
                    .collect();
                sample_points.push(ArrayD::from_shape_vec(IxDyn(&[hidden]), point).unwrap());
            }
            sample_points.push(ArrayD::zeros(IxDyn(&[hidden])));

            for s in 0..n_random_samples {
                let point: Vec<f32> = (0..hidden)
                    .map(|d| {
                        let hash = ((s * 7919 + d * 104729 + 31) % 10000) as f32 / 10000.0;
                        (hash * 2.0 - 1.0) * epsilon
                    })
                    .collect();
                sample_points.push(ArrayD::from_shape_vec(IxDyn(&[hidden]), point).unwrap());
            }

            for (sample_idx, point) in sample_points.iter().enumerate() {
                let point_bt = BoundedTensor::new(point.clone(), point.clone()).unwrap();
                let concrete_bounds = graph
                    .collect_block_ibp_bounds(nodes_in_block, &point_bt)
                    .unwrap();
                let last_node = nodes_in_block.last().unwrap();
                let concrete_output = concrete_bounds.get(last_node).unwrap();
                let concrete_vals = concrete_output.lower();

                for d in 0..output_dim {
                    let val = concrete_vals[[d]];
                    let lo = crown_lower[[d]];
                    let hi = crown_upper[[d]];
                    assert!(
                        val >= lo - 1e-6 && val <= hi + 1e-6,
                        "Fixed block-CROWN soundness violation: sample {} dim {}: \
                     val={:.8} not in [{:.8}, {:.8}] (gap: lo={:.2e}, hi={:.2e})",
                        sample_idx,
                        d,
                        val,
                        lo,
                        hi,
                        lo - val,
                        val - hi,
                    );
                }
            }
        }
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_per_block_crown_soundness_larger_scale() {
    // Larger-scale soundness test: hidden=16, expansion=4 (compared to toy hidden=4).
    //
    // R1 flagged that all prior tests use hidden=4 which has only 2^4=16 corners.
    // At hidden=16 corner enumeration is 2^16=65536, so we use random sampling
    // with more samples to exercise higher-dimensional interactions.
    //
    // Part of #3221.
    //
    // Serialized against budget=0 tests to prevent CROWN fallback (#3515).
    tests::with_crown_dense_budget_mb("2048", || {
        let hidden = 16;
        let expansion = 4;
        let epsilon = 0.02_f32;
        let n_random_samples = 200;

        let graph = build_two_block_ffn_graph(hidden, expansion);

        let exec_order = graph.exec_order().unwrap();
        let block_nodes_map = GraphNetwork::collect_block_nodes(exec_order);
        assert!(!block_nodes_map.is_empty(), "Should detect blocks");

        for nodes_in_block in block_nodes_map.values() {
            let block_input =
                BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();

            let block_node_bounds = graph
                .collect_block_ibp_bounds(nodes_in_block, &block_input)
                .unwrap();

            let (crown_bounds, _stats, provenance) = graph
                .crown_backward_within_block(nodes_in_block, &block_node_bounds, &block_input)
                .unwrap();
            assert_eq!(
                provenance,
                BoundsProvenance::Crown,
                "block-wise CROWN on scaled FFN block must not fall back to forward bounds"
            );

            let crown_lower = crown_bounds.lower();
            let crown_upper = crown_bounds.upper();
            let output_dim = crown_lower.len();

            // Verify bounds are finite and non-degenerate.
            for d in 0..output_dim {
                assert!(
                    crown_lower[[d]].is_finite() && crown_upper[[d]].is_finite(),
                    "Non-finite CROWN bound at dim {d}: [{}, {}]",
                    crown_lower[[d]],
                    crown_upper[[d]],
                );
                assert!(
                    crown_upper[[d]] >= crown_lower[[d]] - 1e-6,
                    "Inverted CROWN bound at dim {d}: [{}, {}]",
                    crown_lower[[d]],
                    crown_upper[[d]],
                );
            }

            // Sample random points within the epsilon-ball.
            let mut sample_points: Vec<ArrayD<f32>> = Vec::new();

            // A few extreme corners (all +, all -, alternating).
            let all_pos: Vec<f32> = vec![epsilon; hidden];
            let all_neg: Vec<f32> = vec![-epsilon; hidden];
            let alternating: Vec<f32> = (0..hidden)
                .map(|d| if d % 2 == 0 { epsilon } else { -epsilon })
                .collect();
            sample_points.push(ArrayD::from_shape_vec(IxDyn(&[hidden]), all_pos).unwrap());
            sample_points.push(ArrayD::from_shape_vec(IxDyn(&[hidden]), all_neg).unwrap());
            sample_points.push(ArrayD::from_shape_vec(IxDyn(&[hidden]), alternating).unwrap());
            sample_points.push(ArrayD::zeros(IxDyn(&[hidden])));

            // Random interior points with deterministic pseudo-random.
            for s in 0..n_random_samples {
                let point: Vec<f32> = (0..hidden)
                    .map(|d| {
                        let hash = ((s * 7919 + d * 104729 + 31) % 10000) as f32 / 10000.0;
                        (hash * 2.0 - 1.0) * epsilon
                    })
                    .collect();
                sample_points.push(ArrayD::from_shape_vec(IxDyn(&[hidden]), point).unwrap());
            }

            // Evaluate each point through the block and check containment.
            for (sample_idx, point) in sample_points.iter().enumerate() {
                let point_bt = BoundedTensor::new(point.clone(), point.clone()).unwrap();
                let concrete_bounds = graph
                    .collect_block_ibp_bounds(nodes_in_block, &point_bt)
                    .unwrap();
                let last_node = nodes_in_block.last().unwrap();
                let concrete_output = concrete_bounds.get(last_node).unwrap();
                let concrete_vals = concrete_output.lower();

                for d in 0..output_dim {
                    let val = concrete_vals[[d]];
                    let lo = crown_lower[[d]];
                    let hi = crown_upper[[d]];
                    assert!(
                        val >= lo - 1e-6 && val <= hi + 1e-6,
                        "Larger-scale CROWN soundness violation: sample {} dim {}: \
                     val={:.8} not in [{:.8}, {:.8}] (gap: lo={:.2e}, hi={:.2e})",
                        sample_idx,
                        d,
                        val,
                        lo,
                        hi,
                        lo - val,
                        val - hi,
                    );
                }
            }
        }

        // Also verify that per-block CROWN produces meaningful tightening at this scale.
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();
        let result = graph.propagate_crown_block_wise(&input, epsilon).unwrap();
        for b in &result.blocks {
            assert!(
                b.crown_ibp_ratio < 0.95,
                "Per-block CROWN not tighter than IBP at hidden=16: ratio={:.4}",
                b.crown_ibp_ratio,
            );
        }
    });
}
