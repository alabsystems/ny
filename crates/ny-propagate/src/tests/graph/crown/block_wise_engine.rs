// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Engine-threading regressions for block-wise graph CROWN.

use std::collections::HashMap;

use ndarray::{Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::tests::crown::helpers::{assert_bounds_finite, CountingGemmEngine};

use crate::layers::binary_ops::AddLayer;
use crate::layers::linear::LinearLayer;
use crate::layers::normalization::layer_norm::LayerNormLayer;
use crate::network::BlockAlphaState;
use crate::*;

/// Build a 2-block FFN-only transformer graph for testing.
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

        let ln = LayerNormLayer::new_default(hidden, 1e-5).unwrap();
        graph.add_node(GraphNode::new(
            format!("{}_norm", prefix),
            Layer::LayerNorm(ln),
            vec![block_input_name.clone()],
        ));

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

        let gelu = GELULayer::default();
        graph.add_node(GraphNode::new(
            format!("{}_ffn_act", prefix),
            Layer::GELU(gelu),
            vec![format!("{}_ffn_up", prefix)],
        ));

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

fn assert_alpha_block_wise_results_match(
    engine_result: &BlockWiseCrownResult,
    baseline: &BlockWiseCrownResult,
) {
    assert_eq!(
        engine_result.total_blocks, baseline.total_blocks,
        "#3597 regression: engine-aware alpha block-wise CROWN changed block count"
    );
    for (engine_block, baseline_block) in engine_result.blocks.iter().zip(&baseline.blocks) {
        assert_eq!(
            engine_block.block_name, baseline_block.block_name,
            "#3597 regression: engine-aware alpha block-wise CROWN changed block order"
        );
        assert_eq!(
            engine_block.crown_successful, baseline_block.crown_successful,
            "#3597 regression: engine-aware alpha block-wise CROWN changed success flag for {}",
            engine_block.block_name
        );
        assert!(
            engine_block.crown_max_width.is_finite(),
            "#3597 regression: engine-aware alpha block-wise CROWN produced NaN/Inf fixed-slope width for {}",
            engine_block.block_name
        );
        assert!(
            (engine_block.crown_max_width - baseline_block.crown_max_width).abs() <= 1e-6,
            "#3597 regression: engine-aware alpha block-wise CROWN changed fixed-slope width for {} \
             (engine={}, baseline={})",
            engine_block.block_name,
            engine_block.crown_max_width,
            baseline_block.crown_max_width
        );

        match (
            engine_block.alpha_crown_max_width,
            baseline_block.alpha_crown_max_width,
        ) {
            (Some(engine_width), Some(baseline_width)) => {
                assert!(
                    engine_width.is_finite(),
                    "#3597 regression: engine-aware alpha block-wise CROWN produced NaN/Inf optimized width for {}",
                    engine_block.block_name
                );
                assert!(
                    (engine_width - baseline_width).abs() <= 1e-6,
                    "#3597 regression: engine-aware alpha block-wise CROWN changed optimized width for {} \
                     (engine={}, baseline={})",
                    engine_block.block_name,
                    engine_width,
                    baseline_width
                );
            }
            (None, None) => {}
            _ => panic!(
                "#3597 regression: engine-aware alpha block-wise CROWN changed alpha availability for {}",
                engine_block.block_name
            ),
        }
    }
}

#[test]
fn test_per_block_crown_threads_engine_3597() {
    // Serialized against budget=0 tests to prevent CROWN fallback (#3515).
    tests::with_crown_dense_budget_mb("2048", || {
        let hidden = 4;
        let expansion = 2;
        let epsilon = 0.01_f32;

        let graph = build_two_block_ffn_graph(hidden, expansion);
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();
        let baseline = graph.propagate_crown_block_wise(&input, epsilon).unwrap();

        let engine = CountingGemmEngine::new();
        let engine_result = graph
            .propagate_crown_block_wise_with_engine(&input, epsilon, Some(&engine))
            .unwrap();

        assert_eq!(
            engine_result.total_blocks, baseline.total_blocks,
            "#3597 regression: engine-aware per-block CROWN changed block count"
        );
        for (engine_block, baseline_block) in engine_result.blocks.iter().zip(&baseline.blocks) {
            assert_eq!(
                engine_block.block_name, baseline_block.block_name,
                "#3597 regression: engine-aware per-block CROWN changed block order"
            );
            assert_eq!(
                engine_block.crown_successful, baseline_block.crown_successful,
                "#3597 regression: engine-aware per-block CROWN changed success flag for {}",
                engine_block.block_name
            );
            assert!(
                engine_block.crown_max_width.is_finite(),
                "#3597 regression: engine-aware per-block CROWN produced NaN/Inf width for {}",
                engine_block.block_name
            );
            assert!(
                (engine_block.crown_max_width - baseline_block.crown_max_width).abs() <= 1e-6,
                "#3597 regression: engine-aware per-block CROWN changed width for {} \
                 (engine={}, baseline={})",
                engine_block.block_name,
                engine_block.crown_max_width,
                baseline_block.crown_max_width
            );
        }

        let calls = engine.gemm_calls();
        assert!(
            calls > 0,
            "#3597 regression: per-block CROWN should hit GemmEngine, got {calls} calls"
        );
    });
}

#[test]
fn test_alpha_block_crown_missing_gelu_alpha_threads_engine_3597() {
    // Serialized against budget=0 tests to prevent CROWN fallback (#3515).
    tests::with_crown_dense_budget_mb("2048", || {
        let hidden = 4;
        let expansion = 2;
        let epsilon = 0.01_f32;

        let graph = build_two_block_ffn_graph(hidden, expansion);
        let block_input =
            BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();
        let exec_order = graph.exec_order().unwrap();
        let block_nodes_map = GraphNetwork::collect_block_nodes(exec_order);
        let nodes_in_block = block_nodes_map
            .values()
            .next()
            .expect("test graph should contain at least one transformer block");
        let block_node_bounds = graph
            .collect_block_ibp_bounds(nodes_in_block, &block_input)
            .unwrap();
        let empty_alpha_state = BlockAlphaState {
            gelu_alphas: HashMap::new(),
        };

        let (baseline, _, baseline_prov) = graph
            .crown_backward_within_block_with_engine(
                nodes_in_block,
                &block_node_bounds,
                &block_input,
                None,
                Some(&empty_alpha_state),
                None,
            )
            .unwrap();
        assert_eq!(
            baseline_prov,
            BoundsProvenance::Crown,
            "baseline block-wise CROWN must not fall back to forward bounds"
        );

        let engine = CountingGemmEngine::new();
        let (engine_bounds, _, engine_prov) = graph
            .crown_backward_within_block_with_engine(
                nodes_in_block,
                &block_node_bounds,
                &block_input,
                Some(&engine),
                Some(&empty_alpha_state),
                None,
            )
            .unwrap();
        assert_eq!(
            engine_prov,
            BoundsProvenance::Crown,
            "engine-threaded block-wise CROWN must not fall back to forward bounds"
        );

        assert_bounds_finite(
            &engine_bounds,
            "#3597 missing-alpha GELU fallback block CROWN with engine output",
        );
        for (d, ((el, eu), (bl, bu))) in engine_bounds
            .lower()
            .iter()
            .zip(engine_bounds.upper().iter())
            .zip(baseline.lower().iter().zip(baseline.upper().iter()))
            .enumerate()
        {
            assert!(
                (el - bl).abs() <= 1e-6 && (eu - bu).abs() <= 1e-6,
                "#3597 regression: engine bounds differ at dim {d}: [{el},{eu}] vs [{bl},{bu}]"
            );
        }

        let calls = engine.gemm_calls();
        assert!(
            calls > 0,
            "#3597 regression: missing-alpha GELU fallback should hit GemmEngine, got {calls} calls"
        );
    });
}

#[test]
fn test_alpha_block_crown_threads_engine_3597() {
    // Serialized against budget=0 tests to prevent CROWN fallback (#3515).
    tests::with_crown_dense_budget_mb("2048", || {
        let hidden = 4;
        let expansion = 2;
        let epsilon = 0.01_f32;

        let graph = build_two_block_ffn_graph(hidden, expansion);
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();
        let baseline = graph
            .propagate_alpha_crown_block_wise(&input, epsilon)
            .unwrap();

        let engine = CountingGemmEngine::new();
        let engine_result = graph
            .propagate_alpha_crown_block_wise_with_engine(&input, epsilon, Some(&engine))
            .unwrap();

        assert_alpha_block_wise_results_match(&engine_result, &baseline);

        let calls = engine.gemm_calls();
        assert!(
            calls > 0,
            "#3597 regression: alpha block-wise CROWN should hit GemmEngine, got {calls} calls"
        );
    });
}
