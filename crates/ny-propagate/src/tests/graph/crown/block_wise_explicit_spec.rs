// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit `BlockSpec` regression tests (#4024).
//!
//! Tests that consumer-supplied block boundaries with non-`layer{N}` node
//! names and sparse block indices preserve metadata through
//! `BlockWiseCrownResult`.

use ndarray::{Array2, ArrayD, IxDyn};

use ny_tensor::BoundedTensor;

use crate::layers::binary_ops::AddLayer;
use crate::layers::linear::LinearLayer;
use crate::network::{BlockSpec, BlockSpecEntry};
use crate::*;

/// Build a two-block FFN graph with non-`layer{N}` node names for explicit
/// `BlockSpec` testing. Node names use `trace_ffn_{a,b}_*` prefixes that
/// the legacy `parse_block_index` cannot parse.
fn build_two_block_trace_graph(hidden: usize, expansion: usize) -> GraphNetwork {
    let scale1 = (2.0 / (hidden + hidden * expansion) as f32).sqrt();
    let scale2 = (2.0 / (hidden * expansion + hidden) as f32).sqrt();

    let mut graph = GraphNetwork::new();

    let prefixes = ["trace_ffn_a", "trace_ffn_b"];
    for (i, prefix) in prefixes.iter().enumerate() {
        let block_input_name = if i == 0 {
            NETWORK_INPUT.to_string()
        } else {
            format!("{}_add", prefixes[i - 1])
        };

        // Linear up
        let weight_up = Array2::from_shape_fn((hidden * expansion, hidden), |(r, c)| {
            let phase = (r * 17 + c * 31 + i * 97) as f32;
            scale1 * phase.sin() * 0.15
        });
        let linear_up = LinearLayer::new(weight_up, None).unwrap();
        graph.add_node(GraphNode::new(
            format!("{}_up", prefix),
            Layer::Linear(linear_up),
            vec![block_input_name.clone()],
        ));

        // ReLU (simpler activation for fast test)
        graph.add_node(GraphNode::new(
            format!("{}_relu", prefix),
            Layer::ReLU(ReLULayer),
            vec![format!("{}_up", prefix)],
        ));

        // Linear down
        let weight_down = Array2::from_shape_fn((hidden, hidden * expansion), |(r, c)| {
            let phase = (r * 23 + c * 37 + i * 71) as f32;
            scale2 * phase.cos() * 0.15
        });
        let linear_down = LinearLayer::new(weight_down, None).unwrap();
        graph.add_node(GraphNode::new(
            format!("{}_down", prefix),
            Layer::Linear(linear_down),
            vec![format!("{}_relu", prefix)],
        ));

        // Residual Add
        let add = AddLayer;
        graph.add_node(GraphNode::new(
            format!("{}_add", prefix),
            Layer::Add(add),
            vec![block_input_name, format!("{}_down", prefix)],
        ));
    }

    graph.set_output("trace_ffn_b_add");
    graph
}

/// #4024 regression: explicit `BlockSpec` with non-`layer{N}` node names and
/// sparse block indices preserves consumer-chosen metadata through
/// `BlockWiseCrownResult`.
#[ntest::timeout(10000)]
#[test]
fn test_explicit_block_spec_preserves_sparse_metadata_4024() {
    tests::with_crown_dense_budget_mb("2048", || {
        let hidden = 4;
        let expansion = 2;
        let epsilon = 0.05_f32;

        let graph = build_two_block_trace_graph(hidden, expansion);

        // Legacy discovery finds zero blocks (no `layer{N}` names).
        let input = BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[hidden])), epsilon).unwrap();
        let legacy_result = graph.propagate_crown_block_wise(&input, epsilon).unwrap();
        assert_eq!(
            legacy_result.total_blocks, 0,
            "legacy discovery must find no blocks with trace_ffn_* names"
        );

        // Explicit block spec with sparse indices.
        let spec = BlockSpec {
            blocks: vec![
                BlockSpecEntry {
                    block_index: 3,
                    block_name: "encoder_block_3".to_string(),
                    node_names: vec![
                        "trace_ffn_a_up".to_string(),
                        "trace_ffn_a_relu".to_string(),
                        "trace_ffn_a_down".to_string(),
                        "trace_ffn_a_add".to_string(),
                    ],
                },
                BlockSpecEntry {
                    block_index: 7,
                    block_name: "encoder_block_7".to_string(),
                    node_names: vec![
                        "trace_ffn_b_up".to_string(),
                        "trace_ffn_b_relu".to_string(),
                        "trace_ffn_b_down".to_string(),
                        "trace_ffn_b_add".to_string(),
                    ],
                },
            ],
        };

        let result = graph
            .propagate_crown_with_blocks(&input, epsilon, &spec)
            .expect("explicit BlockSpec should succeed on traced graph");

        // Assert metadata survived.
        assert_eq!(result.total_blocks, 2, "expected 2 explicit blocks");
        assert_eq!(result.blocks[0].block_index, 3);
        assert_eq!(result.blocks[0].block_name, "encoder_block_3");
        assert_eq!(result.blocks[1].block_index, 7);
        assert_eq!(result.blocks[1].block_name, "encoder_block_7");

        // Both blocks should report finite widths.
        for block in &result.blocks {
            assert!(
                block.ibp_max_width.is_finite() && block.ibp_max_width > 0.0,
                "block '{}' IBP width should be finite and positive, got {}",
                block.block_name,
                block.ibp_max_width
            );
            assert!(
                block.crown_max_width.is_finite(),
                "block '{}' CROWN width should be finite, got {}",
                block.block_name,
                block.crown_max_width
            );
        }

        // At least one block should have CROWN tighter than IBP.
        let any_tight = result.blocks.iter().any(|b| b.crown_ibp_ratio < 1.0);
        assert!(
            any_tight,
            "at least one explicit block should have crown_ibp_ratio < 1.0"
        );
    });
}
