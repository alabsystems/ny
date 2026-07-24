// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::MultiBlockConfig;
use super::helpers::{whisper_tiny_encoder, whisper_zero_input};

#[ntest::timeout(120000)]
#[test]
fn test_multi_block_sound_tight_reset_flag_is_semantic_noop() {
    crate::test_fixtures::require_test_model_or_skip!("whisper_tiny_encoder.onnx");
    // Part of #318: the compositional sequential verifier already rebuilds a
    // fresh zonotope from each block's interval input, so toggling the
    // compatibility reset flag must not change the computed bounds.
    let whisper = whisper_tiny_encoder();
    let hidden_dim = whisper.hidden_dim;

    // A single-token sequence still crosses the block boundary while keeping
    // the duplicated sequential run under the test timeout budget.
    let input = whisper_zero_input(hidden_dim, 1, 1e-4);

    let with_reset = MultiBlockConfig::sound_tight();
    let without_reset = MultiBlockConfig::sound_tight().with_reset_zonotope_between_blocks(false);

    let (output_reset, details_reset) = whisper
        .verify_encoder_sequential_with_config(&input, 0, 2, false, false, None, &with_reset)
        .unwrap_or_else(|e| panic!("sound-tight with reset failed: {:?}", e));
    let (output_no_reset, details_no_reset) = whisper
        .verify_encoder_sequential_with_config(&input, 0, 2, false, false, None, &without_reset)
        .unwrap_or_else(|e| panic!("sound-tight without reset failed: {:?}", e));

    assert_eq!(
        details_reset.blocks_completed, details_no_reset.blocks_completed,
        "reset flag must not change blocks_completed"
    );
    assert_eq!(
        details_reset.block_details.len(),
        details_no_reset.block_details.len(),
        "reset flag must not change block detail count"
    );
    assert_eq!(
        details_reset.final_output_width.to_bits(),
        details_no_reset.final_output_width.to_bits(),
        "reset flag must not change final output width"
    );

    for (idx, (reset_block, no_reset_block)) in details_reset
        .block_details
        .iter()
        .zip(details_no_reset.block_details.iter())
        .enumerate()
    {
        assert_eq!(
            reset_block.attention_delta_width.to_bits(),
            no_reset_block.attention_delta_width.to_bits(),
            "block {idx} attention width changed when toggling reset flag"
        );
        assert_eq!(
            reset_block.mlp_delta_width.to_bits(),
            no_reset_block.mlp_delta_width.to_bits(),
            "block {idx} mlp width changed when toggling reset flag"
        );
        assert_eq!(
            reset_block.output_width.to_bits(),
            no_reset_block.output_width.to_bits(),
            "block {idx} output width changed when toggling reset flag"
        );
    }

    for (lhs, rhs) in output_reset
        .lower()
        .iter()
        .zip(output_no_reset.lower().iter())
    {
        assert_eq!(
            lhs.to_bits(),
            rhs.to_bits(),
            "reset flag changed lower-bound values"
        );
    }
    for (lhs, rhs) in output_reset
        .upper()
        .iter()
        .zip(output_no_reset.upper().iter())
    {
        assert_eq!(
            lhs.to_bits(),
            rhs.to_bits(),
            "reset flag changed upper-bound values"
        );
    }
}
