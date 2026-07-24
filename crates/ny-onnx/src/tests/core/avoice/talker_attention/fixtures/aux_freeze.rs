// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::contract::{
    avoice_talker_attention_raw, talker_fixture_contract, TALKER_ATTENTION_FILE,
    TALKER_ATTENTION_SEQ_LEN, TALKER_AUX_INPUTS,
};
use super::*;

/// Build a causal mask: 0 on and below the diagonal, -1e4 above.
/// Unbatched [T, T] to match ny propagation convention.
fn causal_mask(seq_len: usize) -> ArrayD<f32> {
    let mut mask = ArrayD::from_elem(IxDyn(&[seq_len, seq_len]), -1e4f32);
    for i in 0..seq_len {
        for j in 0..=i {
            mask[[i, j]] = 0.0;
        }
    }
    mask
}

/// Compute Qwen3-TTS RoPE frequency tables for a given sequence length.
///
/// Uses the standard Su et al. (2021) formula:
///   inv_freq[i] = 1 / (base^(2i/dim))  for i = 0..dim/2
///   angle[pos, i] = pos * inv_freq[i]
///
/// The output cos/sin tables follow the HuggingFace "repeated" layout:
///   cos_table[pos, :] = [cos(theta_0), ..., cos(theta_{d/2-1}), cos(theta_0), ..., cos(theta_{d/2-1})]
///   sin_table[pos, :] = [sin(theta_0), ..., sin(theta_{d/2-1}), sin(theta_0), ..., sin(theta_{d/2-1})]
///
/// Returns (cos_table, sin_table) both of shape [seq_len, rope_dim].
/// Computation uses f64 for precision, then casts to f32 for the fixture.
///
/// Reference: Su et al. "RoFormer: Enhanced Transformer with Rotary Position
/// Embedding" (2021), Section 3.4.2
pub(in super::super) fn compute_qwen3_rope_tables(
    seq_len: usize,
    rope_dim: usize,
    base: f64,
) -> (ArrayD<f32>, ArrayD<f32>) {
    assert!(
        rope_dim.is_multiple_of(2),
        "rope_dim must be even, got {rope_dim}"
    );
    let num_pairs = rope_dim / 2;

    let inv_freq: Vec<f64> = (0..num_pairs)
        .map(|i| {
            let exponent = 2.0 * i as f64 / rope_dim as f64;
            1.0 / base.powf(exponent)
        })
        .collect();

    let mut cos_data = Vec::with_capacity(seq_len * rope_dim);
    let mut sin_data = Vec::with_capacity(seq_len * rope_dim);

    for pos in 0..seq_len {
        let half_cos: Vec<f32> = inv_freq
            .iter()
            .map(|&freq| (pos as f64 * freq).cos() as f32)
            .collect();
        let half_sin: Vec<f32> = inv_freq
            .iter()
            .map(|&freq| (pos as f64 * freq).sin() as f32)
            .collect();

        cos_data.extend_from_slice(&half_cos);
        cos_data.extend_from_slice(&half_cos);
        sin_data.extend_from_slice(&half_sin);
        sin_data.extend_from_slice(&half_sin);
    }

    let cos_table =
        ArrayD::from_shape_vec(IxDyn(&[seq_len, rope_dim]), cos_data).expect("cos_table shape");
    let sin_table =
        ArrayD::from_shape_vec(IxDyn(&[seq_len, rope_dim]), sin_data).expect("sin_table shape");

    (cos_table, sin_table)
}

fn assert_talker_input_inventory(raw: &OnnxModel) {
    let input_names: Vec<&str> = raw.network.inputs.iter().map(|s| s.name.as_str()).collect();
    assert!(
        input_names.contains(&"hidden_states"),
        "talker attention export should contain hidden_states input, got {:?}",
        input_names
    );
    for &aux in TALKER_AUX_INPUTS {
        assert!(
            input_names.contains(&aux),
            "talker attention export should contain auxiliary input '{aux}', got {:?}",
            input_names
        );
    }
}

/// Load `talker_attention_layer0.onnx` with cos/sin/mask frozen to concrete
/// auxiliary tensors, leaving only `hidden_states` as the activation input.
///
/// This is a test-side wrapper; no production API changes are needed.
pub(in super::super) fn load_talker_attention_with_fixed_aux() -> OnnxModel {
    load_talker_attention_with_fixed_aux_for_seq_len(TALKER_ATTENTION_SEQ_LEN)
}

/// Experimental short-seq lane for the talker-attention test surface.
///
/// The exported contract lane remains fixed at `TALKER_ATTENTION_SEQ_LEN`.
/// Smaller `seq_len` values are only for local smoke tests that isolate shape
/// wiring before paying the full attention runtime.
pub(in super::super) fn load_talker_attention_with_fixed_aux_for_seq_len(
    seq_len: usize,
) -> OnnxModel {
    let raw = avoice_talker_attention_raw();
    let mut model = load_onnx(require_test_model_with_hint(
        TALKER_ATTENTION_FILE,
        AVOICE_TEST_MODEL_HINT,
    ))
    .expect("Failed to reload talker_attention_layer0.onnx for aux freezing");

    assert_talker_input_inventory(raw);

    let contract = talker_fixture_contract();
    assert_eq!(
        contract.mask_kind, "causal_upper_neg_inf",
        "causal_mask helper requires causal_upper_neg_inf, contract says '{}'",
        contract.mask_kind
    );

    // Frozen weights must be unbatched to match ny's propagation
    // convention: the ONNX model uses [batch, ...] shapes, but the graph
    // builder strips batch axis 0 during conversion. Weights injected after
    // load must already be unbatched.
    model.freeze_inputs([
        (
            "cos".to_string(),
            ArrayD::ones(IxDyn(&[seq_len, contract.rope_dim])),
        ),
        (
            "sin".to_string(),
            ArrayD::zeros(IxDyn(&[seq_len, contract.rope_dim])),
        ),
        ("mask".to_string(), causal_mask(seq_len)),
    ]);

    assert_eq!(
        model.network.inputs.len(),
        1,
        "after aux freezing, talker attention should have exactly 1 activation input, got {:?}",
        model
            .network
            .inputs
            .iter()
            .map(|s| &s.name)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        model.network.inputs[0].name, "hidden_states",
        "the remaining activation input should be hidden_states"
    );

    model
}

/// Load talker attention with real Qwen3-TTS RoPE frequency tables.
///
/// Same as `load_talker_attention_with_fixed_aux_for_seq_len` but injects
/// position-dependent cos/sin tables from the standard Su et al. (2021)
/// formula instead of identity rotation (cos=1, sin=0).
pub(in super::super) fn load_talker_attention_with_real_rope_seq_len(seq_len: usize) -> OnnxModel {
    let raw = avoice_talker_attention_raw();
    let mut model = load_onnx(require_test_model_with_hint(
        TALKER_ATTENTION_FILE,
        AVOICE_TEST_MODEL_HINT,
    ))
    .expect("Failed to reload talker_attention_layer0.onnx for real-RoPE aux freezing");

    assert_talker_input_inventory(raw);

    let contract = talker_fixture_contract();
    let (cos_tensor, sin_tensor) =
        compute_qwen3_rope_tables(seq_len, contract.rope_dim, contract.rope_base);
    model.freeze_inputs([
        ("cos".to_string(), cos_tensor),
        ("sin".to_string(), sin_tensor),
        ("mask".to_string(), causal_mask(seq_len)),
    ]);

    assert_eq!(
        model.network.inputs.len(),
        1,
        "after aux freezing, talker attention should have exactly 1 activation input, got {:?}",
        model
            .network
            .inputs
            .iter()
            .map(|s| &s.name)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        model.network.inputs[0].name, "hidden_states",
        "the remaining activation input should be hidden_states"
    );

    model
}
