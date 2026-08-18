// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// Talker attention layer (#3497)
//
// `talker_attention_layer0.onnx` is a single Qwen3-TTS attention layer with
// four inputs: hidden_states [B, T, H], cos [1, 1, T, 64],
// sin [1, 1, T, 64], mask [1, 1, T, T].
//
// The first real-weight slice freezes cos/sin/mask as concrete auxiliary
// tensors and keeps hidden_states as the only bounded activation input.
// Reference: designs/2026-03-11-issue-3497-talker-attention-softmax-surface.md
// ---------------------------------------------------------------------------

pub(super) const TALKER_ATTENTION_FILE: &str = "talker_attention_layer0.onnx";
pub(in super::super) const TALKER_ATTENTION_SEQ_LEN: usize = 16;
pub(in super::super) const TALKER_ATTENTION_SHORT_SEQ_LEN: usize = 4;
pub(in super::super) const TALKER_ATTENTION_HIDDEN_DIM: usize = 1024;
pub(in super::super) const TALKER_ATTENTION_ROPE_DIM: usize = 64;
pub(in super::super) const TALKER_ATTENTION_EPSILON: f32 = 1e-3;

/// Current Qwen3-TTS RoPE frequency scalar (`rope_theta` in avoice config).
/// The fixture contract keeps the historical `rope_base` field name because
/// `compute_qwen3_rope_tables()` still takes the generic RoPE formula input.
pub(in super::super) const QWEN3_TTS_ROPE_BASE: f64 = 1_000_000.0;

/// Names of auxiliary inputs that should be frozen as concrete weight tensors.
pub(super) const TALKER_AUX_INPUTS: &[&str] = &["cos", "sin", "mask"];

// ---------------------------------------------------------------------------
// Contract accessor boundary (#3917)
//
// Centralizes exporter-owned facts behind a cached, fallback-safe accessor.
// When a `.contract.json` sidecar exists, the accessor validates and returns
// sidecar values; otherwise it falls back to today's constants.
// ---------------------------------------------------------------------------

pub(super) struct TalkerFixtureContract {
    pub canonical_seq_len: usize,
    pub hidden_dim: usize,
    pub rope_dim: usize,
    pub rope_base: f64,
    pub mask_kind: &'static str,
}

/// Return the cached talker-attention fixture contract.
///
/// Consults `load_avoice_contract()` once, validates the sidecar if present,
/// and falls back to the current local constants when no sidecar exists.
pub(super) fn talker_fixture_contract() -> &'static TalkerFixtureContract {
    static CONTRACT: OnceLock<TalkerFixtureContract> = OnceLock::new();
    CONTRACT.get_or_init(|| {
        let model_path =
            require_test_model_with_hint(TALKER_ATTENTION_FILE, AVOICE_TEST_MODEL_HINT);
        match load_avoice_contract(&model_path) {
            Ok(Some(contract)) => {
                assert_eq!(
                    contract.activation_input, "hidden_states",
                    "talker attention sidecar activation_input must be hidden_states, \
                     got {:?} at {:?}",
                    contract.activation_input, model_path
                );
                for expected_aux in &["cos", "sin", "mask"] {
                    assert!(
                        contract.aux_inputs.iter().any(|a| a == expected_aux),
                        "talker attention sidecar must list '{expected_aux}' in aux_inputs, \
                         got {:?} at {:?}",
                        contract.aux_inputs,
                        model_path
                    );
                }
                let constraints = &contract.constraints;
                TalkerFixtureContract {
                    canonical_seq_len: contract
                        .canonical_seq_len
                        .unwrap_or(TALKER_ATTENTION_SEQ_LEN),
                    hidden_dim: constraints
                        .hidden_dim
                        .unwrap_or(TALKER_ATTENTION_HIDDEN_DIM),
                    rope_dim: constraints.rope_dim.unwrap_or(TALKER_ATTENTION_ROPE_DIM),
                    rope_base: constraints.rope_base.unwrap_or(QWEN3_TTS_ROPE_BASE),
                    mask_kind: match constraints.mask_kind.as_deref() {
                        Some("causal_upper_neg_inf") | None => "causal_upper_neg_inf",
                        Some(other) => panic!(
                            "talker attention sidecar mask_kind must be \
                             'causal_upper_neg_inf', got '{other}' at {model_path:?}"
                        ),
                    },
                }
            }
            Ok(None) => TalkerFixtureContract {
                canonical_seq_len: TALKER_ATTENTION_SEQ_LEN,
                hidden_dim: TALKER_ATTENTION_HIDDEN_DIM,
                rope_dim: TALKER_ATTENTION_ROPE_DIM,
                rope_base: QWEN3_TTS_ROPE_BASE,
                mask_kind: "causal_upper_neg_inf",
            },
            Err(e) => {
                panic!("failed to load talker attention contract sidecar at {model_path:?}: {e}")
            }
        }
    })
}

pub(in super::super) fn avoice_talker_attention_raw() -> &'static OnnxModel {
    static MODEL: OnceLock<OnnxModel> = OnceLock::new();
    MODEL.get_or_init(|| {
        let path = require_test_model_with_hint(TALKER_ATTENTION_FILE, AVOICE_TEST_MODEL_HINT);
        load_onnx(&path).expect("Failed to load avoice talker_attention_layer0.onnx")
    })
}

// ---------------------------------------------------------------------------
// Contract regression tests (#3917, #4180)
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "external-avoice")]
fn test_talker_fixture_contract_matches_current_avoice_contract_4180() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let contract = talker_fixture_contract();
    assert_eq!(
        contract.canonical_seq_len, TALKER_ATTENTION_SEQ_LEN,
        "talker canonical_seq_len should match the current fixture contract"
    );
    assert_eq!(
        contract.hidden_dim, TALKER_ATTENTION_HIDDEN_DIM,
        "talker hidden_dim should match the current fixture contract"
    );
    assert_eq!(
        contract.rope_dim, 64,
        "talker exported cos/sin width should stay at 64 even though runtime head_dim is 128"
    );
    assert!(
        (contract.rope_base - 1_000_000.0).abs() < f64::EPSILON,
        "talker rope scalar should match the current avoice rope_theta=1_000_000.0 contract"
    );
    assert_eq!(
        contract.rope_dim * 2,
        128,
        "talker exported cos/sin width should remain half of the runtime head_dim"
    );
    assert_eq!(
        contract.mask_kind, "causal_upper_neg_inf",
        "talker mask_kind should remain causal_upper_neg_inf"
    );
}
