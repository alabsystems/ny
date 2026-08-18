// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Kokoro vocoder model loading, shape contracts, and bounded-input builders.

use super::super::common::{input_spec_by_name, unbatched_shape_from_input_spec};
use super::*;

pub(super) const KOKORO_VOCODER_FILE: &str = "kokoro_vocoder.onnx";
// Minimum features temporal window for the fixed-aux harness.
//
// Under the corrected har contract (har_t = 60*features_t + 1), features_t=1
// produces har=[22,61].  The old minimum of 6 (from the equal-axis era, where
// har_t == features_t == 6) makes har=[22,361], whose const-folding alone
// exceeds the 60-120s test budgets.  features_t=1 is verified valid by ORT
// forward (test_kokoro_vocoder_ort_forward_accepts_exported_har_contract_3500).
pub(super) const KOKORO_VOCODER_MIN_FIXED_AUX_T: usize = 1;
pub(super) const KOKORO_VOCODER_STRUCTURAL_T: usize = 1;
const KOKORO_AUX_INPUTS: &[&str] = &["style", "har"];
const KOKORO_F0_FRAMES_PER_FEATURE_FRAME: usize = 2;
const KOKORO_AUDIO_SAMPLES_PER_FEATURE_FRAME: usize = 300;
const KOKORO_HAR_STFT_HOP_SIZE: usize = 5;

// ---------------------------------------------------------------------------
// Contract accessor boundary (#3917)
//
// Centralizes exporter-owned `min_fixed_aux_t` behind a cached, fallback-safe
// accessor so child modules do not need to import the raw constant directly.
// ---------------------------------------------------------------------------

struct KokoroVocoderFixtureContract {
    min_fixed_aux_t: usize,
}

fn kokoro_vocoder_fixture_contract() -> &'static KokoroVocoderFixtureContract {
    static CONTRACT: OnceLock<KokoroVocoderFixtureContract> = OnceLock::new();
    CONTRACT.get_or_init(|| {
        let model_path = require_test_model_with_hint(KOKORO_VOCODER_FILE, AVOICE_TEST_MODEL_HINT);
        match load_avoice_contract(&model_path) {
            Ok(Some(contract)) => {
                assert_eq!(
                    contract.activation_input, "features",
                    "kokoro vocoder sidecar activation_input must be features, \
                     got {:?} at {:?}",
                    contract.activation_input, model_path
                );
                for expected_aux in &["style", "har"] {
                    assert!(
                        contract.aux_inputs.iter().any(|a| a == expected_aux),
                        "kokoro vocoder sidecar must list '{expected_aux}' in aux_inputs, \
                         got {:?} at {:?}",
                        contract.aux_inputs,
                        model_path
                    );
                }
                KokoroVocoderFixtureContract {
                    min_fixed_aux_t: contract
                        .constraints
                        .min_fixed_aux_t
                        .unwrap_or(KOKORO_VOCODER_MIN_FIXED_AUX_T),
                }
            }
            Ok(None) => KokoroVocoderFixtureContract {
                min_fixed_aux_t: KOKORO_VOCODER_MIN_FIXED_AUX_T,
            },
            Err(e) => {
                panic!("failed to load kokoro vocoder contract sidecar at {model_path:?}: {e}")
            }
        }
    })
}

/// Return the minimum features temporal window from the contract.
///
/// Prefers sidecar value when present, falls back to
/// [`KOKORO_VOCODER_MIN_FIXED_AUX_T`].
pub(super) fn kokoro_vocoder_min_fixed_aux_t() -> usize {
    kokoro_vocoder_fixture_contract().min_fixed_aux_t
}

pub(crate) fn load_kokoro_vocoder_with_fixed_aux(dynamic_t: usize) -> OnnxModel {
    let min_t = kokoro_vocoder_min_fixed_aux_t();
    assert!(
        dynamic_t >= min_t,
        "kokoro fixed-aux helper currently requires features T >= {} for the \
         verified runtime harness",
        min_t
    );

    let path = require_test_model_with_hint(KOKORO_VOCODER_FILE, AVOICE_TEST_MODEL_HINT);
    let raw = load_onnx(&path).expect("Failed to load kokoro_vocoder.onnx");

    let input_names: Vec<&str> = raw.network.inputs.iter().map(|s| s.name.as_str()).collect();
    assert!(
        input_names.contains(&"features"),
        "kokoro vocoder export should contain features input, got {:?}",
        input_names
    );
    for &aux in KOKORO_AUX_INPUTS {
        assert!(
            input_names.contains(&aux),
            "kokoro vocoder export should contain auxiliary input '{aux}', got {:?}",
            input_names
        );
    }
    assert_kokoro_vocoder_io_shapes(&raw);

    let style_shape =
        unbatched_shape_from_input_spec(input_spec_by_name(&raw, "style"), dynamic_t, "style");
    let har_shape = unbatched_shape_from_input_spec(
        input_spec_by_name(&raw, "har"),
        kokoro_har_time_for_features_t(dynamic_t),
        "har",
    );

    let mut model = raw;
    model.freeze_inputs([
        ("style".to_string(), ArrayD::zeros(IxDyn(&style_shape))),
        ("har".to_string(), ArrayD::zeros(IxDyn(&har_shape))),
    ]);

    assert_eq!(
        model.network.inputs.len(),
        1,
        "after aux freezing, kokoro vocoder should have exactly 1 activation input, got {:?}",
        model
            .network
            .inputs
            .iter()
            .map(|s| &s.name)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        model.network.inputs[0].name, "features",
        "the remaining activation input should be features"
    );

    model
}

/// The ONNX export does not use `har_t == features_t`.
///
/// Upstream Kokoro produces `f0 [B, 1, 2*T_mel]`, upsamples that signal to
/// audio rate with total factor `10 * 6 * 5 = 300`, then runs a center-padded
/// STFT with `hop=5`, so:
///
/// `T_har = T_audio / hop + 1 = (T_features * 300) / 5 + 1 = 60*T_features + 1`
///
/// Sources:
/// - `./avoice/crates/avoice-tts/src/kokoro/prosody.rs:177-180`
/// - `./avoice/crates/avoice-tts/src/kokoro/config.rs:115-118`
/// - `./avoice/crates/avoice-tts/src/kokoro/generator_unit_tests.rs:137-148`
pub(super) fn kokoro_har_time_for_features_t(features_t: usize) -> usize {
    let f0_t = features_t
        .checked_mul(KOKORO_F0_FRAMES_PER_FEATURE_FRAME)
        .expect("features_t should fit in usize after F0 upsample");
    let audio_t = f0_t
        .checked_mul(KOKORO_AUDIO_SAMPLES_PER_FEATURE_FRAME / KOKORO_F0_FRAMES_PER_FEATURE_FRAME)
        .expect("features_t should fit in usize after audio-rate upsample");
    audio_t / KOKORO_HAR_STFT_HOP_SIZE + 1
}

pub(super) fn assert_kokoro_vocoder_io_shapes(model: &OnnxModel) {
    let features_spec = input_spec_by_name(model, "features");
    assert_eq!(
        features_spec.shape.len(),
        3,
        "kokoro vocoder features input should be rank-3 [B, 512, T], got {:?}",
        features_spec.shape
    );
    assert_eq!(
        features_spec.shape[1], 512,
        "kokoro vocoder features channel axis should be 512, got {:?}",
        features_spec.shape
    );

    let style_spec = input_spec_by_name(model, "style");
    assert_eq!(
        style_spec.shape.len(),
        2,
        "kokoro vocoder style input should be rank-2 [B, 128], got {:?}",
        style_spec.shape
    );
    assert_eq!(
        style_spec.shape[1], 128,
        "kokoro vocoder style embedding axis should be 128, got {:?}",
        style_spec.shape
    );

    let har_spec = input_spec_by_name(model, "har");
    assert_eq!(
        har_spec.shape.len(),
        3,
        "kokoro vocoder har input should be rank-3 [B, 22, T], got {:?}",
        har_spec.shape
    );
    assert_eq!(
        har_spec.shape[1], 22,
        "kokoro vocoder har channel axis should be 22, got {:?}",
        har_spec.shape
    );

    assert_eq!(
        model.network.outputs.len(),
        1,
        "kokoro vocoder should expose one waveform output"
    );
    let output_spec = &model.network.outputs[0];
    assert_eq!(
        output_spec.shape.len(),
        3,
        "kokoro vocoder output should be rank-3 [B, 1, T], got {:?}",
        output_spec.shape
    );
    assert!(
        matches!(output_spec.shape.get(1), Some(&1) | Some(&-1)),
        "kokoro vocoder output channel axis should be 1 or dynamic, got {:?}",
        output_spec.shape
    );
}

/// Build a bounded features input for the kokoro vocoder.
///
/// Reads the `features` input spec from the raw model, strips the batch axis,
/// replaces dynamic axes with `dynamic_t`, and creates an epsilon ball around
/// a zero center.
pub(super) fn bounded_kokoro_features_input(
    model: &OnnxModel,
    dynamic_t: usize,
    epsilon: f32,
) -> BoundedTensor {
    bounded_kokoro_features_input_centered(model, dynamic_t, 0.0, epsilon)
}

/// Build a bounded features input centered at a uniform non-zero value.
///
/// Used for two-chunk crossfade tests where adjacent streaming chunks have
/// different audio content. A non-zero center produces content-dependent
/// vocoder outputs, exercising the real-weight path with different feature
/// distributions in each chunk.
pub(crate) fn bounded_kokoro_features_input_centered(
    model: &OnnxModel,
    dynamic_t: usize,
    center_value: f32,
    epsilon: f32,
) -> BoundedTensor {
    let features_spec = input_spec_by_name(model, "features");
    let shape = unbatched_shape_from_input_spec(features_spec, dynamic_t, "features");
    let center = ArrayD::from_elem(IxDyn(&shape), center_value);
    BoundedTensor::from_epsilon(center, epsilon).expect("valid features epsilon ball")
}

// ---------------------------------------------------------------------------
// Fallback regression test (#3917)
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "external-avoice")]
fn test_kokoro_vocoder_fixture_contract_fallback_matches_constants_3917() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    assert_eq!(
        kokoro_vocoder_min_fixed_aux_t(),
        KOKORO_VOCODER_MIN_FIXED_AUX_T,
        "fallback min_fixed_aux_t should match KOKORO_VOCODER_MIN_FIXED_AUX_T"
    );
}
