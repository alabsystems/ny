// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concrete-center bridge smoke for Kokoro vocoder -> mel128 -> speaker encoder.
//!
//! Uses ONNX Runtime to produce a real concrete Kokoro waveform at features_t=5,
//! then feeds it through the verifier-local mel128 graph and the real speaker
//! encoder graph.  This exercises the full vocoder→mel128→speaker composition
//! with real model weights without the slow `to_graph_network()` const-folding
//! path.
//!
//! Bounded full-waveform IBP is out of scope here (tracked by #3500, #3622).
//!
//! Reference: designs/2026-03-15-issue-3719-ort-concrete-speaker-bridge.md

use super::common::assert_finite_and_ordered;
use ny_propagate::network::dsp::build_qwen3_mel128_graph;

// features_t=5 at 300 audio samples per feature frame. The current Kokoro
// export trims 10 samples from each end of the raw upsampled waveform
// (top-level `/Slice_2`, starts=10/ends=-10), so the emitted waveform is
// exactly features_t * 300 samples; ORT confirms (1, 1, 1500) for
// features_t=5 / har_t=301. The earlier 1520 pin captured a pre-trim export.
const KOKORO_BRIDGE_AUDIO_LEN_3719: usize = 1500;

// Budget: ORT vocoder forward (~10-30s at features_t=5) + mel128 IBP +
// speaker encoder graph conversion and IBP.  Much cheaper than the old
// to_graph_network() path which exceeded 600s at features_t>=5.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_kokoro_to_speaker_bridge_concrete_smoke_3719() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    // Stage 1: Produce a real concrete Kokoro waveform via ORT.
    // The shared speaker floor comes from fixed TDNN pads [2, 2, 3, 4, 0], so
    // T=5 is the smallest valid real-weight sequence for the encoder.
    let waveform = super::kokoro_vocoder::kokoro_vocoder_concrete_waveform_from_ort(
        super::speaker_encoder::SPEAKER_ENCODER_SEQUENCE_LEN,
        0.01,
    );
    assert_finite_and_ordered(&waveform, "bridge vocoder ORT waveform");
    assert_eq!(
        waveform.shape()[0],
        1,
        "waveform batch dim should be 1, got {}",
        waveform.shape()[0]
    );
    let audio_len = waveform.shape()[1];
    // Pin the current exported waveform contract exactly: any audio length in
    // [1280, 1535] still yields a [5, 128] mel tensor, so the bridge needs a
    // direct audio-length assertion to catch silent ORT/export drift.
    assert_eq!(
        audio_len,
        KOKORO_BRIDGE_AUDIO_LEN_3719,
        "features_t={} should produce exactly {KOKORO_BRIDGE_AUDIO_LEN_3719} samples on the current Kokoro export, got {audio_len}",
        super::speaker_encoder::SPEAKER_ENCODER_SEQUENCE_LEN
    );
    eprintln!(
        "ORT vocoder: features_t={} -> audio_len={audio_len}",
        super::speaker_encoder::SPEAKER_ENCODER_SEQUENCE_LEN
    );

    // Stage 2: mel128 bridge — converts real waveform to mel spectrogram.
    // T_mel = (audio_len + 2*pad - window) / hop + 1 where pad=384, window=1024, hop=256
    let mel_graph = build_qwen3_mel128_graph(audio_len)
        .expect("mel128 graph should build from vocoder-compatible waveform length");
    let mel = mel_graph
        .propagate_ibp(&waveform)
        .expect("mel128 bridge propagation should succeed");
    let t_mel = mel.shape()[0];
    assert_eq!(
        mel.shape(),
        &[super::speaker_encoder::SPEAKER_ENCODER_SEQUENCE_LEN, 128],
        "features_t={} bridge should produce a [{}, 128] mel spectrogram, got {:?}",
        super::speaker_encoder::SPEAKER_ENCODER_SEQUENCE_LEN,
        super::speaker_encoder::SPEAKER_ENCODER_SEQUENCE_LEN,
        mel.shape()
    );
    eprintln!("mel128 bridge: audio_len={audio_len} -> T_mel={t_mel}");
    assert_finite_and_ordered(&mel, "kokoro speaker bridge mel");

    // Stage 3: Speaker encoder — unconditional at T_mel=5.
    let speaker = super::speaker_encoder::avoice_speaker_encoder_graph()
        .propagate_ibp(&mel)
        .expect("speaker encoder bridge propagation should succeed");
    assert_finite_and_ordered(&speaker, "kokoro speaker bridge speaker output");
    assert_eq!(
        speaker.lower().shape().last().copied(),
        Some(1024),
        "speaker encoder bridge output should end in 1024 dims, got {:?}",
        speaker.lower().shape()
    );
}
