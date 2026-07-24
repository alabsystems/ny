// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Curated DSP / audio graph surface for external consumers.
//!
//! These builders construct [`graph`](crate::graph) networks that represent
//! common DSP operations (waveshaping, biquad IIR filtering, envelope
//! following, STFT, mel spectrograms) as compositions of existing ny layers,
//! enabling bound propagation (IBP / CROWN / alpha-CROWN) without new layer
//! types. The Kokoro forward-STFT helpers back TTS verification.

/// Build a waveshaper (tanh soft-clip) verification graph for a fixed drive,
/// implementing `f(x) = tanh(d·x) / tanh(d)` as a small pointwise graph.
pub use ny_propagate::network::dsp::build_waveshaper_graph;

/// Biquad IIR filter coefficients (Direct Form II Transposed) plus a builder
/// for a single linear filter step (exact under IBP / CROWN).
pub use ny_propagate::network::dsp::{build_biquad_single_step_graph, BiquadCoefficients};

/// Envelope-follower graph builders and parameters: one-pole step, gain
/// reduction, and a conservative end-to-end verification helper.
pub use ny_propagate::network::dsp::{
    build_envelope_follower_step_graph, build_gain_reduction_graph, build_one_pole_step_graph,
    verify_envelope_follower_conservative, EnvelopeFollowerParams, GainReductionParams,
};

/// Kokoro forward-STFT graph builders (full / magnitude / phase) and the fixed
/// STFT configuration constants used by TTS verification.
pub use ny_propagate::network::dsp::{
    build_kokoro_forward_stft_full_graph, build_kokoro_forward_stft_magnitude_graph,
    build_kokoro_forward_stft_phase_graph, KOKORO_STFT_FREQ_BINS, KOKORO_STFT_HOP,
    KOKORO_STFT_MAG_EPS, KOKORO_STFT_N_FFT, KOKORO_STFT_PAD,
};

/// Build the Qwen3 speaker-encoder mel-128 spectrogram graph.
pub use ny_propagate::network::dsp::build_qwen3_mel128_graph;
