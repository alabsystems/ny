// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact Qwen3-TTS mel128 frontend as a verifier-local [`GraphNetwork`].
//!
//! Source contract:
//! - `./avoice/crates/avoice-tts/src/qwen3/mel128.rs`
//! - `./avoice/crates/avoice-common/src/mel.rs`

use std::f64::consts::PI;
use std::sync::OnceLock;

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};

use crate::layers::{
    AddConstantLayer, AddLayer, ClipLayer, Conv1dLayer, Layer, LogLayer, PadLayer, PadMode,
    PowConstantLayer, SqrtLayer, TransposeLayer,
};
use crate::network::{GraphNetwork, GraphNode};

pub const QWEN3_SPEAKER_N_FFT: usize = 1024;
pub const QWEN3_SPEAKER_HOP: usize = 256;
pub const QWEN3_SPEAKER_PAD: usize = 384;
pub const QWEN3_SPEAKER_FREQ_BINS: usize = QWEN3_SPEAKER_N_FFT / 2 + 1;
pub const QWEN3_SPEAKER_MELS: usize = 128;
pub const QWEN3_SPEAKER_MAG_EPS: f32 = 1e-9;
pub const QWEN3_SPEAKER_LOG_FLOOR: f32 = 1e-5;

const QWEN3_SPEAKER_SAMPLE_RATE_HZ: f64 = 24_000.0;
const SLANEY_F_SP: f64 = 200.0 / 3.0;
const SLANEY_MIN_LOG_HZ: f64 = 1000.0;
const SLANEY_MIN_LOG_MEL: f64 = 1000.0 / (200.0 / 3.0);

/// Compute the mel-frame count for an unbatched waveform shaped `[1, audio_len]`.
pub(crate) fn qwen3_mel128_frame_count(audio_len: usize) -> Result<usize> {
    if audio_len <= QWEN3_SPEAKER_PAD {
        return Err(NyError::InvalidSpec(format!(
            "Qwen3 mel128 reflect pad requires audio_len > {}, got {}",
            QWEN3_SPEAKER_PAD, audio_len
        )));
    }

    let padded_len = audio_len
        .checked_add(2 * QWEN3_SPEAKER_PAD)
        .ok_or_else(|| NyError::InvalidSpec("Qwen3 mel128 padded length overflow".into()))?;
    if padded_len < QWEN3_SPEAKER_N_FFT {
        return Err(NyError::InvalidSpec(format!(
            "Qwen3 mel128 padded length {} smaller than n_fft {}",
            padded_len, QWEN3_SPEAKER_N_FFT
        )));
    }

    Ok((padded_len - QWEN3_SPEAKER_N_FFT) / QWEN3_SPEAKER_HOP + 1)
}

/// Build the exact local `mel128` graph used by Qwen3-TTS speaker cloning.
///
/// Input shape: `[1, audio_len]`.
/// Output shape: `[T_mel, 128]`.
pub fn build_qwen3_mel128_graph(audio_len: usize) -> Result<GraphNetwork> {
    let padded_len = audio_len
        .checked_add(2 * QWEN3_SPEAKER_PAD)
        .ok_or_else(|| NyError::InvalidSpec("Qwen3 mel128 padded length overflow".into()))?;
    let mel_frames = qwen3_mel128_frame_count(audio_len)?;

    let mut graph = GraphNetwork::new();
    add_qwen3_mel128_nodes(
        &mut graph,
        build_qwen3_stft_conv_layer(real_dft_bank(), "Qwen3 mel128 real DFT bank", padded_len)?,
        build_qwen3_stft_conv_layer(imag_dft_bank(), "Qwen3 mel128 imag DFT bank", padded_len)?,
        build_qwen3_mel_conv_layer(mel_frames)?,
    )?;
    graph.set_output("mel_time_major");
    Ok(graph)
}

fn build_qwen3_stft_conv_layer(
    weights: Vec<f32>,
    label: &str,
    padded_len: usize,
) -> Result<Conv1dLayer> {
    Conv1dLayer::with_input_length(
        weights_to_kernel(
            &[QWEN3_SPEAKER_FREQ_BINS, 1, QWEN3_SPEAKER_N_FFT],
            weights,
            label,
        )?,
        None,
        QWEN3_SPEAKER_HOP,
        0,
        padded_len,
    )
}

fn build_qwen3_mel_conv_layer(mel_frames: usize) -> Result<Conv1dLayer> {
    Conv1dLayer::with_input_length(
        weights_to_kernel(
            &[QWEN3_SPEAKER_MELS, QWEN3_SPEAKER_FREQ_BINS, 1],
            mel_filterbank(),
            "Qwen3 mel128 filterbank",
        )?,
        None,
        1,
        0,
        mel_frames,
    )
}

fn add_qwen3_mel128_nodes(
    graph: &mut GraphNetwork,
    real_conv: Conv1dLayer,
    imag_conv: Conv1dLayer,
    mel_conv: Conv1dLayer,
) -> Result<()> {
    graph.try_add_node(GraphNode::from_input(
        "reflect_pad",
        Layer::Pad(PadLayer::new(
            vec![(0, 0), (QWEN3_SPEAKER_PAD, QWEN3_SPEAKER_PAD)],
            PadMode::Reflect,
        )),
    ))?;
    graph.try_add_node(GraphNode::new(
        "stft_real",
        Layer::Conv1d(real_conv),
        vec!["reflect_pad".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "stft_imag",
        Layer::Conv1d(imag_conv),
        vec!["reflect_pad".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "stft_real_sq",
        Layer::PowConstant(PowConstantLayer::square()),
        vec!["stft_real".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "stft_imag_sq",
        Layer::PowConstant(PowConstantLayer::square()),
        vec!["stft_imag".to_string()],
    ))?;
    graph.try_add_node(GraphNode::binary(
        "stft_power",
        Layer::Add(AddLayer),
        "stft_real_sq",
        "stft_imag_sq",
    ))?;
    graph.try_add_node(GraphNode::new(
        "stft_power_eps",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(
            IxDyn(&[]),
            QWEN3_SPEAKER_MAG_EPS,
        ))),
        vec!["stft_power".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "stft_magnitude",
        Layer::Sqrt(SqrtLayer::new()),
        vec!["stft_power_eps".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "mel_filterbank",
        Layer::Conv1d(mel_conv),
        vec!["stft_magnitude".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "mel_floor",
        Layer::Clip(ClipLayer::new(QWEN3_SPEAKER_LOG_FLOOR, f32::MAX)),
        vec!["mel_filterbank".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "mel_log",
        Layer::Log(LogLayer::new()),
        vec!["mel_floor".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "mel_time_major",
        Layer::Transpose(TransposeLayer::new(vec![1, 0])),
        vec!["mel_log".to_string()],
    ))?;
    Ok(())
}

fn weights_to_kernel(shape: &[usize], weights: Vec<f32>, label: &str) -> Result<ArrayD<f32>> {
    ArrayD::from_shape_vec(IxDyn(shape), weights)
        .map_err(|err| NyError::InvalidSpec(format!("{label} shape error: {err}")))
}

fn real_dft_bank() -> Vec<f32> {
    static REAL: OnceLock<Vec<f32>> = OnceLock::new();
    REAL.get_or_init(|| build_dft_bank(false)).clone()
}

fn imag_dft_bank() -> Vec<f32> {
    static IMAG: OnceLock<Vec<f32>> = OnceLock::new();
    IMAG.get_or_init(|| build_dft_bank(true)).clone()
}

fn mel_filterbank() -> Vec<f32> {
    static MEL: OnceLock<Vec<f32>> = OnceLock::new();
    MEL.get_or_init(build_mel_filterbank).clone()
}

fn build_dft_bank(imaginary: bool) -> Vec<f32> {
    let window = hann_window_f64(QWEN3_SPEAKER_N_FFT);
    let mut weights = Vec::with_capacity(
        QWEN3_SPEAKER_FREQ_BINS
            .checked_mul(QWEN3_SPEAKER_N_FFT)
            .expect("Qwen3 mel128 DFT bank size should fit usize"),
    );

    for k in 0..QWEN3_SPEAKER_FREQ_BINS {
        for (n, &window_value) in window.iter().enumerate() {
            let angle = 2.0 * PI * k as f64 * n as f64 / QWEN3_SPEAKER_N_FFT as f64;
            let value = if imaginary { -angle.sin() } else { angle.cos() };
            weights.push((window_value as f64 * value) as f32);
        }
    }

    weights
}

fn hann_window_f64(size: usize) -> Vec<f32> {
    (0..size)
        .map(|idx| {
            let phase = 2.0 * PI * idx as f64 / size as f64;
            (0.5 * (1.0 - phase.cos())) as f32
        })
        .collect()
}

fn hz_to_mel_slaney(hz: f64) -> f64 {
    let logstep = 6.4f64.ln() / 27.0;
    if hz < SLANEY_MIN_LOG_HZ {
        hz / SLANEY_F_SP
    } else {
        SLANEY_MIN_LOG_MEL + (hz / SLANEY_MIN_LOG_HZ).ln() / logstep
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    let logstep = 6.4f64.ln() / 27.0;
    if mel < SLANEY_MIN_LOG_MEL {
        SLANEY_F_SP * mel
    } else {
        SLANEY_MIN_LOG_HZ * (logstep * (mel - SLANEY_MIN_LOG_MEL)).exp()
    }
}

fn build_mel_filterbank() -> Vec<f32> {
    let n_freqs = QWEN3_SPEAKER_FREQ_BINS;
    let freqs: Vec<f64> = (0..n_freqs)
        .map(|idx| idx as f64 * QWEN3_SPEAKER_SAMPLE_RATE_HZ / QWEN3_SPEAKER_N_FFT as f64)
        .collect();

    let min_mel = hz_to_mel_slaney(0.0);
    let max_mel = hz_to_mel_slaney(QWEN3_SPEAKER_SAMPLE_RATE_HZ / 2.0);
    let hz_points: Vec<f64> = (0..QWEN3_SPEAKER_MELS + 2)
        .map(|idx| {
            let mel = min_mel + (max_mel - min_mel) * idx as f64 / (QWEN3_SPEAKER_MELS + 1) as f64;
            mel_to_hz_slaney(mel)
        })
        .collect();

    let mut filters = vec![0.0_f32; QWEN3_SPEAKER_MELS * n_freqs];
    for mel_idx in 0..QWEN3_SPEAKER_MELS {
        let left = hz_points[mel_idx];
        let center = hz_points[mel_idx + 1];
        let right = hz_points[mel_idx + 2];
        let norm = (2.0 / (right - left)) as f32;

        for (freq_idx, &freq_hz) in freqs.iter().enumerate() {
            let rising = if center > left {
                (freq_hz - left) / (center - left)
            } else {
                0.0
            };
            let falling = if right > center {
                (right - freq_hz) / (right - center)
            } else {
                0.0
            };
            filters[mel_idx * n_freqs + freq_idx] = rising.min(falling).max(0.0) as f32 * norm;
        }
    }

    filters
}
