// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph tests for the local Qwen3 mel128 DSP bridge (#3719).

use std::f64::consts::PI;
use std::sync::{Mutex, OnceLock};

use crate::network::dsp::{
    build_qwen3_mel128_graph, qwen3_mel128_frame_count, QWEN3_SPEAKER_FREQ_BINS, QWEN3_SPEAKER_HOP,
    QWEN3_SPEAKER_LOG_FLOOR, QWEN3_SPEAKER_MAG_EPS, QWEN3_SPEAKER_MELS, QWEN3_SPEAKER_N_FFT,
    QWEN3_SPEAKER_PAD,
};
use crate::*;
use ndarray::{ArrayD, IxDyn};

const MEL128_AUDIO_LEN_NY: usize = 1500;

fn with_serialized_mel128_test<T>(f: impl FnOnce() -> T) -> T {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    f()
}

fn deterministic_waveform(audio_len: usize) -> Vec<f32> {
    (0..audio_len)
        .map(|idx| {
            let t = idx as f32 / audio_len as f32;
            0.25 * (2.0 * std::f32::consts::PI * 3.0 * t).sin()
                + 0.10 * (2.0 * std::f32::consts::PI * 7.0 * t).cos()
                + 0.05 * (2.0 * std::f32::consts::PI * 13.0 * t).sin()
        })
        .collect()
}

fn waveform_bounds(audio: &[f32], epsilon: f32) -> BoundedTensor {
    let center = ArrayD::from_shape_vec(IxDyn(&[1, audio.len()]), audio.to_vec())
        .expect("mel128 waveform center shape should be valid");
    BoundedTensor::from_epsilon(center, epsilon).expect("mel128 waveform epsilon box should build")
}

fn point_bounds(audio: &[f32]) -> BoundedTensor {
    let point = ArrayD::from_shape_vec(IxDyn(&[1, audio.len()]), audio.to_vec())
        .expect("mel128 waveform point shape should be valid");
    BoundedTensor::new(point.clone(), point).expect("mel128 point bounds should build")
}

fn assert_close_tensor(actual: &BoundedTensor, expected: &[f32], tol: f32, label: &str) {
    let lower = actual
        .lower()
        .as_slice()
        .expect("mel128 lower bounds should be contiguous");
    let upper = actual
        .upper()
        .as_slice()
        .expect("mel128 upper bounds should be contiguous");
    assert_eq!(
        lower.len(),
        expected.len(),
        "{label}: output length mismatch {} != {}",
        lower.len(),
        expected.len()
    );

    for (idx, ((&lo, &hi), &expected_value)) in lower
        .iter()
        .zip(upper.iter())
        .zip(expected.iter())
        .enumerate()
    {
        let scale = expected_value.abs().max(1.0);
        let limit = tol * scale;
        assert!(
            (lo - expected_value).abs() <= limit,
            "{label}: lower[{idx}]={lo} expected {expected_value} tol {limit}"
        );
        assert!(
            (hi - expected_value).abs() <= limit,
            "{label}: upper[{idx}]={hi} expected {expected_value} tol {limit}"
        );
    }
}

fn assert_concrete_contained(concrete: &BoundedTensor, bounds: &BoundedTensor, label: &str) {
    let concrete_flat = concrete.flatten();
    let bounds_flat = bounds.flatten();
    for (idx, ((&val, &lo), &hi)) in concrete_flat
        .lower()
        .iter()
        .zip(bounds_flat.lower().iter())
        .zip(bounds_flat.upper().iter())
        .enumerate()
    {
        let scale = val.abs().max(lo.abs()).max(hi.abs()).max(1.0);
        let tol = 1e-5 * scale;
        assert!(
            val >= lo - tol,
            "{label}: concrete[{idx}]={val} below lower {lo} tol {tol}"
        );
        assert!(
            val <= hi + tol,
            "{label}: concrete[{idx}]={val} above upper {hi} tol {tol}"
        );
    }
}

fn reflect_pad(audio: &[f32], pad: usize) -> Vec<f32> {
    assert!(
        audio.len() > pad,
        "mel128 reference reflect pad requires audio_len > pad"
    );

    let mut padded = Vec::with_capacity(audio.len() + 2 * pad);
    for idx in (1..=pad).rev() {
        padded.push(audio[idx]);
    }
    padded.extend_from_slice(audio);
    for idx in 0..pad {
        padded.push(audio[audio.len() - 2 - idx]);
    }
    padded
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
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = 1000.0 / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = 1000.0 / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if mel < min_log_mel {
        f_sp * mel
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

fn mel_filterbank() -> Vec<f32> {
    let sample_rate_hz = 24_000.0_f64;
    let n_freqs = QWEN3_SPEAKER_FREQ_BINS;
    let freqs: Vec<f64> = (0..n_freqs)
        .map(|idx| idx as f64 * sample_rate_hz / QWEN3_SPEAKER_N_FFT as f64)
        .collect();
    let min_mel = hz_to_mel_slaney(0.0);
    let max_mel = hz_to_mel_slaney(sample_rate_hz / 2.0);
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

fn qwen3_mel128_reference(audio: &[f32]) -> Vec<f32> {
    let padded = reflect_pad(audio, QWEN3_SPEAKER_PAD);
    let window = hann_window_f64(QWEN3_SPEAKER_N_FFT);
    let filters = mel_filterbank();
    let n_frames = qwen3_mel128_frame_count(audio.len()).expect("reference frame count");
    let mut out = vec![0.0_f32; n_frames * QWEN3_SPEAKER_MELS];

    for frame_idx in 0..n_frames {
        let start = frame_idx * QWEN3_SPEAKER_HOP;
        let mut magnitudes = vec![0.0_f32; QWEN3_SPEAKER_FREQ_BINS];
        for (freq_idx, mag_slot) in magnitudes.iter_mut().enumerate() {
            let mut real = 0.0_f64;
            let mut imag = 0.0_f64;
            for sample_idx in 0..QWEN3_SPEAKER_N_FFT {
                let sample = padded[start + sample_idx] as f64 * window[sample_idx] as f64;
                let angle =
                    2.0 * PI * freq_idx as f64 * sample_idx as f64 / QWEN3_SPEAKER_N_FFT as f64;
                real += sample * angle.cos();
                imag += sample * -angle.sin();
            }
            *mag_slot = ((real * real + imag * imag + QWEN3_SPEAKER_MAG_EPS as f64).sqrt()) as f32;
        }

        for mel_idx in 0..QWEN3_SPEAKER_MELS {
            let mut sum = 0.0_f64;
            for freq_idx in 0..QWEN3_SPEAKER_FREQ_BINS {
                sum += magnitudes[freq_idx] as f64
                    * filters[mel_idx * QWEN3_SPEAKER_FREQ_BINS + freq_idx] as f64;
            }
            out[frame_idx * QWEN3_SPEAKER_MELS + mel_idx] =
                (sum as f32).max(QWEN3_SPEAKER_LOG_FLOOR).ln();
        }
    }

    out
}

#[ntest::timeout(120000)]
#[test]
fn test_qwen3_mel128_graph_shape_matches_formula_3719() {
    with_serialized_mel128_test(|| {
        let graph =
            build_qwen3_mel128_graph(MEL128_AUDIO_LEN_NY).expect("mel128 graph should build");
        let input = point_bounds(&vec![0.0_f32; MEL128_AUDIO_LEN_NY]);
        let output = graph
            .propagate_ibp(&input)
            .expect("mel128 graph shape smoke should run");

        assert_eq!(
            output.shape(),
            &[5, 128],
            "1500-sample waveform should map to [5, 128], got {:?}",
            output.shape()
        );
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_qwen3_mel128_graph_concrete_matches_reference_3719() {
    with_serialized_mel128_test(|| {
        let graph =
            build_qwen3_mel128_graph(MEL128_AUDIO_LEN_NY).expect("mel128 graph should build");
        let waveform = deterministic_waveform(MEL128_AUDIO_LEN_NY);
        let input = point_bounds(&waveform);
        let output = graph
            .propagate_ibp(&input)
            .expect("mel128 concrete graph evaluation should succeed");
        let expected = qwen3_mel128_reference(&waveform);

        assert_close_tensor(&output, &expected, 5e-3, "mel128 concrete parity");
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_qwen3_mel128_graph_bounds_contain_concrete_3719() {
    with_serialized_mel128_test(|| {
        let graph =
            build_qwen3_mel128_graph(MEL128_AUDIO_LEN_NY).expect("mel128 graph should build");
        let waveform = deterministic_waveform(MEL128_AUDIO_LEN_NY);
        let concrete = graph
            .propagate_ibp(&point_bounds(&waveform))
            .expect("mel128 concrete point evaluation should succeed");
        let ibp = graph
            .propagate_ibp(&waveform_bounds(&waveform, 1e-3))
            .expect("mel128 IBP bounds should succeed");

        assert_concrete_contained(&concrete, &ibp, "mel128 containment");
    });
}
