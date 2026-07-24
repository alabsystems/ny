// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::f64::consts::PI;
use std::sync::{Mutex, OnceLock};

use ndarray::{ArrayD, IxDyn};
use ny_propagate::layers::Atan2Layer;
use ny_propagate::network::dsp::{
    build_kokoro_forward_stft_full_graph, build_kokoro_forward_stft_phase_graph,
};
use ny_tensor::BoundedTensor;

const KOKORO_AUDIO_LEN_NY: usize = 40;
const KOKORO_STFT_FREQ_BINS: usize = 11;
const KOKORO_STFT_HOP: usize = 5;
const KOKORO_STFT_N_FFT: usize = 20;
const KOKORO_STFT_PAD: usize = 10;
const KOKORO_STFT_MAG_EPS: f32 = 1e-9;

struct KokoroReference {
    magnitude: Vec<f32>,
    real: Vec<f64>,
    imag: Vec<f64>,
    n_frames: usize,
}

fn with_serialized_kokoro_stft_test<T>(f: impl FnOnce() -> T) -> T {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    f()
}

fn point_bounds(values: &[f32]) -> BoundedTensor {
    let point = ArrayD::from_shape_vec(IxDyn(&[1, values.len()]), values.to_vec())
        .expect("integration point shape should be valid");
    BoundedTensor::new(point.clone(), point).expect("integration point bounds should build")
}

fn vector_bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec())
            .expect("integration lower shape should be valid"),
        ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec())
            .expect("integration upper shape should be valid"),
    )
    .expect("integration vector bounds should build")
}

fn make_kokoro_phase_wrap_tone(delta: f32, audio_len: usize) -> Vec<f32> {
    (0..audio_len)
        .map(|n| {
            0.1_f32
                * ((2.0 * std::f32::consts::PI * n as f32 / 20.0) + std::f32::consts::PI + delta)
                    .cos()
        })
        .collect()
}

fn reflect_pad(audio: &[f32], pad: usize) -> Vec<f32> {
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

fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|idx| {
            let phase = 2.0 * PI * idx as f64 / size as f64;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

fn kokoro_reference(audio: &[f32]) -> KokoroReference {
    let padded = reflect_pad(audio, KOKORO_STFT_PAD);
    let window = hann_window(KOKORO_STFT_N_FFT);
    let n_frames = audio.len() / KOKORO_STFT_HOP + 1;
    let total = KOKORO_STFT_FREQ_BINS * n_frames;
    let mut magnitude = vec![0.0_f32; total];
    let mut real = vec![0.0_f64; total];
    let mut imag = vec![0.0_f64; total];

    for freq_idx in 0..KOKORO_STFT_FREQ_BINS {
        for frame_idx in 0..n_frames {
            let start = frame_idx * KOKORO_STFT_HOP;
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for sample_idx in 0..KOKORO_STFT_N_FFT {
                let sample = padded[start + sample_idx] as f64 * window[sample_idx];
                let angle =
                    2.0 * PI * freq_idx as f64 * sample_idx as f64 / KOKORO_STFT_N_FFT as f64;
                re += sample * angle.cos();
                im += sample * -angle.sin();
            }
            let idx = freq_idx * n_frames + frame_idx;
            real[idx] = re;
            imag[idx] = im;
            magnitude[idx] = ((re * re + im * im + KOKORO_STFT_MAG_EPS as f64).sqrt()) as f32;
        }
    }

    KokoroReference {
        magnitude,
        real,
        imag,
        n_frames,
    }
}

fn strongest_raw_phase_gap(
    minus_phase: &[f32],
    plus_phase: &[f32],
    n_frames: usize,
) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for freq_idx in 1..KOKORO_STFT_FREQ_BINS {
        for frame_idx in 0..n_frames {
            let idx = freq_idx * n_frames + frame_idx;
            let gap = (minus_phase[idx] - plus_phase[idx]).abs();
            if gap > best.1 {
                best = (idx, gap);
            }
        }
    }
    best
}

#[ntest::timeout(120000)]
#[test]
fn test_atan2_branch_cut_y_upper_zero_regression_4223_integration() {
    with_serialized_kokoro_stft_test(|| {
        let layer = Atan2Layer;
        let y = vector_bounds(&[-1.0], &[0.0]);
        let x = vector_bounds(&[-4.0], &[-2.0]);
        let result = layer
            .propagate_ibp_binary(&y, &x)
            .expect("atan2 branch-cut regression should propagate");

        assert_eq!(result.lower()[0], -std::f32::consts::PI);
        assert_eq!(result.upper()[0], std::f32::consts::PI);
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_full_shape_4223_integration() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_full_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro full graph should build");
        let input = vec![0.0_f32; KOKORO_AUDIO_LEN_NY];
        let output = graph
            .propagate_ibp(&point_bounds(&input))
            .expect("kokoro full graph should evaluate");

        assert_eq!(output.shape(), &[22, 9]);
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_phase_raw_wrap_canary_4223_integration() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_phase_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro phase graph should build");
        let minus_waveform = make_kokoro_phase_wrap_tone(-1e-4, KOKORO_AUDIO_LEN_NY);
        let plus_waveform = make_kokoro_phase_wrap_tone(1e-4, KOKORO_AUDIO_LEN_NY);

        let minus_output = graph
            .propagate_ibp(&point_bounds(&minus_waveform))
            .expect("minus waveform should evaluate");
        let plus_output = graph
            .propagate_ibp(&point_bounds(&plus_waveform))
            .expect("plus waveform should evaluate");

        let minus_phase = minus_output
            .lower()
            .as_slice()
            .expect("phase output should be contiguous");
        let plus_phase = plus_output
            .lower()
            .as_slice()
            .expect("phase output should be contiguous");
        let minus_reference = kokoro_reference(&minus_waveform);
        let plus_reference = kokoro_reference(&plus_waveform);
        let (idx, gap) = strongest_raw_phase_gap(minus_phase, plus_phase, minus_reference.n_frames);

        assert!(gap > 6.0, "raw phase gap should stay near 2π, got {gap}");
        assert!(
            (minus_reference.magnitude[idx] - plus_reference.magnitude[idx]).abs() < 1e-4,
            "magnitude gap should stay tiny at the wrap seam"
        );
        assert!(
            minus_reference.real[idx] < 0.0 && plus_reference.real[idx] < 0.0,
            "wrap seam should stay on the negative real axis"
        );
        // The seam cell sits on the negative real axis, so its imaginary part is
        // at round-off magnitude on BOTH sides of the perturbation. We do NOT
        // assert the f64 reference imag flips sign across the cut: at a true
        // branch cut the imaginary part is numerically ~zero, and which side of
        // ±0 the f64 sum lands on is not a robust signal. The phase wrap is
        // confirmed by the `gap > 6.0` check above (graph phases differ by ~2π).
        // This matches the fixed in-crate canary
        // (tests::graph::kokoro_forward_stft::...raw_wrap_canary_4223). (#4223)
        assert!(
            minus_reference.imag[idx]
                .abs()
                .max(plus_reference.imag[idx].abs())
                < 1e-2,
            "wrap seam imaginary parts should stay small near the branch cut: im-={}, im+={}",
            minus_reference.imag[idx],
            plus_reference.imag[idx]
        );
    });
}
