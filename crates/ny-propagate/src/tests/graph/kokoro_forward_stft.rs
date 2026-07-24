// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph tests for the local Kokoro forward-STFT DSP bridge (#4215, #4223).

use std::f64::consts::PI;
use std::sync::{Mutex, OnceLock};

use crate::network::dsp::{
    build_kokoro_forward_stft_full_graph, build_kokoro_forward_stft_magnitude_graph,
    build_kokoro_forward_stft_phase_graph, kokoro_forward_stft_frame_count, KOKORO_STFT_FREQ_BINS,
    KOKORO_STFT_HOP, KOKORO_STFT_MAG_EPS, KOKORO_STFT_N_FFT, KOKORO_STFT_PAD,
};
use crate::*;
use ndarray::{ArrayD, IxDyn};

const KOKORO_AUDIO_LEN_NY: usize = 40;
const KOKORO_EPSILON_NY: f32 = 1e-3;

struct KokoroStftReference {
    magnitude: Vec<f32>,
    phase: Vec<f32>,
    real: Vec<f64>,
    imag: Vec<f64>,
    n_frames: usize,
}

struct RawPhaseGap {
    idx: usize,
    gap: f32,
}

fn with_serialized_kokoro_stft_test<T>(f: impl FnOnce() -> T) -> T {
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
            0.045 * (2.0 * std::f32::consts::PI * 2.0 * t).sin()
                + 0.030 * (2.0 * std::f32::consts::PI * 5.0 * t).cos()
                + 0.015 * (2.0 * std::f32::consts::PI * 9.0 * t).sin()
        })
        .collect()
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

fn waveform_bounds(audio: &[f32], epsilon: f32) -> BoundedTensor {
    let lower: Vec<f32> = audio.iter().map(|&x| x - epsilon).collect();
    let upper: Vec<f32> = audio.iter().map(|&x| x + epsilon).collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, audio.len()]), lower)
            .expect("kokoro waveform lower bounds should be shaped correctly"),
        ArrayD::from_shape_vec(IxDyn(&[1, audio.len()]), upper)
            .expect("kokoro waveform upper bounds should be shaped correctly"),
    )
    .expect("kokoro waveform epsilon box should build")
}

fn point_bounds(audio: &[f32]) -> BoundedTensor {
    let point = ArrayD::from_shape_vec(IxDyn(&[1, audio.len()]), audio.to_vec())
        .expect("kokoro waveform point shape should be valid");
    BoundedTensor::new(point.clone(), point).expect("kokoro point bounds should build")
}

fn reflect_pad(audio: &[f32], pad: usize) -> Vec<f32> {
    assert!(
        audio.len() > pad,
        "kokoro reference reflect pad requires audio_len > pad"
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

fn hann_window_f64(size: usize) -> Vec<f64> {
    (0..size)
        .map(|idx| {
            let phase = 2.0 * PI * idx as f64 / size as f64;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

fn kokoro_forward_stft_reference(audio: &[f32]) -> KokoroStftReference {
    let padded = reflect_pad(audio, KOKORO_STFT_PAD);
    let window = hann_window_f64(KOKORO_STFT_N_FFT);
    let n_frames = kokoro_forward_stft_frame_count(audio.len()).expect("reference frame count");
    let total = KOKORO_STFT_FREQ_BINS * n_frames;
    let mut magnitude = vec![0.0_f32; total];
    let mut phase = vec![0.0_f32; total];
    let mut real = vec![0.0_f64; total];
    let mut imag = vec![0.0_f64; total];

    for freq_idx in 0..KOKORO_STFT_FREQ_BINS {
        for frame_idx in 0..n_frames {
            let start = frame_idx * KOKORO_STFT_HOP;
            let mut real_sum = 0.0_f64;
            let mut imag_sum = 0.0_f64;
            for sample_idx in 0..KOKORO_STFT_N_FFT {
                let sample = padded[start + sample_idx] as f64 * window[sample_idx];
                let angle =
                    2.0 * PI * freq_idx as f64 * sample_idx as f64 / KOKORO_STFT_N_FFT as f64;
                real_sum += sample * angle.cos();
                imag_sum += sample * -angle.sin();
            }

            let idx = freq_idx * n_frames + frame_idx;
            real[idx] = real_sum;
            imag[idx] = imag_sum;
            magnitude[idx] =
                ((real_sum * real_sum + imag_sum * imag_sum + KOKORO_STFT_MAG_EPS as f64).sqrt())
                    as f32;
            phase[idx] = imag_sum.atan2(real_sum) as f32;
        }
    }

    KokoroStftReference {
        magnitude,
        phase,
        real,
        imag,
        n_frames,
    }
}

fn flatten_point_output(bounds: &BoundedTensor) -> &[f32] {
    bounds
        .lower()
        .as_slice()
        .expect("point output should be contiguous")
}

fn assert_close_tensor(actual: &BoundedTensor, expected: &[f32], tol: f32, label: &str) {
    let lower = actual
        .lower()
        .as_slice()
        .expect("kokoro lower bounds should be contiguous");
    let upper = actual
        .upper()
        .as_slice()
        .expect("kokoro upper bounds should be contiguous");
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
        assert!(
            (lo - expected_value).abs() <= tol,
            "{label}: lower[{idx}]={lo} expected {expected_value} tol {tol}"
        );
        assert!(
            (hi - expected_value).abs() <= tol,
            "{label}: upper[{idx}]={hi} expected {expected_value} tol {tol}"
        );
    }
}

/// Smallest absolute difference between two angles, accounting for the 2π
/// wrap. Phase is a circular quantity: +π and -π denote the *same* angle. At a
/// branch-cut cell (real < 0, imag ≈ 0) the f32 graph and the f64 scalar
/// reference can legitimately land on opposite sides of the cut, so a raw
/// |a - b| comparison spuriously reports a ~2π error. (#4223)
fn angular_diff(a: f32, b: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut d = (a - b).rem_euclid(two_pi);
    if d > std::f32::consts::PI {
        d -= two_pi;
    }
    d.abs()
}

/// Compare a phase tensor against a scalar reference using angular distance.
fn assert_close_phase_tensor(actual: &BoundedTensor, expected: &[f32], tol: f32, label: &str) {
    let lower = actual
        .lower()
        .as_slice()
        .expect("kokoro lower bounds should be contiguous");
    let upper = actual
        .upper()
        .as_slice()
        .expect("kokoro upper bounds should be contiguous");
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
        assert!(
            angular_diff(lo, expected_value) <= tol,
            "{label}: lower[{idx}]={lo} expected {expected_value} (angular) tol {tol}"
        );
        assert!(
            angular_diff(hi, expected_value) <= tol,
            "{label}: upper[{idx}]={hi} expected {expected_value} (angular) tol {tol}"
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

fn sample_waveform(
    audio: &[f32],
    epsilon: f32,
    sample_idx: usize,
    total_samples: usize,
) -> Vec<f32> {
    let t = sample_idx as f32 / total_samples as f32;
    audio
        .iter()
        .enumerate()
        .map(|(idx, &center)| {
            let phase = ((t + idx as f32 * 0.31) % 1.0).clamp(0.0, 1.0);
            (center - epsilon) + phase * (2.0 * epsilon)
        })
        .collect()
}

fn strongest_raw_phase_gap(
    minus_phase: &[f32],
    plus_phase: &[f32],
    n_frames: usize,
) -> RawPhaseGap {
    let mut best = RawPhaseGap {
        idx: 0,
        gap: f32::NEG_INFINITY,
    };
    for freq_idx in 1..KOKORO_STFT_FREQ_BINS {
        for frame_idx in 0..n_frames {
            let idx = freq_idx * n_frames + frame_idx;
            let gap = (minus_phase[idx] - plus_phase[idx]).abs();
            if gap > best.gap {
                best = RawPhaseGap { idx, gap };
            }
        }
    }
    best
}

fn assert_raw_phase_wrap_canary(
    minus_phase: &[f32],
    plus_phase: &[f32],
    minus_reference: &KokoroStftReference,
    plus_reference: &KokoroStftReference,
) {
    let best = strongest_raw_phase_gap(minus_phase, plus_phase, minus_reference.n_frames);
    let idx = best.idx;
    let mag_gap = (minus_reference.magnitude[idx] - plus_reference.magnitude[idx]).abs();

    assert!(
        best.gap > 6.0,
        "raw phase gap should stay near 2π, got {} at index {idx}",
        best.gap
    );
    assert!(
        mag_gap < 1e-4,
        "magnitude gap should stay tiny at the wrap seam, got {mag_gap} at index {idx}"
    );
    assert!(
        minus_reference.real[idx] < 0.0 && plus_reference.real[idx] < 0.0,
        "reference seam cell should stay on the negative real axis: re-={}, re+={}",
        minus_reference.real[idx],
        plus_reference.real[idx]
    );
    // The seam cell sits on the negative real axis, so its imaginary part is at
    // round-off magnitude (~1e-16) on BOTH sides of the perturbation. We do NOT
    // assert the f64 reference imag flips sign across the cut: at a true branch
    // cut the imaginary part is numerically zero, and which side of ±0 the f64
    // sum lands on is not a robust, meaningful signal. The phase wrap is instead
    // confirmed by `best.gap > 6.0` above (the graph phases differ by ~2π) and by
    // the angular agreement with the reference below. (#4223)
    assert!(
        minus_reference.imag[idx]
            .abs()
            .max(plus_reference.imag[idx].abs())
            < 1e-2,
        "reference seam cell imaginary parts should stay small near the branch cut: im-={}, im+={}",
        minus_reference.imag[idx],
        plus_reference.imag[idx]
    );
    // Compare graph vs scalar reference by angular distance: ±π are the same
    // angle, so a branch-cut sign difference must not register as a ~2π error.
    let minus_graph_ang = angular_diff(minus_phase[idx], minus_reference.phase[idx]);
    let plus_graph_ang = angular_diff(plus_phase[idx], plus_reference.phase[idx]);
    assert!(
        minus_graph_ang < 1e-4 && plus_graph_ang < 1e-4,
        "graph phase should match the scalar reference at the seam (angular): err-={minus_graph_ang}, err+={plus_graph_ang}"
    );
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_magnitude_shape_matches_formula_4215() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_magnitude_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro magnitude graph should build");
        let output = graph
            .propagate_ibp(&point_bounds(&[0.0_f32; KOKORO_AUDIO_LEN_NY]))
            .expect("kokoro magnitude graph shape smoke should run");

        assert_eq!(
            output.shape(),
            &[11, 9],
            "40-sample waveform should map to [11, 9], got {:?}",
            output.shape()
        );
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_concrete_matches_reference_4215() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_magnitude_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro magnitude graph should build");
        let waveform = deterministic_waveform(KOKORO_AUDIO_LEN_NY);
        let output = graph
            .propagate_ibp(&point_bounds(&waveform))
            .expect("kokoro magnitude concrete evaluation should succeed");
        let reference = kokoro_forward_stft_reference(&waveform);

        assert_close_tensor(
            &output,
            &reference.magnitude,
            1e-4,
            "kokoro magnitude parity",
        );
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_ibp_contains_concrete_4215() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_magnitude_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro magnitude graph should build");
        let waveform = deterministic_waveform(KOKORO_AUDIO_LEN_NY);
        let concrete = graph
            .propagate_ibp(&point_bounds(&waveform))
            .expect("kokoro magnitude concrete evaluation should succeed");
        let ibp = graph
            .propagate_ibp(&waveform_bounds(&waveform, KOKORO_EPSILON_NY))
            .expect("kokoro magnitude IBP bounds should succeed");

        assert_concrete_contained(&concrete, &ibp, "kokoro magnitude containment");
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_crown_width_small_4215() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_magnitude_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro magnitude graph should build");
        let waveform = deterministic_waveform(KOKORO_AUDIO_LEN_NY);
        let input = waveform_bounds(&waveform, KOKORO_EPSILON_NY);

        let crown_bounds = graph
            .propagate_crown(&input)
            .expect("kokoro magnitude CROWN should succeed");
        let ibp_bounds = graph
            .propagate_ibp(&input)
            .expect("kokoro magnitude IBP should succeed");

        // CROWN should be no wider than IBP for every element
        let crown_lower = crown_bounds
            .lower()
            .as_slice()
            .expect("CROWN lower contiguous");
        let crown_upper = crown_bounds
            .upper()
            .as_slice()
            .expect("CROWN upper contiguous");
        let ibp_lower = ibp_bounds.lower().as_slice().expect("IBP lower contiguous");
        let ibp_upper = ibp_bounds.upper().as_slice().expect("IBP upper contiguous");

        for idx in 0..crown_lower.len() {
            let crown_width = crown_upper[idx] - crown_lower[idx];
            let ibp_width = ibp_upper[idx] - ibp_lower[idx];
            assert!(
                crown_width <= ibp_width + 1e-4,
                "CROWN should be no wider than IBP at idx {idx}: CROWN={crown_width:.6}, IBP={ibp_width:.6}"
            );
            // CROWN width budget: magnitude output for small epsilon around SineGen
            // amplitude should stay reasonable (< 0.5 per element)
            assert!(
                crown_width < 0.5,
                "CROWN width {crown_width:.6} at idx {idx} exceeds budget 0.5"
            );
        }
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_phase_shape_matches_formula_4223() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_phase_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro phase graph should build");
        let output = graph
            .propagate_ibp(&point_bounds(&[0.0_f32; KOKORO_AUDIO_LEN_NY]))
            .expect("kokoro phase graph shape smoke should run");

        assert_eq!(
            output.shape(),
            &[11, 9],
            "40-sample waveform should map to [11, 9], got {:?}",
            output.shape()
        );
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_phase_concrete_matches_reference_4223() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_phase_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro phase graph should build");
        let waveform = deterministic_waveform(KOKORO_AUDIO_LEN_NY);
        let output = graph
            .propagate_ibp(&point_bounds(&waveform))
            .expect("kokoro phase graph concrete evaluation should succeed");
        let reference = kokoro_forward_stft_reference(&waveform);

        assert_close_phase_tensor(&output, &reference.phase, 1e-4, "kokoro phase parity");
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_phase_ibp_contains_concrete_4223() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_phase_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro phase graph should build");
        let waveform = deterministic_waveform(KOKORO_AUDIO_LEN_NY);
        let concrete = graph
            .propagate_ibp(&point_bounds(&waveform))
            .expect("kokoro phase concrete point evaluation should succeed");
        let ibp = graph
            .propagate_ibp(&waveform_bounds(&waveform, KOKORO_EPSILON_NY))
            .expect("kokoro phase IBP bounds should succeed");

        assert_concrete_contained(&concrete, &ibp, "kokoro phase containment");
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_phase_ibp_soundness_sampling_4223() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_phase_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro phase graph should build");
        let waveform = deterministic_waveform(KOKORO_AUDIO_LEN_NY);
        let ibp = graph
            .propagate_ibp(&waveform_bounds(&waveform, KOKORO_EPSILON_NY))
            .expect("kokoro phase IBP bounds should succeed");

        for sample_idx in 0..100 {
            let sample = sample_waveform(&waveform, KOKORO_EPSILON_NY, sample_idx, 100);
            let concrete = graph
                .propagate_ibp(&point_bounds(&sample))
                .expect("sampled kokoro phase evaluation should succeed");
            assert_concrete_contained(
                &concrete,
                &ibp,
                &format!("kokoro phase sample containment {sample_idx}"),
            );
        }
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_full_concrete_matches_reference_4223() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_full_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro full graph should build");
        let waveform = deterministic_waveform(KOKORO_AUDIO_LEN_NY);
        let output = graph
            .propagate_ibp(&point_bounds(&waveform))
            .expect("kokoro full graph concrete evaluation should succeed");
        let reference = kokoro_forward_stft_reference(&waveform);

        // The full surface concatenates magnitude (linear) then phase (circular).
        // Compare each half with the appropriate metric: magnitude with absolute
        // tolerance, phase with angular distance so branch-cut cells (±π) match. (#4223)
        let mag_len = reference.magnitude.len();
        let lower = output
            .lower()
            .as_slice()
            .expect("kokoro full lower contiguous")
            .to_vec();
        let upper = output
            .upper()
            .as_slice()
            .expect("kokoro full upper contiguous")
            .to_vec();
        let magnitude_half = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[mag_len]), lower[..mag_len].to_vec())
                .expect("magnitude lower half"),
            ArrayD::from_shape_vec(IxDyn(&[mag_len]), upper[..mag_len].to_vec())
                .expect("magnitude upper half"),
        )
        .expect("magnitude half bounds");
        let phase_half = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[lower.len() - mag_len]), lower[mag_len..].to_vec())
                .expect("phase lower half"),
            ArrayD::from_shape_vec(IxDyn(&[upper.len() - mag_len]), upper[mag_len..].to_vec())
                .expect("phase upper half"),
        )
        .expect("phase half bounds");

        assert_close_tensor(
            &magnitude_half,
            &reference.magnitude,
            1e-4,
            "kokoro full magnitude parity",
        );
        assert_close_phase_tensor(
            &phase_half,
            &reference.phase,
            1e-4,
            "kokoro full phase parity",
        );
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_full_shape_matches_production_4223() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_full_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro full graph should build");
        let output = graph
            .propagate_ibp(&point_bounds(&[0.0_f32; KOKORO_AUDIO_LEN_NY]))
            .expect("kokoro full graph shape smoke should run");

        assert_eq!(
            output.shape(),
            &[22, 9],
            "40-sample waveform should map to [22, 9], got {:?}",
            output.shape()
        );
    });
}

#[ntest::timeout(120000)]
#[test]
fn test_kokoro_forward_stft_phase_raw_wrap_canary_4223() {
    with_serialized_kokoro_stft_test(|| {
        let graph = build_kokoro_forward_stft_phase_graph(KOKORO_AUDIO_LEN_NY)
            .expect("kokoro phase graph should build");
        let minus_waveform = make_kokoro_phase_wrap_tone(-1e-4, KOKORO_AUDIO_LEN_NY);
        let plus_waveform = make_kokoro_phase_wrap_tone(1e-4, KOKORO_AUDIO_LEN_NY);

        let minus_output = graph
            .propagate_ibp(&point_bounds(&minus_waveform))
            .expect("minus phase graph evaluation should succeed");
        let plus_output = graph
            .propagate_ibp(&point_bounds(&plus_waveform))
            .expect("plus phase graph evaluation should succeed");

        let minus_phase = flatten_point_output(&minus_output);
        let plus_phase = flatten_point_output(&plus_output);
        let minus_reference = kokoro_forward_stft_reference(&minus_waveform);
        let plus_reference = kokoro_forward_stft_reference(&plus_waveform);
        assert_raw_phase_wrap_canary(minus_phase, plus_phase, &minus_reference, &plus_reference);
    });
}
