// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for DSP graph builders (waveshaper, biquad, gain reduction).

use super::*;
use ndarray::{arr1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

/// Tolerance for floating-point precision in soundness checks.
const FP_TOLERANCE: f32 = 1e-5;

/// Strategy to generate valid interval bounds [lower, upper] where lower <= upper.
fn valid_interval(range: f32) -> impl Strategy<Value = (f32, f32)> {
    (-range..=range)
        .prop_flat_map(move |a| (-range..=range).prop_map(move |b| (a.min(b), a.max(b))))
}

/// Sample points within an interval for soundness verification.
/// Guards: at least 2 samples (avoids div-by-zero), clamp to [lower, upper]
/// to prevent FP drift outside the interval.
fn sample_points(lower: f32, upper: f32, num_samples: usize) -> Vec<f32> {
    let (lower, upper) = if lower <= upper {
        (lower, upper)
    } else {
        (upper, lower)
    };
    if lower == upper {
        return vec![lower];
    }
    let samples = num_samples.max(2);
    let denom = (samples - 1) as f32;
    (0..samples)
        .map(|i| {
            let t = i as f32 / denom;
            let sample = lower + (upper - lower) * t;
            sample.clamp(lower, upper)
        })
        .collect()
}

// ---- Waveshaper tests ----

/// True waveshaper output for validation.
fn waveshaper_true(x: f32, drive: f32) -> f32 {
    if drive < 1e-6 {
        x
    } else {
        (drive * x).tanh() / drive.tanh()
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_waveshaper_graph_ibp_soundness() {
    for &drive in &[0.1_f32, 1.0, 2.0, 5.0, 10.0] {
        let graph = build_waveshaper_graph(drive).unwrap();
        let input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();

        for x in sample_points(-1.0, 1.0, 20) {
            let y = waveshaper_true(x, drive);
            assert!(
                ibp_bounds.lower()[[0]] - FP_TOLERANCE <= y
                    && y <= ibp_bounds.upper()[[0]] + FP_TOLERANCE,
                "IBP soundness violation: drive={drive}, x={x}, y={y}, \
                 bounds=[{}, {}]",
                ibp_bounds.lower()[[0]],
                ibp_bounds.upper()[[0]],
            );
        }

        // Key acceptance criterion: |f(x,d)| < 1 for |x| <= 1
        assert!(
            ibp_bounds.upper()[[0]] <= 1.0 + FP_TOLERANCE,
            "Upper bound exceeds 1.0 for drive={drive}: {}",
            ibp_bounds.upper()[[0]],
        );
        assert!(
            ibp_bounds.lower()[[0]] >= -1.0 - FP_TOLERANCE,
            "Lower bound below -1.0 for drive={drive}: {}",
            ibp_bounds.lower()[[0]],
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_waveshaper_graph_crown_soundness() {
    for &drive in &[0.5_f32, 1.0, 3.0, 7.0] {
        let graph = build_waveshaper_graph(drive).unwrap();
        let input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

        let crown_bounds = graph.propagate_crown(&input).unwrap();

        for x in sample_points(-1.0, 1.0, 20) {
            let y = waveshaper_true(x, drive);
            assert!(
                crown_bounds.lower()[[0]] - FP_TOLERANCE <= y
                    && y <= crown_bounds.upper()[[0]] + FP_TOLERANCE,
                "CROWN soundness violation: drive={drive}, x={x}, y={y}, \
                 bounds=[{}, {}]",
                crown_bounds.lower()[[0]],
                crown_bounds.upper()[[0]],
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_waveshaper_crown_tighter_than_ibp() {
    let drive = 3.0_f32;
    let graph = build_waveshaper_graph(drive).unwrap();
    let input =
        BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[0.5_f32]).into_dyn()).unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    let ibp_width = ibp_bounds.upper()[[0]] - ibp_bounds.lower()[[0]];
    let crown_width = crown_bounds.upper()[[0]] - crown_bounds.lower()[[0]];

    assert!(
        crown_width <= ibp_width + FP_TOLERANCE,
        "CROWN should be at least as tight as IBP: crown_width={crown_width}, \
         ibp_width={ibp_width}",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_waveshaper_identity_at_low_drive() {
    let graph = build_waveshaper_graph(1e-8).unwrap();
    let input =
        BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[0.5_f32]).into_dyn()).unwrap();

    let bounds = graph.propagate_ibp(&input).unwrap();

    assert!(
        (bounds.lower()[[0]] - (-0.5)).abs() < FP_TOLERANCE,
        "Identity lower bound: {}",
        bounds.lower()[[0]],
    );
    assert!(
        (bounds.upper()[[0]] - 0.5).abs() < FP_TOLERANCE,
        "Identity upper bound: {}",
        bounds.upper()[[0]],
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_waveshaper_multidimensional_input() {
    let drive = 2.0_f32;
    let graph = build_waveshaper_graph(drive).unwrap();

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[4]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[4]), 1.0_f32),
    )
    .unwrap();

    let bounds = graph.propagate_ibp(&input).unwrap();

    for i in 0..4 {
        let y_neg = waveshaper_true(-1.0, drive);
        let y_pos = waveshaper_true(1.0, drive);
        assert!(
            bounds.lower()[[i]] - FP_TOLERANCE <= y_neg,
            "Element {i} lower bound unsound",
        );
        assert!(
            bounds.upper()[[i]] + FP_TOLERANCE >= y_pos,
            "Element {i} upper bound unsound",
        );
    }
}

// ---- Biquad single-step tests ----

/// True biquad single-step output for validation.
fn biquad_single_step_true(
    x: f32,
    z1_prev: f32,
    z2_prev: f32,
    coeff: &BiquadCoefficients,
) -> (f32, f32, f32) {
    let y = coeff.b0 * x + z1_prev;
    let z1_new = (coeff.b1 - coeff.a1 * coeff.b0) * x - coeff.a1 * z1_prev + z2_prev;
    let z2_new = (coeff.b2 - coeff.a2 * coeff.b0) * x - coeff.a2 * z1_prev;
    (y, z1_new, z2_new)
}

/// Example peaking EQ coefficients (1kHz, Q=1, +6dB, sr=44100).
fn test_peaking_eq_coefficients() -> BiquadCoefficients {
    let a = 10.0_f32.powf(6.0 / 40.0); // ~1.413
    let w0 = 2.0 * std::f32::consts::PI * 1000.0 / 44100.0;
    let alpha = w0.sin() / (2.0 * 1.0); // Q=1
    let b0 = (1.0 + alpha * a) / (1.0 + alpha / a);
    let b1 = (-2.0 * w0.cos()) / (1.0 + alpha / a);
    let b2 = (1.0 - alpha * a) / (1.0 + alpha / a);
    let a1 = b1; // same numerator for peaking EQ
    let a2 = (1.0 - alpha / a) / (1.0 + alpha / a);
    BiquadCoefficients { b0, b1, b2, a1, a2 }
}

#[ntest::timeout(10000)]
#[test]
fn test_biquad_single_step_ibp_soundness() {
    let coeff = test_peaking_eq_coefficients();
    let graph = build_biquad_single_step_graph(&coeff).unwrap();

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -2.0, -2.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0, 2.0]).into_dyn(),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    for x in sample_points(-1.0, 1.0, 5) {
        for z1 in sample_points(-2.0, 2.0, 5) {
            for z2 in sample_points(-2.0, 2.0, 5) {
                let (y, z1_new, z2_new) = biquad_single_step_true(x, z1, z2, &coeff);
                let true_outputs = [y, z1_new, z2_new];
                let labels = ["y", "z1_new", "z2_new"];

                for (i, (&val, label)) in true_outputs.iter().zip(labels.iter()).enumerate() {
                    assert!(
                        ibp_bounds.lower()[[i]] - FP_TOLERANCE <= val
                            && val <= ibp_bounds.upper()[[i]] + FP_TOLERANCE,
                        "IBP unsound for {label}: x={x}, z1={z1}, z2={z2}, \
                         val={val}, bounds=[{}, {}]",
                        ibp_bounds.lower()[[i]],
                        ibp_bounds.upper()[[i]],
                    );
                }
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_biquad_single_step_crown_soundness() {
    let coeff = test_peaking_eq_coefficients();
    let graph = build_biquad_single_step_graph(&coeff).unwrap();

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -1.0, -1.0]).into_dyn(),
        arr1(&[0.5_f32, 1.0, 1.0]).into_dyn(),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();

    for x in sample_points(-0.5, 0.5, 5) {
        for z1 in sample_points(-1.0, 1.0, 5) {
            for z2 in sample_points(-1.0, 1.0, 5) {
                let (y, z1_new, z2_new) = biquad_single_step_true(x, z1, z2, &coeff);
                let true_outputs = [y, z1_new, z2_new];
                let labels = ["y", "z1_new", "z2_new"];

                for (i, (&val, label)) in true_outputs.iter().zip(labels.iter()).enumerate() {
                    assert!(
                        crown_bounds.lower()[[i]] - FP_TOLERANCE <= val
                            && val <= crown_bounds.upper()[[i]] + FP_TOLERANCE,
                        "CROWN unsound for {label}: x={x}, z1={z1}, z2={z2}, \
                         val={val}, bounds=[{}, {}]",
                        crown_bounds.lower()[[i]],
                        crown_bounds.upper()[[i]],
                    );
                }
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_biquad_linear_exact_bounds() {
    // For a purely linear system, IBP and CROWN should give identical bounds
    let coeff = test_peaking_eq_coefficients();
    let graph = build_biquad_single_step_graph(&coeff).unwrap();

    let input = BoundedTensor::new(
        arr1(&[-0.3_f32, -0.5, -0.5]).into_dyn(),
        arr1(&[0.3_f32, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    for i in 0..3 {
        assert!(
            (ibp_bounds.lower()[[i]] - crown_bounds.lower()[[i]]).abs() < FP_TOLERANCE,
            "IBP/CROWN lower mismatch at {i}: ibp={}, crown={}",
            ibp_bounds.lower()[[i]],
            crown_bounds.lower()[[i]],
        );
        assert!(
            (ibp_bounds.upper()[[i]] - crown_bounds.upper()[[i]]).abs() < FP_TOLERANCE,
            "IBP/CROWN upper mismatch at {i}: ibp={}, crown={}",
            ibp_bounds.upper()[[i]],
            crown_bounds.upper()[[i]],
        );
    }
}

// ---- Gain reduction tests ----

/// True gain reduction for validation.
///
/// Reference: avoice `crates/avoice-mixer/src/ducking.rs:58-86`.
/// Giannoulis et al., "Digital Dynamic Range Compressor Design" (JAES, 2012).
fn gain_reduction_true(envelope: f32, params: &GainReductionParams) -> f32 {
    if envelope <= params.threshold {
        1.0
    } else {
        let compression = ((envelope - params.threshold) / envelope).min(1.0);
        (1.0 - params.ratio * compression).max(params.max_atten)
    }
}

/// Default test parameters: -30 dB threshold, full ratio, -40 dB floor.
fn test_gain_params() -> GainReductionParams {
    GainReductionParams {
        threshold: 10.0_f32.powf(-30.0 / 20.0), // ≈ 0.0316
        ratio: 0.8,
        max_atten: 0.01, // -40 dB
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_gain_reduction_ibp_soundness() {
    let params = test_gain_params();
    let graph = build_gain_reduction_graph(&params).unwrap();

    // Full range: envelope ∈ [0, 1]
    let input =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    for e in sample_points(0.0, 1.0, 50) {
        let g = gain_reduction_true(e, &params);
        assert!(
            ibp_bounds.lower()[[0]] - FP_TOLERANCE <= g
                && g <= ibp_bounds.upper()[[0]] + FP_TOLERANCE,
            "IBP soundness violation: e={e}, gain={g}, bounds=[{}, {}]",
            ibp_bounds.lower()[[0]],
            ibp_bounds.upper()[[0]],
        );
    }

    // Acceptance criterion from #3260: gain ∈ [max_atten, 1.0]
    assert!(
        ibp_bounds.lower()[[0]] >= params.max_atten - FP_TOLERANCE,
        "Lower bound below max_atten: {}",
        ibp_bounds.lower()[[0]],
    );
    assert!(
        ibp_bounds.upper()[[0]] <= 1.0 + FP_TOLERANCE,
        "Upper bound exceeds 1.0: {}",
        ibp_bounds.upper()[[0]],
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_gain_reduction_crown_soundness() {
    let params = test_gain_params();
    let graph = build_gain_reduction_graph(&params).unwrap();

    // Narrow range above threshold for CROWN precision
    let input =
        BoundedTensor::new(arr1(&[0.05_f32]).into_dyn(), arr1(&[0.5_f32]).into_dyn()).unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();

    for e in sample_points(0.05, 0.5, 20) {
        let g = gain_reduction_true(e, &params);
        assert!(
            crown_bounds.lower()[[0]] - FP_TOLERANCE <= g
                && g <= crown_bounds.upper()[[0]] + FP_TOLERANCE,
            "CROWN soundness violation: e={e}, gain={g}, bounds=[{}, {}]",
            crown_bounds.lower()[[0]],
            crown_bounds.upper()[[0]],
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_gain_reduction_below_threshold() {
    let params = test_gain_params();
    let graph = build_gain_reduction_graph(&params).unwrap();

    // Entirely below threshold: gain should be 1.0
    let input = BoundedTensor::new(
        arr1(&[0.0_f32]).into_dyn(),
        arr1(&[params.threshold * 0.5]).into_dyn(),
    )
    .unwrap();

    let bounds = graph.propagate_ibp(&input).unwrap();

    assert!(
        (bounds.lower()[[0]] - 1.0).abs() < FP_TOLERANCE,
        "Below threshold: lower bound should be 1.0, got {}",
        bounds.lower()[[0]],
    );
    assert!(
        (bounds.upper()[[0]] - 1.0).abs() < FP_TOLERANCE,
        "Below threshold: upper bound should be 1.0, got {}",
        bounds.upper()[[0]],
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_gain_reduction_at_full_envelope() {
    let params = test_gain_params();
    let graph = build_gain_reduction_graph(&params).unwrap();

    // Point evaluation at e = 1.0
    let input =
        BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let bounds = graph.propagate_ibp(&input).unwrap();
    let expected = gain_reduction_true(1.0, &params);

    assert!(
        (bounds.lower()[[0]] - expected).abs() < FP_TOLERANCE,
        "At e=1.0: expected gain={expected}, got bounds=[{}, {}]",
        bounds.lower()[[0]],
        bounds.upper()[[0]],
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_gain_reduction_zero_ratio() {
    // With ratio = 0, gain should always be 1.0 (no compression).
    let params = GainReductionParams {
        threshold: 0.1,
        ratio: 0.0,
        max_atten: 0.01,
    };
    let graph = build_gain_reduction_graph(&params).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let bounds = graph.propagate_ibp(&input).unwrap();

    assert!(
        (bounds.lower()[[0]] - 1.0).abs() < FP_TOLERANCE,
        "Zero ratio: lower bound should be 1.0, got {}",
        bounds.lower()[[0]],
    );
    assert!(
        (bounds.upper()[[0]] - 1.0).abs() < FP_TOLERANCE,
        "Zero ratio: upper bound should be 1.0, got {}",
        bounds.upper()[[0]],
    );
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    #[ntest::timeout(60000)]
    #[test]
    fn proptest_waveshaper_ibp_soundness(
        drive in 0.01_f32..10.0,
        (l, u) in valid_interval(1.0),
    ) {
        let graph = build_waveshaper_graph(drive).unwrap();
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 8) {
            let y = waveshaper_true(x, drive);
            prop_assert!(
                ibp_bounds.lower()[[0]] - FP_TOLERANCE <= y
                    && y <= ibp_bounds.upper()[[0]] + FP_TOLERANCE,
                "IBP unsound: drive={}, x={}, y={}, bounds=[{}, {}]",
                drive, x, y, ibp_bounds.lower()[[0]], ibp_bounds.upper()[[0]],
            );
        }
    }

    #[ntest::timeout(60000)]
    #[test]
    fn proptest_waveshaper_crown_soundness(
        drive in 0.01_f32..10.0,
        (l, u) in valid_interval(1.0),
    ) {
        let graph = build_waveshaper_graph(drive).unwrap();
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let crown_bounds = graph.propagate_crown(&input).unwrap();

        for x in sample_points(l, u, 8) {
            let y = waveshaper_true(x, drive);
            prop_assert!(
                crown_bounds.lower()[[0]] - FP_TOLERANCE <= y
                    && y <= crown_bounds.upper()[[0]] + FP_TOLERANCE,
                "CROWN unsound: drive={}, x={}, y={}, bounds=[{}, {}]",
                drive, x, y, crown_bounds.lower()[[0]], crown_bounds.upper()[[0]],
            );
        }
    }

    #[ntest::timeout(60000)]
    #[test]
    fn proptest_biquad_single_step_ibp_soundness(
        // Coefficients in a stable range (small feedback)
        b0 in -2.0_f32..2.0,
        b1 in -2.0_f32..2.0,
        b2 in -2.0_f32..2.0,
        a1 in -1.5_f32..1.5,
        a2 in -0.9_f32..0.9,
        (xl, xu) in valid_interval(1.0),
        (z1l, z1u) in valid_interval(2.0),
        (z2l, z2u) in valid_interval(2.0),
    ) {
        let coeff = BiquadCoefficients { b0, b1, b2, a1, a2 };
        let graph = build_biquad_single_step_graph(&coeff).unwrap();
        let input = BoundedTensor::new(
            arr1(&[xl, z1l, z2l]).into_dyn(),
            arr1(&[xu, z1u, z2u]).into_dyn(),
        ).unwrap();

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();

        for x in sample_points(xl, xu, 3) {
            for z1 in sample_points(z1l, z1u, 3) {
                for z2 in sample_points(z2l, z2u, 3) {
                    let (y, z1_new, z2_new) = biquad_single_step_true(x, z1, z2, &coeff);
                    let vals = [y, z1_new, z2_new];
                    for (i, &v) in vals.iter().enumerate() {
                        prop_assert!(
                            ibp_bounds.lower()[[i]] - FP_TOLERANCE <= v
                                && v <= ibp_bounds.upper()[[i]] + FP_TOLERANCE,
                            "IBP unsound at output {}: val={}, bounds=[{}, {}]",
                            i, v, ibp_bounds.lower()[[i]], ibp_bounds.upper()[[i]],
                        );
                    }
                }
            }
        }
    }

    #[ntest::timeout(60000)]
    #[test]
    fn proptest_gain_reduction_ibp_soundness(
        // Threshold in (0.001, 0.5) — avoid near-zero for Reciprocal stability
        threshold in 0.001_f32..0.5,
        ratio in 0.0_f32..1.0,
        max_atten in 0.001_f32..0.5,
        (el, eu) in valid_interval(0.5),
    ) {
        // Shift interval to [0, 1] range
        let el = (el + 0.5).clamp(0.0, 1.0);
        let eu = (eu + 0.5).clamp(0.0, 1.0);
        let (el, eu) = if el <= eu { (el, eu) } else { (eu, el) };

        let params = GainReductionParams { threshold, ratio, max_atten };
        let graph = build_gain_reduction_graph(&params).unwrap();
        let input = BoundedTensor::new(
            arr1(&[el]).into_dyn(),
            arr1(&[eu]).into_dyn(),
        ).unwrap();

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();

        for e in sample_points(el, eu, 10) {
            let g = gain_reduction_true(e, &params);
            prop_assert!(
                ibp_bounds.lower()[[0]] - FP_TOLERANCE <= g
                    && g <= ibp_bounds.upper()[[0]] + FP_TOLERANCE,
                "IBP unsound: e={}, gain={}, bounds=[{}, {}], params={:?}",
                e, g, ibp_bounds.lower()[[0]], ibp_bounds.upper()[[0]], params,
            );
        }

        // Structural property: gain ∈ [max_atten, 1.0]
        prop_assert!(
            ibp_bounds.lower()[[0]] >= max_atten - FP_TOLERANCE,
            "Lower bound below max_atten: {} < {}",
            ibp_bounds.lower()[[0]], max_atten,
        );
        prop_assert!(
            ibp_bounds.upper()[[0]] <= 1.0 + FP_TOLERANCE,
            "Upper bound exceeds 1.0: {}",
            ibp_bounds.upper()[[0]],
        );
    }

    #[ntest::timeout(60000)]
    #[test]
    fn proptest_gain_reduction_crown_soundness(
        threshold in 0.001_f32..0.5,
        ratio in 0.0_f32..1.0,
        max_atten in 0.001_f32..0.5,
        (el, eu) in valid_interval(0.5),
    ) {
        // Shift interval to [0, 1] range
        let el = (el + 0.5).clamp(0.0, 1.0);
        let eu = (eu + 0.5).clamp(0.0, 1.0);
        let (el, eu) = if el <= eu { (el, eu) } else { (eu, el) };

        let params = GainReductionParams { threshold, ratio, max_atten };
        let graph = build_gain_reduction_graph(&params).unwrap();
        let input = BoundedTensor::new(
            arr1(&[el]).into_dyn(),
            arr1(&[eu]).into_dyn(),
        ).unwrap();

        let crown_bounds = graph.propagate_crown(&input).unwrap();

        for e in sample_points(el, eu, 10) {
            let g = gain_reduction_true(e, &params);
            prop_assert!(
                crown_bounds.lower()[[0]] - FP_TOLERANCE <= g
                    && g <= crown_bounds.upper()[[0]] + FP_TOLERANCE,
                "CROWN unsound: e={}, gain={}, bounds=[{}, {}], params={:?}",
                e, g, crown_bounds.lower()[[0]], crown_bounds.upper()[[0]], params,
            );
        }

        // Structural property: gain ∈ [max_atten, 1.0]
        prop_assert!(
            crown_bounds.lower()[[0]] >= max_atten - FP_TOLERANCE,
            "Lower bound below max_atten: {} < {}",
            crown_bounds.lower()[[0]], max_atten,
        );
        prop_assert!(
            crown_bounds.upper()[[0]] <= 1.0 + FP_TOLERANCE,
            "Upper bound exceeds 1.0: {}",
            crown_bounds.upper()[[0]],
        );
    }
}

// ---- Waveshaper CROWN tightness (Prover directive P1-24 F1) ----
//
// Re: Prover F1: CROWN is NOT strictly tighter than IBP for the waveshaper
// because the graph has only ONE nonlinear layer (Tanh) flanked by two
// MulConstant (linear) layers. CROWN's linear relaxation of a single
// nonlinear activation produces the same concretized bounds as IBP.
// CROWN only provides tighter bounds when composing ≥2 nonlinear layers,
// where backward-mode relaxation avoids the interval wrapping problem.
//
// This test verifies that CROWN and IBP produce equal bounds (documenting
// the expected behavior) rather than asserting strict tightness.

#[ntest::timeout(10000)]
#[test]
fn test_waveshaper_crown_equals_ibp_single_nonlinearity() {
    let drive = 3.0_f32;
    let graph = build_waveshaper_graph(drive).unwrap();
    let input =
        BoundedTensor::new(arr1(&[-0.5_f32]).into_dyn(), arr1(&[0.5_f32]).into_dyn()).unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    let ibp_width = ibp_bounds.upper()[[0]] - ibp_bounds.lower()[[0]];
    let crown_width = crown_bounds.upper()[[0]] - crown_bounds.lower()[[0]];

    // Single nonlinear layer: CROWN and IBP produce identical bounds.
    assert!(
        (crown_width - ibp_width).abs() < FP_TOLERANCE,
        "With one nonlinear layer, CROWN should match IBP: \
         crown_width={crown_width}, ibp_width={ibp_width}",
    );
}

// ---- One-pole smoother tests (Phase 2) ----

/// True one-pole step output for validation.
fn one_pole_step_true(rms: f32, envelope_prev: f32, coefficient: f32) -> f32 {
    coefficient * rms + (1.0 - coefficient) * envelope_prev
}

/// Default attack/release coefficients at sr=24000, τ_attack=5ms, τ_release=100ms.
fn test_envelope_params() -> EnvelopeFollowerParams {
    EnvelopeFollowerParams {
        c_attack: 1.0 - (-1.0 / (24000.0 * 0.005_f32)).exp(), // ≈ 0.0083
        c_release: 1.0 - (-1.0 / (24000.0 * 0.1_f32)).exp(),  // ≈ 0.00042
        gain: test_gain_params(),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_one_pole_step_ibp_soundness() {
    let params = test_envelope_params();

    for &c in &[params.c_attack, params.c_release] {
        let graph = build_one_pole_step_graph(c).unwrap();

        // rms ∈ [0, 1], envelope_prev ∈ [0, 1]
        let input = BoundedTensor::new(
            arr1(&[0.0_f32, 0.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .unwrap();

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();

        for rms in sample_points(0.0, 1.0, 10) {
            for env in sample_points(0.0, 1.0, 10) {
                let y = one_pole_step_true(rms, env, c);
                assert!(
                    ibp_bounds.lower()[[0]] - FP_TOLERANCE <= y
                        && y <= ibp_bounds.upper()[[0]] + FP_TOLERANCE,
                    "IBP unsound: c={c}, rms={rms}, env={env}, y={y}, \
                     bounds=[{}, {}]",
                    ibp_bounds.lower()[[0]],
                    ibp_bounds.upper()[[0]],
                );
            }
        }

        // Convex combination of [0,1] inputs → output must be in [0,1]
        assert!(
            ibp_bounds.lower()[[0]] >= -FP_TOLERANCE,
            "One-pole lower below 0: {}",
            ibp_bounds.lower()[[0]],
        );
        assert!(
            ibp_bounds.upper()[[0]] <= 1.0 + FP_TOLERANCE,
            "One-pole upper above 1: {}",
            ibp_bounds.upper()[[0]],
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_one_pole_step_crown_soundness() {
    let params = test_envelope_params();

    for &c in &[params.c_attack, params.c_release] {
        let graph = build_one_pole_step_graph(c).unwrap();

        let input = BoundedTensor::new(
            arr1(&[0.0_f32, 0.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .unwrap();

        let crown_bounds = graph.propagate_crown(&input).unwrap();

        for rms in sample_points(0.0, 1.0, 10) {
            for env in sample_points(0.0, 1.0, 10) {
                let y = one_pole_step_true(rms, env, c);
                assert!(
                    crown_bounds.lower()[[0]] - FP_TOLERANCE <= y
                        && y <= crown_bounds.upper()[[0]] + FP_TOLERANCE,
                    "CROWN unsound: c={c}, rms={rms}, env={env}, y={y}, \
                     bounds=[{}, {}]",
                    crown_bounds.lower()[[0]],
                    crown_bounds.upper()[[0]],
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_one_pole_linear_exact_bounds() {
    // For a purely linear system, IBP and CROWN should give identical bounds.
    let c = 0.5_f32;
    let graph = build_one_pole_step_graph(c).unwrap();

    let input = BoundedTensor::new(
        arr1(&[0.1_f32, 0.2]).into_dyn(),
        arr1(&[0.8_f32, 0.9]).into_dyn(),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    assert!(
        (ibp_bounds.lower()[[0]] - crown_bounds.lower()[[0]]).abs() < FP_TOLERANCE,
        "IBP/CROWN lower mismatch: ibp={}, crown={}",
        ibp_bounds.lower()[[0]],
        crown_bounds.lower()[[0]],
    );
    assert!(
        (ibp_bounds.upper()[[0]] - crown_bounds.upper()[[0]]).abs() < FP_TOLERANCE,
        "IBP/CROWN upper mismatch: ibp={}, crown={}",
        ibp_bounds.upper()[[0]],
        crown_bounds.upper()[[0]],
    );
}

// ---- Combined envelope follower tests (Phase 3) ----

#[ntest::timeout(10000)]
#[test]
fn test_envelope_follower_step_ibp_soundness() {
    let params = test_envelope_params();

    for &c in &[params.c_attack, params.c_release] {
        let graph = build_envelope_follower_step_graph(c, &params.gain).unwrap();

        // rms ∈ [0, 1], envelope_prev ∈ [0, 1]
        let input = BoundedTensor::new(
            arr1(&[0.0_f32, 0.0]).into_dyn(),
            arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .unwrap();

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();

        // Acceptance criterion: gain ∈ [max_atten, 1.0]
        assert!(
            ibp_bounds.lower()[[0]] >= params.gain.max_atten - FP_TOLERANCE,
            "Gain below max_atten: {} < {}",
            ibp_bounds.lower()[[0]],
            params.gain.max_atten,
        );
        assert!(
            ibp_bounds.upper()[[0]] <= 1.0 + FP_TOLERANCE,
            "Gain above 1.0: {}",
            ibp_bounds.upper()[[0]],
        );

        // Sample points should all be within bounds
        for rms in sample_points(0.0, 1.0, 8) {
            for env in sample_points(0.0, 1.0, 8) {
                let envelope_new = one_pole_step_true(rms, env, c);
                let gain = gain_reduction_true(envelope_new, &params.gain);
                assert!(
                    ibp_bounds.lower()[[0]] - FP_TOLERANCE <= gain
                        && gain <= ibp_bounds.upper()[[0]] + FP_TOLERANCE,
                    "Combined IBP unsound: c={c}, rms={rms}, env={env}, \
                     envelope_new={envelope_new}, gain={gain}, bounds=[{}, {}]",
                    ibp_bounds.lower()[[0]],
                    ibp_bounds.upper()[[0]],
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_envelope_follower_step_crown_soundness() {
    let params = test_envelope_params();

    for &c in &[params.c_attack, params.c_release] {
        let graph = build_envelope_follower_step_graph(c, &params.gain).unwrap();

        // Narrow range for CROWN precision through the nonlinear gain chain
        let input = BoundedTensor::new(
            arr1(&[0.1_f32, 0.1]).into_dyn(),
            arr1(&[0.8_f32, 0.8]).into_dyn(),
        )
        .unwrap();

        let crown_bounds = graph.propagate_crown(&input).unwrap();

        for rms in sample_points(0.1, 0.8, 8) {
            for env in sample_points(0.1, 0.8, 8) {
                let envelope_new = one_pole_step_true(rms, env, c);
                let gain = gain_reduction_true(envelope_new, &params.gain);
                assert!(
                    crown_bounds.lower()[[0]] - FP_TOLERANCE <= gain
                        && gain <= crown_bounds.upper()[[0]] + FP_TOLERANCE,
                    "Combined CROWN unsound: c={c}, rms={rms}, env={env}, \
                     envelope_new={envelope_new}, gain={gain}, bounds=[{}, {}]",
                    crown_bounds.lower()[[0]],
                    crown_bounds.upper()[[0]],
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_conservative_envelope_follower_gain_bounds() {
    // Phase 3 conservative verification: union of attack-only and release-only
    // gain bounds should satisfy the acceptance criterion.
    let params = test_envelope_params();

    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let conservative_bounds = verify_envelope_follower_conservative(&params, &input).unwrap();

    // Acceptance criterion: gain ∈ [max_atten, 1.0]
    assert!(
        conservative_bounds.lower()[[0]] >= params.gain.max_atten - FP_TOLERANCE,
        "Conservative lower below max_atten: {} < {}",
        conservative_bounds.lower()[[0]],
        params.gain.max_atten,
    );
    assert!(
        conservative_bounds.upper()[[0]] <= 1.0 + FP_TOLERANCE,
        "Conservative upper above 1.0: {}",
        conservative_bounds.upper()[[0]],
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_conservative_below_threshold_gain_is_one() {
    // When envelope stays below threshold, gain should be exactly 1.0
    // regardless of attack/release coefficient.
    let params = test_envelope_params();
    let threshold = params.gain.threshold;

    // rms and envelope both below threshold → envelope_new below threshold
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[threshold * 0.5, threshold * 0.5]).into_dyn(),
    )
    .unwrap();

    let conservative_bounds = verify_envelope_follower_conservative(&params, &input).unwrap();

    assert!(
        (conservative_bounds.lower()[[0]] - 1.0).abs() < FP_TOLERANCE,
        "Below threshold: lower should be 1.0, got {}",
        conservative_bounds.lower()[[0]],
    );
    assert!(
        (conservative_bounds.upper()[[0]] - 1.0).abs() < FP_TOLERANCE,
        "Below threshold: upper should be 1.0, got {}",
        conservative_bounds.upper()[[0]],
    );
}
