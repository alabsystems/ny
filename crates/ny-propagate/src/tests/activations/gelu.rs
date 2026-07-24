// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Adaptive GELU Relaxation tests ====================

use crate::layers::{
    adaptive_gelu_linear_relaxation, gelu_eval, gelu_linear_relaxation, GeluApproximation,
    RelaxationMode,
};

/// Helper: compute GELU at a point
fn gelu_at(x: f32) -> f32 {
    gelu_eval(x, GeluApproximation::Erf)
}

/// Helper: verify relaxation is sound (lower <= GELU <= upper over interval)
fn verify_relaxation_sound(l: f32, u: f32, relaxation: (f32, f32, f32, f32)) {
    let (ls, li, us, ui) = relaxation;
    let num_samples = 100;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let gx = gelu_at(x);
        let lower_bound = ls * x + li;
        let upper_bound = us * x + ui;

        assert!(
            lower_bound <= gx + 1e-4,
            "Lower bound violated at x={}: lower_bound={} > GELU({})={}",
            x,
            lower_bound,
            x,
            gx
        );
        assert!(
            gx <= upper_bound + 1e-4,
            "Upper bound violated at x={}: GELU({})={} > upper_bound={}",
            x,
            x,
            gx,
            upper_bound
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_relaxation_mode_chord_soundness() {
    // Test chord mode on various intervals
    let test_intervals = [
        (-2.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (-1.0, 1.0),
        (-3.0, 3.0),
    ];

    for (l, u) in test_intervals {
        let relaxation =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Chord);
        verify_relaxation_sound(l, u, relaxation);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_relaxation_mode_tangent_soundness() {
    // Test tangent mode on various intervals
    let test_intervals = [
        (-2.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (-1.0, 1.0),
        (-3.0, 3.0),
    ];

    for (l, u) in test_intervals {
        let relaxation =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Tangent);
        verify_relaxation_sound(l, u, relaxation);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_relaxation_mode_two_slope_soundness() {
    // Test two-slope mode on various intervals
    let test_intervals = [
        (-2.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (-1.0, 1.0),
        (-3.0, 3.0),
    ];

    for (l, u) in test_intervals {
        let relaxation =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::TwoSlope);
        verify_relaxation_sound(l, u, relaxation);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_relaxation_mode_adaptive_soundness() {
    // Test adaptive mode on various intervals
    let test_intervals = [
        (-2.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (-1.0, 1.0),
        (-3.0, 3.0),
    ];

    for (l, u) in test_intervals {
        let relaxation =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Adaptive);
        verify_relaxation_sound(l, u, relaxation);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_adaptive_is_at_least_as_tight_as_chord() {
    // Adaptive mode should produce bounds at least as tight as chord
    let test_intervals = [
        (-2.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (-1.0, 1.0),
        (-0.5, 0.5),
    ];

    for (l, u) in test_intervals {
        let chord =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Chord);
        let adaptive =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Adaptive);

        let c = f32::midpoint(l, u);
        let chord_width = (chord.2 * c + chord.3) - (chord.0 * c + chord.1);
        let adaptive_width = (adaptive.2 * c + adaptive.3) - (adaptive.0 * c + adaptive.1);

        assert!(
            adaptive_width <= chord_width + 1e-5,
            "Adaptive should be at least as tight as chord for [{}, {}]: adaptive_width={} > chord_width={}",
            l, u, adaptive_width, chord_width
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_relaxation_modes_small_interval() {
    // For small intervals, tangent should be very tight
    let l = -0.1;
    let u = 0.1;

    let chord =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Chord);
    let tangent =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Tangent);
    let two_slope =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::TwoSlope);
    let adaptive =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Adaptive);

    let c = f32::midpoint(l, u);
    let chord_width = (chord.2 * c + chord.3) - (chord.0 * c + chord.1);
    let tangent_width = (tangent.2 * c + tangent.3) - (tangent.0 * c + tangent.1);
    let two_slope_width = (two_slope.2 * c + two_slope.3) - (two_slope.0 * c + two_slope.1);
    let adaptive_width = (adaptive.2 * c + adaptive.3) - (adaptive.0 * c + adaptive.1);

    // All modes should be sound
    verify_relaxation_sound(l, u, chord);
    verify_relaxation_sound(l, u, tangent);
    verify_relaxation_sound(l, u, two_slope);
    verify_relaxation_sound(l, u, adaptive);

    // Adaptive should be the tightest or equal
    assert!(adaptive_width <= chord_width + 1e-5);
    assert!(adaptive_width <= tangent_width + 1e-5);
    assert!(adaptive_width <= two_slope_width + 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_gelu_layer_with_relaxation_modes() {
    // Test GELULayer with different relaxation modes produces valid bounds
    let lower = ArrayD::from_elem(IxDyn(&[4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Test with default (Chord) mode
    let gelu_chord = GELULayer::new(GeluApproximation::Erf);
    let output_chord = gelu_chord.propagate_ibp(&input).unwrap();

    // Test with adaptive mode
    let gelu_adaptive = GELULayer::adaptive(GeluApproximation::Erf);
    let output_adaptive = gelu_adaptive.propagate_ibp(&input).unwrap();

    // Both should produce valid bounds
    for i in 0..4 {
        assert!(output_chord.lower()[[i]] <= output_chord.upper()[[i]]);
        assert!(output_adaptive.lower()[[i]] <= output_adaptive.upper()[[i]]);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_relaxation_modes_wide_interval() {
    // For wide intervals, two-slope may help
    let l = -3.0;
    let u = 3.0;

    let chord =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Chord);
    let tangent =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Tangent);
    let two_slope =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::TwoSlope);
    let adaptive =
        adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Adaptive);

    // All modes should be sound
    verify_relaxation_sound(l, u, chord);
    verify_relaxation_sound(l, u, tangent);
    verify_relaxation_sound(l, u, two_slope);
    verify_relaxation_sound(l, u, adaptive);

    // Adaptive should pick the best
    let c = f32::midpoint(l, u);
    let chord_width = (chord.2 * c + chord.3) - (chord.0 * c + chord.1);
    let adaptive_width = (adaptive.2 * c + adaptive.3) - (adaptive.0 * c + adaptive.1);

    assert!(
        adaptive_width <= chord_width + 1e-5,
        "Adaptive should be at least as tight as chord"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_chord_vs_original_gelu_linear_relaxation() {
    // Verify chord mode matches original function
    let test_intervals = [(-2.0, -1.0), (-1.0, 0.0), (0.0, 1.0), (-1.0, 1.0)];

    for (l, u) in test_intervals {
        let original = gelu_linear_relaxation(l, u, GeluApproximation::Erf);
        let chord =
            adaptive_gelu_linear_relaxation(l, u, GeluApproximation::Erf, RelaxationMode::Chord);

        assert!((original.0 - chord.0).abs() < 1e-6, "Lower slope mismatch");
        assert!(
            (original.1 - chord.1).abs() < 1e-6,
            "Lower intercept mismatch"
        );
        assert!((original.2 - chord.2).abs() < 1e-6, "Upper slope mismatch");
        assert!(
            (original.3 - chord.3).abs() < 1e-6,
            "Upper intercept mismatch"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_relaxation_improvement_metrics() {
    // Print improvement metrics for adaptive relaxation
    // This test always passes but logs useful information

    let test_intervals = [
        ("small_near_zero", -0.1_f32, 0.1_f32),
        ("medium_symmetric", -1.0, 1.0),
        ("wide_symmetric", -3.0, 3.0),
        ("negative_region", -2.0, -0.5),
        ("positive_region", 0.5, 2.0),
        ("critical_region", -1.0, 0.0),
    ];

    let mut total_chord = 0.0_f32;
    let mut total_adaptive = 0.0_f32;
    let mut improvements = Vec::new();

    for (name, l, u) in test_intervals.iter() {
        let chord =
            adaptive_gelu_linear_relaxation(*l, *u, GeluApproximation::Erf, RelaxationMode::Chord);
        let adaptive = adaptive_gelu_linear_relaxation(
            *l,
            *u,
            GeluApproximation::Erf,
            RelaxationMode::Adaptive,
        );

        let c = (l + u) / 2.0;
        let chord_width = (chord.2 * c + chord.3) - (chord.0 * c + chord.1);
        let adaptive_width = (adaptive.2 * c + adaptive.3) - (adaptive.0 * c + adaptive.1);

        let improvement = if chord_width > 0.0 {
            (chord_width - adaptive_width) / chord_width * 100.0
        } else {
            0.0
        };

        improvements.push((name, chord_width, adaptive_width, improvement));
        total_chord += chord_width;
        total_adaptive += adaptive_width;
    }

    // Log results (visible with --nocapture)
    eprintln!(
        "
=== Adaptive GELU Relaxation Improvement ==="
    );
    eprintln!(
        "{:<20} {:>12} {:>12} {:>12}",
        "Interval", "Chord", "Adaptive", "Improvement"
    );
    for (name, chord, adaptive, improvement) in &improvements {
        eprintln!(
            "{:<20} {:>12.6} {:>12.6} {:>11.1}%",
            name, chord, adaptive, improvement
        );
    }

    let avg_improvement = if total_chord > 0.0 {
        (total_chord - total_adaptive) / total_chord * 100.0
    } else {
        0.0
    };
    eprintln!(
        "
Average improvement: {:.1}%",
        avg_improvement
    );

    // Verify adaptive is never worse than chord
    for (name, chord_width, adaptive_width, _) in improvements {
        assert!(
            adaptive_width <= chord_width + 1e-5,
            "Adaptive should not be worse than chord for {}",
            name
        );
    }
}

// ==================== Sound GELU Relaxation tests ====================
// Tests for the precomputed tangent-based sound relaxation (no sampling).
// Reference: auto_LiRPA's BoundGelu @ 9d100ec070868440b48d34e2f1dd21b97aab9172

use crate::layers::gelu_sound_linear_relaxation;

#[ntest::timeout(10000)]
#[test]
fn test_sound_gelu_relaxation_basic_soundness() {
    // Test sound relaxation on various intervals covering all case splits
    let test_intervals = [
        // Entirely negative intervals
        (-5.0, -3.0), // Both < -sqrt(2)
        (-2.0, -1.0), // l < -sqrt(2), u in concave region
        (-1.0, -0.5), // Both in concave region (between -sqrt(2) and 0)
        // Entirely positive intervals
        (0.5, 1.0), // Both in concave region (between 0 and sqrt(2))
        (1.0, 2.0), // l < sqrt(2), u > sqrt(2)
        (2.0, 5.0), // Both > sqrt(2)
        // Cross-zero intervals
        (-1.0, 1.0), // Within (-sqrt(2), sqrt(2))
        (-3.0, 0.5), // l < -sqrt(2), crosses zero
        (-0.5, 3.0), // u > sqrt(2), crosses zero
        (-3.0, 3.0), // Wide interval crossing everything
    ];

    for (l, u) in test_intervals {
        let relaxation = gelu_sound_linear_relaxation(l, u);
        verify_relaxation_sound(l, u, relaxation);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sound_gelu_relaxation_edge_cases() {
    // Test edge cases: very small intervals, points near critical values
    let edge_cases = [
        // Near sqrt(2)
        (1.41, 1.42),
        (1.414, 1.415),
        // Near -sqrt(2)
        (-1.42, -1.41),
        (-1.415, -1.414),
        // Near zero
        (-0.1, 0.1),
        (-0.01, 0.01),
        // Very small intervals
        (0.5, 0.501),
        (-0.5, -0.499),
        // Around the critical point (GELU minimum near -0.75)
        (-0.8, -0.7),
        (-1.0, -0.5),
    ];

    for (l, u) in edge_cases {
        let relaxation = gelu_sound_linear_relaxation(l, u);
        verify_relaxation_sound(l, u, relaxation);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sound_gelu_relaxation_degenerate() {
    // Test degenerate cases: point intervals, infinite/NaN bounds

    // Point intervals should use derivative
    let point = gelu_sound_linear_relaxation(1.0, 1.0);
    let (ls, li, us, ui) = point;
    assert!(
        (ls - us).abs() < 1e-5,
        "Point interval should have same slope"
    );
    assert!(
        (li - ui).abs() < 1e-5,
        "Point interval should have same intercept"
    );

    // Infinite bounds should return maximally loose (NOT identity — identity is unsound, #1837).
    // GELU(x) = x·Φ(x), so GELU(x) ≥ x fails for 0 < x where Φ(x) < 1.
    let inf_case = gelu_sound_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY);
    assert_eq!(
        inf_case,
        (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY),
        "Infinite bounds should return maximally loose (fix #1837)"
    );

    // NaN bounds should return maximally loose (NOT identity — #1837).
    let nan_case = gelu_sound_linear_relaxation(f32::NAN, 1.0);
    assert_eq!(
        nan_case,
        (0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY),
        "NaN bounds should return maximally loose (fix #1837)"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sound_gelu_layer_flag() {
    // Test that GELULayer::sound() creates a layer with sound=true
    use crate::layers::GELULayer;

    let sound_layer = GELULayer::sound(GeluApproximation::Erf);
    assert!(sound_layer.is_sound(), "sound() should create sound layer");
    assert!(sound_layer.sound, "sound flag should be true");
    assert_eq!(sound_layer.approximation, GeluApproximation::Erf);

    let sound_tanh = GELULayer::sound(GeluApproximation::Tanh);
    assert!(sound_tanh.is_sound(), "sound() should create sound layer");
    assert!(sound_tanh.sound, "sound flag should be true");
    assert_eq!(sound_tanh.approximation, GeluApproximation::Tanh);

    // Default layer should be sound (changed from false to true per #1735)
    let default_layer = GELULayer::default();
    assert!(default_layer.is_sound(), "default should be sound (#1735)");
    assert!(
        default_layer.sound,
        "sound flag should be true by default (#1735)"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sound_gelu_and_sampling_both_sound() {
    // Verify both sound (analytical) and sampling-based GELU relaxations are
    // individually sound across representative intervals. They may differ in
    // tightness but both must satisfy the soundness envelope.
    let test_intervals = [(-2.0, -1.0), (0.0, 1.0), (-1.0, 1.0), (1.5, 3.0)];

    for (l, u) in test_intervals {
        let sound = gelu_sound_linear_relaxation(l, u);
        let sampling = gelu_linear_relaxation(l, u, GeluApproximation::Erf);

        verify_relaxation_sound(l, u, sound);
        verify_relaxation_sound(l, u, sampling);
    }
}
