// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// Licensed under the Apache License, Version 2.0

use super::*;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{Array1, Array2, ArrayD};

// ── Constructor ──────────────────────────────────────────────────────

#[test]
fn test_new() {
    let clip = ClipLayer::new(-1.0, 1.0);
    assert_eq!(clip.min, -1.0);
    assert_eq!(clip.max, 1.0);
}

#[test]
fn test_new_zero_width() {
    let clip = ClipLayer::new(0.5, 0.5);
    assert_eq!(clip.min, clip.max);
}

#[test]
fn test_try_new_rejects_invalid_bounds_2551() {
    for (min, max) in [(f32::NAN, 1.0), (-1.0, f32::NAN), (2.0, 1.0)] {
        let err = ClipLayer::try_new(min, max).expect_err("invalid bounds should be rejected");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }
}

// ── IBP ──────────────────────────────────────────────────────────────

#[test]
fn test_ibp_entirely_within() {
    // Input [0.2, 0.8] with clip [-1, 1] → no clamping
    let clip = ClipLayer::new(-1.0, 1.0);
    let input = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[3]), 0.2f32),
        ArrayD::from_elem(ndarray::IxDyn(&[3]), 0.8f32),
    )
    .unwrap();
    let result = clip.propagate_ibp(&input).unwrap();
    for &v in result.lower().iter() {
        assert!((v - 0.2).abs() < 1e-6);
    }
    for &v in result.upper().iter() {
        assert!((v - 0.8).abs() < 1e-6);
    }
}

#[test]
fn test_ibp_clamps_both_sides() {
    // Input [-5, 5] with clip [-1, 1] → [-1, 1]
    let clip = ClipLayer::new(-1.0, 1.0);
    let input = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[2]), -5.0f32),
        ArrayD::from_elem(ndarray::IxDyn(&[2]), 5.0f32),
    )
    .unwrap();
    let result = clip.propagate_ibp(&input).unwrap();
    for &v in result.lower().iter() {
        assert!((v - (-1.0)).abs() < 1e-6);
    }
    for &v in result.upper().iter() {
        assert!((v - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_ibp_entirely_below() {
    // Input [-5, -3] with clip [-1, 1] → [-1, -1]
    let clip = ClipLayer::new(-1.0, 1.0);
    let input = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[2]), -5.0f32),
        ArrayD::from_elem(ndarray::IxDyn(&[2]), -3.0f32),
    )
    .unwrap();
    let result = clip.propagate_ibp(&input).unwrap();
    for &v in result.lower().iter() {
        assert!((v - (-1.0)).abs() < 1e-6);
    }
    for &v in result.upper().iter() {
        assert!((v - (-1.0)).abs() < 1e-6);
    }
}

#[test]
fn test_ibp_entirely_above() {
    // Input [3, 5] with clip [-1, 1] → [1, 1]
    let clip = ClipLayer::new(-1.0, 1.0);
    let input = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[2]), 3.0f32),
        ArrayD::from_elem(ndarray::IxDyn(&[2]), 5.0f32),
    )
    .unwrap();
    let result = clip.propagate_ibp(&input).unwrap();
    for &v in result.lower().iter() {
        assert!((v - 1.0).abs() < 1e-6);
    }
    for &v in result.upper().iter() {
        assert!((v - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_ibp_point_input() {
    let clip = ClipLayer::new(0.0, 1.0);
    let input = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.5f32),
        ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.5f32),
    )
    .unwrap();
    let result = clip.propagate_ibp(&input).unwrap();
    for (&lo, &hi) in result.lower().iter().zip(result.upper().iter()) {
        assert!((lo - 0.5).abs() < 1e-6);
        assert!((hi - 0.5).abs() < 1e-6);
    }
}

#[test]
fn test_ibp_soundness_grid() {
    // Exhaustive soundness: for many (l,u) pairs, verify clip(x) ∈ [out_l, out_u]
    let clip = ClipLayer::new(-0.5, 0.5);
    let vals: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.15).collect();
    for &lo in &vals {
        for &hi in &vals {
            if lo > hi {
                continue;
            }
            let input = BoundedTensor::new(
                ArrayD::from_elem(ndarray::IxDyn(&[1]), lo),
                ArrayD::from_elem(ndarray::IxDyn(&[1]), hi),
            )
            .unwrap();
            let result = clip.propagate_ibp(&input).unwrap();
            let out_lo = result.lower().iter().next().unwrap();
            let out_hi = result.upper().iter().next().unwrap();
            // Check concrete points
            for &x in &[lo, hi, f32::midpoint(lo, hi)] {
                let y = x.clamp(-0.5, 0.5);
                assert!(
                    *out_lo <= y + 1e-6 && y <= *out_hi + 1e-6,
                    "clip({}) = {} not in [{}, {}] for input [{}, {}]",
                    x,
                    y,
                    out_lo,
                    out_hi,
                    lo,
                    hi,
                );
            }
        }
    }
}

// ── CROWN backward ──────────────────────────────────────────────────

/// Helper: make identity CROWN bounds for n neurons
fn identity_bounds(n: usize) -> LinearBounds {
    LinearBounds::new(
        Array2::eye(n),
        Array1::zeros(n),
        Array2::eye(n),
        Array1::zeros(n),
    )
    .unwrap()
}

#[test]
fn test_crown_entirely_below_min() {
    // pre-activation [−3, −2], clip [−1, 1] → constant at min
    let clip = ClipLayer::new(-1.0, 1.0);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[2]), -3.0f32),
        ArrayD::from_elem(ndarray::IxDyn(&[2]), -2.0f32),
    )
    .unwrap();
    let bounds = identity_bounds(2);
    let result = clip.propagate_linear_with_bounds(&bounds, &pre).unwrap();
    // slope=0, intercept=min → lower_a=0, lower_b=min, upper_a=0, upper_b=min
    for &v in result.lower_a.iter() {
        assert!(v.abs() < 1e-6, "lower_a should be 0, got {}", v);
    }
    for &v in result.lower_b.iter() {
        assert!((v - (-1.0)).abs() < 1e-6, "lower_b should be -1, got {}", v);
    }
    for &v in result.upper_a.iter() {
        assert!(v.abs() < 1e-6, "upper_a should be 0, got {}", v);
    }
    for &v in result.upper_b.iter() {
        assert!((v - (-1.0)).abs() < 1e-6, "upper_b should be -1, got {}", v);
    }
}

#[test]
fn test_crown_entirely_above_max() {
    // pre-activation [2, 3], clip [−1, 1] → constant at max
    let clip = ClipLayer::new(-1.0, 1.0);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[2]), 2.0f32),
        ArrayD::from_elem(ndarray::IxDyn(&[2]), 3.0f32),
    )
    .unwrap();
    let bounds = identity_bounds(2);
    let result = clip.propagate_linear_with_bounds(&bounds, &pre).unwrap();
    for &v in result.lower_a.iter() {
        assert!(v.abs() < 1e-6);
    }
    for &v in result.lower_b.iter() {
        assert!((v - 1.0).abs() < 1e-6);
    }
    for &v in result.upper_a.iter() {
        assert!(v.abs() < 1e-6);
    }
    for &v in result.upper_b.iter() {
        assert!((v - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_crown_identity_region() {
    // pre-activation [−0.5, 0.5], clip [−1, 1] → identity (slope=1, intercept=0)
    let clip = ClipLayer::new(-1.0, 1.0);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[2]), -0.5f32),
        ArrayD::from_elem(ndarray::IxDyn(&[2]), 0.5f32),
    )
    .unwrap();
    let bounds = identity_bounds(2);
    let result = clip.propagate_linear_with_bounds(&bounds, &pre).unwrap();
    // Identity: lower_a=I, lower_b=0, upper_a=I, upper_b=0
    for i in 0..2 {
        assert!((result.lower_a[[i, i]] - 1.0).abs() < 1e-6);
        assert!((result.upper_a[[i, i]] - 1.0).abs() < 1e-6);
    }
    for &v in result.lower_b.iter() {
        assert!(v.abs() < 1e-6);
    }
    for &v in result.upper_b.iter() {
        assert!(v.abs() < 1e-6);
    }
}

#[test]
fn test_crown_crosses_both_boundaries() {
    // pre-activation [−2, 2], clip [−1, 1] → case 4
    let clip = ClipLayer::new(-1.0, 1.0);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[1]), -2.0f32),
        ArrayD::from_elem(ndarray::IxDyn(&[1]), 2.0f32),
    )
    .unwrap();
    let bounds = identity_bounds(1);
    let result = clip.propagate_linear_with_bounds(&bounds, &pre).unwrap();
    // Upper: su = (1-(-1))/(1-(-2)) = 2/3
    let su = 2.0 / 3.0;
    assert!((result.upper_a[[0, 0]] - su).abs() < 1e-5, "upper slope");
    // Lower: sl = (1-(-1))/(2-(-1)) = 2/3
    let sl = 2.0 / 3.0;
    assert!((result.lower_a[[0, 0]] - sl).abs() < 1e-5, "lower slope");
}

#[test]
fn test_crown_crosses_lower_boundary_adaptive_identity() {
    // pre-activation [−0.5, 1.0], clip [0.0, 2.0] → crosses lower only
    // su = (1.0 - 0.0) / (1.0 - (-0.5)) = 1.0/1.5 = 2/3 > 0.5 → lower is identity
    let clip = ClipLayer::new(0.0, 2.0);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[1]), -0.5f32),
        ArrayD::from_elem(ndarray::IxDyn(&[1]), 1.0f32),
    )
    .unwrap();
    let bounds = identity_bounds(1);
    let result = clip.propagate_linear_with_bounds(&bounds, &pre).unwrap();
    // Lower: identity (slope=1, intercept=0) because su > 0.5
    assert!(
        (result.lower_a[[0, 0]] - 1.0).abs() < 1e-5,
        "lower slope should be 1 (identity)"
    );
    assert!(
        result.lower_b[0].abs() < 1e-5,
        "lower intercept should be 0"
    );
    // Upper: chord slope su = 2/3
    let su = 2.0 / 3.0;
    assert!(
        (result.upper_a[[0, 0]] - su).abs() < 1e-5,
        "upper slope should be 2/3"
    );
}

#[test]
fn test_crown_crosses_lower_boundary_adaptive_flat() {
    // pre-activation [−5.0, 0.1], clip [0.0, 1.0] → crosses lower only
    // su = (0.1 - 0.0) / (0.1 - (-5.0)) = 0.1/5.1 ≈ 0.0196 < 0.5 → lower is flat at min
    let clip = ClipLayer::new(0.0, 1.0);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[1]), -5.0f32),
        ArrayD::from_elem(ndarray::IxDyn(&[1]), 0.1f32),
    )
    .unwrap();
    let bounds = identity_bounds(1);
    let result = clip.propagate_linear_with_bounds(&bounds, &pre).unwrap();
    // Lower: flat at min (slope=0, intercept=min)
    assert!(
        result.lower_a[[0, 0]].abs() < 1e-5,
        "lower slope should be 0 (flat)"
    );
    assert!(
        (result.lower_b[0] - 0.0).abs() < 1e-5,
        "lower intercept should be min=0"
    );
}

#[test]
fn test_crown_crosses_upper_boundary_adaptive_identity() {
    // pre-activation [0.0, 1.5], clip [−1.0, 1.0] → crosses upper only
    // sl = (1.0 - 0.0) / (1.5 - 0.0) = 2/3 > 0.5 → upper is identity
    let clip = ClipLayer::new(-1.0, 1.0);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[1]), 0.0f32),
        ArrayD::from_elem(ndarray::IxDyn(&[1]), 1.5f32),
    )
    .unwrap();
    let bounds = identity_bounds(1);
    let result = clip.propagate_linear_with_bounds(&bounds, &pre).unwrap();
    // Upper: identity because sl > 0.5
    assert!(
        (result.upper_a[[0, 0]] - 1.0).abs() < 1e-5,
        "upper slope should be 1 (identity)"
    );
    assert!(
        result.upper_b[0].abs() < 1e-5,
        "upper intercept should be 0"
    );
}

#[test]
fn test_crown_crosses_upper_boundary_adaptive_flat() {
    // pre-activation [0.9, 10.0], clip [−1.0, 1.0] → crosses upper only
    // sl = (1.0 - 0.9) / (10.0 - 0.9) = 0.1/9.1 ≈ 0.011 < 0.5 → upper flat at max
    let clip = ClipLayer::new(-1.0, 1.0);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[1]), 0.9f32),
        ArrayD::from_elem(ndarray::IxDyn(&[1]), 10.0f32),
    )
    .unwrap();
    let bounds = identity_bounds(1);
    let result = clip.propagate_linear_with_bounds(&bounds, &pre).unwrap();
    // Upper: flat at max (slope=0, intercept=max)
    assert!(
        result.upper_a[[0, 0]].abs() < 1e-5,
        "upper slope should be 0 (flat)"
    );
    assert!(
        (result.upper_b[0] - 1.0).abs() < 1e-5,
        "upper intercept should be max=1"
    );
}

/// Post-#2977: domain_guard rejects NaN pre-activation bounds with
/// NumericalInstability error. The old behavior (conservative [min,max]
/// fallback) is now unreachable because the guard fires first.
#[test]
fn test_crown_nan_guard() {
    let clip = ClipLayer::new(-1.0, 1.0);
    let pre = BoundedTensor::new_unchecked(
        ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::NAN),
        ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::NAN),
    )
    .unwrap();
    let bounds = identity_bounds(1);
    let result = clip.propagate_linear_with_bounds(&bounds, &pre);
    assert!(
        result.is_err(),
        "NaN pre-activation should be rejected by domain_guard (#2977)"
    );
}

#[test]
fn test_crown_propagate_linear_requires_bounds() {
    // propagate_linear without pre-activation should return error
    let clip = ClipLayer::new(-1.0, 1.0);
    let bounds = identity_bounds(2);
    let result = clip.propagate_linear(&bounds);
    assert!(result.is_err());
    assert!(clip.requires_pre_activation_bounds());
}

#[test]
fn test_crown_shape_mismatch() {
    // Pre-activation size != bounds num_inputs
    let clip = ClipLayer::new(-1.0, 1.0);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[3]), -1.0f32),
        ArrayD::from_elem(ndarray::IxDyn(&[3]), 1.0f32),
    )
    .unwrap();
    let bounds = identity_bounds(5); // mismatch: 5 vs 3
    let result = clip.propagate_linear_with_bounds(&bounds, &pre);
    assert!(result.is_err());
}

#[test]
fn test_crown_soundness_grid() {
    // Verify CROWN relaxation soundness: for each (l, u), the linear bounds
    // must enclose the actual clip output for any x ∈ [l, u].
    let clip = ClipLayer::new(-1.0, 1.0);
    let test_ranges: Vec<(f32, f32)> = vec![
        (-3.0, -2.0),  // entirely below
        (2.0, 3.0),    // entirely above
        (-0.5, 0.5),   // identity
        (-2.0, 2.0),   // crosses both
        (-2.0, 0.5),   // crosses lower
        (0.5, 2.0),    // crosses upper
        (-0.01, 0.01), // tiny around origin
        (-1.0, 1.0),   // exactly at boundaries
    ];
    for (lo, hi) in test_ranges {
        let pre = BoundedTensor::new(
            ArrayD::from_elem(ndarray::IxDyn(&[1]), lo),
            ArrayD::from_elem(ndarray::IxDyn(&[1]), hi),
        )
        .unwrap();
        let bounds = identity_bounds(1);
        let result = clip.propagate_linear_with_bounds(&bounds, &pre).unwrap();
        // Evaluate concrete output bounds
        let la = result.lower_a[[0, 0]];
        let lb = result.lower_b[0];
        let ua = result.upper_a[[0, 0]];
        let ub = result.upper_b[0];
        // Check at sample points
        let n = 21;
        for k in 0..=n {
            let x = lo + (hi - lo) * (k as f32 / n as f32);
            let y = x.clamp(-1.0, 1.0);
            let lower_bound = la * x + lb;
            let upper_bound = ua * x + ub;
            assert!(
                lower_bound <= y + 1e-5,
                "lower_bound {} > y {} at x={} for [{}, {}]",
                lower_bound,
                y,
                x,
                lo,
                hi,
            );
            assert!(
                upper_bound >= y - 1e-5,
                "upper_bound {} < y {} at x={} for [{}, {}]",
                upper_bound,
                y,
                x,
                lo,
                hi,
            );
        }
    }
}

// ── Batched CROWN ───────────────────────────────────────────────────

#[test]
fn test_batched_crown_identity_region() {
    let clip = ClipLayer::new(-1.0, 1.0);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[2]), -0.5f32),
        ArrayD::from_elem(ndarray::IxDyn(&[2]), 0.5f32),
    )
    .unwrap();
    let batched = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ArrayD::zeros(ndarray::IxDyn(&[2])),
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ArrayD::zeros(ndarray::IxDyn(&[2])),
        vec![2],
        vec![2],
    );
    let result = clip
        .propagate_linear_batched_with_bounds(&batched, &pre)
        .unwrap();
    // Identity region: bounds should pass through
    for &v in result.lower_a.iter() {
        // Either 1.0 (diag) or 0.0 (off-diag), same as input
        assert!(v.abs() < 1e-5 || (v - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_batched_crown_clamped_region() {
    let clip = ClipLayer::new(-1.0, 1.0);
    let pre = BoundedTensor::new(
        ArrayD::from_elem(ndarray::IxDyn(&[2]), -5.0f32),
        ArrayD::from_elem(ndarray::IxDyn(&[2]), -3.0f32),
    )
    .unwrap();
    let batched = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ArrayD::zeros(ndarray::IxDyn(&[2])),
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ArrayD::zeros(ndarray::IxDyn(&[2])),
        vec![2],
        vec![2],
    );
    let result = clip
        .propagate_linear_batched_with_bounds(&batched, &pre)
        .unwrap();
    // All below min: slope=0, intercept=min
    for &v in result.lower_a.iter() {
        assert!(v.abs() < 1e-5, "lower_a should be 0");
    }
    for &v in result.lower_b.iter() {
        assert!((v - (-1.0)).abs() < 1e-5, "lower_b should be -1");
    }
}

// ── IBP guard regression tests (#3278) ────────────────────────────

#[test]
fn test_ibp_nan_input_lower_rejected_3278() {
    let clip = ClipLayer::new(0.0, 6.0);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::NAN),
        ArrayD::from_elem(ndarray::IxDyn(&[1]), 3.0),
    )
    .unwrap();
    let err = clip.propagate_ibp(&input).expect_err("NaN input lower");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

#[test]
fn test_ibp_nan_input_upper_rejected_3278() {
    let clip = ClipLayer::new(0.0, 6.0);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_elem(ndarray::IxDyn(&[1]), 1.0),
        ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::NAN),
    )
    .unwrap();
    let err = clip.propagate_ibp(&input).expect_err("NaN input upper");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

/// #3278 originally rejected ±Inf here too. That was wrong: ±Inf is what an
/// upstream OpaqueSkip legitimately emits, and `NumericalInstability` is not
/// degradable, so one tainted element aborted the whole graph-IBP pass. Clip
/// must now widen instead — and because it saturates, `[-inf, +inf]` in yields
/// the exact `[min, max]` out.
#[test]
fn test_ibp_inf_input_propagates_widened_3278() {
    let clip = ClipLayer::new(0.0, 6.0);
    let input = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::NEG_INFINITY),
        ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::INFINITY),
    )
    .unwrap();
    let out = clip
        .propagate_ibp(&input)
        .expect("a tainted element must widen, not abort the pass");
    assert_eq!(out.lower()[[0]], 0.0);
    assert_eq!(out.upper()[[0]], 6.0);
}

// ── OpaqueSkip taint propagation (#3278 follow-up) ─────────────────────

/// Probe: a mixed tensor where one element carries an upstream OpaqueSkip's
/// `[-inf, +inf]` and the other is a normal finite interval. The tainted
/// element must saturate to the clip range; the finite element is unchanged.
#[test]
fn test_ibp_opaque_skip_taint_widens_only_tainted_element() {
    let clip = ClipLayer::new(-1.0, 1.0);
    let input = BoundedTensor::new_allow_infinite(
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![f32::NEG_INFINITY, -0.5]).unwrap(),
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![f32::INFINITY, 0.5]).unwrap(),
    )
    .unwrap();
    let out = clip
        .propagate_ibp(&input)
        .expect("[-inf, +inf] is a sound enclosure, not an error");

    assert_eq!(out.lower()[[0]], -1.0);
    assert_eq!(out.upper()[[0]], 1.0);
    assert_eq!(out.lower()[[1]], -0.5);
    assert_eq!(out.upper()[[1]], 0.5);
}

/// Soundness: the saturated output must still enclose clip over the whole
/// (unbounded) input interval.
#[test]
fn test_ibp_inf_input_output_encloses_true_range() {
    let (min_val, max_val) = (0.0f32, 6.0f32);
    let clip = ClipLayer::new(min_val, max_val);
    let input = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::NEG_INFINITY),
        ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::INFINITY),
    )
    .unwrap();
    let out = clip.propagate_ibp(&input).unwrap();
    for x in [-1e30f32, -100.0, -1.0, 0.0, 3.0, 6.0, 100.0, 1e30] {
        let y = x.clamp(min_val, max_val);
        assert!(
            out.lower()[[0]] <= y,
            "lower {} > clip({x})={y}",
            out.lower()[[0]]
        );
        assert!(
            out.upper()[[0]] >= y,
            "upper {} < clip({x})={y}",
            out.upper()[[0]]
        );
    }
}

/// An infinite clip bound keeps the taint infinite — this is exactly why the
/// output uses `new_allow_infinite` and not the strict constructor.
#[test]
fn test_ibp_infinite_clip_bound_keeps_output_infinite() {
    let clip = ClipLayer::try_new(f32::NEG_INFINITY, 6.0).expect("infinite min is permitted");
    let input = BoundedTensor::new_allow_infinite(
        ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::NEG_INFINITY),
        ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::INFINITY),
    )
    .unwrap();
    let out = clip.propagate_ibp(&input).expect("must not abort");
    assert_eq!(out.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(out.upper()[[0]], 6.0);
}

/// The relaxation must NOT relax the NaN firewall: NaN from finite inputs
/// (or any NaN endpoint) is still a hard error.
#[test]
fn test_ibp_nan_still_rejected_after_inf_relaxation() {
    let clip = ClipLayer::new(0.0, 6.0);
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![f32::NEG_INFINITY, f32::NAN]).unwrap(),
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![f32::INFINITY, 1.0]).unwrap(),
    )
    .unwrap();
    let err = clip
        .propagate_ibp(&input)
        .expect_err("NaN must not be absorbed alongside a tainted element");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}
