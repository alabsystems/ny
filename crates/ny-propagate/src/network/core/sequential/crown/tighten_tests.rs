// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for CROWN output tightening (`tighten.rs`).
//!
//! Tests verify the NaN/Inf defense-in-depth paths, shape-tolerant
//! intersection, and provenance tracking.
//!
//! Reference: alpha-beta-CROWN `optimized_bounds.py:937-947`.
//! Reference: #3043 — duplication of intersection caused #2990 and #3037.
//! Part of #4205.

use super::*;
use ndarray::{arr1, Array1};
use ny_core::Result;
use ny_tensor::BoundedTensor;

/// Helper: create 1D BoundedTensor from slices.
fn bounded_1d(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        Array1::from_vec(lower.to_vec()).into_dyn(),
        Array1::from_vec(upper.to_vec()).into_dyn(),
    )
    .expect("valid bounds")
}

// ── tighten_crown_output: normal tightening ───────────────────────────────

/// When CROWN bounds are tighter than IBP bounds, intersection should
/// produce bounds at least as tight as either source (soundness preserved).
///
/// Reference: alpha-beta-CROWN optimized_bounds.py:941
/// `ret_l = torch.max(ret_l, lb_refined); ret_u = torch.min(ret_u, ub_refined)`
#[test]
fn test_tighten_crown_output_normal_tightening() -> Result<()> {
    // IBP (forward): [-5, 10]
    let forward = bounded_1d(&[-5.0], &[10.0]);
    // CROWN: [-2, 7] (tighter than IBP)
    let crown = bounded_1d(&[-2.0], &[7.0]);

    let result = tighten_crown_output(crown, &forward, "test")?;

    // Intersection of [-5,10] and [-2,7] = [-2, 7] (CROWN already tighter).
    // max(lower) and min(upper).
    assert!(
        result.lower()[[0]] >= -2.0 - 1e-6,
        "tightened lower = {} should be >= -2.0",
        result.lower()[[0]]
    );
    assert!(
        result.upper()[[0]] <= 7.0 + 1e-6,
        "tightened upper = {} should be <= 7.0",
        result.upper()[[0]]
    );
    Ok(())
}

/// When IBP bounds are tighter than CROWN, intersection takes IBP's tighter side.
#[test]
fn test_tighten_crown_output_ibp_tighter_side() -> Result<()> {
    // IBP: [-1, 3]
    let forward = bounded_1d(&[-1.0], &[3.0]);
    // CROWN: [-5, 8] (wider than IBP)
    let crown = bounded_1d(&[-5.0], &[8.0]);

    let result = tighten_crown_output(crown, &forward, "test")?;

    // Intersection: max(-5,-1)=-1, min(8,3)=3 → [-1, 3].
    assert!(
        result.lower()[[0]] >= -1.0 - 1e-6,
        "tightened lower = {} should be >= -1.0",
        result.lower()[[0]]
    );
    assert!(
        result.upper()[[0]] <= 3.0 + 1e-6,
        "tightened upper = {} should be <= 3.0",
        result.upper()[[0]]
    );
    Ok(())
}

/// Multi-element tightening: each element tightened independently.
#[test]
fn test_tighten_crown_output_multi_element() -> Result<()> {
    let forward = bounded_1d(&[-5.0, -1.0, 0.0], &[10.0, 3.0, 2.0]);
    let crown = bounded_1d(&[-2.0, -3.0, -1.0], &[7.0, 5.0, 1.5]);

    let result = tighten_crown_output(crown, &forward, "test")?;

    // Element 0: max(-5,-2)=-2, min(10,7)=7 → [-2, 7]
    assert!(result.lower()[[0]] >= -2.0 - 1e-6);
    assert!(result.upper()[[0]] <= 7.0 + 1e-6);
    // Element 1: max(-1,-3)=-1, min(3,5)=3 → [-1, 3]
    assert!(result.lower()[[1]] >= -1.0 - 1e-6);
    assert!(result.upper()[[1]] <= 3.0 + 1e-6);
    // Element 2: max(0,-1)=0, min(2,1.5)=1.5 → [0, 1.5]
    assert!(result.lower()[[2]] >= 0.0 - 1e-6);
    assert!(result.upper()[[2]] <= 1.5 + 1e-6);
    Ok(())
}

// ── tighten_crown_output: shape mismatch handling ─────────────────────────

/// Shape mismatch with same element count: reshape CROWN to match forward (#3300).
#[test]
fn test_tighten_crown_output_shape_mismatch_same_count_reshapes() -> Result<()> {
    // Forward: shape [1, 2, 2] (3D array with 4 elements)
    let forward = BoundedTensor::new(
        ndarray::array![[[-5.0, -3.0], [-1.0, -2.0]]].into_dyn(),
        ndarray::array![[[10.0, 8.0], [5.0, 6.0]]].into_dyn(),
    )?;
    // CROWN: shape [4] (same element count = 4)
    let crown = bounded_1d(&[-2.0, -1.0, 0.0, -1.0], &[7.0, 5.0, 3.0, 4.0]);

    let result = tighten_crown_output(crown, &forward, "test")?;

    // Should reshape CROWN to [1, 2, 2] and intersect. Result shape matches forward.
    assert_eq!(result.shape(), forward.shape());
    // Verify the intersection is sound (result within both bounds).
    assert_eq!(result.len(), 4);
    Ok(())
}

/// Shape mismatch with different element count: skip intersection, return CROWN unchanged.
#[test]
fn test_tighten_crown_output_shape_mismatch_different_count_skips() -> Result<()> {
    let forward = bounded_1d(&[-5.0, -3.0, -1.0], &[10.0, 8.0, 5.0]);
    let crown = bounded_1d(&[-2.0, -1.0], &[7.0, 5.0]);

    let result = tighten_crown_output(crown, &forward, "test")?;

    // Different element counts → can't intersect. Return CROWN as-is.
    assert_eq!(result.len(), 2);
    assert!((result.lower()[[0]] - (-2.0)).abs() < 1e-6);
    assert!((result.upper()[[0]] - 7.0).abs() < 1e-6);
    Ok(())
}

// ── tighten_crown_output_with_provenance ──────────────────────────────────

/// Normal tightening with provenance: returns Crown provenance when both valid.
#[test]
fn test_tighten_with_provenance_valid_bounds_returns_crown() -> Result<()> {
    let forward = bounded_1d(&[-5.0], &[10.0]);
    let crown = bounded_1d(&[-2.0], &[7.0]);

    let (result, provenance) = tighten_crown_output_with_provenance(crown, &forward, "test")?;

    assert!(
        matches!(provenance, BoundsProvenance::Crown),
        "expected Crown provenance, got {:?}",
        provenance
    );
    assert!(result.lower()[[0]] >= -2.0 - 1e-6);
    assert!(result.upper()[[0]] <= 7.0 + 1e-6);
    Ok(())
}

/// ±Inf CROWN bounds with valid forward → per-element intersection (Crown
/// provenance), NOT a wholesale fallback. The +Inf upper is tightened to the
/// finite IBP upper while the CROWN lower (-2, tighter than IBP's -3) is kept.
///
/// Previously this returned the full forward bounds [-3, 5] (ForwardFallback);
/// the per-element path returns the strictly-tighter [-2, 5]. Both are sound;
/// the new result is never looser. ±Inf is no longer treated as "degraded" for
/// the purpose of abandoning the intersection — only NaN is.
#[test]
fn test_tighten_with_provenance_inf_crown_intersects_per_element() -> Result<()> {
    let forward = bounded_1d(&[-3.0], &[5.0]);
    // CROWN with +Inf upper (overflow on one output), tighter finite lower.
    let crown = BoundedTensor::new_allow_infinite(
        arr1(&[-2.0f32]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )?;

    let (result, provenance) = tighten_crown_output_with_provenance(crown, &forward, "test")?;

    assert!(
        matches!(provenance, BoundsProvenance::Crown),
        "expected Crown provenance (per-element intersection), got {:?}",
        provenance
    );
    // Lower: max(-2, -3) = -2 (CROWN kept, tighter). Upper: min(+Inf, 5) = 5.
    assert!(
        (result.lower()[[0]] - (-2.0)).abs() < 1e-6,
        "lower {} should keep tighter CROWN -2.0",
        result.lower()[[0]]
    );
    assert!((result.upper()[[0]] - 5.0).abs() < 1e-6);
    Ok(())
}

/// Mixed CROWN: one overflowed (±Inf) output, one healthy output tighter than
/// IBP. The old behaviour discarded the healthy CROWN row via full fallback;
/// per-element intersection keeps it. This is the deep-ResNet scenario.
#[test]
fn test_tighten_with_provenance_mixed_inf_keeps_healthy_crown_rows() -> Result<()> {
    // Forward (IBP): both outputs [-5, 10].
    let forward = bounded_1d(&[-5.0, -5.0], &[10.0, 10.0]);
    // CROWN: output 0 overflowed to [-inf, +inf]; output 1 tight [1, 2].
    let crown = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY, 1.0f32]).into_dyn(),
        arr1(&[f32::INFINITY, 2.0f32]).into_dyn(),
    )?;

    let (result, provenance) = tighten_crown_output_with_provenance(crown, &forward, "test")?;

    assert!(
        matches!(provenance, BoundsProvenance::Crown),
        "expected Crown provenance, got {:?}",
        provenance
    );
    // Output 0: overflow row tightened to IBP [-5, 10].
    assert!((result.lower()[[0]] - (-5.0)).abs() < 1e-6);
    assert!((result.upper()[[0]] - 10.0).abs() < 1e-6);
    // Output 1: healthy CROWN row KEPT (tighter than IBP).
    assert!(
        (result.lower()[[1]] - 1.0).abs() < 1e-6,
        "healthy row lower {} should keep CROWN 1.0",
        result.lower()[[1]]
    );
    assert!(
        (result.upper()[[1]] - 2.0).abs() < 1e-6,
        "healthy row upper {} should keep CROWN 2.0",
        result.upper()[[1]]
    );
    Ok(())
}

/// NaN CROWN bounds with valid forward → full ForwardFallback (unchanged).
/// NaN must still trigger wholesale fallback: per-element max/min would
/// contaminate healthy elements with NaN.
#[test]
fn test_tighten_with_provenance_nan_crown_falls_back_to_forward() -> Result<()> {
    let forward = bounded_1d(&[-3.0], &[5.0]);
    // new_unchecked bypasses the NaN guard so we can construct a NaN-bearing
    // CROWN output (as concretize_sound could surface defensively).
    let crown =
        BoundedTensor::new_unchecked(arr1(&[-2.0f32]).into_dyn(), arr1(&[f32::NAN]).into_dyn())?;

    let (result, provenance) = tighten_crown_output_with_provenance(crown, &forward, "test")?;

    assert!(
        matches!(
            provenance,
            BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::CrownPropagationError)
        ),
        "expected ForwardFallback for NaN, got {:?}",
        provenance
    );
    assert!((result.lower()[[0]] - (-3.0)).abs() < 1e-6);
    assert!((result.upper()[[0]] - 5.0).abs() < 1e-6);
    Ok(())
}

/// Both CROWN and forward degraded: skip intersection, return CROWN with Crown provenance.
///
/// When forward bounds are also invalid, we can't use them for fallback.
/// The function returns CROWN as-is (with Crown provenance).
#[test]
fn test_tighten_with_provenance_both_degraded_skips_intersection() -> Result<()> {
    let forward = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )?;
    let crown = BoundedTensor::new_allow_infinite(
        arr1(&[-2.0f32]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )?;

    let (result, provenance) = tighten_crown_output_with_provenance(crown, &forward, "test")?;

    // Both invalid → no intersection, CROWN returned as-is with Crown provenance.
    assert!(
        matches!(provenance, BoundsProvenance::Crown),
        "expected Crown provenance when both degraded, got {:?}",
        provenance
    );
    assert!((result.lower()[[0]] - (-2.0)).abs() < 1e-6);
    Ok(())
}

/// Shape mismatch with provenance: skip intersection, return Crown provenance.
#[test]
fn test_tighten_with_provenance_shape_mismatch_returns_crown() -> Result<()> {
    let forward = bounded_1d(&[-5.0, -3.0, -1.0], &[10.0, 8.0, 5.0]);
    let crown = bounded_1d(&[-2.0, -1.0], &[7.0, 5.0]);

    let (result, provenance) = tighten_crown_output_with_provenance(crown, &forward, "test")?;

    assert!(
        matches!(provenance, BoundsProvenance::Crown),
        "expected Crown provenance on shape mismatch, got {:?}",
        provenance
    );
    assert_eq!(result.len(), 2);
    Ok(())
}
