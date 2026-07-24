// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared CROWN utilities used by both sequential `Network` and `GraphNetwork`
//! propagation paths.
//!
//! Moved from `graph_crown/propagation.rs` to break the bidirectional dependency
//! between `core/` and `graph_crown/` (#2380).

use ny_core::{nan_propagating_max, nan_propagating_min, NyError, Result};
use ny_tensor::BoundedTensor;

/// Tighten CROWN output bounds by intersecting with forward (IBP/CROWN-IBP) bounds.
///
/// When CROWN and forward bounds overlap, takes the intersection (tighter).
/// When they are disjoint (indicating at least one side is unsound), takes
/// the union to preserve soundness.
///
/// Returns the tightened bounds and the number of disjoint element intervals.
///
/// Reference: alpha-beta-CROWN `optimized_bounds.py:941-946`.
pub(crate) fn tighten_crown_with_forward_bounds(
    crown_output: &BoundedTensor,
    output_bounds: &BoundedTensor,
) -> Result<(BoundedTensor, usize)> {
    // Upgraded from debug_assert_eq! to runtime check (#2875, #2920 WP-C).
    // In release mode, mismatched shapes silently truncate the zip, producing
    // partial tightening with incorrect disjoint_count.
    if crown_output.shape() != output_bounds.shape() {
        return Err(NyError::ShapeMismatch {
            expected: crown_output.shape().to_vec(),
            got: output_bounds.shape().to_vec(),
        });
    }

    let mut crown_lower = crown_output.lower().clone();
    let mut crown_upper = crown_output.upper().clone();
    let output_lower = output_bounds.lower();
    let output_upper = output_bounds.upper();
    let mut disjoint_count = 0usize;

    for ((cl, cu), (il, iu)) in crown_lower
        .iter_mut()
        .zip(crown_upper.iter_mut())
        .zip(output_lower.iter().zip(output_upper.iter()))
    {
        // NaN-safe: nan_propagating_{max,min} return NaN if either operand is NaN,
        // preventing silent absorption (IEEE 754: NaN.max(x) = x). (#2643)
        let tightened_lower = nan_propagating_max(*cl, *il);
        let tightened_upper = nan_propagating_min(*cu, *iu);
        if tightened_lower <= tightened_upper {
            // Intersection is non-empty: tighten to the overlap.
            *cl = tightened_lower;
            *cu = tightened_upper;
        } else {
            // Non-overlapping intervals imply at least one input bound is unsound.
            // Use union to preserve soundness regardless of which side is wrong.
            // NaN comparison: <= returns false for NaN, so NaN values land here
            // and propagate through the union path.
            disjoint_count += 1;
            *cl = nan_propagating_min(*cl, *il);
            *cu = nan_propagating_max(*cu, *iu);
        }
    }

    // Use new_allow_infinite: CROWN output may contain ±Inf from #2681
    // non-finite row fallback. After per-element intersection with finite IBP
    // bounds, affected elements become finite. If both CROWN and IBP are
    // infinite for an element, ±Inf is the correct (maximally loose) result.
    Ok((
        BoundedTensor::new_allow_infinite(crown_lower, crown_upper)?,
        disjoint_count,
    ))
}

#[cfg(test)]
mod tests {
    use super::tighten_crown_with_forward_bounds;
    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    #[test]
    fn test_tighten_crown_with_forward_bounds_uses_intersection_when_overlapping() {
        let crown = BoundedTensor::new(
            arr1(&[-3.0_f32, -2.0]).into_dyn(),
            arr1(&[5.0_f32, 6.0]).into_dyn(),
        )
        .unwrap();
        let forward = BoundedTensor::new(
            arr1(&[-1.0_f32, -4.0]).into_dyn(),
            arr1(&[2.0_f32, 3.0]).into_dyn(),
        )
        .unwrap();

        let (tightened, disjoint_count) =
            tighten_crown_with_forward_bounds(&crown, &forward).unwrap();

        assert_eq!(disjoint_count, 0);
        assert_eq!(tightened.lower()[[0]], -1.0);
        assert_eq!(tightened.upper()[[0]], 2.0);
        assert_eq!(tightened.lower()[[1]], -2.0);
        assert_eq!(tightened.upper()[[1]], 3.0);
    }

    #[test]
    fn test_tighten_crown_with_forward_bounds_uses_union_when_disjoint() {
        let crown =
            BoundedTensor::new(arr1(&[3.0_f32]).into_dyn(), arr1(&[4.0_f32]).into_dyn()).unwrap();
        let forward =
            BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn()).unwrap();

        let (tightened, disjoint_count) =
            tighten_crown_with_forward_bounds(&crown, &forward).unwrap();

        assert_eq!(disjoint_count, 1);
        assert_eq!(tightened.lower()[[0]], 1.0);
        assert_eq!(tightened.upper()[[0]], 4.0);
    }

    /// #2681: ±Inf CROWN bounds (from non-finite A-matrix row fallback) should
    /// be tightened to IBP via per-element intersection. This is the core
    /// mechanism that makes per-row IBP fallback work.
    #[test]
    fn test_tighten_crown_with_forward_bounds_inf_crown_tightened_to_ibp() {
        // Row 0: CROWN overflowed → [-inf, +inf], forward has [1.0, 5.0]
        // Row 1: CROWN [2.0, 4.0] is tighter than forward [0.0, 6.0]
        let crown = BoundedTensor::new_allow_infinite(
            arr1(&[f32::NEG_INFINITY, 2.0_f32]).into_dyn(),
            arr1(&[f32::INFINITY, 4.0_f32]).into_dyn(),
        )
        .unwrap();
        let forward = BoundedTensor::new(
            arr1(&[1.0_f32, 0.0_f32]).into_dyn(),
            arr1(&[5.0_f32, 6.0_f32]).into_dyn(),
        )
        .unwrap();

        let (tightened, disjoint_count) =
            tighten_crown_with_forward_bounds(&crown, &forward).unwrap();

        assert_eq!(disjoint_count, 0, "[-inf,+inf] ∩ [1,5] is non-empty");

        // Row 0: max(-inf, 1.0) = 1.0, min(+inf, 5.0) = 5.0
        assert_eq!(
            tightened.lower()[[0]],
            1.0,
            "overflow row tightened to IBP lower"
        );
        assert_eq!(
            tightened.upper()[[0]],
            5.0,
            "overflow row tightened to IBP upper"
        );

        // Row 1: max(2.0, 0.0) = 2.0, min(4.0, 6.0) = 4.0
        assert_eq!(tightened.lower()[[1]], 2.0, "healthy row keeps CROWN lower");
        assert_eq!(tightened.upper()[[1]], 4.0, "healthy row keeps CROWN upper");
    }
}
