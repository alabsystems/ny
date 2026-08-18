// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::{
    next_down_f32, next_up_f32, repair_inverted_bounds, repair_inverted_bounds_nd, BoundedTensor,
    InversionRepair,
};
use ndarray::{arr1, arr2, arr3, IxDyn};
use ny_core::NyError;

#[test]
fn test_concrete_tensor() {
    let values = arr1(&[1.0, 2.0, 3.0]).into_dyn();
    let t = BoundedTensor::concrete(values).expect("invariant: valid concrete tensor");
    assert_eq!(t.max_width(), 0.0);
}

#[test]
fn test_epsilon_perturbation() {
    let values = arr1(&[0.0, 1.0]).into_dyn();
    let t =
        BoundedTensor::from_epsilon(values, 0.1).expect("invariant: valid epsilon perturbation");
    assert!(
        (t.lower()[[0]] - (-0.1)).abs() < 1e-6,
        "lower[0] should be ≈ -0.1, got {}",
        t.lower()[[0]]
    );
    assert!(
        (t.upper()[[0]] - 0.1).abs() < 1e-6,
        "upper[0] should be ≈ 0.1, got {}",
        t.upper()[[0]]
    );
}

#[test]
fn test_addition() {
    let a = BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn()).unwrap();
    let b = BoundedTensor::new(arr1(&[0.5, 0.5]).into_dyn(), arr1(&[1.5, 1.5]).into_dyn()).unwrap();

    let c = a.add(&b).unwrap();
    assert_eq!(c.lower()[[0]], 0.5);
    assert_eq!(c.upper()[[0]], 2.5);
}

/// Regression test for #2743: BoundedTensor::add must repair NaN from inf + (-inf).
///
/// When lower[i] = -inf and other.lower[i] = +inf (or vice versa), IEEE-754
/// produces NaN. The add() method must repair this to the conservative
/// unbounded direction, matching the pattern in AddLayer::propagate_ibp_binary.
#[test]
fn test_add_inf_minus_inf_repairs_nan_conservatively() {
    // a.lower = [-inf], a.upper = [5.0]
    let a = BoundedTensor::from_parts_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[5.0]).into_dyn(),
    );
    // b.lower = [+inf], b.upper = [10.0]
    let b = BoundedTensor::from_parts_unchecked(
        arr1(&[f32::INFINITY]).into_dyn(),
        arr1(&[10.0]).into_dyn(),
    );

    // -inf + inf = NaN under IEEE-754; must be repaired
    let result = a.add(&b).expect("add should not fail on matching shapes");
    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];

    assert!(
        !lower.is_nan(),
        "lower bound must not be NaN after repair, got {lower}"
    );
    assert!(
        !upper.is_nan(),
        "upper bound must not be NaN after repair, got {upper}"
    );
    // Lower was NaN (from -inf + inf), repaired to -inf: a NaN endpoint proves
    // nothing, so the only sound repair is the unbounded direction.
    assert_eq!(
        lower,
        f32::NEG_INFINITY,
        "NaN lower should be repaired to -inf"
    );
    // Upper was 5.0 + 10.0 = 15.0 (finite), preserved as-is
    assert_eq!(upper, 15.0, "finite upper should be preserved");
}

/// Regression test for #2743: add with both endpoints producing NaN.
#[test]
fn test_add_both_endpoints_nan_repaired() {
    // a = [-inf, +inf], b = [+inf, -inf] → lower: -inf+inf=NaN, upper: inf+(-inf)=NaN
    let a = BoundedTensor::from_parts_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    );
    let b = BoundedTensor::from_parts_unchecked(
        arr1(&[f32::INFINITY]).into_dyn(),
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
    );

    let result = a.add(&b).expect("add should succeed");
    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];

    assert!(!lower.is_nan(), "lower must not be NaN");
    assert!(!upper.is_nan(), "upper must not be NaN");
    // Both endpoints were NaN; the only sound repair is the unbounded interval.
    assert_eq!(lower, f32::NEG_INFINITY);
    assert_eq!(upper, f32::INFINITY);
}

/// Verify add preserves finite results unchanged (no false widening).
#[test]
fn test_add_finite_values_unchanged() {
    let a = BoundedTensor::new(arr1(&[1.0, -3.0]).into_dyn(), arr1(&[2.0, -1.0]).into_dyn())
        .expect("invariant: valid ascending bounds");
    let b = BoundedTensor::new(arr1(&[0.5, 0.5]).into_dyn(), arr1(&[1.5, 1.5]).into_dyn())
        .expect("invariant: valid ascending bounds");

    let result = a.add(&b).expect("invariant: matching shapes");
    // 1.0 + 0.5 = 1.5
    assert_eq!(result.lower()[[0]], 1.5);
    // 2.0 + 1.5 = 3.5
    assert_eq!(result.upper()[[0]], 3.5);
    // -3.0 + 0.5 = -2.5
    assert_eq!(result.lower()[[1]], -2.5);
    // -1.0 + 1.5 = 0.5
    assert_eq!(result.upper()[[1]], 0.5);
}

/// Regression test for #2764: scale(0) must repair NaN from 0*inf.
#[test]
fn test_scale_zero_times_infinite_bounds_repairs_non_finite() {
    let t = BoundedTensor::from_parts_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    );
    let result = t.scale(0.0);
    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];

    assert!(!lower.is_nan(), "lower bound must not be NaN");
    assert!(!upper.is_nan(), "upper bound must not be NaN");
    // 0 * ±inf = NaN on both endpoints → repaired to the unbounded interval.
    assert_eq!(lower, f32::NEG_INFINITY);
    assert_eq!(upper, f32::INFINITY);
}

/// Regression test for #2764: shift(neg_inf) on positive bounds must repair NaN.
///
/// inf + (-inf) = NaN under IEEE-754; must be repaired to the conservative
/// unbounded direction. new_repaired preserves ±inf endpoints as-is (#3423).
#[test]
fn test_shift_inf_plus_neg_inf_repairs_nan_conservatively() {
    // upper = +inf, shift by -inf → inf + (-inf) = NaN
    let t = BoundedTensor::from_parts_unchecked(
        arr1(&[-1.0]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    );
    let result = t.shift(f32::NEG_INFINITY);
    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];

    assert!(
        !lower.is_nan(),
        "lower bound must not be NaN after repair, got {lower}"
    );
    assert!(
        !upper.is_nan(),
        "upper bound must not be NaN after repair, got {upper}"
    );
    // -1.0 + (-inf) = -inf (genuine arithmetic result, preserved)
    assert_eq!(lower, f32::NEG_INFINITY);
    // inf + (-inf) = NaN → repaired to +inf (conservative direction)
    assert_eq!(upper, f32::INFINITY);
}

/// shift() with finite inputs should preserve exact arithmetic.
#[test]
fn test_shift_finite_values_unchanged() {
    let t = BoundedTensor::new(arr1(&[-2.0, 5.0]).into_dyn(), arr1(&[3.0, 10.0]).into_dyn())
        .expect("invariant: valid ascending bounds");

    let result = t.shift(1.5);
    assert_eq!(result.lower()[[0]], -0.5);
    assert_eq!(result.upper()[[0]], 4.5);
    assert_eq!(result.lower()[[1]], 6.5);
    assert_eq!(result.upper()[[1]], 11.5);
}

/// scale() with finite inputs should preserve exact arithmetic.
#[test]
fn test_scale_finite_values_unchanged() {
    let t = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[5.0]).into_dyn())
        .expect("invariant: valid ascending bounds");

    let positive = t.scale(3.0);
    assert_eq!(positive.lower()[[0]], -6.0);
    assert_eq!(positive.upper()[[0]], 15.0);

    let negative = t.scale(-2.0);
    assert_eq!(negative.lower()[[0]], -10.0);
    assert_eq!(negative.upper()[[0]], 4.0);
}

/// Regression test (#2895): scale(NaN) must return conservative bounds, not swap bounds.
///
/// Without the guard, `NaN >= 0.0` → false (IEEE 754), so NaN was treated as negative,
/// swapping bounds unnecessarily before new_repaired fixed them to conservative (#3423).
/// The guard now returns conservative bounds directly without the pointless swap.
#[test]
fn test_scale_nan_returns_conservative_2895() {
    let t = BoundedTensor::new(arr1(&[-2.0, 5.0]).into_dyn(), arr1(&[3.0, 10.0]).into_dyn())
        .expect("invariant: valid ascending bounds");

    let result = t.scale(f32::NAN);
    assert_eq!(
        result.lower()[[0]],
        f32::NEG_INFINITY,
        "NaN scalar must produce conservative lower bound"
    );
    assert_eq!(
        result.upper()[[0]],
        f32::INFINITY,
        "NaN scalar must produce conservative upper bound"
    );
    // Bounds must still be valid (lower <= upper)
    assert!(
        result.lower()[[0]] <= result.upper()[[0]],
        "element 0: lower {} must not exceed upper {}",
        result.lower()[[0]],
        result.upper()[[0]]
    );
    assert!(
        result.lower()[[1]] <= result.upper()[[1]],
        "element 1: lower {} must not exceed upper {}",
        result.lower()[[1]],
        result.upper()[[1]]
    );
}

/// Regression test (#2895): scale(Inf) must return conservative bounds.
///
/// Inf * finite = Inf, and Inf * 0 = NaN. The guard catches Inf scalars
/// before any per-element multiplication.
#[test]
fn test_scale_inf_returns_conservative_2895() {
    let t = BoundedTensor::new(arr1(&[-2.0, 0.0]).into_dyn(), arr1(&[3.0, 0.0]).into_dyn())
        .expect("invariant: valid ascending bounds");

    let pos_inf = t.scale(f32::INFINITY);
    assert_eq!(pos_inf.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(pos_inf.upper()[[0]], f32::INFINITY);

    let neg_inf = t.scale(f32::NEG_INFINITY);
    assert_eq!(neg_inf.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(neg_inf.upper()[[0]], f32::INFINITY);
}

#[test]
fn test_mul_zero_times_unbounded_interval_never_returns_nan() {
    let zero = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[0.0]).into_dyn()).unwrap();
    let unbounded = BoundedTensor::from_parts_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    );

    let result = zero.mul(&unbounded).unwrap();
    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];

    assert!(!lower.is_nan(), "lower bound must not be NaN");
    assert!(!upper.is_nan(), "upper bound must not be NaN");
    assert!(lower <= upper, "bounds must remain ordered");

    // Accept either exact-zero tightening or conservative unbounded widening.
    let exact_zero = lower == 0.0 && upper == 0.0;
    let conservative_unbounded = lower == f32::NEG_INFINITY && upper == f32::INFINITY;
    assert!(
        exact_zero || conservative_unbounded,
        "expected [0,0] or [-inf,inf], got [{lower}, {upper}]"
    );
}

#[test]
fn test_mul_crosses_zero_with_unbounded_interval_never_returns_nan() {
    let crosses_zero =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let unbounded = BoundedTensor::from_parts_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    );

    let result = crosses_zero.mul(&unbounded).unwrap();
    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];

    assert!(!lower.is_nan(), "lower bound must not be NaN");
    assert!(!upper.is_nan(), "upper bound must not be NaN");
    assert!(lower <= upper, "bounds must remain ordered");

    // This case can be represented as finite bounds in tighter implementations,
    // or conservatively widened to [-inf, inf].
    let finite_bounds = lower.is_finite() && upper.is_finite();
    let conservative_unbounded = lower == f32::NEG_INFINITY && upper == f32::INFINITY;
    assert!(
        finite_bounds || conservative_unbounded,
        "expected finite bounds or [-inf,inf], got [{lower}, {upper}]"
    );
}

/// Regression test: mixed NaN products must widen to [-inf, inf].
///
/// `[-1, 0] * [inf, inf]` produces products `(-inf, -inf, NaN, NaN)`.
/// Dropping the NaN products would give `[-inf, -inf]`, which is too tight
/// because by continuous extension, 0 * inf should include 0 in the upper bound.
/// The sound result is `[-inf, inf]` (conservative widening on any NaN).
///
/// Reference: alpha-beta-CROWN interval_bound.py:73-93 widens any NaN to ±inf.
#[test]
fn test_mul_mixed_nan_products_widen_to_unbounded() {
    let neg_to_zero =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[0.0]).into_dyn()).unwrap();
    let pos_inf = BoundedTensor::from_parts_unchecked(
        arr1(&[f32::INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    );

    let result = neg_to_zero.mul(&pos_inf).unwrap();
    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];

    assert!(!lower.is_nan(), "lower bound must not be NaN");
    assert!(!upper.is_nan(), "upper bound must not be NaN");
    assert!(lower <= upper, "bounds must remain ordered");

    // Must be [-inf, inf] because 0*inf=NaN makes the result indeterminate.
    // Dropping the NaN and returning [-inf, -inf] would be unsound.
    assert_eq!(
        lower,
        f32::NEG_INFINITY,
        "lower must be -inf (conservative widening)"
    );
    assert_eq!(
        upper,
        f32::INFINITY,
        "upper must be +inf (conservative widening)"
    );
}

#[test]
fn test_slice_axis_basic() {
    // Shape: [2, 3]
    let lower = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
    let upper = arr2(&[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]).into_dyn();
    let t = BoundedTensor::new(lower, upper).unwrap();

    // Slice along axis 0 (select row)
    let row0 = t.slice_axis(0, 0).unwrap();
    assert_eq!(row0.shape(), &[3]);
    assert_eq!(row0.lower()[[0]], 1.0);
    assert_eq!(row0.lower()[[2]], 3.0);

    let row1 = t.slice_axis(0, 1).unwrap();
    assert_eq!(row1.shape(), &[3]);
    assert_eq!(row1.lower()[[0]], 4.0);

    // Slice along axis 1 (select column)
    let col1 = t.slice_axis(1, 1).unwrap();
    assert_eq!(col1.shape(), &[2]);
    assert_eq!(col1.lower()[[0]], 2.0);
    assert_eq!(col1.lower()[[1]], 5.0);
}

#[test]
fn test_slice_axis_3d() {
    // Shape: [batch=1, seq=4, hidden=3]
    let lower = arr3(&[[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]])
    .into_dyn();
    let upper = lower.mapv(|x| x + 0.5);
    let t = BoundedTensor::new(lower, upper).unwrap();

    // Slice position 2 along seq axis (axis=1)
    let pos2 = t.slice_axis(1, 2).unwrap();
    assert_eq!(pos2.shape(), &[1, 3]); // [batch, hidden]
    assert_eq!(pos2.lower()[[0, 0]], 7.0);
    assert_eq!(pos2.lower()[[0, 2]], 9.0);
}

#[test]
fn test_slice_axis_range() {
    // Shape: [batch=1, seq=4, hidden=3]
    let lower = arr3(&[[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]])
    .into_dyn();
    let upper = lower.mapv(|x| x + 0.5);
    let t = BoundedTensor::new(lower, upper).unwrap();

    // Slice range [1, 3) along seq axis
    let mid = t.slice_axis_range(1, 1, 3).unwrap();
    assert_eq!(mid.shape(), &[1, 2, 3]); // [batch, seq=2, hidden]
    assert_eq!(mid.lower()[[0, 0, 0]], 4.0); // Was position 1
    assert_eq!(mid.lower()[[0, 1, 0]], 7.0); // Was position 2
}

#[test]
fn test_expand_axis() {
    // Shape: [3]
    let lower = arr1(&[1.0, 2.0, 3.0]).into_dyn();
    let upper = arr1(&[1.5, 2.5, 3.5]).into_dyn();
    let t = BoundedTensor::new(lower, upper).unwrap();

    // Expand at axis 0
    let expanded = t.expand_axis(0).unwrap();
    assert_eq!(expanded.shape(), &[1, 3]);
    assert_eq!(expanded.lower()[[0, 1]], 2.0);

    // Expand at axis 1
    let expanded2 = t.expand_axis(1).unwrap();
    assert_eq!(expanded2.shape(), &[3, 1]);
    assert_eq!(expanded2.lower()[[1, 0]], 2.0);
}

#[test]
fn test_stack_positions() {
    // Simulate stacking position outputs back together
    // Each position has shape [batch=1, hidden=3]
    let pos0 = BoundedTensor::new(
        arr2(&[[1.0, 2.0, 3.0]]).into_dyn(),
        arr2(&[[1.5, 2.5, 3.5]]).into_dyn(),
    )
    .unwrap();

    let pos1 = BoundedTensor::new(
        arr2(&[[4.0, 5.0, 6.0]]).into_dyn(),
        arr2(&[[4.5, 5.5, 6.5]]).into_dyn(),
    )
    .unwrap();

    let pos2 = BoundedTensor::new(
        arr2(&[[7.0, 8.0, 9.0]]).into_dyn(),
        arr2(&[[7.5, 8.5, 9.5]]).into_dyn(),
    )
    .unwrap();

    // Stack along axis 1 to get [batch=1, seq=3, hidden=3]
    let stacked = BoundedTensor::stack(&[pos0, pos1, pos2], 1).unwrap();
    assert_eq!(stacked.shape(), &[1, 3, 3]);
    assert_eq!(stacked.lower()[[0, 0, 0]], 1.0);
    assert_eq!(stacked.lower()[[0, 1, 0]], 4.0);
    assert_eq!(stacked.lower()[[0, 2, 0]], 7.0);
}

#[test]
fn test_concat_positions() {
    // Simulate concatenating position outputs with expand_axis
    // Each position has shape [batch=1, hidden=3]
    let pos0 = BoundedTensor::new(
        arr2(&[[1.0, 2.0, 3.0]]).into_dyn(),
        arr2(&[[1.5, 2.5, 3.5]]).into_dyn(),
    )
    .unwrap();

    let pos1 = BoundedTensor::new(
        arr2(&[[4.0, 5.0, 6.0]]).into_dyn(),
        arr2(&[[4.5, 5.5, 6.5]]).into_dyn(),
    )
    .unwrap();

    // First expand to [batch=1, seq=1, hidden=3]
    let pos0_exp = pos0.expand_axis(1).unwrap();
    let pos1_exp = pos1.expand_axis(1).unwrap();

    // Then concat along axis 1
    let combined = BoundedTensor::concat(&[pos0_exp, pos1_exp], 1).unwrap();
    assert_eq!(combined.shape(), &[1, 2, 3]);
    assert_eq!(combined.lower()[[0, 0, 0]], 1.0);
    assert_eq!(combined.lower()[[0, 1, 0]], 4.0);
}

#[test]
fn test_slice_and_stack_roundtrip() {
    // Start with [batch=1, seq=4, hidden=3]
    let lower = arr3(&[[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]])
    .into_dyn();
    let upper = lower.mapv(|x| x + 0.5);
    let original = BoundedTensor::new(lower, upper).unwrap();

    // Slice into individual positions
    let positions: Vec<_> = (0..4).map(|i| original.slice_axis(1, i).unwrap()).collect();

    // Stack them back together
    let reconstructed = BoundedTensor::stack(&positions, 1).unwrap();

    // Should be identical
    assert_eq!(reconstructed.shape(), original.shape());
    assert!(
        reconstructed
            .lower()
            .iter()
            .zip(original.lower().iter())
            .all(|(a, b)| (a - b).abs() < 1e-6),
        "stack(lower) roundtrip should preserve all elements within 1e-6"
    );
}

#[test]
fn test_slice_axis_errors() {
    let t = BoundedTensor::concrete(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn())
        .expect("invariant: valid concrete tensor");

    // Axis out of bounds
    assert!(
        t.slice_axis(5, 0).is_err(),
        "slice_axis should reject axis 5 for a rank-2 tensor"
    );

    // Index out of bounds
    assert!(
        t.slice_axis(0, 10).is_err(),
        "slice_axis should reject index 10 for axis 0 with len 2"
    );
}

#[test]
fn test_flatten_preserves_row_major_order_for_standard_layout() {
    let lower = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
    let upper = arr2(&[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]).into_dyn();
    let t = BoundedTensor::new(lower, upper).unwrap();

    let flat = t.flatten();

    assert_eq!(flat.shape(), &[6]);
    assert_eq!(
        flat.lower().as_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        flat.upper().as_slice().unwrap(),
        &[1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    );
}

#[test]
fn test_flatten_to_ix1_matches_flatten_bounds() {
    let lower = arr2(&[[1.0, -2.0, 3.5], [4.0, 5.0, -6.0]]).into_dyn();
    let upper = arr2(&[[1.5, -1.0, 4.0], [4.5, 6.0, -5.0]]).into_dyn();
    let t = BoundedTensor::new(lower, upper).unwrap();

    let flat = t.flatten();
    let (lower_ix1, upper_ix1) = t
        .flatten_to_ix1("test_flatten_to_ix1_matches_flatten_bounds")
        .unwrap();

    assert_eq!(lower_ix1.len(), flat.len());
    assert_eq!(upper_ix1.len(), flat.len());

    assert_eq!(
        lower_ix1.as_slice().unwrap(),
        flat.lower().as_slice().unwrap()
    );
    assert_eq!(
        upper_ix1.as_slice().unwrap(),
        flat.upper().as_slice().unwrap()
    );
}

#[test]
fn test_flatten_handles_non_standard_layout_without_panicking() {
    let lower = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        .permuted_axes([1, 0])
        .into_dyn();
    let upper = arr2(&[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
        .permuted_axes([1, 0])
        .into_dyn();

    assert!(
        !lower.is_standard_layout(),
        "Test precondition failed: expected non-standard layout"
    );

    // This reshape path used to panic for non-standard layout.
    assert!(
        lower.clone().into_shape_with_order(IxDyn(&[6])).is_err(),
        "non-standard layout lower should reject flat reshape"
    );
    assert!(
        upper.clone().into_shape_with_order(IxDyn(&[6])).is_err(),
        "non-standard layout upper should reject flat reshape"
    );

    let t = BoundedTensor::new(lower, upper).unwrap();
    let flat = t.flatten();

    // Flatten should follow logical row-major order of the permuted [3,2] view:
    // [[1,4],[2,5],[3,6]].
    assert_eq!(flat.shape(), &[6]);
    assert_eq!(
        flat.lower().as_slice().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    assert_eq!(
        flat.upper().as_slice().unwrap(),
        &[1.5, 4.5, 2.5, 5.5, 3.5, 6.5]
    );
}

// === NaN/Inf Detection Tests ===

#[test]
fn test_has_nan_or_inf_normal_values() {
    let arr = arr1(&[1.0, 2.0, 3.0, -1.0, 0.0]).into_dyn();
    let tensor = BoundedTensor::new(arr.clone(), arr).unwrap();
    assert!(
        !tensor.has_overflow(),
        "normal finite values should not trigger overflow"
    );
}

#[test]
fn test_new_rejects_nan_bounds() {
    let lower = arr1(&[f32::NAN, 0.0]).into_dyn();
    let upper = arr1(&[1.0, 1.0]).into_dyn();
    let err = BoundedTensor::new(lower, upper).unwrap_err();
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "NaN bounds should return NumericalInstability, got {err:?}"
    );
}

#[test]
fn test_new_rejects_infinite_bounds() {
    let lower = arr1(&[0.0, 0.0]).into_dyn();
    let upper = arr1(&[1.0, f32::INFINITY]).into_dyn();
    let err = BoundedTensor::new(lower, upper).unwrap_err();
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "infinite bounds should return NumericalInstability, got {err:?}"
    );
}

#[test]
fn test_new_conservative_constructs_unbounded_bounds() {
    let tensor = BoundedTensor::new_conservative(&[2, 3]);

    assert_eq!(tensor.shape(), &[2, 3]);
    assert!(
        tensor.lower().iter().all(|&v| v == f32::NEG_INFINITY),
        "conservative lower should be all -inf"
    );
    assert!(
        tensor.upper().iter().all(|&v| v == f32::INFINITY),
        "conservative upper should be all +inf"
    );
}

#[test]
fn test_new_conservative_supports_scalar_shape() {
    let tensor = BoundedTensor::new_conservative(&[]);

    assert_eq!(tensor.shape(), &[] as &[usize]);
    assert_eq!(tensor.lower()[IxDyn(&[])], f32::NEG_INFINITY);
    assert_eq!(tensor.upper()[IxDyn(&[])], f32::INFINITY);
}

#[test]
fn test_try_get_returns_infinite_bound_for_conservative_tensor() {
    let tensor = BoundedTensor::new_conservative(&[1]);

    let bound = tensor
        .try_get(&[0])
        .expect("valid scalar index should not error")
        .expect("conservative element should remain feasible");

    assert_eq!(bound.lower(), f32::NEG_INFINITY);
    assert_eq!(bound.upper(), f32::INFINITY);
}

#[test]
fn test_get_accepts_infinite_feasible_bounds() {
    let tensor = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[5.0]).into_dyn(),
    )
    .expect("mixed finite/infinite bound should be valid");

    let bound = tensor.get(&[0]);
    assert_eq!(bound.lower(), f32::NEG_INFINITY);
    assert_eq!(bound.upper(), 5.0);
}

#[test]
fn test_new_rejects_inverted_bounds() {
    let lower = arr1(&[2.0, -1.0]).into_dyn();
    let upper = arr1(&[1.0, 0.0]).into_dyn();
    let err = BoundedTensor::new(lower, upper).unwrap_err();
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "inverted bounds should return InvalidSpec, got {err:?}"
    );
}

#[test]
fn test_has_nan_or_inf_with_nan() {
    let arr = arr1(&[1.0, f32::NAN, 3.0]).into_dyn();
    let tensor = BoundedTensor::new_unchecked(arr.clone(), arr).unwrap();
    assert!(tensor.has_overflow(), "NaN element should trigger overflow");
}

#[test]
fn test_has_nan_or_inf_with_pos_inf() {
    let arr = arr1(&[1.0, f32::INFINITY, 3.0]).into_dyn();
    let tensor = BoundedTensor::new_unchecked(arr.clone(), arr).unwrap();
    assert!(
        tensor.has_overflow(),
        "+inf element should trigger overflow"
    );
}

#[test]
fn test_has_nan_or_inf_with_neg_inf() {
    let arr = arr1(&[1.0, f32::NEG_INFINITY, 3.0]).into_dyn();
    let tensor = BoundedTensor::new_unchecked(arr.clone(), arr).unwrap();
    assert!(
        tensor.has_overflow(),
        "-inf element should trigger overflow"
    );
}

// === Edge Case Tests (f32::MAX, denormals, negative zero) ===

#[test]
fn test_f32_max_bounds() {
    // f32::MAX is a valid bound value (not infinity)
    let lower = arr1(&[f32::MIN, -f32::MAX, 0.0]).into_dyn();
    let upper = arr1(&[0.0, 0.0, f32::MAX]).into_dyn();
    let t = BoundedTensor::new(lower, upper).unwrap();
    assert_eq!(t.upper()[[2]], f32::MAX);
    assert_eq!(t.lower()[[0]], f32::MIN);
}

#[test]
fn test_denormal_values() {
    // Denormal (subnormal) numbers are valid - smallest positive f32
    let tiny = f32::MIN_POSITIVE / 2.0; // Subnormal value
    let lower = arr1(&[-tiny, 0.0]).into_dyn();
    let upper = arr1(&[0.0, tiny]).into_dyn();
    let t = BoundedTensor::new(lower, upper).unwrap();
    assert!(
        t.upper()[[1]] > 0.0 && t.upper()[[1]] < f32::MIN_POSITIVE,
        "denormal upper {} should be positive but below MIN_POSITIVE",
        t.upper()[[1]]
    );
}

#[test]
fn test_negative_zero() {
    // -0.0 is valid and equals 0.0 for comparison
    let lower = arr1(&[-0.0, -1.0]).into_dyn();
    let upper = arr1(&[0.0, 1.0]).into_dyn();
    let t = BoundedTensor::new(lower, upper).unwrap();
    // -0.0 == 0.0 in IEEE 754
    assert_eq!(t.lower()[[0]], 0.0);
    assert_eq!(t.lower()[[0]], -0.0);
}

#[test]
fn test_very_small_epsilon() {
    // Very small but non-denormal epsilon
    let values = arr1(&[0.0, 1.0, -1.0]).into_dyn();
    let epsilon = f32::EPSILON;
    let t = BoundedTensor::from_epsilon(values, epsilon)
        .expect("invariant: valid epsilon perturbation");
    assert!(
        (t.lower()[[0]] - (-f32::EPSILON)).abs() < 1e-10,
        "lower[0] should be ≈ -EPSILON, got {}",
        t.lower()[[0]]
    );
    assert!(
        (t.upper()[[0]] - f32::EPSILON).abs() < 1e-10,
        "upper[0] should be ≈ +EPSILON, got {}",
        t.upper()[[0]]
    );
}

#[test]
fn test_zero_width_bounds() {
    // Zero-width bounds (concrete values)
    let lower = arr1(&[1.0, 2.0, 3.0]).into_dyn();
    let upper = arr1(&[1.0, 2.0, 3.0]).into_dyn();
    let t = BoundedTensor::new(lower, upper).unwrap();
    assert_eq!(t.max_width(), 0.0);
}

#[test]
fn test_large_width_bounds() {
    // Wide bounds approaching but not reaching infinity
    let lower = arr1(&[-f32::MAX / 2.0]).into_dyn();
    let upper = arr1(&[f32::MAX / 2.0]).into_dyn();
    let t = BoundedTensor::new(lower, upper).unwrap();
    assert!(
        t.max_width().is_finite(),
        "half-MAX width {} should still be finite",
        t.max_width()
    );
}

// === Directed Rounding Tests ===

#[test]
fn test_round_for_soundness_widens_bounds() {
    // Directed rounding should widen bounds by 1 ULP
    let lower = arr1(&[1.0_f32, 0.1, -1.0]).into_dyn();
    let upper = arr1(&[1.0_f32, 0.1, -1.0]).into_dyn();
    let t = BoundedTensor::new(lower, upper).unwrap();

    let rounded = t.round_for_soundness();

    // Each lower should be < original, each upper should be > original
    for i in 0..3 {
        assert!(
            rounded.lower()[[i]] < t.lower()[[i]],
            "Lower bound should decrease at index {}: {} < {}",
            i,
            rounded.lower()[[i]],
            t.lower()[[i]]
        );
        assert!(
            rounded.upper()[[i]] > t.upper()[[i]],
            "Upper bound should increase at index {}: {} > {}",
            i,
            rounded.upper()[[i]],
            t.upper()[[i]]
        );
    }

    // Width should increase by exactly 2 ULPs (1 down + 1 up)
    assert!(
        rounded.max_width() > 0.0,
        "rounded width should be positive after ULP widening, got {}",
        rounded.max_width()
    );
}

#[test]
fn test_round_for_soundness_1ulp_difference() {
    // Test that bounds widen by exactly 1 ULP
    let val = 1.0_f32;
    let lower = arr1(&[val]).into_dyn();
    let upper = arr1(&[val]).into_dyn();
    let t = BoundedTensor::new(lower, upper).unwrap();

    let rounded = t.round_for_soundness();

    // next_down(1.0) and next_up(1.0) should differ by exactly 1 ULP each
    assert_eq!(rounded.lower()[[0]], next_down_f32(1.0));
    assert_eq!(rounded.upper()[[0]], next_up_f32(1.0));
    assert_eq!(rounded.upper()[[0]], 1.0 + f32::EPSILON);
}

#[test]
fn test_round_for_soundness_preserves_infinity() {
    // Infinity should stay infinity (no ULP beyond infinity)
    let lower = arr1(&[f32::NEG_INFINITY]).into_dyn();
    let upper = arr1(&[f32::INFINITY]).into_dyn();
    let t = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let rounded = t.round_for_soundness();

    assert_eq!(rounded.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(rounded.upper()[[0]], f32::INFINITY);
}

#[test]
fn test_round_for_soundness_inplace() {
    // Test in-place rounding
    let lower = arr1(&[1.0_f32]).into_dyn();
    let upper = arr1(&[2.0_f32]).into_dyn();
    let mut t = BoundedTensor::new(lower, upper).unwrap();

    t.round_for_soundness_inplace();

    assert!(
        t.lower()[[0]] < 1.0,
        "in-place rounding should widen lower below 1.0, got {}",
        t.lower()[[0]]
    );
    assert!(
        t.upper()[[0]] > 2.0,
        "in-place rounding should widen upper above 2.0, got {}",
        t.upper()[[0]]
    );
}

#[test]
fn test_set_lower_and_upper_updates_bounds() {
    let lower = arr1(&[0.0, 1.0]).into_dyn();
    let upper = arr1(&[2.0, 3.0]).into_dyn();
    let mut t = BoundedTensor::new(lower, upper).expect("valid bounds");

    t.set_lower(arr1(&[-1.0, 1.0]).into_dyn())
        .expect("valid lower");
    t.set_upper(arr1(&[2.0, 4.0]).into_dyn())
        .expect("valid upper");

    assert_eq!(t.lower()[[0]], -1.0);
    assert_eq!(t.upper()[[1]], 4.0);
}

#[test]
fn test_set_lower_updates_bounds_when_valid() {
    let lower = arr1(&[0.0, 1.0]).into_dyn();
    let upper = arr1(&[2.0, 3.0]).into_dyn();
    let mut t = BoundedTensor::new(lower, upper).expect("valid bounds");

    t.set_lower(arr1(&[-1.0, 1.5]).into_dyn())
        .expect("set_lower should accept finite, ordered bounds");

    assert_eq!(t.lower()[[0]], -1.0);
    assert_eq!(t.lower()[[1]], 1.5);
    assert_eq!(t.upper()[[0]], 2.0);
    assert_eq!(t.upper()[[1]], 3.0);
}

#[test]
fn test_set_upper_rejects_inversion() {
    let lower = arr1(&[0.0, 1.0]).into_dyn();
    let upper = arr1(&[2.0, 3.0]).into_dyn();
    let mut t = BoundedTensor::new(lower, upper).expect("valid bounds");

    let err = t
        .set_upper(arr1(&[-0.5, 2.5]).into_dyn())
        .expect_err("set_upper must reject lower > upper");

    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "set_upper inversion should return InvalidSpec, got {err:?}"
    );
}

#[test]
fn test_set_lower_rejects_non_finite() {
    let lower = arr1(&[0.0, 1.0]).into_dyn();
    let upper = arr1(&[2.0, 3.0]).into_dyn();
    let mut t = BoundedTensor::new(lower, upper).expect("valid bounds");

    let err = t
        .set_lower(arr1(&[0.0, f32::NAN]).into_dyn())
        .expect_err("set_lower must reject NaN");

    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "set_lower NaN should return NumericalInstability, got {err:?}"
    );
}

#[test]
fn test_mark_infeasible_helpers() {
    let mut t = BoundedTensor::new(
        arr2(&[[0.0, 1.0], [2.0, 3.0]]).into_dyn(),
        arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(),
    )
    .expect("valid 2x2 bounds");

    t.mark_infeasible_at(0, 1)
        .expect("axis-0 index 1 should exist");
    assert_eq!(t.lower()[[1, 0]], f32::INFINITY);
    assert_eq!(t.upper()[[1, 0]], f32::NEG_INFINITY);
    assert_eq!(t.lower()[[0, 0]], 0.0);
    assert_eq!(t.upper()[[0, 0]], 1.0);

    let err = t
        .mark_infeasible_at(1, 5)
        .expect_err("out-of-bounds index should error");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "mark_infeasible_at out-of-bounds should return InvalidSpec, got {err:?}"
    );

    t.mark_infeasible_all();
    assert!(
        t.lower().iter().all(|v| *v == f32::INFINITY),
        "mark_infeasible_all should set all lower bounds to +inf"
    );
    assert!(
        t.upper().iter().all(|v| *v == f32::NEG_INFINITY),
        "mark_infeasible_all should set all upper bounds to -inf"
    );
}

#[test]
fn test_try_get_returns_none_for_infeasible_element() {
    let mut t = BoundedTensor::new(
        arr2(&[[0.0, 1.0], [2.0, 3.0]]).into_dyn(),
        arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(),
    )
    .expect("valid 2x2 bounds");

    t.mark_infeasible_at(0, 1)
        .expect("axis-0 index 1 should exist");

    assert_eq!(
        t.try_get(&[1, 0])
            .expect("infeasible sentinel should not error"),
        None
    );
    assert_eq!(
        t.try_get(&[1, 1])
            .expect("infeasible sentinel should not error"),
        None
    );
    assert!(
        t.try_get(&[0, 0])
            .expect("feasible element should not error")
            .is_some(),
        "untouched element should remain feasible"
    );
}

#[test]
fn test_try_get_out_of_bounds_returns_invalid_spec() {
    let t = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid 1D bounds");

    let err = t
        .try_get(&[1])
        .expect_err("out-of-bounds scalar access should error");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("out of bounds")),
        "try_get out-of-bounds should return InvalidSpec, got {err:?}"
    );
}

#[test]
fn test_try_get_rank_mismatch_returns_invalid_spec() {
    let t = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid 1D bounds");

    let err = t.try_get(&[0, 0]).expect_err("rank mismatch should error");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("index rank")),
        "try_get rank mismatch should return InvalidSpec, got {err:?}"
    );
}

#[test]
#[should_panic(expected = "use try_get()")]
fn test_get_panics_with_try_get_hint_on_infeasible_element() {
    let mut t = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid 1D bounds");
    t.mark_infeasible_all();

    let _ = t.get(&[0]);
}

// === intersection_per_element additional tests (migrated from deprecated intersection(), #4253) ===

#[test]
fn test_intersection_per_element_touching_single_point() {
    // [0, 1] ∩ [1, 2] = [1, 1] (touching at boundary)
    let a = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let b = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();

    let (result, disjoint) = a
        .intersection_per_element(&b)
        .expect("invariant: no NaN, shapes match");
    assert_eq!(disjoint, 0);
    assert_eq!(result.lower()[[0]], 1.0);
    assert_eq!(result.upper()[[0]], 1.0);
}

#[test]
fn test_intersection_per_element_one_contains_other() {
    // [0, 10] ∩ [2, 5] = [2, 5]
    let a = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[10.0]).into_dyn()).unwrap();
    let b = BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[5.0]).into_dyn()).unwrap();

    let (result, disjoint) = a
        .intersection_per_element(&b)
        .expect("invariant: no NaN, shapes match");
    assert_eq!(disjoint, 0);
    assert_eq!(result.lower()[[0]], 2.0);
    assert_eq!(result.upper()[[0]], 5.0);
}

#[test]
fn test_intersection_per_element_symmetry() {
    // a ∩ b == b ∩ a
    let a = BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap();
    let b = BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[5.0, 6.0]).into_dyn()).unwrap();

    let (ab, ab_disjoint) = a
        .intersection_per_element(&b)
        .expect("invariant: no NaN, shapes match");
    let (ba, ba_disjoint) = b
        .intersection_per_element(&a)
        .expect("invariant: no NaN, shapes match");

    assert_eq!(ab_disjoint, ba_disjoint);
    assert_eq!(ab.lower()[[0]], ba.lower()[[0]]);
    assert_eq!(ab.upper()[[0]], ba.upper()[[0]]);
    assert_eq!(ab.lower()[[1]], ba.lower()[[1]]);
    assert_eq!(ab.upper()[[1]], ba.upper()[[1]]);
}

#[test]
fn test_intersection_per_element_infinite_bounds() {
    // [-inf, 5] ∩ [0, inf] = [0, 5]
    let a = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[5.0]).into_dyn(),
    )
    .expect("invariant: allow infinite bounds");
    let b = BoundedTensor::new_allow_infinite(
        arr1(&[0.0]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .expect("invariant: allow infinite bounds");

    let (result, disjoint) = a
        .intersection_per_element(&b)
        .expect("invariant: no NaN, shapes match");
    assert_eq!(disjoint, 0);
    assert_eq!(result.lower()[[0]], 0.0);
    assert_eq!(result.upper()[[0]], 5.0);
}

// =============================================================================
// Regression tests for #1809: runtime validation in release builds
// =============================================================================

/// concrete() must return Err on NaN in all builds (not just debug).
#[test]
fn test_concrete_rejects_nan() {
    assert!(
        BoundedTensor::concrete(arr1(&[f32::NAN]).into_dyn()).is_err(),
        "concrete() must reject NaN in release builds"
    );
}

/// concrete() must return Err on Inf in all builds.
#[test]
fn test_concrete_rejects_inf() {
    assert!(
        BoundedTensor::concrete(arr1(&[f32::INFINITY]).into_dyn()).is_err(),
        "concrete() must reject +inf in release builds"
    );
}

/// concrete() must return Err on -Inf in all builds.
#[test]
fn test_concrete_rejects_neg_inf() {
    assert!(
        BoundedTensor::concrete(arr1(&[f32::NEG_INFINITY]).into_dyn()).is_err(),
        "concrete() must reject -inf in release builds"
    );
}

/// from_epsilon() must return Err on NaN values in all builds.
#[test]
fn test_from_epsilon_rejects_nan_values() {
    assert!(
        BoundedTensor::from_epsilon(arr1(&[f32::NAN]).into_dyn(), 0.1).is_err(),
        "from_epsilon should reject NaN input values"
    );
}

/// from_epsilon() must return Err on Inf values in all builds.
#[test]
fn test_from_epsilon_rejects_inf_values() {
    assert!(
        BoundedTensor::from_epsilon(arr1(&[f32::INFINITY]).into_dyn(), 0.1).is_err(),
        "from_epsilon should reject +inf input values"
    );
}

/// from_epsilon() must return Err on negative epsilon.
#[test]
fn test_from_epsilon_rejects_negative_epsilon() {
    assert!(
        BoundedTensor::from_epsilon(arr1(&[1.0]).into_dyn(), -0.1).is_err(),
        "from_epsilon should reject negative epsilon"
    );
}

/// from_epsilon() must return Err on NaN epsilon.
#[test]
fn test_from_epsilon_rejects_nan_epsilon() {
    assert!(
        BoundedTensor::from_epsilon(arr1(&[1.0]).into_dyn(), f32::NAN).is_err(),
        "from_epsilon should reject NaN epsilon"
    );
}

/// from_epsilon() must return Err on infinite epsilon.
#[test]
fn test_from_epsilon_rejects_inf_epsilon() {
    assert!(
        BoundedTensor::from_epsilon(arr1(&[1.0]).into_dyn(), f32::INFINITY).is_err(),
        "from_epsilon should reject infinite epsilon"
    );
}

/// from_epsilon() must clamp overflow instead of producing Inf bounds.
#[test]
fn test_from_epsilon_clamps_overflow() {
    let t = BoundedTensor::from_epsilon(arr1(&[f32::MAX]).into_dyn(), 1.0)
        .expect("invariant: f32::MAX with epsilon=1 should clamp, not error");
    assert!(
        t.upper()[[0]].is_finite(),
        "upper bound should be clamped to f32::MAX, got {}",
        t.upper()[[0]]
    );
    assert_eq!(t.upper()[[0]], f32::MAX);

    let t2 = BoundedTensor::from_epsilon(arr1(&[f32::MIN]).into_dyn(), 1.0)
        .expect("invariant: f32::MIN with epsilon=1 should clamp, not error");
    assert!(
        t2.lower()[[0]].is_finite(),
        "lower bound should be clamped to f32::MIN, got {}",
        t2.lower()[[0]]
    );
    assert_eq!(t2.lower()[[0]], f32::MIN);
}

// ── intersection_per_element tests (#2935) ──────────────────────────

/// All elements overlap: intersection_per_element tightens every element.
#[test]
fn test_intersection_per_element_all_overlapping() {
    let a = BoundedTensor::new(
        arr1(&[0.0, 1.0, 2.0]).into_dyn(),
        arr1(&[3.0, 4.0, 5.0]).into_dyn(),
    )
    .expect("invariant: test construction");
    let b = BoundedTensor::new(
        arr1(&[1.0, 2.0, 3.0]).into_dyn(),
        arr1(&[2.0, 3.0, 4.0]).into_dyn(),
    )
    .expect("invariant: test construction");
    let (result, disjoint) = a
        .intersection_per_element(&b)
        .expect("invariant: no NaN, shapes match");
    assert_eq!(disjoint, 0);
    assert_eq!(result.lower()[[0]], 1.0);
    assert_eq!(result.upper()[[0]], 2.0);
    assert_eq!(result.lower()[[1]], 2.0);
    assert_eq!(result.upper()[[1]], 3.0);
    assert_eq!(result.lower()[[2]], 3.0);
    assert_eq!(result.upper()[[2]], 4.0);
}

/// 10 elements, 1 disjoint: per-element keeps 9 tightened + 1 union fallback.
#[test]
fn test_intersection_per_element_one_disjoint_of_many() {
    // 10 overlapping elements
    let lower_a = vec![0.0_f32; 10];
    let upper_a = vec![2.0_f32; 10];
    let mut lower_b = vec![1.0_f32; 10];
    let mut upper_b = vec![3.0_f32; 10];
    // Make element 5 disjoint: a=[0,2], b=[4,6]
    lower_b[5] = 4.0;
    upper_b[5] = 6.0;

    let a = BoundedTensor::new(arr1(&lower_a).into_dyn(), arr1(&upper_a).into_dyn())
        .expect("invariant: test construction");
    let b = BoundedTensor::new(arr1(&lower_b).into_dyn(), arr1(&upper_b).into_dyn())
        .expect("invariant: test construction");

    // Per-element keeps 9 good + 1 union
    let (result, disjoint) = a
        .intersection_per_element(&b)
        .expect("invariant: no NaN, shapes match");
    assert_eq!(disjoint, 1);

    // Overlapping elements are tightened (max lower, min upper)
    for i in 0..10 {
        if i == 5 {
            // Disjoint element: union fallback = min(0,4)=0, max(2,6)=6
            assert_eq!(result.lower()[[i]], 0.0);
            assert_eq!(result.upper()[[i]], 6.0);
        } else {
            // Overlapping: intersection = max(0,1)=1, min(2,3)=2
            assert_eq!(result.lower()[[i]], 1.0);
            assert_eq!(result.upper()[[i]], 2.0);
        }
    }
}

/// All elements disjoint: everything gets union fallback.
#[test]
fn test_intersection_per_element_all_disjoint() {
    let a = BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
        .expect("invariant: valid bounds (lower <= upper)");
    let b = BoundedTensor::new(arr1(&[2.0, 3.0]).into_dyn(), arr1(&[4.0, 5.0]).into_dyn())
        .expect("invariant: valid bounds (lower <= upper)");
    let (result, disjoint) = a
        .intersection_per_element(&b)
        .expect("invariant: no NaN, shapes match");
    assert_eq!(disjoint, 2);
    // Union: min(0,2)=0, max(1,4)=4
    assert_eq!(result.lower()[[0]], 0.0);
    assert_eq!(result.upper()[[0]], 4.0);
    // Union: min(0,3)=0, max(1,5)=5
    assert_eq!(result.lower()[[1]], 0.0);
    assert_eq!(result.upper()[[1]], 5.0);
}

/// NaN in any endpoint returns None.
#[test]
fn test_intersection_per_element_nan_returns_none() {
    // Use new_unchecked to bypass NaN validation in constructor (#2935).
    // This tests that intersection_per_element itself rejects NaN endpoints.
    let a = BoundedTensor::new_unchecked(
        arr1(&[f32::NAN, 1.0]).into_dyn(),
        arr1(&[2.0, 3.0]).into_dyn(),
    )
    .expect("new_unchecked: shape check only");
    let b = BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[4.0, 4.0]).into_dyn())
        .expect("invariant: valid bounds (lower <= upper)");
    assert!(
        a.intersection_per_element(&b).is_none(),
        "NaN in lower should return None"
    );
}

/// Shape mismatch returns None.
#[test]
fn test_intersection_per_element_shape_mismatch_returns_none() {
    let a = BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn())
        .expect("invariant: valid bounds (lower <= upper)");
    let b = BoundedTensor::new(arr1(&[0.5]).into_dyn(), arr1(&[1.5]).into_dyn())
        .expect("invariant: valid bounds (lower <= upper)");
    assert!(
        a.intersection_per_element(&b).is_none(),
        "shape mismatch should return None"
    );
}

// =============================================================================
// Regression tests for #2929: repair of large/non-finite endpoints must never
// tighten a bound (an unbounded endpoint proves nothing, so the repaired
// interval has to contain the true range).
// =============================================================================

/// Regression test for #2929: add() with a finite lower sum and NaN upper.
///
/// Scenario: A.lower = [1.5e10], A.upper = [inf], B.lower = [1.5e10], B.upper = [-inf].
/// add() computes: lower = 3e10 (finite, preserved), upper = inf + (-inf) = NaN.
/// The NaN upper carries no bound information, so the only sound repair is +inf;
/// any finite replacement would invert the bounds or exclude the true range.
#[test]
fn test_add_repair_nan_upper_2929() {
    let a = BoundedTensor::from_parts_unchecked(
        arr1(&[1.5e10]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    );
    let b = BoundedTensor::from_parts_unchecked(
        arr1(&[1.5e10]).into_dyn(),
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
    );

    let result = a.add(&b).expect("add should not fail on matching shapes");
    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];

    assert!(!lower.is_nan(), "lower must not be NaN after repair");
    assert!(!upper.is_nan(), "upper must not be NaN after repair");
    assert!(
        lower <= upper,
        "bounds must not be inverted after add: lower={lower}, upper={upper}"
    );
    assert_eq!(lower, 3e10, "finite lower sum must be preserved");
    assert_eq!(
        upper,
        f32::INFINITY,
        "NaN upper must be repaired to +inf, never a finite value"
    );
}

/// Regression test for #2929: scale() with an upper product overflowing to inf.
///
/// tensor = [lower=[2.0], upper=[f32::MAX]] (valid input accepted by `new`).
/// scale(1e10): lower = 2e10 (finite), upper = f32::MAX * 1e10 = inf.
/// The true range is [2e10, 3.4e48], so the overflowed upper must stay +inf:
/// clamping it to any finite value would exclude essentially the whole range.
#[test]
fn test_scale_overflow_keeps_unbounded_upper_2929() {
    let t = BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[f32::MAX]).into_dyn())
        .expect("invariant: valid ascending bounds");

    let result = t.scale(1e10);
    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];

    assert!(
        lower <= upper,
        "bounds must not be inverted after scale: lower={lower}, upper={upper}"
    );
    // Containment of the true range [2e10, f32::MAX * 1e10]:
    assert_eq!(lower, 2e10, "finite lower product must be preserved");
    assert_eq!(
        upper,
        f32::INFINITY,
        "overflowed upper must stay +inf, never a finite value"
    );
}

/// Regression test for #2929: shift() with an infinite upper endpoint.
///
/// tensor = [lower=[1.5e10], upper=[inf]] (valid via new_allow_infinite).
/// shift(1.5e10): lower = 3e10 (finite), upper = inf + 1.5e10 = inf.
/// The true range is [3e10, inf); the upper endpoint must stay +inf.
#[test]
fn test_shift_keeps_unbounded_upper_2929() {
    let t = BoundedTensor::new_allow_infinite(
        arr1(&[1.5e10]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .expect("invariant: valid bounds with infinite upper");

    let result = t.shift(1.5e10);
    let lower = result.lower()[[0]];
    let upper = result.upper()[[0]];

    assert!(
        lower <= upper,
        "bounds must not be inverted after shift: lower={lower}, upper={upper}"
    );
    // Containment of the true range [3e10, inf):
    assert_eq!(lower, 3e10, "finite lower sum must be preserved");
    assert_eq!(
        upper,
        f32::INFINITY,
        "infinite upper must stay +inf, never a finite value"
    );
}

/// The enforce_bound_ordering fix must not affect already-valid bounds.
#[test]
fn test_add_valid_bounds_unaffected_by_ordering_fix_2929() {
    let a = BoundedTensor::new(arr1(&[1.0, -5.0]).into_dyn(), arr1(&[3.0, -2.0]).into_dyn())
        .expect("invariant: valid ascending bounds");
    let b = BoundedTensor::new(arr1(&[2.0, 1.0]).into_dyn(), arr1(&[4.0, 3.0]).into_dyn())
        .expect("invariant: valid ascending bounds");

    let result = a.add(&b).expect("invariant: matching shapes");
    // 1+2=3, 3+4=7, -5+1=-4, -2+3=1
    assert_eq!(result.lower()[[0]], 3.0);
    assert_eq!(result.upper()[[0]], 7.0);
    assert_eq!(result.lower()[[1]], -4.0);
    assert_eq!(result.upper()[[1]], 1.0);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "from_parts_unchecked: lower contains NaN")]
fn test_from_parts_unchecked_rejects_nan_in_debug() {
    let _ =
        BoundedTensor::from_parts_unchecked(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn());
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "from_parts_unchecked: found finite lower > upper")]
fn test_from_parts_unchecked_rejects_finite_inversion_in_debug() {
    let _ = BoundedTensor::from_parts_unchecked(arr1(&[2.0]).into_dyn(), arr1(&[1.0]).into_dyn());
}

// =============================================================================
// Tests for new_allow_infinite (#2944)
// =============================================================================

/// new_allow_infinite accepts [-inf, 0.0].
#[test]
fn test_new_allow_infinite_neg_inf_to_zero() {
    let t = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
    )
    .expect("invariant: [-inf, 0.0] is valid");
    assert_eq!(t.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(t.upper()[[0]], 0.0);
}

/// new_allow_infinite accepts [0.0, +inf].
#[test]
fn test_new_allow_infinite_zero_to_inf() {
    let t = BoundedTensor::new_allow_infinite(
        arr1(&[0.0]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .expect("invariant: [0.0, +inf] is valid");
    assert_eq!(t.lower()[[0]], 0.0);
    assert_eq!(t.upper()[[0]], f32::INFINITY);
}

/// new_allow_infinite accepts [-inf, +inf].
#[test]
fn test_new_allow_infinite_full_range() {
    let t = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .expect("invariant: [-inf, +inf] is valid");
    assert_eq!(t.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(t.upper()[[0]], f32::INFINITY);
}

/// new_allow_infinite rejects [+inf, -inf] (inverted infinite).
#[test]
fn test_new_allow_infinite_rejects_inverted_infinite() {
    let err = BoundedTensor::new_allow_infinite(
        arr1(&[f32::INFINITY]).into_dyn(),
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
    )
    .expect_err("should reject [+inf, -inf]");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec for inverted bounds, got: {:?}",
        err
    );
}

/// new_allow_infinite rejects NaN in lower bounds.
#[test]
fn test_new_allow_infinite_rejects_nan_lower() {
    let err =
        BoundedTensor::new_allow_infinite(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect_err("should reject NaN lower");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability for NaN, got: {:?}",
        err
    );
}

/// new_allow_infinite rejects NaN in upper bounds.
#[test]
fn test_new_allow_infinite_rejects_nan_upper() {
    let err =
        BoundedTensor::new_allow_infinite(arr1(&[1.0]).into_dyn(), arr1(&[f32::NAN]).into_dyn())
            .expect_err("should reject NaN upper");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability for NaN, got: {:?}",
        err
    );
}

/// new_allow_infinite rejects NaN in both bounds.
#[test]
fn test_new_allow_infinite_rejects_both_nan() {
    let err = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NAN]).into_dyn(),
        arr1(&[f32::NAN]).into_dyn(),
    )
    .expect_err("should reject NaN/NaN");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability for NaN, got: {:?}",
        err
    );
}

/// new_allow_infinite rejects finite inverted bounds.
#[test]
fn test_new_allow_infinite_rejects_inverted_finite() {
    let err = BoundedTensor::new_allow_infinite(arr1(&[1.0]).into_dyn(), arr1(&[0.0]).into_dyn())
        .expect_err("should reject [1.0, 0.0]");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec for inverted bounds, got: {:?}",
        err
    );
}

/// new_allow_infinite rejects [-inf, NaN] (NaN upper with infinite lower).
#[test]
fn test_new_allow_infinite_rejects_neg_inf_nan() {
    let err = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::NAN]).into_dyn(),
    )
    .expect_err("should reject [-inf, NaN]");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability for NaN, got: {:?}",
        err
    );
}

/// new_allow_infinite works with mixed finite/infinite multi-element bounds.
#[test]
fn test_new_allow_infinite_multi_element() {
    let t = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY, -1.0, 0.0]).into_dyn(),
        arr1(&[0.0, f32::INFINITY, f32::INFINITY]).into_dyn(),
    )
    .expect("invariant: mixed finite/infinite bounds are valid");
    assert_eq!(t.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(t.upper()[[1]], f32::INFINITY);
    assert_eq!(t.lower()[[1]], -1.0);
    assert_eq!(t.upper()[[0]], 0.0);
}

/// new_allow_infinite rejects shape mismatch.
#[test]
fn test_new_allow_infinite_rejects_shape_mismatch() {
    let err =
        BoundedTensor::new_allow_infinite(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[1.0]).into_dyn())
            .expect_err("should reject shape mismatch");
    assert!(
        matches!(err, NyError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got: {:?}",
        err
    );
}

// =============================================================================
// Tests for round_for_soundness_n_ulps (#2944)
// =============================================================================

/// n=0: no widening occurs.
#[test]
fn test_round_for_soundness_n_ulps_zero_no_widening() {
    let t = BoundedTensor::new(
        arr1(&[1.0_f32, -1.0]).into_dyn(),
        arr1(&[2.0_f32, 0.0]).into_dyn(),
    )
    .expect("invariant: valid test bounds");

    let rounded = t.round_for_soundness_n_ulps(0);
    assert_eq!(rounded.lower()[[0]], t.lower()[[0]]);
    assert_eq!(rounded.upper()[[0]], t.upper()[[0]]);
    assert_eq!(rounded.lower()[[1]], t.lower()[[1]]);
    assert_eq!(rounded.upper()[[1]], t.upper()[[1]]);
}

/// n=1: matches the 1-ULP round_for_soundness() output exactly.
#[test]
fn test_round_for_soundness_n_ulps_one_matches_1ulp() {
    let t = BoundedTensor::new(
        arr1(&[1.0_f32, 0.1, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 0.1, -1.0]).into_dyn(),
    )
    .expect("invariant: valid test bounds");

    let n1 = t.round_for_soundness_n_ulps(1);
    let ref1 = t.round_for_soundness();

    for i in 0..3 {
        assert_eq!(
            n1.lower()[[i]],
            ref1.lower()[[i]],
            "n_ulps(1) lower should match round_for_soundness at index {}",
            i
        );
        assert_eq!(
            n1.upper()[[i]],
            ref1.upper()[[i]],
            "n_ulps(1) upper should match round_for_soundness at index {}",
            i
        );
    }
}

/// n=2: strictly wider than n=1.
#[test]
fn test_round_for_soundness_n_ulps_two_wider_than_one() {
    let t = BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("invariant: valid test bounds");

    let r1 = t.round_for_soundness_n_ulps(1);
    let r2 = t.round_for_soundness_n_ulps(2);

    assert!(
        r2.lower()[[0]] < r1.lower()[[0]],
        "2-ULP lower ({}) should be below 1-ULP lower ({})",
        r2.lower()[[0]],
        r1.lower()[[0]]
    );
    assert!(
        r2.upper()[[0]] > r1.upper()[[0]],
        "2-ULP upper ({}) should be above 1-ULP upper ({})",
        r2.upper()[[0]],
        r1.upper()[[0]]
    );
}

/// Near f32::MAX: must saturate to +inf, not overflow to NaN.
#[test]
fn test_round_for_soundness_n_ulps_near_max_saturates() {
    let t = BoundedTensor::new(arr1(&[f32::MAX]).into_dyn(), arr1(&[f32::MAX]).into_dyn())
        .expect("invariant: valid test bounds");

    let rounded = t.round_for_soundness_n_ulps(10);
    assert_eq!(
        rounded.upper()[[0]],
        f32::INFINITY,
        "ULP shift past f32::MAX should saturate to +inf"
    );
    assert!(
        rounded.lower()[[0]] < f32::MAX,
        "lower should shift down from f32::MAX"
    );
    assert!(
        rounded.lower()[[0]].is_finite(),
        "lower should remain finite"
    );
}

/// Near -f32::MAX: lower must saturate to -inf.
#[test]
fn test_round_for_soundness_n_ulps_near_neg_max_saturates() {
    let t = BoundedTensor::new(arr1(&[-f32::MAX]).into_dyn(), arr1(&[-f32::MAX]).into_dyn())
        .expect("invariant: valid test bounds");

    let rounded = t.round_for_soundness_n_ulps(10);
    assert_eq!(
        rounded.lower()[[0]],
        f32::NEG_INFINITY,
        "ULP shift past -f32::MAX should saturate to -inf"
    );
    assert!(
        rounded.upper()[[0]] > -f32::MAX,
        "upper should shift up from -f32::MAX"
    );
    assert!(
        rounded.upper()[[0]].is_finite(),
        "upper should remain finite"
    );
}

/// Subnormal values: ULPs near the smallest representable positive.
#[test]
fn test_round_for_soundness_n_ulps_subnormal() {
    let tiny = f32::MIN_POSITIVE / 2.0; // Subnormal
    assert!(
        tiny > 0.0 && tiny < f32::MIN_POSITIVE,
        "precondition: should be subnormal"
    );

    let t = BoundedTensor::new(arr1(&[tiny]).into_dyn(), arr1(&[tiny]).into_dyn())
        .expect("invariant: valid test bounds");

    let rounded = t.round_for_soundness_n_ulps(2);
    assert!(
        rounded.lower()[[0]] < tiny,
        "subnormal lower should shift down"
    );
    assert!(
        rounded.upper()[[0]] > tiny,
        "subnormal upper should shift up"
    );
}

/// Bounds spanning zero: ULP size changes at the zero boundary.
#[test]
fn test_round_for_soundness_n_ulps_zero_crossing() {
    let t = BoundedTensor::new(
        arr1(&[-f32::MIN_POSITIVE]).into_dyn(),
        arr1(&[f32::MIN_POSITIVE]).into_dyn(),
    )
    .expect("invariant: valid test bounds");

    let rounded = t.round_for_soundness_n_ulps(2);
    assert!(
        rounded.lower()[[0]] < -f32::MIN_POSITIVE,
        "lower should shift further negative"
    );
    assert!(
        rounded.upper()[[0]] > f32::MIN_POSITIVE,
        "upper should shift further positive"
    );
    assert!(
        rounded.lower()[[0]].is_finite(),
        "subnormal lower should remain finite after n-ULP rounding, got {}",
        rounded.lower()[[0]]
    );
    assert!(
        rounded.upper()[[0]].is_finite(),
        "subnormal upper should remain finite after n-ULP rounding, got {}",
        rounded.upper()[[0]]
    );
}

/// Infinity preserved for any n.
#[test]
fn test_round_for_soundness_n_ulps_preserves_infinity() {
    let t = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .expect("invariant: valid infinite test bounds");

    let rounded = t.round_for_soundness_n_ulps(100);
    assert_eq!(rounded.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(rounded.upper()[[0]], f32::INFINITY);
}

/// In-place variant matches the returning variant.
#[test]
fn test_round_for_soundness_n_ulps_inplace_matches() {
    let lower = arr1(&[1.0_f32, -2.0]).into_dyn();
    let upper = arr1(&[3.0_f32, 5.0]).into_dyn();
    let t = BoundedTensor::new(lower.clone(), upper.clone()).expect("invariant: valid test bounds");
    let mut t_inplace = BoundedTensor::new(lower, upper).expect("invariant: valid test bounds");

    let rounded = t.round_for_soundness_n_ulps(3);
    t_inplace.round_for_soundness_n_ulps_inplace(3);

    for i in 0..2 {
        assert_eq!(
            rounded.lower()[[i]],
            t_inplace.lower()[[i]],
            "inplace lower should match at index {}",
            i
        );
        assert_eq!(
            rounded.upper()[[i]],
            t_inplace.upper()[[i]],
            "inplace upper should match at index {}",
            i
        );
    }
}

/// center() must return 0.0 for symmetric infinite bounds [-inf, +inf],
/// not NaN. Such bounds arise from new_conservative() and new_repaired(Widen).
/// IEEE 754: (-inf + inf) / 2.0 = NaN. (#3291 F3, #3423)
#[test]
fn test_center_infinite_bounds_returns_zero() {
    let lower = arr1(&[f32::NEG_INFINITY, -1.0, f32::NEG_INFINITY]).into_dyn();
    let upper = arr1(&[f32::INFINITY, 1.0, f32::INFINITY]).into_dyn();
    let t = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let center = t.center();

    // [-inf, +inf] → 0.0, not NaN
    assert_eq!(center[[0]], 0.0, "center of [-inf, +inf] should be 0.0");
    // [-1.0, 1.0] → 0.0 (normal case)
    assert_eq!(center[[1]], 0.0, "center of [-1.0, 1.0] should be 0.0");
    // [-inf, +inf] → 0.0, not NaN
    assert_eq!(center[[2]], 0.0, "center of [-inf, +inf] should be 0.0");

    // Verify no NaN in output
    assert!(
        !center.iter().any(|v| v.is_nan()),
        "center() must not produce NaN"
    );
}

/// center() with one-sided infinite bounds (not symmetric) should still produce
/// correct non-NaN results. [-inf, 10.0] → -inf (correct), [5.0, +inf] → +inf (correct).
#[test]
fn test_center_one_sided_infinite_bounds() {
    let lower = arr1(&[f32::NEG_INFINITY, 5.0]).into_dyn();
    let upper = arr1(&[10.0, f32::INFINITY]).into_dyn();
    let t = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let center = t.center();

    // [-inf, 10.0] → (-inf + 10) / 2 = -inf (correct)
    assert!(
        center[[0]].is_infinite() && center[[0]] < 0.0,
        "center of [-inf, 10.0] should be -inf, got {}",
        center[[0]]
    );
    // [5.0, +inf] → (5.0 + inf) / 2 = +inf (correct)
    assert!(
        center[[1]].is_infinite() && center[[1]] > 0.0,
        "center of [5.0, +inf] should be +inf, got {}",
        center[[1]]
    );
}

/// center() on conservative bounds from new_repaired(Widen) should work (#3423).
#[test]
fn test_center_after_widen_repair() {
    use crate::RepairStrategy;
    let lower = arr1(&[1.0, f32::NAN, 3.0]).into_dyn();
    let upper = arr1(&[2.0, f32::NAN, 4.0]).into_dyn();
    let t = BoundedTensor::new_repaired(lower, upper, RepairStrategy::Widen)
        .expect("new_repaired(Widen) should succeed");

    let center = t.center();
    // Repaired NaN element is [-inf, +inf] → center should be 0.0
    assert_eq!(center[[1]], 0.0, "Repaired center should be 0.0, not NaN");
    // Normal elements
    assert!(
        (center[[0]] - 1.5).abs() < 1e-6,
        "center[0] should be ≈ 1.5, got {}",
        center[[0]]
    );
    assert!(
        (center[[2]] - 3.5).abs() < 1e-6,
        "center[2] should be ≈ 3.5, got {}",
        center[[2]]
    );
}

#[test]
fn test_repair_inverted_bounds_swap_repairs_only_inverted_elements() {
    let mut lower = vec![0.0, 5.0, -2.0];
    let mut upper = vec![1.0, 1.0, -2.0];

    let repaired = repair_inverted_bounds(&mut lower, &mut upper, InversionRepair::Swap);

    assert_eq!(repaired, 1, "only the inverted element should be repaired");
    assert_eq!(lower, vec![0.0, 1.0, -2.0]);
    assert_eq!(upper, vec![1.0, 5.0, -2.0]);
}

#[test]
fn test_repair_inverted_bounds_swap_leaves_nan_for_caller_sanitization() {
    let mut lower = vec![f32::NAN, 5.0];
    let mut upper = vec![1.0, 1.0];

    let repaired = repair_inverted_bounds(&mut lower, &mut upper, InversionRepair::Swap);

    assert_eq!(repaired, 1, "only the finite inversion should be repaired");
    assert!(lower[0].is_nan(), "swap strategy must not sanitize NaN");
    assert_eq!(upper[0], 1.0, "non-NaN endpoint remains unchanged");
    assert_eq!(lower[1], 1.0, "finite inversion is still swapped");
    assert_eq!(upper[1], 5.0, "finite inversion is still swapped");
}

#[test]
fn test_repair_inverted_bounds_widen_to_inf_repairs_nan_and_inversions() {
    let mut lower = vec![0.0, 5.0, f32::NAN, 2.0];
    let mut upper = vec![1.0, 1.0, 4.0, f32::NAN];

    let repaired = repair_inverted_bounds(&mut lower, &mut upper, InversionRepair::WidenToInf);

    assert_eq!(
        repaired, 3,
        "two NaN elements and one inversion should be repaired"
    );
    assert_eq!(lower[0], 0.0, "valid bounds must remain unchanged");
    assert_eq!(upper[0], 1.0, "valid bounds must remain unchanged");
    assert_eq!(lower[1], f32::NEG_INFINITY);
    assert_eq!(upper[1], f32::INFINITY);
    assert_eq!(lower[2], f32::NEG_INFINITY);
    assert_eq!(upper[2], f32::INFINITY);
    assert_eq!(lower[3], f32::NEG_INFINITY);
    assert_eq!(upper[3], f32::INFINITY);
}

#[test]
fn test_repair_inverted_bounds_widen_to_fallback_repairs_infinite_inversion() {
    let mut lower = vec![f32::INFINITY, -1.0];
    let mut upper = vec![f32::NEG_INFINITY, 2.0];

    let repaired = repair_inverted_bounds(
        &mut lower,
        &mut upper,
        InversionRepair::WidenToFallback(7.0),
    );

    assert_eq!(
        repaired, 1,
        "only the infinite inverted element should be repaired"
    );
    assert_eq!(lower, vec![-7.0, -1.0]);
    assert_eq!(upper, vec![7.0, 2.0]);
}

#[test]
fn test_repair_inverted_bounds_nd_repairs_mixed_tensor() {
    let mut lower = arr1(&[0.0, 5.0, f32::NAN]).into_dyn();
    let mut upper = arr1(&[1.0, 1.0, 4.0]).into_dyn();

    let repaired = repair_inverted_bounds_nd(
        &mut lower,
        &mut upper,
        InversionRepair::WidenToFallback(9.0),
    );

    assert_eq!(repaired, 2, "one inversion and one NaN should be repaired");
    assert_eq!(lower[[0]], 0.0, "valid element must remain unchanged");
    assert_eq!(upper[[0]], 1.0, "valid element must remain unchanged");
    assert_eq!(lower[[1]], -9.0);
    assert_eq!(upper[[1]], 9.0);
    assert_eq!(lower[[2]], -9.0);
    assert_eq!(upper[[2]], 9.0);
}

// =============================================================================
// new_repaired() constructor tests (Part of #3423)
// =============================================================================

#[test]
fn test_new_repaired_conservative_nan() {
    use crate::RepairStrategy;

    let lower = arr1(&[1.0, f32::NAN, 3.0]).into_dyn();
    let upper = arr1(&[2.0, f32::NAN, 4.0]).into_dyn();
    let t = BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
        .expect("Conservative should repair NaN");
    // NaN carries no bound information: repair to the unbounded direction.
    assert_eq!(t.lower()[[1]], f32::NEG_INFINITY);
    assert_eq!(t.upper()[[1]], f32::INFINITY);
    // Finite values preserved
    assert_eq!(t.lower()[[0]], 1.0);
    assert_eq!(t.upper()[[2]], 4.0);
}

#[test]
fn test_new_repaired_conservative_preserves_inf() {
    use crate::RepairStrategy;

    let lower = arr1(&[f32::NEG_INFINITY, 1.0]).into_dyn();
    let upper = arr1(&[2.0, f32::INFINITY]).into_dyn();
    let t = BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
        .expect("Conservative should accept Inf");
    // ±Inf endpoints must be preserved: clamping them to any finite value
    // would tighten an interval that was never proven.
    assert_eq!(t.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(t.upper()[[1]], f32::INFINITY);
    // Finite values preserved
    assert_eq!(t.upper()[[0]], 2.0);
    assert_eq!(t.lower()[[1]], 1.0);
}

#[test]
fn test_new_repaired_conservative_fixes_inverted() {
    use crate::RepairStrategy;

    // After NaN repair: lower=[5.0, -inf], upper=[1.0, +inf]
    // Then fix_inverted swaps element 0: lower=[1.0, -inf], upper=[5.0, +inf]
    let lower = arr1(&[5.0, f32::NAN]).into_dyn();
    let upper = arr1(&[1.0, f32::NAN]).into_dyn();
    let t = BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
        .expect("Conservative should fix inverted bounds");
    // Verify specific values, not just ordering
    assert_eq!(
        t.lower()[[0]],
        1.0,
        "Inverted lower should become 1.0 (was upper)"
    );
    assert_eq!(
        t.upper()[[0]],
        5.0,
        "Inverted upper should become 5.0 (was lower)"
    );
    // Verify NaN element was repaired to the unbounded interval
    assert_eq!(
        t.lower()[[1]],
        f32::NEG_INFINITY,
        "NaN lower should be repaired to -inf"
    );
    assert_eq!(
        t.upper()[[1]],
        f32::INFINITY,
        "NaN upper should be repaired to +inf"
    );
    assert!(
        t.lower()[[1]] <= t.upper()[[1]],
        "Repaired NaN element should have lower <= upper"
    );
}

#[test]
fn test_new_repaired_widen_nan() {
    use crate::RepairStrategy;

    let lower = arr1(&[1.0, f32::NAN]).into_dyn();
    let upper = arr1(&[2.0, f32::NAN]).into_dyn();
    let t = BoundedTensor::new_repaired(lower, upper, RepairStrategy::Widen)
        .expect("Widen should repair NaN");
    assert_eq!(t.lower()[[1]], f32::NEG_INFINITY);
    assert_eq!(t.upper()[[1]], f32::INFINITY);
    // Finite values preserved
    assert_eq!(t.lower()[[0]], 1.0);
}

#[test]
fn test_new_repaired_widen_preserves_inf() {
    use crate::RepairStrategy;

    let lower = arr1(&[f32::NEG_INFINITY]).into_dyn();
    let upper = arr1(&[f32::INFINITY]).into_dyn();
    let t = BoundedTensor::new_repaired(lower, upper, RepairStrategy::Widen)
        .expect("Widen should preserve Inf");
    assert_eq!(t.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(t.upper()[[0]], f32::INFINITY);
}

#[test]
fn test_new_repaired_strict_rejects_nan() {
    use crate::RepairStrategy;

    let lower = arr1(&[1.0, f32::NAN]).into_dyn();
    let upper = arr1(&[2.0, 3.0]).into_dyn();
    let result = BoundedTensor::new_repaired(lower, upper, RepairStrategy::Strict);
    assert!(result.is_err(), "Strict should reject NaN");
}

#[test]
fn test_new_repaired_strict_rejects_inf() {
    use crate::RepairStrategy;

    let lower = arr1(&[1.0]).into_dyn();
    let upper = arr1(&[f32::INFINITY]).into_dyn();
    let result = BoundedTensor::new_repaired(lower, upper, RepairStrategy::Strict);
    assert!(result.is_err(), "Strict should reject Inf");
}

#[test]
fn test_new_repaired_shape_mismatch() {
    use crate::RepairStrategy;

    let lower = arr1(&[1.0, 2.0]).into_dyn();
    let upper = arr1(&[3.0]).into_dyn();
    let result = BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative);
    assert!(result.is_err(), "Shape mismatch should error");
}

#[test]
fn test_new_repaired_with_poll_matches_conservative_semantics() {
    use crate::RepairStrategy;

    let lower = arr1(&[5.0, f32::NAN, f32::NEG_INFINITY, 1.0]).into_dyn();
    let upper = arr1(&[1.0, f32::NAN, 2.0, f32::INFINITY]).into_dyn();
    let expected =
        BoundedTensor::new_repaired(lower.clone(), upper.clone(), RepairStrategy::Conservative)
            .expect("ordinary conservative repair");
    let mut polls = 0usize;
    let actual =
        BoundedTensor::new_repaired_with_poll(lower, upper, RepairStrategy::Conservative, || {
            polls += 1;
            Ok(())
        })
        .expect("polling conservative repair");

    assert_eq!(actual.lower(), expected.lower());
    assert_eq!(actual.upper(), expected.upper());
    assert!(
        polls >= 5,
        "entry, three semantic passes, and publication must all poll"
    );
}

#[test]
fn test_new_repaired_with_poll_matches_strict_rejections() {
    use crate::RepairStrategy;

    let cases = [
        (arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn()),
        (arr1(&[0.0]).into_dyn(), arr1(&[f32::INFINITY]).into_dyn()),
        (arr1(&[2.0]).into_dyn(), arr1(&[1.0]).into_dyn()),
        (
            arr1(&[2.0, 0.0]).into_dyn(),
            arr1(&[1.0, f32::INFINITY]).into_dyn(),
        ),
    ];
    for (lower, upper) in cases {
        let expected =
            BoundedTensor::new_repaired(lower.clone(), upper.clone(), RepairStrategy::Strict)
                .expect_err("ordinary Strict must reject");
        let actual =
            BoundedTensor::new_repaired_with_poll(lower, upper, RepairStrategy::Strict, || Ok(()))
                .expect_err("polling Strict must reject");
        assert_eq!(
            std::mem::discriminant(&actual),
            std::mem::discriminant(&expected),
            "polling Strict must preserve the error category"
        );
    }
}

#[test]
fn test_new_repaired_with_poll_checks_every_bounded_chunk() {
    use crate::RepairStrategy;

    let lower = ndarray::ArrayD::zeros(IxDyn(&[8_193]));
    let upper = ndarray::ArrayD::ones(IxDyn(&[8_193]));
    let mut polls = 0usize;
    BoundedTensor::new_repaired_with_poll(lower, upper, RepairStrategy::Conservative, || {
        polls += 1;
        Ok(())
    })
    .expect("polling repair");
    assert_eq!(
        polls, 11,
        "expected entry, three offsets in each semantic pass, and publication"
    );
}

#[test]
fn test_new_repaired_with_poll_returns_final_poll_error_before_publication() {
    use crate::RepairStrategy;

    let mut polls = 0usize;
    let error = BoundedTensor::new_repaired_with_poll(
        arr1(&[0.0, 1.0]).into_dyn(),
        arr1(&[2.0, 3.0]).into_dyn(),
        RepairStrategy::Conservative,
        || {
            polls += 1;
            if polls == 5 {
                Err(NyError::DeadlineExceeded(
                    "injected final publication poll".to_string(),
                ))
            } else {
                Ok(())
            }
        },
    )
    .expect_err("final poll error must prevent publication");
    assert!(matches!(error, NyError::DeadlineExceeded(_)));
    assert_eq!(polls, 5);
}

#[test]
fn test_pollable_sound_rounding_matches_and_polls_8193_elements() {
    let lower = ndarray::ArrayD::from_shape_fn(IxDyn(&[8_193]), |index| {
        -2.0_f32 + index[0] as f32 * 0.0001
    });
    let upper =
        ndarray::ArrayD::from_shape_fn(IxDyn(&[8_193]), |index| 3.0_f32 + index[0] as f32 * 0.0001);

    let mut expected_one =
        BoundedTensor::new(lower.clone(), upper.clone()).expect("ordinary input");
    expected_one.round_for_soundness_inplace();
    let mut actual_one = BoundedTensor::new(lower.clone(), upper.clone()).expect("pollable input");
    let mut one_polls = 0usize;
    actual_one
        .round_for_soundness_inplace_with_poll(|| {
            one_polls += 1;
            Ok(())
        })
        .expect("pollable one-ULP widening");
    assert_eq!(actual_one.lower(), expected_one.lower());
    assert_eq!(actual_one.upper(), expected_one.upper());
    assert_eq!(one_polls, 8);

    let mut expected_n = BoundedTensor::new(lower.clone(), upper.clone()).expect("ordinary input");
    expected_n.round_for_soundness_n_ulps_inplace(17);
    let mut actual_n = BoundedTensor::new(lower, upper).expect("pollable input");
    let mut n_polls = 0usize;
    actual_n
        .round_for_soundness_n_ulps_inplace_with_poll(17, || {
            n_polls += 1;
            Ok(())
        })
        .expect("pollable n-ULP widening");
    assert_eq!(actual_n.lower(), expected_n.lower());
    assert_eq!(actual_n.upper(), expected_n.upper());
    assert_eq!(n_polls, 8);

    let mut cancelled = actual_n;
    let mut injected_polls = 0usize;
    let error = cancelled
        .round_for_soundness_inplace_with_poll(|| {
            injected_polls += 1;
            if injected_polls == 8 {
                Err(NyError::DeadlineExceeded(
                    "injected rounding publication poll".to_string(),
                ))
            } else {
                Ok(())
            }
        })
        .expect_err("final poll must return the injected error");
    assert!(matches!(error, NyError::DeadlineExceeded(_)));
    assert_eq!(injected_polls, 8);
}

#[test]
fn test_pollable_center_and_concrete_match_and_poll_8193_elements() {
    let lower = ndarray::ArrayD::from_shape_fn(IxDyn(&[8_193]), |index| {
        -4.0_f32 + index[0] as f32 * 0.0001
    });
    let upper =
        ndarray::ArrayD::from_shape_fn(IxDyn(&[8_193]), |index| 6.0_f32 + index[0] as f32 * 0.0001);
    let bounds = BoundedTensor::new(lower, upper).expect("bounds");
    let expected_center = bounds.center();
    let mut center_polls = 0usize;
    let actual_center = bounds
        .center_with_poll(|| {
            center_polls += 1;
            Ok(())
        })
        .expect("pollable center");
    assert_eq!(actual_center, expected_center);
    assert_eq!(center_polls, 7);

    let mut injected_center_polls = 0usize;
    let center_error = bounds
        .center_with_poll(|| {
            injected_center_polls += 1;
            if injected_center_polls == 7 {
                Err(NyError::DeadlineExceeded(
                    "injected center publication poll".to_string(),
                ))
            } else {
                Ok(())
            }
        })
        .expect_err("final center poll must prevent publication");
    assert!(matches!(center_error, NyError::DeadlineExceeded(_)));
    assert_eq!(injected_center_polls, 7);

    let expected_concrete =
        BoundedTensor::concrete(expected_center.clone()).expect("ordinary concrete");
    let mut concrete_polls = 0usize;
    let actual_concrete = BoundedTensor::concrete_with_poll(actual_center, || {
        concrete_polls += 1;
        Ok(())
    })
    .expect("pollable concrete");
    assert_eq!(actual_concrete.lower(), expected_concrete.lower());
    assert_eq!(actual_concrete.upper(), expected_concrete.upper());
    assert_eq!(concrete_polls, 9);

    let mut injected_concrete_polls = 0usize;
    let concrete_error = BoundedTensor::concrete_with_poll(expected_center, || {
        injected_concrete_polls += 1;
        if injected_concrete_polls == 9 {
            Err(NyError::DeadlineExceeded(
                "injected concrete publication poll".to_string(),
            ))
        } else {
            Ok(())
        }
    })
    .expect_err("final concrete poll must prevent publication");
    assert!(matches!(concrete_error, NyError::DeadlineExceeded(_)));
    assert_eq!(injected_concrete_polls, 9);
}

#[test]
fn test_pollable_intersection_matches_and_cancels_before_publication_8193_elements() {
    let a = BoundedTensor::new(
        ndarray::ArrayD::from_elem(IxDyn(&[8_193]), -2.0),
        ndarray::ArrayD::from_elem(IxDyn(&[8_193]), 3.0),
    )
    .expect("a");
    let b = BoundedTensor::new(
        ndarray::ArrayD::from_elem(IxDyn(&[8_193]), -1.0),
        ndarray::ArrayD::from_elem(IxDyn(&[8_193]), 4.0),
    )
    .expect("b");
    let expected = a
        .intersection_per_element(&b)
        .expect("ordinary intersection");
    let mut polls = 0usize;
    let actual = a
        .intersection_per_element_with_poll(&b, || {
            polls += 1;
            Ok(())
        })
        .expect("pollable intersection")
        .expect("matching shapes");
    assert_eq!(actual.0.lower(), expected.0.lower());
    assert_eq!(actual.0.upper(), expected.0.upper());
    assert_eq!(actual.1, expected.1);
    assert_eq!(polls, 20);

    let mut injected_polls = 0usize;
    let error = a
        .intersection_per_element_with_poll(&b, || {
            injected_polls += 1;
            if injected_polls == 20 {
                Err(NyError::DeadlineExceeded(
                    "injected publication poll".to_string(),
                ))
            } else {
                Ok(())
            }
        })
        .expect_err("final poll must prevent publication");
    assert!(matches!(error, NyError::DeadlineExceeded(_)));
    assert_eq!(injected_polls, 20);
}

#[test]
fn test_pollable_conservative_fill_matches_and_polls_8193_elements() {
    let expected = BoundedTensor::new_conservative(&[8_193]);
    let mut polls = 0usize;
    let actual = BoundedTensor::new_conservative_with_poll(&[8_193], || {
        polls += 1;
        Ok(())
    })
    .expect("pollable conservative bounds");
    assert_eq!(actual.lower(), expected.lower());
    assert_eq!(actual.upper(), expected.upper());
    assert_eq!(polls, 10);

    let mut injected_polls = 0usize;
    let error = BoundedTensor::new_conservative_with_poll(&[8_193], || {
        injected_polls += 1;
        if injected_polls == 10 {
            Err(NyError::DeadlineExceeded(
                "injected conservative publication poll".to_string(),
            ))
        } else {
            Ok(())
        }
    })
    .expect_err("final poll must prevent publication");
    assert!(matches!(error, NyError::DeadlineExceeded(_)));
    assert_eq!(injected_polls, 10);
}

#[test]
fn test_new_allow_infinite_with_poll_is_atomic_and_matches_legacy() {
    let lower = arr1(&[f32::NEG_INFINITY, -1.0, 0.0]).into_dyn();
    let upper = arr1(&[1.0, f32::INFINITY, 0.0]).into_dyn();
    let expected = BoundedTensor::new_allow_infinite(lower.clone(), upper.clone()).unwrap();

    let mut polls = 0usize;
    let actual = BoundedTensor::new_allow_infinite_with_poll(lower.clone(), upper.clone(), || {
        polls += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(actual.lower(), expected.lower());
    assert_eq!(actual.upper(), expected.upper());
    assert_eq!(polls, 5);

    let mut injected_polls = 0usize;
    let error = BoundedTensor::new_allow_infinite_with_poll(lower, upper, || {
        injected_polls += 1;
        if injected_polls == 5 {
            Err(NyError::DeadlineExceeded(
                "injected final constructor poll".to_string(),
            ))
        } else {
            Ok(())
        }
    })
    .expect_err("final poll must prevent publication");
    assert!(matches!(error, NyError::DeadlineExceeded(_)));
    assert_eq!(injected_polls, 5);
}

#[test]
fn test_into_reshape_with_poll_moves_standard_buffers_and_refuses_hidden_copy() {
    let tensor = BoundedTensor::new(
        arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]).into_dyn(),
        arr2(&[[5.0_f32, 6.0], [7.0, 8.0]]).into_dyn(),
    )
    .unwrap();
    let lower_ptr = tensor.lower().as_ptr();
    let upper_ptr = tensor.upper().as_ptr();
    let mut polls = 0usize;
    let reshaped = tensor
        .into_reshape_with_poll(&[4], || {
            polls += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(polls, 3);
    assert_eq!(reshaped.shape(), &[4]);
    assert_eq!(reshaped.lower().as_ptr(), lower_ptr);
    assert_eq!(reshaped.upper().as_ptr(), upper_ptr);

    let nonstandard = BoundedTensor::new(
        arr2(&[[1.0_f32, 2.0], [3.0, 4.0]])
            .reversed_axes()
            .into_dyn(),
        arr2(&[[5.0_f32, 6.0], [7.0, 8.0]])
            .reversed_axes()
            .into_dyn(),
    )
    .unwrap();
    assert!(matches!(
        nonstandard.into_reshape_with_poll(&[4], || Ok(())),
        Err(NyError::UnsupportedConfiguration(_))
    ));
}
