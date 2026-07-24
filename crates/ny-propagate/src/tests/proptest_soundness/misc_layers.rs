// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest IBP soundness tests for misc layers: Gather, Where, NonZero,
//! SkipMerge, OpaqueSkip, Floor, Ceil, Round, Sign, Reciprocal (zero-crossing).
//!
//! Part of #40, #2564.

use crate::layers::common::BoundPropagation;
use crate::layers::{
    CeilLayer, FloorLayer, GatherLayer, NonZeroLayer, OpaqueSkipLayer, ReciprocalLayer, RoundLayer,
    SignLayer, SkipMergeLayer, TruncLayer, WhereLayer,
};
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{valid_interval, FP_TOLERANCE};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bounded_from_intervals(shape: &[usize], intervals: &[(f32, f32)]) -> BoundedTensor {
    let total: usize = shape.iter().product();
    assert_eq!(
        total,
        intervals.len(),
        "interval count must match shape product"
    );
    let lower: Vec<f32> = intervals.iter().map(|(l, _)| *l).collect();
    let upper: Vec<f32> = intervals.iter().map(|(_, u)| *u).collect();
    let lower = ArrayD::from_shape_vec(IxDyn(shape), lower).expect("valid lower shape");
    let upper = ArrayD::from_shape_vec(IxDyn(shape), upper).expect("valid upper shape");
    BoundedTensor::new(lower, upper).expect("valid bounded tensor")
}

fn point_tensor(shape: &[usize], values: &[f32]) -> BoundedTensor {
    let point = ArrayD::from_shape_vec(IxDyn(shape), values.to_vec()).expect("valid point shape");
    BoundedTensor::new(point.clone(), point).expect("valid point tensor")
}

fn representative_points(intervals: &[(f32, f32)]) -> Vec<Vec<f32>> {
    let lower: Vec<f32> = intervals.iter().map(|(l, _)| *l).collect();
    let upper: Vec<f32> = intervals.iter().map(|(_, u)| *u).collect();
    let midpoint: Vec<f32> = intervals.iter().map(|(l, u)| 0.5 * (l + u)).collect();

    let mut points = vec![lower.clone(), upper.clone(), midpoint];
    for idx in 0..intervals.len() {
        let mut mixed = lower.clone();
        mixed[idx] = upper[idx];
        points.push(mixed);
    }
    points
}

fn assert_contains_bounds(
    bounds: &BoundedTensor,
    concrete: &BoundedTensor,
    layer_name: &str,
) -> Result<(), TestCaseError> {
    prop_assert_eq!(
        bounds.shape(),
        concrete.shape(),
        "{} shape mismatch: bounds={:?}, concrete={:?}",
        layer_name,
        bounds.shape(),
        concrete.shape()
    );

    for (idx, ((&l, &u), &y)) in bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .zip(concrete.lower().iter())
        .enumerate()
    {
        // Skip infinite bounds (OpaqueSkipLayer, etc.) — trivially sound
        if l.is_infinite() && l.is_sign_negative() && u.is_infinite() && u.is_sign_positive() {
            continue;
        }
        prop_assert!(
            l - FP_TOLERANCE <= y && y <= u + FP_TOLERANCE,
            "{} soundness violation at output index {}: y={} not in [{}, {}]",
            layer_name,
            idx,
            y,
            l,
            u
        );
    }
    Ok(())
}

/// Assert IBP soundness for a unary layer: propagate interval bounds, then check
/// that representative concrete points map inside those bounds.
fn assert_unary_ibp_sound<L: BoundPropagation>(
    layer: &L,
    input_shape: &[usize],
    intervals: &[(f32, f32)],
    layer_name: &str,
) -> Result<(), TestCaseError> {
    let input_bounds = bounded_from_intervals(input_shape, intervals);
    let output_bounds = layer.propagate_ibp(&input_bounds).map_err(|e| {
        TestCaseError::fail(format!(
            "{} propagate_ibp on interval input failed: {e}",
            layer_name
        ))
    })?;

    for point in representative_points(intervals) {
        let point_input = point_tensor(input_shape, &point);
        let point_output = layer.propagate_ibp(&point_input).map_err(|e| {
            TestCaseError::fail(format!(
                "{} propagate_ibp on point input failed: {e}",
                layer_name
            ))
        })?;
        assert_contains_bounds(&output_bounds, &point_output, layer_name)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GatherLayer — static indices
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Gather axis=0 with static indices [0, 2] from a [3, 2] input.
    /// Output is [2, 2] — selects rows 0 and 2.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_gather_static_axis0(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 2]).unwrap();
        let layer = GatherLayer::new(0, Some(indices), vec![]);
        assert_unary_ibp_sound(&layer, &[3, 2], &intervals, "Gather(axis0,static)")?;
    }

    /// Gather axis=1 with static indices [0, 2] from a [2, 3] input.
    /// Output is [2, 2] — selects columns 0 and 2.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_gather_static_axis1(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 2]).unwrap();
        let layer = GatherLayer::new(1, Some(indices), vec![]);
        assert_unary_ibp_sound(&layer, &[2, 3], &intervals, "Gather(axis1,static)")?;
    }

    /// Gather axis=0 with negative index [-1] from a [4] input.
    /// Output is [1] — selects last element.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_gather_static_negative(intervals in prop::collection::vec(valid_interval(10.0), 4)) {
        let indices = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1i64]).unwrap();
        let layer = GatherLayer::new(0, Some(indices), vec![]);
        assert_unary_ibp_sound(&layer, &[4], &intervals, "Gather(negative)")?;
    }
}

// ---------------------------------------------------------------------------
// GatherLayer — dynamic indices (conservative min/max fallback)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Dynamic gather on axis=0 from [3, 2] with indices_shape=[2].
    /// Conservative: output lower = min(lower) across axis, upper = max(upper).
    /// Every possible static gather result must lie within these bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_gather_dynamic_axis0(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let layer = GatherLayer::new(0, None, vec![2]);
        let input_shape = [3, 2];
        let input_bounds = bounded_from_intervals(&input_shape, &intervals);
        let output_bounds = layer.propagate_ibp(&input_bounds).map_err(|e| {
            TestCaseError::fail(format!(
                "Gather(dynamic) propagate_ibp failed: {e}"
            ))
        })?;

        // Verify that every possible static index produces outputs within dynamic bounds.
        for idx in 0..3i64 {
            let static_indices = ArrayD::from_shape_vec(IxDyn(&[1]), vec![idx]).unwrap();
            let static_layer = GatherLayer::new(0, Some(static_indices), vec![]);
            let static_output = static_layer.propagate_ibp(&input_bounds).map_err(|e| {
                TestCaseError::fail(format!(
                    "Gather(static idx={idx}) propagate_ibp failed: {e}"
                ))
            })?;

            // Static output [1, 2] must fit within dynamic output [2, 2].
            // The dynamic bounds broadcast to cover all possible indices,
            // so each column of static output should be within corresponding
            // column of dynamic bounds.
            for col in 0..2 {
                let y_lower = static_output.lower()[[0, col]];
                let y_upper = static_output.upper()[[0, col]];
                // Dynamic bounds are broadcast, check column
                let dyn_lower = output_bounds.lower()[[0, col]];
                let dyn_upper = output_bounds.upper()[[0, col]];
                prop_assert!(
                    dyn_lower - FP_TOLERANCE <= y_lower,
                    "Gather(dynamic) lower violation: col={col} idx={idx}: dyn_lower={dyn_lower} > static_lower={y_lower}"
                );
                prop_assert!(
                    y_upper <= dyn_upper + FP_TOLERANCE,
                    "Gather(dynamic) upper violation: col={col} idx={idx}: static_upper={y_upper} > dyn_upper={dyn_upper}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WhereLayer — ternary IBP (no embedded constants)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// WhereLayer ternary: output bounds must contain both x and y branches.
    /// For any concrete condition, the result is either x[i] or y[i], so the
    /// union bounds [min(xl, yl), max(xu, yu)] are sound.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_where_ternary(
        x_intervals in prop::collection::vec(valid_interval(10.0), 4),
        y_intervals in prop::collection::vec(valid_interval(10.0), 4),
    ) {
        let shape = [2, 2];
        let x_bounds = bounded_from_intervals(&shape, &x_intervals);
        let y_bounds = bounded_from_intervals(&shape, &y_intervals);
        // Condition doesn't matter for soundness — Where takes union of x and y.
        let condition = bounded_from_intervals(
            &shape,
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        );

        let layer = WhereLayer::new();
        let output = layer.propagate_ibp_ternary(&condition, &x_bounds, &y_bounds)
            .map_err(|e| TestCaseError::fail(
                format!("Where ternary failed: {e}")
            ))?;

        // Every concrete x point must be in output bounds.
        for point in representative_points(&x_intervals) {
            let concrete = point_tensor(&shape, &point);
            assert_contains_bounds(&output, &concrete, "Where(x-branch)")?;
        }
        // Every concrete y point must also be in output bounds.
        for point in representative_points(&y_intervals) {
            let concrete = point_tensor(&shape, &point);
            assert_contains_bounds(&output, &concrete, "Where(y-branch)")?;
        }
    }
}

// ---------------------------------------------------------------------------
// WhereLayer — embedded constants IBP
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// WhereLayer with embedded scalar constants: true_val and false_val are
    /// fixed scalars. Output bounds must contain both possible values for every
    /// element regardless of condition.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_where_embedded_constants(
        true_val in -10.0f32..10.0,
        false_val in -10.0f32..10.0,
    ) {
        let shape = [2, 2];
        let const_true = ArrayD::from_elem(IxDyn(&[1]), true_val);
        let const_false = ArrayD::from_elem(IxDyn(&[1]), false_val);
        let layer = WhereLayer::with_constants(Some(const_true), Some(const_false));

        // Condition is arbitrary — bounds should hold regardless.
        let condition = bounded_from_intervals(
            &shape,
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        );

        let output = layer.propagate_ibp(&condition).map_err(|e| {
            TestCaseError::fail(format!(
                "Where(embedded) propagate_ibp failed: {e}"
            ))
        })?;

        // Both constant values must be within the output bounds.
        let expected_lower = true_val.min(false_val);
        let expected_upper = true_val.max(false_val);

        for idx in 0..output.lower().len() {
            let l = output.lower().iter().nth(idx).unwrap();
            let u = output.upper().iter().nth(idx).unwrap();
            prop_assert!(
                *l - FP_TOLERANCE <= expected_lower,
                "Where(embedded) lower bound too tight at {idx}: bound={l}, need<={expected_lower}"
            );
            prop_assert!(
                *u + FP_TOLERANCE >= expected_upper,
                "Where(embedded) upper bound too tight at {idx}: bound={u}, need>={expected_upper}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// WhereLayer — propagate_ibp returns error without embedded constants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(50) })]

    /// WhereLayer without embedded constants should return UnsupportedOp from
    /// the trait propagate_ibp (callers must use propagate_ibp_ternary).
    #[ntest::timeout(10000)]
    #[test]
    fn where_no_constants_returns_error(intervals in prop::collection::vec(valid_interval(10.0), 4)) {
        let layer = WhereLayer::new();
        let input = bounded_from_intervals(&[2, 2], &intervals);
        let result = layer.propagate_ibp(&input);
        prop_assert!(result.is_err(), "WhereLayer::propagate_ibp should fail without embedded constants");
    }
}

// ---------------------------------------------------------------------------
// NonZeroLayer — IBP soundness
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// NonZero returns index bounds. For any concrete input, the actual nonzero
    /// indices must lie within the declared index bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_nonzero_ibp(intervals in prop::collection::vec(valid_interval(5.0), 6)) {
        let shape = [2, 3];
        let input_bounds = bounded_from_intervals(&shape, &intervals);
        let layer = NonZeroLayer;

        let output_bounds = layer.propagate_ibp(&input_bounds).map_err(|e| {
            TestCaseError::fail(format!(
                "NonZero propagate_ibp failed: {e}"
            ))
        })?;

        // For each representative concrete point, find actual nonzero indices
        // and verify they fall within the output bounds.
        for point in representative_points(&intervals) {
            let concrete_arr = ArrayD::from_shape_vec(IxDyn(&shape), point).unwrap();

            // Compute actual nonzero indices using flat iteration.
            // Shape is [2, 3] so row = flat / 3, col = flat % 3.
            let mut nonzero_indices: Vec<Vec<usize>> = vec![Vec::new(); shape.len()];
            for (flat, &val) in concrete_arr.iter().enumerate() {
                if val != 0.0 {
                    let row = flat / shape[1];
                    let col = flat % shape[1];
                    nonzero_indices[0].push(row);
                    nonzero_indices[1].push(col);
                }
            }

            let num_nonzero = nonzero_indices[0].len();

            // If the concrete point has more nonzero elements than the output
            // can hold, that's a soundness issue (output shape is too small).
            // The output has shape [rank, max_possibly_nonzero].
            if output_bounds.shape().len() == 2 && output_bounds.shape()[1] > 0 {
                prop_assert!(
                    num_nonzero <= output_bounds.shape()[1],
                    "NonZero: concrete has {} nonzero but output shape is {:?}",
                    num_nonzero,
                    output_bounds.shape()
                );

                // Verify each actual index is within the index bounds.
                for (dim, indices) in nonzero_indices.iter().enumerate() {
                    for (col, &actual_idx) in indices.iter().enumerate() {
                        let lower_bound = output_bounds.lower()[[dim, col]];
                        let upper_bound = output_bounds.upper()[[dim, col]];
                        let actual_f = actual_idx as f32;
                        prop_assert!(
                            lower_bound - FP_TOLERANCE <= actual_f
                                && actual_f <= upper_bound + FP_TOLERANCE,
                            "NonZero index violation: dim={dim} col={col} actual={actual_idx} \
                             not in [{lower_bound}, {upper_bound}]"
                        );
                    }
                }
            }
        }
    }

    /// NonZero with all-zero input should produce empty output (shape [rank, 0]).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_nonzero_all_zero(_dummy in 0..1u32) {
        let shape = [2, 3];
        // Intervals all exactly [0, 0]
        let intervals = vec![(0.0f32, 0.0f32); 6];
        let input = bounded_from_intervals(&shape, &intervals);
        let layer = NonZeroLayer;
        let output = layer.propagate_ibp(&input).unwrap();
        // All intervals are [0, 0], so count_possibly_nonzero should be 0.
        prop_assert_eq!(output.shape(), &[2, 0],
            "NonZero all-zero input should produce [rank, 0] output, got {:?}", output.shape());
    }
}

// ---------------------------------------------------------------------------
// SkipMergeLayer — identity IBP soundness
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// SkipMergeLayer is identity: output bounds must exactly equal input bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_skip_merge_ibp(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let layer = SkipMergeLayer;
        assert_unary_ibp_sound(&layer, &[2, 3], &intervals, "SkipMerge")?;
    }

    /// SkipMergeLayer preserves exact values — output == input for all shapes.
    #[ntest::timeout(10000)]
    #[test]
    fn skip_merge_exact_identity(intervals in prop::collection::vec(valid_interval(10.0), 4)) {
        let shape = [2, 2];
        let input = bounded_from_intervals(&shape, &intervals);
        let layer = SkipMergeLayer;
        let output = layer.propagate_ibp(&input).unwrap();

        prop_assert_eq!(output.shape(), input.shape());
        for ((&il, &iu), (&ol, &ou)) in input.lower().iter()
            .zip(input.upper().iter())
            .zip(output.lower().iter().zip(output.upper().iter()))
        {
            prop_assert_eq!(il, ol, "SkipMerge lower must match exactly");
            prop_assert_eq!(iu, ou, "SkipMerge upper must match exactly");
        }
    }
}

// ---------------------------------------------------------------------------
// OpaqueSkipLayer — conservative unbounded IBP soundness
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// OpaqueSkipLayer returns [-inf, +inf] for every element.
    /// Trivially sound — any concrete output is contained.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_opaque_skip_ibp(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let layer = OpaqueSkipLayer::new();
        // This always succeeds since output is [-inf, inf].
        assert_unary_ibp_sound(&layer, &[2, 3], &intervals, "OpaqueSkip")?;
    }

    /// OpaqueSkipLayer output bounds must be [-inf, +inf] for every element.
    #[ntest::timeout(10000)]
    #[test]
    fn opaque_skip_bounds_are_infinite(intervals in prop::collection::vec(valid_interval(10.0), 4)) {
        let shape = [2, 2];
        let input = bounded_from_intervals(&shape, &intervals);
        let layer = OpaqueSkipLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();

        prop_assert_eq!(output.shape(), input.shape());
        for (&l, &u) in output.lower().iter().zip(output.upper().iter()) {
            prop_assert!(l.is_infinite() && l.is_sign_negative(),
                "OpaqueSkip lower must be -inf, got {l}");
            prop_assert!(u.is_infinite() && u.is_sign_positive(),
                "OpaqueSkip upper must be +inf, got {u}");
        }
    }
}

// ---------------------------------------------------------------------------
// FloorLayer — IBP soundness
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Floor IBP soundness: for any x in [l, u], floor(x) is within IBP bounds.
    /// Part of #2564.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_floor_ibp(intervals in prop::collection::vec(valid_interval(20.0), 4)) {
        let layer = FloorLayer::new();
        assert_unary_ibp_sound(&layer, &[4], &intervals, "Floor")?;
    }
}

// ---------------------------------------------------------------------------
// CeilLayer — IBP soundness
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Ceil IBP soundness: for any x in [l, u], ceil(x) is within IBP bounds.
    /// Part of #2564.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_ceil_ibp(intervals in prop::collection::vec(valid_interval(20.0), 4)) {
        let layer = CeilLayer::new();
        assert_unary_ibp_sound(&layer, &[4], &intervals, "Ceil")?;
    }
}

// ---------------------------------------------------------------------------
// RoundLayer — IBP soundness
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Round IBP soundness: for any x in [l, u], round(x) is within IBP bounds.
    /// Part of #2564.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_round_ibp(intervals in prop::collection::vec(valid_interval(20.0), 4)) {
        let layer = RoundLayer::new();
        assert_unary_ibp_sound(&layer, &[4], &intervals, "Round")?;
    }
}

// ---------------------------------------------------------------------------
// TruncLayer — IBP soundness (ONNX Cast-to-int lowering, #cctsdb B1)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Trunc IBP soundness: for any x in [l, u], trunc(x) is within IBP bounds.
    /// This is the enclosure property that the previous identity-drop of
    /// float->int Cast VIOLATED (trunc(0.5)=0 not in [0.5, 62]).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_trunc_ibp(intervals in prop::collection::vec(valid_interval(20.0), 4)) {
        let layer = TruncLayer::new();
        assert_unary_ibp_sound(&layer, &[4], &intervals, "Trunc")?;
    }
}

/// The exact counterexample from the cctsdb design: identity does NOT enclose
/// trunc on [0.5, 62] (trunc(0.5) = 0 lies outside), but the Trunc layer does.
#[ntest::timeout(10000)]
#[test]
fn trunc_encloses_identity_unsound_case() {
    use ndarray::{ArrayD, IxDyn};
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 0.5_f32),
        ArrayD::from_elem(IxDyn(&[1]), 62.0_f32),
    )
    .unwrap();
    let out = TruncLayer::new().propagate_ibp(&input).unwrap();
    // Exact hull: [trunc(0.5), trunc(62)] = [0, 62]; must contain trunc(0.5)=0.
    assert_eq!(out.lower()[[0]], 0.0);
    assert_eq!(out.upper()[[0]], 62.0);
    // Identity bounds [0.5, 62] would NOT contain 0 — the old unsoundness.
    assert!(out.lower()[[0]] < 0.5);
}

/// Trunc differs from Floor on negatives: trunc(-1.5) = -1 (toward zero),
/// floor(-1.5) = -2. The IBP hull must use trunc, not floor.
#[ntest::timeout(10000)]
#[test]
fn trunc_negative_values_round_toward_zero() {
    use ndarray::{ArrayD, IxDyn};
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -1.5_f32),
        ArrayD::from_elem(IxDyn(&[1]), -0.5_f32),
    )
    .unwrap();
    let out = TruncLayer::new().propagate_ibp(&input).unwrap();
    assert_eq!(out.lower()[[0]], -1.0);
    assert_eq!(out.upper()[[0]], -0.0);
}

// ---------------------------------------------------------------------------
// SignLayer — IBP soundness
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Sign IBP soundness: for any x in [l, u], sign(x) is within IBP bounds.
    /// Exercises all sign_interval_bounds branches: purely positive, purely
    /// negative, crossing zero, touching zero from one side.
    /// Part of #2564.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_sign_ibp(intervals in prop::collection::vec(valid_interval(20.0), 4)) {
        let layer = SignLayer::new();
        assert_unary_ibp_sound(&layer, &[4], &intervals, "Sign")?;
    }

    /// Sign IBP with intervals guaranteed to cross zero — exercises the
    /// (-1, 1) fallback path in sign_interval_bounds.
    /// Part of #2564.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_sign_ibp_crossing_zero(
        neg_part in 0.01f32..10.0,
        pos_part in 0.01f32..10.0,
    ) {
        let l = -neg_part;
        let u = pos_part;
        let input = bounded_from_intervals(&[1], &[(l, u)]);
        let layer = SignLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();

        // Crossing zero: sign can be -1, 0, or 1
        prop_assert!(
            output.lower()[[0]] <= -1.0 + FP_TOLERANCE,
            "Sign IBP crossing-zero lower should be <= -1, got {}",
            output.lower()[[0]]
        );
        prop_assert!(
            output.upper()[[0]] >= 1.0 - FP_TOLERANCE,
            "Sign IBP crossing-zero upper should be >= 1, got {}",
            output.upper()[[0]]
        );
    }
}

// ---------------------------------------------------------------------------
// ReciprocalLayer — IBP soundness (normal intervals)
// ---------------------------------------------------------------------------
// Single-element positive/negative IBP tests are in elementwise.rs.
// This section covers multi-element tensor propagation and edge cases.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Reciprocal IBP soundness for multi-element tensor with all-positive intervals.
    /// Verifies the elementwise propagation is correct across multiple dimensions.
    /// Complements the single-element tests in elementwise.rs.
    /// Part of #2564.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reciprocal_ibp_multi_element(
        l0 in 0.1f32..5.0, d0 in 0.0f32..5.0,
        l1 in 0.1f32..5.0, d1 in 0.0f32..5.0,
        l2 in 0.1f32..5.0, d2 in 0.0f32..5.0,
    ) {
        let intervals = vec![(l0, l0 + d0), (l1, l1 + d1), (l2, l2 + d2)];
        let layer = ReciprocalLayer::new();
        assert_unary_ibp_sound(&layer, &[3], &intervals, "Reciprocal(multi)")?;
    }

    /// Reciprocal IBP soundness for multi-element tensor with mixed positive/negative.
    /// Each element is independently positive or negative (no zero-crossing per element).
    /// Part of #2564.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reciprocal_ibp_multi_mixed(
        l0 in 0.1f32..5.0, d0 in 0.0f32..5.0,
        u1_abs in 0.1f32..5.0, d1 in 0.0f32..5.0,
    ) {
        let intervals = vec![(l0, l0 + d0), (-u1_abs - d1, -u1_abs)];
        let layer = ReciprocalLayer::new();
        assert_unary_ibp_sound(&layer, &[2], &intervals, "Reciprocal(mixed)")?;
    }
}

// ---------------------------------------------------------------------------
// ReciprocalLayer — IBP zero-crossing soundness
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Reciprocal IBP with zero-crossing interval returns Ok with conservative
    /// fallback bounds since #3030 added new_repaired before BoundedTensor::new.
    /// The Inf endpoints are preserved (a non-finite endpoint carries no proven
    /// bound, so no finite substitute is sound). Verify the bounds contain
    /// sampled reciprocal values at the interval endpoints (away from zero).
    /// Part of #2564.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reciprocal_ibp_zero_crossing(
        neg_part in 0.01f32..10.0,
        pos_part in 0.01f32..10.0,
    ) {
        let l = -neg_part;
        let u = pos_part;
        let input = bounded_from_intervals(&[1], &[(l, u)]);
        let layer = ReciprocalLayer::new();
        let result = layer.propagate_ibp(&input);

        // #3030 new_repaired accepts ±Inf endpoints (NaN widens to ±inf) → Ok
        let output = result.expect("Reciprocal IBP zero-crossing should return Ok after #3030 repair");
        let out_lower = output.lower()[[0]];
        let out_upper = output.upper()[[0]];

        // Bounds must contain 1/l and 1/u (the reciprocal at the endpoints)
        let recip_l = 1.0 / l; // negative
        let recip_u = 1.0 / u; // positive
        prop_assert!(
            out_lower <= recip_l + FP_TOLERANCE,
            "Reciprocal IBP [{l}, {u}]: lower bound {out_lower} should be <= 1/{l} = {recip_l}"
        );
        prop_assert!(
            out_upper >= recip_u - FP_TOLERANCE,
            "Reciprocal IBP [{l}, {u}]: upper bound {out_upper} should be >= 1/{u} = {recip_u}"
        );
    }

    /// Reciprocal IBP with zero endpoint returns Ok with conservative fallback
    /// bounds since #3030 added new_repaired. Part of #2564.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reciprocal_ibp_zero_endpoint(val in 0.01f32..10.0, side: bool) {
        let (l, u) = if side { (0.0, val) } else { (-val, 0.0) };
        let input = bounded_from_intervals(&[1], &[(l, u)]);
        let layer = ReciprocalLayer::new();
        let result = layer.propagate_ibp(&input);

        // #3030 new_repaired accepts ±Inf endpoints (NaN widens to ±inf) → Ok
        let output = result.expect("Reciprocal IBP zero-endpoint should return Ok after #3030 repair");
        let out_lower = output.lower()[[0]];
        let out_upper = output.upper()[[0]];

        // Bounds must contain the reciprocal at the non-zero endpoint
        if side {
            // [0, val]: 1/val is the finite endpoint reciprocal
            let recip_val = 1.0 / val;
            prop_assert!(
                out_upper >= recip_val - FP_TOLERANCE,
                "Reciprocal IBP [0, {val}]: upper {out_upper} should be >= 1/{val} = {recip_val}"
            );
        } else {
            // [-val, 0]: 1/(-val) is the finite endpoint reciprocal
            let recip_val = 1.0 / (-val);
            prop_assert!(
                out_lower <= recip_val + FP_TOLERANCE,
                "Reciprocal IBP [-{val}, 0]: lower {out_lower} should be <= 1/-{val} = {recip_val}"
            );
        }
    }
}
