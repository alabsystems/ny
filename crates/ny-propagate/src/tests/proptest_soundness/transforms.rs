// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest IBP and CROWN backward soundness tests for shape-transform layers.
//!
//! These operators are value-preserving index transforms, so for any concrete input
//! point `x` inside input bounds, transformed output `f(x)` must lie inside
//! `propagate_ibp([l, u])`.
//!
//! Part of #40.
//! CROWN backward tests: Part of #1825.

use crate::layers::common::BoundPropagation;
use crate::layers::{
    FlattenLayer, GatherLayer, Layer, PadLayer, PadMode, ReshapeLayer, SliceLayer, SqueezeLayer,
    TileLayer, TransposeLayer, UnsqueezeLayer,
};
use crate::LinearBounds;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{sample_points, valid_interval, FP_TOLERANCE};

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

fn assert_transform_ibp_sound<L: BoundPropagation>(
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

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reshape_ibp(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let layer = ReshapeLayer::new(vec![3, 2]);
        assert_transform_ibp_sound(&layer, &[2, 3], &intervals, "Reshape")?;
    }

    #[ntest::timeout(10000)]
    #[test]
    fn soundness_flatten_ibp_axis1(intervals in prop::collection::vec(valid_interval(10.0), 8)) {
        let layer = FlattenLayer::new(1);
        assert_transform_ibp_sound(&layer, &[2, 2, 2], &intervals, "Flatten")?;
    }

    #[ntest::timeout(10000)]
    #[test]
    fn soundness_transpose_ibp_2d(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let layer = TransposeLayer::new(vec![1, 0]);
        assert_transform_ibp_sound(&layer, &[2, 3], &intervals, "Transpose")?;
    }

    #[ntest::timeout(10000)]
    #[test]
    fn soundness_tile_ibp_negative_axis(intervals in prop::collection::vec(valid_interval(10.0), 4)) {
        let layer = TileLayer::new(-1, 3);
        assert_transform_ibp_sound(&layer, &[2, 2], &intervals, "Tile")?;
    }

    #[ntest::timeout(10000)]
    #[test]
    fn soundness_slice_ibp_negative_axis(intervals in prop::collection::vec(valid_interval(10.0), 8)) {
        let layer = SliceLayer::new(-1, 1, 3);
        assert_transform_ibp_sound(&layer, &[2, 4], &intervals, "Slice")?;
    }

    #[ntest::timeout(10000)]
    #[test]
    fn soundness_squeeze_ibp(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let layer = SqueezeLayer::new(1);
        assert_transform_ibp_sound(&layer, &[2, 1, 3], &intervals, "Squeeze")?;
    }

    #[ntest::timeout(10000)]
    #[test]
    fn soundness_unsqueeze_ibp(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let layer = UnsqueezeLayer::new(1);
        assert_transform_ibp_sound(&layer, &[2, 3], &intervals, "Unsqueeze")?;
    }
}

// =============================================================================
// TIGHTNESS: Shape-transform layers must preserve exact bound values (#2131)
// =============================================================================
//
// Shape-transform layers (Reshape, Flatten, Transpose, Squeeze, Unsqueeze) are
// value-preserving index transforms. Their output bounds must exactly equal the
// input bounds (after reindexing). A regression that widens these to [-inf, +inf]
// would pass soundness tests but fail these tightness checks.

/// Assert that a shape-transform layer produces output bounds with the same
/// flat-order element values as the input bounds (exact identity on values).
fn assert_transform_exact_values<L: BoundPropagation>(
    layer: &L,
    input_shape: &[usize],
    intervals: &[(f32, f32)],
    layer_name: &str,
) -> Result<(), TestCaseError> {
    let input = bounded_from_intervals(input_shape, intervals);
    let output = layer
        .propagate_ibp(&input)
        .map_err(|e| TestCaseError::fail(format!("{layer_name} propagate_ibp failed: {e}")))?;

    // Total element count must be preserved.
    let in_len: usize = input.shape().iter().product();
    let out_len: usize = output.shape().iter().product();
    prop_assert_eq!(
        in_len,
        out_len,
        "{} element count changed: input={}, output={}",
        layer_name,
        in_len,
        out_len
    );

    // Flat-order values must match exactly (no tolerance — these are index
    // transforms, not arithmetic).
    for (idx, ((&il, &iu), (&ol, &ou))) in input
        .lower()
        .iter()
        .zip(input.upper().iter())
        .zip(output.lower().iter().zip(output.upper().iter()))
        .enumerate()
    {
        prop_assert_eq!(
            il,
            ol,
            "{} lower bound at flat index {} changed: input={}, output={}",
            layer_name,
            idx,
            il,
            ol
        );
        prop_assert_eq!(
            iu,
            ou,
            "{} upper bound at flat index {} changed: input={}, output={}",
            layer_name,
            idx,
            iu,
            ou
        );
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Reshape preserves exact bound values (#2131 tightness).
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_reshape_exact_values(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let layer = ReshapeLayer::new(vec![3, 2]);
        assert_transform_exact_values(&layer, &[2, 3], &intervals, "Reshape")?;
    }

    /// Flatten preserves exact bound values (#2131 tightness).
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_flatten_exact_values(intervals in prop::collection::vec(valid_interval(10.0), 8)) {
        let layer = FlattenLayer::new(1);
        assert_transform_exact_values(&layer, &[2, 2, 2], &intervals, "Flatten")?;
    }

    /// Squeeze preserves exact bound values (#2131 tightness).
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_squeeze_exact_values(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let layer = SqueezeLayer::new(1);
        assert_transform_exact_values(&layer, &[2, 1, 3], &intervals, "Squeeze")?;
    }

    /// Unsqueeze preserves exact bound values (#2131 tightness).
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_unsqueeze_exact_values(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let layer = UnsqueezeLayer::new(1);
        assert_transform_exact_values(&layer, &[2, 3], &intervals, "Unsqueeze")?;
    }

    /// Transpose preserves exact bound values under reindexing (#2131 tightness).
    /// Unlike Reshape/Flatten/Squeeze, Transpose changes element order, so we
    /// compare input[i,j] == output[j,i] rather than flat-order equality.
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_transpose_exact_values(intervals in prop::collection::vec(valid_interval(10.0), 6)) {
        let shape = [2usize, 3usize];
        let input = bounded_from_intervals(&shape, &intervals);
        let layer = TransposeLayer::new(vec![1, 0]);
        let output = layer.propagate_ibp(&input).unwrap();

        // Output shape must be [3, 2].
        prop_assert_eq!(output.shape(), &[3, 2]);

        // Each element: input[i, j] == output[j, i].
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                prop_assert_eq!(
                    input.lower()[[i, j]], output.lower()[[j, i]],
                    "Transpose lower at [{},{}] differs", i, j
                );
                prop_assert_eq!(
                    input.upper()[[i, j]], output.upper()[[j, i]],
                    "Transpose upper at [{},{}] differs", i, j
                );
            }
        }
    }
}

// =============================================================================
// CROWN BACKWARD SOUNDNESS: Tile and Slice (#1825)
// =============================================================================

/// Concretize LinearBounds to concrete lower/upper bound vectors.
fn concretize_crown_1d(
    result: &LinearBounds,
    pre_activation: &BoundedTensor,
) -> (Vec<f32>, Vec<f32>) {
    let concrete = result.concretize(pre_activation);
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Tile CROWN backward with identity incoming bounds (#1825).
    ///
    /// For input shape [2, 3] tiled along axis 1 with reps=2, output is [2, 6].
    /// Identity CROWN backward over the 12-element output should produce bounds
    /// that match IBP exactly (Tile is a linear operation).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_tile_crown_identity_incoming(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let mut tile = TileLayer::new(-1, 2);
        tile.set_input_shape(vec![2, 3]);
        let layer = Layer::Tile(tile.clone());

        // Output size = 2*6 = 12
        let identity = LinearBounds::identity(12);

        let result = layer
            .propagate_crown_backward(&identity, None)
            .map_err(|e| TestCaseError::fail(
                format!("Tile CROWN backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // Compare against IBP
        let ibp_result = tile.propagate_ibp(&pre_activation).unwrap();
        let ibp_lower: Vec<f32> = ibp_result.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_result.upper().iter().copied().collect();

        for i in 0..12 {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() < FP_TOLERANCE,
                "Tile CROWN-IBP lower mismatch at {i}: crown={} ibp={}",
                crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() < FP_TOLERANCE,
                "Tile CROWN-IBP upper mismatch at {i}: crown={} ibp={}",
                crown_upper[i], ibp_upper[i]
            );
        }
    }

    /// Tile CROWN backward with non-identity incoming bounds (#1825).
    ///
    /// Random coefficient matrix — verify that for every sampled input point,
    /// the true composed output lies within the concretized CROWN bounds.
    /// Tile(axis=-1, reps=2) on [2, 2] -> [2, 4], output_size=8.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_tile_crown_nonidentity_incoming(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
    ) {
        // Input [2, 2], output [2, 4] (8 elements)
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), vec![l0, l1, l2, l3],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), vec![u0, u1, u2, u3],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let mut tile = TileLayer::new(-1, 2);
        tile.set_input_shape(vec![2, 2]);

        // 1-row incoming: coefficients over 8-element tiled output
        let incoming = LinearBounds::new(
            Array2::from_shape_vec(
                (1, 8), vec![c0, c1, c0, c1, c0, c1, c0, c1],
            ).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec(
                (1, 8), vec![c0, c1, c0, c1, c0, c1, c0, c1],
            ).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let layer = Layer::Tile(tile);

        let result = layer
            .propagate_crown_backward(&incoming, None)
            .map_err(|e| TestCaseError::fail(
                format!("Tile CROWN nonidentity failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // tile([x0, x1, x2, x3]) = [x0, x1, x0, x1, x2, x3, x2, x3]
        // f(tile(x)) = c0*(x0+x0+x2+x2) + c1*(x1+x1+x3+x3)
        let samples: Vec<Vec<f32>> = (0..4)
            .map(|j| {
                let (lj, uj) = match j {
                    0 => (l0, u0), 1 => (l1, u1), 2 => (l2, u2), _ => (l3, u3),
                };
                sample_points(lj, uj, 3)
            })
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        let true_output = c0 * (s0 + s0 + s2 + s2)
                            + c1 * (s1 + s1 + s3 + s3);

                        let scale_tol = FP_TOLERANCE * true_output.abs().max(1.0);
                        prop_assert!(
                            crown_lower[0] - scale_tol <= true_output,
                            "Tile CROWN lower violated: lb={} > true={true_output}",
                            crown_lower[0]
                        );
                        prop_assert!(
                            true_output <= crown_upper[0] + scale_tol,
                            "Tile CROWN upper violated: true={true_output} > ub={}",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }

    /// Tile CROWN backward with asymmetric lower/upper coefficients (#1825).
    ///
    /// Tests the case where lower_a != upper_a, which occurs after passing
    /// through nonlinear layers in multi-layer CROWN backward.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_tile_crown_asymmetric_bounds(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        lc0 in -3.0f32..3.0,
        lc1 in -3.0f32..3.0,
        uc0 in -3.0f32..3.0,
        uc1 in -3.0f32..3.0,
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), vec![l0, l1, l2, l3],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), vec![u0, u1, u2, u3],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let mut tile = TileLayer::new(-1, 2);
        tile.set_input_shape(vec![2, 2]);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec(
                (1, 8), vec![lc0, lc1, lc0, lc1, lc0, lc1, lc0, lc1],
            ).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec(
                (1, 8), vec![uc0, uc1, uc0, uc1, uc0, uc1, uc0, uc1],
            ).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let layer = Layer::Tile(tile);

        let result = layer
            .propagate_crown_backward(&incoming, None)
            .map_err(|e| TestCaseError::fail(
                format!("Tile CROWN asymmetric failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let samples: Vec<Vec<f32>> = (0..4)
            .map(|j| {
                let (lj, uj) = match j {
                    0 => (l0, u0), 1 => (l1, u1), 2 => (l2, u2), _ => (l3, u3),
                };
                sample_points(lj, uj, 3)
            })
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        let lower_val = lc0 * (s0 + s0 + s2 + s2)
                            + lc1 * (s1 + s1 + s3 + s3);
                        let upper_val = uc0 * (s0 + s0 + s2 + s2)
                            + uc1 * (s1 + s1 + s3 + s3);

                        let lower_tol = FP_TOLERANCE * lower_val.abs().max(1.0);
                        let upper_tol = FP_TOLERANCE * upper_val.abs().max(1.0);

                        prop_assert!(
                            crown_lower[0] - lower_tol <= lower_val,
                            "Tile CROWN asym lower violated: lb={} > val={lower_val}",
                            crown_lower[0]
                        );
                        prop_assert!(
                            upper_val <= crown_upper[0] + upper_tol,
                            "Tile CROWN asym upper violated: val={upper_val} > ub={}",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }

    /// Slice CROWN backward with identity incoming bounds (#1825).
    ///
    /// For input [2, 4] sliced along axis -1 with start=1, end=3, output is [2, 2].
    /// Identity CROWN backward should produce bounds matching IBP.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_slice_crown_identity_incoming(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
        (l6, u6) in valid_interval(10.0),
        (l7, u7) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 4]), vec![l0, l1, l2, l3, l4, l5, l6, l7],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 4]), vec![u0, u1, u2, u3, u4, u5, u6, u7],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let slice = SliceLayer::new(-1, 1, 3);
        let layer = Layer::Slice(slice.clone());

        // Output = [2, 2] = 4 elements
        let identity = LinearBounds::identity(4);

        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("Slice CROWN backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let ibp_result = slice.propagate_ibp(&pre_activation).unwrap();
        let ibp_lower: Vec<f32> = ibp_result.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_result.upper().iter().copied().collect();

        for i in 0..4 {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() < FP_TOLERANCE,
                "Slice CROWN-IBP lower mismatch at {i}: crown={} ibp={}",
                crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() < FP_TOLERANCE,
                "Slice CROWN-IBP upper mismatch at {i}: crown={} ibp={}",
                crown_upper[i], ibp_upper[i]
            );
        }
    }

    /// Slice CROWN backward with non-identity incoming bounds (#1825).
    ///
    /// Random coefficients — verify sampled input points produce outputs within bounds.
    /// Slice(axis=-1, start=1, end=3) on [2, 4] -> [2, 2].
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_slice_crown_nonidentity_incoming(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        (l4, u4) in valid_interval(5.0),
        (l5, u5) in valid_interval(5.0),
        (l6, u6) in valid_interval(5.0),
        (l7, u7) in valid_interval(5.0),
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 4]), vec![l0, l1, l2, l3, l4, l5, l6, l7],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 4]), vec![u0, u1, u2, u3, u4, u5, u6, u7],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let slice = SliceLayer::new(-1, 1, 3);
        let layer = Layer::Slice(slice);

        // 1-row with 4 coefficients over 4-element output
        // slice output = [x1, x2, x5, x6] (indices 1,2 from row 0; 1,2 from row 1)
        let incoming = LinearBounds::new(
            Array2::from_shape_vec(
                (1, 4), vec![c0, c1, c0, c1],
            ).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec(
                (1, 4), vec![c0, c1, c0, c1],
            ).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("Slice CROWN nonidentity failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // slice([x0..x7]) = [x1, x2, x5, x6]
        // f(slice(x)) = c0*x1 + c1*x2 + c0*x5 + c1*x6
        let intervals = [
            (l0, u0), (l1, u1), (l2, u2), (l3, u3),
            (l4, u4), (l5, u5), (l6, u6), (l7, u7),
        ];
        let samples: Vec<Vec<f32>> = intervals
            .iter()
            .map(|&(lj, uj)| sample_points(lj, uj, 2))
            .collect();

        // Only the slice-relevant indices matter: x1, x2, x5, x6
        for &s1 in &samples[1] {
            for &s2 in &samples[2] {
                for &s5 in &samples[5] {
                    for &s6 in &samples[6] {
                        let true_output = c0 * s1 + c1 * s2 + c0 * s5 + c1 * s6;
                        let scale_tol = FP_TOLERANCE * true_output.abs().max(1.0);

                        prop_assert!(
                            crown_lower[0] - scale_tol <= true_output,
                            "Slice CROWN lower violated: lb={} > true={true_output}",
                            crown_lower[0]
                        );
                        prop_assert!(
                            true_output <= crown_upper[0] + scale_tol,
                            "Slice CROWN upper violated: true={true_output} > ub={}",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }

    /// Slice CROWN backward with asymmetric lower/upper coefficients (#1825).
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_slice_crown_asymmetric_bounds(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        (l4, u4) in valid_interval(5.0),
        (l5, u5) in valid_interval(5.0),
        (l6, u6) in valid_interval(5.0),
        (l7, u7) in valid_interval(5.0),
        lc0 in -3.0f32..3.0,
        lc1 in -3.0f32..3.0,
        uc0 in -3.0f32..3.0,
        uc1 in -3.0f32..3.0,
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 4]), vec![l0, l1, l2, l3, l4, l5, l6, l7],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 4]), vec![u0, u1, u2, u3, u4, u5, u6, u7],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let slice = SliceLayer::new(-1, 1, 3);
        let layer = Layer::Slice(slice);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec(
                (1, 4), vec![lc0, lc1, lc0, lc1],
            ).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec(
                (1, 4), vec![uc0, uc1, uc0, uc1],
            ).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("Slice CROWN asymmetric failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let intervals = [
            (l0, u0), (l1, u1), (l2, u2), (l3, u3),
            (l4, u4), (l5, u5), (l6, u6), (l7, u7),
        ];
        let samples: Vec<Vec<f32>> = intervals
            .iter()
            .map(|&(lj, uj)| sample_points(lj, uj, 2))
            .collect();

        for &s1 in &samples[1] {
            for &s2 in &samples[2] {
                for &s5 in &samples[5] {
                    for &s6 in &samples[6] {
                        let lower_val = lc0 * s1 + lc1 * s2 + lc0 * s5 + lc1 * s6;
                        let upper_val = uc0 * s1 + uc1 * s2 + uc0 * s5 + uc1 * s6;

                        let lower_tol = FP_TOLERANCE * lower_val.abs().max(1.0);
                        let upper_tol = FP_TOLERANCE * upper_val.abs().max(1.0);

                        prop_assert!(
                            crown_lower[0] - lower_tol <= lower_val,
                            "Slice CROWN asym lower violated: lb={} > val={lower_val}",
                            crown_lower[0]
                        );
                        prop_assert!(
                            upper_val <= crown_upper[0] + upper_tol,
                            "Slice CROWN asym upper violated: val={upper_val} > ub={}",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }

    /// Batched CROWN Slice backward + concretize regression (#3188).
    ///
    /// Before the input_shape fix, `concretize` would fail with ShapeMismatch
    /// because the returned `BatchedLinearBounds` carried the post-slice shape
    /// instead of the pre-slice (input) shape.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_slice_batched_crown_concretize(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        use crate::BatchedLinearBounds;

        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let slice = SliceLayer::new(0, 1, 3);
        // Output = [2], identity batched bounds
        let bounds = BatchedLinearBounds::identity(&[2]).map_err(|e|
            TestCaseError::fail(format!("identity failed: {e}"))
        )?;

        let result = slice.propagate_linear_batched(&bounds, &pre_activation).map_err(|e|
            TestCaseError::fail(format!("batched backward failed: {e}"))
        )?;

        // Key assertion: concretize must succeed (was ShapeMismatch before fix)
        let concrete = result.concretize(&pre_activation).map_err(|e|
            TestCaseError::fail(format!("concretize failed: {e}"))
        )?;

        // Verify soundness: concretized bounds match IBP for identity incoming
        let ibp = slice.propagate_ibp(&pre_activation).unwrap();
        let conc_lower: Vec<f32> = concrete.lower().iter().copied().collect();
        let conc_upper: Vec<f32> = concrete.upper().iter().copied().collect();
        let ibp_lower: Vec<f32> = ibp.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp.upper().iter().copied().collect();

        for i in 0..2 {
            prop_assert!(
                (conc_lower[i] - ibp_lower[i]).abs() < FP_TOLERANCE,
                "Slice batched CROWN-IBP lower mismatch at {i}: batched={} ibp={}",
                conc_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (conc_upper[i] - ibp_upper[i]).abs() < FP_TOLERANCE,
                "Slice batched CROWN-IBP upper mismatch at {i}: batched={} ibp={}",
                conc_upper[i], ibp_upper[i]
            );
        }
    }
}

// =============================================================================
// BATCHED CROWN BACKWARD SOUNDNESS: Tile (#287)
// =============================================================================
//
// These tests verify the batched CROWN propagation path (propagate_linear_batched)
// which is used in per-block CROWN and the graph-level batched CROWN backward.
// The non-batched (DAG-CROWN) path is tested above; these test the N-D batched
// coefficient path which sums across tile replicas.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Tile batched CROWN backward with identity incoming bounds (#287).
    ///
    /// For input [6] tiled along axis 0 with reps=2, output is [12].
    /// Identity CROWN should produce concretized bounds matching IBP.
    /// Uses 1D input because Tile's flat A matrices are incompatible with
    /// concretize's batch-dim broadcasting for multi-dim inputs.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_tile_batched_crown_identity(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
    ) {
        use crate::BatchedLinearBounds;

        let lower = ArrayD::from_shape_vec(IxDyn(&[6]), vec![l0, l1, l2, l3, l4, l5]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[6]), vec![u0, u1, u2, u3, u4, u5]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let mut tile = TileLayer::new(0, 2);
        tile.set_input_shape(vec![6]);

        let bounds = BatchedLinearBounds::identity(&[12]).map_err(|e|
            TestCaseError::fail(format!("identity failed: {e}"))
        )?;

        let result = tile.propagate_linear_batched(&bounds, &pre_activation).map_err(|e|
            TestCaseError::fail(format!("Tile batched CROWN failed: {e}"))
        )?;

        let concrete = result.concretize(&pre_activation).map_err(|e|
            TestCaseError::fail(format!("concretize failed: {e}"))
        )?;

        let ibp = tile.propagate_ibp(&pre_activation).unwrap();
        let conc_lower: Vec<f32> = concrete.lower().iter().copied().collect();
        let conc_upper: Vec<f32> = concrete.upper().iter().copied().collect();
        let ibp_lower: Vec<f32> = ibp.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp.upper().iter().copied().collect();

        for i in 0..12 {
            prop_assert!(
                (conc_lower[i] - ibp_lower[i]).abs() < FP_TOLERANCE,
                "Tile batched CROWN-IBP lower mismatch at {i}: batched={} ibp={}",
                conc_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (conc_upper[i] - ibp_upper[i]).abs() < FP_TOLERANCE,
                "Tile batched CROWN-IBP upper mismatch at {i}: batched={} ibp={}",
                conc_upper[i], ibp_upper[i]
            );
        }
    }

    /// Tile batched CROWN backward for 1D reps=3 (#287).
    ///
    /// For Tile(axis=0, reps=3) on [4], output [12]. With identity batched bounds
    /// the backward should sum contributions from 3 replicas per input position.
    /// Verifies the core summation logic.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_tile_batched_crown_1d_reps3(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        use crate::BatchedLinearBounds;

        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let mut tile = TileLayer::new(0, 3);
        tile.set_input_shape(vec![4]);

        // Output shape: [12]
        let bounds = BatchedLinearBounds::identity(&[12]).map_err(|e|
            TestCaseError::fail(format!("identity failed: {e}"))
        )?;

        let result = tile.propagate_linear_batched(&bounds, &pre_activation).map_err(|e|
            TestCaseError::fail(format!("Tile batched CROWN 1D failed: {e}"))
        )?;

        let concrete = result.concretize(&pre_activation).map_err(|e|
            TestCaseError::fail(format!("concretize failed: {e}"))
        )?;

        let ibp = tile.propagate_ibp(&pre_activation).unwrap();
        let conc_lower: Vec<f32> = concrete.lower().iter().copied().collect();
        let conc_upper: Vec<f32> = concrete.upper().iter().copied().collect();
        let ibp_lower: Vec<f32> = ibp.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp.upper().iter().copied().collect();

        for i in 0..12 {
            prop_assert!(
                (conc_lower[i] - ibp_lower[i]).abs() < FP_TOLERANCE,
                "Tile 1D reps=3 lower mismatch at {i}: batched={} ibp={}",
                conc_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (conc_upper[i] - ibp_upper[i]).abs() < FP_TOLERANCE,
                "Tile 1D reps=3 upper mismatch at {i}: batched={} ibp={}",
                conc_upper[i], ibp_upper[i]
            );
        }
    }

    /// Tile batched CROWN with pointwise soundness verification (#287).
    ///
    /// For Tile(axis=0, reps=2) on [4] -> [8], verify that for every
    /// sampled input point within bounds, the tiled output lies within the
    /// concretized batched CROWN bounds.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_tile_batched_crown_pointwise(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
    ) {
        use crate::BatchedLinearBounds;

        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let mut tile = TileLayer::new(0, 2);
        tile.set_input_shape(vec![4]);

        let bounds = BatchedLinearBounds::identity(&[8]).map_err(|e|
            TestCaseError::fail(format!("identity failed: {e}"))
        )?;

        let result = tile.propagate_linear_batched(&bounds, &pre_activation).map_err(|e|
            TestCaseError::fail(format!("Tile batched CROWN failed: {e}"))
        )?;

        let concrete = result.concretize(&pre_activation).map_err(|e|
            TestCaseError::fail(format!("concretize failed: {e}"))
        )?;

        // Sample input points and verify tiled outputs are within bounds
        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3)];
        for point_vals in representative_points(&intervals) {
            let point = ArrayD::from_shape_vec(IxDyn(&[4]), point_vals).unwrap();
            let point_bt = BoundedTensor::new(point.clone(), point).unwrap();
            let tiled = tile.propagate_ibp(&point_bt).unwrap();

            for (idx, ((&tl, &tu), (&cl, &cu))) in tiled.lower().iter()
                .zip(tiled.upper().iter())
                .zip(concrete.lower().iter().zip(concrete.upper().iter()))
                .enumerate()
            {
                prop_assert!(
                    cl - FP_TOLERANCE <= tl && tu <= cu + FP_TOLERANCE,
                    "Tile batched CROWN pointwise violation at {idx}: \
                     tiled=[{tl},{tu}] not in [{cl},{cu}]"
                );
            }
        }
    }
}

// =============================================================================
// CROWN BACKWARD SOUNDNESS: Gather (#3400)
// =============================================================================
//
// GatherLayer is a linear selection operation: it picks elements along an axis.
// The CROWN backward pass scatters incoming A-matrix columns back to their
// original input positions. These proptests verify that the scatter is correct
// by checking concretized CROWN bounds match IBP (identity case) and contain
// all sampled true outputs (non-identity case).

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Gather CROWN backward with identity incoming bounds (#3400).
    ///
    /// For input [4], gather indices=[1, 3], output [2].
    /// Identity CROWN backward should produce bounds matching IBP exactly.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_gather_crown_identity_1d(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1i64, 3]).unwrap();
        let mut layer = GatherLayer::new(0, Some(indices), vec![]);
        layer.set_input_shape(vec![4]);
        let layer_enum = Layer::Gather(layer.clone());

        // Output [2], identity incoming
        let identity = LinearBounds::identity(2);
        let result = layer_enum
            .propagate_crown_backward(&identity, None)
            .map_err(|e| TestCaseError::fail(
                format!("Gather CROWN backward failed: {e}")
            ))?;
        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // IBP reference
        let ibp = layer.propagate_ibp(&pre_activation).unwrap();
        let ibp_lower: Vec<f32> = ibp.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp.upper().iter().copied().collect();

        for i in 0..2 {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() < FP_TOLERANCE,
                "Gather CROWN-IBP lower mismatch at {i}: crown={} ibp={}",
                crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() < FP_TOLERANCE,
                "Gather CROWN-IBP upper mismatch at {i}: crown={} ibp={}",
                crown_upper[i], ibp_upper[i]
            );
        }
    }

    /// Gather CROWN backward with identity incoming bounds, 2D axis=0 (#3400).
    ///
    /// Input [3, 2], gather axis=0 indices=[0, 2], output [2, 2] (4 elements).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_gather_crown_identity_2d_axis0(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[3, 2]), vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[3, 2]), vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 2]).unwrap();
        let mut layer = GatherLayer::new(0, Some(indices), vec![]);
        layer.set_input_shape(vec![3, 2]);
        let layer_enum = Layer::Gather(layer.clone());

        let identity = LinearBounds::identity(4);
        let result = layer_enum
            .propagate_crown_backward(&identity, None)
            .map_err(|e| TestCaseError::fail(
                format!("Gather 2D CROWN backward failed: {e}")
            ))?;
        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let ibp = layer.propagate_ibp(&pre_activation).unwrap();
        let ibp_lower: Vec<f32> = ibp.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp.upper().iter().copied().collect();

        for i in 0..4 {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() < FP_TOLERANCE,
                "Gather 2D CROWN-IBP lower mismatch at {i}: crown={} ibp={}",
                crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() < FP_TOLERANCE,
                "Gather 2D CROWN-IBP upper mismatch at {i}: crown={} ibp={}",
                crown_upper[i], ibp_upper[i]
            );
        }
    }

    /// Gather CROWN backward with non-identity incoming bounds (#3400).
    ///
    /// Random coefficients — verify that for every sampled input point,
    /// the true composed output lies within the concretized CROWN bounds.
    /// Gather(axis=0, indices=[1, 3]) on [4] -> [2].
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_gather_crown_nonidentity_incoming(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
    ) {
        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1i64, 3]).unwrap();
        let mut layer = GatherLayer::new(0, Some(indices), vec![]);
        layer.set_input_shape(vec![4]);
        let layer_enum = Layer::Gather(layer);

        // 1-row incoming: coefficients over 2-element gathered output
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer_enum
            .propagate_crown_backward(&incoming, None)
            .map_err(|e| TestCaseError::fail(
                format!("Gather CROWN nonidentity failed: {e}")
            ))?;
        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // gather([x0, x1, x2, x3]) = [x1, x3]
        // f(gather(x)) = c0 * x1 + c1 * x3
        let samples: Vec<Vec<f32>> = [(l0, u0), (l1, u1), (l2, u2), (l3, u3)]
            .iter()
            .map(|&(lj, uj)| sample_points(lj, uj, 3))
            .collect();

        // Only x1 and x3 affect the output
        for &s1 in &samples[1] {
            for &s3 in &samples[3] {
                let true_output = c0 * s1 + c1 * s3;
                let scale_tol = FP_TOLERANCE * true_output.abs().max(1.0);

                prop_assert!(
                    crown_lower[0] - scale_tol <= true_output,
                    "Gather CROWN lower violated: lb={} > true={true_output}",
                    crown_lower[0]
                );
                prop_assert!(
                    true_output <= crown_upper[0] + scale_tol,
                    "Gather CROWN upper violated: true={true_output} > ub={}",
                    crown_upper[0]
                );
            }
        }
    }

    /// Gather CROWN backward with asymmetric lower/upper coefficients (#3400).
    ///
    /// Tests the case where lower_a != upper_a, which occurs after passing
    /// through nonlinear layers in multi-layer CROWN backward.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_gather_crown_asymmetric_bounds(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        lc0 in -3.0f32..3.0,
        lc1 in -3.0f32..3.0,
        uc0 in -3.0f32..3.0,
        uc1 in -3.0f32..3.0,
    ) {
        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1i64, 3]).unwrap();
        let mut layer = GatherLayer::new(0, Some(indices), vec![]);
        layer.set_input_shape(vec![4]);
        let layer_enum = Layer::Gather(layer);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lc0, lc1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![uc0, uc1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer_enum
            .propagate_crown_backward(&incoming, None)
            .map_err(|e| TestCaseError::fail(
                format!("Gather CROWN asymmetric failed: {e}")
            ))?;
        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let samples: Vec<Vec<f32>> = [(l0, u0), (l1, u1), (l2, u2), (l3, u3)]
            .iter()
            .map(|&(lj, uj)| sample_points(lj, uj, 3))
            .collect();

        for &s1 in &samples[1] {
            for &s3 in &samples[3] {
                let lower_val = lc0 * s1 + lc1 * s3;
                let upper_val = uc0 * s1 + uc1 * s3;

                let lower_tol = FP_TOLERANCE * lower_val.abs().max(1.0);
                let upper_tol = FP_TOLERANCE * upper_val.abs().max(1.0);

                prop_assert!(
                    crown_lower[0] - lower_tol <= lower_val,
                    "Gather CROWN asym lower violated: lb={} > val={lower_val}",
                    crown_lower[0]
                );
                prop_assert!(
                    upper_val <= crown_upper[0] + upper_tol,
                    "Gather CROWN asym upper violated: val={upper_val} > ub={}",
                    crown_upper[0]
                );
            }
        }
    }

    /// Gather CROWN backward with duplicate indices (#3400).
    ///
    /// Gather(axis=0, indices=[1, 1]) on [3] -> [2]. Both output positions
    /// select the same input element. The backward scatter must accumulate
    /// (+=), not overwrite. This tests a subtle correctness property.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_gather_crown_duplicate_indices(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
    ) {
        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1i64, 1]).unwrap();
        let mut layer = GatherLayer::new(0, Some(indices), vec![]);
        layer.set_input_shape(vec![3]);
        let layer_enum = Layer::Gather(layer);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer_enum
            .propagate_crown_backward(&incoming, None)
            .map_err(|e| TestCaseError::fail(
                format!("Gather CROWN duplicate failed: {e}")
            ))?;
        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // gather([x0, x1, x2]) with indices=[1,1] = [x1, x1]
        // f(gather(x)) = c0 * x1 + c1 * x1 = (c0 + c1) * x1
        for &s1 in &sample_points(l1, u1, 5) {
            let true_output = (c0 + c1) * s1;
            let scale_tol = FP_TOLERANCE * true_output.abs().max(1.0);

            prop_assert!(
                crown_lower[0] - scale_tol <= true_output,
                "Gather CROWN dup lower violated: lb={} > true={true_output}",
                crown_lower[0]
            );
            prop_assert!(
                true_output <= crown_upper[0] + scale_tol,
                "Gather CROWN dup upper violated: true={true_output} > ub={}",
                crown_upper[0]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// PadLayer CROWN backward proptests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Constant pad CROWN backward with identity incoming bounds.
    ///
    /// Input [4], pad (1,1) with constant 0, output [6].
    /// Identity CROWN should match IBP exactly since Pad is linear.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_pad_constant_crown_identity(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let layer = Layer::Pad(PadLayer::new(vec![(1, 1)], PadMode::Constant(0.0)));

        // Output size = 6
        let identity = LinearBounds::identity(6);

        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("Pad constant CROWN identity failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let pad_layer = PadLayer::new(vec![(1, 1)], PadMode::Constant(0.0));
        let ibp_result = pad_layer.propagate_ibp(&pre_activation).unwrap();
        let ibp_lower: Vec<f32> = ibp_result.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_result.upper().iter().copied().collect();

        for i in 0..6 {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() < FP_TOLERANCE,
                "Pad constant CROWN-IBP lower mismatch at {i}: crown={} ibp={}",
                crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() < FP_TOLERANCE,
                "Pad constant CROWN-IBP upper mismatch at {i}: crown={} ibp={}",
                crown_upper[i], ibp_upper[i]
            );
        }
    }

    /// Constant pad CROWN backward with asymmetric coefficients.
    ///
    /// Input [3], pad (1,1) with constant 5.0, output [5].
    /// For sampled input points, verify that applying coefficients to the
    /// padded output lies within the concretized CROWN bounds.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_pad_constant_crown_asymmetric(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        lc in -3.0f32..3.0,
        uc in -3.0f32..3.0,
    ) {
        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let pad_value = 5.0f32;
        let pad_layer = PadLayer::new(vec![(1, 1)], PadMode::Constant(pad_value));

        // Output [5]: 1-row with coefficients [c, c, c, c, c]
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 5), vec![lc; 5]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 5), vec![uc; 5]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let layer = Layer::Pad(pad_layer);
        let result = layer
            .propagate_crown_backward(&incoming, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("Pad constant CROWN asymmetric failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // Padded output = [5, x0, x1, x2, 5]
        // lower_val = lc * (5 + x0 + x1 + x2 + 5)
        // upper_val = uc * (5 + x0 + x1 + x2 + 5)
        let samples: Vec<Vec<f32>> = (0..3)
            .map(|j| {
                let (lj, uj) = match j {
                    0 => (l0, u0), 1 => (l1, u1), _ => (l2, u2),
                };
                sample_points(lj, uj, 5)
            })
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    let padded = [pad_value, s0, s1, s2, pad_value];
                    let lower_val: f32 = padded.iter().map(|v| lc * v).sum();
                    let upper_val: f32 = padded.iter().map(|v| uc * v).sum();

                    let lower_tol = FP_TOLERANCE * lower_val.abs().max(1.0);
                    let upper_tol = FP_TOLERANCE * upper_val.abs().max(1.0);

                    prop_assert!(
                        crown_lower[0] - lower_tol <= lower_val,
                        "Pad constant CROWN lower violated: lb={} > val={lower_val}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        upper_val <= crown_upper[0] + upper_tol,
                        "Pad constant CROWN upper violated: val={upper_val} > ub={}",
                        crown_upper[0]
                    );
                }
            }
        }
    }

    /// Reflect pad CROWN backward with asymmetric coefficients.
    ///
    /// Input [1, 4], pad [(0,0), (2,2)] with reflect mode, output [1, 8].
    /// For sampled input points, verify that applying coefficients to the
    /// reflected output lies within the concretized CROWN bounds.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_pad_reflect_crown_asymmetric(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        lc in -3.0f32..3.0,
        uc in -3.0f32..3.0,
    ) {
        let lower = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let pad_layer = PadLayer::new(vec![(0, 0), (2, 2)], PadMode::Reflect);

        // Output [1, 8] = 8 elements. 1-row incoming.
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 8), vec![lc; 8]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 8), vec![uc; 8]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let layer = Layer::Pad(pad_layer);
        let result = layer
            .propagate_crown_backward(&incoming, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("Pad reflect CROWN asymmetric failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // Reflect pad [1,4] with (0,0),(2,2):
        // axis 0: no pad
        // axis 1: input [a,b,c,d] -> [c,b, a,b,c,d, c,b]
        let samples: Vec<Vec<f32>> = (0..4)
            .map(|j| {
                let (lj, uj) = match j {
                    0 => (l0, u0), 1 => (l1, u1), 2 => (l2, u2), _ => (l3, u3),
                };
                sample_points(lj, uj, 5)
            })
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        // Reflect: [c,b, a,b,c,d, c,b]
                        let padded = [s2, s1, s0, s1, s2, s3, s2, s1];
                        let lower_val: f32 = padded.iter().map(|v| lc * v).sum();
                        let upper_val: f32 = padded.iter().map(|v| uc * v).sum();

                        let lower_tol = FP_TOLERANCE * lower_val.abs().max(1.0);
                        let upper_tol = FP_TOLERANCE * upper_val.abs().max(1.0);

                        prop_assert!(
                            crown_lower[0] - lower_tol <= lower_val,
                            "Pad reflect CROWN lower violated: lb={} > val={lower_val}",
                            crown_lower[0]
                        );
                        prop_assert!(
                            upper_val <= crown_upper[0] + upper_tol,
                            "Pad reflect CROWN upper violated: val={upper_val} > ub={}",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }

    /// Reflect pad CROWN backward with identity incoming bounds.
    ///
    /// Input [1, 3], pad [(0,0), (1,1)] with reflect mode, output [1, 5].
    /// CROWN should match IBP exactly since Pad is a linear operation.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_pad_reflect_crown_identity(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![u0, u1, u2]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let pad_layer = PadLayer::new(vec![(0, 0), (1, 1)], PadMode::Reflect);
        let layer = Layer::Pad(pad_layer.clone());

        let identity = LinearBounds::identity(5);

        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("Pad reflect CROWN identity failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let ibp_result = pad_layer.propagate_ibp(&pre_activation).unwrap();
        let ibp_lower: Vec<f32> = ibp_result.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_result.upper().iter().copied().collect();

        for i in 0..5 {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() < FP_TOLERANCE,
                "Pad reflect CROWN-IBP lower mismatch at {i}: crown={} ibp={}",
                crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() < FP_TOLERANCE,
                "Pad reflect CROWN-IBP upper mismatch at {i}: crown={} ibp={}",
                crown_upper[i], ibp_upper[i]
            );
        }
    }
}
