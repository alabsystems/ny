// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Brute-force soundness proptests for bounded-index dynamic ScatterND
//! (#cctsdb B4).
//!
//! Property: the IBP hull produced from indices INTERVALS must enclose the
//! concrete ScatterND output for EVERY integer index assignment inside those
//! intervals (last-write-wins in row order, ONNX semantics), at representative
//! data/updates samples. Assignments containing an out-of-range index are
//! excluded: ONNX leaves them undefined, so no enclosure obligation exists.

use crate::layers::ScatterNdLayer;
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::FP_TOLERANCE;

/// Concrete ONNX ScatterND with depth-`d` indices over `data_shape`,
/// slice_len 1 (full addressing): last write wins in row order.
/// Returns `None` if any index is out of range (undefined behavior).
fn concrete_scatter(
    data: &[f32],
    data_shape: &[usize],
    rows: &[Vec<i64>],
    updates: &[f32],
) -> Option<Vec<f32>> {
    let mut strides = vec![1usize; data_shape.len()];
    for idx in (1..data_shape.len()).rev() {
        strides[idx - 1] = strides[idx] * data_shape[idx];
    }
    let mut out = data.to_vec();
    for (row, update) in rows.iter().zip(updates.iter()) {
        let mut target = 0usize;
        for (axis, &raw) in row.iter().enumerate() {
            let len = data_shape[axis] as i64;
            let normalized = if raw < 0 { raw + len } else { raw };
            if normalized < 0 || normalized >= len {
                return None; // undefined behavior — no obligation
            }
            target += (normalized as usize) * strides[axis];
        }
        out[target] = *update;
    }
    Some(out)
}

/// Enumerate every integer assignment of the index intervals (odometer).
fn all_assignments(intervals: &[(i64, i64)]) -> Vec<Vec<i64>> {
    let mut result = vec![vec![]];
    for &(lo, hi) in intervals {
        let mut next = Vec::new();
        for assignment in &result {
            for value in lo..=hi {
                let mut extended = assignment.clone();
                extended.push(value);
                next.push(extended);
            }
        }
        result = next;
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn assert_bounded_scatter_sound(
    data_shape: &[usize],
    index_depth: usize,
    num_rows: usize,
    index_intervals: &[(i64, i64)],
    data_intervals: &[(f32, f32)],
    update_intervals: &[(f32, f32)],
) -> Result<(), TestCaseError> {
    let data_size: usize = data_shape.iter().product();
    assert_eq!(data_intervals.len(), data_size);
    assert_eq!(update_intervals.len(), num_rows);
    assert_eq!(index_intervals.len(), num_rows * index_depth);

    let data = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(data_shape),
            data_intervals.iter().map(|(l, _)| *l).collect(),
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(data_shape),
            data_intervals.iter().map(|(_, u)| *u).collect(),
        )
        .unwrap(),
    )
    .unwrap();
    let indices = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[num_rows, index_depth]),
            index_intervals.iter().map(|(l, _)| *l as f32).collect(),
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[num_rows, index_depth]),
            index_intervals.iter().map(|(_, u)| *u as f32).collect(),
        )
        .unwrap(),
    )
    .unwrap();
    let updates = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[num_rows]),
            update_intervals.iter().map(|(l, _)| *l).collect(),
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[num_rows]),
            update_intervals.iter().map(|(_, u)| *u).collect(),
        )
        .unwrap(),
    )
    .unwrap();

    let layer = ScatterNdLayer::new(None, None, None);
    let bounds = layer
        .propagate_ibp_ternary(&data, &indices, &updates)
        .map_err(|e| TestCaseError::fail(format!("bounded scatter IBP failed: {e}")))?;

    // Representative data/updates value samples: endpoints and midpoint.
    let sample = |intervals: &[(f32, f32)], pick: usize| -> Vec<f32> {
        intervals
            .iter()
            .map(|&(l, u)| match pick {
                0 => l,
                1 => u,
                _ => f32::midpoint(l, u),
            })
            .collect()
    };

    for assignment in all_assignments(index_intervals) {
        let rows: Vec<Vec<i64>> = assignment
            .chunks(index_depth)
            .map(|chunk| chunk.to_vec())
            .collect();
        for data_pick in 0..3 {
            for update_pick in 0..3 {
                let data_point = sample(data_intervals, data_pick);
                let update_point = sample(update_intervals, update_pick);
                let Some(concrete) =
                    concrete_scatter(&data_point, data_shape, &rows, &update_point)
                else {
                    continue;
                };
                for (element, &value) in concrete.iter().enumerate() {
                    let lo = bounds.lower().as_slice().unwrap()[element];
                    let up = bounds.upper().as_slice().unwrap()[element];
                    prop_assert!(
                        lo - FP_TOLERANCE <= value && value <= up + FP_TOLERANCE,
                        "bounded ScatterND enclosure violated at element {element}: \
                         concrete {value} not in [{lo}, {up}] \
                         (assignment {rows:?}, data_pick {data_pick}, update_pick {update_pick})"
                    );
                }
            }
        }
    }
    Ok(())
}

fn small_interval(radius: f32) -> impl Strategy<Value = (f32, f32)> {
    (-radius..radius, 0.0..radius).prop_map(|(lo, width)| (lo, lo + width))
}

fn index_interval(max_abs: i64, max_width: i64) -> impl Strategy<Value = (i64, i64)> {
    (-max_abs..=max_abs, 0..=max_width).prop_map(|(lo, width)| (lo, lo + width))
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(128) })]

    /// Depth-1 scatter over a [5] vector: hull encloses every concrete
    /// instantiation over the index ranges (incl. duplicates and negatives).
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_bounded_scatter_depth1(
        index_intervals in prop::collection::vec(index_interval(6, 2), 2),
        data_intervals in prop::collection::vec(small_interval(10.0), 5),
        update_intervals in prop::collection::vec(small_interval(10.0), 2),
    ) {
        assert_bounded_scatter_sound(
            &[5], 1, 2, &index_intervals, &data_intervals, &update_intervals,
        )?;
    }

    /// Depth-2 scatter over a [2, 3] matrix (full addressing): hull encloses
    /// every concrete instantiation over the 2-D index boxes.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_bounded_scatter_depth2(
        index_intervals in prop::collection::vec(index_interval(4, 1), 4),
        data_intervals in prop::collection::vec(small_interval(10.0), 6),
        update_intervals in prop::collection::vec(small_interval(10.0), 2),
    ) {
        assert_bounded_scatter_sound(
            &[2, 3], 2, 2, &index_intervals, &data_intervals, &update_intervals,
        )?;
    }
}
