// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Selection-style reduction layers: TopK values/indices, ArgMax/ArgMin, ArgSort.
//!
//! These layers primarily serve cross-repo bounds propagation for traced models.
//! IBP is sound and intentionally conservative; CROWN backward is left as an
//! explicit fallback because these operators are piecewise-constant or
//! data-dependent selection maps.

use std::borrow::Cow;

use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::resolve_reduction_axes;
use crate::{BoundPropagation, LinearBounds};

/// Enclose an integer index in binary32 without assuming it is exactly
/// representable (indices above 2^24 need two adjacent f32 endpoints).
fn index_interval(index: usize) -> (f32, f32) {
    let rounded = index as f32;
    // Compare in a wider integer domain. Casting a rounded value just above
    // `usize::MAX` back to usize saturates and can falsely look exact at the
    // theoretical maximum index.
    let represented = rounded as u128;
    let exact = index as u128;
    if represented < exact {
        (rounded, ny_tensor::next_up_f32(rounded))
    } else if represented > exact {
        (ny_tensor::next_down_f32(rounded), rounded)
    } else {
        (rounded, rounded)
    }
}

fn index_upper_bound(index: usize) -> f32 {
    index_interval(index).1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopkOutputKind {
    Values,
    Indices,
}

#[derive(Debug, Clone)]
pub struct TopkLayer {
    pub k: usize,
    pub axis: i64,
    pub output: TopkOutputKind,
}

impl TopkLayer {
    pub fn new(k: usize, axis: i64, output: TopkOutputKind) -> Self {
        Self { k, axis, output }
    }

    pub fn values(k: usize, axis: i64) -> Self {
        Self::new(k, axis, TopkOutputKind::Values)
    }

    pub fn indices(k: usize, axis: i64) -> Self {
        Self::new(k, axis, TopkOutputKind::Indices)
    }

    fn resolve_axis(&self, ndim: usize) -> Result<usize> {
        Ok(resolve_reduction_axes(&[self.axis], ndim, "Topk")?[0])
    }

    fn output_shape(&self, input_shape: &[usize], axis: usize) -> Result<Vec<usize>> {
        let axis_len = input_shape[axis];
        if self.k == 0 || self.k > axis_len {
            return Err(NyError::InvalidSpec(format!(
                "Topk k={} out of range for axis {} with size {}",
                self.k, axis, axis_len
            )));
        }
        let mut out_shape = input_shape.to_vec();
        out_shape[axis] = self.k;
        Ok(out_shape)
    }

    fn propagate_values_ibp(&self, input: &BoundedTensor, axis: usize) -> Result<BoundedTensor> {
        let out_shape = self.output_shape(input.shape(), axis)?;
        let has_non_finite = input.lower().iter().any(|&v| !v.is_finite())
            || input.upper().iter().any(|&v| !v.is_finite());
        if has_non_finite {
            let lower = ArrayD::from_elem(IxDyn(&out_shape), f32::NEG_INFINITY);
            let upper = ArrayD::from_elem(IxDyn(&out_shape), f32::INFINITY);
            return BoundedTensor::new_allow_infinite(lower, upper);
        }

        let axis_obj = Axis(axis);
        let mut lower = ArrayD::zeros(IxDyn(&out_shape));
        let mut upper = ArrayD::zeros(IxDyn(&out_shape));
        for (((mut out_lower, mut out_upper), lower_lane), upper_lane) in lower
            .lanes_mut(axis_obj)
            .into_iter()
            .zip(upper.lanes_mut(axis_obj))
            .zip(input.lower().lanes(axis_obj))
            .zip(input.upper().lanes(axis_obj))
        {
            let mut lane_lower = lower_lane.iter().copied().collect::<Vec<_>>();
            let mut lane_upper = upper_lane.iter().copied().collect::<Vec<_>>();
            lane_lower.sort_unstable_by(|a, b| b.total_cmp(a));
            lane_upper.sort_unstable_by(|a, b| b.total_cmp(a));
            for (slot, value) in out_lower
                .iter_mut()
                .zip(lane_lower.into_iter().take(self.k))
            {
                *slot = value;
            }
            for (slot, value) in out_upper
                .iter_mut()
                .zip(lane_upper.into_iter().take(self.k))
            {
                *slot = value;
            }
        }

        BoundedTensor::new(lower, upper)
    }

    fn propagate_indices_ibp(&self, input: &BoundedTensor, axis: usize) -> Result<BoundedTensor> {
        let out_shape = self.output_shape(input.shape(), axis)?;
        let axis_len = input.shape()[axis];
        let upper_index = if axis_len <= 1 {
            0.0
        } else {
            index_upper_bound(axis_len - 1)
        };
        let lower = ArrayD::from_elem(IxDyn(&out_shape), 0.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&out_shape), upper_index);
        BoundedTensor::new(lower, upper)
    }
}

impl BoundPropagation for TopkLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let axis = self.resolve_axis(input.shape().len())?;
        match self.output {
            TopkOutputKind::Values => self.propagate_values_ibp(input, axis),
            TopkOutputKind::Indices => self.propagate_indices_ibp(input, axis),
        }
    }

    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Topk CROWN backward is not implemented".to_string(),
        ))
    }
}

#[derive(Debug, Clone)]
pub struct ArgMaxLayer {
    pub axis: i64,
    pub keepdims: bool,
    pub select_last_index: bool,
}

impl ArgMaxLayer {
    pub fn new(axis: i64, keepdims: bool) -> Self {
        Self {
            axis,
            keepdims,
            select_last_index: false,
        }
    }

    pub fn with_select_last_index(mut self, select_last_index: bool) -> Self {
        self.select_last_index = select_last_index;
        self
    }

    fn resolve_axis(&self, ndim: usize) -> Result<usize> {
        Ok(resolve_reduction_axes(&[self.axis], ndim, "ArgMax")?[0])
    }
}

#[derive(Debug, Clone)]
pub struct ArgMinLayer {
    pub axis: i64,
    pub keepdims: bool,
    pub select_last_index: bool,
}

impl ArgMinLayer {
    pub fn new(axis: i64, keepdims: bool) -> Self {
        Self {
            axis,
            keepdims,
            select_last_index: false,
        }
    }

    pub fn with_select_last_index(mut self, select_last_index: bool) -> Self {
        self.select_last_index = select_last_index;
        self
    }

    fn resolve_axis(&self, ndim: usize) -> Result<usize> {
        Ok(resolve_reduction_axes(&[self.axis], ndim, "ArgMin")?[0])
    }
}

#[derive(Debug, Clone)]
pub struct ArgSortLayer {
    pub axis: i64,
    pub descending: bool,
}

impl ArgSortLayer {
    pub fn new(axis: i64, descending: bool) -> Self {
        Self { axis, descending }
    }

    fn resolve_axis(&self, ndim: usize) -> Result<usize> {
        Ok(resolve_reduction_axes(&[self.axis], ndim, "ArgSort")?[0])
    }
}

fn argext_output_shape(input_shape: &[usize], axis: usize, keepdims: bool) -> Vec<usize> {
    let mut out_shape = input_shape.to_vec();
    if keepdims {
        out_shape[axis] = 1;
    } else {
        out_shape.remove(axis);
    }
    out_shape
}

fn exact_argext_index(
    lower_lane: &[f32],
    upper_lane: &[f32],
    use_argmax: bool,
    select_last_index: bool,
) -> Option<usize> {
    for candidate in 0..lower_lane.len() {
        let cand_lower = lower_lane[candidate];
        let cand_upper = upper_lane[candidate];
        if !cand_lower.is_finite() || !cand_upper.is_finite() {
            return None;
        }
        let mut exact = true;
        for other in 0..lower_lane.len() {
            if other == candidate {
                continue;
            }
            let other_lower = lower_lane[other];
            let other_upper = upper_lane[other];
            if !other_lower.is_finite() || !other_upper.is_finite() {
                return None;
            }
            // ONNX breaks exact ties by source index. A competitor on the
            // preferred side must be strictly dominated; one on the other
            // side may tie because the candidate wins that tie.
            let other_wins_tie = if select_last_index {
                other > candidate
            } else {
                other < candidate
            };
            let dominates = if use_argmax {
                if other_wins_tie {
                    cand_lower > other_upper
                } else {
                    cand_lower >= other_upper
                }
            } else if other_wins_tie {
                cand_upper < other_lower
            } else {
                cand_upper <= other_lower
            };
            if !dominates {
                exact = false;
                break;
            }
        }
        if exact {
            return Some(candidate);
        }
    }
    None
}

/// Exact ArgSort permutation for a single lane when the sort order is certain.
///
/// Returns `Some(indices)` only when every adjacent pair in the resolved order is
/// strictly separated by intervals (so no input perturbation within the bounds can
/// reorder them). In that case the permutation is the unique sound output and we
/// can emit it as a degenerate (tight) interval. When any adjacency is ambiguous
/// — including ties — we return `None` and the caller falls back to the sound
/// uniform `[0, axis_len - 1]` range. Non-finite bounds also force the fallback.
fn exact_argsort_lane(
    lower_lane: &[f32],
    upper_lane: &[f32],
    descending: bool,
) -> Option<Vec<usize>> {
    let n = lower_lane.len();
    if n == 0 {
        return Some(Vec::new());
    }
    if lower_lane
        .iter()
        .chain(upper_lane.iter())
        .any(|v| !v.is_finite())
    {
        return None;
    }

    // Candidate order: sort indices by the bound that defines rank (lower for
    // ascending, upper for descending) so the comparison key matches the
    // separation check below.
    let mut order: Vec<usize> = (0..n).collect();
    if descending {
        // Largest first: order by upper bound descending.
        order.sort_by(|&a, &b| {
            upper_lane[b]
                .partial_cmp(&upper_lane[a])
                .expect("finite bounds compared above")
        });
    } else {
        // Smallest first: order by lower bound ascending.
        order.sort_by(|&a, &b| {
            lower_lane[a]
                .partial_cmp(&lower_lane[b])
                .expect("finite bounds compared above")
        });
    }

    // The order is unambiguous only when each element strictly dominates the
    // next in the resolved order across the *entire* interval. For ascending,
    // upper[curr] < lower[next]; for descending, lower[curr] > upper[next].
    for window in order.windows(2) {
        let (curr, next) = (window[0], window[1]);
        let separated = if descending {
            lower_lane[curr] > upper_lane[next]
        } else {
            upper_lane[curr] < lower_lane[next]
        };
        if !separated {
            return None;
        }
    }

    // ArgSort output position p holds the source index that sorts into rank p.
    Some(order)
}

fn propagate_argext_ibp(
    input: &BoundedTensor,
    axis: usize,
    keepdims: bool,
    use_argmax: bool,
    select_last_index: bool,
    layer_name: &str,
) -> Result<BoundedTensor> {
    let axis_len = input.shape()[axis];
    if axis_len == 0 {
        return Err(NyError::InvalidSpec(format!(
            "{layer_name}: reduction axis {axis} has size 0"
        )));
    }
    let out_shape = argext_output_shape(input.shape(), axis, keepdims);
    let out_elems = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let mut out_lower = Vec::with_capacity(out_elems);
    let mut out_upper = Vec::with_capacity(out_elems);
    let uncertain_upper = if axis_len <= 1 {
        0.0
    } else {
        index_upper_bound(axis_len - 1)
    };

    for (lower_lane, upper_lane) in input
        .lower()
        .lanes(Axis(axis))
        .into_iter()
        .zip(input.upper().lanes(Axis(axis)))
    {
        let lower_vec = lower_lane.iter().copied().collect::<Vec<_>>();
        let upper_vec = upper_lane.iter().copied().collect::<Vec<_>>();
        if let Some(exact) =
            exact_argext_index(&lower_vec, &upper_vec, use_argmax, select_last_index)
        {
            let (lower, upper) = index_interval(exact);
            out_lower.push(lower);
            out_upper.push(upper);
        } else {
            out_lower.push(0.0);
            out_upper.push(uncertain_upper);
        }
    }

    let lower = ArrayD::from_shape_vec(IxDyn(&out_shape), out_lower)
        .map_err(|e| NyError::InvalidSpec(format!("{layer_name} lower reshape failed: {e}")))?;
    let upper = ArrayD::from_shape_vec(IxDyn(&out_shape), out_upper)
        .map_err(|e| NyError::InvalidSpec(format!("{layer_name} upper reshape failed: {e}")))?;
    BoundedTensor::new(lower, upper)
}

impl BoundPropagation for ArgMaxLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let axis = self.resolve_axis(input.shape().len())?;
        propagate_argext_ibp(
            input,
            axis,
            self.keepdims,
            true,
            self.select_last_index,
            "ArgMax",
        )
    }

    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "ArgMax CROWN backward is not implemented".to_string(),
        ))
    }
}

impl BoundPropagation for ArgMinLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let axis = self.resolve_axis(input.shape().len())?;
        propagate_argext_ibp(
            input,
            axis,
            self.keepdims,
            false,
            self.select_last_index,
            "ArgMin",
        )
    }

    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "ArgMin CROWN backward is not implemented".to_string(),
        ))
    }
}

impl BoundPropagation for ArgSortLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let axis = self.resolve_axis(input.shape().len())?;
        let axis_len = input.shape()[axis];
        if axis_len == 0 {
            return Err(NyError::InvalidSpec(format!(
                "ArgSort: axis {} has size 0",
                axis
            )));
        }
        let uncertain_upper = if axis_len <= 1 {
            0.0
        } else {
            index_upper_bound(axis_len - 1)
        };

        // ArgSort output shape equals the input shape; the sorted indices live
        // along `axis`. When a lane's ordering is certain (e.g. concrete inputs
        // or fully-separated intervals) we emit the exact permutation as a tight
        // degenerate interval; otherwise we fall back to the sound uniform range
        // [0, axis_len - 1] for that lane.
        let mut lower = ArrayD::from_elem(IxDyn(input.shape()), 0.0_f32);
        let mut upper = ArrayD::from_elem(IxDyn(input.shape()), uncertain_upper);

        for (((lower_lane_in, upper_lane_in), mut lower_lane_out), mut upper_lane_out) in input
            .lower()
            .lanes(Axis(axis))
            .into_iter()
            .zip(input.upper().lanes(Axis(axis)))
            .zip(lower.lanes_mut(Axis(axis)))
            .zip(upper.lanes_mut(Axis(axis)))
        {
            let lower_vec = lower_lane_in.iter().copied().collect::<Vec<_>>();
            let upper_vec = upper_lane_in.iter().copied().collect::<Vec<_>>();
            if let Some(exact) = exact_argsort_lane(&lower_vec, &upper_vec, self.descending) {
                for ((lower_slot, upper_slot), index) in lower_lane_out
                    .iter_mut()
                    .zip(upper_lane_out.iter_mut())
                    .zip(exact)
                {
                    let (lower, upper) = index_interval(index);
                    *lower_slot = lower;
                    *upper_slot = upper;
                }
            }
            // else: keep the pre-filled sound uniform [0, axis_len - 1] range.
        }

        BoundedTensor::new(lower, upper)
    }

    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "ArgSort CROWN backward is not implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    fn bounded_2d(lower: [[f32; 4]; 2], upper: [[f32; 4]; 2]) -> BoundedTensor {
        BoundedTensor::new(arr2(&lower).into_dyn(), arr2(&upper).into_dyn()).unwrap()
    }

    #[test]
    fn integer_index_enclosure_is_outward_above_f32_exact_range() {
        let index = 16_777_217_usize;
        let (lower, upper) = index_interval(index);
        assert!((lower as f64) <= index as f64);
        assert!((upper as f64) >= index as f64);
        assert!(
            lower < upper,
            "the nonrepresentable integer needs an interval"
        );
    }

    #[test]
    fn test_topk_values_ibp_selects_top_k_bounds() {
        let layer = TopkLayer::values(2, 1);
        let input = bounded_2d(
            [[1.0, 5.0, 3.0, 4.0], [0.0, -2.0, 7.0, 6.0]],
            [[2.0, 8.0, 4.0, 6.0], [1.0, -1.0, 9.0, 8.0]],
        );
        let out = layer.propagate_ibp(&input).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.lower(), &arr2(&[[5.0, 4.0], [7.0, 6.0]]).into_dyn());
        assert_eq!(out.upper(), &arr2(&[[8.0, 6.0], [9.0, 8.0]]).into_dyn());
    }

    #[test]
    fn topk_values_preserve_non_trailing_axis_layout() {
        let layer = TopkLayer::values(2, 0);
        let input = BoundedTensor::concrete(
            arr2(&[[10.0_f32, 1.0, 7.0], [5.0, 9.0, 2.0], [8.0, 3.0, 6.0]]).into_dyn(),
        )
        .unwrap();
        let output = layer.propagate_ibp(&input).unwrap();
        let expected = arr2(&[[10.0_f32, 9.0, 7.0], [8.0, 3.0, 6.0]]).into_dyn();
        assert_eq!(output.lower(), &expected);
        assert_eq!(output.upper(), &expected);
    }

    #[test]
    fn test_topk_indices_ibp_returns_axis_index_range() {
        let layer = TopkLayer::indices(2, 1);
        let input = bounded_2d(
            [[1.0, 5.0, 3.0, 4.0], [0.0, -2.0, 7.0, 6.0]],
            [[2.0, 8.0, 4.0, 6.0], [1.0, -1.0, 9.0, 8.0]],
        );
        let out = layer.propagate_ibp(&input).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert!(out.lower().iter().all(|&v| v == 0.0));
        assert!(out.upper().iter().all(|&v| v == 3.0));
    }

    #[test]
    fn test_argmax_ibp_detects_exact_winner() {
        let layer = ArgMaxLayer::new(1, false);
        let input = bounded_2d(
            [[1.0, 8.0, 3.0, 4.0], [0.0, -2.0, 7.0, 6.0]],
            [[2.0, 9.0, 4.0, 6.0], [1.0, 8.5, 9.0, 8.0]],
        );
        let out = layer.propagate_ibp(&input).unwrap();
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.lower(), &arr1(&[1.0, 0.0]).into_dyn());
        assert_eq!(out.upper(), &arr1(&[1.0, 3.0]).into_dyn());
    }

    #[test]
    fn test_argmin_keepdims_preserves_axis() {
        let layer = ArgMinLayer::new(1, true);
        let input = bounded_2d(
            [[-5.0, 8.0, 3.0, 4.0], [0.0, -2.0, 7.0, 6.0]],
            [[-4.0, 9.0, 4.0, 6.0], [1.0, -1.0, 9.0, 8.0]],
        );
        let out = layer.propagate_ibp(&input).unwrap();
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(out.lower(), &arr2(&[[0.0], [1.0]]).into_dyn());
        assert_eq!(out.upper(), &arr2(&[[0.0], [1.0]]).into_dyn());
    }

    #[test]
    fn arg_extrema_exact_ties_obey_onnx_first_and_last_index() {
        let input = BoundedTensor::concrete(arr1(&[4.0_f32, 4.0, 1.0]).into_dyn()).unwrap();

        let first_max = ArgMaxLayer::new(0, false).propagate_ibp(&input).unwrap();
        assert_eq!(first_max.lower()[[]], 0.0);
        assert_eq!(first_max.upper()[[]], 0.0);

        let last_max = ArgMaxLayer::new(0, false)
            .with_select_last_index(true)
            .propagate_ibp(&input)
            .unwrap();
        assert_eq!(last_max.lower()[[]], 1.0);
        assert_eq!(last_max.upper()[[]], 1.0);

        let min_input = BoundedTensor::concrete(arr1(&[-2.0_f32, 3.0, -2.0]).into_dyn()).unwrap();
        let first_min = ArgMinLayer::new(0, false)
            .propagate_ibp(&min_input)
            .unwrap();
        assert_eq!(first_min.lower()[[]], 0.0);
        let last_min = ArgMinLayer::new(0, false)
            .with_select_last_index(true)
            .propagate_ibp(&min_input)
            .unwrap();
        assert_eq!(last_min.lower()[[]], 2.0);
    }

    #[test]
    fn test_argsort_ibp_returns_uniform_index_range() {
        let layer = ArgSortLayer::new(1, true);
        let input = bounded_2d(
            [[1.0, 5.0, 3.0, 4.0], [0.0, -2.0, 7.0, 6.0]],
            [[2.0, 8.0, 4.0, 6.0], [1.0, -1.0, 9.0, 8.0]],
        );
        let out = layer.propagate_ibp(&input).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
        assert!(out.lower().iter().all(|&v| v == 0.0));
        assert!(out.upper().iter().all(|&v| v == 3.0));
    }

    #[test]
    fn test_topk_rejects_invalid_k() {
        let layer = TopkLayer::values(5, 1);
        let input = bounded_2d(
            [[1.0, 5.0, 3.0, 4.0], [0.0, -2.0, 7.0, 6.0]],
            [[2.0, 8.0, 4.0, 6.0], [1.0, -1.0, 9.0, 8.0]],
        );
        assert!(layer.propagate_ibp(&input).is_err());
    }
}
