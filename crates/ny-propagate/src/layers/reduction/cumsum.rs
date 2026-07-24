// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CumSum (cumulative sum / prefix sum) layer for bound propagation.
//!
//! CumSum is a linear operator: `y[i] = sum(x[0..=i])`. Its Jacobian is a
//! lower-triangular matrix of ones. Because it is monotone in each input,
//! IBP bounds are exact: `cumsum(lower) <= cumsum(x) <= cumsum(upper)`.
//!
//! CROWN backward: the transpose of the lower-triangular Jacobian is an
//! upper-triangular matrix of ones, equivalent to a suffix sum (reverse
//! cumsum) of the A-matrix coefficients along the reduction axis. This is
//! O(T) per row, NOT O(T^2) matmul.
//!
//! Needed for Kokoro TTS (T=24000). The O(n) AddLayer decomposition
//! creates T layers; a native CumSum is a single layer with O(T) backward.
//!
//! Part of #3919.

use ndarray::{Array2, Axis};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{BoundedTensor, RepairStrategy};
use std::borrow::Cow;

use super::super::common::{compute_strides, BoundPropagation};
use crate::LinearBounds;

/// Cumulative sum layer: computes prefix sum along a specified axis.
///
/// Forward: `y[..., i, ...] = sum(x[..., 0..=i, ...])` along `axis`.
///
/// Attributes:
/// - `axis`: the axis along which to compute cumsum.
/// - `exclusive`: if true, `y[i] = sum(x[0..i])` (excludes current element).
/// - `reverse`: if true, compute suffix sum instead of prefix sum.
#[derive(Debug, Clone)]
pub struct CumsumLayer {
    /// Axis along which to compute cumulative sum (ONNX convention, may be negative).
    pub axis: i64,
    /// If true, exclusive cumsum: y[i] = sum(x[0..i]), with y[0] = 0.
    pub exclusive: bool,
    /// If true, reverse (suffix sum): y[i] = sum(x[i..T]).
    pub reverse: bool,
}

impl CumsumLayer {
    /// Create a new cumulative sum layer.
    pub fn new(axis: i64, exclusive: bool, reverse: bool) -> Self {
        Self {
            axis,
            exclusive,
            reverse,
        }
    }

    /// Resolve negative axis to positive.
    pub(super) fn resolve_axis(&self, ndim: usize) -> Result<usize> {
        super::super::common::resolve_axis(self.axis, ndim, "CumSum")
    }

    /// Apply cumsum to a single ndarray along the resolved axis.
    ///
    /// Handles exclusive and reverse variants.
    fn apply_cumsum(
        &self,
        data: &ndarray::ArrayD<f32>,
        axis: usize,
        round_up: bool,
    ) -> Result<ndarray::ArrayD<f32>> {
        let axis_len = data.shape()[axis];
        let mut result = data.clone();

        if axis_len == 0 {
            return Ok(result);
        }

        // SOUND (#vnncomp-aw-soundness): cumsum is monotone in REAL arithmetic, but
        // each f32 `+=` below rounds to nearest, which can pull the lower partial sum
        // UP or the upper partial sum DOWN (inward) -> uncertified. Directed-round each
        // accumulated partial OUTWARD (lower -> -inf via next_down, upper -> +inf via
        // next_up), as AveragePool does for its window sum. Sound by induction even
        // under cancellation: next_down(a (+)f32 b) <= a + b (real), so the stored
        // partial never exceeds the exact partial. For a T-step prefix sum the
        // round-to-nearest drift reaches gamma_{T-1}*Sum|x| (unbounded relative to the
        // result under cancellation) — a single final 1-ULP rounding would NOT cover it.
        let round: fn(f32) -> f32 = if round_up {
            ny_tensor::next_up_f32
        } else {
            ny_tensor::next_down_f32
        };

        // For reverse: iterate from end to start
        // For forward: iterate from start to end
        // For exclusive: shift before accumulating
        if self.reverse {
            // Suffix sum: y[i] = sum(x[i..T])
            // Iterate from second-to-last to first, accumulating backward
            for i in (0..axis_len - 1).rev() {
                // result[..., i, ...] += result[..., i+1, ...]
                let next = result.index_axis(Axis(axis), i + 1).to_owned();
                let mut current = result.index_axis_mut(Axis(axis), i);
                current += &next;
                current.mapv_inplace(round);
            }
            if self.exclusive {
                // Exclusive reverse: y[i] = sum(x[i+1..T]), y[T-1] = 0
                // Shift forward: y[i] = y[i+1] (of the inclusive result)
                let mut shifted = ndarray::ArrayD::<f32>::zeros(data.raw_dim());
                for i in 0..axis_len - 1 {
                    let src = result.index_axis(Axis(axis), i + 1).to_owned();
                    let mut dst = shifted.index_axis_mut(Axis(axis), i);
                    dst.assign(&src);
                }
                // Last element is 0 (already zeroed)
                result = shifted;
            }
        } else {
            // Prefix sum: y[i] = sum(x[0..=i])
            // Iterate from second to last, accumulating forward
            for i in 1..axis_len {
                let prev = result.index_axis(Axis(axis), i - 1).to_owned();
                let mut current = result.index_axis_mut(Axis(axis), i);
                current += &prev;
                current.mapv_inplace(round);
            }
            if self.exclusive {
                // Exclusive forward: y[i] = sum(x[0..i]), y[0] = 0
                // Shift backward: y[i] = y[i-1] (of the inclusive result)
                let mut shifted = ndarray::ArrayD::<f32>::zeros(data.raw_dim());
                for i in 1..axis_len {
                    let src = result.index_axis(Axis(axis), i - 1).to_owned();
                    let mut dst = shifted.index_axis_mut(Axis(axis), i);
                    dst.assign(&src);
                }
                // First element is 0 (already zeroed)
                result = shifted;
            }
        }

        Ok(result)
    }
}

impl BoundPropagation for CumsumLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // CumSum is monotone in each input element (all Jacobian entries are non-negative).
        // Therefore IBP is exact: cumsum(lower) <= cumsum(x) <= cumsum(upper).
        let ndim = input.lower().ndim();
        let axis = self.resolve_axis(ndim)?;

        let lower = self.apply_cumsum(input.lower(), axis, false)?;
        let upper = self.apply_cumsum(input.upper(), axis, true)?;

        BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // CumSum backward needs pre-activation shape for axis resolution.
        Err(NyError::UnsupportedOp(
            "CumSum linear propagation requires pre-activation bounds - use propagate_crown_backward"
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        CumsumLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl CumsumLayer {
    /// CROWN backward propagation through CumSum layer.
    ///
    /// The Jacobian of cumsum is a lower-triangular matrix of ones:
    /// ```text
    /// J = [[1, 0, 0, ...],
    ///      [1, 1, 0, ...],
    ///      [1, 1, 1, ...],
    ///      ...           ]
    /// ```
    ///
    /// The CROWN backward pass computes `new_A = A @ J`, where J^T is upper-triangular.
    /// Multiplying by J^T is equivalent to computing the suffix sum (reverse cumsum)
    /// of the A-matrix rows along the cumsum axis.
    ///
    /// For exclusive cumsum, the Jacobian has the diagonal zeroed (strictly lower-triangular).
    /// For reverse cumsum, J is upper-triangular.
    ///
    /// Complexity: O(T * num_outputs) — a single scan per row, NOT O(T^2) matmul.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        let input_shape = pre_activation.shape();
        let ndim = input_shape.len();
        let axis = self.resolve_axis(ndim)?;
        let axis_len = input_shape[axis];

        let input_len = checked_shape_product(input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "CumSum CROWN: input shape product overflows: {:?}",
                input_shape
            ))
        })?;

        if input_shape.contains(&0) {
            return Err(NyError::InvalidSpec(
                "CumSum CROWN: input has zero-sized dimension".to_string(),
            ));
        }

        // For CumSum, input and output shapes are identical.
        if bounds.num_inputs() != input_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![input_len],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        let input_strides = compute_strides(input_shape)?;

        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_len));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_len));
        // Certified per-coefficient error on the suffix/prefix-sum coefficients
        // (#vnncomp-aw-soundness). The scan below accumulates the A-matrix in f32,
        // which ROUNDS at every step; Higham's bound |fl(Σx) - Σx| <= γ_n·Σ|x|
        // certifies that error, and any incoming coefficient error is re-summed
        // through the SAME transpose-Jacobian scan. Both are carried to concretize via
        // new_or_conservative_with_err so the verdict box widens OUTWARD. Without this
        // the f32 scan error is silently dropped and a stored bound can be TIGHTER
        // than the true real value — a false-proof.
        let mut new_lower_a_err = Array2::<f32>::zeros((num_outputs, input_len));
        let mut new_upper_a_err = Array2::<f32>::zeros((num_outputs, input_len));

        // For each output (row of A), compute the backward transformation.
        // We iterate over all positions in the non-cumsum dimensions, and for each
        // such "fiber" along the cumsum axis, apply a suffix sum (for forward cumsum)
        // or prefix sum (for reverse cumsum) to the A-matrix coefficients.
        //
        // This is equivalent to: new_A[row, input_idx] = sum over all output positions j
        // that depend on input_idx of A[row, j].

        // Build A-matrices first, then apply suffix/prefix sums per fiber.
        // Start by copying the original coefficients.
        new_lower_a.assign(bounds.lower_a());
        new_upper_a.assign(bounds.upper_a());

        // Now apply the transposed Jacobian via scan.
        // For forward (non-reverse) cumsum: J^T is upper-triangular, so we need suffix sum.
        // For reverse cumsum: J^T is lower-triangular, so we need prefix sum.
        //
        // We iterate over "fibers" — lines along the cumsum axis — and apply the scan.

        // Compute the stride along the cumsum axis
        let axis_stride = input_strides[axis];

        // Number of fibers = total elements / axis_len
        let num_fibers = input_len / axis_len;

        // Iterate over each fiber. A fiber is identified by fixing all coordinates
        // except the cumsum axis. The flat index of position k along the fiber is:
        // fiber_base + k * axis_stride.
        //
        // To enumerate all fibers, we iterate over a "reduced" index space.
        let mut fiber_base_indices = Vec::with_capacity(num_fibers);
        {
            // Generate all flat indices where the cumsum-axis coordinate is 0
            let mut coords = vec![0usize; ndim];
            for flat_idx in 0..input_len {
                // Decompose flat_idx into coordinates
                let mut remaining = flat_idx;
                for d in 0..ndim {
                    coords[d] = remaining / input_strides[d];
                    remaining %= input_strides[d];
                }
                if coords[axis] == 0 {
                    fiber_base_indices.push(flat_idx);
                }
            }
        }

        // Higham growth factor γ_n for an f32-accumulated sum of up to `axis_len`
        // terms, plus the propagated incoming certified coefficient error (if any).
        let gamma = crate::layers::linear::crown_single_gamma_n_f32(axis_len);
        let in_lower_err = bounds.lower_a_err();
        let in_upper_err = bounds.upper_a_err();

        for row in 0..num_outputs {
            for &base in &fiber_base_indices {
                // Running f64 accumulators for the certified error: the absolute sum S
                // (scaled by γ_n) and the propagated incoming coefficient error, both
                // over EXACTLY the terms folded into the stored coefficient.
                let mut abs_lower = 0.0f64;
                let mut abs_upper = 0.0f64;
                let mut prop_lower = 0.0f64;
                let mut prop_upper = 0.0f64;
                if self.reverse {
                    if self.exclusive {
                        // Exclusive reverse: stored coeff EXCLUDES the current term, so
                        // its error is taken BEFORE folding this term in.
                        let mut acc_lower = 0.0f32;
                        let mut acc_upper = 0.0f32;
                        for k in 0..axis_len {
                            let idx = base + k * axis_stride;
                            let orig_lower = new_lower_a[[row, idx]];
                            let orig_upper = new_upper_a[[row, idx]];
                            new_lower_a[[row, idx]] = acc_lower;
                            new_upper_a[[row, idx]] = acc_upper;
                            new_lower_a_err[[row, idx]] =
                                ny_tensor::next_up_f32((gamma * abs_lower + prop_lower) as f32);
                            new_upper_a_err[[row, idx]] =
                                ny_tensor::next_up_f32((gamma * abs_upper + prop_upper) as f32);
                            acc_lower += orig_lower;
                            acc_upper += orig_upper;
                            abs_lower += (orig_lower as f64).abs();
                            abs_upper += (orig_upper as f64).abs();
                            prop_lower += in_lower_err.map_or(0.0, |e| e[[row, idx]] as f64);
                            prop_upper += in_upper_err.map_or(0.0, |e| e[[row, idx]] as f64);
                        }
                    } else {
                        // Inclusive reverse: stored coeff INCLUDES the current term, so
                        // its error is taken AFTER folding this term in.
                        let mut acc_lower = 0.0f32;
                        let mut acc_upper = 0.0f32;
                        for k in 0..axis_len {
                            let idx = base + k * axis_stride;
                            let orig_lower = new_lower_a[[row, idx]];
                            let orig_upper = new_upper_a[[row, idx]];
                            acc_lower += orig_lower;
                            acc_upper += orig_upper;
                            abs_lower += (orig_lower as f64).abs();
                            abs_upper += (orig_upper as f64).abs();
                            prop_lower += in_lower_err.map_or(0.0, |e| e[[row, idx]] as f64);
                            prop_upper += in_upper_err.map_or(0.0, |e| e[[row, idx]] as f64);
                            new_lower_a[[row, idx]] = acc_lower;
                            new_upper_a[[row, idx]] = acc_upper;
                            new_lower_a_err[[row, idx]] =
                                ny_tensor::next_up_f32((gamma * abs_lower + prop_lower) as f32);
                            new_upper_a_err[[row, idx]] =
                                ny_tensor::next_up_f32((gamma * abs_upper + prop_upper) as f32);
                        }
                    }
                } else if self.exclusive {
                    // Exclusive forward: stored coeff EXCLUDES the current term.
                    let mut acc_lower = 0.0f32;
                    let mut acc_upper = 0.0f32;
                    for k in (0..axis_len).rev() {
                        let idx = base + k * axis_stride;
                        let orig_lower = new_lower_a[[row, idx]];
                        let orig_upper = new_upper_a[[row, idx]];
                        new_lower_a[[row, idx]] = acc_lower;
                        new_upper_a[[row, idx]] = acc_upper;
                        new_lower_a_err[[row, idx]] =
                            ny_tensor::next_up_f32((gamma * abs_lower + prop_lower) as f32);
                        new_upper_a_err[[row, idx]] =
                            ny_tensor::next_up_f32((gamma * abs_upper + prop_upper) as f32);
                        acc_lower += orig_lower;
                        acc_upper += orig_upper;
                        abs_lower += (orig_lower as f64).abs();
                        abs_upper += (orig_upper as f64).abs();
                        prop_lower += in_lower_err.map_or(0.0, |e| e[[row, idx]] as f64);
                        prop_upper += in_upper_err.map_or(0.0, |e| e[[row, idx]] as f64);
                    }
                } else {
                    // Inclusive forward: stored coeff INCLUDES the current term.
                    let mut acc_lower = 0.0f32;
                    let mut acc_upper = 0.0f32;
                    for k in (0..axis_len).rev() {
                        let idx = base + k * axis_stride;
                        let orig_lower = new_lower_a[[row, idx]];
                        let orig_upper = new_upper_a[[row, idx]];
                        acc_lower += orig_lower;
                        acc_upper += orig_upper;
                        abs_lower += (orig_lower as f64).abs();
                        abs_upper += (orig_upper as f64).abs();
                        prop_lower += in_lower_err.map_or(0.0, |e| e[[row, idx]] as f64);
                        prop_upper += in_upper_err.map_or(0.0, |e| e[[row, idx]] as f64);
                        new_lower_a[[row, idx]] = acc_lower;
                        new_upper_a[[row, idx]] = acc_upper;
                        new_lower_a_err[[row, idx]] =
                            ny_tensor::next_up_f32((gamma * abs_lower + prop_lower) as f32);
                        new_upper_a_err[[row, idx]] =
                            ny_tensor::next_up_f32((gamma * abs_upper + prop_upper) as f32);
                    }
                }
            }
        }

        // Bias unchanged — CumSum has no additive constant. The certified f32 scan
        // error (and any re-summed incoming error) is carried so concretize widens
        // the box OUTWARD instead of trusting the rounded coefficients.
        LinearBounds::new_or_conservative_with_err(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
            new_lower_a_err,
            new_upper_a_err,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    fn bounded_from_1d(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        let l = ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap();
        BoundedTensor::new(l, u).unwrap()
    }

    /// The sound cumsum IBP forward directed-rounds every accumulation step OUTWARD
    /// (#vnncomp-aw-soundness), so the lower bound sits at or just below the exact
    /// prefix/suffix sum and the upper bound at or just above it (≤ a few ULPs of
    /// widening per step). Assert that sound enclosure rather than bit-exact equality.
    fn assert_sound_cumsum(
        actual_lower: &[f32],
        actual_upper: &[f32],
        exact_l: &[f32],
        exact_u: &[f32],
    ) {
        let tol = 1e-4_f32;
        for (i, (&a, &e)) in actual_lower.iter().zip(exact_l).enumerate() {
            assert!(
                a <= e + f32::EPSILON,
                "lower[{i}]={a} must be <= exact {e} (sound)"
            );
            assert!(a >= e - tol, "lower[{i}]={a} too far below exact {e}");
        }
        for (i, (&a, &e)) in actual_upper.iter().zip(exact_u).enumerate() {
            assert!(
                a >= e - f32::EPSILON,
                "upper[{i}]={a} must be >= exact {e} (sound)"
            );
            assert!(a <= e + tol, "upper[{i}]={a} too far above exact {e}");
        }
    }

    #[test]
    fn test_cumsum_ibp_forward_inclusive() {
        let layer = CumsumLayer::new(0, false, false);
        let input = bounded_from_1d(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let output = layer.propagate_ibp(&input).unwrap();

        // cumsum([1,2,3]) = [1,3,6], cumsum([4,5,6]) = [4,9,15]
        assert_sound_cumsum(
            output.lower().as_slice().unwrap(),
            output.upper().as_slice().unwrap(),
            &[1.0, 3.0, 6.0],
            &[4.0, 9.0, 15.0],
        );
    }

    #[test]
    fn test_cumsum_ibp_forward_exclusive() {
        let layer = CumsumLayer::new(0, true, false);
        let input = bounded_from_1d(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let output = layer.propagate_ibp(&input).unwrap();

        // exclusive cumsum([1,2,3]) = [0, 1, 3], exclusive cumsum([4,5,6]) = [0, 4, 9]
        assert_sound_cumsum(
            output.lower().as_slice().unwrap(),
            output.upper().as_slice().unwrap(),
            &[0.0, 1.0, 3.0],
            &[0.0, 4.0, 9.0],
        );
    }

    #[test]
    fn test_cumsum_ibp_reverse_inclusive() {
        let layer = CumsumLayer::new(0, false, true);
        let input = bounded_from_1d(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let output = layer.propagate_ibp(&input).unwrap();

        // reverse cumsum([1,2,3]) = [6, 5, 3], reverse cumsum([4,5,6]) = [15, 11, 6]
        assert_sound_cumsum(
            output.lower().as_slice().unwrap(),
            output.upper().as_slice().unwrap(),
            &[6.0, 5.0, 3.0],
            &[15.0, 11.0, 6.0],
        );
    }

    #[test]
    fn test_cumsum_ibp_reverse_exclusive() {
        let layer = CumsumLayer::new(0, true, true);
        let input = bounded_from_1d(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        let output = layer.propagate_ibp(&input).unwrap();

        // exclusive reverse cumsum([1,2,3]) = [5, 3, 0]
        // exclusive reverse cumsum([4,5,6]) = [11, 6, 0]
        assert_sound_cumsum(
            output.lower().as_slice().unwrap(),
            output.upper().as_slice().unwrap(),
            &[5.0, 3.0, 0.0],
            &[11.0, 6.0, 0.0],
        );
    }

    #[test]
    fn test_cumsum_crown_backward_forward_inclusive() {
        // CROWN backward through forward inclusive cumsum.
        // J = [[1,0,0],[1,1,0],[1,1,1]] (lower-triangular)
        // new_A = A @ J, which is the suffix sum of A's columns.
        let layer = CumsumLayer::new(0, false, false);
        let input = bounded_from_1d(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);

        // Identity A-matrix: A = I(3x3)
        let lower_a = Array2::<f32>::eye(3);
        let upper_a = Array2::<f32>::eye(3);
        let lower_b = Array1::<f32>::zeros(3);
        let upper_b = Array1::<f32>::zeros(3);
        let bounds = LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b).unwrap();

        let result = layer.propagate_linear_with_bounds(&bounds, &input).unwrap();

        // new_A[row, col] = sum_{j>=col} A[row, j] (suffix sum of each row)
        // A = I: row k gets suffix sum [1,1,...,1,0,...,0] with ones from 0..=k
        let expected = ndarray::array![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]];
        assert_eq!(result.lower_a(), &expected);
        assert_eq!(result.upper_a(), &expected);
    }

    #[test]
    fn test_cumsum_crown_backward_reverse_inclusive() {
        let layer = CumsumLayer::new(0, false, true);
        let input = bounded_from_1d(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);

        let lower_a = Array2::<f32>::eye(3);
        let upper_a = Array2::<f32>::eye(3);
        let lower_b = Array1::<f32>::zeros(3);
        let upper_b = Array1::<f32>::zeros(3);
        let bounds = LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b).unwrap();

        let result = layer.propagate_linear_with_bounds(&bounds, &input).unwrap();

        // Reverse inclusive: J[j, col] = 1 if col >= j (upper-triangular)
        // new_A[row, col] = sum_{j<=col} A[row, j] = prefix sum
        //
        // Row 0: A = [1,0,0], prefix sums: [1, 1, 1]
        // Row 1: A = [0,1,0], prefix sums: [0, 1, 1]
        // Row 2: A = [0,0,1], prefix sums: [0, 0, 1]
        let expected = ndarray::array![[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]];
        assert_eq!(result.lower_a(), &expected);
        assert_eq!(result.upper_a(), &expected);
    }

    #[test]
    fn test_cumsum_soundness_ibp_contains_concrete() {
        // Verify that IBP bounds always contain the concrete cumsum output.
        use ndarray::ArrayD;

        let layer = CumsumLayer::new(-1, false, false);

        // 2D input: [3, 4] — cumsum along last axis
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[3, 4]),
            vec![
                -1.0, -2.0, 0.0, 1.0, 0.5, -0.5, 1.0, -1.0, -3.0, 2.0, -1.0, 0.5,
            ],
        )
        .unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[3, 4]),
            vec![1.0, 0.0, 2.0, 3.0, 2.0, 1.0, 3.0, 1.0, -1.0, 4.0, 1.0, 2.0],
        )
        .unwrap();

        let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
        let output = layer.propagate_ibp(&input).unwrap();

        // Test with a concrete point in the middle of the bounds
        let concrete = (&lower + &upper) / 2.0;
        let concrete_bt = BoundedTensor::concrete(concrete).unwrap();
        let concrete_output = layer.propagate_ibp(&concrete_bt).unwrap();

        // Verify containment: lower <= concrete <= upper
        for ((&l, &c), &u) in output
            .lower()
            .iter()
            .zip(concrete_output.lower().iter())
            .zip(output.upper().iter())
        {
            assert!(
                l <= c + 1e-6,
                "Lower bound {} > concrete {} (violation: {})",
                l,
                c,
                l - c
            );
            assert!(
                c <= u + 1e-6,
                "Concrete {} > upper bound {} (violation: {})",
                c,
                u,
                c - u
            );
        }
    }

    /// Fail-before / pass-after repro for #vnncomp-aw-soundness (cumsum IBP forward).
    /// A long prefix sum where f32 round-to-nearest swallows the small additions: the
    /// upper fiber starts at 2^24 (ULP = 2) then adds 1.0 a hundred times. A plain
    /// round-to-nearest scan loses every +1 (stays 2^24), giving a certified UPPER
    /// bound BELOW the true reachable max (2^24 + 100) — a false proof. The directed
    /// (next_up) rounding per step keeps the upper bound at/above the true value.
    #[test]
    fn test_cumsum_ibp_upper_encloses_under_f32_cancellation() {
        let n = 101usize;
        let p = (1u32 << 24) as f32; // 2^24, ULP = 2
        let mut up = vec![1.0f32; n];
        up[0] = p;
        let lo = vec![0.0f32; n];
        let layer = CumsumLayer::new(0, false, false); // forward inclusive
        let output = layer.propagate_ibp(&bounded_from_1d(&lo, &up)).unwrap();
        let upper = output.upper().as_slice().unwrap();
        // True max prefix sum at the last position = 2^24 + (n-1) added ones.
        let true_last = p as f64 + (n as f64 - 1.0);
        assert!(
            upper[n - 1] as f64 >= true_last,
            "cumsum upper[{}]={} must be >= true reachable {} (round-to-nearest would give ~{})",
            n - 1,
            upper[n - 1],
            true_last,
            p
        );
    }
}
