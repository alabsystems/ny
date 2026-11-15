// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array2, ArrayD, Axis};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};
use std::borrow::Cow;
use tracing::debug;

use super::super::common::BoundPropagation;
use crate::bounds::nan_propagating_max;
use crate::LinearBounds;

mod batched;

/// LogSumExp layer: y = log(sum(exp(x))) over specified axes.
///
/// This is commonly used in loss fusion paths (log-softmax and log-sum-exp losses).
#[derive(Debug, Clone)]
pub struct LogSumExpLayer {
    /// Axes to reduce over (e.g., [-1] for last axis).
    pub axes: Vec<i64>,
    /// Whether to keep reduced dimensions (size 1) in output.
    pub keepdims: bool,
}

impl LogSumExpLayer {
    /// Create a new LogSumExp layer.
    pub fn new(axes: Vec<i64>, keepdims: bool) -> Self {
        Self { axes, keepdims }
    }

    fn resolve_axes(&self, ndim: usize) -> Result<Vec<usize>> {
        if ndim == 0 {
            return Ok(Vec::new());
        }
        if self.axes.is_empty() {
            return Ok((0..ndim).collect());
        }
        self.axes
            .iter()
            .map(|&axis| crate::layers::common::resolve_axis(axis, ndim, "LogSumExp"))
            .collect()
    }

    fn logsumexp_reduce(
        values: ArrayD<f32>,
        axis: usize,
        keepdims: bool,
        round_f32: fn(f32) -> f32,
    ) -> Result<ArrayD<f32>> {
        // NaN-propagating fold: NaN in values must propagate through the max.
        let max_vals = values.fold_axis(Axis(axis), f32::NEG_INFINITY, |&acc, &x| {
            nan_propagating_max(acc, x)
        });
        let max_expanded = max_vals.clone().insert_axis(Axis(axis));
        // f64 accumulation for exp and sum_axis to prevent precision loss
        // for large reduction axes (seq_len=512+).
        //
        // Limit-correct shift at ±inf (dc9da0fde class): a plain `x - m`
        // yields inf - inf = NaN exactly on the element that ATTAINS an
        // infinite max (and on every element of an all-(-inf) slice, where
        // m = -inf). Pinning `x == m` to a shift of 0 is exact for finite
        // values (x - x = 0) and gives each max-attaining element its true
        // limit contribution exp(0) = 1, so sum >= 1 and:
        //   m = +inf  →  +inf + ln(sum)  =  +inf   (LSE(.., +inf, ..) = +inf)
        //   m = -inf  →  -inf + ln(n)    =  -inf   (LSE(-inf, .., -inf) = -inf)
        //   m finite, x = -inf → shift -inf, exp 0 (exact, unchanged).
        // NaN still propagates: NaN == m is false, NaN - m = NaN.
        let max_broadcast = max_expanded.broadcast(values.raw_dim()).ok_or_else(|| {
            NyError::InternalError(
                "LogSumExp: max_expanded must broadcast to the input shape".to_string(),
            )
        })?;
        let exp_shifted_f64 =
            ndarray::Zip::from(&values)
                .and(&max_broadcast)
                .map_collect(|&x, &m| {
                    if x == m {
                        1.0_f64 // exp(0): the element attaining the max, exact at ±inf too
                    } else {
                        f64::from(x - m).exp()
                    }
                });
        let sum_exp_f64 = exp_shifted_f64.sum_axis(Axis(axis));
        let logsumexp = max_vals.mapv(|x| x as f64) + sum_exp_f64.mapv(|x| x.ln());
        // Directed rounding on f64→f32 cast: next_down for lower bounds,
        // next_up for upper bounds.
        let logsumexp_f32 = logsumexp.mapv(|x| round_f32(x as f32));
        if keepdims {
            Ok(logsumexp_f32.insert_axis(Axis(axis)).into_dyn())
        } else {
            Ok(logsumexp_f32.into_dyn())
        }
    }
}

impl BoundPropagation for LogSumExpLayer {
    /// IBP for LogSumExp: y = log(sum(exp(x))) over specified axes.
    ///
    /// LogSumExp is monotonically increasing in each argument, so IBP
    /// applies logsumexp independently to lower and upper bounds — including
    /// at the extended-real limits, which `logsumexp_reduce` maps exactly:
    /// a +inf element makes the reduction +inf, an all-(-inf) slice reduces
    /// to -inf, and a -inf element among finite ones contributes exp(-inf) =
    /// 0. Only NaN input bounds fail closed.
    ///
    /// Category B per domain validation policy:
    /// NaN carries no bound information and must be surfaced, not masked; a
    /// ±inf endpoint is the conservative-bounds convention for "unbounded in
    /// that direction" (deep-stack IBP overflow saturation, dc9da0fde class)
    /// and propagates as the corresponding exact limit.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let ndim = input.shape().len();
        if ndim == 0 {
            return Ok(input.clone());
        }

        let axes = self.resolve_axes(ndim)?;

        // Guard: NaN input bounds fail closed (Category B strict mode). ±Inf
        // is admitted and handled limit-exactly inside `logsumexp_reduce`.
        if input.lower().iter().any(|&v| v.is_nan()) || input.upper().iter().any(|&v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "LogSumExp IBP: NaN input bounds".to_string(),
            ));
        }

        let mut lower = input.lower().clone();
        let mut upper = input.upper().clone();

        let mut sorted_axes = axes;
        sorted_axes.sort_by(|a, b| b.cmp(a));

        for &axis in &sorted_axes {
            lower = Self::logsumexp_reduce(lower, axis, self.keepdims, next_down_f32)?;
            upper = Self::logsumexp_reduce(upper, axis, self.keepdims, next_up_f32)?;
        }

        // Conservative construction: ±Inf endpoints are preserved — both the
        // exact ±inf limits from saturated inputs and Inf produced from large
        // FINITE inputs via exp() overflow are sound conservative bounds
        //NaN cannot arise from the NaN-guarded input, and
        // would still be repaired to the conservative direction here rather
        // than silently kept.
        BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "LogSumExp is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
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
        LogSumExpLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl LogSumExpLayer {
    /// Expose resolve_axes for testing.
    #[cfg(test)]
    pub(crate) fn resolve_axes_pub(&self, ndim: usize) -> Result<Vec<usize>> {
        self.resolve_axes(ndim)
    }

    /// CROWN backward propagation with pre-activation bounds.
    ///
    /// Uses IBP-derived constant bounds for soundness. Since LogSumExp has no
    /// closed-form linear relaxation, the CROWN path concretizes against
    /// IBP output (zero-slope linear bounds).
    ///
    /// Returns `NumericalInstability` if pre-activation bounds are non-finite.
    /// Returns conservative `[-inf, +inf]` constant bounds when asymmetric
    /// incoming coefficients cause bound inversion during concretization.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("LogSumExp layer CROWN backward propagation with pre-activation bounds");

        // Guard: reject non-finite pre-activation bounds (Category B strict mode).
        if pre_activation.lower().iter().any(|&v| !v.is_finite())
            || pre_activation.upper().iter().any(|&v| !v.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "LogSumExp CROWN: non-finite pre-activation bounds".to_string(),
            ));
        }

        let output_bounds = self.propagate_ibp(pre_activation)?;

        let output_flat = output_bounds.flatten();
        let output_len = output_flat.len();
        if bounds.num_inputs() != output_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![bounds.num_inputs()],
                got: vec![output_len],
            });
        }

        let concretized = bounds.concretize_sound(&output_flat); // #2236: directed rounding for soundness
        let lower_shape = concretized.lower().shape().to_vec();
        let lower = concretized
            .lower()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![bounds.num_outputs()],
                got: lower_shape,
            })?;
        let upper_shape = concretized.upper().shape().to_vec();
        let upper = concretized
            .upper()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| NyError::ShapeMismatch {
                expected: vec![bounds.num_outputs()],
                got: upper_shape,
            })?;

        let input_len: usize = checked_shape_product(pre_activation.shape()).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "LogSumExp CROWN: shape product overflows usize: {:?}",
                pre_activation.shape(),
            ))
        })?;
        LinearBounds::new_or_conservative(
            Array2::zeros((bounds.num_outputs(), input_len)),
            lower,
            Array2::zeros((bounds.num_outputs(), input_len)),
            upper,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn make_bt(lower: &[f32], upper: &[f32], shape: &[usize]) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    // Reference: log(sum(exp(x))) for a slice
    fn reference_logsumexp(xs: &[f32]) -> f32 {
        let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = xs.iter().map(|&x| (x - max).exp()).sum();
        max + sum_exp.ln()
    }

    // ── resolve_axes tests ────────────────────────────────────────────

    #[test]
    fn test_resolve_axes_negative() {
        let layer = LogSumExpLayer::new(vec![-1], false);
        let axes = layer.resolve_axes_pub(3).unwrap();
        assert_eq!(axes, vec![2]);
    }

    #[test]
    fn test_resolve_axes_positive() {
        let layer = LogSumExpLayer::new(vec![0, 1], false);
        let axes = layer.resolve_axes_pub(3).unwrap();
        assert_eq!(axes, vec![0, 1]);
    }

    #[test]
    fn test_resolve_axes_empty_means_all() {
        let layer = LogSumExpLayer::new(vec![], false);
        let axes = layer.resolve_axes_pub(3).unwrap();
        assert_eq!(axes, vec![0, 1, 2]);
    }

    #[test]
    fn test_resolve_axes_out_of_bounds() {
        let layer = LogSumExpLayer::new(vec![5], false);
        let err = layer
            .resolve_axes_pub(3)
            .expect_err("axis 5 out of bounds for 3 dims");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    // ── IBP tests ──────────────────────────────────────────────────────

    #[test]
    fn test_ibp_1d_last_axis() {
        let layer = LogSumExpLayer::new(vec![-1], false);
        // Input: 1D tensor [3], bounds [1,2,3] to [2,3,4]
        let input = make_bt(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0], &[3]);
        let result = layer.propagate_ibp(&input).unwrap();
        // Output should be scalar (reduce over axis 0 for 1D)
        let lower_expected = reference_logsumexp(&[1.0, 2.0, 3.0]);
        let upper_expected = reference_logsumexp(&[2.0, 3.0, 4.0]);
        assert!(
            (result.lower().iter().next().unwrap() - lower_expected).abs() < 1e-4,
            "lower: got {}, expected {}",
            result.lower().iter().next().unwrap(),
            lower_expected
        );
        assert!(
            (result.upper().iter().next().unwrap() - upper_expected).abs() < 1e-4,
            "upper: got {}, expected {}",
            result.upper().iter().next().unwrap(),
            upper_expected
        );
    }

    #[test]
    fn test_ibp_2d_last_axis() {
        let layer = LogSumExpLayer::new(vec![-1], false);
        // Shape [2, 3]
        let input = make_bt(
            &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            &[2.0, 3.0, 4.0, 1.0, 1.0, 1.0],
            &[2, 3],
        );
        let result = layer.propagate_ibp(&input).unwrap();
        // Output shape [2] (axis -1 reduced)
        assert_eq!(result.shape(), &[2]);
        let lower0 = reference_logsumexp(&[1.0, 2.0, 3.0]);
        assert!((result.lower()[0] - lower0).abs() < 1e-4);
    }

    #[test]
    fn test_ibp_keepdims() {
        let layer = LogSumExpLayer::new(vec![-1], true);
        let input = make_bt(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0], &[3]);
        let result = layer.propagate_ibp(&input).unwrap();
        // With keepdims, output should be [1]
        assert_eq!(result.shape(), &[1]);
    }

    #[test]
    fn test_ibp_identical_bounds() {
        // When lower == upper, output bounds should also be equal (point interval)
        let layer = LogSumExpLayer::new(vec![-1], false);
        let input = make_bt(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &[3]);
        let result = layer.propagate_ibp(&input).unwrap();
        let expected = reference_logsumexp(&[1.0, 2.0, 3.0]);
        assert!((result.lower().iter().next().unwrap() - expected).abs() < 1e-4);
        assert!((result.upper().iter().next().unwrap() - expected).abs() < 1e-4);
    }

    #[test]
    fn test_ibp_nan_error() {
        let layer = LogSumExpLayer::new(vec![-1], false);
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        )
        .unwrap();
        let err = layer.propagate_ibp(&input).expect_err("NaN fails closed");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    /// A -inf endpoint among finite ones contributes exp(-inf) = 0 exactly
    /// (dc9da0fde saturation class — supersedes the old non-finite refusal).
    #[test]
    fn test_ibp_neg_inf_among_finite_is_exact() {
        let layer = LogSumExpLayer::new(vec![-1], false);
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        )
        .unwrap();
        let result = layer
            .propagate_ibp(&input)
            .expect("-inf among finite values must propagate");
        // Lower: LSE(-inf, 1) = ln(e^1) = 1 exactly (up to directed rounding).
        assert!(
            (result.lower().iter().next().unwrap() - 1.0).abs() < 1e-5,
            "LSE(-inf, 1) = 1, got {}",
            result.lower().iter().next().unwrap()
        );
    }

    /// Regression test, updated for the dc9da0fde saturation class:
    /// all-NEG_INFINITY reduces to the exact -inf limit — sum(exp) = 0, so
    /// LSE = -inf — and must NOT produce NaN (the original failure
    /// was -inf - -inf = NaN inside the max-shift).
    #[test]
    fn test_ibp_all_neg_infinity_no_nan_2713() {
        let layer = LogSumExpLayer::new(vec![-1], false);
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY; 3]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY; 3]).unwrap(),
        )
        .unwrap();
        let result = layer
            .propagate_ibp(&input)
            .expect("all -inf must propagate as the exact -inf limit");
        for v in result.lower().iter().chain(result.upper().iter()) {
            assert!(!v.is_nan(), "all -inf reduction must not produce NaN");
            assert!(
                v.is_infinite() && v.is_sign_negative(),
                "LSE(-inf, ..., -inf) = -inf, got {v}"
            );
        }
    }

    /// A +inf endpoint dominates the reduction: LSE(.., +inf, ..) = +inf,
    /// with no NaN from the inf - inf shift on the max-attaining element.
    #[test]
    fn test_ibp_pos_inf_dominates_reduction() {
        let layer = LogSumExpLayer::new(vec![-1], false);
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, f32::NEG_INFINITY, 0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, f32::INFINITY, 1.0]).unwrap(),
        )
        .unwrap();
        let result = layer.propagate_ibp(&input).expect("+inf must propagate");
        let upper = *result.upper().iter().next().unwrap();
        assert!(
            upper.is_infinite() && upper.is_sign_positive(),
            "LSE with a +inf element is +inf, got {upper}"
        );
        let lower = *result.lower().iter().next().unwrap();
        // Lower: LSE(0, -inf, 0) = ln(2) exactly (the -inf term drops out).
        assert!(
            (lower - (2.0f32).ln()).abs() < 1e-5,
            "LSE(0, -inf, 0) = ln 2, got {lower}"
        );
        assert!(!lower.is_nan() && !upper.is_nan());
    }

    #[test]
    fn test_ibp_soundness_monotonicity() {
        // LogSumExp is monotonically increasing: increasing any input increases output.
        // So lower(LSE(l)) <= LSE(x) <= upper(LSE(u)) for all l <= x <= u.
        let layer = LogSumExpLayer::new(vec![-1], false);
        let lower_vals = [0.0, 1.0, -1.0];
        let upper_vals = [2.0, 3.0, 1.0];
        let input = make_bt(&lower_vals, &upper_vals, &[3]);
        let result = layer.propagate_ibp(&input).unwrap();
        let bound_lower = *result.lower().iter().next().unwrap();
        let bound_upper = *result.upper().iter().next().unwrap();

        // Sample points and verify containment
        for k in 0..=10 {
            let t = k as f32 / 10.0;
            let xs: Vec<f32> = (0..3)
                .map(|i| lower_vals[i] + (upper_vals[i] - lower_vals[i]) * t)
                .collect();
            let lse = reference_logsumexp(&xs);
            assert!(
                lse >= bound_lower - 1e-4,
                "lse={} < lower={}",
                lse,
                bound_lower
            );
            assert!(
                lse <= bound_upper + 1e-4,
                "lse={} > upper={}",
                lse,
                bound_upper
            );
        }
    }

    // ── CROWN backward tests ──────────────────────────────────────────

    #[test]
    fn test_crown_identity_constant_bounds() {
        // CROWN backward for LogSumExp returns zero-slope bounds (IBP-derived constants)
        let layer = LogSumExpLayer::new(vec![-1], false);
        let pre = make_bt(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0], &[3]);
        let ibp_result = layer.propagate_ibp(&pre).unwrap();

        let bounds = LinearBounds::identity(1); // output is scalar
        let crown_result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        // Zero slopes
        assert!(crown_result.lower_a.iter().all(|&v| v.abs() < 1e-7));
        assert!(crown_result.upper_a.iter().all(|&v| v.abs() < 1e-7));
        // Bias should match IBP output
        assert!((crown_result.lower_b[0] - ibp_result.lower().iter().next().unwrap()).abs() < 1e-4);
        assert!((crown_result.upper_b[0] - ibp_result.upper().iter().next().unwrap()).abs() < 1e-4);
    }

    #[test]
    fn test_crown_non_finite_preact_error() {
        let layer = LogSumExpLayer::new(vec![-1], false);
        let pre = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        )
        .unwrap();
        let bounds = LinearBounds::identity(1);
        let err = layer
            .propagate_linear_with_bounds(&bounds, &pre)
            .expect_err("non-finite preact");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    /// Regression test: asymmetric incoming coefficients must not
    /// cause NumericalInstability error. The concretize_sound fallback to
    /// [-inf, +inf] is a sound conservative result.
    #[test]
    fn test_crown_asymmetric_coefficients_returns_conservative_bounds_2236() {
        use ndarray::Array1;

        // Reproduce the proptest regression seed:
        // l0=0, d0=0.01, l1=0, d1=0.01, l2=0, d2=0.01,
        // cl0=0, cl1=-1.563345, cu0=0, cu1=-1.7472557
        let layer = LogSumExpLayer::new(vec![-1], false);

        // Two rows of 3 elements each, with narrow intervals near zero.
        let pre = make_bt(
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            &[2, 3],
        );

        // Asymmetric incoming: lower and upper use different coefficients.
        // This is valid CROWN state — different linear combinations for
        // bounding from below vs above.
        let bounds = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![0.0, -1.563345]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![0.0, -1.7472557]).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();

        // Before the fix, this returned Err(NumericalInstability).
        // After the fix, it returns Ok with conservative constant bounds.
        let result = layer
            .propagate_linear_with_bounds(&bounds, &pre)
            .expect("asymmetric coefficients should succeed, not return NumericalInstability");

        // A matrices must be zero (concretized to constant bounds).
        assert!(result.lower_a.iter().all(|&v| v == 0.0));
        assert!(result.upper_a.iter().all(|&v| v == 0.0));

        // Bounds must be sound: lower_b <= upper_b (or both infinite).
        // With these asymmetric coefficients, concretize_sound may return [-inf, +inf].
        for i in 0..result.lower_b.len() {
            assert!(
                result.lower_b[i] <= result.upper_b[i],
                "inverted bounds at [{i}]: lower={} > upper={}",
                result.lower_b[i],
                result.upper_b[i]
            );
        }
    }

    #[test]
    fn test_propagate_linear_requires_preact() {
        let layer = LogSumExpLayer::new(vec![-1], false);
        assert!(layer.requires_pre_activation_bounds());
        let bounds = LinearBounds::identity(3);
        let err = layer
            .propagate_linear(&bounds)
            .expect_err("requires preact");
        assert!(matches!(err, NyError::UnsupportedOp(_)));
    }
}
