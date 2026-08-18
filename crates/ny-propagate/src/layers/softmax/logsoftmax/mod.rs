// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, Axis};
use ny_core::{NyError, Result, VerificationSoundnessMode};
use ny_tensor::{
    next_down_f32, next_up_f32, sub_down_f32, sub_up_f32, BoundedTensor, RepairStrategy,
};
use std::borrow::Cow;

use super::super::common::BoundPropagation;
use crate::bounds::nan_propagating_max;
use crate::LinearBounds;

mod batched;
mod crown;

/// LogSoftmax layer: y = log(softmax(x)) = x - logsumexp(x)
///
/// LogSoftmax is more numerically stable than computing log(softmax(x))
/// directly. It's commonly used with NLLLoss for classification.
#[derive(Debug, Clone)]
pub struct LogSoftmaxLayer {
    /// Dimension along which to apply logsoftmax (default: -1)
    pub axis: i32,
    /// Use sound (no sampling) relaxation for CROWN.
    ///
    /// When true, uses LSE-based affine bounds (sound) instead of heuristic sampling.
    pub sound: bool,
}

impl LogSoftmaxLayer {
    /// Create a new LogSoftmax layer.
    pub fn new(axis: i32) -> Self {
        Self { axis, sound: true }
    }

    /// Enable or disable sound (no sampling) CROWN mode.
    pub fn with_sound_mode(mut self, enabled: bool) -> Self {
        self.sound = enabled;
        self
    }

    /// Enable heuristic sampling-based CROWN relaxation (not provably sound).
    pub fn with_heuristic_sampling(mut self, enabled: bool) -> Self {
        self.sound = !enabled;
        self
    }

    /// Returns the current verification soundness mode (Sound or Heuristic).
    pub fn soundness_mode(&self) -> VerificationSoundnessMode {
        if self.sound {
            VerificationSoundnessMode::Sound
        } else {
            VerificationSoundnessMode::Heuristic
        }
    }

    /// IBP propagation that accounts for a prepended restart axis.
    ///
    /// When restart batching adds a leading axis, positive stored axes (which
    /// used unbatched convention `axis - 1` at load time) must shift right by
    /// one to resolve against the correct sample-space dimension.
    ///
    /// Part of #4096.
    pub fn propagate_ibp_preserve_leading_axis(
        &self,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let ndim = input.shape().len();
        if ndim == 0 {
            return Err(NyError::InvalidSpec(
                "LogSoftmax requires at least 1D input".to_string(),
            ));
        }
        let has_non_finite_input = input.lower().iter().any(|&v| !v.is_finite())
            || input.upper().iter().any(|&v| !v.is_finite());
        if has_non_finite_input {
            let lower = ArrayD::from_elem(input.lower().raw_dim(), f32::NEG_INFINITY);
            let upper = ArrayD::from_elem(input.upper().raw_dim(), f32::INFINITY);
            return BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative);
        }
        let axis = crate::layers::common::resolve_axis_i32_with_restored_leading_axis(
            self.axis,
            ndim,
            "LogSoftmax",
        )?;
        Self::propagate_ibp_with_axis(input, axis)
    }
}

impl Default for LogSoftmaxLayer {
    fn default() -> Self {
        Self {
            axis: -1,
            sound: true,
        }
    }
}

/// Compute logsumexp with directed rounding in f64 along the given axis.
/// Returns (logsumexp_upper, logsumexp_lower) where upper is rounded UP and lower DOWN.
/// Ref: #3245 — f64 pipeline prevents intermediate f32 truncation errors.
fn logsumexp_directed(input: &BoundedTensor, axis: usize) -> (ArrayD<f32>, ArrayD<f32>) {
    let max_upper = input
        .upper()
        .fold_axis(Axis(axis), f32::NEG_INFINITY, |&acc, &x| {
            nan_propagating_max(acc, x)
        });
    let max_lower = input
        .lower()
        .fold_axis(Axis(axis), f32::NEG_INFINITY, |&acc, &x| {
            nan_propagating_max(acc, x)
        });

    let max_upper_expanded = max_upper.clone().insert_axis(Axis(axis));
    let max_lower_expanded = max_lower.clone().insert_axis(Axis(axis));

    let exp_upper = (input.upper() - &max_upper_expanded).mapv(|x| (x as f64).exp());
    let exp_lower = (input.lower() - &max_lower_expanded).mapv(|x| (x as f64).exp());

    let sum_upper = exp_upper.sum_axis(Axis(axis));
    let sum_lower = exp_lower.sum_axis(Axis(axis));

    let lse_upper = max_upper.mapv(|x| x as f64) + sum_upper.mapv(|x| x.ln());
    let lse_lower = max_lower.mapv(|x| x as f64) + sum_lower.mapv(|x| x.ln());

    (
        lse_upper.mapv(|x| next_up_f32(x as f32)),
        lse_lower.mapv(|x| next_down_f32(x as f32)),
    )
}

/// Shared IBP implementation parameterized by resolved axis.
impl LogSoftmaxLayer {
    fn propagate_ibp_with_axis(input: &BoundedTensor, axis: usize) -> Result<BoundedTensor> {
        // Compute logsumexp with f64 directed rounding (#3245).
        let (lse_upper, lse_lower) = logsumexp_directed(input, axis);
        let lse_upper_expanded = lse_upper.insert_axis(Axis(axis));
        let lse_lower_expanded = lse_lower.insert_axis(Axis(axis));

        // logsoftmax_i = x_i - logsumexp(x)
        // Lower: x_i^L - logsumexp(x^U) → round DOWN
        // Upper: x_i^U - logsumexp(x^L) → round UP
        // `sub_down_f32` rather than `(a - b).mapv(next_down_f32)`: the latter
        // is a plain round-to-nearest subtract followed by an UNCONDITIONAL ULP
        // step, so it gives away a full ULP even when the subtraction was
        // exact. The directed form steps only when it must.
        // NOTE: `lse_*_expanded` carries a size-1 axis at `axis`, which the `-`
        // operator used to broadcast implicitly. `Zip` does NOT broadcast, so
        // the views must be widened explicitly first.
        let target = input.lower().raw_dim();
        let lse_upper_b = lse_upper_expanded
            .broadcast(target.clone())
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: input.lower().shape().to_vec(),
                got: lse_upper_expanded.shape().to_vec(),
            })?;
        let lse_lower_b =
            lse_lower_expanded
                .broadcast(target)
                .ok_or_else(|| NyError::ShapeMismatch {
                    expected: input.lower().shape().to_vec(),
                    got: lse_lower_expanded.shape().to_vec(),
                })?;

        let lower = ndarray::Zip::from(input.lower())
            .and(&lse_upper_b)
            .map_collect(|&x, &l| sub_down_f32(x, l));
        // Soundness tightening: log_softmax(x)_i = x_i - logsumexp(x) <= 0 for ALL
        // inputs, since logsumexp(x) >= max_j x_j >= x_i. Hence 0 is an exact,
        // input-independent upper bound on every output. The interval upper above
        // (x_i^U - logsumexp(x^L), with x_i^U and x^L from possibly different corners)
        // over-approximates and can exceed 0; clamping to min(upper, 0) only
        // TIGHTENS and never drops a reachable value. This preserves lower <= upper:
        // lower <= true_value <= 0, so lower <= min(upper, 0). The lower bound is
        // left untouched.
        let upper = ndarray::Zip::from(input.upper())
            .and(&lse_lower_b)
            .map_collect(|&x, &l| sub_up_f32(x, l).min(0.0));

        // Repair non-finite outputs: logsumexp subtraction can produce Inf/NaN
        // from large finite inputs. Clamp to FALLBACK_BOUND for consistency
        // with the IBP overflow strategy (#3030, #3060).
        BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
    }
}

impl BoundPropagation for LogSoftmaxLayer {
    /// IBP for LogSoftmax: y = x - logsumexp(x)
    ///
    /// For bounds on logsoftmax_i = x_i - log(sum_j exp(x_j)):
    /// - Lower bound on logsoftmax_i: x_i^L - logsumexp(x^U)
    /// - Upper bound on logsoftmax_i: x_i^U - logsumexp(x^L)
    ///
    /// This is a simplified but sound bound: we use the max logsumexp for all lower bounds
    /// and min logsumexp for all upper bounds.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let ndim = input.shape().len();
        if ndim == 0 {
            return Err(NyError::InvalidSpec(
                "LogSoftmax requires at least 1D input".to_string(),
            ));
        }
        let has_non_finite_input = input.lower().iter().any(|&v| !v.is_finite())
            || input.upper().iter().any(|&v| !v.is_finite());
        if has_non_finite_input {
            let lower = ArrayD::from_elem(input.lower().raw_dim(), f32::NEG_INFINITY);
            let upper = ArrayD::from_elem(input.upper().raw_dim(), f32::INFINITY);
            return BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative);
        }

        let axis = crate::layers::common::resolve_axis_i32(self.axis, ndim, "LogSoftmax")?;
        Self::propagate_ibp_with_axis(input, axis)
    }

    /// CROWN requires pre-activation bounds for nonlinear LogSoftmax.
    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "LogSoftmax is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
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
        LogSoftmaxLayer::propagate_linear_with_bounds(
            self,
            bounds,
            pre_activation,
            self.soundness_mode(),
        )
    }
}

#[cfg(test)]
mod tests;
