// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Power layer: y = x^p where p is a constant (element-wise).

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};
use std::borrow::Cow;
use tracing::debug;

use super::pow_relaxation::{
    pow2_linear_relaxation, pow_neg1_linear_relaxation,
    pow_positive_integer_nonnegative_linear_relaxation,
};
use crate::bounds::nan_propagating_min;
use crate::layers::activations::validate::validate_finite;
use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, non_finite_domain_guard, BoundPropagation,
};
use crate::{BatchedLinearBounds, LinearBounds};

/// Power layer: y = x^p where p is a constant (element-wise).
///
/// Used in LayerNorm for computing variance: (x - mean)^2
/// For p=2 (square), this is always non-negative.
#[derive(Debug, Clone)]
pub struct PowConstantLayer {
    /// The constant exponent.
    pub(crate) exponent: f32,
}

impl PowConstantLayer {
    /// Validate and create a new power layer with constant exponent.
    pub fn try_new(exponent: f32) -> Result<Self> {
        Ok(Self {
            exponent: validate_finite(exponent, "PowConstantLayer", "exponent")?,
        })
    }

    /// Create a new power layer with constant exponent.
    pub fn new(exponent: f32) -> Self {
        Self::try_new(exponent).expect("invariant: PowConstantLayer::new requires finite exponent")
    }

    /// Create a square layer (x^2).
    pub fn square() -> Self {
        Self::new(2.0)
    }

    /// Return the configured exponent.
    pub fn exponent(&self) -> f32 {
        self.exponent
    }
}

impl BoundPropagation for PowConstantLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // For y = x^p:
        // - Even integer (±2, ±4, ±6, ...): U-shaped, sign-aware
        // - Odd integer (±1, ±3, ...): preserves sign, monotonic (inc for p>0, dec for p<0)
        // - Non-integer: requires x >= 0 (and x > 0 for p < 0)
        // - p < 0: x^p undefined at x=0, reject intervals containing zero

        let p = self.exponent;
        let is_integer_exp = (p - p.round()).abs() < 1e-6;
        // Guard: extreme float exponents saturate to i32::MAX/MIN under `as i32`,
        // which can misclassify even exponents as odd. Reject exponents that
        // overflow i32 range. (#2911)
        let p_rounded = p.round();
        if is_integer_exp && (!p_rounded.is_finite() || p_rounded.abs() > i32::MAX as f32) {
            return Err(NyError::InvalidSpec(format!(
                "PowConstantLayer exponent {} overflows i32 range for parity check",
                p,
            )));
        }
        let is_even_integer = is_integer_exp && (p_rounded as i32) % 2 == 0;

        // Negative exponents: x^p is undefined at x=0.
        // Reject any element whose interval contains zero for all p < 0.
        // Per-element check: each [l_i, u_i] must not span zero. A global
        // min/max check would falsely reject tensors where some elements are
        // positive and others are negative but none individually contain zero.
        // See #1699.
        if p < 0.0 {
            for (i, (l, u)) in input.lower().iter().zip(input.upper().iter()).enumerate() {
                if *l <= 0.0 && *u >= 0.0 {
                    return Err(NyError::InvalidSpec(format!(
                        "PowConstantLayer with negative exponent {} requires inputs excluding zero, \
                         element {} has interval containing zero ([{}, {}])",
                        p, i, l, u
                    )));
                }
            }
        }

        // Even integer exponents (x^2, x^4, ... and x^{-2}, x^{-4}, ...):
        // U-shaped, sign-aware.
        // For p > 0: output non-negative, minimum at x=0
        // For p < 0: output positive, maximum at x→0 (but zero already rejected above)
        //   - For p < 0, x > 0: decreasing. For p < 0, x < 0: increasing. (flipped from p > 0)
        if is_even_integer {
            let mut out_lower = ArrayD::zeros(IxDyn(input.shape()));
            let mut out_upper = ArrayD::zeros(IxDyn(input.shape()));
            let p64 = p as f64;

            for (idx, &l) in input.lower().indexed_iter() {
                let u = input.upper()[idx.clone()];
                // Compute in f64 for precision, then cast with directed rounding.
                // f32 powf can round either direction; directed rounding guarantees
                // lower bounds round DOWN and upper bounds round UP. (#1483)
                let l64 = l as f64;
                let u64 = u as f64;
                let lp64 = l64.powf(p64);
                let up64 = u64.powf(p64);

                if p > 0.0 {
                    // Positive even: U-shaped with minimum at x=0
                    if l >= 0.0 {
                        out_lower[idx.clone()] = next_down_f32(lp64 as f32);
                        out_upper[idx] = next_up_f32(up64 as f32);
                    } else if u <= 0.0 {
                        out_lower[idx.clone()] = next_down_f32(up64 as f32);
                        out_upper[idx] = next_up_f32(lp64 as f32);
                    } else {
                        out_lower[idx.clone()] = 0.0;
                        out_upper[idx] = next_up_f32((lp64 as f32).max(up64 as f32));
                    }
                } else {
                    // Negative even (p=-2,-4,...): ∪-shaped with maximum at x→0
                    // For x > 0: monotonically decreasing (larger x → smaller output)
                    // For x < 0: monotonically increasing (more negative x → smaller output)
                    // Zero already rejected above, so interval is entirely positive or negative.
                    if l > 0.0 {
                        // All positive, decreasing: lower=u^p, upper=l^p
                        out_lower[idx.clone()] = next_down_f32(up64 as f32);
                        out_upper[idx] = next_up_f32(lp64 as f32);
                    } else {
                        // All negative, increasing: lower=l^p, upper=u^p
                        out_lower[idx.clone()] = next_down_f32(lp64 as f32);
                        out_upper[idx] = next_up_f32(up64 as f32);
                    }
                }
            }

            // Centralized NaN/Inf repair at constructor (#3423, replaces ad-hoc #3030).
            return BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative);
        }

        // Non-integer exponents: x^p is undefined for x < 0 in reals.
        // For p < 0 non-integer: also need x > 0 (zero already rejected above).
        if !is_integer_exp {
            // NaN-propagating fold — see #2577.
            let min_input = input
                .lower()
                .iter()
                .copied()
                .fold(f32::INFINITY, nan_propagating_min);
            if min_input.is_nan() || min_input < 0.0 {
                return Err(NyError::InvalidSpec(format!(
                    "PowConstantLayer with non-integer exponent {} requires non-negative inputs, got minimum bound {}",
                    p, min_input
                )));
            }
        }

        // Odd integer or non-negative non-integer: monotonic.
        // p > 0: monotonically increasing → lower^p maps to lower, upper^p maps to upper
        // p < 0 odd integer: monotonically decreasing → swap bounds
        //
        // Directed rounding: compute in f64, cast to f32 with next_down/next_up
        // to guarantee lower bounds are below and upper bounds are above the
        // true value. Raw f32 powf can round either direction. (#1483)
        let p64 = p as f64;
        if p < 0.0 {
            // Odd negative integer (p=-1,-3,...): monotonically decreasing.
            // For x > 0: larger x → smaller output. For x < 0: more negative x → less negative output.
            let out_lower = input
                .upper()
                .mapv(|v| next_down_f32((v as f64).powf(p64) as f32));
            let out_upper = input
                .lower()
                .mapv(|v| next_up_f32((v as f64).powf(p64) as f32));
            // Centralized NaN/Inf repair at constructor (#3423, replaces ad-hoc #3030).
            BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
        } else {
            let out_lower = input
                .lower()
                .mapv(|v| next_down_f32((v as f64).powf(p64) as f32));
            let out_upper = input
                .upper()
                .mapv(|v| next_up_f32((v as f64).powf(p64) as f32));
            // Centralized NaN/Inf repair at constructor (#3423, replaces ad-hoc #3030).
            BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
        }
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Power is nonlinear, linear bounds not supported
        Err(NyError::UnsupportedOp(
            "Pow is nonlinear - use propagate_ibp".to_string(),
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
        PowConstantLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl PowConstantLayer {
    /// CROWN backward propagation with pre-activation bounds.
    ///
    /// Supports x^2, 1/x, and positive integer exponents >= 2 on x >= 0.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("PowConstant", pre_activation)?;
        let p = self.exponent;
        if (p - 2.0).abs() < 1e-6 {
            return crown_elementwise_backward(bounds, pre_activation, pow2_linear_relaxation);
        }
        if (p + 1.0).abs() < 1e-6 {
            return crown_elementwise_backward(bounds, pre_activation, pow_neg1_linear_relaxation);
        }
        let rounded = p.round();
        if (p - rounded).abs() < 1e-6
            && rounded >= 2.0
            && rounded <= i32::MAX as f32
            && pre_activation.lower().iter().all(|&v| v >= 0.0)
        {
            let exponent = rounded as i32;
            return crown_elementwise_backward(bounds, pre_activation, |l, u| {
                pow_positive_integer_nonnegative_linear_relaxation(exponent, l, u)
            });
        }
        Err(NyError::UnsupportedOp(format!(
            "CROWN for PowConstant exponent {} not supported",
            p
        )))
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        non_finite_domain_guard("PowConstant", pre_activation)?;
        let p = self.exponent;
        if (p - 2.0).abs() < 1e-6 {
            debug!("PowConstant (x^2) layer batched CROWN backward propagation");
            return crown_elementwise_backward_batched(
                bounds,
                pre_activation,
                pow2_linear_relaxation,
            );
        }
        if (p + 1.0).abs() < 1e-6 {
            debug!("PowConstant (1/x) layer batched CROWN backward propagation");
            return crown_elementwise_backward_batched(
                bounds,
                pre_activation,
                pow_neg1_linear_relaxation,
            );
        }
        let rounded = p.round();
        if (p - rounded).abs() < 1e-6
            && rounded >= 2.0
            && rounded <= i32::MAX as f32
            && pre_activation.lower().iter().all(|&v| v >= 0.0)
        {
            let exponent = rounded as i32;
            return crown_elementwise_backward_batched(bounds, pre_activation, |l, u| {
                pow_positive_integer_nonnegative_linear_relaxation(exponent, l, u)
            });
        }
        Err(NyError::UnsupportedOp(format!(
            "Batched CROWN for PowConstant exponent {} not supported",
            p
        )))
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("PowConstant", pre_activation)?;
        let p = self.exponent;
        if (p - 2.0).abs() < 1e-6 {
            return crown_elementwise_backward_patches(
                bounds,
                pre_activation,
                pow2_linear_relaxation,
            );
        }
        if (p + 1.0).abs() < 1e-6 {
            return crown_elementwise_backward_patches(
                bounds,
                pre_activation,
                pow_neg1_linear_relaxation,
            );
        }
        let rounded = p.round();
        if (p - rounded).abs() < 1e-6
            && rounded >= 2.0
            && rounded <= i32::MAX as f32
            && pre_activation.lower().iter().all(|&v| v >= 0.0)
        {
            let exponent = rounded as i32;
            return crown_elementwise_backward_patches(bounds, pre_activation, |l, u| {
                pow_positive_integer_nonnegative_linear_relaxation(exponent, l, u)
            });
        }
        Err(NyError::UnsupportedOp(format!(
            "Patches CROWN for PowConstant exponent {} not supported",
            p
        )))
    }
}
