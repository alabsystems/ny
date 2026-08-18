// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::layers::common::{impl_elementwise_activation, BoundPropagation};

use super::validate::validate_clip_bounds;
use super::LinearRelaxation;

// Re-export for test module's `use super::*` — cfg(test) items are used by
// the separate tests.rs file but the linter can't see the cross-file usage.
#[cfg(test)]
#[allow(unused_imports)]
use crate::{BatchedLinearBounds, LinearBounds};

/// Clip layer: clamp values to [min, max] range.
///
/// Clip is commonly used in quantization-aware training and to limit activations.
/// clip(x, min, max) = max(min, min(max, x))
#[derive(Debug, Clone)]
pub struct ClipLayer {
    /// Minimum value (inclusive)
    pub min: f32,
    /// Maximum value (inclusive)
    pub max: f32,
}

impl ClipLayer {
    /// Validate and create a new Clip layer with the given bounds.
    pub fn try_new(min: f32, max: f32) -> Result<Self> {
        let (min, max) = validate_clip_bounds(min, max)?;
        Ok(Self { min, max })
    }

    /// Create a new Clip layer with the given bounds.
    pub fn new(min: f32, max: f32) -> Self {
        Self::try_new(min, max).expect("invariant: ClipLayer::new requires validated bounds")
    }
}

impl BoundPropagation for ClipLayer {
    /// IBP for Clip: y = clip(x, min, max)
    ///
    /// For x in [l, u]:
    /// - lower_bound = clip(l, min, max)
    /// - upper_bound = clip(u, min, max)
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Guard: NaN input bounds propagate through f32::clamp, producing NaN
        // output. NaN ONLY — ±Inf is a legitimate input here. An upstream node
        // that failed closed to an OpaqueSkip hands its consumers `[-inf, +inf]`
        // (`OpaqueSkipLayer::unbounded_like` builds exactly that); rejecting it
        // as `NumericalInstability` aborted the WHOLE graph-IBP pass, because
        // that variant is not in `is_degradable_error`. Pattern: AddConstant
        // (add_constant.rs:69-79). CROWN path guards separately at
        // clip_linear_relaxation:124.
        if input.lower().iter().any(|x| x.is_nan()) || input.upper().iter().any(|x| x.is_nan()) {
            return Err(NyError::NumericalInstability(
                "Clip IBP: NaN input bounds".to_string(),
            ));
        }
        let min_val = self.min;
        let max_val = self.max;
        let lower = input.lower().mapv(|v| v.clamp(min_val, max_val));
        let upper = input.upper().mapv(|v| v.clamp(min_val, max_val));
        // `new_allow_infinite`, not the strict `new`: `f32::clamp` is pure
        // comparison, never arithmetic, so none of the NaN-producing inf
        // patterns (inf - inf, 0 * inf, inf / inf) exists here and no repair is
        // needed. With finite min/max an infinite endpoint saturates to the
        // clip range, so a fully tainted `[-inf, +inf]` input recovers the
        // EXACT output range [min, max] — Clip is a taint sink, not just a
        // pass-through. `new_allow_infinite` is still required because
        // `validate_clip_bounds` permits ±Inf min/max (it rejects only NaN and
        // min > max), in which case the saturated output is itself infinite.
        // NaN can only come from a NaN input, which the guard above rejects,
        // and NaN reaching here anyway still hard-errors in this constructor.
        BoundedTensor::new_allow_infinite(lower, upper)
    }

    impl_elementwise_activation!(
        @trait_methods
        ClipLayer,
        NyError::InvalidSpec(
            "Clip CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string()
        )
    );
}

impl ClipLayer {
    impl_elementwise_activation!(
        @inherent_methods_stateful
        ClipLayer,
        |layer: &ClipLayer, l, u| clip_linear_relaxation(l, u, layer.min, layer.max),
        domain_guard: |pre_activation: &BoundedTensor| {
            crate::layers::common::non_finite_domain_guard("Clip", pre_activation)
        }
    );
}

/// Compute CROWN linear relaxation for Clip on interval [l, u].
///
/// Clip: y = clamp(x, min_val, max_val)
///
/// Piecewise linear:
/// - x < min: y = min (constant)
/// - min <= x <= max: y = x (identity)
/// - x > max: y = max (constant)
///
/// Returns (lower_slope, lower_intercept, upper_slope, upper_intercept).
///
/// Reference: alpha-beta-CROWN BoundHardTanh.bound_relax()
/// at auto_LiRPA/operators/activations.py:275-403
fn clip_linear_relaxation(l: f32, u: f32, min_val: f32, max_val: f32) -> LinearRelaxation {
    // NaN/Inf guard: BoundedTensor::new rejects non-finite inputs, but
    // callers using new_unchecked can still feed invalid intervals.
    // Clip output is always in [min_val, max_val], so constant bounds
    // at the full range are always sound.
    // Reference: same pattern as SiLU (silu.rs:228), Softsign, HardSwish.
    if l.is_nan() || u.is_nan() || !l.is_finite() || !u.is_finite() {
        return LinearRelaxation::new(0.0, min_val, 0.0, max_val);
    }

    // Near-degenerate interval: avoid division by near-zero (u - l).
    // Pattern: SiLU, Mish, Softsign, HardSwish, GELU.
    if (u - l).abs() < 1e-8 {
        // SOUNDNESS (false-proof fix): clip is monotone non-decreasing, so a single eval(l)
        // misses clip(u) → a certified bound under the true value when that gap exceeds the ULP.
        // Cover the endpoint range with directed outward rounding.
        let y_l = l.clamp(min_val, max_val);
        let y_u = u.clamp(min_val, max_val);
        return LinearRelaxation::new(
            0.0,
            next_down_f32(y_l.min(y_u)),
            0.0,
            next_up_f32(y_l.max(y_u)),
        );
    }

    if u <= min_val {
        // Entirely below min: constant output = min
        LinearRelaxation::new(0.0, min_val, 0.0, min_val)
    } else if l >= max_val {
        // Entirely above max: constant output = max
        LinearRelaxation::new(0.0, max_val, 0.0, max_val)
    } else if l >= min_val && u <= max_val {
        // Entirely within [min, max]: identity
        LinearRelaxation::identity()
    } else if l < min_val && u > max_val {
        // Case 4: Crosses both boundaries. Directed rounding (#3337).
        // Upper: line through (l, min) to (max, max). Lower: (min, min) to (u, max).
        let max_abs = l.abs().max(u.abs()) as f64;
        let (l64, u64) = (l as f64, u as f64);
        let (min64, max64) = (min_val as f64, max_val as f64);
        let su64 = (max64 - min64) / (max64 - l64);
        let su = su64 as f32;
        let su_err = next_up_f32(((su64 - su as f64).abs() * max_abs) as f32);
        let iu = next_up_f32((max64 - su64 * max64) as f32 + su_err);
        let sl64 = (max64 - min64) / (u64 - min64);
        let sl = sl64 as f32;
        let sl_err = next_up_f32(((sl64 - sl as f64).abs() * max_abs) as f32);
        let il = next_down_f32((min64 - sl64 * min64) as f32 - sl_err);
        LinearRelaxation::new(sl, il, su, iu)
    } else if l < min_val {
        // Case 5: Crosses lower boundary only. Directed rounding (#3337).
        let max_abs = l.abs().max(u.abs()) as f64;
        let (l64, u64) = (l as f64, u as f64);
        let su64 = (u64 - min_val as f64) / (u64 - l64);
        let su = su64 as f32;
        let su_err = next_up_f32(((su64 - su as f64).abs() * max_abs) as f32);
        let (ls, li) = if su > 0.5 { (1.0, 0.0) } else { (0.0, min_val) };
        let iu = next_up_f32((u64 - su64 * u64) as f32 + su_err);
        LinearRelaxation::new(ls, li, su, iu)
    } else {
        // Case 6: Crosses upper boundary only. Directed rounding (#3337).
        let max_abs = l.abs().max(u.abs()) as f64;
        let (l64, u64) = (l as f64, u as f64);
        let sl64 = (max_val as f64 - l64) / (u64 - l64);
        let sl = sl64 as f32;
        let sl_err = next_up_f32(((sl64 - sl as f64).abs() * max_abs) as f32);
        let (us, ui) = if sl > 0.5 { (1.0, 0.0) } else { (0.0, max_val) };
        let il = next_down_f32((l64 - sl64 * l64) as f32 - sl_err);
        LinearRelaxation::new(sl, il, us, ui)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) fn audit_clip_relax(l: f32, u: f32, min_val: f32, max_val: f32) -> LinearRelaxation {
    clip_linear_relaxation(l, u, min_val, max_val)
}
