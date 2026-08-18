// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::layers::activations::validate::{validate_finite, validate_nonnegative_finite};
use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{impl_elementwise_activation, BoundPropagation};

// Re-export for test module's `use super::*` — cfg(test) items are used by
// the separate tests.rs file but the linter can't see the cross-file usage.
#[cfg(test)]
#[allow(unused_imports)]
use crate::LinearBounds;

/// Shrink layer: soft thresholding / shrinkage operation
///
/// y = x - bias if x > lambd
/// y = x + bias if x < -lambd
/// y = 0 otherwise
///
/// This implements soft thresholding used in sparse coding and LASSO.
/// Default: bias = 0.0, lambd = 0.5
#[derive(Debug, Clone)]
pub struct ShrinkLayer {
    /// Bias value (default: 0.0)
    pub(crate) bias: f32,
    /// Lambda threshold (default: 0.5)
    pub(crate) lambd: f32,
}

impl ShrinkLayer {
    /// Validate and create a new Shrink layer.
    pub fn try_new(bias: f32, lambd: f32) -> Result<Self> {
        Ok(Self {
            bias: validate_finite(bias, "ShrinkLayer", "bias")?,
            lambd: validate_nonnegative_finite(lambd, "ShrinkLayer", "lambd")?,
        })
    }

    /// Create a new Shrink layer.
    pub fn new(bias: f32, lambd: f32) -> Self {
        Self::try_new(bias, lambd)
            .expect("invariant: ShrinkLayer::new requires validated parameters")
    }

    /// Return the configured bias value.
    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// Return the configured lambda threshold.
    pub fn lambd(&self) -> f32 {
        self.lambd
    }
}

impl Default for ShrinkLayer {
    fn default() -> Self {
        Self::new(0.0, 0.5)
    }
}

/// Evaluate Shrink function: soft thresholding
fn shrink_scalar(x: f32, bias: f32, lambd: f32) -> f32 {
    if x > lambd {
        x - bias
    } else if x < -lambd {
        x + bias
    } else {
        0.0
    }
}

impl BoundPropagation for ShrinkLayer {
    /// IBP for Shrink: soft thresholding
    ///
    /// The function has three linear pieces:
    /// - x < -lambd: y = x + bias (slope 1)
    /// - -lambd <= x <= lambd: y = 0 (slope 0)
    /// - x > lambd: y = x - bias (slope 1)
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Guard: NaN input bounds → NaN comparisons silently fall into the
        // dead-zone branch, returning (0.0, 0.0) — unsound. NaN ONLY — ±Inf is
        // a legitimate input here. An upstream node that failed closed to an
        // OpaqueSkip hands its consumers `[-inf, +inf]`
        // (`OpaqueSkipLayer::unbounded_like` builds exactly that); rejecting it
        // as `NumericalInstability` aborted the WHOLE graph-IBP pass, because
        // that variant is not in `is_degradable_error`. The ±Inf branch
        // selection below is exact: `-inf < -lambd` and `+inf > lambd` hold for
        // every finite lambd, so an infinite endpoint never lands in the
        // dead-zone branch. Pattern: AddConstant (add_constant.rs:69-79).
        if input.lower().iter().any(|x| x.is_nan()) || input.upper().iter().any(|x| x.is_nan()) {
            return Err(NyError::NumericalInstability(
                "Shrink IBP: NaN input bounds".to_string(),
            ));
        }

        let bias = self.bias;
        let lambd = self.lambd;

        let lower_shape = input.lower().shape().to_vec();
        let mut lower_data = Vec::with_capacity(input.lower().len());
        let mut upper_data = Vec::with_capacity(input.upper().len());

        for (&l, &u) in input.lower().iter().zip(input.upper().iter()) {
            // Compute bounds by evaluating at endpoints and checking critical points
            let fl = shrink_scalar(l, bias, lambd);
            let fu = shrink_scalar(u, bias, lambd);

            // The function is piecewise linear with breakpoints at -lambd and lambd
            // Within each piece, it's monotonic

            // Check if interval spans multiple pieces
            let spans_neg_break = l < -lambd && u > -lambd;
            let spans_pos_break = l < lambd && u > lambd;
            let in_dead_zone = l >= -lambd && u <= lambd;

            let (nl, nu) = if in_dead_zone {
                // Entirely in dead zone
                (0.0, 0.0)
            } else if !spans_neg_break && !spans_pos_break {
                // Entirely in one linear piece (outside dead zone)
                // Monotonically increasing (slope 1)
                (nan_propagating_min(fl, fu), nan_propagating_max(fl, fu))
            } else {
                // Spans one or more breakpoints.
                // Shrink has discontinuities at ±lambd when bias ≠ 0:
                //   shrink(lambd-ε) = 0, shrink(lambd+ε) = lambd - bias
                //   shrink(-lambd+ε) = 0, shrink(-lambd-ε) = -lambd + bias
                // We must include the one-sided limits from both sides of each
                // breakpoint, not just the value at the breakpoint (which is 0).
                let mut candidates = vec![fl, fu];
                if spans_neg_break {
                    candidates.push(0.0); // dead-zone side of -lambd
                    candidates.push(-lambd + bias); // linear side of -lambd
                }
                if spans_pos_break {
                    candidates.push(0.0); // dead-zone side of +lambd
                    candidates.push(lambd - bias); // linear side of +lambd
                }

                // NaN-propagating folds — see #2577.
                (
                    candidates
                        .iter()
                        .cloned()
                        .fold(f32::INFINITY, nan_propagating_min),
                    candidates
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, nan_propagating_max),
                )
            };

            lower_data.push(nl);
            upper_data.push(nu);
        }

        let lower = ArrayD::from_shape_vec(IxDyn(&lower_shape), lower_data)
            .map_err(|e| NyError::InvalidSpec(format!("Shrink lower reshape: {}", e)))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&lower_shape), upper_data)
            .map_err(|e| NyError::InvalidSpec(format!("Shrink upper reshape: {}", e)))?;

        // `new_allow_infinite`, not the strict `new`: an infinite endpoint from
        // an upstream OpaqueSkip flows through cleanly, so no repair is needed.
        // `bias` and `lambd` are validated finite at construction, so the only
        // arithmetic on an infinite input is `x - bias` / `x + bias` with a
        // finite bias → ±inf, never inf - inf. The `candidates` folds are
        // comparisons (nan_propagating_min/max), not arithmetic, and 0 * inf /
        // inf / inf never appear. So an infinite endpoint yields either ±Inf or
        // a finite bound (e.g. [-inf, 0] with lambd = 0.5 caps the upper at 0),
        // and NaN can only come from a NaN input, which the guard above
        // rejects; a NaN that reached here anyway still hard-errors in this
        // constructor.
        BoundedTensor::new_allow_infinite(lower, upper)
    }

    impl_elementwise_activation!(
        @trait_methods
        ShrinkLayer,
        NyError::InvalidSpec(
            "Shrink CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string()
        )
    );
}

impl ShrinkLayer {
    impl_elementwise_activation!(
        @inherent_methods_stateful
        ShrinkLayer,
        |layer: &ShrinkLayer, l, u| layer.relaxation(l, u),
        domain_guard: |pre_activation: &BoundedTensor| {
            crate::layers::common::non_finite_domain_guard("Shrink", pre_activation)
        }
    );

    /// Compute the linear relaxation (lower_slope, lower_intercept, upper_slope, upper_intercept)
    /// for a single neuron with pre-activation bounds [l, u].
    ///
    /// 6-case analytical relaxation following the BoundHardTanh pattern from
    /// alpha-beta-CROWN (activations.py:275-403), adapted for Shrink's 3-piece
    /// structure with possible discontinuities at ±lambd.
    ///
    /// Reference: alpha-beta-CROWN auto_LiRPA/operators/activations.py BoundHardTanh
    pub(super) fn relaxation(&self, l: f32, u: f32) -> LinearRelaxation {
        let bias = self.bias;
        let lambd = self.lambd;

        // NaN guard: treat unknown interval endpoints as unconstrained.
        if l.is_nan() || u.is_nan() {
            return LinearRelaxation::nan_fallback();
        }

        // Inf guard
        if l.is_infinite() || u.is_infinite() {
            let fl = shrink_scalar(l, bias, lambd);
            let fu = shrink_scalar(u, bias, lambd);
            let mut fmax = nan_propagating_max(nan_propagating_max(fl, fu), 0.0);
            let mut fmin = nan_propagating_min(nan_propagating_min(fl, fu), 0.0);
            // Include breakpoint approach values when breakpoints are within [l, u].
            // Shrink has discontinuities at ±lambd when bias ≠ lambd: the function
            // approaches -lambd + bias from the left at -lambd, and lambd - bias
            // from the right at +lambd. These interior extrema are not captured by
            // the endpoint values fl, fu alone. Part of #3322.
            if l < -lambd && u > -lambd {
                let f_at_neg_break = -lambd + bias;
                fmax = nan_propagating_max(fmax, f_at_neg_break);
                fmin = nan_propagating_min(fmin, f_at_neg_break);
            }
            if l < lambd && u > lambd {
                let f_at_pos_break = lambd - bias;
                fmax = nan_propagating_max(fmax, f_at_pos_break);
                fmin = nan_propagating_min(fmin, f_at_pos_break);
            }
            return LinearRelaxation::new(
                0.0,
                if fmin.is_finite() {
                    fmin
                } else {
                    f32::NEG_INFINITY
                },
                0.0,
                if fmax.is_finite() {
                    fmax
                } else {
                    f32::INFINITY
                },
            );
        }

        // Near-degenerate interval: avoid division by near-zero (u - l).
        // Pattern: SiLU, Mish, Softsign, HardSwish, GELU.
        if (u - l).abs() < 1e-8 {
            // SOUNDNESS (false-proof fix): shrink is monotone non-decreasing, so a single
            // eval(l) misses f(u) → a certified bound under the true value when that gap exceeds
            // the ULP. Cover the endpoint range with directed outward rounding.
            let y_l = shrink_scalar(l, bias, lambd);
            let y_u = shrink_scalar(u, bias, lambd);
            return LinearRelaxation::new(
                0.0,
                next_down_f32(y_l.min(y_u)),
                0.0,
                next_up_f32(y_l.max(y_u)),
            );
        }

        let in_neg = u < -lambd;
        let in_dead = l >= -lambd && u <= lambd;
        let in_pos = l > lambd;
        let spans_neg = l < -lambd && u > -lambd;
        let spans_pos = l < lambd && u > lambd;

        if in_neg {
            // Case 1: Entirely in region A: y = x + bias (exact)
            LinearRelaxation::new(1.0, bias, 1.0, bias)
        } else if in_dead {
            // Case 2: Entirely in dead zone: y = 0 (exact)
            LinearRelaxation::zero()
        } else if in_pos {
            // Case 3: Entirely in region C: y = x - bias (exact)
            LinearRelaxation::new(1.0, -bias, 1.0, -bias)
        } else if spans_neg && !spans_pos {
            // Case 4: l < -lambd, u in dead zone.
            // Lower chord from (l, fl) to (u, 0) in f64. Part of #3313.
            // Compute fmax/fmin in f64 to avoid f32 rounding (#3321).
            let l_f64 = l as f64;
            let u_f64 = u as f64;
            let bias_f64 = bias as f64;
            let lambd_f64 = lambd as f64;
            let fl_f64 = l_f64 + bias_f64; // shrink(l) = l + bias (l < -lambd)
            let f_at_neg_break_f64 = -lambd_f64 + bias_f64;
            let fmax_f64 = fl_f64.max(f_at_neg_break_f64).max(0.0);

            let us = 0.0;
            let ui = next_up_f32(fmax_f64 as f32);

            // Use f64 fl for chord to match f64 reference (#3321).
            let chord_s_f64 = -fl_f64 / (u_f64 - l_f64);
            let chord_s = chord_s_f64 as f32;
            let (ls, li) = if chord_s <= 1.0 + f32::EPSILON && fl_f64 <= f32::EPSILON as f64 {
                let int_at_l = fl_f64 - (chord_s as f64) * l_f64;
                let int_at_u = -(chord_s as f64) * u_f64;
                (chord_s, next_down_f32(int_at_l.min(int_at_u) as f32))
            } else if bias <= lambd {
                (1.0, bias)
            } else {
                let fmin_f64 = fl_f64.min(0.0);
                (0.0, next_down_f32(fmin_f64 as f32))
            };
            LinearRelaxation::new(ls, li, us, ui)
        } else if !spans_neg && spans_pos {
            // Case 5: l in dead zone, u > lambd.
            // Lower chord from (l, 0) to (u, fu) in f64. Part of #3313.
            // Compute fmax/fmin in f64 to avoid f32 rounding (#3321).
            let l_f64 = l as f64;
            let u_f64 = u as f64;
            let bias_f64 = bias as f64;
            let lambd_f64 = lambd as f64;
            let fu_f64 = u_f64 - bias_f64; // shrink(u) = u - bias (u > lambd)
            let f_at_pos_break_f64 = lambd_f64 - bias_f64;
            let fmax_f64 = fu_f64.max(f_at_pos_break_f64).max(0.0);

            let us = 0.0;
            let ui = next_up_f32(fmax_f64 as f32);

            // Use f64 fu for chord to match f64 reference (#3321).
            let chord_s_f64 = fu_f64 / (u_f64 - l_f64);
            let chord_s = chord_s_f64 as f32;
            let (ls, li) = if chord_s <= 1.0 + f32::EPSILON && fu_f64 <= f32::EPSILON as f64 {
                let int_at_l = -(chord_s as f64) * l_f64;
                let int_at_u = fu_f64 - (chord_s as f64) * u_f64;
                let chord_t = next_down_f32(int_at_l.min(int_at_u) as f32);
                // Breakpoint validity: chord must stay below function at +lambd.
                // Dead zone gives 0 at lambd; pos piece approaches lambd - bias.
                // When bias > lambd, the function dips below the chord line at
                // the breakpoint discontinuity. Part of #3322.
                let chord_at_pos_break = chord_s * lambd + chord_t;
                let f_pos_break_f32 = lambd - bias;
                if chord_at_pos_break <= f32::EPSILON
                    && chord_at_pos_break <= f_pos_break_f32 + f32::EPSILON
                {
                    (chord_s, chord_t)
                } else if bias >= lambd {
                    (1.0, -bias)
                } else {
                    let fmin_f64 = fu_f64.min(f_at_pos_break_f64).min(0.0);
                    (0.0, next_down_f32(fmin_f64 as f32))
                }
            } else if bias >= lambd {
                (1.0, -bias)
            } else {
                let fmin_f64 = fu_f64.min(f_at_pos_break_f64).min(0.0);
                (0.0, next_down_f32(fmin_f64 as f32))
            };
            LinearRelaxation::new(ls, li, us, ui)
        } else {
            // Case 6: Spans both breakpoints.
            // Chord from (l, fl) to (u, fu) in f64 with separate directed
            // rounding for upper vs lower intercept. Part of #3313.
            // Compute fmax/fmin in f64 to avoid f32 rounding (#3321).
            let l_f64 = l as f64;
            let u_f64 = u as f64;
            let bias_f64 = bias as f64;
            let lambd_f64 = lambd as f64;
            let fl_f64 = l_f64 + bias_f64; // shrink(l) = l + bias (l < -lambd)
            let fu_f64 = u_f64 - bias_f64; // shrink(u) = u - bias (u > lambd)
            let f_at_neg_break_f64 = -lambd_f64 + bias_f64;
            let f_at_pos_break_f64 = lambd_f64 - bias_f64;
            let fmax_f64 = fl_f64
                .max(fu_f64)
                .max(f_at_neg_break_f64)
                .max(f_at_pos_break_f64)
                .max(0.0);
            let fmin_f64 = fl_f64
                .min(fu_f64)
                .min(f_at_neg_break_f64)
                .min(f_at_pos_break_f64)
                .min(0.0);

            let f_at_neg_break = -lambd + bias;
            let f_at_pos_break = lambd - bias;

            // Use f64-computed fl/fu for chord to match f64 reference (#3321).
            let chord_s = if (u - l).abs() > f32::EPSILON {
                let s = (fu_f64 - fl_f64) / (u_f64 - l_f64);
                s as f32
            } else {
                0.0
            };
            let int_at_l = fl_f64 - (chord_s as f64) * l_f64;
            let int_at_u = fu_f64 - (chord_s as f64) * u_f64;

            // Upper bound: intercept rounded up, full validity over all 3 pieces.
            let chord_t_upper = next_up_f32(int_at_l.max(int_at_u) as f32);
            let chord_at_neg_u_brk = chord_s * (-lambd) + chord_t_upper;
            let chord_at_pos_u_brk = chord_s * lambd + chord_t_upper;
            let chord_at_l_u = chord_s * l + chord_t_upper;
            let chord_at_u_u = chord_s * u + chord_t_upper;
            let fl_f32_up = next_up_f32(fl_f64 as f32);
            let fu_f32_up = next_up_f32(fu_f64 as f32);
            let chord_valid_upper =
                // Dead zone: chord >= 0 at ±lambd
                chord_at_neg_u_brk >= -f32::EPSILON
                && chord_at_pos_u_brk >= -f32::EPSILON
                // Neg piece: chord >= x + bias at -lambd and l
                && chord_at_neg_u_brk >= f_at_neg_break - f32::EPSILON
                && chord_at_l_u >= fl_f32_up - f32::EPSILON
                // Pos piece: chord >= x - bias at +lambd and u
                && chord_at_pos_u_brk >= f_at_pos_break - f32::EPSILON
                && chord_at_u_u >= fu_f32_up - f32::EPSILON;

            let (us, ui) = if chord_valid_upper {
                (chord_s, chord_t_upper)
            } else {
                (0.0, next_up_f32(fmax_f64 as f32))
            };

            // Lower bound: intercept rounded down, full validity over all 3 pieces.
            // Must be below: dead zone (0), neg piece (x+bias), pos piece (x-bias).
            // Each piece is linear, so checking at segment endpoints suffices.
            let chord_t_lower = next_down_f32(int_at_l.min(int_at_u) as f32);
            let chord_at_neg_lambd = chord_s * (-lambd) + chord_t_lower;
            let chord_at_pos_lambd = chord_s * lambd + chord_t_lower;
            let chord_at_l = chord_s * l + chord_t_lower;
            let chord_at_u = chord_s * u + chord_t_lower;
            let fl_f32 = next_down_f32(fl_f64 as f32);
            let fu_f32 = next_down_f32(fu_f64 as f32);
            let chord_valid_lower =
                // Dead zone: chord <= 0 at ±lambd
                chord_at_neg_lambd <= f32::EPSILON
                && chord_at_pos_lambd <= f32::EPSILON
                // Neg piece: chord <= x + bias at -lambd and l
                && chord_at_neg_lambd <= f_at_neg_break + f32::EPSILON
                && chord_at_l <= fl_f32 + f32::EPSILON
                // Pos piece: chord <= x - bias at +lambd and u
                && chord_at_pos_lambd <= f_at_pos_break + f32::EPSILON
                && chord_at_u <= fu_f32 + f32::EPSILON
                // Slope constraint: chord_s <= 1 (pieces have slope 1)
                && chord_s <= 1.0 + f32::EPSILON;

            let (ls, li) = if chord_valid_lower {
                (chord_s, chord_t_lower)
            } else {
                (0.0, next_down_f32(fmin_f64 as f32))
            };

            LinearRelaxation::new(ls, li, us, ui)
        }
    }
}

#[cfg(test)]
mod tests;
