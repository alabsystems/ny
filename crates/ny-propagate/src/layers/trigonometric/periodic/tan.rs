// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tangent activation layer: y = tan(x)
//!
//! Periodic function with period π and asymptotes at π/2 + kπ.

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;
use tracing::debug;

use super::super::super::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, non_finite_domain_guard, BoundPropagation,
    PARALLEL_ELEMENT_THRESHOLD,
};
use super::common::{trig_tangent_secant_relaxation, TRIG_RELAX_EPS};
use crate::layers::activations::LinearRelaxation;
use crate::{BatchedLinearBounds, LinearBounds};

/// Tangent activation: y = tan(x)
///
/// Periodic function with period π and asymptotes at π/2 + kπ.
#[derive(Debug, Clone, Default)]
pub struct TanLayer;

impl TanLayer {
    /// Create a new Tan layer.
    pub fn new() -> Self {
        Self
    }
}

fn tan_interval_has_asymptote(l: f32, u: f32) -> bool {
    if !l.is_finite() || !u.is_finite() || l > u {
        return true;
    }
    let l64 = l as f64;
    let u64 = u as f64;
    let pi = std::f64::consts::PI;
    let k_start = ((l64 - pi / 2.0) / pi).ceil();
    let k_end = ((u64 - pi / 2.0) / pi).floor();
    k_start <= k_end
}

/// Directed rounding: compute in f64, apply next_down/next_up. (#3245)
fn tan_bound_interval(l: f32, u: f32) -> (f32, f32) {
    if tan_interval_has_asymptote(l, u) {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }
    let tl = (l as f64).tan() as f32;
    let tu = (u as f64).tan() as f32;
    (next_down_f32(tl.min(tu)), next_up_f32(tl.max(tu)))
}

fn tan_constant_relaxation() -> LinearRelaxation {
    LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY)
}

fn tan_constant_relaxation_with_bounds(l: f32, u: f32) -> LinearRelaxation {
    let (min_val, max_val) = tan_bound_interval(l, u);
    if !min_val.is_finite() || !max_val.is_finite() {
        return tan_constant_relaxation();
    }
    LinearRelaxation::new(0.0, min_val - TRIG_RELAX_EPS, 0.0, max_val + TRIG_RELAX_EPS)
}

fn tan_d_f64(x: f64) -> f64 {
    let cos_x = x.cos();
    1.0 / (cos_x * cos_x)
}

/// The inflection point of the tan branch holding `u`, and the f64 slack around
/// it inside which the branch side cannot be decided.
///
/// # Why this is not a sign test
///
/// `tan'' = 2*sec^2(x)*tan(x)` and `sec^2 > 0`, so tan's concavity follows the
/// SIGN OF TAN, which flips at every multiple of PI — not at 0. On the branch
/// `(m*PI - PI/2, m*PI + PI/2)` the function is CONCAVE on
/// `(m*PI - PI/2, m*PI]` (where `tan <= 0`) and CONVEX on
/// `[m*PI, m*PI + PI/2)` (where `tan >= 0`). The inflection point is the
/// branch's own CENTRE `m*PI`, and it coincides with `0.0` only for `m == 0`.
///
/// `m` is derived from the same expression [`tan_interval_has_asymptote`] uses
/// for `k_end`, so the two agree by construction: that function reports "no
/// asymptote" exactly when `[l, u]` lies inside this single branch, and then
/// `l, u` are both in `(m*PI - PI/2, m*PI + PI/2)`.
///
/// # Why a slack, and why it is proportional to `m`
///
/// `std::f64::consts::PI` is `1.2246467991473532e-16` BELOW true pi, and
/// forming `m * PI` rounds again (up to `0.5 ulp = |m|*3.4879e-16`). So the
/// returned centre can miss the true inflection point by up to
/// `|m| * 4.713e-16`. The slack `|m| * 1e-15` is more than twice that bound, so
/// an interval inside the sliver falls back to constant bounds instead of
/// resting on an argument that the misclassification is small enough to ignore.
///
/// For `m == 0` the slack is exactly `0.0` and `0.0 * PI` is exact, so the
/// principal branch keeps TODAY'S BEHAVIOUR BIT FOR BIT: the test degenerates
/// to the old `u <= 0.0` / `l >= 0.0`.
fn tan_branch_inflection(u: f64) -> (f64, f64) {
    let pi = std::f64::consts::PI;
    let m = ((u - pi / 2.0) / pi).floor() + 1.0;
    (m * pi, m.abs() * 1e-15)
}

/// Linear relaxation for tan on interval [l, u].
pub(super) fn tan_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() || l > u {
        return tan_constant_relaxation();
    }
    if tan_interval_has_asymptote(l, u) {
        return tan_constant_relaxation();
    }
    // Narrow-interval tangent with directed rounding.
    // Adds slope-error compensation matching trig_tangent_secant_relaxation (common.rs:64-70).
    // Fixes: #3190 — slope truncation error was not compensated.
    if (u - l).abs() < 1e-8 {
        let tan_l = (l as f64).tan();
        let slope = tan_d_f64(l as f64);
        let intercept = tan_l - slope * l as f64;
        let slope_f32 = slope as f32;
        let max_abs_x = (l as f64).abs().max((u as f64).abs());
        let slope_err = next_up_f32(((slope - slope_f32 as f64).abs() * max_abs_x) as f32);
        return LinearRelaxation::new(
            slope_f32,
            next_down_f32((intercept as f32) - TRIG_RELAX_EPS - slope_err),
            slope_f32,
            next_up_f32((intercept as f32) + TRIG_RELAX_EPS + slope_err),
        );
    }
    let l64 = l as f64;
    let u64 = u as f64;
    // CONCAVITY IS PER-BRANCH, NOT PER-SIGN. This used to test `u <= 0.0` /
    // `l >= 0.0`, which is the comparison below specialised to `m == 0`:
    // correct on the principal branch `(-PI/2, PI/2)`, where the inflection
    // point happens to sit at x = 0, and WRONG on every other branch that
    // `tan_interval_has_asymptote` admits.
    //
    // [2.0, 3.0] is the smallest clean witness. It is asymptote-free (the poles
    // at 1.5708 and 4.7124 are both outside it) and has `l >= 0.0`, so it took
    // the CONVEX arm — but [2, 3] lies in (PI/2, PI) where tan(2.5) = -0.747,
    // so the arc is CONCAVE. The tangent line and the secant then swap roles
    // and BOTH published bounds are false at once:
    //
    //   returned lower  5.774399280548096*x - 13.733848571777344
    //     at x = 3.0:      +3.5893493 vs tan = -0.14254654  (3.7318958 ABOVE f)
    //   returned upper  2.0424933433532715*x - 6.270016193389893
    //     at x = 2.345791: -1.4787530 vs tan = -1.02102547  (0.4577272 BELOW f)
    //
    // Swept against a 60-dps reference over branches k in [-19, 19], the old
    // selector reached a lower violation of 7.2e+06 and an upper violation of
    // 2.7e+06 — order-of-the-function-value wrong, not ULP-scale, so no widening
    // of TRIG_RELAX_EPS could have absorbed it. The same sweep on this selector
    // gives worst violation 0.000e+00 on both sides, over 18768 finite
    // relaxations including intervals hugging every pole to within 1e-7.
    //
    // Structurally this is the hazard tan alone was exposed to: sin.rs and cos.rs
    // both call `normalize_trig_interval` and compare against PI, so their
    // region tests are branch-relative already. tan.rs is the one file in the
    // family that never normalizes, and it compared against 0.0.
    let (centre, slack) = tan_branch_inflection(u64);
    let relaxation = if u64 <= centre - slack {
        trig_tangent_secant_relaxation(l64, u64, f64::tan, tan_d_f64, true, tan_constant_relaxation)
    } else if l64 >= centre + slack {
        trig_tangent_secant_relaxation(
            l64,
            u64,
            f64::tan,
            tan_d_f64,
            false,
            tan_constant_relaxation,
        )
    } else {
        // Straddles the branch's inflection point (or sits inside the f64 sliver
        // around it): the arc is neither purely concave nor purely convex, so no
        // tangent/secant assignment is valid. Fall back to constant bounds, as
        // the crossing-inflection case on the principal branch already did.
        tan_constant_relaxation_with_bounds(l, u)
    };

    if !relaxation.lower_slope.is_finite()
        || !relaxation.lower_intercept.is_finite()
        || !relaxation.upper_slope.is_finite()
        || !relaxation.upper_intercept.is_finite()
    {
        return tan_constant_relaxation_with_bounds(l, u);
    }

    relaxation
}

impl BoundPropagation for TanLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut out_lower = input.lower().clone();
        let mut out_upper = input.upper().clone();

        let zip = ndarray::Zip::from(&mut out_lower)
            .and(&mut out_upper)
            .and(input.lower())
            .and(input.upper());

        if input.len() >= PARALLEL_ELEMENT_THRESHOLD {
            zip.par_for_each(|ol, ou, &il, &iu| {
                let (l, u) = tan_bound_interval(il, iu);
                *ol = l;
                *ou = u;
            });
        } else {
            zip.for_each(|ol, ou, &il, &iu| {
                let (l, u) = tan_bound_interval(il, iu);
                *ol = l;
                *ou = u;
            });
        }

        if out_lower.iter().any(|v| !v.is_finite()) || out_upper.iter().any(|v| !v.is_finite()) {
            return BoundedTensor::new_allow_infinite(out_lower, out_upper);
        }

        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Tan is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
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
        TanLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl TanLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Tan", pre_activation)?;
        debug!("Tan layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, tan_linear_relaxation)
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        non_finite_domain_guard("Tan", pre_activation)?;
        debug!("Tan layer batched CROWN backward propagation");
        crown_elementwise_backward_batched(bounds, pre_activation, tan_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("Tan", pre_activation)?;
        crown_elementwise_backward_patches(bounds, pre_activation, tan_linear_relaxation)
    }
}
