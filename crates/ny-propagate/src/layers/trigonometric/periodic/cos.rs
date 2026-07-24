// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cosine activation layer: y = cos(x)
//!
//! Periodic function with range [-1, 1] and period 2π.
//! Used in positional encodings for transformers.

use ny_core::{NyError, Result, VerificationSoundnessMode};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;
use std::f64::consts::PI;
use tracing::debug;

use super::super::super::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, ibp_bound_interval_parallel, non_finite_domain_guard,
    BoundPropagation,
};
use super::common::{
    constant_bounds_from_output, normalize_trig_interval, trig_constant_relaxation,
    trig_tangent_secant_relaxation, TRIG_RELAX_EPS,
};
use crate::layers::activations::LinearRelaxation;
use crate::{BatchedLinearBounds, LinearBounds};

/// Cosine activation: y = cos(x)
///
/// Periodic function with range [-1, 1] and period 2π.
/// Used in positional encodings for transformers.
#[derive(Debug, Clone, Default)]
pub struct CosLayer {
    /// Force conservative CROWN relaxations using IBP-derived constant bounds.
    pub sound: bool,
}

impl CosLayer {
    /// Create a new Cos layer with heuristic CROWN relaxation by default.
    pub fn new() -> Self {
        Self { sound: false }
    }

    /// Enable or disable conservative (IBP-only) CROWN relaxations.
    pub fn with_sound_mode(mut self, enabled: bool) -> Self {
        self.sound = enabled;
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
}

/// Compute cos bound interval for [l, u].
pub(super) fn cos_bound_interval(l: f32, u: f32) -> (f32, f32) {
    if !l.is_finite() || !u.is_finite() || l > u {
        return (-1.0, 1.0);
    }

    // Directed rounding: compute in f64, apply next_down/next_up for endpoint
    // evaluations. Exact extrema (-1.0, 1.0) don't need rounding. (#3245)
    let cl = (l as f64).cos() as f32;
    let cu = (u as f64).cos() as f32;
    // Range clamp: cos(x) ∈ [-1, 1]. Directed rounding can push past ±1
    // for inputs near extrema (e.g., cos(ε) ≈ 1 → next_up > 1). (#3316)
    let mut min_val = next_down_f32(cl.min(cu)).max(-1.0);
    let mut max_val = next_up_f32(cl.max(cu)).min(1.0);

    let pi64 = PI;
    let l64 = l as f64;
    let u64 = u as f64;
    let k_max_start = (l64 / (2.0 * pi64)).ceil();
    let k_max_end = (u64 / (2.0 * pi64)).floor();
    if k_max_start <= k_max_end {
        max_val = 1.0;
    }

    let k_min_start = ((l64 - pi64) / (2.0 * pi64)).ceil();
    let k_min_end = ((u64 - pi64) / (2.0 * pi64)).floor();
    if k_min_start <= k_min_end {
        min_val = -1.0;
    }

    (min_val, max_val)
}

impl BoundPropagation for CosLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        ibp_bound_interval_parallel(input, cos_bound_interval)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Cos is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
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
        CosLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

/// Linear relaxation for cos on interval [l, u].
pub(super) fn cos_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() || l > u {
        return trig_constant_relaxation();
    }

    // Narrow-interval tangent with directed rounding.
    // Adds slope-error compensation matching trig_tangent_secant_relaxation (common.rs:64-70).
    // Fixes: #3190 — slope truncation error was not compensated.
    if (u - l).abs() < 1e-8 {
        let cos_l = (l as f64).cos();
        let slope = -(l as f64).sin();
        let intercept = cos_l - slope * l as f64;
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
    let (l_norm, u_norm) = match normalize_trig_interval(l64, u64) {
        Some(interval) => interval,
        None => return trig_constant_relaxation(),
    };

    let half_pi = 0.5 * PI;
    let three_half_pi = 1.5 * PI;
    if (l_norm < half_pi && u_norm > half_pi) || (l_norm < three_half_pi && u_norm > three_half_pi)
    {
        return trig_constant_relaxation();
    }

    if u_norm <= half_pi || l_norm >= three_half_pi {
        trig_tangent_secant_relaxation(
            l64,
            u64,
            f64::cos,
            |x| -x.sin(),
            true,
            trig_constant_relaxation,
        )
    } else if l_norm >= half_pi && u_norm <= three_half_pi {
        trig_tangent_secant_relaxation(
            l64,
            u64,
            f64::cos,
            |x| -x.sin(),
            false,
            trig_constant_relaxation,
        )
    } else {
        trig_constant_relaxation()
    }
}

impl CosLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Cos", pre_activation)?;
        debug!("Cos layer CROWN backward propagation with pre-activation bounds");
        if self.sound {
            debug!("Cos sound mode: using IBP-derived constant bounds");
            let output_bounds = self.propagate_ibp(pre_activation)?;
            return constant_bounds_from_output(bounds, &output_bounds);
        }
        crown_elementwise_backward(bounds, pre_activation, cos_linear_relaxation)
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    ///
    /// Sound mode is not supported in batched CROWN — returns UnsupportedOp
    /// so the dispatch fallback uses unbatched CROWN instead.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        non_finite_domain_guard("Cos", pre_activation)?;
        if self.sound {
            return Err(NyError::UnsupportedOp(
                "Cos batched CROWN not supported in sound mode".to_string(),
            ));
        }
        debug!("Cos layer batched CROWN backward propagation");
        crown_elementwise_backward_batched(bounds, pre_activation, cos_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("Cos", pre_activation)?;
        if self.sound {
            return Err(NyError::NumericalInstability(
                "Cos Patches CROWN not supported in sound mode — falling back to Dense".into(),
            ));
        }
        crown_elementwise_backward_patches(bounds, pre_activation, cos_linear_relaxation)
    }
}
