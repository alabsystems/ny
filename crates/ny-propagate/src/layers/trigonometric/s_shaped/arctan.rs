// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Arctan activation layer for bound propagation.

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::borrow::Cow;
use std::sync::OnceLock;
use tracing::debug;

use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, ibp_bound_interval_parallel, non_finite_domain_guard,
    BoundPropagation,
};
use crate::{BatchedLinearBounds, LinearBounds};

use super::shared::{s_shaped_linear_relaxation, SShapedPrecomputeTables, S_SHAPED_RELAX_EPS};

/// Arctangent activation: y = atan(x)
///
/// Monotonically increasing S-shaped function with range (-π/2, π/2).
#[derive(Debug, Clone, Default)]
pub struct ArctanLayer;

impl ArctanLayer {
    /// Create a new Arctan layer.
    pub fn new() -> Self {
        Self
    }
}

/// Directed rounding: compute in f64, apply next_down/next_up. (#3245)
fn arctan_bound_interval(l: f32, u: f32) -> (f32, f32) {
    // Range clamp: -π/2 < arctan(x) < π/2 for all finite x. Directed rounding can
    // push past ±π/2 for extreme inputs. (#3316)
    (
        next_down_f32((l as f64).atan() as f32).max(-std::f32::consts::FRAC_PI_2),
        next_up_f32((u as f64).atan() as f32).min(std::f32::consts::FRAC_PI_2),
    )
}

fn arctan_f64(x: f64) -> f64 {
    x.atan()
}

fn arctan_d_f64(x: f64) -> f64 {
    1.0 / (1.0 + x * x)
}

fn arctan_constant_relaxation() -> LinearRelaxation {
    let half_pi = std::f32::consts::FRAC_PI_2;
    LinearRelaxation::constant(-half_pi - S_SHAPED_RELAX_EPS, half_pi + S_SHAPED_RELAX_EPS)
}

pub(super) fn arctan_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    static TABLES: OnceLock<SShapedPrecomputeTables> = OnceLock::new();
    let tables = TABLES.get_or_init(|| SShapedPrecomputeTables::new(arctan_f64, arctan_d_f64));
    let relaxation = s_shaped_linear_relaxation(
        l,
        u,
        arctan_f64,
        arctan_d_f64,
        tables,
        arctan_constant_relaxation,
    );

    if !relaxation.lower_slope.is_finite()
        || !relaxation.lower_intercept.is_finite()
        || !relaxation.upper_slope.is_finite()
        || !relaxation.upper_intercept.is_finite()
    {
        return arctan_constant_relaxation();
    }

    relaxation
}

impl BoundPropagation for ArctanLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        ibp_bound_interval_parallel(input, arctan_bound_interval)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Arctan is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
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
        ArctanLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl ArctanLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Arctan", pre_activation)?;
        debug!("Arctan layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, arctan_linear_relaxation)
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        non_finite_domain_guard("Arctan", pre_activation)?;
        debug!("Arctan layer batched CROWN backward propagation");
        crown_elementwise_backward_batched(bounds, pre_activation, arctan_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("Arctan", pre_activation)?;
        crown_elementwise_backward_patches(bounds, pre_activation, arctan_linear_relaxation)
    }
}
