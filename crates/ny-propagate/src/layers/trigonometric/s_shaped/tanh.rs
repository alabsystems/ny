// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tanh activation layer for bound propagation.

use crate::bounds::{MonotoneSShapedAlpha, MonotoneSShapedPathAlpha};
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

use super::shared::{
    crown_elementwise_backward_dual_indexed, s_shaped_linear_relaxation,
    s_shaped_linear_relaxation_with_alpha, SShapedPrecomputeTables, S_SHAPED_RELAX_EPS,
};

/// Hyperbolic tangent activation: y = tanh(x)
///
/// Monotonically increasing function with range (-1, 1).
/// Properties:
/// - tanh(0) = 0
/// - tanh(-x) = -tanh(x) (odd function)
/// - Derivative: sech²(x) = 1 - tanh²(x)
#[derive(Debug, Clone, Default)]
pub struct TanhLayer;

impl TanhLayer {
    /// Create a new Tanh layer.
    pub fn new() -> Self {
        Self
    }
}

/// Compute tanh bound interval for [l, u].
/// Since tanh is monotonically increasing: tanh(l) <= tanh(x) <= tanh(u) for all x in [l, u].
/// Directed rounding: compute in f64, apply next_down/next_up to guarantee
/// lower bounds round DOWN and upper bounds round UP. (#3245)
fn tanh_bound_interval(l: f32, u: f32) -> (f32, f32) {
    // Range clamp: -1 < tanh(x) < 1 for all finite x. Directed rounding can
    // push past ±1 for extreme inputs (e.g., tanh(1000) → 1 → next_up > 1). (#3316)
    (
        next_down_f32((l as f64).tanh() as f32).max(-1.0),
        next_up_f32((u as f64).tanh() as f32).min(1.0),
    )
}

pub(super) fn tanh_f64(x: f64) -> f64 {
    x.tanh()
}

pub(super) fn tanh_d_f64(x: f64) -> f64 {
    let t = x.tanh();
    1.0 - t * t
}

fn tanh_constant_relaxation() -> LinearRelaxation {
    LinearRelaxation::constant(-1.0 - S_SHAPED_RELAX_EPS, 1.0 + S_SHAPED_RELAX_EPS)
}

pub(crate) fn tanh_crossing_default_tangents(l: f32, u: f32) -> (f32, f32) {
    static TABLES: OnceLock<SShapedPrecomputeTables> = OnceLock::new();
    let tables = TABLES.get_or_init(|| SShapedPrecomputeTables::new(tanh_f64, tanh_d_f64));
    (tables.lower_tangent(u, l), tables.upper_tangent(l, u))
}

/// Linear relaxation for tanh on interval [l, u].
/// Since tanh is monotonically increasing and S-shaped (concave for x > 0, convex for x < 0),
/// we use precomputed tangent points and chord/tangent case splits.
pub(crate) fn tanh_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    static TABLES: OnceLock<SShapedPrecomputeTables> = OnceLock::new();
    let tables = TABLES.get_or_init(|| SShapedPrecomputeTables::new(tanh_f64, tanh_d_f64));
    s_shaped_linear_relaxation(l, u, tanh_f64, tanh_d_f64, tables, tanh_constant_relaxation)
}

pub(super) fn tanh_linear_relaxation_with_alpha(
    l: f32,
    u: f32,
    alpha: MonotoneSShapedPathAlpha,
) -> LinearRelaxation {
    s_shaped_linear_relaxation_with_alpha(
        l,
        u,
        tanh_f64,
        tanh_d_f64,
        tanh_constant_relaxation,
        alpha,
    )
}

impl BoundPropagation for TanhLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        ibp_bound_interval_parallel(input, tanh_bound_interval)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Tanh is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
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
        TanhLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl TanhLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Tanh", pre_activation)?;
        debug!("Tanh layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, tanh_linear_relaxation)
    }

    /// Dense alpha-CROWN backward with per-neuron tangent-point controls.
    pub(crate) fn propagate_linear_with_alpha(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &MonotoneSShapedAlpha,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Tanh", pre_activation)?;
        // `flatten().len()` allocated two full element-copy Vecs solely to read
        // the count; `len()` returns the same total element count (flatten
        // preserves element count by construction) with no allocation.
        let pre_len = pre_activation.len();
        if alpha.len() != pre_len {
            return Err(NyError::ShapeMismatch {
                expected: vec![pre_len],
                got: vec![alpha.len()],
            });
        }
        crown_elementwise_backward_dual_indexed(
            bounds,
            pre_activation,
            |l, u, i| tanh_linear_relaxation_with_alpha(l, u, alpha.lower_path_alpha(i)),
            |l, u, i| tanh_linear_relaxation_with_alpha(l, u, alpha.upper_path_alpha(i)),
        )
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        non_finite_domain_guard("Tanh", pre_activation)?;
        debug!("Tanh layer batched CROWN backward propagation");
        crown_elementwise_backward_batched(bounds, pre_activation, tanh_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("Tanh", pre_activation)?;
        crown_elementwise_backward_patches(bounds, pre_activation, tanh_linear_relaxation)
    }
}
