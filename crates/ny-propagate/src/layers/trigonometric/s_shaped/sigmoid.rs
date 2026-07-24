// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sigmoid activation layer for bound propagation.

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

/// Sigmoid activation: y = 1 / (1 + exp(-x))
///
/// Monotonically increasing function with range (0, 1).
/// Properties:
/// - sigmoid(0) = 0.5
/// - sigmoid(-x) = 1 - sigmoid(x)
/// - Derivative: sigmoid(x) * (1 - sigmoid(x))
#[derive(Debug, Clone, Default)]
pub struct SigmoidLayer;

impl SigmoidLayer {
    /// Create a new Sigmoid layer.
    pub fn new() -> Self {
        Self
    }
}

/// Compute sigmoid bound interval for [l, u].
/// Since sigmoid is monotonically increasing: sigmoid(l) <= sigmoid(x) <= sigmoid(u).
/// Directed rounding: compute in f64, apply next_down/next_up. (#3245)
fn sigmoid_bound_interval(l: f32, u: f32) -> (f32, f32) {
    // Range clamp: 0 < sigmoid(x) < 1 for all finite x. Directed rounding can
    // push past 0 or 1 for extreme inputs (e.g., sigmoid(-1000) → 0 → -1e-45). (#3316)
    (
        next_down_f32(sigmoid_f64(l as f64) as f32).max(0.0),
        next_up_f32(sigmoid_f64(u as f64) as f32).min(1.0),
    )
}

/// Sigmoid function in f64 precision.
/// Re-exported by the `s_shaped` facade for use by softplus.
pub(in crate::layers::trigonometric) fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_d_f64(x: f64) -> f64 {
    let s = sigmoid_f64(x);
    s * (1.0 - s)
}

fn sigmoid_constant_relaxation() -> LinearRelaxation {
    LinearRelaxation::constant(-S_SHAPED_RELAX_EPS, 1.0 + S_SHAPED_RELAX_EPS)
}

pub(crate) fn sigmoid_crossing_default_tangents(l: f32, u: f32) -> (f32, f32) {
    static TABLES: OnceLock<SShapedPrecomputeTables> = OnceLock::new();
    let tables = TABLES.get_or_init(|| SShapedPrecomputeTables::new(sigmoid_f64, sigmoid_d_f64));
    (tables.lower_tangent(u, l), tables.upper_tangent(l, u))
}

/// Linear relaxation for sigmoid on interval [l, u].
pub(crate) fn sigmoid_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    static TABLES: OnceLock<SShapedPrecomputeTables> = OnceLock::new();
    let tables = TABLES.get_or_init(|| SShapedPrecomputeTables::new(sigmoid_f64, sigmoid_d_f64));
    s_shaped_linear_relaxation(
        l,
        u,
        sigmoid_f64,
        sigmoid_d_f64,
        tables,
        sigmoid_constant_relaxation,
    )
}

pub(super) fn sigmoid_linear_relaxation_with_alpha(
    l: f32,
    u: f32,
    alpha: MonotoneSShapedPathAlpha,
) -> LinearRelaxation {
    s_shaped_linear_relaxation_with_alpha(
        l,
        u,
        sigmoid_f64,
        sigmoid_d_f64,
        sigmoid_constant_relaxation,
        alpha,
    )
}

impl BoundPropagation for SigmoidLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        ibp_bound_interval_parallel(input, sigmoid_bound_interval)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Sigmoid is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
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
        SigmoidLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

impl SigmoidLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Sigmoid", pre_activation)?;
        debug!("Sigmoid layer CROWN backward propagation with pre-activation bounds");
        crown_elementwise_backward(bounds, pre_activation, sigmoid_linear_relaxation)
    }

    /// Dense alpha-CROWN backward with per-neuron tangent-point controls.
    pub(crate) fn propagate_linear_with_alpha(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
        alpha: &MonotoneSShapedAlpha,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Sigmoid", pre_activation)?;
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
            |l, u, i| sigmoid_linear_relaxation_with_alpha(l, u, alpha.lower_path_alpha(i)),
            |l, u, i| sigmoid_linear_relaxation_with_alpha(l, u, alpha.upper_path_alpha(i)),
        )
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        non_finite_domain_guard("Sigmoid", pre_activation)?;
        debug!("Sigmoid layer batched CROWN backward propagation");
        crown_elementwise_backward_batched(bounds, pre_activation, sigmoid_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("Sigmoid", pre_activation)?;
        crown_elementwise_backward_patches(bounds, pre_activation, sigmoid_linear_relaxation)
    }
}
