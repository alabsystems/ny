// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Traits for bound propagation layers.
//!
//! Defines the core `BoundPropagation` trait and the optional `PatchesPropagation`
//! extension for layers that natively support Patches-mode backward.

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;

use crate::LinearBounds;

/// Trait for layers that support bound propagation.
pub trait BoundPropagation {
    /// Propagate bounds through the layer using IBP.
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor>;

    /// Propagate linear bounds through the layer (for CROWN).
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>>;

    /// Whether this layer needs pre-activation bounds for CROWN backward propagation.
    ///
    /// Linear layers (Linear, Conv, AddConstant, Reshape, etc.) return `false` — their
    /// CROWN backward pass is an exact linear transformation that doesn't need input bounds.
    ///
    /// Nonlinear layers (ReLU, Sigmoid, Exp, Softmax, etc.) return `true` — they need
    /// pre-activation bounds to compute a linear relaxation.
    ///
    /// Matches alpha-beta-CROWN's `requires_input_bounds` pattern.
    /// Reference: `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/operators/base.py:143`
    fn requires_pre_activation_bounds(&self) -> bool {
        false
    }

    /// CROWN backward propagation with pre-activation bounds (for nonlinear layers).
    ///
    /// Override this for nonlinear layers that need pre-activation bounds to compute
    /// linear relaxation slopes and intercepts. The default returns `UnsupportedOp`.
    ///
    /// Linear layers should NOT override this — they use `propagate_linear()` instead.
    fn propagate_linear_with_bounds(
        &self,
        _bounds: &LinearBounds,
        _pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        Err(NyError::UnsupportedOp(format!(
            "{} does not implement propagate_linear_with_bounds",
            std::any::type_name::<Self>()
        )))
    }

    /// CROWN backward propagation with optional pre-activation bounds.
    ///
    /// This is the unified backward method. Linear layers delegate to
    /// `propagate_linear()`. Nonlinear layers (where `requires_pre_activation_bounds()`
    /// returns `true`) delegate to `propagate_linear_with_bounds()`.
    ///
    /// Most layers should NOT override this — override `propagate_linear_with_bounds()`
    /// or `propagate_linear()` instead. The default handles the dispatch.
    ///
    /// Reference: alpha-beta-CROWN `bound_backward` at
    /// `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/operators/base.py:401-419`
    fn propagate_crown_backward(
        &self,
        bounds: &LinearBounds,
        pre_activation: Option<&BoundedTensor>,
    ) -> Result<LinearBounds> {
        if self.requires_pre_activation_bounds() {
            let pre_act = pre_activation.ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "{} requires pre-activation bounds for CROWN backward",
                    std::any::type_name::<Self>()
                ))
            })?;
            self.propagate_linear_with_bounds(bounds, pre_act)
        } else {
            // Linear layers: delegate to propagate_linear
            self.propagate_linear(bounds).map(|cow| cow.into_owned())
        }
    }
}

/// Optional extension for layers that natively support Patches backward.
///
/// Layers that DON'T implement this trait get automatic Dense fallback
/// via the default implementation in the backward engine.
///
/// Linear layers implement `propagate_patches()`. Nonlinear layers (activations)
/// implement `propagate_patches_with_bounds()` to receive pre-activation bounds
/// for computing relaxation slopes.
///
/// Reference: designs/2026-02-28-patches-mode-wrapper-enum-design.md
/// Part of #2613
pub(crate) trait PatchesPropagation {
    /// CROWN backward with Patches coefficients (for LINEAR layers).
    /// Returns either Patches (preserving sparse structure) or Dense
    /// (if the layer terminates Patches mode, e.g., Linear).
    fn propagate_patches(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
    ) -> Result<crate::bounds::patches::CrownBounds>;

    /// CROWN backward with Patches coefficients + pre-activation bounds
    /// (for NONLINEAR layers like ReLU, activations).
    ///
    /// Default: delegates to `propagate_patches()` (ignoring pre-activation).
    /// Nonlinear layers MUST override this to use pre-activation bounds
    /// for computing linear relaxation slopes.
    fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        _pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        self.propagate_patches(bounds)
    }
}
