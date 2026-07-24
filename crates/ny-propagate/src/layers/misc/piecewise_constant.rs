// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Piecewise-constant layers (Floor, Ceil, Round, Sign) for bound propagation.
//!
//! These layers share identical structure — they differ only in the
//! function applied to each element. The CROWN relaxation uses
//! slope = 0 with constant intercepts.

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use tracing::debug;

use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, non_finite_domain_guard, BoundPropagation,
};
use crate::{BatchedLinearBounds, LinearBounds};

/// Generates a piecewise-constant layer (Floor, Ceil, Round, Sign).
///
/// Two forms:
/// - Simple: `piecewise_constant_layer!(Name, method, "OpStr")` — IBP applies
///   `method()` independently to lower and upper (Floor, Ceil, Round).
/// - Custom IBP: `piecewise_constant_layer!(Name, "OpStr", ibp_fn, crown_fn)` —
///   IBP uses a `(l, u) -> (out_l, out_u)` function for paired bounds analysis,
///   and CROWN uses a `(l, u) -> LinearRelaxation`
///   function (Sign).
macro_rules! piecewise_constant_layer {
    // Simple form: IBP applies a scalar method independently to lower and upper.
    (
        $(#[$meta:meta])*
        $name:ident, $fn_name:ident, $op_str:expr
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Default)]
        pub struct $name;

        impl $name {
            /// Create a new layer.
            pub fn new() -> Self {
                Self
            }
        }

        impl BoundPropagation for $name {
            #[inline]
            fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
                let lower = input.lower().mapv(|x: f32| x.$fn_name());
                let upper = input.upper().mapv(|x: f32| x.$fn_name());
                BoundedTensor::new(lower, upper)
            }

            piecewise_constant_layer!(@shared_trait_methods $name, $op_str);
        }

        piecewise_constant_layer!(@crown_methods $name, $op_str,
            |l: f32, u: f32| { LinearRelaxation::new(0.0, l.$fn_name(), 0.0, u.$fn_name()) }
        );
    };

    // Custom IBP form: IBP uses paired bounds function, CROWN uses explicit relaxation.
    (
        $(#[$meta:meta])*
        $name:ident, $op_str:expr,
        ibp_bounds_fn: $ibp_fn:expr,
        crown_relaxation_fn: $crown_fn:expr
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Default)]
        pub struct $name;

        impl $name {
            /// Create a new layer.
            pub fn new() -> Self {
                Self
            }
        }

        impl BoundPropagation for $name {
            #[inline]
            fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
                let ibp_fn: fn(f32, f32) -> (f32, f32) = $ibp_fn;
                let mut out_lower = ArrayD::zeros(IxDyn(input.shape()));
                let mut out_upper = ArrayD::zeros(IxDyn(input.shape()));
                for (idx, &lb) in input.lower().indexed_iter() {
                    let ub = input.upper()[idx.clone()];
                    let (ol, ou) = ibp_fn(lb, ub);
                    out_lower[idx.clone()] = ol;
                    out_upper[idx] = ou;
                }
                BoundedTensor::new(out_lower, out_upper)
            }

            piecewise_constant_layer!(@shared_trait_methods $name, $op_str);
        }

        piecewise_constant_layer!(@crown_methods $name, $op_str, $crown_fn);
    };

    // Shared trait methods (propagate_linear, requires_pre_activation_bounds, etc.)
    (@shared_trait_methods $name:ident, $op_str:expr) => {
        #[inline]
        fn propagate_linear<'a>(
            &self,
            _bounds: &'a LinearBounds,
        ) -> Result<Cow<'a, LinearBounds>> {
            Err(NyError::UnsupportedOp(
                concat!(
                    $op_str,
                    " is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
                )
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
            $name::propagate_linear_with_bounds(self, bounds, pre_activation)
        }
    };

    // CROWN inherent methods (propagate_linear_with_bounds, batched variant).
    (@crown_methods $name:ident, $op_str:expr, $crown_relaxation:expr) => {
        impl $name {
            /// CROWN backward propagation with pre-activation bounds.
            ///
            /// Piecewise constant (step) function with zero derivative.
            /// The linear relaxation uses constant bounds (slope = 0):
            /// - Lower bound: y >= f(l) (constant)
            /// - Upper bound: y <= f(u) (constant)
            ///
            /// This is equivalent to IBP but expressed in the CROWN linear form.
            pub fn propagate_linear_with_bounds(
                &self,
                bounds: &LinearBounds,
                pre_activation: &BoundedTensor,
            ) -> Result<LinearBounds> {
                non_finite_domain_guard($op_str, pre_activation)?;
                debug!(
                    concat!($op_str, " layer CROWN backward propagation with pre-activation bounds")
                );
                crown_elementwise_backward(bounds, pre_activation, $crown_relaxation)
            }

            /// Batched CROWN backward propagation with pre-activation bounds.
            pub fn propagate_linear_batched_with_bounds(
                &self,
                bounds: &BatchedLinearBounds,
                pre_activation: &BoundedTensor,
            ) -> Result<BatchedLinearBounds> {
                non_finite_domain_guard($op_str, pre_activation)?;
                debug!(concat!($op_str, " layer batched CROWN backward propagation"));
                crown_elementwise_backward_batched(bounds, pre_activation, $crown_relaxation)
            }

            /// Patches CROWN backward propagation with pre-activation bounds.
            /// Part of #2613 Phase 2: generic activation Patches support.
            pub(crate) fn propagate_patches_with_bounds(
                &self,
                bounds: &crate::bounds::patches::PatchesLinearBounds,
                pre_activation: &BoundedTensor,
            ) -> Result<crate::bounds::patches::CrownBounds> {
                non_finite_domain_guard($op_str, pre_activation)?;
                crown_elementwise_backward_patches(bounds, pre_activation, $crown_relaxation)
            }
        }
    };
}

piecewise_constant_layer!(
    /// Floor layer: y = floor(x) - rounds towards negative infinity.
    ///
    /// Common in quantization and index computation.
    FloorLayer, floor, "Floor"
);

piecewise_constant_layer!(
    /// Ceil layer: y = ceil(x) - rounds towards positive infinity.
    ///
    /// Common in quantization and index computation.
    CeilLayer, ceil, "Ceil"
);

piecewise_constant_layer!(
    /// Round layer: y = round(x) - rounds to nearest integer (0.5 rounds away from zero).
    ///
    /// Common in quantization and rounding operations.
    /// Uses Rust's round() (round half away from zero).
    RoundLayer, round, "Round"
);

piecewise_constant_layer!(
    /// Trunc layer: y = trunc(x) - rounds toward zero (fractional part discarded).
    ///
    /// Lowered from ONNX Cast with an integer target dtype on non-constant
    /// input (op semantics: float->int casts truncate). Identity would NOT
    /// enclose the true output (trunc(0.5)=0 not in [0.5, 62]) — see the
    /// cctsdb_yolo_2023 design. trunc is monotone non-decreasing, so applying
    /// it independently to lower/upper yields the exact interval hull.
    TruncLayer, trunc, "Trunc"
);

/// Compute sign bounds for an interval [l, u].
///
/// Returns (lower_bound, upper_bound) of sign(x) for x in [l, u].
/// Output is in {-1, 0, 1}. Bounds depend on whether interval crosses zero.
fn sign_interval_bounds(l: f32, u: f32) -> (f32, f32) {
    if l > 0.0 {
        (1.0, 1.0) // Entire interval is positive
    } else if u < 0.0 {
        (-1.0, -1.0) // Entire interval is negative
    } else if l == 0.0 && u == 0.0 {
        (0.0, 0.0) // Exactly zero
    } else if l == 0.0 {
        (0.0, 1.0) // lb == 0, ub > 0: could be 0 or 1
    } else if u == 0.0 {
        (-1.0, 0.0) // lb < 0, ub == 0: could be -1 or 0
    } else {
        (-1.0, 1.0) // Interval spans zero: could be -1, 0, or 1
    }
}

/// Sign CROWN relaxation with non-zero slopes for boundary cases.
///
/// Reference: alpha-beta-CROWN `BoundSign.bound_relax` (relu.py:824-853).
///
/// | Interval         | Lower bound         | Upper bound         |
/// |------------------|---------------------|---------------------|
/// | `l > 0`          | `y = 1` (exact)     | `y = 1` (exact)     |
/// | `u < 0`          | `y = -1` (exact)    | `y = -1` (exact)    |
/// | `l == 0, u == 0` | `y = 0` (exact)     | `y = 0` (exact)     |
/// | `l == 0, u > 0`  | `y = x/u` (slope)   | `y = 1` (constant)  |
/// | `l < 0, u == 0`  | `y = -1` (constant) | `y = -x/l` (slope)  |
/// | `l < 0, u > 0`   | `y = -1` (constant) | `y = 1` (constant)  |
///
/// Soundness: for `l == 0, u > 0`, lower line `y = x/u` passes through
/// `(0, 0)` and `(u, 1)`. `sign(x) >= x/u` for all `x in [0, u]` because
/// `sign(x) in {0, 1}` and `x/u in [0, 1]`, with equality at endpoints.
fn sign_crown_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if l > 0.0 {
        LinearRelaxation::new(0.0, 1.0, 0.0, 1.0)
    } else if u < 0.0 {
        LinearRelaxation::new(0.0, -1.0, 0.0, -1.0)
    } else if l == 0.0 && u == 0.0 {
        LinearRelaxation::new(0.0, 0.0, 0.0, 0.0)
    } else if l == 0.0 {
        // l == 0, u > 0: lower slope through (0,0)→(u,1), upper constant 1
        let lower_slope = 1.0 / u.max(1e-8);
        LinearRelaxation::new(lower_slope, 0.0, 0.0, 1.0)
    } else if u == 0.0 {
        // l < 0, u == 0: lower constant -1, upper slope through (l,-1)→(0,0)
        let upper_slope = -1.0 / l.min(-1e-8);
        LinearRelaxation::new(0.0, -1.0, upper_slope, 0.0)
    } else {
        // Spans zero: lower = -1, upper = 1 (no tighter linear bound exists)
        LinearRelaxation::new(0.0, -1.0, 0.0, 1.0)
    }
}

piecewise_constant_layer!(
    /// Sign layer: y = -1 if x < 0, 0 if x == 0, 1 if x > 0.
    ///
    /// Useful for conditional logic and gradient sign analysis.
    /// Unlike Floor/Ceil/Round, Sign's IBP requires paired bounds analysis
    /// since the output depends on the interval's position relative to zero.
    SignLayer, "Sign",
    ibp_bounds_fn: sign_interval_bounds,
    crown_relaxation_fn: sign_crown_relaxation
);
