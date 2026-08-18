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

            /// TEMPORARY (envelope audit): exposes the EXACT relaxation closure that
            /// `propagate_linear_with_bounds` hands to `crown_elementwise_backward`.
            #[cfg(test)]
            #[allow(dead_code)] // Not every macro instantiation has a direct audit case.
            pub(crate) fn crown_relaxation_for_audit(l: f32, u: f32) -> LinearRelaxation {
                let f: fn(f32, f32) -> LinearRelaxation = $crown_relaxation;
                f(l, u)
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

/// The smallest value ONNX `Round` can take at `x`, over BOTH tie conventions.
///
/// ONNX-13 §Round specifies rounding half to EVEN; Rust's `f32::round` is half
/// AWAY FROM ZERO. They agree everywhere except exact half-integers, where they
/// differ by exactly 1 — `Round(-2.5)` is `-2` by the spec and `-3` in Rust.
/// Bounding only the Rust convention is what the field-wide audit caught: 802
/// violations, the worst a full unit, with the upper envelope published at `-3`
/// over an interval whose true ONNX maximum is `-2`. A whole-unit gap in the
/// false-proof direction on a step function is not a rounding artefact.
///
/// Rather than pick a convention and hope the model's runtime agrees, the bounds
/// enclose BOTH. Outside half-integers that is not a widening at all — the two
/// functions are equal there, so `min == max` and the envelope is unchanged. It
/// costs one unit of width only where the semantics are genuinely ambiguous, and
/// Round appears in NY's corpus mostly on QDQ paths, where operands sit
/// deliberately close to the half-integers this disagreement lives on.
#[inline]
fn round_lower_over_tie_conventions(x: f32) -> f32 {
    x.round().min(round_half_to_even_f32(x))
}

/// The largest value ONNX `Round` can take at `x`, over both tie conventions.
/// See [`round_lower_over_tie_conventions`].
#[inline]
fn round_upper_over_tie_conventions(x: f32) -> f32 {
    x.round().max(round_half_to_even_f32(x))
}

/// ONNX-13 `Round`: ties go to the EVEN neighbour.
#[inline]
fn round_half_to_even_f32(x: f32) -> f32 {
    x.round_ties_even()
}

piecewise_constant_layer!(
    /// Round layer: y = round(x) — nearest integer, with the tie case bounded
    /// over BOTH ONNX's half-to-even and Rust's half-away-from-zero.
    ///
    /// Common in quantization and rounding operations. See
    /// [`round_lower_over_tie_conventions`] for why both are enclosed.
    RoundLayer,
    "Round",
    ibp_bounds_fn: |l: f32, u: f32| {
        (
            round_lower_over_tie_conventions(l),
            round_upper_over_tie_conventions(u),
        )
    },
    crown_relaxation_fn: |l: f32, u: f32| {
        LinearRelaxation::new(
            0.0,
            round_lower_over_tie_conventions(l),
            0.0,
            round_upper_over_tie_conventions(u),
        )
    }
);

piecewise_constant_layer!(TruncKernelLayer, trunc, "Trunc");

/// Destination domain carried by an ONNX floating-point-to-integer Cast.
///
/// ONNX defines such a Cast only while the truncated value is representable in
/// the destination fixed-point type.  The positive endpoint is exclusive:
/// `2^31` is already outside INT32 even though `i32::MAX as f32` rounds to that
/// same value (and likewise for INT64 at `2^63`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegerCastDomain {
    Int32,
    Int64,
}

impl IntegerCastDomain {
    fn label(self) -> &'static str {
        match self {
            Self::Int32 => "INT32",
            Self::Int64 => "INT64",
        }
    }

    fn bounds(self) -> (f64, f64) {
        match self {
            Self::Int32 => (-2.0_f64.powi(31), 2.0_f64.powi(31)),
            Self::Int64 => (-2.0_f64.powi(63), 2.0_f64.powi(63)),
        }
    }

    fn contains_interval(self, lower: f64, upper: f64) -> bool {
        let (minimum, maximum_exclusive) = self.bounds();
        lower.is_finite() && upper.is_finite() && lower >= minimum && upper < maximum_exclusive
    }
}

/// Trunc layer: `y = trunc(x)` (fractional part discarded toward zero).
///
/// A native ONNX `Trunc` is defined on every finite floating-point value and
/// uses [`TruncLayer::new`].  A `Cast(FLOAT32 -> INT32/INT64)` uses one of the
/// target-specific constructors and carries an additional proof obligation:
/// every reachable pre-activation must be finite and inside the destination
/// integer domain.  That obligation is checked at every verdict-bearing IBP,
/// CROWN, Patches, and exact-f64-cell propagation boundary.
///
/// Keeping the target here, rather than merely calling the result a generic
/// `Trunc`, is essential.  ONNX specifies floating-point-to-fixed-point Cast as
/// undefined out of range; selecting Rust's `trunc` result there would choose
/// one behavior and could exclude the deployed runtime's value.
#[derive(Debug, Clone, Default)]
pub struct TruncLayer {
    cast_domain: Option<IntegerCastDomain>,
    kernel: TruncKernelLayer,
}

impl TruncLayer {
    /// Construct a native, unrestricted ONNX `Trunc` layer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct the guarded lowering of `Cast(..., to=INT32)`.
    pub fn for_int32_cast() -> Self {
        Self {
            cast_domain: Some(IntegerCastDomain::Int32),
            kernel: TruncKernelLayer::new(),
        }
    }

    /// Construct the guarded lowering of `Cast(..., to=INT64)`.
    pub fn for_int64_cast() -> Self {
        Self {
            cast_domain: Some(IntegerCastDomain::Int64),
            kernel: TruncKernelLayer::new(),
        }
    }

    fn domain_refusal(domain: IntegerCastDomain, lower: f64, upper: f64) -> NyError {
        let (minimum, maximum_exclusive) = domain.bounds();
        NyError::SoundnessRefusal(format!(
            "Cast to {} requires finite pre-activation bounds inside [{minimum}, \
             {maximum_exclusive}); got [{lower}, {upper}]",
            domain.label(),
        ))
    }

    fn validate_f32_domain(&self, input: &BoundedTensor) -> Result<()> {
        let Some(domain) = self.cast_domain else {
            return Ok(());
        };
        for (&lower, &upper) in input.lower().iter().zip(input.upper().iter()) {
            let (lower, upper) = (f64::from(lower), f64::from(upper));
            if !domain.contains_interval(lower, upper) {
                return Err(Self::domain_refusal(domain, lower, upper));
            }
        }
        Ok(())
    }

    /// Validate the exact-f64 cell evaluator's pre-activation interval.
    ///
    /// That evaluator dispatches directly rather than through
    /// [`BoundPropagation::propagate_ibp`], so it must invoke the same domain
    /// certificate explicitly before applying `f64::trunc`.
    pub(crate) fn validate_f64_domain(
        &self,
        lower: &ArrayD<f64>,
        upper: &ArrayD<f64>,
    ) -> Result<()> {
        let Some(domain) = self.cast_domain else {
            return Ok(());
        };
        for (&lower, &upper) in lower.iter().zip(upper.iter()) {
            if !domain.contains_interval(lower, upper) {
                return Err(Self::domain_refusal(domain, lower, upper));
            }
        }
        Ok(())
    }

    /// Batched CROWN backward propagation with the Cast-domain certificate.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        self.validate_f32_domain(pre_activation)?;
        self.kernel
            .propagate_linear_batched_with_bounds(bounds, pre_activation)
    }

    /// Patches CROWN backward propagation with the Cast-domain certificate.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        self.validate_f32_domain(pre_activation)?;
        self.kernel
            .propagate_patches_with_bounds(bounds, pre_activation)
    }

    /// Dense CROWN backward propagation with the Cast-domain certificate.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        self.validate_f32_domain(pre_activation)?;
        self.kernel
            .propagate_linear_with_bounds(bounds, pre_activation)
    }

    /// Exact relaxation closure used by the envelope audit.
    #[cfg(test)]
    pub(crate) fn crown_relaxation_for_audit(l: f32, u: f32) -> LinearRelaxation {
        TruncKernelLayer::crown_relaxation_for_audit(l, u)
    }
}

impl BoundPropagation for TruncLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.validate_f32_domain(input)?;
        self.kernel.propagate_ibp(input)
    }

    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        self.kernel.propagate_linear(bounds)
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        TruncLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

#[cfg(test)]
mod trunc_cast_domain_tests {
    use super::*;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};

    fn interval(lower: f32, upper: f32) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1]), lower),
            ArrayD::from_elem(IxDyn(&[1]), upper),
        )
        .expect("ordered finite interval")
    }

    fn spatial_interval(lower: f32, upper: f32) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 1, 1]), lower),
            ArrayD::from_elem(IxDyn(&[1, 1, 1]), upper),
        )
        .expect("ordered finite spatial interval")
    }

    #[test]
    fn integer_cast_domains_accept_the_largest_representable_in_range_values() {
        for (layer, exponent) in [
            (TruncLayer::for_int32_cast(), 31),
            (TruncLayer::for_int64_cast(), 63),
        ] {
            let upper_exclusive = 2.0_f32.powi(exponent);
            let upper = ny_tensor::next_down_f32(upper_exclusive);
            let input = interval(-upper_exclusive, upper);
            let output = layer
                .propagate_ibp(&input)
                .expect("the closed interval immediately inside the Cast domain must pass");
            assert_eq!(output.lower()[[0]], -upper_exclusive);
            assert_eq!(output.upper()[[0]], upper);
        }
    }

    #[test]
    fn integer_cast_domains_reject_nan_infinity_and_both_out_of_range_edges() {
        for (layer, domain, exponent) in [
            (TruncLayer::for_int32_cast(), IntegerCastDomain::Int32, 31),
            (TruncLayer::for_int64_cast(), IntegerCastDomain::Int64, 63),
        ] {
            let limit = 2.0_f32.powi(exponent);
            assert!(layer.propagate_ibp(&interval(0.0, limit)).is_err());
            assert!(layer
                .propagate_ibp(&interval(ny_tensor::next_down_f32(-limit), 0.0))
                .is_err());

            let infinite = BoundedTensor::new_allow_infinite(
                arr1(&[f32::NEG_INFINITY]).into_dyn(),
                arr1(&[f32::INFINITY]).into_dyn(),
            )
            .expect("BoundedTensor permits explicit infinite enclosures");
            assert!(layer.propagate_ibp(&infinite).is_err());

            // BoundedTensor itself rejects NaN; pin the layer's raw domain
            // predicate too so a future tensor representation cannot bypass it.
            assert!(!domain.contains_interval(f64::NAN, 0.0));
            assert!(!domain.contains_interval(0.0, f64::NAN));
        }
    }

    #[test]
    fn integer_cast_domain_guard_covers_dense_batched_and_patches_crown() {
        let dense = LinearBounds::identity(1);
        let batched = BatchedLinearBounds::new(
            arr2(&[[1.0]]).into_dyn(),
            arr1(&[0.0]).into_dyn(),
            arr2(&[[1.0]]).into_dyn(),
            arr1(&[0.0]).into_dyn(),
            vec![1],
            vec![1],
        )
        .expect("one-dimensional batched identity");
        let patches = crate::bounds::patches::PatchesLinearBounds::identity((1, 1, 1), (1, 1, 1));

        for (layer, exponent) in [
            (TruncLayer::for_int32_cast(), 31),
            (TruncLayer::for_int64_cast(), 63),
        ] {
            let limit = 2.0_f32.powi(exponent);
            let in_range = interval(-1.0, 1.0);
            let _ = layer
                .propagate_linear_with_bounds(&dense, &in_range)
                .expect("dense CROWN must accept an in-range Cast domain");
            let _ = layer
                .propagate_linear_batched_with_bounds(&batched, &in_range)
                .expect("batched CROWN must accept an in-range Cast domain");
            layer
                .propagate_patches_with_bounds(&patches, &spatial_interval(-1.0, 1.0))
                .expect("Patches CROWN must accept an in-range Cast domain");

            let out_of_range = interval(0.0, limit);
            assert!(matches!(
                layer.propagate_linear_with_bounds(&dense, &out_of_range),
                Err(NyError::SoundnessRefusal(_))
            ));
            assert!(matches!(
                layer.propagate_linear_batched_with_bounds(&batched, &out_of_range),
                Err(NyError::SoundnessRefusal(_))
            ));
            assert!(matches!(
                layer.propagate_patches_with_bounds(&patches, &spatial_interval(0.0, limit),),
                Err(NyError::SoundnessRefusal(_))
            ));
        }
    }

    #[test]
    fn native_trunc_is_not_accidentally_restricted_to_an_integer_cast_domain() {
        let huge = interval(-f32::MAX, f32::MAX);
        TruncLayer::new()
            .propagate_ibp(&huge)
            .expect("native ONNX Trunc remains defined over finite f32 values");
    }
}

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
/// `1/d` for `d > 0`, rounded TOWARD ZERO so the result never exceeds the exact
/// reciprocal.
///
/// Both boundary arms of [`sign_crown_relaxation`] carry an envelope obligation
/// that is tight at the far endpoint and needs `slope <= 1/d` to hold EXACTLY.
/// Plain `1.0 / d` rounds to nearest and can land one ULP above, which makes the
/// relaxation non-enclosing in the false-`unsat` direction.
///
/// The obligation is tested EXACTLY. `1.0f64 / f64::from(d)` would NOT do: it is
/// itself a rounded quotient, so comparing against it both misses cases and
/// needlessly loosens others. Instead test the obligation in its own terms —
/// `slope * d <= 1` — which is exact in f64, because the product of two f32
/// values needs at most 48 significand bits and f64 carries 53.
fn reciprocal_rounded_toward_zero(d: f32) -> f32 {
    debug_assert!(d > 0.0, "reciprocal_rounded_toward_zero requires d > 0");
    let mut s = 1.0f32 / d;
    // Exact obligation check; `next_down` is monotone so this terminates, and in
    // practice it never runs more than once.
    while s > 0.0 && f64::from(s) * f64::from(d) > 1.0 {
        s = ny_tensor::next_down_f32(s);
    }
    s
}

fn sign_crown_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if l > 0.0 {
        LinearRelaxation::new(0.0, 1.0, 0.0, 1.0)
    } else if u < 0.0 {
        LinearRelaxation::new(0.0, -1.0, 0.0, -1.0)
    } else if l == 0.0 && u == 0.0 {
        LinearRelaxation::new(0.0, 0.0, 0.0, 0.0)
    } else if l == 0.0 {
        // l == 0, u > 0: lower slope through (0,0)→(u,1), upper constant 1.
        //
        // The envelope obligation is `sign(x) >= slope * x` on [0, u], and it is
        // TIGHT at x = u: it needs `slope * u <= sign(u) = 1`, i.e. `slope <= 1/u`
        // EXACTLY. Round-to-nearest `1.0 / u` can land ABOVE the exact reciprocal,
        // which over-claims the lower bound by ~1 ULP — the false-`unsat` direction.
        // MEASURED PREVALENCE: this is the common case, not a corner. Scanning 4096
        // mantissas in each of the 254 normal exponents, **49.4% violate**, and every
        // single exponent is affected (range 1.18e-38 .. 1.70e38). A scan that
        // evaluates `slope * u` in f32 sees zero violations — the f32 product rounds
        // back to 1.0 — which is why this hid: the defect is in the STORED slope
        // exceeding 1/u, so the line sits above `sign` in exact arithmetic.
        // Round toward zero so the obligation holds by construction.
        let lower_slope = reciprocal_rounded_toward_zero(u.max(1e-8));
        LinearRelaxation::new(lower_slope, 0.0, 0.0, 1.0)
    } else if u == 0.0 {
        // l < 0, u == 0: lower constant -1, upper slope through (l,-1)→(0,0).
        // Mirror obligation: `sign(x) <= slope * x` on [l, 0], tight at x = l where
        // it needs `slope * l >= -1`, i.e. `slope <= 1/|l|` exactly. Same rounding
        // hazard, same fix.
        let upper_slope = reciprocal_rounded_toward_zero(-l.min(-1e-8));
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

#[cfg(test)]
mod sign_relaxation_boundary_tests {
    use super::*;

    /// Every f32 whose reciprocal is SUBNORMAL — the regime where round-to-nearest
    /// `1.0 / u` lands above the exact reciprocal. Reciprocals go subnormal for
    /// `u > 1/f32::MIN_POSITIVE ~= 8.5e37`.
    fn subnormal_reciprocal_witnesses() -> Vec<f32> {
        vec![
            2.061_146_3e38,
            3.264_381_8e38,
            2.090_681_2e38,
            1.497_343_2e38,
            3.019_198e38,
            f32::MAX,
            1.0e38,
            9.0e37,
        ]
    }

    /// THE OBLIGATION: for `l == 0, u > 0` the lower line `y = slope*x` must satisfy
    /// `slope * u <= sign(u) = 1`. Anything above 1 is a non-enclosing lower bound,
    /// i.e. the false-`unsat` direction.
    #[test]
    fn sign_lower_line_never_exceeds_one_at_the_far_endpoint() {
        for u in subnormal_reciprocal_witnesses() {
            let r = sign_crown_relaxation(0.0, u);
            let at_u = f64::from(r.lower_slope) * f64::from(u);
            assert!(
                at_u <= 1.0,
                "lower line over-claims at x=u: u={u:e} slope={:e} slope*u={at_u:.17e} > 1",
                r.lower_slope
            );
        }
    }

    /// Mirror obligation for `l < 0, u == 0`: `slope * l >= sign(l) = -1`.
    #[test]
    fn sign_upper_line_never_undercuts_minus_one_at_the_far_endpoint() {
        for u in subnormal_reciprocal_witnesses() {
            let l = -u;
            let r = sign_crown_relaxation(l, 0.0);
            let at_l = f64::from(r.upper_slope) * f64::from(l);
            assert!(
                at_l >= -1.0,
                "upper line under-claims at x=l: l={l:e} slope={:e} slope*l={at_l:.17e} < -1",
                r.upper_slope
            );
        }
    }

    /// TEETH. Reproduces the ORIGINAL round-to-nearest arithmetic and asserts it
    /// ACTUALLY VIOLATES the obligation on these witnesses. Without this, the two
    /// tests above would pass against the buggy code and prove nothing.
    #[test]
    fn round_to_nearest_reciprocal_is_actually_caught() {
        let mut violations = 0usize;
        for u in subnormal_reciprocal_witnesses() {
            let old_slope = 1.0f32 / u.max(1e-8); // the pre-fix expression
            if f64::from(old_slope) * f64::from(u) > 1.0 {
                violations += 1;
            }
        }
        assert!(
            violations > 0,
            "witness set no longer exercises the round-to-nearest defect — \
             the teeth are gone and the obligation tests above are vacuous"
        );
    }

    /// The fix must never loosen a slope that ALREADY satisfies the obligation —
    /// otherwise it would pay tightness for nothing. Note this is emphatically NOT
    /// "bit-identical for normal reciprocals": measured exhaustively by exponent
    /// (4096 mantissas x all 254 normal exponents), **49.4% of normal f32 violate**
    /// the obligation, in every exponent, across the whole range 1.18e-38..1.70e38.
    #[test]
    fn never_loosens_a_slope_that_already_satisfies_the_obligation() {
        for &u in &[1e-8f32, 1e-3, 0.5, 1.0, 2.0, 3.7, 100.0, 1e6, 1e20, 1e30] {
            let rtn = 1.0f32 / u;
            let got = reciprocal_rounded_toward_zero(u);
            if f64::from(rtn) * f64::from(u) <= 1.0 {
                assert_eq!(
                    got.to_bits(),
                    rtn.to_bits(),
                    "needless loosening at u={u:e}: {got:e} vs round-to-nearest {rtn:e}"
                );
            } else {
                // Had to step down; it must step down by exactly one ULP, no more.
                assert_eq!(
                    got.to_bits(),
                    ny_tensor::next_down_f32(rtn).to_bits(),
                    "over-corrected at u={u:e}"
                );
            }
        }
    }

    /// Breadth check on the real invariant, over a wide spread of exponents and
    /// mantissas: the returned slope ALWAYS satisfies `slope * d <= 1` exactly.
    #[test]
    fn obligation_holds_across_the_whole_normal_range() {
        let mut stepped = 0usize;
        let mut total = 0usize;
        for exp in 1u32..255 {
            for m in (0u32..(1 << 23)).step_by(1 << 19) {
                let d = f32::from_bits((exp << 23) | m);
                if !(d > 0.0 && d.is_finite()) {
                    continue;
                }
                let s = reciprocal_rounded_toward_zero(d);
                if s == 0.0 {
                    continue;
                }
                total += 1;
                assert!(
                    f64::from(s) * f64::from(d) <= 1.0,
                    "obligation violated: d={d:e} slope={s:e}"
                );
                if s.to_bits() != (1.0f32 / d).to_bits() {
                    stepped += 1;
                }
            }
        }
        assert!(total > 3000, "coverage too thin: {total}");
        // Pins the measured prevalence: this is a common case, not a rare corner.
        assert!(
            stepped * 100 / total >= 30,
            "expected a large fraction to need correction, got {stepped}/{total}"
        );
    }
}
