// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Common traits and utilities for bound propagation layers.
//!
//! Sub-modules:
//! - [`compose`] — Shared CROWN backward composition helpers
//! - [`traits`] — `BoundPropagation` and `PatchesPropagation` traits
//! - [`crown_dense`] — Dense (Array2) CROWN backward for element-wise activations
//! - [`crown_batched`] — Batched (ArrayD) CROWN backward for element-wise activations
//! - [`crown_patches`] — Patches (6D) CROWN backward for element-wise activations

pub(crate) mod compose;
mod crown_batched;
mod crown_dense;
mod crown_patches;
mod crown_patches_alpha;
mod crown_patches_sparse;
pub(crate) mod per_channel;
mod traits;

// Re-export traits
pub use traits::BoundPropagation;
pub(crate) use traits::PatchesPropagation;

// Re-export CROWN backward functions
pub(crate) use crown_dense::crown_elementwise_backward;
pub use crown_dense::crown_elementwise_backward_indexed;

pub use crown_batched::crown_elementwise_backward_batched;
pub use crown_batched::crown_elementwise_backward_batched_indexed;

pub(crate) use crown_patches::crown_elementwise_backward_patches;
pub(crate) use crown_patches_alpha::{
    crown_relu_backward_patches_with_alpha, crown_relu_backward_patches_with_alpha_bound_only,
};

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

/// Minimum number of elements to use parallel iteration for element-wise operations.
/// Below this threshold, sequential iteration is faster due to parallelization overhead.
/// Benchmark results: 24K elements shows regression, 98K+ elements shows 2-3x speedup.
pub(crate) const PARALLEL_ELEMENT_THRESHOLD: usize = 65536;

/// IBP propagation for element-wise activations with a non-monotonic or S-shaped
/// bound function that maps `(lower, upper) -> (out_lower, out_upper)`.
///
/// Factors out the repeated parallel-zip pattern used by Tanh, Sigmoid, Arctan,
/// Softplus, Sin, Cos, and similar layers whose IBP differs only in the bound function.
/// Part of #2914.
pub(crate) fn ibp_bound_interval_parallel(
    input: &BoundedTensor,
    bound_fn: fn(f32, f32) -> (f32, f32),
) -> Result<BoundedTensor> {
    let mut out_lower = input.lower().clone();
    let mut out_upper = input.upper().clone();

    let zip = ndarray::Zip::from(&mut out_lower)
        .and(&mut out_upper)
        .and(input.lower())
        .and(input.upper());

    if input.len() >= PARALLEL_ELEMENT_THRESHOLD {
        zip.par_for_each(|ol, ou, &il, &iu| {
            let (l, u) = bound_fn(il, iu);
            *ol = l;
            *ou = u;
        });
    } else {
        zip.for_each(|ol, ou, &il, &iu| {
            let (l, u) = bound_fn(il, iu);
            *ol = l;
            *ou = u;
        });
    }

    BoundedTensor::new(out_lower, out_upper)
}

/// Domain guard that rejects non-finite pre-activation bounds (NaN or ±Inf).
///
/// Use this as the `domain_guard` parameter in `impl_elementwise_activation!` for
/// activation layers that are defined on all finite reals but cannot soundly relax
/// non-finite inputs (SiLU, ELU, SELU, CELU, LeakyReLU, etc.).
///
/// Returns `Err(NumericalInstability)` so the backward dispatch caller falls back
/// to IBP, which is always sound for these layers.
///
/// Reference: exp.rs and log.rs use domain-specific guards; this is the generic
/// "reject non-finite" guard for activations without restricted domains.
/// Part of #2836.
pub(crate) fn non_finite_domain_guard(
    layer_name: &str,
    pre_activation: &BoundedTensor,
) -> Result<()> {
    if pre_activation.lower().iter().any(|x| !x.is_finite())
        || pre_activation.upper().iter().any(|x| !x.is_finite())
    {
        return Err(NyError::NumericalInstability(format!(
            "{layer_name} CROWN: non-finite pre-activation bounds"
        )));
    }
    Ok(())
}

/// Domain guard that rejects only NaN pre-activation bounds, accepting ±Inf.
///
/// Use this for piecewise-linear activations (ReLU, LeakyReLU) whose
/// `*_linear_relaxation` functions contain *proven* over-approximation branches
/// for infinite pre-activation endpoints (l=-inf and/or u=+inf). Letting those
/// branches run recovers a tight, sound bound on unbounded input domains that the
/// stricter [`non_finite_domain_guard`] would otherwise discard by falling back to IBP.
///
/// NaN must still bail: a NaN endpoint cannot be soundly bounded (its true range is
/// undefined), so the only sound response is to refuse and let the caller fall back.
///
/// SOUNDNESS: this guard is only appropriate for relaxation functions where EVERY
/// infinite branch is an in-code-proven over-approximation
/// (`lower(x) <= f(x) <= upper(x)` over the whole, possibly-unbounded, range), with
/// the genuinely-unbounded sub-cases failing closed to a conservative ±Inf plane.
/// Do NOT use it for activations whose infinite branches are missing or unproven.
pub(crate) fn nan_only_domain_guard(
    layer_name: &str,
    pre_activation: &BoundedTensor,
) -> Result<()> {
    if pre_activation.lower().iter().any(|x| x.is_nan())
        || pre_activation.upper().iter().any(|x| x.is_nan())
    {
        return Err(NyError::NumericalInstability(format!(
            "{layer_name} CROWN: NaN pre-activation bounds"
        )));
    }
    Ok(())
}

/// Compute strides for a multi-dimensional array shape.
///
/// For shape [d0, d1, ..., dn], returns strides [s0, s1, ..., sn] where
/// s_i = product of d_(i+1) to d_n. This allows converting between flat
/// indices and multi-dimensional coordinates.
pub(crate) fn compute_strides(shape: &[usize]) -> Result<Vec<usize>> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].checked_mul(shape[i + 1]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "compute_strides: stride overflow at dim {i}: {} * {}",
                strides[i + 1],
                shape[i + 1]
            ))
        })?;
    }
    Ok(strides)
}

/// Resolve a single ONNX axis index (i64) to a positive `usize` with bounds validation.
///
/// Delegates to `ny_core::resolve_axis`. See that module for details.
pub(crate) fn resolve_axis(axis: i64, ndim: usize, layer_name: &str) -> Result<usize> {
    ny_core::resolve_axis(axis, ndim, layer_name)
}

/// Resolve a single ONNX axis (i32) to a positive `usize` with bounds validation.
///
/// Convenience wrapper for layers that store axis as `i32`.
pub(crate) fn resolve_axis_i32(axis: i32, ndim: usize, layer_name: &str) -> Result<usize> {
    ny_core::resolve_axis_i32(axis, ndim, layer_name)
}

/// Resolve a stored unbatched-convention axis when a leading restart axis
/// has been prepended.
///
/// ny's ONNX loader stores positive axes as `axis - 1` (unbatched convention,
/// see `adjust_softmax_axis_for_unbatched` in ny-build). When restart batching
/// prepends a leading axis, positive stored axes must shift right by one to
/// restore sample-space semantics. Negative axes are relative to the trailing
/// end and need no adjustment.
///
/// Contract:
/// - Only for operators whose positive ONNX axes were stored as `axis - 1`.
/// - Negative stored axes pass through to `resolve_axis_i32` unchanged.
/// - Non-negative stored axes are shifted right by one (`axis + 1`).
///
/// Part of #4096.
pub(crate) fn resolve_axis_i32_with_restored_leading_axis(
    axis: i32,
    ndim: usize,
    layer_name: &str,
) -> Result<usize> {
    if axis < 0 {
        resolve_axis_i32(axis, ndim, layer_name)
    } else {
        resolve_axis_i32(
            axis.checked_add(1).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "{layer_name} preserve-leading-axis: axis overflow: {axis} + 1"
                ))
            })?,
            ndim,
            layer_name,
        )
    }
}

/// Shared CROWN boilerplate for element-wise nonlinear activation layers.
///
/// This macro factors out repeated trait/inherent methods where activations only differ by:
/// - IBP logic (kept manual in each layer)
/// - Relaxation function used by CROWN backward
/// - Optional domain guards on pre-activation bounds
///
/// Usage pattern inside a layer module:
/// - Inside `impl BoundPropagation for Layer`: `impl_elementwise_activation!(@trait_methods Layer, error_expr);`
/// - Inside `impl Layer`: `impl_elementwise_activation!(@inherent_methods Layer, relax_fn[, domain_guard: guard_fn]);`
macro_rules! impl_elementwise_activation {
    (@trait_methods $layer:ty, $error_expr:expr) => {
        #[inline]
        fn propagate_linear<'a>(
            &self,
            _bounds: &'a crate::LinearBounds,
        ) -> ny_core::Result<std::borrow::Cow<'a, crate::LinearBounds>> {
            Err($error_expr)
        }

        #[inline]
        fn requires_pre_activation_bounds(&self) -> bool {
            true
        }

        #[inline]
        fn propagate_linear_with_bounds(
            &self,
            bounds: &crate::LinearBounds,
            pre_activation: &ny_tensor::BoundedTensor,
        ) -> ny_core::Result<crate::LinearBounds> {
            <$layer>::propagate_linear_with_bounds(self, bounds, pre_activation)
        }
    };

    (@inherent_methods $layer:ty, $relax_fn:expr, domain_guard: $guard:expr) => {
        /// CROWN backward propagation with pre-activation bounds.
        #[inline]
        pub fn propagate_linear_with_bounds(
            &self,
            bounds: &crate::LinearBounds,
            pre_activation: &ny_tensor::BoundedTensor,
        ) -> ny_core::Result<crate::LinearBounds> {
            ($guard)(pre_activation)?;
            crate::layers::common::crown_elementwise_backward(bounds, pre_activation, $relax_fn)
        }

        /// Batched CROWN backward propagation with pre-activation bounds.
        #[inline]
        pub fn propagate_linear_batched_with_bounds(
            &self,
            bounds: &crate::BatchedLinearBounds,
            pre_activation: &ny_tensor::BoundedTensor,
        ) -> ny_core::Result<crate::BatchedLinearBounds> {
            ($guard)(pre_activation)?;
            crate::layers::common::crown_elementwise_backward_batched(
                bounds,
                pre_activation,
                $relax_fn,
            )
        }

        /// Patches CROWN backward propagation with pre-activation bounds.
        ///
        /// Scales 6D patches coefficients by per-neuron relaxation slopes,
        /// keeping bounds in sparse Patches form for CNN memory optimization.
        /// Part of #2613 Phase 2 step 11.
        #[inline]
        pub(crate) fn propagate_patches_with_bounds(
            &self,
            bounds: &crate::bounds::patches::PatchesLinearBounds,
            pre_activation: &ny_tensor::BoundedTensor,
        ) -> ny_core::Result<crate::bounds::patches::CrownBounds> {
            ($guard)(pre_activation)?;
            crate::layers::common::crown_elementwise_backward_patches(
                bounds,
                pre_activation,
                $relax_fn,
            )
        }
    };

    (@inherent_methods_stateful $layer:ty, $relax_fn:expr, domain_guard: $guard:expr) => {
        /// CROWN backward propagation with pre-activation bounds.
        #[inline]
        pub fn propagate_linear_with_bounds(
            &self,
            bounds: &crate::LinearBounds,
            pre_activation: &ny_tensor::BoundedTensor,
        ) -> ny_core::Result<crate::LinearBounds> {
            ($guard)(pre_activation)?;
            let relax_fn = |l: f32, u: f32| ($relax_fn)(self, l, u);
            crate::layers::common::crown_elementwise_backward(bounds, pre_activation, relax_fn)
        }

        /// Batched CROWN backward propagation with pre-activation bounds.
        #[inline]
        pub fn propagate_linear_batched_with_bounds(
            &self,
            bounds: &crate::BatchedLinearBounds,
            pre_activation: &ny_tensor::BoundedTensor,
        ) -> ny_core::Result<crate::BatchedLinearBounds> {
            ($guard)(pre_activation)?;
            let relax_fn = |l: f32, u: f32| ($relax_fn)(self, l, u);
            crate::layers::common::crown_elementwise_backward_batched(
                bounds,
                pre_activation,
                relax_fn,
            )
        }

        /// Patches CROWN backward propagation with pre-activation bounds.
        ///
        /// Scales 6D patches coefficients by per-neuron relaxation slopes,
        /// keeping bounds in sparse Patches form for CNN memory optimization.
        /// Part of #2613 Phase 2 step 11.
        #[inline]
        pub(crate) fn propagate_patches_with_bounds(
            &self,
            bounds: &crate::bounds::patches::PatchesLinearBounds,
            pre_activation: &ny_tensor::BoundedTensor,
        ) -> ny_core::Result<crate::bounds::patches::CrownBounds> {
            ($guard)(pre_activation)?;
            let relax_fn = |l: f32, u: f32| ($relax_fn)(self, l, u);
            crate::layers::common::crown_elementwise_backward_patches(
                bounds,
                pre_activation,
                relax_fn,
            )
        }
    };
}

pub(crate) use impl_elementwise_activation;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_indexed;
#[cfg(test)]
mod tests_overflow;
