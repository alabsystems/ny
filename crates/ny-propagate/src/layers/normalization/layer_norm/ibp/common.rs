// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for LayerNorm IBP: error constructors, directed-rounding
//! mean utilities, input validation, and fallback bounds.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use super::super::types::{LayerNormLayer, LayerNormMode};

/// Return `InternalError` when an ndarray `mean()`/`first()` returns `None`.
pub(super) fn ln_internal_err(ctx: &str) -> NyError {
    NyError::InternalError(format!("LayerNorm: {ctx}"))
}

/// Compute mean along an axis using f64 accumulation, with directed rounding
/// toward negative infinity (sound for lower bounds).
///
/// Replaces ndarray `mean_axis()` which accumulates in f32, losing ~log2(n) bits
/// of precision for norm_size=768+. Uses `next_down_f32` on the f64->f32 cast so
/// the result is always <= the true f64 mean -- required when the mean is used as
/// a lower bound (e.g., `xu - mean_lower`). Part of #2423.
pub(super) fn mean_axis_f64_lower(arr: &ArrayD<f32>, axis: ndarray::Axis) -> Option<ArrayD<f32>> {
    let n = arr.shape().get(axis.0).copied()?;
    if n == 0 {
        return None;
    }
    let sum_f64 = arr.fold_axis(axis, 0.0_f64, |&acc, &x| acc + x as f64);
    let nf = n as f64;
    Some(sum_f64.mapv(|x| next_down_f32((x / nf) as f32)).into_dyn())
}

/// Compute mean along an axis using f64 accumulation, with directed rounding
/// toward positive infinity (sound for upper bounds).
///
/// Uses `next_up_f32` on the f64->f32 cast so the result is always >= the true
/// f64 mean -- required when the mean is used as an upper bound (e.g.,
/// `xl - mean_upper`). Part of #2423.
pub(super) fn mean_axis_f64_upper(arr: &ArrayD<f32>, axis: ndarray::Axis) -> Option<ArrayD<f32>> {
    let n = arr.shape().get(axis.0).copied()?;
    if n == 0 {
        return None;
    }
    let sum_f64 = arr.fold_axis(axis, 0.0_f64, |&acc, &x| acc + x as f64);
    let nf = n as f64;
    Some(sum_f64.mapv(|x| next_up_f32((x / nf) as f32)).into_dyn())
}

/// Validated shape context for LayerNorm IBP, consolidating the repeated
/// ndim/norm_size/zero-dimension/non-finite guards from the dispatch entry point.
pub(super) struct IbpShapeContext {
    pub ndim: usize,
    pub norm_size: usize,
}

/// Classification of a structurally valid LayerNorm IBP input.
pub(super) enum ValidatedIbpInput {
    /// All bounds finite: proceed with interval arithmetic.
    Finite(IbpShapeContext),
    /// Bounds contain ±inf (but no NaN): a legitimately saturated upstream box
    /// (deep-stack IBP overflow — outward-rounded Linear IBP under
    /// production-scale weights overflows f32). Not an error: the caller must
    /// substitute the conservative fallback enclosure, which is sound for
    /// every finite real input regardless of magnitude.
    Saturated,
}

/// Validate input for LayerNorm IBP and extract shape context.
///
/// Consolidates the validation checks previously repeated inline in
/// `propagate_ibp` and partially re-derived in `propagate_ibp_forward_mode`.
pub(super) fn validate_ibp_input(
    layer: &LayerNormLayer,
    input: &BoundedTensor,
) -> Result<ValidatedIbpInput> {
    let shape = input.shape();
    let ndim = shape.len();

    if ndim == 0 {
        return Err(NyError::InvalidSpec(
            "LayerNorm requires at least 1D input".to_string(),
        ));
    }

    // The per-slice index buffer is a fixed `[0usize; 8]` (`full_idx` in
    // standard/forward_mode/mean_only/slices), so a rank > 8 input would index it
    // out of bounds and PANIC. Fail closed on higher-rank tensors (e.g. a malformed
    // or adversarial ONNX model). (Trust verifier: closes the index_out_of_bounds
    // obligations in the LayerNorm IBP path.)
    if ndim > 8 {
        return Err(NyError::InvalidSpec(format!(
            "LayerNorm IBP supports tensors up to rank 8, got rank {ndim}"
        )));
    }

    // Guard: zero-valued dimensions cause division-by-zero in batch index
    // decoding. (#2806)
    if shape.contains(&0) {
        return Err(NyError::InvalidSpec(
            "LayerNorm: zero-valued dimension in input shape".to_string(),
        ));
    }

    // Category B per domain validation policy (designs/2026-02-07-domain-validation-policy.md):
    // NaN input bounds fail closed — NaN carries no bound information and would
    // silently poison the normalization arithmetic (0/0, inf/inf). ±Inf
    // endpoints are handled separately below: they are the conservative-bounds
    // convention for "unbounded in that direction" and must saturate, not
    // error (consistent with the CROWN path's constant-bounds fallback, #3259).
    if input.lower().iter().any(|x| x.is_nan()) || input.upper().iter().any(|x| x.is_nan()) {
        return Err(NyError::NumericalInstability(
            "LayerNorm IBP: NaN input bounds".to_string(),
        ));
    }

    let norm_size = shape[ndim - 1];
    if norm_size > (1 << 24) {
        // f32 precision guard (#2136)
        return Err(NyError::InternalError(format!(
            "LayerNorm dimension {norm_size} exceeds f32 exact integer range"
        )));
    }
    if layer.ny.len() != norm_size {
        return Err(NyError::ShapeMismatch {
            expected: vec![norm_size],
            got: vec![layer.ny.len()],
        });
    }
    if layer.beta.len() != norm_size {
        return Err(NyError::ShapeMismatch {
            expected: vec![norm_size],
            got: vec![layer.beta.len()],
        });
    }

    // Deep-stack IBP overflow saturation: an input box with infinite endpoints
    // is sound (it over-approximates the true range) and must produce a sound
    // conservative output, not abort the whole verification pass.
    if input.lower().iter().any(|x| x.is_infinite())
        || input.upper().iter().any(|x| x.is_infinite())
    {
        return Ok(ValidatedIbpInput::Saturated);
    }

    Ok(ValidatedIbpInput::Finite(IbpShapeContext {
        ndim,
        norm_size,
    }))
}

impl LayerNormLayer {
    pub(super) fn fallback_output_bounds(&self, shape: &[usize]) -> Result<BoundedTensor> {
        let ndim = shape.len();
        if ndim == 0 {
            return Err(NyError::InvalidSpec(
                "LayerNorm requires at least 1D input".to_string(),
            ));
        }

        let norm_size = shape[ndim - 1];
        if self.ny.len() != norm_size || self.beta.len() != norm_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![norm_size],
                got: vec![self.ny.len()],
            });
        }

        if self.mode == LayerNormMode::MeanOnly {
            // MeanOnly fallback: no bound information available. Use FALLBACK_BOUND
            // for consistency with the IBP overflow strategy (#3030, #3060).
            // Previously used raw [-inf, +inf] which BoundedTensor::new() rejects.
            let out_lower = ArrayD::from_elem(IxDyn(shape), f32::NEG_INFINITY);
            let out_upper = ArrayD::from_elem(IxDyn(shape), f32::INFINITY);
            return BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative);
        }

        if norm_size == 0 {
            return BoundedTensor::new(ArrayD::zeros(IxDyn(shape)), ArrayD::zeros(IxDyn(shape)));
        }

        let z_max = if norm_size <= 1 {
            // #3344: round UP
            0.0
        } else {
            next_up_f32(((norm_size as f32) - 1.0).sqrt())
        };

        let z_lower = -z_max;
        let z_upper = z_max;

        let mut per_dim_lower = Array1::<f32>::zeros(norm_size);
        let mut per_dim_upper = Array1::<f32>::zeros(norm_size);
        for i in 0..norm_size {
            let g = self.ny[i];
            let b = self.beta[i];

            if !g.is_finite() || !b.is_finite() {
                per_dim_lower.fill(f32::NEG_INFINITY);
                per_dim_upper.fill(f32::INFINITY);
                break;
            }

            if g >= 0.0 {
                // #3344: directed rounding on fallback
                per_dim_lower[i] = affine_fallback_down(b, g, z_lower);
                per_dim_upper[i] = affine_fallback_up(b, g, z_upper);
            } else {
                per_dim_lower[i] = affine_fallback_down(b, g, z_upper);
                per_dim_upper[i] = affine_fallback_up(b, g, z_lower);
            }
        }

        let mut out_lower = ArrayD::<f32>::zeros(IxDyn(shape));
        let mut out_upper = ArrayD::<f32>::zeros(IxDyn(shape));
        for mut lane in out_lower.lanes_mut(ndarray::Axis(ndim - 1)) {
            lane.assign(&per_dim_lower);
        }
        for mut lane in out_upper.lanes_mut(ndarray::Axis(ndim - 1)) {
            lane.assign(&per_dim_upper);
        }

        // Repair non-finite outputs: non-finite ny/beta -> [-inf, +inf] fill above,
        // or extreme ny * z_max can overflow. Consistent with IBP overflow
        // strategy (#3030, #3060).
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }
}

/// `beta + gamma * z` rounded OUTWARD (`_up` / `_down`), computed in f64.
///
/// SOUNDNESS-CRITICAL — do not inline this back into f32. `f32 * f32` is
/// EXACT in f64 (24 + 24 = 48 <= 53 mantissa bits), so the only rounding here
/// is the f64 addition (<= 0.5 ulp of the TRUE sum) plus the final f32 cast,
/// and the single directed f32 step dominates both.
///
/// Computing it in f32 is UNSOUND whenever `beta` cancels the product: the
/// product's round-to-nearest error lives at the PRODUCT's magnitude while the
/// one-ULP outward step lands at the (possibly hundreds of ULPs smaller) SUM's
/// magnitude, so the step cannot cover it and the "bound" excludes reachable
/// outputs. Found by adversarial review 2026-08-26; repro (InstanceNorm1d,
/// T = 4, gamma = 1.1550927, beta = -2.0006793, eps = 1e-5): the f32 form
/// returns an upper bound of 1e-45 while x = [1e5, 0, 0, 0] evaluates to
/// 1.7408787e-8.
#[inline]
fn affine_fallback_up(beta: f32, gamma: f32, z: f32) -> f32 {
    next_up_f32((f64::from(beta) + f64::from(gamma) * f64::from(z)) as f32)
}

/// Downward-rounded mirror of [`affine_fallback_up`]; same f64 rationale.
#[inline]
fn affine_fallback_down(beta: f32, gamma: f32, z: f32) -> f32 {
    next_down_f32((f64::from(beta) + f64::from(gamma) * f64::from(z)) as f32)
}
