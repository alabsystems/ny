// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Decomposed RmsNorm CROWN backward propagation.
//!
//! Propagates CROWN backward through a decomposed RmsNorm chain:
//!   x → x² → mean(x²) → var+eps → sqrt(var+eps) → 1/rms → x*inv_rms → γ·norm
//!
//! Part of #3387, #3447.

use ndarray::{Array1, Array2};
use ny_core::{checked_dim_product, checked_shape_product};
use ny_core::{nan_propagating_max, nan_propagating_min, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::debug;

use crate::bounds::BatchedLinearBounds;
use crate::layers::arithmetic::sqrt_linear_relaxation;
use crate::layers::common::BoundPropagation;
use crate::layers::misc::reciprocal::reciprocal_linear_relaxation;
use crate::layers::normalization::decomposed::{
    finalize_decomposed_norm_bounds, validate_norm_against_fused_ibp, DecomposedNormBackwardResult,
    DecomposedNormFinalizeMetadata, RowValidationCounts,
};
use crate::layers::normalization::math_common::square_interval_bounds;
use crate::{contiguous_flat_slice, RmsNormLayer};

use super::bilinear::accumulate_mccormick_bilinear_term;
use super::variance_chain::accumulate_variance_chain;

/// A per-row narrowing of the internal `inv_rms = 1/sqrt(mean(x²)+eps)` range,
/// supplied by GenBaB norm branching (#norm-genbab).
///
/// `decomposed_rms_norm_crown_backward` ordinarily derives the `inv_rms`
/// interval from the input box `x_ibp` by sound interval arithmetic. That
/// interval is very wide for wide inputs (e.g. `[1, 316]` for `x ∈ [-1,1]`,
/// `eps=1e-5`), so the fixed-midpoint reciprocal/sqrt relaxations are loose and
/// the row collapses to the fused-RmsNorm IBP in `validate_norm_against_fused_ibp`.
///
/// When a [`InvRmsOverride`] is supplied, the IBP-derived `[inv_rms_l,
/// inv_rms_u]` is **intersected** with `[lo, hi]` (per batch row). Intersection
/// only ever narrows the interval, so the resulting reciprocal/sqrt relaxation
/// is a sound over-approximation of the function on the narrowed set — which is
/// exactly the input subregion `{x : inv_rms(x) ∈ [lo,hi]}` that the requesting
/// GenBaB child subdomain is responsible for. The sibling child covers the
/// complementary range, so the two children union-cover the parent `inv_rms`
/// range (hence the full input box) and the combined BaB verdict is sound.
///
/// SOUNDNESS: the override may only be a *sub-interval* of the IBP-derived
/// range. We intersect (not replace) so a malformed/too-wide override can never
/// widen the certified range; a too-narrow override is still sound because the
/// sibling subdomain reclaims the excluded portion.
///
/// PER-GROUP (#norm-genbab soundness): the window is keyed by normalization
/// group (batch row `b`). A split narrows the `inv_rms` of ONE group; the other
/// groups keep their full IBP range. This is essential: a single shared window
/// would create a join gap — an input `x` whose group 0 has `inv_rms ≤ mid` but
/// group 1 has `inv_rms ≥ mid` would fall in NEITHER sibling child. Splitting
/// one group at a time keeps the union-cover argument valid (the constrained
/// group is partitioned; unconstrained groups span their full range in both
/// children).
#[derive(Debug, Clone)]
pub(crate) struct InvRmsOverride {
    /// Per-group `inv_rms` clamp window, indexed by normalization group (batch
    /// row). `windows[b] = Some((lo, hi))` narrows group `b`; `None` leaves it
    /// at the full IBP range. A length shorter than the batch count leaves the
    /// trailing groups unconstrained.
    pub windows: Vec<Option<(f32, f32)>>,
}

impl InvRmsOverride {
    /// A single-group window: clamp group `group_idx` to `[lo, hi]`, leaving all
    /// other groups at their full IBP range.
    pub(crate) fn single_group(group_idx: usize, lo: f32, hi: f32) -> Self {
        let mut windows = vec![None; group_idx + 1];
        windows[group_idx] = Some((lo, hi));
        Self { windows }
    }

    #[inline]
    fn window_for(&self, b: usize) -> Option<(f32, f32)> {
        self.windows.get(b).copied().flatten()
    }
}

/// Decomposed RmsNorm CROWN backward propagation.
///
/// Propagates CROWN backward through a decomposed RmsNorm chain:
///   x → x² → mean(x²) → var+eps → sqrt(var+eps) → 1/rms → x*inv_rms → γ·norm
///
/// Simpler than LayerNorm decomposition: no mean subtraction, no beta offset.
/// Fan-out at x (not d): x feeds both the product path (x*inv_rms) and the
/// variance path (x² → mean(x²) → sqrt → 1/rms).
///
/// Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019.
/// alpha-beta-CROWN decomposes via `_split_complex` (normalization.py:303-331).
///
/// Part of #3387. Design: designs/2026-03-06-rmsnorm-decomposed-crown-backward.md
pub(crate) fn decomposed_rms_norm_crown_backward(
    a_output: &BatchedLinearBounds,
    ny: &Array1<f32>,
    eps: f32,
    x_ibp: &BoundedTensor,
) -> Result<DecomposedNormBackwardResult> {
    decomposed_rms_norm_crown_backward_with_override(a_output, ny, eps, x_ibp, None)
}

/// [`decomposed_rms_norm_crown_backward`] with an optional per-row `inv_rms`
/// range override from GenBaB norm branching (#norm-genbab).
///
/// `inv_rms_override`, when `Some`, applies the SAME narrowing to every batch
/// row (the GenBaB split point is one scalar shared across the normalization
/// groups of a node). It is intersected with the IBP-derived range; see
/// [`InvRmsOverride`] for the soundness argument.
pub(crate) fn decomposed_rms_norm_crown_backward_with_override(
    a_output: &BatchedLinearBounds,
    ny: &Array1<f32>,
    eps: f32,
    x_ibp: &BoundedTensor,
    inv_rms_override: Option<InvRmsOverride>,
) -> Result<DecomposedNormBackwardResult> {
    let a_shape = a_output.lower_a().shape();
    let ndim = a_shape.len();
    if ndim < 2 {
        return Err(NyError::InvalidSpec(
            "decomposed_rms_norm_crown_backward: A must have at least 2 dimensions".into(),
        ));
    }

    let n = a_shape[ndim - 1]; // normalization dimension
    let out_dim = a_shape[ndim - 2];
    let batch_dims = &a_shape[..ndim - 2];
    let total_batch = checked_shape_product(batch_dims)
        .ok_or_else(|| {
            NyError::InvalidSpec("decomposed_rms_norm: batch dimensions overflow".into())
        })?
        .max(1);
    let nf = n as f32;

    if n == 0 {
        return Err(NyError::InvalidSpec(
            "decomposed_rms_norm_crown_backward: normalization dimension is 0".into(),
        ));
    }
    if ny.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![ny.len()],
        });
    }

    // Reshape A matrices to 3D: [total_batch, out_dim, n]
    let a_l_3d = a_output
        .lower_a()
        .view()
        .into_shape_with_order((total_batch, out_dim, n))
        .map_err(|e| NyError::InvalidSpec(format!("reshape lower_a: {}", e)))?;
    let a_u_3d = a_output
        .upper_a()
        .view()
        .into_shape_with_order((total_batch, out_dim, n))
        .map_err(|e| NyError::InvalidSpec(format!("reshape upper_a: {}", e)))?;
    let b_l_2d = a_output
        .lower_b()
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|e| NyError::InvalidSpec(format!("reshape lower_b: {}", e)))?;
    let b_u_2d = a_output
        .upper_b()
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|e| NyError::InvalidSpec(format!("reshape upper_b: {}", e)))?;

    // Reshape x_ibp to [total_batch, n]
    let x_l_2d = x_ibp
        .lower()
        .view()
        .into_shape_with_order((total_batch, n))
        .map_err(|e| NyError::InvalidSpec(format!("reshape x_lower: {}", e)))?;
    let x_u_2d = x_ibp
        .upper()
        .view()
        .into_shape_with_order((total_batch, n))
        .map_err(|e| NyError::InvalidSpec(format!("reshape x_upper: {}", e)))?;

    // Output arrays
    let total_rows = checked_dim_product(
        &[total_batch, out_dim],
        "decomposed_rms_norm_crown_backward total rows",
    )?;
    let mut new_a_l = Array2::<f32>::zeros((total_rows, n));
    let mut new_a_u = Array2::<f32>::zeros((total_rows, n));
    let mut new_b_l = Array2::<f64>::zeros((total_batch, out_dim));
    let mut new_b_u = Array2::<f64>::zeros((total_batch, out_dim));

    // Non-finite tracking (#3009)
    let mut lower_nonfinite_rows = vec![false; total_rows];
    let mut upper_nonfinite_rows = vec![false; total_rows];

    // Copy initial biases
    for b in 0..total_batch {
        for j in 0..out_dim {
            new_b_l[[b, j]] = b_l_2d[[b, j]] as f64;
            new_b_u[[b, j]] = b_u_2d[[b, j]] as f64;
        }
    }

    // Per-group narrowed inv_rms (after override intersection). Captured so the
    // fused-IBP fallback can be tightened to the narrowed window (#norm-genbab):
    // for a narrow inv_rms window the fused interval ny·x·[inv_lo,inv_hi] is far
    // tighter than the full-range fused IBP, making EVERY narrowed subdomain
    // informative even when the decomposed relaxation itself does not survive.
    let mut narrowed_inv_rms: Vec<(f32, f32)> =
        vec![(f32::NEG_INFINITY, f32::INFINITY); total_batch];

    for b in 0..total_batch {
        let x_l_row = x_l_2d.row(b);
        let x_u_row = x_u_2d.row(b);
        let x_l_row = contiguous_flat_slice(&x_l_row);
        let x_u_row = contiguous_flat_slice(&x_u_row);

        // === Forward IBP: intermediate bounds for decomposed RmsNorm chain ===
        // No mean computation — RmsNorm uses x directly, not d = x - mean.
        // f64 accumulation for variance sum (#2423)
        let mut var_l_f64 = 0.0_f64;
        let mut var_u_f64 = 0.0_f64;
        for i in 0..n {
            let (sq_l, sq_u) = square_interval_bounds(x_l_row[i], x_u_row[i]);
            var_l_f64 += sq_l as f64;
            var_u_f64 += sq_u as f64;
        }

        // Directed rounding on variance/rms/inv_rms (#3270)
        let var_l = next_down_f32((var_l_f64 / nf as f64) as f32);
        let var_u = next_up_f32((var_u_f64 / nf as f64) as f32);
        let mut var_eps_l = next_down_f32((var_l as f64 + eps as f64) as f32);
        let mut var_eps_u = next_up_f32((var_u as f64 + eps as f64) as f32);
        let mut rms_l = next_down_f32(((var_eps_l as f64).sqrt()) as f32);
        let mut rms_u = next_up_f32(((var_eps_u as f64).sqrt()) as f32);
        let mut inv_rms_l = next_down_f32(1.0 / rms_u);
        let mut inv_rms_u = next_up_f32(1.0 / rms_l);

        // === GenBaB norm branching: narrow inv_rms / rms / var_eps to the
        // requesting child subdomain (#norm-genbab). ===
        //
        // The override is a sub-interval of inv_rms supplied by a GenBaB norm
        // split. We INTERSECT (never widen) the IBP-derived [inv_rms_l,
        // inv_rms_u] with [override.lo, override.hi], then derive the matching
        // rms and var_eps ranges so the reciprocal (over rms) and sqrt (over
        // var_eps) relaxations are built over the SAME narrowed set. The result
        // is sound on {x : inv_rms(x) ∈ narrowed} — the child's input subregion
        // — and the sibling child covers the complement, so the union of
        // children covers the parent input box (see `InvRmsOverride`).
        //
        // Directed rounding: widen the narrowed interval outward (lo down, hi
        // up) at every transform so the certified range never excludes a
        // reachable value of the child's subregion.
        if let Some((ov_lo, ov_hi)) = inv_rms_override.as_ref().and_then(|ov| ov.window_for(b)) {
            // Intersect inv_rms (clamp inward toward the override).
            let new_inv_l = nan_propagating_max(inv_rms_l, ov_lo);
            let new_inv_u = nan_propagating_min(inv_rms_u, ov_hi);
            // Only apply a well-formed, non-empty, strictly-positive narrowing.
            // inv_rms is always > 0 (rms ≥ sqrt(eps) > 0); a non-positive or
            // inverted intersection means the override is degenerate or the
            // child is empty — fall back to the un-narrowed IBP range (sound:
            // wider is always sound; emptiness is handled by the BaB layer).
            if new_inv_l.is_finite()
                && new_inv_u.is_finite()
                && new_inv_l > 0.0
                && new_inv_l <= new_inv_u
            {
                inv_rms_l = new_inv_l;
                inv_rms_u = new_inv_u;
                // rms = 1/inv_rms: monotone decreasing, so the rms interval is
                // [1/inv_u, 1/inv_l]. Round outward.
                rms_l = next_down_f32(1.0 / inv_rms_u);
                rms_u = next_up_f32(1.0 / inv_rms_l);
                // var_eps = rms²: monotone increasing on rms ≥ 0. Round outward.
                var_eps_l = next_down_f32(((rms_l as f64) * (rms_l as f64)) as f32);
                var_eps_u = next_up_f32(((rms_u as f64) * (rms_u as f64)) as f32);
            }
        }
        // inv_rms_l / inv_rms_u are read directly by the bilinear McCormick
        // term below (Phase A); rms_l/rms_u feed the reciprocal relaxation and
        // var_eps_l/var_eps_u feed the sqrt relaxation, all now consistently
        // narrowed.
        narrowed_inv_rms[b] = (inv_rms_l, inv_rms_u);

        // === Precompute scalar relaxations for the variance path ===
        let recip_relax = reciprocal_linear_relaxation(rms_l, rms_u);
        let sqrt_relax = sqrt_linear_relaxation(var_eps_l, var_eps_u);

        // Per-element A_x accumulators (fan-out: product path + variance path).
        // f64 to avoid compounding rounding error over out_dim iterations (#3344).
        let mut a_x_total_l = vec![0.0_f64; n];
        let mut a_x_total_u = vec![0.0_f64; n];

        // === CROWN backward through decomposed RmsNorm chain ===
        for j in 0..out_dim {
            let row_idx = b * out_dim + j;

            // Scalar accumulator for A_inv_rms
            let mut a_inv_rms_l_f64 = 0.0_f64;
            let mut a_inv_rms_u_f64 = 0.0_f64;

            // Reset per-element A_x accumulators
            a_x_total_l.fill(0.0);
            a_x_total_u.fill(0.0);

            // --- Phase A: ny scaling + McCormick for x[i] * inv_rms ---
            // y[i] = ny[i] * x[i] * inv_rms (no beta in RmsNorm)
            for i in 0..n {
                let g = ny[i];

                // Ny scaling: w = A_output[j,i] * ny[i]
                let w_l = a_l_3d[[b, j, i]] * g;
                let w_u = a_u_3d[[b, j, i]] * g;

                // No beta contribution (RmsNorm has no bias offset)
                let (lower_nonfinite, upper_nonfinite) = accumulate_mccormick_bilinear_term(
                    w_l,
                    w_u,
                    x_l_row[i],
                    x_u_row[i],
                    inv_rms_l,
                    inv_rms_u,
                    &mut a_x_total_l[i],
                    &mut a_x_total_u[i],
                    &mut a_inv_rms_l_f64,
                    &mut a_inv_rms_u_f64,
                    &mut new_b_l[[b, j]],
                    &mut new_b_u[[b, j]],
                );
                lower_nonfinite_rows[row_idx] |= lower_nonfinite;
                upper_nonfinite_rows[row_idx] |= upper_nonfinite;
            }

            // --- Phase B: Variance path (inv_rms → Reciprocal → Sqrt → Mean → Square) ---
            let (lower_nonfinite, upper_nonfinite) = accumulate_variance_chain(
                a_inv_rms_l_f64,
                a_inv_rms_u_f64,
                &recip_relax,
                &sqrt_relax,
                x_l_row.as_ref(),
                x_u_row.as_ref(),
                n,
                eps,
                &mut a_x_total_l,
                &mut a_x_total_u,
                &mut new_b_l[[b, j]],
                &mut new_b_u[[b, j]],
            );
            lower_nonfinite_rows[row_idx] |= lower_nonfinite;
            upper_nonfinite_rows[row_idx] |= upper_nonfinite;

            // --- No Phase C: RmsNorm has no mean subtraction ---
            // Fan-out at x merges directly — a_x_total already has both paths.
            // Single directed rounding at final f64→f32 conversion.
            for i in 0..n {
                new_a_l[[row_idx, i]] = next_down_f32(a_x_total_l[i] as f32);
                new_a_u[[row_idx, i]] = next_up_f32(a_x_total_u[i] as f32);
            }
        }
    }

    let mut result = finalize_decomposed_norm_bounds(
        new_a_l,
        new_a_u,
        new_b_l,
        new_b_u,
        DecomposedNormFinalizeMetadata {
            lower_nonfinite_rows: &lower_nonfinite_rows,
            upper_nonfinite_rows: &upper_nonfinite_rows,
            total_rows,
            out_dim,
            n,
            batch_dims,
            input_shape: a_output.input_shape(),
            output_shape: a_output.output_shape(),
            label: "Decomposed RmsNorm",
        },
    )?;

    // Fused-IBP fallback envelope. With a GenBaB inv_rms override we tighten it
    // per group to `ny·x·[inv_lo_b, inv_hi_b]` (#norm-genbab): for a narrow
    // inv_rms window this interval-product is far tighter than the full-range
    // fused RmsNorm IBP, so a row that collapses to the fallback still carries
    // the narrowed information instead of the global loose bound. This is what
    // lets BaB make monotone progress as it splits the inv_rms range, rather
    // than every not-yet-survived subdomain reporting the same global IBP.
    //
    // SOUNDNESS: for any x in the child subregion {x : inv_rms(x) ∈ [inv_lo_b,
    // inv_hi_b]}, RmsNorm(x)_i = ny_i·x_i·inv_rms(x) ∈ ny_i·[x_l_i,x_u_i]·[inv_lo_b,
    // inv_hi_b] (sound interval product, directed-rounded). Groups without an
    // override keep their exact full-range fused IBP via `min`/`max` with the
    // standard fused tensor below, so the fallback is never looser than before.
    let standard_fused = RmsNormLayer::new(ny.to_owned(), eps)?.propagate_ibp(x_ibp)?;
    let fused_ibp = if inv_rms_override.is_some() {
        tighten_fused_ibp_with_inv_rms(
            &standard_fused,
            ny,
            &narrowed_inv_rms,
            &x_l_2d,
            &x_u_2d,
            total_batch,
            n,
        )?
    } else {
        standard_fused
    };
    let fallback_rows =
        validate_norm_against_fused_ibp(&mut result, a_output, &fused_ibp, x_ibp, total_rows, n)?;
    if fallback_rows > 0 {
        debug!(
            "Decomposed RmsNorm: collapsed {fallback_rows}/{total_rows} rows to fused RmsNorm IBP"
        );
    }

    Ok(DecomposedNormBackwardResult {
        bounds: result,
        validation: RowValidationCounts {
            fallback_rows,
            total_rows,
        },
    })
}

/// Tighten the fused-RmsNorm IBP fallback per group using the narrowed `inv_rms`
/// window from a GenBaB norm split (#norm-genbab).
///
/// `standard_fused` is `RmsNormLayer::propagate_ibp(x)` over the full input box,
/// shape `[batch..., n]`. For each group `b` with a finite narrowed window
/// `[inv_lo, inv_hi]`, we intersect each element's fused interval with the sound
/// interval product `ny_i · [x_l, x_u] · [inv_lo, inv_hi]`. Groups whose window
/// is unbounded (no override applied) keep the standard fused interval.
///
/// The intersection (`max` of lowers, `min` of uppers) can only TIGHTEN the
/// envelope, so the fallback remains a sound over-approximation of the group's
/// restricted subregion `{x : inv_rms(x) ∈ [inv_lo, inv_hi]}`.
fn tighten_fused_ibp_with_inv_rms(
    standard_fused: &BoundedTensor,
    ny: &Array1<f32>,
    narrowed_inv_rms: &[(f32, f32)],
    x_l_2d: &ndarray::ArrayView2<'_, f32>,
    x_u_2d: &ndarray::ArrayView2<'_, f32>,
    total_batch: usize,
    n: usize,
) -> Result<BoundedTensor> {
    let shape = standard_fused.shape().to_vec();
    let std_lower = standard_fused
        .lower()
        .view()
        .into_shape_with_order((total_batch, n))
        .map_err(|e| NyError::InternalError(format!("tighten fused lower reshape: {e}")))?
        .to_owned();
    let std_upper = standard_fused
        .upper()
        .view()
        .into_shape_with_order((total_batch, n))
        .map_err(|e| NyError::InternalError(format!("tighten fused upper reshape: {e}")))?
        .to_owned();
    let mut lower = std_lower.clone();
    let mut upper = std_upper.clone();

    for b in 0..total_batch {
        let (inv_lo, inv_hi) = narrowed_inv_rms[b];
        if !inv_lo.is_finite() || !inv_hi.is_finite() || inv_lo > inv_hi {
            continue; // no (valid) override for this group — keep standard fused.
        }
        // NOTE (#norm-genbab): RmsNorm output is SCALE-INVARIANT in ‖x‖, so a
        // narrowed inv_rms (= narrowed ‖x‖) does NOT by itself bound x·inv_rms
        // tighter (|x_i·inv_rms| ≤ √n regardless of ‖x‖). The tightening below
        // therefore helps only where the inv_rms interval [inv_lo, inv_hi] is
        // genuinely narrow (small (inv_hi−inv_lo)·|x_i|), i.e. the low-inv_rms
        // region where x saturates the box (direction constrained). That is
        // exactly where the worst-case objective lives, so it is the region
        // that matters; the high-inv_rms tail stays at the (sound) standard
        // fused envelope.
        for i in 0..n {
            // Sound interval product ny_i · [x_l, x_u] · [inv_lo, inv_hi].
            // Compute the corner products in f64 and round outward.
            let g = ny[i] as f64;
            let xl = x_l_2d[[b, i]] as f64;
            let xu = x_u_2d[[b, i]] as f64;
            let il = inv_lo as f64;
            let ih = inv_hi as f64;
            // ny_i · x_i first (sign-aware), then × inv_rms (inv_rms > 0).
            let (gx_l, gx_u) = if g >= 0.0 {
                (g * xl, g * xu)
            } else {
                (g * xu, g * xl)
            };
            let corners = [gx_l * il, gx_l * ih, gx_u * il, gx_u * ih];
            let prod_l =
                next_down_f32(corners.iter().copied().fold(f64::INFINITY, f64::min) as f32);
            let prod_u =
                next_up_f32(corners.iter().copied().fold(f64::NEG_INFINITY, f64::max) as f32);
            // Intersect (tighten) with the standard fused interval.
            let new_l = nan_propagating_max(std_lower[[b, i]], prod_l);
            let new_u = nan_propagating_min(std_upper[[b, i]], prod_u);
            // Guard against a degenerate inverted interval from rounding: keep the
            // (sound, wider) standard fused values if the tightened bounds cross.
            if new_l <= new_u {
                lower[[b, i]] = new_l;
                upper[[b, i]] = new_u;
            }
        }
    }

    let lower = lower
        .into_shape_with_order(shape.clone())
        .map_err(|e| NyError::InternalError(format!("tighten fused lower restore: {e}")))?;
    let upper = upper
        .into_shape_with_order(shape)
        .map_err(|e| NyError::InternalError(format!("tighten fused upper restore: {e}")))?;
    BoundedTensor::new(lower.into_dyn(), upper.into_dyn())
}
