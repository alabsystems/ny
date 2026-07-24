// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-mode CROWN backward for element-wise activation functions.
//!
//! Operates on 6D patches tensors [oc, oh, ow, ic, ki, kj] from Conv2d backward,
//! or 4D sparse patches tensors [sparse_idx, ic, ki, kj] when unstable_idx is set.

use ndarray::ArrayD;
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use rayon::prelude::*;

use super::compose;
use super::crown_patches_sparse::backward_patches_sparse;
use crate::bounds::patches::{CrownBounds, PatchesData, PatchesLinearBounds};

/// CROWN backward for element-wise activations in Patches mode.
///
/// This is the Patches-mode equivalent of [`super::crown_elementwise_backward`].
/// Instead of operating on Dense A-matrices (Array2), it scales the 6D patches
/// tensor by per-INPUT-neuron relaxation slopes and updates the bias vectors
/// with intercept contributions.
///
/// Each patches coefficient at [oc, oh, ow, ic, ki, kj] connects a specification
/// output (oc, oh, ow) to an INPUT neuron at position:
///   ih = oh * stride_h + ki - pad_top
///   iw = ow * stride_w + kj - pad_left
/// The relaxation slope for that coefficient is determined by the pre-activation
/// bounds of the mapped input neuron, not the output position.
///
/// Handles identity patches by materializing them first, since element-wise
/// scaling produces non-identity results.
///
/// # Arguments
/// * `bounds` - Incoming Patches linear bounds from layers above
/// * `pre_activation` - Pre-activation bounds for the activation's input neurons.
///   Must have shape matching `bounds.lower_a.input_shape` (the space the patches
///   reference into).
/// * `relaxation_fn` - `(l, u) -> LinearRelaxation`
///
/// Reference: alpha-beta-CROWN auto_LiRPA/operators/relu.py (Patches backward)
/// Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md Phase 2, Step 8
/// Part of #2613
pub(crate) fn crown_elementwise_backward_patches<F>(
    bounds: &PatchesLinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
) -> Result<CrownBounds>
where
    F: Fn(f32, f32) -> crate::layers::activations::LinearRelaxation,
{
    let pre_flat = pre_activation.flatten();
    let pre_lower_nd = pre_flat
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![pre_flat.len()],
            got: pre_flat.lower().shape().to_vec(),
        })?;
    let pre_upper_nd = pre_flat
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![pre_flat.len()],
            got: pre_flat.upper().shape().to_vec(),
        })?;

    let (out_c, out_h, out_w) = bounds.lower_a.output_shape;

    // Pre-activation bounds must match the patches' input_shape (the neuron space)
    let (in_c_shape, in_h_shape, in_w_shape) = bounds.lower_a.input_shape;
    let num_input_neurons = in_c_shape * in_h_shape * in_w_shape;
    if pre_lower_nd.len() != num_input_neurons {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_input_neurons],
            got: vec![pre_lower_nd.len()],
        });
    }

    // Precompute per-input-neuron relaxation parameters
    let pre_lower_slice = pre_lower_nd
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous pre_lower array".into()))?;
    let pre_upper_slice = pre_upper_nd
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous pre_upper array".into()))?;
    let relaxations =
        compose::precompute_relaxations(pre_lower_slice, pre_upper_slice, &|l, u, _i| {
            relaxation_fn(l, u)
        });

    // Materialize identity patches if needed — element-wise scaling makes them non-identity
    let lower_a_data = if bounds.lower_a.identity {
        bounds.lower_a.materialize_identity()
    } else {
        bounds.lower_a.clone()
    };
    let upper_a_data = if bounds.upper_a.identity {
        bounds.upper_a.materialize_identity()
    } else {
        bounds.upper_a.clone()
    };

    let lower_patches = lower_a_data.patches.as_ref().ok_or_else(|| {
        NyError::InternalError("Non-identity PatchesData has no patches tensor".into())
    })?;
    let upper_patches = upper_a_data.patches.as_ref().ok_or_else(|| {
        NyError::InternalError("Non-identity PatchesData has no patches tensor".into())
    })?;

    // Sparse patches: 4D (unstable_size, in_c, kH, kW). Delegate to sparse path.
    // Part of #2613 Phase 4 step 19
    if lower_a_data.unstable_idx.is_some() {
        return backward_patches_sparse(
            &lower_a_data,
            &upper_a_data,
            lower_patches,
            upper_patches,
            bounds,
            &relaxations,
            (in_c_shape, in_h_shape, in_w_shape),
        );
    }

    let shape = lower_patches.shape();
    let explicit_rows = match shape.len() {
        6 => false,
        7 => {
            if shape[0] != bounds.row_count {
                return Err(NyError::ShapeMismatch {
                    expected: vec![bounds.row_count],
                    got: vec![shape[0]],
                });
            }
            true
        }
        _ => {
            return Err(NyError::ShapeMismatch {
                expected: vec![6, 7],
                got: vec![shape.len()],
            });
        }
    };
    let (in_c, kh, kw) = if explicit_rows {
        (shape[4], shape[5], shape[6])
    } else {
        (shape[3], shape[4], shape[5])
    };
    let patch_volume = in_c * kh * kw;
    // Contiguous chunk size of one explicit-rows SPEC row (7D layout only;
    // hoisted so both the compose pass and the err pass share it, spec §6.3).
    let row_volume = out_c * out_h * out_w * patch_volume;
    let logical_rows = if explicit_rows {
        bounds.row_count
    } else {
        checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "crown_elementwise_backward_patches: output dims overflow: {out_c} * {out_h} * {out_w}"
            ))
        })?
    };

    // Hard length check on the 7D explicit-rows path (spec I6,
    // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md): a carried coeff_err is indexed by
    // SPEC row, so `Some` with `len != row_count` is a construction bug —
    // return Err(ShapeMismatch) (routing the caller to its sound dense
    // fallback) rather than silently under-counting (the false-proof
    // direction). The 6D arm keeps its silent `.get(j).unwrap_or(0.0)` reads
    // for byte-identity (hardening deferred).
    if explicit_rows {
        for err in [
            lower_a_data.coeff_err.as_ref(),
            upper_a_data.coeff_err.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if err.len() != bounds.row_count {
                return Err(NyError::ShapeMismatch {
                    expected: vec![bounds.row_count],
                    got: vec![err.len()],
                });
            }
        }
    }

    // Stride and padding for mapping patches positions to input neuron positions.
    // Both sides must have matching stride/padding since they originate from the
    // same Conv2d backward (activation backward doesn't change stride/padding).
    debug_assert_eq!(
        lower_a_data.stride, upper_a_data.stride,
        "Patches stride mismatch between lower ({:?}) and upper ({:?})",
        lower_a_data.stride, upper_a_data.stride,
    );
    debug_assert_eq!(
        lower_a_data.padding, upper_a_data.padding,
        "Patches padding mismatch between lower ({:?}) and upper ({:?})",
        lower_a_data.padding, upper_a_data.padding,
    );
    let (sh, sw) = lower_a_data.stride;
    let (pad_left, _pad_right, pad_top, _pad_bottom) = lower_a_data.padding;

    // Create output patches and bias. Bias uses f64 to prevent catastrophic
    // cancellation (#1745), matching the Dense path in crown_elementwise_backward_indexed.
    let mut new_lower_patches = ArrayD::<f32>::zeros(lower_patches.raw_dim());
    let mut new_upper_patches = ArrayD::<f32>::zeros(upper_patches.raw_dim());
    let mut new_lower_b_f64 = bounds.lower_b.mapv(|x| x as f64);
    let mut new_upper_b_f64 = bounds.upper_b.mapv(|x| x as f64);

    // Track non-finite rows for ±Inf fallback (#3009)
    let mut lower_nonfinite = vec![false; logical_rows];
    let mut upper_nonfinite = vec![false; logical_rows];

    // Each output row owns a disjoint contiguous chunk of the standard-layout
    // patches tensors plus its own b/nonfinite slot — no cross-row state — so
    // rows compose in parallel with the per-row tap order unchanged
    // (value-identical to the serial loop). Non-standard layout (as_slice
    // returns None) falls back to the serial path.
    if explicit_rows {
        let compose_row_7d = |lp_r: &[f32],
                              up_r: &[f32],
                              nlp_r: &mut [f32],
                              nup_r: &mut [f32],
                              nlb_r: &mut f64,
                              nub_r: &mut f64,
                              lnf_r: &mut bool,
                              unf_r: &mut bool| {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        for ic in 0..in_c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                    let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                                    if ih_raw < 0
                                        || (ih_raw as usize) >= in_h_shape
                                        || iw_raw < 0
                                        || (iw_raw as usize) >= in_w_shape
                                    {
                                        continue;
                                    }

                                    let ih = ih_raw as usize;
                                    let iw = iw_raw as usize;
                                    let input_flat =
                                        ic * in_h_shape * in_w_shape + ih * in_w_shape + iw;
                                    let relax = &relaxations[input_flat];
                                    // Flat tap index within the row's contiguous chunk
                                    let t = ((((oc * out_h + oh) * out_w + ow) * in_c + ic) * kh
                                        + ki)
                                        * kw
                                        + kj;

                                    let lr = compose::compose_lower(lp_r[t], relax);
                                    nlp_r[t] = lr.new_coeff;
                                    *nlb_r += lr.intercept_contrib;
                                    *lnf_r |= lr.nonfinite;

                                    let ur = compose::compose_upper(up_r[t], relax);
                                    nup_r[t] = ur.new_coeff;
                                    *nub_r += ur.intercept_contrib;
                                    *unf_r |= ur.nonfinite;
                                }
                            }
                        }
                    }
                }
            }
        };

        let ran_parallel = row_volume > 0
            && match (
                lower_patches.as_slice(),
                upper_patches.as_slice(),
                new_lower_patches.as_slice_mut(),
                new_upper_patches.as_slice_mut(),
                new_lower_b_f64.as_slice_mut(),
                new_upper_b_f64.as_slice_mut(),
            ) {
                (Some(lp), Some(up), Some(nlp), Some(nup), Some(nlb), Some(nub)) => {
                    nlp.par_chunks_mut(row_volume)
                        .zip(nup.par_chunks_mut(row_volume))
                        .zip(lp.par_chunks(row_volume))
                        .zip(up.par_chunks(row_volume))
                        .zip(&mut nlb[..bounds.row_count])
                        .zip(&mut nub[..bounds.row_count])
                        .zip(&mut lower_nonfinite)
                        .zip(&mut upper_nonfinite)
                        .for_each(
                            |(((((((nlp_r, nup_r), lp_r), up_r), nlb_r), nub_r), lnf_r), unf_r)| {
                                compose_row_7d(
                                    lp_r, up_r, nlp_r, nup_r, nlb_r, nub_r, lnf_r, unf_r,
                                );
                            },
                        );
                    true
                }
                _ => false,
            };

        if !ran_parallel {
            for row in 0..bounds.row_count {
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            for ic in 0..in_c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                        let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                                        if ih_raw < 0
                                            || (ih_raw as usize) >= in_h_shape
                                            || iw_raw < 0
                                            || (iw_raw as usize) >= in_w_shape
                                        {
                                            continue;
                                        }

                                        let ih = ih_raw as usize;
                                        let iw = iw_raw as usize;
                                        let input_flat =
                                            ic * in_h_shape * in_w_shape + ih * in_w_shape + iw;
                                        let relax = &relaxations[input_flat];

                                        let la = lower_patches[[row, oc, oh, ow, ic, ki, kj]];
                                        let lr = compose::compose_lower(la, relax);
                                        new_lower_patches[[row, oc, oh, ow, ic, ki, kj]] =
                                            lr.new_coeff;
                                        new_lower_b_f64[row] += lr.intercept_contrib;
                                        lower_nonfinite[row] |= lr.nonfinite;

                                        let ua = upper_patches[[row, oc, oh, ow, ic, ki, kj]];
                                        let ur = compose::compose_upper(ua, relax);
                                        new_upper_patches[[row, oc, oh, ow, ic, ki, kj]] =
                                            ur.new_coeff;
                                        new_upper_b_f64[row] += ur.intercept_contrib;
                                        upper_nonfinite[row] |= ur.nonfinite;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        let compose_row_6d = |j: usize,
                              lp_j: &[f32],
                              up_j: &[f32],
                              nlp_j: &mut [f32],
                              nup_j: &mut [f32],
                              nlb_j: &mut f64,
                              nub_j: &mut f64,
                              lnf_j: &mut bool,
                              unf_j: &mut bool| {
            let oh = (j % (out_h * out_w)) / out_w;
            let ow = j % out_w;
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        // Map patches position to input neuron position
                        let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                        let iw_raw = (ow * sw + kj) as isize - pad_left as isize;

                        // Skip out-of-bounds (zero-padding positions)
                        if ih_raw < 0
                            || (ih_raw as usize) >= in_h_shape
                            || iw_raw < 0
                            || (iw_raw as usize) >= in_w_shape
                        {
                            continue;
                        }

                        let ih = ih_raw as usize;
                        let iw = iw_raw as usize;
                        let input_flat = ic * in_h_shape * in_w_shape + ih * in_w_shape + iw;
                        let relax = &relaxations[input_flat];
                        // Flat tap index within the row's contiguous chunk
                        let t = (ic * kh + ki) * kw + kj;

                        let lr = compose::compose_lower(lp_j[t], relax);
                        nlp_j[t] = lr.new_coeff;
                        *nlb_j += lr.intercept_contrib;
                        *lnf_j |= lr.nonfinite;

                        let ur = compose::compose_upper(up_j[t], relax);
                        nup_j[t] = ur.new_coeff;
                        *nub_j += ur.intercept_contrib;
                        *unf_j |= ur.nonfinite;
                    }
                }
            }
        };

        let ran_parallel = patch_volume > 0
            && match (
                lower_patches.as_slice(),
                upper_patches.as_slice(),
                new_lower_patches.as_slice_mut(),
                new_upper_patches.as_slice_mut(),
                new_lower_b_f64.as_slice_mut(),
                new_upper_b_f64.as_slice_mut(),
            ) {
                (Some(lp), Some(up), Some(nlp), Some(nup), Some(nlb), Some(nub)) => {
                    nlp.par_chunks_mut(patch_volume)
                        .zip(nup.par_chunks_mut(patch_volume))
                        .zip(lp.par_chunks(patch_volume))
                        .zip(up.par_chunks(patch_volume))
                        .zip(&mut nlb[..logical_rows])
                        .zip(&mut nub[..logical_rows])
                        .zip(&mut lower_nonfinite)
                        .zip(&mut upper_nonfinite)
                        .enumerate()
                        .for_each(
                            |(
                                j,
                                (((((((nlp_j, nup_j), lp_j), up_j), nlb_j), nub_j), lnf_j), unf_j),
                            )| {
                                compose_row_6d(
                                    j, lp_j, up_j, nlp_j, nup_j, nlb_j, nub_j, lnf_j, unf_j,
                                );
                            },
                        );
                    true
                }
                _ => false,
            };

        if !ran_parallel {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let j = oc * out_h * out_w + oh * out_w + ow;

                        for ic in 0..in_c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    // Map patches position to input neuron position
                                    let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                    let iw_raw = (ow * sw + kj) as isize - pad_left as isize;

                                    // Skip out-of-bounds (zero-padding positions)
                                    if ih_raw < 0
                                        || (ih_raw as usize) >= in_h_shape
                                        || iw_raw < 0
                                        || (iw_raw as usize) >= in_w_shape
                                    {
                                        continue;
                                    }

                                    let ih = ih_raw as usize;
                                    let iw = iw_raw as usize;
                                    let input_flat =
                                        ic * in_h_shape * in_w_shape + ih * in_w_shape + iw;
                                    let relax = &relaxations[input_flat];

                                    let la = lower_patches[[oc, oh, ow, ic, ki, kj]];
                                    let lr = compose::compose_lower(la, relax);
                                    new_lower_patches[[oc, oh, ow, ic, ki, kj]] = lr.new_coeff;
                                    new_lower_b_f64[j] += lr.intercept_contrib;
                                    lower_nonfinite[j] |= lr.nonfinite;

                                    let ua = upper_patches[[oc, oh, ow, ic, ki, kj]];
                                    let ur = compose::compose_upper(ua, relax);
                                    new_upper_patches[[oc, oh, ow, ic, ki, kj]] = ur.new_coeff;
                                    new_upper_b_f64[j] += ur.intercept_contrib;
                                    upper_nonfinite[j] |= ur.nonfinite;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // #3009: Non-finite row fallback for Patches activation CROWN backward.
    let lower_affected = lower_nonfinite.iter().filter(|&&r| r).count();
    let upper_affected = upper_nonfinite.iter().filter(|&&r| r).count();
    compose::log_nonfinite_fallback(
        "Patches activation",
        lower_affected,
        upper_affected,
        logical_rows,
    );

    // Certified coefficient error + intercept-error discharge
    // (#patches-coeff-err-soundness; 7D lift per
    // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §6). Mirrors the Dense
    // `crown_activation_error_step`: the activation backward scales each
    // incoming coefficient `a` by a per-neuron relaxation slope, so the stored
    // f32 coefficient `next_down/up_f32(fl32(a·slope))` differs from the true
    // real coefficient by (1) the incoming per-row error `a_err` possibly
    // flipping `a`'s sign and selecting the OTHER envelope slope — bounded by
    // `a_err·(|lower_slope|+|upper_slope|)` — and (2) the EXACT f32-multiply +
    // directed-rounding gap `|a·slope_used − stored|` (computed here, reduced
    // to the per-row max; both are over-bounds of the true per-coefficient
    // max, since max(x+y) ≤ max x + max y). The relaxation intercept folded
    // into the bias likewise picks up
    // `a_err·(|lower_intercept|+|upper_intercept|)`, discharged OUTWARD into
    // the f64 bias BEFORE the directed cast below.
    //
    // 6D arm: err index = flat output position (out_c·out_h·out_w rows);
    // byte-identical to the certified 6D design. 7D explicit-rows arm: the
    // err index is the SPEC row (axis 0, len row_count == bias len, spec I1);
    // the identical per-tap formulas are reduced over the WHOLE spec row —
    // MAX for the err terms (one scalar must cover every coefficient of the
    // row), SUM for the bias discharges (every output position's fold lands
    // in the row's single bias slot) — plus two 7D-only f64-summation
    // discharges beyond the literal 6D mirror (spec §6.1/R4, adjudication
    // A1): `(1+gbar)` on the intercept sum IS covers the nearest-f64
    // summation under-estimate of IS at 7D tap counts, and `gbar·ABS`
    // certifies the f64 rounding of the compose pass's own intercept fold
    // (up to row_volume adds per row — ~2^-29 relative at cifar scale, which
    // can escape the 0.5-ulp32 cast slack under bias cancellation), where
    // gbar = γ_(8·row_volume+16). Both terms only widen and vanish as
    // row_volume shrinks.
    let (lower_coeff_err, upper_coeff_err) = if explicit_rows {
        let old_lower_err = lower_a_data.coeff_err.as_ref();
        let old_upper_err = upper_a_data.coeff_err.as_ref();
        let mut new_lower_err = ndarray::Array1::<f32>::zeros(logical_rows);
        let mut new_upper_err = ndarray::Array1::<f32>::zeros(logical_rows);

        // gbar = γ_(8·row_volume+16) (Higham, f64 unit roundoff): ≥ 4×
        // headroom over the γ_(2·rv+4) needed by the IS/ABS accumulation
        // deficits, `+16` covering the small-row_volume corner (spec §6.2
        // (2d)). Saturating: absurd row volumes drive gbar → +INF, which
        // poisons the bias outward rather than under-counting.
        let gamma_bar = crate::layers::linear::crown_single_gamma_n_f64(
            row_volume.saturating_mul(8).saturating_add(16),
        );

        // I5 sanitize at consumption: non-finite or NEGATIVE carried err
        // poisons to +INF (outward degrade), NEVER NaN -> 0 (false-proof
        // hazard). Direct index is total: length was hard-checked above.
        let sanitize = |v: f32| -> f64 {
            if v.is_finite() && v >= 0.0 {
                f64::from(v)
            } else {
                f64::INFINITY
            }
        };

        // Per-row err pass (READ-ONLY over all coefficient tensors, runs
        // after compose so the stored new coefficients give EXACT gaps; spec
        // I3). Per spec row r, per side σ, in f64 over the fixed serial tap
        // order oc->oh->ow->ic->ki->kj (same padding predicate and
        // input_flat mapping as compose_row_7d):
        //   MSS  = max_t (|ls|+|us|)              (err term, MAX-lift)
        //   IS   = Σ_t (|li|+|ui|)                (every valid tap, incl a==0)
        //   GAP_σ = max_{t, a_σ≠0} |f64(a_σ)·f64(s_σ(a_σ)) − f64(stored_σ)|
        //   ABS_σ = |f64(b_σ[r])| + Σ_{t, a_σ≠0} |f64(a_σ)·f64(i_σ(a_σ))|
        //   D_σ  = gbar·ABS_σ + (oe_σ≠0 ? oe_σ·(IS·(1+gbar)) : 0)
        //     -> lower b −= D_l / upper b += D_u (non-finite D poisons ∓INF)
        //   err_σ[r] = 0.0 on σ-nonfinite rows (vacuous certificate),
        //     else next_up_f32((oe_σ·MSS [if oe_σ≠0] + GAP_σ) as f32),
        //     +INF if non-finite (never NaN).
        // Writes only err[r] and b[r]; rows are disjoint, so the parallel
        // driver is bitwise identical to the serial fallback.
        let err_row_7d = |row: usize,
                          lp_r: &[f32],
                          up_r: &[f32],
                          nlp_r: &[f32],
                          nup_r: &[f32],
                          nlb_r: &mut f64,
                          nub_r: &mut f64,
                          nle_r: &mut f32,
                          nue_r: &mut f32| {
            let oe_l = old_lower_err.map_or(0.0, |e| sanitize(e[row]));
            let oe_u = old_upper_err.map_or(0.0, |e| sanitize(e[row]));

            let mut max_slope_sum = 0.0f64;
            let mut int_sum = 0.0f64;
            let mut max_lower_gap = 0.0f64;
            let mut max_upper_gap = 0.0f64;
            // ABS_σ initialized with |b_σ[r]| — the compose fold accumulates
            // starting from the incoming bias, so Higham's γ_n·ABS bound
            // must include it (spec §6.2 (2b)).
            let mut abs_lower_sum = f64::from(bounds.lower_b[row]).abs();
            let mut abs_upper_sum = f64::from(bounds.upper_b[row]).abs();
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        for ic in 0..in_c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                    let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                                    if ih_raw < 0
                                        || (ih_raw as usize) >= in_h_shape
                                        || iw_raw < 0
                                        || (iw_raw as usize) >= in_w_shape
                                    {
                                        continue;
                                    }
                                    let ih = ih_raw as usize;
                                    let iw = iw_raw as usize;
                                    let input_flat =
                                        ic * in_h_shape * in_w_shape + ih * in_w_shape + iw;
                                    let relax = &relaxations[input_flat];
                                    let t = ((((oc * out_h + oh) * out_w + ow) * in_c + ic) * kh
                                        + ki)
                                        * kw
                                        + kj;

                                    let ss = f64::from(relax.lower_slope).abs()
                                        + f64::from(relax.upper_slope).abs();
                                    if ss > max_slope_sum {
                                        max_slope_sum = ss;
                                    }
                                    int_sum += f64::from(relax.lower_intercept).abs()
                                        + f64::from(relax.upper_intercept).abs();

                                    // EXACT directed-rounding gap + |a·i| fold
                                    // magnitude per side (mirror compose_*):
                                    // compose_lower uses lower slope/intercept
                                    // for a>0 else upper; compose_upper the
                                    // reverse. a==0 taps skip both — compose
                                    // stores 0 exactly and folds no intercept.
                                    let la = lp_r[t];
                                    if la != 0.0 {
                                        let (slope_used, intercept_used) = if la > 0.0 {
                                            (
                                                f64::from(relax.lower_slope),
                                                f64::from(relax.lower_intercept),
                                            )
                                        } else {
                                            (
                                                f64::from(relax.upper_slope),
                                                f64::from(relax.upper_intercept),
                                            )
                                        };
                                        let stored = f64::from(nlp_r[t]);
                                        let gap = (f64::from(la) * slope_used - stored).abs();
                                        if gap > max_lower_gap {
                                            max_lower_gap = gap;
                                        }
                                        abs_lower_sum += (f64::from(la) * intercept_used).abs();
                                    }
                                    let ua = up_r[t];
                                    if ua != 0.0 {
                                        let (slope_used, intercept_used) = if ua > 0.0 {
                                            (
                                                f64::from(relax.upper_slope),
                                                f64::from(relax.upper_intercept),
                                            )
                                        } else {
                                            (
                                                f64::from(relax.lower_slope),
                                                f64::from(relax.lower_intercept),
                                            )
                                        };
                                        let stored = f64::from(nup_r[t]);
                                        let gap = (f64::from(ua) * slope_used - stored).abs();
                                        if gap > max_upper_gap {
                                            max_upper_gap = gap;
                                        }
                                        abs_upper_sum += (f64::from(ua) * intercept_used).abs();
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Bias discharge D_σ into the f64 accumulator BEFORE the directed
            // cast (spec I4). `oe == 0` short-circuits `0·∞ = NaN` for
            // degenerate ±∞ relaxation intercepts (spec I5); any non-finite D
            // (∞ from a degenerate intercept in the receptive field, or NaN
            // from ∞·0) poisons the bias OUTWARD — skipping would emit a
            // finite bound the true range can escape (false-VERIFIED class).
            let disc_l = gamma_bar * abs_lower_sum
                + if oe_l != 0.0 {
                    oe_l * (int_sum * (1.0 + gamma_bar))
                } else {
                    0.0
                };
            if disc_l.is_finite() {
                *nlb_r -= disc_l;
            } else {
                *nlb_r = f64::NEG_INFINITY;
            }
            let disc_u = gamma_bar * abs_upper_sum
                + if oe_u != 0.0 {
                    oe_u * (int_sum * (1.0 + gamma_bar))
                } else {
                    0.0
                };
            if disc_u.is_finite() {
                *nub_r += disc_u;
            } else {
                *nub_r = f64::INFINITY;
            }

            // Err emission: f64 compute, one outward next_up_f32 at the f32
            // cast (spec I4). Non-finite (∞ overflow or NaN from ∞·0) emits
            // +INF — the degrade poison — NEVER NaN (spec I5). Nonfinite
            // rows are zeroed + bias-poisoned by the #3009 fallback below, so
            // err 0.0 is exact there (vacuous certificate).
            let lterm = if oe_l != 0.0 {
                oe_l * max_slope_sum
            } else {
                0.0
            };
            let uterm = if oe_u != 0.0 {
                oe_u * max_slope_sum
            } else {
                0.0
            };
            let lv = lterm + max_lower_gap;
            let uv = uterm + max_upper_gap;
            *nle_r = if lower_nonfinite[row] {
                0.0
            } else if !lv.is_finite() {
                f32::INFINITY
            } else {
                next_up_f32(lv as f32)
            };
            *nue_r = if upper_nonfinite[row] {
                0.0
            } else if !uv.is_finite() {
                f32::INFINITY
            } else {
                next_up_f32(uv as f32)
            };
        };

        let ran_parallel = row_volume > 0
            && match (
                lower_patches.as_slice(),
                upper_patches.as_slice(),
                new_lower_patches.as_slice(),
                new_upper_patches.as_slice(),
                new_lower_b_f64.as_slice_mut(),
                new_upper_b_f64.as_slice_mut(),
                new_lower_err.as_slice_mut(),
                new_upper_err.as_slice_mut(),
            ) {
                (
                    Some(lp),
                    Some(up),
                    Some(nlp),
                    Some(nup),
                    Some(nlb),
                    Some(nub),
                    Some(nle),
                    Some(nue),
                ) => {
                    nle.par_iter_mut()
                        .zip(nue.par_iter_mut())
                        .zip(&mut nlb[..bounds.row_count])
                        .zip(&mut nub[..bounds.row_count])
                        .zip(lp.par_chunks(row_volume))
                        .zip(up.par_chunks(row_volume))
                        .zip(nlp.par_chunks(row_volume))
                        .zip(nup.par_chunks(row_volume))
                        .enumerate()
                        .for_each(
                            |(
                                row,
                                (((((((nle_r, nue_r), nlb_r), nub_r), lp_r), up_r), nlp_r), nup_r),
                            )| {
                                err_row_7d(
                                    row, lp_r, up_r, nlp_r, nup_r, nlb_r, nub_r, nle_r, nue_r,
                                );
                            },
                        );
                    true
                }
                _ => false,
            };

        if !ran_parallel {
            for row in 0..bounds.row_count {
                let oe_l = old_lower_err.map_or(0.0, |e| sanitize(e[row]));
                let oe_u = old_upper_err.map_or(0.0, |e| sanitize(e[row]));

                let mut max_slope_sum = 0.0f64;
                let mut int_sum = 0.0f64;
                let mut max_lower_gap = 0.0f64;
                let mut max_upper_gap = 0.0f64;
                let mut abs_lower_sum = f64::from(bounds.lower_b[row]).abs();
                let mut abs_upper_sum = f64::from(bounds.upper_b[row]).abs();
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            for ic in 0..in_c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                        let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                                        if ih_raw < 0
                                            || (ih_raw as usize) >= in_h_shape
                                            || iw_raw < 0
                                            || (iw_raw as usize) >= in_w_shape
                                        {
                                            continue;
                                        }
                                        let ih = ih_raw as usize;
                                        let iw = iw_raw as usize;
                                        let input_flat =
                                            ic * in_h_shape * in_w_shape + ih * in_w_shape + iw;
                                        let relax = &relaxations[input_flat];

                                        let ss = f64::from(relax.lower_slope).abs()
                                            + f64::from(relax.upper_slope).abs();
                                        if ss > max_slope_sum {
                                            max_slope_sum = ss;
                                        }
                                        int_sum += f64::from(relax.lower_intercept).abs()
                                            + f64::from(relax.upper_intercept).abs();

                                        // EXACT gap + |a·i| magnitude per side
                                        // (see the closure above).
                                        let la = lower_patches[[row, oc, oh, ow, ic, ki, kj]];
                                        if la != 0.0 {
                                            let (slope_used, intercept_used) = if la > 0.0 {
                                                (
                                                    f64::from(relax.lower_slope),
                                                    f64::from(relax.lower_intercept),
                                                )
                                            } else {
                                                (
                                                    f64::from(relax.upper_slope),
                                                    f64::from(relax.upper_intercept),
                                                )
                                            };
                                            let stored = f64::from(
                                                new_lower_patches[[row, oc, oh, ow, ic, ki, kj]],
                                            );
                                            let gap = (f64::from(la) * slope_used - stored).abs();
                                            if gap > max_lower_gap {
                                                max_lower_gap = gap;
                                            }
                                            abs_lower_sum += (f64::from(la) * intercept_used).abs();
                                        }
                                        let ua = upper_patches[[row, oc, oh, ow, ic, ki, kj]];
                                        if ua != 0.0 {
                                            let (slope_used, intercept_used) = if ua > 0.0 {
                                                (
                                                    f64::from(relax.upper_slope),
                                                    f64::from(relax.upper_intercept),
                                                )
                                            } else {
                                                (
                                                    f64::from(relax.lower_slope),
                                                    f64::from(relax.lower_intercept),
                                                )
                                            };
                                            let stored = f64::from(
                                                new_upper_patches[[row, oc, oh, ow, ic, ki, kj]],
                                            );
                                            let gap = (f64::from(ua) * slope_used - stored).abs();
                                            if gap > max_upper_gap {
                                                max_upper_gap = gap;
                                            }
                                            abs_upper_sum += (f64::from(ua) * intercept_used).abs();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Discharge + err write — same rule as the closure above
                // (parallel and serial drivers are bitwise identical).
                let disc_l = gamma_bar * abs_lower_sum
                    + if oe_l != 0.0 {
                        oe_l * (int_sum * (1.0 + gamma_bar))
                    } else {
                        0.0
                    };
                if disc_l.is_finite() {
                    new_lower_b_f64[row] -= disc_l;
                } else {
                    new_lower_b_f64[row] = f64::NEG_INFINITY;
                }
                let disc_u = gamma_bar * abs_upper_sum
                    + if oe_u != 0.0 {
                        oe_u * (int_sum * (1.0 + gamma_bar))
                    } else {
                        0.0
                    };
                if disc_u.is_finite() {
                    new_upper_b_f64[row] += disc_u;
                } else {
                    new_upper_b_f64[row] = f64::INFINITY;
                }

                let lterm = if oe_l != 0.0 {
                    oe_l * max_slope_sum
                } else {
                    0.0
                };
                let uterm = if oe_u != 0.0 {
                    oe_u * max_slope_sum
                } else {
                    0.0
                };
                let lv = lterm + max_lower_gap;
                let uv = uterm + max_upper_gap;
                new_lower_err[row] = if lower_nonfinite[row] {
                    0.0
                } else if !lv.is_finite() {
                    f32::INFINITY
                } else {
                    next_up_f32(lv as f32)
                };
                new_upper_err[row] = if upper_nonfinite[row] {
                    0.0
                } else if !uv.is_finite() {
                    f32::INFINITY
                } else {
                    next_up_f32(uv as f32)
                };
            }
        }
        // Always Some/Some: the GAP terms (and the gbar·ABS discharge) are
        // intrinsic to the compose pass, even with exact (None) inputs.
        (Some(new_lower_err), Some(new_upper_err))
    } else {
        let old_lower_err = lower_a_data.coeff_err.as_ref();
        let old_upper_err = upper_a_data.coeff_err.as_ref();
        let mut new_lower_err = ndarray::Array1::<f32>::zeros(logical_rows);
        let mut new_upper_err = ndarray::Array1::<f32>::zeros(logical_rows);

        // Per-row parallel error step: row j reads its own old err / patches
        // chunk / already-written new patches chunk and writes only err[j] and
        // b[j] — no cross-row state, tap order within the row unchanged
        // (int_sum accumulates in the serial ic/ki/kj order). Value-identical.
        let err_row_6d = |j: usize,
                          lp_j: &[f32],
                          up_j: &[f32],
                          nlp_j: &[f32],
                          nup_j: &[f32],
                          nlb_j: &mut f64,
                          nub_j: &mut f64,
                          nle_j: &mut f32,
                          nue_j: &mut f32| {
            let oh = (j % (out_h * out_w)) / out_w;
            let ow = j % out_w;
            let oe_l = old_lower_err.map_or(0.0, |e| f64::from(e.get(j).copied().unwrap_or(0.0)));
            let oe_u = old_upper_err.map_or(0.0, |e| f64::from(e.get(j).copied().unwrap_or(0.0)));

            let mut max_slope_sum = 0.0f64;
            let mut int_sum = 0.0f64;
            let mut max_lower_gap = 0.0f64;
            let mut max_upper_gap = 0.0f64;
            for ic in 0..in_c {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                        let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                        if ih_raw < 0
                            || (ih_raw as usize) >= in_h_shape
                            || iw_raw < 0
                            || (iw_raw as usize) >= in_w_shape
                        {
                            continue;
                        }
                        let ih = ih_raw as usize;
                        let iw = iw_raw as usize;
                        let input_flat = ic * in_h_shape * in_w_shape + ih * in_w_shape + iw;
                        let relax = &relaxations[input_flat];
                        let t = (ic * kh + ki) * kw + kj;

                        let ss =
                            f64::from(relax.lower_slope).abs() + f64::from(relax.upper_slope).abs();
                        if ss > max_slope_sum {
                            max_slope_sum = ss;
                        }
                        int_sum += f64::from(relax.lower_intercept).abs()
                            + f64::from(relax.upper_intercept).abs();

                        // EXACT directed-rounding gap per side (mirror compose_*):
                        // compose_lower uses lower_slope for a>0 else upper_slope;
                        // compose_upper uses upper_slope for a>0 else lower_slope.
                        let la = lp_j[t];
                        if la != 0.0 {
                            let slope_used = if la > 0.0 {
                                f64::from(relax.lower_slope)
                            } else {
                                f64::from(relax.upper_slope)
                            };
                            let stored = f64::from(nlp_j[t]);
                            let gap = (f64::from(la) * slope_used - stored).abs();
                            if gap > max_lower_gap {
                                max_lower_gap = gap;
                            }
                        }
                        let ua = up_j[t];
                        if ua != 0.0 {
                            let slope_used = if ua > 0.0 {
                                f64::from(relax.upper_slope)
                            } else {
                                f64::from(relax.lower_slope)
                            };
                            let stored = f64::from(nup_j[t]);
                            let gap = (f64::from(ua) * slope_used - stored).abs();
                            if gap > max_upper_gap {
                                max_upper_gap = gap;
                            }
                        }
                    }
                }
            }

            // Discharge the incoming-error intercept perturbation OUTWARD into the
            // f64 bias. `oe == 0` short-circuits `0·∞ = NaN` (nothing to
            // discharge). An INFINITE discharge (oe > 0 with a degenerate ±∞
            // relaxation intercept in the receptive field) means the certificate
            // admits a coefficient sign that folds an infinite intercept — the
            // only sound bound is vacuous, so poison the bias outward rather
            // than skipping (skipping would emit a finite bound the true range
            // can escape: false-VERIFIED class).
            if oe_l != 0.0 {
                let disc_l = oe_l * int_sum;
                if disc_l.is_finite() {
                    *nlb_j -= disc_l;
                } else {
                    *nlb_j = f64::NEG_INFINITY;
                }
            }
            if oe_u != 0.0 {
                let disc_u = oe_u * int_sum;
                if disc_u.is_finite() {
                    *nub_j += disc_u;
                } else {
                    *nub_j = f64::INFINITY;
                }
            }

            // `if oe == 0` short-circuits `0·∞ = NaN` for degenerate ∞ slopes.
            let lterm = if oe_l != 0.0 {
                oe_l * max_slope_sum
            } else {
                0.0
            };
            let uterm = if oe_u != 0.0 {
                oe_u * max_slope_sum
            } else {
                0.0
            };
            *nle_j = if lower_nonfinite[j] {
                0.0
            } else {
                next_up_f32((lterm + max_lower_gap) as f32)
            };
            *nue_j = if upper_nonfinite[j] {
                0.0
            } else {
                next_up_f32((uterm + max_upper_gap) as f32)
            };
        };

        let ran_parallel = patch_volume > 0
            && match (
                lower_patches.as_slice(),
                upper_patches.as_slice(),
                new_lower_patches.as_slice(),
                new_upper_patches.as_slice(),
                new_lower_b_f64.as_slice_mut(),
                new_upper_b_f64.as_slice_mut(),
                new_lower_err.as_slice_mut(),
                new_upper_err.as_slice_mut(),
            ) {
                (
                    Some(lp),
                    Some(up),
                    Some(nlp),
                    Some(nup),
                    Some(nlb),
                    Some(nub),
                    Some(nle),
                    Some(nue),
                ) => {
                    nle.par_iter_mut()
                        .zip(nue.par_iter_mut())
                        .zip(&mut nlb[..logical_rows])
                        .zip(&mut nub[..logical_rows])
                        .zip(lp.par_chunks(patch_volume))
                        .zip(up.par_chunks(patch_volume))
                        .zip(nlp.par_chunks(patch_volume))
                        .zip(nup.par_chunks(patch_volume))
                        .enumerate()
                        .for_each(
                            |(
                                j,
                                (((((((nle_j, nue_j), nlb_j), nub_j), lp_j), up_j), nlp_j), nup_j),
                            )| {
                                err_row_6d(j, lp_j, up_j, nlp_j, nup_j, nlb_j, nub_j, nle_j, nue_j);
                            },
                        );
                    true
                }
                _ => false,
            };

        if !ran_parallel {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let j = oc * out_h * out_w + oh * out_w + ow;
                        let oe_l = old_lower_err
                            .map_or(0.0, |e| f64::from(e.get(j).copied().unwrap_or(0.0)));
                        let oe_u = old_upper_err
                            .map_or(0.0, |e| f64::from(e.get(j).copied().unwrap_or(0.0)));

                        let mut max_slope_sum = 0.0f64;
                        let mut int_sum = 0.0f64;
                        let mut max_lower_gap = 0.0f64;
                        let mut max_upper_gap = 0.0f64;
                        for ic in 0..in_c {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                    let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                                    if ih_raw < 0
                                        || (ih_raw as usize) >= in_h_shape
                                        || iw_raw < 0
                                        || (iw_raw as usize) >= in_w_shape
                                    {
                                        continue;
                                    }
                                    let ih = ih_raw as usize;
                                    let iw = iw_raw as usize;
                                    let input_flat =
                                        ic * in_h_shape * in_w_shape + ih * in_w_shape + iw;
                                    let relax = &relaxations[input_flat];

                                    let ss = f64::from(relax.lower_slope).abs()
                                        + f64::from(relax.upper_slope).abs();
                                    if ss > max_slope_sum {
                                        max_slope_sum = ss;
                                    }
                                    int_sum += f64::from(relax.lower_intercept).abs()
                                        + f64::from(relax.upper_intercept).abs();

                                    // EXACT directed-rounding gap per side (mirror compose_*):
                                    // compose_lower uses lower_slope for a>0 else upper_slope;
                                    // compose_upper uses upper_slope for a>0 else lower_slope.
                                    let la = lower_patches[[oc, oh, ow, ic, ki, kj]];
                                    if la != 0.0 {
                                        let slope_used = if la > 0.0 {
                                            f64::from(relax.lower_slope)
                                        } else {
                                            f64::from(relax.upper_slope)
                                        };
                                        let stored =
                                            f64::from(new_lower_patches[[oc, oh, ow, ic, ki, kj]]);
                                        let gap = (f64::from(la) * slope_used - stored).abs();
                                        if gap > max_lower_gap {
                                            max_lower_gap = gap;
                                        }
                                    }
                                    let ua = upper_patches[[oc, oh, ow, ic, ki, kj]];
                                    if ua != 0.0 {
                                        let slope_used = if ua > 0.0 {
                                            f64::from(relax.upper_slope)
                                        } else {
                                            f64::from(relax.lower_slope)
                                        };
                                        let stored =
                                            f64::from(new_upper_patches[[oc, oh, ow, ic, ki, kj]]);
                                        let gap = (f64::from(ua) * slope_used - stored).abs();
                                        if gap > max_upper_gap {
                                            max_upper_gap = gap;
                                        }
                                    }
                                }
                            }
                        }

                        // Discharge the incoming-error intercept perturbation OUTWARD into
                        // the f64 bias — see the parallel path above for the soundness
                        // argument (infinite discharge poisons the bias outward).
                        if oe_l != 0.0 {
                            let disc_l = oe_l * int_sum;
                            if disc_l.is_finite() {
                                new_lower_b_f64[j] -= disc_l;
                            } else {
                                new_lower_b_f64[j] = f64::NEG_INFINITY;
                            }
                        }
                        if oe_u != 0.0 {
                            let disc_u = oe_u * int_sum;
                            if disc_u.is_finite() {
                                new_upper_b_f64[j] += disc_u;
                            } else {
                                new_upper_b_f64[j] = f64::INFINITY;
                            }
                        }

                        // `if oe == 0` short-circuits `0·∞ = NaN` for degenerate ∞ slopes.
                        let lterm = if oe_l != 0.0 {
                            oe_l * max_slope_sum
                        } else {
                            0.0
                        };
                        let uterm = if oe_u != 0.0 {
                            oe_u * max_slope_sum
                        } else {
                            0.0
                        };
                        new_lower_err[j] = if lower_nonfinite[j] {
                            0.0
                        } else {
                            next_up_f32((lterm + max_lower_gap) as f32)
                        };
                        new_upper_err[j] = if upper_nonfinite[j] {
                            0.0
                        } else {
                            next_up_f32((uterm + max_upper_gap) as f32)
                        };
                    }
                }
            }
        }
        (Some(new_lower_err), Some(new_upper_err))
    };

    let mut new_lower_b = new_lower_b_f64.mapv(|x| next_down_f32(x as f32));
    let mut new_upper_b = new_upper_b_f64.mapv(|x| next_up_f32(x as f32));

    if explicit_rows {
        for row in 0..bounds.row_count {
            if lower_nonfinite[row] {
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            for ic in 0..in_c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        new_lower_patches[[row, oc, oh, ow, ic, ki, kj]] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
                new_lower_b[row] = f32::NEG_INFINITY;
            }
            if upper_nonfinite[row] {
                for oc in 0..out_c {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            for ic in 0..in_c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        new_upper_patches[[row, oc, oh, ow, ic, ki, kj]] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
                new_upper_b[row] = f32::INFINITY;
            }
        }
    } else {
        for j in 0..logical_rows {
            let oc = j / (out_h * out_w);
            let oh = (j % (out_h * out_w)) / out_w;
            let ow = j % out_w;
            if lower_nonfinite[j] {
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            new_lower_patches[[oc, oh, ow, ic, ki, kj]] = 0.0;
                        }
                    }
                }
                new_lower_b[j] = f32::NEG_INFINITY;
            }
            if upper_nonfinite[j] {
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            new_upper_patches[[oc, oh, ow, ic, ki, kj]] = 0.0;
                        }
                    }
                }
                new_upper_b[j] = f32::INFINITY;
            }
        }
    }

    Ok(CrownBounds::Patches(Box::new(PatchesLinearBounds {
        row_count: bounds.row_count,
        lower_a: PatchesData {
            coeff_err: lower_coeff_err,
            patches: Some(new_lower_patches),
            stride: lower_a_data.stride,
            padding: lower_a_data.padding,
            identity: false,
            output_shape: lower_a_data.output_shape,
            input_shape: lower_a_data.input_shape,
            unstable_idx: None,
        },
        lower_b: new_lower_b,
        upper_a: PatchesData {
            coeff_err: upper_coeff_err,
            patches: Some(new_upper_patches),
            stride: upper_a_data.stride,
            padding: upper_a_data.padding,
            identity: false,
            output_shape: upper_a_data.output_shape,
            input_shape: upper_a_data.input_shape,
            unstable_idx: None,
        },
        upper_b: new_upper_b,
    })))
}
