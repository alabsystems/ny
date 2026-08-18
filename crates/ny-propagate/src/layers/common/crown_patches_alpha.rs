// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Patches-mode CROWN backward for ReLU with optimizable alpha parameters.
//!
//! Split from `crown_patches.rs` for file size compliance.
//! This module contains the alpha-aware variant used by alpha-CROWN on CNN networks.

use ndarray::{Array1, ArrayD};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use rayon::prelude::*;
use std::mem::size_of;

use super::compose;
use crate::bounds::patches::{CrownBounds, PatchesData, PatchesLinearBounds};

/// Contiguous flat views + chunk geometry for the parallel patch compose
/// (#alpha-patches-par). See the call site for the disjointness argument.
struct PatchesParState<'a> {
    lp: &'a [f32],
    up: &'a [f32],
    nlp: &'a mut [f32],
    nup: &'a mut [f32],
    /// Number of chunks = length of the outermost patch axis.
    outer: usize,
    /// Coefficients per chunk.
    chunk: usize,
    /// Bias / non-finite slots per chunk (1 for 7D rows, `out_h*out_w` for 6D).
    bias_per_chunk: usize,
}

/// Admit the parallel compose, or `None` to keep the sequential loops.
///
/// Declines on any non-standard layout (non-contiguous arrays, an outer axis
/// that does not divide the buffer) and on a single chunk, where parallelism
/// cannot pay. Also declines when per-chunk gradient partials would not fit a
/// 512 MiB budget — the sequential path has no such allocation, so a very wide
/// outer axis must not turn into an OOM.
#[allow(clippy::too_many_arguments)]
fn patches_par_state<'a>(
    lower_patches: &'a ArrayD<f32>,
    upper_patches: &'a ArrayD<f32>,
    new_lower_patches: &'a mut ArrayD<f32>,
    new_upper_patches: &'a mut ArrayD<f32>,
    explicit_rows: bool,
    row_count: usize,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    track_gradients: bool,
    num_input_neurons: usize,
) -> Option<PatchesParState<'a>> {
    let outer = if explicit_rows { row_count } else { out_c };
    if outer < 2 {
        return None;
    }
    if track_gradients {
        const GRAD_PARTIAL_BUDGET_BYTES: usize = 512 << 20;
        let bytes = outer
            .checked_mul(num_input_neurons)?
            .checked_mul(size_of::<f32>())?;
        if bytes > GRAD_PARTIAL_BUDGET_BYTES {
            return None;
        }
    }
    let lp = lower_patches.as_slice()?;
    let up = upper_patches.as_slice()?;
    let total = lp.len();
    if up.len() != total || total == 0 || !total.is_multiple_of(outer) {
        return None;
    }
    let bias_per_chunk = if explicit_rows { 1 } else { out_h * out_w };
    if bias_per_chunk == 0 {
        return None;
    }
    let chunk = total / outer;
    let nlp = new_lower_patches.as_slice_mut()?;
    if nlp.len() != total {
        return None;
    }
    let nup = new_upper_patches.as_slice_mut()?;
    if nup.len() != total {
        return None;
    }
    Some(PatchesParState {
        lp,
        up,
        nlp,
        nup,
        outer,
        chunk,
        bias_per_chunk,
    })
}

/// CROWN backward for ReLU in Patches mode with optimizable alpha parameters.
///
/// This is the Patches-mode equivalent of [`ReLULayer::propagate_linear_with_alpha`].
/// For each crossing neuron i, the lower bound slope is `alpha[i]` (optimizable)
/// instead of the heuristic value. Returns both the propagated bounds and a
/// per-neuron gradient `d(lower_bound_sum)/d(alpha[i])`.
///
/// The gradient for crossing neuron i equals the sum of positive lower-A
/// coefficients across all output positions that map to neuron i. This is the
/// same mathematical quantity as in Dense mode, just scattered across patches.
///
/// Reference: alpha-beta-CROWN auto_LiRPA/operators/relu.py (Patches backward with alpha)
/// Part of #3293
pub(crate) fn crown_relu_backward_patches_with_alpha(
    bounds: &PatchesLinearBounds,
    pre_activation: &BoundedTensor,
    alpha: &Array1<f32>,
) -> Result<(CrownBounds, Array1<f32>)> {
    crown_relu_backward_patches_with_alpha_impl(bounds, pre_activation, alpha, true)
}

/// Bound-only counterpart of [`crown_relu_backward_patches_with_alpha`].
///
/// Certified coefficient and bias arithmetic is shared with the gradient
/// route; only gradient allocation/accumulation is disabled.
pub(crate) fn crown_relu_backward_patches_with_alpha_bound_only(
    bounds: &PatchesLinearBounds,
    pre_activation: &BoundedTensor,
    alpha: &Array1<f32>,
) -> Result<CrownBounds> {
    crown_relu_backward_patches_with_alpha_impl(bounds, pre_activation, alpha, false)
        .map(|(bounds, _)| bounds)
}

fn crown_relu_backward_patches_with_alpha_impl(
    bounds: &PatchesLinearBounds,
    pre_activation: &BoundedTensor,
    alpha: &Array1<f32>,
    track_gradients: bool,
) -> Result<(CrownBounds, Array1<f32>)> {
    use crate::layers::activations::{relu_crossing_upper_chord, LinearRelaxation};

    // Alpha-ReLU is still affine-only. Refuse Anchored in O(1) before paired
    // common validation walks every origin under a finite graph deadline.
    let affine_geometry = bounds
        .lower_a
        .geometry
        .require_affine("alpha-ReLU Patches backward")?;
    bounds
        .upper_a
        .geometry
        .require_affine("alpha-ReLU Patches backward")?;
    bounds.lower_a.validate_common_geometry(&bounds.upper_a)?;

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
    let num_outputs = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "crown_relu_backward_patches_with_alpha: output dims overflow: {out_c} * {out_h} * {out_w}"
        ))
    })?;
    let (in_c_shape, in_h_shape, in_w_shape) = bounds.lower_a.input_shape;
    let num_input_neurons = checked_shape_product(&[in_c_shape, in_h_shape, in_w_shape]).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "crown_relu_backward_patches_with_alpha: input dims overflow: {in_c_shape} * {in_h_shape} * {in_w_shape}"
        ))
    })?;

    if pre_lower_nd.len() != num_input_neurons {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_input_neurons],
            got: vec![pre_lower_nd.len()],
        });
    }
    if alpha.len() != num_input_neurons {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_input_neurons],
            got: vec![alpha.len()],
        });
    }

    let pre_lower_slice = pre_lower_nd
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous pre_lower array".into()))?;
    let pre_upper_slice = pre_upper_nd
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous pre_upper array".into()))?;

    // Build per-neuron relaxations using alpha for lower slopes of crossing neurons.
    // For stable neurons (always positive/negative), alpha has no effect.
    let relaxations: Vec<LinearRelaxation> = pre_lower_slice
        .iter()
        .zip(pre_upper_slice.iter())
        .enumerate()
        .map(|(i, (&l, &u))| {
            if l.is_nan() || u.is_nan() {
                LinearRelaxation::new(0.0, 0.0, 0.0, f32::INFINITY)
            } else if l >= 0.0 {
                LinearRelaxation::identity()
            } else if u <= 0.0 {
                LinearRelaxation::zero()
            } else if l.is_infinite() && u.is_infinite() {
                LinearRelaxation::new(alpha[i], 0.0, 0.0, f32::INFINITY)
            } else if u.is_infinite() {
                // l < 0, u = +inf: chord limit slope -> 1, intercept -> -l.
                LinearRelaxation::new(alpha[i], 0.0, 1.0, -l)
            } else if l.is_infinite() {
                // l = -inf, u > 0: constant upper y <= u.
                LinearRelaxation::new(alpha[i], 0.0, 0.0, u)
            } else {
                // Crossing: alpha for lower, chord for upper.
                let (lambda, lambda_intercept) = relu_crossing_upper_chord(l, u, None);
                LinearRelaxation::new(alpha[i], 0.0, lambda, lambda_intercept)
            }
        })
        .collect();

    // Track which neurons are crossing (for gradient computation).
    let is_crossing: Vec<bool> = if track_gradients {
        pre_lower_slice
            .iter()
            .zip(pre_upper_slice.iter())
            .map(|(&l, &u)| !l.is_nan() && !u.is_nan() && l < 0.0 && u > 0.0)
            .collect()
    } else {
        Vec::new()
    };

    // Materialize identity patches only when needed; otherwise borrow the
    // existing patches tensor to avoid an O(patch_size) deep clone (perf #3293).
    // `Cow` keeps the owned materialized tensor alive when identity, and borrows
    // the input tensor directly otherwise. Geometry is validated up front and
    // cloned only into the returned carrier, so no tensor clone is needed.
    let lower_owned;
    let lower_patches: &ArrayD<f32> = if bounds.lower_a.identity {
        lower_owned = bounds.lower_a.try_materialize_identity()?;
        lower_owned.patches.as_ref().ok_or_else(|| {
            NyError::InternalError("Materialized identity PatchesData has no patches tensor".into())
        })?
    } else {
        bounds.lower_a.patches.as_ref().ok_or_else(|| {
            NyError::InternalError("Non-identity PatchesData has no patches tensor".into())
        })?
    };
    let upper_owned;
    let upper_patches: &ArrayD<f32> = if bounds.upper_a.identity {
        upper_owned = bounds.upper_a.try_materialize_identity()?;
        upper_owned.patches.as_ref().ok_or_else(|| {
            NyError::InternalError("Materialized identity PatchesData has no patches tensor".into())
        })?
    } else {
        bounds.upper_a.patches.as_ref().ok_or_else(|| {
            NyError::InternalError("Non-identity PatchesData has no patches tensor".into())
        })?
    };
    if upper_patches.shape() != lower_patches.shape() {
        return Err(NyError::ShapeMismatch {
            expected: lower_patches.shape().to_vec(),
            got: upper_patches.shape().to_vec(),
        });
    }

    // Metadata for the output bounds. Anchored layouts are refused above until
    // this coordinate transform is generalized to per-position origins.
    let lower_output_shape = bounds.lower_a.output_shape;
    let lower_input_shape = bounds.lower_a.input_shape;
    let upper_output_shape = bounds.upper_a.output_shape;
    let upper_input_shape = bounds.upper_a.input_shape;

    // Sparse patches: convert to dense for now (sparse alpha not yet implemented).
    if bounds.lower_a.unstable_idx.is_some() {
        let dense_lb = bounds.to_dense()?;
        // Delegate to the Dense propagate_linear_with_alpha.
        // NOTE(#3782): `alpha_upper` is `None` here because patches-mode ReLU
        // is fundamentally single-alpha — the helper models one optimizable lower
        // slope (`alpha[i]`) plus a fixed upper chord (`lambda`). A true
        // dual-alpha patches implementation requires a dedicated relaxation
        // redesign where upper-path slopes are independently optimizable.
        let relu = crate::layers::ReLULayer;
        if track_gradients {
            let (dense_result, grad, _grad_upper) =
                relu.propagate_linear_with_alpha(&dense_lb, pre_activation, alpha, None)?;
            return Ok((CrownBounds::Dense(dense_result), grad));
        }
        let dense_result =
            relu.propagate_linear_with_alpha_bound_only(&dense_lb, pre_activation, alpha, None)?;
        return Ok((CrownBounds::Dense(dense_result), Array1::zeros(0)));
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
    let expected_shape = if explicit_rows {
        vec![bounds.row_count, out_c, out_h, out_w, in_c_shape, kh, kw]
    } else {
        vec![out_c, out_h, out_w, in_c_shape, kh, kw]
    };
    if shape != expected_shape.as_slice() {
        return Err(NyError::ShapeMismatch {
            expected: expected_shape,
            got: shape.to_vec(),
        });
    }
    let logical_rows = if explicit_rows {
        bounds.row_count
    } else {
        num_outputs
    };
    if bounds.lower_b.len() != logical_rows || bounds.upper_b.len() != logical_rows {
        return Err(NyError::ShapeMismatch {
            expected: vec![logical_rows, logical_rows],
            got: vec![bounds.lower_b.len(), bounds.upper_b.len()],
        });
    }

    for err in [
        bounds.lower_a.coeff_err.as_ref(),
        bounds.upper_a.coeff_err.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if err.len() != logical_rows {
            return Err(NyError::ShapeMismatch {
                expected: vec![logical_rows],
                got: vec![err.len()],
            });
        }
    }

    // Both compose sides use the geometry validated before materialization.
    let (sh, sw) = affine_geometry.stride();
    let (pad_left, _pad_right, pad_top, _pad_bottom) = affine_geometry.padding();

    let mut new_lower_patches = ArrayD::<f32>::zeros(lower_patches.raw_dim());
    let mut new_upper_patches = ArrayD::<f32>::zeros(upper_patches.raw_dim());
    let mut new_lower_b_f64 = bounds.lower_b.mapv(|x| x as f64);
    let mut new_upper_b_f64 = bounds.upper_b.mapv(|x| x as f64);
    let mut lower_nonfinite = vec![false; logical_rows];
    let mut upper_nonfinite = vec![false; logical_rows];
    let gradient_len = if track_gradients {
        num_input_neurons
    } else {
        0
    };
    let mut gradient = Array1::<f32>::zeros(gradient_len);

    // #alpha-patches-par: parallel compose over the outermost patch axis.
    //
    // This module had ZERO parallel constructs while its non-alpha twin
    // (`crown_patches.rs`) had thirty, so every root alpha-CROWN iteration ran
    // this 7-deep nest on one thread. Measured on CIFAR100_resnet_medium: one
    // alpha iteration cost ~96s with `crown_backward_step_patches` busy on ~2
    // threads out of 32 and ~87% of all samples parked in rayon's idle path.
    //
    // The outer axis (`row` for 7D explicit rows, `oc` for 6D) partitions the
    // coefficient, bias and non-finite writes into DISJOINT slices, so those
    // stay bit-identical: each output element is still produced by the same
    // `compose_lower`/`compose_upper` call on the same inputs, and each bias
    // accumulates over its own chunk in the same inner order as before.
    //
    // The one cross-chunk quantity is `gradient`, which scatters into shared
    // input-neuron slots. Each chunk accumulates a private partial that is then
    // summed in ASCENDING CHUNK ORDER, so the result is deterministic and
    // run-to-run reproducible, though not bit-identical to the old fully
    // sequential accumulation (a partial restarts its running sum at 0 rather
    // than continuing the global one). That is sound: the gradient only steers
    // the alpha ascent's search direction and never enters a bound — any
    // alpha in [0,1] the ascent visits is a valid relaxation.
    let par_state = patches_par_state(
        lower_patches,
        upper_patches,
        &mut new_lower_patches,
        &mut new_upper_patches,
        explicit_rows,
        bounds.row_count,
        out_c,
        out_h,
        out_w,
        track_gradients,
        num_input_neurons,
    );
    let composed_in_parallel = if let Some(st) = par_state {
        let PatchesParState {
            lp,
            up,
            nlp,
            nup,
            outer,
            chunk,
            bias_per_chunk,
        } = st;
        let lp_all: &[f32] = lp;
        let _: Vec<()> = nlp
            .par_chunks_mut(chunk)
            .zip(nup.par_chunks_mut(chunk))
            .zip(lp.par_chunks(chunk))
            .zip(up.par_chunks(chunk))
            .zip(
                new_lower_b_f64.as_slice_mut().expect("contiguous lower_b")
                    [..outer * bias_per_chunk]
                    .par_chunks_mut(bias_per_chunk),
            )
            .zip(
                new_upper_b_f64.as_slice_mut().expect("contiguous upper_b")
                    [..outer * bias_per_chunk]
                    .par_chunks_mut(bias_per_chunk),
            )
            .zip(lower_nonfinite[..outer * bias_per_chunk].par_chunks_mut(bias_per_chunk))
            .zip(upper_nonfinite[..outer * bias_per_chunk].par_chunks_mut(bias_per_chunk))
            .enumerate()
            .map(|(outer_idx, item)| {
                let (((((((nlp_c, nup_c), lp_c), up_c), nlb_c), nub_c), lnf_c), unf_c) = item;
                // Reproduce the original iteration order exactly within the chunk.
                let (oc_lo, oc_hi) = if explicit_rows {
                    (0, out_c)
                } else {
                    (outer_idx, outer_idx + 1)
                };
                let mut flat = 0usize;
                for oc in oc_lo..oc_hi {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            // Bias/non-finite slot within this chunk: the whole
                            // row for 7D, the (oh,ow) cell of this `oc` for 6D.
                            let slot = if explicit_rows { 0 } else { oh * out_w + ow };
                            let _ = oc;
                            for ic in 0..in_c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let idx = flat;
                                        flat += 1;
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

                                        let la = lp_c[idx];
                                        let lr = compose::compose_lower(la, relax);
                                        nlp_c[idx] = lr.new_coeff;
                                        nlb_c[slot] += lr.intercept_contrib;
                                        lnf_c[slot] |= lr.nonfinite;

                                        let ua = up_c[idx];
                                        let ur = compose::compose_upper(ua, relax);
                                        nup_c[idx] = ur.new_coeff;
                                        nub_c[slot] += ur.intercept_contrib;
                                        unf_c[slot] |= ur.nonfinite;
                                    }
                                }
                            }
                        }
                    }
                }
            })
            .collect();
        // GRADIENT stays sequential, in the original global order.
        //
        // Floating-point addition is not associative, so no parallel reduction
        // of a scatter-accumulate can be bit-identical to the sequential one:
        // a per-chunk partial restarts its running sum at 0 instead of
        // continuing the global prefix, which measurably moved `gradient` by
        // 1 ULP and broke the optimized-vs-reference bit-identity pins. The
        // gradient is one predicated FMA per coefficient against two
        // `compose_*` calls plus four writes in the loop above, so keeping it
        // sequential costs a small fraction of the pass while preserving exact
        // equality with the pre-parallel behavior for EVERY output.
        if track_gradients {
            let mut flat = 0usize;
            let outer_span = if explicit_rows { outer } else { 1 };
            for _ in 0..outer_span {
                for oc in 0..out_c {
                    let _ = oc;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            for ic in 0..in_c {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let idx = flat;
                                        flat += 1;
                                        let ih_raw = (oh * sh + ki) as isize - pad_top as isize;
                                        let iw_raw = (ow * sw + kj) as isize - pad_left as isize;
                                        if ih_raw < 0
                                            || (ih_raw as usize) >= in_h_shape
                                            || iw_raw < 0
                                            || (iw_raw as usize) >= in_w_shape
                                        {
                                            continue;
                                        }
                                        let input_flat = ic * in_h_shape * in_w_shape
                                            + (ih_raw as usize) * in_w_shape
                                            + iw_raw as usize;
                                        let la = lp_all[idx];
                                        if la > 0.0 && is_crossing[input_flat] {
                                            gradient[input_flat] +=
                                                la * pre_lower_slice[input_flat];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        true
    } else {
        false
    };

    if composed_in_parallel {
        // handled above
    } else if explicit_rows {
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
                                    new_lower_patches[[row, oc, oh, ow, ic, ki, kj]] = lr.new_coeff;
                                    new_lower_b_f64[row] += lr.intercept_contrib;
                                    lower_nonfinite[row] |= lr.nonfinite;

                                    if track_gradients && la > 0.0 && is_crossing[input_flat] {
                                        gradient[input_flat] += la * pre_lower_slice[input_flat];
                                    }

                                    let ua = upper_patches[[row, oc, oh, ow, ic, ki, kj]];
                                    let ur = compose::compose_upper(ua, relax);
                                    new_upper_patches[[row, oc, oh, ow, ic, ki, kj]] = ur.new_coeff;
                                    new_upper_b_f64[row] += ur.intercept_contrib;
                                    upper_nonfinite[row] |= ur.nonfinite;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let j = oc * out_h * out_w + oh * out_w + ow;

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

                                let la = lower_patches[[oc, oh, ow, ic, ki, kj]];
                                let lr = compose::compose_lower(la, relax);
                                new_lower_patches[[oc, oh, ow, ic, ki, kj]] = lr.new_coeff;
                                new_lower_b_f64[j] += lr.intercept_contrib;
                                lower_nonfinite[j] |= lr.nonfinite;

                                if track_gradients && la > 0.0 && is_crossing[input_flat] {
                                    gradient[input_flat] += la * pre_lower_slice[input_flat];
                                }

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

    let lower_affected = lower_nonfinite.iter().filter(|&&r| r).count();
    let upper_affected = upper_nonfinite.iter().filter(|&&r| r).count();
    compose::log_nonfinite_fallback(
        "Patches ReLU alpha",
        lower_affected,
        upper_affected,
        logical_rows,
    );

    // Certified coefficient error + intercept-error discharge
    // (#patches-coeff-err-soundness) — the SAME rule as the non-alpha path in
    // `crown_patches.rs`; previously this alpha variant silently DROPPED the
    // incoming coeff_err (emitted None), leaving the --method alpha patches
    // path uncertified. With alpha, `relaxations[..]` carries `alpha[i]` as its
    // lower slope, so the sign-flip envelope term below is exactly
    // `a_err·(|alpha_slope|+|upper_slope|)`; the exact f32-multiply +
    // directed-rounding gap is computed per coefficient and reduced to the
    // per-row max; the relaxation-intercept perturbation `a_err·Σ|intercepts|`
    // is discharged OUTWARD into the f64 bias BEFORE the directed cast below.
    //
    // Two layouts (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §7):
    // - 6D dense: logical row = output position `j`; per-position reductions
    //   (the arm below is byte-identical to the certified 6D design).
    // - 7D explicit-rows: the err index is the SPEC row (axis 0, length
    //   `row_count` == the bias length). Identical per-tap arithmetic, but the
    //   reduction domain is the WHOLE spec row: MAX-lift for the slope
    //   envelope and the exact directed-rounding gaps (one scalar must cover
    //   every coefficient of the row), SUM-lift for the intercept discharge
    //   (the compose loop above folds EVERY output position into the one
    //   spec-row bias slot — a per-position sum would under-discharge by up
    //   to the position count).
    let (lower_coeff_err, upper_coeff_err) = if explicit_rows {
        // ---- 7D explicit-rows arm ----
        let old_lower_err = bounds.lower_a.coeff_err.as_ref();
        let old_upper_err = bounds.upper_a.coeff_err.as_ref();
        // Hard length checks (spec I6): a `Some` err whose length differs
        // from the spec-row count is a construction bug. The shared preflight
        // already enforces this for both 6D and 7D; retain these local checks
        // beside the direct row indexing as defense in depth.
        if let Some(e) = old_lower_err {
            if e.len() != bounds.row_count {
                return Err(NyError::ShapeMismatch {
                    expected: vec![bounds.row_count],
                    got: vec![e.len()],
                });
            }
        }
        if let Some(e) = old_upper_err {
            if e.len() != bounds.row_count {
                return Err(NyError::ShapeMismatch {
                    expected: vec![bounds.row_count],
                    got: vec![e.len()],
                });
            }
        }
        let mut new_lower_err = Array1::<f32>::zeros(bounds.row_count);
        let mut new_upper_err = Array1::<f32>::zeros(bounds.row_count);

        // Per the lead adjudication (spec §14 A1) the 7D bias fold ALSO
        // carries the two `γ̄` f64-summation discharges of the non-alpha rule
        // (spec §6.1), keeping the two activation files in lockstep:
        //   `γ̄·ABS`      — certifies the compose loop's OWN f64 intercept-fold
        //                  accumulation rounding (up to `row_volume` adds per
        //                  row/side; at 7D row sizes this can escape the
        //                  0.5-ulp32 cast slack under bias cancellation), and
        //   `(1+γ̄)` on IS — covers the f64 nearest-summation under-estimate of
        //                  the intercept-magnitude sum itself.
        // `γ̄ = γ_n(8·row_volume + 16)` has ≥ 4x headroom over the Higham
        // factor `γ_n(row_volume + 1)` actually needed, absorbing the
        // over-bound ingredients' own roundings (spec §6.2 (2d)). Both terms
        // are widening-only and vanish as `row_volume` shrinks. Cast-slack
        // dominance needs the row addend count `n << 2^28` (spec §14 E3/F3);
        // cifar-scale rows (~4e6 taps) are 60x under it.
        let row_volume = out_c
            .saturating_mul(out_h)
            .saturating_mul(out_w)
            .saturating_mul(in_c)
            .saturating_mul(kh)
            .saturating_mul(kw);
        debug_assert!(
            (row_volume as u128) < (1u128 << 28),
            "7D alpha coeff_err pass: row addend count {row_volume} breaches the \
             n < 2^28 cast-dominance bound (spec E3/F3)"
        );
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

        // SERIAL row loop (spec I8): this whole function is serial — the
        // gradient `+=` in the compose loop accumulates across rows — and the
        // err pass mirrors that (read-only over the coefficient tensors, so
        // coefficient/bias VALUE accumulation order is untouched; I3).
        // #alpha-err-hoist: `max_slope_sum` and `int_sum` are ROW-INVARIANT.
        //
        // Inside the tap nest below, `input_flat` is
        // `ic*in_h*in_w + ih*in_w + iw` with `ih`/`iw` derived only from
        // `(oh,ow,ki,kj)` — it does not depend on `row` or on any coefficient.
        // So `relax`, and therefore `ss` and the intercept term, are identical
        // for every one of `row_count` rows, and this pass was recomputing the
        // same two f64 scalars `row_count` times over the whole tap volume.
        // Hoisted here and computed once.
        //
        // BIT-IDENTICAL: `max_slope_sum` is a max (order-free), and `int_sum`
        // accumulates over exactly the same non-padding taps in exactly the
        // same order as each row's copy did, so it reproduces that copy's
        // value bit for bit.
        let mut max_slope_sum = 0.0f64;
        let mut int_sum = 0.0f64;
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let _ = oc;
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
                                let input_flat = ic * in_h_shape * in_w_shape
                                    + (ih_raw as usize) * in_w_shape
                                    + iw_raw as usize;
                                let relax = &relaxations[input_flat];
                                let ss = f64::from(relax.lower_slope).abs()
                                    + f64::from(relax.upper_slope).abs();
                                if ss > max_slope_sum {
                                    max_slope_sum = ss;
                                }
                                int_sum += f64::from(relax.lower_intercept).abs()
                                    + f64::from(relax.upper_intercept).abs();
                            }
                        }
                    }
                }
            }
        }

        // #alpha-err-par: each row reads only its own coefficient slice and
        // writes only its own `[row]` bias/err slots, so the rows are a pure
        // independent map — the last fully serial pass in this function after
        // #alpha-patches-par parallelized the compose. Per-row f64
        // accumulation order is untouched (each row still folds its own taps
        // in the original nest order), so every emitted value is bit-identical.
        let new_lower_patches_ro = &new_lower_patches;
        let new_upper_patches_ro = &new_upper_patches;
        let row_terms: Vec<(f64, f64, f64, f64)> = (0..bounds.row_count)
            .into_par_iter()
            .map(|row| {
                let oe_l = old_lower_err.map_or(0.0, |e| sanitize(e[row]));
                let oe_u = old_upper_err.map_or(0.0, |e| sanitize(e[row]));

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

                                        // EXACT directed-rounding gap per side
                                        // (mirror compose_*): compose_lower uses
                                        // the lower slope/intercept for a>0 else
                                        // the upper pair; compose_upper the
                                        // reverse.
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
                                                new_lower_patches_ro[[row, oc, oh, ow, ic, ki, kj]],
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
                                                new_upper_patches_ro[[row, oc, oh, ow, ic, ki, kj]],
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

                // Bias discharge D_σ into the f64 accumulator BEFORE the directed
                // cast (spec I4). `oe == 0` short-circuits `0·∞ = NaN` for
                // degenerate ±∞ relaxation intercepts (spec I5); any non-finite D
                // (∞ from a poisoned carried err or a degenerate intercept in the
                // receptive field, or NaN from ∞·0) poisons the bias OUTWARD —
                // skipping would emit a finite bound the true range can escape
                // (false-VERIFIED class; spec §14 A2).
                let disc_l = gamma_bar * abs_lower_sum
                    + if oe_l != 0.0 {
                        oe_l * (int_sum * (1.0 + gamma_bar))
                    } else {
                        0.0
                    };

                let disc_u = gamma_bar * abs_upper_sum
                    + if oe_u != 0.0 {
                        oe_u * (int_sum * (1.0 + gamma_bar))
                    } else {
                        0.0
                    };

                // Err emission: slope envelope (row-wide MAX — one scalar covers
                // every coefficient of the row) + exact gap; f64 compute, one
                // outward next_up_f32 at the f32 cast (spec I4). Non-finite (a
                // poisoned +INF carried err, or NaN from `∞·0` on an all-padding
                // row) emits +INF — the degrade poison — NEVER NaN (spec I5).
                // Nonfinite rows are zeroed + bias-poisoned by the fallback
                // below, so err 0.0 is exact there (vacuous certificate).
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
                (disc_l, disc_u, lv, uv)
            })
            .collect();

        // Apply serially (O(row_count)); identical branch structure to the
        // pre-parallel code, just fed from the collected per-row terms.
        for (row, (disc_l, disc_u, lv, uv)) in row_terms.into_iter().enumerate() {
            if disc_l.is_finite() {
                new_lower_b_f64[row] -= disc_l;
            } else {
                new_lower_b_f64[row] = f64::NEG_INFINITY;
            }
            if disc_u.is_finite() {
                new_upper_b_f64[row] += disc_u;
            } else {
                new_upper_b_f64[row] = f64::INFINITY;
            }
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
        (Some(new_lower_err), Some(new_upper_err))
    } else {
        let old_lower_err = bounds.lower_a.coeff_err.as_ref();
        let old_upper_err = bounds.upper_a.coeff_err.as_ref();
        let mut new_lower_err = Array1::<f32>::zeros(logical_rows);
        let mut new_upper_err = Array1::<f32>::zeros(logical_rows);
        let sanitize = |value: f32| -> f64 {
            if value.is_finite() && value >= 0.0 {
                f64::from(value)
            } else {
                f64::INFINITY
            }
        };
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let j = oc * out_h * out_w + oh * out_w + ow;
                    let oe_l = old_lower_err.map_or(0.0, |e| sanitize(e[j]));
                    let oe_u = old_upper_err.map_or(0.0, |e| sanitize(e[j]));

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
                                // compose_lower uses lower_slope (= alpha) for a>0 else
                                // upper_slope; compose_upper the reverse.
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
                    // the f64 bias. `oe == 0` short-circuits `0·∞ = NaN` (nothing to
                    // discharge). An INFINITE discharge (oe > 0 with a degenerate ±∞
                    // relaxation intercept in the receptive field) requires a VACUOUS
                    // bound — poison outward rather than skip (skipping would emit a
                    // finite bound the true range can escape). Unreachable on this
                    // alpha path today (non_finite_domain_guard rejects ±∞/NaN
                    // pre-activations upstream) but kept identical to the non-alpha
                    // rule so the two files cannot drift.
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
                    let lower_total = lterm + max_lower_gap;
                    let upper_total = uterm + max_upper_gap;
                    new_lower_err[j] = if lower_nonfinite[j] {
                        0.0
                    } else if !lower_total.is_finite() {
                        f32::INFINITY
                    } else {
                        next_up_f32(lower_total as f32)
                    };
                    new_upper_err[j] = if upper_nonfinite[j] {
                        0.0
                    } else if !upper_total.is_finite() {
                        f32::INFINITY
                    } else {
                        next_up_f32(upper_total as f32)
                    };
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
        for j in 0..num_outputs {
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

    let mut folded = PatchesLinearBounds {
        row_count: bounds.row_count,
        lower_a: PatchesData {
            coeff_err: lower_coeff_err,
            patches: Some(new_lower_patches),
            geometry: bounds.lower_a.geometry.clone(),
            identity: false,
            output_shape: lower_output_shape,
            input_shape: lower_input_shape,
            unstable_idx: None,
        },
        lower_b: new_lower_b,
        upper_a: PatchesData {
            coeff_err: upper_coeff_err,
            patches: Some(new_upper_patches),
            geometry: bounds.upper_a.geometry.clone(),
            identity: false,
            output_shape: upper_output_shape,
            input_shape: upper_input_shape,
            unstable_idx: None,
        },
        upper_b: new_upper_b,
    };

    // #patches-eager-err: discharge the carried coefficient error HERE, against
    // the pre-activation cut these columns multiply — the same policy the dense
    // path applies at every activation backward step
    // (`LinearBounds::fold_coeff_err_over_box_eager`, dispatched at
    // layers/layer_enum/dispatch.rs). Without it a conv stack carries the error
    // to the network input and pays for it after ABS-composition through every
    // remaining layer, which grows at IBP scale with depth. Rows with a
    // non-finite penalty, and layouts the fold does not model exactly, keep
    // carrying — never a new degrade. See bounds/patches/eager_err.rs.
    if crate::bounds::patches::eager_err_enabled() {
        folded.fold_coeff_err_over_box_eager(pre_activation);
    }

    let result = CrownBounds::Patches(Box::new(folded));

    Ok((result, gradient))
}
